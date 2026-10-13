from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path("/root/data2/fyc/fyc/GRAT-main")
NEW_ROOT = ROOT / "new"
if str(NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(NEW_ROOT))

from theory_common import (
    SynapsePaths,
    build_support_mask,
    decompose_drift_energy,
    device_from_arg,
    extract_case_name,
    get_2d_loader,
    get_synapse_depth,
    get_synapse_loader,
    load_medsa_model,
    load_samed_encoder,
    load_source_sam,
    prepare_2d_image,
    prepare_synapse_slice,
    setup_seed,
    write_json,
)
from generate_directional_attacks import (
    build_init_direction,
    calculate_loss_directional,
    dataset_num_classes,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Synapse", choices=["Synapse", "ISIC", "REFUGE"])
    parser.add_argument("--vit_name", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_ckpt", type=str, required=True)
    parser.add_argument("--samed_ckpt", type=str, required=True)
    parser.add_argument("--medsa_ckpt", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--att_numstep", type=int, default=5)
    parser.add_argument("--shared_weight", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def get_example(args, device):
    if args.dataset == "Synapse":
        loader = get_synapse_loader(SynapsePaths().volume_path, SynapsePaths().list_dir)
        sampled_batch = next(iter(loader))
        case_name = extract_case_name(sampled_batch)
        slice_idx = get_synapse_depth(sampled_batch) // 2
        image, _ = prepare_synapse_slice(sampled_batch, slice_idx, device)
        return case_name, image
    loader = get_2d_loader(args.dataset, args.image_size)
    sampled_batch = next(iter(loader))
    case_name = extract_case_name(sampled_batch)
    image, _ = prepare_2d_image(sampled_batch, device)
    return case_name, image


def select_coeffs(feat_s: torch.Tensor, topk: int = 128):
    gs = torch.fft.fft2(feat_s, norm="ortho")
    support_mask = build_support_mask(gs, energy_quantile=0.9)
    amp = torch.abs(gs)
    masked_amp = amp[support_mask]
    topk = min(topk, masked_amp.numel())
    values, _ = torch.topk(masked_amp.reshape(-1), k=topk)
    threshold = values[-1]
    mask = support_mask & (amp >= threshold)
    return gs, mask


def flatten_coeffs(freq: torch.Tensor, mask: torch.Tensor):
    coeffs = freq[mask]
    return coeffs.real.detach().cpu().numpy(), coeffs.imag.detach().cpu().numpy()


def local_alpha_beta(source_freq: torch.Tensor, target_freq: torch.Tensor, mask: torch.Tensor):
    carrier = torch.exp(1j * torch.angle(source_freq))
    delta = (target_freq - source_freq) * torch.conj(carrier)
    coeffs = delta[mask]
    alpha = coeffs.real.detach().cpu().numpy()
    beta = coeffs.imag.detach().cpu().numpy()
    return alpha, beta


def anisotropy_ratio(alpha: np.ndarray, beta: np.ndarray) -> float:
    alpha_var = float(np.var(alpha)) if alpha.size else 0.0
    beta_var = float(np.var(beta)) if beta.size else 0.0
    return alpha_var / (beta_var + 1e-12)


def trajectory_points(model, image, args, direction_mode: str):
    perturb = image.clone().detach().requires_grad_(True)
    with torch.no_grad():
        targets, targets_layer = model.image_encoder(image)
        src_freq, mask = select_coeffs(targets, topk=48)
    points = []
    momentum = torch.zeros_like(image)
    step_size = float((image.max() - image.min()) * 2.0 / 255.0)
    for step in range(args.att_numstep):
        features, features_layer = model.image_encoder(perturb)
        loss = calculate_loss_directional(
            features=features,
            targets=targets,
            features_layer=features_layer,
            targets_layer=targets_layer,
            direction_mode=direction_mode,
            ads_rho=1.0,
            ads_gamma=1e-4,
        )
        grad = torch.autograd.grad(loss, perturb, retain_graph=False, create_graph=False)[0]
        grad = grad.detach()
        if step == 0:
            momentum = build_init_direction(grad, "none", args.shared_weight, None)
        else:
            momentum = 0.5 * momentum + grad
        perturb = (perturb.detach() + momentum.sign() * step_size).clamp(image.min(), image.max()).requires_grad_(True)
        with torch.no_grad():
            feat_adv, _ = model.image_encoder(perturb)
            freq_adv = torch.fft.fft2(feat_adv, norm="ortho")
            alpha, beta = local_alpha_beta(src_freq, freq_adv, mask)
            points.append((alpha, beta))
    alpha0 = np.zeros(int(mask.sum().item()), dtype=np.float32)
    beta0 = np.zeros(int(mask.sum().item()), dtype=np.float32)
    return (alpha0, beta0), points


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = device_from_arg(args.device)
    args.num_classes = dataset_num_classes(args.dataset, args.num_classes)
    case_name, image = get_example(args, device)
    source_sam = load_source_sam(args.vit_name, args.sam_ckpt, args.image_size, args.num_classes, device)
    _, samed_encoder = load_samed_encoder(args.vit_name, args.sam_ckpt, args.samed_ckpt, args.image_size, args.num_classes, args.rank, device)
    medsa_model = load_medsa_model(args.vit_name, args.sam_ckpt, args.medsa_ckpt, args.image_size, args.num_classes + 1, device)

    with torch.no_grad():
        feat_s, _ = source_sam.image_encoder(image)
        feat_t_samed, _ = samed_encoder(image)
        medsa_out = medsa_model.image_encoder(image)
        feat_t_medsa = medsa_out[0] if isinstance(medsa_out, (list, tuple)) else medsa_out
        src_freq, mask = select_coeffs(feat_s)
        samed_freq = torch.fft.fft2(feat_t_samed, norm="ortho")
        medsa_freq = torch.fft.fft2(feat_t_medsa, norm="ortho")
        support_mask = build_support_mask(src_freq, energy_quantile=0.9)
        drift_stats = {
            "SAMed": decompose_drift_energy(feat_s, feat_t_samed, support_mask=support_mask),
            "MedSA": decompose_drift_energy(feat_s, feat_t_medsa, support_mask=support_mask),
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samed_alpha, samed_beta = local_alpha_beta(src_freq, samed_freq, mask)
    medsa_alpha, medsa_beta = local_alpha_beta(src_freq, medsa_freq, mask)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(samed_alpha, samed_beta, s=8, alpha=0.35, label="SAMed")
    axes[0].scatter(medsa_alpha, medsa_beta, s=8, alpha=0.35, label="Med-SA")
    axes[0].axhline(0.0, color="#666666", linewidth=0.8)
    axes[0].axvline(0.0, color="#666666", linewidth=0.8)
    axes[0].set_title("Drift Distribution in Alpha-Beta")
    axes[0].set_xlabel("Alpha (radial)")
    axes[0].set_ylabel("Beta (tangential)")
    axes[0].legend()

    (base_alpha, base_beta), radial_points = trajectory_points(source_sam, image, args, "radial")
    (_, _), isotropic_points = trajectory_points(source_sam, image, args, "isotropic")
    (_, _), tangential_points = trajectory_points(source_sam, image, args, "tangential")
    axes[1].scatter(base_alpha, base_beta, s=8, alpha=0.25, label="Source")
    sample_idx = np.linspace(0, len(base_alpha) - 1, num=min(16, len(base_alpha)), dtype=int)
    for idx in sample_idx:
        radial_x = [base_alpha[idx]] + [p[0][idx] for p in radial_points]
        radial_y = [base_beta[idx]] + [p[1][idx] for p in radial_points]
        iso_x = [base_alpha[idx]] + [p[0][idx] for p in isotropic_points]
        iso_y = [base_beta[idx]] + [p[1][idx] for p in isotropic_points]
        tan_x = [base_alpha[idx]] + [p[0][idx] for p in tangential_points]
        tan_y = [base_beta[idx]] + [p[1][idx] for p in tangential_points]
        axes[1].plot(radial_x, radial_y, color="#d04a02", alpha=0.4)
        axes[1].plot(iso_x, iso_y, color="#1f77b4", alpha=0.25)
        axes[1].plot(tan_x, tan_y, color="#2a9d8f", alpha=0.25)
    axes[1].axhline(0.0, color="#666666", linewidth=0.8)
    axes[1].axvline(0.0, color="#666666", linewidth=0.8)
    axes[1].set_title("Attack Trajectory in Alpha-Beta")
    axes[1].set_xlabel("Alpha (radial)")
    axes[1].set_ylabel("Beta (tangential)")
    axes[1].plot([], [], color="#d04a02", label="Radial")
    axes[1].plot([], [], color="#1f77b4", label="Isotropic")
    axes[1].plot([], [], color="#2a9d8f", label="Tangential")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "geometry.png", dpi=200)
    plt.close(fig)

    write_json(
        out_dir / "geometry_summary.json",
        {
            "dataset": args.dataset,
            "vit_name": args.vit_name,
            "case_name": case_name,
            "drift_stats": drift_stats,
            "alpha_beta_stats": {
                "SAMed": {"anisotropy_ratio": anisotropy_ratio(samed_alpha, samed_beta)},
                "MedSA": {"anisotropy_ratio": anisotropy_ratio(medsa_alpha, medsa_beta)},
            },
            "figure": str(out_dir / "geometry.png"),
        },
    )
    print(json.dumps({"case_name": case_name, "drift_stats": drift_stats}, indent=2))


if __name__ == "__main__":
    main()
