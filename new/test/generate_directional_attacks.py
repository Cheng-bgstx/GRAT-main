from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

ROOT = Path("/root/data2/fyc/fyc/GRAT-main")
NEW_ROOT = ROOT / "new"
if str(NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(NEW_ROOT))

from shared_subspace_codd import load_shared_subspace, normalize_l1, project_to_shared_subspace
from theory_common import (
    SynapsePaths,
    device_from_arg,
    ensure_dir,
    extract_case_name,
    get_2d_loader,
    get_synapse_depth,
    get_synapse_loader,
    load_source_sam,
    prepare_2d_image,
    prepare_synapse_slice,
    repack_synapse_png_to_h5,
    save_2d_adv,
    setup_seed,
    write_json,
)


criterion = torch.nn.L1Loss()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Synapse", choices=["Synapse", "ISIC", "REFUGE"])
    parser.add_argument("--vit_name", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_ckpt", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--att_step_size", type=float, default=2.0)
    parser.add_argument("--att_bound", type=float, default=10.0)
    parser.add_argument("--att_numstep", type=int, default=5)
    parser.add_argument("--momentum_decay", type=float, default=0.5)
    parser.add_argument("--ads_rho", type=float, default=1.0)
    parser.add_argument("--ads_gamma", type=float, default=1e-4)
    parser.add_argument(
        "--direction_mode",
        type=str,
        default="radial",
        choices=["radial", "tangential", "isotropic", "misaligned_radial"],
    )
    parser.add_argument("--shared_subspace_path", type=str, default="")
    parser.add_argument("--shared_mode", type=str, default="none", choices=["none", "random", "mean", "top1", "topk"])
    parser.add_argument("--shared_weight", type=float, default=0.15)
    parser.add_argument("--max_cases", type=int, default=2)
    parser.add_argument("--max_slices", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--repack_path", type=str, default="")
    return parser.parse_args()


def dataset_num_classes(dataset: str, num_classes: int) -> int:
    if dataset == "ISIC":
        return 1
    if dataset == "REFUGE":
        return 2
    return num_classes


def build_loader(args):
    if args.dataset == "Synapse":
        synapse = SynapsePaths()
        return get_synapse_loader(synapse.volume_path, synapse.list_dir)
    return get_2d_loader(args.dataset, args.image_size)


def return_mean_value(input_tensor: torch.Tensor) -> torch.Tensor:
    input_tensor = torch.abs(input_tensor)
    list_mean = [input_tensor[i].mean() for i in range(len(input_tensor))]
    input_mean = torch.stack(list_mean)
    input_mean = input_mean.view(-1, *([1] * (input_tensor.dim() - 1)))
    return input_mean


def get_edge_map(input_tensor: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(input_tensor[:, :, :, :-1] - input_tensor[:, :, :, 1:])
    dy = torch.abs(input_tensor[:, :, :-1, :] - input_tensor[:, :, 1:, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    edge_min = edge.view(edge.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    edge_max = edge.view(edge.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    return (edge - edge_min) / (edge_max - edge_min + 1e-6)


def return_eta(momentum: torch.Tensor, step_size: float) -> torch.Tensor:
    upd_momentum = momentum * (step_size / (torch.mean(torch.abs(momentum)) + 1e-12))
    device_local = momentum.device
    momentum_binary = torch.where(momentum > 0, torch.tensor(1.0, device=device_local), torch.tensor(0.0, device=device_local))
    eta = torch.clamp(upd_momentum, -1 * step_size, -0.25 * step_size) * (1 - momentum_binary) + momentum_binary * torch.clamp(
        upd_momentum, 0.25 * step_size, 1 * step_size
    )
    return eta


def match_energy(xi: torch.Tensor, target_rms: torch.Tensor) -> torch.Tensor:
    current_rms = torch.sqrt(torch.mean(xi.square(), dim=(-3, -2, -1), keepdim=True) + 1e-12)
    return xi * (target_rms / current_rms)


def synthesize_directional_probe(
    f_clean: torch.Tensor,
    f_adv: torch.Tensor,
    direction_mode: str,
    ads_rho: float,
    ads_gamma: float,
) -> torch.Tensor:
    f_freq = torch.fft.fftn(f_clean, dim=(-2, -1), norm="ortho")
    amp = torch.abs(f_freq)
    phase_clean = f_freq.angle()
    delta_ref = f_adv.detach() * 0.999 - f_clean.detach()
    target_rms = torch.sqrt(torch.mean(delta_ref.square(), dim=(-3, -2, -1), keepdim=True) + 1e-12)
    spectrum_energy = torch.mean(amp, dim=(-3, -2, -1), keepdim=True)
    lambda_val = target_rms / (spectrum_energy + 1e-6)
    sigma_freq = lambda_val * (amp + 1e-12).pow(ads_rho) + ads_gamma
    n_rad = torch.randn_like(amp)
    n_tan = torch.randn_like(amp)
    carrier = torch.exp(1j * phase_clean)
    if direction_mode == "radial":
        coeff = (n_rad * sigma_freq) * carrier
    elif direction_mode == "tangential":
        coeff = (n_tan * sigma_freq) * (1j * carrier)
    elif direction_mode == "isotropic":
        scale = 1.0 / math.sqrt(2.0)
        coeff = scale * (n_rad * sigma_freq) * carrier + scale * (n_tan * sigma_freq) * (1j * carrier)
    elif direction_mode == "misaligned_radial":
        phase_offset = (torch.rand_like(phase_clean) * 2.0 - 1.0) * math.pi
        wrong_carrier = torch.exp(1j * (phase_clean + phase_offset))
        coeff = (n_rad * sigma_freq) * wrong_carrier
    else:
        raise ValueError(direction_mode)
    xi = torch.fft.ifftn(coeff, dim=(-2, -1), norm="ortho").real
    return match_energy(xi, target_rms)


def calculate_loss_directional(
    features: torch.Tensor,
    targets: torch.Tensor,
    features_layer,
    targets_layer,
    direction_mode: str,
    ads_rho: float,
    ads_gamma: float,
):
    loss_org = criterion(features * 0.9999, targets)
    pre_layer = int(len(features_layer) * 0.9)
    num_layers = min(6, pre_layer)
    selected_indices = list(range(pre_layer - num_layers, pre_layer))
    feature_difference_list = []
    for idx in selected_indices[-4:]:
        f_clean = targets_layer[idx].unsqueeze(0)
        f_adv = features_layer[idx].unsqueeze(0)
        xi_struct = synthesize_directional_probe(f_clean, f_adv, direction_mode, ads_rho, ads_gamma)
        feature_difference = (f_adv * 0.999 - f_clean) - xi_struct
        feature_difference_list.append(feature_difference.squeeze(0))
    selected_targets = torch.stack([targets_layer[idx] for idx in selected_indices[-4:]])
    input_mean = return_mean_value(selected_targets)
    feature_difference = torch.stack(feature_difference_list)
    feature_difference = feature_difference / (input_mean + 1e-6)
    l1_losses = (torch.abs(torch.mean(feature_difference, dim=0))).mean()
    return l1_losses + loss_org


def build_init_direction(grad: torch.Tensor, shared_mode: str, shared_weight: float, bundle) -> torch.Tensor:
    g = normalize_l1(grad)
    if shared_mode == "none" or bundle is None:
        return g
    if shared_mode == "random":
        return normalize_l1(torch.randn_like(grad))
    if shared_mode == "mean":
        return normalize_l1(g + shared_weight * normalize_l1(bundle.mean_direction.to(g.device)))
    if shared_mode == "top1":
        top1 = bundle.basis[:1]
        shape = bundle.shape
        coeff = g.reshape(1, -1) @ top1.T.to(g.device)
        proj = (coeff @ top1.to(g.device)).reshape(shape)
        return normalize_l1(g + shared_weight * normalize_l1(proj))
    if shared_mode == "topk":
        proj = project_to_shared_subspace(g, bundle)
        return normalize_l1(g + shared_weight * proj)
    raise ValueError(shared_mode)


def save_synapse_slice(out_path: str, case_name: str, slice_index: int, image_tensor: np.ndarray) -> None:
    folder_path = Path(out_path) / case_name
    ensure_dir(folder_path)
    image_data_out = np.transpose(image_tensor, (1, 2, 0))
    image_data_out = np.clip(image_data_out, 0, 1)
    gray = (image_data_out[:, :, 0] * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    from PIL import Image

    Image.fromarray(rgb).save(folder_path / f"{slice_index}.png")


def run_single_attack(sam, input_image_torch: torch.Tensor, args, shared_bundle):
    anatomical_edge = get_edge_map(input_image_torch).detach()
    with torch.no_grad():
        targets, targets_layer = sam.image_encoder(input_image_torch)
    max_val, min_val = input_image_torch.max(), input_image_torch.min()
    step_size = float((max_val - min_val) * args.att_step_size / 255.0)
    epsilon = float((max_val - min_val) * args.att_bound / 255.0)
    perturb_img = Variable(input_image_torch.data + torch.randn_like(input_image_torch) * 1e-5, requires_grad=True)
    momentum = torch.zeros_like(input_image_torch).detach()
    losses = []
    for t in range(args.att_numstep):
        sam.image_encoder.zero_grad()
        features, features_layer = sam.image_encoder(perturb_img)
        loss = calculate_loss_directional(
            features=features,
            targets=targets,
            features_layer=features_layer,
            targets_layer=targets_layer,
            direction_mode=args.direction_mode,
            ads_rho=args.ads_rho,
            ads_gamma=args.ads_gamma,
        )
        losses.append(float(loss.detach().item()))
        grad = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]
        grad = grad * (1.0 + 3.0 * anatomical_edge)
        grad = normalize_l1(grad.detach())
        if t == 0:
            momentum = build_init_direction(grad, args.shared_mode, args.shared_weight, shared_bundle)
        else:
            momentum = normalize_l1(args.momentum_decay * momentum.detach() + grad)
        eta = return_eta(momentum, step_size).detach()
        perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
        eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
        perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)
    return perturb_img[0].detach().cpu().numpy(), losses


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = device_from_arg(args.device)
    args.num_classes = dataset_num_classes(args.dataset, args.num_classes)
    loader = build_loader(args)
    sam = load_source_sam(args.vit_name, args.sam_ckpt, args.image_size, args.num_classes, device)
    shared_bundle = load_shared_subspace(args.shared_subspace_path, device) if args.shared_subspace_path else None
    trace_rows = []
    processed_cases = 0
    for sampled_batch in loader:
        if processed_cases >= args.max_cases:
            break
        case_name = extract_case_name(sampled_batch)
        if args.dataset == "Synapse":
            depth = get_synapse_depth(sampled_batch)
            slice_indices = list(range(depth))
            if args.max_slices > 0:
                slice_indices = slice_indices[: args.max_slices]
            for slice_index in slice_indices:
                start_time = time.time()
                input_image_torch, _ = prepare_synapse_slice(sampled_batch, slice_index, device)
                image_out, losses = run_single_attack(sam, input_image_torch, args, shared_bundle)
                save_synapse_slice(args.out_path, case_name, slice_index, image_out)
                trace_rows.append({"dataset": args.dataset, "case_name": case_name, "unit_name": f"{case_name}:{slice_index}", "losses": losses})
                print(f"[{case_name}][slice {slice_index}] done in {time.time() - start_time:.2f}s")
        else:
            start_time = time.time()
            input_image_torch, _ = prepare_2d_image(sampled_batch, device)
            image_out, losses = run_single_attack(sam, input_image_torch, args, shared_bundle)
            save_2d_adv(args.out_path, case_name, image_out)
            trace_rows.append({"dataset": args.dataset, "case_name": case_name, "unit_name": case_name, "losses": losses})
            print(f"[{case_name}] done in {time.time() - start_time:.2f}s")
        processed_cases += 1
    if args.dataset == "Synapse" and args.repack_path:
        repack_synapse_png_to_h5(args.out_path, args.repack_path, SynapsePaths().volume_path)
        print(f"Repacked to {args.repack_path}")
    max_steps = max((len(row["losses"]) for row in trace_rows), default=0)
    mean_loss_by_step = []
    for step_idx in range(max_steps):
        step_values = [row["losses"][step_idx] for row in trace_rows if step_idx < len(row["losses"])]
        mean_loss_by_step.append(float(np.mean(step_values)))
    write_json(
        Path(args.out_path).parent / f"{Path(args.out_path).name}_trace.json",
        {
            "config": vars(args),
            "num_units": len(trace_rows),
            "mean_loss_by_step": mean_loss_by_step,
            "rows": trace_rows,
        },
    )


if __name__ == "__main__":
    main()
