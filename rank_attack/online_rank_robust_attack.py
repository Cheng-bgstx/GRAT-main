"""Online rank-robust transfer attack for medically adapted SAM models.

The attack uses only the open-source SAM encoder plus an offline rank profile.
Victim weights are never accessed by this script.

At each iteration:
  1) estimate exact first-order adaptation sensitivity matrices Q_c on selected
     PEFT channels;
  2) compute the worst rank-r, Frobenius-bounded local adaptation DeltaW_c*;
  3) if the relative first-order adaptation risk exceeds the offline threshold,
     optimize a mixture of source gain and worst-case adapted gain;
  4) update the image perturbation with momentum under the same L_inf interface
     as the user's existing attack code.

This is a stop-gradient inner minimax approximation: DeltaW_c* is recomputed
from the current attack state but detached during the outer image update.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from rank_adapt_utils import (
    ROOT,
    LowRankInjector,
    RankProfileBundle,
    compute_gain,
    cosine_flat,
    encoder_forward,
    estimate_channel_q,
    normalize_l1,
    resolve_profile_channels,
    return_eta,
    risk_from_q,
    worst_updates_from_q,
)

from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
from datasets.refuge import REFUGE
from datasets.isic import ISIC2016


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Synapse", choices=["Synapse", "ISIC", "REFUGE"])
    p.add_argument("--vit_name", type=str, default="vit_b")
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--ckpt", type=str, default="checkpoints/sam_vit_b_01ec64.pth")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--volume_path", type=str, default="data/Synapse_preprocess/test_vol_h5")
    p.add_argument("--list_dir", type=str, default="data/Synapse_preprocess/lists")
    p.add_argument("--input_image", type=str, default=None,
                   help="optional PNG/JPG/NPY input; if set, dataset loader is bypassed")
    p.add_argument("--test_batch", type=str, default=None)
    p.add_argument("--max_slices", type=int, default=-1)
    p.add_argument("--indices", type=str, default=None,
                   help="optional comma-separated dataset indices for a fixed 2D gate")
    p.add_argument("--max_cases", type=int, default=-1,
                   help="optional number of dataset cases; applied after --indices filtering")

    p.add_argument("--profile", type=str, required=True,
                   help="offline RankProfileBundle .pth")
    p.add_argument("--robust_lambda", type=float, default=0.5,
                   help="when gated: J=(1-lambda)G_source + lambda G_worst")
    p.add_argument("--tau_risk", type=float, default=None,
                   help="override offline risk threshold")
    p.add_argument("--disable_gate", action="store_true",
                   help="always use the robust objective")
    p.add_argument("--q_refresh", type=int, default=1,
                   help="recompute Q / worst adaptation every N attack steps")

    p.add_argument("--layer_weight", type=float, default=None,
                   help="default: value stored in profile")
    p.add_argument("--core_num_layers", type=int, default=None,
                   help="default: value stored in profile")
    p.add_argument("--svd_oversample", type=int, default=8)
    p.add_argument("--svd_niter", type=int, default=4)

    # image attack budget, preserving the old interface
    p.add_argument("--momentum_decay", type=float, default=0.5)
    p.add_argument("--att_step_size", type=float, default=2.0)
    p.add_argument("--att_bound", type=float, default=10.0)
    p.add_argument("--att_numstep", type=int, default=10)
    p.add_argument("--step_rule", type=str, default="asymmetric", choices=["asymmetric", "sign"])
    p.add_argument("--init_jitter", type=float, default=1e-5,
                   help="only breaks zero-gradient symmetry at x_adv=x_clean")

    # optional legacy spatial edge weighting; off by default
    p.add_argument("--edge_guidance", type=int, default=0)
    p.add_argument("--edge_gamma", type=float, default=3.0)

    p.add_argument("--out_path", type=str, default="testset/adv_rank_robust")
    p.add_argument("--trace_path", type=str, default=None,
                   help="defaults to <out_path>/trace.jsonl")
    p.add_argument("--diagnostic_cosine", type=int, default=0,
                   help="if 1, re-forward source objective to log source-vs-final gradient cosine")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(args):
    tfs = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
    tfs_msk = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    if args.dataset == "Synapse":
        db = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test_vol")
        is_3d = True
    elif args.dataset == "REFUGE":
        db = REFUGE(args=args, data_path=args.volume_path, mode="Test", transform=tfs, transform_msk=tfs_msk)
        is_3d = False
    elif args.dataset == "ISIC":
        db = ISIC2016(args=args, data_path=args.volume_path, mode="Test", transform=tfs, transform_msk=tfs_msk)
        is_3d = False
    else:
        raise ValueError(args.dataset)
    return DataLoader(db, batch_size=1, shuffle=False, num_workers=4), is_3d


def get_case_name(batch: Dict) -> str:
    if "case_name" in batch:
        cn = batch["case_name"]
        return cn[0] if isinstance(cn, (list, tuple)) else str(cn)
    if "image_meta_dict" in batch:
        return str(batch["image_meta_dict"]["filename_or_obj"][0])
    return "unknown"


def batch_to_slices(batch: Dict, dataset: str) -> np.ndarray:
    img_tensor = batch["image"]
    if dataset == "Synapse":
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(1)
        vol = img_tensor.squeeze(0).cpu().detach().numpy()
        if len(vol.shape) == 2:
            vol = vol[np.newaxis, np.newaxis, :, :]
        elif len(vol.shape) == 3:
            vol = vol[:, np.newaxis, :, :]
        return vol
    return img_tensor.cpu().detach().numpy()


def load_direct_input(path: str, img_size: int) -> torch.Tensor:
    """Return [1,C,H,W] float tensor in [0,1]."""
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3:
            raise ValueError(f"unsupported NPY image shape: {arr.shape}")
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = arr.astype(np.float32)
        if arr.max() > 1.1:
            arr = arr / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
        return t.clamp(0, 1)
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_adv(image_np: np.ndarray, case_name: str, ind: int, is_3d: bool, out_path: str):
    os.makedirs(out_path, exist_ok=True)
    image_np = np.clip(image_np, 0.0, 1.0)
    if is_3d:
        folder = os.path.join(out_path, case_name)
        os.makedirs(folder, exist_ok=True)
        plt.imsave(os.path.join(folder, f"{ind}.png"), image_np[0], cmap="gray", vmin=0.0, vmax=1.0)
    else:
        file_base = os.path.splitext(os.path.basename(case_name))[0]
        hwc = np.transpose(image_np, (1, 2, 0))
        np.save(os.path.join(out_path, f"{file_base}.npy"), hwc)
        quantized = np.rint(hwc * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(quantized, mode="RGB").save(os.path.join(out_path, f"{file_base}.png"))


def get_edge_map(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    emin = edge.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1)
    emax = edge.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
    return (edge - emin) / (emax - emin + 1e-6)


def run_attack(
    image_encoder,
    x_clean: torch.Tensor,
    profile: RankProfileBundle,
    args,
    norm_transform,
    is_synapse: bool,
) -> Tuple[np.ndarray, Dict]:
    selected_channels, selected_meta = resolve_profile_channels(image_encoder, profile)

    layer_weight = float(args.layer_weight if args.layer_weight is not None else profile.meta.get("layer_weight", 0.25))
    core_num_layers = int(args.core_num_layers if args.core_num_layers is not None else profile.meta.get("core_num_layers", 4))
    tau = float(args.tau_risk if args.tau_risk is not None else profile.tau_risk)

    max_val, min_val = float(x_clean.max()), float(x_clean.min())
    value_range = max(max_val - min_val, 1e-6)
    step_size = value_range * float(args.att_step_size) / 255.0
    epsilon = value_range * float(args.att_bound) / 255.0

    jitter = torch.randn_like(x_clean) * min(float(args.init_jitter), epsilon * 0.01)
    perturb_img = torch.clamp(x_clean + jitter, min_val, max_val).detach().requires_grad_(True)
    momentum = torch.zeros_like(x_clean)
    edge_map = get_edge_map(x_clean).detach() if args.edge_guidance else None

    cached_q = None
    cached_updates = None
    cached_risk = None
    cached_bound = None
    cached_diag = None

    trace = {
        "gain_source": [], "gain_robust": [], "objective": [],
        "risk": [], "bound": [], "gate": [], "grad_cos_source_vs_final": [],
    }

    for t in range(int(args.att_numstep)):
        refresh = cached_q is None or (t % max(1, int(args.q_refresh)) == 0)
        if refresh:
            q_gain, qdict = estimate_channel_q(
                image_encoder,
                x_clean,
                perturb_img.detach(),
                selected_channels,
                is_synapse,
                norm_transform,
                layer_weight,
                core_num_layers,
            )
            bound, risk, diag = risk_from_q(
                qdict,
                selected_meta,
                q_gain,
                profile.gain_floor,
                args.svd_oversample,
                args.svd_niter,
            )
            updates, _ = worst_updates_from_q(
                qdict,
                selected_meta,
                args.svd_oversample,
                args.svd_niter,
            )
            cached_q, cached_updates = qdict, updates
            cached_risk, cached_bound, cached_diag = risk, bound, diag

        risk = float(cached_risk)
        bound = float(cached_bound)
        use_robust = bool(args.disable_gate or risk > tau)

        # Source objective on the live attack graph.
        gain_source = compute_gain(
            image_encoder, x_clean, perturb_img, is_synapse, norm_transform,
            layer_weight, core_num_layers, injector=None,
        )

        if use_robust and float(args.robust_lambda) > 0:
            injector = LowRankInjector(selected_channels, cached_updates)
            gain_robust = compute_gain(
                image_encoder, x_clean, perturb_img, is_synapse, norm_transform,
                layer_weight, core_num_layers, injector=injector,
            )
            lam = float(args.robust_lambda)
            objective = (1.0 - lam) * gain_source + lam * gain_robust
        else:
            gain_robust = gain_source.detach()
            objective = gain_source

        grad = torch.autograd.grad(objective, perturb_img, retain_graph=False, create_graph=False)[0]
        # Optional diagnostic only; disabled by default to avoid an extra source
        # forward/backward on every gated iteration.
        if use_robust and args.diagnostic_cosine:
            xa_diag = perturb_img.detach().clone().requires_grad_(True)
            gs_diag = compute_gain(
                image_encoder, x_clean, xa_diag, is_synapse, norm_transform,
                layer_weight, core_num_layers, injector=None,
            )
            g_source_diag = torch.autograd.grad(gs_diag, xa_diag)[0].detach()
            grad_cos = cosine_flat(g_source_diag, grad.detach())
        else:
            grad_cos = 1.0 if not use_robust else float("nan")

        if edge_map is not None:
            grad = grad * (1.0 + float(args.edge_gamma) * edge_map)
        grad = normalize_l1(grad)
        momentum = float(args.momentum_decay) * momentum.detach() + grad.detach()
        eta = return_eta(momentum, step_size, args.step_rule).detach()

        new_img = perturb_img.detach() + eta
        total_delta = torch.clamp(new_img - x_clean, -epsilon, epsilon)
        new_img = torch.clamp(x_clean + total_delta, min_val, max_val)
        perturb_img = new_img.detach().requires_grad_(True)

        trace["gain_source"].append(float(gain_source.detach().item()))
        trace["gain_robust"].append(float(gain_robust.detach().item()))
        trace["objective"].append(float(objective.detach().item()))
        trace["risk"].append(risk)
        trace["bound"].append(bound)
        trace["gate"].append(bool(use_robust))
        trace["grad_cos_source_vs_final"].append(float(grad_cos))

    trace["tau_risk"] = tau
    trace["selected_channels"] = [x.key for x in profile.selected]
    trace["ranks"] = {x.key: x.rank for x in profile.selected}
    trace["weights"] = {x.key: x.weight for x in profile.selected}
    trace["rhos"] = {x.key: x.rho for x in profile.selected}
    return perturb_img[0].detach().cpu().numpy(), trace


def main():
    args = parse_args()
    set_seed(args.random_seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    profile_path = args.profile if os.path.isabs(args.profile) else os.path.join(ROOT, args.profile)
    profile = RankProfileBundle.load(profile_path)
    if not profile.gate1_pass:
        print("[warning] profile Gate-1 was not passed; run only as a diagnostic unless intentionally overridden.")

    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    image_encoder = sam.image_encoder.to(device).eval()
    for p in image_encoder.parameters():
        p.requires_grad_(False)

    if profile.meta.get("vit_name") and profile.meta["vit_name"] != args.vit_name:
        raise ValueError(f"profile vit_name={profile.meta['vit_name']} but attack vit_name={args.vit_name}")

    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    is_synapse = args.dataset == "Synapse"
    out_path = args.out_path if os.path.isabs(args.out_path) else os.path.join(ROOT, args.out_path)
    os.makedirs(out_path, exist_ok=True)
    trace_path = args.trace_path or os.path.join(out_path, "trace.jsonl")
    if not os.path.isabs(trace_path):
        trace_path = os.path.join(ROOT, trace_path)
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)

    with open(trace_path, "w", encoding="utf-8") as trace_file:
        if args.input_image:
            x = load_direct_input(args.input_image, args.img_size).to(device)
            adv, trace = run_attack(image_encoder, x, profile, args, norm_transform, is_synapse=False)
            name = os.path.basename(args.input_image)
            save_adv(adv, name, 0, False, out_path)
            trace_file.write(json.dumps({"case": name, "slice": 0, **trace}, ensure_ascii=False) + "\n")
            print(f"[online] done -> {out_path}")
            return

        loader, is_3d = load_dataset(args)
        selected_indices = None
        if args.indices is not None:
            selected_indices = {int(value) for value in args.indices.split(",") if value.strip()}
            if not selected_indices or min(selected_indices) < 0:
                raise ValueError(f"invalid dataset indices: {args.indices!r}")
        processed_cases = 0
        for dataset_index, batch in enumerate(
            tqdm(loader, desc=f"rank-robust [{args.dataset}/{args.vit_name}]")
        ):
            if selected_indices is not None and dataset_index > max(selected_indices):
                break
            if selected_indices is not None and dataset_index not in selected_indices:
                continue
            if args.max_cases >= 0 and processed_cases >= args.max_cases:
                break
            case_name = get_case_name(batch)
            if args.test_batch is not None and case_name != args.test_batch:
                continue
            processed_cases += 1
            slices = batch_to_slices(batch, args.dataset)
            n_slices = slices.shape[0] if args.max_slices < 0 else min(args.max_slices, slices.shape[0])
            for ind in range(n_slices):
                inp = slices[ind]
                x = torch.from_numpy(inp).unsqueeze(0).float().to(device)
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                if float(x.max()) > 1.1:
                    x = x / 255.0
                adv, trace = run_attack(image_encoder, x, profile, args, norm_transform, is_synapse)
                save_adv(adv, case_name, ind, is_3d, out_path)
                trace_file.write(json.dumps({"case": case_name, "slice": ind, **trace}, ensure_ascii=False) + "\n")
                trace_file.flush()
            torch.cuda.empty_cache()

    print(f"[online] done -> {out_path}")
    print(f"[online] trace -> {trace_path}")


if __name__ == "__main__":
    main()
