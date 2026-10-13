#!/usr/bin/env python3
"""
Verify spectral bias: ΔA vs ΔΦ between frozen SAM ViT-L (f_s) and finetuned f_t (SAMed LoRA / MSA).
Hooks align with root SAM feature_layer: ViT Block output is post-residual (same as
segment_anything ImageEncoderViT collecting x after each blk).

Does not modify Sam_attack_SE-CGR.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

# PS 后端在此环境可稳定写 EPS；Agg 的 PNG 路径会触发 _renderer.draw_path 错误。
matplotlib.use("PS")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.cm as mpl_cm
from matplotlib import colors as mcolors

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.spectral_hypothesis.lora_analysis import (
    analyze_msa_adapters,
    analyze_samed_lora,
    collect_lora_singular_values,
    plot_lora_singular_histograms,
    random_baseline_lora_fro,
    random_baseline_singular_values,
)
from experiments.spectral_hypothesis.spectral_metrics import (
    aggregate_channels,
    delta_spectrum_masked,
    fft2_amplitude_phase,
    paired_wilcoxon_greater,
    process_pair_features,
    radial_profile,
    radial_triplet_neck,
)


def save_spectral_heatmaps(
    neck_s: torch.Tensor,
    neck_t: torch.Tensor,
    mask_tau: float,
    use_log_amp: bool,
    out_dir: str,
    prefix: str = "neck",
) -> None:
    """Appendix: 2D |ΔA| and |ΔΦ| (first batch item)."""
    fs = aggregate_channels(neck_s, "mean")
    ft = aggregate_channels(neck_t, "mean")
    if fs.shape[-2:] != ft.shape[-2:]:
        ft = F.interpolate(ft, size=fs.shape[-2:], mode="bilinear", align_corners=False)
    g_s, _, _ = fft2_amplitude_phase(fs)
    g_t, _, _ = fft2_amplitude_phase(ft)
    d_a, d_phi, mask = delta_spectrum_masked(g_s, g_t, mask_tau=mask_tau, use_log_amp=use_log_amp)
    da = d_a[0, 0].detach().cpu().numpy()
    dp = d_phi[0, 0].detach().cpu().numpy()
    mk = mask[0, 0].detach().cpu().numpy()
    da = np.where(mk, da, np.nan)
    dp = np.where(mk, dp, np.nan)
    for name, arr in [("deltaA", da), ("deltaPhi", dp)]:
        # 不用 imshow：部分 matplotlib 后端在 quad_mesh 上会触发 __array__ 错误；直接写 PNG。
        arr_f = np.asarray(arr, dtype=np.float64)
        finite = np.isfinite(arr_f)
        if finite.any():
            vmin, vmax = float(np.nanmin(arr_f)), float(np.nanmax(arr_f))
            if vmin == vmax:
                vmax = vmin + 1e-8
        else:
            vmin, vmax = 0.0, 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        rgba = mpl_cm.magma(norm(np.nan_to_num(arr_f, nan=vmin)))
        rgb = (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"{prefix}_{name}_heatmap.png"))


def sam_preprocess_pad(x: torch.Tensor, img_size: int, pixel_mean, pixel_std) -> torch.Tensor:
    """x [B,3,H,W] in [0,1], match segment_anything Sam.preprocess."""
    mean = torch.tensor(pixel_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(pixel_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    return F.pad(x, (0, padw, 0, padh))


def load_fs_sam(ckpt: str, img_size: int, num_classes: int, device: torch.device):
    from segment_anything import sam_model_registry

    sam, _ = sam_model_registry["vit_l"](
        image_size=img_size,
        num_classes=num_classes,
        checkpoint=ckpt,
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )
    return sam.to(device).eval()


def load_samed_lora(
    ckpt: str,
    lora_ckpt: Optional[str],
    img_size: int,
    num_classes: int,
    rank: int,
    device: torch.device,
):
    samed = os.path.join(REPO_ROOT, "SAMed_h")
    if samed not in sys.path:
        sys.path.insert(0, samed)
    from sam_lora_image_encoder import LoRA_Sam
    from segment_anything import sam_model_registry

    sam, _ = sam_model_registry["vit_l"](
        image_size=img_size,
        num_classes=num_classes,
        checkpoint=ckpt,
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )
    net = LoRA_Sam(sam, r=rank)
    if lora_ckpt and os.path.isfile(lora_ckpt):
        net.load_lora_parameters(lora_ckpt)
    return net.to(device).eval()


def load_msa(
    sam_ckpt: str,
    finetune_ckpt: Optional[str],
    img_size: int,
    multimask_output: int,
    dataset: str,
    device: torch.device,
):
    msa = os.path.join(REPO_ROOT, "Medical-SAM-Adapter-main")
    if msa not in sys.path:
        sys.path.insert(0, msa)
    from models.sam.build_sam import sam_model_registry

    args = SimpleNamespace(
        image_size=img_size,
        multimask_output=multimask_output,
        dataset=dataset,
        mod="sam_adpt",
        mid_dim=None,
        thd=False,
        chunk=None,
    )
    net = sam_model_registry["vit_l"](args, checkpoint=sam_ckpt)
    if finetune_ckpt and os.path.isfile(finetune_ckpt):
        sd = torch.load(finetune_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        net.load_state_dict(sd, strict=False)
    return net.to(device).eval()


def register_block_neck_hooks(
    image_encoder: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], List]:
    """
    Capture each ViT Block module output as [B,C,H,W] after permute.
    This matches root SAM ImageEncoderViT: tensor after full Block (attn+MLP residuals),
    same index as feature_layer[i] in segment_anything/modeling/image_encoder.py.
    (Not the tensor between norm1 and attn — use block output for cross-repo alignment.)
    Capture neck output [B,256,H,W].
    """
    store: Dict[str, torch.Tensor] = {}
    handles = []

    def permute_out(name):
        def hook(_m, _inp, out):
            # out: [B,H,W,C]
            store[name] = out.permute(0, 3, 1, 2).contiguous()

        return hook

    for i, blk in enumerate(image_encoder.blocks):
        handles.append(blk.register_forward_hook(permute_out(f"block_{i}")))

    def neck_hook(_m, _inp, out):
        store["neck"] = out

    if hasattr(image_encoder, "neck"):
        handles.append(image_encoder.neck.register_forward_hook(neck_hook))
    return store, handles


def forward_fs(sam, x_in: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Root SAM image_encoder returns (neck, feature_layer list)."""
    neck, flist = sam.image_encoder(x_in)
    return neck, flist


def forward_ft_lora(net, x_in: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], List]:
    store, handles = register_block_neck_hooks(net.sam.image_encoder)
    with torch.no_grad():
        _ = net.sam.image_encoder(x_in)
    for h in handles:
        h.remove()
    neck = store.get("neck")
    depth = len(net.sam.image_encoder.blocks)
    flist = [store[f"block_{i}"] for i in range(depth)]
    return neck, flist


def forward_ft_msa(net, x_prep: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], List]:
    """Use same preprocessed tensor as f_s (avoid net.preprocess distribution shift)."""
    store, handles = register_block_neck_hooks(net.image_encoder)
    with torch.no_grad():
        _ = net.image_encoder(x_prep)
    for h in handles:
        h.remove()
    neck = store.get("neck")
    depth = len(net.image_encoder.blocks)
    flist = [store[f"block_{i}"] for i in range(depth)]
    return neck, flist


def synthetic_batch(n: int, img_size: int, device) -> torch.Tensor:
    return torch.rand(n, 3, img_size, img_size, device=device)


def load_images_from_dir(folder: str, n: int, img_size: int, device) -> torch.Tensor:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
        if len(paths) >= n:
            break
    paths = sorted(paths)[:n]
    if not paths:
        raise FileNotFoundError(f"No images under {folder}")
    tfm = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    )
    tensors = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        tensors.append(tfm(im))
    return torch.stack(tensors, dim=0).to(device)


def preprocess_pixel_params(mode: str) -> Tuple[List[float], List[float]]:
    if mode == "synapse":
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    if mode == "imagenet":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    raise ValueError(mode)


def run_spectral_compare(
    sam_fs,
    ft_model,
    target: str,
    x_batch: torch.Tensor,
    img_size: int,
    block_indices: List[int],
    mask_tau: float,
    use_log_amp: bool,
    pixel_mean: List[float],
    pixel_std: List[float],
) -> Dict[str, Any]:
    """One batch; same x_in for f_s and f_t."""
    x_in = sam_preprocess_pad(x_batch, img_size, pixel_mean, pixel_std)
    with torch.no_grad():
        neck_s, flist_s = forward_fs(sam_fs, x_in)
    if target == "samed":
        neck_t, flist_t = forward_ft_lora(ft_model, x_in)
    else:
        neck_t, flist_t = forward_ft_msa(ft_model, x_batch)

    layers = [("neck", neck_s, neck_t)]
    for bi in block_indices:
        if bi < len(flist_s) and bi < len(flist_t):
            layers.append((f"block_{bi}", flist_s[bi], flist_t[bi]))

    rows = []
    for name, fs, ft in layers:
        # align spatial size (MSA pos embed interpolation may differ slightly — interpolate ft)
        if fs.shape[-2:] != ft.shape[-2:]:
            ft = F.interpolate(ft, size=fs.shape[-2:], mode="bilinear", align_corners=False)
        b = fs.size(0)
        for ib in range(b):
            st = process_pair_features(
                fs[ib : ib + 1], ft[ib : ib + 1], mask_tau=mask_tau, use_log_amp=use_log_amp
            )
            st["layer"] = name
            st["img_idx"] = ib
            rows.append(st)
    return {"rows": rows, "neck_s": neck_s, "neck_t": neck_t}


def plot_radial_shaded(
    rs_stack: np.ndarray,
    rt_stack: np.ndarray,
    da_stack: np.ndarray,
    out_path: str,
) -> None:
    """rs_stack [N, n_bins] — mean ± std across images."""
    x = np.arange(rs_stack.shape[1])
    mean_rs, std_rs = rs_stack.mean(0), rs_stack.std(0)
    mean_rt, std_rt = rt_stack.mean(0), rt_stack.std(0)
    mean_da, std_da = da_stack.mean(0), da_stack.std(0)
    plt.figure(figsize=(7, 4))
    plt.fill_between(x, mean_rs - std_rs, mean_rs + std_rs, alpha=0.25)
    plt.plot(x, mean_rs, label="|G| f_s")
    plt.fill_between(x, mean_rt - std_rt, mean_rt + std_rt, alpha=0.25)
    plt.plot(x, mean_rt, label="|G| f_t")
    plt.fill_between(x, mean_da - std_da, mean_da + std_da, alpha=0.25)
    plt.plot(x, mean_da, label="|ΔA|")
    plt.xlabel("Radial bin")
    plt.ylabel("Mean amplitude")
    plt.legend(loc="upper right")
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_radial(neck_s: torch.Tensor, neck_t: torch.Tensor, out_path: str):
    fs = aggregate_channels(neck_s, "mean")
    ft = aggregate_channels(neck_t, "mean")
    _, a_s, _ = fft2_amplitude_phase(fs)
    _, a_t, _ = fft2_amplitude_phase(ft)
    n_bins = 32
    rs = radial_profile(a_s, n_bins)
    rt = radial_profile(a_t, n_bins)
    da = radial_profile((a_t - a_s).abs(), n_bins)
    x = np.arange(n_bins)
    plt.figure(figsize=(6, 4))
    plt.plot(x, rs, label="|G| fs")
    plt.plot(x, rt, label="|G| ft")
    plt.plot(x, da, label="|ΔA|")
    plt.xlabel("Radial bin")
    plt.ylabel("Mean amplitude")
    plt.legend(loc="upper right")
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_variance_bars(summary: List[Dict], out_path: str):
    layers = [s["layer"] for s in summary]
    va = [s["var_delta_a_mean"] for s in summary]
    vp = [s["circ_var_delta_phi_mean"] for s in summary]
    ea = [s.get("var_delta_a_std", 0.0) for s in summary]
    ep = [s.get("circ_var_delta_phi_std", 0.0) for s in summary]
    x = np.arange(len(layers))
    w = 0.35
    plt.figure(figsize=(max(8, len(layers)), 4))
    plt.bar(x - w / 2, va, w, yerr=ea, capsize=2, label="Var(ΔA)")
    plt.bar(x + w / 2, vp, w, yerr=ep, capsize=2, label="CircVar(ΔΦ)")
    plt.xticks(x, layers, rotation=25, ha="right")
    plt.legend(loc="upper right")
    plt.ylabel("Metric")
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_lora_fro(data: Dict[str, List[float]], out_path: str):
    idx = data["layer_idx"]
    plt.figure(figsize=(8, 4))
    plt.plot(idx, data["fro_q"], marker="o", label="Frobenius ΔW_q")
    plt.plot(idx, data["fro_v"], marker="s", label="Frobenius ΔW_v")
    plt.xlabel("LoRA layer index")
    plt.ylabel("Frobenius norm")
    plt.legend(loc="upper right")
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_msa_adapters(data: Dict[str, List[float]], out_path: str):
    idx = data["block_idx"]
    plt.figure(figsize=(8, 4))
    plt.plot(idx, data["space_adapter_fro"], label="Space_Adapter")
    plt.plot(idx, data["mlp_adapter_fro"], label="MLP_Adapter")
    plt.plot(idx, data["depth_adapter_fro"], label="Depth_Adapter")
    plt.xlabel("Block index")
    plt.ylabel("||W2 W1||_F")
    plt.legend(loc="upper right")
    plt.savefig(out_path, dpi=150)
    plt.close()


def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    from collections import defaultdict

    g = defaultdict(list)
    for r in rows:
        g[r["layer"]].append(r)
    out = []
    for layer, lst in g.items():
        va = [x["var_delta_a"] for x in lst if not np.isnan(x["var_delta_a"])]
        vp = [x["circ_var_delta_phi"] for x in lst if not np.isnan(x["circ_var_delta_phi"])]
        out.append(
            {
                "layer": layer,
                "var_delta_a_mean": float(np.mean(va)) if va else float("nan"),
                "var_delta_a_std": float(np.std(va)) if len(va) > 1 else 0.0,
                "circ_var_delta_phi_mean": float(np.mean(vp)) if vp else float("nan"),
                "circ_var_delta_phi_std": float(np.std(vp)) if len(vp) > 1 else 0.0,
                "n": len(lst),
            }
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["samed", "msa"], default="samed")
    p.add_argument("--ckpt_fs", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "sam_vit_l_0b3195.pth"))
    p.add_argument("--lora_ckpt", type=str, default="", help="SAMed LoRA+decoder .pth")
    p.add_argument("--msa_ckpt", type=str, default="", help="MSA finetuned full checkpoint (optional)")
    p.add_argument("--sam_ckpt_msa", type=str, default="", help="Base SAM ckpt for MSA build")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--num_classes", type=int, default=9)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--multimask_output", type=int, default=9)
    p.add_argument("--dataset_tag", type=str, default="Synapse", help="For MSA args.dataset")
    p.add_argument("--num_images", type=int, default=32)
    p.add_argument("--data_path", type=str, default="", help="Folder with images; empty = synthetic")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--block_indices", type=str, default="0,5,11,17,23")
    p.add_argument("--mask_tau", type=float, default=1e-3)
    p.add_argument("--tau_sweep", type=str, default="1e-4,1e-3,1e-2")
    p.add_argument("--use_log_amp", action="store_true")
    p.add_argument(
        "--preprocess",
        choices=["synapse", "imagenet"],
        default="synapse",
        help="Input normalization: synapse [0,1] with mean=0 std=1 (matches Sam_attack); "
        "imagenet uses ImageNet mean/std on [0,1] RGB.",
    )
    p.add_argument("--out_dir", type=str, default=os.path.join(REPO_ROOT, "experiments", "spectral_hypothesis", "outputs"))
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    block_indices = [int(x) for x in args.block_indices.split(",") if x.strip()]
    pixel_mean, pixel_std = preprocess_pixel_params(args.preprocess)

    if not os.path.isfile(args.ckpt_fs):
        raise FileNotFoundError(f"--ckpt_fs not found: {args.ckpt_fs}")
    sam_fs = load_fs_sam(args.ckpt_fs, args.img_size, args.num_classes, device)
    if args.target == "samed":
        ft = load_samed_lora(args.ckpt_fs, args.lora_ckpt or None, args.img_size, args.num_classes, args.rank, device)
    else:
        sc = args.sam_ckpt_msa or args.ckpt_fs
        if not os.path.isfile(sc):
            raise FileNotFoundError(f"MSA base SAM ckpt not found: {sc}")
        ft = load_msa(sc, args.msa_ckpt or None, args.img_size, args.multimask_output, args.dataset_tag, device)
    sam_fs.eval()
    ft.eval()

    # Data
    if args.data_path and os.path.isdir(args.data_path):
        x_all = load_images_from_dir(args.data_path, args.num_images, args.img_size, device)
    else:
        x_all = synthetic_batch(args.num_images, args.img_size, device)

    all_rows: List[Dict] = []
    radial_rs: List[np.ndarray] = []
    radial_rt: List[np.ndarray] = []
    radial_da: List[np.ndarray] = []
    n = x_all.size(0)
    for start in range(0, n, args.batch_size):
        xb = x_all[start : start + args.batch_size]
        res = run_spectral_compare(
            sam_fs,
            ft,
            args.target,
            xb,
            args.img_size,
            block_indices,
            args.mask_tau,
            args.use_log_amp,
            pixel_mean,
            pixel_std,
        )
        all_rows.extend(res["rows"])
        neck_s, neck_t = res["neck_s"], res["neck_t"]
        for ib in range(neck_s.size(0)):
            rs, rt, da = radial_triplet_neck(neck_s[ib : ib + 1], neck_t[ib : ib + 1])
            radial_rs.append(rs)
            radial_rt.append(rt)
            radial_da.append(da)
        if start == 0:
            plot_radial(neck_s, neck_t, os.path.join(args.out_dir, "radial_neck_firstbatch.eps"))
            save_spectral_heatmaps(
                neck_s,
                neck_t,
                args.mask_tau,
                args.use_log_amp,
                args.out_dir,
                prefix="neck",
            )
    if radial_rs:
        rs_stack = np.stack(radial_rs, axis=0)
        rt_stack = np.stack(radial_rt, axis=0)
        da_stack = np.stack(radial_da, axis=0)
        plot_radial_shaded(
            rs_stack,
            rt_stack,
            da_stack,
            os.path.join(args.out_dir, "radial_neck_shaded.eps"),
        )

    summary = aggregate_rows(all_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if all_rows:
        with open(os.path.join(args.out_dir, "per_image_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    else:
        print("Warning: no metric rows (empty batch or failure).")

    if summary:
        plot_variance_bars(summary, os.path.join(args.out_dir, "var_deltaA_vs_circvar_phi.eps"))

    # Wilcoxon: paired per-image var_delta_a vs circ_var per layer
    from collections import defaultdict

    tests = []
    if all_rows:
        by_layer = defaultdict(lambda: {"va": [], "vp": []})
        for r in all_rows:
            by_layer[r["layer"]]["va"].append(r["var_delta_a"])
            by_layer[r["layer"]]["vp"].append(r["circ_var_delta_phi"])
        for layer, d in by_layer.items():
            va = np.array(d["va"], dtype=np.float64)
            vp = np.array(d["vp"], dtype=np.float64)
            stat, pval = paired_wilcoxon_greater(va, vp)
            tests.append({"layer": layer, "wilcoxon_stat": stat, "p_value": pval})
    with open(os.path.join(args.out_dir, "wilcoxon.json"), "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2)

    # tau sensitivity
    tau_list = [float(x) for x in args.tau_sweep.split(",") if x.strip()]
    sens = []
    for tau in tau_list:
        rows_tau = []
        for start in range(0, n, args.batch_size):
            xb = x_all[start : start + args.batch_size]
            res = run_spectral_compare(
                sam_fs,
                ft,
                args.target,
                xb,
                args.img_size,
                block_indices,
                tau,
                args.use_log_amp,
                pixel_mean,
                pixel_std,
            )
            rows_tau.extend(res["rows"])
        sens.append({"tau": tau, "summary": aggregate_rows(rows_tau)})
    with open(os.path.join(args.out_dir, "tau_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(sens, f, indent=2)

    # LoRA / Adapter weight analysis
    if args.target == "samed" and hasattr(ft, "w_As"):
        ldata = analyze_samed_lora(ft)
        with open(os.path.join(args.out_dir, "lora_svd.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in ldata.items()}, f, indent=2)
        plot_lora_fro(ldata, os.path.join(args.out_dir, "lora_frobenius.eps"))
        dim = int(ft.w_As[0].in_features)
        rand_fro = random_baseline_lora_fro(dim, args.rank, len(ldata["layer_idx"]))
        with open(os.path.join(args.out_dir, "lora_random_baseline.json"), "w", encoding="utf-8") as f:
            json.dump({"random_fro": rand_fro}, f, indent=2)
        sq, sv = collect_lora_singular_values(ft)
        s_rand = random_baseline_singular_values(dim, args.rank, len(ft.w_As))
        plot_lora_singular_histograms(
            sq,
            sv,
            s_rand,
            os.path.join(args.out_dir, "lora_singular_histogram.eps"),
        )
    if args.target == "msa":
        adata = analyze_msa_adapters(ft)
        with open(os.path.join(args.out_dir, "msa_adapter_norms.json"), "w", encoding="utf-8") as f:
            json.dump(adata, f, indent=2)
        plot_msa_adapters(adata, os.path.join(args.out_dir, "msa_adapter_fro.eps"))

    cfg = vars(args).copy()
    cfg["note"] = (
        "Block hooks: ViT Block forward output after full residual (matches root SAM feature_layer). "
        "Preprocess: same pixel_mean/std for f_s and f_t; MSA uses same x_in as SAM (no separate net.preprocess). "
        f"Mode={args.preprocess}."
    )
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("Done. Outputs in", args.out_dir)


if __name__ == "__main__":
    main()
