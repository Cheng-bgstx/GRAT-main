from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from theory_common import (
    SynapsePaths,
    build_support_mask,
    device_from_arg,
    extract_case_name,
    get_synapse_depth,
    get_synapse_loader,
    load_medsa_model,
    load_samed_encoder,
    load_source_sam,
    prepare_synapse_slice,
    setup_seed,
)


ROOT = Path("/root/data2/fyc/fyc/GRAT-main")
OUT_DIR = ROOT / "new/test_runs/codd_paper_figures"

BLUE = (54, 104, 164, 120)
ORANGE = (228, 132, 48, 72)
BLACK = (28, 26, 24, 255)
GRID = (224, 224, 224, 255)
BG = (255, 255, 255, 255)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit_name", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_ckpt", type=str, required=True)
    parser.add_argument("--samed_ckpt", type=str, required=True)
    parser.add_argument("--medsa_ckpt", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max_cases", type=int, default=3)
    parser.add_argument("--max_slices", type=int, default=4)
    parser.add_argument("--max_points_per_image", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


AXIS_FONT = load_font(22, bold=True)
LEGEND_FONT = load_font(18, bold=True)


def local_drift_points(
    feat_s: torch.Tensor,
    feat_t: torch.Tensor,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    gs = torch.fft.fft2(feat_s, norm="ortho")
    gt = torch.fft.fft2(feat_t, norm="ortho")
    support = build_support_mask(gs, min_radius_ratio=0.05, max_radius_ratio=1.0, energy_quantile=0.9)
    delta = gt - gs
    carrier = torch.exp(-1j * torch.angle(gs))
    local = delta * carrier
    amp_s = torch.abs(gs)
    alpha = torch.real(local)
    beta = torch.imag(local)

    alpha = alpha[support]
    beta = beta[support]
    amp = amp_s[support]
    if alpha.numel() == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # Normalize by source spectral magnitude so the aligned drift is comparable across coefficients.
    alpha = alpha / (amp + 1e-8)
    beta = beta / (amp + 1e-8)

    num = min(max_points, alpha.numel())
    if alpha.numel() > num:
        idx = torch.randperm(alpha.numel(), device=alpha.device)[:num]
        alpha = alpha[idx]
        beta = beta[idx]
    return alpha.detach().cpu().numpy(), beta.detach().cpu().numpy()


def collect_points(args):
    device = device_from_arg(args.device)
    source_sam = load_source_sam(args.vit_name, args.sam_ckpt, args.image_size, args.num_classes, device)
    _, samed_encoder = load_samed_encoder(
        vit_name=args.vit_name,
        sam_ckpt=args.sam_ckpt,
        lora_ckpt=args.samed_ckpt,
        image_size=args.image_size,
        num_classes=args.num_classes,
        rank=args.rank,
        device=device,
    )
    medsa_model = load_medsa_model(
        vit_name=args.vit_name,
        sam_ckpt=args.sam_ckpt,
        msa_ckpt=args.medsa_ckpt,
        image_size=args.image_size,
        multimask_output=args.num_classes + 1,
        device=device,
    )

    loader = get_synapse_loader(SynapsePaths().volume_path, SynapsePaths().list_dir)
    samed_pts = []
    medsa_pts = []

    with torch.no_grad():
        for case_idx, sampled_batch in enumerate(loader):
            if case_idx >= args.max_cases:
                break
            _ = extract_case_name(sampled_batch)
            depth = get_synapse_depth(sampled_batch)
            slice_ids = np.linspace(0, depth - 1, num=min(args.max_slices, depth), dtype=int).tolist()
            for slice_idx in slice_ids:
                image_t, _ = prepare_synapse_slice(sampled_batch, slice_idx, device)
                feat_s, _ = source_sam.image_encoder(image_t)
                feat_samed, _ = samed_encoder(image_t)
                medsa_out = medsa_model.image_encoder(image_t)
                feat_medsa = medsa_out[0] if isinstance(medsa_out, (list, tuple)) else medsa_out

                samed_pts.append(local_drift_points(feat_s, feat_samed, args.max_points_per_image))
                medsa_pts.append(local_drift_points(feat_s, feat_medsa, args.max_points_per_image))

    samed_alpha = np.concatenate([p[0] for p in samed_pts if len(p[0])], axis=0)
    samed_beta = np.concatenate([p[1] for p in samed_pts if len(p[1])], axis=0)
    medsa_alpha = np.concatenate([p[0] for p in medsa_pts if len(p[0])], axis=0)
    medsa_beta = np.concatenate([p[1] for p in medsa_pts if len(p[1])], axis=0)
    return (samed_alpha, samed_beta), (medsa_alpha, medsa_beta)


def draw_scatter(samed, medsa, out_path: Path):
    width, height = 1300, 900
    img = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(img, "RGBA")

    left, top, right, bottom = 120, 85, 1185, 760
    cx = (left + right) // 2
    cy = (top + bottom) // 2

    all_alpha = np.concatenate([samed[0], medsa[0]])
    all_beta = np.concatenate([samed[1], medsa[1]])
    max_abs = float(max(np.max(np.abs(all_alpha)), np.max(np.abs(all_beta)), 1e-3))
    limit = max_abs * 1.08

    for frac in [0.2, 0.4, 0.6, 0.8]:
        x = int(left + (right - left) * frac)
        y = int(top + (bottom - top) * frac)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.line((left, y, right, y), fill=GRID, width=1)

    draw.line((left, cy, right, cy), fill=BLACK, width=3)
    draw.line((cx, bottom, cx, top), fill=BLACK, width=3)
    draw.polygon([(right, cy), (right - 18, cy - 8), (right - 18, cy + 8)], fill=BLACK)
    draw.polygon([(cx, top), (cx - 8, top + 18), (cx + 8, top + 18)], fill=BLACK)

    def to_px(alpha: np.ndarray, beta: np.ndarray):
        x = cx + (alpha / limit) * ((right - left) / 2.2)
        y = cy - (beta / limit) * ((bottom - top) / 2.2)
        return x, y

    def paint(points, color, radius):
        x, y = to_px(points[0], points[1])
        for px, py in zip(x, y):
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    def draw_cov_ellipse(points, color, scale=2.0):
        alpha, beta = points
        pts = np.stack([alpha, beta], axis=1)
        if pts.shape[0] < 3:
            return
        mean = pts.mean(axis=0)
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        radii = scale * np.sqrt(np.maximum(eigvals, 1e-12))
        theta = np.linspace(0, 2 * math.pi, 240)
        circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
        ellipse = (eigvecs @ (radii[:, None] * circle)).T + mean[None, :]
        ex, ey = to_px(ellipse[:, 0], ellipse[:, 1])
        poly = [(float(x), float(y)) for x, y in zip(ex, ey)]
        draw.line(poly + [poly[0]], fill=color, width=5)

    # Paint Med-SA first and SAMed second so the blue cloud stays visible.
    paint(medsa, ORANGE, radius=3)
    paint(samed, BLUE, radius=3)

    # Draw fitted covariance ellipses last, with stronger contrast.
    draw_cov_ellipse(medsa, (228, 132, 48, 255))
    draw_cov_ellipse(samed, (30, 84, 150, 255))

    bbox = draw.textbbox((0, 0), "Carrier-aligned Radial Drift  α", font=AXIS_FONT)
    tw = bbox[2] - bbox[0]
    draw.text(((left + right - tw) / 2, 806), "Carrier-aligned Radial Drift  α", fill=BLACK, font=AXIS_FONT)

    y_label = "Tangential Drift  β"
    bbox = draw.textbbox((0, 0), y_label, font=AXIS_FONT)
    label_w = bbox[2] - bbox[0]
    tmp = Image.new("RGBA", (label_w + 20, 56), (255, 255, 255, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    tmp_draw.text((10, 10), y_label, fill=BLACK, font=AXIS_FONT)
    tmp = tmp.rotate(90, expand=True)
    img.alpha_composite(tmp, (20, int((top + bottom) / 2 - tmp.size[1] / 2)))

    legend_x0, legend_y0 = 842, 115
    legend_x1, legend_y1 = 1115, 214
    draw.rounded_rectangle((legend_x0, legend_y0, legend_x1, legend_y1), radius=18, fill=(255, 255, 255, 245), outline=GRID, width=2)
    draw.ellipse((862, 136, 880, 154), fill=BLUE)
    draw.text((900, 130), "SAMed drift cloud", fill=BLACK, font=LEGEND_FONT)
    draw.ellipse((862, 172, 880, 190), fill=ORANGE)
    draw.text((900, 166), "Med-SA drift cloud", fill=BLACK, font=LEGEND_FONT)

    img.convert("RGB").save(out_path)


def main():
    args = parse_args()
    setup_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samed, medsa = collect_points(args)
    out_path = OUT_DIR / "figure2_empirical_real_feature_drift.png"
    draw_scatter(samed, medsa, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
