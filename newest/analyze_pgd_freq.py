#!/usr/bin/env python3
"""
Analyze PGD perturbation in frequency domain (A, Φ).
Measures: for each frequency, how much of PGD's perturbation goes to
amplitude vs phase change. This reveals what PGD implicitly does.

Theory prediction:
- Low freq: PGD should prefer AMPLITUDE perturbation (phase is expensive)
- High freq: PGD should prefer PHASE perturbation (phase is cheap, A is small)
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder
from exp_paid_optimal_attack import attack_pgd, build_freq_mask, pixel_to_fft, fft_to_pixel, load_synapse_slices

IMAGE_SIZE = 512
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
EPSILON = 16.0 / 255.0
STEP = 1.6 / 255.0
STEPS = 20

device = torch.device("cuda:0")

source_sam = load_sam_encoder("vit_b", SAM_CKPT, IMAGE_SIZE, device=device)
image_encoder = source_sam.image_encoder

images = load_synapse_slices(5)
print(f"Loaded {len(images)} Synapse images")

all_results = []

for idx, (img_data, lbl) in enumerate(images):
    x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
    rg = x.max().item() - x.min().item()
    eps_pix = EPSILON * rg
    step_pix = STEP * rg

    x_adv = attack_pgd(image_encoder, x, eps_pix, step_pix, STEPS)

    # Analyze perturbation in frequency domain
    A_clean, Phi_clean = pixel_to_fft(x)
    A_adv, Phi_adv = pixel_to_fft(x_adv)

    delta_A = (A_adv - A_clean).abs().squeeze().cpu().numpy()  # [C, H, W]
    delta_Phi = (Phi_adv - Phi_clean).abs().squeeze().cpu().numpy()

    A_spec = A_clean.squeeze().cpu().numpy()
    eta_pix = (x_adv - x).abs().squeeze().cpu().numpy()

    # Radial frequency bins
    H, W = delta_A.shape[-2:]
    y = np.arange(H) - H // 2
    x_coord = np.arange(W) - W // 2
    Y, X = np.meshgrid(y, x_coord)
    R = np.sqrt(Y**2 + X**2)
    max_r = min(H, W) // 2
    bins = np.linspace(0, max_r, 21)

    band_stats = []
    for i in range(len(bins) - 1):
        mask = (R >= bins[i]) & (R < bins[i+1])
        if mask.sum() == 0: continue
        band_stats.append({
            "freq_radius": float((bins[i] + bins[i+1]) / 2),
            "delta_A_mean": float(delta_A[:, mask].mean()),
            "delta_Phi_mean": float(delta_Phi[:, mask].mean()),
            "A_clean_mean": float(A_spec[:, mask].mean()),
            "delta_pixel_mean": float(eta_pix[:, mask].mean()),
        })

    # Also: per-frequency correlation between A and Δ effectiveness
    all_results.append({
        "image_idx": idx,
        "eps_pix": float(eps_pix),
        "eta_max": float(eta_pix.max()),
        "eta_mean": float(eta_pix.mean()),
        "band_stats": band_stats,
    })

    print(f"Image {idx}: eta_max={eta_pix.max():.4f}, eta_mean={eta_pix.mean():.4f}")
    print(f"  Low freq  (band 0-2): δA={band_stats[0]['delta_A_mean']:.6f}, δΦ={band_stats[0]['delta_Phi_mean']:.6f}, A={band_stats[0]['A_clean_mean']:.4f}")
    print(f"  Mid freq  (band 8-10): δA={band_stats[8]['delta_A_mean']:.6f}, δΦ={band_stats[8]['delta_Phi_mean']:.6f}, A={band_stats[8]['A_clean_mean']:.4f}")
    print(f"  High freq (band 18-20): δA={band_stats[-1]['delta_A_mean']:.6f}, δΦ={band_stats[-1]['delta_Phi_mean']:.6f}, A={band_stats[-1]['A_clean_mean']:.6f}")

# Save
out = os.path.join(NEWEST, "ablation_results", "pgd_freq_analysis.json")
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out}")

# Summary across images
print("\n" + "=" * 60)
print("CROSS-IMAGE SUMMARY")
print("=" * 60)
n_bands = len(all_results[0]["band_stats"])
for b in range(n_bands):
    da = np.mean([r["band_stats"][b]["delta_A_mean"] for r in all_results])
    dp = np.mean([r["band_stats"][b]["delta_Phi_mean"] for r in all_results])
    ac = np.mean([r["band_stats"][b]["A_clean_mean"] for r in all_results])
    fr = all_results[0]["band_stats"][b]["freq_radius"]
    ratio = dp / (da + 1e-10)
    print(f"  Band {b:2d} (r={fr:5.1f}): A_clean={ac:.4f}, δA={da:.6f}, δΦ={dp:.6f}, Φ/A ratio={ratio:.2f}")
