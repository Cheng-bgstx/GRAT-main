#!/usr/bin/env python3
"""
PAID-Prescribed Decoupled Attack (PPA):
  Decouple A and Φ → apply PAID-derived prescriptions → recombine → ε-ball project.
  NO GRADIENTS. Pure decoupled perturbation based on PAID analysis.

PAID prescriptions:
  - Φ_high: randomize (PAID: I_Φ dominates, Φ_high is information-rich)
  - A: flatten toward geometric mean (PAID: amplitude perturbation most destructive, Δ=+0.66)
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model
from exp_paid_optimal_attack import (
    pixel_to_fft, fft_to_pixel, build_freq_mask,
    load_synapse_slices, evaluate, attack_pgd, attack_fd_pgd,
)

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
CKPT_REL = "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"
EPSILON = 16.0 / 255.0
STEP = 1.6 / 255.0
STEPS = 20

device = torch.device("cuda:0")

net = load_samed_model("vit_b", SAM_CKPT, os.path.join(CKPT_BASE, CKPT_REL),
                       IMAGE_SIZE, 8, device=device).to(device)
source_sam = load_sam_encoder("vit_b", SAM_CKPT, IMAGE_SIZE, device=device)
image_encoder = source_sam.image_encoder
images = load_synapse_slices(5)
print(f"Loaded {len(images)} images")


def attack_decoupled(x, eps_pix, cutoff=0.25, a_strength=0.5, phi_strength=1.0):
    """
    PAID-Prescribed Decoupled Attack (NO GRADIENTS).

    Φ prescription: randomize high-freq phase (PAID: Φ_high has information)
    A prescription: flatten amplitude spectrum (PAID: A perturbation most destructive)

    a_strength: how much to flatten amplitude (0=none, 1=fully flat)
    phi_strength: scaling factor for phase randomization
    """
    max_val, min_val = x.max(), x.min()
    B, C, H, W = x.shape

    A, Phi = pixel_to_fft(x)
    mask_high = build_freq_mask(H, W, cutoff, x.device, 'high').unsqueeze(0).unsqueeze(0)
    mask_low = build_freq_mask(H, W, cutoff, x.device, 'low').unsqueeze(0).unsqueeze(0)

    # --- Amplitude prescription: flatten toward geometric mean ---
    # Compute geometric mean amplitude per channel per frequency band
    # Actually: compute a smoothed (low-pass) version as target
    A_smooth = torch.zeros_like(A)
    for b in range(B):
        for c in range(C):
            # Low-pass filter the amplitude spectrum
            A_lp = A[b, c] * mask_low[0, 0]
            # IFFT to spatial, then FFT back (this gives smooth envelope)
            A_spatial = torch.fft.ifft2(A_lp, norm='ortho').real
            A_smooth_full = torch.abs(torch.fft.fft2(A_spatial, norm='ortho'))
            A_smooth[b, c] = A_smooth_full

    # Target: blend original and smoothed
    A_target = (1 - a_strength) * A + a_strength * A_smooth

    # --- Phase prescription: randomize high frequencies ---
    Phi_random = torch.rand_like(Phi) * 2 * np.pi - np.pi  # uniform [-π, π]
    Phi_target = Phi.clone()
    Phi_target = Phi_target + phi_strength * (Phi_random - Phi) * mask_high

    # --- Reconstruct ---
    x_adv = fft_to_pixel(A_target, Phi_target)

    # --- Project to ε-ball ---
    eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
    x_adv = torch.clamp(x + eta, min_val, max_val)

    return x_adv.detach()


def attack_decoupled_iter(x, eps_pix, cutoff=0.25, a_strength=0.3, phi_strength=0.5, steps=5):
    """
    Iterative version: apply decoupled perturbation in small steps,
    re-projecting to ε-ball after each step.
    """
    max_val, min_val = x.max(), x.min()
    B, C, H, W = x.shape

    A0, Phi0 = pixel_to_fft(x)
    mask_high = build_freq_mask(H, W, cutoff, x.device, 'high').unsqueeze(0).unsqueeze(0)
    mask_low = build_freq_mask(H, W, cutoff, x.device, 'low').unsqueeze(0).unsqueeze(0)

    # Target amplitude: smoothed version
    A_smooth = torch.zeros_like(A0)
    for b in range(B):
        for c in range(C):
            A_lp = A0[b, c] * mask_low[0, 0]
            A_spatial = torch.fft.ifft2(A_lp, norm='ortho').real
            A_smooth_full = torch.abs(torch.fft.fft2(A_spatial, norm='ortho'))
            A_smooth[b, c] = A_smooth_full
    A_target_full = (1 - a_strength) * A0 + a_strength * A_smooth

    # Target phase: random at high freq
    Phi_random = torch.rand_like(Phi0) * 2 * np.pi - np.pi
    Phi_target_full = Phi0 + phi_strength * (Phi_random - Phi0) * mask_high

    x_adv = x.clone()
    for step in range(steps):
        A_curr, Phi_curr = pixel_to_fft(x_adv)

        # Interpolate toward target
        alpha = 1.0 / (steps - step)  # increasing fraction
        A_new = (1 - alpha) * A_curr + alpha * A_target_full
        Phi_new = (1 - alpha) * Phi_curr + alpha * Phi_target_full

        x_new = fft_to_pixel(A_new, Phi_new)
        eta = torch.clamp(x_new - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val)

    return x_adv.detach()


# Sweep over prescription strengths
configs = [
    ("PGD (reference)", None),
    ("FD-PGD (reference)", None),
    ("A-flat=0.5, Φ-rand=1.0 (1-step)", 0.5, 1.0, 1),
    ("A-flat=0.7, Φ-rand=1.0 (1-step)", 0.7, 1.0, 1),
    ("A-flat=0.5, Φ-rand=2.0 (1-step)", 0.5, 2.0, 1),
    ("A-flat=0.3, Φ-rand=0.5 (iter=5)", 0.3, 0.5, 5),
    ("A-flat=0.5, Φ-rand=0.5 (iter=5)", 0.5, 0.5, 5),
    ("A-flat=0.3, Φ-rand=1.0 (iter=10)", 0.3, 1.0, 10),
    ("A-flat=0.5, Φ-rand=2.0 (iter=10)", 0.5, 2.0, 10),
]

print(f"\n{'Config':<40} {'Δ Dice':>12} {'Time':>8}")
print("-" * 60)

for cfg in configs:
    name = cfg[0]
    deltas = []
    t0 = time.time()

    for img_data, lbl in images:
        x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        clean_d = evaluate(net, img_data, lbl, "Synapse", device)
        if clean_d < 0.01: continue
        rg = x.max().item() - x.min().item()
        eps_pix = EPSILON * rg
        step_pix = STEP * rg

        if name == "PGD (reference)":
            x_adv = attack_pgd(image_encoder, x, eps_pix, step_pix, STEPS)
        elif name == "FD-PGD (reference)":
            x_adv = attack_fd_pgd(image_encoder, x, eps_pix, step_pix, STEPS)
        elif len(cfg) == 4:
            a_s, p_s, n_iter = cfg[1], cfg[2], cfg[3]
            if n_iter == 1:
                x_adv = attack_decoupled(x, eps_pix, cutoff=0.25,
                                         a_strength=a_s, phi_strength=p_s)
            else:
                x_adv = attack_decoupled_iter(x, eps_pix, cutoff=0.25,
                                              a_strength=a_s, phi_strength=p_s, steps=n_iter)

        adv_np = x_adv.squeeze(0).cpu().numpy()
        att_d = evaluate(net, adv_np, lbl, "Synapse", device)
        deltas.append(clean_d - att_d)

    dt = time.time() - t0
    md, sd = float(np.mean(deltas)), float(np.std(deltas))
    marker = " ← BEST" if md == max(d[1] for d in [(0,0)] + [(0, float(np.mean(deltas)))]) else ""
    print(f"  {name:<40} Δ={md:+.4f}±{sd:.3f} {dt:.1f}s")

del net, source_sam
torch.cuda.empty_cache()
print("\nDone")
