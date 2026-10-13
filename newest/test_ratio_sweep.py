#!/usr/bin/env python3
"""
Ratio Sweep: Find optimal phase/amplitude perturbation ratio.
PGD implicitly uses ~100:1 Φ:A ratio. Is this optimal?
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
    load_synapse_slices, evaluate,
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

def attack_ratio_pgd(image_encoder, x, eps_pix, steps, phi_weight=1.0, amp_weight=1.0):
    """
    PGD with explicit phase/amplitude weighting.
    phi_weight > amp_weight → more phase perturbation
    amp_weight > phi_weight → more amplitude perturbation
    """
    max_val, min_val = x.max(), x.min()
    step_pix = STEP * (max_val - min_val)
    B, C, H, W = x.shape

    with torch.no_grad():
        targets, _ = image_encoder(x)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    momentum = torch.zeros_like(x)

    def L_img(xa):
        f, _ = image_encoder(xa)
        return sum(F.l1_loss(a, b.detach()) for a, b in zip(f, targets))

    for _ in range(steps):
        A_c, Phi_c = pixel_to_fft(x_adv)
        A_param = A_c.clone().detach().requires_grad_(True)
        Phi_param = Phi_c.clone().detach().requires_grad_(True)
        x_recon = fft_to_pixel(A_param, Phi_param)
        loss = L_img(x_recon)
        g_A, g_Phi = torch.autograd.grad(loss, [A_param, Phi_param])

        # Apply weights
        g_A_w = g_A * amp_weight
        g_Phi_w = g_Phi * phi_weight

        # Synthesize pixel-space gradient
        F_c = A_c * torch.exp(1j * Phi_c)
        delta_A = torch.fft.ifft2(g_A_w * torch.exp(1j * Phi_c), norm='ortho').real
        delta_Phi = torch.fft.ifft2(1j * F_c * g_Phi_w, norm='ortho').real
        g_pixel = delta_A + delta_Phi

        g_pixel = g_pixel / (torch.norm(g_pixel, p=1) + 1e-12)
        momentum = 0.5 * momentum + g_pixel
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# Also: standard PGD as reference
def attack_pgd_ref(image_encoder, x, eps_pix, steps):
    max_val, min_val = x.max(), x.min()
    step_pix = STEP * (max_val - min_val)
    with torch.no_grad():
        targets, _ = image_encoder(x)
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    momentum = torch.zeros_like(x)
    def L(f): return sum(F.l1_loss(a, b.detach()) for a, b in zip(f, targets))
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        f, _ = image_encoder(x_adv)
        g = torch.autograd.grad(L(f), x_adv)[0]
        g = g / (torch.norm(g, p=1) + 1e-12)
        momentum = 0.5 * momentum + g
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_balanced_fd_pgd(image_encoder, x, eps_pix, steps, cutoff=0.25):
    """Balanced FD-PGD: high-pass mask + 1:1 Φ:A gradient ratio."""
    max_val, min_val = x.max(), x.min()
    step_pix = STEP * (max_val - min_val)
    B, C, H, W = x.shape

    with torch.no_grad():
        targets, _ = image_encoder(x)

    from exp_paid_optimal_attack import build_freq_mask, pixel_to_fft, fft_to_pixel
    mask_high = build_freq_mask(H, W, cutoff, x.device, 'high').unsqueeze(0).unsqueeze(0)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    momentum = torch.zeros_like(x)

    def L_img(xa):
        f, _ = image_encoder(xa)
        return sum(F.l1_loss(a, b.detach()) for a, b in zip(f, targets))

    for _ in range(steps):
        A_c, Phi_c = pixel_to_fft(x_adv)
        A_param = A_c.clone().detach().requires_grad_(True)
        Phi_param = Phi_c.clone().detach().requires_grad_(True)
        x_recon = fft_to_pixel(A_param, Phi_param)
        loss = L_img(x_recon)
        g_A, g_Phi = torch.autograd.grad(loss, [A_param, Phi_param])

        # Apply high-freq mask to both channels, then balanced 1:1
        g_A_w = g_A * mask_high
        g_Phi_w = g_Phi * mask_high

        F_c = A_c * torch.exp(1j * Phi_c)
        delta_A = torch.fft.ifft2(g_A_w * torch.exp(1j * Phi_c), norm='ortho').real
        delta_Phi = torch.fft.ifft2(1j * F_c * g_Phi_w, norm='ortho').real
        g_pixel = delta_A + delta_Phi

        g_pixel = g_pixel / (torch.norm(g_pixel, p=1) + 1e-12)
        momentum = 0.5 * momentum + g_pixel
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# Sweep over Φ:A ratios
ratios = [
    ("Φ-only", 1.0, 0.0),
    ("Φ:A=100:1", 1.0, 0.01),
    ("Φ:A=10:1", 1.0, 0.1),
    ("Φ:A=3:1", 1.0, 0.33),
    ("Φ:A=1:1", 1.0, 1.0),
    ("Φ:A=1:3", 0.33, 1.0),
    ("Φ:A=1:10", 0.1, 1.0),
    ("A-only", 0.0, 1.0),
]

print("\nSweeping Φ:A ratios...")
print(f"{'Ratio':<20} {'Δ Dice':>12} {'Time':>8}")
print("-" * 40)

for name, pw, aw in ratios:
    deltas = []
    t0 = time.time()
    for img_data, lbl in images:
        x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        clean_d = evaluate(net, img_data, lbl, "Synapse", device)
        if clean_d < 0.01: continue
        rg = x.max().item() - x.min().item()
        eps_pix = EPSILON * rg

        if pw == aw == 1.0:
            x_adv = attack_pgd_ref(image_encoder, x, eps_pix, STEPS)
        else:
            x_adv = attack_ratio_pgd(image_encoder, x, eps_pix, STEPS, pw, aw)

        adv_np = x_adv.squeeze(0).cpu().numpy()
        att_d = evaluate(net, adv_np, lbl, "Synapse", device)
        deltas.append(clean_d - att_d)

    dt = time.time() - t0
    md, sd = float(np.mean(deltas)), float(np.std(deltas))
    marker = " ← BEST" if md == max(deltas) else ""
    print(f"  {name:<20} Δ={md:+.4f}±{sd:.3f} {dt:.1f}s{marker}")

# Also test PGD in pixel space
deltas = []
t0 = time.time()
for img_data, lbl in images:
    x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
    clean_d = evaluate(net, img_data, lbl, "Synapse", device)
    if clean_d < 0.01: continue
    rg = x.max().item() - x.min().item()
    eps_pix = EPSILON * rg
    x_adv = attack_pgd_ref(image_encoder, x, eps_pix, STEPS)
    adv_np = x_adv.squeeze(0).cpu().numpy()
    att_d = evaluate(net, adv_np, lbl, "Synapse", device)
    deltas.append(clean_d - att_d)
dt = time.time() - t0
print(f"  {'PGD (pixel-space)':<20} Δ={np.mean(deltas):+.4f}±{np.std(deltas):.3f} {dt:.1f}s")

# Also test: FD-PGD with balanced 1:1 ratio (high-freq only + balanced channels)
from exp_paid_optimal_attack import attack_fd_pgd, build_freq_mask

for cutoff in [0.25]:
    deltas = []
    t0 = time.time()
    for img_data, lbl in images:
        x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        clean_d = evaluate(net, img_data, lbl, "Synapse", device)
        if clean_d < 0.01: continue
        rg = x.max().item() - x.min().item()
        eps_pix = EPSILON * rg
        x_adv = attack_balanced_fd_pgd(image_encoder, x, eps_pix, STEPS, cutoff)
        adv_np = x_adv.squeeze(0).cpu().numpy()
        att_d = evaluate(net, adv_np, lbl, "Synapse", device)
        deltas.append(clean_d - att_d)
    dt = time.time() - t0
    print(f"  {'Balanced-FD-PGD':<20} Δ={np.mean(deltas):+.4f}±{np.std(deltas):.3f} {dt:.1f}s")

del net, source_sam
torch.cuda.empty_cache()
print("\nDone")
