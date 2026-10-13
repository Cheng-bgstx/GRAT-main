#!/usr/bin/env python3
"""
AB-PGD (Amplitude-Biased PGD): Explicitly decompose pixel gradient into
amplitude (g_A) and phase (g_Φ) components, then BIAS toward amplitude.

Derivation:
  For x = IFFT(A · e^{jΦ}), a small pixel change δx affects:
    δA(f) = Re(e^{-jΦ(f)} · FFT(δx)(f))
    δΦ(f) = -Im(e^{-jΦ(f)} · FFT(δx)(f)) / A(f)

  The gradient decomposes as:
    g_A = IFFT(Re(e^{-jΦ} · G) · e^{jΦ})      ← changes amplitude
    g_Φ = IFFT(j · Im(e^{-jΦ} · G) · e^{jΦ})   ← changes phase
    g = g_A + g_Φ                                ← full pixel gradient

  AB-PGD: g_biased = α·g_A + β·g_Φ  with α > β (favor amplitude)
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


def decompose_gradient_autograd(x_adv, image_encoder, targets):
    """
    Decompose pixel gradient into g_A and g_Φ using autograd.
    g_A = IFFT(∂L/∂A · e^{jΦ}) — changes only amplitude
    g_Φ = IFFT(j · A · ∂L/∂Φ · e^{jΦ}) — changes only phase
    g_pixel = g_A + g_Φ (EXACT, by autograd)

    Returns (g_pixel, g_A, g_Φ)
    """
    B, C, H, W = x_adv.shape
    A, Phi = pixel_to_fft(x_adv)

    A_param = A.clone().detach().requires_grad_(True)
    Phi_param = Phi.clone().detach().requires_grad_(True)

    x_recon = fft_to_pixel(A_param, Phi_param)
    f, _ = image_encoder(x_recon)
    loss = sum(F.l1_loss(a, b.detach()) for a, b in zip(f, targets))

    g_A_freq, g_Phi_freq = torch.autograd.grad(loss, [A_param, Phi_param])

    # Synthesize pixel-space components
    g_A_pixel = torch.zeros_like(x_adv)
    g_Phi_pixel = torch.zeros_like(x_adv)

    exp_jPhi = torch.exp(1j * Phi)
    for b in range(B):
        for c in range(C):
            g_A_pixel[b, c] = torch.fft.ifft2(
                g_A_freq[b, c] * exp_jPhi[b, c], norm='ortho').real
            g_Phi_pixel[b, c] = torch.fft.ifft2(
                1j * A[b, c] * g_Phi_freq[b, c] * exp_jPhi[b, c], norm='ortho').real

    return g_A_pixel + g_Phi_pixel, g_A_pixel, g_Phi_pixel


def attack_ab_pgd(image_encoder, x, eps_pix, step_pix, steps,
                  amp_weight=2.0, phi_weight=1.0, cutoff=0.0):
    """
    AB-PGD: Amplitude-Biased PGD (autograd decomposition).
    Uses autograd through IFFT to get EXACT g_A and g_Φ components.
    """
    max_val, min_val = x.max(), x.min()
    B, C, H, W = x.shape

    with torch.no_grad():
        targets, _ = image_encoder(x)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    momentum_A = torch.zeros_like(x)
    momentum_Phi = torch.zeros_like(x)

    mask_high = None
    if cutoff > 0:
        mask_high = build_freq_mask(H, W, cutoff, x.device, 'high')
        mask_high = mask_high.unsqueeze(0).unsqueeze(0)

    for _ in range(steps):
        # Compute g_A and g_Φ via autograd through IFFT
        g_full, g_A, g_Phi = decompose_gradient_autograd(x_adv, image_encoder, targets)

        # Apply frequency mask (optional)
        if mask_high is not None:
            for b in range(B):
                for c in range(C):
                    GA = torch.fft.fft2(g_A[b, c], norm='ortho')
                    GP = torch.fft.fft2(g_Phi[b, c], norm='ortho')
                    GA = GA * mask_high[0, 0]
                    GP = GP * mask_high[0, 0]
                    g_A[b, c] = torch.fft.ifft2(GA, norm='ortho').real
                    g_Phi[b, c] = torch.fft.ifft2(GP, norm='ortho').real
                    g_full[b, c] = g_A[b, c] + g_Phi[b, c]

        # Bias toward one channel
        g_biased = amp_weight * g_A + phi_weight * g_Phi

        # Normalize and momentum
        g_biased = g_biased / (torch.norm(g_biased, p=1) + 1e-12)
        momentum_A = 0.5 * momentum_A + g_biased
        x_adv = x_adv.detach() + step_pix * momentum_A.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# Verify autograd decomposition
print("Verifying gradient decomposition (autograd)...")
x_test = torch.from_numpy(images[0][0]).unsqueeze(0).float().to(device)
with torch.no_grad():
    targets_test, _ = image_encoder(x_test)
x_test_adv = x_test.clone().detach() + torch.randn_like(x_test) * 1e-5
g_full, g_A, g_Phi = decompose_gradient_autograd(x_test_adv, image_encoder, targets_test)
g_recon = g_A + g_Phi
error = (g_full - g_recon).abs().max().item()
g_max = g_full.abs().max().item()
print(f"  Decomposition error: max={error:.2e}, g_max={g_max:.4f}")
print(f"  |g_full|={g_full.abs().mean():.4f}, |g_A|={g_A.abs().mean():.4f}, |g_Phi|={g_Phi.abs().mean():.4f}")
g_A_mean = g_A.abs().mean().item()
g_Phi_mean = g_Phi.abs().mean().item()
print(f"  Natural |g_A|/|g_Phi| ratio = {g_A_mean / max(g_Phi_mean, 1e-10):.2f}")

# Sweep over amp_weight / phi_weight ratios
configs = [
    ("PGD (baseline)", "pgd", 1.0, 1.0, 0.0),
    ("FD-PGD (baseline)", "fd_pgd", 1.0, 1.0, 0.0),
    ("AB-PGD A:Φ=1:1 (no bias)", "ab", 1.0, 1.0, 0.0),
    ("AB-PGD A:Φ=2:1", "ab", 2.0, 1.0, 0.0),
    ("AB-PGD A:Φ=5:1", "ab", 5.0, 1.0, 0.0),
    ("AB-PGD A:Φ=10:1", "ab", 10.0, 1.0, 0.0),
    ("AB-PGD A:Φ=1:2", "ab", 1.0, 2.0, 0.0),
    ("AB-PGD A:Φ=1:5", "ab", 1.0, 5.0, 0.0),
    ("AB-FD-PGD A:Φ=2:1", "ab", 2.0, 1.0, 0.25),
    ("AB-FD-PGD A:Φ=5:1", "ab", 5.0, 1.0, 0.25),
]

print(f"\n{'Method':<35} {'Δ Dice':>12} {'Time':>8}")
print("-" * 55)

best_delta = -999
for name, method, aw, pw, cutoff in configs:
    deltas = []
    t0 = time.time()

    for img_data, lbl in images:
        x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        clean_d = evaluate(net, img_data, lbl, "Synapse", device)
        if clean_d < 0.01: continue
        rg = x.max().item() - x.min().item()
        eps_pix = EPSILON * rg
        step_pix = STEP * rg

        if method == "pgd":
            x_adv = attack_pgd(image_encoder, x, eps_pix, step_pix, STEPS)
        elif method == "fd_pgd":
            x_adv = attack_fd_pgd(image_encoder, x, eps_pix, step_pix, STEPS)
        else:
            x_adv = attack_ab_pgd(image_encoder, x, eps_pix, step_pix, STEPS,
                                  amp_weight=aw, phi_weight=pw, cutoff=cutoff)

        adv_np = x_adv.squeeze(0).cpu().numpy()
        att_d = evaluate(net, adv_np, lbl, "Synapse", device)
        deltas.append(clean_d - att_d)

    dt = time.time() - t0
    md, sd = float(np.mean(deltas)), float(np.std(deltas))
    marker = ""
    if md > best_delta:
        best_delta = md
        marker = " ← BEST"
    print(f"  {name:<35} Δ={md:+.4f}±{sd:.3f} {dt:.1f}s{marker}")

del net, source_sam
torch.cuda.empty_cache()
print("\nDone")
