#!/usr/bin/env python3
"""Quick test: Phase-PGD vs PGD on Synapse vit_b only."""
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
    attack_pgd, attack_phase_pgd, attack_fd_pgd,
    build_freq_mask, pixel_to_fft, fft_to_pixel, load_synapse_slices, evaluate,
)

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
CKPT_REL = "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"

EPSILON = 16.0 / 255.0
STEP = 1.6 / 255.0
STEPS = 20
CUTOFF = 0.25

device = torch.device("cuda:0")

# Load models
net = load_samed_model("vit_b", SAM_CKPT, os.path.join(CKPT_BASE, CKPT_REL),
                       IMAGE_SIZE, 8, device=device).to(device)
source_sam = load_sam_encoder("vit_b", SAM_CKPT, IMAGE_SIZE, device=device)
image_encoder = source_sam.image_encoder

# Load images
images = load_synapse_slices(5)
print(f"Loaded {len(images)} Synapse images")

# Clean evaluation
clean_dices = [evaluate(net, img, lbl, "Synapse", device) for img, lbl in images]
clean_dices = [d for d in clean_dices if d > 0.01]
print(f"Clean: {np.mean(clean_dices):.4f}")

def attack_amp_pgd(image_encoder, x, eps_pix, step_pix, steps, cutoff=0.0):
    """Amplitude-PGD: optimize only amplitude A(f), keep phase Φ fixed."""
    max_val, min_val = x.max(), x.min()
    B, C, H, W = x.shape

    with torch.no_grad():
        targets, _ = image_encoder(x)

    A0, Phi = pixel_to_fft(x)
    A = A0.clone().detach()

    def L_img(x_adv):
        f, _ = image_encoder(x_adv)
        return sum(F.l1_loss(a, b.detach()) for a, b in zip(f, targets))

    # Optimize in log-A space to ensure positivity
    log_A = torch.log(A.clamp(min=1e-6))
    momentum = torch.zeros_like(log_A)

    for _ in range(steps):
        log_A.requires_grad_(True)
        A_curr = torch.exp(log_A)
        x_recon = fft_to_pixel(A_curr, Phi)
        loss = L_img(x_recon)
        g = torch.autograd.grad(loss, log_A)[0]
        g = g / (torch.norm(g, p=1) + 1e-12)
        momentum = 0.5 * momentum + g
        log_A = log_A.detach() + step_pix * 0.1 * momentum.sign()
        A_curr = torch.exp(log_A)
        x_recon = fft_to_pixel(A_curr, Phi)
        eta = torch.clamp(x_recon - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val)
        A_curr, _ = pixel_to_fft(x_adv)
        log_A = torch.log(A_curr.clamp(min=1e-6)).detach()

    return x_adv.detach()


# Compare attacks
for att_name, att_fn in [("PGD", attack_pgd), ("FD-PGD", attack_fd_pgd),
                          ("Phase-PGD", attack_phase_pgd), ("Amp-PGD", attack_amp_pgd)]:
    deltas = []
    t0 = time.time()
    for img_data, lbl in images:
        if torch.is_tensor(img_data):
            x = img_data.unsqueeze(0).float().to(device)
        else:
            x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)

        clean_d = evaluate(net, img_data, lbl, "Synapse", device)
        if clean_d < 0.01: continue

        rg = x.max().item() - x.min().item()
        eps_pix = EPSILON * rg
        step_pix = STEP * rg

        if att_name == "PGD":
            x_adv = att_fn(image_encoder, x, eps_pix, step_pix, STEPS)
        else:
            x_adv = att_fn(image_encoder, x, eps_pix, step_pix, STEPS, CUTOFF)

        adv_np = x_adv.squeeze(0).cpu().numpy()
        att_d = evaluate(net, adv_np, lbl, "Synapse", device)
        deltas.append(clean_d - att_d)

    dt = time.time() - t0
    print(f"  {att_name:<20} Δ={np.mean(deltas):+.4f}±{np.std(deltas):.3f} ({dt:.1f}s)")

del net, source_sam
torch.cuda.empty_cache()
print("Done")
