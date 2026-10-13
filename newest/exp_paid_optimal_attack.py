#!/usr/bin/env python3
"""
PAID Theory: Amplitude-Weighted Optimal Attack.

KEY THEORETICAL INSIGHT:
=========================
The pixel-space L∞ constraint ||x' - x||_∞ ≤ ε is ASYMMETRIC in frequency domain
for natural images with 1/f amplitude spectra:

  max ΔΦ(f) ≈ ε / A(f)    (first-order approximation)

Since A(f) is LARGE at low frequencies and SMALL at high frequencies:
  - Low-freq phase can barely change (A is large → even tiny ΔΦ violates ε)
  - High-freq phase can change A LOT (A is small → large ΔΦ stays within ε)

This means the ε-ball NATURALLY biases perturbation toward high frequencies.
The optimal attack EXPLOITS this by using AMPLITUDE-WEIGHTED phase perturbation:
put MORE phase noise where amplitude is SMALL (high frequencies).

This is DERIVED from PAID theory (vulnerability in Φ_high, not Φ_low) and
CONSTRAINED by the ε-ball asymmetry. The result is a principled, theory-driven
attack that should outperform both PGD and FD-PGD.

METHODS COMPARED:
- PGD: Pixel-space iterative (baseline, wastes budget on Φ_low)
- FD-PGD: Frequency-masked PGD (our previous best, +9.5%)
- AW-PGD: Amplitude-Weighted PGD — per-frequency adaptive step size
- AW-Phase: ONE-STEP amplitude-weighted phase randomization (no gradient!)
- AW-Hybrid: AW-Phase init + AW-PGD refinement (MAXIMAL attack)
"""
import os, sys, json, gc, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model
from attack_codd import AttackConfig, run_codd_attack

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}

SAMDED_CFG = {
    "Synapse": {
        "vit_b": ("SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 8),
        "vit_l": ("SAMed/Synapse_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 8),
        "vit_h": ("SAMed/Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_299.pth", 8),
    },
    "ISIC": {
        "vit_b": ("SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 1),
        "vit_l": ("SAMed/isic_512_lrexp5.0_pretrain_vit_l_epo300_bs8_lr0.005/epoch_59.pth", 1),
        "vit_h": ("SAMed/isic_512_lrexp3.0_pretrain_vit_h_epo300_bs4_lr0.005/epoch_159.pth", 1),
    },
    "REFUGE": {
        "vit_b": ("SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 2),
        "vit_l": ("SAMed/REFUGE_512_pretrain_vit_l_epo300_bs8_lr0.005/epoch_149.pth", 2),
        "vit_h": ("SAMed/REFUGE_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_159.pth", 2),
    },
}

# ═══════════════════════════════════════════════════════════════════════
# FREQUENCY UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def build_freq_mask(H, W, cutoff_ratio, device, band='high'):
    y = torch.arange(H, device=device).float() - H // 2
    x = torch.arange(W, device=device).float() - W // 2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    R = torch.sqrt(Y**2 + X**2)
    cutoff = cutoff_ratio * min(H, W) / 2
    return (R > cutoff).float() if band == 'high' else (R <= cutoff).float()


def fft_to_pixel(A, Phi):
    return torch.fft.ifft2(A * torch.exp(1j * Phi), norm='ortho').real


def pixel_to_fft(x):
    F = torch.fft.fft2(x, norm='ortho')
    return torch.abs(F), torch.angle(F)


# ═══════════════════════════════════════════════════════════════════════
# ATTACK METHODS
# ═══════════════════════════════════════════════════════════════════════

def attack_pgd(image_encoder, x, eps_pix, step_pix, steps):
    """Standard PGD (pixel space, momentum)."""
    max_val, min_val = x.max(), x.min()
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


def attack_fd_pgd(image_encoder, x, eps_pix, step_pix, steps, cutoff=0.25):
    """FD-PGD: PGD with frequency-masked gradient."""
    max_val, min_val = x.max(), x.min()
    B, C, H, W = x.shape
    with torch.no_grad():
        targets, _ = image_encoder(x)
    mask = build_freq_mask(H, W, cutoff, x.device, 'high').unsqueeze(0).unsqueeze(0)
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    momentum = torch.zeros_like(x)
    def L(f): return sum(F.l1_loss(a, b.detach()) for a, b in zip(f, targets))
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        f, _ = image_encoder(x_adv)
        g = torch.autograd.grad(L(f), x_adv)[0]
        # Mask gradient in frequency domain per channel
        for b in range(B):
            for c in range(C):
                G = torch.fft.fft2(g[b, c], norm='ortho')
                G = G * mask[0, 0]
                g[b, c] = torch.fft.ifft2(G, norm='ortho').real
        g = g / (torch.norm(g, p=1) + 1e-12)
        momentum = 0.5 * momentum + g
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_aw_phase(x, eps_pix, cutoff=0.25):
    """
    AW-Phase: Amplitude-Weighted Phase Attack (ONE-STEP, NO GRADIENT).

    Theory: The maximum phase perturbation at frequency f is ΔΦ_max(f) ≈ ε / A(f).
    We apply random phase noise proportional to this bound in high frequencies,
    zero in low frequencies. This is the OPTIMAL one-step phase attack under
    the ε-ball constraint, derived from PAID theory.

    Key property: Perturbation is concentrated in Φ_high where A(f) is small,
    allowing large phase changes that disrupt the model's phase coherence.
    """
    max_val, min_val = x.max(), x.min()
    B, C, H, W = x.shape

    A, Phi = pixel_to_fft(x)
    mask_high = build_freq_mask(H, W, cutoff, x.device, 'high')
    mask_high = mask_high.unsqueeze(0).unsqueeze(0)

    # Amplitude-weighted max phase change: ΔΦ_max(f) ∝ ε / A(f)
    # For stability, use A(f) clamped to minimum 1e-6
    A_safe = A.clamp(min=1e-6)
    delta_phi_max = eps_pix / A_safe

    # Random phase perturbation scaled by max allowed change
    delta_phi = torch.randn_like(Phi) * delta_phi_max * mask_high

    Phi_new = Phi + delta_phi
    x_adv = fft_to_pixel(A, Phi_new)

    # Project to ε-ball (should already be close)
    eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
    x_adv = torch.clamp(x + eta, min_val, max_val)
    return x_adv.detach()


def attack_aw_pgd(image_encoder, x, eps_pix, step_pix, steps, cutoff=0.25):
    """
    AW-PGD: Amplitude-Weighted PGD — per-frequency adaptive step size.

    Instead of masking low frequencies to zero (FD-PGD), we apply
    AMPLITUDE-WEIGHTED gradient scaling in frequency domain:
      g_weighted(f) = g(f) / A(f)  (normalized per frequency)

    This gives LARGER steps at high frequencies (where A is small)
    and SMALLER steps at low frequencies (where A is large),
    matching the ε-ball's natural asymmetry.

    This is more PRECISE than FD-PGD because instead of binary mask (keep/discard),
    it applies soft weighting proportional to each frequency's vulnerability.
    """
    max_val, min_val = x.max(), x.min()
    B, C, H, W = x.shape

    with torch.no_grad():
        targets, _ = image_encoder(x)

    mask_high = build_freq_mask(H, W, cutoff, x.device, 'high')
    mask_high = mask_high.unsqueeze(0).unsqueeze(0)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    momentum = torch.zeros_like(x)

    # Pre-compute amplitude for weighting
    A_img, _ = pixel_to_fft(x)
    # Weight: 1/A in high frequencies, keep original in low
    weight = mask_high / (A_img.clamp(min=1e-6)) + (1 - mask_high)

    def L(f): return sum(F.l1_loss(a, b.detach()) for a, b in zip(f, targets))

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        f, _ = image_encoder(x_adv)
        g = torch.autograd.grad(L(f), x_adv)[0]

        # Amplitude-weighted frequency masking per channel
        for b in range(B):
            for c in range(C):
                G = torch.fft.fft2(g[b, c], norm='ortho')
                G = G * weight[b, c]
                g[b, c] = torch.fft.ifft2(G, norm='ortho').real

        g = g / (torch.norm(g, p=1) + 1e-12)
        momentum = 0.5 * momentum + g
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_phase_pgd(image_encoder, x, eps_pix, step_pix, steps, cutoff=0.25):
    """
    Phase-PGD: PGD in PHASE SPACE (not pixel space).

    Decompose image into amplitude A and phase Φ via FFT.
    OPTIMIZE ONLY Φ (keep A fixed) using adversarial gradients.
    The ε-ball projection happens after IFFT back to pixel space.

    This directly targets the model's phase sensitivity (PAID: I_Φ dominates)
    while respecting the pixel-space L∞ constraint.
    Key difference from AW-Phase: uses ADVERSARIAL GRADIENTS, not random noise.
    """
    max_val, min_val = x.max(), x.min()
    B, C, H, W = x.shape

    with torch.no_grad():
        targets, _ = image_encoder(x)

    A, Phi0 = pixel_to_fft(x)
    mask_high = build_freq_mask(H, W, cutoff, x.device, 'high')
    mask_high = mask_high.unsqueeze(0).unsqueeze(0)

    # Start from clean phase
    Phi = Phi0.clone().detach()
    momentum = torch.zeros_like(Phi)

    def L_img(x_adv):
        f, _ = image_encoder(x_adv)
        return sum(F.l1_loss(a, b.detach()) for a, b in zip(f, targets))

    x_adv = x.clone()
    for _ in range(steps):
        # Reconstruct from A (fixed) + Phi (optimizable)
        Phi.requires_grad_(True)
        x_recon = fft_to_pixel(A, Phi)

        loss = L_img(x_recon)
        g_phi = torch.autograd.grad(loss, Phi)[0]

        # Mask to high frequencies only
        g_phi = g_phi * mask_high

        # Normalize and momentum
        g_phi = g_phi / (torch.norm(g_phi, p=1) + 1e-12)
        momentum = 0.5 * momentum + g_phi
        Phi = Phi.detach() + step_pix * momentum.sign()

        # Reconstruct and project to ε-ball
        x_recon = fft_to_pixel(A, Phi)
        eta = torch.clamp(x_recon - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val)

        # Re-extract phase from projected image for next iteration
        _, Phi = pixel_to_fft(x_adv)
        Phi = Phi.detach()

    return x_adv.detach()


def attack_aw_hybrid(image_encoder, x, eps_pix, step_pix, steps, cutoff=0.25):
    """
    AW-Hybrid: AW-Phase initialization + AW-PGD refinement.

    Stage 1 (ONE-STEP): Apply amplitude-weighted phase randomization to
      directly destroy Φ_high coherence. This exploits the ε-ball asymmetry
      to maximally disrupt phase structure.

    Stage 2 (ITERATIVE): Run AW-PGD on the residual ε budget to fine-tune
      the perturbation using adversarial gradients.

    This should be the STRONGEST attack because it combines:
    (a) Direct channel disruption (theory-driven, phase channel)
    (b) Adversarial refinement (gradient-driven, both channels)
    """
    max_val, min_val = x.max(), x.min()

    # Stage 1: AW-Phase with 60% of budget
    eps_stage1 = eps_pix * 0.6
    x_adv = attack_aw_phase(x, eps_stage1, cutoff)

    # Stage 2: AW-PGD with remaining 40% budget + reduced step
    eps_stage2 = eps_pix * 0.4
    x_adv = attack_aw_pgd(image_encoder, x_adv, eps_stage2, step_pix * 0.5,
                          max(steps // 2, 5), cutoff)

    # Final projection
    eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
    x_adv = torch.clamp(x + eta, min_val, max_val)
    return x_adv.detach()


def attack_secgr(image_encoder, x, eps_pix, step_pix, steps):
    """SE-CGR: CODD radial attack (baseline)."""
    config = AttackConfig(
        att_step_size=1.6,   # 1.6/255 in pixel scale → step_pix after scaling
        att_bound=16.0,      # 16/255 in pixel scale → eps_pix after scaling
        att_numstep=steps,
        direction_mode="radial", probe_amplification=2.0,
        edge_guidance=True, seed=42,
    )
    adv_np, _ = run_codd_attack(image_encoder, x, config, return_trace=True)
    # Returns [3,H,W] numpy, convert back to [1,3,H,W] tensor
    return torch.from_numpy(adv_np.copy()).float().to(x.device).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════
# DATA & EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def load_synapse_slices(n=5):
    import h5py
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])
    slices = []
    for h5f in h5s:
        if len(slices) >= n: break
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 500],
                       key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:1]
        for s in valid:
            if len(slices) >= n: break
            slc = img[s].astype(np.float32)
            slices.append((np.stack([slc, slc, slc], axis=0), lbl[s]))
    return slices


def load_isic_images(n=5):
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    csv_path = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv")
    if not os.path.exists(csv_path): return []
    df = pd.read_csv(csv_path)
    T = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imgs = []
    for _, row in df.iterrows():
        if len(imgs) >= n: break
        ip = os.path.join(data_dir, row['img'])
        gp = os.path.join(data_dir, row['seg'])
        if not os.path.exists(ip) or not os.path.exists(gp): continue
        imgs.append((T(Image.open(ip).convert('RGB')),
                      (np.array(Image.open(gp).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST),
                                dtype=np.float32) / 255.0 > 0.5).astype(np.float32)))
    return imgs


def load_refuge_images(n=5):
    from PIL import Image
    from torchvision import transforms
    data_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    folders = sorted([f for f in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, f))])
    T = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imgs = []
    for folder in folders:
        if len(imgs) >= n: break
        ip = os.path.join(data_dir, folder, f"{folder}.jpg")
        dp = os.path.join(data_dir, folder, f"{folder}_disc.bmp")
        cp = os.path.join(data_dir, folder, f"{folder}_cup.bmp")
        if not os.path.exists(ip): continue
        img_t = T(Image.open(ip).convert('RGB'))
        disc = np.array(Image.open(dp).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        cup = np.array(Image.open(cp).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        lbl = np.stack([(cup < 128).astype(np.float32), (disc <= 128).astype(np.float32),
                         1.0 - (disc <= 128).astype(np.float32) - (cup < 128).astype(np.float32)], axis=0)
        imgs.append((img_t, lbl))
    return imgs


def evaluate(net, img_data, lbl, ds_name, device):
    if ds_name == "Synapse":
        if img_data.ndim == 2: img_data = img_data[np.newaxis, :, :]
        inp = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        with torch.no_grad():
            masks = net(inp, True, IMAGE_SIZE)['masks']
        masks = torch.softmax(masks, dim=1).squeeze(0).cpu().numpy()
        pred = np.argmax(masks, axis=0)
        dc_sum, n_valid = 0.0, 0
        for c in range(1, 9):
            pc, gc = (pred == c).astype(np.float32), (lbl == c).astype(np.float32)
            if gc.sum() > 0:
                inter = (pc * gc).sum()
                denom = pc.sum() + gc.sum()
                if denom > 0: dc_sum += 2.0 * inter / denom; n_valid += 1
        return dc_sum / max(n_valid, 1)
    elif ds_name == "ISIC":
        if torch.is_tensor(img_data):
            inp = img_data.unsqueeze(0).to(device) if img_data.dim() == 3 else img_data.to(device)
        else:
            inp = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        with torch.no_grad():
            probs = torch.sigmoid(net(inp, True, IMAGE_SIZE)['masks']).squeeze().cpu().numpy()
        gt = (lbl > 0.5).astype(np.float32)
        best = 0.0
        for th in np.arange(0.1, 1.0, 0.1):
            pb = (probs > th).astype(np.float32)
            inter = (pb * gt).sum()
            denom = pb.sum() + gt.sum()
            d = 2.0 * inter / (denom + 1e-8)
            if d > best: best = d
        return float(best)
    else:
        if torch.is_tensor(img_data):
            inp = img_data.unsqueeze(0).to(device) if img_data.dim() == 3 else img_data.to(device)
        else:
            inp = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        with torch.no_grad():
            masks = torch.softmax(net(inp, True, IMAGE_SIZE)['masks'], dim=1).squeeze(0).cpu().numpy()
        pred = np.argmax(masks, axis=0)
        disc_gt = (lbl[1] > 0.5).astype(np.float32)
        cup_gt = (lbl[0] > 0.5).astype(np.float32)
        disc_p, cup_p = (pred == 1).astype(np.float32), (pred == 2).astype(np.float32)
        d_disc = 2.0 * (disc_p * disc_gt).sum() / (disc_p.sum() + disc_gt.sum() + 1e-8)
        d_cup = 2.0 * (cup_p * cup_gt).sum() / (cup_p.sum() + cup_gt.sum() + 1e-8)
        return float((d_disc + d_cup) / 2.0)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

EPSILON = 16.0 / 255.0
STEP = 1.6 / 255.0
STEPS = 20
CUTOFF = 0.25

ATTACKS = {
    "SE-CGR":          ("secgr",     "CODD radial"),
    "PGD":             ("pgd",       "Standard PGD"),
    "FD-PGD":          ("fd_pgd",    "Freq-masked PGD (our previous best)"),
    "AW-Phase":        ("aw_phase",  "Amp-Weighted Phase (1-step, THEORY-OPTIMAL phase)"),
    "AW-PGD":          ("aw_pgd",    "Amp-Weighted PGD (adaptive freq weighting)"),
    "AW-Hybrid":       ("aw_hybrid", "AW-Phase + AW-PGD (MAXIMAL attack)"),
    "Phase-PGD":       ("phase_pgd","PGD in phase space (ITERATIVE gradient, Φ-only)"),
}

def run_attack(key, image_encoder, x, eps_pix, step_pix, steps):
    if key == "secgr":     return attack_secgr(image_encoder, x, eps_pix, step_pix, steps)
    elif key == "pgd":     return attack_pgd(image_encoder, x, eps_pix, step_pix, steps)
    elif key == "fd_pgd":  return attack_fd_pgd(image_encoder, x, eps_pix, step_pix, steps, CUTOFF)
    elif key == "aw_phase": return attack_aw_phase(x, eps_pix, CUTOFF)
    elif key == "aw_pgd":  return attack_aw_pgd(image_encoder, x, eps_pix, step_pix, steps, CUTOFF)
    elif key == "aw_hybrid": return attack_aw_hybrid(image_encoder, x, eps_pix, step_pix, steps, CUTOFF)
    elif key == "phase_pgd": return attack_phase_pgd(image_encoder, x, eps_pix, step_pix, steps, CUTOFF)
    else: raise ValueError(f"Unknown attack: {key}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--vit", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    print(f"PAID AW-Attack: GPU={args.gpu}, ViT={args.vit}")
    print(f"ε={EPSILON:.4f}, step={STEP:.4f}, steps={STEPS}, cutoff={CUTOFF}\n")

    all_results = {}

    for ds_name in ["Synapse", "ISIC", "REFUGE"]:
        ckpt_rel, ncls = SAMDED_CFG[ds_name][args.vit]
        ckpt_path = os.path.join(CKPT_BASE, ckpt_rel)
        if not os.path.exists(ckpt_path):
            print(f"SKIP {ds_name}: no checkpoint"); continue

        print(f"\n{'='*60}")
        print(f"  {ds_name} ({args.vit}, {ncls} classes)")
        print(f"{'='*60}")

        net = load_samed_model(args.vit, SAM_CKPTS[args.vit], ckpt_path,
                               IMAGE_SIZE, ncls, device=device).to(device)
        source_sam = load_sam_encoder(args.vit, SAM_CKPTS[args.vit],
                                       IMAGE_SIZE, device=device)
        image_encoder = source_sam.image_encoder

        if ds_name == "Synapse": images = load_synapse_slices(5)
        elif ds_name == "ISIC": images = load_isic_images(5)
        else: images = load_refuge_images(5)
        print(f"  Loaded {len(images)} images")

        clean_dices = [evaluate(net, img, lbl, ds_name, device)
                       for img, lbl in images]
        clean_dices = [d for d in clean_dices if d > 0.01]
        if not clean_dices:
            print("  No valid samples"); continue
        mean_clean = float(np.mean(clean_dices))
        print(f"  Clean: {mean_clean:.4f} (n={len(clean_dices)})")

        ds_results = {"clean_dice": mean_clean}

        for att_name, (att_key, att_desc) in ATTACKS.items():
            deltas = []
            print(f"  {att_name:<20} ...", end=" ", flush=True)
            t0 = time.time()

            for img_data, lbl in images:
                if torch.is_tensor(img_data):
                    x = img_data.unsqueeze(0).float().to(device)
                else:
                    x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)

                clean_d = evaluate(net, img_data, lbl, ds_name, device)
                if clean_d < 0.01: continue

                rg = x.max().item() - x.min().item()
                eps_pix = EPSILON * rg
                step_pix = STEP * rg

                if att_key == "aw_phase":
                    # AW-Phase is one-step, non-iterative
                    x_adv = run_attack(att_key, image_encoder, x, eps_pix, step_pix, 1)
                else:
                    x_adv = run_attack(att_key, image_encoder, x, eps_pix, step_pix, STEPS)

                if ds_name == "Synapse":
                    adv_np = x_adv.squeeze(0).cpu().numpy()
                else:
                    adv_np = x_adv.squeeze(0).cpu()
                att_d = evaluate(net, adv_np, lbl, ds_name, device)
                deltas.append(clean_d - att_d)

            dt = time.time() - t0
            if deltas:
                md, sd = float(np.mean(deltas)), float(np.std(deltas))
                ds_results[att_name] = {"mean_delta": md, "std_delta": sd,
                                        "n": len(deltas), "time": dt}
                print(f"Δ={md:+.4f}±{sd:.3f} ({dt:.1f}s)")
            else:
                ds_results[att_name] = {"mean_delta": 0.0, "std_delta": 0.0,
                                        "n": 0, "time": dt}
                print("no valid")

        all_results[f"{ds_name}_SAMed_{args.vit}"] = ds_results
        del net, source_sam; gc.collect(); torch.cuda.empty_cache()

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"paid_aw_{args.vit}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n" + "=" * 90)
    print("FINAL: PAID Amplitude-Weighted Attack Validation")
    print("=" * 90)
    for ds_key, data in all_results.items():
        ds = ds_key.replace(f"_SAMed_{args.vit}", "")
        clean = data["clean_dice"]
        print(f"\n{ds} (clean={clean:.4f})")
        entries = [(k, v["mean_delta"]) for k, v in data.items() if k != "clean_dice"]
        entries.sort(key=lambda e: -e[1])
        for rank, (name, delta) in enumerate(entries):
            marker = " ← BEST" if rank == 0 else ""
            print(f"  #{rank+1} {name:<20} Δ={delta:+.4f}{marker}")

        # Theory check
        pgd = data.get("PGD", {}).get("mean_delta", 0)
        fd = data.get("FD-PGD", {}).get("mean_delta", 0)
        aw_h = data.get("AW-Hybrid", {}).get("mean_delta", 0)
        aw_p = data.get("AW-PGD", {}).get("mean_delta", 0)
        best = max(pgd, fd, aw_h, aw_p)
        print(f"  Best Δ={best:+.4f} | AW-Hybrid vs PGD: ΔΔ={aw_h-pgd:+.4f} | AW-PGD vs FD-PGD: ΔΔ={aw_p-fd:+.4f}")


if __name__ == "__main__":
    main()
