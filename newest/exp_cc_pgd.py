#!/usr/bin/env python3
"""
CC-PGD (Cross-Channel PGD): Derives optimal attack from phase-amplitude theory.

THEORY (ε-Ball Frequency Asymmetry):
=====================================
For natural images with 1/f amplitude spectrum A(f) ∝ 1/f, the pixel-space
L∞ constraint ||x' - x||_∞ ≤ ε is ASYMMETRIC in frequency domain:

  Phase budget:   max |δΦ(f)| ∝ ε / A(f)    (large at high freq where A is small)
  Amplitude budget: max |δA(f)| ∝ ε          (uniform across frequencies)

This creates a frequency-dependent trade-off:
  - Low frequencies (A large): phase perturbation is EXPENSIVE (tiny δΦ → large pixel change)
                               → should use AMPLITUDE perturbation
  - High frequencies (A small): phase perturbation is CHEAP (large δΦ → small pixel change)
                                → should use PHASE perturbation

The optimal attack ALLOCATES the ε-budget across channels based on frequency:
  δx = IFFT( δA_low * e^{jΦ} ) + IFFT( j * A * δΦ_high * e^{jΦ} )
       \_____ amplitude @ low freq _____/   \_____ phase @ high freq _____/

METHODS COMPARED:
  - PGD: pixel-space, both channels, all frequencies (BASELINE, wastes budget)
  - FD-PGD: pixel-space, both channels, high-freq only (BINARY mask)
  - CC-PGD: FREQUENCY-DOMAIN, amplitude@low + phase@high (THEORY-DERIVED optimal)
  - CC-PGD (iterative): multi-step CC-PGD
"""
import os, sys, json, time
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
    """Standard PGD (pixel space)."""
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
    """FD-PGD: PGD with frequency-masked gradient (binary high-pass)."""
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


def attack_cc_pgd(image_encoder, x, eps_pix, step_pix, steps, cutoff=0.25):
    """
    CC-PGD: Cross-Channel PGD — THEORY-DERIVED optimal attack.

    Theory: The ε-ball asymmetry means:
    - Low frequencies: phase perturbation is EXPENSIVE (A is large → tiny ΔΦ
      already fills ε budget). Use AMPLITUDE perturbation instead.
    - High frequencies: phase perturbation is CHEAP (A is small → large ΔΦ
      stays within ε). Use PHASE perturbation for maximum effect.

    Implementation:
    1. FFT to get (A, Φ)
    2. Compute gradients in (A, Φ) space via autograd through IFFT
    3. Apply channel masks: g_A at LOW freq, g_Φ at HIGH freq
    4. Synthesize pixel-space update via IFFT:
       δx = IFFT(g_A_low * e^{jΦ}) + IFFT(j * A * g_Φ_high * e^{jΦ})
    5. Sign + momentum + ε-ball projection
    """
    max_val, min_val = x.max(), x.min()
    B, C, H, W = x.shape

    with torch.no_grad():
        targets, _ = image_encoder(x)

    mask_low = build_freq_mask(H, W, cutoff, x.device, 'low')
    mask_high = build_freq_mask(H, W, cutoff, x.device, 'high')
    mask_low = mask_low.unsqueeze(0).unsqueeze(0)
    mask_high = mask_high.unsqueeze(0).unsqueeze(0)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    momentum = torch.zeros_like(x)

    def L_img(x_adv):
        f, _ = image_encoder(x_adv)
        return sum(F.l1_loss(a, b.detach()) for a, b in zip(f, targets))

    for _ in range(steps):
        # Decompose current x_adv into A and Φ
        A_curr, Phi_curr = pixel_to_fft(x_adv)

        # Reconstruct via IFFT with autograd on BOTH A and Φ
        A_param = A_curr.clone().detach().requires_grad_(True)
        Phi_param = Phi_curr.clone().detach().requires_grad_(True)

        x_recon = fft_to_pixel(A_param, Phi_param)
        loss = L_img(x_recon)

        # Gradients w.r.t. amplitude and phase
        g_A, g_Phi = torch.autograd.grad(loss, [A_param, Phi_param])

        # Channel allocation:
        # LOW freq → amplitude channel: δx_A = IFFT(g_A_low * e^{jΦ})
        # HIGH freq → phase channel: δx_Φ = IFFT(j * A * g_Phi_high * e^{jΦ})
        g_A_low = g_A * mask_low
        g_Phi_high = g_Phi * mask_high

        # Synthesize pixel-space update
        # Amplitude contribution: ∂x/∂A = e^{jΦ} → δx_A = IFFT(g_A * e^{jΦ})
        # Phase contribution: ∂x/∂Φ = j*A*e^{jΦ} → δx_Φ = IFFT(j*A*g_Φ*e^{jΦ})
        F_curr = A_curr * torch.exp(1j * Phi_curr)
        update_A = torch.fft.ifft2(g_A_low * torch.exp(1j * Phi_curr), norm='ortho').real
        update_Phi = torch.fft.ifft2(1j * F_curr * g_Phi_high, norm='ortho').real

        g_pixel = update_A + update_Phi

        # Normalize and momentum
        g_pixel = g_pixel / (torch.norm(g_pixel, p=1) + 1e-12)
        momentum = 0.5 * momentum + g_pixel
        x_adv = x_adv.detach() + step_pix * momentum.sign()

        # Project to ε-ball
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


def attack_cc_pgd_v2(image_encoder, x, eps_pix, step_pix, steps, cutoff=0.25):
    """
    CC-PGD v2: Direct (A, Φ) optimization.

    Instead of synthesizing pixel-space updates, we directly optimize A and Φ
    in frequency domain, with explicit per-channel masking. After each step,
    we IFFT and project to ε-ball.

    This is the FULL frequency-domain PGD that PGD approximates.
    """
    max_val, min_val = x.max(), x.min()
    B, C, H, W = x.shape

    with torch.no_grad():
        targets, _ = image_encoder(x)

    A0, Phi0 = pixel_to_fft(x)
    mask_low = build_freq_mask(H, W, cutoff, x.device, 'low').unsqueeze(0).unsqueeze(0)
    mask_high = build_freq_mask(H, W, cutoff, x.device, 'high').unsqueeze(0).unsqueeze(0)

    # Initialize
    A_adv = A0.clone().detach()
    Phi_adv = Phi0.clone().detach()

    def L_img(x_adv):
        f, _ = image_encoder(x_adv)
        return sum(F.l1_loss(a, b.detach()) for a, b in zip(f, targets))

    # Optimize in log-A for positivity
    log_A = torch.log(A_adv.clamp(min=1e-6))
    mom_A = torch.zeros_like(log_A)
    mom_Phi = torch.zeros_like(Phi_adv)

    for _ in range(steps):
        log_A.requires_grad_(True)
        Phi_adv.requires_grad_(True)
        A_curr = torch.exp(log_A)

        x_recon = fft_to_pixel(A_curr, Phi_adv)
        loss = L_img(x_recon)
        g_logA, g_Phi = torch.autograd.grad(loss, [log_A, Phi_adv])

        # Channel masks
        g_logA = g_logA * mask_low   # Only amplitude at LOW frequencies
        g_Phi = g_Phi * mask_high    # Only phase at HIGH frequencies

        # Normalize and momentum
        g_logA = g_logA / (torch.norm(g_logA, p=1) + 1e-12)
        g_Phi = g_Phi / (torch.norm(g_Phi, p=1) + 1e-12)

        mom_A = 0.5 * mom_A + g_logA
        mom_Phi = 0.5 * mom_Phi + g_Phi

        # Update in frequency domain
        log_A = log_A.detach() + step_pix * 0.1 * mom_A.sign()
        Phi_adv = Phi_adv.detach() + step_pix * 0.1 * mom_Phi.sign()

        # IFFT + project to ε-ball
        A_curr = torch.exp(log_A)
        x_recon = fft_to_pixel(A_curr, Phi_adv)
        eta = torch.clamp(x_recon - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val)

        # Re-extract (A, Φ) from projected image
        A_new, Phi_new = pixel_to_fft(x_adv)
        log_A = torch.log(A_new.clamp(min=1e-6)).detach()
        Phi_adv = Phi_new.detach()

    return x_adv.detach()


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
    "PGD":             ("pgd",       "Standard PGD"),
    "FD-PGD":          ("fd_pgd",    "Freq-masked PGD (binary mask)"),
    "CC-PGD":          ("cc_pgd",    "Cross-Channel PGD (THEORY-DERIVED, pixel synthesis)"),
    "CC-PGD-v2":       ("cc_pgd_v2", "Cross-Channel PGD v2 (direct A,Φ optimization)"),
}

def run_attack(key, image_encoder, x, eps_pix, step_pix, steps):
    if key == "pgd":       return attack_pgd(image_encoder, x, eps_pix, step_pix, steps)
    elif key == "fd_pgd":  return attack_fd_pgd(image_encoder, x, eps_pix, step_pix, steps, CUTOFF)
    elif key == "cc_pgd":  return attack_cc_pgd(image_encoder, x, eps_pix, step_pix, steps, CUTOFF)
    elif key == "cc_pgd_v2": return attack_cc_pgd_v2(image_encoder, x, eps_pix, step_pix, steps, CUTOFF)
    else: raise ValueError(f"Unknown attack: {key}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--vit", type=str, default="vit_b")
    parser.add_argument("--quick", action="store_true", help="Only Synapse, for testing")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    print(f"CC-PGD Attack: GPU={args.gpu}, ViT={args.vit}")
    print(f"ε={EPSILON:.4f}, step={STEP:.4f}, steps={STEPS}, cutoff={CUTOFF}\n")

    all_results = {}
    datasets = ["Synapse"] if args.quick else ["Synapse", "ISIC", "REFUGE"]

    for ds_name in datasets:
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

        clean_dices = [evaluate(net, img, lbl, ds_name, device) for img, lbl in images]
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
                ds_results[att_name] = {"mean_delta": 0.0, "std_delta": 0.0, "n": 0, "time": dt}
                print("no valid")

        all_results[f"{ds_name}_SAMed_{args.vit}"] = ds_results
        del net, source_sam; import gc; gc.collect(); torch.cuda.empty_cache()

    # Save
    tag = "quick" if args.quick else args.vit
    out_path = os.path.join(OUTPUT_DIR, f"cc_pgd_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n" + "=" * 90)
    print("FINAL: Cross-Channel PGD — Theory-Derived Optimal Attack")
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


if __name__ == "__main__":
    main()
