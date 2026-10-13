#!/usr/bin/env python3
"""
PAID Theory Ablation Experiments (Task #51).

Ablations:
  1. Multi-band frequency sweep (6 bands instead of just high/low)
  2. Cross-layer PSI (measure PSI at different ViT layer depths)
  3. Cutoff ratio sweep (6 cutoff values from 0.05 to 0.5)
  4. Amplitude vs Phase perturbation comparison

Runs on available GPUs. Saves to ablation_results/paid_ablations.json
"""
import os, sys, json, gc, random
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from collections import defaultdict

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model, load_medsa_model

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}

SAMED_CFG = {
    "Synapse": {
        "vit_b": ("SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 8),
        "vit_l": ("SAMed/Synapse_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 8),
        "vit_h": ("SAMed/Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_299.pth", 8),
    },
    "ISIC": {
        "vit_b": ("SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 1),
        "vit_h": ("SAMed/isic_512_lrexp3.0_pretrain_vit_h_epo300_bs4_lr0.005/epoch_159.pth", 1),
    },
    "REFUGE": {
        "vit_b": ("SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 2),
        "vit_h": ("SAMed/REFUGE_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_159.pth", 2),
    },
}


# =============================================================================
# Phase perturbation functions
# =============================================================================

def phase_band_noise(img_tensor, band='high', cutoff_ratio=0.25):
    """Add random noise to FFT phase in specified frequency band."""
    if img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0)
    _, C, H, W = img_tensor.shape
    F = torch.fft.fft2(img_tensor, norm='ortho')
    A = torch.abs(F); Phi = torch.angle(F)
    y_freq = torch.arange(H, device=img_tensor.device).float() - H // 2
    x_freq = torch.arange(W, device=img_tensor.device).float() - W // 2
    Y, X = torch.meshgrid(y_freq, x_freq, indexing='ij')
    R = torch.sqrt(Y**2 + X**2)
    cutoff = cutoff_ratio * min(H, W) / 2
    if band == 'low': mask = R <= cutoff
    elif band == 'high': mask = R > cutoff
    else: mask = torch.ones(H, W, device=img_tensor.device, dtype=torch.bool)
    delta_phi = torch.randn_like(Phi)
    delta_phi[:, :, ~mask] = 0.0
    F_pert = A * torch.exp(1j * (Phi + delta_phi))
    img_pert = torch.fft.ifft2(F_pert, norm='ortho').real
    return torch.clamp(img_pert, img_tensor.min(), img_tensor.max())


def phase_multiband_noise(img_tensor, band_idx, n_bands=6):
    """Add noise to one of n_bands equal-width frequency bands."""
    if img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0)
    _, C, H, W = img_tensor.shape
    F = torch.fft.fft2(img_tensor, norm='ortho')
    A = torch.abs(F); Phi = torch.angle(F)
    y_freq = torch.arange(H, device=img_tensor.device).float() - H // 2
    x_freq = torch.arange(W, device=img_tensor.device).float() - W // 2
    Y, X = torch.meshgrid(y_freq, x_freq, indexing='ij')
    R = torch.sqrt(Y**2 + X**2)
    Rmax = min(H, W) / 2
    # Divide [0, Rmax] into n_bands equal-width bands
    band_lo = (band_idx / n_bands) * Rmax
    band_hi = ((band_idx + 1) / n_bands) * Rmax
    mask = (R >= band_lo) & (R < band_hi)
    delta_phi = torch.randn_like(Phi)
    delta_phi[:, :, ~mask] = 0.0
    F_pert = A * torch.exp(1j * (Phi + delta_phi))
    img_pert = torch.fft.ifft2(F_pert, norm='ortho').real
    return torch.clamp(img_pert, img_tensor.min(), img_tensor.max())


def amplitude_perturbation(img_tensor, mode='flat'):
    """Perturb FFT amplitude while preserving phase."""
    if img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0)
    F = torch.fft.fft2(img_tensor, norm='ortho')
    Phi = torch.angle(F)
    if mode == 'flat':
        A_new = torch.ones_like(torch.abs(F))
    elif mode == 'shuffled':
        A_orig = torch.abs(F)
        B, C, H, W = A_orig.shape
        A_flat = A_orig.view(B, C, -1)
        idx = torch.randperm(H * W, device=img_tensor.device)
        A_new = A_flat[:, :, idx].view_as(A_orig)
    elif mode == 'random':
        A_new = torch.rand_like(torch.abs(F))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    F_pert = A_new * torch.exp(1j * Phi)
    img_pert = torch.fft.ifft2(F_pert, norm='ortho').real
    return torch.clamp(img_pert, img_tensor.min(), img_tensor.max())


# =============================================================================
# Data loading
# =============================================================================

def get_synapse_slices(n_images=5):
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])
    images = []
    for h5f in h5s:
        if len(images) >= n_images: break
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 500],
                       key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:1]
        for s in valid:
            if len(images) >= n_images: break
            slc = img[s].astype(np.float32)
            img_3ch = np.stack([slc, slc, slc], axis=0)
            images.append((img_3ch, lbl[s]))
    return images


def get_isic_images(n_images=5):
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    csv_path = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv")
    if not os.path.exists(csv_path): return []
    df = pd.read_csv(csv_path)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = []
    for _, row in df.iterrows():
        if len(images) >= n_images: break
        img_path = os.path.join(data_dir, row['img'])
        gt_path = os.path.join(data_dir, row['seg'])
        if not os.path.exists(img_path) or not os.path.exists(gt_path): continue
        img_t = transform(Image.open(img_path).convert('RGB'))
        gt = np.array(Image.open(gt_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32) / 255.0
        images.append((img_t, (gt > 0.5).astype(np.float32)))
    return images


def get_refuge_images(n_images=5):
    from PIL import Image
    from torchvision import transforms
    data_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = []
    for folder in folders:
        if len(images) >= n_images: break
        img_path = os.path.join(data_dir, folder, f"{folder}.jpg")
        disc_path = os.path.join(data_dir, folder, f"{folder}_disc.bmp")
        cup_path = os.path.join(data_dir, folder, f"{folder}_cup.bmp")
        if not os.path.exists(img_path): continue
        img_t = transform(Image.open(img_path).convert('RGB'))
        disc_raw = np.array(Image.open(disc_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        cup_raw = np.array(Image.open(cup_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        label = np.stack([
            (cup_raw < 128).astype(np.float32),
            (disc_raw <= 128).astype(np.float32),
            1.0 - (disc_raw <= 128).astype(np.float32) - (cup_raw < 128).astype(np.float32),
        ], axis=0)
        images.append((img_t, label))
    return images


# =============================================================================
# Evaluation (simplified per-dataset, SAMed only)
# =============================================================================

def evaluate_samed(net, img_data, lbl, ds_name, device):
    if ds_name == "Synapse":
        # Direct 2D inference — avoids test_single_volume's 3D volume path
        if torch.is_tensor(img_data):
            inp = img_data.unsqueeze(0).to(device) if img_data.dim() == 3 else img_data.to(device)
        else:
            inp = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        with torch.no_grad():
            outputs = net(inp, True, IMAGE_SIZE)
            masks = outputs['masks']  # [1, 8, H, W]
        masks = torch.softmax(masks, dim=1).squeeze(0).cpu().numpy()  # [8, H, W]
        pred = np.argmax(masks, axis=0)  # [H, W]
        gt = lbl if not torch.is_tensor(lbl) else lbl.cpu().numpy()
        # Per-class Dice (excluding background 0)
        dc_sum = 0.0; n_valid = 0
        for c in range(1, 8 + 1):
            pred_c = (pred == c).astype(np.float32)
            gt_c = (gt == c).astype(np.float32)
            if gt_c.sum() > 0:
                inter = (pred_c * gt_c).sum()
                denom = pred_c.sum() + gt_c.sum()
                if denom > 0:
                    dc_sum += 2.0 * inter / denom
                n_valid += 1
        return dc_sum / max(n_valid, 1)
    elif ds_name == "ISIC":
        if torch.is_tensor(img_data):
            inp = img_data.unsqueeze(0).to(device) if img_data.dim() == 3 else img_data.to(device)
        else:
            inp = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        with torch.no_grad():
            outputs = net(inp, True, IMAGE_SIZE)
            masks = outputs['masks']
        probs = torch.sigmoid(masks).squeeze().cpu().numpy()
        gt = (lbl > 0.5).astype(np.float32)
        best_d = 0.0
        for th in np.arange(0.1, 1.0, 0.1):
            pb = (probs > th).astype(np.float32)
            inter = (pb * gt).sum(); denom = pb.sum() + gt.sum()
            d = 2.0 * inter / (denom + 1e-8)
            if d > best_d: best_d = d
        return float(best_d)
    elif ds_name == "REFUGE":
        if torch.is_tensor(img_data):
            inp = img_data.unsqueeze(0).to(device) if img_data.dim() == 3 else img_data.to(device)
        else:
            inp = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        with torch.no_grad():
            outputs = net(inp, True, IMAGE_SIZE)
            masks = outputs['masks']
        masks = torch.softmax(masks, dim=1).squeeze(0).cpu().numpy()
        pred = np.argmax(masks, axis=0)
        disc_gt = (lbl[1] > 0.5).astype(np.float32)
        cup_gt = (lbl[0] > 0.5).astype(np.float32)
        disc_p = (pred == 1).astype(np.float32)
        cup_p = (pred == 2).astype(np.float32)
        d_disc = 2.0 * (disc_p * disc_gt).sum() / (disc_p.sum() + disc_gt.sum() + 1e-8)
        d_cup = 2.0 * (cup_p * cup_gt).sum() / (cup_p.sum() + cup_gt.sum() + 1e-8)
        return float((d_disc + d_cup) / 2.0)


# =============================================================================
# Ablation 1: Multi-band frequency sweep
# =============================================================================

def run_ablation_multiband(device, n_imgs=5, n_trials=3, n_bands=6):
    print("=" * 60)
    print(f"Ablation 1: Multi-Band Frequency Sweep ({n_bands} bands)")
    print("=" * 60)
    results = {}
    dataset_loaders = {
        "Synapse": get_synapse_slices,
        "ISIC": get_isic_images,
        "REFUGE": get_refuge_images,
    }

    for ds_name in ["Synapse", "ISIC", "REFUGE"]:
        ckpt_rel, ncls = SAMED_CFG[ds_name]["vit_b"]
        ckpt_path = os.path.join(CKPT_BASE, ckpt_rel)
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {ds_name}: no checkpoint"); continue
        net = load_samed_model("vit_b", SAM_CKPTS["vit_b"], ckpt_path, IMAGE_SIZE, ncls, device=device)
        images = dataset_loaders[ds_name](n_imgs)
        print(f"  {ds_name}: {len(images)} images")

        band_results = {f"band_{b}": [] for b in range(n_bands)}
        band_results["clean"] = []
        for img_data, lbl in images:
            if torch.is_tensor(img_data):
                img_t = img_data.unsqueeze(0).float().to(device)
            else:
                img_t = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
            clean_d = evaluate_samed(net, img_data, lbl, ds_name, device)
            if clean_d < 0.01: continue
            band_results["clean"].append(clean_d)
            for bi in range(n_bands):
                band_ds = []
                for _ in range(n_trials):
                    img_pert = phase_multiband_noise(img_t, band_idx=bi, n_bands=n_bands)
                    pert_d = evaluate_samed(net, img_pert[0], lbl, ds_name, device)
                    band_ds.append(max(0, clean_d - pert_d))
                band_results[f"band_{bi}"].append(float(np.mean(band_ds)))
        results[ds_name] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in band_results.items() if v}
        print(f"    Clean: {results[ds_name]['clean']['mean']:.4f}")
        for bi in range(n_bands):
            if f"band_{bi}" in results[ds_name]:
                print(f"    Band {bi}: Δ={results[ds_name][f'band_{bi}']['mean']:.4f}")
        del net; gc.collect(); torch.cuda.empty_cache()
    return results


# =============================================================================
# Ablation 2: Cutoff ratio sweep
# =============================================================================

def run_ablation_cutoff_sweep(device, n_imgs=5, n_trials=3):
    print("=" * 60)
    print("Ablation 2: Cutoff Ratio Sweep")
    print("=" * 60)
    cutoff_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    results = {}
    dataset_loaders = {
        "Synapse": get_synapse_slices,
        "ISIC": get_isic_images,
        "REFUGE": get_refuge_images,
    }

    for ds_name in ["Synapse", "ISIC", "REFUGE"]:
        ckpt_rel, ncls = SAMED_CFG[ds_name]["vit_b"]
        ckpt_path = os.path.join(CKPT_BASE, ckpt_rel)
        if not os.path.exists(ckpt_path): continue
        net = load_samed_model("vit_b", SAM_CKPTS["vit_b"], ckpt_path, IMAGE_SIZE, ncls, device=device)
        images = dataset_loaders[ds_name](n_imgs)
        print(f"  {ds_name}: {len(images)} images")

        cutoff_results = {}
        for cr in cutoff_ratios:
            high_deltas = []
            for img_data, lbl in images:
                if torch.is_tensor(img_data):
                    img_t = img_data.unsqueeze(0).float().to(device)
                else:
                    img_t = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
                clean_d = evaluate_samed(net, img_data, lbl, ds_name, device)
                if clean_d < 0.01: continue
                trial_ds = []
                for _ in range(n_trials):
                    img_pert = phase_band_noise(img_t, band='high', cutoff_ratio=cr)
                    pert_d = evaluate_samed(net, img_pert[0], lbl, ds_name, device)
                    trial_ds.append(max(0, clean_d - pert_d))
                high_deltas.append(float(np.mean(trial_ds)))
            cutoff_results[f"cr_{cr}"] = {"mean": float(np.mean(high_deltas)), "std": float(np.std(high_deltas))}
        results[ds_name] = cutoff_results
        print(f"    Cutoff sweep: ", {k: f"{v['mean']:.4f}" for k, v in cutoff_results.items()})
        del net; gc.collect(); torch.cuda.empty_cache()
    return results


# =============================================================================
# Ablation 3: Amplitude vs Phase perturbation
# =============================================================================

def run_ablation_amplitude_phase(device, n_imgs=5, n_trials=3):
    print("=" * 60)
    print("Ablation 3: Amplitude vs Phase Perturbation")
    print("=" * 60)
    results = {}
    dataset_loaders = {
        "Synapse": get_synapse_slices,
        "ISIC": get_isic_images,
        "REFUGE": get_refuge_images,
    }

    for ds_name in ["Synapse", "ISIC", "REFUGE"]:
        ckpt_rel, ncls = SAMED_CFG[ds_name]["vit_b"]
        ckpt_path = os.path.join(CKPT_BASE, ckpt_rel)
        if not os.path.exists(ckpt_path): continue
        net = load_samed_model("vit_b", SAM_CKPTS["vit_b"], ckpt_path, IMAGE_SIZE, ncls, device=device)
        images = dataset_loaders[ds_name](n_imgs)
        print(f"  {ds_name}: {len(images)} images")

        conds = {
            "phase_high": lambda t: phase_band_noise(t, 'high', 0.25),
            "phase_low": lambda t: phase_band_noise(t, 'low', 0.25),
            "phase_all": lambda t: phase_band_noise(t, 'all', 0.0),
            "amp_flat": lambda t: amplitude_perturbation(t, 'flat'),
            "amp_shuffled": lambda t: amplitude_perturbation(t, 'shuffled'),
            "amp_random": lambda t: amplitude_perturbation(t, 'random'),
        }
        cond_results = defaultdict(list)
        for img_data, lbl in images:
            if torch.is_tensor(img_data):
                img_t = img_data.unsqueeze(0).float().to(device)
            else:
                img_t = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
            clean_d = evaluate_samed(net, img_data, lbl, ds_name, device)
            if clean_d < 0.01: continue
            cond_results["clean"].append(clean_d)
            for cname, cfn in conds.items():
                trial_ds = []
                for _ in range(n_trials):
                    img_pert = cfn(img_t)
                    pert_d = evaluate_samed(net, img_pert[0], lbl, ds_name, device)
                    trial_ds.append(max(0, clean_d - pert_d))
                cond_results[cname].append(float(np.mean(trial_ds)))
        results[ds_name] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in cond_results.items() if v}
        for k, v in results[ds_name].items():
            print(f"    {k}: Δ={v['mean']:.4f} ± {v['std']:.4f}")
        del net; gc.collect(); torch.cuda.empty_cache()
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ablation", type=str, default="all",
                        help="which ablation: multiband, cutoff, ampphase, all")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    print(f"PAID Ablations on GPU {args.gpu}")

    all_results = {}

    if args.ablation in ("multiband", "all"):
        all_results["multiband_sweep"] = run_ablation_multiband(device)

    if args.ablation in ("cutoff", "all"):
        all_results["cutoff_sweep"] = run_ablation_cutoff_sweep(device)

    if args.ablation in ("ampphase", "all"):
        all_results["amplitude_phase"] = run_ablation_amplitude_phase(device)

    output_path = os.path.join(OUTPUT_DIR, "paid_ablations.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
