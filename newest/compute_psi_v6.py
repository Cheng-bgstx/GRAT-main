#!/usr/bin/env python3
"""
PAID Theory v6: Gradient Spectral PSI + Frequency-Band PSI Combined.

Key insight: The phase-shuffled attack randomizes the GRADIENT's phase, not the input's.
We measure: (1) input-level band PSI, (2) gradient spectral power distribution.

Hypothesis: Models whose GRADIENT has more high-freq power are more vulnerable
to phase-shuffled attack (since attack destroys gradient phase coherence at high freqs).
"""
import os, sys, json, gc, random
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import zoom

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model
from utils import calculate_metric_percase

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


# =========================================================================
# Data loading (matching Exp14 preprocessing)
# =========================================================================

def get_synapse_test_images(n_images=5):
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


def get_isic_test_images(n_images=5):
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
        img_path = os.path.join(data_dir, str(row['img']))
        if os.path.exists(img_path):
            img_t = transform(Image.open(img_path).convert('RGB'))
            gt = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            if 'seg' in row:
                gt_path = os.path.join(data_dir, str(row['seg']))
                if os.path.exists(gt_path):
                    gt = (np.array(Image.open(gt_path).resize((IMAGE_SIZE, IMAGE_SIZE))) > 128).astype(np.float32)
            images.append((img_t, gt))
    return images


def get_refuge_test_images(n_images=5):
    from PIL import Image
    from torchvision import transforms
    test_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    if not os.path.isdir(test_dir): return []
    folders = sorted([f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))])
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = []
    for folder in folders:
        if len(images) >= n_images: break
        img_path = os.path.join(test_dir, folder, f"{folder}.jpg")
        disc_path = os.path.join(test_dir, folder, f"{folder}_disc.bmp")
        cup_path = os.path.join(test_dir, folder, f"{folder}_cup.bmp")
        if not os.path.exists(img_path) or not os.path.exists(disc_path) or not os.path.exists(cup_path):
            continue
        img_t = transform(Image.open(img_path).convert('RGB'))
        disc_raw = np.array(Image.open(disc_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        cup_raw = np.array(Image.open(cup_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        disc_mask = (disc_raw <= 128).astype(np.float32)
        cup_mask = (cup_raw < 128).astype(np.float32)
        label = np.stack([cup_mask, disc_mask], axis=0)
        images.append((img_t, label))
    return images


# =========================================================================
# Evaluation (matching Exp14)
# =========================================================================

def evaluate_samed(net, img_data, lbl, ds_name, device):
    """Unified SAMed evaluation."""
    if torch.is_tensor(img_data):
        img_t = img_data
    else:
        img_t = torch.from_numpy(img_data).float()
    inputs = img_t.unsqueeze(0).to(device)

    net.eval()
    with torch.no_grad():
        outputs = net(inputs, True, IMAGE_SIZE)
        masks = torch.softmax(outputs['masks'], dim=1).squeeze(0).cpu().numpy()

    if ds_name == "REFUGE":
        pred = np.argmax(masks, axis=0)
        if torch.is_tensor(lbl): lbl = lbl.cpu().numpy()
        disc_gt = (lbl[1] > 0.5).astype(np.float32)
        cup_gt = (lbl[0] > 0.5).astype(np.float32)
        disc_p = (pred == 1).astype(np.float32)
        cup_p = (pred == 2).astype(np.float32)
        d_disc = 2.0 * (disc_p * disc_gt).sum() / (disc_p.sum() + disc_gt.sum() + 1e-8)
        d_cup = 2.0 * (cup_p * cup_gt).sum() / (cup_p.sum() + cup_gt.sum() + 1e-8)
        return float((d_disc + d_cup) / 2.0)
    elif ds_name == "ISIC":
        out = np.argmax(masks, axis=0)
        pred_bin = (out > 0.5).astype(np.float32)
        gt_bin = (lbl > 0.5).astype(np.float32)
        if gt_bin.sum() == 0: return 0.0
        metric = calculate_metric_percase(pred_bin, gt_bin)
        return float(metric[0]) if isinstance(metric, (tuple, list, np.ndarray)) else float(metric)
    else:  # Synapse
        if lbl.ndim == 2: lbl = lbl[np.newaxis, :, :]
        out = np.argmax(masks, axis=0)
        prediction = np.tile(out[np.newaxis, :, :], (lbl.shape[0], 1, 1))
        ncls = max(2, 8)
        metric_list = []
        for i in range(1, ncls):
            gt_bin = (lbl == i)
            if gt_bin.sum() > 0:
                metric_list.append(calculate_metric_percase(prediction == i, gt_bin))
        if len(metric_list) == 0:
            metric_list = [calculate_metric_percase(out > 0.5, lbl > 0)]
        return float(np.mean(metric_list, axis=0)[0])


# =========================================================================
# Frequency-band phase perturbation
# =========================================================================

def phase_band_noise(img_tensor, band='high', cutoff_ratio=0.25):
    if img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0)
    _, C, H, W = img_tensor.shape
    F = torch.fft.fft2(img_tensor, norm='ortho')
    A = torch.abs(F)
    Phi = torch.angle(F)
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


# =========================================================================
# NEW: Gradient spectral analysis
# =========================================================================

def compute_gradient_spectral_psi(net, img_tensor, lbl, ds_name, device):
    """
    Compute PSI from the FFT spectrum of input gradient.

    The gradient of model output entropy w.r.t. input tells us which
    frequency components the model relies on for its predictions.
    High-freq gradient power → model relies on fine details → vulnerable to phase shuffling.
    """
    # Prepare input with gradient tracking
    if torch.is_tensor(img_tensor):
        img = img_tensor.clone().detach().float().to(device)
    else:
        img = torch.from_numpy(img_tensor).float().to(device)
    img = img.unsqueeze(0).requires_grad_(True)

    net.eval()
    outputs = net(img, True, IMAGE_SIZE)
    masks = outputs['masks']  # [1, C, H, W]

    # Pixel-wise entropy as loss (higher entropy = more uncertain)
    probs = F.softmax(masks, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

    # Backward pass
    grad = torch.autograd.grad(entropy, img, create_graph=False)[0]  # [1, 3, H, W]

    # FFT and spectral analysis
    grad_fft = torch.fft.fft2(grad, norm='ortho')
    power = torch.abs(grad_fft) ** 2  # [1, 3, H, W]

    H, W = power.shape[-2:]
    y_freq = torch.arange(H, device=device).float() - H // 2
    x_freq = torch.arange(W, device=device).float() - W // 2
    Y, X = torch.meshgrid(y_freq, x_freq, indexing='ij')
    R = torch.sqrt(Y**2 + X**2)
    R_max = min(H, W) / 2

    # Three frequency bands
    low_cut = 0.125 * R_max   # r < 32
    high_cut = 0.25 * R_max    # r > 64

    low_mask = R <= low_cut
    mid_mask = (R > low_cut) & (R <= high_cut)
    high_mask = R > high_cut

    low_power = power[:, :, low_mask].mean().item()
    mid_power = power[:, :, mid_mask].mean().item()
    high_power = power[:, :, high_mask].mean().item()
    total_power = low_power + mid_power + high_power + 1e-12

    psi_grad = high_power / total_power
    return psi_grad, low_power / total_power, mid_power / total_power, high_power / total_power


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("PAID Theory v6: Gradient Spectral PSI + Band PSI")
    print("=" * 70)

    device = torch.device("cuda:0")
    n_imgs = 3
    n_trials = 3
    cutoff = 0.25

    all_results = {}

    for ds_name, cfg in SAMED_CFG.items():
        for vit_name, (ckpt_rel, ncls) in cfg.items():
            target_name = f"{ds_name}_SAMed_{vit_name}"
            ckpt_path = os.path.join(CKPT_BASE, ckpt_rel)
            if not os.path.exists(ckpt_path):
                print(f"SKIP {target_name}: no checkpoint")
                continue

            print(f"\n{'=' * 60}")
            print(f"  {target_name} (classes={ncls})")
            print(f"{'=' * 60}")

            try:
                sam_ckpt = SAM_CKPTS[vit_name]
                net = load_samed_model(vit_name, sam_ckpt, ckpt_path, IMAGE_SIZE, ncls, device=device)

                if ds_name == "Synapse": test_imgs = get_synapse_test_images(n_imgs)
                elif ds_name == "ISIC": test_imgs = get_isic_test_images(n_imgs)
                else: test_imgs = get_refuge_test_images(n_imgs)

                if not test_imgs:
                    print(f"  SKIP: no test images")
                    continue

                print(f"  Loaded {len(test_imgs)} test images")

                all_psi_band, all_high_d, all_low_d, all_clean = [], [], [], []
                all_psi_grad = []

                for img_idx, (img_data, lbl) in enumerate(test_imgs):
                    if torch.is_tensor(img_data):
                        img_t = img_data.unsqueeze(0).float().to(device)
                    else:
                        img_t = torch.from_numpy(img_data).unsqueeze(0).float().to(device)

                    # (A) Band PSI
                    clean_d = evaluate_samed(net, img_data, lbl, ds_name, device)
                    if clean_d < 0.01:
                        print(f"    img={img_idx + 1}: SKIP (clean={clean_d:.4f})")
                        continue

                    high_deltas, low_deltas = [], []
                    for _ in range(n_trials):
                        img_high = phase_band_noise(img_t, band='high', cutoff_ratio=cutoff)
                        dice_high = evaluate_samed(net, img_high[0], lbl, ds_name, device)
                        high_deltas.append(max(0, clean_d - dice_high))

                        img_low = phase_band_noise(img_t, band='low', cutoff_ratio=cutoff)
                        dice_low = evaluate_samed(net, img_low[0], lbl, ds_name, device)
                        low_deltas.append(max(0, clean_d - dice_low))

                    mean_high = np.mean(high_deltas)
                    mean_low = np.mean(low_deltas)
                    psi_band = mean_high / (mean_high + mean_low + 1e-12)

                    # (B) Gradient spectral PSI
                    psi_grad, low_frac, mid_frac, high_frac = compute_gradient_spectral_psi(
                        net, img_data, lbl, ds_name, device)

                    all_psi_band.append(psi_band)
                    all_high_d.append(mean_high)
                    all_low_d.append(mean_low)
                    all_clean.append(clean_d)
                    all_psi_grad.append(psi_grad)

                    print(f"    img={img_idx + 1}: band_PSI={psi_band:.4f} (hΔ={mean_high:.4f}, lΔ={mean_low:.4f}, cl={clean_d:.4f}) "
                          f"grad_PSI={psi_grad:.4f} (L={low_frac:.3f} M={mid_frac:.3f} H={high_frac:.3f})")

                if len(all_clean) == 0:
                    continue

                all_results[target_name] = {
                    "psi_band": float(np.mean(all_psi_band)),
                    "high_delta": float(np.mean(all_high_d)),
                    "low_delta": float(np.mean(all_low_d)),
                    "clean_dice": float(np.mean(all_clean)),
                    "psi_grad": float(np.mean(all_psi_grad)),
                    "arch": "SAMed", "dataset": ds_name, "vit": vit_name,
                    "n_images": len(all_clean),
                }
                print(f"  => band_PSI={all_results[target_name]['psi_band']:.4f} "
                      f"high_Δ={all_results[target_name]['high_delta']:.4f} "
                      f"grad_PSI={all_results[target_name]['psi_grad']:.4f}")

                del net; gc.collect(); torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
                all_results[target_name] = {"error": str(e)}

    # Save
    with open(os.path.join(OUTPUT_DIR, "paid_psi_v6_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # =====================================================================
    # VALIDATION
    # =====================================================================
    print("\n" + "=" * 70)
    print("VALIDATION: PSI metrics vs Exp14 Phase-Shuffled ΔDice")
    print("=" * 70)

    exp14_path = os.path.join(OUTPUT_DIR, "exp14_full_matrix.json")
    with open(exp14_path) as f:
        exp14 = json.load(f)

    ps_deltas = {}
    for model_name, conditions in exp14.get("results", {}).items():
        if not isinstance(conditions, dict): continue
        for cond_name, cond_val in conditions.items():
            if "phase_shuffled" in cond_name.lower():
                v = cond_val.get("mean_delta", 0) if isinstance(cond_val, dict) else float(cond_val or 0)
                ps_deltas[model_name] = float(v) if v else 0

    pairs = []
    for name, data in all_results.items():
        if "error" in data: continue
        if name in ps_deltas and data.get("clean_dice", 0) > 0.01:
            pairs.append({
                "model": name, "high_delta": data["high_delta"],
                "psi_grad": data["psi_grad"], "clean_dice": data["clean_dice"],
                "exp14_delta": ps_deltas[name], "dataset": data.get("dataset", ""),
            })

    if len(pairs) >= 5:
        for metric_name, metric_key in [("high_ΔDice (band)", "high_delta"), ("gradient PSI", "psi_grad")]:
            vals = [p[metric_key] for p in pairs]
            exp14_vals = [p["exp14_delta"] for p in pairs]
            r, pv = pearsonr(vals, exp14_vals)
            rs, ps = spearmanr(vals, exp14_vals)
            print(f"\n  {metric_name}: Pearson r={r:.4f} (p={pv:.4f}), Spearman ρ={rs:.4f} (p={ps:.4f})")

        print(f"\n  {'Model':<30s} {'band_Δ':>10s} {'grad_PSI':>10s} {'clean':>10s} {'Exp14_Δ':>10s}")
        print(f"  {'-' * 72}")
        for p in sorted(pairs, key=lambda x: x["high_delta"], reverse=True):
            print(f"  {p['model']:<30s} {p['high_delta']:10.4f} {p['psi_grad']:10.4f} {p['clean_dice']:10.4f} {p['exp14_delta']:10.4f}")

    return all_results


if __name__ == "__main__":
    main()
