#!/usr/bin/env python3
"""
PAID Theory v5: Frequency-Band PSI with Corrected Input Processing.

Matches Exp14 exactly:
- Synapse: raw [0,255] grayscale stacked to 3ch
- ISIC: ImageNet-normalized RGB
- REFUGE: ImageNet-normalized RGB

Uses same data loading as Exp14.
"""
import os, sys, json, gc, random
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import zoom

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model, load_medsa_model
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
# Data loading — matching Exp14 exactly
# =========================================================================

def get_synapse_test_images(n_images=5):
    """Synapse: raw CT values in [0,255], stacked to 3ch. Matches Exp14."""
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])
    images = []
    for h5f in h5s:
        if len(images) >= n_images:
            break
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 500],
                       key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:1]
        for s in valid:
            if len(images) >= n_images:
                break
            slc = img[s].astype(np.float32)
            # Stack grayscale to 3ch (matching Exp14: Lambda(x*255) gives [0,255])
            img_3ch = np.stack([slc, slc, slc], axis=0)  # [3, H, W]
            images.append((img_3ch, lbl[s]))
    return images


def get_isic_test_images(n_images=5):
    """ISIC: ImageNet-normalized RGB. Matches Exp14."""
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    csv_path = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = []
    for _, row in df.iterrows():
        if len(images) >= n_images:
            break
        img_path = os.path.join(data_dir, str(row['img']))
        if os.path.exists(img_path):
            img_t = transform(Image.open(img_path).convert('RGB'))  # [3, H, W] normalized
            gt = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            if 'seg' in row:
                gt_path = os.path.join(data_dir, str(row['seg']))
                if os.path.exists(gt_path):
                    gt = (np.array(Image.open(gt_path).resize((IMAGE_SIZE, IMAGE_SIZE))) > 128).astype(np.float32)
            images.append((img_t, gt))
    return images


def get_refuge_test_images(n_images=5):
    """REFUGE: ImageNet-normalized RGB. Returns (img_tensor, label_2ch).
    label: (2, H, W) [cup, disc] float32
    Matches Exp14 exactly.
    """
    from PIL import Image
    from torchvision import transforms
    test_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    if not os.path.isdir(test_dir):
        return []
    folders = sorted([f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))])
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = []
    for folder in folders:
        if len(images) >= n_images:
            break
        img_path = os.path.join(test_dir, folder, f"{folder}.jpg")
        disc_path = os.path.join(test_dir, folder, f"{folder}_disc.bmp")
        cup_path = os.path.join(test_dir, folder, f"{folder}_cup.bmp")
        if not os.path.exists(img_path) or not os.path.exists(disc_path) or not os.path.exists(cup_path):
            continue
        img_t = transform(Image.open(img_path).convert('RGB'))  # [3, H, W] normalized
        disc_raw = np.array(Image.open(disc_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        cup_raw = np.array(Image.open(cup_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        disc_mask = (disc_raw <= 128).astype(np.float32)
        cup_mask = (cup_raw < 128).astype(np.float32)
        label = np.stack([cup_mask, disc_mask], axis=0)  # [cup, disc]
        images.append((img_t, label))
    return images


# =========================================================================
# Evaluation functions
# =========================================================================

def evaluate_samed_synapse(net, img_data, label_np, device):
    """Evaluate SAMed Synapse. img_data: numpy [3, H, W] in [0,255] or tensor."""
    if torch.is_tensor(img_data):
        img_tensor = img_data
    else:
        img_tensor = torch.from_numpy(img_data).float()
    if label_np.ndim == 2:
        label_np = label_np[np.newaxis, :, :]
    prediction = np.zeros_like(label_np)
    for ind in range(label_np.shape[0]):
        inputs = img_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, True, IMAGE_SIZE)
            out = torch.argmax(torch.softmax(outputs['masks'], dim=1), dim=1).squeeze(0).cpu().numpy()
            prediction[ind] = out
    ncls = max(2, 8)
    metric_list = []
    for i in range(1, ncls):
        gt_bin = (label_np == i)
        if gt_bin.sum() > 0:
            metric_list.append(calculate_metric_percase(prediction == i, gt_bin))
    if len(metric_list) == 0:
        metric_list = [calculate_metric_percase(prediction > 0.5, label_np > 0)]
    return float(np.mean(metric_list, axis=0)[0])


def evaluate_samed_isic(net, img_tensor, gt_mask, device):
    """Evaluate SAMed ISIC with normalized RGB tensor [3, H, W]."""
    inputs = img_tensor.unsqueeze(0).to(device)
    net.eval()
    with torch.no_grad():
        outputs = net(inputs, True, IMAGE_SIZE)
        out = torch.argmax(torch.softmax(outputs['masks'], dim=1), dim=1).squeeze(0).cpu().numpy()
    pred_bin = (out > 0.5).astype(np.float32)
    gt_bin = (gt_mask > 0.5).astype(np.float32)
    if gt_bin.sum() == 0:
        return 0.0
    metric = calculate_metric_percase(pred_bin, gt_bin)
    return float(metric[0]) if isinstance(metric, (tuple, list, np.ndarray)) else float(metric)


def evaluate_samed_refuge(net, img_tensor, label, device):
    """Evaluate SAMed REFUGE. img_tensor: [3, H, W] normalized, label: [2, H, W] [cup, disc]."""
    inputs = img_tensor.unsqueeze(0).to(device)
    net.eval()
    with torch.no_grad():
        outputs = net(inputs, True, IMAGE_SIZE)
        masks = torch.softmax(outputs['masks'], dim=1).squeeze(0).cpu().numpy()
    pred = np.argmax(masks, axis=0)
    if torch.is_tensor(label):
        label = label.cpu().numpy()
    disc_gt = (label[1] > 0.5).astype(np.float32)
    cup_gt = (label[0] > 0.5).astype(np.float32)
    disc_p = (pred == 1).astype(np.float32)
    cup_p = (pred == 2).astype(np.float32)
    d_disc = 2.0 * (disc_p * disc_gt).sum() / (disc_p.sum() + disc_gt.sum() + 1e-8)
    d_cup = 2.0 * (cup_p * cup_gt).sum() / (cup_p.sum() + cup_gt.sum() + 1e-8)
    return float((d_disc + d_cup) / 2.0)


# =========================================================================
# Phase band perturbation (operates on tensors)
# =========================================================================

def phase_band_noise(img_tensor, band='high', cutoff_ratio=0.25):
    """Randomize phase in frequency band of image tensor [1, C, H, W] or [C, H, W]."""
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    _, C, H, W = img_tensor.shape
    F = torch.fft.fft2(img_tensor, norm='ortho')
    A = torch.abs(F)
    Phi = torch.angle(F)

    y_freq = torch.arange(H, device=img_tensor.device).float() - H // 2
    x_freq = torch.arange(W, device=img_tensor.device).float() - W // 2
    Y, X = torch.meshgrid(y_freq, x_freq, indexing='ij')
    R = torch.sqrt(Y ** 2 + X ** 2)
    cutoff = cutoff_ratio * min(H, W) / 2

    if band == 'low':
        mask = R <= cutoff
    elif band == 'high':
        mask = R > cutoff
    else:
        mask = torch.ones(H, W, device=img_tensor.device, dtype=torch.bool)

    delta_phi = torch.randn_like(Phi)
    delta_phi[:, :, ~mask] = 0.0
    F_pert = A * torch.exp(1j * (Phi + delta_phi))
    img_pert = torch.fft.ifft2(F_pert, norm='ortho').real
    # Clamp to original range
    return torch.clamp(img_pert, img_tensor.min(), img_tensor.max())


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("PAID Theory v5: Frequency-Band PSI — Corrected Input Processing")
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

                if ds_name == "Synapse":
                    test_imgs = get_synapse_test_images(n_imgs)
                elif ds_name == "ISIC":
                    test_imgs = get_isic_test_images(n_imgs)
                else:
                    test_imgs = get_refuge_test_images(n_imgs)

                if not test_imgs:
                    print(f"  SKIP: no test images")
                    all_results[target_name] = {"error": "no test images"}
                    continue

                print(f"  Loaded {len(test_imgs)} test images")

                all_psi, all_high, all_low, all_clean = [], [], [], []
                for img_idx, (img_data, lbl) in enumerate(test_imgs):
                    # img_data is either numpy [3,H,W] or tensor [3,H,W]
                    if torch.is_tensor(img_data):
                        img_t = img_data.unsqueeze(0).float().to(device)
                    else:
                        img_t = torch.from_numpy(img_data).unsqueeze(0).float().to(device)

                    # Clean baseline
                    if ds_name == "Synapse":
                        clean_d = evaluate_samed_synapse(net, img_data, lbl, device)
                    elif ds_name == "ISIC":
                        clean_d = evaluate_samed_isic(net, img_data, lbl, device)
                    else:
                        clean_d = evaluate_samed_refuge(net, img_data, lbl, device)

                    if clean_d < 0.01:
                        print(f"    img={img_idx + 1}: SKIP (clean={clean_d:.4f})")
                        continue

                    high_deltas, low_deltas = [], []
                    for _ in range(n_trials):
                        # High-freq phase noise
                        img_high = phase_band_noise(img_t, band='high', cutoff_ratio=cutoff)
                        if ds_name == "Synapse":
                            dice_high = evaluate_samed_synapse(net, img_high[0], lbl, device)
                        elif ds_name == "ISIC":
                            dice_high = evaluate_samed_isic(net, img_high[0], lbl, device)
                        else:
                            dice_high = evaluate_samed_refuge(net, img_high[0], lbl, device)
                        high_deltas.append(max(0, clean_d - dice_high))

                        # Low-freq phase noise
                        img_low = phase_band_noise(img_t, band='low', cutoff_ratio=cutoff)
                        if ds_name == "Synapse":
                            dice_low = evaluate_samed_synapse(net, img_low[0], lbl, device)
                        elif ds_name == "ISIC":
                            dice_low = evaluate_samed_isic(net, img_low[0], lbl, device)
                        else:
                            dice_low = evaluate_samed_refuge(net, img_low[0], lbl, device)
                        low_deltas.append(max(0, clean_d - dice_low))

                    mean_high = np.mean(high_deltas)
                    mean_low = np.mean(low_deltas)
                    psi = mean_high / (mean_high + mean_low + 1e-12)
                    all_psi.append(psi)
                    all_high.append(mean_high)
                    all_low.append(mean_low)
                    all_clean.append(clean_d)
                    print(f"    img={img_idx + 1}: PSI={psi:.4f} (high_Δ={mean_high:.4f}, low_Δ={mean_low:.4f}, clean={clean_d:.4f})")

                if len(all_clean) == 0:
                    all_results[target_name] = {"error": "all clean=0"}
                    continue

                all_results[target_name] = {
                    "psi_band": float(np.mean(all_psi)),
                    "high_delta": float(np.mean(all_high)),
                    "low_delta": float(np.mean(all_low)),
                    "clean_dice": float(np.mean(all_clean)),
                    "arch": "SAMed",
                    "dataset": ds_name,
                    "vit": vit_name,
                    "n_images": len(all_clean),
                }
                print(f"  => PSI={all_results[target_name]['psi_band']:.4f} "
                      f"high_Δ={all_results[target_name]['high_delta']:.4f} "
                      f"low_Δ={all_results[target_name]['low_delta']:.4f} "
                      f"clean={all_results[target_name]['clean_dice']:.4f}")

                del net
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results[target_name] = {"error": str(e)}

    # Save
    with open(os.path.join(OUTPUT_DIR, "paid_psi_v5_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # =====================================================================
    # VALIDATION
    # =====================================================================
    print("\n" + "=" * 70)
    print("VALIDATION: High-ΔDice vs Exp14 Phase-Shuffled ΔDice")
    print("=" * 70)

    exp14_path = os.path.join(OUTPUT_DIR, "exp14_full_matrix.json")
    if not os.path.exists(exp14_path):
        print("Exp14 results not found — skipping")
        return all_results

    with open(exp14_path) as f:
        exp14 = json.load(f)

    ps_deltas = {}
    results = exp14.get("results", {})
    for model_name, conditions in results.items():
        if not isinstance(conditions, dict):
            continue
        for cond_name, cond_val in conditions.items():
            if "phase_shuffled" in cond_name.lower():
                if isinstance(cond_val, dict):
                    v = cond_val.get("mean_delta", 0)
                else:
                    v = float(cond_val) if cond_val else 0
                ps_deltas[model_name] = float(v) if v else 0
    print(f"Loaded {len(ps_deltas)} Exp14 phase-shuffled ΔDice entries")

    pairs = []
    for name, data in all_results.items():
        if "error" in data:
            continue
        if name in ps_deltas and data.get("clean_dice", 0) > 0.01:
            pairs.append({
                "model": name,
                "high_delta": data["high_delta"],
                "clean_dice": data["clean_dice"],
                "exp14_delta": ps_deltas[name],
                "dataset": data.get("dataset", ""),
            })

    print(f"\n  Valid pairs: {len(pairs)}")
    if len(pairs) >= 5:
        hd = [p["high_delta"] for p in pairs]
        ed = [p["exp14_delta"] for p in pairs]
        r_hd, p_hd = pearsonr(hd, ed)
        r_s, p_s = spearmanr(hd, ed)
        print(f"  Using high_ΔDice: Pearson r={r_hd:.4f} (p={p_hd:.4f}), Spearman ρ={r_s:.4f} (p={p_s:.4f})")

        # Normalized
        hd_n = [p["high_delta"] / max(p["clean_dice"], 0.01) for p in pairs]
        r_n, p_n = pearsonr(hd_n, ed)
        r_sn, p_sn = spearmanr(hd_n, ed)
        print(f"  Using high_Δ/clean: Pearson r={r_n:.4f} (p={p_n:.4f}), Spearman ρ={r_sn:.4f} (p={p_sn:.4f})")

        print(f"\n  {'Model':<30s} {'high_Δ':>10s} {'clean':>10s} {'Exp14_Δ':>10s}")
        print(f"  {'-' * 62}")
        for p in sorted(pairs, key=lambda x: x["high_delta"], reverse=True):
            print(f"  {p['model']:<30s} {p['high_delta']:10.4f} {p['clean_dice']:10.4f} {p['exp14_delta']:10.4f}")

        assessment = "strong" if r_s > 0.7 else "moderate" if r_s > 0.5 else "weak"
        print(f"\n  Assessment: {assessment.upper()} (ρ={r_s:.3f})")

        with open(os.path.join(OUTPUT_DIR, "paid_correlation_v5.json"), "w") as f:
            json.dump({"pairs": pairs, "pearson_r": r_hd, "spearman_r": r_s, "n": len(pairs), "assessment": assessment}, f, indent=2)
    else:
        print(f"\n  Insufficient pairs ({len(pairs)}). Listing all:")
        for p in sorted(pairs, key=lambda x: x["high_delta"], reverse=True):
            print(f"  {p['model']:<30s} high_Δ={p['high_delta']:.4f} clean={p['clean_dice']:.4f} exp14_Δ={p['exp14_delta']:.4f}")

    return all_results


if __name__ == "__main__":
    main()
