#!/usr/bin/env python3
"""
PAID Theory v3: Frequency-Band Phase Sensitivity.

Key insight: The hallmark of phase-dependent models is sensitivity to HIGH-FREQ
phase disruption. Phase shuffling destroys all phase structure, but a model that
relies on low-freq phase (gross structure) will be less affected than one that
relies on high-freq phase (fine details, textures, edges).

Method:
1. Split FFT into low-freq (center) and high-freq (periphery) bands
2. Randomize phase in each band separately
3. Measure ΔDice for low-band phase noise vs high-band phase noise
4. PSI_band = ΔDice_highband / (ΔDice_highband + ΔDice_lowband)

Hypothesis: Models with high PSI_band are more vulnerable to full phase-shuffled attack.
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
        "vit_l": ("SAMed/isic_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 1),
        "vit_h": ("SAMed/isic_512_lrexp3.0_pretrain_vit_h_epo300_bs4_lr0.005/epoch_159.pth", 1),
    },
    "REFUGE": {
        "vit_b": ("SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 2),
        "vit_l": ("SAMed/REFUGE_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 2),
        "vit_h": ("SAMed/REFUGE_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_159.pth", 2),
    },
}


def evaluate_slice(net, image_np, label_np, num_classes=8):
    """Device-safe SAMed evaluation."""
    device = next(net.parameters()).device
    if image_np.ndim == 2: image_np = image_np[np.newaxis, :, :]
    if label_np.ndim == 2: label_np = label_np[np.newaxis, :, :]
    prediction = np.zeros_like(label_np)
    for ind in range(image_np.shape[0]):
        slc = image_np[ind, :, :]
        x, y = slc.shape
        if x != IMAGE_SIZE or y != IMAGE_SIZE:
            slc = zoom(slc, (IMAGE_SIZE / x, IMAGE_SIZE / y), order=3)
        inputs = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).float().to(device)
        inputs = inputs.repeat(1, 3, 1, 1)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, True, IMAGE_SIZE)
            out = torch.argmax(torch.softmax(outputs['masks'], dim=1), dim=1).squeeze(0).cpu().numpy()
            if x != out.shape[0]:
                pred = zoom(out, (x / out.shape[0], y / out.shape[1]), order=0)
            else:
                pred = out
            prediction[ind] = pred
    ncls = max(2, num_classes)
    metric_list = [calculate_metric_percase(prediction == i, label_np == i) for i in range(1, ncls)]
    if len(metric_list) == 0:
        metric_list = [calculate_metric_percase(prediction > 0.5, label_np > 0)]
    return float(np.mean(metric_list, axis=0)[0])


def phase_band_noise(img_tensor, band='high', cutoff_ratio=0.25, noise_std=1.0):
    """
    Randomize phase in low or high frequency band.

    Args:
        img_tensor: [1, 3, H, W]
        band: 'low' or 'high' or 'all'
        cutoff_ratio: radius threshold as fraction of min(H,W)/2
        noise_std: std of Gaussian noise added (1.0 = full randomization)

    Returns perturbed image tensor.
    """
    _, C, H, W = img_tensor.shape
    F = torch.fft.fft2(img_tensor, norm='ortho')
    A = torch.abs(F)
    Phi = torch.angle(F)

    # Create frequency grid
    y_freq = torch.arange(H, device=img_tensor.device).float() - H // 2
    x_freq = torch.arange(W, device=img_tensor.device).float() - W // 2
    Y, X = torch.meshgrid(y_freq, x_freq, indexing='ij')
    R = torch.sqrt(Y**2 + X**2)

    cutoff = cutoff_ratio * min(H, W) / 2

    # Create mask for the band
    if band == 'low':
        mask = R <= cutoff  # [H, W]
    elif band == 'high':
        mask = R > cutoff
    else:  # 'all'
        mask = torch.ones(H, W, device=img_tensor.device, dtype=torch.bool)

    # Apply phase noise to selected band
    delta_phi = torch.randn_like(Phi) * noise_std
    delta_phi[:, :, ~mask] = 0.0  # zero out non-target band

    F_pert = A * torch.exp(1j * (Phi + delta_phi))
    img_pert = torch.fft.ifft2(F_pert, norm='ortho').real
    return torch.clamp(img_pert, img_tensor.min(), img_tensor.max())


def get_synapse_test_images(n_images=5):
    """Load Synapse test slices with labels."""
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
            images.append((img[s].astype(np.float32), lbl[s]))
    return images


def get_isic_test_images(n_images=5):
    """Load ISIC test images with segmentation labels."""
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    csv_path = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv")
    if not os.path.exists(csv_path): return []
    df = pd.read_csv(csv_path)
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = []
    for _, row in df.iterrows():
        if len(images) >= n_images: break
        img_path = os.path.join(data_dir, str(row['img']))
        if os.path.exists(img_path):
            img = (transform(Image.open(img_path).convert('RGB'))[0] * 255).numpy()
            gt = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            if 'seg' in row:
                gt_path = os.path.join(data_dir, str(row['seg']))
                if os.path.exists(gt_path):
                    gt = (np.array(Image.open(gt_path).resize((IMAGE_SIZE, IMAGE_SIZE))) > 128).astype(np.float32)
            images.append((img.astype(np.float32), gt))
    return images


def get_refuge_test_images(n_images=5):
    """Load REFUGE test images."""
    from PIL import Image
    from torchvision import transforms
    test_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    if not os.path.isdir(test_dir): return []
    folders = sorted([f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))])[:n_images]
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = []
    for folder in folders:
        img_path = os.path.join(test_dir, folder, f"{folder}.jpg")
        gt_path = os.path.join(test_dir, folder, f"{folder}_seg.png")
        if os.path.exists(img_path):
            img = (transform(Image.open(img_path).convert('RGB'))[0] * 255).numpy()
            gt = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            if os.path.exists(gt_path):
                gt = (np.array(Image.open(gt_path).resize((IMAGE_SIZE, IMAGE_SIZE))) > 128).astype(np.float32)
            images.append((img.astype(np.float32), gt))
    return images


def compute_band_psi(net, arch, image_np, label_np, num_classes, n_trials=5, cutoff_ratio=0.25):
    """
    Compute frequency-band PSI.

    Measures ΔDice for:
    - High-frequency phase noise (randomize periphery phase)
    - Low-frequency phase noise (randomize center phase)

    PSI_band = ΔDice_high / (ΔDice_high + ΔDice_low)
    """
    device = next(net.parameters()).device

    # Clean baseline
    clean_dice = evaluate_slice(net, image_np, label_np, num_classes)

    high_deltas = []
    low_deltas = []

    for _ in range(n_trials):
        # Prepare image tensor
        img_3ch = np.stack([image_np if image_np.ndim == 2 else image_np[:,:,0]]*3, axis=0)
        if image_np.ndim == 3 and image_np.shape[0] >= 3:
            img_3ch = image_np[:3]
        img_t = torch.from_numpy(img_3ch).unsqueeze(0).float().to(device)

        # High-freq phase noise
        img_high = phase_band_noise(img_t, band='high', cutoff_ratio=cutoff_ratio, noise_std=1.0)
        dice_high = evaluate_slice(net, img_high[0, 0].cpu().numpy(), label_np, num_classes)
        high_deltas.append(max(0, clean_dice - dice_high))

        # Low-freq phase noise
        img_low = phase_band_noise(img_t, band='low', cutoff_ratio=cutoff_ratio, noise_std=1.0)
        dice_low = evaluate_slice(net, img_low[0, 0].cpu().numpy(), label_np, num_classes)
        low_deltas.append(max(0, clean_dice - dice_low))

    mean_high = np.mean(high_deltas)
    mean_low = np.mean(low_deltas)

    psi_band = mean_high / (mean_high + mean_low + 1e-12)

    return psi_band, mean_high, mean_low, clean_dice


def main():
    print("="*70)
    print("PAID Theory v3: Frequency-Band Phase Sensitivity across SAMed Models")
    print("="*70)

    # Build available models
    models = []
    for ds_name, cfg in SAMED_CFG.items():
        for vit_name, (ckpt_rel, ncls) in cfg.items():
            ckpt_path = os.path.join(CKPT_BASE, ckpt_rel)
            if os.path.exists(ckpt_path):
                models.append(("SAMed", ds_name, vit_name, ckpt_path, ncls))

    print(f"Found {len(models)} SAMed models")

    all_results = {}
    n_imgs = 4  # images per model
    n_trials = 5  # noise trials per condition per image
    cutoff_ratios = [0.125, 0.25, 0.5]  # test multiple cutoffs

    for arch, ds, vit, ckpt_path, num_classes in models:
        target_name = f"{ds}_{arch}_{vit}"

        print(f"\n{'='*60}")
        print(f"  {target_name} (classes={num_classes})")
        print(f"{'='*60}")

        try:
            device = torch.device("cuda:0")  # single GPU to avoid issues
            sam_ckpt = SAM_CKPTS[vit]
            net = load_samed_model(vit, sam_ckpt, ckpt_path, IMAGE_SIZE, num_classes, device=device)

            if ds == "Synapse":
                test_imgs = get_synapse_test_images(n_imgs)
            elif ds == "ISIC":
                test_imgs = get_isic_test_images(n_imgs)
            else:
                test_imgs = get_refuge_test_images(n_imgs)

            if not test_imgs:
                print(f"  SKIP: no test images")
                all_results[target_name] = {"error": "no test images"}
                continue

            print(f"  Loaded {len(test_imgs)} test images")

            best_cutoff_results = {}
            for cutoff in cutoff_ratios:
                all_psi = []
                all_high = []
                all_low = []
                for img_idx, (img_np, lbl_np) in enumerate(test_imgs):
                    psi, high_d, low_d, clean_d = compute_band_psi(
                        net, arch, img_np, lbl_np, num_classes,
                        n_trials=n_trials, cutoff_ratio=cutoff
                    )
                    all_psi.append(psi)
                    all_high.append(high_d)
                    all_low.append(low_d)
                    print(f"    cutoff={cutoff:.3f} img={img_idx+1}: PSI={psi:.4f} (high_Δ={high_d:.4f}, low_Δ={low_d:.4f}, clean={clean_d:.4f})")

                best_cutoff_results[f"cutoff_{cutoff}"] = {
                    "mean_psi": float(np.mean(all_psi)),
                    "std_psi": float(np.std(all_psi)),
                    "mean_high_delta": float(np.mean(all_high)),
                    "mean_low_delta": float(np.mean(all_low)),
                }

            # Use cutoff=0.25 as primary
            primary = best_cutoff_results["cutoff_0.25"]
            all_results[target_name] = {
                "psi_band": primary["mean_psi"],
                "high_delta": primary["mean_high_delta"],
                "low_delta": primary["mean_low_delta"],
                "n_images": len(test_imgs),
                "all_cutoffs": best_cutoff_results,
            }
            print(f"  => PSI_band={primary['mean_psi']:.4f} (high_Δ={primary['mean_high_delta']:.4f}, low_Δ={primary['mean_low_delta']:.4f})")

            del net; gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[target_name] = {"error": str(e)}

    # Save
    with open(os.path.join(OUTPUT_DIR, "paid_psi_v3_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ============================================================
    # VALIDATION
    # ============================================================
    print("\n" + "="*70)
    print("VALIDATION: Band-PSI vs Phase-Shuffled Attack ΔDice")
    print("="*70)

    # Load Exp14 results
    exp14_path = os.path.join(OUTPUT_DIR, "exp14_full_matrix.json")
    with open(exp14_path) as f:
        exp14 = json.load(f)

    ps_deltas = {}
    for entry in exp14.get("entries", []):
        ds = entry.get("dataset", "")
        arch = entry.get("arch", "").upper()
        vit = entry.get("vit", "")
        model_name = f"{ds}_{arch}_{vit}"
        for cond_name, cond_val in entry.get("conditions", {}).items():
            if "phase_shuffled" in cond_name.lower():
                if isinstance(cond_val, dict):
                    v = cond_val.get("mean_delta_dice", cond_val.get("mean_delta", 0))
                else:
                    v = float(cond_val) if cond_val else 0
                ps_deltas[model_name] = float(v) if v else 0

    pairs = []
    for name, data in all_results.items():
        if "error" in data: continue
        if name in ps_deltas:
            pairs.append({"model": name, "psi": data["psi_band"], "delta_dice": ps_deltas[name]})

    if len(pairs) >= 5:
        psis = [p["psi"] for p in pairs]
        deltas = [p["delta_dice"] for p in pairs]
        r_p, p_p = pearsonr(psis, deltas)
        r_s, p_s = spearmanr(psis, deltas)

        print(f"\n  Pearson r = {r_p:.4f} (p={p_p:.4f})")
        print(f"  Spearman ρ = {r_s:.4f} (p={p_s:.4f})")
        print(f"\n  {'Model':<30s} {'PSI_band':>10s} {'ΔDice':>10s}")
        print(f"  {'-'*52}")
        for p in sorted(pairs, key=lambda x: x["psi"]):
            print(f"  {p['model']:<30s} {p['psi']:10.4f} {p['delta_dice']:10.4f}")

        assessment = "strong" if r_s > 0.7 else "moderate" if r_s > 0.5 else "weak"
        print(f"\n  Assessment: {assessment.upper()} (ρ={r_s:.3f})")

        with open(os.path.join(OUTPUT_DIR, "paid_correlation_v3.json"), "w") as f:
            json.dump({"pairs": pairs, "pearson_r": r_p, "spearman_r": r_s, "n": len(pairs), "assessment": assessment}, f, indent=2)
    else:
        print(f"  Only {len(pairs)} pairs — insufficient")

    return all_results


if __name__ == "__main__":
    main()
