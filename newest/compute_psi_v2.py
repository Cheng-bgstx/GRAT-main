#!/usr/bin/env python3
"""
PAID Theory v2: Task-Level PSI Computation.

Key insight: PSI must be measured on the TASK OUTPUT (segmentation), not features.
Small phase vs amplitude perturbations to INPUT IMAGE → measure ΔDice.
Linear sensitivity (PSI) should predict nonlinear disruption vulnerability.

Theory: PSI = ΔDice(phase_noise) / (ΔDice(phase_noise) + ΔDice(amplitude_noise))

Prediction: Models with higher PSI are more vulnerable to phase-shuffled attack.
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
        "vit_b": ("Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 8),
        "vit_l": ("Synapse_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 8),
        "vit_h": ("Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_299.pth", 8),
    },
    "ISIC": {
        "vit_b": ("isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 1),
        "vit_l": ("isic_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 1),
        "vit_h": ("isic_512_lrexp3.0_pretrain_vit_h_epo300_bs4_lr0.005/epoch_159.pth", 1),
    },
    "REFUGE": {
        "vit_b": ("REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 2),
        "vit_l": ("REFUGE_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 2),
        "vit_h": ("REFUGE_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_159.pth", 2),
    },
}

MSA_CFG = {
    "Synapse": {
        "vit_b": ("MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 9),
        "vit_l": ("MedicalSAMAdapter/Synapse_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 9),
        "vit_h": ("MedicalSAMAdapter/Synapse_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24/Model/checkpoint_best.pth", 9),
    },
    "ISIC": {
        "vit_b": ("MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 1),
        "vit_l": ("MedicalSAMAdapter/isic_512_pretrain_vit_l_epo100_bs16_lr0.0001_s24/Model/checkpoint_best.pth", 1),
        "vit_h": ("MedicalSAMAdapter/isic_512_pretrain_vit_h_epo100_bs8_lr0.0001_s24/Model/checkpoint_best.pth", 1),
    },
    "REFUGE": {
        "vit_b": ("MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 2),
        "vit_l": ("MedicalSAMAdapter/REFUGE_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 2),
        "vit_h": ("MedicalSAMAdapter/REFUGE_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24/Model/checkpoint_best.pth", 2),
    },
}

# GPU assignment
GPU_MAP = {}
for gpu_id, ds in enumerate(["Synapse", "ISIC", "REFUGE"]):
    for arch in ["SAMed", "MSA"]:
        for vit in ["vit_b", "vit_l", "vit_h"]:
            GPU_MAP[f"{ds}_{arch}_{vit}"] = gpu_id


# ============================================================
# TASK-LEVEL PSI: Perturb input image FFT → measure ΔDice
# ============================================================
def perturb_fft_phase(img_tensor, noise_std=0.2):
    """Add Gaussian noise to FFT phase of image, keep magnitude."""
    # img_tensor: [1, 3, H, W] in [0, 255]
    F = torch.fft.fft2(img_tensor, norm='ortho')
    A = torch.abs(F)
    Phi = torch.angle(F)
    delta_phi = torch.randn_like(Phi) * noise_std
    F_pert = A * torch.exp(1j * (Phi + delta_phi))
    img_pert = torch.fft.ifft2(F_pert, norm='ortho').real
    return torch.clamp(img_pert, img_tensor.min(), img_tensor.max())


def perturb_fft_amplitude(img_tensor, noise_std=0.2):
    """Add Gaussian noise to FFT magnitude of image, keep phase."""
    F = torch.fft.fft2(img_tensor, norm='ortho')
    A = torch.abs(F)
    Phi = torch.angle(F)
    delta_A = torch.randn_like(A) * noise_std * A.std(dim=[2,3], keepdim=True)
    F_pert = (A + delta_A) * torch.exp(1j * Phi)
    img_pert = torch.fft.ifft2(F_pert, norm='ortho').real
    return torch.clamp(img_pert, img_tensor.min(), img_tensor.max())


def evaluate_samed_slice(net, image_np, label_np, num_classes=8):
    """Device-safe evaluation for SAMed models."""
    from utils import calculate_metric_percase
    device = next(net.parameters()).device
    if image_np.ndim == 2:
        image_np = image_np[np.newaxis, :, :]
    if label_np.ndim == 2:
        label_np = label_np[np.newaxis, :, :]
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


def evaluate_msa_slice(net, image_np, label_np, num_classes=8):
    """Evaluation for MSA models using prompt-based inference.

    MSA uses the Medical-SAM-Adapter which expects ImageNet-normalized input
    and uses prompt-based interactive inference.
    """
    from scipy.ndimage import zoom
    from utils import calculate_metric_percase
    device = next(net.parameters()).device
    if image_np.ndim == 2:
        image_np = image_np[np.newaxis, :, :]
    if label_np.ndim == 2:
        label_np = label_np[np.newaxis, :, :]
    prediction = np.zeros_like(label_np)
    for ind in range(image_np.shape[0]):
        slc = image_np[ind, :, :]
        x, y = slc.shape
        if x != IMAGE_SIZE or y != IMAGE_SIZE:
            slc = zoom(slc, (IMAGE_SIZE / x, IMAGE_SIZE / y), order=3)
        slc_3ch = np.stack([slc]*3, axis=-1).astype(np.float32)
        img_t = torch.from_numpy(slc_3ch).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            img_pre = net.preprocess(img_t)
            enc_out = net.image_encoder(img_pre)
            if isinstance(enc_out, tuple):
                img_feats = enc_out[0]
            else:
                img_feats = enc_out

            # Box-only prompt (avoids point coordinate issues)
            h2, w2 = slc.shape
            box_np = np.array([[0, 0, w2, h2]], dtype=np.float32)
            from segment_anything.utils.transforms import ResizeLongestSide
            transform = ResizeLongestSide(IMAGE_SIZE)
            box_t = transform.apply_boxes_torch(
                torch.from_numpy(box_np), (h2, w2)
            ).to(device)

            sparse_emb, dense_emb = net.prompt_encoder(
                points=None, boxes=box_t, masks=None
            )
            lr_masks, _ = net.mask_decoder(
                image_embeddings=img_feats,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            lr_masks = F.interpolate(lr_masks, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear')
            out = (lr_masks.squeeze(0) > 0.5).float().cpu().numpy()
            if out.ndim == 3:
                out = out[0]
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


# ============================================================
# DATA LOADING
# ============================================================
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
            images.append((img[s].astype(np.float32), lbl[s]))
    return images


def get_isic_test_images(n_images=5):
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
        gt_path = os.path.join(data_dir, str(row['seg'])) if 'seg' in row else None
        if os.path.exists(img_path):
            img = transform(Image.open(img_path).convert('RGB')) * 255
            img_np = img[0].numpy()  # Use first channel
            if gt_path and os.path.exists(gt_path):
                gt = np.array(Image.open(gt_path).resize((IMAGE_SIZE, IMAGE_SIZE))) > 128
                images.append((img_np.astype(np.float32), gt.astype(np.float32)))
            else:
                images.append((img_np.astype(np.float32), np.zeros((IMAGE_SIZE, IMAGE_SIZE))))
    return images


def get_refuge_test_images(n_images=5):
    from PIL import Image
    from torchvision import transforms
    test_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    if not os.path.isdir(test_dir): return []
    folders = sorted([f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))])[:n_images]
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = []
    for folder in folders:
        img_path = os.path.join(test_dir, folder, f"{folder}.jpg")
        if os.path.exists(img_path):
            img = transform(Image.open(img_path).convert('RGB')) * 255
            img_np = img[0].numpy()
            images.append((img_np.astype(np.float32), np.zeros((IMAGE_SIZE, IMAGE_SIZE))))
    return images


# ============================================================
# COMPUTE TASK-LEVEL PSI
# ============================================================
def compute_task_psi(net, arch, image_np, label_np, num_classes, n_trials=5, noise_std=0.3):
    """
    Compute PSI as: how much does phase noise vs amplitude noise affect Dice?

    For each trial:
    1. Add random phase noise → evaluate → measure ΔDice
    2. Add random amplitude noise → evaluate → measure ΔDice
    PSI = mean(ΔDice_phase) / (mean(ΔDice_phase) + mean(ΔDice_amplitude) + 1e-12)
    """
    device = next(net.parameters()).device

    # Get clean baseline
    if arch == "SAMed":
        clean_dice = evaluate_samed_slice(net, image_np, label_np, num_classes)
    else:
        clean_dice = evaluate_msa_slice(net, image_np, label_np, num_classes)

    phase_deltas = []
    ampl_deltas = []

    for _ in range(n_trials):
        # Prepare 3-channel tensor
        if image_np.ndim == 2:
            img_3ch = np.stack([image_np]*3, axis=0)
        else:
            img_3ch = image_np[:3] if image_np.shape[0] >= 3 else np.tile(image_np, (3, 1, 1))
        img_t = torch.from_numpy(img_3ch).unsqueeze(0).float().to(device)

        # Phase perturbation
        img_phase = perturb_fft_phase(img_t, noise_std)
        if arch == "SAMed":
            dice_phase = evaluate_samed_slice(net, img_phase[0, 0].cpu().numpy(), label_np, num_classes)
        else:
            dice_phase = evaluate_msa_slice(net, img_phase[0, 0].cpu().numpy(), label_np, num_classes)
        phase_deltas.append(max(0, clean_dice - dice_phase))  # degradation (positive = worse)

        # Amplitude perturbation
        img_ampl = perturb_fft_amplitude(img_t, noise_std)
        if arch == "SAMed":
            dice_ampl = evaluate_samed_slice(net, img_ampl[0, 0].cpu().numpy(), label_np, num_classes)
        else:
            dice_ampl = evaluate_msa_slice(net, img_ampl[0, 0].cpu().numpy(), label_np, num_classes)
        ampl_deltas.append(max(0, clean_dice - dice_ampl))

    mean_phase = np.mean(phase_deltas)
    mean_ampl = np.mean(ampl_deltas)
    psi = mean_phase / (mean_phase + mean_ampl + 1e-12)

    return psi, mean_phase, mean_ampl, clean_dice


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*70)
    print("PAID Theory v2: Task-Level PSI across 18 Models")
    print("="*70)

    # Build model list
    models_to_test = []
    for arch, cfg in [("SAMed", SAMED_CFG), ("MSA", MSA_CFG)]:
        for ds in ["Synapse", "ISIC", "REFUGE"]:
            for vit in ["vit_b", "vit_l", "vit_h"]:
                ckpt_rel, ncls = cfg[ds][vit]
                ckpt_path = os.path.join(CKPT_BASE, "SAMed" if arch == "SAMed" else "", ckpt_rel)
                if arch == "MSA":
                    ckpt_path = os.path.join(CKPT_BASE, ckpt_rel)
                if os.path.exists(ckpt_path):
                    models_to_test.append((arch, ds, vit, ckpt_path, ncls))

    print(f"Found {len(models_to_test)} available models")

    all_results = {}
    n_imgs_per_model = 3  # fewer images, more trials per image

    for arch, ds, vit, ckpt_path, num_classes in models_to_test:
        target_name = f"{ds}_{arch}_{vit}"
        gpu_id = GPU_MAP.get(target_name, 0)
        device = torch.device(f"cuda:{gpu_id}")

        print(f"\n{'='*60}")
        print(f"  {target_name} (GPU {gpu_id}, classes={num_classes})")
        print(f"{'='*60}")

        try:
            sam_ckpt = SAM_CKPTS[vit]
            if arch == "SAMed":
                net = load_samed_model(vit, sam_ckpt, ckpt_path, IMAGE_SIZE, num_classes, device=device)
            else:
                net = load_medsa_model(vit, sam_ckpt, ckpt_path, IMAGE_SIZE, num_classes, device=device)

            # Get test images
            if ds == "Synapse":
                test_imgs = get_synapse_test_images(n_imgs_per_model)
            elif ds == "ISIC":
                test_imgs = get_isic_test_images(n_imgs_per_model)
            else:
                test_imgs = get_refuge_test_images(n_imgs_per_model)

            if not test_imgs:
                print(f"  SKIP: no test images")
                all_results[target_name] = {"error": "no test images"}
                del net; gc.collect(); torch.cuda.empty_cache()
                continue

            print(f"  Loaded {len(test_imgs)} test images")

            all_psi = []
            for img_idx, (img_np, lbl_np) in enumerate(test_imgs):
                # Use a higher noise_std for Synapse (3D → single slice is different)
                noise_std = 0.4 if ds == "Synapse" else 0.3
                psi, phase_d, ampl_d, clean_d = compute_task_psi(
                    net, arch, img_np, lbl_np, num_classes, n_trials=5, noise_std=noise_std
                )
                all_psi.append({
                    "psi": psi, "phase_delta": phase_d, "ampl_delta": ampl_d, "clean_dice": clean_d
                })
                print(f"    [{img_idx+1}/{len(test_imgs)}] PSI={psi:.4f} (phase_Δ={phase_d:.4f}, ampl_Δ={ampl_d:.4f}, clean={clean_d:.4f})")

            mean_psi = float(np.mean([p["psi"] for p in all_psi]))
            mean_phase_delta = float(np.mean([p["phase_delta"] for p in all_psi]))
            mean_ampl_delta = float(np.mean([p["ampl_delta"] for p in all_psi]))

            all_results[target_name] = {
                "mean_psi": mean_psi,
                "std_psi": float(np.std([p["psi"] for p in all_psi])),
                "mean_phase_delta": mean_phase_delta,
                "mean_ampl_delta": mean_ampl_delta,
                "per_image": all_psi,
            }
            print(f"  => PSI={mean_psi:.4f} (phase_Δ={mean_phase_delta:.4f}, ampl_Δ={mean_ampl_delta:.4f})")

            del net; gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[target_name] = {"error": str(e)}
            gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(os.path.join(OUTPUT_DIR, "paid_psi_v2_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ============================================================
    # VALIDATION
    # ============================================================
    print("\n" + "="*70)
    print("VALIDATION: Task PSI vs Phase-Shuffled Attack ΔDice")
    print("="*70)

    # Load Exp14 phase-shuffled results
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

    # Also try structured results
    for group_name, group_data in exp14.get("results", {}).items():
        if isinstance(group_data, dict):
            for model_key, model_vals in group_data.items():
                if isinstance(model_vals, dict) and "phase_shuffled" in model_vals:
                    ps_deltas[model_key] = float(model_vals["phase_shuffled"])

    print(f"  Phase-shuffled ΔDice available for {len(ps_deltas)} models")

    # Build correlation
    pairs = []
    for name, data in all_results.items():
        if "error" in data: continue
        if name in ps_deltas:
            pairs.append({"model": name, "psi": data["mean_psi"], "delta_dice": ps_deltas[name]})

    if len(pairs) >= 5:
        psis = [p["psi"] for p in pairs]
        deltas = [p["delta_dice"] for p in pairs]

        r_p, p_p = pearsonr(psis, deltas)
        r_s, p_s = spearmanr(psis, deltas)

        print(f"\n  Correlation Results:")
        print(f"  Pearson r = {r_p:.4f} (p={p_p:.4f})")
        print(f"  Spearman ρ = {r_s:.4f} (p={p_s:.4f})")

        print(f"\n  Per-Model Breakdown:")
        print(f"  {'Model':<30s} {'PSI':>8s} {'ΔDice':>8s}")
        print(f"  {'-'*48}")
        for p in sorted(pairs, key=lambda x: x["psi"]):
            print(f"  {p['model']:<30s} {p['psi']:8.4f} {p['delta_dice']:8.4f}")

        if r_p > 0.7:
            print(f"\n  ✅ STRONG: PSI predicts attack vulnerability (r={r_p:.3f})")
        elif r_p > 0.5:
            print(f"\n  ⚠️ MODERATE: PSI has predictive power (r={r_p:.3f})")
        else:
            print(f"\n  ❌ WEAK: r={r_p:.3f}")

        with open(os.path.join(OUTPUT_DIR, "paid_correlation_v2.json"), "w") as f:
            json.dump({"pairs": pairs, "pearson_r": r_p, "pearson_p": p_p,
                       "spearman_r": r_s, "spearman_p": p_s, "n": len(pairs)}, f, indent=2)
    else:
        print(f"  Only {len(pairs)} paired models — insufficient for correlation")

    return all_results


if __name__ == "__main__":
    main()
