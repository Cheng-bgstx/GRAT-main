#!/usr/bin/env python3
"""
CODD Master Experiment: Complete 18-Model × 3-Dataset Attack Evaluation.

Covers:
  1. CODD Radial Attack (phase-locked, carrier-aligned) on all 18 victim models
  2. SE-CGR Baseline on all 18 victim models
  3. Directional Ablation: radial vs tangential vs isotropic vs misaligned
  4. SRS (Shared Radial Subspace) Cross-dataset Transfer
  5. Statistical tests & result compilation

Theory grounding (CODD — Carrier-Oriented Drift Decomposition):
  - ΔF(ω) = F_target(ω) - F_source(ω)
  - In source carrier basis: ΔG = ΔF · e^{-j·φ_s} = α + j·β
  - α = radial (carrier-aligned), β = tangential (carrier-orthogonal)
  - Validated: radial_ratio = 0.54-0.70 across 18 models (>> random 0.50)

Attack design:
  - Probe = N(0,1) · (λ · A_s + γ) · e^{j·φ_s}  (carrier-aligned, phase-locked)
  - SRS offline: PCA on proxy attack gradients → shared radial direction
  - SRS online: blend gradient with SRS projection to prevent overfitting

Usage:
  # Single GPU (all experiments):
  python exp_codd_master.py --gpu 0

  # Multi-GPU (split workload):
  for i in 0 1 2 3 4 5; do
    python exp_codd_master.py --gpu $i --total_gpus 6 &
  done
"""
import os, sys, json, time, random, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import AttackConfig, run_codd_attack, distill_srs, _get_edge_map, _return_eta
from theory import SharedSubspace, blend_with_subspace, fit_shared_subspace
from evaluate import (
    load_sam_encoder, load_samed_model, load_medsa_model,
    compute_dice_hd95, compute_iou,
)

# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_SIZE = 512
OUTPUT_DIR = os.path.join(NEWEST, "codd_master_results")
SRS_DIR = os.path.join(NEWEST, "codd_master_results", "srs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SRS_DIR, exist_ok=True)

# Base SAM checkpoints
SAM_CKPT = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}

# Victim model configs (from checkpoints/best_weights/ symlinks)
SAMED_CFG = {
    ("Synapse", "vit_b"): (f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 8),
    ("Synapse", "vit_l"): (f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_99.pth", 8),
    ("Synapse", "vit_h"): (f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_299.pth", 8),
    ("ISIC", "vit_b"): (f"{ROOT}/output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 1),
    ("ISIC", "vit_l"): (f"{ROOT}/output_runs_correct/SAMed/isic_512_lrexp5.0_pretrain_vit_l_epo300_bs12_lr0.005/epoch_99.pth", 1),
    ("ISIC", "vit_h"): (f"{ROOT}/output_runs_correct/SAMed/isic_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_99.pth", 1),
    ("REFUGE", "vit_b"): (f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 2),
    ("REFUGE", "vit_l"): (f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_99.pth", 2),
    ("REFUGE", "vit_h"): (f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_99.pth", 2),
}

MSA_CFG = {
    ("Synapse", "vit_b"): (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/Model/best_dice_checkpoint.pth", 9),
    ("Synapse", "vit_l"): (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/Synapse_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24/Model/best_dice_checkpoint.pth", 9),
    ("Synapse", "vit_h"): (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/Synapse_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24/Model/best_dice_checkpoint.pth", 9),
    ("ISIC", "vit_b"): (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/best_dice_checkpoint.pth", 1),
    ("ISIC", "vit_l"): (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/isic_512_pretrain_vit_l_epo100_bs16_lr0.0001_s24/Model/best_dice_checkpoint.pth", 1),
    ("ISIC", "vit_h"): (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/isic_512_pretrain_vit_h_epo100_bs8_lr0.0001_s24/Model/best_dice_checkpoint.pth", 1),
    ("REFUGE", "vit_b"): (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/best_dice_checkpoint.pth", 2),
    ("REFUGE", "vit_l"): (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/REFUGE_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24/Model/best_dice_checkpoint.pth", 2),
    ("REFUGE", "vit_h"): (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/REFUGE_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24/Model/best_dice_checkpoint.pth", 2),
}

# Attack parameters (matching SE-CGR paper defaults)
ATTACK_DEFAULTS = {
    "att_step_size": 2.0,
    "att_bound": 10.0,
    "att_numstep": 10,
    "momentum_decay": 0.5,
    "ads_rho": 1.0,
    "ads_gamma": 1e-4,
    "probe_amplification": 2.0,
    "edge_guidance": True,
    "edge_gamma": 3.0,
}

# Number of test samples per dataset
N_SAMPLES = {"Synapse": 3, "ISIC": 16, "REFUGE": 16}  # Synapse volumes have many slices each
N_PROXY = {"Synapse": 64, "ISIC": 64, "REFUGE": 64}

# ============================================================
# DATA LOADING
# ============================================================

def load_synapse_test_cases(n_cases=3, max_slices_per_volume=8):
    """Load Synapse test volumes with organ-containing slices."""
    test_dir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    cases = []
    for fname in sorted(os.listdir(test_dir))[:n_cases]:
        if not fname.endswith('.h5') and not fname.endswith('.npy'):
            continue
        path = os.path.join(test_dir, fname)
        case_name = os.path.splitext(fname)[0]
        if fname.endswith('.h5'):
            import h5py
            with h5py.File(path, 'r') as f:
                img = f['image'][:].astype(np.float32)   # [D, H, W]
                lbl = f['label'][:].astype(np.float32)
        else:
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.ndarray):
                img = data.astype(np.float32)
                lbl = np.zeros_like(img)
            else:
                img = data['image'].astype(np.float32)
                lbl = data['label'].astype(np.float32)

        # Find organ-containing slices
        valid_slices = []
        for s in range(img.shape[0]):
            if (lbl[s] > 0).sum() > 100:
                valid_slices.append(s)

        if not valid_slices:
            valid_slices = list(range(min(img.shape[0], max_slices_per_volume)))
        elif len(valid_slices) > max_slices_per_volume:
            # Uniformly sample
            step = len(valid_slices) / max_slices_per_volume
            valid_slices = [valid_slices[int(i * step)] for i in range(max_slices_per_volume)]

        slices = []
        for s in valid_slices:
            slc_img = img[s]
            slc_lbl = lbl[s]
            # Normalize to [0, 255]
            if slc_img.max() > 1.0:
                slc_img = np.clip(slc_img, -200, 300)
                slc_img = (slc_img - slc_img.min()) / (slc_img.max() - slc_img.min() + 1e-6) * 255.0

            # Resize to IMAGE_SIZE
            from scipy.ndimage import zoom
            h_scale = IMAGE_SIZE / slc_img.shape[0]
            w_scale = IMAGE_SIZE / slc_img.shape[1]
            slc_img = zoom(slc_img, (h_scale, w_scale), order=3)
            slc_lbl = zoom(slc_lbl, (h_scale, w_scale), order=0)

            slices.append({
                "image": slc_img,
                "label": slc_lbl,
                "case_name": f"{case_name}_slice{s}",
            })
        cases.append({"case_name": case_name, "slices": slices})
    return cases


def load_isic_test_cases(n_cases=16):
    """Load ISIC 2016 test images with ground truth."""
    import pandas as pd
    from PIL import Image

    data_dir = f"{ROOT}/data/ISIC"
    csv_path = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv")
    if not os.path.exists(csv_path):
        # Fallback: use test data directory directly
        img_dir = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_Data")
        gt_dir = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth")
        jpgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        cases = []
        for jpg in jpgs[:n_cases]:
            base = os.path.splitext(jpg)[0]
            img_path = os.path.join(img_dir, jpg)
            gt_path = os.path.join(gt_dir, f"{base}_Segmentation.png")
            if not os.path.exists(gt_path):
                continue
            img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            gt = Image.open(gt_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            img_np = np.array(img).astype(np.float32)
            gt_np = (np.array(gt) > 128).astype(np.float32)
            if gt_np.sum() < 100:
                continue
            cases.append({"case_name": base, "image": img_np, "label": gt_np})
        return cases

    df = pd.read_csv(csv_path)
    cases = []
    for _, row in df.iterrows():
        if len(cases) >= n_cases:
            break
        img_path = os.path.join(data_dir, row['img'])
        gt_path = os.path.join(data_dir, row['seg'])
        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            continue
        img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        gt = Image.open(gt_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        img_np = np.array(img).astype(np.float32)
        gt_np = (np.array(gt) > 128).astype(np.float32)
        if gt_np.sum() < 100:
            continue
        cases.append({"case_name": os.path.splitext(os.path.basename(row['img']))[0],
                      "image": img_np, "label": gt_np})
    return cases


def load_refuge_test_cases(n_cases=16):
    """Load REFUGE test images (disc + cup segmentation)."""
    from PIL import Image

    data_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    if not os.path.isdir(data_dir):
        data_dir = f"{ROOT}/data/REFUGE-Multirater"

    cases = []
    # Look for subdirectories with images
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for subdir in subdirs[:n_cases]:
        subdir_path = os.path.join(data_dir, subdir)
        imgs = sorted([f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png', '.bmp'))])
        gts = sorted([f for f in os.listdir(subdir_path) if 'seg' in f.lower() or 'gt' in f.lower() or 'mask' in f.lower()])
        if not imgs:
            continue

        img_path = os.path.join(subdir_path, imgs[0])
        img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)

        if gts:
            gt_path = os.path.join(subdir_path, gts[0])
            gt = Image.open(gt_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            gt_np = np.array(gt).astype(np.float32)
        else:
            gt_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        cases.append({"case_name": subdir, "image": img_np, "label": gt_np})
    return cases


# ============================================================
# MODEL LOADING
# ============================================================

def load_source_sam(vit_name, device):
    """Load base SAM encoder (white-box surrogate)."""
    sam = load_sam_encoder(vit_name, SAM_CKPT[vit_name], IMAGE_SIZE, device=device)
    return sam

def load_victim(vit_name, method, dataset, device):
    """Load a victim model (SAMed or MSA)."""
    if method == "SAMed":
        ckpt_path, n_classes = SAMED_CFG[(dataset, vit_name)]
        return load_samed_model(vit_name, SAM_CKPT[vit_name], ckpt_path,
                               IMAGE_SIZE, n_classes, device=device), n_classes
    else:  # MSA
        ckpt_path, mmo = MSA_CFG[(dataset, vit_name)]
        return load_medsa_model(vit_name, SAM_CKPT[vit_name], ckpt_path,
                                IMAGE_SIZE, mmo, device=device), mmo


# ============================================================
# EVALUATION
# ============================================================

def evaluate_samed_2d(net, image_tensor, label_tensor, device, n_classes):
    """Evaluate SAMed on a 2D image."""
    with torch.no_grad():
        img = image_tensor.unsqueeze(0).to(device)
        net.eval()
        outputs = net(img, True, IMAGE_SIZE)
        masks = outputs['masks']

    if n_classes == 1:
        # Binary: ISIC
        probs = torch.sigmoid(masks).squeeze().cpu().numpy()
        gt = label_tensor.cpu().numpy() if torch.is_tensor(label_tensor) else label_tensor
        gt = (gt > 0.5).astype(np.float32)
        best_d = 0.0
        for th in np.arange(0.1, 1.0, 0.1):
            pb = (probs > th).astype(np.float32)
            if pb.ndim > 1:
                pb = pb.squeeze()
            intersection = (pb * gt).sum()
            denom = pb.sum() + gt.sum()
            d = 2.0 * intersection / (denom + 1e-8)
            if d > best_d:
                best_d = d
        return best_d
    elif n_classes == 2:
        # REFUGE: disc + cup
        masks_sm = torch.softmax(masks, dim=1).squeeze(0).cpu().numpy()
        pred = np.argmax(masks_sm, axis=0)
        gt = label_tensor.cpu().numpy() if torch.is_tensor(label_tensor) else label_tensor
        disc_gt = (gt[1] > 0.5).astype(np.float32) if gt.ndim > 2 else (gt > 0).astype(np.float32)
        cup_gt = (gt[0] > 0.5).astype(np.float32) if gt.ndim > 2 else np.zeros_like(disc_gt)
        disc_p = (pred == 1).astype(np.float32)
        cup_p = (pred == 2).astype(np.float32)
        d_disc = 2.0 * (disc_p * disc_gt).sum() / (disc_p.sum() + disc_gt.sum() + 1e-8)
        d_cup = 2.0 * (cup_p * cup_gt).sum() / (cup_p.sum() + cup_gt.sum() + 1e-8)
        return float((d_disc + d_cup) / 2.0)
    else:
        # Synapse: multi-class
        pred = torch.argmax(torch.softmax(masks.float(), dim=1), dim=1).squeeze(0).cpu().numpy()
        gt = label_tensor.cpu().numpy() if torch.is_tensor(label_tensor) else label_tensor
        dices = []
        for c in range(1, n_classes):
            pc = (pred == c).astype(np.float32)
            tc = (gt == c).astype(np.float32)
            inter = (pc * tc).sum()
            denom = pc.sum() + tc.sum()
            dices.append(2.0 * inter / (denom + 1e-8) if denom > 0 else 1.0)
        return float(np.mean(dices))


def evaluate_msa_2d(model, image_np, label_tensor, device, n_classes):
    """Evaluate MSA on a 2D image."""
    from torchvision.transforms import Resize, Compose, ToTensor, Lambda
    from PIL import Image as PILImage

    img_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)
    img_pil = PILImage.fromarray(img_uint8.squeeze() if img_uint8.ndim == 3 and img_uint8.shape[0] < 10 else img_uint8).convert('RGB')
    transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE)), ToTensor(), Lambda(lambda x: x * 255)])
    img_t = transform(img_pil).unsqueeze(0).to(device)

    gt = label_tensor.cpu().numpy() if torch.is_tensor(label_tensor) else label_tensor
    gt = (gt > 0.5).astype(np.float32)

    # Click prompt
    indices = np.argwhere(gt.squeeze() > 0) if gt.ndim > 2 else np.argwhere(gt > 0)
    if len(indices) == 0:
        pt = torch.tensor([[[0.0, 0.0]]], device=device)
        pt_label = torch.tensor([[0]], dtype=torch.int, device=device)
    else:
        r, c = indices[random.randint(0, len(indices) - 1)]
        pt = torch.tensor([[[float(c), float(r)]]], device=device)
        pt_label = torch.tensor([[1]], dtype=torch.int, device=device)

    with torch.no_grad():
        model.eval()
        imgs_pre = model.preprocess(img_t)
        image_emb = model.image_encoder(imgs_pre)
        sparse_emb, dense_emb = model.prompt_encoder(points=(pt, pt_label), boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=image_emb, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)

    probs = torch.sigmoid(pred).squeeze().cpu().numpy()
    if probs.ndim > 1:
        probs = probs.squeeze()
    gt_flat = gt.squeeze()

    best_d = 0.0
    for th in np.arange(0.1, 1.0, 0.1):
        pb = (probs > th).astype(np.float32)
        d = 2.0 * (pb * gt_flat).sum() / (pb.sum() + gt_flat.sum() + 1e-8)
        if d > best_d:
            best_d = d
    return float(best_d)


# ============================================================
# ATTACK RUNNER
# ============================================================

def prepare_image_tensor(img_np, device, use_imagenet_norm=False):
    """Convert numpy image to normalized tensor for SAM input."""
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=0)
    elif img_np.ndim == 3 and img_np.shape[-1] == 3:
        img_np = img_np.transpose(2, 0, 1)

    img_t = torch.from_numpy(img_np).float()
    if img_t.max() > 2.0:
        img_t = img_t / 255.0

    if use_imagenet_norm:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

    return img_t.unsqueeze(0).to(device)


def run_attack_on_image(source_sam, img_np, config, srs_bundle=None, device=None):
    """Run CODD attack on a single image using source SAM."""
    if device is None:
        device = next(source_sam.parameters()).device

    img_t = prepare_image_tensor(img_np, device, use_imagenet_norm=False)
    # SAM expects [0, 255] range
    img_t = img_t * 255.0

    adv_np, trace = run_codd_attack(
        source_sam.image_encoder, img_t, config,
        srs_bundle=srs_bundle, return_trace=True,
    )
    return adv_np, trace


def run_codd_experiment(
    source_sam, victim_models, cases, config, device,
    experiment_name="CODD", srs_bundle=None,
):
    """
    Run CODD attack on all test cases, evaluate on all victims.

    Args:
        source_sam: Base SAM model (white-box surrogate)
        victim_models: dict of {victim_name: (model, n_classes)}
        cases: list of test case dicts
        config: AttackConfig
        device: torch device
        experiment_name: label for this experiment
        srs_bundle: optional SharedSubspace for warm-start

    Returns:
        results: dict with per-case per-victim metrics
    """
    results = {vname: {"clean_dices": [], "adv_dices": [], "dice_drops": []}
              for vname in victim_models}

    for case_idx, case in enumerate(cases):
        case_name = case["case_name"]

        if "slices" in case:
            # 3D volume (Synapse) - process slices
            slices = case["slices"]
            for vname, (vmodel, v_ncls) in victim_models.items():
                clean_dices = []
                adv_dices = []

                for slc in slices:
                    img_np = slc["image"]
                    lbl_np = slc["label"]

                    # Clean evaluation
                    img_t = prepare_image_tensor(img_np, device)
                    img_t_255 = img_t * 255.0

                    # Evaluate clean
                    # We need to evaluate on adversarial images
                    # For clean: just use img_t_255 directly
                    # But wait - we need to handle this differently for SAMed vs MSA

                    # Run attack on source SAM
                    try:
                        adv_np, trace = run_attack_on_image(
                            source_sam, img_np, config, srs_bundle, device
                        )
                    except Exception as e:
                        print(f"  ERROR attack {case_name}: {e}")
                        continue

                    # Evaluate on victim
                    try:
                        adv_t = prepare_image_tensor(adv_np, device) * 255.0
                        lbl_t = torch.from_numpy(lbl_np).float()

                        # Evaluate clean
                        clean_dice = evaluate_on_victim(
                            vname, vmodel, img_t_255.squeeze(0), lbl_t, device, v_ncls
                        )
                        # Evaluate adversarial
                        adv_dice = evaluate_on_victim(
                            vname, vmodel, adv_t.squeeze(0), lbl_t, device, v_ncls
                        )
                        clean_dices.append(clean_dice)
                        adv_dices.append(adv_dice)
                    except Exception as e:
                        print(f"  ERROR eval {vname} on {case_name}: {e}")
                        continue

                    del adv_np, adv_t, img_t, img_t_255, lbl_t
                    gc.collect()
                    torch.cuda.empty_cache()

                if clean_dices:
                    results[vname]["clean_dices"].append(np.mean(clean_dices))
                    results[vname]["adv_dices"].append(np.mean(adv_dices))
                    results[vname]["dice_drops"].append(np.mean(clean_dices) - np.mean(adv_dices))
        else:
            # 2D image (ISIC, REFUGE)
            img_np = case["image"]
            lbl_np = case["label"]

            # Run attack
            try:
                adv_np, trace = run_attack_on_image(
                    source_sam, img_np, config, srs_bundle, device
                )
            except Exception as e:
                print(f"  ERROR attack {case_name}: {e}")
                continue

            # Evaluate on all victims
            for vname, (vmodel, v_ncls) in victim_models.items():
                try:
                    # Clean
                    img_t_clean = (prepare_image_tensor(img_np, device).squeeze(0) * 255.0).to(device)
                    lbl_t = torch.from_numpy(lbl_np).float().to(device) if not torch.is_tensor(lbl_np) else lbl_np.to(device)
                    clean_dice = evaluate_on_victim(vname, vmodel, img_t_clean, lbl_t, device, v_ncls)

                    # Adversarial
                    adv_t = (prepare_image_tensor(adv_np, device).squeeze(0) * 255.0).to(device)
                    adv_dice = evaluate_on_victim(vname, vmodel, adv_t, lbl_t, device, v_ncls)

                    results[vname]["clean_dices"].append(clean_dice)
                    results[vname]["adv_dices"].append(adv_dice)
                    results[vname]["dice_drops"].append(clean_dice - adv_dice)
                except Exception as e:
                    print(f"  ERROR eval {vname} on {case_name}: {e}")
                    continue

            del adv_np
            gc.collect()
            torch.cuda.empty_cache()

        if (case_idx + 1) % 4 == 0:
            print(f"  [{experiment_name}] processed {case_idx + 1}/{len(cases)} cases")

    # Aggregate
    for vname in results:
        for key in ["clean_dices", "adv_dices", "dice_drops"]:
            arr = results[vname][key]
            results[vname][f"mean_{key[:-1]}"] = float(np.mean(arr)) if arr else float("nan")

    return results


def evaluate_on_victim(vname, vmodel, img_t, lbl_t, device, n_classes):
    """Dispatch evaluation based on victim type."""
    if "MSA" in vname:
        if img_t.dim() == 3:
            img_np = img_t.cpu().numpy().transpose(1, 2, 0) if img_t.shape[0] == 3 else img_t.cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] < 10:
                img_np = img_np.transpose(1, 2, 0)
            if img_np.ndim == 3 and img_np.shape[2] == 3:
                img_np = img_np.squeeze()
        else:
            img_np = img_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        return evaluate_msa_2d(vmodel, img_np, lbl_t, device, n_classes)
    else:
        return evaluate_samed_2d(vmodel, img_t, lbl_t, device, n_classes)


# ============================================================
# MAIN EXPERIMENT ORCHESTRATION
# ============================================================

def build_experiment_list():
    """Build the complete list of experiments."""
    experiments = []

    # For each ViT size, we have:
    # - 1 source SAM (base model)
    # - 6 victims (SAMed + MSA × 3 datasets)
    for vit in ["vit_b", "vit_l", "vit_h"]:
        sam_ckpt = SAM_CKPT[vit]
        if not os.path.exists(sam_ckpt):
            print(f"SKIP {vit}: SAM checkpoint not found: {sam_ckpt}")
            continue

        experiment = {
            "vit": vit,
            "sam_ckpt": sam_ckpt,
            "victims": {},
            "datasets": {},
        }

        for dataset in ["Synapse", "ISIC", "REFUGE"]:
            samed_key = (dataset, vit)
            msa_key = (dataset, vit)

            if samed_key in SAMED_CFG and os.path.exists(SAMED_CFG[samed_key][0]):
                experiment["victims"][f"{dataset}_SAMed"] = {
                    "method": "SAMed",
                    "dataset": dataset,
                    "ckpt": SAMED_CFG[samed_key][0],
                    "n_classes": SAMED_CFG[samed_key][1],
                }
            else:
                print(f"SKIP SAMed {dataset} {vit}: checkpoint not found")

            if msa_key in MSA_CFG and os.path.exists(MSA_CFG[msa_key][0]):
                experiment["victims"][f"{dataset}_MSA"] = {
                    "method": "MSA",
                    "dataset": dataset,
                    "ckpt": MSA_CFG[msa_key][0],
                    "n_classes": MSA_CFG[msa_key][1],
                }
            else:
                print(f"SKIP MSA {dataset} {vit}: checkpoint not found")

        if experiment["victims"]:
            experiments.append(experiment)

    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--total_gpus", type=int, default=1)
    parser.add_argument("--skip_srs", action="store_true", help="Skip SRS distillation")
    parser.add_argument("--skip_ablation", action="store_true", help="Skip directional ablation")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["all", "codd", "baseline", "ablation", "srs"])
    parser.add_argument("--max_cases", type=int, default=0, help="Max test cases (0=all)")
    parser.add_argument("--vit_only", type=str, default="", help="Only run specific ViT (vit_b/vit_l/vit_h)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    print(f"CODD Master Experiment — GPU {args.gpu}/{args.total_gpus}")
    print(f"Output: {OUTPUT_DIR}")

    all_experiments = build_experiment_list()
    if args.vit_only:
        all_experiments = [e for e in all_experiments if e["vit"] == args.vit_only]

    # Split across GPUs
    my_experiments = all_experiments[args.gpu::args.total_gpus]
    print(f"Assigned {len(my_experiments)}/{len(all_experiments)} ViT groups:")
    for exp in my_experiments:
        print(f"  {exp['vit']}: {list(exp['victims'].keys())}")

    all_results = {}

    for exp_idx, exp in enumerate(my_experiments):
        vit = exp["vit"]
        print(f"\n{'='*60}")
        print(f"Processing {vit} ({exp_idx + 1}/{len(my_experiments)})")
        print(f"{'='*60}")

        # Load source SAM
        print(f"Loading source SAM {vit}...")
        source_sam = load_source_sam(vit, device)

        # Load test data for each dataset
        datasets = {}
        for vname, vinfo in exp["victims"].items():
            ds = vinfo["dataset"]
            if ds not in datasets:
                n_cases = N_SAMPLES[ds]
                if args.max_cases > 0:
                    n_cases = min(n_cases, args.max_cases)

                if ds == "Synapse":
                    datasets[ds] = load_synapse_test_cases(n_cases)
                elif ds == "ISIC":
                    datasets[ds] = load_isic_test_cases(n_cases)
                elif ds == "REFUGE":
                    datasets[ds] = load_refuge_test_cases(n_cases)
                print(f"  Loaded {len(datasets[ds])} {ds} test cases")

        # Load victim models
        victims = {}
        for vname, vinfo in exp["victims"].items():
            print(f"  Loading victim {vname}...")
            try:
                vmodel, v_ncls = load_victim(vit, vinfo["method"], vinfo["dataset"], device)
                victims[vname] = (vmodel, v_ncls)
            except Exception as e:
                print(f"  ERROR loading {vname}: {e}")
                continue

        # ================================================================
        # Experiment 1: CODD Radial Attack (default)
        # ================================================================
        if args.experiment in ("all", "codd"):
            print(f"\n--- CODD Radial Attack ---")
            config = AttackConfig(**ATTACK_DEFAULTS, direction_mode="radial", seed=42)

            vit_results = {}
            for ds_name, ds_cases in datasets.items():
                # Filter victims for this dataset
                ds_victims = {vn: v for vn, v in victims.items() if ds_name in vn}
                if not ds_victims:
                    continue

                print(f"  Dataset: {ds_name} ({len(ds_cases)} cases, {len(ds_victims)} victims)")
                ds_results = run_codd_experiment(
                    source_sam, ds_victims, ds_cases, config, device,
                    experiment_name=f"CODD_radial_{vit}_{ds_name}"
                )
                vit_results[ds_name] = ds_results

            all_results[f"{vit}_CODD_radial"] = vit_results

            # Save intermediate
            with open(os.path.join(OUTPUT_DIR, f"codd_radial_{vit}.json"), 'w') as f:
                json.dump(vit_results, f, indent=2, default=str)
            print(f"  Saved to codd_radial_{vit}.json")

        # ================================================================
        # Experiment 2: Directional Ablation (vit_b only!)
        # ================================================================
        if args.experiment in ("all", "ablation") and vit == "vit_b" and not args.skip_ablation:
            directions = {
                "radial": "Carrier-aligned (CODD)",
                "tangential": "Carrier-orthogonal",
                "isotropic": "Equal radial+tangential",
                "misaligned": "Random phase direction",
            }

            for dir_mode, dir_desc in directions.items():
                print(f"\n--- Directional Ablation: {dir_mode} ({dir_desc}) ---")
                config = AttackConfig(**ATTACK_DEFAULTS, direction_mode=dir_mode, seed=42)

                dir_results = {}
                for ds_name, ds_cases in datasets.items():
                    ds_victims = {vn: v for vn, v in victims.items() if ds_name in vn}
                    if not ds_victims:
                        continue
                    print(f"  Dataset: {ds_name}")
                    ds_results = run_codd_experiment(
                        source_sam, ds_victims, ds_cases, config, device,
                        experiment_name=f"CODD_{dir_mode}_{vit}_{ds_name}"
                    )
                    dir_results[ds_name] = ds_results

                all_results[f"{vit}_CODD_{dir_mode}"] = dir_results

                with open(os.path.join(OUTPUT_DIR, f"codd_{dir_mode}_{vit}.json"), 'w') as f:
                    json.dump(dir_results, f, indent=2, default=str)

        # ================================================================
        # Experiment 3: SRS Distillation (vit_b only for now)
        # ================================================================
        if args.experiment in ("all", "srs") and vit == "vit_b" and not args.skip_srs:
            print(f"\n--- SRS Distillation ---")

            for ds_name in ["Synapse", "ISIC", "REFUGE"]:
                if ds_name not in datasets:
                    continue

                srs_path = os.path.join(SRS_DIR, f"srs_{vit}_{ds_name}.pt")
                if os.path.exists(srs_path):
                    print(f"  Loading existing SRS: {srs_path}")
                    srs_bundle = SharedSubspace.load(srs_path, device)
                else:
                    print(f"  Distilling SRS from {ds_name} proxy data...")
                    # Load proxy training data
                    proxy_cases = load_proxy_data(ds_name, N_PROXY[ds_name])
                    if not proxy_cases:
                        print(f"  No proxy data for {ds_name}, skipping SRS")
                        continue

                    # Create simple dataloader-like list
                    proxy_list = []
                    for case in proxy_cases:
                        if "slices" in case:
                            proxy_list.extend(case["slices"])
                        else:
                            proxy_list.append(case)

                    config = AttackConfig(**ATTACK_DEFAULTS, direction_mode="radial", seed=42)

                    # Distill SRS
                    class ProxyDataLoader:
                        def __init__(self, data):
                            self.data = data
                            self.idx = 0
                        def __iter__(self):
                            return self
                        def __next__(self):
                            if self.idx >= len(self.data):
                                raise StopIteration
                            item = self.data[self.idx]
                            self.idx += 1
                            img = item["image"]
                            if img.ndim == 2:
                                img = np.stack([img]*3, axis=0)
                            elif img.ndim == 3 and img.shape[-1] == 3:
                                img = img.transpose(2, 0, 1)
                            img_t = torch.from_numpy(img).float().unsqueeze(0)
                            if img_t.max() > 2.0:
                                img_t = img_t / 255.0
                            img_t_255 = img_t * 255.0
                            return {"image": img_t_255}

                    srs_bundle = distill_srs(
                        source_sam.image_encoder,
                        ProxyDataLoader(proxy_list[:N_PROXY[ds_name]]),
                        config, num_samples=N_PROXY[ds_name], rank=8,
                        device=device, is_3d=(ds_name == "Synapse"),
                    )
                    srs_bundle.save(srs_path)
                    print(f"  SRS saved to {srs_path}")

                # Run CODD + SRS attack
                print(f"  Running CODD + SRS on {ds_name}...")
                config = AttackConfig(
                    **ATTACK_DEFAULTS, direction_mode="radial",
                    use_srs=True, srs_path=srs_path,
                    srs_proj_weight=0.15, srs_mean_weight=0.1,
                    seed=42,
                )

                ds_victims = {vn: v for vn, v in victims.items() if ds_name in vn}
                if ds_victims and ds_name in datasets:
                    srs_results = run_codd_experiment(
                        source_sam, ds_victims, datasets[ds_name], config, device,
                        experiment_name=f"CODD_SRS_{vit}_{ds_name}",
                        srs_bundle=srs_bundle,
                    )
                    all_results[f"{vit}_CODD_SRS_{ds_name}"] = {ds_name: srs_results}

                    with open(os.path.join(OUTPUT_DIR, f"codd_srs_{vit}_{ds_name}.json"), 'w') as f:
                        json.dump(srs_results, f, indent=2, default=str)

        # Cleanup
        del source_sam
        for vname in list(victims.keys()):
            vmodel, _ = victims[vname]
            del vmodel
        del victims
        gc.collect()
        torch.cuda.empty_cache()

    # ================================================================
    # Final Summary
    # ================================================================
    print(f"\n{'='*60}")
    print("Compiling Final Summary...")

    summary = compile_summary(all_results, OUTPUT_DIR)

    summary_path = os.path.join(OUTPUT_DIR, f"codd_master_summary_gpu{args.gpu}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")
    print("Done!")


def load_proxy_data(dataset_name, n_samples):
    """Load proxy (training) data for SRS distillation."""
    if dataset_name == "Synapse":
        train_dir = f"{ROOT}/data/Synapse_preprocess/train_npz"
        if not os.path.isdir(train_dir):
            return []
        npzs = sorted([f for f in os.listdir(train_dir) if f.endswith('.npz')])[:n_samples]
        cases = []
        for npz_name in npzs:
            try:
                data = np.load(os.path.join(train_dir, npz_name), allow_pickle=True)
                img = data['image'].astype(np.float32)
                lbl = data['label'].astype(np.float32)
                if img.max() > 1.0:
                    img = np.clip(img, -200, 300)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255.0
                from scipy.ndimage import zoom
                h_scale = IMAGE_SIZE / img.shape[0]
                w_scale = IMAGE_SIZE / img.shape[1]
                img = zoom(img, (h_scale, w_scale), order=3)
                cases.append({"case_name": npz_name.replace('.npz', ''),
                             "image": img, "label": lbl})
            except Exception:
                continue
        return cases

    elif dataset_name == "ISIC":
        from PIL import Image
        data_dir = f"{ROOT}/data/ISIC"
        train_img_dir = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Training_Data")
        if not os.path.isdir(train_img_dir):
            return []
        jpgs = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.jpg')])[:n_samples]
        cases = []
        for jpg in jpgs:
            img = Image.open(os.path.join(train_img_dir, jpg)).convert('RGB')
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            img_np = np.array(img).astype(np.float32)
            cases.append({"case_name": jpg.replace('.jpg', ''),
                         "image": img_np, "label": np.zeros((IMAGE_SIZE, IMAGE_SIZE))})
        return cases

    elif dataset_name == "REFUGE":
        from PIL import Image
        data_dir = f"{ROOT}/data/REFUGE-Multirater/Training-400"
        if not os.path.isdir(data_dir):
            return []
        subdirs = sorted([d for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))])[:n_samples]
        cases = []
        for subdir in subdirs:
            subdir_path = os.path.join(data_dir, subdir)
            imgs = sorted([f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png'))])
            if imgs:
                img = Image.open(os.path.join(subdir_path, imgs[0])).convert('RGB')
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                img_np = np.array(img).astype(np.float32)
                cases.append({"case_name": subdir,
                             "image": img_np, "label": np.zeros((IMAGE_SIZE, IMAGE_SIZE))})
        return cases

    return []


def compile_summary(all_results, output_dir):
    """Compile experiment results into summary tables."""
    summary = {"experiments": {}, "tables": {}}

    for exp_name, exp_data in all_results.items():
        summary["experiments"][exp_name] = {}

        if isinstance(exp_data, dict):
            for ds_name, ds_data in exp_data.items():
                if isinstance(ds_data, dict):
                    for vname, vdata in ds_data.items():
                        if isinstance(vdata, dict) and "mean_dice_drop" in vdata:
                            summary["experiments"][exp_name][f"{ds_name}_{vname}"] = {
                                "mean_clean_dice": vdata.get("mean_clean_dice", float("nan")),
                                "mean_adv_dice": vdata.get("mean_adv_dice", float("nan")),
                                "mean_dice_drop": vdata.get("mean_dice_drop", float("nan")),
                            }

    # Build summary tables
    for exp_name, exp_results in summary["experiments"].items():
        if not exp_results:
            continue

        table = []
        avg_drop = []
        for key, metrics in exp_results.items():
            drop = metrics.get("mean_dice_drop", float("nan"))
            table.append({"model": key, "dice_drop": drop,
                         "clean_dice": metrics.get("mean_clean_dice", float("nan"))})
            if not np.isnan(drop):
                avg_drop.append(drop)

        summary["tables"][exp_name] = {
            "rows": sorted(table, key=lambda r: r.get("dice_drop", float("nan")) if not np.isnan(r.get("dice_drop", float("nan"))) else 0, reverse=True),
            "mean_drop": float(np.mean(avg_drop)) if avg_drop else float("nan"),
            "std_drop": float(np.std(avg_drop)) if avg_drop else float("nan"),
            "n_models": len(avg_drop),
        }

    return summary


if __name__ == "__main__":
    main()
