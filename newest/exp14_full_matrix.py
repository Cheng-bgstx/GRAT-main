#!/usr/bin/env python3
"""
COMPREHENSIVE Phase Carrier Ablation — Full 18-Model Matrix.

Validates the REVISED CODD theory:
  C3: Phase coherence disruption >> radial attack on ALL models.

Covers:
  3 datasets (Synapse, ISIC, REFUGE) ×
  2 architectures (SAMed, MSA) ×
  3 ViT scales (vit_b, vit_l, vit_h) ×
  5 conditions (SE-CGR, phase-shuffled, tangential, random-phase, no-probe)

Usage:
  python exp14_full_matrix.py                        # all 18 models on GPU 0
  python exp14_full_matrix.py --gpu 0 --subset 0/6   # 1/6 of models on GPU 0
  python exp14_full_matrix.py --gpu 1 --subset 1/6   # etc.

Launch all 6 workers:
  for i in 0 1 2 3 4 5; do
    python exp14_full_matrix.py --gpu $i --subset $i/6 &
  done
"""
import os, sys, json, time, random, gc, argparse
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import AttackConfig, run_codd_attack
from evaluate import load_sam_encoder, load_samed_model, load_medsa_model

IMAGE_SIZE = 512

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAM_CKPT = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}

CONDITIONS = ["se_cgr", "phase_shuffled", "tangential", "random_phase", "no_probe"]
DIRECTION_MAP = {
    "se_cgr": "radial", "phase_shuffled": "radial",
    "tangential": "tangential", "random_phase": "isotropic", "no_probe": "radial",
}

N_SAMPLES = {"Synapse": 3, "ISIC": 8, "REFUGE": 8}  # volumes for Synapse, images for 2D
NUM_SLICES_PER_VOLUME = 4  # Synapse slices per volume

# ---------------------------------------------------------------------------
# All 18 model configurations
# ---------------------------------------------------------------------------

SAMED_CFG = {
    "Synapse": {
        "vit_b": (f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 8),
        "vit_l": (f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 8),
        "vit_h": (f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_299.pth", 8),
    },
    "ISIC": {
        "vit_b": (f"{ROOT}/output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 1),
        "vit_l": (f"{ROOT}/output_runs_correct/SAMed/isic_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 1),
        "vit_h": (f"{ROOT}/output_runs_correct/SAMed/isic_512_lrexp3.0_pretrain_vit_h_epo300_bs4_lr0.005/epoch_159.pth", 1),
    },
    "REFUGE": {
        "vit_b": (f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 2),
        "vit_l": (f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 2),
        "vit_h": (f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_159.pth", 2),
    },
}

MSA_CFG = {
    "Synapse": {
        "vit_b": (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 9),
        "vit_l": (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/Synapse_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 9),
        "vit_h": (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/Synapse_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24/Model/checkpoint_best.pth", 9),
    },
    "ISIC": {
        "vit_b": (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 1),
        "vit_l": (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/isic_512_pretrain_vit_l_epo100_bs16_lr0.0001_s24/Model/checkpoint_best.pth", 1),
        "vit_h": (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/isic_512_pretrain_vit_h_epo100_bs8_lr0.0001_s24/Model/checkpoint_best.pth", 1),
    },
    "REFUGE": {
        "vit_b": (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 2),
        "vit_l": (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/REFUGE_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 2),
        "vit_h": (f"{ROOT}/output_runs_correct/MedicalSAMAdapter/REFUGE_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24/Model/checkpoint_best.pth", 2),
    },
}


def build_model_list():
    """Build flat list of all 18 model configs."""
    models = []
    for dataset in ["Synapse", "ISIC", "REFUGE"]:
        for vit in ["vit_b", "vit_l", "vit_h"]:
            ckpt, ncls = SAMED_CFG[dataset][vit]
            if os.path.exists(ckpt):
                models.append((f"{dataset}_SAMed_{vit}", dataset, "SAMed", vit, ckpt, ncls))
            ckpt, mmo = MSA_CFG[dataset][vit]
            if os.path.exists(ckpt):
                models.append((f"{dataset}_MSA_{vit}", dataset, "MSA", vit, ckpt, mmo))
    return models


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_samed_slice(net, image_np, label_np, case_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location("root_utils", f"{ROOT}/utils.py")
    root_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root_utils)
    test_single_volume = root_utils.test_single_volume
    if image_np.ndim == 2:
        image_np = image_np[np.newaxis, :, :]
    if label_np.ndim == 2:
        label_np = label_np[np.newaxis, :, :]
    img = torch.from_numpy(image_np.copy()).unsqueeze(0).float()
    lbl = torch.from_numpy(label_np.copy()).unsqueeze(0).float()
    metric = test_single_volume(
        image=img, label=lbl, net=net, classes=8, multimask_output=True,
        patch_size=[IMAGE_SIZE, IMAGE_SIZE], test_save_path=None, case=case_name, z_spacing=1,
    )
    return float(np.mean(metric, axis=0)[0])


def evaluate_samed_isic(net, image, label, device):
    with torch.no_grad():
        outputs = net(image.unsqueeze(0).to(device), True, IMAGE_SIZE)
        masks = outputs['masks']
    probs = torch.sigmoid(masks).squeeze().cpu().numpy()
    label_np = label.cpu().numpy() if torch.is_tensor(label) else label
    gt = (label_np > 0.5).astype(np.float32)
    best_d = 0.0
    for th in np.arange(0.1, 1.0, 0.1):
        pb = (probs > th).astype(np.float32)
        intersection = (pb * gt).sum()
        denom = pb.sum() + gt.sum()
        d = 2.0 * intersection / (denom + 1e-8)
        if d > best_d: best_d = d
    return float(best_d)


def evaluate_samed_refuge(net, image, label, device):
    with torch.no_grad():
        outputs = net(image.unsqueeze(0).to(device), True, IMAGE_SIZE)
        masks = outputs['masks']
    masks = torch.softmax(masks, dim=1).squeeze(0).cpu().numpy()
    pred = np.argmax(masks, axis=0)
    label_np = label.cpu().numpy() if torch.is_tensor(label) else label
    disc_gt = (label_np[1] > 0.5).astype(np.float32)
    cup_gt = (label_np[0] > 0.5).astype(np.float32)
    disc_p = (pred == 1).astype(np.float32)
    cup_p = (pred == 2).astype(np.float32)
    d_disc = 2.0 * (disc_p * disc_gt).sum() / (disc_p.sum() + disc_gt.sum() + 1e-8)
    d_cup = 2.0 * (cup_p * cup_gt).sum() / (cup_p.sum() + cup_gt.sum() + 1e-8)
    return float((d_disc + d_cup) / 2.0)


def evaluate_msa_isic(model, image_np, gt_mask, device):
    from PIL import Image as PILImage
    from torchvision.transforms import Resize, Compose, ToTensor, Lambda
    img_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)
    img_pil = PILImage.fromarray(img_uint8).convert('RGB')
    transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE)), ToTensor(), Lambda(lambda x: x * 255)])
    img_t = transform(img_pil).unsqueeze(0).to(device)
    if torch.is_tensor(gt_mask): gt_mask = gt_mask.cpu().numpy()
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    indices = np.argwhere(gt_mask > 0)
    if len(indices) == 0:
        pt = torch.tensor([[[0.0, 0.0]]], device=device)
        pt_label = torch.tensor([[0]], dtype=torch.int, device=device)
    else:
        r, c = indices[random.randint(0, len(indices)-1)]
        pt = torch.tensor([[[float(c), float(r)]]], device=device)
        pt_label = torch.tensor([[1]], dtype=torch.int, device=device)
    with torch.no_grad():
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
    best_d = 0.0
    for th in np.arange(0.1, 1.0, 0.1):
        pb = (probs > th).astype(np.float32)
        d = 2.0 * (pb * gt_mask).sum() / (pb.sum() + gt_mask.sum() + 1e-8)
        if d > best_d: best_d = d
    return float(best_d)


def evaluate_msa_refuge(model, image_np, gt_label, device):
    from PIL import Image as PILImage
    from torchvision.transforms import Resize, Compose, ToTensor, Lambda
    img_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)
    img_pil = PILImage.fromarray(img_uint8).convert('RGB')
    transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE)), ToTensor(), Lambda(lambda x: x * 255)])
    img_t = transform(img_pil).unsqueeze(0).to(device)
    pt_coords = torch.tensor([[[0.0, 0.0]]], device=device)
    pt_labels = torch.tensor([[1]], dtype=torch.int, device=device)
    with torch.no_grad():
        imgs_pre = model.preprocess(img_t)
        image_emb = model.image_encoder(imgs_pre)
        sparse_emb, dense_emb = model.prompt_encoder(points=(pt_coords, pt_labels), boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=image_emb, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    probs = torch.sigmoid(pred[0])
    cup_pred = (probs[0] > 0.5).float().cpu().numpy()
    disc_pred = (probs[1] > 0.5).float().cpu().numpy()
    label_np = gt_label.cpu().numpy() if torch.is_tensor(gt_label) else gt_label
    disc_gt = (label_np[1] > 0.5).astype(np.float32)
    cup_gt = (label_np[0] > 0.5).astype(np.float32)
    d_disc = 2.0 * (disc_pred * disc_gt).sum() / (disc_pred.sum() + disc_gt.sum() + 1e-8)
    d_cup = 2.0 * (cup_pred * cup_gt).sum() / (cup_pred.sum() + cup_gt.sum() + 1e-8)
    return float((d_disc + d_cup) / 2.0)


def evaluate_msa_synapse_slice(model, image_np, label_slice, device):
    img_3ch = np.stack([image_np]*3, axis=0)
    img_t = torch.from_numpy(img_3ch).unsqueeze(0).float().to(device)
    pt_list, lbl_list = [], []
    for c in range(1, 9):
        class_mask = (label_slice == c).astype(np.uint8)
        indices = np.argwhere(class_mask > 0)
        if len(indices) > 0:
            r, cc = indices[random.randint(0, len(indices)-1)]
            pt_list.append(np.array([cc, r])); lbl_list.append(1)
        else:
            pt_list.append(np.array([0, 0])); lbl_list.append(0)
    coords = torch.tensor(np.stack(pt_list, axis=0), dtype=torch.float, device=device).unsqueeze(0)
    labels = torch.tensor(lbl_list, dtype=torch.int, device=device).unsqueeze(0)
    with torch.no_grad():
        imgs_pre = model.preprocess(img_t)
        image_emb = model.image_encoder(imgs_pre)
        sparse_emb, dense_emb = model.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=image_emb, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    pred_classes = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    dices = []
    for c in range(1, 9):
        pc = (pred_classes == c).astype(np.float32)
        tc = (label_slice == c).astype(np.float32)
        inter = (pc * tc).sum(); denom = pc.sum() + tc.sum()
        dices.append(2.0 * inter / (denom + 1e-8) if denom > 0 else 1.0)
    return float(np.mean(dices))


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def get_isic_samples(n_cases):
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    df = pd.read_csv(os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv"))
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    samples = []
    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, row['img'])
        gt_path = os.path.join(data_dir, row['seg'])
        if not os.path.exists(img_path) or not os.path.exists(gt_path): continue
        img = Image.open(img_path).convert('RGB')
        img_raw = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
        gt = np.array(Image.open(gt_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32) / 255.0
        samples.append((transform(img), (gt > 0.5).astype(np.float32), img_raw))
        if len(samples) >= n_cases: break
    return samples


def get_refuge_samples(n_cases):
    from PIL import Image
    from torchvision import transforms
    data_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    samples = []
    for folder in folders:
        img_path = os.path.join(data_dir, folder, f"{folder}.jpg")
        disc_path = os.path.join(data_dir, folder, f"{folder}_disc.bmp")
        cup_path = os.path.join(data_dir, folder, f"{folder}_cup.bmp")
        if not os.path.exists(img_path): continue
        img = Image.open(img_path).convert('RGB')
        img_raw = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
        disc_raw = np.array(Image.open(disc_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        cup_raw = np.array(Image.open(cup_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        label = np.stack([
            (cup_raw < 128).astype(np.float32),
            (disc_raw <= 128).astype(np.float32),
            1.0 - (disc_raw <= 128).astype(np.float32) - (cup_raw < 128).astype(np.float32),
        ], axis=0)
        samples.append((transform(img), label, img_raw))
        if len(samples) >= n_cases: break
    return samples


def normalized_gray_to_uint8(gray_normalized):
    if gray_normalized.ndim == 3 and gray_normalized.shape[0] == 3:
        gray_normalized = gray_normalized[0]
    MEAN_AVG = (0.485 + 0.456 + 0.406) / 3.0
    STD_AVG = (0.229 + 0.224 + 0.225) / 3.0
    return np.clip((gray_normalized * STD_AVG + MEAN_AVG) * 255, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_experiment(tag, dataset, model_type, vit, ckpt, ncls_or_mmo, device):
    """Run phase carrier ablation on one model."""
    print(f"\n{'='*60}\n{tag} (vit={vit}, device={device})\n{'='*60}")

    source_sam = load_sam_encoder(vit, SAM_CKPT[vit], IMAGE_SIZE, 8, device)
    target = None
    n_cases = N_SAMPLES[dataset]

    if model_type == "SAMed":
        target = load_samed_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, ncls_or_mmo, device=device)
    else:
        target = load_medsa_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, ncls_or_mmo, device)

    results = {cond: [] for cond in CONDITIONS}

    if dataset == "Synapse":
        _run_synapse(tag, dataset, model_type, vit, source_sam, target, results, n_cases, device)
    elif dataset == "ISIC":
        _run_isic(tag, model_type, source_sam, target, results, n_cases, device)
    elif dataset == "REFUGE":
        _run_refuge(tag, model_type, source_sam, target, results, n_cases, device)

    summary = {}
    for cond in CONDITIONS:
        deltas = [r["delta"] for r in results[cond]]
        summary[cond] = {
            "mean_delta": float(np.mean(deltas)), "std_delta": float(np.std(deltas)),
            "mean_dice": float(np.mean([r["dice"] for r in results[cond]])),
            "n": len(deltas),
        }
    print(f"  [{tag}] " + " | ".join(f"{c}={summary[c]['mean_delta']:.4f}" for c in CONDITIONS))

    del source_sam, target; gc.collect(); torch.cuda.empty_cache()
    return summary, results


def _run_synapse(tag, dataset, model_type, vit, source_sam, target, results, n_volumes, device):
    import h5py
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:n_volumes]

    for vol_idx, h5f in enumerate(h5s):
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        organ_slices = sorted(
            [s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
            key=lambda s: (lbl[s] > 0).sum(), reverse=True,
        )[:NUM_SLICES_PER_VOLUME]
        if not organ_slices: continue

        for si in organ_slices:
            slc = img[si].astype(np.float32); label_slice = lbl[si]
            slc_3ch = np.stack([slc]*3, axis=0)
            img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(device)

            # Clean dice
            if model_type == "SAMed":
                clean_d = evaluate_samed_slice(target, slc, label_slice, f"{h5f}_s{si}")
            else:
                clean_d = evaluate_msa_synapse_slice(target, slc, label_slice, device)

            for cond in CONDITIONS:
                if cond == "no_probe":
                    results[cond].append({"case": h5f, "slice": si, "dice": clean_d, "delta": 0.0})
                    continue

                img_for_attack = _prepare_attack_input(img_t, cond)
                config = AttackConfig(
                    att_step_size=2.0, att_bound=10.0, att_numstep=10,
                    direction_mode=DIRECTION_MAP[cond], probe_amplification=2.0,
                    edge_guidance=True, seed=42,
                )
                random.seed(42); np.random.seed(42); torch.manual_seed(42)
                adv_np, _ = run_codd_attack(source_sam.image_encoder, img_for_attack, config, return_trace=True)

                adv_gray = adv_np[0].copy()
                if model_type == "SAMed":
                    adv_d = evaluate_samed_slice(target, adv_gray, label_slice, f"{h5f}_s{si}")
                else:
                    adv_d = evaluate_msa_synapse_slice(target, adv_gray, label_slice, device)

                results[cond].append({"case": h5f, "slice": si, "dice": float(adv_d), "delta": float(clean_d - adv_d), "clean_dice": float(clean_d)})

        print(f"  [{tag}] Volume {vol_idx+1}/{n_volumes} done ({len(organ_slices)} slices)")


def _run_isic(tag, model_type, source_sam, target, results, n_cases, device):
    samples = get_isic_samples(n_cases)
    for case_idx, (img_t, label, img_raw) in enumerate(samples):
        # Clean dice
        if model_type == "SAMed":
            clean_d = evaluate_samed_isic(target, img_t, torch.from_numpy(label).float(), device)
            img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        else:
            img_uint8_np = img_raw.mean(axis=2).astype(np.float32)
            label_t = torch.from_numpy(label).float()
            clean_d = evaluate_msa_isic(target, img_uint8_np, label_t, device)
            img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1)

        img_tensor = img_gray.unsqueeze(0).float().to(device)

        for cond in CONDITIONS:
            if cond == "no_probe":
                results[cond].append({"case": case_idx, "dice": clean_d, "delta": 0.0})
                continue

            img_for_attack = _prepare_attack_input(img_tensor, cond)
            config = AttackConfig(
                att_step_size=2.0, att_bound=10.0, att_numstep=10,
                direction_mode=DIRECTION_MAP[cond], probe_amplification=2.0,
                edge_guidance=True, seed=42,
            )
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            adv_np, _ = run_codd_attack(source_sam.image_encoder, img_for_attack, config, return_trace=True)

            if model_type == "SAMed":
                adv_gray = torch.from_numpy(adv_np.copy()).float()
                adv_d = evaluate_samed_isic(target, adv_gray, torch.from_numpy(label).float(), device)
            else:
                adv_uint8 = normalized_gray_to_uint8(adv_np[0].copy())
                label_t = torch.from_numpy(label).float()
                adv_d = evaluate_msa_isic(target, adv_uint8, label_t, device)

            results[cond].append({"case": case_idx, "dice": float(adv_d), "delta": float(clean_d - adv_d), "clean_dice": float(clean_d)})

        print(f"  [{tag}] Case {case_idx+1}/{n_cases}: clean={clean_d:.4f}")


def _run_refuge(tag, model_type, source_sam, target, results, n_cases, device):
    samples = get_refuge_samples(n_cases)
    for case_idx, (img_t, label, img_raw) in enumerate(samples):
        if model_type == "SAMed":
            clean_d = evaluate_samed_refuge(target, img_t, torch.from_numpy(label).float(), device)
            img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        else:
            img_uint8_np = img_raw.mean(axis=2).astype(np.float32)
            label_t = torch.from_numpy(label).float()
            clean_d = evaluate_msa_refuge(target, img_uint8_np, label_t, device)
            img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1)

        img_tensor = img_gray.unsqueeze(0).float().to(device)

        for cond in CONDITIONS:
            if cond == "no_probe":
                results[cond].append({"case": case_idx, "dice": clean_d, "delta": 0.0})
                continue

            img_for_attack = _prepare_attack_input(img_tensor, cond)
            config = AttackConfig(
                att_step_size=2.0, att_bound=10.0, att_numstep=10,
                direction_mode=DIRECTION_MAP[cond], probe_amplification=2.0,
                edge_guidance=True, seed=42,
            )
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            adv_np, _ = run_codd_attack(source_sam.image_encoder, img_for_attack, config, return_trace=True)

            if model_type == "SAMed":
                adv_gray = torch.from_numpy(adv_np.copy()).float()
                adv_d = evaluate_samed_refuge(target, adv_gray, torch.from_numpy(label).float(), device)
            else:
                adv_uint8 = normalized_gray_to_uint8(adv_np[0].copy())
                adv_d = evaluate_msa_refuge(target, adv_uint8, label_t, device)

            results[cond].append({"case": case_idx, "dice": float(adv_d), "delta": float(clean_d - adv_d), "clean_dice": float(clean_d)})

        print(f"  [{tag}] Case {case_idx+1}/{n_cases}: clean={clean_d:.4f}")


def _prepare_attack_input(img_tensor, cond):
    """Apply phase manipulation before attack."""
    if cond == "phase_shuffled":
        G = torch.fft.fft2(img_tensor, norm="ortho")
        mag = torch.abs(G); phase = torch.angle(G)
        B, C, H, W = phase.shape
        phase_shuffled = phase.view(B, C, -1)[:, :, torch.randperm(H*W, device=img_tensor.device)].view_as(phase)
        G_shuffled = mag * torch.exp(1j * phase_shuffled)
        return torch.fft.ifft2(G_shuffled, norm="ortho").real
    elif cond == "random_phase":
        G = torch.fft.fft2(img_tensor, norm="ortho")
        mag = torch.abs(G)
        rand_phase = torch.rand_like(G, device=img_tensor.device) * 2 * np.pi
        G_rand = mag * torch.exp(1j * rand_phase)
        return torch.fft.ifft2(G_rand, norm="ortho").real
    else:
        return img_tensor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--subset", type=str, default="0/1", help="subset_index/total_subsets")
    parser.add_argument("--models", type=str, default="", help="comma-separated model tags to run (overrides subset)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using device: {device}")

    all_models = build_model_list()
    print(f"Total available models: {len(all_models)}")

    # Select model subset
    if args.models:
        requested = set(args.models.split(","))
        models = [m for m in all_models if m[0] in requested]
    else:
        subset_idx, total = map(int, args.subset.split("/"))
        models = [m for i, m in enumerate(all_models) if i % total == subset_idx]

    print(f"Running {len(models)} models: {[m[0] for m in models]}")

    # Per-worker output to avoid race conditions
    worker_id = f"gpu{args.gpu}_sub{args.subset.replace('/', '_')}"
    output_path = os.path.join(OUTPUT_DIR, f"exp14_worker_{worker_id}.json")
    all_summaries = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            all_summaries = json.load(f).get("results", {})

    for tag, dataset, model_type, vit, ckpt, ncls_or_mmo in models:
        if tag in all_summaries:
            print(f"SKIP {tag}: already completed")
            continue

        try:
            summary, _ = run_experiment(tag, dataset, model_type, vit, ckpt, ncls_or_mmo, device)
            all_summaries[tag] = summary
        except Exception as e:
            print(f"  [ERROR] {tag}: {e}")
            import traceback; traceback.print_exc()

        # Save incrementally
        output = {"config": {"conditions": CONDITIONS, "n_models": len(all_models)},
                  "results": all_summaries}
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(output, f, indent=2)
        os.rename(tmp_path, output_path)
        print(f"  [SAVED] {tag} → {output_path} ({len(all_summaries)} models)")

    print(f"\nDONE. {len(all_summaries)} models completed.")


if __name__ == "__main__":
    main()
