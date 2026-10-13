#!/usr/bin/env python3
"""
Experiment 15: Complete remaining experiments.
Modes:
  --mode heatmap     : Exp12 full 18-target SE-CGR heatmap (11 missing models)
  --mode defense_msa : Exp10 MSA defense on 3 datasets
  --mode theory     : Exp1+2 appendix — radial dominance + carrier null on vit_l/vit_h
  --mode amplitude  : Exp5 expansion — amplitude ablation on vit_l/vit_h
  --mode hyperparams: Exp9 expansion — hyperparameter ablation on vit_l/vit_h

Parallel launch:
  python exp15_remaining.py --gpu 0 --mode heatmap --subset 0/3 &
  python exp15_remaining.py --gpu 1 --mode heatmap --subset 1/3 &
  python exp15_remaining.py --gpu 2 --mode heatmap --subset 2/3 &
  python exp15_remaining.py --gpu 3 --mode defense_msa &
  python exp15_remaining.py --gpu 4 --mode theory &
  python exp15_remaining.py --gpu 5 --mode amplitude &
  python exp15_remaining.py --gpu 0 --mode hyperparams &
"""
import os, sys, json, time, random, gc, argparse, copy
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from utils import calculate_metric_percase

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

# ---------------------------------------------------------------------------
# Model configs (same as exp14)
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


def build_model_list(datasets=None, architectures=None, vits=None):
    models = []
    for ds in datasets or ["Synapse", "ISIC", "REFUGE"]:
        for vit in vits or ["vit_b", "vit_l", "vit_h"]:
            if architectures is None or "SAMed" in architectures:
                ckpt, ncls = SAMED_CFG[ds][vit]
                if os.path.exists(ckpt):
                    models.append((f"{ds}_SAMed_{vit}", ds, "SAMed", vit, ckpt, ncls))
            if architectures is None or "MSA" in architectures:
                ckpt, mmo = MSA_CFG[ds][vit]
                if os.path.exists(ckpt):
                    models.append((f"{ds}_MSA_{vit}", ds, "MSA", vit, ckpt, mmo))
    return models


# ---------------------------------------------------------------------------
# Evaluation functions (same as exp14)
# ---------------------------------------------------------------------------

def evaluate_samed_slice(net, image_np, label_np, case_name):
    """Evaluate SAMed on a single Synapse slice — device-safe version."""
    from scipy.ndimage import zoom
    device = next(net.parameters()).device
    if image_np.ndim == 2: image_np = image_np[np.newaxis, :, :]
    if label_np.ndim == 2: label_np = label_np[np.newaxis, :, :]
    prediction = np.zeros_like(label_np)
    for ind in range(image_np.shape[0]):
        slc = image_np[ind, :, :]
        x, y = slc.shape[0], slc.shape[1]
        if x != IMAGE_SIZE or y != IMAGE_SIZE:
            slc = zoom(slc, (IMAGE_SIZE / x, IMAGE_SIZE / y), order=3)
        nx, ny = slc.shape[0], slc.shape[1]
        if nx != IMAGE_SIZE or ny != IMAGE_SIZE:
            slc = zoom(slc, (IMAGE_SIZE / nx, IMAGE_SIZE / ny), order=3)
        inputs = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).float().to(device)
        inputs = inputs.repeat(1, 3, 1, 1)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, True, IMAGE_SIZE)
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            oh, ow = out.shape
            if x != oh or y != ow:
                pred = zoom(out, (x / oh, y / ow), order=0)
            else:
                pred = out
            prediction[ind] = pred
    metric_list = []
    for i in range(1, 9):
        metric_list.append(calculate_metric_percase(prediction == i, label_np == i))
    return float(np.mean(metric_list, axis=0)[0])


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
        d = 2.0 * (pb * gt).sum() / (pb.sum() + gt.sum() + 1e-8)
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
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb, multimask_output=False)
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
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb, multimask_output=False)
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
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb, multimask_output=True)
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    pred_classes = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    dices = []
    for c in range(1, 9):
        pc = (pred_classes == c).astype(np.float32); tc = (label_slice == c).astype(np.float32)
        inter = (pc * tc).sum(); denom = pc.sum() + tc.sum()
        dices.append(2.0 * inter / (denom + 1e-8) if denom > 0 else 1.0)
    return float(np.mean(dices))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_isic_samples(n_cases):
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    df = pd.read_csv(os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv"))
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    samples = []
    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, row['img']); gt_path = os.path.join(data_dir, row['seg'])
        if not os.path.exists(img_path): continue
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
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    samples = []
    for folder in folders:
        img_path = os.path.join(data_dir, folder, f"{folder}.jpg")
        if not os.path.exists(img_path): continue
        img = Image.open(img_path).convert('RGB')
        img_raw = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
        disc_raw = np.array(Image.open(os.path.join(data_dir, folder, f"{folder}_disc.bmp")).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        cup_raw = np.array(Image.open(os.path.join(data_dir, folder, f"{folder}_cup.bmp")).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        label = np.stack([(cup_raw < 128).astype(np.float32), (disc_raw <= 128).astype(np.float32),
                          1.0 - (disc_raw <= 128).astype(np.float32) - (cup_raw < 128).astype(np.float32)], axis=0)
        samples.append((transform(img), label, img_raw))
        if len(samples) >= n_cases: break
    return samples


def normalized_gray_to_uint8(gray_normalized):
    if gray_normalized.ndim == 3 and gray_normalized.shape[0] == 3:
        gray_normalized = gray_normalized[0]
    MEAN_AVG = (0.485 + 0.456 + 0.406) / 3.0; STD_AVG = (0.229 + 0.224 + 0.225) / 3.0
    return np.clip((gray_normalized * STD_AVG + MEAN_AVG) * 255, 0, 255).astype(np.float32)


def run_se_cgr_attack(source_sam, img_tensor, device):
    config = AttackConfig(att_step_size=2.0, att_bound=10.0, att_numstep=10,
                          direction_mode="radial", probe_amplification=2.0,
                          edge_guidance=True, seed=42)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    adv_np, _ = run_codd_attack(source_sam.image_encoder, img_tensor, config, return_trace=True)
    return adv_np


# ===========================================================================
# MODE: heatmap — Full 18-target SE-CGR cross-arch heatmap
# ===========================================================================

def run_heatmap(models, device):
    results = {}
    for tag, dataset, model_type, vit, ckpt, ncls_or_mmo in models:
        print(f"\n{'='*50}\nHEATMAP: {tag}\n{'='*50}")
        try:
            source_sam = load_sam_encoder(vit, SAM_CKPT[vit], IMAGE_SIZE, 8, device)
            if model_type == "SAMed":
                target = load_samed_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, ncls_or_mmo, device=device)
            else:
                target = load_medsa_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, ncls_or_mmo, device)

            n_cases = 8 if dataset != "Synapse" else 3
            all_deltas = []

            if dataset == "Synapse":
                import h5py
                tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
                h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:n_cases]
                for h5f in h5s:
                    with h5py.File(os.path.join(tdir, h5f), 'r') as f:
                        img, lbl = f['image'][:], f['label'][:]
                    organ_slices = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
                                          key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:4]
                    for si in organ_slices:
                        slc = img[si].astype(np.float32); label_slice = lbl[si]
                        slc_3ch = np.stack([slc]*3, axis=0)
                        img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(device)
                        if model_type == "SAMed":
                            clean_d = evaluate_samed_slice(target, slc, label_slice, f"{h5f}_s{si}")
                        else:
                            clean_d = evaluate_msa_synapse_slice(target, slc, label_slice, device)
                        adv_np = run_se_cgr_attack(source_sam, img_t, device)
                        if model_type == "SAMed":
                            adv_d = evaluate_samed_slice(target, adv_np[0].copy(), label_slice, f"{h5f}_s{si}")
                        else:
                            adv_d = evaluate_msa_synapse_slice(target, adv_np[0].copy(), label_slice, device)
                        all_deltas.append(clean_d - adv_d)
            elif dataset == "ISIC":
                samples = get_isic_samples(n_cases)
                for img_t, label, img_raw in samples:
                    img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1)
                    img_tensor = img_gray.unsqueeze(0).float().to(device)
                    if model_type == "SAMed":
                        clean_d = evaluate_samed_isic(target, img_t, torch.from_numpy(label).float(), device)
                    else:
                        clean_d = evaluate_msa_isic(target, img_raw.mean(axis=2).astype(np.float32),
                                                    torch.from_numpy(label).float(), device)
                    adv_np = run_se_cgr_attack(source_sam, img_tensor, device)
                    if model_type == "SAMed":
                        adv_d = evaluate_samed_isic(target, torch.from_numpy(adv_np.copy()).float(),
                                                    torch.from_numpy(label).float(), device)
                    else:
                        adv_d = evaluate_msa_isic(target, adv_np[0].copy(),
                                                  torch.from_numpy(label).float(), device)
                    all_deltas.append(clean_d - adv_d)
            elif dataset == "REFUGE":
                samples = get_refuge_samples(n_cases)
                for img_t, label, img_raw in samples:
                    img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1)
                    img_tensor = img_gray.unsqueeze(0).float().to(device)
                    if model_type == "SAMed":
                        clean_d = evaluate_samed_refuge(target, img_t, torch.from_numpy(label).float(), device)
                    else:
                        clean_d = evaluate_msa_refuge(target, img_raw.mean(axis=2).astype(np.float32),
                                                      torch.from_numpy(label).float(), device)
                    adv_np = run_se_cgr_attack(source_sam, img_tensor, device)
                    if model_type == "SAMed":
                        adv_d = evaluate_samed_refuge(target, torch.from_numpy(adv_np.copy()).float(),
                                                      torch.from_numpy(label).float(), device)
                    else:
                        adv_d = evaluate_msa_refuge(target, normalized_gray_to_uint8(adv_np[0].copy()),
                                                    torch.from_numpy(label).float(), device)
                    all_deltas.append(clean_d - adv_d)

            deltas = np.array(all_deltas)
            results[tag] = {"se_cgr_mean_delta": float(np.mean(deltas)), "se_cgr_std_delta": float(np.std(deltas)),
                            "no_probe_mean_delta": 0.0, "improvement": float(np.mean(deltas)), "n": len(deltas)}
            print(f"  [{tag}] ΔDice={np.mean(deltas):+.4f} ± {np.std(deltas):.4f} (n={len(deltas)})")
            del source_sam, target; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] {tag}: {e}"); import traceback; traceback.print_exc()
    return results


# ===========================================================================
# MODE: defense_msa — Defense robustness on MSA models
# ===========================================================================

DEFENSE_METHODS = ["none", "jpeg_q75", "jpeg_q50", "gaussian_blur_s1", "gaussian_blur_s2",
                   "median_3x3", "median_5x5", "bitdepth_4", "resize_pad"]

def apply_defense(x, method):
    """Apply defense to adversarial image. x: [H,W] numpy float32."""
    from scipy.ndimage import gaussian_filter, median_filter
    from PIL import Image as PILImage
    from io import BytesIO
    if method == "none": return x.copy()
    x_u8 = np.clip(x, 0, 255).astype(np.uint8)
    if method == "jpeg_q75":
        buf = BytesIO(); PILImage.fromarray(x_u8).save(buf, format='JPEG', quality=75); buf.seek(0)
        return np.array(PILImage.open(buf)).astype(np.float32)
    if method == "jpeg_q50":
        buf = BytesIO(); PILImage.fromarray(x_u8).save(buf, format='JPEG', quality=50); buf.seek(0)
        return np.array(PILImage.open(buf)).astype(np.float32)
    if method == "gaussian_blur_s1": return gaussian_filter(x, sigma=1.0)
    if method == "gaussian_blur_s2": return gaussian_filter(x, sigma=2.0)
    if method == "median_3x3": return median_filter(x, size=3)
    if method == "median_5x5": return median_filter(x, size=5)
    if method == "bitdepth_4":
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x_q = np.round(x_norm * 15) / 15
        return x_q * (x.max() - x.min()) + x.min()
    if method == "resize_pad":
        from PIL import Image as PILImage
        s = IMAGE_SIZE // 2
        img = PILImage.fromarray(x_u8).resize((s, s), PILImage.BILINEAR).resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR)
        return np.array(img).astype(np.float32)
    return x.copy()


def run_defense_msa(models, device):
    results = {}
    for tag, dataset, model_type, vit, ckpt, ncls_or_mmo in models:
        if model_type != "MSA": continue
        print(f"\n{'='*50}\nDEFENSE MSA: {tag}\n{'='*50}")
        try:
            source_sam = load_sam_encoder(vit, SAM_CKPT[vit], IMAGE_SIZE, 8, device)
            target = load_medsa_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, ncls_or_mmo, device)
            results[tag] = {}
            n_cases = 4 if dataset == "Synapse" else 6

            if dataset == "Synapse":
                import h5py
                tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
                h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:n_cases]
                for method in DEFENSE_METHODS:
                    deltas = []
                    for h5f in h5s:
                        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
                            img, lbl = f['image'][:], f['label'][:]
                        organ_slices = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
                                              key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:4]
                        for si in organ_slices:
                            slc = img[si].astype(np.float32); label_slice = lbl[si]
                            clean_d = evaluate_msa_synapse_slice(target, slc, label_slice, device)
                            if method == "none":
                                deltas.append(0.0); continue
                            slc_3ch = np.stack([slc]*3, axis=0)
                            img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(device)
                            adv_np = run_se_cgr_attack(source_sam, img_t, device)
                            adv_def = apply_defense(adv_np[0].copy(), method)
                            adv_d = evaluate_msa_synapse_slice(target, adv_def, label_slice, device)
                            deltas.append(clean_d - adv_d)
                    results[tag][method] = {"mean_delta": float(np.mean(deltas)), "std_delta": float(np.std(deltas)), "n": len(deltas)}
                    print(f"  {method}: mean_delta={np.mean(deltas):+.4f}")
            elif dataset == "ISIC":
                samples = get_isic_samples(n_cases)
                for method in DEFENSE_METHODS:
                    deltas = []
                    for img_t, label, img_raw in samples:
                        img_uint8 = img_raw.mean(axis=2).astype(np.float32)
                        label_t = torch.from_numpy(label).float()
                        clean_d = evaluate_msa_isic(target, img_uint8, label_t, device)
                        if method == "none":
                            deltas.append(0.0); continue
                        img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1)
                        img_tensor = img_gray.unsqueeze(0).float().to(device)
                        adv_np = run_se_cgr_attack(source_sam, img_tensor, device)
                        adv_def = apply_defense(normalized_gray_to_uint8(adv_np[0].copy()), method)
                        adv_d = evaluate_msa_isic(target, adv_def, label_t, device)
                        deltas.append(clean_d - adv_d)
                    results[tag][method] = {"mean_delta": float(np.mean(deltas)), "std_delta": float(np.std(deltas)), "n": len(deltas)}
                    print(f"  {method}: mean_delta={np.mean(deltas):+.4f}")
            elif dataset == "REFUGE":
                samples = get_refuge_samples(n_cases)
                for method in DEFENSE_METHODS:
                    deltas = []
                    for img_t, label, img_raw in samples:
                        img_uint8 = img_raw.mean(axis=2).astype(np.float32)
                        label_t = torch.from_numpy(label).float()
                        clean_d = evaluate_msa_refuge(target, img_uint8, label_t, device)
                        if method == "none":
                            deltas.append(0.0); continue
                        img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1)
                        img_tensor = img_gray.unsqueeze(0).float().to(device)
                        adv_np = run_se_cgr_attack(source_sam, img_tensor, device)
                        adv_def = apply_defense(normalized_gray_to_uint8(adv_np[0].copy()), method)
                        adv_d = evaluate_msa_refuge(target, adv_def, label_t, device)
                        deltas.append(clean_d - adv_d)
                    results[tag][method] = {"mean_delta": float(np.mean(deltas)), "std_delta": float(np.std(deltas)), "n": len(deltas)}
                    print(f"  {method}: mean_delta={np.mean(deltas):+.4f}")
            del source_sam, target; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] {tag}: {e}"); import traceback; traceback.print_exc()
    return results


# ===========================================================================
# MODE: theory — Radial dominance + carrier null on vit_l/vit_h
# ===========================================================================

def compute_radial_ratio(source_feat, target_feat):
    """Compute radial ratio between source and target features."""
    diff = target_feat - source_feat
    G = torch.fft.fft2(diff.float(), norm="ortho")
    mag = torch.abs(G); phase = torch.angle(G)
    G_source = torch.fft.fft2(source_feat.float(), norm="ortho")
    source_phase = torch.angle(G_source)
    radial = (mag * torch.cos(phase - source_phase)).abs().mean().item()
    tangential = (mag * torch.sin(phase - source_phase)).abs().mean().item()
    total = radial + tangential + 1e-12
    return radial / total


def run_theory_appendix(models, device):
    results = {}
    for tag, dataset, model_type, vit, ckpt, ncls_or_mmo in models:
        if dataset != "Synapse": continue  # Only Synapse for appendix
        print(f"\n{'='*50}\nTHEORY APPENDIX: {tag}\n{'='*50}")
        try:
            source_sam = load_sam_encoder(vit, SAM_CKPT[vit], IMAGE_SIZE, 8, device)
            if model_type == "SAMed":
                target = load_samed_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, ncls_or_mmo, device=device)
            else:
                target = load_medsa_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, ncls_or_mmo, device)

            import h5py
            tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
            h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:10]
            layer_ratios = {l: [] for l in range(12 if vit == "vit_b" else 24 if vit == "vit_l" else 32)}
            carrier_results = {"source_carrier": [], "shuffled_carrier": [], "random_carrier": []}

            for h5f in h5s:
                with h5py.File(os.path.join(tdir, h5f), 'r') as f:
                    img, lbl = f['image'][:], f['label'][:]
                organ_slices = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
                                      key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:2]
                for si in organ_slices:
                    slc = img[si].astype(np.float32)
                    slc_3ch = np.stack([slc]*3, axis=0)
                    img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(device)

                    # Layer-wise radial ratio
                    with torch.no_grad():
                        s_feats, s_layers = source_sam.image_encoder(img_t)
                        if model_type == "SAMed":
                            t_feats, t_layers = target.sam.image_encoder(img_t)
                        else:
                            imgs_pre = target.preprocess(img_t)
                            enc_out = target.image_encoder(imgs_pre)
                            if isinstance(enc_out, tuple):
                                t_feats, t_layers = enc_out
                            else:
                                t_feats = enc_out
                                t_layers = []

                    n_layers = len(s_layers)
                    if len(t_layers) > 0:
                        indices = list(range(n_layers))
                        random.shuffle(indices)
                        for li in indices[:min(8, n_layers)]:
                            rr = compute_radial_ratio(s_layers[li], t_layers[li] if li < len(t_layers) else t_feats)
                            layer_ratios[li].append(rr)

                    # Carrier null test
                    with torch.no_grad():
                        s_final = s_feats
                        t_final = t_feats
                    diff = t_final - s_final
                    G = torch.fft.fft2(diff.float(), norm="ortho")
                    # Source carrier
                    G_s = torch.fft.fft2(s_final.float(), norm="ortho")
                    s_phase = torch.angle(G_s)
                    rad_s = (torch.abs(G) * torch.cos(torch.angle(G) - s_phase)).abs().mean().item()
                    tan_s = (torch.abs(G) * torch.sin(torch.angle(G) - s_phase)).abs().mean().item()
                    carrier_results["source_carrier"].append(rad_s / (rad_s + tan_s + 1e-12))
                    # Shuffled carrier
                    B, C, H, W = s_phase.shape
                    s_phase_shuf = s_phase.view(B, C, -1)[:, :, torch.randperm(H*W, device=device)].view_as(s_phase)
                    rad_sh = (torch.abs(G) * torch.cos(torch.angle(G) - s_phase_shuf)).abs().mean().item()
                    tan_sh = (torch.abs(G) * torch.sin(torch.angle(G) - s_phase_shuf)).abs().mean().item()
                    carrier_results["shuffled_carrier"].append(rad_sh / (rad_sh + tan_sh + 1e-12))
                    # Random carrier
                    rand_phase = torch.rand_like(G) * 2 * np.pi
                    rad_rand = (torch.abs(G) * torch.cos(torch.angle(G) - rand_phase)).abs().mean().item()
                    tan_rand = (torch.abs(G) * torch.sin(torch.angle(G) - rand_phase)).abs().mean().item()
                    carrier_results["random_carrier"].append(rad_rand / (rad_rand + tan_rand + 1e-12))

            summary = {}
            mean_rrs = []
            for li, vals in sorted(layer_ratios.items()):
                if vals:
                    summary[str(li)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                    mean_rrs.extend(vals)
            results[tag] = {
                "mean_radial_ratio": float(np.mean(mean_rrs)),
                "std_radial_ratio": float(np.std(mean_rrs)),
                "num_samples": len(mean_rrs),
                "layers_above_05": sum(1 for li in summary.values() if li["mean"] > 0.5),
                "total_layers": len(summary),
                "carrier_null": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                                 for k, v in carrier_results.items()},
            }
            print(f"  [{tag}] RR={results[tag]['mean_radial_ratio']:.4f}, "
                  f"source={results[tag]['carrier_null']['source_carrier']['mean']:.4f}, "
                  f"shuffled={results[tag]['carrier_null']['shuffled_carrier']['mean']:.4f}")
            del source_sam, target; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] {tag}: {e}"); import traceback; traceback.print_exc()
    return results


# ===========================================================================
# MODE: amplitude — Exp5 expansion on vit_l/vit_h Synapse
# ===========================================================================

def run_amplitude_expansion(models, device):
    results = {}
    conditions = ["clean_spectrum", "flat_spectrum", "shuffled_spectrum", "lowpass", "highpass"]
    for tag, dataset, model_type, vit, ckpt, ncls_or_mmo in models:
        if dataset != "Synapse" or model_type != "SAMed": continue
        print(f"\n{'='*50}\nAMPLITUDE: {tag}\n{'='*50}")
        try:
            source_sam = load_sam_encoder(vit, SAM_CKPT[vit], IMAGE_SIZE, 8, device)
            target = load_samed_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, 8, device=device)
            import h5py
            tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
            h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:6]
            cond_results = {c: [] for c in conditions}
            for h5f in h5s:
                with h5py.File(os.path.join(tdir, h5f), 'r') as f:
                    img, lbl = f['image'][:], f['label'][:]
                organ_slices = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
                                      key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:2]
                for si in organ_slices:
                    slc = img[si].astype(np.float32); label_slice = lbl[si]
                    clean_d = evaluate_samed_slice(target, slc, label_slice, f"{h5f}_s{si}")
                    slc_3ch = np.stack([slc]*3, axis=0)
                    img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(device)
                    G = torch.fft.fft2(img_t, norm="ortho")
                    mag = torch.abs(G); phase = torch.angle(G)
                    H, W = mag.shape[-2], mag.shape[-1]
                    for cond in conditions:
                        if cond == "clean_spectrum":
                            amp = 2.0 * mag
                        elif cond == "flat_spectrum":
                            amp = 2.0 * torch.ones_like(mag)
                        elif cond == "shuffled_spectrum":
                            amp = 2.0 * mag.view(mag.shape[0], -1)[:, torch.randperm(mag.shape[1]*H*W, device=device)].view_as(mag)
                        elif cond == "lowpass":
                            yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                            r = torch.sqrt((yy - H/2)**2 + (xx - W/2)**2)
                            amp = 2.0 * mag * (r < min(H, W) * 0.15).float().unsqueeze(0).unsqueeze(0)
                        elif cond == "highpass":
                            yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                            r = torch.sqrt((yy - H/2)**2 + (xx - W/2)**2)
                            amp = 2.0 * mag * (r > min(H, W) * 0.15).float().unsqueeze(0).unsqueeze(0)
                        G_mod = amp * torch.exp(1j * phase)
                        img_mod = torch.fft.ifft2(G_mod, norm="ortho").real
                        adv_np = run_se_cgr_attack(source_sam, img_mod, device)
                        adv_d = evaluate_samed_slice(target, adv_np[0].copy(), label_slice, f"{h5f}_s{si}")
                        cond_results[cond].append(clean_d - adv_d)
            summary = {}
            for cond in conditions:
                summary[cond] = {"mean_delta": float(np.mean(cond_results[cond])), "std_delta": float(np.std(cond_results[cond])), "n": len(cond_results[cond])}
            results[tag] = summary
            print(f"  [{tag}] " + " | ".join(f"{c}={summary[c]['mean_delta']:+.4f}" for c in conditions))
            del source_sam, target; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] {tag}: {e}"); import traceback; traceback.print_exc()
    return results


# ===========================================================================
# MODE: hyperparams — Exp9 expansion on vit_l/vit_h Synapse
# ===========================================================================

def run_hyperparams_expansion(models, device):
    results = {}
    for tag, dataset, model_type, vit, ckpt, ncls_or_mmo in models:
        if dataset != "Synapse" or model_type != "SAMed": continue
        print(f"\n{'='*50}\nHYPERPARAMS: {tag}\n{'='*50}")
        try:
            source_sam = load_sam_encoder(vit, SAM_CKPT[vit], IMAGE_SIZE, 8, device)
            target = load_samed_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, 8, device=device)
            import h5py
            tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
            h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:3]
            configs = [("eps=8_steps=20", 8, 20), ("eps=16_steps=20", 16, 20),
                       ("eps=16_steps=10", 16, 10), ("eps=16_steps=40", 16, 40)]
            for cfg_name, eps, steps in configs:
                deltas = []
                for h5f in h5s:
                    with h5py.File(os.path.join(tdir, h5f), 'r') as f:
                        img, lbl = f['image'][:], f['label'][:]
                    organ_slices = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
                                          key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:2]
                    for si in organ_slices:
                        slc = img[si].astype(np.float32); label_slice = lbl[si]
                        clean_d = evaluate_samed_slice(target, slc, label_slice, f"{h5f}_s{si}")
                        slc_3ch = np.stack([slc]*3, axis=0)
                        img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(device)
                        config = AttackConfig(att_step_size=2.0, att_bound=float(eps), att_numstep=steps,
                                              direction_mode="radial", probe_amplification=2.0,
                                              edge_guidance=True, seed=42)
                        random.seed(42); np.random.seed(42); torch.manual_seed(42)
                        adv_np, _ = run_codd_attack(source_sam.image_encoder, img_t, config, return_trace=True)
                        adv_d = evaluate_samed_slice(target, adv_np[0].copy(), label_slice, f"{h5f}_s{si}")
                        deltas.append(clean_d - adv_d)
                results[f"{tag}_{cfg_name}"] = {"mean_delta": float(np.mean(deltas)), "std_delta": float(np.std(deltas)), "n": len(deltas)}
                print(f"  {cfg_name}: mean_delta={np.mean(deltas):+.4f}")
            del source_sam, target; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] {tag}: {e}"); import traceback; traceback.print_exc()
    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["heatmap", "defense_msa", "theory", "amplitude", "hyperparams"])
    parser.add_argument("--subset", type=str, default="0/1")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Mode: {args.mode}, Device: {device}")

    # Select models based on mode
    if args.mode == "heatmap":
        # All 18 models — run subset for parallelization
        all_models = build_model_list()
        subset_idx, total = map(int, args.subset.split("/"))
        models = [m for i, m in enumerate(all_models) if i % total == subset_idx]
        output_file = f"exp15_heatmap_gpu{args.gpu}.json"
        results = run_heatmap(models, device)
    elif args.mode == "defense_msa":
        models = build_model_list(architectures=["MSA"], vits=["vit_b"])
        output_file = "exp15_defense_msa.json"
        results = run_defense_msa(models, device)
    elif args.mode == "theory":
        models = build_model_list(datasets=["Synapse"], vits=["vit_l", "vit_h"])
        output_file = "exp15_theory_appendix.json"
        results = run_theory_appendix(models, device)
    elif args.mode == "amplitude":
        models = build_model_list(datasets=["Synapse"], architectures=["SAMed"], vits=["vit_l", "vit_h"])
        output_file = "exp15_amplitude_expansion.json"
        results = run_amplitude_expansion(models, device)
    elif args.mode == "hyperparams":
        models = build_model_list(datasets=["Synapse"], architectures=["SAMed"], vits=["vit_l", "vit_h"])
        output_file = "exp15_hyperparams_expansion.json"
        results = run_hyperparams_expansion(models, device)

    # Save
    out_path = os.path.join(OUTPUT_DIR, output_file)
    with open(out_path, "w") as f:
        json.dump({"mode": args.mode, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path} ({len(results)} entries)")
    print("DONE")


if __name__ == "__main__":
    main()
