#!/usr/bin/env python3
"""
PAID Theory v4: Frequency-Band Phase Sensitivity with Fixed Evaluation.

Key fix: REFUGE uses _disc.bmp + _cup.bmp, not _seg.png.
Uses dataset-specific evaluation functions matching Exp14 exactly.

PSI = high-freq ΔDice (normalized by clean Dice for cross-dataset comparison).
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

MSA_CFG = {
    "Synapse": {
        "vit_b": ("MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs24_lr0.0001_s42/Model/checkpoint_best.pth", 8),
        "vit_l": ("MedicalSAMAdapter/Synapse_512_pretrain_vit_l_epo100_bs12_lr0.0001_s42/Model/checkpoint_best.pth", 8),
        "vit_h": ("MedicalSAMAdapter/Synapse_512_pretrain_vit_h_epo100_bs6_lr0.0001_s42/Model/checkpoint_best.pth", 8),
    },
    "ISIC": {
        "vit_b": ("MedicalSAMAdapter/ISIC_512_pretrain_vit_b_epo100_bs24_lr0.0001_s42/Model/checkpoint_best.pth", 1),
        "vit_l": ("MedicalSAMAdapter/ISIC_512_pretrain_vit_l_epo100_bs12_lr0.0001_s42/Model/checkpoint_best.pth", 1),
        "vit_h": ("MedicalSAMAdapter/ISIC_512_pretrain_vit_h_epo100_bs6_lr0.0001_s42/Model/checkpoint_best.pth", 1),
    },
    "REFUGE": {
        "vit_b": ("MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 2),
        "vit_l": ("MedicalSAMAdapter/REFUGE_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 2),
        "vit_h": ("MedicalSAMAdapter/REFUGE_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24/Model/checkpoint_best.pth", 2),
    },
}


# =========================================================================
# Evaluation functions (matching Exp14 exactly)
# =========================================================================

def evaluate_samed_generic(net, image_np, label_np, num_classes):
    """Generic SAMed evaluation for Synapse/ISIC."""
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
    metric_list = []
    for i in range(1, ncls):
        pred_bin = (prediction == i)
        gt_bin = (label_np == i)
        if gt_bin.sum() > 0:
            metric_list.append(calculate_metric_percase(pred_bin, gt_bin))
    if len(metric_list) == 0:
        metric_list = [calculate_metric_percase(prediction > 0.5, label_np > 0)]
    return float(np.mean(metric_list, axis=0)[0])


def evaluate_samed_refuge(net, image_np, label, device):
    """SAMed REFUGE evaluation matching Exp14 exactly.
    label: (2, H, W) float32 [cup, disc]
    """
    if image_np.ndim == 2:
        image_np = image_np[np.newaxis, :, :]
    img_3ch = np.stack([image_np[0] if image_np.shape[0] == 1 else image_np[0]] * 3, axis=0)
    img_t = torch.from_numpy(img_3ch).unsqueeze(0).float().to(device)
    with torch.no_grad():
        outputs = net(img_t, True, IMAGE_SIZE)
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


def evaluate_msa_synapse(net, image_np, label_slice, device):
    """MSA Synapse evaluation matching Exp14 exactly."""
    img_3ch = np.stack([image_np] * 3, axis=0) if image_np.ndim == 2 else image_np[:3]
    img_t = torch.from_numpy(img_3ch).unsqueeze(0).float().to(device)
    pt_list, lbl_list = [], []
    for c in range(1, 9):
        class_mask = (label_slice == c).astype(np.uint8)
        indices = np.argwhere(class_mask > 0)
        if len(indices) > 0:
            r, cc = indices[random.randint(0, len(indices) - 1)]
            pt_list.append(np.array([cc, r]))
            lbl_list.append(1)
        else:
            pt_list.append(np.array([0, 0]))
            lbl_list.append(0)
    coords = torch.tensor(np.stack(pt_list, axis=0), dtype=torch.float, device=device).unsqueeze(0)
    labels_pt = torch.tensor(lbl_list, dtype=torch.int, device=device).unsqueeze(0)
    with torch.no_grad():
        imgs_pre = net.preprocess(img_t)
        image_emb = net.image_encoder(imgs_pre)
        sparse_emb, dense_emb = net.prompt_encoder(points=(coords, labels_pt), boxes=None, masks=None)
        pred, _ = net.mask_decoder(
            image_embeddings=image_emb, image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    pred_classes = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    dices = []
    for c in range(1, 9):
        pc = (pred_classes == c).astype(np.float32)
        tc = (label_slice == c).astype(np.float32)
        inter = (pc * tc).sum()
        denom = pc.sum() + tc.sum()
        dices.append(2.0 * inter / (denom + 1e-8) if denom > 0 else 1.0)
    return float(np.mean(dices))


def evaluate_msa_isic(net, image_np, gt_mask, device):
    """MSA ISIC evaluation matching Exp14 exactly."""
    from PIL import Image as PILImage
    from torchvision.transforms import Resize, Compose, ToTensor, Lambda
    img_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)
    if img_uint8.ndim == 3 and img_uint8.shape[0] == 3:
        img_uint8 = img_uint8.transpose(1, 2, 0)
    img_pil = PILImage.fromarray(img_uint8).convert('RGB')
    transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE)), ToTensor(), Lambda(lambda x: x * 255)])
    img_t = transform(img_pil).unsqueeze(0).to(device)
    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.cpu().numpy()
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    indices = np.argwhere(gt_mask > 0)
    if len(indices) == 0:
        pt = torch.tensor([[[0.0, 0.0]]], device=device)
        pt_label = torch.tensor([[0]], dtype=torch.int, device=device)
    else:
        r, c = indices[random.randint(0, len(indices) - 1)]
        pt = torch.tensor([[[float(c), float(r)]]], device=device)
        pt_label = torch.tensor([[1]], dtype=torch.int, device=device)
    with torch.no_grad():
        imgs_pre = net.preprocess(img_t)
        image_emb = net.image_encoder(imgs_pre)
        sparse_emb, dense_emb = net.prompt_encoder(points=(pt, pt_label), boxes=None, masks=None)
        pred, _ = net.mask_decoder(
            image_embeddings=image_emb, image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    probs = torch.sigmoid(pred).squeeze().cpu().numpy()
    best_d = 0.0
    for th in np.arange(0.1, 1.0, 0.1):
        pb = (probs > th).astype(np.float32)
        d = 2.0 * (pb * gt_mask).sum() / (pb.sum() + gt_mask.sum() + 1e-8)
        if d > best_d:
            best_d = d
    return float(best_d)


def evaluate_msa_refuge(net, image_np, gt_label, device):
    """MSA REFUGE evaluation matching Exp14 exactly.
    gt_label: (2, H, W) float32 [cup, disc]
    """
    from PIL import Image as PILImage
    from torchvision.transforms import Resize, Compose, ToTensor, Lambda
    img_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)
    if img_uint8.ndim == 3 and img_uint8.shape[0] == 3:
        img_uint8 = img_uint8.transpose(1, 2, 0)
    img_pil = PILImage.fromarray(img_uint8).convert('RGB')
    transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE)), ToTensor(), Lambda(lambda x: x * 255)])
    img_t = transform(img_pil).unsqueeze(0).to(device)
    pt_coords = torch.tensor([[[0.0, 0.0]]], device=device)
    pt_labels = torch.tensor([[1]], dtype=torch.int, device=device)
    with torch.no_grad():
        imgs_pre = net.preprocess(img_t)
        image_emb = net.image_encoder(imgs_pre)
        sparse_emb, dense_emb = net.prompt_encoder(points=(pt_coords, pt_labels), boxes=None, masks=None)
        pred, _ = net.mask_decoder(
            image_embeddings=image_emb, image_pe=net.prompt_encoder.get_dense_pe(),
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


# =========================================================================
# Phase band perturbation
# =========================================================================

def phase_band_noise(img_tensor, band='high', cutoff_ratio=0.25, noise_std=1.0):
    """Randomize phase in low or high frequency band."""
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

    delta_phi = torch.randn_like(Phi) * noise_std
    delta_phi[:, :, ~mask] = 0.0
    F_pert = A * torch.exp(1j * (Phi + delta_phi))
    img_pert = torch.fft.ifft2(F_pert, norm='ortho').real
    return torch.clamp(img_pert, img_tensor.min(), img_tensor.max())


# =========================================================================
# Data loading
# =========================================================================

def get_synapse_test_images(n_images=5):
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
            images.append((img[s].astype(np.float32), lbl[s]))
    return images


def get_isic_test_images(n_images=5):
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    csv_path = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = []
    for _, row in df.iterrows():
        if len(images) >= n_images:
            break
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
    """Load REFUGE test images with disc+cup labels.
    Returns: list of (image_np, label) where label is (2, H, W) [cup, disc]
    """
    from PIL import Image
    test_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    if not os.path.isdir(test_dir):
        return []
    folders = sorted([f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))])
    images = []
    for folder in folders:
        if len(images) >= n_images:
            break
        img_path = os.path.join(test_dir, folder, f"{folder}.jpg")
        disc_path = os.path.join(test_dir, folder, f"{folder}_disc.bmp")
        cup_path = os.path.join(test_dir, folder, f"{folder}_cup.bmp")
        if not os.path.exists(img_path) or not os.path.exists(disc_path) or not os.path.exists(cup_path):
            continue
        # Load image
        img_pil = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_np = np.array(img_pil, dtype=np.float32)  # (H, W, 3)
        img_gray = img_np.mean(axis=-1)  # (H, W) grayscale
        # Load disc and cup masks
        disc_raw = np.array(Image.open(disc_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        cup_raw = np.array(Image.open(cup_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        # REFUGE masks: dark pixels are the region of interest
        disc_mask = (disc_raw <= 128).astype(np.float32)
        cup_mask = (cup_raw < 128).astype(np.float32)
        label = np.stack([cup_mask, disc_mask], axis=0)  # [cup, disc] matching Exp14
        images.append((img_gray, label))
    return images


# =========================================================================
# Core PSI computation
# =========================================================================

def compute_band_psi_samed(net, ds_name, image_np, label, num_classes, device, n_trials=5, cutoff_ratio=0.25):
    """Compute band PSI for SAMed models."""
    # Clean baseline
    if ds_name == "REFUGE":
        clean_dice = evaluate_samed_refuge(net, image_np, label, device)
    else:
        clean_dice = evaluate_samed_generic(net, image_np, label, num_classes)

    high_deltas = []
    low_deltas = []

    for _ in range(n_trials):
        # Build 3-channel image tensor
        if image_np.ndim == 2:
            img_3ch = np.stack([image_np] * 3, axis=0)
        elif image_np.shape[0] == 1:
            img_3ch = np.stack([image_np[0]] * 3, axis=0)
        else:
            img_3ch = image_np[:3]
        img_t = torch.from_numpy(img_3ch).unsqueeze(0).float().to(device)

        # High-freq phase noise
        img_high = phase_band_noise(img_t, band='high', cutoff_ratio=cutoff_ratio, noise_std=1.0)
        img_high_np = img_high[0, 0].cpu().numpy()
        if ds_name == "REFUGE":
            dice_high = evaluate_samed_refuge(net, img_high_np, label, device)
        else:
            dice_high = evaluate_samed_generic(net, img_high_np, label, num_classes)
        high_deltas.append(max(0, clean_dice - dice_high))

        # Low-freq phase noise
        img_low = phase_band_noise(img_t, band='low', cutoff_ratio=cutoff_ratio, noise_std=1.0)
        img_low_np = img_low[0, 0].cpu().numpy()
        if ds_name == "REFUGE":
            dice_low = evaluate_samed_refuge(net, img_low_np, label, device)
        else:
            dice_low = evaluate_samed_generic(net, img_low_np, label, num_classes)
        low_deltas.append(max(0, clean_dice - dice_low))

    mean_high = np.mean(high_deltas)
    mean_low = np.mean(low_deltas)
    psi_band = mean_high / (mean_high + mean_low + 1e-12)
    return psi_band, mean_high, mean_low, clean_dice


def compute_band_psi_msa(net, ds_name, image_np, label, device, n_trials=5, cutoff_ratio=0.25):
    """Compute band PSI for MSA models."""
    # Clean baseline
    if ds_name == "Synapse":
        clean_dice = evaluate_msa_synapse(net, image_np, label, device)
    elif ds_name == "ISIC":
        clean_dice = evaluate_msa_isic(net, image_np, label, device)
    else:  # REFUGE
        clean_dice = evaluate_msa_refuge(net, image_np, label, device)

    high_deltas = []
    low_deltas = []

    for _ in range(n_trials):
        if image_np.ndim == 2:
            img_3ch = np.stack([image_np] * 3, axis=0)
        elif image_np.shape[0] == 1:
            img_3ch = np.stack([image_np[0]] * 3, axis=0)
        else:
            img_3ch = image_np[:3]
        img_t = torch.from_numpy(img_3ch).unsqueeze(0).float().to(device)

        # High-freq
        img_high = phase_band_noise(img_t, band='high', cutoff_ratio=cutoff_ratio, noise_std=1.0)
        img_high_np = img_high[0, 0].cpu().numpy()
        if ds_name == "Synapse":
            dice_high = evaluate_msa_synapse(net, img_high_np, label, device)
        elif ds_name == "ISIC":
            dice_high = evaluate_msa_isic(net, img_high_np, label, device)
        else:
            dice_high = evaluate_msa_refuge(net, img_high_np, label, device)
        high_deltas.append(max(0, clean_dice - dice_high))

        # Low-freq
        img_low = phase_band_noise(img_t, band='low', cutoff_ratio=cutoff_ratio, noise_std=1.0)
        img_low_np = img_low[0, 0].cpu().numpy()
        if ds_name == "Synapse":
            dice_low = evaluate_msa_synapse(net, img_low_np, label, device)
        elif ds_name == "ISIC":
            dice_low = evaluate_msa_isic(net, img_low_np, label, device)
        else:
            dice_low = evaluate_msa_refuge(net, img_low_np, label, device)
        low_deltas.append(max(0, clean_dice - dice_low))

    mean_high = np.mean(high_deltas)
    mean_low = np.mean(low_deltas)
    psi_band = mean_high / (mean_high + mean_low + 1e-12)
    return psi_band, mean_high, mean_low, clean_dice


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("PAID Theory v4: Frequency-Band PSI with Fixed Evaluation")
    print("=" * 70)

    device = torch.device("cuda:0")
    n_imgs = 3
    n_trials = 3
    cutoff_ratios = [0.25]  # primary cutoff only for speed

    all_results = {}

    # ==== SAMed models ====
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

                # Test all cutoff ratios
                best_cutoff_results = {}
                for cutoff in cutoff_ratios:
                    all_psi, all_high, all_low, all_clean = [], [], [], []
                    for img_idx, (img_np, lbl) in enumerate(test_imgs):
                        psi, high_d, low_d, clean_d = compute_band_psi_samed(
                            net, ds_name, img_np, lbl, ncls, device,
                            n_trials=n_trials, cutoff_ratio=cutoff
                        )
                        all_psi.append(psi)
                        all_high.append(high_d)
                        all_low.append(low_d)
                        all_clean.append(clean_d)
                        print(f"    cutoff={cutoff:.3f} img={img_idx + 1}: PSI={psi:.4f} "
                              f"(high_Δ={high_d:.4f}, low_Δ={low_d:.4f}, clean={clean_d:.4f})")

                    best_cutoff_results[f"cutoff_{cutoff}"] = {
                        "mean_psi": float(np.mean(all_psi)),
                        "std_psi": float(np.std(all_psi)),
                        "mean_high_delta": float(np.mean(all_high)),
                        "mean_low_delta": float(np.mean(all_low)),
                        "mean_clean_dice": float(np.mean(all_clean)),
                    }

                primary = best_cutoff_results["cutoff_0.25"]
                all_results[target_name] = {
                    "psi_band": primary["mean_psi"],
                    "high_delta": primary["mean_high_delta"],
                    "low_delta": primary["mean_low_delta"],
                    "clean_dice": primary["mean_clean_dice"],
                    "arch": "SAMed",
                    "dataset": ds_name,
                    "vit": vit_name,
                    "n_images": len(test_imgs),
                    "all_cutoffs": best_cutoff_results,
                }
                print(f"  => PSI={primary['mean_psi']:.4f} high_Δ={primary['mean_high_delta']:.4f} "
                      f"low_Δ={primary['mean_low_delta']:.4f} clean={primary['mean_clean_dice']:.4f}")

                del net
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results[target_name] = {"error": str(e)}

    # ==== MSA models ====
    for ds_name, cfg in MSA_CFG.items():
        for vit_name, (ckpt_rel, ncls) in cfg.items():
            target_name = f"{ds_name}_MSA_{vit_name}"
            ckpt_path = os.path.join(CKPT_BASE, ckpt_rel)
            if not os.path.exists(ckpt_path):
                print(f"SKIP {target_name}: no checkpoint")
                continue

            print(f"\n{'=' * 60}")
            print(f"  {target_name} (classes={ncls})")
            print(f"{'=' * 60}")

            try:
                net = load_medsa_model(vit_name, ckpt_path, IMAGE_SIZE, ncls, device=device)

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

                best_cutoff_results = {}
                for cutoff in cutoff_ratios:
                    all_psi, all_high, all_low, all_clean = [], [], [], []
                    for img_idx, (img_np, lbl) in enumerate(test_imgs):
                        psi, high_d, low_d, clean_d = compute_band_psi_msa(
                            net, ds_name, img_np, lbl, device,
                            n_trials=n_trials, cutoff_ratio=cutoff
                        )
                        all_psi.append(psi)
                        all_high.append(high_d)
                        all_low.append(low_d)
                        all_clean.append(clean_d)
                        print(f"    cutoff={cutoff:.3f} img={img_idx + 1}: PSI={psi:.4f} "
                              f"(high_Δ={high_d:.4f}, low_Δ={low_d:.4f}, clean={clean_d:.4f})")

                    best_cutoff_results[f"cutoff_{cutoff}"] = {
                        "mean_psi": float(np.mean(all_psi)),
                        "std_psi": float(np.std(all_psi)),
                        "mean_high_delta": float(np.mean(all_high)),
                        "mean_low_delta": float(np.mean(all_low)),
                        "mean_clean_dice": float(np.mean(all_clean)),
                    }

                primary = best_cutoff_results["cutoff_0.25"]
                all_results[target_name] = {
                    "psi_band": primary["mean_psi"],
                    "high_delta": primary["mean_high_delta"],
                    "low_delta": primary["mean_low_delta"],
                    "clean_dice": primary["mean_clean_dice"],
                    "arch": "MSA",
                    "dataset": ds_name,
                    "vit": vit_name,
                    "n_images": len(test_imgs),
                    "all_cutoffs": best_cutoff_results,
                }
                print(f"  => PSI={primary['mean_psi']:.4f} high_Δ={primary['mean_high_delta']:.4f} "
                      f"low_Δ={primary['mean_low_delta']:.4f} clean={primary['mean_clean_dice']:.4f}")

                del net
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results[target_name] = {"error": str(e)}

    # Save
    with open(os.path.join(OUTPUT_DIR, "paid_psi_v4_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # =====================================================================
    # VALIDATION against Exp14
    # =====================================================================
    print("\n" + "=" * 70)
    print("VALIDATION: High-ΔDice (PSI) vs Exp14 Phase-Shuffled ΔDice")
    print("=" * 70)

    exp14_path = os.path.join(OUTPUT_DIR, "exp14_full_matrix.json")
    if not os.path.exists(exp14_path):
        print("Exp14 results not found — skipping validation")
        return all_results

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

    # Build pairs: use high_delta as predictor
    pairs = []
    for name, data in all_results.items():
        if "error" in data:
            continue
        if name in ps_deltas and data.get("clean_dice", 0) > 0.01:
            pairs.append({
                "model": name,
                "high_delta": data["high_delta"],
                "psi_band": data["psi_band"],
                "clean_dice": data["clean_dice"],
                "exp14_delta": ps_deltas[name],
                "arch": data.get("arch", ""),
                "dataset": data.get("dataset", ""),
            })

    if len(pairs) < 5:
        print(f"  Only {len(pairs)} valid pairs — insufficient for correlation")
        for p in pairs:
            print(f"  {p['model']}: high_Δ={p['high_delta']:.4f} exp14_Δ={p['exp14_delta']:.4f}")
        return all_results

    # Correlation: high_delta vs exp14_delta
    hd = [p["high_delta"] for p in pairs]
    ed = [p["exp14_delta"] for p in pairs]
    r_hd, p_hd = pearsonr(hd, ed)
    r_s, p_s = spearmanr(hd, ed)

    print(f"\n  Using raw high_ΔDice as PSI predictor:")
    print(f"  Pearson r = {r_hd:.4f} (p={p_hd:.4f})")
    print(f"  Spearman ρ = {r_s:.4f} (p={p_s:.4f})")

    # Also try normalized: high_delta / clean_dice
    hd_norm = [p["high_delta"] / max(p["clean_dice"], 0.01) for p in pairs]
    r_norm, p_norm = pearsonr(hd_norm, ed)
    r_sn, p_sn = spearmanr(hd_norm, ed)
    print(f"\n  Using normalized high_Δ/clean_Dice:")
    print(f"  Pearson r = {r_norm:.4f} (p={p_norm:.4f})")
    print(f"  Spearman ρ = {r_sn:.4f} (p={p_sn:.4f})")

    print(f"\n  {'Model':<30s} {'high_Δ':>10s} {'clean':>10s} {'Exp14_Δ':>10s}")
    print(f"  {'-' * 62}")
    for p in sorted(pairs, key=lambda x: x["high_delta"], reverse=True):
        print(f"  {p['model']:<30s} {p['high_delta']:10.4f} {p['clean_dice']:10.4f} {p['exp14_delta']:10.4f}")

    assessment = "strong" if r_s > 0.7 else "moderate" if r_s > 0.5 else "weak"
    print(f"\n  Assessment: {assessment.upper()} (ρ={r_s:.3f})")

    with open(os.path.join(OUTPUT_DIR, "paid_correlation_v4.json"), "w") as f:
        json.dump({
            "pairs": pairs,
            "pearson_r_raw": r_hd,
            "spearman_r_raw": r_s,
            "pearson_r_norm": r_norm,
            "spearman_r_norm": r_sn,
            "n": len(pairs),
            "assessment": assessment,
        }, f, indent=2)

    return all_results


if __name__ == "__main__":
    main()
