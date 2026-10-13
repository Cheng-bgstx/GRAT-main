#!/usr/bin/env python3
"""
PAID Theory v7: Complete 18-Model Frequency-Band PSI.

Includes both SAMed (7 working) and MSA (9) models.
Uses Exp14-matched data loading and evaluation.
"""
import os, sys, json, gc, random
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from scipy.stats import pearsonr, spearmanr

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

# SAMed: (ckpt_rel, num_classes). Skip vit_l for ISIC/REFUGE (broken)
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

# MSA: (ckpt_rel, multimask_output). Uses exp14 paths (s24 seed)
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
# SAMed evaluation
# =========================================================================

def evaluate_samed(net, img_data, lbl, ds_name, device):
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
# MSA evaluation (matching Exp14)
# =========================================================================

def evaluate_msa_synapse(net, img_data, label_slice, device):
    if torch.is_tensor(img_data):
        img_t = img_data.float().to(device)
    else:
        img_t = torch.from_numpy(img_data).float().to(device)
    if img_t.dim() == 3: img_t = img_t.unsqueeze(0)
    pt_list, lbl_list = [], []
    for c in range(1, 9):
        class_mask = (label_slice == c).astype(np.uint8)
        indices = np.argwhere(class_mask > 0)
        if len(indices) > 0:
            r, cc = indices[random.randint(0, len(indices) - 1)]
            pt_list.append(np.array([cc, r])); lbl_list.append(1)
        else:
            pt_list.append(np.array([0, 0])); lbl_list.append(0)
    coords = torch.tensor(np.stack(pt_list, axis=0), dtype=torch.float, device=device).unsqueeze(0)
    labels_pt = torch.tensor(lbl_list, dtype=torch.int, device=device).unsqueeze(0)
    net.eval()
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
        inter = (pc * tc).sum(); denom = pc.sum() + tc.sum()
        dices.append(2.0 * inter / (denom + 1e-8) if denom > 0 else 1.0)
    return float(np.mean(dices))


def evaluate_msa_isic(net, img_data, gt_mask, device):
    from PIL import Image as PILImage
    from torchvision.transforms import Resize, Compose, ToTensor, Lambda
    # Convert normalized tensor back to uint8 for MSA preprocessing
    if torch.is_tensor(img_data):
        # ImageNet denormalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_data.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_data.device)
        img_unnorm = img_data * std[:, None, None] + mean[:, None, None]
        img_uint8 = (img_unnorm * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
    else:
        img_uint8 = np.clip(img_data, 0, 255).astype(np.uint8)
        if img_uint8.ndim == 3 and img_uint8.shape[0] == 3:
            img_uint8 = img_uint8.transpose(1, 2, 0)
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
        if d > best_d: best_d = d
    return float(best_d)


def evaluate_msa_refuge(net, img_data, gt_label, device):
    from PIL import Image as PILImage
    from torchvision.transforms import Resize, Compose, ToTensor, Lambda
    if torch.is_tensor(img_data):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_data.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_data.device)
        img_unnorm = img_data * std[:, None, None] + mean[:, None, None]
        img_uint8 = (img_unnorm * 255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
    else:
        img_uint8 = np.clip(img_data, 0, 255).astype(np.uint8)
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
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("PAID Theory v7: Complete 18-Model Frequency-Band PSI")
    print("=" * 70)

    device = torch.device("cuda:0")
    n_imgs = 3
    n_trials = 3
    cutoff = 0.25

    all_results = {}

    # ---- SAMed models ----
    for ds_name, cfg in SAMED_CFG.items():
        for vit_name, (ckpt_rel, ncls) in cfg.items():
            target_name = f"{ds_name}_SAMed_{vit_name}"
            ckpt_path = os.path.join(CKPT_BASE, ckpt_rel)
            if not os.path.exists(ckpt_path):
                print(f"SKIP {target_name}: no checkpoint"); continue

            print(f"\n{'=' * 60}\n  {target_name}\n{'=' * 60}")
            try:
                sam_ckpt = SAM_CKPTS[vit_name]
                net = load_samed_model(vit_name, sam_ckpt, ckpt_path, IMAGE_SIZE, ncls, device=device)
                if ds_name == "Synapse": test_imgs = get_synapse_test_images(n_imgs)
                elif ds_name == "ISIC": test_imgs = get_isic_test_images(n_imgs)
                else: test_imgs = get_refuge_test_images(n_imgs)
                if not test_imgs: continue
                print(f"  Loaded {len(test_imgs)} images")

                all_high_d, all_low_d, all_clean = [], [], []
                for img_idx, (img_data, lbl) in enumerate(test_imgs):
                    if torch.is_tensor(img_data):
                        img_t = img_data.unsqueeze(0).float().to(device)
                    else:
                        img_t = torch.from_numpy(img_data).unsqueeze(0).float().to(device)

                    clean_d = evaluate_samed(net, img_data, lbl, ds_name, device)
                    if clean_d < 0.01: continue

                    high_ds, low_ds = [], []
                    for _ in range(n_trials):
                        img_high = phase_band_noise(img_t, band='high', cutoff_ratio=cutoff)
                        high_ds.append(max(0, clean_d - evaluate_samed(net, img_high[0], lbl, ds_name, device)))
                        img_low = phase_band_noise(img_t, band='low', cutoff_ratio=cutoff)
                        low_ds.append(max(0, clean_d - evaluate_samed(net, img_low[0], lbl, ds_name, device)))

                    mh, ml = np.mean(high_ds), np.mean(low_ds)
                    all_high_d.append(mh); all_low_d.append(ml); all_clean.append(clean_d)
                    print(f"    img={img_idx + 1}: hΔ={mh:.4f} lΔ={ml:.4f} clean={clean_d:.4f}")

                if all_clean:
                    all_results[target_name] = {
                        "high_delta": float(np.mean(all_high_d)), "low_delta": float(np.mean(all_low_d)),
                        "clean_dice": float(np.mean(all_clean)), "arch": "SAMed", "dataset": ds_name, "vit": vit_name,
                        "psi_band": float(np.mean(all_high_d) / (np.mean(all_high_d) + np.mean(all_low_d) + 1e-12)),
                    }
                    print(f"  => high_Δ={all_results[target_name]['high_delta']:.4f} clean={all_results[target_name]['clean_dice']:.4f}")
                del net; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
                all_results[target_name] = {"error": str(e)}

    # ---- MSA models ----
    for ds_name, cfg in MSA_CFG.items():
        for vit_name, (ckpt_rel, mmo) in cfg.items():
            target_name = f"{ds_name}_MSA_{vit_name}"
            ckpt_path = os.path.join(CKPT_BASE, ckpt_rel)
            if not os.path.exists(ckpt_path):
                print(f"SKIP {target_name}: no checkpoint"); continue

            print(f"\n{'=' * 60}\n  {target_name}\n{'=' * 60}")
            try:
                sam_ckpt = SAM_CKPTS[vit_name]
                net = load_medsa_model(vit_name, sam_ckpt, ckpt_path, IMAGE_SIZE, multimask_output=mmo, device=device)
                if ds_name == "Synapse": test_imgs = get_synapse_test_images(n_imgs)
                elif ds_name == "ISIC": test_imgs = get_isic_test_images(n_imgs)
                else: test_imgs = get_refuge_test_images(n_imgs)
                if not test_imgs: continue
                print(f"  Loaded {len(test_imgs)} images")

                all_high_d, all_low_d, all_clean = [], [], []
                for img_idx, (img_data, lbl) in enumerate(test_imgs):
                    if torch.is_tensor(img_data):
                        img_t = img_data.unsqueeze(0).float().to(device)
                    else:
                        img_t = torch.from_numpy(img_data).unsqueeze(0).float().to(device)

                    # Clean
                    if ds_name == "Synapse": clean_d = evaluate_msa_synapse(net, img_data, lbl, device)
                    elif ds_name == "ISIC": clean_d = evaluate_msa_isic(net, img_data, lbl, device)
                    else: clean_d = evaluate_msa_refuge(net, img_data, lbl, device)
                    if clean_d < 0.01: continue

                    high_ds, low_ds = [], []
                    for _ in range(n_trials):
                        img_high = phase_band_noise(img_t, band='high', cutoff_ratio=cutoff)
                        if ds_name == "Synapse": dh = evaluate_msa_synapse(net, img_high[0], lbl, device)
                        elif ds_name == "ISIC": dh = evaluate_msa_isic(net, img_high[0], lbl, device)
                        else: dh = evaluate_msa_refuge(net, img_high[0], lbl, device)
                        high_ds.append(max(0, clean_d - dh))

                        img_low = phase_band_noise(img_t, band='low', cutoff_ratio=cutoff)
                        if ds_name == "Synapse": dl = evaluate_msa_synapse(net, img_low[0], lbl, device)
                        elif ds_name == "ISIC": dl = evaluate_msa_isic(net, img_low[0], lbl, device)
                        else: dl = evaluate_msa_refuge(net, img_low[0], lbl, device)
                        low_ds.append(max(0, clean_d - dl))

                    mh, ml = np.mean(high_ds), np.mean(low_ds)
                    all_high_d.append(mh); all_low_d.append(ml); all_clean.append(clean_d)
                    print(f"    img={img_idx + 1}: hΔ={mh:.4f} lΔ={ml:.4f} clean={clean_d:.4f}")

                if all_clean:
                    all_results[target_name] = {
                        "high_delta": float(np.mean(all_high_d)), "low_delta": float(np.mean(all_low_d)),
                        "clean_dice": float(np.mean(all_clean)), "arch": "MSA", "dataset": ds_name, "vit": vit_name,
                        "psi_band": float(np.mean(all_high_d) / (np.mean(all_high_d) + np.mean(all_low_d) + 1e-12)),
                    }
                    print(f"  => high_Δ={all_results[target_name]['high_delta']:.4f} clean={all_results[target_name]['clean_dice']:.4f}")
                del net; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
                all_results[target_name] = {"error": str(e)}

    # Save
    with open(os.path.join(OUTPUT_DIR, "paid_psi_v7_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # =====================================================================
    # VALIDATION
    # =====================================================================
    print("\n" + "=" * 70)
    print("VALIDATION: high_ΔDice vs Exp14 Phase-Shuffled ΔDice")
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
        if name in ps_deltas and data["clean_dice"] > 0.01:
            pairs.append({"model": name, "high_delta": data["high_delta"],
                          "clean_dice": data["clean_dice"], "exp14_delta": ps_deltas[name]})

    if len(pairs) >= 5:
        hd = [p["high_delta"] for p in pairs]
        ed = [p["exp14_delta"] for p in pairs]
        r_pe, p_pe = pearsonr(hd, ed)
        r_sp, p_sp = spearmanr(hd, ed)
        print(f"\n  n={len(pairs)}  Pearson r={r_pe:.4f} (p={p_pe:.6f})  Spearman ρ={r_sp:.4f} (p={p_sp:.6f})")

        # Normalized
        hd_n = [p["high_delta"] / max(p["clean_dice"], 0.01) for p in pairs]
        r_n, p_n = pearsonr(hd_n, ed)
        r_sn, p_sn = spearmanr(hd_n, ed)
        print(f"  Normalized: Pearson r={r_n:.4f} (p={p_n:.6f})  Spearman ρ={r_sn:.4f} (p={p_sn:.6f})")

        print(f"\n  {'Model':<28s} {'high_Δ':>9s} {'clean':>9s} {'Exp14_Δ':>9s} {'Arch':>6s}")
        print(f"  {'-' * 66}")
        for p in sorted(pairs, key=lambda x: x["high_delta"], reverse=True):
            arch = "SAMed" if "SAMed" in p["model"] else "MSA"
            print(f"  {p['model']:<28s} {p['high_delta']:9.4f} {p['clean_dice']:9.4f} {p['exp14_delta']:9.4f} {arch:>6s}")

        assessment = "strong" if r_sp > 0.7 else "moderate" if r_sp > 0.5 else "weak"
        print(f"\n  Assessment: {assessment.upper()} (ρ={r_sp:.3f})")

        with open(os.path.join(OUTPUT_DIR, "paid_correlation_v7.json"), "w") as f:
            json.dump({"pairs": pairs, "pearson_r": r_pe, "spearman_r": r_sp,
                       "pearson_r_norm": r_n, "spearman_r_norm": r_sn,
                       "n": len(pairs), "assessment": assessment}, f, indent=2)
    else:
        for p in sorted(pairs, key=lambda x: x["high_delta"], reverse=True):
            print(f"  {p['model']:<28s} {p['high_delta']:9.4f} {p['clean_dice']:9.4f} {p['exp14_delta']:9.4f}")

    return all_results


if __name__ == "__main__":
    main()
