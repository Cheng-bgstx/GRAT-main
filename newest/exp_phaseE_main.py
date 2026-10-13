#!/usr/bin/env python3
"""
Phase E: Main Attack Experiments — Full 18-Model Matrix × 8 Attack Methods.

Produces Tables 2-4 from EXPERIMENT_PLAN.md:
  Table 2: Synapse (8-class)
  Table 3: ISIC (binary)
  Table 4: REFUGE (3-class)

8 attack methods:
  PGD, MI-FGSM, NI-FGSM, DI-FGSM, TI-FGSM, SI-FGSM, SE-CGR, SE-CGR+SRS

Parallel usage (6 GPUs):
  for i in 0 1 2 3 4 5; do
    python exp_phaseE_main.py --gpu $i --subset $i/6 &
  done
"""
import os, sys, json, time, random, gc, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import AttackConfig, run_codd_attack, distill_srs, _get_edge_map, _return_eta
from evaluate import load_sam_encoder, load_samed_model, load_medsa_model

DEVICE = None  # set in main
IMAGE_SIZE = 512
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAM_CKPT = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}

# ── Attack method definitions ──────────────────────────────────────────────
ATTACK_METHODS = [
    "PGD", "MI-FGSM", "NI-FGSM", "DI-FGSM",
    "TI-FGSM", "SI-FGSM", "SE-CGR", "SE-CGR+SRS",
]

# Attack hyperparameters (matching EXPERIMENT_PLAN.md)
EPSILON = 16.0    # L_inf bound (out of 255)
STEP_SIZE = 1.6   # per-step size (EPSILON / NUMSTEPS with small adjust)
NUMSTEPS = 20
MOMENTUM_DECAY = 0.5

# Samples per dataset
N_SAMPLES = {"Synapse": 3, "ISIC": 8, "REFUGE": 8}
NUM_SLICES_PER_VOLUME = 4

# ── Model configurations ───────────────────────────────────────────────────
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

# ── Build model list ────────────────────────────────────────────────────────
def build_model_list():
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


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS (dataset-specific, matching exp14)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_samed_synapse_slice(net, image_np, label_np, case_name):
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


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE LOADING
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE ATTACK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

criterion_l1 = nn.L1Loss()


def _pgd_attack(image_encoder, input_image, epsilon, step_size, steps):
    max_val, min_val = input_image.max(), input_image.min()
    with torch.no_grad():
        targets, _ = image_encoder(input_image)
    perturb_img = input_image.clone().detach()
    perturb_img = perturb_img + torch.randn_like(perturb_img) * 1e-5
    for _ in range(steps):
        perturb_img = perturb_img.clone().detach().requires_grad_(True)
        features, _ = image_encoder(perturb_img)
        loss = criterion_l1(features, targets)
        grad = torch.autograd.grad(loss, perturb_img)[0]
        perturb_img = perturb_img.detach() + step_size * grad.sign()
        eta = torch.clamp(perturb_img - input_image, -epsilon, epsilon)
        perturb_img = torch.clamp(input_image + eta, min_val, max_val).detach()
    return perturb_img.detach()


def _mi_fgsm_attack(image_encoder, input_image, epsilon, step_size, steps, decay=1.0):
    max_val, min_val = input_image.max(), input_image.min()
    with torch.no_grad():
        targets, _ = image_encoder(input_image)
    perturb_img = input_image.clone().detach() + torch.randn_like(input_image) * 1e-5
    momentum = torch.zeros_like(input_image)
    for _ in range(steps):
        perturb_img = perturb_img.clone().detach().requires_grad_(True)
        features, _ = image_encoder(perturb_img)
        loss = criterion_l1(features, targets)
        grad = torch.autograd.grad(loss, perturb_img)[0]
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = decay * momentum + grad
        perturb_img = perturb_img.detach() + step_size * momentum.sign()
        eta = torch.clamp(perturb_img - input_image, -epsilon, epsilon)
        perturb_img = torch.clamp(input_image + eta, min_val, max_val).detach()
    return perturb_img.detach()


def _ni_fgsm_attack(image_encoder, input_image, epsilon, step_size, steps, decay=0.5):
    max_val, min_val = input_image.max(), input_image.min()
    with torch.no_grad():
        targets, _ = image_encoder(input_image)
    perturb_img = input_image.clone().detach() + torch.randn_like(input_image) * 1e-5
    momentum = torch.zeros_like(input_image)
    for _ in range(steps):
        nes_img = perturb_img.detach() + decay * step_size * momentum
        nes_img = nes_img.clone().detach().requires_grad_(True)
        features, _ = image_encoder(nes_img)
        loss = criterion_l1(features, targets)
        grad = torch.autograd.grad(loss, nes_img)[0]
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = decay * momentum + grad
        perturb_img = perturb_img.detach() + step_size * momentum.sign()
        eta = torch.clamp(perturb_img - input_image, -epsilon, epsilon)
        perturb_img = torch.clamp(input_image + eta, min_val, max_val).detach()
    return perturb_img.detach()


def _di_fgsm_attack(image_encoder, input_image, epsilon, step_size, steps, prob=0.5):
    max_val, min_val = input_image.max(), input_image.min()
    with torch.no_grad():
        targets, _ = image_encoder(input_image)
    perturb_img = input_image.clone().detach()
    momentum = torch.zeros_like(input_image)
    H, W = input_image.shape[2], input_image.shape[3]
    for _ in range(steps):
        grad_img = perturb_img.clone().detach().requires_grad_(True)
        if random.random() < prob:
            r = random.randint(int(0.9 * H), H)
            grad_img_resized = F.interpolate(grad_img, size=(r, r), mode='bilinear')
            grad_img_resized = F.interpolate(grad_img_resized, size=(H, W), mode='bilinear')
        else:
            grad_img_resized = grad_img
        features, _ = image_encoder(grad_img_resized)
        loss = criterion_l1(features, targets)
        grad = torch.autograd.grad(loss, grad_img)[0]
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 1.0 * momentum + grad
        perturb_img = perturb_img.detach() + step_size * momentum.sign()
        eta = torch.clamp(perturb_img - input_image, -epsilon, epsilon)
        perturb_img = torch.clamp(input_image + eta, min_val, max_val).detach()
    return perturb_img.detach()


def _ti_fgsm_attack(image_encoder, input_image, epsilon, step_size, steps, kernel_size=5):
    max_val, min_val = input_image.max(), input_image.min()
    with torch.no_grad():
        targets, _ = image_encoder(input_image)
    perturb_img = input_image.clone().detach() + torch.randn_like(input_image) * 1e-5
    momentum = torch.zeros_like(input_image)
    ks = kernel_size
    k = torch.ones(1, 1, ks, ks, device=input_image.device) / (ks * ks)
    if input_image.shape[1] == 3:
        k = k.repeat(3, 1, 1, 1)
    for _ in range(steps):
        perturb_img = perturb_img.clone().detach().requires_grad_(True)
        features, _ = image_encoder(perturb_img)
        loss = criterion_l1(features, targets)
        grad = torch.autograd.grad(loss, perturb_img)[0]
        grad = F.conv2d(grad, k, padding=ks//2, groups=input_image.shape[1])
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 1.0 * momentum + grad
        perturb_img = perturb_img.detach() + step_size * momentum.sign()
        eta = torch.clamp(perturb_img - input_image, -epsilon, epsilon)
        perturb_img = torch.clamp(input_image + eta, min_val, max_val).detach()
    return perturb_img.detach()


def _si_fgsm_attack(image_encoder, input_image, epsilon, step_size, steps, num_scales=3):
    max_val, min_val = input_image.max(), input_image.min()
    with torch.no_grad():
        targets, _ = image_encoder(input_image)
    perturb_img = input_image.clone().detach()
    momentum = torch.zeros_like(input_image)
    H, W = input_image.shape[2], input_image.shape[3]
    for _ in range(steps):
        total_grad = torch.zeros_like(input_image)
        for s in range(num_scales):
            scale = 1.0 / (2 ** s)
            scaled = F.interpolate(perturb_img.detach(), scale_factor=scale, mode='bilinear')
            scaled = F.interpolate(scaled, size=(H, W), mode='bilinear')
            scaled = scaled.clone().detach().requires_grad_(True)
            features, _ = image_encoder(scaled)
            loss = criterion_l1(features, targets)
            grad = torch.autograd.grad(loss, scaled)[0]
            total_grad += grad
        grad = total_grad / num_scales
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 1.0 * momentum + grad
        perturb_img = perturb_img.detach() + step_size * momentum.sign()
        eta = torch.clamp(perturb_img - input_image, -epsilon, epsilon)
        perturb_img = torch.clamp(input_image + eta, min_val, max_val).detach()
    return perturb_img.detach()


# ══════════════════════════════════════════════════════════════════════════════
# SRS DISTILLATION (on-the-fly)
# ══════════════════════════════════════════════════════════════════════════════

def distill_srs_for_config(source_encoder, dataset, vit, device, n_proxy=64, rank=8):
    """Distill SRS bundle for a given (dataset, vit) config."""
    from theory import SharedSubspace

    # Cache path
    cache_dir = os.path.join(OUTPUT_DIR, "srs_bundles")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"srs_{dataset}_{vit}.pth")

    if os.path.exists(cache_path):
        bundle = torch.load(cache_path, map_location=device)
        if isinstance(bundle, dict) and 'mean_direction' in bundle:
            subspace = SharedSubspace(
                mean_direction=bundle['mean_direction'],
                basis=bundle['basis'],
                shape=bundle.get('shape', [1, 3, IMAGE_SIZE, IMAGE_SIZE]),
                singular_values=bundle['singular_values'],
                explained_variance=bundle['explained_variance'],
            )
            print(f"  [SRS] Loaded cached bundle: {cache_path}")
            return subspace

    print(f"  [SRS] Distilling for {dataset}/{vit} ({n_proxy} samples, rank={rank})...")

    config = AttackConfig(
        att_step_size=2.0, att_bound=10.0, att_numstep=3,
        direction_mode="radial", probe_amplification=2.0,
        edge_guidance=True, seed=42,
    )

    gradient_list = []

    if dataset == "Synapse":
        import h5py
        tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
        h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:10]
        for h5f in h5s:
            if len(gradient_list) >= n_proxy: break
            with h5py.File(os.path.join(tdir, h5f), 'r') as f:
                img, lbl = f['image'][:], f['label'][:]
            organ_slices = [s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100]
            for si in organ_slices[:4]:
                if len(gradient_list) >= n_proxy: break
                slc = img[si].astype(np.float32)
                slc_3ch = np.stack([slc]*3, axis=0)
                img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(device)
                _collect_srs_gradients(source_encoder, img_t, config, gradient_list, device)
                if len(gradient_list) % 16 == 0:
                    print(f"    [SRS] {len(gradient_list)}/{n_proxy}")
    else:
        # ISIC or REFUGE
        if dataset == "ISIC":
            samples = get_isic_samples(n_proxy)
        else:
            samples = get_refuge_samples(n_proxy)

        for img_t, _, _ in samples:
            if len(gradient_list) >= n_proxy: break
            img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1).unsqueeze(0).to(device)
            _collect_srs_gradients(source_encoder, img_gray, config, gradient_list, device)
            if len(gradient_list) % 16 == 0:
                print(f"    [SRS] {len(gradient_list)}/{n_proxy}")

    print(f"  [SRS] Fitting subspace from {len(gradient_list)} gradients")
    from theory import fit_shared_subspace
    subspace = fit_shared_subspace(gradient_list, rank=rank)

    # Cache
    torch.save({
        'mean_direction': subspace.mean_direction,
        'basis': subspace.basis,
        'shape': subspace.shape,
        'singular_values': subspace.singular_values,
        'explained_variance': subspace.explained_variance,
    }, cache_path)
    print(f"  [SRS] Saved: {cache_path} (var={subspace.explained_variance:.4f})")
    return subspace


def _collect_srs_gradients(source_sam, img_tensor, config, gradient_list, device):
    encoder = source_sam.image_encoder
    max_val, min_val = img_tensor.max(), img_tensor.min()
    step_size = float((max_val - min_val) * config.att_step_size / 255.0)
    epsilon = float((max_val - min_val) * config.att_bound / 255.0)

    with torch.no_grad():
        targets, targets_layer = encoder(img_tensor)

    perturb_img = img_tensor.clone().detach() + torch.randn_like(img_tensor) * 1e-5
    perturb_img = perturb_img.detach().requires_grad_(True)
    edge_map = _get_edge_map(img_tensor).detach()

    local_grads = []
    for _ in range(3):
        encoder.zero_grad()
        features, features_layer = encoder(perturb_img)

        # CODD loss
        loss_org = criterion_l1(features * 0.9999, targets)
        pre_layer = int(len(features_layer) * 0.9)
        num_layers = min(6, pre_layer)
        n_pick = max(1, num_layers - 2)
        indices = random.sample(range(num_layers), n_pick)
        dir_energy = 0.0
        from attack_codd import synthesize_codd_probe
        for idx in indices:
            F_clean_i = targets_layer[idx + pre_layer - num_layers]
            F_adv_i = features_layer[idx + pre_layer - num_layers]
            probe = synthesize_codd_probe(
                F_clean=F_clean_i.unsqueeze(0), F_adv=F_adv_i.unsqueeze(0),
                direction="radial", ads_rho=config.ads_rho,
                ads_gamma=config.ads_gamma, use_random_noise=False, detach_adv=False,
            )
            dir_energy += (probe ** 2).mean()
        dir_energy = dir_energy / n_pick
        loss = loss_org - dir_energy
        grad = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]
        if config.edge_guidance:
            grad = grad * (1.0 + config.edge_gamma * edge_map)
        grad_norm = grad.detach() / (grad.detach().abs().sum() + 1e-12)
        local_grads.append(grad_norm.cpu())
        perturb_img = (perturb_img.detach() + step_size * grad_norm.sign()).clamp(min_val, max_val)
        eta = torch.clamp(perturb_img - img_tensor, -epsilon, epsilon)
        perturb_img = (img_tensor + eta).detach().requires_grad_(True)

    avg_grad = torch.stack(local_grads, dim=0).mean(dim=0)
    gradient_list.append(avg_grad)


# ══════════════════════════════════════════════════════════════════════════════
# CORE: Run all attacks on one model
# ══════════════════════════════════════════════════════════════════════════════

def run_phaseE_model(tag, dataset, model_type, vit, ckpt, ncls_or_mmo, device):
    """Run all 8 attack methods on a single model."""
    print(f"\n{'='*70}\n  {tag}\n{'='*70}")

    # Load source SAM encoder
    source_sam = load_sam_encoder(vit, SAM_CKPT[vit], IMAGE_SIZE, 8, device)

    # Load target model
    if model_type == "SAMed":
        target = load_samed_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, ncls_or_mmo, device=device)
    else:
        target = load_medsa_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, ncls_or_mmo, device)

    # Distill SRS (cached)
    srs_bundle = distill_srs_for_config(source_sam, dataset, vit, device)

    # Gather samples
    n_cases = N_SAMPLES[dataset]
    case_results = []  # list of per-case dicts

    if dataset == "Synapse":
        _run_phaseE_synapse(tag, model_type, source_sam, target, srs_bundle, n_cases, device, case_results)
    elif dataset == "ISIC":
        _run_phaseE_isic(tag, model_type, source_sam, target, srs_bundle, n_cases, device, case_results)
    elif dataset == "REFUGE":
        _run_phaseE_refuge(tag, model_type, source_sam, target, srs_bundle, n_cases, device, case_results)

    # Aggregate
    summary = {}
    for method in ATTACK_METHODS:
        deltas = [r[f"{method}_delta"] for r in case_results if f"{method}_delta" in r]
        if deltas:
            summary[method] = {
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas)),
                "mean_dice": float(np.mean([r[f"{method}_dice"] for r in case_results if f"{method}_dice" in r])),
                "n": len(deltas),
            }
    summary["clean_dice"] = float(np.mean([r["clean_dice"] for r in case_results]))

    print(f"  [{tag}] Clean={summary['clean_dice']:.4f} | " + " | ".join(
        f"{m}={summary[m]['mean_delta']:.4f}" for m in ATTACK_METHODS if m in summary))

    del source_sam, target; gc.collect(); torch.cuda.empty_cache()
    return summary, case_results


# ── Synapse runner ──────────────────────────────────────────────────────────

def _run_phaseE_synapse(tag, model_type, source_sam, target, srs_bundle, n_volumes, device, case_results):
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

            # Compute attack bounds from actual image range
            max_val_pix, min_val_pix = img_t.max().item(), img_t.min().item()
            eps_pix = float((max_val_pix - min_val_pix) * EPSILON / 255.0)
            step_pix = float((max_val_pix - min_val_pix) * STEP_SIZE / 255.0)

            # Clean evaluation
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            if model_type == "SAMed":
                clean_d = evaluate_samed_synapse_slice(target, slc, label_slice, f"{h5f}_s{si}")
            else:
                clean_d = evaluate_msa_synapse_slice(target, slc, label_slice, device)

            result = {"case": h5f, "slice": int(si), "clean_dice": float(clean_d)}
            print(f"    [{tag}] {h5f}/s{si} clean={clean_d:.4f}", end="")

            # Run each attack method
            for method in ATTACK_METHODS:
                random.seed(42); np.random.seed(42); torch.manual_seed(42)
                torch.cuda.empty_cache()

                try:
                    if method in ("PGD", "MI-FGSM", "NI-FGSM", "DI-FGSM", "TI-FGSM", "SI-FGSM"):
                        adv_tensor = _run_baseline_attack(method, source_sam.image_encoder, img_t, eps_pix, step_pix)
                    elif method == "SE-CGR":
                        adv_tensor = _run_secgr_attack(source_sam.image_encoder, img_t, eps_pix, step_pix)
                    elif method == "SE-CGR+SRS":
                        adv_tensor = _run_secgr_srs_attack(source_sam.image_encoder, img_t, eps_pix, step_pix, srs_bundle)
                    else:
                        continue

                    adv_np = adv_tensor[0].detach().cpu().numpy()
                    adv_gray = adv_np[0] if adv_np.shape[0] == 3 else adv_np

                    if model_type == "SAMed":
                        adv_d = evaluate_samed_synapse_slice(target, adv_gray, label_slice, f"{h5f}_s{si}")
                    else:
                        adv_d = evaluate_msa_synapse_slice(target, adv_gray, label_slice, device)

                    result[f"{method}_dice"] = float(adv_d)
                    result[f"{method}_delta"] = float(clean_d - adv_d)
                    print(f" {method}={adv_d:.4f}", end="")
                except Exception as e:
                    print(f" {method}=ERR({e})", end="")
                    result[f"{method}_dice"] = None
                    result[f"{method}_delta"] = None

            print()
            case_results.append(result)

        print(f"  [{tag}] Volume {vol_idx+1}/{n_volumes} done")


# ── ISIC runner ─────────────────────────────────────────────────────────────

def _run_phaseE_isic(tag, model_type, source_sam, target, srs_bundle, n_cases, device, case_results):
    samples = get_isic_samples(n_cases)

    for case_idx, (img_t, label, img_raw) in enumerate(samples):
        max_val, min_val = img_t.max(), img_t.min()
        eps_pix = float((max_val - min_val) * EPSILON / 255.0)
        step_pix = float((max_val - min_val) * STEP_SIZE / 255.0)

        # Clean evaluation
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        if model_type == "SAMed":
            clean_d = evaluate_samed_isic(target, img_t, torch.from_numpy(label).float(), device)
        else:
            img_uint8_np = img_raw.mean(axis=2).astype(np.float32)
            label_t = torch.from_numpy(label).float()
            clean_d = evaluate_msa_isic(target, img_uint8_np, label_t, device)

        # For attacks: use 3-channel grayscale version (matching exp14)
        img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1).unsqueeze(0).to(device)

        result = {"case": int(case_idx), "clean_dice": float(clean_d)}
        print(f"    [{tag}] case {case_idx} clean={clean_d:.4f}", end="")

        for method in ATTACK_METHODS:
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            torch.cuda.empty_cache()

            try:
                if method in ("PGD", "MI-FGSM", "NI-FGSM", "DI-FGSM", "TI-FGSM", "SI-FGSM"):
                    adv_tensor = _run_baseline_attack(method, source_sam.image_encoder, img_gray, eps_pix, step_pix)
                elif method == "SE-CGR":
                    adv_tensor = _run_secgr_attack(source_sam.image_encoder, img_gray, eps_pix, step_pix)
                elif method == "SE-CGR+SRS":
                    adv_tensor = _run_secgr_srs_attack(source_sam.image_encoder, img_gray, eps_pix, step_pix, srs_bundle)
                else:
                    continue

                adv_np = adv_tensor[0].detach().cpu().numpy()

                if model_type == "SAMed":
                    adv_gray = torch.from_numpy(adv_np.copy()).float()
                    adv_d = evaluate_samed_isic(target, adv_gray, torch.from_numpy(label).float(), device)
                else:
                    adv_uint8 = normalized_gray_to_uint8(adv_np[0].copy())
                    adv_d = evaluate_msa_isic(target, adv_uint8, torch.from_numpy(label).float(), device)

                result[f"{method}_dice"] = float(adv_d)
                result[f"{method}_delta"] = float(clean_d - adv_d)
                print(f" {method}={adv_d:.4f}", end="")
            except Exception as e:
                print(f" {method}=ERR({e})", end="")
                result[f"{method}_dice"] = None
                result[f"{method}_delta"] = None

        print()
        case_results.append(result)


# ── REFUGE runner ───────────────────────────────────────────────────────────

def _run_phaseE_refuge(tag, model_type, source_sam, target, srs_bundle, n_cases, device, case_results):
    samples = get_refuge_samples(n_cases)

    for case_idx, (img_t, label, img_raw) in enumerate(samples):
        max_val, min_val = img_t.max(), img_t.min()
        eps_pix = float((max_val - min_val) * EPSILON / 255.0)
        step_pix = float((max_val - min_val) * STEP_SIZE / 255.0)

        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        if model_type == "SAMed":
            clean_d = evaluate_samed_refuge(target, img_t, torch.from_numpy(label).float(), device)
        else:
            img_uint8_np = img_raw.mean(axis=2).astype(np.float32)
            label_t = torch.from_numpy(label).float()
            clean_d = evaluate_msa_refuge(target, img_uint8_np, label_t, device)

        img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1).unsqueeze(0).to(device)

        result = {"case": int(case_idx), "clean_dice": float(clean_d)}
        print(f"    [{tag}] case {case_idx} clean={clean_d:.4f}", end="")

        for method in ATTACK_METHODS:
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            torch.cuda.empty_cache()

            try:
                if method in ("PGD", "MI-FGSM", "NI-FGSM", "DI-FGSM", "TI-FGSM", "SI-FGSM"):
                    adv_tensor = _run_baseline_attack(method, source_sam.image_encoder, img_gray, eps_pix, step_pix)
                elif method == "SE-CGR":
                    adv_tensor = _run_secgr_attack(source_sam.image_encoder, img_gray, eps_pix, step_pix)
                elif method == "SE-CGR+SRS":
                    adv_tensor = _run_secgr_srs_attack(source_sam.image_encoder, img_gray, eps_pix, step_pix, srs_bundle)
                else:
                    continue

                adv_np = adv_tensor[0].detach().cpu().numpy()

                if model_type == "SAMed":
                    adv_gray = torch.from_numpy(adv_np.copy()).float()
                    adv_d = evaluate_samed_refuge(target, adv_gray, torch.from_numpy(label).float(), device)
                else:
                    adv_uint8 = normalized_gray_to_uint8(adv_np[0].copy())
                    adv_d = evaluate_msa_refuge(target, adv_uint8, torch.from_numpy(label).float(), device)

                result[f"{method}_dice"] = float(adv_d)
                result[f"{method}_delta"] = float(clean_d - adv_d)
                print(f" {method}={adv_d:.4f}", end="")
            except Exception as e:
                print(f" {method}=ERR({e})", end="")
                result[f"{method}_dice"] = None
                result[f"{method}_delta"] = None

        print()
        case_results.append(result)


# ══════════════════════════════════════════════════════════════════════════════
# ATTACK METHOD WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_baseline_attack(method, image_encoder, img_tensor, eps_pix, step_pix):
    attacks = {
        "PGD": _pgd_attack,
        "MI-FGSM": _mi_fgsm_attack,
        "NI-FGSM": lambda ie, ii, ep, sp, st: _ni_fgsm_attack(ie, ii, ep, sp, st, decay=0.5),
        "DI-FGSM": _di_fgsm_attack,
        "TI-FGSM": _ti_fgsm_attack,
        "SI-FGSM": _si_fgsm_attack,
    }
    return attacks[method](image_encoder, img_tensor, eps_pix, step_pix, NUMSTEPS)


def _run_secgr_attack(image_encoder, img_tensor, eps_pix, step_pix):
    config = AttackConfig(
        att_step_size=STEP_SIZE, att_bound=EPSILON, att_numstep=NUMSTEPS,
        direction_mode="radial", probe_amplification=2.0,
        edge_guidance=True, seed=42,
    )
    adv_np, _ = run_codd_attack(image_encoder, img_tensor, config, return_trace=True)
    return torch.from_numpy(adv_np.copy()).float().to(img_tensor.device).unsqueeze(0)


def _run_secgr_srs_attack(image_encoder, img_tensor, eps_pix, step_pix, srs_bundle):
    config = AttackConfig(
        att_step_size=STEP_SIZE, att_bound=EPSILON, att_numstep=NUMSTEPS,
        direction_mode="radial", probe_amplification=2.0,
        edge_guidance=True, seed=42,
        use_srs=True, srs_proj_weight=0.15, srs_mean_weight=0.1,
    )
    adv_np, _ = run_codd_attack(image_encoder, img_tensor, config, srs_bundle=srs_bundle, return_trace=True)
    return torch.from_numpy(adv_np.copy()).float().to(img_tensor.device).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--subset", type=str, default="0/1")
    parser.add_argument("--models", type=str, default="", help="comma-separated model tags")
    args = parser.parse_args()

    DEVICE = torch.device(f"cuda:{args.gpu}")
    print(f"Phase E: Main Attack Experiments — GPU {args.gpu}")

    all_models = build_model_list()
    print(f"Total available models: {len(all_models)}")
    for m in all_models:
        print(f"  {m[0]}")

    if args.models:
        requested = set(args.models.split(","))
        models = [m for m in all_models if m[0] in requested]
    else:
        subset_idx, total = map(int, args.subset.split("/"))
        models = [m for i, m in enumerate(all_models) if i % total == subset_idx]

    print(f"Running {len(models)} models: {[m[0] for m in models]}")

    worker_id = f"gpu{args.gpu}_sub{args.subset.replace('/', '_')}"
    output_path = os.path.join(OUTPUT_DIR, f"phaseE_worker_{worker_id}.json")
    all_summaries = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            all_summaries = json.load(f).get("summaries", {})

    for tag, dataset, model_type, vit, ckpt, ncls_or_mmo in models:
        if tag in all_summaries:
            print(f"SKIP {tag}: already completed")
            continue

        try:
            summary, case_results = run_phaseE_model(tag, dataset, model_type, vit, ckpt, ncls_or_mmo, DEVICE)
            all_summaries[tag] = summary

            # Save incrementally
            with open(output_path, "w") as f:
                json.dump({
                    "config": {"epsilon": EPSILON, "step_size": STEP_SIZE, "numsteps": NUMSTEPS},
                    "summaries": all_summaries,
                }, f, indent=2)
        except Exception as e:
            print(f"  [FATAL] {tag}: {e}")
            import traceback; traceback.print_exc()

    print(f"\nPhase E worker {worker_id} DONE. Results: {output_path}")


if __name__ == "__main__":
    main()
