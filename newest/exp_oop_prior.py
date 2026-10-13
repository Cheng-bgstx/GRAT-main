#!/usr/bin/env python3
"""
Optimized Offline Prior (OOP) Experiment.

Tests whether averaging D_r across proxy training images provides
a useful prior for improving attack effectiveness.

PAID justification:
  Φ (radial direction f/|f|) is cross-model invariant → D_r should
  share structure across images of the same modality processed by
  the same model. Averaging isolates shared model-specific structure.

Experiment:
  1. Measure D_r cosine similarity across training images (diagnostic)
  2. Compute offline prior from N proxy images
  3. Test prior_weight ∈ {0, 0.25, 0.5, 0.75, 1.0}
  4. Compare per-channel vs global prior computation
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_samed_model
from attack_codd import _get_edge_map
from attack_radial_optimized import (
    attack_radial_optimized, compute_offline_radial_prior,
    make_radial_probe,
)

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
CKPT_REL = "SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0")
IM_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ── Data Loading ─────────────────────────────────────────────────────────────

def load_images(img_dir, mask_dir, n=None):
    """Load ISIC images as tensors."""
    jpgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    if n:
        jpgs = jpgs[:n]
    imgs = []
    for jpg in jpgs:
        img = Image.open(os.path.join(img_dir, jpg)).convert('RGB').resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)
        base = jpg.replace('.jpg', '')
        mask_path = os.path.join(mask_dir, base + '_Segmentation.png')
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        gt = (np.array(mask) > 128).astype(np.float32)
        if gt.sum() < 100:
            continue
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        img_t = (img_t / 255.0 - IM_MEAN_CPU) / IM_STD_CPU
        imgs.append(img_t)
    return imgs


def evaluate_isic(net, img_t, gt):
    with torch.no_grad():
        out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        probs = torch.softmax(out['masks'].float(), dim=1)
        lesion_prob = probs[0, 1, :, :].cpu().numpy()
    best = 0.0
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        pred_bin = (lesion_prob > thresh).astype(np.float32)
        inter = (pred_bin * gt).sum()
        dice = float(2.0 * inter / (pred_bin.sum() + gt.sum() + 1e-8))
        if dice > best:
            best = dice
    return best


# ── Load Model ───────────────────────────────────────────────────────────────

print("Loading model...")
net = load_samed_model("vit_b", SAM_CKPT, os.path.join(CKPT_BASE, CKPT_REL),
                       IMAGE_SIZE, 1, device=device)
net.eval()
image_encoder = net.sam.image_encoder
image_encoder.eval()

# ── Step 1: Diagnostic — D_r similarity across training images ───────────────

print("\n" + "=" * 70)
print("STEP 1: D_r Cosine Similarity Across Training Images")
print("=" * 70)

train_dir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Training_Data"
train_mask_dir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Training_GroundTruth"
train_imgs = load_images(train_dir, train_mask_dir, n=32)
print(f"Loaded {len(train_imgs)} training images for diagnostics")

# Compute D_r for each image
drs = []
with torch.no_grad():
    for img_t in train_imgs[:16]:  # 16 for similarity matrix
        x = img_t.to(device)
        _, targets_layer = image_encoder(x)
        pre_layer = int(len(targets_layer) * 0.9)
        num_layers = min(6, pre_layer)
        layer_drs = []
        for i in range(num_layers):
            idx = i + pre_layer - num_layers
            F_i = targets_layer[idx]
            d_r = make_radial_probe(F_i).reshape(-1)
            layer_drs.append(d_r.cpu())
        drs.append(layer_drs)

# Compute pairwise cosine similarity per layer
n_imgs = len(drs)
for l in range(num_layers):
    sims = []
    for i in range(n_imgs):
        for j in range(i + 1, n_imgs):
            sim = F.cosine_similarity(drs[i][l], drs[j][l], dim=0).item()
            sims.append(sim)
    ms = float(np.mean(sims))
    ss = float(np.std(sims))
    print(f"  Layer {l}: mean cos_sim = {ms:.4f} ± {ss:.4f}")

# ── Step 2: Compute offline prior with different N ───────────────────────────

print(f"\n{'='*70}")
print("STEP 2: Offline Prior with Different Numbers of Proxy Images")
print("=" * 70)

for n_proxy in [8, 16, 32]:
    print(f"\n  Computing prior from {n_proxy} proxy images...")
    proxy_imgs = train_imgs[:n_proxy]

    t0 = time.time()
    layer_priors_global = compute_offline_radial_prior(
        image_encoder, proxy_imgs, num_layers=6, per_channel=False)
    dt_global = time.time() - t0

    t0 = time.time()
    layer_priors_perchan = compute_offline_radial_prior(
        image_encoder, proxy_imgs, num_layers=6, per_channel=True)
    dt_perchan = time.time() - t0

    # Measure internal consistency of priors
    for name, priors, dt in [("global", layer_priors_global, dt_global),
                               ("per-channel", layer_priors_perchan, dt_perchan)]:
        norms = [p.norm().item() for p in priors]
        print(f"    {name:<15}: prior norms={[f'{n:.3f}' for n in norms]}, compute_time={dt:.1f}s")

# ── Step 3: Test prior weights on test images ────────────────────────────────

print(f"\n{'='*70}")
print("STEP 3: Prior Weight Sweep on Test Images")
print("=" * 70)

test_dir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_Data"
test_mask_dir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_GroundTruth"
test_data = load_images(test_dir, test_mask_dir, n=8)
test_masks = []
jpgs = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])[:16]
for jpg in jpgs:
    base = jpg.replace('.jpg', '')
    mask = Image.open(os.path.join(test_mask_dir, base + '_Segmentation.png')).convert('L')
    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    gt = (np.array(mask) > 128).astype(np.float32)
    if gt.sum() < 100:
        continue
    test_masks.append(gt)
    if len(test_masks) >= len(test_data):
        break

# Use 16 proxy images for the prior
n_proxy = 16
proxy_imgs = train_imgs[:n_proxy]
print(f"Using prior from {n_proxy} proxy images")

layer_priors_global = compute_offline_radial_prior(
    image_encoder, proxy_imgs, num_layers=6, per_channel=False)

eps_frac = 8.0 / 255.0
step_frac = 0.8 / 255.0
n_steps = 40

all_results = []

for prior_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
    dices, clean_dices, times = [], [], []
    for img_idx, (img_t, gt) in enumerate(zip(test_data, test_masks)):
        x = img_t.to(device)
        rg = x.max().item() - x.min().item()

        clean_dice = evaluate_isic(net, x, gt)
        if clean_dice < 0.01:
            continue

        use_prior = prior_weight > 0
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps_frac * rg, step_frac * rg, n_steps,
            amplitude_alpha=0.0, lambda_amp=0.1, amp_loss_mode="spatial",
            lambda_phase=0.0,
            offline_layer_priors=layer_priors_global if use_prior else None,
            prior_weight=prior_weight,
            radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_isic(net, x_adv, gt)
        dices.append(adv_dice)
        clean_dices.append(clean_dice)
        times.append(time.time() - t0)

    if len(dices) == 0:
        continue

    md_clean = float(np.mean(clean_dices))
    md_adv = float(np.mean(dices))
    md_delta = float(np.mean([c - a for c, a in zip(clean_dices, dices)]))
    sd_delta = float(np.std([c - a for c, a in zip(clean_dices, dices)]))
    mt = float(np.mean(times))
    all_results.append({
        "prior_weight": prior_weight, "n_proxy": n_proxy,
        "mean_clean": md_clean, "mean_adv": md_adv,
        "mean_delta": md_delta, "std_delta": sd_delta, "mean_time": mt,
    })
    print(f"  prior_w={prior_weight:.2f}: Clean={md_clean:.4f}  Adv={md_adv:.4f}  "
          f"Δ={md_delta:+.4f}±{sd_delta:.4f}  ({mt:.1f}s/img, n={len(dices)})")

# ── Step 4: Best prior weight with per-channel prior ─────────────────────────

print(f"\n{'='*70}")
print("STEP 4: Per-Channel Prior Comparison")
print("=" * 70)

layer_priors_perchan = compute_offline_radial_prior(
    image_encoder, proxy_imgs, num_layers=6, per_channel=True)

# Find best prior_weight from step 3
best_pw = max(all_results, key=lambda r: r["mean_delta"])["prior_weight"]
print(f"Best prior_weight from global prior: {best_pw:.2f}")

for name, priors in [("no prior", None), ("global prior", layer_priors_global),
                       ("per-channel prior", layer_priors_perchan)]:
    dices, clean_dices, times = [], [], []
    for img_idx, (img_t, gt) in enumerate(zip(test_data, test_masks)):
        x = img_t.to(device)
        rg = x.max().item() - x.min().item()
        clean_dice = evaluate_isic(net, x, gt)
        if clean_dice < 0.01:
            continue

        use_prior = (priors is not None)
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps_frac * rg, step_frac * rg, n_steps,
            amplitude_alpha=0.0, lambda_amp=0.1, amp_loss_mode="spatial",
            lambda_phase=0.0,
            offline_layer_priors=priors if use_prior else None,
            prior_weight=best_pw if use_prior else 0.0,
            radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_isic(net, x_adv, gt)
        dices.append(adv_dice)
        clean_dices.append(clean_dice)
        times.append(time.time() - t0)

    if len(dices) == 0:
        continue
    md_clean = float(np.mean(clean_dices))
    md_adv = float(np.mean(dices))
    md_delta = float(np.mean([c - a for c, a in zip(clean_dices, dices)]))
    mt = float(np.mean(times))
    all_results.append({
        "prior_type": name, "prior_weight": best_pw if use_prior else 0.0,
        "mean_clean": md_clean, "mean_adv": md_adv,
        "mean_delta": md_delta, "mean_time": mt,
    })
    print(f"  {name:<20}: Clean={md_clean:.4f}  Adv={md_adv:.4f}  "
          f"Δ={md_delta:+.4f}  ({mt:.1f}s/img, n={len(dices)})")

# ── Save ─────────────────────────────────────────────────────────────────────

out_path = os.path.join(OUTPUT_DIR, "oop_prior_experiment.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")

del net
torch.cuda.empty_cache()
print("Done — OOP experiment complete")
