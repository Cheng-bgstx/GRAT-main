#!/usr/bin/env python3
"""
ISIC Comparison: Clean vs Radial Probe vs SE-CGR on SAMed-ViT-L

Measures Dice and HD95 for:
  1. Clean (no attack)
  2. Radial-Directed Probe Attack (our method, λr=1.0)
  3. SE-CGR (previous best)

On ISIC 2016 test set with SAMed ViT-L backbone.
ISIC model outputs 2 channels (background, lesion) — use softmax+argmax.
"""

import os, sys, time, json, gc
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
from scipy.ndimage import distance_transform_edt

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth"
CKPT_REL = "SAMed/isic_512_lrexp5.0_pretrain_vit_l_epo300_bs12_lr0.005/epoch_59.pth"
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0")
criterion_l1 = torch.nn.L1Loss()

IMAGENET_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD_CPU  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

print("Loading models...")
# ISIC: num_classes=1 (1 fg class) but model outputs 2 channels [bg, fg]
net = load_samed_model("vit_l", SAM_CKPT, os.path.join(CKPT_BASE, CKPT_REL),
                       IMAGE_SIZE, 1, device=device).to(device)
net.eval()
# Use LoRA-tuned image encoder as attack target (not clean SAM encoder)
# Clean encoder features ≠ LoRA encoder features (cos ~0.63-0.72 for ISIC ViT-L)
image_encoder = net.sam.image_encoder
image_encoder.eval()


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_isic_images():
    tdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_Data"
    mdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_GroundTruth"
    jpgs = sorted([f for f in os.listdir(tdir) if f.endswith('.jpg')])
    imgs = []
    for jpg in jpgs:
        img = Image.open(os.path.join(tdir, jpg)).convert('RGB').resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)
        base = jpg.replace('.jpg', '')
        mask = Image.open(os.path.join(mdir, base + '_Segmentation.png')).convert('L')
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        gt = (np.array(mask) > 128).astype(np.float32)
        if gt.sum() < 100:
            continue
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        img_t = (img_t / 255.0 - IMAGENET_MEAN_CPU) / IMAGENET_STD_CPU
        imgs.append((img_t, gt, jpg))
        if len(imgs) >= 50:
            break
    return imgs

images = load_isic_images()
print(f"Loaded {len(images)} ISIC test images\n")


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_dice(pred, gt):
    eps = 1e-8
    intersection = (pred * gt).sum()
    return float(2.0 * intersection / (pred.sum() + gt.sum() + eps))

def compute_hd95(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2:
        return float('nan')
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float('nan')
    d1 = distance_transform_edt(1 - pred)
    d2 = distance_transform_edt(1 - gt)
    try:
        h1 = np.percentile(d1[gt > 0], 95)
        h2 = np.percentile(d2[pred > 0], 95)
        return float(max(h1, h2))
    except (IndexError, ValueError):
        return float('nan')


def evaluate_isic(img_t, gt):
    """SAMed ISIC inference: 2-channel output → softmax → class 1 = lesion."""
    with torch.no_grad():
        out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        logits = out['masks']  # [1, 2, H, W]
        probs = torch.softmax(logits.float(), dim=1)  # [1, 2, H, W]
        lesion_prob = probs[0, 1, :, :].cpu().numpy()  # [H, W], class 1

    best_dice, best_hd95 = 0.0, float('nan')
    best_thresh = 0.5
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        pred_bin = (lesion_prob > thresh).astype(np.float32)
        dice = compute_dice(pred_bin, gt)
        if dice > best_dice:
            best_dice = dice
            best_thresh = thresh
            best_hd95 = compute_hd95(pred_bin, gt)

    return best_dice, best_hd95


# ── Attack Methods ───────────────────────────────────────────────────────────

def make_radial_probe(F_clean):
    shape = F_clean.shape
    f_flat = F_clean.reshape(-1)
    d_r = F.normalize(f_flat, dim=0)
    return d_r.reshape(shape)


def attack_radial_probe(image_encoder, x, eps_pix, step_pix, steps):
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = image_encoder(x)
    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)
    radial_maps = []
    for i in range(num_layers):
        idx = i + pre_layer - num_layers
        radial_maps.append(make_radial_probe(targets_layer[idx]))
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)
        loss_org = criterion_l1(features * 0.9999, targets)
        import random
        indices = random.sample(range(num_layers), n_pick)
        fd_list = []
        for idx in indices:
            layer_idx = idx + pre_layer - num_layers
            F_c = targets_layer[layer_idx]
            F_a = features_layer[layer_idx]
            probe = radial_maps[idx]
            scale = 0.01 / (probe.abs().mean() + 1e-8)
            probe = probe * scale
            fd_list.append((F_a * 0.999 - F_c - 2.0 * probe))
        fd_stack = torch.stack(fd_list)
        selected_t = torch.stack([targets_layer[i + pre_layer - num_layers] for i in indices])
        im = torch.stack([selected_t[j].abs().mean() for j in range(len(selected_t))])
        im = im.view(-1, *([1] * (fd_stack.dim() - 1)))
        loss = loss_org + torch.abs(torch.mean(fd_stack / (im + 1e-6), dim=0)).mean()
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_secgr(image_encoder, x, eps_pix, step_pix, steps):
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = image_encoder(x)
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)
    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)
    from theory import synthesize_codd_probe
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)
        loss_org = criterion_l1(features * 0.9999, targets)
        import random
        indices = random.sample(range(num_layers), n_pick)
        fd_list = []
        for idx in indices:
            F_c = targets_layer[idx + pre_layer - num_layers]
            F_a = features_layer[idx + pre_layer - num_layers]
            probe = synthesize_codd_probe(
                F_clean=F_c.unsqueeze(0), F_adv=F_a.unsqueeze(0),
                direction="radial", use_random_noise=True)
            fd_list.append((F_a.unsqueeze(0)*0.999 - F_c.unsqueeze(0)) - 2.0*probe)
        fd_stack = torch.stack([f.squeeze(0) for f in fd_list])
        selected_t = torch.stack([targets_layer[i + pre_layer - num_layers] for i in indices])
        from attack_codd import _return_mean_value
        im = _return_mean_value(selected_t)
        if len(im.shape) > len(fd_stack.shape):
            im = im.view(fd_stack.shape[0], 1, 1, 1)
        loss = loss_org + torch.abs(torch.mean(fd_stack/(im+1e-6), dim=0)).mean()
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


# ── Experiment ───────────────────────────────────────────────────────────────

budgets = [
    ("eps=8/255",  8.0/255.0,  0.8/255.0,  40),
    ("eps=16/255", 16.0/255.0, 1.6/255.0,  40),
]

all_results = []
per_image_results = []

for budget_name, eps, step, steps in budgets:
    print(f"\n{'='*80}")
    print(f"  {budget_name}  |  SAMed ViT-L  |  ISIC 2016  |  {len(images)} images")
    print(f"{'='*80}")

    for condition in ["clean", "radial_probe", "secgr"]:
        dices, hd95s, hd95_valid, times = [], [], [], []
        for img_idx, (img_t, gt, name) in enumerate(images):
            x = img_t.to(device)

            if condition == "clean":
                t0 = time.time()
                dice, hd95 = evaluate_isic(x, gt)
                dt = time.time() - t0

            elif condition == "radial_probe":
                rg = x.max().item() - x.min().item()
                t0 = time.time()
                x_adv = attack_radial_probe(image_encoder, x, eps*rg, step*rg, steps)
                dice, hd95 = evaluate_isic(x_adv, gt)
                dt = time.time() - t0

            elif condition == "secgr":
                rg = x.max().item() - x.min().item()
                t0 = time.time()
                x_adv = attack_secgr(image_encoder, x, eps*rg, step*rg, steps)
                dice, hd95 = evaluate_isic(x_adv, gt)
                dt = time.time() - t0

            dices.append(dice)
            if not np.isnan(hd95):
                hd95_valid.append(hd95)
            times.append(dt)
            per_image_results.append({
                "budget": budget_name, "method": condition,
                "img_idx": img_idx, "dice": float(dice),
                "hd95": float(hd95) if not np.isnan(hd95) else None,
            })

            if img_idx < 3 or img_idx == len(images) - 1:
                label = {"clean": "Clean       ", "radial_probe": "Radial Probe",
                         "secgr": "SE-CGR      "}[condition]
                hd_str = f"{hd95:.1f}" if not np.isnan(hd95) else "nan"
                print(f"  [{img_idx}] {label}  Dice={dice:.4f}  HD95={hd_str}")

        md = float(np.mean(dices))
        sd = float(np.std(dices))
        mh = float(np.mean(hd95_valid)) if hd95_valid else float('nan')
        mdt = float(np.mean(times))
        all_results.append({
            "budget": budget_name, "method": condition,
            "mean_dice": md, "std_dice": sd,
            "mean_hd95": mh, "n_hd95_valid": len(hd95_valid),
            "time_per_image": mdt,
        })

        print(f"  {'─'*50}")
        print(f"  {'SUMMARY':>14}: Dice={md:.4f}±{sd:.4f}  HD95={mh:.1f}  "
              f"(valid: {len(hd95_valid)}/{len(dices)})  {mdt:.1f}s/img\n")


# ── Final Comparison Table ───────────────────────────────────────────────────

print(f"\n{'='*90}")
print("ISIC COMPARISON: Clean vs Radial Probe vs SE-CGR on SAMed ViT-L")
print(f"{'='*90}")

for budget_name in ["eps=8/255", "eps=16/255"]:
    print(f"\n── {budget_name} ──")
    print(f"  {'Condition':<20} {'Dice':>12} {'ΔDice':>12} {'HD95':>10} {'ΔHD95':>10}")
    print(f"  {'-'*64}")

    entries = [r for r in all_results if r["budget"] == budget_name]
    clean = [r for r in entries if r["method"] == "clean"]
    if clean:
        cd, ch = clean[0]["mean_dice"], clean[0]["mean_hd95"]
        print(f"  {'Clean (no attack)':<20} {cd:>12.4f} {'—':>12} {ch:>10.1f} {'—':>10}")

    for method, label in [("radial_probe", "Radial Probe (ours)"),
                           ("secgr", "SE-CGR")]:
        m = [r for r in entries if r["method"] == method]
        if m and clean:
            dd = m[0]["mean_dice"] - cd
            dh = m[0]["mean_hd95"] - ch if not np.isnan(m[0]["mean_hd95"]) and not np.isnan(ch) else float('nan')
            print(f"  {label:<20} {m[0]['mean_dice']:>12.4f} {dd:>+12.4f} "
                  f"{m[0]['mean_hd95']:>10.1f} {dh:>+10.1f}")

    # Per-pair paired diffs
    print(f"\n  Per-image breakdown (first 10):")
    clean_pi = {r["img_idx"]: r["dice"] for r in per_image_results
                if r["budget"] == budget_name and r["method"] == "clean"}
    for method, label in [("radial_probe", "Radial Probe"), ("secgr", "SE-CGR")]:
        pi = {r["img_idx"]: r["dice"] for r in per_image_results
              if r["budget"] == budget_name and r["method"] == method}
        diffs = [pi[i] - clean_pi[i] for i in sorted(clean_pi.keys()) if i in pi]
        print(f"    {label:<15} ΔDice: " + " ".join(f"{d:+.3f}" for d in diffs[:10]) +
              (f"  mean={np.mean(diffs):+.4f}" if diffs else ""))


# ── Save ─────────────────────────────────────────────────────────────────────

out_path = os.path.join(OUTPUT_DIR, "isic_comparison_results.json")
with open(out_path, "w") as f:
    json.dump({"summary": all_results, "per_image": per_image_results}, f, indent=2)
print(f"\nSaved to {out_path}")

del net, source_sam
gc.collect()
try:
    del source_sam
except:
    pass
torch.cuda.empty_cache()
print("Done")
