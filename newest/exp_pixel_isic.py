#!/usr/bin/env python3
"""
Pixel-Space PGD vs Pure L1 — ISIC SAMed models only.

Covers ISIC_vit_{b,l,h} for the pixel-vs-feature comparison.
Loads prior results from pixel_vs_feature_space.json if available.
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

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0")
IM_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

MODELS = {
    "ISIC_vit_b": {"vit": "vit_b", "ckpt": "SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", "n_class": 1, "ds": "ISIC"},
    "ISIC_vit_l": {"vit": "vit_l", "ckpt": "SAMed/isic_512_lrexp5.0_pretrain_vit_l_epo300_bs12_lr0.005/epoch_59.pth", "n_class": 1, "ds": "ISIC"},
    "ISIC_vit_h": {"vit": "vit_h", "ckpt": "SAMed/isic_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_249.pth", "n_class": 1, "ds": "ISIC"},
}

def load_isic_data(n=12):
    tdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_Data"
    mdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_GroundTruth"
    jpgs = sorted([f for f in os.listdir(tdir) if f.endswith('.jpg')])[:n]
    imgs = []
    for jpg in jpgs:
        img = Image.open(os.path.join(tdir, jpg)).convert('RGB').resize((512, 512), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)
        base = jpg.replace('.jpg', '')
        mask = Image.open(os.path.join(mdir, base + '_Segmentation.png')).convert('L')
        mask = mask.resize((512, 512), Image.NEAREST)
        gt = (np.array(mask) > 128).astype(np.float32)
        if gt.sum() < 100: continue
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        img_t = (img_t / 255.0 - IM_MEAN_CPU) / IM_STD_CPU
        imgs.append((img_t, gt))
    return imgs

def evaluate_samed_isic(net, img_t, gt):
    with torch.no_grad():
        out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        m = out['masks']
    probs = torch.softmax(m.float(), dim=1)
    lesion_prob = probs[0, 1, :, :].cpu().numpy()
    best = 0.0
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        pred_bin = (lesion_prob > thresh).astype(np.float32)
        inter = (pred_bin * gt).sum()
        dice = float(2.0 * inter / (pred_bin.sum() + gt.sum() + 1e-8))
        if dice > best: best = dice
    return best

def attack_pure_l1(encoder, x, eps_pix, step_pix, steps, edge_map_fn=None):
    criterion = torch.nn.L1Loss()
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, _ = encoder(x)
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = edge_map_fn(x).detach() if edge_map_fn else torch.zeros_like(x)
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, _ = encoder(x_adv)
        loss = criterion(features, targets)
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()

def attack_pixel_pgd(net, x, eps_pix, step_pix, steps):
    max_val, min_val = x.max(), x.min()
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    with torch.no_grad():
        clean_out = net(x, multimask_output=False, image_size=IMAGE_SIZE)
        clean_probs = torch.softmax(clean_out['masks'].float(), dim=1)
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        out = net(x_adv, multimask_output=False, image_size=IMAGE_SIZE)
        probs = torch.softmax(out['masks'].float(), dim=1)
        inter = (probs[:, 1:2] * clean_probs[:, 1:2]).sum()
        denom = probs[:, 1:2].sum() + clean_probs[:, 1:2].sum() + 1e-8
        loss = 2.0 * inter / denom
        grad = torch.autograd.grad(-loss, x_adv)[0]
        grad = grad / (torch.norm(grad, p=2) + 1e-12)
        x_adv = x_adv.detach() + step_pix * grad.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()

def attack_pixel_mse(net, x, eps_pix, step_pix, steps):
    max_val, min_val = x.max(), x.min()
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    with torch.no_grad():
        clean_out = net(x, multimask_output=False, image_size=IMAGE_SIZE)
        clean_probs = torch.softmax(clean_out['masks'].float(), dim=1).detach()
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        out = net(x_adv, multimask_output=False, image_size=IMAGE_SIZE)
        probs = torch.softmax(out['masks'].float(), dim=1)
        loss = -F.mse_loss(probs, clean_probs)
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad / (torch.norm(grad, p=2) + 1e-12)
        x_adv = x_adv.detach() + step_pix * grad.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()

# ── Main ────────────────────────────────────────────────────────────────────────

EPS_LIST = [4.0/255, 8.0/255, 16.0/255]
STEPS_LIST = [10, 20, 40]
N_IMAGES = 12

prior_path = os.path.join(OUTPUT_DIR, "pixel_vs_feature_isic.json")
all_results = []
if os.path.exists(prior_path):
    with open(prior_path) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} prior results")

completed_models = set(r["model"] for r in all_results)

print("=" * 90)
print("PIXEL-SPACE vs FEATURE-SPACE PGD — ISIC Models")
print("=" * 90)

for model_key in sorted(MODELS.keys()):
    cfg = MODELS[model_key]
    ds = cfg["ds"]

    if model_key in completed_models:
        print(f"\n  SKIP {model_key}: already in prior results")
        continue

    print(f"\n{'='*80}")
    print(f"  {model_key}  |  {cfg['vit']}  |  {ds}")
    print(f"{'='*80}")

    try:
        net = load_samed_model(cfg["vit"], SAM_CKPTS[cfg["vit"]],
                               os.path.join(CKPT_BASE, cfg["ckpt"]),
                               IMAGE_SIZE, cfg["n_class"], device=device)
        net.eval()
        encoder = net.sam.image_encoder
        encoder.eval()
    except Exception as e:
        print(f"  SKIP: Load failed: {e}")
        continue

    images = load_isic_data(n=N_IMAGES)
    print(f"  Loaded {len(images)} images")

    for eps_frac in EPS_LIST:
        for n_steps in STEPS_LIST:
            for method_key in ["pure_l1", "pixel_pgd", "pixel_mse"]:
                dices, clean_dices, times = [], [], []

                for img_t, gt in images:
                    x = img_t.to(device)
                    rg = x.max().item() - x.min().item()
                    eps_pix = eps_frac * rg
                    step_pix = eps_pix / max(n_steps, 1)

                    clean_dice = evaluate_samed_isic(net, x, gt)
                    if clean_dice < 0.01: continue

                    try:
                        t0 = time.time()
                        if method_key == "pure_l1":
                            x_adv = attack_pure_l1(encoder, x, eps_pix, step_pix, n_steps, _get_edge_map)
                        elif method_key == "pixel_pgd":
                            x_adv = attack_pixel_pgd(net, x, eps_pix, step_pix, n_steps)
                        elif method_key == "pixel_mse":
                            x_adv = attack_pixel_mse(net, x, eps_pix, step_pix, n_steps)
                        adv_dice = evaluate_samed_isic(net, x_adv, gt)
                        times.append(time.time() - t0)
                        dices.append(adv_dice)
                        clean_dices.append(clean_dice)
                    except Exception as e:
                        print(f"    Error [{method_key}]: {e}")
                        continue

                if len(dices) == 0: continue

                mc = float(np.mean(clean_dices))
                ma = float(np.mean(dices))
                md = mc - ma
                mt = float(np.mean(times))

                all_results.append({
                    "model": model_key, "dataset": ds, "vit": cfg["vit"],
                    "eps": f"{eps_frac*255:.0f}/255", "steps": n_steps,
                    "method": method_key,
                    "mean_clean": mc, "mean_adv": ma, "mean_delta": md,
                    "mean_time": mt, "n_valid": len(dices),
                })

                labels = {"pure_l1":"Pure L1 (feat)", "pixel_pgd":"Pixel PGD (dice)", "pixel_mse":"Pixel PGD (MSE)"}
                print(f"  eps={eps_frac*255:4.0f}/255 steps={n_steps:2d} {labels.get(method_key,method_key):<22}"
                      f"Clean={mc:.4f} Adv={ma:.4f} Δ={md:+.4f} ({mt:.1f}s)")

    del net; torch.cuda.empty_cache(); gc.collect()
    with open(prior_path, "w") as f:
        json.dump(all_results, f, indent=2)

print(f"\nSaved to {prior_path}")
print("Done — ISIC pixel-vs-feature complete")
