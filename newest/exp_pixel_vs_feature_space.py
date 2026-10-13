#!/usr/bin/env python3
"""
Pixel-Space PGD vs Pure L1 Feature-Space — The Critical Basline Comparison.

Pure L1 Feature-Space PGD optimizes L1 in encoder feature space.
Pixel-Space PGD optimizes Dice loss directly on segmentation output.

Theory prediction: Feature-space attack should be MORE effective because:
1. It targets the full information bottleneck (encoder output)
2. It avoids the mask decoder's robustness to pixel perturbations
3. The encoder collapse disrupts ALL downstream tasks

This experiment compares both across all 9 SAMed models.
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

SAMED_MODELS = {
    "Syn_vit_b": {"vit": "vit_b", "ckpt": "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", "n_class": 8, "ds": "Synapse"},
    "Syn_vit_l": {"vit": "vit_l", "ckpt": "SAMed/Synapse_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_249.pth", "n_class": 8, "ds": "Synapse"},
    "Syn_vit_h": {"vit": "vit_h", "ckpt": "SAMed/Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_299.pth", "n_class": 8, "ds": "Synapse"},
    "ISIC_vit_b": {"vit": "vit_b", "ckpt": "SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", "n_class": 1, "ds": "ISIC"},
    "ISIC_vit_l": {"vit": "vit_l", "ckpt": "SAMed/isic_512_lrexp5.0_pretrain_vit_l_epo300_bs12_lr0.005/epoch_59.pth", "n_class": 1, "ds": "ISIC"},
    "ISIC_vit_h": {"vit": "vit_h", "ckpt": "SAMed/isic_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_249.pth", "n_class": 1, "ds": "ISIC"},
    "REF_vit_b": {"vit": "vit_b", "ckpt": "SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_199.pth", "n_class": 2, "ds": "REFUGE"},
    "REF_vit_l": {"vit": "vit_l", "ckpt": "SAMed/REFUGE_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_299.pth", "n_class": 2, "ds": "REFUGE"},
    "REF_vit_h": {"vit": "vit_h", "ckpt": "SAMed/REFUGE_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_299.pth", "n_class": 2, "ds": "REFUGE"},
}

# ── Data Loading ────────────────────────────────────────────────────────────────

def load_data(ds, n=12):
    if ds == "Synapse":
        from exp_paid_optimal_attack import load_synapse_slices
        return load_synapse_slices(n)
    elif ds == "ISIC":
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
    elif ds == "REFUGE":
        from exp_paid_optimal_attack import load_refuge_images
        return load_refuge_images(n)
    raise ValueError(f"Unknown dataset: {ds}")

# ── Evaluation ──────────────────────────────────────────────────────────────────

def evaluate_samed(net, img_t, gt, ds):
    with torch.no_grad():
        out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        m = out['masks']

    if ds == "Synapse":
        pred = torch.argmax(torch.softmax(m.float(), dim=1), dim=1).squeeze(0).cpu().numpy()
        dices = []
        for c in range(1, 8):
            pred_c = (pred == c).astype(np.float32)
            gt_c = (gt == c).astype(np.float32)
            if gt_c.sum() == 0: continue
            inter = (pred_c * gt_c).sum()
            dices.append(float(2.0 * inter / (pred_c.sum() + gt_c.sum() + 1e-8)))
        return float(np.mean(dices)) if dices else 0.0
    elif ds == "ISIC":
        probs = torch.softmax(m.float(), dim=1)
        lesion_prob = probs[0, 1, :, :].cpu().numpy()
        best = 0.0
        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            pred_bin = (lesion_prob > thresh).astype(np.float32)
            inter = (pred_bin * gt).sum()
            dice = float(2.0 * inter / (pred_bin.sum() + gt.sum() + 1e-8))
            if dice > best: best = dice
        return best
    elif ds == "REFUGE":
        probs = torch.softmax(m.float(), dim=1)
        cup_gt, disc_gt = gt[0], gt[1]
        dices = []
        for cls_idx, gt_mask in [(2, cup_gt), (1, disc_gt)]:
            cls_prob = probs[0, cls_idx, :, :].cpu().numpy()
            best = 0.0
            for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
                pred_bin = (cls_prob > thresh).astype(np.float32)
                inter = (pred_bin * gt_mask).sum()
                dice = float(2.0 * inter / (pred_bin.sum() + gt_mask.sum() + 1e-8))
                if dice > best: best = dice
            dices.append(best)
        return float(np.mean(dices))

# ── Attacks ─────────────────────────────────────────────────────────────────────

def attack_pure_l1(encoder, x, eps_pix, step_pix, steps, edge_map_fn=None):
    """Pure L1 feature-space PGD."""
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


def attack_pixel_pgd(net, x, eps_pix, step_pix, steps, dataset_name, n_class):
    """Pixel-space PGD optimizing Dice loss on model output."""
    max_val, min_val = x.max(), x.min()
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5

    # Get clean prediction for reference (used in dice targets)
    with torch.no_grad():
        clean_out = net(x, multimask_output=False, image_size=IMAGE_SIZE)
        clean_probs = torch.softmax(clean_out['masks'].float(), dim=1)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        out = net(x_adv, multimask_output=False, image_size=IMAGE_SIZE)
        probs = torch.softmax(out['masks'].float(), dim=1)

        # Dice loss — maximize Dice with clean prediction (inverting later)
        if dataset_name == "ISIC":
            # Binary: maximize difference between lesion and background
            lesion_prob = probs[0, 1, :, :]
            clean_lesion = clean_probs[0, 1, :, :]
            # Use soft Dice as differentiable surrogate
            inter = (lesion_prob * clean_lesion).sum()
            denom = lesion_prob.sum() + clean_lesion.sum() + 1e-8
            dice = 2.0 * inter / denom
            loss = dice  # Maximize → use negative grad
            grad = torch.autograd.grad(-loss, x_adv)[0]
        elif dataset_name == "Synapse":
            loss = 0.0
            for c in range(1, n_class + 1):
                inter = (probs[:, c:c+1, :, :] * clean_probs[:, c:c+1, :, :]).sum()
                denom = probs[:, c:c+1, :, :].sum() + clean_probs[:, c:c+1, :, :].sum() + 1e-8
                loss = loss + 2.0 * inter / denom
            loss = loss / n_class
            grad = torch.autograd.grad(-loss, x_adv)[0]
        elif dataset_name == "REFUGE":
            loss = 0.0
            for c in [1, 2]:
                inter = (probs[:, c:c+1, :, :] * clean_probs[:, c:c+1, :, :]).sum()
                denom = probs[:, c:c+1, :, :].sum() + clean_probs[:, c:c+1, :, :].sum() + 1e-8
                loss = loss + 2.0 * inter / denom
            loss = loss / 2
            grad = torch.autograd.grad(-loss, x_adv)[0]
        else:
            # Fallback: maximize L2 distance from clean output
            loss = -F.mse_loss(probs, clean_probs)
            grad = torch.autograd.grad(loss, x_adv)[0]

        grad = grad / (torch.norm(grad, p=2) + 1e-12)
        x_adv = x_adv.detach() + step_pix * grad.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


def attack_clean_reference(net, x, eps_pix, step_pix, steps, dataset_name, n_class):
    """PGD that maximizes MSE between clean and adversarial output."""
    max_val, min_val = x.max(), x.min()
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5

    with torch.no_grad():
        clean_out = net(x, multimask_output=False, image_size=IMAGE_SIZE)
        clean_masks = clean_out['masks'].float().detach()

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        out = net(x_adv, multimask_output=False, image_size=IMAGE_SIZE)
        loss = -F.mse_loss(out['masks'].float(), clean_masks)
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

all_results = []

print("=" * 90)
print("PIXEL-SPACE PGD vs PURE L1 FEATURE-SPACE — Critical Comparison")
print("=" * 90)

for model_key in sorted(SAMED_MODELS.keys()):
    cfg = SAMED_MODELS[model_key]
    ds = cfg["ds"]
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

    images = load_data(ds, n=N_IMAGES)
    print(f"  Loaded {len(images)} images")

    for eps_frac in EPS_LIST:
        for n_steps in STEPS_LIST:
            for method_key in ["pure_l1", "pixel_pgd", "pixel_mse"]:
                dices, clean_dices, times = [], [], []

                for img_data in images:
                    if ds == "Synapse":
                        x = torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
                        gt = img_data[1]
                    elif ds == "ISIC":
                        x = img_data[0].to(device)
                        gt = img_data[1]
                    elif ds == "REFUGE":
                        x = img_data[0].unsqueeze(0).float().to(device)
                        gt = img_data[1]

                    rg = x.max().item() - x.min().item()
                    eps_pix = eps_frac * rg
                    step_pix = eps_pix / max(n_steps, 1)

                    clean_dice = evaluate_samed(net, x, gt, ds)
                    if clean_dice < 0.01:
                        continue

                    try:
                        t0 = time.time()
                        if method_key == "pure_l1":
                            x_adv = attack_pure_l1(encoder, x, eps_pix, step_pix, n_steps, _get_edge_map)
                        elif method_key == "pixel_pgd":
                            x_adv = attack_pixel_pgd(net, x, eps_pix, step_pix, n_steps, ds, cfg["n_class"])
                        elif method_key == "pixel_mse":
                            x_adv = attack_clean_reference(net, x, eps_pix, step_pix, n_steps, ds, cfg["n_class"])
                        adv_dice = evaluate_samed(net, x_adv, gt, ds)
                        times.append(time.time() - t0)
                        dices.append(adv_dice)
                        clean_dices.append(clean_dice)
                    except Exception as e:
                        print(f"    Error [{method_key}]: {e}")
                        continue

                if len(dices) == 0:
                    print(f"  eps={eps_frac*255:4.0f}/255 steps={n_steps:2d} {method_key}: NO VALID RESULTS")
                    continue

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

    del net
    torch.cuda.empty_cache()
    gc.collect()

# ── Summary ─────────────────────────────────────────────────────────────────────

print(f"\n{'='*90}")
print("FEATURE-SPACE vs PIXEL-SPACE SUMMARY")
print(f"{'='*90}")

for eps_str in sorted(set(r["eps"] for r in all_results)):
    for n_steps in sorted(set(r["steps"] for r in all_results)):
        print(f"\n  eps={eps_str} steps={n_steps:2d}")
        print(f"  {'Model':<20} {'Pure L1 (feat)':>14} {'Pixel PGD (dice)':>16} {'Pixel PGD (MSE)':>16} {'Winner':<20}")
        print(f"  {'-'*85}")
        for model_key in sorted(SAMED_MODELS.keys()):
            methods = {}
            for m in ["pure_l1", "pixel_pgd", "pixel_mse"]:
                rows = [r for r in all_results if r["model"]==model_key and r["eps"]==eps_str
                        and r["steps"]==n_steps and r["method"]==m]
                methods[m] = rows[0]["mean_delta"] if rows else 0.0
            winner = "Pure L1 (feat)" if methods["pure_l1"] >= max(methods["pixel_pgd"], methods["pixel_mse"]) else \
                     "Pixel PGD (dice)" if methods["pixel_pgd"] >= methods["pixel_mse"] else "Pixel PGD (MSE)"
            print(f"  {model_key:<20} {methods['pure_l1']:>+14.4f} {methods['pixel_pgd']:>+16.4f} {methods['pixel_mse']:>+16.4f} {winner:<20}")

out_path = os.path.join(OUTPUT_DIR, "pixel_vs_feature_space.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
print("Done — feature-space vs pixel-space comparison complete")
