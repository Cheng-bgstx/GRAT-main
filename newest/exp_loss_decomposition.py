#!/usr/bin/env python3
"""
Feature-Space Loss Decomposition — Optimizing PAID Channel Attack.

Tests different loss functions for feature-space attack:
  1. L1 only (current gold standard)
  2. L1 + λ_cos * (1 - cos_sim)  — emphasize Φ attack
  3. L1 + λ_amp * | |f_adv|-|f_clean| | — emphasize A attack
  4. L1 + λ_cos * (...) + λ_amp * (...) — both channels
  5. L2 only
  6. SmoothL1

Also tests multi-layer: use features from multiple encoder layers, not just final.
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F

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
}
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0")
IM_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

MODELS = [
    ("ISIC_vit_b", "vit_b", "SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 1, "ISIC"),
    ("ISIC_vit_l", "vit_l", "SAMed/isic_512_lrexp5.0_pretrain_vit_l_epo300_bs12_lr0.005/epoch_59.pth", 1, "ISIC"),
]


def load_data(ds_name, n=12):
    from PIL import Image
    if ds_name == "ISIC":
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
        if dice > best: best = dice
    return best


def attack_decomposed(encoder, x, eps_pix, step_pix, steps, edge_map_fn=None,
                      loss_type="l1", lambda_cos=0.0, lambda_amp=0.0,
                      use_multilayer=False, num_layers=6):
    """
    Feature-space attack with configurable loss decomposition.

    loss_type: "l1", "l2", "smooth_l1"
    lambda_cos: weight on (1 - cos_sim) → attacks Φ
    lambda_amp: weight on | |f_adv| - |f_clean| | → attacks A
    use_multilayer: if True, use intermediate layer features too
    """
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = encoder(x)

    if use_multilayer:
        pre_layer = int(len(targets_layer) * 0.9)
        nl = min(num_layers, pre_layer)
        layer_targets = [targets_layer[i + pre_layer - nl] for i in range(nl)]

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = edge_map_fn(x).detach() if edge_map_fn else torch.zeros_like(x)
    momentum = torch.zeros_like(x)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = encoder(x_adv)

        # Base feature loss
        if loss_type == "l1":
            loss = F.l1_loss(features, targets)
        elif loss_type == "l2":
            loss = F.mse_loss(features, targets)
        elif loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(features, targets)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Multi-layer: add loss from intermediate layers
        if use_multilayer:
            pre_layer = int(len(features_layer) * 0.9)
            nl = min(num_layers, pre_layer)
            for i in range(nl):
                lidx = i + pre_layer - nl
                layer_loss = F.l1_loss(features_layer[lidx], layer_targets[i])
                weight = (i + 1) / nl  # deeper layers weighted more
                loss = loss + weight * layer_loss

        # Cosine loss (attacks Φ — direction)
        if lambda_cos > 0:
            cos_sim = F.cosine_similarity(
                features.reshape(-1), targets.reshape(-1), dim=0)
            loss = loss + lambda_cos * (1.0 - cos_sim)

        # Amplitude loss (attacks A — magnitude)
        if lambda_amp > 0:
            amp_diff = torch.abs(torch.abs(features) - torch.abs(targets)).mean()
            loss = loss + lambda_amp * amp_diff

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# ── Main Experiment ──────────────────────────────────────────────────────────

# loss_type, lambda_cos, lambda_amp, use_multilayer
LOSS_CONFIGS = [
    ("L1", "l1", 0.0, 0.0, False),
    ("L2", "l2", 0.0, 0.0, False),
    ("SmoothL1", "smooth_l1", 0.0, 0.0, False),
    ("L1+cos(0.1)", "l1", 0.1, 0.0, False),
    ("L1+cos(0.5)", "l1", 0.5, 0.0, False),
    ("L1+cos(1.0)", "l1", 1.0, 0.0, False),
    ("L1+amp(0.1)", "l1", 0.0, 0.1, False),
    ("L1+amp(0.5)", "l1", 0.0, 0.5, False),
    ("L1+amp(1.0)", "l1", 0.0, 1.0, False),
    ("L1+cos+amp(0.1)", "l1", 0.1, 0.1, False),
    ("L1+cos+amp(0.5)", "l1", 0.5, 0.5, False),
    ("L1-multilayer", "l1", 0.0, 0.0, True),
]

EPS_LIST = [4.0/255, 8.0/255]
STEPS_LIST = [10, 20, 40]

print("=" * 90)
print("LOSS DECOMPOSITION: Optimizing Φ/A Channel Attack")
print("=" * 90)

all_results = []

for model_name, vit, ckpt_rel, n_class, ds in MODELS:
    print(f"\n{'='*80}")
    print(f"  {model_name}  |  {vit}  |  {ds}")
    print(f"{'='*80}")

    net = load_samed_model(vit, SAM_CKPTS[vit],
                           os.path.join(CKPT_BASE, ckpt_rel),
                           IMAGE_SIZE, n_class, device=device)
    net.eval()
    encoder = net.sam.image_encoder
    encoder.eval()

    images = load_data(ds, n=12)
    print(f"  Loaded {len(images)} images")

    for config_label, loss_type, lam_cos, lam_amp, use_ml in LOSS_CONFIGS:
        for eps_frac in EPS_LIST:
            for n_steps in STEPS_LIST:
                dices, clean_dices, times = [], [], []

                for img_t, gt in images:
                    x = img_t.to(device)
                    rg = x.max().item() - x.min().item()
                    eps_pix = eps_frac * rg
                    step_pix = eps_pix / max(n_steps, 1)

                    cd = evaluate_isic(net, x, gt)
                    if cd < 0.01: continue

                    try:
                        t0 = time.time()
                        x_adv = attack_decomposed(
                            encoder, x, eps_pix, step_pix, n_steps,
                            edge_map_fn=_get_edge_map,
                            loss_type=loss_type, lambda_cos=lam_cos,
                            lambda_amp=lam_amp, use_multilayer=use_ml)
                        ad = evaluate_isic(net, x_adv, gt)
                        times.append(time.time() - t0)
                        dices.append(ad)
                        clean_dices.append(cd)
                    except Exception as e:
                        continue

                if len(dices) == 0: continue

                mc = float(np.mean(clean_dices))
                ma = float(np.mean(dices))
                md = mc - ma
                mt = float(np.mean(times))

                all_results.append({
                    "model": model_name, "config": config_label,
                    "loss_type": loss_type, "lambda_cos": lam_cos,
                    "lambda_amp": lam_amp, "multilayer": use_ml,
                    "eps": f"{eps_frac*255:.0f}/255", "steps": n_steps,
                    "mean_clean": mc, "mean_adv": ma, "mean_delta": md,
                    "mean_time": mt, "n": len(dices),
                })

                print(f"  {config_label:<22} eps={eps_frac*255:4.0f}/255 steps={n_steps:2d}  "
                      f"Clean={mc:.4f} Adv={ma:.4f} Δ={md:+.4f} ({mt:.1f}s)")

    del net
    torch.cuda.empty_cache()

# ── Summary: Best Config ─────────────────────────────────────────────────────

print(f"\n{'='*90}")
print("BEST CONFIGURATION PER MODEL")
print(f"{'='*90}")

for model_name, _, _, _, _ in MODELS:
    model_results = [r for r in all_results if r["model"] == model_name]
    print(f"\n── {model_name} ──")

    for eps_str in [f"{e*255:.0f}/255" for e in EPS_LIST]:
        for n_steps in STEPS_LIST:
            best = max([r for r in model_results
                       if r["eps"] == eps_str and r["steps"] == n_steps],
                       key=lambda r: r["mean_delta"], default=None)
            if best:
                l1_base = next((r for r in model_results
                               if r["eps"] == eps_str and r["steps"] == n_steps
                               and r["config"] == "L1"), None)
                l1_delta = l1_base["mean_delta"] if l1_base else 0.0
                improvement = best["mean_delta"] - l1_delta
                marker = " *** IMPROVEMENT!" if improvement > 0.01 else ""
                print(f"  eps={eps_str} steps={n_steps:2d}: best={best['config']:<22} "
                      f"Δ={best['mean_delta']:+.4f} (vs L1 Δ={l1_delta:+.4f}, {improvement:+.4f}){marker}")

out_path = os.path.join(OUTPUT_DIR, "loss_decomposition.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
print("Done — loss decomposition complete")
