#!/usr/bin/env python3
"""
Amplitude-Decay Attack — Combines L1+amp at low steps with pure L1 at high steps.

Tests different decay schedules for λ_amp(t):
  - linear: λ(t) = λ_max * (1 - t/T)
  - step: λ(t) = λ_max if t < T/2 else 0
  - constant: λ(t) = λ_max (no decay)

Also tests λ_max ∈ {0.5, 1.0, 2.0}.
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
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
CKPT_REL = "SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0")
IM_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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


def attack_amp_decay(encoder, x, eps_pix, step_pix, steps, edge_map_fn=None,
                     lam_max=1.0, decay="linear"):
    """L1 + amplitude loss with decaying λ_amp."""
    criterion = torch.nn.L1Loss()
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, _ = encoder(x)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = edge_map_fn(x).detach() if edge_map_fn else torch.zeros_like(x)
    momentum = torch.zeros_like(x)

    for t in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, _ = encoder(x_adv)

        loss = criterion(features, targets)

        # Compute lambda_amp for this step
        if decay == "linear":
            lam = lam_max * (1.0 - t / max(steps - 1, 1))
        elif decay == "step":
            lam = lam_max if t < steps // 2 else 0.0
        elif decay == "constant":
            lam = lam_max
        else:
            lam = 0.0

        # Amplitude loss
        if lam > 0:
            amp_diff = torch.abs(torch.abs(features) - torch.abs(targets)).mean()
            loss = loss + lam * amp_diff

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# ── Main ─────────────────────────────────────────────────────────────────────

print("=" * 80)
print("AMPLITUDE-DECAY ATTACK: Combining Amp Loss + Pure L1")
print("=" * 80)

net = load_samed_model("vit_b", SAM_CKPT, os.path.join(CKPT_BASE, CKPT_REL),
                       IMAGE_SIZE, 1, device=device)
net.eval()
encoder = net.sam.image_encoder
encoder.eval()

images = load_isic_data(n=12)
print(f"Loaded {len(images)} images")

EPS_LIST = [4.0/255, 8.0/255]
STEPS_LIST = [10, 20, 40]
CONFIGS = [
    ("L1 (baseline)", 0.0, "constant"),
    ("L1+amp(1.0)", 1.0, "constant"),
    ("AmpDecay-linear(1.0)", 1.0, "linear"),
    ("AmpDecay-step(1.0)", 1.0, "step"),
    ("AmpDecay-linear(2.0)", 2.0, "linear"),
]

all_results = []

for config_label, lam_max, decay in CONFIGS:
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
                    x_adv = attack_amp_decay(encoder, x, eps_pix, step_pix, n_steps,
                                             edge_map_fn=_get_edge_map,
                                             lam_max=lam_max, decay=decay)
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
                "config": config_label, "lam_max": lam_max, "decay": decay,
                "eps": f"{eps_frac*255:.0f}/255", "steps": n_steps,
                "mean_clean": mc, "mean_adv": ma, "mean_delta": md,
                "mean_time": mt, "n": len(dices),
            })

            l1_delta = next((r["mean_delta"] for r in all_results
                           if r["config"] == "L1 (baseline)" and r["eps"] == f"{eps_frac*255:.0f}/255"
                           and r["steps"] == n_steps), 0)
            improvement = md - l1_delta
            marker = " *** BETTER!" if improvement > 0.005 else ""
            print(f"  {config_label:<28} eps={eps_frac*255:4.0f}/255 steps={n_steps:2d} "
                  f"Δ={md:+.4f} (vs L1 {l1_delta:+.4f}, {improvement:+.4f}){marker}")

del net
torch.cuda.empty_cache()

# Summary
print(f"\n{'='*80}")
print("BEST CONFIGURATION")
print(f"{'='*80}")
for eps_str in [f"{e*255:.0f}/255" for e in EPS_LIST]:
    for n_steps in STEPS_LIST:
        best = max([r for r in all_results
                   if r["eps"] == eps_str and r["steps"] == n_steps],
                   key=lambda r: r["mean_delta"], default=None)
        if best:
            print(f"  eps={eps_str} steps={n_steps:2d}: {best['config']:<28} Δ={best['mean_delta']:+.4f}")

out_path = os.path.join(OUTPUT_DIR, "amp_decay.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
print("Done")
