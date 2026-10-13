#!/usr/bin/env python3
"""
Cross-Model Transfer Attack — Validates PAID Φ Claim.

PAID Theory: Φ (radial direction f/|f|) is cross-model invariant.
Prediction: Φ-directed attacks transfer better than unconstrained attacks.

Experiment:
  Source: ISIC SAMed vit_b → Target: ISIC SAMed vit_l, vit_h
  Attack vit_b encoder, evaluate transfer to vit_l/vit_h models.
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
from attack_radial_optimized import attack_radial_optimized, attack_random_probe

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

SAMED_ISIC = {
    "ISIC_vit_b": {"vit": "vit_b", "ckpt": "SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", "n_class": 1},
    "ISIC_vit_l": {"vit": "vit_l", "ckpt": "SAMed/isic_512_lrexp5.0_pretrain_vit_l_epo300_bs12_lr0.005/epoch_59.pth", "n_class": 1},
    "ISIC_vit_h": {"vit": "vit_h", "ckpt": "SAMed/isic_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_249.pth", "n_class": 1},
}


def load_isic_test_data(n=12):
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
        if gt.sum() < 100:
            continue
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


def attack_pure_l1(encoder, x, eps_pix, step_pix, steps, edge_map_fn=None):
    """Pure L1 feature-space PGD — optimal white-box attack."""
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


# ── Main Experiment ──────────────────────────────────────────────────────────

print("=" * 90)
print("CROSS-MODEL TRANSFER: Validating PAID Φ Claim")
print("Φ (radial dir) is cross-model invariant → Radial transfers better")
print("=" * 90)

SOURCE_CKPT = os.path.join(CKPT_BASE, SAMED_ISIC["ISIC_vit_b"]["ckpt"])
TARGET_CFGS = {
    "ISIC_vit_l": SAMED_ISIC["ISIC_vit_l"],
    "ISIC_vit_h": SAMED_ISIC["ISIC_vit_h"],
}

print("\nLoading source model: ISIC_vit_b...")
src_net = load_samed_model("vit_b", SAM_CKPTS["vit_b"], SOURCE_CKPT, IMAGE_SIZE, 1, device=device)
src_net.eval()
src_encoder = src_net.sam.image_encoder
src_encoder.eval()

target_nets = {}
for tgt_name, tgt_cfg in TARGET_CFGS.items():
    print(f"Loading target model: {tgt_name}...")
    tgt_ckpt_path = os.path.join(CKPT_BASE, tgt_cfg["ckpt"])
    tgt_net = load_samed_model(tgt_cfg["vit"], SAM_CKPTS[tgt_cfg["vit"]],
                                tgt_ckpt_path, IMAGE_SIZE, tgt_cfg["n_class"], device=device)
    tgt_net.eval()
    target_nets[tgt_name] = tgt_net

images = load_isic_test_data(n=12)
print(f"\nLoaded {len(images)} test images")

EPS_LIST = [4.0/255, 8.0/255]
STEPS_LIST = [10, 20, 40]
METHODS = {
    "pure_l1": ("Pure L1 (WHITE-BOX)", lambda enc, x, ep, sp, st: attack_pure_l1(enc, x, ep, sp, st, _get_edge_map)),
    "radial": ("Radial Probe (Φ-DIR)", lambda enc, x, ep, sp, st: attack_radial_optimized(
        enc, x, ep, sp, st, amplitude_alpha=0.0, lambda_amp=0.0, lambda_phase=0.0,
        radial_weight=1.0, use_random_noise=False, edge_map_fn=_get_edge_map)),
    "random": ("Random Probe", lambda enc, x, ep, sp, st: attack_random_probe(enc, x, ep, sp, st, edge_map_fn=_get_edge_map)),
}

all_results = []

for eps_frac in EPS_LIST:
    for n_steps in STEPS_LIST:
        for method_key, (method_label, attack_fn) in METHODS.items():
            src_dices, tgt_l_dices, tgt_h_dices = [], [], []
            clean_src_dices, clean_tgt_l_dices, clean_tgt_h_dices = [], [], []

            for img_t, gt in images:
                x = img_t.to(device)
                rg = x.max().item() - x.min().item()
                eps_pix = eps_frac * rg
                step_pix = eps_pix / max(n_steps, 1)

                clean_src = evaluate_isic(src_net, x, gt)
                if clean_src < 0.01:
                    continue
                clean_src_dices.append(clean_src)
                clean_tgt_l_dices.append(evaluate_isic(target_nets["ISIC_vit_l"], x, gt))
                clean_tgt_h_dices.append(evaluate_isic(target_nets["ISIC_vit_h"], x, gt))

                # Attack source encoder only
                try:
                    x_adv = attack_fn(src_encoder, x, eps_pix, step_pix, n_steps)
                except Exception as e:
                    continue

                src_dices.append(evaluate_isic(src_net, x_adv, gt))
                tgt_l_dices.append(evaluate_isic(target_nets["ISIC_vit_l"], x_adv, gt))
                tgt_h_dices.append(evaluate_isic(target_nets["ISIC_vit_h"], x_adv, gt))

            if len(src_dices) == 0:
                continue

            mc_src = float(np.mean(clean_src_dices))
            mc_l = float(np.mean(clean_tgt_l_dices))
            mc_h = float(np.mean(clean_tgt_h_dices))
            ma_src = float(np.mean(src_dices))
            ma_l = float(np.mean(tgt_l_dices))
            ma_h = float(np.mean(tgt_h_dices))
            d_src = mc_src - ma_src
            d_l = mc_l - ma_l
            d_h = mc_h - ma_h
            tr_l = d_l / (d_src + 1e-8)
            tr_h = d_h / (d_src + 1e-8)

            row = {"eps": f"{eps_frac*255:.0f}/255", "steps": n_steps,
                   "method": method_key, "src_delta": d_src,
                   "vit_l_delta": d_l, "vit_l_transfer": tr_l,
                   "vit_h_delta": d_h, "vit_h_transfer": tr_h}
            all_results.append(row)

            print(f"eps={eps_frac*255:4.0f}/255 steps={n_steps:2d} {method_label:<22} "
                  f"SrcΔ={d_src:+.4f} →L:{d_l:+.4f}({tr_l:.2f}) →H:{d_h:+.4f}({tr_h:.2f})")

# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'='*90}")
print("TRANSFER SUMMARY: Φ-directed (Radial) vs Unconstrained (Pure L1)")
print("Transfer ratio = Δ(target) / Δ(source). Higher = better transfer.")
print("PAID predicts: Radial > Pure L1 for transfer (Φ is shared).")
print(f"{'='*90}")

for eps_str in [f"{e*255:.0f}/255" for e in EPS_LIST]:
    for n_steps in STEPS_LIST:
        print(f"\n── eps={eps_str}, steps={n_steps} ──")
        print(f"  {'Method':<25} {'Src Δ':>8} {'→ vit_l ratio':>14} {'→ vit_h ratio':>14}")
        for method_key in ["pure_l1", "radial", "random"]:
            rows = [r for r in all_results if r["eps"]==eps_str and r["steps"]==n_steps and r["method"]==method_key]
            if rows:
                r = rows[0]
                print(f"  {method_key:<25} {r['src_delta']:>+8.4f} {r['vit_l_transfer']:>14.3f} {r['vit_h_transfer']:>14.3f}")

        # PAID validation check
        p1 = next((r for r in all_results if r["eps"]==eps_str and r["steps"]==n_steps and r["method"]=="pure_l1"), None)
        rad = next((r for r in all_results if r["eps"]==eps_str and r["steps"]==n_steps and r["method"]=="radial"), None)
        if p1 and rad:
            for tgt in ["vit_l", "vit_h"]:
                adv = rad[f"{tgt}_transfer"] - p1[f"{tgt}_transfer"]
                if adv > 0.05:
                    print(f"  >>> PAID VALIDATED: Radial > L1 for {tgt} transfer ({adv:+.3f})")

out_path = os.path.join(OUTPUT_DIR, "transfer_attack.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")

for tgt_net in target_nets.values():
    del tgt_net
del src_net
torch.cuda.empty_cache()
print("Done — transfer experiment complete")
