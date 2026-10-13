#!/usr/bin/env python3
"""
Same-Backbone Transfer: Validates PAID Φ Invariance Within Same ViT.

PAID Theory (refined): Φ (radial direction) is cross-model invariant
WITHIN the same ViT backbone. Different fine-tuning methods (SAMed, MSA)
on the same ViT should share Φ structure.

Experiment:
  Source: ISIC SAMed vit_b → Target: ISIC MSA vit_b (SAME ViT backbone)
  Also test: ISIC SAMed vit_l → ISIC MSA vit_l

Compare transfer ratios:
  Cross-backbone (vit_b → vit_l): ~0% (confirmed)
  Same-backbone (SAMed vit_b → MSA vit_b): predicted >0% if Φ is shared
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

from evaluate import load_samed_model, load_medsa_model
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

# Source (SAMed) and Target (MSA) pairs — same ViT backbone
TRANSFER_PAIRS = [
    {
        "name": "ISIC vit_b SAMed→MSA",
        "src_vit": "vit_b", "src_ckpt": "SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth",
        "src_n_class": 1, "src_type": "samed",
        "tgt_vit": "vit_b", "tgt_ckpt_dir": "MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24",
        "tgt_mm_output": 9, "tgt_type": "msa",
    },
]


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
        if gt.sum() < 100: continue
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        img_t = (img_t / 255.0 - IM_MEAN_CPU) / IM_STD_CPU
        imgs.append((img_t, gt))
    return imgs


def evaluate_isic(net, img_t, gt, model_type="samed"):
    with torch.no_grad():
        if model_type == "samed":
            out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        else:
            out = net(img_t, image_size=IMAGE_SIZE)
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
print("SAME-BACKBONE TRANSFER: Testing Refined PAID Φ Invariance")
print("Φ shared within same ViT backbone → transfer should be > 0")
print("=" * 90)

EPS_LIST = [4.0/255, 8.0/255]
STEPS_LIST = [10, 20, 40]

all_results = []

for pair in TRANSFER_PAIRS:
    print(f"\n{'='*80}")
    print(f"  {pair['name']}")
    print(f"{'='*80}")

    # Load source (SAMed)
    print(f"Loading source: SAMed {pair['src_vit']}...")
    src_net = load_samed_model(pair["src_vit"], SAM_CKPTS[pair["src_vit"]],
                               os.path.join(CKPT_BASE, pair["src_ckpt"]),
                               IMAGE_SIZE, pair["src_n_class"], device=device)
    src_net.eval()
    src_encoder = src_net.sam.image_encoder
    src_encoder.eval()

    # Load target (MSA)
    print(f"Loading target: MSA {pair['tgt_vit']}...")
    tgt_msa_dir = os.path.join(CKPT_BASE, pair["tgt_ckpt_dir"])
    tgt_ckpt_path = os.path.join(tgt_msa_dir, "Model", "best_dice_checkpoint.pth")
    tgt_net = load_medsa_model(pair["tgt_vit"], SAM_CKPTS[pair["tgt_vit"]],
                               tgt_ckpt_path, IMAGE_SIZE,
                               multimask_output=pair["tgt_mm_output"], device=device)
    tgt_net.eval()

    images = load_isic_test_data(n=12)
    print(f"Loaded {len(images)} test images")

    for eps_frac in EPS_LIST:
        for n_steps in STEPS_LIST:
            src_dices, tgt_dices, clean_src_dices, clean_tgt_dices = [], [], [], []

            for img_t, gt in images:
                x = img_t.to(device)
                rg = x.max().item() - x.min().item()
                eps_pix = eps_frac * rg
                step_pix = eps_pix / max(n_steps, 1)

                # Clean evaluation
                cd_src = evaluate_isic(src_net, x, gt, "samed")
                cd_tgt = evaluate_isic(tgt_net, x, gt, "msa")
                if cd_src < 0.01: continue
                clean_src_dices.append(cd_src)
                clean_tgt_dices.append(cd_tgt)

                # Attack source encoder
                try:
                    x_adv = attack_pure_l1(src_encoder, x, eps_pix, step_pix, n_steps, _get_edge_map)
                except Exception as e:
                    continue

                src_dices.append(evaluate_isic(src_net, x_adv, gt, "samed"))
                tgt_dices.append(evaluate_isic(tgt_net, x_adv, gt, "msa"))

            if len(src_dices) == 0: continue

            mc_src = float(np.mean(clean_src_dices))
            mc_tgt = float(np.mean(clean_tgt_dices))
            ma_src = float(np.mean(src_dices))
            ma_tgt = float(np.mean(tgt_dices))
            d_src = mc_src - ma_src
            d_tgt = mc_tgt - ma_tgt
            transfer_ratio = d_tgt / (d_src + 1e-8)

            all_results.append({
                "pair": pair["name"], "eps": f"{eps_frac*255:.0f}/255", "steps": n_steps,
                "src_clean": mc_src, "src_adv": ma_src, "src_delta": d_src,
                "tgt_clean": mc_tgt, "tgt_adv": ma_tgt, "tgt_delta": d_tgt,
                "transfer_ratio": transfer_ratio,
            })

            print(f"  eps={eps_frac*255:4.0f}/255 steps={n_steps:2d} "
                  f"Src Δ={d_src:+.4f} Tgt Δ={d_tgt:+.4f} Transfer={transfer_ratio:.3f} "
                  f"{'*** Φ TRANSFER!' if transfer_ratio > 0.05 else ''}")

    del src_net, tgt_net
    torch.cuda.empty_cache()

# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'='*90}")
print("SAME-BACKBONE TRANSFER SUMMARY")
print(f"{'='*90}")

for eps_str in [f"{e*255:.0f}/255" for e in EPS_LIST]:
    for n_steps in STEPS_LIST:
        for pair_name in [p["name"] for p in TRANSFER_PAIRS]:
            rows = [r for r in all_results if r["eps"]==eps_str and r["steps"]==n_steps and r["pair"]==pair_name]
            if rows:
                r = rows[0]
                verdict = "Φ SHARED ✓" if r["transfer_ratio"] > 0.05 else "Φ NOT SHARED ✗"
                print(f"  eps={eps_str} steps={n_steps:2d} {pair_name}: "
                      f"ratio={r['transfer_ratio']:.3f} → {verdict}")

out_path = os.path.join(OUTPUT_DIR, "same_backbone_transfer.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
print("Done — same-backbone transfer complete")
