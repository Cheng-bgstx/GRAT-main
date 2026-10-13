#!/usr/bin/env python3
"""
Pure L1 Feature-Space Attack — The Optimal Method.

Key finding: Pure PGD on L1(f_adv, f_clean) beats ALL probe-based methods.
The probe alignment is unnecessary and counterproductive.

This experiment validates Pure L1 across all 9 SAMed models.
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
from attack_radial_optimized import attack_bare_radial, attack_random_probe

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


def load_data(dataset_name, n=12):
    if dataset_name == "Synapse":
        from exp_paid_optimal_attack import load_synapse_slices
        return load_synapse_slices(n)
    elif dataset_name == "ISIC":
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
    elif dataset_name == "REFUGE":
        from exp_paid_optimal_attack import load_refuge_images
        return load_refuge_images(n)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate_model(net, img_t, gt, dataset_name):
    with torch.no_grad():
        out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        m = out['masks']

    if dataset_name == "Synapse":
        pred = torch.argmax(torch.softmax(m.float(), dim=1), dim=1).squeeze(0).cpu().numpy()
        dices = []
        for c in range(1, 8):
            pred_c = (pred == c).astype(np.float32)
            gt_c = (gt == c).astype(np.float32)
            if gt_c.sum() == 0: continue
            inter = (pred_c * gt_c).sum()
            dices.append(float(2.0 * inter / (pred_c.sum() + gt_c.sum() + 1e-8)))
        return float(np.mean(dices)) if dices else 0.0

    elif dataset_name == "ISIC":
        probs = torch.softmax(m.float(), dim=1)
        lesion_prob = probs[0, 1, :, :].cpu().numpy()
        best = 0.0
        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            pred_bin = (lesion_prob > thresh).astype(np.float32)
            inter = (pred_bin * gt).sum()
            dice = float(2.0 * inter / (pred_bin.sum() + gt.sum() + 1e-8))
            if dice > best: best = dice
        return best

    elif dataset_name == "REFUGE":
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


def attack_pure_l1(encoder, x, eps_pix, step_pix, steps, edge_map_fn=None):
    """Pure L1 feature-space PGD — the optimal attack."""
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

EPS_LIST = [4.0/255, 8.0/255, 16.0/255]
STEPS_LIST = [10, 20, 40]
N_IMAGES = 12

METHODS = {
    "pure_l1": ("Pure L1", lambda enc, x, ep, sp, st: attack_pure_l1(enc, x, ep, sp, st, _get_edge_map)),
    "bare_radial": ("Bare Radial", lambda enc, x, ep, sp, st: attack_bare_radial(enc, x, ep, sp, st, edge_map_fn=_get_edge_map)),
    "random": ("Random Probe", lambda enc, x, ep, sp, st: attack_random_probe(enc, x, ep, sp, st, edge_map_fn=_get_edge_map)),
}

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
prior_path = os.path.join(OUTPUT_DIR, "pure_l1_validation.json")
all_results = []
if os.path.exists(prior_path):
    with open(prior_path) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} prior results")

completed_models = set(r["model"] for r in all_results)

print("=" * 90)
print("PURE L1 FEATURE-SPACE ATTACK: The Optimal Method")
print("=" * 90)

for model_key in sorted(SAMED_MODELS.keys()):
    cfg = SAMED_MODELS[model_key]
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

    images = load_data(ds, n=N_IMAGES)
    print(f"  Loaded {len(images)} images")

    for eps_frac in EPS_LIST:
        for n_steps in STEPS_LIST:

            for method_key, (method_label, attack_fn) in METHODS.items():
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

                    clean_dice = evaluate_model(net, x, gt, ds)
                    if clean_dice < 0.01:
                        continue

                    try:
                        t0 = time.time()
                        x_adv = attack_fn(encoder, x, eps_pix, step_pix, n_steps)
                        adv_dice = evaluate_model(net, x_adv, gt, ds)
                        times.append(time.time() - t0)
                        dices.append(adv_dice)
                        clean_dices.append(clean_dice)
                    except Exception as e:
                        continue

                if len(dices) == 0:
                    continue

                md_clean = float(np.mean(clean_dices))
                md_adv = float(np.mean(dices))
                md_delta = float(np.mean([c - a for c, a in zip(clean_dices, dices)]))
                mt = float(np.mean(times))

                all_results.append({
                    "model": model_key, "dataset": ds, "vit": cfg["vit"],
                    "eps": f"{eps_frac*255:.0f}/255", "steps": n_steps,
                    "method": method_key, "method_label": method_label,
                    "mean_clean": md_clean, "mean_adv": md_adv,
                    "mean_delta": md_delta, "mean_time": mt, "n_valid": len(dices),
                })

                print(f"  eps={eps_frac*255:4.0f}/255 steps={n_steps:2d} {method_label:<15} "
                      f"Clean={md_clean:.4f} Adv={md_adv:.4f} Δ={md_delta:+.4f} ({mt:.1f}s)")

    del net
    torch.cuda.empty_cache()
    gc.collect()
    # Save incrementally
    with open(prior_path, "w") as f:
        json.dump(all_results, f, indent=2)

# ── Summary: Pure L1 vs Baselines ────────────────────────────────────────────

print(f"\n{'='*90}")
print("SUMMARY: Pure L1 Advantage Over Baselines")
print("Δ(Pure L1) - Δ(Baseline) — positive = Pure L1 better")
print(f"{'='*90}")

for eps_str in [f"{e*255:.0f}/255" for e in EPS_LIST]:
    for steps in STEPS_LIST:
        print(f"\n── eps={eps_str}, steps={steps} ──")
        print(f"  {'Model':<14} {'Pure L1 Δ':>12} {'vs Radial':>12} {'vs Random':>12}")

        pure_deltas = {}
        for r in all_results:
            if r["eps"] == eps_str and r["steps"] == steps:
                if r["method"] == "pure_l1":
                    pure_deltas[r["model"]] = r["mean_delta"]

        for model_key in sorted(set(r["model"] for r in all_results)):
            if model_key not in pure_deltas:
                continue
            pdl = pure_deltas[model_key]

            rdl = next((r["mean_delta"] for r in all_results
                       if r["model"]==model_key and r["eps"]==eps_str
                       and r["steps"]==steps and r["method"]=="bare_radial"), None)
            rnd = next((r["mean_delta"] for r in all_results
                       if r["model"]==model_key and r["eps"]==eps_str
                       and r["steps"]==steps and r["method"]=="random"), None)

            vs_r = f"{pdl-rdl:+.4f}" if rdl is not None else "N/A"
            vs_n = f"{pdl-rnd:+.4f}" if rnd is not None else "N/A"
            print(f"  {model_key:<14} {pdl:>+12.4f} {vs_r:>12} {vs_n:>12}")

# ── Save ─────────────────────────────────────────────────────────────────────

out_path = os.path.join(OUTPUT_DIR, "pure_l1_validation.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
print("Done — Pure L1 validation complete")
