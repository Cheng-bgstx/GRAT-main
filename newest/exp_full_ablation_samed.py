#!/usr/bin/env python3
"""
Phase 3: Full Component Ablation Matrix — All SAMed Models.

Compares across 9 SAMed models:
  0. Clean (no attack)
  1. Random Probe (IFFT noise baseline)
  2. Bare Radial (α=0, no aux losses)
  3. + Amp Loss (spatial λ_amp=0.1) — key improvement
  4. Full Optimized (α=1.0, λ_amp=0.1, λ_phase=0.05)
  5. SE-CGR (previous best, where available)

Models: {Synapse, ISIC, REFUGE} × {vit_b, vit_l, vit_h}
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

from evaluate import load_samed_model, load_medsa_model
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

# ── Model Registry ──────────────────────────────────────────────────────────

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

# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data(dataset_name, n=16):
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
        from exp_paid_optimal_attack import load_refuge_test_data
        return load_refuge_test_data(n)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(net, img_t, gt, dataset_name):
    with torch.no_grad():
        out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        m = out['masks']

    if dataset_name == "Synapse":
        pred = torch.argmax(torch.softmax(m.float(), dim=1), dim=1).squeeze(0).cpu().numpy()
        n_classes = 8
        dices = []
        for c in range(1, n_classes):
            pred_c = (pred == c).astype(np.float32)
            gt_c = (gt == c).astype(np.float32)
            if gt_c.sum() == 0:
                continue
            inter = (pred_c * gt_c).sum()
            dices.append(float(2.0 * inter / (pred_c.sum() + gt_c.sum() + 1e-8)))
        return float(np.mean(dices)) if dices else 0.0

    elif dataset_name == "ISIC":
        probs = torch.softmax(m.float(), dim=1)
        # Try multiple thresholds for best Dice
        lesion_prob = probs[0, 1, :, :].cpu().numpy()
        best = 0.0
        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            pred_bin = (lesion_prob > thresh).astype(np.float32)
            inter = (pred_bin * gt).sum()
            dice = float(2.0 * inter / (pred_bin.sum() + gt.sum() + 1e-8))
            if dice > best:
                best = dice
        return best

    elif dataset_name == "REFUGE":
        # REFUGE has 3 output channels: bg, disc, cup. gt = [cup_mask, disc_mask]
        probs = torch.softmax(m.float(), dim=1)  # [1, 3, H, W]
        cup_gt, disc_gt = gt[0], gt[1]
        dices = []
        for cls_idx, gt_mask in [(2, cup_gt), (1, disc_gt)]:
            cls_prob = probs[0, cls_idx, :, :].cpu().numpy()
            best = 0.0
            for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
                pred_bin = (cls_prob > thresh).astype(np.float32)
                inter = (pred_bin * gt_mask).sum()
                dice = float(2.0 * inter / (pred_bin.sum() + gt_mask.sum() + 1e-8))
                if dice > best:
                    best = dice
            dices.append(best)
        return float(np.mean(dices))


# ── Attack Helpers ───────────────────────────────────────────────────────────

def run_attack(image_encoder, x, eps_pix, step_pix, steps, method, **kwargs):
    t0 = time.time()
    if method == "random":
        x_adv = attack_random_probe(image_encoder, x, eps_pix, step_pix, steps,
                                     edge_map_fn=_get_edge_map, **kwargs)
    elif method == "bare_radial":
        x_adv = attack_radial_optimized(
            image_encoder, x, eps_pix, step_pix, steps,
            amplitude_alpha=0.0, weight_mode="magnitude",
            lambda_amp=0.0, lambda_phase=0.0,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map, **kwargs)
    elif method == "amp_loss":
        x_adv = attack_radial_optimized(
            image_encoder, x, eps_pix, step_pix, steps,
            amplitude_alpha=0.0, weight_mode="magnitude",
            lambda_amp=0.1, amp_loss_mode="spatial",
            lambda_phase=0.0,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map, **kwargs)
    elif method == "full_optimized":
        x_adv = attack_radial_optimized(
            image_encoder, x, eps_pix, step_pix, steps,
            amplitude_alpha=1.0, weight_mode="magnitude",
            lambda_amp=0.1, amp_loss_mode="spatial",
            lambda_phase=0.05,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map, **kwargs)
    elif method == "secgr":
        x_adv = attack_secgr(image_encoder, x, eps_pix, step_pix, steps,
                              edge_map_fn=_get_edge_map, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    return x_adv, time.time() - t0


def attack_secgr(image_encoder, x, eps_pix, step_pix, steps, edge_map_fn=None, **kwargs):
    """SE-CGR baseline using CODD theory probes."""
    from theory import synthesize_codd_probe
    criterion_l1 = torch.nn.L1Loss()
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = image_encoder(x)
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = edge_map_fn(x).detach() if edge_map_fn else torch.zeros_like(x)
    momentum = torch.zeros_like(x)
    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)
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
        im = torch.stack([t.abs().mean() for t in selected_t])
        im = im.view(-1, 1, 1, 1)
        loss = loss_org + torch.abs(torch.mean(fd_stack/(im+1e-6), dim=0)).mean()
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


# ── Main Experiment ──────────────────────────────────────────────────────────

EPS_FRAC = 8.0 / 255.0
STEP_FRAC = 0.8 / 255.0
N_STEPS = 40  # 40 steps for reliable results
N_IMAGES = 16

METHODS = ["random", "bare_radial", "amp_loss", "full_optimized", "secgr"]
METHOD_LABELS = {
    "random": "Random Probe",
    "bare_radial": "Bare Radial",
    "amp_loss": "+ Amp Loss",
    "full_optimized": "Full Optimized",
    "secgr": "SE-CGR",
}

all_results = []

print("=" * 90)
print("FULL ABLATION MATRIX: 9 SAMed Models x 5 Conditions")
print(f"eps={EPS_FRAC*255:.0f}/255, {N_STEPS} steps, {N_IMAGES} images per model")
print("=" * 90)

for model_key in sorted(SAMED_MODELS.keys()):
    cfg = SAMED_MODELS[model_key]
    ds = cfg["ds"]
    print(f"\n{'='*80}")
    print(f"  {model_key}  |  {cfg['vit']}  |  {ds}  |  {cfg['n_class']} classes")
    print(f"{'='*80}")

    # Load model
    try:
        net = load_samed_model(cfg["vit"], SAM_CKPTS[cfg["vit"]],
                               os.path.join(CKPT_BASE, cfg["ckpt"]),
                               IMAGE_SIZE, cfg["n_class"], device=device)
        net.eval()
        image_encoder = net.sam.image_encoder
        image_encoder.eval()
    except Exception as e:
        print(f"  SKIP: Failed to load model: {e}")
        continue

    # Load data
    images = load_data(ds, n=N_IMAGES)
    print(f"  Loaded {len(images)} images")

    if len(images) == 0:
        print("  SKIP: No valid images")
        del net
        torch.cuda.empty_cache()
        continue

    # Run conditions
    for method in METHODS:
        dices, clean_dices, times = [], [], []

        for img_idx, img_data in enumerate(images):
            if ds == "Synapse":
                x = torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
                gt = img_data[1]
            elif ds == "ISIC":
                x = img_data[0].to(device)
                gt = img_data[1]
            elif ds == "REFUGE":
                x = img_data[0].to(device) if isinstance(img_data[0], torch.Tensor) else torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
                gt = img_data[1]
            else:
                raise ValueError(f"Unknown dataset: {ds}")

            rg = x.max().item() - x.min().item()
            eps_pix = EPS_FRAC * rg
            step_pix = STEP_FRAC * rg

            # Clean dice
            clean_dice = evaluate_model(net, x, gt, ds)
            if clean_dice < 0.01:
                continue

            try:
                x_adv, dt = run_attack(image_encoder, x, eps_pix, step_pix, N_STEPS, method)
                adv_dice = evaluate_model(net, x_adv, gt, ds)
            except Exception as e:
                print(f"    [{img_idx}] {METHOD_LABELS[method]}: ERROR - {e}")
                continue

            dices.append(adv_dice)
            clean_dices.append(clean_dice)
            times.append(dt)

            if img_idx < 2 or img_idx == len(images) - 1:
                print(f"    [{img_idx}] {METHOD_LABELS[method]:<18} Clean={clean_dice:.4f}  Adv={adv_dice:.4f}  Δ={clean_dice-adv_dice:+.4f}")

        if len(dices) == 0:
            continue

        md_clean = float(np.mean(clean_dices))
        md_adv = float(np.mean(dices))
        md_delta = float(np.mean([c - a for c, a in zip(clean_dices, dices)]))
        sd_delta = float(np.std([c - a for c, a in zip(clean_dices, dices)]))
        mt = float(np.mean(times))

        result = {
            "model": model_key, "dataset": ds, "vit": cfg["vit"],
            "method": method, "method_label": METHOD_LABELS[method],
            "mean_clean_dice": md_clean, "mean_adv_dice": md_adv,
            "mean_delta": md_delta, "std_delta": sd_delta,
            "mean_time": mt, "n_valid": len(dices),
            "per_image_deltas": [float(c - a) for c, a in zip(clean_dices, dices)],
        }
        all_results.append(result)

        print(f"    {'─'*50}")
        print(f"    SUMMARY {METHOD_LABELS[method]:<18}: Clean={md_clean:.4f}  Adv={md_adv:.4f}  "
              f"Δ={md_delta:+.4f}±{sd_delta:.4f}  ({mt:.1f}s/img, n={len(dices)})")

    del net
    torch.cuda.empty_cache()
    gc.collect()

# ── Summary Tables ───────────────────────────────────────────────────────────

print(f"\n{'='*90}")
print("ABLATION SUMMARY: All Models")
print(f"{'='*90}")

for ds_name in ["Synapse", "ISIC", "REFUGE"]:
    print(f"\n── {ds_name} ──")
    ds_results = [r for r in all_results if r["dataset"] == ds_name]
    models_in_ds = sorted(set(r["model"] for r in ds_results))

    # Header
    header = f"  {'Model':<14}"
    for m in METHODS:
        header += f"  {METHOD_LABELS[m]:>16}"
    header += f"  {'Best Method':>20}"
    print(header)
    print(f"  {'─'*len(header)}")

    for model_key in models_in_ds:
        row = f"  {model_key:<14}"
        best_delta = -999
        best_method = ""
        for method in METHODS:
            r = [r for r in ds_results if r["model"] == model_key and r["method"] == method]
            if r:
                delta = r[0]["mean_delta"]
                row += f"  {delta:>+14.4f}"
                if delta > best_delta:
                    best_delta = delta
                    best_method = r[0]["method_label"]
            else:
                row += f"  {'N/A':>16}"
        row += f"  {best_method:>20}"
        print(row)

    # Averages
    avg_row = f"  {'AVERAGE':<14}"
    for method in METHODS:
        deltas = [r["mean_delta"] for r in ds_results if r["method"] == method]
        if deltas:
            avg_row += f"  {np.mean(deltas):>+14.4f}"
        else:
            avg_row += f"  {'N/A':>16}"
    print(f"  {'─'*len(header)}")
    print(avg_row)

# ── Save ─────────────────────────────────────────────────────────────────────

out_path = os.path.join(OUTPUT_DIR, "full_ablation_samed.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
print("Done — full SAMed ablation complete")
