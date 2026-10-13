#!/usr/bin/env python3
"""
Phase 1: Hyperparameter Tuning for PAID-Optimized Radial Probe Attack.

Sweeps:
  1. α (AE-RP amplitude weighting) ∈ {0, 0.25, 0.5, 0.75, 1.0}
  2. λ_amp (spatial amplitude loss) ∈ {0, 0.01, 0.05, 0.1, 0.2}
  3. λ_phase (direct phase/cosine loss) ∈ {0, 0.001, 0.005, 0.01, 0.05}

Two-stage:
  Stage A: Grid search on ISIC vit_b (best differentiation in smoke test)
  Stage B: Validate best params on 3 models

Models: Synapse SAMed vit_b, ISIC SAMed vit_b, ISIC SAMed vit_l
"""
import os, sys, time, json
import numpy as np
import torch

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_samed_model
from attack_codd import _get_edge_map
from attack_radial_optimized import attack_radial_optimized, attack_bare_radial, attack_random_probe

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0")

MODEL_CONFIGS = {
    "Synapse_vit_b": {
        "vit": "vit_b",
        "sam_ckpt": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
        "ckpt_rel": "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth",
        "num_classes": 8,
        "dataset": "Synapse",
    },
    "ISIC_vit_b": {
        "vit": "vit_b",
        "sam_ckpt": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
        "ckpt_rel": "SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth",
        "num_classes": 1,
        "dataset": "ISIC",
    },
    "ISIC_vit_l": {
        "vit": "vit_l",
        "sam_ckpt": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
        "ckpt_rel": "SAMed/isic_512_lrexp5.0_pretrain_vit_l_epo300_bs12_lr0.005/epoch_59.pth",
        "num_classes": 1,
        "dataset": "ISIC",
    },
}


def load_data(dataset_name, n=8):
    from PIL import Image
    if dataset_name == "Synapse":
        from exp_paid_optimal_attack import load_synapse_slices
        return load_synapse_slices(n)
    else:
        tdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_Data"
        mdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_GroundTruth"
        IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
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
            img_t = (img_t / 255.0 - IM_MEAN) / IM_STD
            imgs.append((img_t, gt))
        return imgs


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
        else:
            probs = torch.softmax(m.float(), dim=1)
            pred_bin = (torch.argmax(probs, dim=1).squeeze(0).cpu().numpy() == 1).astype(np.float32)
            intersection = (pred_bin * gt).sum()
            return float(2.0 * intersection / (pred_bin.sum() + gt.sum() + 1e-8))


# ── Stage A: Grid Search on ISIC_vit_b ─────────────────────────────────────

print("=" * 80)
print("STAGE A: Hyperparameter Grid Search on ISIC SAMed vit_b")
print("=" * 80)

model_name = "ISIC_vit_b"
cfg = MODEL_CONFIGS[model_name]
print(f"Loading {model_name}...")
net = load_samed_model(cfg["vit"], cfg["sam_ckpt"],
                       os.path.join(CKPT_BASE, cfg["ckpt_rel"]),
                       IMAGE_SIZE, cfg["num_classes"], device=device)
net.eval()
image_encoder = net.sam.image_encoder
image_encoder.eval()

images = load_data(cfg["dataset"], n=8)
print(f"Loaded {len(images)} images\n")

eps_frac = 8.0 / 255.0
step_frac = 0.8 / 255.0
n_steps = 20

all_sweeps = []

# ── Sweep 1: α (AE-RP magnitude) ──
print("─" * 60)
print("Sweep 1: AE-RP α (amplitude weighting, magnitude mode)")
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
alpha_results = []
for alpha in alphas:
    dices, times = [], []
    for img_idx, (img_t, gt) in enumerate(images):
        x = img_t.to(device)
        rg = x.max().item() - x.min().item()
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps_frac * rg, step_frac * rg, n_steps,
            amplitude_alpha=alpha, weight_mode="magnitude",
            lambda_amp=0.0, lambda_phase=0.0,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, cfg["dataset"])
        dices.append(adv_dice)
        times.append(time.time() - t0)
    md, mt = float(np.mean(dices)), float(np.mean(times))
    print(f"  α={alpha:.2f}: Dice={md:.4f}  ({mt:.1f}s/img)")
    alpha_results.append({"alpha": alpha, "mean_dice": md, "mean_time": mt})
all_sweeps.append({"sweep": "alpha_magnitude", "results": alpha_results})

best_alpha = min(alpha_results, key=lambda r: r["mean_dice"])["alpha"]
print(f"  → Best α={best_alpha:.2f}")

# ── Sweep 2: α (AE-RP edge mode) ──
print("\n" + "─" * 60)
print("Sweep 2: AE-RP α (edge-weighted mode)")
edge_results = []
for alpha in alphas:
    dices, times = [], []
    for img_idx, (img_t, gt) in enumerate(images):
        x = img_t.to(device)
        rg = x.max().item() - x.min().item()
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps_frac * rg, step_frac * rg, n_steps,
            amplitude_alpha=alpha, weight_mode="edge",
            lambda_amp=0.0, lambda_phase=0.0,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, cfg["dataset"])
        dices.append(adv_dice)
        times.append(time.time() - t0)
    md, mt = float(np.mean(dices)), float(np.mean(times))
    print(f"  α_edge={alpha:.2f}: Dice={md:.4f}  ({mt:.1f}s/img)")
    edge_results.append({"alpha": alpha, "mean_dice": md, "mean_time": mt})
all_sweeps.append({"sweep": "alpha_edge", "results": edge_results})

# ── Sweep 3: λ_amp (spatial amplitude loss) ──
print("\n" + "─" * 60)
print("Sweep 3: λ_amp (spatial amplitude loss) with best α")
lambda_amps = [0.0, 0.01, 0.05, 0.1, 0.2]
amp_results = []
for lam in lambda_amps:
    dices, times = [], []
    for img_idx, (img_t, gt) in enumerate(images):
        x = img_t.to(device)
        rg = x.max().item() - x.min().item()
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps_frac * rg, step_frac * rg, n_steps,
            amplitude_alpha=best_alpha, weight_mode="magnitude",
            lambda_amp=lam, amp_loss_mode="spatial",
            lambda_phase=0.0,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, cfg["dataset"])
        dices.append(adv_dice)
        times.append(time.time() - t0)
    md, mt = float(np.mean(dices)), float(np.mean(times))
    print(f"  λ_amp={lam:.3f}: Dice={md:.4f}  ({mt:.1f}s/img)")
    amp_results.append({"lambda_amp": lam, "mean_dice": md, "mean_time": mt})
all_sweeps.append({"sweep": "lambda_amp_spatial", "results": amp_results})

best_lambda_amp = min(amp_results, key=lambda r: r["mean_dice"])["lambda_amp"]
print(f"  → Best λ_amp={best_lambda_amp:.3f}")

# ── Sweep 4: λ_phase (direct phase/cosine loss) ──
print("\n" + "─" * 60)
print("Sweep 4: λ_phase (direct phase/cosine loss) with best α")
lambda_phases = [0.0, 0.001, 0.005, 0.01, 0.05]
phase_results = []
for lam in lambda_phases:
    dices, times = [], []
    for img_idx, (img_t, gt) in enumerate(images):
        x = img_t.to(device)
        rg = x.max().item() - x.min().item()
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps_frac * rg, step_frac * rg, n_steps,
            amplitude_alpha=best_alpha, weight_mode="magnitude",
            lambda_amp=0.0, amp_loss_mode="spatial",
            lambda_phase=lam,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, cfg["dataset"])
        dices.append(adv_dice)
        times.append(time.time() - t0)
    md, mt = float(np.mean(dices)), float(np.mean(times))
    print(f"  λ_phase={lam:.4f}: Dice={md:.4f}  ({mt:.1f}s/img)")
    phase_results.append({"lambda_phase": lam, "mean_dice": md, "mean_time": mt})
all_sweeps.append({"sweep": "lambda_phase", "results": phase_results})

best_lambda_phase = min(phase_results, key=lambda r: r["mean_dice"])["lambda_phase"]
print(f"  → Best λ_phase={best_lambda_phase:.4f}")

# ── Sweep 5: Combined best params ──
print("\n" + "─" * 60)
print("Sweep 5: Combined best params vs baselines")
combos = {
    "Bare Radial (α=0, no aux)": {"amplitude_alpha": 0.0, "lambda_amp": 0.0, "lambda_phase": 0.0},
    "Best AE-RP only": {"amplitude_alpha": best_alpha, "lambda_amp": 0.0, "lambda_phase": 0.0},
    "Best AE-RP + Amp": {"amplitude_alpha": best_alpha, "lambda_amp": best_lambda_amp, "lambda_phase": 0.0},
    "Best AE-RP + Amp + Phase": {"amplitude_alpha": best_alpha, "lambda_amp": best_lambda_amp, "lambda_phase": best_lambda_phase},
    "Random Probe": {"amplitude_alpha": 0.0, "lambda_amp": 0.0, "lambda_phase": 0.0, "radial_weight": 0.0, "use_random_noise": True},
}
combo_results = []
for name, params in combos.items():
    dices, times = [], []
    for img_idx, (img_t, gt) in enumerate(images):
        x = img_t.to(device)
        rg = x.max().item() - x.min().item()
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps_frac * rg, step_frac * rg, n_steps,
            amplitude_alpha=params.get("amplitude_alpha", 0.0),
            weight_mode="magnitude",
            lambda_amp=params.get("lambda_amp", 0.0),
            amp_loss_mode="spatial",
            lambda_phase=params.get("lambda_phase", 0.0),
            prior_weight=0.0,
            radial_weight=params.get("radial_weight", 1.0),
            use_random_noise=params.get("use_random_noise", False),
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, cfg["dataset"])
        dices.append(adv_dice)
        times.append(time.time() - t0)
    md, mt = float(np.mean(dices)), float(np.mean(times))
    print(f"  {name:<30}: Dice={md:.4f}  ({mt:.1f}s/img)")
    combo_results.append({"name": name, "mean_dice": md, "mean_time": mt})

all_sweeps.append({"sweep": "best_combos", "results": combo_results})

BEST_PARAMS = {"alpha": best_alpha, "lambda_amp": best_lambda_amp, "lambda_phase": best_lambda_phase}
print(f"\n{'='*60}")
print(f"BEST PARAMS on ISIC vit_b: α={best_alpha:.2f}, λ_amp={best_lambda_amp:.3f}, λ_phase={best_lambda_phase:.4f}")

del net
torch.cuda.empty_cache()

# ── Stage B: Validate on 3 models ─────────────────────────────────────────

print(f"\n{'='*80}")
print("STAGE B: Validate Best Params on 3 Models")
print("=" * 80)

validation_results = []

for model_name in MODEL_CONFIGS:
    cfg = MODEL_CONFIGS[model_name]
    print(f"\n{'─'*60}")
    print(f"  {model_name}")
    print(f"{'─'*60}")

    net = load_samed_model(cfg["vit"], cfg["sam_ckpt"],
                           os.path.join(CKPT_BASE, cfg["ckpt_rel"]),
                           IMAGE_SIZE, cfg["num_classes"], device=device)
    net.eval()
    image_encoder = net.sam.image_encoder
    image_encoder.eval()
    images = load_data(cfg["dataset"], n=8)
    print(f"  Loaded {len(images)} images")

    conditions = [
        ("Bare Radial", {"amplitude_alpha": 0.0, "lambda_amp": 0.0, "lambda_phase": 0.0}),
        ("Best AE-RP", {"amplitude_alpha": best_alpha, "lambda_amp": 0.0, "lambda_phase": 0.0}),
        ("Best Full", {"amplitude_alpha": best_alpha, "lambda_amp": best_lambda_amp, "lambda_phase": best_lambda_phase}),
        ("Random Probe", {"amplitude_alpha": 0.0, "lambda_amp": 0.0, "lambda_phase": 0.0, "radial_weight": 0.0, "use_random_noise": True}),
    ]

    for cond_name, params in conditions:
        dices, clean_dices, times = [], [], []
        for img_idx, (img_t, gt) in enumerate(images):
            if cfg["dataset"] == "Synapse":
                x = torch.from_numpy(img_t).unsqueeze(0).float().to(device)
                gt_arr = gt
            else:
                x = img_t.to(device)
                gt_arr = gt
            rg = x.max().item() - x.min().item()

            clean_dice = evaluate_model(net, x, gt_arr, cfg["dataset"])
            clean_dices.append(clean_dice)

            t0 = time.time()
            x_adv = attack_radial_optimized(
                image_encoder, x, eps_frac * rg, step_frac * rg, n_steps,
                amplitude_alpha=params.get("amplitude_alpha", 0.0),
                weight_mode="magnitude",
                lambda_amp=params.get("lambda_amp", 0.0),
                amp_loss_mode="spatial",
                lambda_phase=params.get("lambda_phase", 0.0),
                prior_weight=0.0,
                radial_weight=params.get("radial_weight", 1.0),
                use_random_noise=params.get("use_random_noise", False),
                edge_map_fn=_get_edge_map,
            )
            adv_dice = evaluate_model(net, x_adv, gt_arr, cfg["dataset"])
            dices.append(adv_dice)
            times.append(time.time() - t0)

        md_clean = float(np.mean(clean_dices))
        md = float(np.mean(dices))
        md_delta = float(np.mean([c - a for c, a in zip(clean_dices, dices)]))
        mt = float(np.mean(times))
        print(f"  {cond_name:<20} Clean={md_clean:.4f}  Adv={md:.4f}  Δ={md_delta:+.4f}  {mt:.1f}s/img")
        validation_results.append({
            "model": model_name, "condition": cond_name,
            "mean_clean_dice": md_clean, "mean_adv_dice": md,
            "mean_delta": md_delta, "mean_time": mt,
        })

    del net
    torch.cuda.empty_cache()

# ── Save ──────────────────────────────────────────────────────────────────

out = {
    "best_params": BEST_PARAMS,
    "stage_a_sweeps": all_sweeps,
    "stage_b_validation": validation_results,
}
out_path = os.path.join(OUTPUT_DIR, "hyperparameter_tuning.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")
print("Done — hyperparameter tuning complete")
