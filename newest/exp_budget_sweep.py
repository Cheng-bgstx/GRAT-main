#!/usr/bin/env python3
"""
Budget Sweep: Find the regime where Radial Probe dominates Random Probe.

Tests:
  Steps ∈ {5, 10, 20, 40}
  Eps ∈ {2/255, 4/255, 8/255, 16/255}
  Methods: Random Probe, Bare Radial, + Amp Loss (λ=0.1), Full Optimized

Hypothesis: Radial probe is more sample-efficient — dominates at low steps/eps.
At high budgets, all methods converge similarly.

Models: Synapse vit_b, ISIC vit_b, ISIC vit_l
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
from attack_radial_optimized import attack_radial_optimized, attack_random_probe

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
    ("Syn_vit_b", "vit_b", "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 8, "Synapse"),
    ("ISIC_vit_b", "vit_b", "SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 1, "ISIC"),
    ("ISIC_vit_l", "vit_l", "SAMed/isic_512_lrexp5.0_pretrain_vit_l_epo300_bs12_lr0.005/epoch_59.pth", 1, "ISIC"),
]

EPS_LIST = [2.0/255, 4.0/255, 8.0/255, 16.0/255]
STEPS_LIST = [5, 10, 20, 40]
N_IMAGES = 12

METHODS = {
    "random": {"amplitude_alpha": 0.0, "lambda_amp": 0.0, "lambda_phase": 0.0,
               "radial_weight": 0.0, "use_random_noise": True},
    "bare_radial": {"amplitude_alpha": 0.0, "lambda_amp": 0.0, "lambda_phase": 0.0,
                    "radial_weight": 1.0, "use_random_noise": False},
    "amp_loss": {"amplitude_alpha": 0.0, "lambda_amp": 0.1, "lambda_phase": 0.0,
                 "radial_weight": 1.0, "use_random_noise": False, "amp_loss_mode": "spatial"},
    "full_optimized": {"amplitude_alpha": 1.0, "lambda_amp": 0.1, "lambda_phase": 0.05,
                       "radial_weight": 1.0, "use_random_noise": False, "amp_loss_mode": "spatial"},
}
METHOD_LABELS = {"random": "Random Probe", "bare_radial": "Bare Radial",
                  "amp_loss": "+ Amp Loss", "full_optimized": "Full Optimized"}


def load_data(dataset_name, n=12):
    from PIL import Image
    import numpy as np
    if dataset_name == "Synapse":
        from exp_paid_optimal_attack import load_synapse_slices
        return load_synapse_slices(n)
    else:
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
            if gt_c.sum() == 0:
                continue
            inter = (pred_c * gt_c).sum()
            dices.append(float(2.0 * inter / (pred_c.sum() + gt_c.sum() + 1e-8)))
        return float(np.mean(dices)) if dices else 0.0
    else:
        probs = torch.softmax(m.float(), dim=1)
        lesion_prob = probs[0, 1, :, :].cpu().numpy()
        best = 0.0
        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            pred_bin = (lesion_prob > thresh).astype(np.float32)
            inter = (pred_bin * gt).sum()
            dice = float(2.0 * inter / (pred_bin.sum() + gt.sum() + 1e-8))
            if dice > best:
                best = dice
        return best


all_results = []

print("=" * 90)
print("BUDGET SWEEP: Find Optimal Regime for Radial Probe")
print("=" * 90)

for model_name, vit, ckpt_rel, n_class, ds in MODELS:
    print(f"\n{'='*80}")
    print(f"  {model_name}  |  {vit}  |  {ds}")
    print(f"{'='*80}")

    net = load_samed_model(vit, SAM_CKPTS[vit],
                           os.path.join(CKPT_BASE, ckpt_rel),
                           IMAGE_SIZE, n_class, device=device)
    net.eval()
    image_encoder = net.sam.image_encoder
    image_encoder.eval()

    images = load_data(ds, n=N_IMAGES)
    print(f"  Loaded {len(images)} images")

    for eps_frac in EPS_LIST:
        for n_steps in STEPS_LIST:
            step_frac = eps_frac / 10.0  # standard step size

            for method_key, params in METHODS.items():
                dices, clean_dices, times = [], [], []

                for img_idx, img_data in enumerate(images):
                    if ds == "Synapse":
                        x = torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
                        gt = img_data[1]
                    else:
                        x = img_data[0].to(device)
                        gt = img_data[1]

                    rg = x.max().item() - x.min().item()
                    eps_pix = eps_frac * rg
                    step_pix = step_frac * rg

                    clean_dice = evaluate_model(net, x, gt, ds)
                    if clean_dice < 0.01:
                        continue

                    t0 = time.time()
                    x_adv = attack_radial_optimized(
                        image_encoder, x, eps_pix, step_pix, n_steps,
                        amplitude_alpha=params.get("amplitude_alpha", 0.0),
                        weight_mode="magnitude",
                        lambda_amp=params.get("lambda_amp", 0.0),
                        amp_loss_mode=params.get("amp_loss_mode", "spatial"),
                        lambda_phase=params.get("lambda_phase", 0.0),
                        prior_weight=0.0,
                        radial_weight=params.get("radial_weight", 1.0),
                        use_random_noise=params.get("use_random_noise", False),
                        edge_map_fn=_get_edge_map,
                    )
                    adv_dice = evaluate_model(net, x_adv, gt, ds)
                    dices.append(adv_dice)
                    clean_dices.append(clean_dice)
                    times.append(time.time() - t0)

                if len(dices) == 0:
                    continue

                md_clean = float(np.mean(clean_dices))
                md_adv = float(np.mean(dices))
                md_delta = float(np.mean([c - a for c, a in zip(clean_dices, dices)]))

                all_results.append({
                    "model": model_name, "dataset": ds,
                    "eps": f"{eps_frac*255:.0f}/255", "steps": n_steps,
                    "method": method_key, "method_label": METHOD_LABELS[method_key],
                    "mean_clean": md_clean, "mean_adv": md_adv,
                    "mean_delta": md_delta, "n_valid": len(dices),
                })

                print(f"  eps={eps_frac*255:.0f}/255 steps={n_steps:2d} {METHOD_LABELS[method_key]:<18} "
                      f"Clean={md_clean:.4f} Adv={md_adv:.4f} Δ={md_delta:+.4f}")

    del net
    torch.cuda.empty_cache()

# ── Summary: Find regime where radial beats random ──────────────────────────

print(f"\n{'='*90}")
print("RADIAL ADVANTAGE: Δ(Radial) - Δ(Random) — positive = radial better")
print(f"{'='*90}")

for model_name, _, _, _, ds in MODELS:
    model_results = [r for r in all_results if r["model"] == model_name]
    print(f"\n── {model_name} ──")
    print(f"  {'eps':>10} {'steps':>6} ", end="")
    for m in ["bare_radial", "amp_loss", "full_optimized"]:
        print(f"  {METHOD_LABELS[m]:>20}", end="")
    print()

    for eps_frac in EPS_LIST:
        for n_steps in STEPS_LIST:
            eps_str = f"{eps_frac*255:.0f}/255"
            random_r = [r for r in model_results
                       if r["eps"] == eps_str and r["steps"] == n_steps
                       and r["method"] == "random"]
            if not random_r:
                continue
            random_delta = random_r[0]["mean_delta"]

            print(f"  {eps_str:>10} {n_steps:>6} ", end="")
            for m in ["bare_radial", "amp_loss", "full_optimized"]:
                mr = [r for r in model_results
                      if r["eps"] == eps_str and r["steps"] == n_steps
                      and r["method"] == m]
                if mr:
                    advantage = mr[0]["mean_delta"] - random_delta
                    marker = " ★" if advantage > 0.01 else ""
                    print(f"  {advantage:>+19.4f}{marker}", end="")
                else:
                    print(f"  {'N/A':>20}", end="")
            print()

# ── Save ─────────────────────────────────────────────────────────────────────

out_path = os.path.join(OUTPUT_DIR, "budget_sweep.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
print("Done — budget sweep complete")
