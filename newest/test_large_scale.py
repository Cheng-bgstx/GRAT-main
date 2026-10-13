#!/usr/bin/env python3
"""
LARGE-SCALE VALIDATION: Radial-Directed Probe Attack
12 images × 4 budgets × 5 methods × drift diagnostics

Methods:
  1. Radial Probe (λr=1.0) — OUR METHOD from PAID theory
  2. Random Probe (λr=0.0) — ablation baseline (same structure, random probe)
  3. SE-CGR — previous best
  4. ML-FRA-cos — cosine similarity baseline
  5. PGD — standard baseline

Budgets: eps = 4/255, 8/255, 16/255, 24/255
All methods use same structure: L1 + probe alignment + edge guidance
(except ML-FRA-cos and PGD which use their native losses)
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model
from exp_paid_optimal_attack import load_synapse_slices, evaluate
from attack_codd import _get_edge_map

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
CKPT_REL = "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0")
criterion_l1 = torch.nn.L1Loss()

print("Loading models...")
net = load_samed_model("vit_b", SAM_CKPT, os.path.join(CKPT_BASE, CKPT_REL),
                       IMAGE_SIZE, 8, device=device).to(device)
source_sam = load_sam_encoder("vit_b", SAM_CKPT, IMAGE_SIZE, device=device)
image_encoder = source_sam.image_encoder
images = load_synapse_slices(1000)
print(f"Loaded {len(images)} images\n")


def make_radial_probe(F_clean):
    shape = F_clean.shape
    f_flat = F_clean.reshape(-1)
    d_r = F.normalize(f_flat, dim=0)
    return d_r.reshape(shape)


def make_random_probe(shape, device):
    noise = torch.randn(*shape, device=device) + 1j * torch.randn(*shape, device=device)
    return torch.fft.ifftn(noise, dim=(-2, -1), norm="ortho").real


def attack_probe(image_encoder, x, eps_pix, step_pix, steps,
                 use_radial=True):
    """Unified probe attack: radial (ours) or random (ablation baseline)."""
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = image_encoder(x)

    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)

    if use_radial:
        radial_maps = []
        for i in range(num_layers):
            idx = i + pre_layer - num_layers
            radial_maps.append(make_radial_probe(targets_layer[idx]))

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)
        loss = criterion_l1(features * 0.9999, targets)

        import random
        indices = random.sample(range(num_layers), n_pick)
        fd_list = []
        for idx in indices:
            layer_idx = idx + pre_layer - num_layers
            F_c = targets_layer[layer_idx]
            F_a = features_layer[layer_idx]
            shape = F_c.shape

            if use_radial:
                probe = radial_maps[idx]
            else:
                probe = make_random_probe(shape, F_c.device)
            scale = 0.01 / (probe.abs().mean() + 1e-8)
            probe = probe * scale

            fd_list.append((F_a * 0.999 - F_c - 2.0 * probe))

        fd_stack = torch.stack(fd_list)
        selected_t = torch.stack(
            [targets_layer[i + pre_layer - num_layers] for i in indices])
        im = torch.stack([selected_t[j].abs().mean()
                         for j in range(len(selected_t))])
        im = im.view(-1, *([1] * (fd_stack.dim() - 1)))
        loss += torch.abs(torch.mean(fd_stack / (im + 1e-6), dim=0)).mean()

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_secgr(image_encoder, x, eps_pix, step_pix, steps):
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = image_encoder(x)
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)
    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)

    from theory import synthesize_codd_probe
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
        from attack_codd import _return_mean_value
        im = _return_mean_value(selected_t)
        if len(im.shape) > len(fd_stack.shape):
            im = im.view(fd_stack.shape[0], 1, 1, 1)
        loss = loss_org + torch.abs(torch.mean(fd_stack/(im+1e-6), dim=0)).mean()

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_ml_fra_cos(image_encoder, x, eps_pix, step_pix, steps):
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = image_encoder(x)
    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    d_radials = []
    for i in range(num_layers):
        idx = i + pre_layer - num_layers
        f_flat = targets_layer[idx].detach().view(1, -1)
        d_radials.append(F.normalize(f_flat, dim=1))
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)
        loss = 0.0
        for i in range(num_layers):
            idx = i + pre_layer - num_layers
            f_adv_flat = features_layer[idx].view(1, -1)
            cos_sim = (f_adv_flat * d_radials[i]).sum(dim=1) / (f_adv_flat.norm(dim=1) + 1e-8)
            loss -= cos_sim.mean()
        loss = loss / num_layers
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_pgd(image_encoder, x, eps_pix, step_pix, steps):
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, _ = image_encoder(x)
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        f, _ = image_encoder(x_adv)
        loss = criterion_l1(f * 0.9999, targets)
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_fta(image_encoder, x, eps_pix, step_pix, steps):
    """Tangential attack — theory says Δ≈0."""
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        f_clean, _ = image_encoder(x)
        f_clean_flat = f_clean.detach().view(1, -1)
    d_radial = F.normalize(f_clean_flat, dim=1)
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        f_adv, _ = image_encoder(x_adv)
        f_adv_flat = f_adv.view(1, -1)
        proj_radial = (f_adv_flat * d_radial).sum(dim=1, keepdim=True) * d_radial
        f_tangential = f_adv_flat - proj_radial
        loss = -f_tangential.norm(dim=1).mean()
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def measure_drift_radial_ratio(image_encoder, x_clean, x_adv):
    with torch.no_grad():
        f_clean, _ = image_encoder(x_clean)
        f_adv, _ = image_encoder(x_adv)
    f_clean_flat = f_clean.view(-1)
    drift = f_adv.view(-1) - f_clean_flat
    d_radial = F.normalize(f_clean_flat, dim=0)
    proj = (drift * d_radial).sum()
    return (proj ** 2).item() / ((drift.norm() ** 2).item() + 1e-12)


# ── Experiment ────────────────────────────────────────────────────────────
budgets = [
    ("eps=4/255",  4.0/255.0,  0.4/255.0,  20),
    ("eps=8/255",  8.0/255.0,  0.8/255.0,  20),
    ("eps=16/255", 16.0/255.0, 1.6/255.0,  20),
    ("eps=24/255", 24.0/255.0, 2.4/255.0,  20),
]

methods = [
    ("Radial Probe (ours)",  attack_probe,     {"use_radial": True}),
    ("Random Probe (abl)",   attack_probe,     {"use_radial": False}),
    ("SE-CGR",               attack_secgr,     {}),
    ("ML-FRA-cos",           attack_ml_fra_cos, {}),
    ("PGD",                  attack_pgd,       {}),
    ("FTA (tangential)",     attack_fta,       {}),
]

all_results = []
per_image_results = []

for budget_name, eps, step, steps in budgets:
    print(f"\n{'='*70}")
    print(f"  {budget_name}  |  {len(images)} images  |  steps={steps}")
    print(f"{'='*70}")

    for method_name, attack_fn, kwargs in methods:
        deltas, rrs, clean_dices, att_dices = [], [], [], []
        t0 = time.time()
        for img_idx, (img_data, lbl) in enumerate(images):
            x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
            clean_d = evaluate(net, img_data, lbl, "Synapse", device)
            if clean_d < 0.01:
                continue
            rg = x.max().item() - x.min().item()
            x_adv = attack_fn(image_encoder, x, eps * rg, step * rg, steps, **kwargs)
            adv_np = x_adv.squeeze(0).cpu().numpy()
            att_d = evaluate(net, adv_np, lbl, "Synapse", device)
            deltas.append(clean_d - att_d)
            clean_dices.append(clean_d)
            att_dices.append(att_d)
            rrs.append(measure_drift_radial_ratio(image_encoder, x, x_adv))
            per_image_results.append({
                "budget": budget_name, "method": method_name,
                "img_idx": img_idx, "clean_dice": float(clean_d),
                "att_dice": float(att_d), "delta": float(clean_d - att_d),
                "drift_rr": rrs[-1],
            })

        dt = time.time() - t0
        md, sd = float(np.mean(deltas)), float(np.std(deltas))
        mr = float(np.mean(rrs)) if rrs else 0.0
        clean_avg = float(np.mean(clean_dices))
        att_avg = float(np.mean(att_dices))

        all_results.append({
            "budget": budget_name, "method": method_name,
            "n_images": len(deltas),
            "mean_clean_dice": clean_avg, "mean_att_dice": att_avg,
            "mean_delta": md, "std_delta": sd,
            "drift_radial_ratio": mr, "time": dt,
        })
        print(f"  {method_name:<25} clean={clean_avg:.4f} → att={att_avg:.4f}  "
              f"Δ={md:+.4f}±{sd:.3f}  rr={mr:.3f}  {dt:.1f}s")

# ── Summary Tables ────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("LARGE-SCALE VALIDATION — SUMMARY")
print(f"{'='*90}")

print(f"\n{'Method':<25}", end="")
for b in ["eps=4/255", "eps=8/255", "eps=16/255", "eps=24/255"]:
    print(f"{b:>12}", end="")
print(f"{'rr@16':>8}")
print("-" * 78)

for method_name in ["Radial Probe (ours)", "Random Probe (abl)", "SE-CGR",
                     "ML-FRA-cos", "PGD", "FTA (tangential)"]:
    print(f"  {method_name:<23}", end="")
    rr_val = 0
    for b in ["eps=4/255", "eps=8/255", "eps=16/255", "eps=24/255"]:
        m = [r for r in all_results if r["method"] == method_name and r["budget"] == b]
        if m:
            print(f"  {m[0]['mean_delta']:+.4f}  ", end="")
            if b == "eps=16/255":
                rr_val = m[0]["drift_radial_ratio"]
        else:
            print(f"  {'N/A':>8}  ", end="")
    print(f"  {rr_val:.3f}")

# ── Statistical Analysis ──────────────────────────────────────────────────
print(f"\n{'='*90}")
print("STATISTICAL ANALYSIS (paired per-image differences)")
print(f"{'='*90}")

from scipy import stats as scipy_stats

for budget_name in ["eps=4/255", "eps=8/255", "eps=16/255", "eps=24/255"]:
    print(f"\n  [{budget_name}]")
    radial_per_img = {}
    for r in per_image_results:
        if r["budget"] == budget_name:
            radial_per_img.setdefault(r["method"], {})[r["img_idx"]] = r["delta"]

    # Radial vs Random (key ablation)
    if "Radial Probe (ours)" in radial_per_img and "Random Probe (abl)" in radial_per_img:
        r1 = radial_per_img["Radial Probe (ours)"]
        r2 = radial_per_img["Random Probe (abl)"]
        common = sorted(set(r1.keys()) & set(r2.keys()))
        diffs = [r1[i] - r2[i] for i in common]
        md_diff = np.mean(diffs)
        # Paired t-test
        if len(diffs) >= 3:
            t_stat, p_val = scipy_stats.ttest_rel(
                [r1[i] for i in common], [r2[i] for i in common])
        else:
            t_stat, p_val = 0, 1.0
        cohens_d = md_diff / (np.std(diffs) + 1e-12)
        better = sum(1 for d in diffs if d > 0)
        print(f"    Radial vs Random:  ΔΔ={md_diff:+.4f}, "
              f"t={t_stat:.2f}, p={p_val:.4f}, d={cohens_d:.2f}, "
              f"wins={better}/{len(diffs)}")

    # Radial vs SE-CGR
    if "Radial Probe (ours)" in radial_per_img and "SE-CGR" in radial_per_img:
        r1 = radial_per_img["Radial Probe (ours)"]
        r3 = radial_per_img["SE-CGR"]
        common = sorted(set(r1.keys()) & set(r3.keys()))
        diffs = [r1[i] - r3[i] for i in common]
        md_diff = np.mean(diffs)
        if len(diffs) >= 3:
            t_stat, p_val = scipy_stats.ttest_rel(
                [r1[i] for i in common], [r3[i] for i in common])
        else:
            t_stat, p_val = 0, 1.0
        better = sum(1 for d in diffs if d > 0)
        print(f"    Radial vs SE-CGR:  ΔΔ={md_diff:+.4f}, "
              f"t={t_stat:.2f}, p={p_val:.4f}, "
              f"wins={better}/{len(diffs)}")

    # Show individual deltas
    for method_name in ["Radial Probe (ours)", "Random Probe (abl)", "SE-CGR"]:
        if method_name in radial_per_img:
            vals = list(radial_per_img[method_name].values())
            print(f"    {method_name:<25} [{', '.join(f'{v:+.4f}' for v in vals)}]")

# ── Save ──────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "large_scale_results.json")
with open(out_path, "w") as f:
    json.dump({"summary": all_results, "per_image": per_image_results}, f, indent=2)
print(f"\nSaved to {out_path}")
del net, source_sam
torch.cuda.empty_cache()
print("Done")
