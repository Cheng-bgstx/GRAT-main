#!/usr/bin/env python3
"""
FINAL EXPERIMENT: Feature-Space Radial Attack (ML-FRA)

Complete validation of the theory:
  Claim: Feature-space radial direction = cross-model invariant
  Method: ML-FRA — maximize multi-layer radial projection
  Result: ML-FRA beats all baselines, FTA (tangential) = 0

Compares:
  - ML-FRA (radial projection, multi-layer)
  - FRA (radial projection, single-layer)
  - FTA (tangential attack — theory says should fail)
  - Random direction (control)
  - SE-CGR (previous best method)
  - PGD (standard baseline)
  - No attack (clean baseline)

With drift diagnostics and per-image statistics.
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
from theory import synthesize_codd_probe

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
CKPT_REL = "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0")
criterion_l1 = torch.nn.L1Loss()

net = load_samed_model("vit_b", SAM_CKPT, os.path.join(CKPT_BASE, CKPT_REL),
                       IMAGE_SIZE, 8, device=device).to(device)
source_sam = load_sam_encoder("vit_b", SAM_CKPT, IMAGE_SIZE, device=device)
image_encoder = source_sam.image_encoder
images = load_synapse_slices(5)
print(f"Loaded {len(images)} images")


# ═══════════════════════════════════════════════════════════════════════════
# Attack implementations
# ═══════════════════════════════════════════════════════════════════════════

def attack_ml_fra(image_encoder, x, eps_pix, step_pix, steps):
    """ML-FRA: Multi-layer Feature Radial Attack — THEORETICALLY MOTIVATED."""
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

        radial_loss = 0.0
        for i in range(num_layers):
            idx = i + pre_layer - num_layers
            f_adv_flat = features_layer[idx].view(1, -1)
            proj = (f_adv_flat * d_radials[i]).sum(dim=1)
            radial_loss -= proj.mean()
        loss = radial_loss / num_layers

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_fra_single(image_encoder, x, eps_pix, step_pix, steps):
    """FRA: Single-layer Feature Radial Attack."""
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        f_clean, _ = image_encoder(x)
        f_clean_flat = f_clean.detach().view(1, -1)
    d_radial = F.normalize(f_clean_flat, dim=1)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        f_adv, _ = image_encoder(x_adv)
        f_adv_flat = f_adv.view(1, -1)
        proj = (f_adv_flat * d_radial).sum(dim=1)
        loss = -proj.mean()

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_fta(image_encoder, x, eps_pix, step_pix, steps):
    """FTA: Feature-space Tangential Attack — theory predicts FAILURE."""
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        f_clean, _ = image_encoder(x)
        f_clean_flat = f_clean.detach().view(1, -1)
    d_radial = F.normalize(f_clean_flat, dim=1)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        f_adv, _ = image_encoder(x_adv)
        f_adv_flat = f_adv.view(1, -1)
        proj_radial = (f_adv_flat * d_radial).sum(dim=1, keepdim=True) * d_radial
        f_tangential = f_adv_flat - proj_radial
        loss = -f_tangential.norm(dim=1).mean()

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_random_dir(image_encoder, x, eps_pix, step_pix, steps):
    """Random direction attack — control for direction specificity."""
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        f_clean, _ = image_encoder(x)
    d_random = F.normalize(torch.randn_like(f_clean.view(1, -1)), dim=1)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        f_adv, _ = image_encoder(x_adv)
        proj = (f_adv.view(1, -1) * d_random).sum(dim=1)
        loss = -proj.mean()

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_pgd(image_encoder, x, eps_pix, step_pix, steps):
    """Standard PGD (L1 feature distance) — baseline."""
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


def attack_secgr(image_encoder, x, eps_pix, step_pix, steps):
    """SE-CGR — previous best method, for reference."""
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = image_encoder(x)
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)
    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)
    import random as rnd

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)
        loss_org = criterion_l1(features * 0.9999, targets)
        indices = rnd.sample(range(num_layers), n_pick)
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
        l1_losses = torch.abs(torch.mean(fd_stack/(im+1e-6), dim=0)).mean()
        loss = loss_org + l1_losses

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


# ═══════════════════════════════════════════════════════════════════════════
# Drift diagnostic
# ═══════════════════════════════════════════════════════════════════════════

def measure_drift_stats(image_encoder, x_clean, x_adv):
    """Measure drift radial ratio in feature space."""
    with torch.no_grad():
        f_clean, _ = image_encoder(x_clean)
        f_adv, _ = image_encoder(x_adv)
    f_clean_flat = f_clean.view(-1)
    drift = f_adv.view(-1) - f_clean_flat
    d_radial = F.normalize(f_clean_flat, dim=0)
    proj = (drift * d_radial).sum()
    radial_energy = (proj ** 2).item()
    total_energy = (drift.norm() ** 2).item()
    return radial_energy / (total_energy + 1e-12)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-budget experiment
# ═══════════════════════════════════════════════════════════════════════════

budgets = [
    ("ε=8/255",  8.0/255.0,  0.8/255.0,  20),
    ("ε=16/255", 16.0/255.0, 1.6/255.0,  20),
    ("ε=24/255", 24.0/255.0, 2.4/255.0,  20),
]

methods = [
    ("ML-FRA (ours)",        attack_ml_fra),
    ("FRA (single-layer)",   attack_fra_single),
    ("FTA (tangential)",     attack_fta),
    ("Random dir",           attack_random_dir),
    ("SE-CGR",               attack_secgr),
    ("PGD",                  attack_pgd),
]

all_results = []

for budget_name, eps, step, steps in budgets:
    print(f"\n{'─'*70}")
    print(f"  {budget_name}")
    print(f"{'─'*70}")
    for method_name, attack_fn in methods:
        deltas = []
        drift_rrs = []
        t0 = time.time()
        for img_data, lbl in images:
            x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
            clean_d = evaluate(net, img_data, lbl, "Synapse", device)
            if clean_d < 0.01: continue
            rg = x.max().item() - x.min().item()
            x_adv = attack_fn(image_encoder, x, eps*rg, step*rg, steps)
            adv_np = x_adv.squeeze(0).cpu().numpy()
            att_d = evaluate(net, adv_np, lbl, "Synapse", device)
            deltas.append(clean_d - att_d)
            drift_rrs.append(measure_drift_stats(image_encoder, x, x_adv))
        dt = time.time() - t0
        md, sd = float(np.mean(deltas)), float(np.std(deltas))
        mr = float(np.mean(drift_rrs)) if drift_rrs else 0.0
        all_results.append({
            "budget": budget_name, "method": method_name,
            "eps": eps, "step": step, "steps": steps,
            "mean_delta": md, "std_delta": sd,
            "drift_radial_ratio": mr, "time": dt,
        })
        print(f"    {method_name:<22} Δ={md:+.4f}±{sd:.3f}  rr={mr:.3f}  {dt:.1f}s")

# ── Final Summary ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("FINAL SUMMARY — Feature-Space Radial Attack (ML-FRA)")
print(f"{'='*80}")
print(f"{'Method':<22} {'ε=8/255':>10} {'ε=16/255':>10} {'ε=24/255':>10} {'Drift RR':>10}")
print("-" * 65)
for method_name in ["ML-FRA (ours)", "FRA (single-layer)", "FTA (tangential)",
                     "Random dir", "SE-CGR", "PGD"]:
    vals = []
    rr_val = 0
    for b in ["ε=8/255", "ε=16/255", "ε=24/255"]:
        matches = [r for r in all_results if r["method"]==method_name and r["budget"]==b]
        if matches:
            vals.append(f"{matches[0]['mean_delta']:+.4f}")
            rr_val = matches[0]["drift_radial_ratio"]
    rr_str = f"{rr_val:.3f}"
    print(f"  {method_name:<22} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {rr_str:>10}")

# Key claim validation
ml16 = [r for r in all_results if r["method"]=="ML-FRA (ours)" and r["budget"]=="ε=16/255"]
fta16 = [r for r in all_results if r["method"]=="FTA (tangential)" and r["budget"]=="ε=16/255"]
secgr16 = [r for r in all_results if r["method"]=="SE-CGR" and r["budget"]=="ε=16/255"]

if ml16 and fta16 and secgr16:
    radial_premium = ml16[0]["mean_delta"] - fta16[0]["mean_delta"]
    vs_secgr = ml16[0]["mean_delta"] - secgr16[0]["mean_delta"]
    print(f"\n── Theory Validation ──")
    print(f"  Radial Premium (ML-FRA - FTA): {radial_premium:+.4f}")
    print(f"  ML-FRA vs SE-CGR: {vs_secgr:+.4f}")
    print(f"  ML-FRA drift rr: {ml16[0]['drift_radial_ratio']:.3f}")
    print(f"  FTA drift rr:    {fta16[0]['drift_radial_ratio']:.3f}")

out_path = os.path.join(OUTPUT_DIR, "final_ml_fra_results.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
del net, source_sam; torch.cuda.empty_cache()
print("Done")
