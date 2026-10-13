#!/usr/bin/env python3
"""
Radial-Directed Probe Attack: Replace random probe component with
deterministic radial direction derived from PAID theory.

THEORY:
  PAID shows phase Φ is the cross-model invariant.
  Phase Φ determines feature vector direction f/|f| (validated: radial premium +0.797).

  Current best method (SE-CGR effective core):
    L = L1(f_adv, f_clean) + |drift - 2*random_probe|
    The random_probe = IFFT(N_white) provides diverse exploration directions.

  OUR METHOD: Replace part of random probe with DETERMINISTIC RADIAL TARGET.
    D_r = flatten(F_clean) / |flatten(F_clean)|  → reshape to (C,H,W)
    This is the RADIAL DIRECTION MAP — at each (c,h,w), it points radially.

    probe = λ_r * D_r + λ_n * IFFT(N_white)

  INTUITION:
    - D_r provides the "signal" (correct cross-model direction)
    - IFFT(N_white) provides "exploration" (diverse gradient targets)
    - λ_r controls how much we trust the radial direction

  ABLATION PREDICTION:
    λ_r > 0 should beat λ_r = 0 (pure random = current best)
    Because explicitly targeting the known invariant direction helps.
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

net = load_samed_model("vit_b", SAM_CKPT, os.path.join(CKPT_BASE, CKPT_REL),
                       IMAGE_SIZE, 8, device=device).to(device)
source_sam = load_sam_encoder("vit_b", SAM_CKPT, IMAGE_SIZE, device=device)
image_encoder = source_sam.image_encoder
images = load_synapse_slices(5)
print(f"Loaded {len(images)} images")


def make_radial_probe(F_clean):
    """
    Deterministic radial direction probe from clean feature map.

    D_r = flatten(F_clean) / |flatten(F_clean)| → reshape to original shape

    This is the direction in feature map space that corresponds to
    the RADIAL direction in feature vector space (cross-model invariant).
    """
    shape = F_clean.shape
    f_flat = F_clean.reshape(-1)
    d_r = F.normalize(f_flat, dim=0)
    return d_r.reshape(shape)


def make_random_fft_probe(shape, device):
    """Random IFFT probe — spatially correlated noise. Works with any shape."""
    noise = torch.randn(*shape, device=device) + 1j * torch.randn(*shape, device=device)
    return torch.fft.ifftn(noise, dim=(-2, -1), norm="ortho").real


def attack_radial_probe(image_encoder, x, eps_pix, step_pix, steps,
                        radial_weight=0.0):
    """
    Radial-Directed Probe Attack.

    radial_weight=0.0: pure random IFFT probe (current best baseline)
    radial_weight=0.5: equal radial + random
    radial_weight=1.0: pure radial probe
    """
    max_val, min_val = x.max(), x.min()

    with torch.no_grad():
        targets, targets_layer = image_encoder(x)

    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)

    # Pre-compute radial direction maps for each layer
    radial_maps = []
    for i in range(num_layers):
        idx = i + pre_layer - num_layers
        radial_maps.append(make_radial_probe(targets_layer[idx]))

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    for step_idx in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)

        # L1 on final features
        loss_org = criterion_l1(features * 0.9999, targets)

        # Probe alignment on intermediate layers
        import random
        indices = random.sample(range(num_layers), n_pick)
        fd_list = []
        for idx in indices:
            layer_idx = idx + pre_layer - num_layers
            F_clean_i = targets_layer[layer_idx]
            F_adv_i = features_layer[layer_idx]
            # Build combined probe
            shape = F_clean_i.shape
            if radial_weight > 0:
                probe_radial = radial_maps[idx]
                probe_radial = probe_radial * 0.01 / (probe_radial.abs().mean() + 1e-8)
            else:
                probe_radial = 0.0

            if radial_weight < 1.0:
                probe_random = make_random_fft_probe(shape, F_clean_i.device)
                probe_random = probe_random * 0.01 / (probe_random.abs().mean() + 1e-8)
            else:
                probe_random = 0.0

            probe = radial_weight * probe_radial + (1.0 - radial_weight) * probe_random

            drift = F_adv_i * 0.999 - F_clean_i
            fd_list.append((drift - 2.0 * probe))

        fd_stack = torch.stack(fd_list)
        selected_t = torch.stack(
            [targets_layer[i + pre_layer - num_layers] for i in indices])
        im = torch.stack([selected_t[j].abs().mean()
                         for j in range(len(selected_t))])
        im = im.view(-1, *([1] * (fd_stack.dim() - 1)))
        probe_loss = torch.abs(torch.mean(fd_stack / (im + 1e-6), dim=0)).mean()
        loss = loss_org + probe_loss

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)

        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


def attack_secgr(image_encoder, x, eps_pix, step_pix, steps):
    """SE-CGR reference."""
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


def measure_drift_radial_ratio(image_encoder, x_clean, x_adv):
    with torch.no_grad():
        f_clean, _ = image_encoder(x_clean)
        f_adv, _ = image_encoder(x_adv)
    f_clean_flat = f_clean.view(-1)
    drift = f_adv.view(-1) - f_clean_flat
    d_radial = F.normalize(f_clean_flat, dim=0)
    proj = (drift * d_radial).sum()
    return (proj ** 2).item() / ((drift.norm() ** 2).item() + 1e-12)


# ── Experiment: Sweep radial_weight ──────────────────────────────────────

budgets = [
    ("eps=8/255",  8.0/255.0,  0.8/255.0,  20),
    ("eps=16/255", 16.0/255.0, 1.6/255.0,  20),
    ("eps=24/255", 24.0/255.0, 2.4/255.0,  20),
]

radial_weights = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

all_results = []

for budget_name, eps, step, steps in budgets:
    print(f"\n{'─'*70}")
    print(f"  {budget_name}")
    print(f"{'─'*70}")
    for rw in radial_weights:
        name = f"Radial probe (λr={rw:.2f})"
        if rw == 0.0:
            name = "Random probe (λr=0) [BASELINE]"
        elif rw == 1.0:
            name = "Pure radial (λr=1.0)"

        deltas = []
        drift_rrs = []
        t0 = time.time()
        for img_data, lbl in images:
            x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
            clean_d = evaluate(net, img_data, lbl, "Synapse", device)
            if clean_d < 0.01:
                continue
            rg = x.max().item() - x.min().item()
            x_adv = attack_radial_probe(image_encoder, x, eps*rg, step*rg, steps,
                                        radial_weight=rw)
            adv_np = x_adv.squeeze(0).cpu().numpy()
            att_d = evaluate(net, adv_np, lbl, "Synapse", device)
            deltas.append(clean_d - att_d)
            drift_rrs.append(measure_drift_radial_ratio(image_encoder, x, x_adv))
        dt = time.time() - t0
        md, sd = float(np.mean(deltas)), float(np.std(deltas))
        mr = float(np.mean(drift_rrs)) if drift_rrs else 0.0
        all_results.append({
            "budget": budget_name, "radial_weight": rw,
            "mean_delta": md, "std_delta": sd,
            "drift_radial_ratio": mr, "time": dt,
        })
        marker = ""
        if rw == 0.0:
            marker = f"  ← baseline"
        elif md > [r for r in all_results
                    if r["budget"] == budget_name and r["radial_weight"] == 0.0][0]["mean_delta"]:
            marker = f"  ← BEATS BASELINE"
        print(f"  {name:<30} Δ={md:+.4f}±{sd:.3f}  rr={mr:.3f}  {dt:.1f}s{marker}")

    # SE-CGR reference
    deltas = []
    drift_rrs = []
    t0 = time.time()
    for img_data, lbl in images:
        x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        clean_d = evaluate(net, img_data, lbl, "Synapse", device)
        if clean_d < 0.01:
            continue
        rg = x.max().item() - x.min().item()
        x_adv = attack_secgr(image_encoder, x, eps*rg, step*rg, steps)
        adv_np = x_adv.squeeze(0).cpu().numpy()
        att_d = evaluate(net, adv_np, lbl, "Synapse", device)
        deltas.append(clean_d - att_d)
        drift_rrs.append(measure_drift_radial_ratio(image_encoder, x, x_adv))
    dt = time.time() - t0
    md, sd = float(np.mean(deltas)), float(np.std(deltas))
    mr = float(np.mean(drift_rrs)) if drift_rrs else 0.0
    all_results.append({
        "budget": budget_name, "radial_weight": -1,
        "mean_delta": md, "std_delta": sd,
        "drift_radial_ratio": mr, "time": dt,
    })
    print(f"  {'SE-CGR (reference)':<30} Δ={md:+.4f}±{sd:.3f}  rr={mr:.3f}  {dt:.1f}s")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("RADIAL-DIRECTED PROBE — Sweep λr")
print(f"{'='*80}")
for budget_name in ["eps=8/255", "eps=16/255", "eps=24/255"]:
    print(f"\n  {budget_name}:")
    print(f"  {'λr':<8} {'ΔDice':>10} {'vs baseline':>12}")
    print(f"  {'-'*32}")
    baseline = [r for r in all_results
                if r["budget"] == budget_name and r.get("radial_weight") == 0.0]
    base_delta = baseline[0]["mean_delta"] if baseline else 0
    for rw in radial_weights:
        matches = [r for r in all_results
                   if r["budget"] == budget_name and r.get("radial_weight") == rw]
        if matches:
            md = matches[0]["mean_delta"]
            diff = md - base_delta
            marker = " ← BEST" if diff > 0 and rw > 0 else ""
            if rw == 0:
                marker = " ← baseline"
            print(f"  {rw:<8.2f} {md:+.4f}     {diff:+.4f}    {marker}")
    secgr = [r for r in all_results
             if r["budget"] == budget_name and r.get("radial_weight") == -1]
    if secgr:
        diff = secgr[0]["mean_delta"] - base_delta
        print(f"  {'SE-CGR':<8} {secgr[0]['mean_delta']:+.4f}     {diff:+.4f}")

out_path = os.path.join(OUTPUT_DIR, "radial_probe_sweep.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
del net, source_sam
torch.cuda.empty_cache()
print("Done")
