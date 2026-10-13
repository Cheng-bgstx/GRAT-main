#!/usr/bin/env python3
"""
Radial-Directed Probe Attack v2: Validated method from PAID theory.

THEORETICAL FRAMEWORK:
  PAID → Phase Φ is cross-model invariant (spatial structure)
  Phase Φ → Feature direction f/|f| = radial direction
  Radial direction → D_r = flatten(F_clean)/|flatten(F_clean)| (per layer)

METHOD:
  1. Pre-compute radial direction map D_r for each layer from clean features
  2. Each step: probe = λ_r * D_r + (1-λ_r) * IFFT(N_complex)
     - λ_r = 0.5 optimal (balances signal vs exploration)
  3. L1 on final features + probe alignment on intermediate layers
  4. Edge-guided gradient (PSAA)

ABLATION:
  Sweep λ_r ∈ {0, 0.25, 0.5, 0.75, 1.0}
  λ_r = 0: pure random (baseline)
  λ_r = 1: pure radial
  Theory predicts: 0 < λ_r < 1 beats both extremes

COMPARISON:
  vs SE-CGR, ML-FRA-cos, PGD, FTA, Random dir
  At multiple budgets: eps = 8/255, 16/255, 24/255
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
    """D_r = flatten(F_clean) / |flatten(F_clean)| — radial direction map."""
    shape = F_clean.shape
    f_flat = F_clean.reshape(-1)
    d_r = F.normalize(f_flat, dim=0)
    return d_r.reshape(shape)


def make_random_probe(shape, device):
    """IFFT(complex white noise) — spatially correlated random pattern."""
    noise = torch.randn(*shape, device=device) + 1j * torch.randn(*shape, device=device)
    return torch.fft.ifftn(noise, dim=(-2, -1), norm="ortho").real


def attack_radial_directed(image_encoder, x, eps_pix, step_pix, steps,
                           radial_weight=0.5):
    """
    Radial-Directed Probe Attack.

    probe = λ_r * D_r + (1-λ_r) * IFFT(N_complex)

    D_r targets the cross-model invariant radial direction.
    IFFT(N) provides diverse exploration.
    """
    max_val, min_val = x.max(), x.min()

    with torch.no_grad():
        targets, targets_layer = image_encoder(x)

    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)

    # Pre-compute radial direction maps
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

        loss_org = criterion_l1(features * 0.9999, targets)

        import random
        indices = random.sample(range(num_layers), n_pick)
        fd_list = []
        for idx in indices:
            layer_idx = idx + pre_layer - num_layers
            F_clean_i = targets_layer[layer_idx]
            F_adv_i = features_layer[layer_idx]
            shape = F_clean_i.shape

            # Build probe: radial + random
            probe_radial = radial_maps[idx]
            r_scale = 0.01 / (probe_radial.abs().mean() + 1e-8)
            probe_radial = probe_radial * r_scale

            probe_random = make_random_probe(shape, F_clean_i.device)
            n_scale = 0.01 / (probe_random.abs().mean() + 1e-8)
            probe_random = probe_random * n_scale

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


def attack_ml_fra_cos(image_encoder, x, eps_pix, step_pix, steps):
    """ML-FRA-cos — cosine similarity only."""
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
    """PGD baseline — L1 on final features."""
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
    ("eps=8/255",  8.0/255.0,  0.8/255.0,  20),
    ("eps=16/255", 16.0/255.0, 1.6/255.0,  20),
    ("eps=24/255", 24.0/255.0, 2.4/255.0,  20),
]

all_results = []

for budget_name, eps, step, steps in budgets:
    print(f"\n{'─'*70}")
    print(f"  {budget_name}")
    print(f"{'─'*70}")

    # Radial probe sweep
    for rw in [0.0, 0.25, 0.5, 0.75, 1.0]:
        name = f"Radial λr={rw:.2f}"
        deltas, rrs = [], []
        t0 = time.time()
        for img_data, lbl in images:
            x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
            clean_d = evaluate(net, img_data, lbl, "Synapse", device)
            if clean_d < 0.01:
                continue
            rg = x.max().item() - x.min().item()
            x_adv = attack_radial_directed(image_encoder, x, eps*rg, step*rg, steps,
                                           radial_weight=rw)
            adv_np = x_adv.squeeze(0).cpu().numpy()
            att_d = evaluate(net, adv_np, lbl, "Synapse", device)
            deltas.append(clean_d - att_d)
            rrs.append(measure_drift_radial_ratio(image_encoder, x, x_adv))
        dt = time.time() - t0
        md, sd, mr = float(np.mean(deltas)), float(np.std(deltas)), float(np.mean(rrs))
        all_results.append({"budget": budget_name, "method": name,
                            "mean_delta": md, "std_delta": sd,
                            "drift_radial_ratio": mr, "time": dt})
        base = [r for r in all_results if r["budget"] == budget_name
                and r["method"] == "Radial λr=0.00"]
        base_d = base[0]["mean_delta"] if base else md
        flag = "← BEST" if md > base_d + 0.005 else ("← same" if abs(md - base_d) < 0.005 else "")
        if rw == 0:
            flag = "← baseline"
        print(f"  {name:<20} Δ={md:+.4f}±{sd:.3f}  rr={mr:.3f}  {dt:.1f}s  {flag}")

    # Baselines
    baselines = [
        ("SE-CGR",          attack_secgr),
        ("ML-FRA-cos",      attack_ml_fra_cos),
        ("PGD (L1)",        attack_pgd),
    ]
    for bname, bfn in baselines:
        deltas, rrs = [], []
        t0 = time.time()
        for img_data, lbl in images:
            x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
            clean_d = evaluate(net, img_data, lbl, "Synapse", device)
            if clean_d < 0.01:
                continue
            rg = x.max().item() - x.min().item()
            x_adv = bfn(image_encoder, x, eps*rg, step*rg, steps)
            adv_np = x_adv.squeeze(0).cpu().numpy()
            att_d = evaluate(net, adv_np, lbl, "Synapse", device)
            deltas.append(clean_d - att_d)
            rrs.append(measure_drift_radial_ratio(image_encoder, x, x_adv))
        dt = time.time() - t0
        md, sd, mr = float(np.mean(deltas)), float(np.std(deltas)), float(np.mean(rrs))
        all_results.append({"budget": budget_name, "method": bname,
                            "mean_delta": md, "std_delta": sd,
                            "drift_radial_ratio": mr, "time": dt})
        print(f"  {bname:<20} Δ={md:+.4f}±{sd:.3f}  rr={mr:.3f}  {dt:.1f}s")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("RADIAL-DIRECTED PROBE — Final Summary")
print(f"{'='*80}")
print(f"{'Method':<20} {'eps=8/255':>10} {'eps=16/255':>10} {'eps=24/255':>10}")
print("-" * 56)
for method in ["Radial λr=0.00", "Radial λr=0.25", "Radial λr=0.50",
               "Radial λr=0.75", "Radial λr=1.00",
               "SE-CGR", "ML-FRA-cos", "PGD (L1)"]:
    vals = []
    for b in ["eps=8/255", "eps=16/255", "eps=24/255"]:
        m = [r for r in all_results if r["method"] == method and r["budget"] == b]
        vals.append(f"{m[0]['mean_delta']:+.4f}" if m else "       N/A")
    if any(v != "       N/A" for v in vals):
        print(f"  {method:<20} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

# Best method
print(f"\n── Best per budget ──")
for b in ["eps=8/255", "eps=16/255", "eps=24/255"]:
    matches = [(r["mean_delta"], r["method"]) for r in all_results if r["budget"] == b]
    best = max(matches, key=lambda x: x[0])
    print(f"  {b}: {best[1]} at {best[0]:+.4f}")

out_path = os.path.join(OUTPUT_DIR, "radial_probe_v2_results.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
del net, source_sam
torch.cuda.empty_cache()
print("Done")
