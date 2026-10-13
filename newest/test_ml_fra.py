#!/usr/bin/env python3
"""
Multi-Layer Feature Radial Attack (ML-FRA): Extend FRA to multiple encoder
layers, matching SE-CGR's multi-layer structure while using the theoretically
motivated feature-space radial direction.

Goal: Beat SE-CGR (+0.852) with a method directly derived from the theory.

Variants tested:
  1. ML-FRA:      multi-layer radial projection maximization
  2. ML-FRA+L1:   multi-layer radial + L1 on final output
  3. ML-FRA probe: multi-layer probe alignment (SE-CGR structure, FRA direction)
  4. FRA+L1 tuned: single-layer with tuned radial_weight
  5. SE-CGR:       reference
  6. PGD:          baseline
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
from attack_codd import _get_edge_map, _return_mean_value
from theory import synthesize_codd_probe

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
CKPT_REL = "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"
EPSILON = 16.0 / 255.0
STEP = 1.6 / 255.0
STEPS = 20
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
# 1. ML-FRA: Multi-Layer Feature Radial Attack
# ═══════════════════════════════════════════════════════════════════════════

def attack_ml_fra(image_encoder, x, eps_pix, step_pix, steps,
                 radial_weight=1.0, l1_weight=0.0, mode="projection"):
    """
    Multi-Layer Feature Radial Attack.

    mode="projection": maximize radial projection sum across layers
    mode="probe": SE-CGR-style probe alignment using feature-space radial direction
    """
    max_val, min_val = x.max(), x.min()

    with torch.no_grad():
        targets, targets_layer = image_encoder(x)

    # Prepare radial directions for each layer
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

        if mode == "projection":
            # Maximize radial projection at each layer
            if radial_weight > 0:
                radial_loss = 0.0
                for i in range(num_layers):
                    idx = i + pre_layer - num_layers
                    f_adv_flat = features_layer[idx].view(1, -1)
                    proj = (f_adv_flat * d_radials[i]).sum(dim=1)
                    radial_loss -= proj.mean()
                loss += radial_weight * radial_loss / num_layers

            if l1_weight > 0:
                loss += l1_weight * criterion_l1(features * 0.9999, targets)

        elif mode == "probe":
            # SE-CGR style: align drift with radial probe at each layer
            loss_org = criterion_l1(features * 0.9999, targets)

            import random
            n_pick = max(1, num_layers - 2)
            indices = random.sample(range(num_layers), n_pick)
            fd_list = []
            for i in indices:
                idx = i + pre_layer - num_layers
                F_clean_i = targets_layer[idx]
                F_adv_i = features_layer[idx]

                # Feature-space radial probe:
                # Extract radial component of drift = proj(F_adv - F_clean onto d_radial)
                f_clean_flat = F_clean_i.detach().view(-1)
                d_r = F.normalize(f_clean_flat, dim=0)
                f_adv_flat = F_adv_i.view(-1)
                drift_flat = f_adv_flat - f_clean_flat
                proj_radial = (drift_flat * d_r).sum() * d_r
                probe_spatial = proj_radial.view_as(F_clean_i)

                fd_list.append((F_adv_i * 0.999 - F_clean_i) - 2.0 * probe_spatial)

            fd_stack = torch.stack(fd_list)
            selected_t = torch.stack([targets_layer[i + pre_layer - num_layers]
                                      for i in indices])
            input_mean = torch.stack([selected_t[j].abs().mean()
                                      for j in range(len(selected_t))])
            input_mean = input_mean.view(-1, *([1] * (fd_stack.dim() - 1)))
            fd_norm = fd_stack / (input_mean + 1e-6)
            l1_losses = torch.abs(torch.mean(fd_norm, dim=0)).mean()
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
# 2. Single-layer FRA with tuned weights
# ═══════════════════════════════════════════════════════════════════════════

def attack_fra_tuned(image_encoder, x, eps_pix, step_pix, steps,
                    radial_weight=1.0, l1_weight=0.0):
    """Single-layer FRA with tunable weights."""
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

        loss = 0.0
        if radial_weight > 0:
            proj_radial = (f_adv_flat * d_radial).sum(dim=1)
            loss -= radial_weight * proj_radial.mean()
        if l1_weight > 0:
            with torch.no_grad():
                t, _ = image_encoder(x)
            loss += l1_weight * criterion_l1(f_adv * 0.9999, t)

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# ═══════════════════════════════════════════════════════════════════════════
# 3. SE-CGR reference
# ═══════════════════════════════════════════════════════════════════════════

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

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)
        loss_org = criterion_l1(features * 0.9999, targets)

        import random
        indices = random.sample(range(num_layers), n_pick)
        fd_list = []
        for idx in indices:
            F_clean_i = targets_layer[idx + pre_layer - num_layers]
            F_adv_i = features_layer[idx + pre_layer - num_layers]
            probe = synthesize_codd_probe(
                F_clean=F_clean_i.unsqueeze(0), F_adv=F_adv_i.unsqueeze(0),
                direction="radial", use_random_noise=True)
            fd_list.append((F_adv_i.unsqueeze(0)*0.999 - F_clean_i.unsqueeze(0)) - 2.0*probe)

        fd_stack = torch.stack([f.squeeze(0) for f in fd_list])
        selected_t = torch.stack([targets_layer[i + pre_layer - num_layers] for i in indices])
        input_mean = torch.stack([selected_t[j].abs().mean() for j in range(len(selected_t))])
        input_mean = input_mean.view(-1, *([1]*(fd_stack.dim()-1)))
        l1_losses = torch.abs(torch.mean(fd_stack/(input_mean+1e-6), dim=0)).mean()
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
# Experiment
# ═══════════════════════════════════════════════════════════════════════════

configs = [
    # (name, fn, kwargs)
    ("SE-CGR (reference)",      attack_secgr,     {}),
    ("ML-FRA proj (r=1,L1=0)",  attack_ml_fra,    {"radial_weight": 1.0, "l1_weight": 0.0, "mode": "projection"}),
    ("ML-FRA proj (r=1,L1=1)",  attack_ml_fra,    {"radial_weight": 1.0, "l1_weight": 1.0, "mode": "projection"}),
    ("ML-FRA proj (r=2,L1=1)",  attack_ml_fra,    {"radial_weight": 2.0, "l1_weight": 1.0, "mode": "projection"}),
    ("ML-FRA proj (r=5,L1=1)",  attack_ml_fra,    {"radial_weight": 5.0, "l1_weight": 1.0, "mode": "projection"}),
    ("ML-FRA probe",            attack_ml_fra,    {"mode": "probe"}),
    ("FRA tuned (r=2,L1=1)",    attack_fra_tuned, {"radial_weight": 2.0, "l1_weight": 1.0}),
    ("FRA tuned (r=5,L1=1)",    attack_fra_tuned, {"radial_weight": 5.0, "l1_weight": 1.0}),
    ("FRA tuned (r=2,L1=0)",    attack_fra_tuned, {"radial_weight": 2.0, "l1_weight": 0.0}),
    ("FRA tuned (r=5,L1=0)",    attack_fra_tuned, {"radial_weight": 5.0, "l1_weight": 0.0}),
    ("PGD (L1)",                attack_fra_tuned, {"radial_weight": 0.0, "l1_weight": 1.0}),
]

print(f"\n{'='*80}")
print("MULTI-LAYER FEATURE RADIAL ATTACK — TUNING")
print(f"{'='*80}")

best_delta = -999
for name, fn, kwargs in configs:
    deltas = []
    t0 = time.time()
    for img_data, lbl in images:
        x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        clean_d = evaluate(net, img_data, lbl, "Synapse", device)
        if clean_d < 0.01: continue
        rg = x.max().item() - x.min().item()
        eps_pix, step_pix = EPSILON * rg, STEP * rg
        x_adv = fn(image_encoder, x, eps_pix, step_pix, STEPS, **kwargs)
        adv_np = x_adv.squeeze(0).cpu().numpy()
        att_d = evaluate(net, adv_np, lbl, "Synapse", device)
        deltas.append(clean_d - att_d)
    dt = time.time() - t0
    md, sd = float(np.mean(deltas)), float(np.std(deltas))
    marker = " ← NEW BEST" if md > best_delta else ""
    if md > best_delta:
        best_delta = md
    print(f"  {name:<35} Δ={md:+.4f}±{sd:.3f}  {dt:.1f}s{marker}")

print(f"\nBest: {best_delta:+.4f}")

# Save
out_path = os.path.join(OUTPUT_DIR, "ml_fra_tuning.json")
with open(out_path, "w") as f:
    json.dump({"best_delta": best_delta, "results": []}, f)
print(f"Saved to {out_path}")
del net, source_sam; torch.cuda.empty_cache()
print("Done")
