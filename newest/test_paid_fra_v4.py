#!/usr/bin/env python3
"""
PAID-FRA v4: Deterministic radial-direction probe.

THEORETICAL INSIGHT:

  SE-CGR's probe works at low budgets because it provides a non-zero
  structured gradient target: sign(drift - 2*probe) ≈ -sign(probe) when f_adv≈f_clean.

  But SE-CGR's probe is RANDOM (white noise × A × exp(j*Phase)). The ablation
  showed A-weighting and phase carrier don't matter — the probe structure comes
  purely from IFFT(random noise) creating spatial correlation.

  PAID-FRA v4 replaces the random probe with a DETERMINISTIC RADIAL PROBE:
    D_l = flatten(F_l_clean) / |flatten(F_l_clean)|  → reshape to (C,H,W)
    probe_l = -α * D_l  (anti-radial direction)

  This directly targets the RADIAL direction in feature space, which IS the
  cross-model invariant. The loss:

    L = L1(f_adv, f_clean) + Σ_l w_l * mean(|drift_l + α_l * D_l|)

  Where:
  - w_l ∝ E[|FFT(F_l_clean)|] (amplitude A from PAID)
  - D_l is the radial direction map (phase Φ from PAID)
  - α_l adapts to the current drift magnitude

  This is the FIRST method that:
  1. Uses amplitude A for layer weighting (explicit PAID A)
  2. Uses phase Φ via radial direction D_l (explicit PAID Φ)
  3. Has non-vanishing gradient at all ranges (deterministic probe target)
  4. Directly targets the cross-model invariant (radial direction)

ABLATION VARIANTS:
  - Full PAID-FRA v4: radial probe + amplitude weights
  - Uniform weights: radial probe only
  - Random probe: SE-CGR style random target (control)
  - Cosine only: no probe, cosine similarity (baseline)
  - SE-CGR: reference
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


def compute_amplitude_weights(targets_layer, pre_layer, num_layers):
    """w_l = E[|FFT(f_l_clean)|] — amplitude A from PAID."""
    amp_energies = []
    for i in range(num_layers):
        idx = i + pre_layer - num_layers
        f_map = targets_layer[idx]
        fft = torch.fft.fftn(f_map, dim=(-2, -1), norm="ortho")
        A = torch.abs(fft)
        amp_energies.append(A.mean().item())
    total = sum(amp_energies) + 1e-12
    return [e / total for e in amp_energies]


def attack_paid_fra_v4(image_encoder, x, eps_pix, step_pix, steps,
                       use_amp_weight=True, probe_type="radial"):
    """
    PAID-FRA v4: Deterministic probe targeting radial direction.

    probe_type="radial":  probe = -α * D_l where D_l is radial direction map
    probe_type="random":  probe = IFFT(randn) — random structured (SE-CGR baseline)
    probe_type="cosine":  cosine similarity only, no probe
    """
    max_val, min_val = x.max(), x.min()

    with torch.no_grad():
        targets, targets_layer = image_encoder(x)

    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)

    # Amplitude weights from PAID
    if use_amp_weight:
        amp_weights = compute_amplitude_weights(targets_layer, pre_layer, num_layers)
    else:
        amp_weights = [1.0 / num_layers] * num_layers

    # Pre-compute clean feature maps and radial direction maps
    F_clean_list = []
    D_radial_list = []  # radial direction maps
    for i in range(num_layers):
        idx = i + pre_layer - num_layers
        F_c = targets_layer[idx].detach()  # (C, H, W)
        F_clean_list.append(F_c)
        # Radial direction: flatten → normalize → reshape back
        f_flat = F_c.view(-1)
        d_flat = F.normalize(f_flat, dim=0)
        D_radial_list.append(d_flat.view_as(F_c))

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    for step_idx in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)

        # L_org: push features away from clean
        loss_org = criterion_l1(features * 0.9999, targets)

        loss = loss_org

        if probe_type in ("radial", "random"):
            probe_loss = 0.0
            for i in range(num_layers):
                idx = i + pre_layer - num_layers
                F_adv_i = features_layer[idx]  # (C, H, W)
                F_clean_i = F_clean_list[i]

                if probe_type == "radial":
                    # Deterministic radial probe
                    # α_i = current drift magnitude at this layer
                    drift_i = F_adv_i.detach() - F_clean_i
                    alpha_i = drift_i.abs().mean() + 1e-4
                    probe_i = -alpha_i * D_radial_list[i]
                else:
                    # Random structured probe (SE-CGR baseline)
                    F_np = F_clean_i.cpu().numpy()
                    fft = np.fft.fftn(F_np, axes=(-2, -1), norm="ortho")
                    N_white = np.random.randn(*fft.shape).astype(np.float32)
                    fft_noise = N_white + 1j * np.random.randn(*fft.shape).astype(np.float32)
                    probe_np = np.fft.ifftn(fft_noise, axes=(-2, -1), norm="ortho").real
                    probe_i = torch.from_numpy(probe_np).to(device) * 0.01

                # Encourage drift to align with probe direction
                diff = (F_adv_i - F_clean_i) - 2.0 * probe_i
                norm_factor = F_clean_i.abs().mean() + 1e-6
                probe_loss += amp_weights[i] * (diff / norm_factor).abs().mean()

            loss += probe_loss

        elif probe_type == "cosine":
            # Cosine similarity loss (no probe)
            cos_loss = 0.0
            for i in range(num_layers):
                idx = i + pre_layer - num_layers
                f_adv_flat = features_layer[idx].view(1, -1)
                d_r = D_radial_list[i].view(1, -1)
                f_adv_norm = f_adv_flat.norm(dim=1) + 1e-8
                cos_sim = (f_adv_flat * d_r).sum(dim=1) / f_adv_norm
                cos_loss -= amp_weights[i] * cos_sim.mean()
            loss += cos_loss

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
    radial_energy = (proj ** 2).item()
    total_energy = (drift.norm() ** 2).item()
    return radial_energy / (total_energy + 1e-12)


# ── Experiment ────────────────────────────────────────────────────────────

budgets = [
    ("eps=8/255",  8.0/255.0,  0.8/255.0,  20),
    ("eps=16/255", 16.0/255.0, 1.6/255.0,  20),
    ("eps=24/255", 24.0/255.0, 2.4/255.0,  20),
]

configs = [
    ("PAID-FRA v4 radial (amp)",  attack_paid_fra_v4,
     {"use_amp_weight": True, "probe_type": "radial"}),
    ("PAID-FRA v4 radial (uni)",  attack_paid_fra_v4,
     {"use_amp_weight": False, "probe_type": "radial"}),
    ("PAID-FRA v4 random (amp)",  attack_paid_fra_v4,
     {"use_amp_weight": True, "probe_type": "random"}),
    ("PAID-FRA v4 cosine (amp)",  attack_paid_fra_v4,
     {"use_amp_weight": True, "probe_type": "cosine"}),
    ("SE-CGR (reference)",        attack_secgr, {}),
]

all_results = []

for budget_name, eps, step, steps in budgets:
    print(f"\n{'─'*70}")
    print(f"  {budget_name}")
    print(f"{'─'*70}")
    for method_name, attack_fn, kwargs in configs:
        deltas = []
        drift_rrs = []
        t0 = time.time()
        for img_data, lbl in images:
            x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
            clean_d = evaluate(net, img_data, lbl, "Synapse", device)
            if clean_d < 0.01:
                continue
            rg = x.max().item() - x.min().item()
            x_adv = attack_fn(image_encoder, x, eps * rg, step * rg, steps, **kwargs)
            adv_np = x_adv.squeeze(0).cpu().numpy()
            att_d = evaluate(net, adv_np, lbl, "Synapse", device)
            deltas.append(clean_d - att_d)
            drift_rrs.append(measure_drift_radial_ratio(image_encoder, x, x_adv))
        dt = time.time() - t0
        md, sd = float(np.mean(deltas)), float(np.std(deltas))
        mr = float(np.mean(drift_rrs)) if drift_rrs else 0.0
        all_results.append({
            "budget": budget_name, "method": method_name,
            "mean_delta": md, "std_delta": sd,
            "drift_radial_ratio": mr, "time": dt,
        })
        print(f"  {method_name:<30} Δ={md:+.4f}±{sd:.3f}  rr={mr:.3f}  {dt:.1f}s")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print("PAID-FRA v4 — Deterministic Radial Probe")
print(f"{'='*85}")
print(f"{'Method':<30} {'eps=8/255':>10} {'eps=16/255':>10} {'eps=24/255':>10} {'rr':>8}")
print("-" * 72)
for method_name in ["PAID-FRA v4 radial (amp)", "PAID-FRA v4 radial (uni)",
                     "PAID-FRA v4 random (amp)", "PAID-FRA v4 cosine (amp)",
                     "SE-CGR (reference)"]:
    vals = []
    rr_val = 0
    for b in ["eps=8/255", "eps=16/255", "eps=24/255"]:
        matches = [r for r in all_results if r["method"] == method_name and r["budget"] == b]
        if matches:
            vals.append(f"{matches[0]['mean_delta']:+.4f}")
            rr_val = matches[0]["drift_radial_ratio"]
    if len(vals) >= 2:
        print(f"  {method_name:<30} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {rr_val:>8.3f}")

best = {}
for r in all_results:
    b = r["budget"]
    if b not in best or r["mean_delta"] > best[b][0]:
        best[b] = (r["mean_delta"], r["method"])
print(f"\nBest per budget:")
for b in ["eps=8/255", "eps=16/255", "eps=24/255"]:
    if b in best:
        print(f"  {b}: {best[b][1]} at {best[b][0]:+.4f}")

out_path = os.path.join(OUTPUT_DIR, "paid_fra_v4_results.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
del net, source_sam
torch.cuda.empty_cache()
print("Done")
