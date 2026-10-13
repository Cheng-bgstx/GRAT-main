#!/usr/bin/env python3
"""
PAID-FRA FINAL: Two-Component Attack Derived from Phase-Amplitude Theory.

THEORETICAL FRAMEWORK:

  PAID decomposes feature information:
    I(F) = I_Φ(Φ) + I_A(A) + ε

  where Φ = phase of FFT(F) encodes spatial structure (CROSS-MODEL INVARIANT)
        A = |FFT(F)| encodes frequency energy (IMPORTANCE WEIGHTING)

  The cross-model invariant is the DIRECTION of feature vectors f/|f|.
  This direction is determined by the collective phase structure of feature maps.

  Two mechanisms are needed for optimal attack:

  1. DIRECTIONAL TARGETING (uses Φ):
     L_dir = -Σ_l w_l · cos(f_l_adv, f_l_clean)
     Directly attacks the cross-model invariant by maximizing angular deviation.
     This IS the phase attack — feature direction IS the phase information.

  2. STRUCTURED EXPLORATION (uses A):
     L_exp = Σ_l w_l · |drift_l - 2 · probe_l|
     Provides non-vanishing gradient via structured FFT-domain probe.
     The probe uses amplitude A and phase Φ from clean features.

  Where w_l ∝ E[A_l] = E[|FFT(f_l_clean)|] are the AMPLITUDE weights from PAID.

  KEY INSIGHT: Unlike SE-CGR where A-modulation and Φ-carrier are redundant
  (because the gradient already captures them), here:
    - Φ is used via cosine similarity → EXPLICITLY targets the invariant direction
    - A is used via layer weights → EXPLICITLY prioritizes important layers
    - The FFT probe uses A and Φ for structured exploration

  ABLATION PREDICTIONS (each component should measurably contribute):
    1. Remove A-weighting (w_l = uniform)        → Δ DECREASES (A matters)
    2. Remove Φ-targeting (no cosine term)        → Δ DECREASES (Φ matters)
    3. Remove probe structure (random phase)       → Δ DECREASES (A/Φ structure matters)
    4. Remove both A and Φ (uniform + no cosine)   → Δ DECREASES most
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
    """
    w_l = E[|FFT(f_l_clean)|] — AMPLITUDE A from PAID.

    These weights prioritize layers whose feature maps carry more
    frequency-domain energy, which corresponds to more information
    in the PAID decomposition.
    """
    energies = []
    for i in range(num_layers):
        idx = i + pre_layer - num_layers
        f_map = targets_layer[idx]
        fft = torch.fft.fftn(f_map, dim=(-2, -1), norm="ortho")
        energies.append(torch.abs(fft).mean().item())
    total = sum(energies) + 1e-12
    return [e / total for e in energies]


def synthesize_probe_from_clean(F_clean, use_amplitude=True, use_phase=True):
    """
    Synthesize probe from clean feature map using A and Φ.

    F_clean: (C, H, W) feature map

    use_amplitude=True:  Sigma = λ*A + γ (amplitude-weighted noise)
    use_amplitude=False: Sigma = constant (flat spectrum)

    use_phase=True:   carrier = exp(j*Φ_clean) (phase-aligned)
    use_phase=False:  carrier = exp(j*rand_phase) (random phase)
    """
    G_clean = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
    A = torch.abs(G_clean)
    Phase_clean = torch.angle(G_clean)

    if use_amplitude:
        spectrum_energy = torch.mean(A)
        lambda_val = 0.1 / (spectrum_energy + 1e-6)
        Sigma_freq = lambda_val * A + 1e-4
    else:
        Sigma_freq = torch.ones_like(A) * 0.01

    if use_phase:
        carrier = torch.exp(1j * Phase_clean)
    else:
        rand_phase = torch.rand_like(Phase_clean) * 2 * np.pi - np.pi
        carrier = torch.exp(1j * rand_phase)

    N_white = torch.randn_like(A)
    fft_probe = (N_white * Sigma_freq) * carrier
    return torch.fft.ifftn(fft_probe, dim=(-2, -1), norm="ortho").real


def attack_paid_final(image_encoder, x, eps_pix, step_pix, steps,
                      use_amp_weight=True, use_cos_loss=True,
                      use_amp_probe=True, use_phase_probe=True):
    """
    PAID-FRA FINAL: Two-component attack.

    Component 1 (Φ-targeting): Amplitude-weighted cosine similarity
      - Directly attacks feature direction = phase information = cross-model invariant

    Component 2 (A/Φ exploration): Structured FFT probe
      - Provides non-vanishing gradient for bootstrap
      - Uses amplitude A and phase Φ from clean features

    Ablation controls:
      use_amp_weight:   toggle amplitude-weighted layer contribution
      use_cos_loss:     toggle cosine similarity (phase targeting)
      use_amp_probe:    toggle amplitude modulation in probe
      use_phase_probe:  toggle phase carrier in probe
    """
    max_val, min_val = x.max(), x.min()

    with torch.no_grad():
        targets, targets_layer = image_encoder(x)

    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)

    # Amplitude weights (PAID A)
    if use_amp_weight:
        amp_weights = compute_amplitude_weights(targets_layer, pre_layer, num_layers)
    else:
        amp_weights = [1.0 / num_layers] * num_layers

    # Radial directions for cosine similarity (PAID Φ)
    d_radials = []
    for i in range(num_layers):
        idx = i + pre_layer - num_layers
        f_flat = targets_layer[idx].detach().view(1, -1)
        d_radials.append(F.normalize(f_flat, dim=1))

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    for step_idx in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)

        # ── Component 1: L1 on final features (base loss) ──
        loss = criterion_l1(features * 0.9999, targets)

        # ── Component 2: Cosine similarity (Φ-targeting) ──
        if use_cos_loss:
            cos_term = 0.0
            for i in range(num_layers):
                idx = i + pre_layer - num_layers
                f_adv_flat = features_layer[idx].view(1, -1)
                f_norm = f_adv_flat.norm(dim=1) + 1e-8
                cos_sim = (f_adv_flat * d_radials[i]).sum(dim=1) / f_norm
                cos_term -= amp_weights[i] * cos_sim.mean()
            loss += 0.1 * cos_term / num_layers

        # ── Component 3: Structured FFT probe (A/Φ exploration) ──
        import random
        indices = random.sample(range(num_layers), n_pick)
        fd_list = []
        for idx in indices:
            layer_idx = idx + pre_layer - num_layers
            F_clean_i = targets_layer[layer_idx]
            F_adv_i = features_layer[layer_idx]

            probe = synthesize_probe_from_clean(
                F_clean_i, use_amplitude=use_amp_probe, use_phase=use_phase_probe)
            drift = F_adv_i * 0.999 - F_clean_i
            fd_list.append((drift - 2.0 * probe))

        fd_stack = torch.stack(fd_list)
        selected_t = torch.stack(
            [targets_layer[i + pre_layer - num_layers] for i in indices])
        im = torch.stack([selected_t[j].abs().mean()
                         for j in range(len(selected_t))])
        im = im.view(-1, *([1] * (fd_stack.dim() - 1)))
        probe_loss = torch.abs(torch.mean(fd_stack / (im + 1e-6), dim=0)).mean()
        loss += probe_loss

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


# ── Experiment ────────────────────────────────────────────────────────────

budgets = [
    ("eps=8/255",  8.0/255.0,  0.8/255.0,  20),
    ("eps=16/255", 16.0/255.0, 1.6/255.0,  20),
    ("eps=24/255", 24.0/255.0, 2.4/255.0,  20),
]

# ABLATION: Test each component independently
configs = [
    # Full method
    ("PAID-Final FULL",        attack_paid_final,
     {"use_amp_weight": True, "use_cos_loss": True,
      "use_amp_probe": True, "use_phase_probe": True}),
    # Remove A-weighting (test if amplitude matters)
    ("  - A-weight (uniform w)", attack_paid_final,
     {"use_amp_weight": False, "use_cos_loss": True,
      "use_amp_probe": True, "use_phase_probe": True}),
    # Remove Φ-targeting (test if cosine/phase matters)
    ("  - Φ-target (no cosine)", attack_paid_final,
     {"use_amp_weight": True, "use_cos_loss": False,
      "use_amp_probe": True, "use_phase_probe": True}),
    # Remove probe A-modulation
    ("  - Probe A-mod",         attack_paid_final,
     {"use_amp_weight": True, "use_cos_loss": True,
      "use_amp_probe": False, "use_phase_probe": True}),
    # Remove probe Φ-carrier
    ("  - Probe Φ-carrier",     attack_paid_final,
     {"use_amp_weight": True, "use_cos_loss": True,
      "use_amp_probe": True, "use_phase_probe": False}),
    # Remove both A and Φ from probe
    ("  - Probe A&Φ (random)",  attack_paid_final,
     {"use_amp_weight": True, "use_cos_loss": True,
      "use_amp_probe": False, "use_phase_probe": False}),
    # Remove everything (just L1 + random probe)
    ("  - All A/Φ (bare L1+probe)", attack_paid_final,
     {"use_amp_weight": False, "use_cos_loss": False,
      "use_amp_probe": False, "use_phase_probe": False}),
    # Reference
    ("SE-CGR (reference)",      attack_secgr, {}),
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
            "budget": budget_name, "method": method_name.strip(),
            "mean_delta": md, "std_delta": sd,
            "drift_radial_ratio": mr, "time": dt,
        })
        print(f"  {method_name:<35} Δ={md:+.4f}±{sd:.3f}  rr={mr:.3f}  {dt:.1f}s")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("PAID-FRA FINAL — Two-Component Ablation")
print(f"{'='*90}")

method_names = [c[0].strip() for c in configs]
print(f"{'Method':<35} {'eps=8/255':>10} {'eps=16/255':>10} {'eps=24/255':>10}")
print("-" * 78)
for method_name in method_names:
    vals = []
    for b in ["eps=8/255", "eps=16/255", "eps=24/255"]:
        matches = [r for r in all_results if r["method"] == method_name and r["budget"] == b]
        vals.append(f"{matches[0]['mean_delta']:+.4f}" if matches else "       N/A")
    if any(v != "       N/A" for v in vals):
        print(f"  {method_name:<35} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

# ── Component contribution analysis ────────────────────────────────────────
print(f"\n{'='*90}")
print("COMPONENT CONTRIBUTION ANALYSIS")
print(f"{'='*90}")

results_map = {}
for r in all_results:
    results_map[(r["method"], r["budget"])] = r["mean_delta"]

for budget in ["eps=8/255", "eps=16/255", "eps=24/255"]:
    print(f"\n  [{budget}]")
    full = results_map.get(("PAID-Final FULL", budget), 0)
    no_aw = results_map.get(("- A-weight (uniform w)", budget), 0)
    no_cos = results_map.get(("- Φ-target (no cosine)", budget), 0)
    no_amp = results_map.get(("- Probe A-mod", budget), 0)
    no_phase = results_map.get(("- Probe Φ-carrier", budget), 0)
    no_ap = results_map.get(("- Probe A&Φ (random)", budget), 0)
    bare = results_map.get(("- All A/Φ (bare L1+probe)", budget), 0)
    secgr = results_map.get(("SE-CGR (reference)", budget), 0)

    a_weight_effect = full - no_aw
    cos_effect = full - no_cos
    probe_amp_effect = full - no_amp
    probe_phase_effect = full - no_phase
    probe_struct_effect = full - no_ap
    total_a_phi_effect = full - bare

    print(f"    FULL = {full:+.4f}")
    print(f"    A-weight contribution:       {a_weight_effect:+.4f}  (FULL - uniform w)")
    print(f"    Φ-target contribution:       {cos_effect:+.4f}  (FULL - no cosine)")
    print(f"    Probe A-mod contribution:    {probe_amp_effect:+.4f}  (FULL - flat spectrum)")
    print(f"    Probe Φ-carrier contribution:{probe_phase_effect:+.4f}  (FULL - random phase)")
    print(f"    Probe A+Φ structure:         {probe_struct_effect:+.4f}  (FULL - random probe)")
    print(f"    Total A+Φ contribution:      {total_a_phi_effect:+.4f}  (FULL - bare)")
    print(f"    vs SE-CGR:                   {full - secgr:+.4f}")

out_path = os.path.join(OUTPUT_DIR, "paid_final_ablation.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
del net, source_sam
torch.cuda.empty_cache()
print("Done")
