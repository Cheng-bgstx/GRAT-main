#!/usr/bin/env python3
"""
PAID-FRA v5: Hilbert Phase Probe — Optimal Phase Disruption from PAID Theory.

THEORETICAL DERIVATION:

  In the complex plane of FFT(feature_maps), the clean feature at frequency f is:
    F̃_clean(f) = A(f) · exp(j·Φ(f))

  Adding a perturbation δF̃ changes the phase by:
    ΔΦ ≈ Im(δF̃ · conj(F̃_clean)) / |F̃_clean|²  (first-order)

  The direction that MAXIMIZES phase change per unit |δF̃| is:
    δF̃ ∝ j · F̃_clean = A · exp(j·(Φ + π/2))

  This is PERPENDICULAR to F̃_clean in the complex plane — the Hilbert direction.
  In contrast, SE-CGR's carrier aligns with F̃_clean (0 shift), which changes
  AMPLITUDE (not phase), so it's redundant with the gradient.

  The spatial-domain probe is: P = H(F_clean) where H is the Hilbert transform
  (π/2 phase shift at each frequency). This produces a REAL spatial pattern
  that maximally disrupts phase while respecting amplitude structure.

  The key difference from SE-CGR:
    SE-CGR:  probe = IFFT(N_white · A · exp(j·Φ))     → targets amplitude (0 shift)
    v5:      probe = H(A-weighted F_clean)             → targets phase (π/2 shift)

  PAID THEORY PREDICTS:
    1. Hilbert probe (π/2 shift) >> SE-CGR probe (0 shift)
    2. A-weighting matters (unlike in SE-CGR where gradient already captures it)
    3. Both A and Φ components are NON-REDUNDANT and experimentally verifiable

ABLATION:
  1. Full PAID Hilbert:     A-weighted π/2 phase shift probe
  2. No A-weighting:        uniform π/2 phase shift
  3. SE-CGR-style (0 shift): probe aligned with F_clean (amplitude direction)
  4. Random phase:           random direction at each frequency
  5. No probe:               L1 + edge only
  6. SE-CGR:                 reference
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


def hilbert_transform_2d(x):
    """
    2D Hilbert-like transform: π/2 phase shift at each frequency.

    In frequency domain: H(f) = -j · sign(k_x, k_y) · F̃(f)
    where sign(k_x, k_y) = sign(k_x + ε) for non-DC frequencies.

    The result is a REAL signal whose each frequency component is
    phase-shifted by ±π/2 relative to the original.

    This is the DIRECTION OF MAXIMAL PHASE CHANGE at each frequency.
    """
    # FFT
    Fx = torch.fft.fftn(x, dim=(-2, -1), norm="ortho")

    # Build sign function in frequency domain
    H, W = x.shape[-2], x.shape[-1]
    ky = torch.fft.fftfreq(H, device=x.device).view(-1, 1)
    kx = torch.fft.fftfreq(W, device=x.device).view(1, -1)

    # sign(k) — odd function, preserves Hermitian symmetry
    # kx dominates for non-zero kx, else ky
    sign_k = torch.sign(kx + ky + 1e-10)

    # Apply π/2 phase shift: multiply by -j * sign(k)
    # -j = exp(-jπ/2): shifts phase by -π/2 in complex plane
    H_fft = -1j * sign_k.unsqueeze(0).unsqueeze(0) * Fx

    # IFFT back to spatial domain
    return torch.fft.ifftn(H_fft, dim=(-2, -1), norm="ortho").real


def synthesize_hilbert_probe(F_clean, use_amp_weight=True):
    """
    Synthesize deterministic Hilbert probe from clean feature map.

    F_clean: (1, C, H, W) or (1, H, W, C) feature map tensor

    use_amp_weight=True:  weight by amplitude A(f) before IFFT
    use_amp_weight=False: uniform amplitude (Hilbert of normalized features)
    """
    if use_amp_weight:
        # Hilbert transform preserves amplitude structure
        return hilbert_transform_2d(F_clean)

    # Hilbert of direction-only features (unit norm per freq)
    Fx = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
    A = torch.abs(Fx) + 1e-8
    Fx_norm = Fx / A  # unit complex number, phase only

    # Build sign function matching FFT spatial dims
    H_sp, W_sp = Fx.shape[-2], Fx.shape[-1]
    ky = torch.fft.fftfreq(H_sp, device=F_clean.device).view(-1, 1)
    kx = torch.fft.fftfreq(W_sp, device=F_clean.device).view(1, -1)
    sign_k = torch.sign(kx + ky + 1e-10)

    # Broadcast: sign_k is (H, W), need to match Fx_norm which may be (B,C,H,W) or (B,H,W,C)
    ndim = Fx_norm.dim()
    if ndim == 4:
        # Standard: (B, C, H, W) — sign_k broadcasts as (1, 1, H, W)
        sign_k = sign_k.view(1, 1, H_sp, W_sp)
    else:
        sign_k = sign_k.view(1, H_sp, W_sp)

    H_norm = -1j * sign_k * Fx_norm
    probe = torch.fft.ifftn(H_norm, dim=(-2, -1), norm="ortho").real
    probe = probe * F_clean.abs().mean()
    return probe


def synthesize_se_cgr_style_probe(F_clean, use_random_phase=False):
    """
    SE-CGR style probe for comparison.

    use_random_phase=False: probe aligned with F_clean (0 phase shift = amplitude dir)
    use_random_phase=True:  random phase (SE-CGR without A-weighting)
    """
    Fx = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
    A = torch.abs(Fx)
    Phi = torch.angle(Fx)

    if use_random_phase:
        carrier = torch.exp(1j * torch.rand_like(Phi) * 2 * np.pi - np.pi)
    else:
        carrier = torch.exp(1j * Phi)  # aligned with clean = 0 shift

    N_white = torch.randn_like(A)
    fft_probe = N_white * A * carrier
    probe = torch.fft.ifftn(fft_probe, dim=(-2, -1), norm="ortho").real
    return probe


def compute_amplitude_weights(targets_layer, pre_layer, num_layers):
    """Layer weights from amplitude A (PAID)."""
    energies = []
    for i in range(num_layers):
        idx = i + pre_layer - num_layers
        f_map = targets_layer[idx]
        fft = torch.fft.fftn(f_map, dim=(-2, -1), norm="ortho")
        energies.append(torch.abs(fft).mean().item())
    total = sum(energies) + 1e-12
    return [e / total for e in energies]


def attack_hilbert_probe(image_encoder, x, eps_pix, step_pix, steps,
                         probe_type="hilbert_amp"):
    """
    Attack with controllable probe type.

    probe_type:
      "hilbert_amp"  — Hilbert probe with amplitude weighting (FULL PAID)
      "hilbert_uni"  — Hilbert probe without amplitude weighting
      "secgr_0shift" — SE-CGR style (0 phase shift = amplitude direction)
      "secgr_random" — Random phase probe (SE-CGR without structure)
      "none"         — No probe (L1 + edge only)
    """
    max_val, min_val = x.max(), x.min()

    with torch.no_grad():
        targets, targets_layer = image_encoder(x)

    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)
    amp_weights = compute_amplitude_weights(targets_layer, pre_layer, num_layers)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    for step_idx in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)

        loss_org = criterion_l1(features * 0.9999, targets)

        if probe_type == "none":
            loss = loss_org
        else:
            import random
            indices = random.sample(range(num_layers), n_pick)
            fd_list = []
            for idx in indices:
                layer_idx = idx + pre_layer - num_layers
                F_clean_i = targets_layer[layer_idx]
                F_adv_i = features_layer[layer_idx]

                if probe_type == "hilbert_amp":
                    probe = synthesize_hilbert_probe(
                        F_clean_i.unsqueeze(0), use_amp_weight=True)
                elif probe_type == "hilbert_uni":
                    probe = synthesize_hilbert_probe(
                        F_clean_i.unsqueeze(0), use_amp_weight=False)
                elif probe_type == "secgr_0shift":
                    probe = synthesize_se_cgr_style_probe(
                        F_clean_i.unsqueeze(0), use_random_phase=False)
                elif probe_type == "secgr_random":
                    probe = synthesize_se_cgr_style_probe(
                        F_clean_i.unsqueeze(0), use_random_phase=True)

                drift = F_adv_i.unsqueeze(0) * 0.999 - F_clean_i.unsqueeze(0)
                fd_list.append((drift - 2.0 * probe).squeeze(0))

            fd_stack = torch.stack(fd_list)
            selected_t = torch.stack(
                [targets_layer[i + pre_layer - num_layers] for i in indices])

            # Normalize by input mean (same as SE-CGR)
            im = torch.stack([selected_t[j].abs().mean()
                             for j in range(len(selected_t))])
            im = im.view(-1, *([1] * (fd_stack.dim() - 1)))
            l1_losses = torch.abs(torch.mean(fd_stack / (im + 1e-6), dim=0)).mean()
            loss = loss_org + l1_losses

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)

        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


def attack_secgr(image_encoder, x, eps_pix, step_pix, steps):
    """SE-CGR reference (original implementation)."""
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
    # PAID Hilbert probes (π/2 phase shift — targets PHASE)
    ("PAID Hilbert (A-weighted)",   attack_hilbert_probe,
     {"probe_type": "hilbert_amp"}),
    ("PAID Hilbert (uniform)",      attack_hilbert_probe,
     {"probe_type": "hilbert_uni"}),
    # SE-CGR style probes (0 phase shift — targets AMPLITUDE)
    ("SE-CGR style (0-shift)",      attack_hilbert_probe,
     {"probe_type": "secgr_0shift"}),
    ("SE-CGR style (random Φ)",     attack_hilbert_probe,
     {"probe_type": "secgr_random"}),
    # Baselines
    ("No probe (L1 + edge)",        attack_hilbert_probe,
     {"probe_type": "none"}),
    ("SE-CGR (original)",           attack_secgr, {}),
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
print("PAID Hilbert Probe — π/2 Phase Shift (Targets Phase Φ)")
print(f"{'='*85}")
print(f"{'Method':<30} {'eps=8/255':>10} {'eps=16/255':>10} {'eps=24/255':>10} {'rr':>8}")
print("-" * 72)

for method_name in ["PAID Hilbert (A-weighted)", "PAID Hilbert (uniform)",
                     "SE-CGR style (0-shift)", "SE-CGR style (random Φ)",
                     "No probe (L1 + edge)", "SE-CGR (original)"]:
    vals = []
    rr_val = 0
    for b in ["eps=8/255", "eps=16/255", "eps=24/255"]:
        matches = [r for r in all_results if r["method"] == method_name and r["budget"] == b]
        if matches:
            vals.append(f"{matches[0]['mean_delta']:+.4f}")
            rr_val = matches[0]["drift_radial_ratio"]
    if len(vals) >= 2:
        print(f"  {method_name:<30} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {rr_val:>8.3f}")

# Key comparisons
results_by_name = {}
for r in all_results:
    results_by_name[(r["method"], r["budget"])] = r["mean_delta"]

print(f"\n── Ablation Analysis ──")
for budget in ["eps=8/255", "eps=16/255", "eps=24/255"]:
    print(f"\n  {budget}:")
    hilbert_amp = results_by_name.get(("PAID Hilbert (A-weighted)", budget), 0)
    hilbert_uni = results_by_name.get(("PAID Hilbert (uniform)", budget), 0)
    secgr_0 = results_by_name.get(("SE-CGR style (0-shift)", budget), 0)
    secgr_rand = results_by_name.get(("SE-CGR style (random Φ)", budget), 0)
    no_probe = results_by_name.get(("No probe (L1 + edge)", budget), 0)
    secgr_orig = results_by_name.get(("SE-CGR (original)", budget), 0)

    a_effect = hilbert_amp - hilbert_uni
    phase_effect = hilbert_amp - secgr_0
    print(f"    Amplitude weighting effect (A): {a_effect:+.4f}")
    print(f"    Phase targeting effect (π/2 vs 0): {phase_effect:+.4f}")
    print(f"    vs No probe: {hilbert_amp - no_probe:+.4f}")
    print(f"    vs SE-CGR original: {hilbert_amp - secgr_orig:+.4f}")

out_path = os.path.join(OUTPUT_DIR, "hilbert_probe_results.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
del net, source_sam
torch.cuda.empty_cache()
print("Done")
