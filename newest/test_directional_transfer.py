#!/usr/bin/env python3
"""
Directional Transfer Experiment: Validate that RADIAL direction is the
cross-model invariant that enables attack transferability.

Theory (CODD):
  - Feature drift Delta_F in the complex spectral plane decomposes into:
    radial = Re(Delta * conj(carrier)) * carrier      (stable across models)
    tangential = Im(Delta * conj(carrier)) * (j*carrier)  (random across models)
  - Cross-model radial ratio: 0.76-0.90
  - Therefore: attacking radial direction → transfers
               attacking tangential direction → does NOT transfer

Experiment:
  6 attack types, all attacking SOURCE SAM encoder, evaluating on SAMed:
    1. Radial (det): maximize radial drift energy ONLY
    2. Tangential (det): maximize tangential drift energy ONLY
    3. Isotropic (det): maximize total drift energy
    4. Misaligned (det): maximize drift along random directions
    5. SE-CGR (random): original random-noise radial probe (reference)
    6. PGD (pixel): standard pixel-space PGD (baseline)

Key: ALL deterministic probes use use_random_noise=False → GENUINE directional
     separation (unlike matched_ablation_v3 which used random noise).

Prediction: Radial >> Tangential on SAMed (transfer via invariant).
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from theory import synthesize_codd_probe, decompose_drift
from attack_codd import AttackConfig, _get_edge_map, _return_eta, _return_mean_value
from evaluate import load_sam_encoder, load_samed_model, evaluate_samed

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

# ── Load models ──────────────────────────────────────────────────────────
net = load_samed_model("vit_b", SAM_CKPT, os.path.join(CKPT_BASE, CKPT_REL),
                       IMAGE_SIZE, 8, device=device).to(device)
source_sam = load_sam_encoder("vit_b", SAM_CKPT, IMAGE_SIZE, device=device)
image_encoder = source_sam.image_encoder

# ── Load data ────────────────────────────────────────────────────────────
from exp_paid_optimal_attack import load_synapse_slices, evaluate
images = load_synapse_slices(5)
print(f"Loaded {len(images)} images")


# ═══════════════════════════════════════════════════════════════════════════
# Directional attack (deterministic probe, use_random_noise=False)
# ═══════════════════════════════════════════════════════════════════════════

def attack_directional(image_encoder, x, eps_pix, step_pix, steps,
                       direction="radial", directional_weight=1.0):
    """
    Attack maximizing directional drift energy in the complex spectral plane.

    Uses synthesize_codd_probe(use_random_noise=False) to extract the
    ACTUAL radial/tangential component of the current drift, then maximizes
    its energy via gradient ascent. This gives GENUINE directional separation.
    """
    max_val, min_val = x.max(), x.min()
    B, C, H, W = x.shape

    with torch.no_grad():
        targets, targets_layer = image_encoder(x)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)

    for step_idx in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)

        # L_org: push features away from clean
        loss_org = criterion_l1(features * 0.9999, targets)

        # L_dir: maximize directional drift energy
        import random
        indices = random.sample(range(num_layers), n_pick)
        dir_energy = 0.0
        for idx in indices:
            F_clean_i = targets_layer[idx + pre_layer - num_layers]
            F_adv_i = features_layer[idx + pre_layer - num_layers]

            probe = synthesize_codd_probe(
                F_clean=F_clean_i.unsqueeze(0),
                F_adv=F_adv_i.unsqueeze(0),
                direction=direction,
                use_random_noise=False,   # ← DETERMINISTIC decomposition
                detach_adv=False,          # ← gradients flow through probe
            )
            dir_energy += (probe ** 2).mean()
        dir_energy = dir_energy / n_pick

        loss = loss_org - directional_weight * dir_energy

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)  # PSAA edge guidance
        grad = grad / (torch.norm(grad, p=1) + 1e-12)

        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# ═══════════════════════════════════════════════════════════════════════════
# SE-CGR style attack (random noise, radial probe) — reference
# ═══════════════════════════════════════════════════════════════════════════

def attack_secgr_random(image_encoder, x, eps_pix, step_pix, steps):
    """Original SE-CGR: random noise modulated by A, carried by Phase_clean."""
    max_val, min_val = x.max(), x.min()

    with torch.no_grad():
        targets, targets_layer = image_encoder(x)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)

    for step_idx in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)

        loss_org = criterion_l1(features * 0.9999, targets)

        import random
        indices = random.sample(range(num_layers), n_pick)
        feature_difference_list = []
        for idx in indices:
            F_clean_i = targets_layer[idx + pre_layer - num_layers]
            F_adv_i = features_layer[idx + pre_layer - num_layers]

            probe = synthesize_codd_probe(
                F_clean=F_clean_i.unsqueeze(0),
                F_adv=F_adv_i.unsqueeze(0),
                direction="radial",
                use_random_noise=True,    # ← RANDOM noise (original SE-CGR)
            )
            feature_difference = (F_adv_i.unsqueeze(0) * 0.999 - F_clean_i.unsqueeze(0)) - 2.0 * probe
            feature_difference_list.append(feature_difference.squeeze(0))

        fd_stack = torch.stack(feature_difference_list)
        input_mean = _return_mean_value(torch.stack(
            [targets_layer[i + pre_layer - num_layers] for i in indices]))
        if len(input_mean.shape) > len(fd_stack.shape):
            input_mean = input_mean.view(fd_stack.shape[0], 1, 1, 1)
        fd_norm = fd_stack / (input_mean + 1e-6)
        l1_losses = (torch.abs(torch.mean(fd_norm, dim=0))).mean()

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
# Standard PGD (pixel space) — baseline
# ═══════════════════════════════════════════════════════════════════════════

def attack_pgd_pixel(image_encoder, x, eps_pix, step_pix, steps):
    """Standard PGD in pixel space."""
    max_val, min_val = x.max(), x.min()

    with torch.no_grad():
        targets, _ = image_encoder(x)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    momentum = torch.zeros_like(x)

    def L(f):
        return sum(F.l1_loss(a, b.detach()) for a, b in zip(f, targets))

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        f, _ = image_encoder(x_adv)
        loss = L(f)
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# ═══════════════════════════════════════════════════════════════════════════
# Drift diagnostic: measure radial_ratio of attack-induced feature drift
# ═══════════════════════════════════════════════════════════════════════════

def measure_attack_drift_radiality(image_encoder, x_clean, x_adv):
    """Measure radial_ratio of the attack-induced feature drift."""
    with torch.no_grad():
        f_clean, layers_clean = image_encoder(x_clean)
        f_adv, layers_adv = image_encoder(x_adv)

    pre_layer = int(len(layers_clean) * 0.9)
    num_layers = min(6, pre_layer)

    ratios = []
    for i in range(num_layers):
        idx = i + pre_layer - num_layers
        stats = decompose_drift(layers_clean[idx], layers_adv[idx])
        ratios.append(stats.radial_ratio)

    return float(np.mean(ratios)), ratios


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

configs = [
    ("Radial (det)",        "radial",     "det"),
    ("Tangential (det)",    "tangential", "det"),
    ("Isotropic (det)",     "isotropic",  "det"),
    ("Misaligned (det)",    "misaligned", "det"),
    ("SE-CGR (random)",     "radial",     "random"),
    ("PGD (pixel)",         "pgd",        "pgd"),
]

print(f"\n{'='*80}")
print("DIRECTIONAL TRANSFER EXPERIMENT")
print(f"Attack: source SAM encoder  →  Evaluate: SAMed (fine-tuned)")
print(f"Budget: ε={EPSILON:.4f}, step={STEP:.4f}, steps={STEPS}")
print(f"{'='*80}")

all_results = []

for name, direction, mode in configs:
    deltas = []
    drift_ratios = []
    t0 = time.time()

    for img_data, lbl in images:
        x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        clean_d = evaluate(net, img_data, lbl, "Synapse", device)
        if clean_d < 0.01:
            continue
        rg = x.max().item() - x.min().item()
        eps_pix = EPSILON * rg
        step_pix = STEP * rg

        if mode == "pgd":
            x_adv = attack_pgd_pixel(image_encoder, x, eps_pix, step_pix, STEPS)
        elif mode == "random":
            x_adv = attack_secgr_random(image_encoder, x, eps_pix, step_pix, STEPS)
        else:
            x_adv = attack_directional(image_encoder, x, eps_pix, step_pix, STEPS,
                                       direction=direction)

        adv_np = x_adv.squeeze(0).cpu().numpy()
        att_d = evaluate(net, adv_np, lbl, "Synapse", device)
        deltas.append(clean_d - att_d)

        # Measure drift radiality
        rr, _ = measure_attack_drift_radiality(image_encoder, x, x_adv)
        drift_ratios.append(rr)

    dt = time.time() - t0
    md, sd = float(np.mean(deltas)), float(np.std(deltas))
    mr = float(np.mean(drift_ratios)) if drift_ratios else 0.0
    all_results.append({
        "name": name, "direction": direction, "mode": mode,
        "mean_delta": md, "std_delta": sd,
        "mean_drift_radial_ratio": mr,
        "time": dt,
    })
    print(f"  {name:<25} ΔDice={md:+.4f}±{sd:.3f}  drift_radial={mr:.3f}  {dt:.1f}s")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"{'Method':<25} {'ΔDice':>10} {'Drift RR':>10} {'Interpretation'}")
print("-" * 70)
for r in all_results:
    interp = ""
    if "Radial" in r["name"] and "det" in r["mode"]:
        interp = "← should be HIGHEST (invariant transfers)"
    elif "Tangential" in r["name"]:
        interp = "← should be LOWEST (random, no transfer)"
    elif "Isotropic" in r["name"]:
        interp = "← intermediate"
    elif "Misaligned" in r["name"]:
        interp = "← near tangential"
    elif "SE-CGR" in r["name"]:
        interp = "← reference (random noise radial)"
    elif "PGD" in r["name"]:
        interp = "← baseline"
    print(f"  {r['name']:<25} {r['mean_delta']:+.4f}     {r['mean_drift_radial_ratio']:.3f}      {interp}")

# ── Key metric: Radial Premium ──────────────────────────────────────────
radial_det = [r for r in all_results if r["name"] == "Radial (det)"]
tangential_det = [r for r in all_results if r["name"] == "Tangential (det)"]
if radial_det and tangential_det:
    premium = radial_det[0]["mean_delta"] - tangential_det[0]["mean_delta"]
    print(f"\n  Radial Premium = ΔRadial - ΔTangential = {premium:+.4f}")
    print(f"  (Positive = radial transfers better → theory validated)")
    print(f"  (Near zero or negative = theory FAILS)")

# Save results
out_path = os.path.join(OUTPUT_DIR, "directional_transfer.json")
with open(out_path, "w") as f:
    json.dump({
        "config": {"epsilon": EPSILON, "step": STEP, "steps": STEPS},
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, f, indent=2)
print(f"\nSaved to {out_path}")

del net, source_sam
torch.cuda.empty_cache()
print("Done")
