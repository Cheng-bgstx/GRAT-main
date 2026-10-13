#!/usr/bin/env python3
"""
Feature-Space Radial Attack (FRA): Directly maximize feature drift along
the RADIAL direction (f_clean direction), which is the cross-model invariant.

Theory:
  - f_clean defines the radial direction in feature space
  - Fine-tuning drift is ~83% radial (direction stable across models)
  - Tangential drift is random (doesn't transfer)
  - Therefore: attack radial → transfers; attack tangential → doesn't

Key difference from previous experiments:
  - Operates in FEATURE VECTOR SPACE, not complex spectral (FFT) plane
  - Radial = direction of f_clean itself (not exp(j*phase(FFT(f))))
  - This is a DIFFERENT decomposition than CODD's complex-plane carrier

Ablation (6 methods):
  1. FRA (radial):     maximize proj(f_adv - f_clean, f_clean_dir)
  2. FTA (tangential): maximize |drift - proj_radial(drift)|
  3. FRA+L1:           radial + L1 distance (combined)
  4. PGD (L1):         maximize L1(f_adv, f_clean) — baseline
  5. Random dir:       maximize proj onto random direction — control
  6. SE-CGR:           original amplitude-modulated probe — reference

Prediction: Radial attack >> Tangential >> Random direction
             on the FINE-TUNED model (SAMed).
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
from theory import synthesize_codd_probe, decompose_drift

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
# 1. Feature-Space Radial Attack (FRA)
#    Maximize drift projection onto f_clean direction
# ═══════════════════════════════════════════════════════════════════════════

def attack_fra(image_encoder, x, eps_pix, step_pix, steps,
              radial_weight=1.0, l1_weight=0.0,
              use_random_direction=False):
    """
    Feature-space Radial Attack.

    radial_weight > 0, l1_weight = 0: pure radial (maximize projection onto f_clean)
    radial_weight = 0, l1_weight > 0: standard PGD (L1 distance)
    radial_weight > 0, l1_weight > 0: combined

    use_random_direction=True: use random direction instead of f_clean (control)
    """
    max_val, min_val = x.max(), x.min()
    B, C_img, H, W = x.shape

    with torch.no_grad():
        f_clean, _ = image_encoder(x)
        f_clean_flat = f_clean.detach().view(1, -1)

    # Radial direction: unit vector in feature space
    if use_random_direction:
        d_random = torch.randn_like(f_clean_flat)
        d_radial = F.normalize(d_random, dim=1)
    else:
        d_radial = F.normalize(f_clean_flat, dim=1)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        f_adv, _ = image_encoder(x_adv)
        f_adv_flat = f_adv.view(1, -1)

        loss = 0.0

        # Radial loss: -projection onto clean feature direction
        # (maximize drift along f_clean direction)
        if radial_weight > 0:
            proj_radial = (f_adv_flat * d_radial).sum(dim=1)
            # Maximize |f_adv · d_radial| → pushes features away from origin
            # along the same direction as clean features
            loss_radial = -proj_radial.mean()
            loss = loss + radial_weight * loss_radial

        # L1 loss: standard feature disruption (optional)
        if l1_weight > 0:
            with torch.no_grad():
                targets, _ = image_encoder(x)
            loss_l1 = criterion_l1(f_adv * 0.9999, targets)
            loss = loss + l1_weight * loss_l1

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)

        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Feature-Space Tangential Attack (FTA)
#    Maximize drift orthogonal to f_clean direction
# ═══════════════════════════════════════════════════════════════════════════

def attack_fta(image_encoder, x, eps_pix, step_pix, steps):
    """
    Feature-space Tangential Attack.
    Maximizes drift component ORTHOGONAL to f_clean direction.
    Theory predicts: does NOT transfer (tangential = random across models).
    """
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

        # Project out radial component → tangential component
        proj_radial = (f_adv_flat * d_radial).sum(dim=1, keepdim=True) * d_radial
        f_tangential = f_adv_flat - proj_radial

        # Maximize tangential norm
        loss = -f_tangential.norm(dim=1).mean()

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)

        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# ═══════════════════════════════════════════════════════════════════════════
# 3. SE-CGR style attack (original, for reference)
# ═══════════════════════════════════════════════════════════════════════════

def attack_secgr(image_encoder, x, eps_pix, step_pix, steps,
                use_amplitude_mod=True, use_phase_carrier=True):
    """
    SE-CGR with controllable amplitude modulation and phase carrier.
    use_amplitude_mod=False → flat spectrum (Sigma = constant)
    use_phase_carrier=False → random phase (misaligned)
    """
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
        feature_difference_list = []
        for idx in indices:
            F_clean_i = targets_layer[idx + pre_layer - num_layers]
            F_adv_i = features_layer[idx + pre_layer - num_layers]

            probe = synthesize_codd_probe(
                F_clean=F_clean_i.unsqueeze(0),
                F_adv=F_adv_i.unsqueeze(0),
                direction="radial",
                use_random_noise=True,
            )
            feature_difference = (F_adv_i.unsqueeze(0) * 0.999
                                 - F_clean_i.unsqueeze(0)) - 2.0 * probe
            feature_difference_list.append(feature_difference.squeeze(0))

        fd_stack = torch.stack(feature_difference_list)

        # Input mean normalization
        selected_t = torch.stack(
            [targets_layer[i + pre_layer - num_layers] for i in indices])
        input_mean = torch.stack([selected_t[i].abs().mean()
                                  for i in range(len(selected_t))])
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
# Drift diagnostic
# ═══════════════════════════════════════════════════════════════════════════

def measure_drift_radiality(image_encoder, x_clean, x_adv):
    """Measure radial ratio of attack drift in FEATURE SPACE."""
    with torch.no_grad():
        f_clean, _ = image_encoder(x_clean)
        f_adv, _ = image_encoder(x_adv)

    f_clean_flat = f_clean.view(-1)
    f_adv_flat = f_adv.view(-1)
    drift = f_adv_flat - f_clean_flat

    # Project onto clean feature direction
    d_radial = F.normalize(f_clean_flat, dim=0)
    proj = (drift * d_radial).sum()
    radial_energy = proj ** 2
    total_energy = drift.norm() ** 2
    radial_ratio = float(radial_energy / (total_energy + 1e-12))

    return radial_ratio


# ═══════════════════════════════════════════════════════════════════════════
# Experiment
# ═══════════════════════════════════════════════════════════════════════════

configs = [
    # (name, attack_fn_key, kwargs)
    ("FRA (radial only)",     "fra_radial",    {}),
    ("FRA+L1 (radial + L1)",  "fra_combined",  {}),
    ("FTA (tangential only)", "fta",           {}),
    ("PGD (L1 baseline)",     "pgd_l1",        {}),
    ("Random dir (control)",  "fra_random",    {}),
    ("SE-CGR (reference)",    "secgr_full",    {}),
    ("SE-CGR (no amp mod)",   "secgr_noamp",   {}),
    ("SE-CGR (no phase)",     "secgr_nophase", {}),
]

print(f"\n{'='*80}")
print("FEATURE-SPACE RADIAL ATTACK EXPERIMENT")
print(f"Radial direction = direction of f_clean in feature vector space")
print(f"Attack: source SAM encoder → Evaluate: SAMed (fine-tuned)")
print(f"ε={EPSILON:.4f}, step={STEP:.4f}, steps={STEPS}")
print(f"{'='*80}")

all_results = []

for name, key, kwargs in configs:
    deltas = []
    drift_rrs = []
    t0 = time.time()

    for img_data, lbl in images:
        x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        clean_d = evaluate(net, img_data, lbl, "Synapse", device)
        if clean_d < 0.01:
            continue
        rg = x.max().item() - x.min().item()
        eps_pix = EPSILON * rg
        step_pix = STEP * rg

        if key == "fra_radial":
            x_adv = attack_fra(image_encoder, x, eps_pix, step_pix, STEPS,
                              radial_weight=1.0, l1_weight=0.0)
        elif key == "fra_combined":
            x_adv = attack_fra(image_encoder, x, eps_pix, step_pix, STEPS,
                              radial_weight=1.0, l1_weight=1.0)
        elif key == "fra_random":
            x_adv = attack_fra(image_encoder, x, eps_pix, step_pix, STEPS,
                              radial_weight=1.0, l1_weight=0.0,
                              use_random_direction=True)
        elif key == "fta":
            x_adv = attack_fta(image_encoder, x, eps_pix, step_pix, STEPS)
        elif key == "pgd_l1":
            x_adv = attack_fra(image_encoder, x, eps_pix, step_pix, STEPS,
                              radial_weight=0.0, l1_weight=1.0)
        elif key == "secgr_full":
            x_adv = attack_secgr(image_encoder, x, eps_pix, step_pix, STEPS)
        elif key == "secgr_noamp":
            x_adv = attack_secgr(image_encoder, x, eps_pix, step_pix, STEPS,
                                use_amplitude_mod=False)
        elif key == "secgr_nophase":
            x_adv = attack_secgr(image_encoder, x, eps_pix, step_pix, STEPS,
                                use_phase_carrier=False)

        adv_np = x_adv.squeeze(0).cpu().numpy()
        att_d = evaluate(net, adv_np, lbl, "Synapse", device)
        deltas.append(clean_d - att_d)

        rr = measure_drift_radiality(image_encoder, x, x_adv)
        drift_rrs.append(rr)

    dt = time.time() - t0
    md, sd = float(np.mean(deltas)), float(np.std(deltas))
    mr = float(np.mean(drift_rrs)) if drift_rrs else 0.0
    all_results.append({
        "name": name, "key": key,
        "mean_delta": md, "std_delta": sd,
        "mean_drift_radial_ratio": mr,
        "time": dt,
    })
    print(f"  {name:<30} ΔDice={md:+.4f}±{sd:.3f}  drift_radial={mr:.3f}  {dt:.1f}s")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("SUMMARY — Feature-Space Directional Transfer")
print(f"{'='*80}")
print(f"{'Method':<30} {'ΔDice':>10} {'Drift RR':>10}")
print("-" * 55)
for r in all_results:
    print(f"  {r['name']:<30} {r['mean_delta']:+.4f}     {r['mean_drift_radial_ratio']:.3f}")

# Key comparisons
fra = {r["key"]: r["mean_delta"] for r in all_results}
print(f"\n── Key Metrics ──")
if "fra_radial" in fra and "fta" in fra:
    premium = fra["fra_radial"] - fra["fta"]
    print(f"  Radial Premium (FRA - FTA): {premium:+.4f}")
    if premium > 0.02:
        print(f"  → THEORY VALIDATED: radial transfers better than tangential")
    else:
        print(f"  → THEORY FAILS: tangential ≈ radial")

if "fra_radial" in fra and "fra_random" in fra:
    dir_effect = fra["fra_radial"] - fra["fra_random"]
    print(f"  Direction Effect (radial - random): {dir_effect:+.4f}")
    print(f"  → Clean direction matters: {dir_effect:+.4f}")

if "secgr_full" in fra:
    print(f"  SE-CGR vs FRA: {fra['secgr_full'] - fra.get('fra_radial', 0):+.4f}")
    print(f"  SE-CGR vs PGD: {fra['secgr_full'] - fra.get('pgd_l1', 0):+.4f}")

# Save
out_path = os.path.join(OUTPUT_DIR, "feature_radial_attack.json")
with open(out_path, "w") as f:
    json.dump({
        "config": {"epsilon": EPSILON, "step": STEP, "steps": STEPS},
        "results": all_results,
    }, f, indent=2)
print(f"\nSaved to {out_path}")

del net, source_sam
torch.cuda.empty_cache()
print("Done")
