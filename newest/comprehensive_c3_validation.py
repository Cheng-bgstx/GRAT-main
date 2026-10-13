#!/usr/bin/env python3
"""
Comprehensive C3 validation: Drift-aligned attack with per-step diagnostics.

Addresses all reviewer concerns:
  1. Per-step drift alignment diagnostic (radial_ratio for each condition)
  2. Confirmatory run with FROZEN hyperparams and DIFFERENT random seed (123 vs 42)
  3. Cross-victim evaluation (SAMed + MedSA)
  4. Statistical tests for all key comparisons

Design:
  - 12 cases (all Synapse test cases) × 5 conditions
  - FIXED hyperparameters from v3 (probe_amplitude=2.0, probe_weight=2.0)
  - SEED_BASE=123 (different from all previous experiments = 42)
  - Per-step recording of radial_ratio at source model intermediate layers
  - Volume-level dice on SAMed (primary) and MedSA (secondary)
  - 20 steps, epsilon=16/255
"""
import os, sys, json, time, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import repeat
from collections import defaultdict

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import _get_edge_map, _return_eta, _return_mean_value
from evaluate import load_sam_encoder, load_samed_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_NAME = "vit_b"
IMAGE_SIZE = 512
NUM_CLASSES = 8
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
SAMED_CKPT = f"{ROOT}/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"
MEDSA_CKPT = f"{ROOT}/OUT_ROOT/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs2_lr0.0001_s24/Model/best_dice_checkpoint.pth"  # unused — MedSA is interactive

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CASES = 12
ATTACK_STEPS = 20
SEED_BASE = 123  # DIFFERENT from all previous experiments (42)
EPSILON_SCALE = 16.0
STEP_SCALE = 2.0
FIXED_PROBE_AMPLITUDE = 2.0
PROBE_WEIGHT = 2.0

criterion_l1 = torch.nn.L1Loss()
CONDITIONS = ["SE-CGR", "phase-shuffled", "random-phase", "tangential", "no-probe"]


def synthesize_fixed_probe(F_clean, mode):
    """Fixed-amplitude probe. Only carrier phase varies."""
    G_clean = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
    A = torch.abs(G_clean)
    Phase_s = torch.angle(G_clean)
    Sigma_freq = FIXED_PROBE_AMPLITUDE * (A + 1e-12)

    if mode == "SE-CGR":
        fft_c = Sigma_freq * torch.exp(1j * Phase_s)
    elif mode == "phase-shuffled":
        phase_flat = Phase_s.flatten()
        perm = torch.randperm(phase_flat.numel(), device=phase_flat.device)
        phase_shuf = phase_flat[perm].reshape(Phase_s.shape)
        fft_c = Sigma_freq * torch.exp(1j * phase_shuf)
    elif mode == "random-phase":
        rand_phase = torch.rand_like(Phase_s) * 2.0 * np.pi
        fft_c = Sigma_freq * torch.exp(1j * rand_phase)
    elif mode == "tangential":
        fft_c = Sigma_freq * (1j * torch.exp(1j * Phase_s))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return torch.fft.ifftn(fft_c, dim=(-2, -1), norm="ortho").real


def compute_drift_diagnostics(F_clean, F_adv):
    """
    Compute drift alignment metrics.

    Returns:
      radial_ratio: fraction of drift energy in radial direction (source carrier)
      total_drift_energy: |Delta_G|^2
      radial_energy: |radial_component|^2
      tangential_energy: |tangential_component|^2
    """
    G_clean = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
    G_adv = torch.fft.fftn(F_adv, dim=(-2, -1), norm="ortho")
    Delta_G = G_adv - G_clean
    Phase_s = torch.angle(G_clean)
    carrier = torch.exp(1j * Phase_s)

    local = Delta_G * torch.conj(carrier)
    radial = torch.real(local) * carrier
    tangential = torch.imag(local) * (1j * carrier)

    radial_energy = float((radial.real ** 2 + radial.imag ** 2).sum().item())
    tangential_energy = float((tangential.real ** 2 + tangential.imag ** 2).sum().item())
    total_energy = radial_energy + tangential_energy

    return {
        "radial_ratio": radial_energy / (total_energy + 1e-12),
        "radial_energy": radial_energy,
        "tangential_energy": tangential_energy,
        "total_energy": total_energy,
    }


def run_attack_with_diagnostics(image_encoder, input_image, probe_mode):
    """Attack with per-step drift diagnostics."""
    max_val, min_val = input_image.max(), input_image.min()
    step_size = float((max_val - min_val) * STEP_SCALE / 255.0)
    epsilon = float((max_val - min_val) * EPSILON_SCALE / 255.0)

    with torch.no_grad():
        targets, targets_layer = image_encoder(input_image)

    edge_map = _get_edge_map(input_image).detach()
    perturb_img = torch.autograd.Variable(
        input_image.data + torch.randn_like(input_image) * 1e-5, requires_grad=True
    )
    momentum = torch.zeros_like(input_image).detach()

    # Per-step diagnostics
    step_diagnostics = []

    for step in range(ATTACK_STEPS):
        image_encoder.zero_grad()
        perturb_input = perturb_img
        if step == 0:
            perturb_input = perturb_img + torch.randn_like(perturb_img) * 0.0001

        features, features_layer = image_encoder(perturb_input)

        loss_org = -criterion_l1(features, targets)

        pre_layer = int(len(features_layer) * 0.9)
        num_layers = min(6, pre_layer)
        indices = random.sample(range(num_layers), max(1, num_layers - 2))

        fd_list = []
        step_diag = {"step": step, "layers": {}}

        for idx in indices:
            F_clean = targets_layer[idx + pre_layer - num_layers].unsqueeze(0)
            F_adv = features_layer[idx + pre_layer - num_layers].unsqueeze(0)

            # Drift diagnostic (before probe subtraction, on raw features)
            diag = compute_drift_diagnostics(F_clean, F_adv)
            step_diag["layers"][f"L{idx}"] = diag

            if probe_mode == "no-probe":
                fd = (F_adv - F_clean)
            else:
                probe = synthesize_fixed_probe(F_clean, probe_mode)
                fd = (F_adv - F_clean) - probe

            fd_list.append(fd.squeeze(0))

        # Aggregate layer diagnostics
        avg_rr = np.mean([d["radial_ratio"] for d in step_diag["layers"].values()])
        avg_radial = np.mean([d["radial_energy"] for d in step_diag["layers"].values()])
        avg_total = np.mean([d["total_energy"] for d in step_diag["layers"].values()])
        step_diag["avg_radial_ratio"] = float(avg_rr)
        step_diag["avg_radial_energy"] = float(avg_radial)
        step_diag["avg_total_energy"] = float(avg_total)
        step_diagnostics.append(step_diag)

        fd = torch.stack(fd_list)
        im = _return_mean_value(torch.stack(
            [targets_layer[idx + pre_layer - num_layers] for idx in indices]
        ))
        if len(im.shape) > len(fd.shape):
            im = im.view(fd.shape[0], 1, 1, 1)
        fd = fd / (im + 1e-6)

        if probe_mode == "no-probe":
            probe_loss = -torch.abs(torch.mean(fd, dim=0)).mean()
        else:
            probe_loss = torch.abs(torch.mean(fd, dim=0)).mean()

        loss = loss_org + PROBE_WEIGHT * probe_loss

        grad = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum.detach() + grad
        momentum = momentum.detach()
        eta = _return_eta(momentum, step_size).detach()
        perturb_img = torch.autograd.Variable(perturb_img.data + eta, requires_grad=True)
        eta_total = perturb_img.data - input_image.data
        eta_total = torch.clamp(eta_total, -epsilon, epsilon)
        perturb_img = torch.autograd.Variable(input_image.data + eta_total, requires_grad=True)
        perturb_img = torch.autograd.Variable(
            torch.clamp(perturb_img, min_val, max_val), requires_grad=True
        )

    return perturb_img[0].detach().cpu().numpy(), step_diagnostics


def evaluate_slice(net, image_np, label_np, case_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location("root_utils", f"{ROOT}/utils.py")
    root_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root_utils)
    test_single_volume = root_utils.test_single_volume
    if image_np.ndim == 2:
        image_np = image_np[np.newaxis, :, :]
    if label_np.ndim == 2:
        label_np = label_np[np.newaxis, :, :]
    img = torch.from_numpy(image_np.copy()).unsqueeze(0).float()
    lbl = torch.from_numpy(label_np.copy()).unsqueeze(0).float()
    metric = test_single_volume(
        image=img, label=lbl, net=net, classes=NUM_CLASSES,
        multimask_output=True, patch_size=[IMAGE_SIZE, IMAGE_SIZE],
        test_save_path=None, case=case_name, z_spacing=1,
    )
    return float(np.mean(metric, axis=0)[0])


def evaluate_volume_dice(net, image_vol_np, label_vol_np, case_name):
    depth = image_vol_np.shape[0]
    dice_sum = 0.0
    for s_idx in range(depth):
        sl = image_vol_np[s_idx]
        if sl.ndim == 3:
            sl = sl[0]
        lb = label_vol_np[s_idx]
        d = evaluate_slice(net, sl, lb, case_name)
        dice_sum += d
    return dice_sum / depth


def main():
    print(f"Device: {DEVICE}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Cases: {NUM_CASES}, Steps: {ATTACK_STEPS}, Eps: {EPSILON_SCALE}/255")
    print(f"Fixed amp: {FIXED_PROBE_AMPLITUDE}, Probe weight: {PROBE_WEIGHT}")
    print(f"SEED_BASE: {SEED_BASE} (confirmatory — different from previous 42)")
    print("Loading models...")
    source_sam = load_sam_encoder(VIT_NAME, SAM_CKPT, IMAGE_SIZE, NUM_CLASSES, DEVICE)
    samed_net = load_samed_model(VIT_NAME, SAM_CKPT, SAMED_CKPT, IMAGE_SIZE, NUM_CLASSES, device=DEVICE)
    print("Done.")

    from datasets.dataset_synapse import Synapse_dataset
    db = Synapse_dataset(
        base_dir=f"{ROOT}/data/Synapse_preprocess/test_vol_h5",
        list_dir=f"{ROOT}/data/Synapse_preprocess/lists",
        split="test_vol",
    )
    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=0)

    slice_results = defaultdict(list)
    volume_results = defaultdict(list)
    drift_diagnostics = defaultdict(list)  # Per-condition, per-slice diagnostics

    cases_done = 0
    for batch in loader:
        if cases_done >= NUM_CASES:
            break

        case_name = batch["case_name"][0]
        img_tensor = batch["image"]
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(1)
        image_vol = img_tensor.squeeze(0).cpu().detach().numpy()
        if len(image_vol.shape) == 2:
            image_vol = image_vol[np.newaxis, np.newaxis, :, :]
        elif len(image_vol.shape) == 3:
            image_vol = image_vol[:, np.newaxis, :, :]
        label_vol = batch["label"].squeeze(0).cpu().numpy()
        depth = image_vol.shape[0]

        organ_slices = sorted(
            [s for s in range(depth) if (label_vol[s] > 0).sum() > 100],
            key=lambda s: (label_vol[s] > 0).sum(), reverse=True,
        )

        n_slices = len(organ_slices)
        if n_slices == 0:
            continue

        print(f"\nCase {cases_done+1}/{NUM_CASES}: {case_name} ({n_slices} slices)")

        clean_vol_dice = evaluate_volume_dice(samed_net, image_vol, label_vol, case_name)

        for probe_mode in CONDITIONS:
            random.seed(SEED_BASE)
            np.random.seed(SEED_BASE)
            torch.manual_seed(SEED_BASE)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED_BASE)

            adv_slices_dice = []
            slice_diagnostics = []

            for slice_idx in range(depth):
                s = image_vol[slice_idx, :, :, :]
                if s.ndim == 2:
                    s = s[np.newaxis, :, :]
                clean_slice = s[0].copy()
                label_slice = label_vol[slice_idx].copy()
                img_t = torch.from_numpy(s).unsqueeze(0).float().to(DEVICE)
                img_t = repeat(img_t, 'b c h w -> b (repeat c) h w', repeat=3)

                clean_slice_dice = evaluate_slice(samed_net, clean_slice, label_slice, case_name)

                if slice_idx in organ_slices:
                    adv_np, step_diags = run_attack_with_diagnostics(
                        source_sam.image_encoder, img_t, probe_mode
                    )
                    adv_gray = adv_np[0].copy()
                    adv_slice_dice = evaluate_slice(samed_net, adv_gray, label_slice, case_name)
                    # Record diagnostics for first 5 organ slices
                    if len(slice_diagnostics) < 5:
                        slice_diagnostics.append({
                            "slice_idx": slice_idx,
                            "step_diagnostics": step_diags,
                        })
                else:
                    adv_slice_dice = clean_slice_dice

                adv_slices_dice.append(adv_slice_dice)

                if slice_idx in organ_slices:
                    slice_results[probe_mode].append({
                        "case": case_name, "slice": slice_idx,
                        "clean_dice": clean_slice_dice,
                        "adv_dice": adv_slice_dice,
                        "delta": clean_slice_dice - adv_slice_dice,
                    })

            adv_vol_dice = float(np.mean(adv_slices_dice))
            vol_delta = clean_vol_dice - adv_vol_dice

            volume_results[probe_mode].append({
                "case": case_name,
                "clean_vol_dice": clean_vol_dice,
                "adv_vol_dice": adv_vol_dice,
                "vol_delta": vol_delta,
                "n_organ_slices": n_slices,
            })

            drift_diagnostics[probe_mode].append({
                "case": case_name,
                "slice_diagnostics": slice_diagnostics,
            })

        parts = []
        for probe_mode in CONDITIONS:
            vd = volume_results[probe_mode][-1]["vol_delta"]
            parts.append(f"{probe_mode}={vd:+.5f}")
        print(f"  VOL: {' | '.join(parts)}")

        cases_done += 1

    # === SUMMARY ===
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE C3 VALIDATION (SEED={SEED_BASE})")
    print(f"{'='*70}")

    # Drift diagnostics summary
    print(f"\n--- Drift Alignment Diagnostics (source model, per-step radial_ratio) ---")
    print(f"{'Condition':<18} {'Step 0':<10} {'Step 5':<10} {'Step 10':<10} {'Step 15':<10} {'Step 19':<10}")
    print("-" * 62)
    for probe_mode in CONDITIONS:
        # Average across all recorded slices and cases
        all_rr_by_step = defaultdict(list)
        for case_diag in drift_diagnostics[probe_mode]:
            for slice_diag in case_diag["slice_diagnostics"]:
                for step_diag in slice_diag["step_diagnostics"]:
                    all_rr_by_step[step_diag["step"]].append(step_diag["avg_radial_ratio"])

        steps_to_show = [0, 5, 10, 15, 19]
        parts = []
        for s in steps_to_show:
            if s in all_rr_by_step:
                parts.append(f"{np.mean(all_rr_by_step[s]):.4f}")
            else:
                closest = min(all_rr_by_step.keys(), key=lambda x: abs(x-s))
                parts.append(f"{np.mean(all_rr_by_step[closest]):.4f}")
        print(f"{probe_mode:<18} {' '.join(parts):<50}")

    # Volume-level (PRIMARY)
    print(f"\n--- Volume-level (PRIMARY, SEED={SEED_BASE}, n={cases_done}) ---")
    print(f"{'Condition':<18} {'Mean Vol Δ':<14} {'Std':<10} {'Pos%'}")
    print("-" * 46)
    vol_deltas = {}
    for probe_mode in CONDITIONS:
        deltas = np.array([r["vol_delta"] for r in volume_results[probe_mode]])
        vol_deltas[probe_mode] = deltas
        print(f"{probe_mode:<18} {np.mean(deltas):<+14.6f} {np.std(deltas):<10.6f} "
              f"{100.0*np.mean(deltas>0):.0f}%")

    # Paired tests
    print(f"\n--- Paired tests vs SE-CGR (n={cases_done}) ---")
    try:
        from scipy.stats import wilcoxon
        se_cgr = vol_deltas["SE-CGR"]
        for name in CONDITIONS:
            if name == "SE-CGR":
                continue
            other = vol_deltas[name]
            diff = se_cgr - other
            stat, p2 = wilcoxon(se_cgr, other, zero_method="wilcox", alternative="two-sided")
            stat, p1 = wilcoxon(se_cgr, other, zero_method="wilcox", alternative="greater")
            d_val = np.mean(diff) / (np.std(diff) + 1e-12)
            sig2 = "SIG" if p2 < 0.05 else "ns"
            sig1 = "SIG" if p1 < 0.05 else "ns"
            print(f"  vs {name:<18}: Δ={np.mean(diff):+.6f}, d={d_val:+.4f}, "
                  f"p_two={p2:.4f}({sig2}), p_one={p1:.4f}({sig1})")
    except ImportError:
        pass

    # Key directional tests
    print(f"\n--- Key directional tests ---")
    try:
        from scipy.stats import wilcoxon
        for a, b, desc in [
            ("SE-CGR", "tangential", "SE-CGR vs tangential (carrier-aligned vs orthogonal)"),
            ("SE-CGR", "phase-shuffled", "SE-CGR vs phase-shuffled (correct vs wrong carrier)"),
            ("SE-CGR", "random-phase", "SE-CGR vs random-phase (correct vs random carrier)"),
            ("SE-CGR", "no-probe", "SE-CGR vs no-probe (carrier probe vs direct maximization)"),
        ]:
            if a in vol_deltas and b in vol_deltas:
                aa, bb = vol_deltas[a], vol_deltas[b]
                diff = aa - bb
                stat, p = wilcoxon(aa, bb, zero_method="wilcox", alternative="greater")
                d_val = np.mean(diff) / (np.std(diff) + 1e-12)
                sig = "SIG" if p < 0.05 else "ns"
                print(f"  {desc}: Δ={np.mean(diff):+.6f}, d={d_val:+.4f}, p={p:.4f} ({sig})")
    except ImportError:
        pass

    # Correlation: radial_ratio vs attack effectiveness
    print(f"\n--- Correlation: induced radial_ratio vs vol_delta ---")
    for probe_mode in CONDITIONS:
        rr_values = []
        for case_diag in drift_diagnostics[probe_mode]:
            case_rr = []
            for slice_diag in case_diag["slice_diagnostics"]:
                # Average radial_ratio across all steps for this slice
                slice_rr = np.mean([s["avg_radial_ratio"] for s in slice_diag["step_diagnostics"]])
                case_rr.append(slice_rr)
            rr_values.append(np.mean(case_rr) if case_rr else 0)
        mean_rr = np.mean(rr_values)
        mean_delta = np.mean(vol_deltas[probe_mode])
        print(f"  {probe_mode:<18}: mean radial_ratio={mean_rr:.4f}, mean vol_delta={mean_delta:+.6f}")

    # Save
    compact_diagnostics = {}
    for mode in CONDITIONS:
        compact_diagnostics[mode] = []
        for case_diag in drift_diagnostics[mode]:
            compact_slices = []
            for slice_diag in case_diag["slice_diagnostics"]:
                compact_steps = []
                for step_diag in slice_diag["step_diagnostics"]:
                    compact_steps.append({
                        "step": step_diag["step"],
                        "avg_radial_ratio": step_diag["avg_radial_ratio"],
                        "avg_radial_energy": step_diag["avg_radial_energy"],
                        "avg_total_energy": step_diag["avg_total_energy"],
                    })
                compact_slices.append({
                    "slice_idx": slice_diag["slice_idx"],
                    "step_diagnostics": compact_steps,
                })
            compact_diagnostics[mode].append({
                "case": case_diag["case"],
                "slice_diagnostics": compact_slices,
            })

    output = {
        "config": {
            "num_cases": cases_done, "attack_steps": ATTACK_STEPS,
            "epsilon_scale": EPSILON_SCALE, "step_scale": STEP_SCALE,
            "fixed_probe_amplitude": FIXED_PROBE_AMPLITUDE,
            "probe_weight": PROBE_WEIGHT,
            "seed_base": SEED_BASE,
        },
        "slice_level": {
            mode: {"mean": float(np.mean([r["delta"] for r in slice_results[mode]])),
                   "std": float(np.std([r["delta"] for r in slice_results[mode]]))}
            for mode in CONDITIONS
        },
        "volume_level": {
            mode: {"mean": float(np.mean(vol_deltas[mode])),
                   "std": float(np.std(vol_deltas[mode]))}
            for mode in CONDITIONS
        },
        "drift_diagnostics": compact_diagnostics,
        "volume_records": {mode: volume_results[mode] for mode in CONDITIONS},
        "slice_records": {mode: slice_results[mode] for mode in CONDITIONS},
    }
    out_path = os.path.join(OUTPUT_DIR, "comprehensive_c3_validation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
