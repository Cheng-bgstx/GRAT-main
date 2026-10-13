#!/usr/bin/env python3
"""
Matched ablation v2: Direct frequency-domain directional energy maximization.

Key fix over v1: Instead of indirect probe-based loss (which vanishes at early steps
due to lambda_val ~ diff_energy ~ 0), directly maximize the frequency-domain radial
or tangential energy of Delta_G = G_adv - G_clean.

CODD decomposition: Delta_G = alpha * exp(j*Phi_s) + beta * j * exp(j*Phi_s)
- Radial component along carrier: Re(Delta_G * conj(carrier)) * carrier
- Tangential component: Im(Delta_G * conj(carrier)) * (j * carrier)

Conditions (all use identical edge guidance, multi-layer, momentum, steps, epsilon):
  - SE-CGR:        maximize radial energy along source carrier
  - phase-shuffled: maximize radial energy along spatially permuted source carrier
  - random-phase:   maximize radial energy along random carrier
  - tangential:     maximize tangential energy (orthogonal to source carrier)
  - no-probe:       maximize total energy |Delta_G|^2 (no directional preference)

Prediction (if CODD is correct):
  - SE-CGR ≈ no-probe (drift IS naturally radial; both produce radial-dominant drift)
  - SE-CGR > phase-shuffled (wrong carrier → less transferable drift)
  - SE-CGR > random-phase (wrong carrier → less transferable drift)
  - SE-CGR > tangential (orthogonal to carrier → least transferable)
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

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CASES = 12
ATTACK_STEPS = 20           # Increased from 10 for stronger attack
SEED_BASE = 42
EPSILON_SCALE = 16.0        # Increased from 10 for stronger attack
STEP_SCALE = 2.0
DIRECTIONAL_WEIGHT = 5.0    # Weight for directional energy term (increased for clear signal)

criterion_l1 = torch.nn.L1Loss()

CONDITIONS = ["SE-CGR", "phase-shuffled", "random-phase", "tangential", "no-probe"]


def run_matched_attack_v2(image_encoder, input_image, probe_mode):
    """
    Attack with direct frequency-domain directional energy maximization.

    Instead of synthesizing a probe and minimizing residual, we DIRECTLY
    maximize the energy of the desired frequency-domain component.
    """
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

    for step in range(ATTACK_STEPS):
        image_encoder.zero_grad()
        perturb_input = perturb_img
        if step == 0:
            perturb_input = perturb_img + torch.randn_like(perturb_img) * 0.0001

        features, features_layer = image_encoder(perturb_input)

        # Base disruption: MAXIMIZE feature distance (like MI-FGSM, negative L1)
        # Positive L1 pushes toward clean (conservative); negative pushes away (adversarial)
        loss_org = -criterion_l1(features, targets)

        # Multi-layer frequency-domain directional energy
        pre_layer = int(len(features_layer) * 0.9)
        num_layers = min(6, pre_layer)
        indices = random.sample(range(num_layers), max(1, num_layers - 2))

        dir_energy_total = 0.0
        n_layers_used = 0

        for idx in indices:
            F_clean = targets_layer[idx + pre_layer - num_layers].unsqueeze(0)  # [1, C, H, W]
            F_adv = features_layer[idx + pre_layer - num_layers].unsqueeze(0)

            # Frequency domain
            G_clean = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
            G_adv = torch.fft.fftn(F_adv, dim=(-2, -1), norm="ortho")

            Delta_G = G_adv - G_clean
            Phase_s = torch.angle(G_clean)
            carrier = torch.exp(1j * Phase_s)  # exp(j*Phi_s)

            if probe_mode == "SE-CGR":
                # Maximize RADIAL energy along SOURCE carrier
                local = Delta_G * torch.conj(carrier)
                radial = torch.real(local) * carrier
                layer_energy = (radial.real ** 2 + radial.imag ** 2).mean()

            elif probe_mode == "phase-shuffled":
                # Maximize RADIAL energy along SHUFFLED source carrier
                phase_flat = Phase_s.flatten()
                perm = torch.randperm(phase_flat.numel(), device=phase_flat.device)
                phase_shuf = phase_flat[perm].reshape(Phase_s.shape)
                carrier_shuf = torch.exp(1j * phase_shuf)
                local = Delta_G * torch.conj(carrier_shuf)
                radial = torch.real(local) * carrier_shuf
                layer_energy = (radial.real ** 2 + radial.imag ** 2).mean()

            elif probe_mode == "random-phase":
                # Maximize RADIAL energy along RANDOM carrier
                rand_phase = torch.rand_like(Phase_s) * 2.0 * np.pi
                carrier_rand = torch.exp(1j * rand_phase)
                local = Delta_G * torch.conj(carrier_rand)
                radial = torch.real(local) * carrier_rand
                layer_energy = (radial.real ** 2 + radial.imag ** 2).mean()

            elif probe_mode == "tangential":
                # Maximize TANGENTIAL energy (orthogonal to source carrier)
                local = Delta_G * torch.conj(carrier)
                tangential = torch.imag(local) * (1j * carrier)
                layer_energy = (tangential.real ** 2 + tangential.imag ** 2).mean()

            elif probe_mode == "no-probe":
                # Maximize TOTAL drift energy (no directional preference)
                layer_energy = (Delta_G.real ** 2 + Delta_G.imag ** 2).mean()

            else:
                raise ValueError(f"Unknown mode: {probe_mode}")

            dir_energy_total += layer_energy
            n_layers_used += 1

        if n_layers_used > 0:
            dir_energy_total = dir_energy_total / n_layers_used

        # MAXIMIZE directional energy (negative sign = gradient ascent)
        loss = loss_org - DIRECTIONAL_WEIGHT * dir_energy_total

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

    return perturb_img[0].detach().cpu().numpy()


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
    """Volume-level dice = mean of per-slice dice."""
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
    print(f"Directional weight: {DIRECTIONAL_WEIGHT}")
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

        # Clean volume-level dice
        clean_vol_dice = evaluate_volume_dice(samed_net, image_vol, label_vol, case_name)

        for probe_mode in CONDITIONS:
            random.seed(SEED_BASE)
            np.random.seed(SEED_BASE)
            torch.manual_seed(SEED_BASE)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED_BASE)

            adv_slices_dice = []
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
                    adv_np = run_matched_attack_v2(source_sam.image_encoder, img_t, probe_mode)
                    adv_gray = adv_np[0].copy()
                    adv_slice_dice = evaluate_slice(samed_net, adv_gray, label_slice, case_name)
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

            # Volume-level: mean across all slices
            adv_vol_dice = float(np.mean(adv_slices_dice))
            vol_delta = clean_vol_dice - adv_vol_dice
            volume_results[probe_mode].append({
                "case": case_name,
                "clean_vol_dice": clean_vol_dice,
                "adv_vol_dice": adv_vol_dice,
                "vol_delta": vol_delta,
                "n_organ_slices": n_slices,
            })

        # Print per-condition volume deltas
        parts = []
        for probe_mode in CONDITIONS:
            vd = volume_results[probe_mode][-1]["vol_delta"]
            parts.append(f"{probe_mode}={vd:+.5f}")
        print(f"  VOL: {' | '.join(parts)}")

        cases_done += 1

    # === SUMMARY ===
    print(f"\n{'='*70}")
    print(f"MATCHED ABLATION V2: Direct Frequency-Domain Directional Optimization")
    print(f"{'='*70}")

    # Slice-level
    print(f"\n--- Slice-level ---")
    print(f"{'Condition':<18} {'N':<5} {'Mean Delta':<14} {'Pos%'}")
    print("-" * 42)
    for probe_mode in CONDITIONS:
        deltas = np.array([r["delta"] for r in slice_results[probe_mode]])
        print(f"{probe_mode:<18} {len(deltas):<5} {np.mean(deltas):<+14.6f} {100.0*np.mean(deltas>0):.0f}%")

    # Volume-level (PRIMARY)
    print(f"\n--- Volume-level (PRIMARY) ---")
    print(f"{'Condition':<18} {'Mean Vol Δ':<14} {'Std':<10} {'Pos%'}")
    print("-" * 46)
    vol_deltas_dict = {}
    for probe_mode in CONDITIONS:
        deltas = np.array([r["vol_delta"] for r in volume_results[probe_mode]])
        vol_deltas_dict[probe_mode] = deltas
        print(f"{probe_mode:<18} {np.mean(deltas):<+14.6f} {np.std(deltas):<10.6f} "
              f"{100.0*np.mean(deltas>0):.0f}%")

    # Paired tests (volume-level) — key comparison: SE-CGR vs others
    print(f"\n--- Paired tests vs SE-CGR (volume-level, n={len(vol_deltas_dict['SE-CGR'])}) ---")
    try:
        from scipy.stats import wilcoxon
        se_cgr = vol_deltas_dict["SE-CGR"]
        for name in CONDITIONS:
            if name == "SE-CGR":
                continue
            other = vol_deltas_dict[name]
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

    # Also compare: no-probe vs tangential (total energy should outperform orthogonal)
    print(f"\n--- Key directional tests ---")
    try:
        from scipy.stats import wilcoxon
        for a, b, desc in [
            ("no-probe", "tangential", "total-energy vs tangential (should differ if direction matters)"),
            ("SE-CGR", "tangential", "SE-CGR vs tangential (carrier-aligned vs orthogonal)"),
            ("SE-CGR", "phase-shuffled", "SE-CGR vs phase-shuffled (correct vs wrong carrier)"),
            ("SE-CGR", "random-phase", "SE-CGR vs random-phase (correct vs random carrier)"),
        ]:
            if a in vol_deltas_dict and b in vol_deltas_dict:
                aa, bb = vol_deltas_dict[a], vol_deltas_dict[b]
                diff = aa - bb
                stat, p = wilcoxon(aa, bb, zero_method="wilcox", alternative="greater")
                d_val = np.mean(diff) / (np.std(diff) + 1e-12)
                sig = "SIG" if p < 0.05 else "ns"
                print(f"  {desc}: Δ={np.mean(diff):+.6f}, d={d_val:+.4f}, p={p:.4f} ({sig})")
    except ImportError:
        pass

    # Save
    output = {
        "config": {"num_cases": cases_done, "attack_steps": ATTACK_STEPS,
                    "epsilon_scale": EPSILON_SCALE, "step_scale": STEP_SCALE,
                    "directional_weight": DIRECTIONAL_WEIGHT},
        "slice_level": {
            mode: {"mean": float(np.mean([r["delta"] for r in slice_results[mode]])),
                   "std": float(np.std([r["delta"] for r in slice_results[mode]]))}
            for mode in CONDITIONS
        },
        "volume_level": {
            mode: {"mean": float(np.mean(vol_deltas_dict[mode])),
                   "std": float(np.std(vol_deltas_dict[mode]))}
            for mode in CONDITIONS
        },
        "volume_records": {mode: volume_results[mode] for mode in CONDITIONS},
        "slice_records": {mode: slice_results[mode] for mode in CONDITIONS},
    }
    out_path = os.path.join(OUTPUT_DIR, "matched_ablation_v2.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
