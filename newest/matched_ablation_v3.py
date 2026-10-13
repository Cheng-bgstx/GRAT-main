#!/usr/bin/env python3
"""
Matched ablation v3: Fixed-amplitude probe-based carrier isolation.

ROOT CAUSE ANALYSIS (v1 & v2 failures):
  v1: Probe amplitude lambda_val ∝ |F_adv - F_clean| ≈ 0 at early steps.
      Gradient direction dominated by loss_org, not probe. Carrier phase irrelevant.
  v2: Frequency-domain energy ∝ |Delta_G|^2. Gradient ∝ |Delta_G| ≈ 0 at init.
      With L1 energy, gradient sign depends on random init, not carrier.

v3 FIX: Fixed-amplitude frequency-domain probe.
  - Probe = IFFT(FIXED_AMPLITUDE * A * exp(j*carrier_phase))
  - Amplitude is FIXED × clean spectrum, independent of current F_adv
  - NON-ZERO from step 1, gradient direction determined by carrier phase
  - Loss: minimize |F_adv - F_clean - probe| → push features along probe direction

Conditions:
  - SE-CGR:        carrier = exp(j*Phase_source)
  - phase-shuffled: carrier = exp(j*Phase_shuffled) [spatially permuted]
  - random-phase:   carrier = exp(j*rand_phase)     [uniform random]
  - tangential:     carrier = j*exp(j*Phase_source)  [orthogonal to source carrier]
  - no-probe:       no probe; directly maximize |F_adv - F_clean|
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
ATTACK_STEPS = 20
SEED_BASE = 42
EPSILON_SCALE = 16.0
STEP_SCALE = 2.0
FIXED_PROBE_AMPLITUDE = 2.0  # Fixed probe strength (independent of current F_adv)
PROBE_WEIGHT = 2.0            # Weight for probe loss term

criterion_l1 = torch.nn.L1Loss()

CONDITIONS = ["SE-CGR", "phase-shuffled", "random-phase", "tangential", "no-probe"]


def synthesize_fixed_probe(F_clean, mode):
    """
    Synthesize probe with FIXED amplitude (no dependence on F_adv).
    Only the CARRIER PHASE varies between conditions.
    """
    G_clean = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
    A = torch.abs(G_clean)
    Phase_s = torch.angle(G_clean)

    # Fixed-amplitude frequency modulation — same scale for ALL conditions
    Sigma_freq = FIXED_PROBE_AMPLITUDE * (A + 1e-12)

    if mode == "SE-CGR":
        # Source carrier — correct phase for radial drift
        fft_c = Sigma_freq * torch.exp(1j * Phase_s)
    elif mode == "phase-shuffled":
        # Spatially permuted source phase — preserves |A| but destroys spatial structure
        phase_flat = Phase_s.flatten()
        perm = torch.randperm(phase_flat.numel(), device=phase_flat.device)
        phase_shuf = phase_flat[perm].reshape(Phase_s.shape)
        fft_c = Sigma_freq * torch.exp(1j * phase_shuf)
    elif mode == "random-phase":
        # Random phase — completely unrelated to source
        rand_phase = torch.rand_like(Phase_s) * 2.0 * np.pi
        fft_c = Sigma_freq * torch.exp(1j * rand_phase)
    elif mode == "tangential":
        # Orthogonal to source carrier (tangential direction)
        fft_c = Sigma_freq * (1j * torch.exp(1j * Phase_s))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return torch.fft.ifftn(fft_c, dim=(-2, -1), norm="ortho").real


def run_matched_attack_v3(image_encoder, input_image, probe_mode):
    """Attack with fixed-amplitude probe. Only carrier phase varies."""
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

        # Base loss: NEGATIVE L1 = maximize feature distance (adversarial)
        # Using 0.9999 dampening on features for numerical stability
        loss_org = -criterion_l1(features, targets)

        # Multi-layer fixed-amplitude probe loss
        pre_layer = int(len(features_layer) * 0.9)
        num_layers = min(6, pre_layer)
        indices = random.sample(range(num_layers), max(1, num_layers - 2))

        fd_list = []
        for idx in indices:
            F_clean = targets_layer[idx + pre_layer - num_layers].unsqueeze(0)
            F_adv = features_layer[idx + pre_layer - num_layers].unsqueeze(0)

            if probe_mode == "no-probe":
                # No probe: directly maximize |F_adv - F_clean|
                fd = (F_adv - F_clean)
            else:
                probe = synthesize_fixed_probe(F_clean, probe_mode)
                # Residual after subtracting probe: push F_adv toward F_clean + probe
                fd = (F_adv - F_clean) - probe

            fd_list.append(fd.squeeze(0))

        fd = torch.stack(fd_list)
        im = _return_mean_value(torch.stack(
            [targets_layer[idx + pre_layer - num_layers] for idx in indices]
        ))
        if len(im.shape) > len(fd.shape):
            im = im.view(fd.shape[0], 1, 1, 1)
        fd = fd / (im + 1e-6)

        if probe_mode == "no-probe":
            # Maximize drift magnitude
            probe_loss = -torch.abs(torch.mean(fd, dim=0)).mean()
        else:
            # Minimize residual → F_adv approaches F_clean + probe
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
    print(f"Fixed probe amplitude: {FIXED_PROBE_AMPLITUDE}, Probe weight: {PROBE_WEIGHT}")
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
                    adv_np = run_matched_attack_v3(source_sam.image_encoder, img_t, probe_mode)
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

            adv_vol_dice = float(np.mean(adv_slices_dice))
            vol_delta = clean_vol_dice - adv_vol_dice
            volume_results[probe_mode].append({
                "case": case_name,
                "clean_vol_dice": clean_vol_dice,
                "adv_vol_dice": adv_vol_dice,
                "vol_delta": vol_delta,
                "n_organ_slices": n_slices,
            })

        parts = []
        for probe_mode in CONDITIONS:
            vd = volume_results[probe_mode][-1]["vol_delta"]
            parts.append(f"{probe_mode}={vd:+.5f}")
        print(f"  VOL: {' | '.join(parts)}")

        cases_done += 1

    # === SUMMARY ===
    print(f"\n{'='*70}")
    print(f"MATCHED ABLATION V3: Fixed-Amplitude Probe Carrier Isolation")
    print(f"{'='*70}")

    print(f"\n--- Slice-level ---")
    print(f"{'Condition':<18} {'N':<5} {'Mean Delta':<14} {'Pos%'}")
    print("-" * 42)
    for probe_mode in CONDITIONS:
        deltas = np.array([r["delta"] for r in slice_results[probe_mode]])
        print(f"{probe_mode:<18} {len(deltas):<5} {np.mean(deltas):<+14.6f} {100.0*np.mean(deltas>0):.0f}%")

    print(f"\n--- Volume-level (PRIMARY) ---")
    print(f"{'Condition':<18} {'Mean Vol Δ':<14} {'Std':<10} {'Pos%'}")
    print("-" * 46)
    vol_deltas_dict = {}
    for probe_mode in CONDITIONS:
        deltas = np.array([r["vol_delta"] for r in volume_results[probe_mode]])
        vol_deltas_dict[probe_mode] = deltas
        print(f"{probe_mode:<18} {np.mean(deltas):<+14.6f} {np.std(deltas):<10.6f} "
              f"{100.0*np.mean(deltas>0):.0f}%")

    # Paired tests
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

    # Key directional comparisons
    print(f"\n--- Key directional tests ---")
    try:
        from scipy.stats import wilcoxon
        for a, b, desc in [
            ("SE-CGR", "tangential", "SE-CGR vs tangential (carrier-aligned vs orthogonal)"),
            ("SE-CGR", "phase-shuffled", "SE-CGR vs phase-shuffled (correct vs wrong carrier)"),
            ("SE-CGR", "random-phase", "SE-CGR vs random-phase (correct vs random carrier)"),
            ("SE-CGR", "no-probe", "SE-CGR vs no-probe (carrier probe vs direct maximization)"),
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

    output = {
        "config": {"num_cases": cases_done, "attack_steps": ATTACK_STEPS,
                    "epsilon_scale": EPSILON_SCALE, "step_scale": STEP_SCALE,
                    "fixed_probe_amplitude": FIXED_PROBE_AMPLITUDE,
                    "probe_weight": PROBE_WEIGHT},
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
    out_path = os.path.join(OUTPUT_DIR, "matched_ablation_v3.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
