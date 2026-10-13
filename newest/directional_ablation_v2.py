#!/usr/bin/env python3
"""
Directional ablation with statistical power.
Tests C3: radial attacks degrade victim segmentation more than tangential.

Uses random-noise mode for consistent attack strength across directions,
with directional modulation via carrier phase (radial = in-phase,
tangential = 90° out-of-phase, isotropic = mixed, misaligned = random phase).

Evaluates per-slice dice on SAMed, then aggregates with paired statistical tests.
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

from theory import synthesize_codd_probe
from attack_codd import AttackConfig, _get_edge_map, _return_eta, _return_mean_value
from evaluate import load_sam_encoder, load_samed_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIT_NAME = "vit_b"
IMAGE_SIZE = 512
NUM_CLASSES = 8
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
SAMED_CKPT = f"{ROOT}/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CASES = 10
MAX_SLICES_PER_CASE = 8
ATTACK_STEPS = 10

DIRECTIONS = ["radial", "tangential", "isotropic", "misaligned"]
SEED_BASE = 42

criterion_l1 = torch.nn.L1Loss()


def evaluate_slice_samed(net, image_np, label_np, case_name):
    """Evaluate SAMed on a single slice. Returns dice score."""
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


def run_attack_slice(image_encoder, input_image, targets, targets_layer,
                     direction, use_random_noise, config):
    """Run directional attack on a single slice. Returns adversarial image numpy [3,H,W]."""
    max_val, min_val = input_image.max(), input_image.min()
    step_size = float((max_val - min_val) * config.att_step_size / 255.0)
    epsilon = float((max_val - min_val) * config.att_bound / 255.0)

    edge_map = _get_edge_map(input_image).detach() if config.edge_guidance else None
    perturb_img = torch.autograd.Variable(
        input_image.data + torch.randn_like(input_image) * 1e-5, requires_grad=True
    )
    momentum = torch.zeros_like(input_image).detach()

    for step in range(config.att_numstep):
        image_encoder.zero_grad()
        perturb_input = perturb_img
        if step == 0:
            perturb_input = perturb_img + torch.randn_like(perturb_img) * 0.0001

        features, features_layer = image_encoder(perturb_input)
        loss_org = criterion_l1(features * 0.9999, targets)

        pre_layer = int(len(features_layer) * 0.9)
        num_layers = min(6, pre_layer)
        indices = random.sample(range(num_layers), max(1, num_layers - 2))

        fd_list = []
        for idx in indices:
            F_clean = targets_layer[idx + pre_layer - num_layers].unsqueeze(0)
            F_adv = features_layer[idx + pre_layer - num_layers].unsqueeze(0)
            probe = synthesize_codd_probe(
                F_clean=F_clean, F_adv=F_adv,
                direction=direction,
                use_random_noise=use_random_noise,
                detach_adv=True,
            )
            fd = (F_adv * 0.999 - F_clean) - probe
            fd_list.append(fd.squeeze(0))

        fd = torch.stack(fd_list)
        im = _return_mean_value(torch.stack(
            [targets_layer[idx + pre_layer - num_layers] for idx in indices]
        ))
        if len(im.shape) > len(fd.shape):
            im = im.view(fd.shape[0], 1, 1, 1)
        fd = fd / (im + 1e-6)
        l1_losses = (torch.abs(torch.mean(fd, dim=0))).mean()
        loss = loss_org + l1_losses

        grad = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]
        if config.edge_guidance and edge_map is not None:
            grad = grad * (1.0 + config.edge_gamma * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)

        momentum = config.momentum_decay * momentum.detach() + grad
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


def main():
    print(f"Device: {DEVICE}")
    print(f"Cases: {NUM_CASES}, Max slices/case: {MAX_SLICES_PER_CASE}")

    # Load models
    print("Loading models...")
    t0 = time.time()
    source_sam = load_sam_encoder(VIT_NAME, SAM_CKPT, IMAGE_SIZE, NUM_CLASSES, DEVICE)
    samed_net = load_samed_model(VIT_NAME, SAM_CKPT, SAMED_CKPT, IMAGE_SIZE, NUM_CLASSES, device=DEVICE)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Load dataset
    from datasets.dataset_synapse import Synapse_dataset
    db = Synapse_dataset(
        base_dir=f"{ROOT}/data/Synapse_preprocess/test_vol_h5",
        list_dir=f"{ROOT}/data/Synapse_preprocess/lists",
        split="test_vol",
    )
    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=0)

    config = AttackConfig(
        att_step_size=2.0, att_bound=10.0, att_numstep=ATTACK_STEPS,
        direction_mode="radial", seed=SEED_BASE,
        edge_guidance=True, edge_gamma=3.0,
        directional_weight=1.0, probe_amplification=1.0,
    )

    # Collect per-slice deltas for each direction
    all_deltas = {d: [] for d in DIRECTIONS}  # list of (clean - adv) per slice
    per_case = defaultdict(lambda: defaultdict(list))

    cases_processed = 0
    for batch in loader:
        if cases_processed >= NUM_CASES:
            break

        case_name = batch["case_name"][0] if "case_name" in batch else f"case_{cases_processed}"

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

        # Organ-containing slices sorted by organ pixels (descending)
        organ_slices = sorted(
            [s for s in range(depth) if (label_vol[s] > 0).sum() > 100],
            key=lambda s: (label_vol[s] > 0).sum(),
            reverse=True,
        )
        step = max(1, len(organ_slices) // MAX_SLICES_PER_CASE)
        selected = organ_slices[:MAX_SLICES_PER_CASE * step:step][:MAX_SLICES_PER_CASE]

        print(f"\nCase {cases_processed+1}/{NUM_CASES}: {case_name} "
              f"(organ slices: {len(organ_slices)}, selected: {len(selected)})")

        for slice_idx in selected:
            s = image_vol[slice_idx, :, :, :]
            if s.ndim == 2:
                s = s[np.newaxis, :, :]
            clean_gray = s[0].copy()
            label = label_vol[slice_idx].copy()
            img_t = torch.from_numpy(s).unsqueeze(0).float().to(DEVICE)
            img_t = repeat(img_t, 'b c h w -> b (repeat c) h w', repeat=3)

            # Clean dice
            clean_dice = evaluate_slice_samed(samed_net, clean_gray, label, case_name)

            # Pre-compute clean features for reuse
            with torch.no_grad():
                targets, targets_layer = source_sam.image_encoder(img_t)

            for direction in DIRECTIONS:
                random.seed(SEED_BASE)
                np.random.seed(SEED_BASE)
                torch.manual_seed(SEED_BASE)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED_BASE)

                adv_np = run_attack_slice(
                    source_sam.image_encoder, img_t, targets, targets_layer,
                    direction=direction, use_random_noise=True, config=config,
                )
                adv_gray = adv_np[0].copy()
                adv_dice = evaluate_slice_samed(samed_net, adv_gray, label, case_name)
                delta = clean_dice - adv_dice
                all_deltas[direction].append(delta)
                per_case[case_name][direction].append(delta)

        cases_processed += 1

    # Statistical analysis
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY ({cases_processed} cases, {len(all_deltas['radial'])} slices)")
    print(f"{'='*70}")

    print(f"\n{'Direction':<14} {'Mean Delta':<12} {'Std':<10} {'Median':<10} {'Pos%':<8} {'Mean Dice':<12}")
    print("-" * 66)

    clean_dices_all = []
    direction_means = {}
    for direction in DIRECTIONS:
        deltas = np.array(all_deltas[direction])
        mean_d = np.mean(deltas)
        std_d = np.std(deltas)
        med_d = np.median(deltas)
        pos_pct = 100.0 * np.mean(deltas > 0)
        direction_means[direction] = mean_d
        print(f"{direction:<14} {mean_d:<+12.6f} {std_d:<10.6f} {med_d:<+10.6f} {pos_pct:<8.1f}")

    # Paired statistical tests
    print(f"\n--- Paired tests (Wilcoxon signed-rank, Cohen's d) ---")
    try:
        from scipy.stats import wilcoxon
        for i, d1 in enumerate(DIRECTIONS):
            for d2 in DIRECTIONS[i+1:]:
                deltas1 = np.array(all_deltas[d1])
                deltas2 = np.array(all_deltas[d2])
                # Wilcoxon test
                stat, p = wilcoxon(deltas1, deltas2, zero_method="wilcox", alternative="two-sided")
                # Cohen's d
                diff = deltas1 - deltas2
                d_val = np.mean(diff) / (np.std(diff) + 1e-12)
                # Rad > Tan?
                better = "radial > tangential" if np.mean(deltas1) > np.mean(deltas2) else "tangential > radial"
                print(f"  {d1} vs {d2}: p={p:.4f}, Cohen's d={d_val:.4f} ({better})")
    except ImportError:
        print("  scipy not available, using numpy-based tests")
        for i, d1 in enumerate(DIRECTIONS):
            for d2 in DIRECTIONS[i+1:]:
                deltas1 = np.array(all_deltas[d1])
                deltas2 = np.array(all_deltas[d2])
                diff = deltas1 - deltas2
                d_val = np.mean(diff) / (np.std(diff) + 1e-12)
                better = "radial > tangential" if np.mean(deltas1) > np.mean(deltas2) else "tangential > radial"
                print(f"  {d1} vs {d2}: Cohen's d={d_val:.4f} ({better})")

    # Per-case summary
    print(f"\n--- Per-case mean deltas ---")
    print(f"{'Case':<16}", end="")
    for d in DIRECTIONS:
        print(f"{d:<12}", end="")
    print()
    for case_name in sorted(per_case.keys()):
        print(f"{case_name:<16}", end="")
        for d in DIRECTIONS:
            mean_delta = np.mean(per_case[case_name][d])
            print(f"{mean_delta:<+12.6f}", end="")
        print()

    # Save results
    output = {
        "config": {
            "num_cases": cases_processed,
            "max_slices_per_case": MAX_SLICES_PER_CASE,
            "total_slices": len(all_deltas["radial"]),
            "attack_steps": ATTACK_STEPS,
            "vit_name": VIT_NAME,
            "mode": "random_noise_directional",
        },
        "direction_means": {d: float(np.mean(all_deltas[d])) for d in DIRECTIONS},
        "direction_stds": {d: float(np.std(all_deltas[d])) for d in DIRECTIONS},
        "direction_medians": {d: float(np.median(all_deltas[d])) for d in DIRECTIONS},
        "direction_pos_pct": {d: float(100.0 * np.mean(np.array(all_deltas[d]) > 0)) for d in DIRECTIONS},
        "all_deltas": {d: [float(x) for x in all_deltas[d]] for d in DIRECTIONS},
        "per_case": {
            case: {d: [float(x) for x in vals] for d, vals in case_data.items()}
            for case, case_data in per_case.items()
        },
    }
    out_path = os.path.join(OUTPUT_DIR, "directional_ablation_v2.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
