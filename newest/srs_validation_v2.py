#!/usr/bin/env python3
"""
SRS (Shared Radial Subspace) validation v2 — C4 test with improved design.

Key improvements over v1:
  1. More proxy cases (8 instead of 4) for better PCA
  2. Separates warm-start and projection effects
  3. Adds UAP warm-start as positive control
  4. Tests across multiple SRS ranks
  5. Per-slice paired comparisons with full statistical analysis

Conditions tested:
  - none: baseline (no warm-start, no projection)
  - srs_ws: SRS mean direction warm-start only (no projection)
  - srs_full: SRS warm-start + per-step projection (original design)
  - uap_ws: UAP warm-start (original SE-CGR approach, positive control)
  - random_ws: random direction warm-start (negative control)
"""
import os, sys, json, time, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from einops import repeat
from collections import defaultdict

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from theory import fit_shared_subspace, blend_with_subspace, decompose_drift
from attack_codd import (
    AttackConfig, _get_edge_map, _return_eta, _return_mean_value,
    synthesize_codd_probe,
)
from evaluate import load_sam_encoder, load_samed_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIT_NAME = "vit_b"
IMAGE_SIZE = 512
NUM_CLASSES = 8
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
SAMED_CKPT = f"{ROOT}/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_PROXY_CASES = 8
NUM_TEST_CASES = 6
MAX_SLICES = 5
ATTACK_STEPS = 10
SRS_RANKS = [8, 16, 32]
SEED_BASE = 42

CONDITIONS = ["none", "srs_ws", "srs_full", "uap_ws", "random_ws"]

criterion_l1 = torch.nn.L1Loss()


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


def collect_gradient(image_encoder, input_image, config):
    max_val, min_val = input_image.max(), input_image.min()
    step_size = config.att_step_size * (max_val - min_val) / 255.0
    epsilon = config.att_bound * (max_val - min_val) / 255.0

    with torch.no_grad():
        targets, targets_layer = image_encoder(input_image)

    edge_map = _get_edge_map(input_image).detach() if config.edge_guidance else None
    perturb_img = torch.autograd.Variable(
        input_image.data + torch.randn_like(input_image) * 1e-5, requires_grad=True
    )

    grad_accum = None
    n_collected = 0

    for step in range(5):  # 5-step for richer gradients
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
                direction="radial", use_random_noise=True, detach_adv=True,
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

        grad_norm = grad.detach() / (grad.detach().abs().sum() + 1e-12)

        if grad_accum is None:
            grad_accum = grad_norm.cpu()
        else:
            grad_accum = grad_accum + grad_norm.cpu()
        n_collected += 1

        perturb_img = torch.autograd.Variable(
            torch.clamp(perturb_img.data + step_size * grad_norm.sign(), min_val, max_val),
            requires_grad=True,
        )
        eta = torch.clamp(perturb_img.data - input_image.data, -epsilon, epsilon)
        perturb_img = torch.autograd.Variable(input_image.data + eta, requires_grad=True)

    return grad_accum / n_collected


def run_attack_srs(image_encoder, input_image, config, srs_bundle=None,
                   use_srs_proj=False, use_srs_ws=False, uap=None,
                   use_uap_ws=False, use_random_ws=False, random_mean=None):
    """Run attack with configurable warm-start and projection."""
    max_val, min_val = input_image.max(), input_image.min()
    step_size = config.att_step_size * (max_val - min_val) / 255.0
    epsilon = config.att_bound * (max_val - min_val) / 255.0

    with torch.no_grad():
        targets, targets_layer = image_encoder(input_image)

    edge_map = _get_edge_map(input_image).detach() if config.edge_guidance else None
    perturb_img = torch.autograd.Variable(
        input_image.data + torch.randn_like(input_image) * 1e-5, requires_grad=True
    )
    momentum = torch.zeros_like(input_image).detach()

    # Initialize momentum via warm-start
    if use_uap_ws and uap is not None:
        uap_norm_val = torch.norm(uap, p=1)
        if uap_norm_val > 1e-6:
            momentum = (uap / uap_norm_val).detach()
    elif use_srs_ws and srs_bundle is not None:
        mean_dir = srs_bundle.mean_direction.to(DEVICE)
        momentum = config.srs_mean_weight * (mean_dir / (mean_dir.abs().sum() + 1e-12))
        momentum = momentum.detach()
    elif use_random_ws and random_mean is not None:
        momentum = config.srs_mean_weight * random_mean.to(DEVICE)
        momentum = momentum.detach()

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
                direction="radial", use_random_noise=True, detach_adv=True,
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

        # SRS projection (per-step)
        if use_srs_proj and srs_bundle is not None:
            grad = blend_with_subspace(
                grad, srs_bundle,
                proj_weight=config.srs_proj_weight,
                mean_weight=0.0,
            )

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
    print(f"Proxy cases: {NUM_PROXY_CASES}, Test cases: {NUM_TEST_CASES}")
    print(f"SRS ranks: {SRS_RANKS}")
    print(f"Conditions: {CONDITIONS}")

    print("\nLoading models...")
    source_sam = load_sam_encoder(VIT_NAME, SAM_CKPT, IMAGE_SIZE, NUM_CLASSES, DEVICE)
    samed_net = load_samed_model(VIT_NAME, SAM_CKPT, SAMED_CKPT, IMAGE_SIZE, NUM_CLASSES, device=DEVICE)
    print("Done.")

    # Load UAP for positive control
    uap_path = f"{ROOT}/common_perb/Synapse.pth"
    uap = None
    if os.path.exists(uap_path):
        uap = torch.load(uap_path, map_location=DEVICE)
        print(f"UAP loaded: shape={uap.shape}")
    else:
        print("WARNING: UAP not found, skipping UAP warm-start condition")

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
        srs_proj_weight=0.15, srs_mean_weight=0.1,
    )

    # ================================================================
    # Phase 1: Collect gradients from proxy cases
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Phase 1: Collecting gradients from {NUM_PROXY_CASES} proxy cases")
    print(f"{'='*60}")

    gradient_list = []
    proxy_cases = []
    case_idx = 0
    uap_gradient_list = []

    for batch in loader:
        if case_idx >= NUM_PROXY_CASES:
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
        )[:MAX_SLICES]

        print(f"  Proxy case {case_idx+1}: {case_name} ({len(organ_slices)} slices)")
        proxy_cases.append(case_name)

        for slice_idx in organ_slices:
            s = image_vol[slice_idx, :, :, :]
            if s.ndim == 2:
                s = s[np.newaxis, :, :]
            img_t = torch.from_numpy(s).unsqueeze(0).float().to(DEVICE)
            img_t = repeat(img_t, 'b c h w -> b (repeat c) h w', repeat=3)

            g = collect_gradient(source_sam.image_encoder, img_t, config)
            gradient_list.append(g)

            if uap is not None:
                uap_slice = uap[:1] if uap.shape[0] > 1 else uap
                uap_norm_val = uap_slice / (uap_slice.abs().sum() + 1e-12)
                uap_gradient_list.append(uap_norm_val.cpu())

        case_idx += 1

    print(f"  Collected {len(gradient_list)} gradient samples ({len(gradient_list[0].shape)}-dim each)")

    # ================================================================
    # Fit SRS at multiple ranks
    # ================================================================
    print(f"\n  Fitting SRS at ranks {SRS_RANKS}...")
    srs_bundles = {}
    for rank in SRS_RANKS:
        b = fit_shared_subspace(gradient_list, rank=rank)
        srs_bundles[rank] = b
        print(f"    Rank {rank}: explained_var={b.explained_variance:.4f}, "
              f"singular_values=[{b.singular_values[0]:.4f}...{b.singular_values[-1]:.4f}]")

    # Random mean for negative control
    shape = gradient_list[0].shape
    D = int(np.prod(shape))
    random_mean = torch.randn(1, D)
    random_mean = random_mean / (random_mean.abs().sum() + 1e-12)
    random_mean = random_mean.reshape(shape)

    # ================================================================
    # Phase 2: Test on held-out cases
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: Testing on {NUM_TEST_CASES} held-out cases")
    print(f"{'='*60}")

    test_loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=0)

    # Results: test_results[condition][rank] = list of records
    test_results = defaultdict(lambda: defaultdict(list))

    test_case_idx = 0
    for batch in test_loader:
        case_name = batch["case_name"][0]
        if case_name in proxy_cases:
            continue
        if test_case_idx >= NUM_TEST_CASES:
            break

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
        )[:MAX_SLICES]

        print(f"\nTest case {test_case_idx+1}: {case_name} ({len(organ_slices)} slices)")

        for slice_idx in organ_slices:
            s = image_vol[slice_idx, :, :, :]
            if s.ndim == 2:
                s = s[np.newaxis, :, :]
            clean_gray = s[0].copy()
            label = label_vol[slice_idx].copy()
            img_t = torch.from_numpy(s).unsqueeze(0).float().to(DEVICE)
            img_t = repeat(img_t, 'b c h w -> b (repeat c) h w', repeat=3)

            clean_dice = evaluate_slice(samed_net, clean_gray, label, case_name)

            # --- Test all conditions ---
            for condition in CONDITIONS:
                for rank in SRS_RANKS if condition in ("srs_ws", "srs_full") else [None]:
                    random.seed(SEED_BASE)
                    np.random.seed(SEED_BASE)
                    torch.manual_seed(SEED_BASE)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(SEED_BASE)

                    bundle = srs_bundles[rank] if condition in ("srs_ws", "srs_full") and rank is not None else None

                    adv_np = run_attack_srs(
                        source_sam.image_encoder, img_t, config,
                        srs_bundle=bundle,
                        use_srs_ws=(condition == "srs_ws" or condition == "srs_full"),
                        use_srs_proj=(condition == "srs_full"),
                        uap=uap,
                        use_uap_ws=(condition == "uap_ws"),
                        use_random_ws=(condition == "random_ws"),
                        random_mean=random_mean,
                    )
                    adv_gray = adv_np[0].copy()
                    adv_dice = evaluate_slice(samed_net, adv_gray, label, case_name)
                    delta = clean_dice - adv_dice

                    test_results[condition][rank].append({
                        "case": case_name, "slice": int(slice_idx),
                        "clean_dice": clean_dice, "adv_dice": adv_dice, "delta": delta,
                    })

            # Print per-condition deltas for this slice
            parts = []
            for condition in CONDITIONS:
                if condition in ("srs_ws", "srs_full"):
                    rank_key = SRS_RANKS[-1]  # show highest rank
                    d = test_results[condition][rank_key][-1]["delta"]
                    marker = "+" if d > 0 else "-"
                    parts.append(f"{condition}(r{rank_key})={d:+.4f}{marker}")
                else:
                    d = test_results[condition][None][-1]["delta"]
                    marker = "+" if d > 0 else "-"
                    parts.append(f"{condition}={d:+.4f}{marker}")
            print(f"  s{slice_idx}: {' | '.join(parts)}")

        test_case_idx += 1

    # ================================================================
    # Summary analysis
    # ================================================================
    print(f"\n{'='*60}")
    print(f"SRS VALIDATION SUMMARY")
    print(f"{'='*60}")

    # Aggregate results
    summary = {}
    for condition in CONDITIONS:
        if condition in ("srs_ws", "srs_full"):
            for rank in SRS_RANKS:
                deltas = np.array([r["delta"] for r in test_results[condition][rank]])
                key = f"{condition}_r{rank}"
                summary[key] = {
                    "n": len(deltas),
                    "mean": float(np.mean(deltas)),
                    "std": float(np.std(deltas)),
                    "median": float(np.median(deltas)),
                    "pos_pct": float(100.0 * np.mean(deltas > 0)),
                }
        else:
            deltas = np.array([r["delta"] for r in test_results[condition][None]])
            summary[condition] = {
                "n": len(deltas),
                "mean": float(np.mean(deltas)),
                "std": float(np.std(deltas)),
                "median": float(np.median(deltas)),
                "pos_pct": float(100.0 * np.mean(deltas > 0)),
            }

    print(f"\n{'Condition':<20} {'N':<5} {'Mean Delta':<14} {'Std':<10} {'Median':<10} {'Pos%':<8}")
    print("-" * 67)
    for cond_name, stats in summary.items():
        print(f"{cond_name:<20} {stats['n']:<5} {stats['mean']:<+14.6f} {stats['std']:<10.6f} "
              f"{stats['median']:<+10.6f} {stats['pos_pct']:<8.1f}")

    # Paired statistical tests: each condition vs none
    print(f"\n--- Paired tests vs None ---")
    none_deltas = np.array([r["delta"] for r in test_results["none"][None]])

    try:
        from scipy.stats import wilcoxon
        for condition in CONDITIONS:
            if condition == "none":
                continue
            if condition in ("srs_ws", "srs_full"):
                for rank in SRS_RANKS:
                    deltas = np.array([r["delta"] for r in test_results[condition][rank]])
                    diff = deltas - none_deltas
                    stat, p = wilcoxon(deltas, none_deltas, zero_method="wilcox", alternative="greater")
                    d_val = np.mean(diff) / (np.std(diff) + 1e-12)
                    sig = "SIG" if p < 0.05 else "ns"
                    print(f"  {condition}_r{rank}: p={p:.4f} ({sig}), Cohen's d={d_val:+.4f}")
            else:
                deltas = np.array([r["delta"] for r in test_results[condition][None]])
                diff = deltas - none_deltas
                stat, p = wilcoxon(deltas, none_deltas, zero_method="wilcox", alternative="greater")
                d_val = np.mean(diff) / (np.std(diff) + 1e-12)
                sig = "SIG" if p < 0.05 else "ns"
                print(f"  {condition}: p={p:.4f} ({sig}), Cohen's d={d_val:+.4f}")
    except ImportError:
        print("  scipy not available")

    # Save
    output = {
        "config": {
            "num_proxy_cases": NUM_PROXY_CASES,
            "num_test_cases": test_case_idx,
            "max_slices": MAX_SLICES,
            "srs_ranks": SRS_RANKS,
            "conditions": CONDITIONS,
        },
        "summary": summary,
        "srs_explained_variance": {str(r): float(srs_bundles[r].explained_variance) for r in SRS_RANKS},
        "results": {
            condition: {
                str(rank): test_results[condition][rank]
                for rank in (SRS_RANKS if condition in ("srs_ws", "srs_full") else [None])
            }
            for condition in CONDITIONS
        },
    }
    out_path = os.path.join(OUTPUT_DIR, "srs_validation_v2.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
