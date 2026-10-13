#!/usr/bin/env python3
"""
C4 SRS validation v3 — Persistent prior (non-decaying bias term).

Key fix: The original SRS warm-start initializes momentum once, which is then
erased by momentum decay (0.5^step ≈ 0 after 5 steps). This version adds
a PERSISTENT bias term that guides ALL attack steps.

Conditions:
  - none: baseline (no prior)
  - uap_persist: UAP as persistent bias (not just warm-start)
  - uap_ws: UAP warm-start only (original SE-CGR)
  - srs_persist: SRS mean direction as persistent bias
  - srs_ws: SRS warm-start only (original design)
  - random_persist: random direction as persistent bias (negative control)
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

from theory import fit_shared_subspace
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
SRS_RANK = 16
SEED_BASE = 42
PERSIST_WEIGHT = 0.05  # weight of persistent bias term

CONDITIONS = ["none", "uap_persist", "uap_ws", "srs_persist", "srs_ws", "random_persist"]

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

    for step in range(3):
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


def run_attack_persistent(image_encoder, input_image, config,
                          uap=None, srs_mean=None, random_mean=None,
                          use_persist_uap=False, use_persist_srs=False,
                          use_persist_random=False, use_ws_uap=False,
                          use_ws_srs=False):
    """Run attack with PERSISTENT prior (non-decaying bias)."""
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

    # Compute persistent bias direction
    persistent_bias = None
    if use_persist_uap and uap is not None:
        persistent_bias = uap / (uap.abs().sum() + 1e-12)
    elif use_persist_srs and srs_mean is not None:
        persistent_bias = srs_mean.to(DEVICE)
        persistent_bias = persistent_bias / (persistent_bias.abs().sum() + 1e-12)
    elif use_persist_random and random_mean is not None:
        persistent_bias = random_mean.to(DEVICE)
        persistent_bias = persistent_bias / (persistent_bias.abs().sum() + 1e-12)

    # Warm-start (one-time momentum init)
    if use_ws_uap and uap is not None:
        uap_norm_val = torch.norm(uap, p=1)
        if uap_norm_val > 1e-6:
            momentum = (uap / uap_norm_val).detach()
    elif use_ws_srs and srs_mean is not None:
        ms = srs_mean.to(DEVICE)
        momentum = 0.1 * (ms / (ms.abs().sum() + 1e-12))
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

        momentum = config.momentum_decay * momentum.detach() + grad

        # Add persistent bias EVERY step (key difference from warm-start)
        if persistent_bias is not None:
            momentum = momentum + PERSIST_WEIGHT * persistent_bias

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
    print(f"Proxy cases: {NUM_PROXY_CASES}, Test: {NUM_TEST_CASES}")
    print(f"Persistent bias weight: {PERSIST_WEIGHT}")
    print(f"Conditions: {CONDITIONS}")

    print("\nLoading models...")
    source_sam = load_sam_encoder(VIT_NAME, SAM_CKPT, IMAGE_SIZE, NUM_CLASSES, DEVICE)
    samed_net = load_samed_model(VIT_NAME, SAM_CKPT, SAMED_CKPT, IMAGE_SIZE, NUM_CLASSES, device=DEVICE)
    print("Done.")

    # Load UAP
    uap_path = f"{ROOT}/common_perb/Synapse.pth"
    uap = torch.load(uap_path, map_location=DEVICE) if os.path.exists(uap_path) else None
    if uap is not None:
        print(f"UAP loaded: shape={uap.shape}")
        uap_slice = uap[:1] if uap.dim() == 4 else uap
    else:
        print("WARNING: UAP not found")

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
    )

    # Phase 1: Collect gradients and fit SRS
    print(f"\n{'='*60}")
    print(f"Phase 1: Collecting gradients, fitting SRS")
    print(f"{'='*60}")

    gradient_list = []
    proxy_cases = []
    case_idx = 0

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

        proxy_cases.append(case_name)
        for slice_idx in organ_slices:
            s = image_vol[slice_idx, :, :, :]
            if s.ndim == 2:
                s = s[np.newaxis, :, :]
            img_t = torch.from_numpy(s).unsqueeze(0).float().to(DEVICE)
            img_t = repeat(img_t, 'b c h w -> b (repeat c) h w', repeat=3)
            g = collect_gradient(source_sam.image_encoder, img_t, config)
            gradient_list.append(g)
        case_idx += 1
        print(f"  Proxy {case_idx}: {case_name}")

    print(f"  Collected {len(gradient_list)} gradient samples")
    srs_bundle = fit_shared_subspace(gradient_list, rank=SRS_RANK)
    print(f"  SRS rank={SRS_RANK}, explained_var={srs_bundle.explained_variance:.4f}")
    srs_mean = srs_bundle.mean_direction

    # Random mean for negative control
    shape = gradient_list[0].shape
    D = int(np.prod(shape))
    random_mean = torch.randn(1, D)
    random_mean = random_mean / (random_mean.abs().sum() + 1e-12)
    random_mean = random_mean.reshape(shape)

    # Phase 2: Test
    print(f"\n{'='*60}")
    print(f"Phase 2: Testing on {NUM_TEST_CASES} held-out cases")
    print(f"{'='*60}")

    test_loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=0)
    test_results = defaultdict(list)
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

            for condition in CONDITIONS:
                random.seed(SEED_BASE)
                np.random.seed(SEED_BASE)
                torch.manual_seed(SEED_BASE)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED_BASE)

                adv_np = run_attack_persistent(
                    source_sam.image_encoder, img_t, config,
                    uap=uap_slice if uap is not None else None,
                    srs_mean=srs_mean,
                    random_mean=random_mean,
                    use_persist_uap=(condition == "uap_persist"),
                    use_persist_srs=(condition == "srs_persist"),
                    use_persist_random=(condition == "random_persist"),
                    use_ws_uap=(condition == "uap_ws"),
                    use_ws_srs=(condition == "srs_ws"),
                )
                adv_gray = adv_np[0].copy()
                adv_dice = evaluate_slice(samed_net, adv_gray, label, case_name)
                delta = clean_dice - adv_dice
                test_results[condition].append({
                    "case": case_name, "slice": int(slice_idx),
                    "clean_dice": clean_dice, "adv_dice": adv_dice, "delta": delta,
                })

            parts = []
            for condition in CONDITIONS:
                d = test_results[condition][-1]["delta"]
                marker = "+" if d > 0 else "-"
                parts.append(f"{condition}={d:+.4f}{marker}")
            print(f"  s{slice_idx}: {' | '.join(parts)}")

        test_case_idx += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"SRS PERSISTENT PRIOR SUMMARY")
    print(f"{'='*60}")

    # Slice-level
    print(f"\n--- Slice-level ---")
    print(f"{'Condition':<18} {'N':<4} {'Mean Delta':<14} {'Std':<10} {'Pos%'}")
    print("-" * 50)
    slice_stats = {}
    for condition in CONDITIONS:
        deltas = np.array([r["delta"] for r in test_results[condition]])
        slice_stats[condition] = {
            "mean": float(np.mean(deltas)), "std": float(np.std(deltas)),
            "pos_pct": float(100.0 * np.mean(deltas > 0)),
        }
        print(f"{condition:<18} {len(deltas):<4} {np.mean(deltas):<+14.6f} "
              f"{np.std(deltas):<10.6f} {100.0*np.mean(deltas>0):.0f}%")

    # Case-level
    print(f"\n--- Case-level ---")
    case_means = {c: {} for c in CONDITIONS}
    for condition in CONDITIONS:
        recs = test_results[condition]
        cases = {}
        for r in recs:
            c = r["case"]
            if c not in cases:
                cases[c] = []
            cases[c].append(r["delta"])
        for c, deltas in cases.items():
            case_means[condition][c] = float(np.mean(deltas))

    for condition in CONDITIONS:
        means = list(case_means[condition].values())
        print(f"  {condition:<18}: mean={np.mean(means):+.6f}, "
              f"pos_cases={100.0*np.mean(np.array(means)>0):.0f}%, n={len(means)}")

    # Paired tests vs none
    none_means_arr = np.array(list(case_means["none"].values()))
    print(f"\n--- Paired vs None (case-level) ---")
    try:
        from scipy.stats import wilcoxon
        for condition in CONDITIONS:
            if condition == "none":
                continue
            common = sorted(set(case_means["none"].keys()) & set(case_means[condition].keys()))
            d1 = np.array([case_means[condition][c] for c in common])
            d0 = np.array([case_means["none"][c] for c in common])
            diff = d1 - d0
            stat, p = wilcoxon(d1, d0, zero_method="wilcox", alternative="greater")
            d_val = np.mean(diff) / (np.std(diff) + 1e-12)
            sig = "SIG" if p < 0.05 else "ns"
            print(f"  {condition:<18}: d={d_val:+.4f}, p={p:.4f} ({sig}), n={len(common)}")
    except ImportError:
        print("  scipy not available")

    # Save
    output = {
        "config": {
            "num_proxy_cases": NUM_PROXY_CASES,
            "num_test_cases": test_case_idx,
            "max_slices": MAX_SLICES,
            "srs_rank": SRS_RANK,
            "persist_weight": PERSIST_WEIGHT,
            "conditions": CONDITIONS,
            "srs_explained_variance": float(srs_bundle.explained_variance),
        },
        "slice_level": slice_stats,
        "case_level": case_means,
        "results": {c: test_results[c] for c in CONDITIONS},
    }
    out_path = os.path.join(OUTPUT_DIR, "srs_validation_v3.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
