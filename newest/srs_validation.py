#!/usr/bin/env python3
"""
SRS (Shared Radial Subspace) validation (CODD C4).

Tests whether SRS warm-start improves attack effectiveness:
  1. Collect attack gradients on proxy data (first N cases)
  2. Fit SRS via PCA
  3. Test on held-out cases with 3 conditions:
     - No warm-start (baseline)
     - SRS warm-start (mean direction + subspace projection)
     - Random warm-start (control for any warm-start benefit)
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

NUM_PROXY_CASES = 4      # cases for fitting SRS
NUM_TEST_CASES = 4       # held-out cases for testing
MAX_SLICES = 4           # slices per case
ATTACK_STEPS = 10
SRS_RANK = 8

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
    """Collect a single normalized attack gradient."""
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

    grad_accum = None
    n_collected = 0

    for step in range(3):  # 3-step lightweight attack for gradient collection
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

        # Simple step
        perturb_img = torch.autograd.Variable(
            torch.clamp(perturb_img.data + step_size * grad_norm.sign(),
                       min_val, max_val),
            requires_grad=True,
        )
        eta = torch.clamp(perturb_img.data - input_image.data, -epsilon, epsilon)
        perturb_img = torch.autograd.Variable(input_image.data + eta, requires_grad=True)

    return grad_accum / n_collected


def run_attack_srs(image_encoder, input_image, config, srs_bundle=None):
    """Run attack with optional SRS warm-start and projection."""
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

    # SRS warm-start
    warm_start_type = "none"
    if srs_bundle is not None:
        mean_dir = srs_bundle.mean_direction.to(DEVICE)
        mean_dir_norm = mean_dir / (mean_dir.abs().sum() + 1e-12)
        momentum = momentum + config.srs_mean_weight * mean_dir_norm
        momentum = momentum.detach()
        warm_start_type = "srs"

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

        # SRS projection
        if srs_bundle is not None:
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

    return perturb_img[0].detach().cpu().numpy(), warm_start_type


def main():
    print(f"Device: {DEVICE}")
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

    config = AttackConfig(
        att_step_size=2.0, att_bound=10.0, att_numstep=ATTACK_STEPS,
        direction_mode="radial", seed=42,
        edge_guidance=True, edge_gamma=3.0,
        srs_proj_weight=0.15, srs_mean_weight=0.1,
    )

    # --- Phase 1: Collect gradients from proxy cases ---
    print(f"\n{'='*60}")
    print(f"Phase 1: Collecting gradients from {NUM_PROXY_CASES} proxy cases")
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

        case_idx += 1

    print(f"  Collected {len(gradient_list)} gradient samples")

    # Fit SRS
    print(f"\n  Fitting SRS (rank={SRS_RANK})...")
    srs_bundle = fit_shared_subspace(gradient_list, rank=SRS_RANK)
    print(f"  Explained variance: {srs_bundle.explained_variance:.4f}")
    print(f"  Singular values: {srs_bundle.singular_values.tolist()}")

    # Create random subspace as control
    shape = gradient_list[0].shape
    D = int(np.prod(shape))
    random_basis = torch.randn(SRS_RANK, D)
    random_basis = random_basis / (random_basis.norm(dim=1, keepdim=True) + 1e-12)
    random_mean = torch.randn(1, D)
    random_mean = random_mean / (random_mean.abs().sum() + 1e-12)

    from theory import SharedSubspace
    random_bundle = SharedSubspace(
        mean_direction=random_mean.reshape(shape),
        basis=random_basis,
        shape=shape,
        singular_values=torch.zeros(SRS_RANK),
        explained_variance=0.0,
    )

    # --- Phase 2: Test on held-out cases ---
    print(f"\n{'='*60}")
    print(f"Phase 2: Testing on {NUM_TEST_CASES} held-out cases")
    print(f"{'='*60}")

    # Skip proxy cases
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

            for mode, bundle in [("none", None), ("srs", srs_bundle), ("random", random_bundle)]:
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)

                adv_np, _ = run_attack_srs(
                    source_sam.image_encoder, img_t, config, srs_bundle=bundle,
                )
                adv_gray = adv_np[0].copy()
                adv_dice = evaluate_slice(samed_net, adv_gray, label, case_name)
                delta = clean_dice - adv_dice

                test_results[mode].append({
                    "case": case_name, "slice": int(slice_idx),
                    "clean_dice": clean_dice, "adv_dice": adv_dice, "delta": delta,
                })

            marker_none = "+" if test_results["none"][-1]["delta"] > 0 else "-"
            marker_srs = "+" if test_results["srs"][-1]["delta"] > 0 else "-"
            marker_rand = "+" if test_results["random"][-1]["delta"] > 0 else "-"
            print(f"  s{slice_idx}: none={test_results['none'][-1]['delta']:+.4f}{marker_none} "
                  f"srs={test_results['srs'][-1]['delta']:+.4f}{marker_srs} "
                  f"rand={test_results['random'][-1]['delta']:+.4f}{marker_rand}")

        test_case_idx += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"SRS VALIDATION SUMMARY")
    print(f"{'='*60}")

    for mode in ["none", "srs", "random"]:
        deltas = np.array([r["delta"] for r in test_results[mode]])
        print(f"\n{mode} (n={len(deltas)}):")
        print(f"  Mean delta:  {np.mean(deltas):+.4f} ± {np.std(deltas):.4f}")
        print(f"  Median:      {np.median(deltas):+.4f}")
        print(f"  Pos%:        {100.0*np.mean(deltas>0):.1f}%")

    # Paired tests
    none_deltas = np.array([r["delta"] for r in test_results["none"]])
    srs_deltas = np.array([r["delta"] for r in test_results["srs"]])
    rand_deltas = np.array([r["delta"] for r in test_results["random"]])

    try:
        from scipy.stats import wilcoxon
        for name, deltas in [("SRS", srs_deltas), ("Random", rand_deltas)]:
            stat, p = wilcoxon(deltas, none_deltas, zero_method="wilcox", alternative="greater")
            d_val = np.mean(deltas - none_deltas) / (np.std(deltas - none_deltas) + 1e-12)
            sig = "SIGNIFICANT" if p < 0.05 else "not sig"
            print(f"\n  {name} vs None: p={p:.4f} ({sig}), Cohen's d={d_val:.4f}")
    except ImportError:
        for name, deltas in [("SRS", srs_deltas), ("Random", rand_deltas)]:
            d_val = np.mean(deltas - none_deltas) / (np.std(deltas - none_deltas) + 1e-12)
            print(f"\n  {name} vs None: Cohen's d={d_val:.4f}")

    # Save
    output = {
        "config": {
            "num_proxy_cases": NUM_PROXY_CASES,
            "num_test_cases": test_case_idx,
            "max_slices": MAX_SLICES,
            "srs_rank": SRS_RANK,
            "explained_variance": float(srs_bundle.explained_variance),
        },
        "results": {
            mode: {
                "mean_delta": float(np.mean([r["delta"] for r in test_results[mode]])),
                "std_delta": float(np.std([r["delta"] for r in test_results[mode]])),
                "records": test_results[mode],
            }
            for mode in ["none", "srs", "random"]
        },
    }
    out_path = os.path.join(OUTPUT_DIR, "srs_validation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
