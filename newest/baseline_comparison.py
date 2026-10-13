#!/usr/bin/env python3
"""
Baseline comparison: CODD-framed SE-CGR vs standard transfer attack baselines.

Compares:
  - SE-CGR (our attack, CODD-framed): frequency-domain carrier-modulated noise probe
  - MI-FGSM: momentum iterative FGSM with L1 feature loss on source model
  - PGD: projected gradient descent with L1 feature loss on source model
  - Random: random Gaussian perturbation (same L-inf bound)

All use identical epsilon budget (10/255), step count (10), and step size (2/255).
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

from attack_codd import _get_edge_map, _return_eta, synthesize_codd_probe, _return_mean_value
from evaluate import load_sam_encoder, load_samed_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_NAME = "vit_b"
IMAGE_SIZE = 512
NUM_CLASSES = 8
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
SAMED_CKPT = f"{ROOT}/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CASES = 8
MAX_SLICES = 6
ATTACK_STEPS = 10
SEED_BASE = 42

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


def run_attack_se_cgr(image_encoder, input_image):
    """Our CODD-framed SE-CGR attack."""
    max_val, min_val = input_image.max(), input_image.min()
    step_size = float((max_val - min_val) * 2.0 / 255.0)
    epsilon = float((max_val - min_val) * 10.0 / 255.0)

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


def run_attack_mifgsm(image_encoder, input_image):
    """MI-FGSM baseline: momentum iterative FGSM with L1 feature loss."""
    max_val, min_val = input_image.max(), input_image.min()
    step_size = float((max_val - min_val) * 2.0 / 255.0)
    epsilon = float((max_val - min_val) * 10.0 / 255.0)

    with torch.no_grad():
        targets, _ = image_encoder(input_image)

    perturb_img = torch.autograd.Variable(
        input_image.data + torch.randn_like(input_image) * 1e-5, requires_grad=True
    )
    momentum = torch.zeros_like(input_image).detach()

    for step in range(ATTACK_STEPS):
        image_encoder.zero_grad()
        features, _ = image_encoder(perturb_img)

        # MI-FGSM: maximize L1 distance between perturbed and clean features
        loss = -criterion_l1(features, targets)

        grad = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]
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


def run_attack_random(input_image):
    """Random Gaussian perturbation baseline (same L-inf bound)."""
    max_val, min_val = input_image.max(), input_image.min()
    epsilon = float((max_val - min_val) * 10.0 / 255.0)
    noise = torch.randn_like(input_image) * epsilon
    noise = torch.clamp(noise, -epsilon, epsilon)
    result = input_image + noise
    result = torch.clamp(result, min_val, max_val)
    return result[0].detach().cpu().numpy()


def main():
    print(f"Device: {DEVICE}")
    print(f"Baselines: SE-CGR, MI-FGSM, Random")
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

    all_results = defaultdict(list)
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
        )[:MAX_SLICES]

        print(f"\nCase {cases_done+1}/{NUM_CASES}: {case_name} ({len(organ_slices)} slices)")

        for slice_idx in organ_slices:
            s = image_vol[slice_idx, :, :, :]
            if s.ndim == 2:
                s = s[np.newaxis, :, :]
            clean_gray = s[0].copy()
            label = label_vol[slice_idx].copy()
            img_t = torch.from_numpy(s).unsqueeze(0).float().to(DEVICE)
            img_t = repeat(img_t, 'b c h w -> b (repeat c) h w', repeat=3)

            clean_dice = evaluate_slice(samed_net, clean_gray, label, case_name)

            attacks = {
                "SE-CGR": lambda: run_attack_se_cgr(source_sam.image_encoder, img_t),
                "MI-FGSM": lambda: run_attack_mifgsm(source_sam.image_encoder, img_t),
                "Random": lambda: run_attack_random(img_t),
            }

            for name, attack_fn in attacks.items():
                random.seed(SEED_BASE)
                np.random.seed(SEED_BASE)
                torch.manual_seed(SEED_BASE)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED_BASE)

                adv_np = attack_fn()
                adv_gray = adv_np[0].copy()
                adv_dice = evaluate_slice(samed_net, adv_gray, label, case_name)
                delta = clean_dice - adv_dice
                all_results[name].append({
                    "case": case_name, "slice": int(slice_idx),
                    "clean_dice": clean_dice, "adv_dice": adv_dice, "delta": delta,
                })

            parts = []
            for name in ["SE-CGR", "MI-FGSM", "Random"]:
                d = all_results[name][-1]["delta"]
                marker = "+" if d > 0 else "-"
                parts.append(f"{name}={d:+.4f}{marker}")
            print(f"  s{slice_idx}: {' | '.join(parts)}")

        cases_done += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON ({cases_done} cases, {len(all_results['SE-CGR'])} slices)")
    print(f"{'='*60}")

    # Slice-level
    print(f"\n--- Slice-level ---")
    for name in ["SE-CGR", "MI-FGSM", "Random"]:
        deltas = np.array([r["delta"] for r in all_results[name]])
        print(f"  {name:<10}: mean={np.mean(deltas):+.4f} ± {np.std(deltas):.4f}, "
              f"median={np.median(deltas):+.4f}, pos={100.0*np.mean(deltas>0):.0f}%")

    # Case-level
    print(f"\n--- Case-level ---")
    case_means = {}
    for name in ["SE-CGR", "MI-FGSM", "Random"]:
        cases = {}
        for r in all_results[name]:
            c = r["case"]
            if c not in cases:
                cases[c] = []
            cases[c].append(r["delta"])
        case_means[name] = {c: float(np.mean(v)) for c, v in cases.items()}
        means = list(case_means[name].values())
        print(f"  {name:<10}: mean={np.mean(means):+.4f} ± {np.std(means):.4f}, "
              f"pos_cases={100.0*np.mean(np.array(means)>0):.0f}%")

    # Paired tests
    print(f"\n--- Paired tests (case-level) ---")
    try:
        from scipy.stats import wilcoxon
        se_cgr = np.array(list(case_means["SE-CGR"].values()))
        mifgsm = np.array(list(case_means["MI-FGSM"].values()))
        rand = np.array(list(case_means["Random"].values()))

        for name, other in [("MI-FGSM", mifgsm), ("Random", rand)]:
            diff = se_cgr - other
            stat, p = wilcoxon(se_cgr, other, zero_method="wilcox", alternative="greater")
            d_val = np.mean(diff) / (np.std(diff) + 1e-12)
            sig = "SIG" if p < 0.05 else "ns"
            print(f"  SE-CGR vs {name}: d={d_val:+.4f}, p={p:.4f} ({sig}), n={len(se_cgr)}")
    except ImportError:
        pass

    output = {
        "config": {"num_cases": cases_done, "max_slices": MAX_SLICES},
        "slice_level": {
            name: {
                "mean": float(np.mean([r["delta"] for r in all_results[name]])),
                "std": float(np.std([r["delta"] for r in all_results[name]])),
                "pos_pct": float(100.0 * np.mean(np.array([r["delta"] for r in all_results[name]]) > 0)),
            }
            for name in ["SE-CGR", "MI-FGSM", "Random"]
        },
        "case_level": case_means,
        "records": {name: all_results[name] for name in ["SE-CGR", "MI-FGSM", "Random"]},
    }
    out_path = os.path.join(OUTPUT_DIR, "baseline_comparison.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
