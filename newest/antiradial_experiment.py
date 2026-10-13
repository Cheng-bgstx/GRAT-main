#!/usr/bin/env python3
"""
Anti-radial experiment: Critical causal test of CODD C3.

Two attack variants:
  1. Normal radial:  feature_difference = drift - probe  (pushes features radially)
  2. Anti-radial:    feature_difference = drift + probe  (pushes features anti-radially)

CODD predicts: anti-radial produces lower radial_ratio AND lower attack effectiveness.

Uses the SE-CGR random noise mechanism for strong gradients in both cases.
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

from theory import decompose_drift, synthesize_codd_probe
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

NUM_CASES = 8
MAX_SLICES = 6
ATTACK_STEPS = 10

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


def run_attack(image_encoder, input_image, anti_radial=False):
    """Run SE-CGR attack. anti_radial=True uses +probe instead of -probe."""
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

            if anti_radial:
                # Push features in ANTI-RADIAL direction (away from source carrier)
                fd = (F_adv * 0.999 - F_clean) + probe
            else:
                # Push features in RADIAL direction (along source carrier) — original
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

    records = []
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

            for mode in ["radial", "antiradial"]:
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)

                anti = (mode == "antiradial")
                adv_np = run_attack(source_sam.image_encoder, img_t, anti_radial=anti)

                # Drift analysis
                adv_tensor = torch.from_numpy(adv_np).unsqueeze(0).float().to(DEVICE)
                with torch.no_grad():
                    src_feat, _ = source_sam.image_encoder(img_t)
                    src_adv_feat, _ = source_sam.image_encoder(adv_tensor)
                drift_stats = decompose_drift(src_feat, src_adv_feat)

                # Dice evaluation
                adv_gray = adv_np[0].copy()
                adv_dice = evaluate_slice(samed_net, adv_gray, label, case_name)

                records.append({
                    "case": case_name, "slice": int(slice_idx),
                    "mode": mode, "clean_dice": clean_dice,
                    "adv_dice": adv_dice, "delta": clean_dice - adv_dice,
                    "radial_ratio": drift_stats.radial_ratio,
                    "tangential_ratio": drift_stats.tangential_ratio,
                })
                marker = "+" if clean_dice - adv_dice > 0 else "-"
                print(f"  s{slice_idx} {mode:<12}: delta={clean_dice-adv_dice:+.4f} {marker} "
                      f"radial={drift_stats.radial_ratio:.4f}")

            sys.stdout.flush()

        cases_done += 1

    # Analysis
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(records)} records from {cases_done} cases")
    print(f"{'='*70}")

    radial_recs = [r for r in records if r["mode"] == "radial"]
    anti_recs = [r for r in records if r["mode"] == "antiradial"]

    for label, recs in [("Radial", radial_recs), ("Anti-radial", anti_recs)]:
        deltas = np.array([r["delta"] for r in recs])
        rr = np.array([r["radial_ratio"] for r in recs])
        pos_pct = 100.0 * np.mean(deltas > 0)
        print(f"\n{label} (n={len(recs)}):")
        print(f"  Delta:          mean={np.mean(deltas):+.4f}, std={np.std(deltas):.4f}, "
              f"median={np.median(deltas):+.4f}")
        print(f"  Radial ratio:   mean={np.mean(rr):.4f}, std={np.std(rr):.4f}")
        print(f"  Pos% (works):   {pos_pct:.1f}%")

    # Paired test
    paired_deltas = []
    paired_rr = []
    for rr_rec, ar_rec in zip(radial_recs, anti_recs):
        paired_deltas.append((rr_rec["delta"], ar_rec["delta"]))
        paired_rr.append((rr_rec["radial_ratio"], ar_rec["radial_ratio"]))

    radial_deltas = np.array([p[0] for p in paired_deltas])
    anti_deltas = np.array([p[1] for p in paired_deltas])
    radial_rr = np.array([p[0] for p in paired_rr])
    anti_rr = np.array([p[1] for p in paired_rr])

    # Wilcoxon test
    try:
        from scipy.stats import wilcoxon
        stat_d, p_d = wilcoxon(radial_deltas, anti_deltas, zero_method="wilcox", alternative="greater")
        stat_r, p_r = wilcoxon(radial_rr, anti_rr, zero_method="wilcox", alternative="greater")
        print(f"\nWilcoxon (radial > anti-radial, one-sided):")
        print(f"  Delta:        p={p_d:.6f} ({'SIGNIFICANT' if p_d < 0.05 else 'not sig'})")
        print(f"  Radial ratio: p={p_r:.6f} ({'SIGNIFICANT' if p_r < 0.05 else 'not sig'})")
    except ImportError:
        mean_diff_d = np.mean(radial_deltas - anti_deltas)
        mean_diff_r = np.mean(radial_rr - anti_rr)
        print(f"\nMean difference (radial - anti):")
        print(f"  Delta:        {mean_diff_d:+.4f}")
        print(f"  Radial ratio: {mean_diff_r:+.4f}")

    # Cohen's d
    d_delta = np.mean(radial_deltas - anti_deltas) / (np.std(radial_deltas - anti_deltas) + 1e-12)
    d_rr = np.mean(radial_rr - anti_rr) / (np.std(radial_rr - anti_rr) + 1e-12)
    print(f"\nCohen's d (radial - anti):")
    print(f"  Delta:        d={d_delta:.4f}")
    print(f"  Radial ratio: d={d_rr:.4f}")

    # Save
    output = {
        "config": {"num_cases": cases_done, "max_slices": MAX_SLICES},
        "radial_summary": {
            "mean_delta": float(np.mean(radial_deltas)),
            "mean_radial_ratio": float(np.mean(radial_rr)),
            "pos_pct": float(100.0 * np.mean(radial_deltas > 0)),
        },
        "anti_radial_summary": {
            "mean_delta": float(np.mean(anti_deltas)),
            "mean_radial_ratio": float(np.mean(anti_rr)),
            "pos_pct": float(100.0 * np.mean(anti_deltas > 0)),
        },
        "wilcoxon_delta_p": float(p_d) if "p_d" in dir() else None,
        "wilcoxon_radial_ratio_p": float(p_r) if "p_r" in dir() else None,
        "cohens_d_delta": float(d_delta),
        "cohens_d_radial_ratio": float(d_rr),
        "records": records,
    }
    out_path = os.path.join(OUTPUT_DIR, "antiradial_experiment.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
