#!/usr/bin/env python3
"""
Observational CODD validation: measure whether attack-induced feature drift
follows the same radial-dominant pattern as natural model drift.

Key tests:
  1. Attack drift decomposition: radial_ratio of Delta_F_attack
  2. Natural drift decomposition: radial_ratio of Delta_F_model (SAMed - SAM)
  3. Correlation: radial_ratio vs attack effectiveness (dice reduction)
  4. Cross-model consistency: SAMed vs MedSA drift patterns

Uses the original SE-CGR attack (with UAP warm-start) for strong attacks.
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
from attack_codd import AttackConfig, _get_edge_map, _return_eta, _return_mean_value
from evaluate import load_sam_encoder, load_samed_model, load_medsa_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIT_NAME = "vit_b"
IMAGE_SIZE = 512
NUM_CLASSES = 8
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
SAMED_CKPT = f"{ROOT}/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"
MEDSA_CKPT = f"{ROOT}/OUT_ROOT/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs2_lr0.0001_s24/Model/best_dice_checkpoint.pth"

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CASES = 8
MAX_SLICES = 6
ATTACK_STEPS = 10

criterion_l1 = torch.nn.L1Loss()


def evaluate_slice(net, image_np, label_np, case_name):
    """Evaluate SAMed on a single slice."""
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


def run_se_cgr_attack(image_encoder, input_image, uap=None):
    """Original SE-CGR attack with optional UAP warm-start."""
    max_val, min_val = input_image.max(), input_image.min()
    step_size = float((max_val - min_val) * 2.0 / 255.0)
    epsilon = float((max_val - min_val) * 10.0 / 255.0)

    with torch.no_grad():
        targets, targets_layer = image_encoder(input_image)

    edge_map = _get_edge_map(input_image).detach()
    perturb_img = torch.autograd.Variable(
        input_image.data + torch.randn_like(input_image) * 1e-5, requires_grad=True
    )

    # Initialize momentum with UAP warm-start
    momentum = torch.zeros_like(input_image).detach()
    if uap is not None:
        uap_norm = torch.norm(uap, p=1)
        if uap_norm > 1e-6:
            momentum = (uap / uap_norm).detach()

    for step in range(ATTACK_STEPS):
        image_encoder.zero_grad()
        perturb_input = perturb_img
        if step == 0 and uap is None:
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


def main():
    print(f"Device: {DEVICE}")
    print("Loading models...")
    t0 = time.time()
    source_sam = load_sam_encoder(VIT_NAME, SAM_CKPT, IMAGE_SIZE, NUM_CLASSES, DEVICE)
    samed_net = load_samed_model(VIT_NAME, SAM_CKPT, SAMED_CKPT, IMAGE_SIZE, NUM_CLASSES, device=DEVICE)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Load UAP
    uap_path = f"{ROOT}/common_perb/Synapse.pth"
    uap = torch.load(uap_path, map_location=DEVICE) if os.path.exists(uap_path) else None
    if uap is not None:
        print(f"  UAP loaded: shape={uap.shape}")

    # Load dataset
    from datasets.dataset_synapse import Synapse_dataset
    db = Synapse_dataset(
        base_dir=f"{ROOT}/data/Synapse_preprocess/test_vol_h5",
        list_dir=f"{ROOT}/data/Synapse_preprocess/lists",
        split="test_vol",
    )
    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=0)

    # Collect data
    records = []  # Each record: {slice_info, drift_stats, dice}

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

            # Clean dice
            clean_dice = evaluate_slice(samed_net, clean_gray, label, case_name)

            # Run attack with UAP warm-start
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            adv_np = run_se_cgr_attack(source_sam.image_encoder, img_t, uap=uap)
            adv_gray = adv_np[0].copy()
            adv_tensor = torch.from_numpy(adv_np).unsqueeze(0).float().to(DEVICE)

            # Attack dice
            adv_dice = evaluate_slice(samed_net, adv_gray, label, case_name)

            # --- Drift analysis ---
            with torch.no_grad():
                # Source features (clean)
                src_feat, src_layers = source_sam.image_encoder(img_t)
                # Source features (attacked)
                src_adv_feat, src_adv_layers = source_sam.image_encoder(adv_tensor)
                # Victim features (clean) — SAMed
                samed_feat, samed_layers = samed_net.sam.image_encoder(img_t)

            # Attack drift: F_source_adv - F_source_clean
            attack_drift = decompose_drift(src_feat, src_adv_feat)
            # Model drift: F_samed - F_source
            model_drift = decompose_drift(src_feat, samed_feat)

            # Per-layer attack drift
            per_layer = []
            for li in range(min(6, len(src_layers))):
                layer_idx = len(src_layers) - 6 + li
                if layer_idx < len(src_layers) and layer_idx < len(src_adv_layers):
                    s = decompose_drift(src_layers[layer_idx], src_adv_layers[layer_idx])
                    per_layer.append({
                        "layer": layer_idx,
                        "radial_ratio": s.radial_ratio,
                        "tangential_ratio": s.tangential_ratio,
                    })

            record = {
                "case": case_name,
                "slice": int(slice_idx),
                "clean_dice": clean_dice,
                "adv_dice": adv_dice,
                "delta": clean_dice - adv_dice,
                "attack_drift": {
                    "radial_ratio": attack_drift.radial_ratio,
                    "tangential_ratio": attack_drift.tangential_ratio,
                    "total_energy": attack_drift.total_energy,
                },
                "model_drift": {
                    "radial_ratio": model_drift.radial_ratio,
                    "tangential_ratio": model_drift.tangential_ratio,
                    "total_energy": model_drift.total_energy,
                },
                "per_layer_attack": per_layer,
            }
            records.append(record)

            print(f"  s{slice_idx}: clean={clean_dice:.4f}, adv={adv_dice:.4f}, "
                  f"delta={clean_dice-adv_dice:+.4f}, "
                  f"attack_radial={attack_drift.radial_ratio:.4f}, "
                  f"model_radial={model_drift.radial_ratio:.4f}")

        cases_done += 1

    # Summary analysis
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(records)} slices from {cases_done} cases")
    print(f"{'='*70}")

    deltas = np.array([r["delta"] for r in records])
    attack_radial = np.array([r["attack_drift"]["radial_ratio"] for r in records])
    model_radial = np.array([r["model_drift"]["radial_ratio"] for r in records])

    print(f"\nAttack drift radial_ratio: mean={np.mean(attack_radial):.4f}, "
          f"std={np.std(attack_radial):.4f}")
    print(f"Model drift radial_ratio:  mean={np.mean(model_radial):.4f}, "
          f"std={np.std(model_radial):.4f}")

    # Correlation: attack radial_ratio vs delta
    corr = np.corrcoef(attack_radial, deltas)[0, 1]
    print(f"\nCorrelation(attack_radial_ratio, delta): r={corr:.4f}")

    # Split by delta sign
    pos_mask = deltas > 0
    neg_mask = ~pos_mask
    if pos_mask.sum() > 0:
        print(f"Positive delta (attack works, n={pos_mask.sum()}): "
              f"attack_radial={np.mean(attack_radial[pos_mask]):.4f}")
    if neg_mask.sum() > 0:
        print(f"Negative delta (attack fails, n={neg_mask.sum()}): "
              f"attack_radial={np.mean(attack_radial[neg_mask]):.4f}")

    # Per-layer average
    print(f"\nPer-layer attack drift radial ratios:")
    layer_data = defaultdict(list)
    for r in records:
        for pl in r["per_layer_attack"]:
            layer_data[pl["layer"]].append(pl["radial_ratio"])
    for layer_idx in sorted(layer_data.keys()):
        vals = layer_data[layer_idx]
        print(f"  Layer {layer_idx}: radial_ratio={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save results
    output = {
        "config": {"num_cases": cases_done, "max_slices": MAX_SLICES, "attack_steps": ATTACK_STEPS},
        "summary": {
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "mean_attack_radial_ratio": float(np.mean(attack_radial)),
            "mean_model_radial_ratio": float(np.mean(model_radial)),
            "correlation_radial_vs_delta": float(corr),
        },
        "records": records,
    }
    out_path = os.path.join(OUTPUT_DIR, "observational_codd.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
