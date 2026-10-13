#!/usr/bin/env python3
"""
Directional ablation: validate C3 claim that radial attacks degrade victim
segmentation more than tangential/isotropic/misaligned.

Tests CODD deterministic probes (use_random_noise=False) for clean directional
comparison, plus an original SE-CGR random-noise baseline.

Multi-case, multi-slice evaluation with statistical testing.
"""
import os, sys, json, time, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import repeat

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from theory import synthesize_codd_probe, decompose_drift
from attack_codd import AttackConfig, run_codd_attack, _get_edge_map, _return_eta, _return_mean_value
from evaluate import load_sam_encoder, load_samed_model, load_medsa_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
VIT_NAME = "vit_b"
DATASET = "Synapse"
IMAGE_SIZE = 512
NUM_CLASSES = 8
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
SAMED_CKPT = f"{ROOT}/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"
MEDSA_CKPT = f"{ROOT}/OUT_ROOT/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs2_lr0.0001_s24/Model/best_dice_checkpoint.pth"

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CASES = 4
MAX_SLICES_PER_CASE = 6
ATTACK_STEPS = 10

DIRECTIONS = ["radial", "tangential", "isotropic", "misaligned"]
SEED = 42

criterion_l1 = torch.nn.L1Loss()


def se_cgr_attack_slice(image_encoder, input_image, direction, use_random_noise, config):
    """Run original SE-CGR style attack on a single slice."""
    max_val, min_val = input_image.max(), input_image.min()
    step_size = float((max_val - min_val) * config.att_step_size / 255.0)
    epsilon = float((max_val - min_val) * config.att_bound / 255.0)

    with torch.no_grad():
        targets, targets_layer = image_encoder(input_image)

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

        # Original SE-CGR loss structure
        loss_org = criterion_l1(features * 0.9999, targets)

        pre_layer = int(len(features_layer) * 0.9)
        num_layers = min(6, pre_layer)
        indices = random.sample(range(num_layers), max(1, num_layers - 2))
        selected_f = torch.stack([features_layer[i + pre_layer - num_layers] for i in indices])
        selected_t = torch.stack([targets_layer[i + pre_layer - num_layers] for i in indices])

        feature_difference_list = []
        for i in range(selected_f.shape[0]):
            F_clean = selected_t[i].unsqueeze(0)
            F_adv = selected_f[i].unsqueeze(0)

            probe = synthesize_codd_probe(
                F_clean=F_clean, F_adv=F_adv,
                direction=direction,
                use_random_noise=use_random_noise,
                detach_adv=True,
            )
            # Apply probe amplification to balance gradient magnitudes across directions
            feature_difference = (F_adv * 0.999 - F_clean) - config.probe_amplification * probe
            feature_difference_list.append(feature_difference.squeeze(0))

        feature_difference = torch.stack(feature_difference_list)
        input_mean = _return_mean_value(selected_t)
        if len(input_mean.shape) > len(feature_difference.shape):
            input_mean = input_mean.view(feature_difference.shape[0], 1, 1, 1)
        feature_difference = feature_difference / (input_mean + 1e-6)
        l1_losses = (torch.abs(torch.mean(feature_difference, dim=0))).mean()

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


def evaluate_slice_samed(net, image_np, label_np, case_name, image_size=512, num_classes=8):
    """Evaluate SAMed on a single slice.

    Args:
        image_np: [H, W] numpy array (grayscale CT image, NOT label)
        label_np: [H, W] numpy array (integer label mask)
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("root_utils", f"{ROOT}/utils.py")
    root_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root_utils)
    test_single_volume = root_utils.test_single_volume

    # Ensure correct shape for image and label
    if image_np.ndim == 2:
        image_np = image_np[np.newaxis, :, :]  # [1, H, W]
    if label_np.ndim == 2:
        label_np = label_np[np.newaxis, :, :]  # [1, H, W]

    img = torch.from_numpy(image_np).unsqueeze(0).float()  # [1, 1, H, W]
    lbl = torch.from_numpy(label_np).unsqueeze(0).float()  # [1, 1, H, W]

    metric = test_single_volume(
        image=img, label=lbl, net=net, classes=num_classes,
        multimask_output=True, patch_size=[image_size, image_size],
        test_save_path=None, case=case_name, z_spacing=1,
    )
    return float(np.mean(metric, axis=0)[0])


def main():
    print(f"Device: {DEVICE}")
    print(f"Cases: {NUM_CASES}, Slices per case: {MAX_SLICES_PER_CASE}")
    print(f"Attack steps: {ATTACK_STEPS}")

    # Load models
    print("\nLoading models...")
    t0 = time.time()
    source_sam = load_sam_encoder(VIT_NAME, SAM_CKPT, IMAGE_SIZE, NUM_CLASSES, DEVICE)
    samed_net = load_samed_model(VIT_NAME, SAM_CKPT, SAMED_CKPT, IMAGE_SIZE, NUM_CLASSES, device=DEVICE)
    print(f"  Models loaded in {time.time()-t0:.1f}s")

    # Load dataset
    from datasets.dataset_synapse import Synapse_dataset
    db = Synapse_dataset(
        base_dir=f"{ROOT}/data/Synapse_preprocess/test_vol_h5",
        list_dir=f"{ROOT}/data/Synapse_preprocess/lists",
        split="test_vol",
    )
    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=0)

    # Attack configs
    config_deterministic = AttackConfig(
        att_step_size=2.0, att_bound=10.0, att_numstep=ATTACK_STEPS,
        direction_mode="radial", seed=SEED,
        edge_guidance=True, edge_gamma=3.0,
        directional_weight=1.0,
        probe_amplification=2.0,  # Balance gradients: amp>1 ensures |drift-amp*probe| comparable
    )
    config_random = AttackConfig(
        att_step_size=2.0, att_bound=10.0, att_numstep=ATTACK_STEPS,
        direction_mode="radial", seed=SEED,
        edge_guidance=True, edge_gamma=3.0,
        directional_weight=1.0,
        probe_amplification=1.0,  # No amplification for random noise (already balanced)
    )

    # Results storage
    all_results = {
        "config": {
            "num_cases": NUM_CASES, "max_slices_per_case": MAX_SLICES_PER_CASE,
            "attack_steps": ATTACK_STEPS, "vit_name": VIT_NAME, "dataset": DATASET,
        },
        "deterministic": {d: {} for d in DIRECTIONS},
        "random_noise": {d: {} for d in DIRECTIONS},
        "clean": {},
    }

    cases_processed = 0
    for batch_idx, batch in enumerate(loader):
        if cases_processed >= NUM_CASES:
            break

        case_name = batch["case_name"][0] if "case_name" in batch else f"case_{batch_idx}"
        print(f"\n{'='*60}")
        print(f"Case {cases_processed+1}/{NUM_CASES}: {case_name}")

        # Extract volume
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

        # Find organ-containing slices
        organ_slices = sorted(
            [s for s in range(depth) if (label_vol[s] > 0).sum() > 100],
            key=lambda s: (label_vol[s] > 0).sum(),
            reverse=True,
        )
        # Select evenly from organ slices
        if len(organ_slices) > MAX_SLICES_PER_CASE:
            step = len(organ_slices) / MAX_SLICES_PER_CASE
            selected_slices = [organ_slices[int(i * step)] for i in range(MAX_SLICES_PER_CASE)]
        else:
            selected_slices = organ_slices[:MAX_SLICES_PER_CASE]

        print(f"  Depth: {depth}, Organ slices: {len(organ_slices)}, Selected: {len(selected_slices)}")
        print(f"  Selected slice indices: {selected_slices}")

        # Prepare slices
        slices_data = []
        for slice_idx in selected_slices:
            s = image_vol[slice_idx, :, :, :]  # [1, H, W] or [H, W]
            if s.ndim == 2:
                s = s[np.newaxis, :, :]
            # Store clean grayscale image as numpy for evaluation
            clean_gray = s[0].copy()  # [H, W] clean CT slice (single channel)
            img_t = torch.from_numpy(s).unsqueeze(0).float().to(DEVICE)
            img_t = repeat(img_t, 'b c h w -> b (repeat c) h w', repeat=3)
            slices_data.append({
                "idx": slice_idx,
                "image": img_t,
                "clean_gray": clean_gray,  # [H, W] numpy for clean eval
                "label": label_vol[slice_idx].copy(),  # [H, W] label mask
            })

        # Evaluate CLEAN
        clean_dices = []
        for sd in slices_data:
            d = evaluate_slice_samed(samed_net, sd["clean_gray"], sd["label"], case_name)
            clean_dices.append(d)
        all_results["clean"][case_name] = {
            "slices": selected_slices,
            "dices": clean_dices,
            "mean_dice": float(np.mean(clean_dices)),
        }
        print(f"  Clean mean dice: {np.mean(clean_dices):.4f}")

        # Attack: deterministic mode
        for direction in DIRECTIONS:
            config_deterministic.direction_mode = direction
            config_deterministic.seed = SEED  # reset seed per direction for consistency
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            adv_dices = []
            t_start = time.time()
            for sd in slices_data:
                adv_np = se_cgr_attack_slice(
                    source_sam.image_encoder, sd["image"],
                    direction=direction, use_random_noise=False,
                    config=config_deterministic,
                )
                # adv_np: [3, H, W] -> take first channel as grayscale
                adv_gray = adv_np[0].copy()
                d = evaluate_slice_samed(samed_net, adv_gray, sd["label"], case_name)
                adv_dices.append(d)
            elapsed = time.time() - t_start

            mean_d = float(np.mean(adv_dices))
            delta = float(np.mean(clean_dices) - mean_d)
            all_results["deterministic"][direction][case_name] = {
                "slices": selected_slices,
                "dices": adv_dices,
                "mean_dice": mean_d,
                "delta": delta,
            }
            print(f"  CODD {direction:<12}: dice={mean_d:.4f}, delta={delta:+.4f}, time={elapsed:.1f}s")

        # Attack: random noise mode (SE-CGR baseline)
        for direction in DIRECTIONS:
            config_random.direction_mode = direction
            config_random.seed = SEED
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            adv_dices = []
            t_start = time.time()
            for sd in slices_data:
                adv_np = se_cgr_attack_slice(
                    source_sam.image_encoder, sd["image"],
                    direction=direction, use_random_noise=True,
                    config=config_random,
                )
                adv_gray = adv_np[0].copy()
                d = evaluate_slice_samed(samed_net, adv_gray, sd["label"], case_name)
                adv_dices.append(d)
            elapsed = time.time() - t_start

            mean_d = float(np.mean(adv_dices))
            delta = float(np.mean(clean_dices) - mean_d)
            all_results["random_noise"][direction][case_name] = {
                "slices": selected_slices,
                "dices": adv_dices,
                "mean_dice": mean_d,
                "delta": delta,
            }
            print(f"  RAND {direction:<12}: dice={mean_d:.4f}, delta={delta:+.4f}, time={elapsed:.1f}s")

        cases_processed += 1

        # Save intermediate results
        with open(os.path.join(OUTPUT_DIR, "directional_ablation.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for mode_name, mode_results in [("deterministic", "CODD Deterministic"), ("random_noise", "SE-CGR Random Noise")]:
        print(f"\n--- {mode_name} ---")
        print(f"{'Direction':<14} {'Mean Dice':<12} {'Mean Delta':<12} {'Cases':<8}")
        print("-" * 46)
        for direction in DIRECTIONS:
            dir_data = all_results[mode_name][direction]
            case_deltas = [v["delta"] for v in dir_data.values() if "delta" in v]
            case_dices = [v["mean_dice"] for v in dir_data.values() if "mean_dice" in v]
            if case_deltas:
                print(f"{direction:<14} {np.mean(case_dices):<12.4f} {np.mean(case_deltas):<+12.4f} {len(case_deltas):<8}")

    print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'directional_ablation.json')}")
    print("Done!")


if __name__ == "__main__":
    main()
