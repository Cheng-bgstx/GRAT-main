#!/usr/bin/env python3
"""
Cross-family C3 validation: SAMed on ISIC (skin lesions) as victim.

ISIC is a GENUINELY different medical domain from Synapse (abdominal CT).
This addresses the external validity concern raised by the reviewer.

Uses 2D evaluation (one Dice per image). 20 test images × 5 conditions.
"""
import os, sys, json, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
NUM_CLASSES = 1  # ISIC is binary segmentation (lesion vs background)

# ISIC SAMed checkpoint
ISIC_CKPT = f"{ROOT}/output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
ISIC_DATA = f"{ROOT}/data/ISIC"
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_IMAGES = 20
ATTACK_STEPS = 20
SEED_BASE = 8899
EPSILON_SCALE = 16.0
STEP_SCALE = 2.0
FIXED_PROBE_AMPLITUDE = 2.0
PROBE_WEIGHT = 2.0

criterion_l1 = torch.nn.L1Loss()
CONDITIONS = ["SE-CGR", "phase-shuffled", "random-phase", "tangential", "no-probe"]


def synthesize_fixed_probe(F_clean, mode):
    G_clean = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
    A = torch.abs(G_clean)
    Phase_s = torch.angle(G_clean)
    Sigma_freq = FIXED_PROBE_AMPLITUDE * (A + 1e-12)
    if mode == "SE-CGR":
        fft_c = Sigma_freq * torch.exp(1j * Phase_s)
    elif mode == "phase-shuffled":
        phase_flat = Phase_s.flatten()
        perm = torch.randperm(phase_flat.numel(), device=phase_flat.device)
        phase_shuf = phase_flat[perm].reshape(Phase_s.shape)
        fft_c = Sigma_freq * torch.exp(1j * phase_shuf)
    elif mode == "random-phase":
        rand_phase = torch.rand_like(Phase_s) * 2.0 * np.pi
        fft_c = Sigma_freq * torch.exp(1j * rand_phase)
    elif mode == "tangential":
        fft_c = Sigma_freq * (1j * torch.exp(1j * Phase_s))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return torch.fft.ifftn(fft_c, dim=(-2, -1), norm="ortho").real


def run_matched_attack(image_encoder, input_image, probe_mode):
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
        loss_org = -criterion_l1(features, targets)

        pre_layer = int(len(features_layer) * 0.9)
        num_layers = min(6, pre_layer)
        indices = random.sample(range(num_layers), max(1, num_layers - 2))

        fd_list = []
        for idx in indices:
            F_clean = targets_layer[idx + pre_layer - num_layers].unsqueeze(0)
            F_adv = features_layer[idx + pre_layer - num_layers].unsqueeze(0)
            if probe_mode == "no-probe":
                fd = (F_adv - F_clean)
            else:
                probe = synthesize_fixed_probe(F_clean, probe_mode)
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
            probe_loss = -torch.abs(torch.mean(fd, dim=0)).mean()
        else:
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


def load_isic_test_images(n):
    """Load ISIC test images and masks."""
    test_dir = os.path.join(ISIC_DATA, "ISBI2016_ISIC_Part1_Test_Data")
    mask_dir = os.path.join(ISIC_DATA, "ISBI2016_ISIC_Part1_Test_GroundTruth")
    jpgs = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
    jpgs = jpgs[:n]
    images, masks, names = [], [], []
    for jpg in jpgs:
        img = Image.open(os.path.join(test_dir, jpg)).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)  # [H, W, 3], 0-255 range

        base = jpg.replace('.jpg', '')
        mask_file = base + '_Segmentation.png'
        mask = Image.open(os.path.join(mask_dir, mask_file)).convert('L')
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        mask_np = (np.array(mask) > 128).astype(np.float32)  # [H, W]

        if mask_np.sum() > 100:  # only use images with lesions
            images.append(img_np)
            masks.append(mask_np)
            names.append(base)
    return images, masks, names


def evaluate_isic_slice(net, image_np, label_np):
    """Evaluate Dice for ISIC 2D image using SAMed forward (like test_single_volume)."""
    img = torch.from_numpy(image_np.copy()).permute(2, 0, 1).float().cuda()
    x, y = img.shape[1], img.shape[2]
    inputs = img.unsqueeze(0)  # [1, 3, H, W]
    net.eval()
    with torch.no_grad():
        outputs = net(inputs, multimask_output=False, image_size=IMAGE_SIZE)
        masks = outputs['masks']  # shape varies
        if masks.shape[1] > 1:
            pred = torch.argmax(torch.softmax(masks.float(), dim=1), dim=1).squeeze(0)
        elif masks.shape[1] == 1:
            pred = (masks.squeeze(0).squeeze(0) > 0)
        else:
            pred = masks.squeeze(0) > 0
        pred_np = pred.cpu().numpy().astype(np.float32)
    # Match size to ground truth
    if pred_np.shape != label_np.shape:
        import scipy.ndimage as ndi
        pred_np = ndi.zoom(pred_np, (label_np.shape[0]/pred_np.shape[0],
                                     label_np.shape[1]/pred_np.shape[1]), order=0)
    intersection = (pred_np * label_np).sum()
    dice = 2.0 * intersection / (pred_np.sum() + label_np.sum() + 1e-8)
    return float(dice)


def main():
    print(f"Device: {DEVICE}")
    print(f"CROSS-FAMILY C3: SAMed on ISIC (skin lesions)")
    print(f"Conditions: {CONDITIONS}")
    print(f"Images: {NUM_IMAGES}, Steps: {ATTACK_STEPS}, Eps: {EPSILON_SCALE}/255")

    print("\nLoading models...")
    source_sam = load_sam_encoder(VIT_NAME, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    # ISIC uses binary segmentation (1 class)
    isic_net = load_samed_model(VIT_NAME, SAM_CKPT, ISIC_CKPT, IMAGE_SIZE, 1, device=DEVICE)
    print("Done.")

    print("\nLoading ISIC test images...")
    images, masks, names = load_isic_test_images(NUM_IMAGES)
    n_actual = len(images)
    print(f"Loaded {n_actual} test images with lesions.")

    results = defaultdict(list)

    for idx in range(n_actual):
        img_np = images[idx]  # [H, W, 3]
        mask_np = masks[idx]  # [H, W]
        name = names[idx]

        # Convert to tensor for attack: [C, H, W]
        img_t = torch.from_numpy(img_np.copy()).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE).contiguous()

        # Clean evaluation
        clean_dice = evaluate_isic_slice(isic_net, img_np, mask_np)
        if clean_dice < 0.01:
            continue  # skip very poor clean performance

        print(f"  Image {idx+1}/{n_actual}: {name} clean={clean_dice:.4f}")

        for probe_mode in CONDITIONS:
            random.seed(SEED_BASE)
            np.random.seed(SEED_BASE)
            torch.manual_seed(SEED_BASE)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED_BASE)

            adv_np = run_matched_attack(source_sam.image_encoder, img_t, probe_mode)
            adv_img = adv_np[0].copy()  # [H, W] grayscale
            if adv_img.ndim == 2:
                adv_img_rgb = np.stack([adv_img, adv_img, adv_img], axis=-1)
            else:
                adv_img_rgb = adv_img

            adv_dice = evaluate_isic_slice(isic_net, adv_img_rgb, mask_np)
            delta = clean_dice - adv_dice
            results[probe_mode].append({
                "name": name, "clean_dice": clean_dice,
                "adv_dice": adv_dice, "delta": delta,
            })

        parts = []
        for pm in CONDITIONS:
            parts.append(f"{pm}={results[pm][-1]['delta']:+.5f}")
        print(f"    Δ: {' | '.join(parts)}")

    print(f"\n{'='*70}")
    print(f"CROSS-FAMILY C3: ISIC (skin lesions, n={len(results['SE-CGR'])})")
    print(f"{'='*70}")

    print(f"\n{'Condition':<18} {'Mean Δ':<14} {'Std':<10} {'Pos%'}")
    print("-" * 46)
    deltas_dict = {}
    for pm in CONDITIONS:
        deltas = np.array([r["delta"] for r in results[pm]])
        deltas_dict[pm] = deltas
        print(f"{pm:<18} {np.mean(deltas):<+14.6f} {np.std(deltas):<10.6f} "
              f"{100.0*np.mean(deltas>0):.0f}%")

    print(f"\n--- Paired tests vs SE-CGR ---")
    try:
        from scipy.stats import wilcoxon
        se = deltas_dict["SE-CGR"]
        for pm in CONDITIONS:
            if pm == "SE-CGR":
                continue
            other = deltas_dict[pm]
            diff = se - other
            stat, p1 = wilcoxon(se, other, alternative="greater")
            stat, p2 = wilcoxon(se, other, alternative="two-sided")
            d_val = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
            s1, s2 = "SIG" if p1 < 0.05 else "ns", "SIG" if p2 < 0.05 else "ns"
            print(f"  vs {pm:<18}: Δ={np.mean(diff):+.6f}, d={d_val:+.4f}, "
                  f"p_one={p1:.4f}({s1}), p_two={p2:.4f}({s2})")
    except ImportError:
        pass

    output = {
        "config": {"victim": "SAMed_ISIC", "n_images": len(results["SE-CGR"]),
                    "attack_steps": ATTACK_STEPS, "epsilon": EPSILON_SCALE,
                    "probe_amplitude": FIXED_PROBE_AMPLITUDE},
        "per_condition": {pm: {"mean": float(np.mean(deltas_dict[pm])),
                                "std": float(np.std(deltas_dict[pm]))}
                          for pm in CONDITIONS},
        "records": {pm: results[pm] for pm in CONDITIONS},
    }
    out_path = os.path.join(OUTPUT_DIR, "cross_family_isic.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
