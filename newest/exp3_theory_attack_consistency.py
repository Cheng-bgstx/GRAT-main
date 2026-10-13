#!/usr/bin/env python3
"""
Experiment 3: Theory-to-Attack Consistency.
Correlates per-image radial ratio (R) with attack effect (ΔDice) across 5 conditions:
  se_cgr (radial), phase_shuffled, tangential, random_phase, no_probe.

Key predictions:
  - se_cgr:     positive correlation with R (higher R → bigger attack)
  - tangential: negative correlation with R (misaligned with drift)
  - random/phase_shuffled: no correlation with R (direction-agnostic)
  - no_probe:   baseline (should be near zero)

Covers: ISIC SAMed vit_b + REFUGE SAMed vit_b (Synapse already done in prior experiments)
"""
import os, sys, json, gc, random
import numpy as np
import torch

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from theory import decompose_drift
from attack_codd import AttackConfig, run_codd_attack
from evaluate import load_sam_encoder, load_samed_model

DEVICE = torch.device("cuda:0")
IMAGE_SIZE = 512

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIT_NAME = "vit_b"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"

CONDITIONS = ["se_cgr", "phase_shuffled", "tangential", "random_phase", "no_probe"]
DIRECTION_MAP = {
    "se_cgr": "radial",
    "phase_shuffled": "radial",
    "tangential": "tangential",
    "random_phase": "isotropic",
    "no_probe": "radial",
}

TARGETS = [
    ("ISIC", "SAMed", f"{ROOT}/output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 1),
    ("REFUGE", "SAMed", f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 2),
]

NUM_CASES = 12


def evaluate_isic(net, image, label, device=DEVICE):
    """Binary Dice for ISIC with optimal threshold."""
    with torch.no_grad():
        outputs = net(image.unsqueeze(0).to(device), True, IMAGE_SIZE)
        masks = outputs['masks']
    probs = torch.sigmoid(masks).squeeze().cpu().numpy()
    label_np = label.cpu().numpy() if torch.is_tensor(label) else label
    gt = (label_np > 0.5).astype(np.float32)
    best_d = 0.0
    for th in np.arange(0.1, 1.0, 0.1):
        pb = (probs > th).astype(np.float32)
        intersection = (pb * gt).sum()
        denom = pb.sum() + gt.sum()
        d = 2.0 * intersection / (denom + 1e-8)
        if d > best_d: best_d = d
    return float(best_d)


def evaluate_refuge(net, image, label, device=DEVICE):
    """Mean Dice for REFUGE (disc + cup)."""
    with torch.no_grad():
        outputs = net(image.unsqueeze(0).to(device), True, IMAGE_SIZE)
        masks = outputs['masks']
    masks = torch.softmax(masks, dim=1).squeeze(0).cpu().numpy()
    pred = np.argmax(masks, axis=0)
    label_np = label.cpu().numpy() if torch.is_tensor(label) else label
    # label: [3, H, W] — [cup, disc, bg]
    disc_gt = (label_np[1] > 0.5).astype(np.float32)
    cup_gt = (label_np[0] > 0.5).astype(np.float32)
    disc_p = (pred == 1).astype(np.float32)
    cup_p = (pred == 2).astype(np.float32)
    d_disc = 2.0 * (disc_p * disc_gt).sum() / (disc_p.sum() + disc_gt.sum() + 1e-8)
    d_cup = 2.0 * (cup_p * cup_gt).sum() / (cup_p.sum() + cup_gt.sum() + 1e-8)
    return float((d_disc + d_cup) / 2.0)


def get_isic_samples():
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    df = pd.read_csv(os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv"))
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    samples = []
    for _, row in df.iterrows():
        img = Image.open(os.path.join(data_dir, row['img'])).convert('RGB')
        gt = np.array(Image.open(os.path.join(data_dir, row['seg'])).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32) / 255.0
        samples.append((transform(img), torch.from_numpy((gt > 0.5).astype(np.float32))))
        if len(samples) >= NUM_CASES: break
    return samples


def get_refuge_samples():
    from PIL import Image
    from torchvision import transforms
    data_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    samples = []
    for folder in folders:
        img_path = os.path.join(data_dir, folder, f"{folder}.jpg")
        disc_path = os.path.join(data_dir, folder, f"{folder}_disc.bmp")
        cup_path = os.path.join(data_dir, folder, f"{folder}_cup.bmp")
        if not os.path.exists(img_path): continue
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img)
        disc_raw = np.array(Image.open(disc_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        cup_raw = np.array(Image.open(cup_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        disc_gt = (disc_raw <= 128).astype(np.float32)
        cup_gt = (cup_raw < 128).astype(np.float32)
        # Stack as [cup, disc, bg]
        label = np.stack([cup_gt, disc_gt, 1.0 - cup_gt - disc_gt], axis=0)
        samples.append((img_t, torch.from_numpy(label)))
        if len(samples) >= NUM_CASES: break
    return samples


def compute_radial_ratio(source_encoder, target_sam, img_t, model_type="SAMed"):
    """Compute radial ratio for one image."""
    with torch.no_grad():
        src_feat, _ = source_encoder.image_encoder(img_t.unsqueeze(0).to(DEVICE))
        if model_type == "SAMed":
            tgt_feat, _ = target_sam.sam.image_encoder(img_t.unsqueeze(0).to(DEVICE))
        else:
            tgt_feat = target_sam.image_encoder(target_sam.preprocess(img_t.unsqueeze(0).to(DEVICE)))
    drift = decompose_drift(src_feat, tgt_feat)
    return drift.radial_ratio


def main():
    print("Loading source SAM (vit_b)...")
    source_sam = load_sam_encoder(VIT_NAME, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    source_sam.eval()

    all_results = {}

    for dataset, model_type, ckpt, num_cls in TARGETS:
        tag = f"{dataset}/{model_type}"
        if not os.path.exists(ckpt):
            print(f"SKIP {tag}: checkpoint not found")
            continue

        print(f"\n{'='*60}\n{tag}\n{'='*60}")

        target = load_samed_model(VIT_NAME, SAM_CKPT, ckpt, IMAGE_SIZE, num_cls, device=DEVICE)
        eval_fn = evaluate_isic if dataset == "ISIC" else evaluate_refuge
        samples = get_isic_samples() if dataset == "ISIC" else get_refuge_samples()

        print(f"  Processing {len(samples)} cases × {len(CONDITIONS)} conditions...")

        per_case_results = []

        for case_idx, (img_t, label) in enumerate(samples):
            # Compute radial ratio for this case
            R = compute_radial_ratio(source_sam, target, img_t, model_type)

            # Clean dice
            clean_d = eval_fn(target, img_t, label)

            case_data = {"case": case_idx, "radial_ratio": float(R), "clean_dice": clean_d, "conditions": {}}

            # Gray image version for attacks
            img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1)
            img_tensor = img_gray.unsqueeze(0).float().to(DEVICE)

            for cond in CONDITIONS:
                if cond == "no_probe":
                    d = clean_d
                    case_data["conditions"][cond] = {"dice": d, "delta": 0.0}
                    continue

                # Prepare attack image
                if cond == "phase_shuffled":
                    G = torch.fft.fft2(img_tensor, norm="ortho")
                    mag = torch.abs(G)
                    phase = torch.angle(G)
                    B, C, H, W = phase.shape
                    phase_shuffled = phase.view(B, C, -1)[:, :, torch.randperm(H*W)].view_as(phase)
                    G_shuffled = mag * torch.exp(1j * phase_shuffled)
                    img_for_attack = torch.fft.ifft2(G_shuffled, norm="ortho").real
                elif cond == "random_phase":
                    G = torch.fft.fft2(img_tensor, norm="ortho")
                    mag = torch.abs(G)
                    rand_phase = torch.rand_like(G) * 2 * np.pi
                    G_rand = mag * torch.exp(1j * rand_phase)
                    img_for_attack = torch.fft.ifft2(G_rand, norm="ortho").real
                else:
                    img_for_attack = img_tensor

                config = AttackConfig(
                    att_step_size=2.0, att_bound=10.0, att_numstep=10,
                    direction_mode=DIRECTION_MAP[cond],
                    probe_amplification=2.0,
                    edge_guidance=True, seed=42,
                )

                random.seed(42); np.random.seed(42); torch.manual_seed(42)
                adv_np, _ = run_codd_attack(
                    source_sam.image_encoder, img_for_attack, config, return_trace=True,
                )

                # Evaluate adversarial image
                adv_gray = torch.from_numpy(adv_np.copy()).float()  # [3, H, W]
                adv_d = eval_fn(target, adv_gray, label)
                delta = clean_d - adv_d
                case_data["conditions"][cond] = {"dice": float(adv_d), "delta": float(delta)}

            per_case_results.append(case_data)
            print(f"  [{case_idx+1}/{len(samples)}] R={R:.4f} clean={clean_d:.4f} "
                  f"se_cgr_delta={case_data['conditions']['se_cgr']['delta']:.4f} "
                  f"tang_delta={case_data['conditions']['tangential']['delta']:.4f}")

        # Compute correlations
        Rs = np.array([c["radial_ratio"] for c in per_case_results])
        correlations = {}
        for cond in CONDITIONS:
            deltas = np.array([c["conditions"][cond]["delta"] for c in per_case_results])
            if len(deltas) > 2:
                corr = float(np.corrcoef(Rs, deltas)[0, 1])
            else:
                corr = float("nan")
            correlations[cond] = {"pearson_r": corr, "mean_delta": float(np.mean(deltas)), "std_delta": float(np.std(deltas))}

        all_results[tag] = {
            "per_case": per_case_results,
            "correlations": correlations,
        }

        print(f"\n  Correlations (R vs ΔDice):")
        for cond in CONDITIONS:
            print(f"    {cond}: r={correlations[cond]['pearson_r']:+.4f} mean_Δ={correlations[cond]['mean_delta']:.4f}")

        del target; gc.collect(); torch.cuda.empty_cache()

    output = {"config": {"conditions": CONDITIONS, "num_cases": NUM_CASES}, "results": all_results}
    out_path = os.path.join(OUTPUT_DIR, "exp3_theory_attack_consistency.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
