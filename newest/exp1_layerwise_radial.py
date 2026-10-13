#!/usr/bin/env python3
"""
Experiment 1: Layer-wise Radial Dominance across all 6 vit_b targets.
Validates C1/C2: radial energy dominates tangential across layers, datasets, architectures.

MSA note: MSA image_encoder returns only final features (no intermediate layers).
For MSA we compute full-model drift only; for SAMed we get per-layer decomposition.
"""
import os, sys, json, gc
import numpy as np
import torch
from collections import defaultdict

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from theory import decompose_drift
from evaluate import load_sam_encoder, load_samed_model, load_medsa_model

DEVICE = torch.device("cuda:0")
IMAGE_SIZE = 512

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All 6 vit_b targets
TARGETS = [
    ("Synapse", "SAMed", f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", None, 8, 9),
    ("Synapse", "MSA", None, f"{ROOT}/output_runs_correct/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 8, 9),
    ("ISIC", "SAMed", f"{ROOT}/output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", None, 1, 1),
    ("ISIC", "MSA", None, f"{ROOT}/output_runs_correct/MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 1, 1),
    ("REFUGE", "SAMed", f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", None, 2, 2),
    ("REFUGE", "MSA", None, f"{ROOT}/output_runs_correct/MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 2, 2),
]

VIT_NAME = "vit_b"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
NUM_SAMPLES = 20


def get_synapse_samples():
    """Get organ-containing slices from Synapse test volumes, scanning ALL slices."""
    import h5py
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    samples = []
    for h5f in sorted(os.listdir(tdir)):
        if len(samples) >= NUM_SAMPLES:
            break
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        # Scan ALL slices for organ content
        valid_slices = sorted(
            [s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
            key=lambda s: (lbl[s] > 0).sum(), reverse=True,
        )
        for s in valid_slices:
            slc = img[s].astype(np.float32)
            slc_3ch = np.stack([slc]*3, axis=0)
            samples.append(torch.from_numpy(slc_3ch))
            if len(samples) >= NUM_SAMPLES:
                break
    return samples


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
        samples.append(transform(img))
        if len(samples) >= NUM_SAMPLES: break
    return samples


def get_refuge_samples():
    from PIL import Image
    from torchvision import transforms
    data_dir = f"{ROOT}/data/REFUGE-Multirater"
    test_dir = os.path.join(data_dir, "Test-400")
    folders = sorted([f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))])
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    samples = []
    for folder in folders:
        img_path = os.path.join(test_dir, folder, f"{folder}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            samples.append(transform(img))
        if len(samples) >= NUM_SAMPLES: break
    return samples


def main():
    print("Loading source SAM (vit_b)...")
    source_sam = load_sam_encoder(VIT_NAME, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    source_sam.eval()

    all_results = {}

    for dataset, model_type, samed_ckpt, msa_ckpt, num_cls, mmo in TARGETS:
        tag = f"{dataset}/{model_type}"
        ckpt_path = samed_ckpt or msa_ckpt
        if not os.path.exists(ckpt_path):
            print(f"\nSKIP {tag}: checkpoint not found")
            continue

        print(f"\n{'='*60}\n{tag}\n{'='*60}")

        # Load target model
        if model_type == "SAMed":
            target = load_samed_model(VIT_NAME, SAM_CKPT, samed_ckpt, IMAGE_SIZE, num_cls, device=DEVICE)
            has_layers = True
        else:
            target = load_medsa_model(VIT_NAME, SAM_CKPT, msa_ckpt, IMAGE_SIZE, mmo, DEVICE)
            has_layers = False  # MSA image_encoder returns only final features

        # Get samples
        if dataset == "Synapse":
            samples = get_synapse_samples()
        elif dataset == "ISIC":
            samples = get_isic_samples()
        else:
            samples = get_refuge_samples()

        print(f"  Processing {len(samples)} samples...")

        layer_radial = defaultdict(list)
        all_radial_ratios = []

        for i, sample in enumerate(samples):
            img_t = sample.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                # Source features (always has intermediate layers)
                src_feat, src_layers = source_sam.image_encoder(img_t)

                if model_type == "SAMed":
                    tgt_feat, tgt_layers = target.sam.image_encoder(img_t)
                else:
                    # MSA: preprocess first, then encode
                    tgt_feat = target.image_encoder(target.preprocess(img_t))
                    tgt_layers = None  # MSA doesn't expose intermediate features

            # Full model drift
            drift = decompose_drift(src_feat, tgt_feat)
            all_radial_ratios.append(drift.radial_ratio)

            # Per-layer drift (SAMed only)
            if has_layers and tgt_layers is not None:
                n_layers = min(len(src_layers), len(tgt_layers))
                for li in range(n_layers):
                    ld = decompose_drift(src_layers[li], tgt_layers[li])
                    layer_radial[li].append(ld.radial_ratio)

        # Summary
        layer_summary = {}
        for l in sorted(layer_radial.keys()):
            vals = layer_radial[l]
            layer_summary[str(l)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        mean_rr = float(np.mean(all_radial_ratios))
        std_rr = float(np.std(all_radial_ratios))
        layers_above_05 = sum(1 for l in layer_summary.values() if l['mean'] > 0.5)
        n_layers = len(layer_summary)

        print(f"  Mean radial ratio: {mean_rr:.4f} ± {std_rr:.4f}")
        if n_layers > 0:
            print(f"  Layers > 0.5: {layers_above_05}/{n_layers}")
            for l in sorted(layer_summary.keys(), key=int):
                print(f"    Layer {l}: R={layer_summary[l]['mean']:.4f}±{layer_summary[l]['std']:.4f}")
        else:
            print(f"  (MSA: no per-layer features available)")

        all_results[tag] = {
            "mean_radial_ratio": mean_rr,
            "std_radial_ratio": std_rr,
            "num_samples": len(samples),
            "has_per_layer": has_layers and n_layers > 0,
            "layers_above_05": layers_above_05,
            "total_layers": n_layers,
            "layer_details": layer_summary,
        }

        del target; gc.collect(); torch.cuda.empty_cache()

    # Save
    output = {"config": {"vit": VIT_NAME, "num_samples": NUM_SAMPLES}, "results": all_results}
    out_path = os.path.join(OUTPUT_DIR, "exp1_layerwise_radial.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
