#!/usr/bin/env python3
"""
Cross-victim validation: Test CODD C2 on MedSA (different victim architecture).
Minimal experiment: drift decomposition of SAM vs SAMed vs MedSA.
"""
import os, sys, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import repeat

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from theory import decompose_drift
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


def main():
    print(f"Device: {DEVICE}")
    print("Loading models...")
    t0 = time.time()
    source_sam = load_sam_encoder(VIT_NAME, SAM_CKPT, IMAGE_SIZE, NUM_CLASSES, DEVICE)
    samed_net = load_samed_model(VIT_NAME, SAM_CKPT, SAMED_CKPT, IMAGE_SIZE, NUM_CLASSES, device=DEVICE)
    medsa_net = load_medsa_model(VIT_NAME, SAM_CKPT, MEDSA_CKPT, IMAGE_SIZE, NUM_CLASSES, DEVICE)
    print(f"  Done in {time.time()-t0:.1f}s")

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
            img_t = torch.from_numpy(s).unsqueeze(0).float().to(DEVICE)
            img_t = repeat(img_t, 'b c h w -> b (repeat c) h w', repeat=3)

            with torch.no_grad():
                # SAM returns (features, layer_features)
                src_feat, src_layers = source_sam.image_encoder(img_t)
                samed_feat, samed_layers = samed_net.sam.image_encoder(img_t)
                # MedSA image_encoder may not return intermediate layers
                medsa_out = medsa_net.image_encoder(img_t)
                medsa_feat = medsa_out[0] if isinstance(medsa_out, tuple) else medsa_out
                # For MedSA layers, just use final feature or empty
                medsa_layers = medsa_out if isinstance(medsa_out, tuple) else [medsa_feat]

            # Model drift: SAMed vs source
            samed_drift = decompose_drift(src_feat, samed_feat)
            # Model drift: MedSA vs source
            medsa_drift = decompose_drift(src_feat, medsa_feat)
            # Cross-victim drift: SAMed vs MedSA
            cross_drift = decompose_drift(samed_feat, medsa_feat)

            records.append({
                "case": case_name, "slice": int(slice_idx),
                "samed_radial_ratio": samed_drift.radial_ratio,
                "medsa_radial_ratio": medsa_drift.radial_ratio,
                "cross_radial_ratio": cross_drift.radial_ratio,
            })
            print(f"  s{slice_idx}: SAMed_radial={samed_drift.radial_ratio:.4f}, "
                  f"MedSA_radial={medsa_drift.radial_ratio:.4f}, "
                  f"cross_radial={cross_drift.radial_ratio:.4f}")

        cases_done += 1

    # Summary
    samed_rr = np.array([r["samed_radial_ratio"] for r in records])
    medsa_rr = np.array([r["medsa_radial_ratio"] for r in records])
    cross_rr = np.array([r["cross_radial_ratio"] for r in records])

    print(f"\n{'='*60}")
    print(f"CROSS-VICTIM DRIFT DECOMPOSITION ({len(records)} slices, {cases_done} cases)")
    print(f"{'='*60}")
    print(f"\n  SAMed vs SAM:    radial_ratio = {np.mean(samed_rr):.4f} ± {np.std(samed_rr):.4f}")
    print(f"  MedSA vs SAM:    radial_ratio = {np.mean(medsa_rr):.4f} ± {np.std(medsa_rr):.4f}")
    print(f"  SAMed vs MedSA:  radial_ratio = {np.mean(cross_rr):.4f} ± {np.std(cross_rr):.4f}")

    # Null baseline
    null_rr = 0.5
    print(f"\n  Null (random carrier): 0.5000")
    print(f"  Cohen's d vs null:")
    print(f"    SAMed:  d={(np.mean(samed_rr)-null_rr)/np.std(samed_rr):.4f}")
    print(f"    MedSA:  d={(np.mean(medsa_rr)-null_rr)/np.std(medsa_rr):.4f}")
    print(f"    Cross:  d={(np.mean(cross_rr)-null_rr)/np.std(cross_rr):.4f}")

    output = {
        "config": {"num_cases": cases_done, "max_slices": MAX_SLICES},
        "summary": {
            "samed_mean_radial_ratio": float(np.mean(samed_rr)),
            "medsa_mean_radial_ratio": float(np.mean(medsa_rr)),
            "cross_mean_radial_ratio": float(np.mean(cross_rr)),
        },
        "records": records,
    }
    out_path = os.path.join(OUTPUT_DIR, "cross_victim_validation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
