#!/usr/bin/env python3
"""
Experiment 10: Hyperparameter Ablation (epsilon / steps).
Tests attack effectiveness across different budgets on Synapse SAMed vit_b.

Epsilon: 4/255, 8/255, 16/255, 32/255 (in pixel space: ~4, 8, 16, 32 out of 255)
Steps: 5, 10, 20, 40

Compares SE-CGR (radial) vs no-probe baseline at each setting.
"""
import os, sys, json, gc, random
import numpy as np
import torch
import h5py

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import AttackConfig, run_codd_attack
from evaluate import load_sam_encoder, load_samed_model

DEVICE = torch.device("cuda:0")
IMAGE_SIZE = 512

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIT = "vit_b"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
CKPT = f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"

EPSILONS = [4, 8, 16, 32]   # pixel values out of 255
STEPS = [5, 10, 20, 40]
N_CASES = 6
NUM_SLICES = 4


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
        image=img, label=lbl, net=net, classes=8, multimask_output=True,
        patch_size=[IMAGE_SIZE, IMAGE_SIZE], test_save_path=None, case=case_name, z_spacing=1,
    )
    return float(np.mean(metric, axis=0)[0])


def main():
    print("Loading models...")
    source_sam = load_sam_encoder(VIT, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    target = load_samed_model(VIT, SAM_CKPT, CKPT, IMAGE_SIZE, 8, device=DEVICE)

    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:N_CASES]

    # Collect slices
    all_slices = []
    for h5f in h5s:
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
                      key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:NUM_SLICES]
        for s in valid:
            all_slices.append((img[s].astype(np.float32), lbl[s], h5f, s))

    print(f"Processing {len(all_slices)} slices...")

    results = {}

    # Epsilon sweep (fixed steps=20)
    for eps in EPSILONS:
        tag = f"eps={eps}_steps=20"
        print(f"\n  {tag}")
        se_cgr_deltas, no_probe_deltas = [], []

        for slc, lbl_slice, h5f, si in all_slices:
            clean_d = evaluate_slice(target, slc, lbl_slice, f"{h5f}_s{si}")

            slc_3ch = np.stack([slc]*3, axis=0)
            img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)

            # SE-CGR
            config = AttackConfig(
                att_step_size=2.0, att_bound=float(eps), att_numstep=20,
                direction_mode="radial", probe_amplification=2.0,
                edge_guidance=True, seed=42,
            )
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            adv_np, _ = run_codd_attack(source_sam.image_encoder, img_t, config, return_trace=True)
            adv_d = evaluate_slice(target, adv_np[0].copy(), lbl_slice, f"{h5f}_s{si}")
            se_cgr_deltas.append(clean_d - adv_d)

            # no-probe = clean
            no_probe_deltas.append(0.0)  # clean, no attack

        results[tag] = {
            "se_cgr_mean_delta": float(np.mean(se_cgr_deltas)),
            "se_cgr_std_delta": float(np.std(se_cgr_deltas)),
            "no_probe_mean_delta": 0.0,
            "ratio": float(np.mean(se_cgr_deltas) / (1e-8 + np.mean(no_probe_deltas))) if np.mean(no_probe_deltas) > 1e-8 else float("inf"),
            "n": len(se_cgr_deltas),
        }
        print(f"    SE-CGR: {results[tag]['se_cgr_mean_delta']:.4f} ± {results[tag]['se_cgr_std_delta']:.4f}")

    # Steps sweep (fixed epsilon=16)
    for nstep in STEPS:
        tag = f"eps=16_steps={nstep}"
        print(f"\n  {tag}")
        deltas = []

        for slc, lbl_slice, h5f, si in all_slices:
            clean_d = evaluate_slice(target, slc, lbl_slice, f"{h5f}_s{si}")
            slc_3ch = np.stack([slc]*3, axis=0)
            img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)

            config = AttackConfig(
                att_step_size=2.0, att_bound=16.0, att_numstep=nstep,
                direction_mode="radial", probe_amplification=2.0,
                edge_guidance=True, seed=42,
            )
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            adv_np, _ = run_codd_attack(source_sam.image_encoder, img_t, config, return_trace=True)
            adv_d = evaluate_slice(target, adv_np[0].copy(), lbl_slice, f"{h5f}_s{si}")
            deltas.append(clean_d - adv_d)

        results[tag] = {
            "se_cgr_mean_delta": float(np.mean(deltas)),
            "se_cgr_std_delta": float(np.std(deltas)),
            "n": len(deltas),
        }
        print(f"    SE-CGR: {results[tag]['se_cgr_mean_delta']:.4f} ± {results[tag]['se_cgr_std_delta']:.4f}")

    output = {"config": {"vit": VIT, "epsilons": EPSILONS, "steps": STEPS}, "results": results}
    out_path = os.path.join(OUTPUT_DIR, "exp10_hyperparams.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
