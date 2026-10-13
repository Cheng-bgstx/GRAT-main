#!/usr/bin/env python3
"""
Master script to run the complete CODD theory validation and attack pipeline.

This is the entry point for the rigorous validation requested:
  1. Drift decomposition (C1, C2)
  2. Directional ablation (C3)
  3. SRS ablation (C4)
  4. Full attack comparison vs baselines

Quick start:
  python run_all.py --dataset Synapse --vit_name vit_b --mode full

Modes:
  - drift_only: Only run drift decomposition validation (G1)
  - ablation: Run G1 + G2 + G3 (theory validation)
  - attack: Run full CODD attack on all test cases
  - full: Everything (drift + ablation + attack)

Output goes to ./output/{dataset}_{vit_name}_{timestamp}/
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/data2/fyc/fyc/GRAT-main")
NEWEST = ROOT / "newest"


# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}

SAMED_CKPTS = {
    ("Synapse", "vit_b"): f"{ROOT}/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth",
    ("Synapse", "vit_l"): f"{ROOT}/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_l_epo200_bs12_lr0.005/epoch_159.pth",
    ("Synapse", "vit_h"): f"{ROOT}/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_h_epo200_bs8_lr0.005/epoch_159.pth",
    ("ISIC", "vit_b"): f"{ROOT}/OUT_ROOT/SAMed/isic_512_pretrain_vit_b_epo200_bs6_lr0.005/epoch_159.pth",
    ("ISIC", "vit_l"): f"{ROOT}/OUT_ROOT/SAMed/isic_512_pretrain_vit_l_epo200_bs16_lr0.005/epoch_159.pth",
    ("ISIC", "vit_h"): f"{ROOT}/OUT_ROOT/SAMed/isic_512_pretrain_vit_h_epo200_bs8_lr0.005/epoch_159.pth",
    ("REFUGE", "vit_b"): f"{ROOT}/OUT_ROOT/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs8_lr0.005/epoch_159.pth",
    ("REFUGE", "vit_l"): f"{ROOT}/OUT_ROOT/SAMed/REFUGE_512_pretrain_vit_l_epo200_bs16_lr0.005/epoch_159.pth",
    ("REFUGE", "vit_h"): f"{ROOT}/OUT_ROOT/SAMed/REFUGE_512_pretrain_vit_h_epo200_bs8_lr0.005/epoch_159.pth",
}

MEDSA_CKPTS = {
    ("Synapse", "vit_b"): f"{ROOT}/OUT_ROOT/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs2_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    ("Synapse", "vit_l"): f"{ROOT}/OUT_ROOT/MedicalSAMAdapter/Synapse_512_pretrain_vit_l_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    ("Synapse", "vit_h"): f"{ROOT}/OUT_ROOT/MedicalSAMAdapter/Synapse_512_pretrain_vit_h_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    ("ISIC", "vit_b"): f"{ROOT}/OUT_ROOT/MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs2_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    ("ISIC", "vit_l"): f"{ROOT}/OUT_ROOT/MedicalSAMAdapter/isic_256_pretrain_vit_l_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    ("ISIC", "vit_h"): f"{ROOT}/OUT_ROOT/MedicalSAMAdapter/isic_512_pretrain_vit_h_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    ("REFUGE", "vit_b"): f"{ROOT}/OUT_ROOT/MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs2_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    ("REFUGE", "vit_l"): f"{ROOT}/OUT_ROOT/MedicalSAMAdapter/REFUGE_256_pretrain_vit_l_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    ("REFUGE", "vit_h"): f"{ROOT}/OUT_ROOT/MedicalSAMAdapter/REFUGE_512_pretrain_vit_h_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth",
}

# SRS paths (from new/)
SRS_PATHS = {
    ("Synapse", "vit_b"): f"{ROOT}/new/offline_codd_synapse_vit_b.pth",
    ("Synapse", "vit_l"): f"{ROOT}/new/offline_codd_synapse_vit_l.pth",
    ("Synapse", "vit_h"): f"{ROOT}/new/offline_codd_synapse_vit_h.pth",
    ("ISIC", "vit_b"): f"{ROOT}/new/offline_codd_isic_vit_b.pth",
    ("ISIC", "vit_l"): f"{ROOT}/new/offline_codd_isic_vit_l.pth",
    ("ISIC", "vit_h"): f"{ROOT}/new/offline_codd_isic_vit_h.pth",
    ("REFUGE", "vit_b"): f"{ROOT}/new/offline_codd_refuge_vit_b.pth",
    ("REFUGE", "vit_l"): f"{ROOT}/new/offline_codd_refuge_vit_l.pth",
    ("REFUGE", "vit_h"): f"{ROOT}/new/offline_codd_refuge_vit_h.pth",
}


def parse_args():
    p = argparse.ArgumentParser(description="CODD Theory Validation & Attack Pipeline")
    p.add_argument("--dataset", type=str, default="Synapse", choices=["Synapse", "ISIC", "REFUGE"])
    p.add_argument("--vit_name", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--mode", type=str, default="full",
                   choices=["drift_only", "ablation", "attack", "full"])
    p.add_argument("--max_cases", type=int, default=10)
    p.add_argument("--max_slices", type=int, default=8)
    p.add_argument("--att_numstep", type=int, default=10)
    p.add_argument("--att_step_size", type=float, default=2.0)
    p.add_argument("--att_bound", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--python_bin", type=str, default="python")
    return p.parse_args()


def _run(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"RUNNING: {desc}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: command exited with code {result.returncode}")
    return result


def main():
    args = parse_args()

    # Verify checkpoints exist
    sam_ckpt = SAM_CKPTS[args.vit_name]
    samed_ckpt = SAMED_CKPTS[(args.dataset, args.vit_name)]
    medsa_ckpt = MEDSA_CKPTS[(args.dataset, args.vit_name)]
    srs_path = SRS_PATHS.get((args.dataset, args.vit_name), "")

    for name, path in [("SAM", sam_ckpt), ("SAMed", samed_ckpt), ("MedSA", medsa_ckpt)]:
        if not os.path.exists(path):
            print(f"ERROR: {name} checkpoint not found: {path}")
            sys.exit(1)

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = str(NEWEST / "output" / f"{args.dataset}_{args.vit_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    validate_script = str(NEWEST / "validate_theory.py")

    # Determine what to run
    do_drift = args.mode in ("drift_only", "ablation", "full")
    do_attack = args.mode in ("ablation", "attack", "full")

    skip_drift_flag = [] if do_drift else ["--skip_drift"]
    skip_attack_flag = [] if do_attack else ["--skip_attack"]

    if do_drift or do_attack:
        cmd = (
            [args.python_bin, validate_script,
             "--dataset", args.dataset,
             "--vit_name", args.vit_name,
             "--source_ckpt", sam_ckpt,
             "--samed_ckpt", samed_ckpt,
             "--medsa_ckpt", medsa_ckpt,
             "--max_cases", str(args.max_cases),
             "--max_slices", str(args.max_slices),
             "--att_numstep", str(args.att_numstep),
             "--att_step_size", str(args.att_step_size),
             "--att_bound", str(args.att_bound),
             "--device", args.device,
             "--seed", str(args.seed),
             "--output_dir", out_dir,
             "--srs_path", srs_path if os.path.exists(srs_path) else "",
             ]
            + skip_drift_flag
            + skip_attack_flag
        )
        _run(cmd, f"CODD Validation: drift={do_drift}, attack={do_attack}")

    print(f"\n{'='*60}")
    print(f"DONE. Results in: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
