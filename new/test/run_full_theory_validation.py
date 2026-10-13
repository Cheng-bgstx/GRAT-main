from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from theory_common import ensure_dir, write_json


SAM_CKPTS = {
    "vit_b": "/root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "/root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "/root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_h_4b8939.pth",
}

SAMED_CKPTS = {
    "Synapse": {
        "vit_b": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth",
        "vit_l": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_l_epo200_bs12_lr0.005/epoch_159.pth",
        "vit_h": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_h_epo200_bs8_lr0.005/epoch_159.pth",
    },
    "ISIC": {
        "vit_b": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/isic_512_pretrain_vit_b_epo200_bs6_lr0.005/epoch_159.pth",
        "vit_l": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/isic_512_pretrain_vit_l_epo200_bs16_lr0.005/epoch_159.pth",
        "vit_h": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/isic_512_pretrain_vit_h_epo200_bs8_lr0.005/epoch_159.pth",
    },
    "REFUGE": {
        "vit_b": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs8_lr0.005/epoch_159.pth",
        "vit_l": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/REFUGE_512_pretrain_vit_l_epo200_bs16_lr0.005/epoch_159.pth",
        "vit_h": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/REFUGE_512_pretrain_vit_h_epo200_bs8_lr0.005/epoch_159.pth",
    },
}

MEDSA_CKPTS = {
    "Synapse": {
        "vit_b": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs2_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        "vit_l": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/Synapse_512_pretrain_vit_l_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        "vit_h": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/Synapse_512_pretrain_vit_h_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    },
    "ISIC": {
        "vit_b": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs2_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        "vit_l": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/isic_256_pretrain_vit_l_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        "vit_h": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/isic_512_pretrain_vit_h_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    },
    "REFUGE": {
        "vit_b": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs2_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        "vit_l": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/REFUGE_256_pretrain_vit_l_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        "vit_h": "/root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/REFUGE_512_pretrain_vit_h_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    },
}

SRS_PATHS = {
    "Synapse": {
        "vit_b": "/root/data2/fyc/fyc/GRAT-main/new/offline_codd_synapse_vit_b.pth",
        "vit_l": "/root/data2/fyc/fyc/GRAT-main/new/offline_codd_synapse_vit_l.pth",
        "vit_h": "/root/data2/fyc/fyc/GRAT-main/new/offline_codd_synapse_vit_h.pth",
    },
    "ISIC": {
        "vit_b": "/root/data2/fyc/fyc/GRAT-main/new/offline_codd_isic_vit_b.pth",
        "vit_l": "/root/data2/fyc/fyc/GRAT-main/new/offline_codd_isic_vit_l.pth",
        "vit_h": "/root/data2/fyc/fyc/GRAT-main/new/offline_codd_isic_vit_h.pth",
    },
    "REFUGE": {
        "vit_b": "/root/data2/fyc/fyc/GRAT-main/new/offline_codd_refuge_vit_b.pth",
        "vit_l": "/root/data2/fyc/fyc/GRAT-main/new/offline_codd_refuge_vit_l.pth",
        "vit_h": "/root/data2/fyc/fyc/GRAT-main/new/offline_codd_refuge_vit_h.pth",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_cases", type=int, default=3)
    parser.add_argument("--max_slices", type=int, default=6)
    parser.add_argument("--att_numstep", type=int, default=5)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    ensure_dir(out_root)
    script_root = Path("/root/data2/fyc/fyc/GRAT-main/new/test")
    directional_script = script_root / "run_directional_ablation.py"
    srs_script = script_root / "run_srs_ablation.py"
    macro_script = script_root / "run_macro_robustness.py"
    geom_script = script_root / "visualize_codd_geometry.py"

    registry = {"group1_directional": {}, "group2_srs": {}, "group3_macro": {}, "group4_geometry": {}}

    for vit_name in ["vit_b", "vit_l", "vit_h"]:
        sam_ckpt = SAM_CKPTS[vit_name]
        # Group 1 and 2 on Synapse across all three source SAM backbones.
        directional_out = out_root / "group1_directional" / vit_name
        run(
            [
                args.python_bin,
                str(directional_script),
                "--dataset",
                "Synapse",
                "--vit_name",
                vit_name,
                "--sam_ckpt",
                sam_ckpt,
                "--samed_ckpt",
                SAMED_CKPTS["Synapse"][vit_name],
                "--medsa_ckpt",
                MEDSA_CKPTS["Synapse"][vit_name],
                "--max_cases",
                str(args.max_cases),
                "--max_slices",
                str(args.max_slices),
                "--att_numstep",
                str(args.att_numstep),
                "--device",
                args.device,
                "--output_dir",
                str(directional_out),
            ]
        )
        registry["group1_directional"][vit_name] = str(directional_out / "directional_summary.json")

        srs_out = out_root / "group2_srs" / vit_name
        run(
            [
                args.python_bin,
                str(srs_script),
                "--dataset",
                "Synapse",
                "--vit_name",
                vit_name,
                "--sam_ckpt",
                sam_ckpt,
                "--samed_ckpt",
                SAMED_CKPTS["Synapse"][vit_name],
                "--medsa_ckpt",
                MEDSA_CKPTS["Synapse"][vit_name],
                "--shared_subspace_path",
                SRS_PATHS["Synapse"][vit_name],
                "--max_cases",
                str(args.max_cases),
                "--max_slices",
                str(args.max_slices),
                "--att_numstep",
                str(args.att_numstep),
                "--device",
                args.device,
                "--output_dir",
                str(srs_out),
            ]
        )
        registry["group2_srs"][vit_name] = str(srs_out / "srs_summary.json")

    for dataset in ["Synapse", "ISIC", "REFUGE"]:
        registry["group3_macro"][dataset] = {}
        for vit_name in ["vit_b", "vit_l", "vit_h"]:
            macro_out = out_root / "group3_macro" / dataset / vit_name
            run(
                [
                    args.python_bin,
                    str(macro_script),
                    "--dataset",
                    dataset,
                    "--vit_name",
                    vit_name,
                    "--sam_ckpt",
                    SAM_CKPTS[vit_name],
                    "--samed_ckpt",
                    SAMED_CKPTS[dataset][vit_name],
                    "--medsa_ckpt",
                    MEDSA_CKPTS[dataset][vit_name],
                    "--shared_subspace_path",
                    SRS_PATHS[dataset][vit_name],
                    "--max_cases",
                    str(args.max_cases),
                    "--max_slices",
                    str(args.max_slices),
                    "--att_numstep",
                    str(args.att_numstep),
                    "--device",
                    args.device,
                    "--output_dir",
                    str(macro_out),
                ]
            )
            registry["group3_macro"][dataset][vit_name] = str(macro_out / "macro_summary.json")

    for dataset in ["Synapse", "ISIC", "REFUGE"]:
        geom_out = out_root / "group4_geometry" / dataset
        run(
            [
                args.python_bin,
                str(geom_script),
                "--dataset",
                dataset,
                "--vit_name",
                "vit_b",
                "--sam_ckpt",
                SAM_CKPTS["vit_b"],
                "--samed_ckpt",
                SAMED_CKPTS[dataset]["vit_b"],
                "--medsa_ckpt",
                MEDSA_CKPTS[dataset]["vit_b"],
                "--att_numstep",
                str(args.att_numstep),
                "--device",
                args.device,
                "--output_dir",
                str(geom_out),
            ]
        )
        registry["group4_geometry"][dataset] = str(geom_out / "geometry_summary.json")

    write_json(out_root / "theory_validation_registry.json", registry)
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
