from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from theory_common import build_qc_flags, ensure_dir, summarize_paired_differences, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--dataset", type=str, default="Synapse", choices=["Synapse", "ISIC", "REFUGE"])
    parser.add_argument("--vit_name", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_ckpt", type=str, required=True)
    parser.add_argument("--samed_ckpt", type=str, required=True)
    parser.add_argument("--medsa_ckpt", type=str, required=True)
    parser.add_argument("--max_cases", type=int, default=2)
    parser.add_argument("--max_slices", type=int, default=4)
    parser.add_argument("--att_numstep", type=int, default=5)
    parser.add_argument("--att_step_size", type=float, default=2.0)
    parser.add_argument("--att_bound", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def load_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_metric(rows, key: str):
    return [float(r[key]) for r in rows]


def build_case_delta_summary(clean_rows, adv_rows):
    clean_dice = get_metric(clean_rows, "mean_dice")
    adv_dice = get_metric(adv_rows, "mean_dice")
    clean_hd95 = get_metric(clean_rows, "mean_hd95")
    adv_hd95 = get_metric(adv_rows, "mean_hd95")
    return {
        "dice_drop": summarize_paired_differences(clean_dice, adv_dice, "clean", "attacked"),
        "hd95_increase": summarize_paired_differences(adv_hd95, clean_hd95, "attacked", "clean"),
        "qc_flags": build_qc_flags(clean_dice, adv_dice, clean_hd95, adv_hd95),
    }


def average_metric(rows, key: str):
    values = [float(r[key]) for r in rows]
    return float(sum(values) / max(1, len(values)))


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    ensure_dir(out_root)
    script_root = Path("/root/data2/fyc/fyc/GRAT-main/new/test")
    attack_script = script_root / "generate_directional_attacks.py"
    eval_script = script_root / "evaluate_victims.py"

    modes = ["radial", "tangential", "isotropic", "misaligned_radial"]
    mode_results = {}

    clean_results = {}
    for victim, ckpt in [("samed", args.samed_ckpt), ("medsa", args.medsa_ckpt)]:
        clean_out = out_root / f"eval_{victim}_clean"
        run(
            [
                args.python_bin,
                str(eval_script),
                "--victim",
                victim,
                "--dataset",
                args.dataset,
                "--vit_name",
                args.vit_name,
                "--sam_ckpt",
                args.sam_ckpt,
                "--victim_ckpt",
                ckpt,
                "--adv_dir",
                "__clean__" if args.dataset != "Synapse" else "/root/data2/fyc/fyc/GRAT-main/data/Synapse_preprocess/test_vol_h5",
                "--max_cases",
                str(args.max_cases),
                "--device",
                args.device,
                "--output_dir",
                str(clean_out),
            ]
        )
        clean_results[victim] = str(clean_out / "per_case_metrics.csv")

    for mode in modes:
        adv_dir = out_root / f"adv_{mode}"
        repack_dir = out_root / f"adv_{mode}_repacked"
        run(
            [
                args.python_bin,
                str(attack_script),
                "--dataset",
                args.dataset,
                "--vit_name",
                args.vit_name,
                "--sam_ckpt",
                args.sam_ckpt,
                "--direction_mode",
                mode,
                "--att_numstep",
                str(args.att_numstep),
                "--att_step_size",
                str(args.att_step_size),
                "--att_bound",
                str(args.att_bound),
                "--max_cases",
                str(args.max_cases),
                "--max_slices",
                str(args.max_slices),
                "--device",
                args.device,
                "--out_path",
                str(adv_dir),
                "--repack_path",
                str(repack_dir),
            ]
        )
        samed_out = out_root / f"eval_samed_{mode}"
        medsa_out = out_root / f"eval_medsa_{mode}"
        run(
            [
                args.python_bin,
                str(eval_script),
                "--victim",
                "samed",
                "--dataset",
                args.dataset,
                "--vit_name",
                args.vit_name,
                "--sam_ckpt",
                args.sam_ckpt,
                "--victim_ckpt",
                args.samed_ckpt,
                "--adv_dir",
                str(adv_dir),
                "--max_cases",
                str(args.max_cases),
                "--device",
                args.device,
                "--output_dir",
                str(samed_out),
            ]
        )
        run(
            [
                args.python_bin,
                str(eval_script),
                "--victim",
                "medsa",
                "--dataset",
                args.dataset,
                "--vit_name",
                args.vit_name,
                "--sam_ckpt",
                args.sam_ckpt,
                "--victim_ckpt",
                args.medsa_ckpt,
                "--adv_dir",
                str(repack_dir if args.dataset == "Synapse" else adv_dir),
                "--max_cases",
                str(args.max_cases),
                "--device",
                args.device,
                "--output_dir",
                str(medsa_out),
            ]
        )
        mode_results[mode] = {
            "samed_csv": str(samed_out / "per_case_metrics.csv"),
            "medsa_csv": str(medsa_out / "per_case_metrics.csv"),
        }

    radial_samed = load_csv(Path(mode_results["radial"]["samed_csv"]))
    tangential_samed = load_csv(Path(mode_results["tangential"]["samed_csv"]))
    isotropic_samed = load_csv(Path(mode_results["isotropic"]["samed_csv"]))
    radial_medsa = load_csv(Path(mode_results["radial"]["medsa_csv"]))
    tangential_medsa = load_csv(Path(mode_results["tangential"]["medsa_csv"]))
    isotropic_medsa = load_csv(Path(mode_results["isotropic"]["medsa_csv"]))
    clean_samed = load_csv(Path(clean_results["samed"]))
    clean_medsa = load_csv(Path(clean_results["medsa"]))

    pairwise = {
        "radial_vs_tangential_dice": summarize_paired_differences(
            get_metric(radial_samed, "mean_dice"),
            get_metric(tangential_samed, "mean_dice"),
            "radial",
            "tangential",
        ),
        "radial_vs_isotropic_dice": summarize_paired_differences(
            get_metric(radial_samed, "mean_dice"),
            get_metric(isotropic_samed, "mean_dice"),
            "radial",
            "isotropic",
        ),
        "radial_vs_misaligned_dice": summarize_paired_differences(
            get_metric(radial_samed, "mean_dice"),
            get_metric(load_csv(Path(mode_results["misaligned_radial"]["samed_csv"])), "mean_dice"),
            "radial",
            "misaligned_radial",
        ),
        "medsa_radial_vs_tangential_dice": summarize_paired_differences(
            get_metric(radial_medsa, "mean_dice"),
            get_metric(tangential_medsa, "mean_dice"),
            "radial",
            "tangential",
        ),
        "medsa_radial_vs_isotropic_dice": summarize_paired_differences(
            get_metric(radial_medsa, "mean_dice"),
            get_metric(isotropic_medsa, "mean_dice"),
            "radial",
            "isotropic",
        ),
        "medsa_radial_vs_misaligned_dice": summarize_paired_differences(
            get_metric(radial_medsa, "mean_dice"),
            get_metric(load_csv(Path(mode_results["misaligned_radial"]["medsa_csv"])), "mean_dice"),
            "radial",
            "misaligned_radial",
        ),
        "radial_vs_tangential_hd95": summarize_paired_differences(
            get_metric(load_csv(Path(mode_results["tangential"]["samed_csv"])), "mean_hd95"),
            get_metric(load_csv(Path(mode_results["radial"]["samed_csv"])), "mean_hd95"),
            "tangential",
            "radial",
        ),
        "radial_vs_isotropic_hd95": summarize_paired_differences(
            get_metric(load_csv(Path(mode_results["isotropic"]["samed_csv"])), "mean_hd95"),
            get_metric(load_csv(Path(mode_results["radial"]["samed_csv"])), "mean_hd95"),
            "isotropic",
            "radial",
        ),
        "radial_vs_misaligned_hd95": summarize_paired_differences(
            get_metric(load_csv(Path(mode_results["misaligned_radial"]["samed_csv"])), "mean_hd95"),
            get_metric(load_csv(Path(mode_results["radial"]["samed_csv"])), "mean_hd95"),
            "misaligned_radial",
            "radial",
        ),
        "medsa_radial_vs_tangential_hd95": summarize_paired_differences(
            get_metric(load_csv(Path(mode_results["tangential"]["medsa_csv"])), "mean_hd95"),
            get_metric(load_csv(Path(mode_results["radial"]["medsa_csv"])), "mean_hd95"),
            "tangential",
            "radial",
        ),
        "medsa_radial_vs_isotropic_hd95": summarize_paired_differences(
            get_metric(load_csv(Path(mode_results["isotropic"]["medsa_csv"])), "mean_hd95"),
            get_metric(load_csv(Path(mode_results["radial"]["medsa_csv"])), "mean_hd95"),
            "isotropic",
            "radial",
        ),
        "medsa_radial_vs_misaligned_hd95": summarize_paired_differences(
            get_metric(load_csv(Path(mode_results["misaligned_radial"]["medsa_csv"])), "mean_hd95"),
            get_metric(load_csv(Path(mode_results["radial"]["medsa_csv"])), "mean_hd95"),
            "misaligned_radial",
            "radial",
        ),
    }
    per_mode = {
        "radial": {
            "samed": build_case_delta_summary(clean_samed, radial_samed),
            "medsa": build_case_delta_summary(clean_medsa, radial_medsa),
        },
        "tangential": {
            "samed": build_case_delta_summary(clean_samed, tangential_samed),
            "medsa": build_case_delta_summary(clean_medsa, tangential_medsa),
        },
        "isotropic": {
            "samed": build_case_delta_summary(clean_samed, isotropic_samed),
            "medsa": build_case_delta_summary(clean_medsa, isotropic_medsa),
        },
        "misaligned_radial": {
            "samed": build_case_delta_summary(clean_samed, load_csv(Path(mode_results["misaligned_radial"]["samed_csv"]))),
            "medsa": build_case_delta_summary(clean_medsa, load_csv(Path(mode_results["misaligned_radial"]["medsa_csv"]))),
        },
    }

    summary = {
        "dataset": args.dataset,
        "clean_mean_dice": {
            "samed": average_metric(clean_samed, "mean_dice"),
            "medsa": average_metric(clean_medsa, "mean_dice"),
        },
        "clean_mean_hd95": {
            "samed": average_metric(clean_samed, "mean_hd95"),
            "medsa": average_metric(clean_medsa, "mean_hd95"),
        },
        "attacked_mean_dice": {
            "radial_samed": average_metric(radial_samed, "mean_dice"),
            "tangential_samed": average_metric(tangential_samed, "mean_dice"),
            "isotropic_samed": average_metric(isotropic_samed, "mean_dice"),
            "misaligned_radial_samed": average_metric(load_csv(Path(mode_results["misaligned_radial"]["samed_csv"])), "mean_dice"),
            "radial_medsa": average_metric(radial_medsa, "mean_dice"),
            "tangential_medsa": average_metric(tangential_medsa, "mean_dice"),
            "isotropic_medsa": average_metric(isotropic_medsa, "mean_dice"),
            "misaligned_radial_medsa": average_metric(load_csv(Path(mode_results["misaligned_radial"]["medsa_csv"])), "mean_dice"),
        },
        "attacked_mean_hd95": {
            "radial_samed": average_metric(radial_samed, "mean_hd95"),
            "tangential_samed": average_metric(tangential_samed, "mean_hd95"),
            "isotropic_samed": average_metric(isotropic_samed, "mean_hd95"),
            "misaligned_radial_samed": average_metric(load_csv(Path(mode_results["misaligned_radial"]["samed_csv"])), "mean_hd95"),
            "radial_medsa": average_metric(radial_medsa, "mean_hd95"),
            "tangential_medsa": average_metric(tangential_medsa, "mean_hd95"),
            "isotropic_medsa": average_metric(isotropic_medsa, "mean_hd95"),
            "misaligned_radial_medsa": average_metric(load_csv(Path(mode_results["misaligned_radial"]["medsa_csv"])), "mean_hd95"),
        },
        "pairwise": pairwise,
        "per_mode": per_mode,
        "artifacts": {"clean": clean_results, **mode_results},
    }
    write_json(out_root / "directional_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
