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
    parser.add_argument("--dataset", type=str, required=True, choices=["Synapse", "ISIC", "REFUGE"])
    parser.add_argument("--vit_name", type=str, required=True, choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_ckpt", type=str, required=True)
    parser.add_argument("--samed_ckpt", type=str, required=True)
    parser.add_argument("--medsa_ckpt", type=str, required=True)
    parser.add_argument("--shared_subspace_path", type=str, required=True)
    parser.add_argument("--max_cases", type=int, default=3)
    parser.add_argument("--max_slices", type=int, default=6)
    parser.add_argument("--att_numstep", type=int, default=5)
    parser.add_argument("--att_step_size", type=float, default=2.0)
    parser.add_argument("--att_bound", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def read_summary(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    ensure_dir(out_root)
    script_root = Path("/root/data2/fyc/fyc/GRAT-main/new/test")
    attack_script = script_root / "generate_directional_attacks.py"
    eval_script = script_root / "evaluate_victims.py"

    adv_dir = out_root / "adv_topk_radial"
    repack_dir = out_root / "adv_topk_radial_repacked"
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
            "radial",
            "--shared_subspace_path",
            args.shared_subspace_path,
            "--shared_mode",
            "topk",
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
    summary = {"dataset": args.dataset, "vit_name": args.vit_name, "victims": {}}
    clean_adv_dir = "__clean__" if args.dataset != "Synapse" else "/root/data2/fyc/fyc/GRAT-main/data/Synapse_preprocess/test_vol_h5"
    for victim, ckpt, attack_path in [
        ("samed", args.samed_ckpt, adv_dir),
        ("medsa", args.medsa_ckpt, repack_dir if args.dataset == "Synapse" else adv_dir),
    ]:
        clean_dir = out_root / f"{victim}_clean"
        adv_eval_dir = out_root / f"{victim}_adv"
        for use_adv, eval_dir, adv_input in [(False, clean_dir, clean_adv_dir), (True, adv_eval_dir, str(attack_path))]:
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
                    str(adv_input),
                    "--max_cases",
                    str(args.max_cases),
                    "--device",
                    args.device,
                    "--output_dir",
                    str(eval_dir),
                ]
            )
        clean_summary = read_summary(clean_dir / "summary.json")
        adv_summary = read_summary(adv_eval_dir / "summary.json")
        summary["victims"][victim] = {
            "clean_mean_dice": clean_summary["mean_dice"],
            "clean_mean_hd95": clean_summary["mean_hd95"],
            "adv_mean_dice": adv_summary["mean_dice"],
            "adv_mean_hd95": adv_summary["mean_hd95"],
            "dice_drop": clean_summary["mean_dice"] - adv_summary["mean_dice"],
            "hd95_increase": adv_summary["mean_hd95"] - clean_summary["mean_hd95"],
        }

        with open(clean_dir / "per_case_metrics.csv", "r", encoding="utf-8") as f:
            clean_rows = list(csv.DictReader(f))
        with open(adv_eval_dir / "per_case_metrics.csv", "r", encoding="utf-8") as f:
            adv_rows = list(csv.DictReader(f))
        summary["victims"][victim]["paired_case_analysis"] = {
            "dice_drop": summarize_paired_differences(
                [float(r["mean_dice"]) for r in clean_rows],
                [float(r["mean_dice"]) for r in adv_rows],
                "clean",
                "adv",
            ),
            "hd95_increase": summarize_paired_differences(
                [float(r["mean_hd95"]) for r in adv_rows],
                [float(r["mean_hd95"]) for r in clean_rows],
                "adv",
                "clean",
            ),
            "qc_flags": build_qc_flags(
                [float(r["mean_dice"]) for r in clean_rows],
                [float(r["mean_dice"]) for r in adv_rows],
                [float(r["mean_hd95"]) for r in clean_rows],
                [float(r["mean_hd95"]) for r in adv_rows],
            ),
        }
    write_json(out_root / "macro_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
