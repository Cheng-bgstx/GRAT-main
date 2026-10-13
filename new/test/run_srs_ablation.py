from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from theory_common import build_qc_flags, ensure_dir, summarize_paired_differences, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--dataset", type=str, default="Synapse", choices=["Synapse", "ISIC", "REFUGE"])
    parser.add_argument("--vit_name", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_ckpt", type=str, required=True)
    parser.add_argument("--samed_ckpt", type=str, required=True)
    parser.add_argument("--medsa_ckpt", type=str, required=True)
    parser.add_argument("--shared_subspace_path", type=str, required=True)
    parser.add_argument("--max_cases", type=int, default=2)
    parser.add_argument("--max_slices", type=int, default=4)
    parser.add_argument("--att_numstep", type=int, default=3)
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


def average_metric(rows, key: str):
    values = [float(r[key]) for r in rows]
    return sum(values) / max(1, len(values))


def load_trace_summary(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    curve = [float(v) for v in payload.get("mean_loss_by_step", [])]
    return {
        "mean_loss_by_step": curve,
        "loss_last": float(curve[-1]) if curve else 0.0,
        "loss_auc": float(np.trapz(curve, dx=1.0)) if len(curve) > 1 else (float(curve[0]) if curve else 0.0),
        "num_units": int(payload.get("num_units", 0)),
    }


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    ensure_dir(out_root)
    script_root = Path("/root/data2/fyc/fyc/GRAT-main/new/test")
    attack_script = script_root / "generate_directional_attacks.py"
    eval_script = script_root / "evaluate_victims.py"

    settings = [
        ("zero", []),
        ("random", ["--shared_mode", "random"]),
        ("mean", ["--shared_subspace_path", args.shared_subspace_path, "--shared_mode", "mean"]),
        ("top1", ["--shared_subspace_path", args.shared_subspace_path, "--shared_mode", "top1"]),
        ("topk", ["--shared_subspace_path", args.shared_subspace_path, "--shared_mode", "topk"]),
    ]
    artifacts = {}
    for name, extra_args in settings:
        adv_dir = out_root / f"adv_{name}"
        repack_dir = out_root / f"adv_{name}_repacked"
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
                *extra_args,
            ]
        )
        samed_out = out_root / f"eval_samed_{name}"
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
        medsa_out = out_root / f"eval_medsa_{name}"
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
        artifacts[name] = {
            "samed_csv": str(samed_out / "per_case_metrics.csv"),
            "medsa_csv": str(medsa_out / "per_case_metrics.csv"),
            "trace_json": str(out_root / f"adv_{name}_trace.json"),
        }

    rows_samed = {name: load_csv(Path(paths["samed_csv"])) for name, paths in artifacts.items()}
    rows_medsa = {name: load_csv(Path(paths["medsa_csv"])) for name, paths in artifacts.items()}
    trace_summaries = {name: load_trace_summary(Path(paths["trace_json"])) for name, paths in artifacts.items()}
    summary = {
        "dataset": args.dataset,
        "mean_dice_samed": {name: average_metric(v, "mean_dice") for name, v in rows_samed.items()},
        "mean_hd95_samed": {name: average_metric(v, "mean_hd95") for name, v in rows_samed.items()},
        "mean_dice_medsa": {name: average_metric(v, "mean_dice") for name, v in rows_medsa.items()},
        "mean_hd95_medsa": {name: average_metric(v, "mean_hd95") for name, v in rows_medsa.items()},
        "source_loss_traces": trace_summaries,
        "pairwise": {
            "samed_topk_vs_zero_dice": summarize_paired_differences(
                [float(r["mean_dice"]) for r in rows_samed["topk"]],
                [float(r["mean_dice"]) for r in rows_samed["zero"]],
                "topk",
                "zero",
            ),
            "samed_topk_vs_random_dice": summarize_paired_differences(
                [float(r["mean_dice"]) for r in rows_samed["topk"]],
                [float(r["mean_dice"]) for r in rows_samed["random"]],
                "topk",
                "random",
            ),
            "samed_topk_vs_mean_dice": summarize_paired_differences(
                [float(r["mean_dice"]) for r in rows_samed["topk"]],
                [float(r["mean_dice"]) for r in rows_samed["mean"]],
                "topk",
                "mean",
            ),
            "samed_topk_vs_top1_dice": summarize_paired_differences(
                [float(r["mean_dice"]) for r in rows_samed["topk"]],
                [float(r["mean_dice"]) for r in rows_samed["top1"]],
                "topk",
                "top1",
            ),
            "medsa_topk_vs_zero_dice": summarize_paired_differences(
                [float(r["mean_dice"]) for r in rows_medsa["topk"]],
                [float(r["mean_dice"]) for r in rows_medsa["zero"]],
                "topk",
                "zero",
            ),
        },
        "qc_flags": {
            name: {
                "samed": build_qc_flags(
                    [float(r["mean_dice"]) for r in rows_samed["zero"]],
                    [float(r["mean_dice"]) for r in rows_samed[name]],
                    [float(r["mean_hd95"]) for r in rows_samed["zero"]],
                    [float(r["mean_hd95"]) for r in rows_samed[name]],
                ),
                "medsa": build_qc_flags(
                    [float(r["mean_dice"]) for r in rows_medsa["zero"]],
                    [float(r["mean_dice"]) for r in rows_medsa[name]],
                    [float(r["mean_hd95"]) for r in rows_medsa["zero"]],
                    [float(r["mean_hd95"]) for r in rows_medsa[name]],
                ),
            }
            for name in rows_samed.keys()
        },
        "artifacts": artifacts,
    }
    write_json(out_root / "srs_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
