#!/usr/bin/env python3
"""Combine frozen development evaluations with paired uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def extract(path: Path, prefix: str):
    payload = json.loads(path.read_text())
    result = {}
    for family, key in (("samed", "SAMed"), ("medsa", "MedSA_per_case_paired")):
        if family == "samed":
            clean = {row["case"]: row["dice"] for row in payload[key]["clean"]["per_case"]}
            candidates = {
                name: {row["case"]: row["dice"] for row in payload[key][name]["per_case"]}
                for name in ("source_feature_B", "gr_only_R")
            }
        else:
            clean = {row["case"]: row["dice"] for row in payload[key]["clean"]}
            candidates = {
                name: {row["case"]: row["dice"] for row in payload[key][name]}
                for name in ("source_feature_B", "gr_only_R")
            }
        result[f"{prefix}_{family}"] = {"clean": clean, **candidates}
    return result


def paired_summary(values, seed: int = 20261011, samples: int = 10000):
    cases = sorted(values["clean"])
    clean = np.asarray([values["clean"][case] for case in cases])
    candidate = np.asarray([values["source_feature_B"][case] for case in cases])
    baseline = np.asarray([values["gr_only_R"][case] for case in cases])
    improvement = baseline - candidate
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cases), size=(samples, len(cases)))
    boot = improvement[indices].mean(axis=1)
    return {
        "cases": len(cases),
        "clean_mean_dice": float(clean.mean()),
        "source_feature_B_mean_dice": float(candidate.mean()),
        "gr_only_R_mean_dice": float(baseline.mean()),
        "B_improvement_over_GR": float(improvement.mean()),
        "B_improvement_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "win_tie_loss": {
            "win": int(np.sum(improvement > 1e-8)),
            "tie": int(np.sum(np.abs(improvement) <= 1e-8)),
            "loss": int(np.sum(improvement < -1e-8)),
        },
        "per_case": [
            {
                "case": case,
                "clean": float(clean[index]),
                "source_feature_B": float(candidate[index]),
                "gr_only_R": float(baseline[index]),
                "B_improvement_over_GR": float(improvement[index]),
            }
            for index, case in enumerate(cases)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank16", nargs="+", required=True)
    parser.add_argument("--seed45", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merged = {}
    for prefix, paths in (("rank16_default", args.rank16), ("seed45", args.seed45)):
        for path in paths:
            extracted = extract(Path(path), prefix)
            for target, values in extracted.items():
                destination = merged.setdefault(target, {
                    "clean": {}, "source_feature_B": {}, "gr_only_R": {}
                })
                for key in destination:
                    overlap = set(destination[key]) & set(values[key])
                    if overlap:
                        raise ValueError(f"duplicate cases for {target}/{key}: {sorted(overlap)}")
                    destination[key].update(values[key])
    summaries = {target: paired_summary(values) for target, values in merged.items()}
    result = {
        "targets": summaries,
        "worst_target_improvement": min(
            row["B_improvement_over_GR"] for row in summaries.values()
        ),
        "all_target_means_improve": all(
            row["B_improvement_over_GR"] > 0 for row in summaries.values()
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        target: {
            "B_improvement_over_GR": row["B_improvement_over_GR"],
            "ci95": row["B_improvement_ci95"],
            "win_tie_loss": row["win_tie_loss"],
        }
        for target, row in summaries.items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

