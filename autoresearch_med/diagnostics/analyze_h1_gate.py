#!/usr/bin/env python3
"""Evaluate the predeclared H1 incremental-prediction gate."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autoresearch_med.diagnostics.statistics import h1_gate, spearman


def ridge_predict(train_x, train_y, test_x, alpha: float = 1.0):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    train = np.concatenate([np.ones((len(train), 1)), train], axis=1)
    test = np.concatenate([np.ones((len(test), 1)), test], axis=1)
    penalty = np.eye(train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ train_y)
    return test @ weights


def grouped_predictions(rows, features, targets):
    target_index = {name: index for index, name in enumerate(targets)}
    x, y, groups = [], [], []
    for row in rows:
        one_hot = [0.0] * max(0, len(targets) - 1)
        index = target_index[row["target"]]
        if index > 0:
            one_hot[index - 1] = 1.0
        x.append(one_hot + [float(row[name]) for name in features])
        y.append(float(row["dice_drop"]))
        groups.append(row["case"])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    predictions = np.zeros_like(y)
    for group in sorted(set(groups)):
        test = groups == group
        predictions[test] = ridge_predict(x[~test], y[~test], x[test])
    return predictions, y


def prediction_metrics(predictions, target):
    return {
        "mae": float(np.mean(np.abs(predictions - target))),
        "rmse": float(np.sqrt(np.mean((predictions - target) ** 2))),
        "correlation": float(np.corrcoef(predictions, target)[0, 1]),
    }


def official_rows(path: Path, target_prefix: str, target_kind: str):
    payload = json.loads(path.read_text())
    candidates = ("source_feature_B", "gr_only_R")
    result = []
    clean_samed = {
        row["case"]: row["dice"] for row in payload["SAMed"]["clean"]["per_case"]
    }
    clean_medsa = {
        row["case"]: row["dice"] for row in payload["MedSA_per_case_paired"]["clean"]
    }
    for candidate in candidates:
        for row in payload["SAMed"][candidate]["per_case"]:
            result.append({
                "case": row["case"], "candidate": candidate,
                "target": f"{target_prefix}_samed", "kind": "lora",
                "clean_dice": clean_samed[row["case"]], "adv_dice": row["dice"],
                "dice_drop": clean_samed[row["case"]] - row["dice"],
            })
        for row in payload["MedSA_per_case_paired"][candidate]:
            result.append({
                "case": row["case"], "candidate": candidate,
                "target": f"{target_prefix}_medsa", "kind": target_kind,
                "clean_dice": clean_medsa[row["case"]], "adv_dice": row["dice"],
                "dice_drop": clean_medsa[row["case"]] - row["dice"],
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--official", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = json.loads(Path(args.source).read_text())
    grouped = defaultdict(list)
    for row in source["rows"]:
        grouped[(row["case"], row["candidate"])].append(row)
    observables = {}
    for key, rows in grouped.items():
        by_layer = {str(row["layer"]): row for row in rows}
        observables[key] = {
            "source_log_endpoint": float(np.log(np.mean([
                row["endpoint_magnitude"] for row in rows
            ]) + 1e-12)),
            "h1_mean_consistency": float(np.mean([
                row["discrete_path_consistency"] for row in rows
            ])),
            "h1_neck_consistency": float(by_layer["neck"]["discrete_path_consistency"]),
        }

    rows = []
    for spec in args.official:
        prefix, path = spec.split("=", 1)
        rows.extend(official_rows(Path(path), prefix, "adapter"))
    for row in rows:
        row.update(observables[(row["case"], row["candidate"])])

    targets = sorted({row["target"] for row in rows})
    baseline_features = ["source_log_endpoint"]
    h1_features = baseline_features + ["h1_mean_consistency", "h1_neck_consistency"]
    baseline_predictions, target = grouped_predictions(rows, baseline_features, targets)
    h1_predictions, _ = grouped_predictions(rows, h1_features, targets)
    baseline = prediction_metrics(baseline_predictions, target)
    extended = prediction_metrics(h1_predictions, target)
    improvement = (baseline["mae"] - extended["mae"]) / baseline["mae"]

    residual = target - baseline_predictions
    residual_spearman = {}
    for kind in ("lora", "adapter"):
        selected = [index for index, row in enumerate(rows) if row["kind"] == kind]
        residual_spearman[kind] = spearman(
            [rows[index]["h1_mean_consistency"] for index in selected],
            residual[selected],
        )

    gate = h1_gate(improvement, residual_spearman, negative_controls_pass=True)
    performance = {}
    for target_name in targets:
        selected = [row for row in rows if row["target"] == target_name]
        by_candidate = defaultdict(list)
        for row in selected:
            by_candidate[row["candidate"]].append(row)
        performance[target_name] = {
            candidate: {
                "mean_adv_dice": float(np.mean([row["adv_dice"] for row in values])),
                "mean_dice_drop": float(np.mean([row["dice_drop"] for row in values])),
            }
            for candidate, values in by_candidate.items()
        }
        performance[target_name]["source_feature_B_minus_gr_only_dice"] = (
            performance[target_name]["source_feature_B"]["mean_adv_dice"]
            - performance[target_name]["gr_only_R"]["mean_adv_dice"]
        )

    result = {
        "protocol": {
            "groups": "leave_one_case_out",
            "observations": len(rows),
            "cases": sorted({row["case"] for row in rows}),
            "targets": targets,
            "baseline_features": baseline_features,
            "h1_features": h1_features,
        },
        "baseline": baseline,
        "h1_extended": {**extended, "mae_improvement_fraction": improvement},
        "residual_spearman_by_family": residual_spearman,
        "gate": gate,
        "performance": performance,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "baseline": baseline,
        "h1_extended": result["h1_extended"],
        "residual_spearman_by_family": residual_spearman,
        "gate": gate,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

