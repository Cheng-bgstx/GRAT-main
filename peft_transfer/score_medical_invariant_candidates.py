#!/usr/bin/env python3
"""Post-hoc test of invariant disruption and environment LCB vs official Dice."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

from peft_transfer.evaluate_synapse_slice import ROOT, SAM
from peft_transfer.medical_invariants import (
    capture_features, centered_scaled, environment_view, load_test_image,
)

sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_sam_encoder


VIEWS = ("identity", "contrast_08", "contrast_12", "gamma_09",
         "gamma_11", "bias_field", "blur")


def official_candidates(paths):
    rows = {}
    for report_path in paths:
        path = Path(report_path)
        match = re.search(r"(case\d+)_slice(\d+)", str(path))
        if match is None:
            raise ValueError(f"Official path must identify case/slice: {path}")
        case, index = match.group(1), int(match.group(2))
        report = json.loads(path.read_text())
        for candidate, values in report["SAMed"].items():
            if candidate == "clean":
                continue
            if candidate not in report.get("MedSA", {}):
                continue
            entry = {
                "case": case, "slice": index, "candidate": candidate,
                "SAMed_dice_drop": report["SAMed"]["clean"]["dice"] - values["dice"],
                "MedSA_dice_drop": report["MedSA"]["clean"]["dice"]
                - report["MedSA"][candidate]["dice"],
            }
            if candidate in rows and rows[candidate] != entry:
                raise ValueError(f"Inconsistent repeated official result for {candidate}")
            rows[candidate] = entry
    return list(rows.values())


def score_views(encoder, image, changed, layers, scales, bases):
    scores = {layer: {"full": [], "invariant": []} for layer in layers}
    with torch.no_grad():
        for view in VIEWS:
            original = capture_features(encoder, environment_view(image, view), layers)
            attacked = capture_features(encoder, environment_view(changed, view), layers)
            for layer in layers:
                scale = scales[layer].to(image.device)
                first = centered_scaled(original[layer], scale)
                second = centered_scaled(attacked[layer], scale)
                displacement = second - first
                basis = bases[layer].to(image.device)
                scores[layer]["full"].append(float(
                    displacement.norm() / (displacement.shape[0] * displacement.shape[1]) ** 0.5
                ))
                scores[layer]["invariant"].append(float(
                    (displacement @ basis).norm() / (displacement.shape[0] * basis.shape[1]) ** 0.5
                ))
    return scores


def correlation(rows, layer, score_key, outcome):
    x = [row["scores"][layer][score_key] for row in rows]
    y = [row[outcome] for row in rows]
    coefficient, pvalue = spearmanr(x, y)
    return {"rho": float(coefficient), "pvalue": float(pvalue)}


def within_case_rankings(rows, layer, score_key, outcome):
    cases = sorted({(row["case"], row["slice"]) for row in rows})
    result = {}
    for case, index in cases:
        subset = [row for row in rows if (row["case"], row["slice"]) == (case, index)]
        if len(subset) < 3:
            continue
        predictions = [row["scores"][layer][score_key] for row in subset]
        drops = [row[outcome] for row in subset]
        rho, _ = spearmanr(predictions, drops)
        result[f"{case}/slice{index}"] = {
            "n_candidates": len(subset),
            "rho": float(rho),
            "predicted_best": subset[int(np.argmax(predictions))]["candidate"],
            "actual_best": subset[int(np.argmax(drops))]["candidate"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--official", nargs="+", required=True)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    artifact = torch.load(args.artifact, map_location="cpu")
    if not artifact["source_only"] or artifact["target_checkpoint_used"]:
        raise ValueError("Expected a frozen source-only subspace")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    layers = tuple(artifact["layers"])
    scales = artifact["scales"]
    bases = {layer: artifact["bases"][layer]["invariant"] for layer in layers}
    rows = official_candidates(args.official)
    model = load_sam_encoder(artifact["vit"], str(SAM[artifact["vit"]]), 512, 8, device)
    encoder = model.image_encoder
    for row in rows:
        image = load_test_image(row["case"], row["slice"], device)
        candidate_path = ROOT / row["candidate"]
        array = np.asarray(Image.open(candidate_path).convert("L"), dtype=np.float32) / 255.0
        changed = torch.from_numpy(array)[None, None].repeat(1, 3, 1, 1).to(device)
        views = score_views(encoder, image, changed, layers, scales, bases)
        row["scores"] = {}
        for layer in layers:
            full = np.asarray(views[layer]["full"])
            invariant = np.asarray(views[layer]["invariant"])
            row["scores"][layer] = {
                "full_identity": float(full[0]),
                "invariant_identity": float(invariant[0]),
                "invariant_mean": float(invariant.mean()),
                "invariant_std": float(invariant.std()),
                "invariant_lcb": float(invariant.mean() - args.kappa * invariant.std()),
                "environment_displacements": dict(zip(VIEWS, map(float, invariant))),
            }
    correlations = {}
    for layer in layers:
        correlations[layer] = {}
        for victim in ("SAMed", "MedSA"):
            outcome = f"{victim}_dice_drop"
            correlations[layer][victim] = {
                "C_full": correlation(rows, layer, "full_identity", outcome),
                "C_invariant": correlation(rows, layer, "invariant_identity", outcome),
                "D_mean": correlation(rows, layer, "invariant_mean", outcome),
                "D_lcb": correlation(rows, layer, "invariant_lcb", outcome),
                "within_case": {
                    key: within_case_rankings(rows, layer, score_key, outcome)
                    for key, score_key in (
                        ("C_full", "full_identity"),
                        ("C_invariant", "invariant_identity"),
                        ("D_mean", "invariant_mean"),
                        ("D_lcb", "invariant_lcb"),
                    )
                },
            }
    payload = {"post_hoc_target_metrics_only": True,
               "source_artifact": args.artifact, "kappa": args.kappa,
               "environment_views": VIEWS, "n_candidates": len(rows),
               "rows": rows, "correlations": correlations}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()
