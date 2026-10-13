#!/usr/bin/env python3
"""Test an image-only, monotone-acquisition-invariant local order proxy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from peft_transfer.evaluate_synapse_slice import ROOT
from peft_transfer.score_medical_invariant_candidates import official_candidates


def pair_score(clean: np.ndarray, attacked: np.ndarray, label: np.ndarray,
               epsilon: float, span: int = 2) -> dict:
    scores = []
    boundary_scores = []
    eligible_counts = []
    boundary_counts = []
    for axis in (0, 1):
        start = [slice(None), slice(None)]
        end = [slice(None), slice(None)]
        start[axis] = slice(None, -2 * span)
        end[axis] = slice(2 * span, None)
        start, end = tuple(start), tuple(end)
        clean_gap = clean[end] - clean[start]
        attacked_gap = attacked[end] - attacked[start]
        eligible = (np.abs(clean_gap) >= epsilon / 4) & (np.abs(clean_gap) <= 2 * epsilon)
        reversal = (clean_gap * attacked_gap < 0) & eligible
        boundary = label[start] != label[end]
        scores.append(int(reversal.sum()))
        eligible_counts.append(int(eligible.sum()))
        boundary_scores.append(int((reversal & boundary).sum()))
        boundary_counts.append(int((eligible & boundary).sum()))
    return {
        "local_order_flip_rate": sum(scores) / max(sum(eligible_counts), 1),
        "label_boundary_flip_rate_post_hoc": sum(boundary_scores) / max(sum(boundary_counts), 1),
        "eligible_pairs": sum(eligible_counts),
        "eligible_label_boundary_pairs_post_hoc": sum(boundary_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = official_candidates(args.official)
    clean_cache = {}
    for row in rows:
        key = (row["case"], row["slice"])
        if key not in clean_cache:
            with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{key[0]}.npy.h5") as handle:
                clean_cache[key] = (handle["image"][key[1]].astype(np.float32),
                                    handle["label"][key[1]].astype(np.int64))
        clean, label = clean_cache[key]
        attacked = np.asarray(Image.open(ROOT / row["candidate"]).convert("L"),
                              dtype=np.float32) / 255.0
        row["local_order"] = pair_score(clean, attacked, label, 10 / 255)
    rankings = {}
    for key in sorted(clean_cache):
        subset = [row for row in rows if (row["case"], row["slice"]) == key]
        if len(subset) < 2:
            continue
        scores = [row["local_order"]["local_order_flip_rate"] for row in subset]
        rankings[f"{key[0]}/slice{key[1]}"] = {
            "predicted_best": subset[int(np.argmax(scores))]["candidate"],
            "SAMed_best": max(subset, key=lambda row: row["SAMed_dice_drop"])["candidate"],
            "MedSA_best": max(subset, key=lambda row: row["MedSA_dice_drop"])["candidate"],
            "SAMed_rho": float(spearmanr(scores, [row["SAMed_dice_drop"] for row in subset])[0]),
            "MedSA_rho": float(spearmanr(scores, [row["MedSA_dice_drop"] for row in subset])[0]),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "source_only_score": True,
        "target_metrics_used_for_post_hoc_ranking_only": True,
        "span": 2, "epsilon_255": 10,
        "rows": rows, "within_case_rankings": rankings,
    }, indent=2))
    print(json.dumps(rankings, indent=2))


if __name__ == "__main__":
    main()
