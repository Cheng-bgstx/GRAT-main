#!/usr/bin/env python3
"""Post-hoc check of organ-boundary contrast changes in frozen attacks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr

from peft_transfer.evaluate_synapse_slice import ROOT


def boundary_contrast(clean: np.ndarray, attacked: np.ndarray,
                      label: np.ndarray, width: int = 3) -> dict[str, float]:
    reductions = []
    perturbations = []
    for class_id in np.unique(label):
        if class_id == 0:
            continue
        mask = label == class_id
        inside = mask & ~binary_erosion(mask, iterations=width)
        outside = binary_dilation(mask, iterations=width) & ~mask
        if inside.sum() < 5 or outside.sum() < 5:
            continue
        before = float(clean[inside].mean() - clean[outside].mean())
        after = float(attacked[inside].mean() - attacked[outside].mean())
        reductions.append(abs(before) - abs(after))
        perturbations.append(float(np.abs(attacked-clean)[inside | outside].mean()))
    if not reductions:
        raise ValueError("No represented organ boundary")
    return {
        "mean_absolute_contrast_reduction": float(np.mean(reductions)),
        "mean_boundary_perturbation": float(np.mean(perturbations)),
        "n_classes": len(reductions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.prediction).read_text())["rows"]
    cache = {}
    for row in rows:
        key = (row["case"], row["slice"])
        if key not in cache:
            with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{key[0]}.npy.h5") as handle:
                cache[key] = (handle["image"][key[1]].astype(np.float32),
                              handle["label"][key[1]].astype(np.int64))
        clean, label = cache[key]
        attacked = np.asarray(Image.open(ROOT / row["candidate"]).convert("L"),
                              dtype=np.float32) / 255.0
        row["boundary_contrast_post_hoc"] = boundary_contrast(clean, attacked, label)
    rankings = {}
    for key in cache:
        subset = [row for row in rows if (row["case"], row["slice"]) == key]
        if len(subset) < 2:
            continue
        case_key = f"{key[0]}/slice{key[1]}"
        rankings[case_key] = {}
        for score in ("mean_absolute_contrast_reduction", "mean_boundary_perturbation"):
            values = [row["boundary_contrast_post_hoc"][score] for row in subset]
            actual = [row["MedSA_dice_drop"] for row in subset]
            rankings[case_key][score] = {
                "predicted_best": subset[int(np.argmax(values))]["candidate"],
                "MedSA_best": subset[int(np.argmax(actual))]["candidate"],
                "MedSA_rho": float(spearmanr(values, actual)[0]),
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "ground_truth_used_for_post_hoc_score": True,
        "not_available_to_strict_attack": True,
        "rows": rows,
        "within_case_rankings": rankings,
    }, indent=2))
    print(json.dumps(rankings, indent=2))


if __name__ == "__main__":
    main()
