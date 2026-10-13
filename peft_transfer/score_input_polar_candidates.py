#!/usr/bin/env python3
"""Audit input Fourier amplitude/phase scores on frozen attack candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from peft_transfer.evaluate_synapse_slice import ROOT


def polar_scores(clean: np.ndarray, attacked: np.ndarray) -> dict[str, float]:
    first = np.fft.rfft2(clean, norm="ortho")
    second = np.fft.rfft2(attacked, norm="ortho")
    first_amp = np.abs(first)
    second_amp = np.abs(second)
    first_unit = first / np.maximum(first_amp, 1e-10)
    second_unit = second / np.maximum(second_amp, 1e-10)
    scale = np.linalg.norm(first_amp).clip(1e-10)
    return {
        "amplitude": float(np.linalg.norm(second_amp - first_amp) / scale),
        "phase": float(np.linalg.norm(first_amp * (second_unit - first_unit)) / scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.prediction).read_text())["rows"]
    clean_cache = {}
    for row in rows:
        key = (row["case"], row["slice"])
        if key not in clean_cache:
            with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{key[0]}.npy.h5") as handle:
                clean_cache[key] = handle["image"][key[1]].astype(np.float64)
        attacked = np.asarray(Image.open(ROOT / row["candidate"]).convert("L"),
                              dtype=np.float64) / 255.0
        row["input_polar"] = polar_scores(clean_cache[key], attacked)
    rankings = {}
    for key in clean_cache:
        subset = [row for row in rows if (row["case"], row["slice"]) == key]
        if len(subset) < 2:
            continue
        case_key = f"{key[0]}/slice{key[1]}"
        rankings[case_key] = {}
        for score in ("amplitude", "phase"):
            values = [row["input_polar"][score] for row in subset]
            actual = [row["MedSA_dice_drop"] for row in subset]
            rankings[case_key][score] = {
                "predicted_best": subset[int(np.argmax(values))]["candidate"],
                "MedSA_best": subset[int(np.argmax(actual))]["candidate"],
                "MedSA_rho": float(spearmanr(values, actual)[0]),
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "source_only_score": True,
        "target_metrics_used_for_post_hoc_ranking_only": True,
        "rows": rows,
        "within_case_rankings": rankings,
    }, indent=2))
    print(json.dumps(rankings, indent=2))


if __name__ == "__main__":
    main()
