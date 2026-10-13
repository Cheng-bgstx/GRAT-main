#!/usr/bin/env python3
"""Convexly blend two source-only attack perturbations within the same budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from peft_transfer.generate import ROOT


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-weights", default="0.25,0.5,0.75")
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean = handle["image"][args.slice].astype(np.float32)
    left = load_gray(ROOT / args.left)
    right = load_gray(ROOT / args.right)
    epsilon = args.epsilon / 255.0
    lower = np.ceil(np.maximum(0.0, clean - epsilon) * 255.0).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean + epsilon) * 255.0).astype(np.int16)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for weight in map(float, args.left_weights.split(",")):
        perturbation = weight * (left - clean) + (1.0 - weight) * (right - clean)
        value = np.clip(clean + perturbation, 0.0, 1.0)
        quantized = np.clip(np.rint(value * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
        name = f"left_{weight:.3f}.png"
        Image.fromarray(quantized).save(output / name)
        records.append({
            "name": name, "left_weight": weight,
            "linf_255": float(np.max(np.abs(quantized.astype(np.float32) / 255.0 - clean)) * 255.0),
            "mean_abs_255": float(np.mean(np.abs(quantized.astype(np.float32) / 255.0 - clean)) * 255.0),
        })
    (output / "blend.json").write_text(json.dumps({"source_only": True, "records": records}, indent=2))


if __name__ == "__main__":
    main()
