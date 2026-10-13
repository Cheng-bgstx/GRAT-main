#!/usr/bin/env python3
"""Fuse source-only path endpoints with the L-infinity support optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-weights", default="0.25,0.5,0.75")
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean = handle["image"][args.slice].astype(np.float32)
    left = load_gray(Path(args.left))
    right = load_gray(Path(args.right))
    directions = [left - clean, right - clean]
    directions = [value / max(float(np.mean(np.abs(value))), 1e-12) for value in directions]
    epsilon = args.epsilon / 255.0
    lower = np.ceil(np.maximum(0.0, clean - epsilon) * 255.0).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean + epsilon) * 255.0).astype(np.int16)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for left_weight in (float(value) for value in args.left_weights.split(",")):
        right_weight = 1.0 - left_weight
        fused = np.clip(clean + epsilon * np.sign(
            left_weight * directions[0] + right_weight * directions[1]
        ), 0.0, 1.0)
        quantized = np.rint(fused * 255.0).astype(np.int16)
        quantized = np.clip(quantized, lower, upper).astype(np.uint8)
        name = f"left_{left_weight:.3f}.png"
        Image.fromarray(quantized).save(output / name)
        records.append({
            "image": name,
            "left_weight": left_weight,
            "right_weight": right_weight,
            "linf_255": float(np.max(np.abs(quantized.astype(np.float32) / 255.0 - clean)) * 255.0),
        })
    with open(output / "fusion_meta.json", "w", encoding="utf-8") as handle:
        json.dump({"source_only": True, "records": records}, handle, indent=2)


if __name__ == "__main__":
    main()
