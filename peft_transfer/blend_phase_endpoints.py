#!/usr/bin/env python3
"""Blend two quantized adversarial phase endpoints inside the exact budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def blend_quantized(clean, left, right, weight, epsilon_255=10.0):
    clean_255 = clean * 255.0
    delta = (1.0 - weight) * (left.astype(np.float32) - clean_255)
    delta += weight * (right.astype(np.float32) - clean_255)
    lower = np.ceil(np.maximum(0.0, clean_255 - epsilon_255)).astype(np.int16)
    upper = np.floor(np.minimum(255.0, clean_255 + epsilon_255)).astype(np.int16)
    return np.clip(np.rint(clean_255 + delta).astype(np.int16), lower, upper).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean = handle["image"][args.slice].astype(np.float32)
    left = np.asarray(Image.open(ROOT / args.left).convert("L"), dtype=np.uint8)
    right = np.asarray(Image.open(ROOT / args.right).convert("L"), dtype=np.uint8)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for weight in args.weights:
        value = blend_quantized(clean, left, right, weight)
        name = f"blend_{int(round(weight * 100)):02d}"
        path = output / f"{name}.png"
        Image.fromarray(value).save(path)
        records.append({
            "name": name, "weight": weight, "path": str(path.relative_to(ROOT)),
            "linf_255": float(np.max(np.abs(value.astype(np.float32) - clean * 255.0))),
        })
    (output / "blend.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
