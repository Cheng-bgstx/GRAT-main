#!/usr/bin/env python3
"""Audit Synapse PNG candidates against the clean float32 H5 volume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--candidates", nargs="+", required=True,
                        help="NAME=directory containing 0.png, 1.png, ...")
    parser.add_argument("--epsilon", type=float, default=10.0,
                        help="L-infinity budget in 0-255 units")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    clean_path = ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5"
    with h5py.File(clean_path, "r") as handle:
        clean = handle["image"][:].astype(np.float32)
    if clean.min() < 0.0 or clean.max() > 1.0:
        raise ValueError(f"clean volume violates [0,1]: [{clean.min()}, {clean.max()}]")

    report = {
        "case": args.case,
        "input_scale": "float32_[0,1]",
        "epsilon_255": args.epsilon,
        "clean_range": [float(clean.min()), float(clean.max())],
        "candidates": {},
    }
    for spec in args.candidates:
        name, folder_value = spec.split("=", 1)
        folder = ROOT / folder_value
        paths = [folder / f"{index}.png" for index in range(len(clean))]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{name}: missing {len(missing)} slices; first {missing[0]}")
        volume = np.stack([
            np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            for path in paths
        ])
        distance = np.abs(volume - clean) * 255.0
        report["candidates"][name] = {
            "range": [float(volume.min()), float(volume.max())],
            "linf_255": float(distance.max()),
            "mean_l1_255": float(distance.mean()),
            "pixels_over_budget": int(np.count_nonzero(distance > args.epsilon + 1e-4)),
            "slice_count": len(paths),
        }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
