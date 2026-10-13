#!/usr/bin/env python3
"""Project legacy matplotlib Synapse PNGs into an exact integer L-inf budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slices", type=int, nargs="+", required=True)
    parser.add_argument("--source-template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epsilon", type=float, default=10.0)
    args = parser.parse_args()
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        for index in args.slices:
            clean = handle["image"][index].astype(np.float32)
            source = np.asarray(
                Image.open(ROOT / args.source_template.format(slice=index)).convert("L"),
                dtype=np.float32,
            )
            lower = np.ceil(np.maximum(0.0, clean * 255.0 - args.epsilon)).astype(np.int16)
            upper = np.floor(np.minimum(255.0, clean * 255.0 + args.epsilon)).astype(np.int16)
            value = np.clip(np.rint(source).astype(np.int16), lower, upper).astype(np.uint8)
            path = output / f"{index}.png"
            Image.fromarray(value).save(path)
            records.append({
                "slice": index, "path": str(path.relative_to(ROOT)),
                "linf_255": float(np.max(np.abs(value.astype(np.float32) - clean * 255.0))),
            })
    (output / "requantize.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
