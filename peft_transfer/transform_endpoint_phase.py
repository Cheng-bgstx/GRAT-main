#!/usr/bin/env python3
"""Apply controlled Fourier phase transforms to a quantized attack endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from peft_transfer.umi_phase_amplitude import phase_amplitude_priors


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=10.0)
    args = parser.parse_args()
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean = handle["image"][args.slice].astype(np.float32)
    endpoint = np.asarray(Image.open(ROOT / args.endpoint).convert("L"), dtype=np.float32) / 255.0
    delta = torch.from_numpy(endpoint - clean)[None, None]
    transformed = phase_amplitude_priors(delta, args.seed)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    clean_255 = clean * 255.0
    lower = np.ceil(np.maximum(0.0, clean_255 - args.epsilon)).astype(np.int16)
    upper = np.floor(np.minimum(255.0, clean_255 + args.epsilon)).astype(np.int16)
    records = []
    for name, candidate_delta in transformed.items():
        value = clean + candidate_delta[0, 0].numpy()
        quantized = np.clip(np.rint(value * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
        path = output / f"{name}.png"
        Image.fromarray(quantized).save(path)
        records.append({
            "name": name, "path": str(path.relative_to(ROOT)),
            "linf_255": float(np.max(np.abs(quantized.astype(np.float32) - clean_255))),
            "delta_rms_255": float(np.sqrt(np.mean((quantized.astype(np.float32) - clean_255) ** 2))),
        })
    (output / "transforms.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
