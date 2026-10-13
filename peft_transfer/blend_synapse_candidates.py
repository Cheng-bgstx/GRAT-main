#!/usr/bin/env python3
"""Fuse two source-only Synapse attacks under the exact shared L-infinity budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from peft_transfer.generate_synapse_dual_peft import ROOT, save_grayscale


def load_gray(path: str):
    value = np.asarray(Image.open(ROOT / path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(value)[None, None].repeat(1, 3, 1, 1)


def fuse(clean, source, semantic, alpha, mode, epsilon=10 / 255):
    source_delta = source - clean
    semantic_delta = semantic - clean
    if mode == "linear":
        delta = alpha * source_delta + (1.0 - alpha) * semantic_delta
    elif mode == "steepest":
        source_scale = source_delta.abs().mean().clamp_min(1e-12)
        semantic_scale = semantic_delta.abs().mean().clamp_min(1e-12)
        direction = alpha * source_delta / source_scale
        direction += (1.0 - alpha) * semantic_delta / semantic_scale
        delta = epsilon * direction.sign()
    else:
        raise ValueError(f"unknown fusion mode: {mode}")
    return (clean + delta.clamp(-epsilon, epsilon)).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    parser.add_argument("--modes", choices=("linear", "steepest"), nargs="+",
                        default=("linear", "steepest"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean_np = handle["image"][args.slice].astype(np.float32)
    clean = torch.from_numpy(clean_np)[None, None].repeat(1, 3, 1, 1)
    source = load_gray(args.source)
    semantic = load_gray(args.semantic)
    rows = []
    for mode in args.modes:
        for alpha in args.alphas:
            adv = fuse(clean, source, semantic, alpha, mode)
            name = f"{mode.upper()}_A{int(round(100 * alpha)):03d}"
            path = ROOT / args.out / name / f"{args.slice}.png"
            rows.append({
                "candidate": name,
                "source": args.source,
                "semantic": args.semantic,
                "source_weight": alpha,
                "mode": mode,
                "target_checkpoints_used": False,
                "linf_255": save_grayscale(path, adv, clean),
            })
    output = ROOT / args.out / "generation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
