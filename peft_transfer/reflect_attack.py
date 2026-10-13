#!/usr/bin/env python3
"""Reflect a frozen perturbation around its clean image for a sign diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from peft_transfer.evaluate_synapse_slice import ROOT


def reflect(clean: np.ndarray, attacked: np.ndarray, epsilon: int) -> np.ndarray:
    clean_pixels = clean.astype(np.float64) * 255.0
    attacked_pixels = attacked.astype(np.float64)
    tolerance = 1e-4
    if np.max(np.abs(attacked_pixels - clean_pixels)) > epsilon + tolerance:
        raise ValueError("Input attack exceeds the requested exact pixel budget")
    lower = np.ceil(np.maximum(0.0, clean_pixels - epsilon - tolerance)).astype(np.int32)
    upper = np.floor(np.minimum(255.0, clean_pixels + epsilon + tolerance)).astype(np.int32)
    reflected = np.rint(2.0 * clean_pixels - attacked_pixels).astype(np.int32)
    output = np.clip(reflected, lower, upper).astype(np.uint8)
    if np.max(np.abs(output.astype(np.float64) - clean_pixels)) > epsilon + tolerance:
        raise AssertionError("Reflected attack exceeds the requested pixel budget")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epsilon", type=int, default=10)
    args = parser.parse_args()
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5") as handle:
        clean = handle["image"][args.slice]
    attacked = np.asarray(Image.open(ROOT / args.candidate).convert("L"), dtype=np.uint8)
    if attacked.shape != clean.shape:
        raise ValueError("Candidate and clean image shapes differ")
    output = reflect(clean, attacked, args.epsilon)
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(path)
    print(f"{path}: max deviation {np.max(np.abs(output.astype(float)-255*clean)):.6f}/255")


if __name__ == "__main__":
    main()
