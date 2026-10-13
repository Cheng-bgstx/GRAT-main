#!/usr/bin/env python3
"""Compose the Fourier amplitude and phase of two adversarial perturbations."""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
from PIL import Image


def circular_mix(left, right, weight):
    unit = (1.0 - weight) * np.exp(1j * left) + weight * np.exp(1j * right)
    return np.angle(unit)


def feasible_quantize(clean, delta, epsilon_255):
    clean_255 = clean * 255.0
    peak = float(np.max(np.abs(delta)))
    if peak > 0:
        delta = delta * (epsilon_255 / peak)
    lower = np.ceil(np.maximum(0.0, clean_255 - epsilon_255)).astype(np.int16)
    upper = np.floor(np.minimum(255.0, clean_255 + epsilon_255)).astype(np.int16)
    return np.clip(np.rint(clean_255 + delta).astype(np.int16), lower, upper).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--left", required=True, help="base-SAM endpoint")
    parser.add_argument("--right", required=True, help="PEFT-proxy endpoint")
    parser.add_argument("--weights", type=float, nargs="+", default=[0, .25, .5, .75, 1])
    parser.add_argument("--epsilon-255", type=float, default=10.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    with h5py.File(root / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean = handle["image"][args.slice].astype(np.float32)
    clean_255 = clean * 255.0
    left = np.asarray(Image.open(root / args.left).convert("L"), dtype=np.float32) - clean_255
    right = np.asarray(Image.open(root / args.right).convert("L"), dtype=np.float32) - clean_255
    left_spectrum = np.fft.rfft2(left, norm="ortho")
    right_spectrum = np.fft.rfft2(right, norm="ortho")
    left_amplitude, right_amplitude = np.abs(left_spectrum), np.abs(right_spectrum)
    left_phase, right_phase = np.angle(left_spectrum), np.angle(right_spectrum)
    output = root / args.out
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for amplitude_weight in args.weights:
        amplitude = (1.0 - amplitude_weight) * left_amplitude + amplitude_weight * right_amplitude
        for phase_weight in args.weights:
            phase = circular_mix(left_phase, right_phase, phase_weight)
            spectrum = amplitude * np.exp(1j * phase)
            delta = np.fft.irfft2(spectrum, s=clean.shape, norm="ortho").real
            value = feasible_quantize(clean, delta, args.epsilon_255)
            name = f"A{int(amplitude_weight * 100):03d}_P{int(phase_weight * 100):03d}"
            path = output / f"{name}.png"
            Image.fromarray(value).save(path)
            records.append({
                "name": name,
                "amplitude_proxy_weight": amplitude_weight,
                "phase_proxy_weight": phase_weight,
                "path": str(path.relative_to(root)),
                "linf_255": float(np.max(np.abs(value.astype(np.float32) - clean_255))),
                "target_checkpoint_used": False,
            })
    (output / "composition.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
