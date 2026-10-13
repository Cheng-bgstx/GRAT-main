#!/usr/bin/env python3
"""Compose an attack with an amplitude-preserving translation phase carrier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from peft_transfer.evaluate_official_slice import ROOT


def phase_translation_carrier(clean: torch.Tensor, shift_fraction: float, epsilon: float):
    """Return a bounded direction toward a Fourier linear-phase translation."""
    if not 0 < shift_fraction <= 0.5:
        raise ValueError("shift fraction must be in (0, 0.5]")
    shifts = tuple(max(1, int(round(size * shift_fraction))) for size in clean.shape[-2:])
    translated = torch.roll(clean, shifts=shifts, dims=(-2, -1))
    return epsilon * (translated - clean).sign()


def compose(clean: torch.Tensor, attack: torch.Tensor, weight: float,
            shift_fraction: float, epsilon: float):
    delta = attack - clean
    carrier = phase_translation_carrier(clean, shift_fraction, epsilon)
    mixed = delta + float(weight) * carrier
    mixed = mixed.clamp(-epsilon, epsilon)
    return (clean + mixed).clamp(0.0, 1.0)


def quantize_budget(adv: torch.Tensor, clean: torch.Tensor, epsilon: float):
    clean_np = clean.cpu().numpy()
    adv_np = adv.cpu().numpy()
    lower = np.ceil(np.maximum(0.0, clean_np - epsilon) * 255).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_np + epsilon) * 255).astype(np.int16)
    return np.clip(np.rint(adv_np * 255).astype(np.int16), lower, upper).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--slices", type=int, nargs="+", required=True)
    parser.add_argument("--attack", required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=(0.1, 0.25))
    parser.add_argument("--shift-fraction", type=float, default=0.25)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    epsilon = args.epsilon / 255.0
    rows = []
    with h5py.File(
        ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r"
    ) as handle:
        for slice_index in args.slices:
            clean = torch.from_numpy(handle["image"][slice_index].astype(np.float32))
            attack_path = ROOT / args.attack.format(slice=slice_index)
            attack = torch.from_numpy(
                np.asarray(Image.open(attack_path).convert("L"), dtype=np.float32) / 255.0
            )
            for weight in args.weights:
                adv = compose(clean, attack, weight, args.shift_fraction, epsilon)
                quantized = quantize_budget(adv, clean, epsilon)
                folder = ROOT / args.out / f"PHASE_TRANSLATION_W{int(round(weight * 100)):03d}"
                folder.mkdir(parents=True, exist_ok=True)
                output = folder / f"{slice_index}.png"
                Image.fromarray(quantized).save(output)
                linf = float(np.max(np.abs(quantized.astype(np.float32) / 255.0 - clean.numpy())))
                rows.append({
                    "case": args.case,
                    "slice": slice_index,
                    "weight": weight,
                    "shift_fraction": args.shift_fraction,
                    "linf_255": linf * 255.0,
                    "source": str(attack_path.relative_to(ROOT)),
                    "target_checkpoints_used": False,
                })
    report = ROOT / args.out / "composition.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
