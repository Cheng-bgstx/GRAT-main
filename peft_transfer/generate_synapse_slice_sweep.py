#!/usr/bin/env python3
"""Generate several Synapse slice candidates with one source-model load."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from peft_transfer.core import ablation_config, attack_image
from peft_transfer.generate import CHECKPOINTS, ROOT
from segment_anything import sam_model_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=CHECKPOINTS, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--configs", nargs="+", required=True,
                        help=("NAME:VARIANT:PRIOR_SCALE:FIRST_STEP_MULTIPLIER"
                              "[:PRIOR_GUIDANCE_WEIGHT[:SPECTRAL_FLOOR:SPECTRAL_WEIGHT]]"))
    parser.add_argument("--prior", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--step-size", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)

    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        image_np = handle["image"][args.slice].astype(np.float32)
    image = torch.from_numpy(image_np)[None, None].repeat(1, 3, 1, 1).to(device)
    model, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=8, checkpoint=str(CHECKPOINTS[args.vit]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    encoder = model.image_encoder.to(device).eval()
    prior = torch.load(args.prior, map_location=device) if args.prior else None
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in args.configs:
        fields = spec.split(":")
        if len(fields) not in {4, 5, 7}:
            raise ValueError(f"invalid config {spec!r}")
        name, variant, prior_scale, first_multiplier = fields[:4]
        guidance_weight = float(fields[4]) if len(fields) == 5 else 0.0
        spectral_floor = float(fields[5]) if len(fields) == 7 else -1.0
        spectral_weight = float(fields[6]) if len(fields) == 7 else 1.0
        cfg = ablation_config(
            variant, epsilon=args.epsilon / 255.0,
            step_size=args.step_size / 255.0, steps=args.steps, seed=args.seed,
            prior_guidance_weight=guidance_weight,
            prior_spectral_floor=spectral_floor,
            prior_spectral_weight=spectral_weight,
        )
        initial = None if prior is None else prior * float(prior_scale)
        adv, trace = attack_image(
            encoder, image, cfg,
            preprocess=lambda value: value[:, :1].repeat(1, 3, 1, 1),
            initial_delta=initial, first_step_multiplier=float(first_multiplier),
        )
        clean = image[0, 0].detach().cpu().numpy()
        value = adv[0, 0].detach().cpu().numpy()
        lower = np.ceil(np.maximum(0.0, clean - cfg.epsilon) * 255.0).astype(np.int16)
        upper = np.floor(np.minimum(1.0, clean + cfg.epsilon) * 255.0).astype(np.int16)
        quantized = np.clip(np.rint(value * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
        path = output / f"{name}.png"
        Image.fromarray(quantized).save(path)
        records.append({
            "name": name, "variant": variant, "prior_scale": float(prior_scale),
            "first_step_multiplier": float(first_multiplier),
            "prior_guidance_weight": guidance_weight,
            "prior_spectral_floor": spectral_floor,
            "prior_spectral_weight": spectral_weight,
            "path": str(path.relative_to(ROOT)),
            "linf_255": float(np.max(np.abs(quantized.astype(np.float32) - clean * 255.0))),
            "trace": {key: values[-1] for key, values in trace.items()},
            "trajectory": trace,
        })
        print(json.dumps(records[-1], indent=2), flush=True)
    (output / "sweep.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
