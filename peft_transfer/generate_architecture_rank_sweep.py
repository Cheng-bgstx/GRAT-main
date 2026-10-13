#!/usr/bin/env python3
"""Generate a small first-adapter rank-tail sweep on one Synapse slice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from peft_transfer.architecture_rank_attack import (
    attack_architecture_embedding,
    attack_architecture_rank_tail,
)
from peft_transfer.core import ablation_config, attack_image
from peft_transfer.generate import CHECKPOINTS, ROOT
from peft_transfer.medsa_surrogate import load_medsa_base_surrogate
from segment_anything import sam_model_registry


def write_exact(path, adv, clean, epsilon):
    value = adv[0, 0].detach().cpu().numpy()
    lower = np.ceil(np.maximum(0.0, clean - epsilon) * 255.0).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean + epsilon) * 255.0).astype(np.int16)
    quantized = np.clip(np.rint(value * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
    Image.fromarray(quantized).save(path)
    return float(np.max(np.abs(quantized.astype(np.float32) - clean * 255.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--specs", nargs="+", default=[
        "AR-P025:plain:0.25", "AR-P05:plain:0.5", "AR-P1:plain:1.0",
        "AR-H05:haar:0.5", "AR-H1:haar:1.0",
    ])
    parser.add_argument("--out", default="testset/architecture_rank_sweep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective", choices=("rank", "embedding"), default="rank")
    args = parser.parse_args()
    device = torch.device(args.device)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean_np = handle["image"][args.slice].astype(np.float32)
    image = torch.from_numpy(clean_np)[None, None].repeat(1, 3, 1, 1).to(device)

    canonical_model, _ = sam_model_registry["vit_b"](
        image_size=512, num_classes=8, checkpoint=str(CHECKPOINTS["vit_b"]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    canonical_encoder = canonical_model.image_encoder.to(device).eval()
    compatible = load_medsa_base_surrogate(
        "vit_b", str(CHECKPOINTS["vit_b"]), 512, 9, device
    )
    cfg = ablation_config(
        "CK", epsilon=10 / 255, step_size=2 / 255, steps=10, seed=args.seed
    )
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    records = []

    control, control_trace = attack_image(
        canonical_encoder, image, cfg,
        preprocess=lambda value: value[:, :1].repeat(1, 3, 1, 1),
        first_step_multiplier=1.0,
    )
    control_path = output / "CK.png"
    control_linf = write_exact(control_path, control, clean_np, cfg.epsilon)
    records.append({
        "name": "CK", "mode": "control", "weight": 0.0,
        "path": str(control_path.relative_to(ROOT)), "linf_255": control_linf,
        "trace": {key: values[-1] for key, values in control_trace.items()},
    })
    print(json.dumps(records[-1]), flush=True)

    for spec in args.specs:
        name, mode, weight = spec.split(":")
        if args.objective == "rank":
            adv, trace = attack_architecture_rank_tail(
                canonical_encoder, compatible.image_encoder, compatible.preprocess,
                image, cfg, float(weight), rank=192, mode=mode,
            )
        else:
            adv, trace = attack_architecture_embedding(
                canonical_encoder, compatible.image_encoder, compatible.preprocess,
                image, cfg, float(weight), mode=mode,
            )
        path = output / f"{name}.png"
        linf = write_exact(path, adv, clean_np, cfg.epsilon)
        records.append({
            "name": name, "mode": mode, "weight": float(weight),
            "path": str(path.relative_to(ROOT)), "linf_255": linf,
            "trace": {key: values[-1] for key, values in trace.items()},
        })
        print(json.dumps(records[-1]), flush=True)
    (output / "sweep.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
