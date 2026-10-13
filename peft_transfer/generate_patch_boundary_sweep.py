#!/usr/bin/env python3
"""Generate UMI-PASD attacks with frozen patch-boundary disruption."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

import h5py
import numpy as np
import torch
from PIL import Image

from peft_transfer.core import ablation_config, attack_image
from peft_transfer.generate import CHECKPOINTS, ROOT
from segment_anything import sam_model_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.05, 0.1, 0.25])
    parser.add_argument("--patch-only", action="store_true")
    parser.add_argument(
        "--prior",
        default="common_perb/common_perturbation_meta_0.025sp_bound2_it7batchit41upsize_ls0.0025mulmd0_10000.pth",
    )
    parser.add_argument("--out", default="testset/patch_boundary_sweep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device(args.device)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean_np = handle["image"][args.slice].astype(np.float32)
    image = torch.from_numpy(clean_np)[None, None].repeat(1, 3, 1, 1).to(device)
    prior = torch.load(ROOT / args.prior, map_location=device).float()
    model, _ = sam_model_registry["vit_b"](
        image_size=512, num_classes=8, checkpoint=str(CHECKPOINTS["vit_b"]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    encoder = model.image_encoder.to(device).eval()
    preprocess = lambda value: value[:, :1].repeat(1, 3, 1, 1)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for weight in args.weights:
        cfg = ablation_config(
            "AG", epsilon=10 / 255, step_size=2 / 255, steps=10, seed=args.seed,
            use_patch_boundary=True, patch_boundary_weight=weight,
        )
        if args.patch_only:
            cfg = replace(cfg, use_pasd=False, output_weight=0.0, use_edge=False)
        adv, trace = attack_image(
            encoder, image, cfg, preprocess=preprocess,
            initial_delta=prior, first_step_multiplier=2.5,
        )
        value = adv[0, 0].detach().cpu().numpy()
        lower = np.ceil(np.maximum(0.0, clean_np - cfg.epsilon) * 255.0).astype(np.int16)
        upper = np.floor(np.minimum(1.0, clean_np + cfg.epsilon) * 255.0).astype(np.int16)
        quantized = np.clip(np.rint(value * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
        prefix = "patch_only" if args.patch_only else "umi_pasd_patch"
        name = f"{prefix}_w{weight:g}".replace(".", "p")
        path = output / f"{name}.png"
        Image.fromarray(quantized).save(path)
        record = {
            "name": name, "weight": weight, "path": str(path.relative_to(ROOT)),
            "linf_255": float(np.max(np.abs(quantized.astype(np.float32) - clean_np * 255.0))),
            "trace": {key: values[-1] for key, values in trace.items()},
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (output / "sweep.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
