#!/usr/bin/env python3
"""Generate UMI-AG candidates from controlled phase/amplitude priors."""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch
from PIL import Image

from peft_transfer.core import ablation_config, attack_image
from peft_transfer.generate import CHECKPOINTS, ROOT
from peft_transfer.umi_phase_amplitude import phase_amplitude_priors
from segment_anything import sam_model_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument(
        "--prior",
        default="common_perb/common_perturbation_meta_0.025sp_bound2_it7batchit41upsize_ls0.0025mulmd0_10000.pth",
    )
    parser.add_argument("--out", default="testset/umi_phase_amplitude_sweep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--names", nargs="+", default=None,
                        help="optional subset of transformed prior names")
    args = parser.parse_args()
    device = torch.device(args.device)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean_np = handle["image"][args.slice].astype(np.float32)
    image = torch.from_numpy(clean_np)[None, None].repeat(1, 3, 1, 1).to(device)
    prior = torch.load(ROOT / args.prior, map_location=device)
    priors = phase_amplitude_priors(prior, args.seed)
    if args.names is not None:
        missing = sorted(set(args.names) - set(priors))
        if missing:
            raise ValueError(f"unknown prior variants: {missing}")
        priors = {name: priors[name] for name in args.names}

    model, _ = sam_model_registry["vit_b"](
        image_size=512, num_classes=8, checkpoint=str(CHECKPOINTS["vit_b"]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    encoder = model.image_encoder.to(device).eval()
    cfg = ablation_config("AG", epsilon=10 / 255, step_size=2 / 255, steps=10, seed=args.seed)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, transformed in priors.items():
        adv, trace = attack_image(
            encoder, image, cfg,
            preprocess=lambda value: value[:, :1].repeat(1, 3, 1, 1),
            initial_delta=transformed, first_step_multiplier=2.5,
        )
        value = adv[0, 0].detach().cpu().numpy()
        lower = np.ceil(np.maximum(0.0, clean_np - cfg.epsilon) * 255.0).astype(np.int16)
        upper = np.floor(np.minimum(1.0, clean_np + cfg.epsilon) * 255.0).astype(np.int16)
        quantized = np.clip(np.rint(value * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
        path = output / f"{name}.png"
        Image.fromarray(quantized).save(path)
        transformed_cpu = transformed.detach().cpu()
        record = {
            "name": name, "path": str(path.relative_to(ROOT)),
            "prior_rms": float(transformed_cpu.square().mean().sqrt()),
            "prior_peak": float(transformed_cpu.abs().max()),
            "linf_255": float(np.max(np.abs(quantized.astype(np.float32) - clean_np * 255.0))),
            "trace": {key: values[-1] for key, values in trace.items()},
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (output / "sweep.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
