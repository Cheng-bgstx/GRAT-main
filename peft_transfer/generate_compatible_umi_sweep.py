#!/usr/bin/env python3
"""Run UMI-PASD on the adapter-free MedSA architecture-compatible base."""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch
from PIL import Image

from peft_transfer.core import ablation_config, attack_image
from peft_transfer.generate import CHECKPOINTS, ROOT
from peft_transfer.medsa_surrogate import IntermediateEncoderWrapper, load_medsa_base_surrogate
from peft_transfer.umi_phase_amplitude import phase_amplitude_priors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--variants", nargs="+", default=["AF", "AG", "AH"])
    parser.add_argument("--prior-names", nargs="+", default=["original"])
    parser.add_argument("--prior", default="common_perb/common_perturbation_meta_0.025sp_bound2_it7batchit41upsize_ls0.0025mulmd0_10000.pth")
    parser.add_argument("--out", default="testset/compatible_umi_sweep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device(args.device)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean_np = handle["image"][args.slice].astype(np.float32)
    image = torch.from_numpy(clean_np)[None, None].repeat(1, 3, 1, 1).to(device)
    prior = torch.load(ROOT / args.prior, map_location=device)
    all_priors = phase_amplitude_priors(prior, args.seed)

    model = load_medsa_base_surrogate("vit_b", str(CHECKPOINTS["vit_b"]), 512, 9, device)
    encoder = IntermediateEncoderWrapper(model.image_encoder).to(device).eval()
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for prior_name in args.prior_names:
        for variant in args.variants:
            cfg = ablation_config(variant, epsilon=10 / 255, step_size=2 / 255, steps=10, seed=args.seed)
            adv, trace = attack_image(
                encoder, image, cfg, preprocess=model.preprocess,
                initial_delta=all_priors[prior_name], first_step_multiplier=2.5,
            )
            value = adv[0, 0].detach().cpu().numpy()
            lower = np.ceil(np.maximum(0.0, clean_np - cfg.epsilon) * 255.0).astype(np.int16)
            upper = np.floor(np.minimum(1.0, clean_np + cfg.epsilon) * 255.0).astype(np.int16)
            quantized = np.clip(np.rint(value * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
            name = f"{prior_name}_{variant}"
            path = output / f"{name}.png"
            Image.fromarray(quantized).save(path)
            record = {
                "name": name, "path": str(path.relative_to(ROOT)),
                "linf_255": float(np.max(np.abs(quantized.astype(np.float32) - clean_np * 255.0))),
                "trace": {key: values[-1] for key, values in trace.items()},
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    (output / "sweep.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
