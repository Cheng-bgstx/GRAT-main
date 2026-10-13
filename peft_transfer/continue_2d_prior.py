#!/usr/bin/env python3
"""Continue saved 2D perturbation priors on the frozen base SAM encoder."""
from __future__ import annotations

import argparse
import json
from argparse import Namespace

import numpy as np
import torch
from torchvision import transforms

from datasets.isic import ISIC2016
from datasets.refuge import REFUGE
from peft_transfer.core import ablation_config, attack_image
from peft_transfer.evaluate_official_slice import ROOT, SAM
from peft_transfer.generate_2d_checkpoint_proxy import save_candidate
from segment_anything import sam_model_registry


DATASETS = {
    "ISIC": (ISIC2016, ROOT / "data/ISIC", 1),
    "REFUGE": (REFUGE, ROOT / "data/REFUGE-Multirater", 2),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--prior-folder", required=True)
    parser.add_argument("--indices", default="0,1,2,3")
    parser.add_argument("--scales", type=float, nargs="+", default=(0.25, 0.375, 0.5))
    parser.add_argument("--guidances", type=float, nargs="+", default=(0.5, 1.0))
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset_cls, data_path, classes = DATASETS[args.dataset]
    image_tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    mask_tf = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = dataset_cls(
        args=Namespace(img_size=512), data_path=str(data_path), mode="Test",
        transform=image_tf, transform_msk=mask_tf,
    )
    source, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=classes, checkpoint=str(SAM[args.vit]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    encoder = source.image_encoder.to(device).eval()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    output = ROOT / args.out
    records = []
    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        sample = dataset[index]
        case = str(sample["image_meta_dict"]["filename_or_obj"])
        clean = sample["image"].float()[None].to(device)
        prior_adv = torch.from_numpy(
            np.transpose(np.load(ROOT / args.prior_folder / f"{case}.npy"), (2, 0, 1))
        )[None].float().to(device)
        prior_delta = prior_adv - clean
        for scale in args.scales:
            for guidance in args.guidances:
                cfg = ablation_config(
                    "CH", epsilon=10 / 255, step_size=2 / 255, steps=10, seed=42,
                    prior_guidance_weight=guidance,
                )
                adv, trace = attack_image(
                    encoder, clean, cfg, preprocess=normalize,
                    initial_delta=scale * prior_delta, first_step_multiplier=2.5,
                )
                candidate = f"R{int(round(scale * 1000)):04d}_G{int(round(guidance * 100)):03d}"
                record = {
                    "index": index, "case": case, "candidate": candidate,
                    "prior_folder": args.prior_folder,
                    "prior_scale": scale, "prior_guidance_weight": guidance,
                    "linf_255": save_candidate(output / candidate, case, adv, clean),
                    "trace": {key: values[-1] for key, values in trace.items()},
                }
                records.append(record)
                print(json.dumps(record, indent=2), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "generation.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
