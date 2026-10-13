#!/usr/bin/env python3
"""Generate 2D attacks from a held-out training-checkpoint encoder proxy."""
from __future__ import annotations

import argparse
import json
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from datasets.isic import ISIC2016
from datasets.refuge import REFUGE
from peft_transfer.core import _asymmetric_step, _unpack_encoder, ablation_config, attack_image
from peft_transfer.evaluate_official_slice import ROOT, SAM, load_samed_model
from segment_anything import sam_model_registry


DATASETS = {
    "ISIC": (ISIC2016, ROOT / "data/ISIC", 1),
    "REFUGE": (REFUGE, ROOT / "data/REFUGE-Multirater", 2),
}


def final_feature(encoder, image):
    return _unpack_encoder(encoder(image))[0]


def proxy_attack(encoder, clean, normalize):
    with torch.no_grad():
        reference = final_feature(encoder, normalize(clean)).detach()
    adv = clean.clone()
    momentum = torch.zeros_like(clean)
    trace = []
    for _ in range(10):
        adv = adv.detach().requires_grad_(True)
        loss = F.l1_loss(final_feature(encoder, normalize(adv)) * 0.9999, reference)
        gradient = torch.autograd.grad(loss, adv)[0]
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = 0.5 * momentum + gradient
        adv = adv.detach() + _asymmetric_step(momentum, 2 / 255)
        adv = torch.maximum(torch.minimum(adv, clean + 10 / 255), clean - 10 / 255).clamp(0, 1)
        trace.append(float(loss.detach()))
    return adv, trace


def save_candidate(folder, case, adv, clean):
    folder.mkdir(parents=True, exist_ok=True)
    chw = adv[0].detach().cpu().numpy().astype(np.float32)
    np.save(folder / f"{case}.npy", np.transpose(chw, (1, 2, 0)))
    clean_hwc = np.transpose(clean[0].cpu().numpy(), (1, 2, 0))
    value_hwc = np.transpose(chw, (1, 2, 0))
    lower = np.ceil(np.maximum(0.0, clean_hwc - 10 / 255) * 255).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_hwc + 10 / 255) * 255).astype(np.int16)
    quantized = np.clip(np.rint(value_hwc * 255).astype(np.int16), lower, upper).astype(np.uint8)
    Image.fromarray(quantized).save(folder / f"{case}.png")
    return float(np.max(np.abs(value_hwc - clean_hwc)) * 255.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--proxy-checkpoint", required=True)
    parser.add_argument("--indices", default="0,1,2,3")
    parser.add_argument("--initial-delta-scale", type=float, default=0.375)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--initial-delta-scales", type=float, nargs="+", default=None)
    parser.add_argument("--guidances", type=float, nargs="+", default=None)
    parser.add_argument(
        "--guidance-schedule",
        choices=("constant", "linear_decay", "linear_ramp", "early", "late"),
        default="constant",
    )
    parser.add_argument("--first-step-multiplier", type=float, default=2.5)
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
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    proxy_model = load_samed_model(
        args.vit, str(SAM[args.vit]), str(ROOT / args.proxy_checkpoint),
        512, classes, device=device,
    )
    proxy_encoder = proxy_model.sam.image_encoder
    source, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=classes, checkpoint=str(SAM[args.vit]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    source_encoder = source.image_encoder.to(device).eval()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    scales = args.initial_delta_scales or [args.initial_delta_scale]
    guidances = args.guidances or [args.guidance]
    output = ROOT / args.out
    records = []
    for index in indices:
        sample = dataset[index]
        case = str(sample["image_meta_dict"]["filename_or_obj"])
        clean = sample["image"].float()[None].to(device)
        proxy_adv, proxy_trace = proxy_attack(proxy_encoder, clean, normalize)
        proxy_record = {
            "index": index, "case": case,
            "target_checkpoint_used": False,
            "proxy_checkpoint": args.proxy_checkpoint,
            "proxy_linf_255": save_candidate(output / "PROXY", case, proxy_adv, clean),
            "proxy_trace": proxy_trace,
        }
        for scale in scales:
            for guidance in guidances:
                cfg = ablation_config(
                    "CH", epsilon=10 / 255, step_size=2 / 255, steps=10, seed=42,
                    prior_guidance_weight=guidance,
                    prior_guidance_schedule=args.guidance_schedule,
                )
                initial_delta = scale * (proxy_adv - clean)
                continuation, trace = attack_image(
                    source_encoder, clean, cfg, preprocess=normalize,
                    initial_delta=initial_delta,
                    first_step_multiplier=args.first_step_multiplier,
                )
                if len(scales) == len(guidances) == 1:
                    candidate = "CONT"
                else:
                    candidate = f"R{int(round(scale * 100)):03d}_G{int(round(guidance * 100)):03d}"
                record = {
                    **proxy_record,
                    "candidate": candidate,
                    "initial_delta_scale": scale,
                    "prior_guidance_weight": guidance,
                    "prior_guidance_schedule": args.guidance_schedule,
                    "continuation_linf_255": save_candidate(
                        output / candidate, case, continuation, clean
                    ),
                    "continuation_trace": {
                        key: values[-1] for key, values in trace.items()
                    },
                }
                records.append(record)
                print(json.dumps(record, indent=2), flush=True)
    (output / "generation.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
