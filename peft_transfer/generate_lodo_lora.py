#!/usr/bin/env python3
"""Generate a Synapse attack from non-Synapse LoRA encoder proxies."""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from peft_transfer.core import _asymmetric_step, _unpack_encoder, ablation_config, attack_image
from peft_transfer.evaluate_official_slice import ROOT, SAM, load_samed_model
from segment_anything import sam_model_registry


def save_feasible(path, adv, clean):
    clean_np = clean[0, 0].cpu().numpy()
    value = adv[0, 0].cpu().numpy()
    lower = np.ceil(np.maximum(0.0, clean_np - 10 / 255) * 255).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_np + 10 / 255) * 255).astype(np.int16)
    quantized = np.clip(np.rint(value * 255).astype(np.int16), lower, upper).astype(np.uint8)
    Image.fromarray(quantized).save(path)
    return float(np.max(np.abs(quantized.astype(np.float32) - clean_np * 255)))


def final_feature(encoder, image):
    return _unpack_encoder(encoder(image))[0]


def proxy_attack(encoders, clean):
    image = clean.repeat(1, 3, 1, 1)
    with torch.no_grad():
        references = [final_feature(encoder, image).detach() for encoder in encoders]
    adv = image.clone()
    momentum = torch.zeros_like(adv)
    trace = []
    for _ in range(10):
        adv = adv.detach().requires_grad_(True)
        gradients, losses = [], []
        for encoder, reference in zip(encoders, references):
            loss = F.l1_loss(final_feature(encoder, adv) * 0.9999, reference)
            gradient = torch.autograd.grad(loss, adv)[0]
            gradients.append(gradient / gradient.abs().mean().clamp_min(1e-12))
            losses.append(float(loss.detach()))
        gradient = torch.stack(gradients).mean(0)
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = 0.5 * momentum + gradient
        adv = adv.detach() + _asymmetric_step(momentum, 2 / 255)
        adv = torch.maximum(torch.minimum(adv, image + 10 / 255), image - 10 / 255).clamp(0, 1)
        trace.append({
            "losses": losses,
            "gradient_cosine": float(F.cosine_similarity(
                gradients[0].flatten(1), gradients[1].flatten(1), dim=1
            ).mean()),
        })
    return adv, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--guidance", type=float, nargs="+", default=[0.5, 1, 2])
    parser.add_argument("--initial-delta-scale", type=float, default=0.5)
    parser.add_argument("--first-step-multiplier", type=float, default=2.5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean = torch.from_numpy(handle["image"][args.slice].astype(np.float32))[None, None].to(device)
    proxy_specs = (
        ("ISIC", 1, "output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"),
        ("REFUGE", 2, "output_runs_correct/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"),
    )
    models = [
        load_samed_model(args.vit, str(SAM[args.vit]), str(ROOT / path), 512, classes, device=device)
        for _dataset, classes, path in proxy_specs
    ]
    encoders = [model.sam.image_encoder for model in models]
    proxy_adv, proxy_trace = proxy_attack(encoders, clean)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    proxy_path = output / "LODO_LORA_PROXY.png"
    records = [{
        "name": "LODO_LORA_PROXY", "path": str(proxy_path.relative_to(ROOT)),
        "linf_255": save_feasible(proxy_path, proxy_adv, clean),
        "held_out_dataset": "Synapse", "held_out_checkpoint_used": False,
        "proxy_datasets": [spec[0] for spec in proxy_specs], "trace": proxy_trace,
    }]
    initial_delta = args.initial_delta_scale * (proxy_adv - clean.repeat(1, 3, 1, 1))
    del models, encoders
    torch.cuda.empty_cache()
    source, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=8, checkpoint=str(SAM[args.vit]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    encoder = source.image_encoder.to(device).eval()
    image = clean.repeat(1, 3, 1, 1)
    for guidance in args.guidance:
        cfg = ablation_config(
            "CH", epsilon=10 / 255, step_size=2 / 255, steps=10,
            seed=42, prior_guidance_weight=guidance,
        )
        adv, trace = attack_image(
            encoder, image, cfg,
            preprocess=lambda value: value[:, :1].repeat(1, 3, 1, 1),
            initial_delta=initial_delta,
            first_step_multiplier=args.first_step_multiplier,
        )
        name = f"LODO_LORA_CH_G{int(guidance * 100):03d}"
        path = output / f"{name}.png"
        records.append({
            "name": name, "path": str(path.relative_to(ROOT)),
            "linf_255": save_feasible(path, adv, clean),
            "held_out_dataset": "Synapse", "held_out_checkpoint_used": False,
            "proxy_datasets": [spec[0] for spec in proxy_specs],
            "initial_delta_scale": args.initial_delta_scale,
            "first_step_multiplier": args.first_step_multiplier,
            "prior_guidance_weight": guidance,
            "trace": {key: values[-1] for key, values in trace.items()},
        })
        print(json.dumps(records[-1], indent=2), flush=True)
    (output / "generation.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
