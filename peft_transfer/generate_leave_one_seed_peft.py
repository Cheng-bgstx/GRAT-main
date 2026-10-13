#!/usr/bin/env python3
"""Generate a leave-one-seed-out PEFT transfer attack for Synapse."""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from peft_transfer.core import _asymmetric_step, ablation_config, attack_image
from peft_transfer.evaluate_official_slice import ROOT, SAM, load_medsa_model
from segment_anything import sam_model_registry


def save_feasible(path, adv, clean):
    clean_np = clean[0, 0].cpu().numpy()
    value = adv[0, 0].cpu().numpy()
    lower = np.ceil(np.maximum(0.0, clean_np - 10 / 255) * 255).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_np + 10 / 255) * 255).astype(np.int16)
    quantized = np.clip(np.rint(value * 255).astype(np.int16), lower, upper).astype(np.uint8)
    Image.fromarray(quantized).save(path)
    return float(np.max(np.abs(quantized.astype(np.float32) - clean_np * 255)))


def attack_proxy_encoder(model, clean):
    with torch.no_grad():
        reference = model.image_encoder(model.preprocess(clean.repeat(1, 3, 1, 1)))
    adv = clean.clone()
    momentum = torch.zeros_like(clean)
    trace = []
    for _ in range(10):
        adv = adv.detach().requires_grad_(True)
        embedding = model.image_encoder(model.preprocess(adv.repeat(1, 3, 1, 1)))
        loss = F.l1_loss(embedding * 0.9999, reference)
        gradient = torch.autograd.grad(loss, adv)[0]
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = 0.5 * momentum + gradient
        adv = adv.detach() + _asymmetric_step(momentum, 2 / 255)
        adv = torch.maximum(torch.minimum(adv, clean + 10 / 255), clean - 10 / 255).clamp(0, 1)
        trace.append(float(loss.detach()))
    return adv, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument(
        "--proxy-vit", choices=SAM, default=None,
        help="optional architecture for a task proxy when no same-ViT independent seed exists",
    )
    parser.add_argument("--proxy-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--guidance", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 2.0]
    )
    parser.add_argument(
        "--schedules", nargs="+", default=["constant"],
        choices=["constant", "linear_decay", "linear_ramp", "early", "late"],
    )
    parser.add_argument("--first-step-multiplier", type=float, default=1.0)
    parser.add_argument("--initial-delta-scale", type=float, default=1.0)
    parser.add_argument("--adaptive-cosine-threshold", type=float, default=0.03)
    args = parser.parse_args()
    proxy_vit = args.proxy_vit or args.vit
    device = torch.device(args.device)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        image_np = handle["image"][args.slice].astype(np.float32)
    clean = torch.from_numpy(image_np)[None, None].to(device)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    proxy = load_medsa_model(
        proxy_vit, str(SAM[proxy_vit]), str(ROOT / args.proxy_checkpoint),
        512, 9, device=device, dataset="Synapse",
    )
    proxy_adv, proxy_trace = attack_proxy_encoder(proxy, clean)
    proxy_path = output / "SEED42_PROXY.png"
    records = [{
        "name": "SEED42_PROXY", "path": str(proxy_path.relative_to(ROOT)),
        "linf_255": save_feasible(proxy_path, proxy_adv, clean),
        "target_seed24_checkpoint_used": False,
        "proxy_checkpoint": args.proxy_checkpoint, "proxy_vit": proxy_vit,
        "source_vit": args.vit, "trace": proxy_trace,
    }]
    initial_delta = (
        args.initial_delta_scale * (proxy_adv - clean)
    ).repeat(1, 3, 1, 1).detach()
    del proxy
    torch.cuda.empty_cache()
    source, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=8, checkpoint=str(SAM[args.vit]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    encoder = source.image_encoder.to(device).eval()
    image = clean.repeat(1, 3, 1, 1)
    for schedule in args.schedules:
        for guidance in args.guidance:
            cfg = ablation_config(
                "CH", epsilon=10 / 255, step_size=2 / 255, steps=10,
                seed=42, prior_guidance_weight=guidance,
                prior_guidance_schedule=schedule,
            )
            adv, trace = attack_image(
                encoder, image, cfg,
                preprocess=lambda value: value[:, :1].repeat(1, 3, 1, 1),
                initial_delta=initial_delta,
                first_step_multiplier=args.first_step_multiplier,
            )
            tag = {"constant": "C", "linear_decay": "D", "linear_ramp": "R", "early": "E", "late": "L"}[schedule]
            name = f"SEED42_CH_{tag}{int(guidance * 100):03d}"
            path = output / f"{name}.png"
            records.append({
                "name": name, "path": str(path.relative_to(ROOT)),
                "linf_255": save_feasible(path, adv, image),
                "target_seed24_checkpoint_used": False,
                "proxy_checkpoint": args.proxy_checkpoint,
                "proxy_vit": proxy_vit, "source_vit": args.vit,
                "prior_guidance_schedule": schedule,
                "prior_guidance_weight": guidance,
                "first_step_multiplier": args.first_step_multiplier,
                "initial_delta_scale": args.initial_delta_scale,
                "trace": {key: values[-1] for key, values in trace.items()},
            })
            print(json.dumps(records[-1], indent=2), flush=True)
            online_cosine = trace["cos_online_prior"][-1]
            adaptive_uses_proxy = (
                proxy_vit == args.vit
                and online_cosine < args.adaptive_cosine_threshold
            )
            adaptive_adv = proxy_adv if adaptive_uses_proxy else adv
            adaptive_name = f"ADAPTIVE_{tag}{int(guidance * 100):03d}"
            adaptive_path = output / f"{adaptive_name}.png"
            records.append({
                "name": adaptive_name,
                "path": str(adaptive_path.relative_to(ROOT)),
                "linf_255": save_feasible(adaptive_path, adaptive_adv, image),
                "target_seed24_checkpoint_used": False,
                "proxy_checkpoint": args.proxy_checkpoint,
                "proxy_vit": proxy_vit, "source_vit": args.vit,
                "selection_uses_target": False,
                "adaptive_cosine_threshold": args.adaptive_cosine_threshold,
                "observed_online_prior_cosine": online_cosine,
                "same_architecture_proxy": proxy_vit == args.vit,
                "selected": "proxy_endpoint" if adaptive_uses_proxy else "continuation",
            })
    (output / "generation.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
