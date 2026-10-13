#!/usr/bin/env python3
"""Generate source-only smooth local phase-flow attacks for one Synapse slice."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from peft_transfer.core import (
    AttackConfig,
    _unpack_encoder,
    ablation_config,
    attack_image,
    carrier_stochastic_disruption,
    phase_amplitude_stochastic_disruption,
)
from peft_transfer.generate import CHECKPOINTS, ROOT
from segment_anything import sam_model_registry


def phase_flow_attack(encoder, image, cfg, grid_size: int, max_flow_px: float):
    """Optimize a smooth displacement field through the frozen source encoder."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    height, width = image.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=image.device),
        torch.linspace(-1.0, 1.0, width, device=image.device),
        indexing="ij",
    )
    base_grid = torch.stack((xx, yy), dim=-1)[None]
    preprocess = lambda value: value[:, :1].repeat(1, 3, 1, 1)
    with torch.no_grad():
        clean_final, clean_layers = _unpack_encoder(encoder(preprocess(image)))
    flow = 1e-4 * torch.randn(1, 2, grid_size, grid_size, device=image.device)
    momentum = torch.zeros_like(flow)
    trace = []
    for step in range(cfg.steps):
        flow = flow.detach().requires_grad_(True)
        dense = F.interpolate(flow, (height, width), mode="bicubic", align_corners=True)
        normalized = torch.stack(
            (2.0 * dense[:, 0] / max(width - 1, 1), 2.0 * dense[:, 1] / max(height - 1, 1)),
            dim=-1,
        )
        warped = F.grid_sample(
            image, base_grid + normalized, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        adv = torch.maximum(torch.minimum(warped, image + cfg.epsilon), image - cfg.epsilon)
        adv = adv.clamp(0.0, 1.0)
        final, layers = _unpack_encoder(encoder(preprocess(adv)))
        feature = F.l1_loss(final * 0.9999, clean_final)
        if step < 4:
            phase_cfg = AttackConfig(
                seed=cfg.seed, pasd_mode="shuffled_white", pasd_samples=2,
                pasd_probe_scale=2.0, pasd_region="final", pasd_layer_count=2,
            )
            auxiliary = phase_amplitude_stochastic_disruption(layers, clean_layers, phase_cfg)
        else:
            auxiliary = carrier_stochastic_disruption(layers, clean_layers, cfg)
        loss = feature + auxiliary
        gradient = torch.autograd.grad(loss, flow)[0]
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = cfg.momentum * momentum + gradient
        flow = flow.detach() + 0.75 * momentum.sign()
        flow = flow.clamp(-max_flow_px, max_flow_px)
        trace.append({
            "loss": float(loss.detach()), "feature": float(feature.detach()),
            "auxiliary": float(auxiliary.detach()),
            "saturated_flow_fraction": float((flow.abs() >= max_flow_px).float().mean()),
        })
    with torch.no_grad():
        dense = F.interpolate(flow, (height, width), mode="bicubic", align_corners=True)
        normalized = torch.stack(
            (2.0 * dense[:, 0] / max(width - 1, 1), 2.0 * dense[:, 1] / max(height - 1, 1)),
            dim=-1,
        )
        warped = F.grid_sample(
            image, base_grid + normalized, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        adv = torch.maximum(torch.minimum(warped, image + cfg.epsilon), image - cfg.epsilon)
        adv = adv.clamp(0.0, 1.0)
    return adv, trace


def save_feasible(path: Path, adv: torch.Tensor, clean: torch.Tensor, epsilon: float):
    clean_np = clean[0, 0].cpu().numpy()
    value = adv[0, 0].cpu().numpy()
    lower = np.ceil(np.maximum(0.0, clean_np - epsilon) * 255.0).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_np + epsilon) * 255.0).astype(np.int16)
    quantized = np.clip(np.rint(value * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
    Image.fromarray(quantized).save(path)
    return float(np.max(np.abs(quantized.astype(np.float32) - clean_np * 255.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--vit", choices=CHECKPOINTS, default="vit_b")
    parser.add_argument("--grids", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--max-flow", type=float, default=6.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
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
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    cfg = AttackConfig(epsilon=10 / 255, step_size=2 / 255, steps=10, momentum=0.5, seed=args.seed)
    records = []
    for grid in args.grids:
        flow_adv, trace = phase_flow_attack(encoder, image, cfg, grid, args.max_flow)
        flow_path = output / f"FLOW{grid}.png"
        flow_linf = save_feasible(flow_path, flow_adv, image, cfg.epsilon)
        delta = (flow_adv - image).detach()
        ch_cfg = ablation_config("CH", epsilon=10 / 255, step_size=2 / 255, steps=10, seed=args.seed)
        refined, ch_trace = attack_image(
            encoder, image, ch_cfg,
            preprocess=lambda value: value[:, :1].repeat(1, 3, 1, 1),
            initial_delta=delta, first_step_multiplier=1.0,
        )
        refined_path = output / f"FLOW{grid}_CH.png"
        refined_linf = save_feasible(refined_path, refined, image, cfg.epsilon)
        records.extend([
            {"name": f"FLOW{grid}", "path": str(flow_path.relative_to(ROOT)), "linf_255": flow_linf, "trace": trace},
            {"name": f"FLOW{grid}_CH", "path": str(refined_path.relative_to(ROOT)), "linf_255": refined_linf,
             "trace": {key: values[-1] for key, values in ch_trace.items()}},
        ])
        print(json.dumps(records[-2:], indent=2), flush=True)
    (output / "generation.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
