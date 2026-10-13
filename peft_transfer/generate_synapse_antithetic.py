#!/usr/bin/env python3
"""Generate strict base-SAM Synapse attacks with two-axis antithetic gradients."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from peft_transfer.core import AttackConfig, VirtualPEFTView, _unpack_encoder, build_virtual_views
from peft_transfer.generate import CHECKPOINTS
from peft_transfer.generate_2d_dual_peft import antithetic_views, hierarchical_gradient
from segment_anything import sam_model_registry


def make_views(encoder, mode: str, seed: int):
    if mode == "base":
        return [VirtualPEFTView(encoder, "source", 1, 0.0, seed)]
    if mode == "mixed":
        return build_virtual_views(
            encoder,
            AttackConfig(ranks=(4, 8), virtual_scale=0.08, seed=seed),
        )
    if mode == "adapter":
        return build_virtual_views(
            encoder,
            AttackConfig(
                adapter_view_count=4,
                adapter_view_rank=192,
                adapter_view_scale=0.25,
                seed=seed,
            ),
        )
    raise ValueError(mode)


def feature(encoder, image):
    return _unpack_encoder(encoder(image))[0]


def attack(encoder, clean, views, sigma, uncertainty_lambda, seed, boundary):
    references = []
    with torch.no_grad():
        for view in views:
            with view.active():
                references.append(feature(encoder, clean).detach())
    adv = clean.clone()
    momentum = torch.zeros_like(clean)
    trace = []
    for step in range(10):
        adv = adv.detach().requires_grad_(True)
        local_views = antithetic_views(
            adv, sigma, seed + 10000 + step, boundary=boundary
        )
        losses = []
        for view, reference in zip(views, references):
            with view.active():
                loss = torch.stack([
                    F.l1_loss(feature(encoder, local) * 0.9999, reference)
                    for local in local_views
                ]).mean()
            losses.append(loss)
        gradients = []
        for index, loss in enumerate(losses):
            gradient = torch.autograd.grad(
                loss, adv, retain_graph=index + 1 < len(losses)
            )[0]
            gradients.append(gradient / gradient.abs().mean().clamp_min(1e-12))
        gradient, retained = hierarchical_gradient(
            gradients, seed_lambda=uncertainty_lambda
        )
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = momentum + gradient
        adv = adv.detach() + (2.0 / 255.0) * momentum.sign()
        adv = torch.maximum(torch.minimum(adv, clean + 10.0 / 255.0), clean - 10.0 / 255.0)
        adv = adv.clamp(0.0, 1.0)
        trace.append({
            "step": step,
            "mean_loss": float(torch.stack([value.detach() for value in losses]).mean()),
            "retained_fraction": retained,
        })
    return adv, trace


def save_quantized(path: Path, adv: torch.Tensor, clean: torch.Tensor):
    clean_np = clean[0, 0].detach().cpu().numpy()
    adv_np = adv[0, 0].detach().cpu().numpy()
    lower = np.ceil(np.maximum(0.0, clean_np - 10.0 / 255.0) * 255.0).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_np + 10.0 / 255.0) * 255.0).astype(np.int16)
    quantized = np.clip(np.rint(adv_np * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(quantized).save(path)
    return float(np.abs(quantized.astype(np.float32) - clean_np * 255.0).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=CHECKPOINTS, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slices", type=int, nargs="+", required=True)
    parser.add_argument("--modes", choices=("base", "mixed", "adapter"), nargs="+",
                        default=("base", "mixed", "adapter"))
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="antithetic radius in 0--255 units")
    parser.add_argument("--uncertainty-lambda", type=float, default=0.5)
    parser.add_argument("--boundary", choices=("clip", "centered"), default="centered")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=8, checkpoint=str(CHECKPOINTS[args.vit]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    encoder = model.image_encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    records = []
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        for slice_index in args.slices:
            image_np = handle["image"][slice_index].astype(np.float32)
            clean = torch.from_numpy(image_np)[None, None].repeat(1, 3, 1, 1).to(device)
            for mode in args.modes:
                views = make_views(encoder, mode, args.seed)
                adv, trace = attack(
                    encoder, clean, views, args.sigma / 255.0,
                    args.uncertainty_lambda if len(views) > 1 else 0.0,
                    args.seed + slice_index * 100, args.boundary,
                )
                name = f"{mode}_smooth{args.sigma:g}_{args.boundary}"
                path = ROOT / args.out / name / f"{slice_index}.png"
                linf = save_quantized(path, adv, clean)
                record = {
                    "case": args.case, "slice": slice_index, "candidate": name,
                    "target_checkpoints_used": False, "view_count": len(views),
                    "sigma_255": args.sigma, "boundary": args.boundary,
                    "uncertainty_lambda": args.uncertainty_lambda,
                    "linf_255": linf, "trace": trace,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    output = ROOT / args.out / "generation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
