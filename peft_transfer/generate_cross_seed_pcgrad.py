#!/usr/bin/env python3
"""Optimize a perturbation against base SAM and a held-out-seed PEFT proxy.

The official target checkpoint is never loaded.  Each update combines gradients
from the frozen base encoder and an independently trained same-task adapter.
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from peft_transfer.core import _asymmetric_step, _unpack_encoder
from peft_transfer.evaluate_official_slice import ROOT, SAM, load_medsa_model
from segment_anything import sam_model_registry


def _normalize(gradient: torch.Tensor) -> torch.Tensor:
    return gradient / gradient.abs().mean().clamp_min(1e-12)


def _pcgrad(left: torch.Tensor, right: torch.Tensor):
    left_flat, right_flat = left.flatten(1), right.flatten(1)
    dot = (left_flat * right_flat).sum(dim=1, keepdim=True)
    if float(dot.item()) >= 0:
        return left, right
    left_norm = left_flat.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    right_norm = right_flat.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    left_projected = left_flat - dot / right_norm * right_flat
    right_projected = right_flat - dot / left_norm * left_flat
    return left_projected.view_as(left), right_projected.view_as(right)


def _save(path, value, clean, epsilon_255=10.0):
    clean_np = clean[0, 0].detach().cpu().numpy()
    value_np = value[0, 0].detach().cpu().numpy()
    lower = np.ceil(np.maximum(0.0, clean_np * 255.0 - epsilon_255)).astype(np.int16)
    upper = np.floor(np.minimum(255.0, clean_np * 255.0 + epsilon_255)).astype(np.int16)
    quantized = np.clip(np.rint(value_np * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
    Image.fromarray(quantized).save(path)
    return float(np.max(np.abs(quantized.astype(np.float32) - clean_np * 255.0)))


def _attack(base, proxy, clean, start, mode, proxy_weight, steps=10):
    image3 = clean.repeat(1, 3, 1, 1)
    with torch.no_grad():
        base_ref = _unpack_encoder(base(image3))[0].detach()
        proxy_ref = proxy.image_encoder(proxy.preprocess(image3)).detach()
    adv = start.repeat(1, 3, 1, 1).detach()
    momentum = torch.zeros_like(adv)
    trace = []
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        base_feature = _unpack_encoder(base(adv))[0]
        base_loss = F.l1_loss(base_feature * 0.9999, base_ref)
        proxy_loss = F.l1_loss(
            proxy.image_encoder(proxy.preprocess(adv)) * 0.9999, proxy_ref
        )
        base_grad = _normalize(torch.autograd.grad(base_loss, adv, retain_graph=True)[0])
        proxy_grad = _normalize(torch.autograd.grad(proxy_loss, adv)[0])
        cosine = F.cosine_similarity(base_grad.flatten(1), proxy_grad.flatten(1), dim=1).mean()
        if mode == "pcgrad":
            base_grad, proxy_grad = _pcgrad(base_grad, proxy_grad)
            base_grad, proxy_grad = _normalize(base_grad), _normalize(proxy_grad)
        elif mode != "sum":
            raise ValueError(mode)
        gradient = _normalize(base_grad + proxy_weight * proxy_grad)
        momentum = 0.5 * momentum + gradient
        adv = adv.detach() + _asymmetric_step(momentum, 2.0 / 255.0)
        adv = torch.maximum(torch.minimum(adv, image3 + 10.0 / 255.0), image3 - 10.0 / 255.0)
        adv = adv.clamp(0.0, 1.0)
        trace.append({
            "base_loss": float(base_loss.detach()),
            "proxy_loss": float(proxy_loss.detach()),
            "gradient_cosine": float(cosine.detach()),
        })
    return adv, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--proxy-checkpoint", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean = torch.from_numpy(handle["image"][args.slice].astype(np.float32))[None, None].to(device)
    start = torch.from_numpy(
        np.asarray(Image.open(ROOT / args.start).convert("L"), dtype=np.float32) / 255.0
    )[None, None].to(device)
    source, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=8, checkpoint=str(SAM[args.vit]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    base = source.image_encoder.to(device).eval()
    proxy = load_medsa_model(
        args.vit, str(SAM[args.vit]), str(ROOT / args.proxy_checkpoint),
        512, 9, device=device, dataset="Synapse",
    )
    for parameter in list(base.parameters()) + list(proxy.parameters()):
        parameter.requires_grad_(False)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for mode in ("sum", "pcgrad"):
        for weight in (0.25, 0.5, 1.0, 2.0, 4.0):
            adv, trace = _attack(base, proxy, clean, start, mode, weight)
            name = f"{mode.upper()}_W{int(weight * 100):03d}"
            path = output / f"{name}.png"
            records.append({
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "linf_255": _save(path, adv, clean),
                "proxy_weight": weight,
                "mode": mode,
                "target_seed24_checkpoint_used": False,
                "proxy_checkpoint": args.proxy_checkpoint,
                "trace": trace,
            })
            print(json.dumps(records[-1], indent=2), flush=True)
    (output / "generation.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
