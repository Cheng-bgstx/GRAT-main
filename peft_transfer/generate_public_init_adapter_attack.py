#!/usr/bin/env python3
"""Attack a public-seed, untrained MedSA adapter encoder without victim weights."""
from __future__ import annotations

import argparse
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from peft_transfer.core import _asymmetric_step, ablation_config, attack_image
from peft_transfer.generate import CHECKPOINTS, ROOT
from segment_anything import sam_model_registry


def save_feasible(path: Path, adv: torch.Tensor, clean: torch.Tensor, epsilon: float) -> float:
    clean_np = clean[0, 0].detach().cpu().numpy()
    value = adv[0, 0].detach().cpu().numpy()
    lower = np.ceil(np.maximum(0.0, clean_np - epsilon) * 255.0).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_np + epsilon) * 255.0).astype(np.int16)
    quantized = np.clip(np.rint(value * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
    Image.fromarray(quantized).save(path)
    return float(np.max(np.abs(quantized.astype(np.float32) - clean_np * 255.0)))


def build_public_init_encoder(vit: str, checkpoint: Path, device: torch.device, seed: int):
    msa_root = str(ROOT / "Medical-SAM-Adapter-main")
    saved_path = list(sys.path)
    sys.path.insert(0, msa_root)
    try:
        from models.sam import sam_model_registry as medsa_registry
        args = Namespace(
            dataset="Synapse", image_size=512, multimask_output=9,
            sam_ckpt=str(checkpoint), encoder=vit, mod="sam_adpt",
            mid_dim=None, thd=False, chunk=None,
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = medsa_registry[vit](args, checkpoint=str(checkpoint)).to(device).eval()
    finally:
        sys.path = saved_path
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def attack_initialized_encoder(model, clean_gray: torch.Tensor, steps: int = 10):
    with torch.no_grad():
        clean_embedding = model.image_encoder(model.preprocess(clean_gray.repeat(1, 3, 1, 1)))
    torch.manual_seed(42)
    adv = (clean_gray + 1e-5 * torch.randn_like(clean_gray)).clamp(0, 1)
    momentum = torch.zeros_like(clean_gray)
    trace = []
    for step in range(steps):
        adv = adv.detach().requires_grad_(True)
        forward = adv + (1e-4 * torch.randn_like(adv) if step == 0 else 0.0)
        embedding = model.image_encoder(model.preprocess(forward.repeat(1, 3, 1, 1)))
        loss = F.l1_loss(embedding * 0.9999, clean_embedding)
        gradient = torch.autograd.grad(loss, adv)[0]
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = 0.5 * momentum + gradient
        adv = adv.detach() + _asymmetric_step(momentum, 2 / 255)
        adv = torch.maximum(torch.minimum(adv, clean_gray + 10 / 255), clean_gray - 10 / 255)
        adv = adv.clamp(0, 1)
        trace.append(float(loss.detach()))
    return adv, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--vit", choices=CHECKPOINTS, default="vit_b")
    parser.add_argument("--public-seed", type=int, default=24)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        image_np = handle["image"][args.slice].astype(np.float32)
    gray = torch.from_numpy(image_np)[None, None].to(device)
    image = gray.repeat(1, 3, 1, 1)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)

    public_model = build_public_init_encoder(
        args.vit, CHECKPOINTS[args.vit], device, args.public_seed
    )
    init_adv, init_trace = attack_initialized_encoder(public_model, gray)
    init_path = output / "PUBLIC_INIT.png"
    init_linf = save_feasible(init_path, init_adv, gray, 10 / 255)
    initial_delta = (init_adv - gray).repeat(1, 3, 1, 1).detach()
    del public_model
    torch.cuda.empty_cache()

    source, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=8, checkpoint=str(CHECKPOINTS[args.vit]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    encoder = source.image_encoder.to(device).eval()
    records = [{
        "name": "PUBLIC_INIT", "path": str(init_path.relative_to(ROOT)),
        "linf_255": init_linf, "public_seed": args.public_seed,
        "victim_checkpoint_used": False, "trace": init_trace,
    }]
    for guidance in (0.1, 0.25, 0.5):
        cfg = ablation_config(
            "CH", epsilon=10 / 255, step_size=2 / 255, steps=10, seed=42,
            prior_guidance_weight=guidance,
        )
        adv, trace = attack_image(
            encoder, image, cfg,
            preprocess=lambda value: value[:, :1].repeat(1, 3, 1, 1),
            initial_delta=initial_delta, first_step_multiplier=1.0,
        )
        name = f"PUBLIC_INIT_CH_G{int(guidance * 100):02d}"
        path = output / f"{name}.png"
        linf = save_feasible(path, adv, image, cfg.epsilon)
        records.append({
            "name": name, "path": str(path.relative_to(ROOT)),
            "linf_255": linf, "public_seed": args.public_seed,
            "victim_checkpoint_used": False,
            "trace": {key: values[-1] for key, values in trace.items()},
        })
        print(json.dumps(records[-1], indent=2), flush=True)
    (output / "generation.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
