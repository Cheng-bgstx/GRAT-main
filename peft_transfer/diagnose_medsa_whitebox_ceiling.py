#!/usr/bin/env python3
"""Measure a MedSA task-loss attack ceiling; never use this as a proposed attack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from peft_transfer.evaluate_official_slice import (
    ROOT,
    SAM,
    load_medsa_function,
    load_medsa_model,
    reset_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-seed", type=int, default=24)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--step-size", type=float, default=2.0)
    parser.add_argument("--objective", choices=("ce", "coverage"), default="ce")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    with h5py.File(
        ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r"
    ) as handle:
        image_np = handle["image"][args.slice].astype(np.float32)
        label_np = handle["label"][args.slice].astype(np.int64)
    clean = torch.from_numpy(image_np)[None, None].to(device)
    label = torch.from_numpy(label_np)[None].long().to(device)
    model = load_medsa_model(
        args.vit,
        str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
        512,
        9,
        device=device,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module = load_medsa_function("Synapse", 9)
    reset_seed(args.prompt_seed)
    _, points, _, point_labels = module.generate_click_prompt(clean.repeat(1, 3, 1, 1), label)
    if points.dim() == 3 and points.shape[1] == 2:
        points = points.permute(0, 2, 1)
    prompt = (points.float().to(device), point_labels.int().to(device))
    epsilon = args.epsilon / 255.0
    step_size = args.step_size / 255.0
    adv = clean.clone()
    losses = []
    for _ in range(args.steps):
        adv.requires_grad_(True)
        embedding = model.image_encoder(model.preprocess(adv.repeat(1, 3, 1, 1)))
        sparse, dense = model.prompt_encoder(points=prompt, boxes=None, masks=None)
        logits, _ = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        logits = F.interpolate(logits, (512, 512), mode="bilinear", align_corners=False)
        if args.objective == "ce":
            loss = F.cross_entropy(logits, label)
        else:
            present = set(int(value) for value in torch.unique(label))
            absent = [class_id for class_id in range(1, 9) if class_id not in present]
            if not absent:
                loss = F.cross_entropy(logits, label)
            else:
                height, width = label.shape[-2:]
                y = torch.arange(height, device=device).view(1, height, 1)
                x = torch.arange(width, device=device).view(1, 1, width)
                regions = ((2 * y // height) * 2 + (2 * x // width)).remainder(len(absent))
                targets = torch.tensor(absent, device=device)[regions].expand_as(label)
                # Targeted descent is expressed as ascent on negative CE.
                loss = -F.cross_entropy(logits, targets)
        gradient = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + step_size * gradient.sign()
        adv = torch.maximum(torch.minimum(adv, clean + epsilon), clean - epsilon).clamp(0, 1)
        losses.append(float(loss.detach()))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_np = clean[0, 0].cpu().numpy()
    adv_np = adv[0, 0].cpu().numpy()
    lower = np.ceil(np.maximum(0.0, clean_np - epsilon) * 255).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_np + epsilon) * 255).astype(np.int16)
    quantized = np.clip(np.rint(adv_np * 255).astype(np.int16), lower, upper).astype(np.uint8)
    Image.fromarray(quantized).save(output)
    report = {
        "diagnostic_only": True,
        "uses_victim_gradient": True,
        "objective": args.objective,
        "losses": losses,
        "measured_linf": float(np.max(np.abs(quantized.astype(np.float32) / 255 - clean_np))),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
