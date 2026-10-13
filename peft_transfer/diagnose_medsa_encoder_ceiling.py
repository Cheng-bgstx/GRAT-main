#!/usr/bin/env python3
"""Diagnostic upper bound from attacking the trained MedSA encoder directly."""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from peft_transfer.evaluate_official_slice import ROOT, SAM, load_medsa_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        image_np = handle["image"][args.slice].astype(np.float32)
    clean = torch.from_numpy(image_np)[None, None].to(device)
    model = load_medsa_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
        512, 9, device=device,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        clean_embedding = model.image_encoder(model.preprocess(clean.repeat(1, 3, 1, 1)))
    adv = clean.clone()
    momentum = torch.zeros_like(adv)
    trace = []
    for _ in range(10):
        adv = adv.detach().requires_grad_(True)
        embedding = model.image_encoder(model.preprocess(adv.repeat(1, 3, 1, 1)))
        loss = F.l1_loss(embedding * 0.9999, clean_embedding)
        gradient = torch.autograd.grad(loss, adv)[0]
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = 0.5 * momentum + gradient
        adv = adv.detach() + (2 / 255) * momentum.sign()
        adv = torch.maximum(torch.minimum(adv, clean + 10 / 255), clean - 10 / 255).clamp(0, 1)
        trace.append(float(loss.detach()))
    clean_255 = image_np * 255.0
    lower = np.ceil(np.maximum(0.0, clean_255 - 10)).astype(np.int16)
    upper = np.floor(np.minimum(255.0, clean_255 + 10)).astype(np.int16)
    value = adv[0, 0].cpu().numpy()
    quantized = np.clip(np.rint(value * 255).astype(np.int16), lower, upper).astype(np.uint8)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(quantized).save(output)
    record = {
        "diagnostic_only": True,
        "uses_trained_medsa_encoder": True,
        "path": str(output.relative_to(ROOT)),
        "linf_255": float(np.max(np.abs(quantized.astype(np.float32) - clean_255))),
        "loss": trace,
    }
    output.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
