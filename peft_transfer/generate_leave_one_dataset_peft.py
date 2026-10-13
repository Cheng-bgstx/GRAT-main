#!/usr/bin/env python3
"""Build a target-checkpoint-free attack from other medical PEFT tasks."""
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


def save_feasible(path, adv, clean, epsilon=10 / 255):
    clean_np = clean[0, 0].cpu().numpy()
    value = adv[0, 0].cpu().numpy()
    lower = np.ceil(np.maximum(0.0, clean_np - epsilon) * 255).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_np + epsilon) * 255).astype(np.int16)
    quantized = np.clip(np.rint(value * 255).astype(np.int16), lower, upper).astype(np.uint8)
    Image.fromarray(quantized).save(path)
    return float(np.max(np.abs(quantized.astype(np.float32) - clean_np * 255)))


def proxy_ensemble_attack(models, clean, steps=10):
    references = []
    with torch.no_grad():
        for model in models:
            references.append(model.image_encoder(model.preprocess(clean.repeat(1, 3, 1, 1))))
    adv = clean.clone()
    momentum = torch.zeros_like(clean)
    trace = []
    for step in range(steps):
        adv = adv.detach().requires_grad_(True)
        gradients, losses = [], []
        for model, reference in zip(models, references):
            embedding = model.image_encoder(model.preprocess(adv.repeat(1, 3, 1, 1)))
            loss = F.l1_loss(embedding * 0.9999, reference)
            gradient = torch.autograd.grad(loss, adv)[0]
            gradients.append(gradient / gradient.abs().mean().clamp_min(1e-12))
            losses.append(loss.detach())
        stack = torch.stack(gradients)
        # Equal weights avoid selecting a held-out task through victim feedback.
        gradient = stack.mean(0)
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = 0.5 * momentum + gradient
        adv = adv.detach() + _asymmetric_step(momentum, 2 / 255)
        adv = torch.maximum(torch.minimum(adv, clean + 10 / 255), clean - 10 / 255).clamp(0, 1)
        trace.append({
            "losses": [float(value) for value in losses],
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        image_np = handle["image"][args.slice].astype(np.float32)
    clean = torch.from_numpy(image_np)[None, None].to(device)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    models = [
        load_medsa_model(
            args.vit, str(SAM[args.vit]),
            str(ROOT / f"checkpoints/best_weights/MSA_{dataset}_{args.vit}.pth"),
            512, classes, device=device, dataset=dataset_name,
        )
        for dataset, dataset_name, classes in (("ISIC", "isic", 1), ("REFUGE", "REFUGE", 2))
    ]
    proxy_adv, proxy_trace = proxy_ensemble_attack(models, clean)
    proxy_path = output / "LODO_PROXY.png"
    proxy_linf = save_feasible(proxy_path, proxy_adv, clean)
    initial_delta = (proxy_adv - clean).repeat(1, 3, 1, 1).detach()
    del models
    torch.cuda.empty_cache()

    source, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=8, checkpoint=str(SAM[args.vit]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    encoder = source.image_encoder.to(device).eval()
    image = clean.repeat(1, 3, 1, 1)
    records = [{
        "name": "LODO_PROXY", "path": str(proxy_path.relative_to(ROOT)),
        "linf_255": proxy_linf, "held_out_checkpoint_used": False,
        "proxy_datasets": ["ISIC", "REFUGE"], "trace": proxy_trace,
    }]
    for guidance in (0.1, 0.25, 0.5, 1.0):
        cfg = ablation_config(
            "CH", epsilon=10 / 255, step_size=2 / 255, steps=10,
            seed=42, prior_guidance_weight=guidance,
        )
        adv, trace = attack_image(
            encoder, image, cfg,
            preprocess=lambda value: value[:, :1].repeat(1, 3, 1, 1),
            initial_delta=initial_delta, first_step_multiplier=1.0,
        )
        name = f"LODO_CH_G{int(guidance * 100):03d}"
        path = output / f"{name}.png"
        records.append({
            "name": name, "path": str(path.relative_to(ROOT)),
            "linf_255": save_feasible(path, adv, image),
            "held_out_checkpoint_used": False,
            "proxy_datasets": ["ISIC", "REFUGE"],
            "trace": {key: values[-1] for key, values in trace.items()},
        })
        print(json.dumps(records[-1], indent=2), flush=True)
    (output / "generation.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
