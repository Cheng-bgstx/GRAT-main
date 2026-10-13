#!/usr/bin/env python3
"""Attack prompt-free PEFT readouts using clean pseudo labels from one image."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from peft_transfer.evaluate_official_slice import SAM, load_medsa_model, load_samed_model
from peft_transfer.generate_2d_dual_peft import (
    antithetic_views,
    hierarchical_gradient,
    project_grayscale_gradient,
)
from peft_transfer.generate_synapse_dual_peft import load_grayscale_prior, save_grayscale


def samed_logits(model, image):
    return model.sam(image, multimask_output=True, image_size=image.shape[-1])["masks"]


def medsa_empty_prompt_logits(model, image):
    embedding = model.image_encoder(model.preprocess(image))
    sparse, dense = model.prompt_encoder(points=None, boxes=None, masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
    )
    return F.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False)


def balanced_pseudo_ce(logits, targets):
    losses = []
    for class_id in torch.unique(targets):
        mask = targets == class_id
        losses.append(F.cross_entropy(
            logits.permute(0, 2, 3, 1)[mask], targets[mask]
        ))
    return torch.stack(losses).mean()


def normalized_gray_gradient(loss, image, retain_graph=False):
    gradient = torch.autograd.grad(loss, image, retain_graph=retain_graph)[0]
    gradient = project_grayscale_gradient(gradient)
    return gradient / gradient.abs().mean().clamp_min(1e-12)


def attack(samed, medsa_models, clean, initial, weight, sigma,
           uncertainty_lambda, seed):
    functions = [
        lambda value: samed_logits(samed, value),
        *[lambda value, model=model: medsa_empty_prompt_logits(model, value)
          for model in medsa_models],
    ]
    with torch.no_grad():
        targets = [function(clean).argmax(1) for function in functions]
    adv = clean.clone()
    if initial is not None:
        adv = torch.maximum(torch.minimum(clean + initial, clean + 10 / 255), clean - 10 / 255)
        adv = adv.clamp(0, 1)
    momentum = torch.zeros_like(clean)
    trace = []
    for step in range(10):
        adv = adv.detach().requires_grad_(True)
        neighborhood = antithetic_views(
            adv, sigma, seed + 10000 + step, boundary="centered"
        )
        losses = [torch.stack([
            balanced_pseudo_ce(function(view), target)
            for view in neighborhood
        ]).mean() for function, target in zip(functions, targets)]
        gradients = [
            normalized_gray_gradient(loss, adv, retain_graph=index + 1 < len(losses))
            for index, loss in enumerate(losses)
        ]
        samed_gradient = gradients[0]
        medsa_gradient, retained = hierarchical_gradient(
            gradients[1:], seed_lambda=uncertainty_lambda
        )
        medsa_gradient = medsa_gradient / medsa_gradient.abs().mean().clamp_min(1e-12)
        gradient = weight * samed_gradient + (1 - weight) * medsa_gradient
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = momentum + gradient
        adv = adv.detach() + (2 / 255) * momentum.sign()
        adv = torch.maximum(torch.minimum(adv, clean + 10 / 255), clean - 10 / 255).clamp(0, 1)
        trace.append({
            "step": step,
            "samed_loss": float(losses[0].detach()),
            "medsa_losses": [float(value.detach()) for value in losses[1:]],
            "family_gradient_cosine": float(F.cosine_similarity(
                samed_gradient.flatten(1), medsa_gradient.flatten(1), dim=1
            ).mean()),
            "adapter_retained_fraction": retained,
        })
    classes = [torch.unique(target).tolist() for target in targets]
    return adv, trace, classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slices", type=int, nargs="+", required=True)
    parser.add_argument("--samed-proxy", required=True)
    parser.add_argument("--medsa-proxies", nargs="+", required=True)
    parser.add_argument("--samed-weights", type=float, nargs="+", default=(0, 0.5, 1))
    parser.add_argument("--initial-perturbation", default="common_perb/Synapse.pth")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--uncertainty-lambda", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    samed = load_samed_model(
        args.vit, str(SAM[args.vit]), str(ROOT / args.samed_proxy), 512, 8,
        device=device,
    )
    medsa_models = [load_medsa_model(
        args.vit, str(SAM[args.vit]), str(ROOT / checkpoint), 512, 9,
        device=device, dataset="Synapse",
    ) for checkpoint in args.medsa_proxies]
    for parameter in list(samed.parameters()) + [
        value for model in medsa_models for value in model.parameters()
    ]:
        parameter.requires_grad_(False)
    initial = load_grayscale_prior(args.initial_perturbation, device)
    rows = []
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        for slice_index in args.slices:
            image = handle["image"][slice_index].astype(np.float32)
            clean = torch.from_numpy(image)[None, None].repeat(1, 3, 1, 1).to(device)
            for weight in args.samed_weights:
                adv, trace, classes = attack(
                    samed, medsa_models, clean, initial, weight,
                    args.sigma / 255.0, args.uncertainty_lambda,
                    args.seed + 100 * slice_index,
                )
                candidate = f"READOUT_W{int(round(100 * weight)):03d}"
                path = ROOT / args.out / candidate / f"{slice_index}.png"
                row = {
                    "case": args.case, "slice": slice_index, "candidate": candidate,
                    "target_checkpoints_used": False,
                    "uses_point_prompt": False,
                    "uses_ground_truth_labels": False,
                    "samed_proxy": args.samed_proxy,
                    "medsa_proxies": args.medsa_proxies,
                    "clean_pseudo_classes": classes,
                    "linf_255": save_grayscale(path, adv, clean), "trace": trace,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    output = ROOT / args.out / "generation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
