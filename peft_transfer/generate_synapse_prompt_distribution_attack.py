#!/usr/bin/env python3
"""Attack held-out prompt risk using independent PEFT proxies and prompt consensus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from peft_transfer.evaluate_official_slice import SAM, load_medsa_model, load_samed_model
from peft_transfer.generate_2d_dual_peft import (
    antithetic_views,
    hierarchical_gradient,
    project_grayscale_gradient,
)
from peft_transfer.generate_synapse_dual_peft import load_grayscale_prior, save_grayscale


def sample_multiclass_points(label, classes, seed):
    """Sample one point from every present class, matching official Synapse prompts."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    coordinates = []
    point_labels = []
    label_cpu = label[0, 0].detach().cpu()
    for class_id in range(classes):
        indices = torch.nonzero(label_cpu == class_id)
        if len(indices):
            choice = int(torch.randint(len(indices), (), generator=generator))
            y, x = indices[choice]
            coordinates.append(torch.tensor([x, y]))
            point_labels.append(1)
        else:
            coordinates.append(torch.tensor([0, 0]))
            point_labels.append(0)
    return (
        torch.stack(coordinates)[None].float().to(label.device),
        torch.tensor(point_labels, device=label.device)[None].int(),
    )


def spatial_risk_weights(label, strength):
    """Upweight background farther from anatomy to target boundary displacement."""
    weights = torch.ones_like(label, dtype=torch.float32)
    if strength <= 0:
        return weights
    foreground = label[0, 0].detach().cpu().numpy() > 0
    distance = distance_transform_edt(~foreground).astype(np.float32)
    maximum = float(distance.max())
    if maximum > 0:
        distance /= maximum
    distance = torch.from_numpy(distance).to(label.device)
    background = label[0, 0] == 0
    weights[0, 0, background] += strength * distance[background]
    return weights


def balanced_label_ce(logits, label, spatial_weights=None):
    losses = []
    targets = label[:, 0].long()
    pixel_weights = (
        torch.ones_like(targets, dtype=logits.dtype)
        if spatial_weights is None else spatial_weights[:, 0].to(logits.dtype)
    )
    for class_id in torch.unique(targets):
        mask = targets == class_id
        per_pixel = F.cross_entropy(
            logits.permute(0, 2, 3, 1)[mask], targets[mask], reduction="none"
        )
        losses.append((per_pixel * pixel_weights[mask]).mean())
    return torch.stack(losses).mean()


def samed_logits(model, image):
    return model.sam(image, multimask_output=True, image_size=image.shape[-1])["masks"]


def medsa_logits(model, image, prompt_embedding):
    embedding = model.image_encoder(model.preprocess(image))
    sparse, dense = prompt_embedding
    logits, _ = model.mask_decoder(
        image_embeddings=embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
    )
    return F.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False)


def normalized_gray_gradient(loss, image, retain_graph=False):
    gradient = torch.autograd.grad(loss, image, retain_graph=retain_graph)[0]
    gradient = project_grayscale_gradient(gradient)
    return gradient / gradient.abs().mean().clamp_min(1e-12)


def attack(samed, medsa_models, clean, label, initial, weight, sigma,
           prompt_samples, seed_lambda, prompt_lambda, spatial_lambda, seed):
    prompt_sets = [
        sample_multiclass_points(label, 9, seed + 7919 * index)
        for index in range(prompt_samples)
    ]
    prompt_embeddings = []
    spatial_weights = spatial_risk_weights(label, spatial_lambda)
    with torch.no_grad():
        for model in medsa_models:
            prompt_embeddings.append([
                tuple(value.detach() for value in model.prompt_encoder(
                    points=prompt, boxes=None, masks=None
                ))
                for prompt in prompt_sets
            ])
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
        samed_loss = torch.stack([
            balanced_label_ce(samed_logits(samed, view), label, spatial_weights)
            for view in neighborhood
        ]).mean()
        medsa_losses = []
        for model, model_prompts in zip(medsa_models, prompt_embeddings):
            for prompt_embedding in model_prompts:
                medsa_losses.append(torch.stack([
                    balanced_label_ce(
                        medsa_logits(model, view, prompt_embedding), label,
                        spatial_weights,
                    ) for view in neighborhood
                ]).mean())
        samed_gradient = normalized_gray_gradient(samed_loss, adv, retain_graph=True)
        medsa_gradients = [
            normalized_gray_gradient(
                loss, adv, retain_graph=index + 1 < len(medsa_losses)
            ) for index, loss in enumerate(medsa_losses)
        ]
        medsa_gradient, retained = hierarchical_gradient(
            medsa_gradients, group_size=prompt_samples,
            seed_lambda=seed_lambda, trajectory_lambda=prompt_lambda,
        )
        medsa_gradient = medsa_gradient / medsa_gradient.abs().mean().clamp_min(1e-12)
        gradient = weight * samed_gradient + (1 - weight) * medsa_gradient
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = momentum + gradient
        adv = adv.detach() + (2 / 255) * momentum.sign()
        adv = torch.maximum(torch.minimum(adv, clean + 10 / 255), clean - 10 / 255).clamp(0, 1)
        trace.append({
            "step": step, "samed_loss": float(samed_loss.detach()),
            "medsa_losses": [float(value.detach()) for value in medsa_losses],
            "family_gradient_cosine": float(F.cosine_similarity(
                samed_gradient.flatten(1), medsa_gradient.flatten(1), dim=1
            ).mean()),
            "retained_fraction": retained,
        })
    return adv, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slices", type=int, nargs="+", required=True)
    parser.add_argument("--samed-proxy", required=True)
    parser.add_argument("--medsa-proxies", nargs="+", required=True)
    parser.add_argument("--samed-weights", type=float, nargs="+", default=(0, 0.5, 1))
    parser.add_argument("--prompt-samples", type=int, default=4)
    parser.add_argument("--seed-lambda", type=float, default=0.0)
    parser.add_argument("--prompt-lambda", type=float, default=0.5)
    parser.add_argument("--spatial-lambda", type=float, default=0.0)
    parser.add_argument("--initial-perturbation", default="common_perb/Synapse.pth")
    parser.add_argument("--sigma", type=float, default=1.0)
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
            label = handle["label"][slice_index].astype(np.int64)
            clean = torch.from_numpy(image)[None, None].repeat(1, 3, 1, 1).to(device)
            label_tensor = torch.from_numpy(label)[None, None].to(device)
            for weight in args.samed_weights:
                adv, trace = attack(
                    samed, medsa_models, clean, label_tensor, initial, weight,
                    args.sigma / 255.0, args.prompt_samples, args.seed_lambda,
                    args.prompt_lambda, args.spatial_lambda,
                    args.seed + 100 * slice_index,
                )
                candidate = (
                    f"PRISK_W{int(round(100 * weight)):03d}_"
                    f"SP{args.spatial_lambda:g}"
                )
                path = ROOT / args.out / candidate / f"{slice_index}.png"
                row = {
                    "case": args.case, "slice": slice_index, "candidate": candidate,
                    "target_checkpoints_used": False,
                    "prompt_conditioned_margin": False,
                    "held_out_target_prompt_seed": 24,
                    "proxy_prompt_samples": args.prompt_samples,
                    "spatial_lambda": args.spatial_lambda,
                    "samed_proxy": args.samed_proxy,
                    "medsa_proxies": args.medsa_proxies,
                    "linf_255": save_grayscale(path, adv, clean), "trace": trace,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    output = ROOT / args.out / "generation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
