#!/usr/bin/env python3
"""Attack a held-out PEFT checkpoint through neighboring trajectory endpoints."""
from __future__ import annotations

import argparse
import json
from argparse import Namespace

import torch
import torch.nn.functional as F
from torchvision import transforms

from datasets.isic import ISIC2016
from datasets.refuge import REFUGE
from peft_transfer.core import _unpack_encoder
from peft_transfer.evaluate_official_slice import ROOT, SAM, load_samed_model
from peft_transfer.generate_2d_checkpoint_proxy import save_candidate
from peft_transfer.generate_2d_robust_cone import robust_linf_direction
from segment_anything import sam_model_registry


DATASETS = {
    "ISIC": (ISIC2016, ROOT / "data/ISIC", 1),
    "REFUGE": (REFUGE, ROOT / "data/REFUGE-Multirater", 2),
}


def feature(encoder, image, kind="final"):
    final, layers = _unpack_encoder(encoder(image))
    return layers[0] if kind == "early" else final


def normalized_gradient(loss, image, retain_graph=False):
    gradient = torch.autograd.grad(loss, image, retain_graph=retain_graph)[0]
    return gradient / gradient.abs().mean().clamp_min(1e-12)


def project_simplex(value):
    """Euclidean projection of a vector onto the probability simplex."""
    sorted_value, _ = torch.sort(value, descending=True)
    cumulative = torch.cumsum(sorted_value, dim=0) - 1.0
    indices = torch.arange(1, value.numel() + 1, device=value.device, dtype=value.dtype)
    active = sorted_value - cumulative / indices > 0
    rho = int(active.nonzero()[-1])
    threshold = cumulative[rho] / indices[rho]
    return (value - threshold).clamp_min(0.0)


def simplex_maxmin_direction(gradients, iterations=40):
    stack = torch.stack(gradients)
    weights = torch.full(
        (len(gradients),), 1.0 / len(gradients),
        device=stack.device, dtype=stack.dtype,
    )
    for iteration in range(iterations):
        mixture = (weights[:, None, None, None, None] * stack).sum(0)
        subgradient = (stack * mixture.sign()[None]).flatten(1).mean(1)
        step_size = 0.25 / ((iteration + 1) ** 0.5)
        weights = project_simplex(weights - step_size * subgradient)
    mixture = (weights[:, None, None, None, None] * stack).sum(0)
    direction = mixture / mixture.abs().mean().clamp_min(1e-12)
    return direction, weights, float(mixture.abs().mean())


def attack(encoders, feature_kinds, clean, normalize, prior, mode):
    with torch.no_grad():
        references = [
            feature(encoder, normalize(clean), kind).detach()
            for encoder, kind in zip(encoders, feature_kinds)
        ]
    adv = clean.clone()
    momentum = torch.zeros_like(clean)
    trace = []
    for step in range(10):
        adv = adv.detach().requires_grad_(True)
        losses = [
            F.l1_loss(feature(encoder, normalize(adv), kind) * 0.9999, reference)
            for encoder, kind, reference in zip(encoders, feature_kinds, references)
        ]
        gradients = [
            normalized_gradient(loss, adv, retain_graph=index < len(losses) - 1)
            for index, loss in enumerate(losses)
        ]
        if mode == "mean":
            gradient = torch.stack(gradients).mean(0)
            diagnostics = {"dual_weight_left": 0.5, "dual_value": 0.0}
        elif mode == "maxmin":
            if len(gradients) == 2:
                gradient, robust = robust_linf_direction(gradients[0], gradients[1])
                weights = [robust["dual_weight_base"], 1.0 - robust["dual_weight_base"]]
                dual_value = robust["dual_value"]
            else:
                gradient, weight_tensor, dual_value = simplex_maxmin_direction(gradients)
                weights = [float(value) for value in weight_tensor]
            diagnostics = {"dual_weights": weights, "dual_value": dual_value}
        else:
            raise ValueError(mode)
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = momentum + gradient
        update = (2.0 / 255.0) * momentum.sign()
        if step == 0 and prior is not None:
            update = 2.5 * update + prior
        adv = adv.detach() + update
        adv = torch.maximum(torch.minimum(adv, clean + 10.0 / 255.0), clean - 10.0 / 255.0)
        adv = adv.clamp(0.0, 1.0)
        cosine = F.cosine_similarity(
            gradients[0].flatten(1), gradients[-1].flatten(1), dim=1
        ).mean()
        trace.append({
            "step": step,
            "losses": [float(value.detach()) for value in losses],
            "endpoint_cosine": float(cosine),
            **diagnostics,
        })
    return adv, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", choices=("mean", "maxmin"),
                        default=("mean", "maxmin"))
    parser.add_argument("--include-base-early", action="store_true")
    parser.add_argument("--umi-prior", default=None)
    parser.add_argument("--indices", default="0,1,2,3")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset_cls, data_path, classes = DATASETS[args.dataset]
    image_tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    mask_tf = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = dataset_cls(
        args=Namespace(img_size=512), data_path=str(data_path), mode="Test",
        transform=image_tf, transform_msk=mask_tf,
    )
    encoders = []
    feature_kinds = []
    for checkpoint in args.checkpoints:
        model = load_samed_model(
            args.vit, str(SAM[args.vit]), str(ROOT / checkpoint),
            512, classes, device=device,
        )
        encoder = model.sam.image_encoder
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        encoders.append(encoder)
        feature_kinds.append("final")
    if args.include_base_early:
        source, _ = sam_model_registry[args.vit](
            image_size=512, num_classes=classes, checkpoint=str(SAM[args.vit]),
            pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
        )
        base_encoder = source.image_encoder.to(device).eval()
        for parameter in base_encoder.parameters():
            parameter.requires_grad_(False)
        encoders.append(base_encoder)
        feature_kinds.append("early")
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    prior = torch.load(ROOT / args.umi_prior, map_location=device) if args.umi_prior else None
    output = ROOT / args.out
    records = []
    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        sample = dataset[index]
        case = str(sample["image_meta_dict"]["filename_or_obj"])
        clean = sample["image"].float()[None].to(device)
        for mode in args.modes:
            adv, trace = attack(encoders, feature_kinds, clean, normalize, prior, mode)
            candidate = mode.upper()
            record = {
                "index": index,
                "case": case,
                "candidate": candidate,
                "target_checkpoint_used": False,
                "trajectory_checkpoints": args.checkpoints,
                "feature_kinds": feature_kinds,
                "linf_255": save_candidate(output / candidate, case, adv, clean),
                "trace": trace,
            }
            records.append(record)
            print(json.dumps(record, indent=2), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "generation.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
