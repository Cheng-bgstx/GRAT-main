#!/usr/bin/env python3
"""Build task-gated PEFT variants using only ISIC Training images and labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from peft_transfer.evaluate_official_slice import ROOT, SAM, load_medsa_model, load_samed_model
from peft_transfer.task_basin import (
    apply_delta, damped_precision, diagonal_empirical_fisher, peft_parameters,
    restore_parameters, sample_delta, split_names,
)


def training_rows():
    path = ROOT / "data/ISIC/ISBI2016_ISIC_Part1_Training_GroundTruth.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["img"]: row for row in rows}


def load_sample(row, device):
    image = Image.open(ROOT / "data/ISIC" / row["img"]).convert("RGB")
    mask = Image.open(ROOT / "data/ISIC" / row["seg"]).convert("L")
    image = transforms.ToTensor()(transforms.Resize((512, 512))(image))[None].to(device)
    mask = transforms.ToTensor()(transforms.Resize(
        (512, 512), interpolation=transforms.InterpolationMode.NEAREST
    )(mask))[None, 0].to(device)
    mask = (mask > 0.5).float()
    coords = (mask[0] > 0).nonzero(as_tuple=False)
    if len(coords):
        center = mask.new_tensor((255.5, 255.5))
        yx = coords[((coords.float() - center).square().sum(1)).argmin()]
        point = image.new_tensor([[[float(yx[1]), float(yx[0])]]])
        point_label = torch.ones((1, 1), device=device, dtype=torch.int)
    else:
        point = image.new_tensor([[[256.0, 256.0]]])
        point_label = torch.zeros((1, 1), device=device, dtype=torch.int)
    return image, mask, point, point_label


def logits_for(model, sample, kind):
    image, _mask, point, point_label = sample
    if kind == "lora":
        mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        logits = model.sam((image - mean) / std, False, 512)["masks"]
    else:
        embedding = model.image_encoder(model.preprocess(image * 255.0))
        sparse, dense = model.prompt_encoder(
            points=(point, point_label), boxes=None, masks=None,
        )
        logits, _ = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
    return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def supervised_nll_and_output(model, sample, kind):
    logits = logits_for(model, sample, kind)
    mask = sample[1]
    if logits.shape[1] == 1:
        foreground = logits[:, 0]
        ce = F.binary_cross_entropy_with_logits(foreground, mask)
        probability = foreground.sigmoid()
        prediction = foreground > 0
    else:
        ce = F.cross_entropy(logits, mask.long())
        probability = logits.softmax(1)[:, 1]
        prediction = logits.argmax(1) == 1
    return ce, probability, prediction


def task_loss_and_dice(model, sample, kind):
    ce, probability, prediction = supervised_nll_and_output(model, sample, kind)
    mask = sample[1]
    soft_dice = (2 * (probability * mask).sum() + 1) / (
        probability.sum() + mask.sum() + 1
    )
    intersection = (prediction * (mask > 0)).sum()
    dice = (2 * intersection + 1) / (prediction.sum() + mask.sum() + 1)
    return ce + (1 - soft_dice), dice


def score(model, samples, kind):
    with torch.no_grad():
        values = [task_loss_and_dice(model, sample, kind) for sample in samples]
    return {
        "risk": float(torch.stack([value[0] for value in values]).mean()),
        "dice": float(torch.stack([value[1] for value in values]).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("lora", "adapter"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--fisher-count", type=int, default=12)
    parser.add_argument("--calibration-count", type=int, default=12)
    parser.add_argument("--damping-ratio", type=float, default=0.01)
    parser.add_argument("--relative-norms", type=float, nargs="+", default=(0.01, 0.03, 0.05))
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 13))
    parser.add_argument("--max-risk-increase", type=float, default=0.05)
    parser.add_argument("--max-dice-drop", type=float, default=0.02)
    parser.add_argument("--max-accepted", type=int, default=2)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.fisher_count < 1 or args.calibration_count < 1:
        parser.error("sample counts must be positive")
    device = torch.device(args.device)
    rows = training_rows()
    train_names, calibration_names = split_names(
        list(rows), args.holdout_fraction, args.split_seed
    )
    fisher_names = train_names[:args.fisher_count]
    calibration_names = calibration_names[:args.calibration_count]
    fisher_samples = [load_sample(rows[name], device) for name in fisher_names]
    calibration_samples = [load_sample(rows[name], device) for name in calibration_names]
    checkpoint = ROOT / args.checkpoint
    if args.kind == "lora":
        model = load_samed_model(
            args.vit, str(SAM[args.vit]), str(checkpoint), 512, 1, device=device
        )
    else:
        model = load_medsa_model(
            args.vit, str(SAM[args.vit]), str(checkpoint), 512, 1,
            device=device, dataset="isic",
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = peft_parameters(model, args.kind)
    for parameter in parameters.values():
        parameter.requires_grad_(True)
    baseline = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    fisher, mean_gradient = diagonal_empirical_fisher(
        model, fisher_samples,
        lambda current, sample: supervised_nll_and_output(current, sample, args.kind)[0],
        parameters, return_mean=True,
    )
    precision, damping = damped_precision(fisher, args.damping_ratio)
    stationarity_dual_norm = sum(
        (mean_gradient[name].square() / precision[name]).sum().item()
        for name in mean_gradient
    ) ** 0.5
    clean = score(model, calibration_samples, args.kind)
    accepted = {"isotropic": [], "fisher": []}
    passed_pairs = {}
    trials = []
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    try:
        for geometry in accepted:
            for relative_norm in args.relative_norms:
                for seed in args.seeds:
                    delta = sample_delta(
                        parameters, precision, relative_norm, seed, geometry
                    )
                    apply_delta(parameters, baseline, delta)
                    positive_metrics = score(model, calibration_samples, args.kind)
                    restore_parameters(parameters, baseline)
                    negative_delta = {name: -value for name, value in delta.items()}
                    apply_delta(parameters, baseline, negative_delta)
                    negative_metrics = score(model, calibration_samples, args.kind)
                    restore_parameters(parameters, baseline)
                    passed = (
                        all(metrics["risk"] <= clean["risk"] + args.max_risk_increase
                            and metrics["dice"] >= clean["dice"] - args.max_dice_drop
                            for metrics in (positive_metrics, negative_metrics))
                    )
                    row = {
                        "geometry": geometry, "relative_norm": relative_norm,
                        "seed": seed, "positive_metrics": positive_metrics,
                        "negative_metrics": negative_metrics, "accepted": passed,
                    }
                    trials.append(row)
                    passed_pairs[(geometry, relative_norm, seed)] = passed
                    print(json.dumps(row), flush=True)
    finally:
        restore_parameters(parameters, baseline)
    common_pairs = sorted((
        (relative_norm, seed) for relative_norm in args.relative_norms
        for seed in args.seeds
        if passed_pairs[("isotropic", relative_norm, seed)]
        and passed_pairs[("fisher", relative_norm, seed)]
    ), reverse=True)[:args.max_accepted]
    for relative_norm, seed in common_pairs:
        for geometry in accepted:
            delta = sample_delta(parameters, precision, relative_norm, seed, geometry)
            for sign in (1, -1):
                signed_delta = {name: sign * value for name, value in delta.items()}
                sign_name = "plus" if sign == 1 else "minus"
                path = output / f"{geometry}_n{relative_norm:g}_s{seed}_{sign_name}.pt"
                torch.save({"delta": signed_delta, "kind": args.kind,
                            "checkpoint": str(args.checkpoint)}, path)
                accepted[geometry].append(str(path.relative_to(ROOT)))
    metadata = {
        "kind": args.kind, "vit": args.vit, "checkpoint": args.checkpoint,
        "split_seed": args.split_seed, "holdout_fraction": args.holdout_fraction,
        "fisher_training_images": fisher_names,
        "calibration_training_images": calibration_names,
        "damping": damping, "clean": clean, "trials": trials, "accepted": accepted,
        "stationarity_dual_norm": stationarity_dual_norm,
        "paired_relative_norm_and_seed": common_pairs,
        "uses_test_images_or_labels": False,
        "uses_victim_checkpoint": False,
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"clean": clean, "accepted": accepted}, indent=2))


if __name__ == "__main__":
    main()
