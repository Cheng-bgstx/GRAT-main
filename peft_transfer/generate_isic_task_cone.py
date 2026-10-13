#!/usr/bin/env python3
"""Small ISIC gate for pseudo-task common ascent across trained PEFT proxies.

Only public proxy checkpoints and unlabeled test images are loaded. The clean
proxy predictions define fixed per-model pseudo labels; victim checkpoints and
test masks are never read during generation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from peft_transfer.common_ascent import common_ascent_direction
from peft_transfer.evaluate_official_slice import ROOT, SAM, load_medsa_model, load_samed_model
from peft_transfer.generate_2d_checkpoint_proxy import save_candidate


def isic_image(index: int):
    csv_path = ROOT / "data/ISIC/ISBI2016_ISIC_Part1_Test_GroundTruth.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    path = ROOT / "data/ISIC" / rows[index]["img"]
    image = Image.open(path).convert("RGB")
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return path.stem, image_transform(image)[None]


def samed_logits(model, image):
    mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return model.sam((image - mean) / std, False, 512)["masks"]


def medsa_logits(model, image):
    embedding = model.image_encoder(model.preprocess(image * 255.0))
    point = image.new_tensor([[[256.0, 256.0]]])
    labels = torch.ones((1, 1), device=image.device, dtype=torch.int)
    sparse, dense = model.prompt_encoder(points=(point, labels), boxes=None, masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def pseudo_task_loss(logits, labels):
    if logits.shape[1] == 1:
        per_pixel = F.binary_cross_entropy_with_logits(logits[:, 0], labels.float(), reduction="none")
    else:
        per_pixel = F.cross_entropy(logits, labels.long(), reduction="none")
    present = [per_pixel[labels == class_id].mean() for class_id in torch.unique(labels)]
    return torch.stack(present).mean()


def attack(models, clean, mode, steps=10, step_size=2 / 255, epsilon=10 / 255):
    functions = [samed_logits, samed_logits, medsa_logits]
    with torch.no_grad():
        labels = [
            (logits[:, 0] > 0).long() if logits.shape[1] == 1 else logits.argmax(1)
            for logits in (function(model, clean) for function, model in zip(functions, models))
        ]
    adv = clean.clone()
    trace = []
    for step in range(steps):
        adv = adv.detach().requires_grad_(True)
        gradients = []
        losses = []
        for function, model, label in zip(functions, models, labels):
            loss = pseudo_task_loss(function(model, adv), label)
            gradient = torch.autograd.grad(loss, adv)[0].detach()
            gradients.append(gradient)
            losses.append(float(loss.detach()))
        normalized = [gradient / gradient.norm().clamp_min(1e-30) for gradient in gradients]
        if mode == "cone":
            direction, weights, margins = common_ascent_direction(
                gradients,
                feasible_low=adv.detach() > (clean - epsilon).clamp_min(0) + 1e-7,
                feasible_high=adv.detach() < (clean + epsilon).clamp_max(1) - 1e-7,
            )
        elif mode == "weighted":
            direction = (0.375 * normalized[0] + 0.375 * normalized[1] + 0.25 * normalized[2]).sign()
            weights = adv.new_tensor((0.375, 0.375, 0.25))
            margins = torch.stack([(gradient * direction).sum() for gradient in normalized])
        elif mode == "lora_only":
            direction = (normalized[0] + normalized[1]).sign()
            weights = adv.new_tensor((0.5, 0.5, 0.0))
            margins = torch.stack([(gradient * direction).sum() for gradient in normalized])
        elif mode == "adapter_only":
            direction = normalized[2].sign()
            weights = adv.new_tensor((0.0, 0.0, 1.0))
            margins = torch.stack([(gradient * direction).sum() for gradient in normalized])
        else:
            raise ValueError(mode)
        proposed = (adv.detach() + step_size * direction).clamp(0, 1)
        adv = torch.maximum(torch.minimum(proposed, clean + epsilon), clean - epsilon).detach()
        trace.append({
            "step": step,
            "pseudo_task_losses": losses,
            "weights": [float(value) for value in weights],
            "first_order_margins": [float(value) for value in margins],
            "min_margin": float(margins.min()),
            "nonzero_update_fraction": float((direction != 0).float().mean()),
        })
    return adv, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--lora-proxies", nargs=2, required=True)
    parser.add_argument("--adapter-proxy", required=True)
    parser.add_argument("--indices", default="0,1")
    parser.add_argument(
        "--modes", nargs="+",
        choices=("cone", "weighted", "lora_only", "adapter_only"),
        default=("cone", "weighted"),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    device = torch.device(args.device)
    paths = [ROOT / path for path in (*args.lora_proxies, args.adapter_proxy)]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError([str(path) for path in paths if not path.is_file()])
    models = [
        load_samed_model(args.vit, str(SAM[args.vit]), str(path), 512, 1, device=device)
        for path in paths[:2]
    ]
    models.append(load_medsa_model(
        args.vit, str(SAM[args.vit]), str(paths[2]), 512, 1,
        device=device, dataset="isic",
    ))
    for model in models:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    output = ROOT / args.out
    metadata_path = output / "generation.json"
    records = json.loads(metadata_path.read_text()) if metadata_path.exists() else []
    for index in (int(value) for value in args.indices.split(",") if value.strip()):
        case, clean = isic_image(index)
        clean = clean.to(device)
        for mode in args.modes:
            adv, trace = attack(models, clean, mode)
            distance = save_candidate(output / mode.upper(), case, adv, clean)
            if distance > 10.0001:
                raise ValueError(f"{case} {mode}: budget exceeded ({distance}/255)")
            record = {
                "index": index, "case": case, "mode": mode,
                "proxy_checkpoints": [str(path.relative_to(ROOT)) for path in paths],
                "uses_test_ground_truth": False,
                "uses_victim_checkpoint": False,
                "proxy_objective": "clean-prediction pseudo-task loss",
                "linf_255": distance, "trace": trace,
            }
            records = [row for row in records if (row["index"], row["mode"]) != (index, mode)]
            records.append(record)
            output.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(records, indent=2))
            print(json.dumps({key: value for key, value in record.items() if key != "trace"}), flush=True)


if __name__ == "__main__":
    main()
