#!/usr/bin/env python3
"""Measure source layer gradients against a 2D Med-SA task gradient.

Victim gradients are used only for post-hoc mechanism analysis.  Attack
generation remains source-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.isic import ISIC2016
from datasets.refuge import REFUGE
from peft_transfer.core import _unpack_encoder
from peft_transfer.evaluate_official_slice import (
    SAM, load_medsa_function, load_medsa_model, reset_seed,
)
from segment_anything import sam_model_registry


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.flatten(1), right.flatten(1), dim=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ISIC", "REFUGE"), default="REFUGE")
    parser.add_argument("--indices", default="0,133,266,399")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--vit", default="vit_b", choices=SAM)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-seed", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)

    image_tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    mask_tf = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    dataset_cls = REFUGE if args.dataset == "REFUGE" else ISIC2016
    data_path = ROOT / ("data/REFUGE-Multirater" if args.dataset == "REFUGE" else "data/ISIC")
    classes = 2 if args.dataset == "REFUGE" else 1
    medsa_name = args.dataset.lower()
    dataset = dataset_cls(
        args=Namespace(img_size=512), data_path=str(data_path),
        mode="Test", transform=image_tf, transform_msk=mask_tf,
    )
    source, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=classes, checkpoint=str(SAM[args.vit]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    source_encoder = source.image_encoder.to(device).eval()
    for parameter in source_encoder.parameters():
        parameter.requires_grad_(False)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    medsa = load_medsa_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/MSA_{args.dataset}_{args.vit}.pth"),
        512, classes, device=device, dataset=medsa_name,
    )
    module = load_medsa_function(medsa_name, classes)
    rows = []
    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        sample = dataset[index]
        case = str(sample["image_meta_dict"]["filename_or_obj"])
        clean = sample["image"][None].to(device)
        adv_np = np.load(ROOT / args.candidate / f"{case}.npy").astype(np.float32)
        adv = torch.from_numpy(adv_np.transpose(2, 0, 1))[None].to(device).requires_grad_(True)
        masks = sample["label"][None].to(device)

        victim_input = adv * 255.0
        reset_seed(args.prompt_seed)
        _images, points, prompt_masks, point_labels = module.generate_click_prompt(victim_input, masks)
        if points.dim() == 3 and points.shape[1] == 2:
            points = points.permute(0, 2, 1)
        prompt = (points.float().to(device), point_labels.int().to(device))
        victim_embedding = medsa.image_encoder(medsa.preprocess(victim_input))
        sparse, dense = medsa.prompt_encoder(points=prompt, boxes=None, masks=None)
        prediction, _ = medsa.mask_decoder(
            image_embeddings=victim_embedding,
            image_pe=medsa.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=classes > 1,
        )
        prediction = F.interpolate(prediction, size=(512, 512), mode="bilinear", align_corners=False)
        victim_loss = F.binary_cross_entropy_with_logits(prediction, prompt_masks)
        victim_gradient = torch.autograd.grad(victim_loss, adv, retain_graph=False)[0].detach()

        with torch.no_grad():
            _clean_final, clean_layers = _unpack_encoder(source_encoder(normalize(clean)))
        _adv_final, adv_layers = _unpack_encoder(source_encoder(normalize(adv)))
        layer_cosines = []
        channel_cosines = []
        for layer_index, (adv_layer, clean_layer) in enumerate(zip(adv_layers, clean_layers)):
            loss = F.l1_loss(adv_layer * 0.9999, clean_layer.detach())
            gradient = torch.autograd.grad(
                loss, adv, retain_graph=layer_index + 1 < len(adv_layers)
            )[0].detach()
            layer_cosines.append(cosine(gradient, victim_gradient))
            channel_cosines.append([
                cosine(gradient[:, channel:channel + 1], victim_gradient[:, channel:channel + 1])
                for channel in range(3)
            ])
        rows.append({
            "case": case, "victim_loss": float(victim_loss.detach()),
            "layer_cosines": layer_cosines, "channel_cosines": channel_cosines,
        })

    layer_matrix = np.asarray([row["layer_cosines"] for row in rows])
    channel_matrix = np.asarray([row["channel_cosines"] for row in rows])
    result = {
        "protocol": vars(args), "rows": rows,
        "mean_layer_cosines": layer_matrix.mean(0).tolist(),
        "positive_case_fraction": (layer_matrix > 0).mean(0).tolist(),
        "mean_channel_cosines": channel_matrix.mean(0).tolist(),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
