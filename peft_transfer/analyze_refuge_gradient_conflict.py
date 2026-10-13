#!/usr/bin/env python3
"""Diagnostic source/Med-SA gradient conflict on fixed REFUGE cases.

Victim gradients are analysis-only and are never consumed by attack generation.
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

from datasets.refuge import REFUGE
from peft_transfer.evaluate_official_slice import (
    SAM, load_medsa_function, load_medsa_model, reset_seed,
)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="0,133,266,399")
    parser.add_argument("--source-gradient-dir", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
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
    dataset = REFUGE(
        args=Namespace(img_size=512), data_path=str(ROOT / "data/REFUGE-Multirater"),
        mode="Test", transform=image_tf, transform_msk=mask_tf,
    )
    candidate_dirs = dict(spec.split("=", 1) for spec in args.candidates)
    medsa = load_medsa_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/MSA_REFUGE_{args.vit}.pth"),
        512, 2, device=device, dataset="refuge",
    )
    module = load_medsa_function("refuge", 2)
    rows = []
    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        sample = dataset[index]
        case = str(sample["image_meta_dict"]["filename_or_obj"])
        clean = sample["image"][None].to(device)
        masks = sample["label"][None].to(device)
        victim_input = (clean * 255.0).detach().requires_grad_(True)
        reset_seed(args.prompt_seed)
        _images, points, prompt_masks, point_labels = module.generate_click_prompt(
            victim_input, masks
        )
        if points.dim() == 3 and points.shape[1] == 2:
            points = points.permute(0, 2, 1)
        prompt = (points.float().to(device), point_labels.int().to(device))
        image_embedding = medsa.image_encoder(medsa.preprocess(victim_input))
        sparse, dense = medsa.prompt_encoder(points=prompt, boxes=None, masks=None)
        prediction, _ = medsa.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=medsa.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        prediction = F.interpolate(prediction, size=(512, 512), mode="bilinear", align_corners=False)
        victim_loss = F.binary_cross_entropy_with_logits(prediction, prompt_masks)
        victim_gradient = torch.autograd.grad(victim_loss, victim_input)[0].detach() * 255.0

        source_record = torch.load(
            ROOT / args.source_gradient_dir / f"{case}.grad.pt", map_location="cpu"
        )
        source_gradient = source_record["final"].to(device)
        row = {
            "case": case,
            "victim_loss": float(victim_loss.detach()),
            "cosine_all": cosine(source_gradient, victim_gradient),
            "cosine_by_channel": [
                cosine(source_gradient[:, channel], victim_gradient[:, channel])
                for channel in range(3)
            ],
            "candidate_victim_gain": {},
            "candidate_source_gain": {},
        }
        clean_np = clean[0].detach().cpu().numpy()
        for name, folder in candidate_dirs.items():
            adv = np.load(ROOT / folder / f"{case}.npy").astype(np.float32).transpose(2, 0, 1)
            delta = torch.from_numpy(adv - clean_np)[None].to(device)
            row["candidate_victim_gain"][name] = float((victim_gradient * delta).sum())
            row["candidate_source_gain"][name] = float((source_gradient * delta).sum())
        rows.append(row)

    summary = {
        "protocol": vars(args),
        "rows": rows,
        "mean_cosine_all": float(np.mean([row["cosine_all"] for row in rows])),
        "mean_cosine_by_channel": np.mean(
            [row["cosine_by_channel"] for row in rows], axis=0
        ).tolist(),
        "mean_candidate_victim_gain": {
            name: float(np.mean([row["candidate_victim_gain"][name] for row in rows]))
            for name in candidate_dirs
        },
        "mean_candidate_source_gain": {
            name: float(np.mean([row["candidate_source_gain"][name] for row in rows]))
            for name in candidate_dirs
        },
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
