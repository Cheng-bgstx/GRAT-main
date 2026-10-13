#!/usr/bin/env python3
"""Select a per-slice attack portfolio by independent PEFT proxy task risk."""
from __future__ import annotations

import argparse
import json
import shutil
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
    load_samed_model,
    reset_seed,
)


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(image)[None, None].repeat(1, 3, 1, 1).to(device)


@torch.no_grad()
def samed_risk(model, image: torch.Tensor, label: torch.Tensor) -> float:
    logits = model(image, True, 512)["masks"]
    logits = F.interpolate(logits, label.shape[-2:], mode="bilinear", align_corners=False)
    return float(F.cross_entropy(logits.float(), label.long()))


@torch.no_grad()
def medsa_expected_risk(model, prompt_module, image: torch.Tensor,
                        label: torch.Tensor, seeds: list[int]) -> float:
    embedding = model.image_encoder(model.preprocess(image))
    losses = []
    for seed in seeds:
        reset_seed(seed)
        _unused, points, _masks, point_labels = prompt_module.generate_click_prompt(
            image, label
        )
        if points.dim() == 3 and points.shape[1] == 2:
            points = points.permute(0, 2, 1)
        prompt = (points.float().to(image.device), point_labels.int().to(image.device))
        sparse, dense = model.prompt_encoder(points=prompt, boxes=None, masks=None)
        logits, _ = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        logits = F.interpolate(logits, label.shape[-2:], mode="bilinear", align_corners=False)
        losses.append(F.cross_entropy(logits.float(), label.long()))
    return float(torch.stack(losses).mean())


def fractional_ranks(values: list[float]) -> list[float]:
    values_array = np.asarray(values)
    order = np.argsort(values_array, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and np.isclose(
            values_array[order[end]], values_array[order[start]], rtol=1e-7, atol=1e-9
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return (ranks / max(1, len(values) - 1)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--samed-proxy", required=True)
    parser.add_argument("--medsa-proxy", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--prompt-seeds", type=int, nargs="+", default=(24, 43, 91))
    parser.add_argument("--samed-weight", type=float, default=0.25)
    parser.add_argument("--start-slice", type=int, default=0)
    parser.add_argument("--end-slice", type=int, default=None)
    parser.add_argument("--fallback", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 <= args.samed_weight <= 1:
        parser.error("--samed-weight must be in [0, 1]")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    candidate_dirs = [ROOT / path for path in args.candidates]
    fallback = ROOT / args.fallback
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)

    samed = load_samed_model(
        args.vit, str(SAM[args.vit]), str(ROOT / args.samed_proxy), 512, 8,
        device=device,
    ).eval()
    medsa = load_medsa_model(
        args.vit, str(SAM[args.vit]), str(ROOT / args.medsa_proxy), 512, 9,
        device=device, dataset="Synapse",
    ).eval()
    prompt_module = load_medsa_function("Synapse", 9)

    rows = []
    h5_path = ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5"
    with h5py.File(h5_path, "r") as handle:
        depth = len(handle["image"])
        end = depth if args.end_slice is None else min(depth, args.end_slice)
        for slice_index in range(args.start_slice, end):
            available = [path for path in candidate_dirs if (path / f"{slice_index}.png").is_file()]
            if not available:
                source = fallback / f"{slice_index}.png"
                shutil.copy2(source, output / source.name)
                continue
            label = torch.from_numpy(
                handle["label"][slice_index].astype(np.int64)
            )[None].to(device)
            samed_losses = []
            medsa_losses = []
            for directory in available:
                image = load_image(directory / f"{slice_index}.png", device)
                samed_losses.append(samed_risk(samed, image, label))
                medsa_losses.append(medsa_expected_risk(
                    medsa, prompt_module, image, label, list(args.prompt_seeds)
                ))
            samed_ranks = fractional_ranks(samed_losses)
            medsa_ranks = fractional_ranks(medsa_losses)
            scores = [
                args.samed_weight * left + (1.0 - args.samed_weight) * right
                for left, right in zip(samed_ranks, medsa_ranks)
            ]
            selected = max(range(len(available)), key=lambda index: scores[index])
            source = available[selected] / f"{slice_index}.png"
            shutil.copy2(source, output / source.name)
            row = {
                "slice": slice_index,
                "selected": str(available[selected].relative_to(ROOT)),
                "samed_losses": samed_losses,
                "medsa_expected_losses": medsa_losses,
                "joint_rank_scores": scores,
                "target_checkpoints_used": False,
                "prompt_marginalized_selection_only": True,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    for slice_index in range(depth):
        destination = output / f"{slice_index}.png"
        if not destination.exists():
            shutil.copy2(fallback / destination.name, destination)
    (output.parent / f"{output.name}_selection.json").write_text(json.dumps({
        "case": args.case,
        "samed_proxy": args.samed_proxy,
        "medsa_proxy": args.medsa_proxy,
        "prompt_seeds": args.prompt_seeds,
        "samed_weight": args.samed_weight,
        "candidates": args.candidates,
        "fallback": args.fallback,
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
