#!/usr/bin/env python3
"""Measure H1 path-response geometry on frozen source-only candidates.

This command loads only public base SAM, clean images, and already generated
candidate images. It never loads a medical target or target result file.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]

from datasets.isic import ISIC2016
from evaluate import load_sam_encoder
from peft_transfer.evaluate_official_2d_gate import load_candidate
from peft_transfer.evaluate_official_slice import SAM

from autoresearch_med.diagnostics.mechanism import (
    discrete_path_statistics,
    path_response_statistics,
)


def parse_candidate(spec: str) -> tuple[str, Path]:
    name, folder = spec.split("=", 1)
    return name, ROOT / folder


def capture_features(encoder, image: torch.Tensor, layers: list[int]) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for index in layers:
        def hook(_module, _inputs, output, index=index):
            value = output[0] if isinstance(output, tuple) else output
            captured[str(index)] = value.detach()

        handles.append(encoder.blocks[index].register_forward_hook(hook))
    try:
        with torch.no_grad():
            final = encoder(image)
            final = final[0] if isinstance(final, tuple) else final
            captured["neck"] = final.detach()
    finally:
        for handle in handles:
            handle.remove()
    return captured


def path_features(
    encoder,
    clean: torch.Tensor,
    delta: torch.Tensor,
    samples: list[float],
    derivative_step: float,
    normalize,
    layers: list[int],
) -> tuple[dict[float, dict[str, torch.Tensor]], dict[float, dict[str, torch.Tensor]]]:
    required = set(samples)
    for sample in samples:
        if sample == 0.0:
            required.add(min(1.0, sample + derivative_step))
        elif sample == 1.0:
            required.add(max(0.0, sample - derivative_step))
        else:
            required.update((sample - derivative_step, sample + derivative_step))
    features = {
        value: capture_features(encoder, normalize(clean + value * delta), layers)
        for value in sorted(required)
    }
    responses: dict[float, dict[str, torch.Tensor]] = {}
    for sample in samples:
        if sample == 0.0:
            left, right = sample, sample + derivative_step
        elif sample == 1.0:
            left, right = sample - derivative_step, sample
        else:
            left, right = sample - derivative_step, sample + derivative_step
        responses[sample] = {
            layer: (features[right][layer] - features[left][layer]) / (right - left)
            for layer in features[left]
        }
    return features, responses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--layers", default="0,5,11")
    parser.add_argument("--path-samples", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--derivative-step", type=float, default=0.02)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    samples = [float(value) for value in args.path_samples.split(",") if value.strip()]
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    if samples[0] != 0.0 or samples[-1] != 1.0:
        raise ValueError("path samples must include endpoints 0 and 1")
    if args.derivative_step <= 0 or args.derivative_step >= 0.25:
        raise ValueError("derivative step must be in (0, 0.25)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    image_tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    dataset = ISIC2016(
        args=Namespace(img_size=512),
        data_path=str(ROOT / "data/ISIC"),
        mode="Test",
        transform=image_tf,
        transform_msk=transforms.ToTensor(),
    )
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    source = load_sam_encoder("vit_b", str(SAM["vit_b"]), 512, 1, device)
    source.eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)

    candidates = dict(parse_candidate(value) for value in args.candidates)
    rows = []
    for index in indices:
        sample = dataset[index]
        case = str(sample["image_meta_dict"]["filename_or_obj"])
        clean = sample["image"].float()[None].to(device)
        for candidate_name, folder in candidates.items():
            adversarial = torch.from_numpy(load_candidate(folder, case, ".npy"))[None].to(device)
            delta = adversarial - clean
            linf_255 = float(delta.abs().max().item() * 255.0)
            if linf_255 > 10.0001:
                raise ValueError(f"{case}/{candidate_name} exceeds budget: {linf_255}")
            features, responses = path_features(
                source.image_encoder,
                clean,
                delta,
                samples,
                args.derivative_step,
                normalize,
                layers,
            )
            for layer in features[0.0]:
                endpoint = features[1.0][layer] - features[0.0][layer]
                ordered = [responses[value][layer] for value in samples]
                stats = path_response_statistics(ordered, samples, endpoint)
                discrete = discrete_path_statistics(
                    [features[value][layer] for value in samples], samples
                )
                alternating = [
                    value if position % 2 == 0 else -value
                    for position, value in enumerate(ordered)
                ]
                control = path_response_statistics(alternating, samples)
                rows.append({
                    "index": index,
                    "case": case,
                    "candidate": candidate_name,
                    "layer": layer,
                    "linf_255": linf_255,
                    **stats,
                    **discrete,
                    "alternating_sign_control_consistency": control["path_consistency"],
                })

    result = {
        "protocol": {
            "source_only": True,
            "target_access": False,
            "backbone": "vit_b",
            "indices": indices,
            "candidates": list(candidates),
            "layers": [*layers, "neck"],
            "path_samples": samples,
            "derivative_step": args.derivative_step,
            "diagnostic_only": True,
        },
        "rows": rows,
        "summary": {
            "mean_discrete_path_consistency": float(np.mean([
                row["discrete_path_consistency"] for row in rows
            ])),
            "max_telescoping_residual_ratio": float(max(
                row["telescoping_residual_ratio"] for row in rows
            )),
            "mean_path_consistency": float(np.mean([row["path_consistency"] for row in rows])),
            "max_quadrature_residual_ratio": float(max(
                row["quadrature_residual_ratio"] for row in rows
            )),
            "mean_alternating_control_consistency": float(np.mean([
                row["alternating_sign_control_consistency"] for row in rows
            ])),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

