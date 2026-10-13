#!/usr/bin/env python3
"""Diagnose prompt-free encoder gradient transfer across PEFT descendants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]

from evaluate import load_medsa_model, load_sam_encoder, load_samed_model


SAM = {
    "vit_b": ROOT / "checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": ROOT / "checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": ROOT / "checkpoints/sam_vit_h_4b8939.pth",
}


def capture_blocks(encoder, image):
    outputs, handles = [], []
    for block in encoder.blocks:
        handles.append(block.register_forward_hook(lambda _m, _i, out: outputs.append(out[0] if isinstance(out, tuple) else out)))
    try:
        encoder(image)
    finally:
        for handle in handles:
            handle.remove()
    return outputs


def layer_indices(depth: int):
    pre = int(depth * 0.9)
    return {
        "early2": list(range(2)),
        "early4": list(range(4)),
        "mixed6": list(range(3)) + list(range(pre - 3, pre)),
        "late6": list(range(pre - 6, pre)),
        "final2": list(range(pre - 2, pre)),
    }


def probe_loss(adv_layers, clean_layers, indices, probes):
    residuals = []
    for index, probe in zip(indices, probes):
        clean = clean_layers[index].detach()
        scale = clean.abs().mean().clamp_min(1e-6)
        residuals.append(((adv_layers[index] * 0.999 - clean) - 2.0 * probe) / scale)
    return torch.stack(residuals).mean(0).abs().mean()


def gradient_for(encoder, clean_input, attack_input, indices, probes):
    with torch.no_grad():
        clean_layers = capture_blocks(encoder, clean_input)
    variable = attack_input.detach().clone().requires_grad_(True)
    adv_layers = capture_blocks(encoder, variable)
    loss = probe_loss(adv_layers, clean_layers, indices, probes)
    gradient = torch.autograd.grad(loss, variable)[0]
    return gradient / gradient.abs().mean().clamp_min(1e-12), float(loss.detach()), clean_layers


def cosine(left, right):
    return float(F.cosine_similarity(left.flatten(1), right.flatten(1), dim=1).mean())


def polar_tangent_responses(gradient, image):
    """Return first-order responses to relative amplitude and phase changes.

    For dX = X (da + i dphi), Parseval gives the two real coefficient
    sensitivities Re(conj(G) X) and -Im(conj(G) X), respectively.
    """
    spectrum = torch.fft.fft2(image.float(), norm="ortho")
    gradient_spectrum = torch.fft.fft2(gradient.float(), norm="ortho")
    cross = gradient_spectrum.conj() * spectrum
    return cross.real, -cross.imag


def masked_cosine(left, right, mask):
    expanded = mask[None, None].expand_as(left)
    return cosine(left.masked_select(expanded)[None], right.masked_select(expanded)[None])


def polar_alignment(gradients, image):
    height, width = image.shape[-2:]
    fy = torch.fft.fftfreq(height, device=image.device)
    fx = torch.fft.fftfreq(width, device=image.device)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / np.sqrt(0.5)
    bands = {
        "low_0_0.2": radius < 0.2,
        "mid_0.05_0.4": (radius >= 0.05) & (radius < 0.4),
        "high_0.2_1": radius >= 0.2,
        "all": torch.ones_like(radius, dtype=torch.bool),
    }
    responses = {
        name: polar_tangent_responses(gradient, image)
        for name, gradient in gradients.items()
    }
    result = {}
    for band_name, mask in bands.items():
        result[band_name] = {}
        for coordinate, coordinate_index in (("log_amplitude", 0), ("phase", 1)):
            source = responses["source"][coordinate_index]
            result[band_name][coordinate] = {
                "cos_source_samed": masked_cosine(
                    source, responses["SAMed"][coordinate_index], mask
                ),
                "cos_source_medsa": masked_cosine(
                    source, responses["MedSA"][coordinate_index], mask
                ),
                "cos_samed_medsa": masked_cosine(
                    responses["SAMed"][coordinate_index],
                    responses["MedSA"][coordinate_index], mask,
                ),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--candidate", default="testset/pasd_exact_gate/AG/case0001/123.png")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="testset/encoder_gradient_alignment.json")
    args = parser.parse_args()
    device = torch.device(args.device)

    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean_np = handle["image"][args.slice].astype(np.float32)
    candidate_np = np.asarray(Image.open(ROOT / args.candidate).convert("L"), dtype=np.float32) / 255.0
    clean = torch.from_numpy(clean_np)[None, None].repeat(1, 3, 1, 1).to(device)
    candidate = torch.from_numpy(candidate_np)[None, None].repeat(1, 3, 1, 1).to(device)

    source_model = load_sam_encoder(args.vit, str(SAM[args.vit]), 512, num_classes=8, device=device)
    samed_model = load_samed_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/SAMed_Synapse_{args.vit}.pth"),
        512, 8, device=device,
    )
    medsa_model = load_medsa_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
        512, 9, device=device,
    )
    encoders = {
        "source": (source_model.image_encoder, lambda value: value),
        "SAMed": (samed_model.sam.image_encoder, lambda value: value),
        "MedSA": (medsa_model.image_encoder, medsa_model.preprocess),
    }
    groups = layer_indices(len(source_model.image_encoder.blocks))
    results = {}
    for point_name, point in {
        "initial": clean + 1e-4 * torch.randn_like(clean),
        args.candidate_name: candidate,
    }.items():
        results[point_name] = {}
        for group_name, indices in groups.items():
            torch.manual_seed(1701 + len(indices))
            with torch.no_grad():
                source_clean_layers = capture_blocks(encoders["source"][0], clean)
            probes = [torch.randn_like(source_clean_layers[index]) for index in indices]
            gradients, losses, clean_layers = {}, {}, {}
            for model_name, (encoder, preprocess) in encoders.items():
                gradient, loss, layers = gradient_for(
                    encoder, preprocess(clean), preprocess(point), indices, probes
                )
                gradients[model_name], losses[model_name], clean_layers[model_name] = gradient, loss, layers
            results[point_name][group_name] = {
                "loss": losses,
                "cos_source_samed": cosine(gradients["source"], gradients["SAMed"]),
                "cos_source_medsa": cosine(gradients["source"], gradients["MedSA"]),
                "cos_samed_medsa": cosine(gradients["SAMed"], gradients["MedSA"]),
                "polar_tangent_alignment": polar_alignment(gradients, point),
                "clean_feature_gap_samed": float(torch.stack([
                    (clean_layers["SAMed"][i] - clean_layers["source"][i]).abs().mean()
                    / clean_layers["source"][i].abs().mean().clamp_min(1e-6) for i in indices
                ]).mean()),
                "clean_feature_gap_medsa": float(torch.stack([
                    (clean_layers["MedSA"][i] - clean_layers["source"][i]).abs().mean()
                    / clean_layers["source"][i].abs().mean().clamp_min(1e-6) for i in indices
                ]).mean()),
            }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
