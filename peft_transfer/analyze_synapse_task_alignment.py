#!/usr/bin/env python3
"""Compare source-only objectives with a Synapse MedSA task gradient.

The victim gradient is diagnostic only. Attack generation never imports or
queries a fine-tuned model.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from peft_transfer.core import (
    AttackConfig,
    QKSiteCapture,
    _unpack_encoder,
    carrier_stochastic_disruption,
    gradient_robust_disruption,
    patch_boundary_disruption,
    phase_amplitude_stochastic_disruption,
    qk_topology_disruption,
    quadrature_collision_objective,
    spatial_feature_polar_disruption_bhwc,
)
from peft_transfer.evaluate_official_slice import (
    SAM,
    load_medsa_function,
    load_medsa_model,
    reset_seed,
)
from segment_anything import sam_model_registry


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.flatten(1), right.flatten(1), dim=1).mean())


def polar_components(
    gradient: torch.Tensor,
    image: torch.Tensor,
    low: float,
    high: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthogonally split a gradient into amplitude and phase FFT tangents."""
    spectrum = torch.fft.fft2(image.detach().float(), norm="ortho")
    grad_spectrum = torch.fft.fft2(gradient.float(), norm="ortho")
    energy = spectrum.abs().square().clamp_min(1e-8)
    radial = (grad_spectrum.conj() * spectrum).real / energy * spectrum
    tangential = grad_spectrum - radial
    height, width = image.shape[-2:]
    fy = torch.fft.fftfreq(height, device=image.device)
    fx = torch.fft.fftfreq(width, device=image.device)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / (0.5 ** 0.5)
    mask = (radius >= low) & (radius < high if high < 1.0 else radius <= high + 1e-6)
    mask = mask[None, None]
    amplitude = torch.fft.ifft2(radial * mask, norm="ortho").real.to(gradient.dtype)
    phase = torch.fft.ifft2(tangential * mask, norm="ortho").real.to(gradient.dtype)
    return amplitude, phase


def polar_response(gradient: torch.Tensor, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    spectrum = torch.fft.fft2(image.detach().float(), norm="ortho")
    grad_spectrum = torch.fft.fft2(gradient.float(), norm="ortho")
    cross = grad_spectrum.conj() * spectrum
    return cross.real, -cross.imag


def masked_cosine(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask[None, None].expand_as(left)
    left_values = left.masked_select(expanded)[None]
    right_values = right.masked_select(expanded)[None]
    return cosine(left_values, right_values)


def principal_subspace_energy(
    clean_feature: torch.Tensor,
    displacement: torch.Tensor,
    ranks: tuple[int, ...] = (8, 32, 64, 192),
) -> dict[str, dict[str, float]]:
    """Measure displacement energy in clean spatial/channel principal spaces."""
    if clean_feature.ndim != 4 or displacement.shape != clean_feature.shape:
        raise ValueError("principal-subspace diagnostic expects matching 4D features")
    # This diagnostic is applied to SAM's final BCHW image embedding.
    clean_matrix = clean_feature[0].reshape(clean_feature.shape[1], -1).T.float()
    delta_matrix = displacement[0].reshape(displacement.shape[1], -1).T.float()
    clean_matrix = clean_matrix - clean_matrix.mean(0, keepdim=True)
    left, _singular, right_h = torch.linalg.svd(clean_matrix, full_matrices=False)
    total = delta_matrix.square().sum().clamp_min(1e-12)
    result = {"channel": {}, "spatial": {}}
    for rank in ranks:
        channel_rank = min(int(rank), right_h.shape[0])
        channel_basis = right_h[:channel_rank].T
        channel_projection = delta_matrix @ channel_basis @ channel_basis.T
        result["channel"][str(rank)] = float(channel_projection.square().sum() / total)
        spatial_rank = min(int(rank), left.shape[1])
        spatial_basis = left[:, :spatial_rank]
        spatial_projection = spatial_basis @ (spatial_basis.T @ delta_matrix)
        result["spatial"][str(rank)] = float(spatial_projection.square().sum() / total)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-seed", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates", nargs="*", default=[])
    args = parser.parse_args()
    device = torch.device(args.device)

    with h5py.File(
        ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r"
    ) as handle:
        image_np = handle["image"][args.slice].astype("float32")
        label_np = handle["label"][args.slice].astype("int64")
    clean = torch.from_numpy(image_np)[None, None].to(device)
    label = torch.from_numpy(label_np)[None].to(device)

    medsa = load_medsa_model(
        args.vit,
        str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
        512,
        9,
        device=device,
    )
    module = load_medsa_function("Synapse", 9)
    victim_gray = clean.detach().clone().requires_grad_(True)
    victim_image = victim_gray.repeat(1, 3, 1, 1)
    reset_seed(args.prompt_seed)
    _unused, points, _masks, point_labels = module.generate_click_prompt(
        victim_image, label
    )
    if points.dim() == 3 and points.shape[1] == 2:
        points = points.permute(0, 2, 1)
    prompt = (points.float().to(device), point_labels.int().to(device))
    embedding = medsa.image_encoder(medsa.preprocess(victim_image))
    sparse, dense = medsa.prompt_encoder(points=prompt, boxes=None, masks=None)
    logits, _ = medsa.mask_decoder(
        image_embeddings=embedding,
        image_pe=medsa.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
    )
    logits = F.interpolate(logits, (512, 512), mode="bilinear", align_corners=False)
    victim_loss = F.cross_entropy(logits, label.long())
    victim_gradient = torch.autograd.grad(victim_loss, victim_gray)[0].detach()
    image_prior_directions = {
        "global_contrast_contraction": clean.mean(dim=(-2, -1), keepdim=True) - clean,
        "local_mean_9": F.avg_pool2d(clean, 9, stride=1, padding=4) - clean,
        "translate_1_1": torch.roll(clean, (1, 1), dims=(-2, -1)) - clean,
        "translate_4_4": torch.roll(clean, (4, 4), dims=(-2, -1)) - clean,
    }
    image_prior_alignment = {
        name: {
            "cosine_medsa_task": cosine(direction, victim_gradient),
            "signed_step_gain": float((direction.sign() * victim_gradient).mean()),
        }
        for name, direction in image_prior_directions.items()
    }
    del medsa, embedding, sparse, dense, logits
    torch.cuda.empty_cache()

    source, _ = sam_model_registry[args.vit](
        image_size=512,
        num_classes=8,
        checkpoint=str(SAM[args.vit]),
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    encoder = source.image_encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    clean_rgb = clean.repeat(1, 3, 1, 1)
    clean_capture = QKSiteCapture(encoder, 0)
    with torch.no_grad(), clean_capture.active():
        clean_final, clean_layers = _unpack_encoder(encoder(clean_rgb))
    clean_qkv = clean_capture.output.detach()

    torch.manual_seed(args.seed)
    variable = (clean + 1e-4 * torch.randn_like(clean)).requires_grad_(True)
    source_capture = QKSiteCapture(encoder, 0)
    with source_capture.active():
        final, layers = _unpack_encoder(encoder(variable.repeat(1, 3, 1, 1)))
    heads = encoder.blocks[0].attn.num_heads
    cfg = AttackConfig(seed=args.seed, grat_noise_sd=0.5)
    pasd_cfg = AttackConfig(
        seed=args.seed,
        pasd_mode="shuffled_white",
        pasd_samples=2,
        pasd_probe_scale=2.0,
        pasd_region="final",
        pasd_layer_count=2,
    )
    phase, amplitude = spatial_feature_polar_disruption_bhwc(
        layers[0], clean_layers[0]
    )
    objectives = {
        "final_l1": F.l1_loss(final * 0.9999, clean_final),
        "early_block0_l1": F.l1_loss(layers[0] * 0.9999, clean_layers[0]),
        "grat": gradient_robust_disruption(layers, clean_layers, cfg),
        "ads": carrier_stochastic_disruption(layers, clean_layers, cfg),
        "final_pasd": phase_amplitude_stochastic_disruption(
            layers, clean_layers, pasd_cfg
        ),
        "early_spatial_phase": phase,
        "early_spatial_amplitude": amplitude,
        "early_boundary": patch_boundary_disruption(layers[0], clean_layers[0]),
        "qk_topology": qk_topology_disruption(
            source_capture.output, clean_qkv, heads
        ),
        "quadrature_collision": quadrature_collision_objective(final, clean_final),
    }
    rows = {}
    names = list(objectives)
    source_gradients = {}
    for index, name in enumerate(names):
        gradient = torch.autograd.grad(
            objectives[name], variable, retain_graph=index + 1 < len(names)
        )[0].detach()
        source_gradients[name] = gradient
        rows[name] = {
            "loss": float(objectives[name].detach()),
            "gradient_l1": float(gradient.abs().mean()),
            "cosine_medsa_task": cosine(gradient, victim_gradient),
            "linearized_medsa_gain": float((gradient.sign() * victim_gradient).mean()),
        }

    height, width = clean.shape[-2:]
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / (0.5 ** 0.5)
    bands = {
        "0.00_0.05": (0.00, 0.05),
        "0.05_0.10": (0.05, 0.10),
        "0.10_0.20": (0.10, 0.20),
        "0.20_0.40": (0.20, 0.40),
        "0.40_0.70": (0.40, 0.70),
        "0.70_1.00": (0.70, 1.00),
        "0.00_0.20": (0.00, 0.20),
        "0.20_1.00": (0.20, 1.00),
        "0.00_1.00": (0.00, 1.00),
    }
    victim_responses = polar_response(victim_gradient, clean)
    for name, gradient in source_gradients.items():
        source_responses = polar_response(gradient, clean)
        rows[name]["polar"] = {}
        for band_name, (low, high) in bands.items():
            mask = (radius >= low) & (radius < high if high < 1.0 else radius <= high + 1e-6)
            source_components = polar_components(gradient, clean, low, high)
            victim_components = polar_components(victim_gradient, clean, low, high)
            coordinate_rows = {}
            for coordinate, coordinate_index in (("amplitude", 0), ("phase", 1)):
                component = source_components[coordinate_index]
                victim_component = victim_components[coordinate_index]
                coordinate_rows[coordinate] = {
                    "response_cosine": masked_cosine(
                        source_responses[coordinate_index],
                        victim_responses[coordinate_index],
                        mask,
                    ),
                    "projected_cosine": cosine(component, victim_component),
                    "signed_step_gain": float((component.sign() * victim_gradient).mean()),
                    "source_l1": float(component.abs().mean()),
                    "victim_l1": float(victim_component.abs().mean()),
                }
            rows[name]["polar"][band_name] = coordinate_rows

    candidate_rows = {}
    for candidate_path in args.candidates:
        candidate_np = np.asarray(
            Image.open(ROOT / candidate_path).convert("L"), dtype=np.float32
        ) / 255.0
        candidate = torch.from_numpy(candidate_np)[None, None].to(device)
        delta = candidate - clean
        with torch.no_grad():
            candidate_final, candidate_layers = _unpack_encoder(
                encoder(candidate.repeat(1, 3, 1, 1))
            )
        candidate_row = {
            "delta_task_cosine": cosine(delta, victim_gradient),
            "delta_signed_task_gain": float((delta.sign() * victim_gradient).mean()),
            "delta_task_gain": float((delta * victim_gradient).mean()),
            "final_relative_l1": float(
                (candidate_final - clean_final).abs().mean()
                / clean_final.abs().mean().clamp_min(1e-8)
            ),
            "layer_relative_l1": [
                float((adv_layer - clean_layer).abs().mean() / clean_layer.abs().mean().clamp_min(1e-8))
                for adv_layer, clean_layer in zip(candidate_layers, clean_layers)
            ],
            "final_principal_subspace_energy": principal_subspace_energy(
                clean_final, candidate_final - clean_final
            ),
            "polar": {},
            "source_gradient_alignment": {
                name: {
                    "cosine": cosine(delta, gradient),
                    "signed_gain": float((delta.sign() * gradient).mean()),
                    "gain": float((delta * gradient).mean()),
                }
                for name, gradient in source_gradients.items()
            },
        }
        delta_responses = polar_response(delta, clean)
        for band_name, (low, high) in bands.items():
            mask = (radius >= low) & (radius < high if high < 1.0 else radius <= high + 1e-6)
            delta_components = polar_components(delta, clean, low, high)
            victim_components = polar_components(victim_gradient, clean, low, high)
            candidate_row["polar"][band_name] = {}
            for coordinate, coordinate_index in (("amplitude", 0), ("phase", 1)):
                component = delta_components[coordinate_index]
                candidate_row["polar"][band_name][coordinate] = {
                    "response_cosine": masked_cosine(
                        delta_responses[coordinate_index],
                        victim_responses[coordinate_index],
                        mask,
                    ),
                    "projected_cosine": cosine(component, victim_components[coordinate_index]),
                    "task_gain": float((component * victim_gradient).mean()),
                    "l1": float(component.abs().mean()),
                }
        candidate_rows[candidate_path] = candidate_row
    result = {
        "protocol": vars(args),
        "victim_gradients_diagnostic_only": True,
        "victim_loss": float(victim_loss.detach()),
        "image_prior_alignment": image_prior_alignment,
        "objectives": rows,
        "candidates": candidate_rows,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
