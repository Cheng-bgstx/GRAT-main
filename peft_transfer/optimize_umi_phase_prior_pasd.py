#!/usr/bin/env python3
"""Learn a bounded UMI phase correction using the inference-time PASD risk."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from peft_transfer.core import (
    AttackConfig,
    _unpack_encoder,
    phase_amplitude_stochastic_disruption,
)
from peft_transfer.generate import CHECKPOINTS, ROOT
from peft_transfer.medsa_surrogate import IntermediateEncoderWrapper, load_medsa_base_surrogate
from peft_transfer.optimize_umi_phase_prior import select_proxy_images, synthesize_phase_prior
from segment_anything import sam_model_registry


def pasd_risk(final_adv, final_clean, layers_adv, layers_clean, cfg):
    """The exact source risk optimized at every online UMI-AG iteration."""
    feature = F.l1_loss(final_adv * 0.9999, final_clean.detach())
    pasd = phase_amplitude_stochastic_disruption(layers_adv, layers_clean, cfg)
    return feature + pasd, feature, pasd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior",
        default="common_perb/common_perturbation_meta_0.025sp_bound2_it7batchit41upsize_ls0.0025mulmd0_10000.pth",
    )
    parser.add_argument("--output", default="testset/optimized_phase_prior_pasd/phase_prior.pth")
    parser.add_argument("--trace", default="testset/optimized_phase_prior_pasd/train_trace.json")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--proxy-images", type=int, default=8)
    parser.add_argument("--phase-radius", type=float, default=0.20)
    parser.add_argument("--phase-height", type=int, default=16)
    parser.add_argument("--phase-width", type=int, default=9)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--prox-weight", type=float, default=0.15)
    parser.add_argument("--tv-weight", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    prior = torch.load(ROOT / args.prior, map_location=device).float()
    prior_spectrum = torch.fft.rfft2(prior, dim=(-2, -1), norm="ortho")
    amplitude = prior_spectrum.abs().detach()
    phase = torch.angle(prior_spectrum).detach()
    peak_bound = prior.abs().max().detach()
    phase_parameter = torch.zeros(
        1, prior.shape[1], args.phase_height, args.phase_width,
        device=device, requires_grad=True,
    )
    optimizer = torch.optim.Adam([phase_parameter], lr=args.lr)

    canonical, _ = sam_model_registry["vit_b"](
        image_size=512, num_classes=8, checkpoint=str(CHECKPOINTS["vit_b"]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    canonical_encoder = canonical.image_encoder.to(device).eval()
    compatible_model = load_medsa_base_surrogate(
        "vit_b", str(CHECKPOINTS["vit_b"]), 512, 9, device
    )
    compatible_encoder = IntermediateEncoderWrapper(compatible_model.image_encoder).eval()
    for model in (canonical_encoder, compatible_encoder):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    proxies = select_proxy_images(args.proxy_images)
    cfg = AttackConfig(
        pasd_mode="shuffled_white", pasd_samples=2,
        pasd_probe_scale=2.0, pasd_region="late", pasd_layer_count=6,
    )
    trace = []
    for step in range(args.steps):
        path, array = proxies[step % len(proxies)]
        image = torch.from_numpy(array)[None, None].repeat(1, 3, 1, 1).to(device)
        with torch.no_grad():
            canonical_clean, canonical_clean_layers = _unpack_encoder(canonical_encoder(image))
            compatible_input = compatible_model.preprocess(image)
            compatible_clean, compatible_clean_layers = _unpack_encoder(
                compatible_encoder(compatible_input)
            )

        delta, correction = synthesize_phase_prior(
            amplitude, phase, phase_parameter, args.phase_radius,
            prior.shape[-2:], peak_bound,
        )
        adv = (image + delta).clamp(0.0, 1.0)

        # Identical seeds make the robust comparison depend on architecture,
        # rather than on different Monte Carlo draws.
        torch.manual_seed(args.seed + step)
        canonical_adv, canonical_layers = _unpack_encoder(canonical_encoder(adv))
        canonical_risk, canonical_feature, canonical_pasd = pasd_risk(
            canonical_adv, canonical_clean, canonical_layers, canonical_clean_layers, cfg
        )
        torch.manual_seed(args.seed + step)
        compatible_adv, compatible_layers = _unpack_encoder(
            compatible_encoder(compatible_model.preprocess(adv))
        )
        compatible_risk, compatible_feature, compatible_pasd = pasd_risk(
            compatible_adv, compatible_clean, compatible_layers, compatible_clean_layers, cfg
        )

        risks = torch.stack([canonical_risk, compatible_risk])
        temperature = max(float(args.temperature), 1e-6)
        robust_risk = -temperature * torch.logsumexp(-risks / temperature, dim=0)
        cosine = F.cosine_similarity(delta.flatten(1), prior.flatten(1), dim=1).mean()
        tv = (correction[..., 1:, :] - correction[..., :-1, :]).abs().mean()
        tv = tv + (correction[..., :, 1:] - correction[..., :, :-1]).abs().mean()
        objective = robust_risk - args.prox_weight * (1.0 - cosine) - args.tv_weight * tv
        optimizer.zero_grad(set_to_none=True)
        (-objective).backward()
        optimizer.step()

        record = {
            "step": step, "proxy": Path(path).name,
            "canonical_risk": float(canonical_risk.detach()),
            "compatible_risk": float(compatible_risk.detach()),
            "canonical_feature": float(canonical_feature.detach()),
            "compatible_feature": float(compatible_feature.detach()),
            "canonical_pasd": float(canonical_pasd.detach()),
            "compatible_pasd": float(compatible_pasd.detach()),
            "robust_risk": float(robust_risk.detach()),
            "prior_cosine": float(cosine.detach()),
            "phase_rms": float(correction.detach().square().mean().sqrt()),
            "phase_max": float(correction.detach().abs().max()),
        }
        trace.append(record)
        if step % 8 == 0 or step + 1 == args.steps:
            print(json.dumps(record), flush=True)

    optimized, correction = synthesize_phase_prior(
        amplitude, phase, phase_parameter, args.phase_radius,
        prior.shape[-2:], peak_bound,
    )
    output = ROOT / args.output
    trace_path = ROOT / args.trace
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(optimized.detach().cpu(), output)
    optimized_spectrum = torch.fft.rfft2(optimized.detach(), dim=(-2, -1), norm="ortho")
    metadata = {
        "source_only": True,
        "training_objective": "inference_aligned_final_plus_pasd_softmin",
        "proxy_files": [Path(path).name for path, _ in proxies],
        "config": vars(args),
        "final": trace[-1],
        "prior_rms": float(prior.square().mean().sqrt()),
        "optimized_rms": float(optimized.detach().square().mean().sqrt()),
        "optimized_peak": float(optimized.detach().abs().max()),
        "phase_rms": float(correction.detach().square().mean().sqrt()),
        "phase_max": float(correction.detach().abs().max()),
        "amplitude_relative_error": float(
            (optimized_spectrum.abs() - amplitude).norm() / amplitude.norm().clamp_min(1e-12)
        ),
        "trace": trace,
    }
    trace_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps({key: value for key, value in metadata.items() if key != "trace"}, indent=2))


if __name__ == "__main__":
    main()
