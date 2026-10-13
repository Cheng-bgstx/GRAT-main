#!/usr/bin/env python3
"""Learn a bounded low-dimensional UMI phase correction on source-only proxies."""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from peft_transfer.architecture_rank_attack import compatible_embedding_objective
from peft_transfer.generate import CHECKPOINTS, ROOT
from peft_transfer.medsa_surrogate import load_medsa_base_surrogate
from segment_anything import sam_model_registry


def select_proxy_images(limit: int):
    groups = {}
    for path in glob.glob(str(ROOT / "data/Synapse_preprocess/train_npz/*.npz")):
        case = Path(path).stem.split("_slice", 1)[0]
        groups.setdefault(case, []).append(path)
    selected = []
    for case in sorted(groups):
        paths = sorted(groups[case])
        center = len(paths) // 2
        order = sorted(range(len(paths)), key=lambda index: abs(index - center))
        for index in order:
            data = np.load(paths[index])
            if int((data["label"] > 0).sum()) > 100:
                selected.append((paths[index], data["image"].astype(np.float32)))
                break
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise RuntimeError(f"found only {len(selected)} organ-containing proxy cases")
    return selected


def synthesize_phase_prior(
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    phase_parameter: torch.Tensor,
    phase_radius: float,
    spatial_shape,
    peak_bound: torch.Tensor,
):
    correction = F.interpolate(
        phase_parameter, size=phase.shape[-2:], mode="bilinear", align_corners=False
    )
    correction = float(phase_radius) * torch.tanh(correction)
    spectrum = amplitude * torch.exp(1j * (phase + correction))
    value = torch.fft.irfft2(spectrum, s=spatial_shape, dim=(-2, -1), norm="ortho")
    scale = torch.minimum(value.new_ones(()), peak_bound / value.abs().max().clamp_min(1e-12))
    return value * scale, correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", default="common_perb/common_perturbation_meta_0.025sp_bound2_it7batchit41upsize_ls0.0025mulmd0_10000.pth")
    parser.add_argument("--output", default="testset/optimized_phase_prior/phase_prior.pth")
    parser.add_argument("--trace", default="testset/optimized_phase_prior/train_trace.json")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--proxy-images", type=int, default=12)
    parser.add_argument("--phase-radius", type=float, default=0.15)
    parser.add_argument("--phase-height", type=int, default=32)
    parser.add_argument("--phase-width", type=int, default=17)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--prox-weight", type=float, default=0.1)
    parser.add_argument("--tv-weight", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    prior = torch.load(ROOT / args.prior, map_location=device).float()
    prior_spectrum = torch.fft.rfft2(prior, dim=(-2, -1), norm="ortho")
    amplitude, phase = prior_spectrum.abs().detach(), torch.angle(prior_spectrum).detach()
    peak_bound = prior.abs().max().detach()
    phase_parameter = torch.zeros(
        1, prior.shape[1], args.phase_height, args.phase_width, device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([phase_parameter], lr=args.lr)

    canonical, _ = sam_model_registry["vit_b"](
        image_size=512, num_classes=8, checkpoint=str(CHECKPOINTS["vit_b"]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    canonical_encoder = canonical.image_encoder.to(device).eval()
    compatible = load_medsa_base_surrogate("vit_b", str(CHECKPOINTS["vit_b"]), 512, 9, device)
    for model in (canonical_encoder, compatible.image_encoder):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    proxies = select_proxy_images(args.proxy_images)
    cached = []
    with torch.no_grad():
        for path, array in proxies:
            image = torch.from_numpy(array)[None, None].repeat(1, 3, 1, 1).to(device)
            canonical_clean, _layers = canonical_encoder(image)
            compatible_clean = compatible.image_encoder(compatible.preprocess(image))
            cached.append((path, image, canonical_clean.detach(), compatible_clean.detach()))

    trace = []
    for step in range(args.steps):
        path, image, canonical_clean, compatible_clean = cached[step % len(cached)]
        delta, correction = synthesize_phase_prior(
            amplitude, phase, phase_parameter, args.phase_radius,
            prior.shape[-2:], peak_bound,
        )
        adv = (image + delta).clamp(0.0, 1.0)
        canonical_adv, _layers = canonical_encoder(adv)
        compatible_adv = compatible.image_encoder(compatible.preprocess(adv))
        canonical_gain, _ = compatible_embedding_objective(
            canonical_adv, canonical_clean, "phase_amp"
        )
        compatible_gain, _ = compatible_embedding_objective(
            compatible_adv, compatible_clean, "phase_amp"
        )
        gains = torch.stack([canonical_gain, compatible_gain])
        robust_gain = -args.temperature * torch.logsumexp(-gains / args.temperature, dim=0)
        cosine = F.cosine_similarity(delta.flatten(1), prior.flatten(1), dim=1).mean()
        tv = (correction[..., 1:, :] - correction[..., :-1, :]).abs().mean()
        tv = tv + (correction[..., :, 1:] - correction[..., :, :-1]).abs().mean()
        objective = robust_gain - args.prox_weight * (1.0 - cosine) - args.tv_weight * tv
        optimizer.zero_grad(set_to_none=True)
        (-objective).backward()
        optimizer.step()
        record = {
            "step": step, "proxy": Path(path).name,
            "canonical_gain": float(canonical_gain.detach()),
            "compatible_gain": float(compatible_gain.detach()),
            "robust_gain": float(robust_gain.detach()),
            "prior_cosine": float(cosine.detach()),
            "phase_rms": float(correction.detach().square().mean().sqrt()),
            "phase_max": float(correction.detach().abs().max()),
        }
        trace.append(record)
        if step % 10 == 0 or step + 1 == args.steps:
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
        "proxy_files": [Path(path).name for path, *_ in cached],
        "config": vars(args),
        "final": trace[-1],
        "prior_rms": float(prior.square().mean().sqrt()),
        "optimized_rms": float(optimized.detach().square().mean().sqrt()),
        "optimized_peak": float(optimized.detach().abs().max()),
        "amplitude_relative_error": float(
            (optimized_spectrum.abs() - amplitude).norm() / amplitude.norm().clamp_min(1e-12)
        ),
        "trace": trace,
    }
    trace_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps({key: value for key, value in metadata.items() if key != "trace"}, indent=2))


if __name__ == "__main__":
    main()
