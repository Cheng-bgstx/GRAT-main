#!/usr/bin/env python3
"""Learn a joint amplitude-phase UMI from source-only robust PASD risk."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from peft_transfer.core import AttackConfig, _unpack_encoder
from peft_transfer.generate import CHECKPOINTS, ROOT
from peft_transfer.medsa_surrogate import IntermediateEncoderWrapper, load_medsa_base_surrogate
from peft_transfer.optimize_umi_phase_prior import select_proxy_images
from peft_transfer.optimize_umi_phase_prior_pasd import pasd_risk
from segment_anything import sam_model_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior",
        default="common_perb/common_perturbation_meta_0.025sp_bound2_it7batchit41upsize_ls0.0025mulmd0_10000.pth",
    )
    parser.add_argument("--output", default="testset/joint_umi_prior/prior.pth")
    parser.add_argument("--trace", default="testset/joint_umi_prior/train_trace.json")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--proxy-images", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=10 / 255)
    parser.add_argument("--step-size", type=float, default=0.25 / 255)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--prox-weight", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    original = torch.load(ROOT / args.prior, map_location=device).float()
    delta = original.clone().clamp(-args.epsilon, args.epsilon)
    velocity = torch.zeros_like(delta)
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
            compatible_clean, compatible_clean_layers = _unpack_encoder(
                compatible_encoder(compatible_model.preprocess(image))
            )
        current = delta.detach().requires_grad_(True)
        adv = (image + current).clamp(0.0, 1.0)
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device)
        canonical_adv, canonical_layers = _unpack_encoder(canonical_encoder(adv))
        canonical_risk, _cf, _cp = pasd_risk(
            canonical_adv, canonical_clean, canonical_layers, canonical_clean_layers, cfg
        )
        cpu_after = torch.random.get_rng_state()
        cuda_after = torch.cuda.get_rng_state(device)
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)
        compatible_adv, compatible_layers = _unpack_encoder(
            compatible_encoder(compatible_model.preprocess(adv))
        )
        compatible_risk, _mf, _mp = pasd_risk(
            compatible_adv, compatible_clean, compatible_layers, compatible_clean_layers, cfg
        )
        torch.random.set_rng_state(cpu_after)
        torch.cuda.set_rng_state(cuda_after, device)
        risks = torch.stack([canonical_risk, compatible_risk])
        robust = -args.temperature * torch.logsumexp(-risks / args.temperature, dim=0)
        cosine = F.cosine_similarity(current.flatten(1), original.flatten(1), dim=1).mean()
        objective = robust - args.prox_weight * (1.0 - cosine)
        gradient = torch.autograd.grad(objective, current)[0]
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        velocity = args.momentum * velocity + gradient
        delta = (delta + args.step_size * velocity.sign()).clamp(-args.epsilon, args.epsilon)
        record = {
            "step": step, "proxy": Path(path).name,
            "canonical_risk": float(canonical_risk.detach()),
            "compatible_risk": float(compatible_risk.detach()),
            "robust_risk": float(robust.detach()),
            "prior_cosine": float(cosine.detach()),
            "rms": float(delta.square().mean().sqrt()),
            "peak": float(delta.abs().max()),
        }
        trace.append(record)
        if step % 8 == 0 or step + 1 == args.steps:
            print(json.dumps(record), flush=True)
    output, trace_path = ROOT / args.output, ROOT / args.trace
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta.detach().cpu(), output)
    metadata = {
        "source_only": True,
        "training_objective": "joint_amplitude_phase_dual_architecture_pasd_softmin",
        "proxy_files": [Path(path).name for path, _ in proxies],
        "config": vars(args), "final": trace[-1], "trace": trace,
    }
    trace_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps({key: value for key, value in metadata.items() if key != "trace"}, indent=2))


if __name__ == "__main__":
    main()
