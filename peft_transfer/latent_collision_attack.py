"""Prompt-free latent collision attacks for PEFT transfer."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from peft_transfer.core import AttackConfig, _asymmetric_step, _unpack_encoder


def collision_distance(observed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.detach()
    scale = target.abs().mean().clamp_min(1e-6)
    relative_l1 = (observed - target).abs().mean() / scale
    angular = 1.0 - F.cosine_similarity(
        observed.flatten(1), target.flatten(1), dim=1
    ).mean()
    return relative_l1 + 0.5 * angular


def attack_latent_collision(
    encoder,
    image: torch.Tensor,
    target_image: torch.Tensor,
    cfg: AttackConfig,
    preprocess=lambda value: value,
    initial_delta: torch.Tensor | None = None,
    first_step_multiplier: float = 2.5,
    layer_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """Move the source representation toward a prompt-free reference image."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if image.is_cuda:
        torch.cuda.manual_seed_all(cfg.seed)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        target_final, target_layers = _unpack_encoder(encoder(preprocess(target_image)))

    prior = None
    if initial_delta is not None:
        prior = initial_delta.to(device=image.device, dtype=image.dtype)
        if prior.shape[-2:] != image.shape[-2:]:
            prior = F.interpolate(prior, image.shape[-2:], mode="bilinear", align_corners=False)
        if prior.shape[1] == 1 and image.shape[1] == 3:
            prior = prior.repeat(1, 3, 1, 1)
        prior = prior.detach()
    adv = image.detach().clone()
    momentum = torch.zeros_like(image)
    trace = {"target_distance": [], "final_distance": [], "layer_distance": []}
    for step in range(cfg.steps):
        adv.requires_grad_(True)
        observed, observed_layers = _unpack_encoder(encoder(preprocess(adv)))
        final_distance = collision_distance(observed, target_final)
        count = min(6, len(observed_layers), len(target_layers))
        if count and layer_weight > 0:
            layer_distance = torch.stack([
                collision_distance(observed_layers[index], target_layers[index])
                for index in range(len(observed_layers) - count, len(observed_layers))
            ]).mean()
        else:
            layer_distance = final_distance.new_zeros(())
        distance = final_distance + float(layer_weight) * layer_distance
        gradient = torch.autograd.grad(-distance, adv)[0]
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = cfg.momentum * momentum + gradient
        update = _asymmetric_step(momentum, cfg.step_size)
        if step == 0 and prior is not None:
            update = float(first_step_multiplier) * update + prior
        adv = adv.detach() + update
        delta = (adv - image).clamp(-cfg.epsilon, cfg.epsilon)
        adv = (image + delta).clamp(0.0, 1.0).detach()
        trace["target_distance"].append(float(distance.detach()))
        trace["final_distance"].append(float(final_distance.detach()))
        trace["layer_distance"].append(float(layer_distance.detach()))
    return adv, trace
