"""Source-only Pareto attack over canonical and adapter-compatible SAM paths."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from peft_transfer.core import (
    AttackConfig,
    _asymmetric_step,
    _edge_map,
    _unpack_encoder,
    phase_amplitude_stochastic_disruption,
)


def _mean_normalized(value: torch.Tensor) -> torch.Tensor:
    return value / value.abs().mean().clamp_min(1e-12)


def pareto_preserving_fusion(
    anchor: torch.Tensor, auxiliary: torch.Tensor, weight: float
) -> Tuple[torch.Tensor, float, float]:
    """Add the auxiliary tangent after removing source-opposing projection.

    In Euclidean geometry, ``<anchor, fused> >= ||anchor||^2`` before the
    final mean normalization. Thus the compatible path cannot reduce the
    canonical path's first-order ascent direction.
    """
    anchor = _mean_normalized(anchor)
    auxiliary = _mean_normalized(auxiliary)
    anchor_unit = anchor / anchor.norm().clamp_min(1e-12)
    projection = (auxiliary * anchor_unit).sum()
    safe_auxiliary = auxiliary - torch.minimum(
        projection, projection.new_zeros(())
    ) * anchor_unit
    safe_auxiliary = _mean_normalized(safe_auxiliary)
    raw_cosine = F.cosine_similarity(
        anchor.flatten(1), auxiliary.flatten(1), dim=1
    ).mean()
    fused = _mean_normalized(anchor + float(weight) * safe_auxiliary)
    anchor_cosine = F.cosine_similarity(
        anchor.flatten(1), fused.flatten(1), dim=1
    ).mean()
    return fused, float(raw_cosine.detach()), float(anchor_cosine.detach())


def sign_cone_fusion(
    anchor: torch.Tensor, auxiliary: torch.Tensor, weight: float
) -> Tuple[torch.Tensor, float, float, float]:
    """Enhance only coordinates where both architecture gradients agree.

    The returned gradient has exactly the anchor sign on every nonzero anchor
    coordinate. This is the relevant preservation condition for an L-infinity
    steepest-ascent step.
    """
    anchor = _mean_normalized(anchor)
    auxiliary = _mean_normalized(auxiliary)
    agreement = anchor * auxiliary > 0
    compatible = torch.where(agreement, auxiliary.abs() * anchor.sign(), torch.zeros_like(anchor))
    fused = _mean_normalized(anchor + float(weight) * compatible)
    raw_cosine = F.cosine_similarity(
        anchor.flatten(1), auxiliary.flatten(1), dim=1
    ).mean()
    fused_cosine = F.cosine_similarity(
        anchor.flatten(1), fused.flatten(1), dim=1
    ).mean()
    sign_preserved = ((fused * anchor >= 0) | (anchor == 0)).float().mean()
    return (
        fused, float(raw_cosine.detach()), float(fused_cosine.detach()),
        float(sign_preserved.detach()),
    )


def source_mass_budget_fusion(
    anchor: torch.Tensor, auxiliary: torch.Tensor, mass_budget: float
) -> Tuple[torch.Tensor, float, float, float, float]:
    """Flip compatible-favored signs under an exact source-gradient budget.

    If selected coordinates carry at most rho of ``sum(abs(anchor))``, the
    L-infinity linear gain of ``sign(fused)`` is at least ``1 - 2*rho`` times
    the source-optimal gain. Coordinates are ranked by compatible/source
    gradient magnitude ratio among sign conflicts.
    """
    rho = min(max(float(mass_budget), 0.0), 0.499)
    anchor = _mean_normalized(anchor)
    auxiliary = _mean_normalized(auxiliary)
    flat_anchor = anchor.flatten()
    flat_auxiliary = auxiliary.flatten()
    mass = flat_anchor.abs()
    total_mass = mass.sum().clamp_min(1e-12)
    conflict = flat_anchor * flat_auxiliary < 0
    indices = torch.nonzero(conflict, as_tuple=False).flatten()
    selected = torch.zeros_like(conflict)
    if indices.numel() and rho > 0:
        score = flat_auxiliary[indices].abs() / mass[indices].clamp_min(1e-12)
        order = indices[torch.argsort(score, descending=True)]
        cumulative = torch.cumsum(mass[order], dim=0)
        allowed = cumulative <= rho * total_mass
        selected[order[allowed]] = True
    flat_fused = flat_anchor.clone()
    flat_fused[selected] = -flat_fused[selected]
    fused = flat_fused.reshape_as(anchor)
    used_mass = mass[selected].sum() / total_mass
    source_gain_ratio = 1.0 - 2.0 * used_mass
    raw_cosine = F.cosine_similarity(
        anchor.flatten(1), auxiliary.flatten(1), dim=1
    ).mean()
    fused_cosine = F.cosine_similarity(
        anchor.flatten(1), fused.flatten(1), dim=1
    ).mean()
    return (
        fused, float(raw_cosine.detach()), float(fused_cosine.detach()),
        float((~selected).float().mean().detach()),
        float(source_gain_ratio.detach()),
    )


def _risk(final, clean_final, layers, clean_layers, cfg):
    feature = F.l1_loss(final * 0.9999, clean_final.detach())
    pasd = phase_amplitude_stochastic_disruption(layers, clean_layers, cfg)
    return feature + pasd, feature, pasd


def _gray3(value: torch.Tensor) -> torch.Tensor:
    return value[:, :1].repeat(1, 3, 1, 1)


def attack_dual_architecture_pasd(
    canonical_encoder,
    compatible_encoder,
    compatible_preprocess,
    image: torch.Tensor,
    cfg: AttackConfig,
    compatible_weight: float,
    initial_delta: torch.Tensor | None = None,
    first_step_multiplier: float = 2.5,
    fusion: str = "pareto",
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """Run inference-aligned PASD on two source-only architecture paths."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if image.is_cuda:
        torch.cuda.manual_seed_all(cfg.seed)
    for encoder in (canonical_encoder, compatible_encoder):
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    with torch.no_grad():
        canonical_clean, canonical_clean_layers = _unpack_encoder(canonical_encoder(_gray3(image)))
        compatible_clean, compatible_clean_layers = _unpack_encoder(
            compatible_encoder(compatible_preprocess(_gray3(image)))
        )

    prior = None
    if initial_delta is not None:
        prior = initial_delta.to(device=image.device, dtype=image.dtype)
        if prior.shape[-2:] != image.shape[-2:]:
            prior = F.interpolate(prior, image.shape[-2:], mode="bilinear", align_corners=False)
        if prior.shape[1] == 1 and image.shape[1] == 3:
            prior = prior.repeat(1, 3, 1, 1)
        if prior.shape != image.shape:
            raise ValueError(f"prior shape {tuple(prior.shape)} != image shape {tuple(image.shape)}")
        prior = prior.detach()

    delta = torch.zeros_like(image)
    if cfg.initial_jitter > 0:
        delta = delta + torch.randn_like(image) * cfg.initial_jitter
    adv = (image + delta).clamp(0.0, 1.0).detach()
    momentum = torch.zeros_like(image)
    edge = _edge_map(image).detach()
    trace = {
        "canonical_risk": [], "compatible_risk": [],
        "canonical_feature": [], "compatible_feature": [],
        "canonical_pasd": [], "compatible_pasd": [],
        "raw_gradient_cosine": [], "fused_anchor_cosine": [],
        "sign_preserved": [], "source_gain_ratio_bound": [],
    }
    for step in range(cfg.steps):
        adv.requires_grad_(True)
        cpu_rng_before = torch.random.get_rng_state()
        cuda_rng_before = torch.cuda.get_rng_state(image.device) if image.is_cuda else None
        canonical_final, canonical_layers = _unpack_encoder(canonical_encoder(_gray3(adv)))
        canonical_risk, canonical_feature, canonical_pasd = _risk(
            canonical_final, canonical_clean, canonical_layers, canonical_clean_layers, cfg
        )
        canonical_gradient = torch.autograd.grad(canonical_risk, adv)[0]
        cpu_rng_after = torch.random.get_rng_state()
        cuda_rng_after = torch.cuda.get_rng_state(image.device) if image.is_cuda else None

        torch.random.set_rng_state(cpu_rng_before)
        if cuda_rng_before is not None:
            torch.cuda.set_rng_state(cuda_rng_before, image.device)
        compatible_final, compatible_layers = _unpack_encoder(
            compatible_encoder(compatible_preprocess(_gray3(adv)))
        )
        compatible_risk, compatible_feature, compatible_pasd = _risk(
            compatible_final, compatible_clean, compatible_layers, compatible_clean_layers, cfg
        )
        compatible_gradient = torch.autograd.grad(compatible_risk, adv)[0]
        torch.random.set_rng_state(cpu_rng_after)
        if cuda_rng_after is not None:
            torch.cuda.set_rng_state(cuda_rng_after, image.device)
        if fusion == "pareto":
            gradient, raw_cosine, anchor_cosine = pareto_preserving_fusion(
                canonical_gradient, compatible_gradient, compatible_weight
            )
            sign_preserved = float(
                ((gradient * canonical_gradient >= 0) | (canonical_gradient == 0)).float().mean()
            )
        elif fusion == "sign_cone":
            gradient, raw_cosine, anchor_cosine, sign_preserved = sign_cone_fusion(
                canonical_gradient, compatible_gradient, compatible_weight
            )
            source_gain_ratio_bound = 1.0
        elif fusion == "mass_budget":
            (
                gradient, raw_cosine, anchor_cosine, sign_preserved,
                source_gain_ratio_bound,
            ) = source_mass_budget_fusion(
                canonical_gradient, compatible_gradient, compatible_weight
            )
        else:
            raise ValueError(f"unknown fusion {fusion!r}")
        if fusion == "pareto":
            source_gain_ratio_bound = float("nan")
        if cfg.use_edge:
            gradient = _mean_normalized(gradient * (1.0 + cfg.edge_gamma * edge))
        momentum = cfg.momentum * momentum + gradient
        update = _asymmetric_step(momentum, cfg.step_size)
        if step == 0 and prior is not None:
            update = float(first_step_multiplier) * update + prior
        adv = adv.detach() + update
        delta = (adv - image).clamp(-cfg.epsilon, cfg.epsilon)
        adv = (image + delta).clamp(0.0, 1.0).detach()

        trace["canonical_risk"].append(float(canonical_risk.detach()))
        trace["compatible_risk"].append(float(compatible_risk.detach()))
        trace["canonical_feature"].append(float(canonical_feature.detach()))
        trace["compatible_feature"].append(float(compatible_feature.detach()))
        trace["canonical_pasd"].append(float(canonical_pasd.detach()))
        trace["compatible_pasd"].append(float(compatible_pasd.detach()))
        trace["raw_gradient_cosine"].append(raw_cosine)
        trace["fused_anchor_cosine"].append(anchor_cosine)
        trace["sign_preserved"].append(sign_preserved)
        trace["source_gain_ratio_bound"].append(source_gain_ratio_bound)
    return adv, trace
