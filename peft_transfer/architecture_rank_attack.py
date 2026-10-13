"""Joint canonical and architecture-compatible rank-tail transfer attack."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from peft_transfer.core import (
    AttackConfig,
    AdapterSiteCapture,
    _asymmetric_step,
    _edge_map,
    _unpack_encoder,
    adapter_irreducible_tail,
    carrier_stochastic_disruption,
    frequency_balanced_irreducible_tail,
)


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return value / value.abs().mean().clamp_min(1e-12)


def _nonconflicting(anchor: torch.Tensor, auxiliary: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Remove only the auxiliary component opposing the anchor direction."""
    anchor_unit = anchor / anchor.norm().clamp_min(1e-12)
    auxiliary_unit = auxiliary / auxiliary.norm().clamp_min(1e-12)
    cosine = (anchor_unit * auxiliary_unit).sum()
    safe = auxiliary_unit - torch.minimum(cosine, cosine.new_zeros(())) * anchor_unit
    return _normalized(safe), float(cosine.detach())


def compatible_embedding_objective(
    adv: torch.Tensor,
    clean: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Prompt-free spatial and phase-amplitude disruption of the final embedding."""
    clean = clean.detach()
    scale = clean.abs().mean().clamp_min(1e-6)
    spatial = (adv - clean).abs().mean() / scale
    if mode == "feature":
        zero = spatial.new_zeros(())
        return spatial, {"spatial": spatial, "amplitude": zero, "phase": zero}
    if mode != "phase_amp":
        raise ValueError(f"unknown compatible embedding mode {mode!r}")
    clean_fft = torch.fft.rfft2(clean.float(), dim=(-2, -1), norm="ortho")
    adv_fft = torch.fft.rfft2(adv.float(), dim=(-2, -1), norm="ortho")
    clean_amplitude = clean_fft.abs()
    amplitude = (adv_fft.abs() - clean_amplitude).abs().mean()
    amplitude = amplitude / clean_amplitude.mean().clamp_min(1e-6)
    phase_chord = 1.0 - torch.cos(torch.angle(adv_fft) - torch.angle(clean_fft))
    phase_weight = torch.sqrt(clean_amplitude / clean_amplitude.mean().clamp_min(1e-6))
    phase_weight = phase_weight.clamp(max=4.0)
    phase = (phase_weight * phase_chord).mean()
    objective = spatial + 0.5 * amplitude + 0.5 * phase
    return objective, {"spatial": spatial, "amplitude": amplitude, "phase": phase}


def attack_architecture_embedding(
    canonical_encoder,
    compatible_encoder,
    compatible_preprocess,
    image: torch.Tensor,
    cfg: AttackConfig,
    compatible_weight: float,
    mode: str = "feature",
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """Fuse canonical ADS with a MedSA-code-path base embedding objective."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if image.is_cuda:
        torch.cuda.manual_seed_all(cfg.seed)
    for encoder in (canonical_encoder, compatible_encoder):
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    with torch.no_grad():
        canonical_clean, canonical_clean_layers = _unpack_encoder(canonical_encoder(image))
        compatible_clean = compatible_encoder(compatible_preprocess(image)).detach()

    generator = torch.Generator(device=image.device).manual_seed(cfg.seed)
    delta = torch.zeros_like(image)
    if cfg.initial_jitter > 0:
        delta = delta + torch.randn(image.shape, generator=generator, device=image.device) * cfg.initial_jitter
    adv = (image + delta).clamp(0.0, 1.0).detach()
    momentum = torch.zeros_like(image)
    edge = _edge_map(image).detach()
    trace = {
        "canonical_loss": [], "compatible_loss": [], "compatible_spatial": [],
        "compatible_amplitude": [], "compatible_phase": [],
        "raw_gradient_cosine": [], "fused_anchor_cosine": [],
    }
    for _step in range(cfg.steps):
        adv.requires_grad_(True)
        canonical_final, canonical_layers = _unpack_encoder(canonical_encoder(adv))
        canonical_loss = F.l1_loss(canonical_final * 0.9999, canonical_clean.detach())
        canonical_loss = canonical_loss + cfg.ads_weight * carrier_stochastic_disruption(
            canonical_layers, canonical_clean_layers, cfg
        )
        canonical_gradient = torch.autograd.grad(canonical_loss, adv)[0]

        compatible_adv = compatible_encoder(compatible_preprocess(adv))
        compatible_loss, components = compatible_embedding_objective(
            compatible_adv, compatible_clean, mode
        )
        compatible_gradient = torch.autograd.grad(compatible_loss, adv)[0]
        anchor = _normalized(canonical_gradient)
        safe_compatible, raw_cosine = _nonconflicting(anchor, compatible_gradient)
        fused = _normalized(anchor + float(compatible_weight) * safe_compatible)
        if cfg.use_edge:
            fused = _normalized(fused * (1.0 + cfg.edge_gamma * edge))
        fused_cosine = F.cosine_similarity(anchor.flatten(1), fused.flatten(1), dim=1).mean()
        momentum = cfg.momentum * momentum + fused
        adv = adv.detach() + _asymmetric_step(momentum, cfg.step_size)
        adv = (image + (adv - image).clamp(-cfg.epsilon, cfg.epsilon)).clamp(0.0, 1.0).detach()

        trace["canonical_loss"].append(float(canonical_loss.detach()))
        trace["compatible_loss"].append(float(compatible_loss.detach()))
        trace["compatible_spatial"].append(float(components["spatial"].detach()))
        trace["compatible_amplitude"].append(float(components["amplitude"].detach()))
        trace["compatible_phase"].append(float(components["phase"].detach()))
        trace["raw_gradient_cosine"].append(raw_cosine)
        trace["fused_anchor_cosine"].append(float(fused_cosine.detach()))
    return adv, trace


def attack_architecture_rank_tail(
    canonical_encoder,
    compatible_encoder,
    compatible_preprocess,
    image: torch.Tensor,
    cfg: AttackConfig,
    rank_weight: float,
    rank: int = 192,
    mode: str = "plain",
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """Attack canonical SAM and the exact pre-adapter MedSA base jointly."""
    if mode not in {"plain", "haar"}:
        raise ValueError(f"unknown rank-tail mode {mode!r}")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if image.is_cuda:
        torch.cuda.manual_seed_all(cfg.seed)
    for encoder in (canonical_encoder, compatible_encoder):
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

    with torch.no_grad():
        canonical_clean, canonical_clean_layers = _unpack_encoder(canonical_encoder(image))
        clean_capture = AdapterSiteCapture(compatible_encoder, (0,), ("attn",))
        with clean_capture.active():
            compatible_encoder(compatible_preprocess(image))
        compatible_clean = clean_capture.outputs[(0, "attn")].detach()

    generator = torch.Generator(device=image.device).manual_seed(cfg.seed)
    delta = torch.zeros_like(image)
    if cfg.initial_jitter > 0:
        delta = delta + torch.randn(image.shape, generator=generator, device=image.device) * cfg.initial_jitter
    adv = (image + delta).clamp(0.0, 1.0).detach()
    momentum = torch.zeros_like(image)
    edge = _edge_map(image).detach()
    trace = {
        "canonical_loss": [], "rank_tail_loss": [], "rank_tail_ratio": [],
        "raw_gradient_cosine": [], "fused_anchor_cosine": [],
    }

    for _step in range(cfg.steps):
        adv.requires_grad_(True)
        canonical_final, canonical_layers = _unpack_encoder(canonical_encoder(adv))
        canonical_loss = F.l1_loss(canonical_final * 0.9999, canonical_clean.detach())
        canonical_loss = canonical_loss + cfg.ads_weight * carrier_stochastic_disruption(
            canonical_layers, canonical_clean_layers, cfg
        )
        canonical_gradient = torch.autograd.grad(canonical_loss, adv)[0]

        capture = AdapterSiteCapture(compatible_encoder, (0,), ("attn",))
        with capture.active():
            compatible_encoder(compatible_preprocess(adv))
        compatible_adv = capture.outputs[(0, "attn")]
        if mode == "plain":
            rank_loss, rank_ratio = adapter_irreducible_tail(
                {(0, "attn"): compatible_adv},
                {(0, "attn"): compatible_clean}, rank,
            )
        else:
            rank_loss, rank_ratio = frequency_balanced_irreducible_tail(
                compatible_adv, compatible_clean, rank
            )
        rank_gradient = torch.autograd.grad(rank_loss, adv)[0]

        anchor = _normalized(canonical_gradient)
        safe_rank, raw_cosine = _nonconflicting(anchor, rank_gradient)
        fused = _normalized(anchor + float(rank_weight) * safe_rank)
        if cfg.use_edge:
            fused = _normalized(fused * (1.0 + cfg.edge_gamma * edge))
        fused_cosine = F.cosine_similarity(
            anchor.flatten(1), fused.flatten(1), dim=1
        ).mean()
        momentum = cfg.momentum * momentum + fused
        update = _asymmetric_step(momentum, cfg.step_size)
        adv = adv.detach() + update
        adv = (image + (adv - image).clamp(-cfg.epsilon, cfg.epsilon)).clamp(0.0, 1.0).detach()

        trace["canonical_loss"].append(float(canonical_loss.detach()))
        trace["rank_tail_loss"].append(float(rank_loss.detach()))
        trace["rank_tail_ratio"].append(float(rank_ratio.detach()))
        trace["raw_gradient_cosine"].append(raw_cosine)
        trace["fused_anchor_cosine"].append(float(fused_cosine.detach()))
    return adv, trace
