"""Controlled spatial Fourier ablations of a universal perturbation."""
from __future__ import annotations

from typing import Dict

import torch


def match_rms_with_peak(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Match reference RMS under its L-infinity bound by monotone rescaling."""
    target_rms = reference.square().mean().sqrt()
    peak = reference.abs().max()
    if value.abs().max() <= 1e-12:
        return torch.zeros_like(value)
    base = value / value.square().mean().sqrt().clamp_min(1e-12)
    low = value.new_zeros(())
    high = target_rms * 64.0
    for _ in range(48):
        middle = (low + high) * 0.5
        candidate_rms = (base * middle).clamp(-peak, peak).square().mean().sqrt()
        if candidate_rms < target_rms:
            low = middle
        else:
            high = middle
    return (base * high).clamp(-peak, peak)


def radial_mean_amplitude(amplitude: torch.Tensor, bins: int = 64) -> torch.Tensor:
    height, half_width = amplitude.shape[-2:]
    width = (half_width - 1) * 2
    fy = torch.fft.fftfreq(height, device=amplitude.device).view(height, 1)
    fx = torch.fft.rfftfreq(width, device=amplitude.device).view(1, half_width)
    radius = torch.sqrt(fx.square() + fy.square())
    indices = torch.clamp((radius / radius.max().clamp_min(1e-12) * bins).long(), max=bins - 1)
    output = torch.empty_like(amplitude)
    for index in range(bins):
        mask = indices == index
        if mask.any():
            mean = amplitude[..., mask].mean(dim=-1, keepdim=True)
            output[..., mask] = mean
    return output


def phase_amplitude_priors(prior: torch.Tensor, seed: int = 42) -> Dict[str, torch.Tensor]:
    """Return controlled priors, all matched to the original RMS and peak."""
    spectrum = torch.fft.rfft2(prior.float(), dim=(-2, -1), norm="ortho")
    amplitude = spectrum.abs()
    phase = torch.angle(spectrum)
    carrier = torch.exp(1j * phase)
    generator = torch.Generator(device=prior.device).manual_seed(seed)
    noise = torch.randn(prior.shape, generator=generator, device=prior.device)
    random_phase = torch.angle(torch.fft.rfft2(noise, dim=(-2, -1), norm="ortho"))
    radial = radial_mean_amplitude(amplitude)
    flat = amplitude.mean(dim=(-2, -1), keepdim=True).expand_as(amplitude)

    original_carrier = torch.exp(1j * phase)
    random_carrier = torch.exp(1j * random_phase)
    spectra = {
        "amplitude_random_phase": amplitude * torch.exp(1j * random_phase),
        "phase_radial_amplitude": radial * carrier,
        "phase_flat_amplitude": flat * carrier,
        "amplitude_zero_phase": amplitude.to(spectrum.dtype),
    }
    for alpha in (0.05, 0.10, 0.20, 0.30, 0.50):
        mixed_carrier = (1.0 - alpha) * original_carrier + alpha * random_carrier
        mixed_carrier = mixed_carrier / mixed_carrier.abs().clamp_min(1e-12)
        spectra[f"phase_mix_{int(alpha * 100):02d}"] = amplitude * mixed_carrier
    transformed = {
        name: match_rms_with_peak(
            torch.fft.irfft2(value, s=prior.shape[-2:], dim=(-2, -1), norm="ortho"), prior
        ).to(prior.dtype)
        for name, value in spectra.items()
    }
    return {"original": prior.clone(), **transformed}
