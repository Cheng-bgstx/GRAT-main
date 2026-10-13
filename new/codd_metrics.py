"""
Metrics for Carrier-Oriented Drift Decomposition (CODD).

Given source and target feature maps [B,C,H,W], we aggregate channels, FFT each map,
and decompose Delta G in the local complex basis defined by the source carrier e^{j phi_s}.

Delta G = alpha * e^{j phi_s} + beta * j * e^{j phi_s}

alpha: carrier-aligned radial coefficient
beta : carrier-orthogonal tangential coefficient
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def aggregate_channels(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if mode == "mean":
        return x.mean(dim=1, keepdim=True)
    if mode == "l2":
        return x.norm(dim=1, keepdim=True)
    raise ValueError(mode)


def fft2_complex(f: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft2(f, norm="ortho")


def radial_tangential_decompose(
    feat_s: torch.Tensor,
    feat_t: torch.Tensor,
    mode: str = "mean",
    eps: float = 1e-8,
) -> Dict[str, float]:
    fs = aggregate_channels(feat_s, mode=mode)
    ft = aggregate_channels(feat_t, mode=mode)
    gs = fft2_complex(fs)
    gt = fft2_complex(ft)

    delta = gt - gs
    phase_s = torch.angle(gs)
    carrier = torch.exp(1j * phase_s)
    local = delta * torch.conj(carrier)

    alpha = torch.real(local)
    beta = torch.imag(local)
    total = torch.abs(delta)

    radial_energy = torch.mean(torch.abs(alpha)).item()
    tangential_energy = torch.mean(torch.abs(beta)).item()
    total_energy = torch.mean(total).item()
    radial_ratio = radial_energy / (total_energy + eps)
    tangential_ratio = tangential_energy / (total_energy + eps)

    return {
        "radial_energy": float(radial_energy),
        "tangential_energy": float(tangential_energy),
        "total_energy": float(total_energy),
        "radial_ratio": float(radial_ratio),
        "tangential_ratio": float(tangential_ratio),
    }


__all__ = [
    "aggregate_channels",
    "fft2_complex",
    "radial_tangential_decompose",
]
