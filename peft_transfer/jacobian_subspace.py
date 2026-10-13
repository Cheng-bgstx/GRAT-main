"""Linear-algebra tools for source-only PEFT Jacobian tangent diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch


@dataclass
class ProjectionResult:
    relative_residual: float
    cosine_reconstruction: float
    coefficient_norm: float
    effective_rank: int
    coefficients: torch.Tensor
    reconstruction: torch.Tensor


def unit_l2(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm().clamp_min(1e-12)


def affine_tangent_projection(
    source: torch.Tensor,
    target: torch.Tensor,
    views: Sequence[torch.Tensor],
    ridge: float = 1e-4,
) -> ProjectionResult:
    """Project ``target-source`` onto source-view tangent columns.

    All inputs are normalized to unit L2 norm before forming the affine model.
    The ridge is scaled by the mean diagonal of the tangent Gram matrix, making
    the reported result insensitive to image resolution and gradient scale.
    """
    source_flat = unit_l2(source.detach().float().reshape(-1))
    target_flat = unit_l2(target.detach().float().reshape(-1))
    if not views:
        reconstruction = source_flat
        return ProjectionResult(
            relative_residual=1.0,
            cosine_reconstruction=float(torch.dot(unit_l2(reconstruction), target_flat)),
            coefficient_norm=0.0,
            effective_rank=0,
            coefficients=torch.empty(0),
            reconstruction=reconstruction,
        )

    tangents = [
        unit_l2(view.detach().float().reshape(-1)) - source_flat for view in views
    ]
    return affine_basis_projection(source, target, tangents, ridge)


def affine_basis_projection(
    source: torch.Tensor,
    target: torch.Tensor,
    tangents: Sequence[torch.Tensor],
    ridge: float = 1e-4,
) -> ProjectionResult:
    """Project normalized target drift onto explicit source-only tangents."""
    source_flat = unit_l2(source.detach().float().reshape(-1))
    target_flat = unit_l2(target.detach().float().reshape(-1))
    residual = target_flat - source_flat
    if not tangents:
        reconstruction = source_flat
        return ProjectionResult(
            relative_residual=1.0,
            cosine_reconstruction=float(torch.dot(unit_l2(reconstruction), target_flat)),
            coefficient_norm=0.0,
            effective_rank=0,
            coefficients=torch.empty(0),
            reconstruction=reconstruction,
        )
    columns = torch.stack([
        tangent.detach().float().reshape(-1) for tangent in tangents
    ], dim=1)
    gram = columns.T @ columns
    scale = gram.diag().mean().clamp_min(1e-12)
    system = gram + float(ridge) * scale * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    coefficients = torch.linalg.solve(system, columns.T @ residual)
    fitted = columns @ coefficients
    reconstruction = source_flat + fitted
    singular_values = torch.linalg.svdvals(columns)
    threshold = singular_values.max().clamp_min(1e-12) * 1e-5
    return ProjectionResult(
        relative_residual=float((residual - fitted).norm() / residual.norm().clamp_min(1e-12)),
        cosine_reconstruction=float(torch.dot(unit_l2(reconstruction), target_flat)),
        coefficient_norm=float(coefficients.norm()),
        effective_rank=int((singular_values > threshold).sum()),
        coefficients=coefficients.detach().cpu(),
        reconstruction=reconstruction,
    )


def riesz_pair(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two real 2-D Riesz quadratures of a BCHW tensor."""
    height, width = value.shape[-2:]
    fy = torch.fft.fftfreq(height, device=value.device, dtype=value.dtype).view(1, 1, height, 1)
    fx = torch.fft.fftfreq(width, device=value.device, dtype=value.dtype).view(1, 1, 1, width)
    radius = torch.sqrt(fx.square() + fy.square()).clamp_min(1e-12)
    spectrum = torch.fft.fft2(value, dim=(-2, -1), norm="ortho")
    horizontal = torch.fft.ifft2(spectrum * (-1j * fx / radius), dim=(-2, -1), norm="ortho").real
    vertical = torch.fft.ifft2(spectrum * (-1j * fy / radius), dim=(-2, -1), norm="ortho").real
    return horizontal, vertical


def carrier_component_coverage(
    source: torch.Tensor,
    target: torch.Tensor,
    reconstruction: torch.Tensor,
) -> Dict[str, float]:
    """Measure radial and tangential residual coverage in the source carrier basis.

    The Fourier transform is spatial (H,W) and channelwise.  For the normalized
    gradient drift, radial coefficients change amplitude along the source phase;
    tangential coefficients rotate that phase to first order.
    """
    shape = source.shape
    source_n = unit_l2(source.detach().float().reshape(-1)).reshape(shape)
    target_n = unit_l2(target.detach().float().reshape(-1)).reshape(shape)
    recon_n = reconstruction.reshape(shape).to(source_n)
    source_fft = torch.fft.rfft2(source_n, dim=(-2, -1), norm="ortho")
    drift_fft = torch.fft.rfft2(target_n - source_n, dim=(-2, -1), norm="ortho")
    fit_fft = torch.fft.rfft2(recon_n - source_n, dim=(-2, -1), norm="ortho")
    carrier = torch.exp(1j * torch.angle(source_fft))
    drift_local = drift_fft * carrier.conj()
    error_local = (drift_fft - fit_fft) * carrier.conj()

    radial_energy = drift_local.real.square().sum().clamp_min(1e-12)
    tangent_energy = drift_local.imag.square().sum().clamp_min(1e-12)
    total = radial_energy + tangent_energy
    return {
        "victim_radial_fraction": float(radial_energy / total),
        "radial_relative_residual": float(error_local.real.norm() / drift_local.real.norm().clamp_min(1e-12)),
        "tangential_relative_residual": float(error_local.imag.norm() / drift_local.imag.norm().clamp_min(1e-12)),
    }
