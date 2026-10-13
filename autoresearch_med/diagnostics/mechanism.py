"""Source-side H1 path-response statistics."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _validate(responses: Sequence[torch.Tensor], path_samples: Sequence[float]) -> None:
    if len(responses) < 2 or len(responses) != len(path_samples):
        raise ValueError("responses and path_samples must have the same length >= 2")
    if any(right <= left for left, right in zip(path_samples, path_samples[1:])):
        raise ValueError("path_samples must be strictly increasing")
    if responses[0].shape != responses[-1].shape:
        raise ValueError("all path responses must share a shape")
    if any(value.shape != responses[0].shape for value in responses):
        raise ValueError("all path responses must share a shape")


def trapezoid_integral(
    responses: Sequence[torch.Tensor], path_samples: Sequence[float]
) -> torch.Tensor:
    _validate(responses, path_samples)
    result = torch.zeros_like(responses[0])
    for left, right, start, end in zip(
        path_samples, path_samples[1:], responses, responses[1:]
    ):
        result = result + 0.5 * (right - left) * (start + end)
    return result


def path_response_statistics(
    responses: Sequence[torch.Tensor],
    path_samples: Sequence[float],
    endpoint_response: torch.Tensor | None = None,
    eta: float = 1e-12,
) -> dict[str, float]:
    """Compute H1 geometry for directional responses sampled along one path."""
    integral = trapezoid_integral(responses, path_samples)
    norm_integral = 0.0
    adjacent_cosines = []
    reversal_fractions = []
    for left, right, start, end in zip(
        path_samples, path_samples[1:], responses, responses[1:]
    ):
        width = right - left
        norm_integral += 0.5 * width * (
            float(start.norm().item()) + float(end.norm().item())
        )
        adjacent_cosines.append(float(F.cosine_similarity(
            start.reshape(1, -1), end.reshape(1, -1), dim=1, eps=eta
        ).item()))
        reversal_fractions.append(float(((start * end) < 0).float().mean().item()))

    consistency = float(integral.norm().item()) / max(norm_integral, eta)
    result = {
        "path_consistency": min(consistency, 1.0 + 1e-6),
        "integrated_response_norm": float(integral.norm().item()),
        "integrated_norm": float(norm_integral),
        "mean_adjacent_cosine": sum(adjacent_cosines) / len(adjacent_cosines),
        "mean_sign_reversal_fraction": sum(reversal_fractions) / len(reversal_fractions),
    }
    if endpoint_response is not None:
        residual = endpoint_response - integral
        result.update({
            "endpoint_magnitude": float(endpoint_response.norm().item()),
            "quadrature_residual_ratio": float(residual.norm().item())
            / max(float(endpoint_response.norm().item()), eta),
        })
    return result


def discrete_path_statistics(
    features: Sequence[torch.Tensor],
    path_samples: Sequence[float],
    eta: float = 1e-12,
) -> dict[str, float]:
    """Stable chord-to-variation estimator of path response consistency.

    The numerator telescopes exactly to the endpoint feature response. The
    denominator is the polygonal path length and converges to total variation
    as path sampling is refined.
    """
    _validate(features, path_samples)
    segments = [right - left for left, right in zip(features, features[1:])]
    endpoint = features[-1] - features[0]
    path_length = sum(float(segment.norm().item()) for segment in segments)
    adjacent_cosines = [
        float(F.cosine_similarity(
            left.reshape(1, -1), right.reshape(1, -1), dim=1, eps=eta
        ).item())
        for left, right in zip(segments, segments[1:])
    ]
    reversal_fractions = [
        float(((left * right) < 0).float().mean().item())
        for left, right in zip(segments, segments[1:])
    ]
    return {
        "discrete_path_consistency": float(endpoint.norm().item()) / max(path_length, eta),
        "endpoint_magnitude": float(endpoint.norm().item()),
        "polygonal_path_length": float(path_length),
        "mean_adjacent_segment_cosine": (
            sum(adjacent_cosines) / len(adjacent_cosines) if adjacent_cosines else 1.0
        ),
        "mean_segment_sign_reversal_fraction": (
            sum(reversal_fractions) / len(reversal_fractions) if reversal_fractions else 0.0
        ),
        "telescoping_residual_ratio": float(
            (torch.stack(segments).sum(0) - endpoint).norm().item()
        ) / max(float(endpoint.norm().item()), eta),
    }

