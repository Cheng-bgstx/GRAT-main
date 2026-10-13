"""Common-ascent directions for a finite family of differentiable losses."""
from __future__ import annotations

import torch


def minimum_norm_convex_combination(gradients, iterations=100, tolerance=1e-6):
    """Minimize ||sum_k weight[k] * gradient[k]||_2 over the simplex.

    Frank-Wolfe uses an exact line search on the selected simplex edge. The
    resulting direction is common ascent when the origin is outside the convex
    hull of the gradients. A zero norm explicitly means no positive margin.
    """
    if not gradients:
        raise ValueError("at least one gradient is required")
    shape = gradients[0].shape
    if any(gradient.shape != shape for gradient in gradients):
        raise ValueError("all gradients must have the same shape")
    matrix = torch.stack([gradient.detach().float().reshape(-1) for gradient in gradients])
    if not torch.isfinite(matrix).all():
        raise ValueError("gradients must be finite")
    gram = matrix @ matrix.T
    weights = torch.full((len(gradients),), 1.0 / len(gradients), device=matrix.device)
    for _ in range(iterations):
        correlations = gram @ weights
        vertex = int(correlations.argmin())
        gap = (weights @ correlations - correlations[vertex]).clamp_min(0)
        if float(gap) <= tolerance:
            break
        difference = gram[vertex, vertex] - 2 * correlations[vertex] + weights @ correlations
        step = (gap / difference.clamp_min(1e-30)).clamp(0, 1)
        weights = weights * (1 - step)
        weights[vertex] += step
    mixture = (weights @ matrix).reshape(shape)
    return mixture, weights


def common_ascent_direction(gradients, feasible_low=None, feasible_high=None):
    """Return an L-inf-feasible first-order ascent direction and its margins.

    A sign step is used only when every model's directional derivative is
    positive. Otherwise the minimum-norm L2 direction is used. Projection can
    remove the common-ascent property, so feasibility is checked explicitly.
    """
    normalized = [gradient / gradient.float().norm().clamp_min(1e-30) for gradient in gradients]
    mixture, weights = minimum_norm_convex_combination(normalized)
    if float(mixture.norm()) < 1e-10:
        direction = torch.zeros_like(mixture)
    else:
        sign = mixture.sign()
        sign_margins = torch.stack([(gradient * sign).sum() for gradient in normalized])
        direction = sign if bool((sign_margins > 0).all()) else mixture / mixture.abs().max()
    if feasible_low is not None or feasible_high is not None:
        if feasible_low is None or feasible_high is None:
            raise ValueError("both feasibility bounds are required")
        direction = torch.where((direction < 0) & ~feasible_low, 0, direction)
        direction = torch.where((direction > 0) & ~feasible_high, 0, direction)
    margins = torch.stack([(gradient * direction).sum() for gradient in normalized])
    return direction, weights, margins
