"""Local, damped task-risk geometry for PEFT model augmentation."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable

import torch


def split_names(names: list[str], holdout_fraction: float, seed: int):
    if not 0 < holdout_fraction < 1 or len(names) < 2:
        raise ValueError("a nonempty train/calibration split is required")
    ordered = sorted(names, key=lambda name: hashlib.sha256(
        f"{seed}:{name}".encode("utf-8")
    ).digest())
    count = max(1, min(len(names) - 1, round(len(names) * holdout_fraction)))
    return ordered[count:], ordered[:count]


def peft_parameters(model: torch.nn.Module, kind: str):
    if kind == "lora":
        selected = {
            name: parameter for name, parameter in model.named_parameters()
            if "linear_a_" in name or "linear_b_" in name
        }
    elif kind == "adapter":
        selected = {
            name: parameter for name, parameter in model.named_parameters()
            if "Adapter" in name
        }
    else:
        raise ValueError(kind)
    if not selected:
        raise ValueError(f"no {kind} PEFT parameters found")
    return selected


def diagonal_empirical_fisher(
    model: torch.nn.Module,
    samples: Iterable,
    loss_fn: Callable,
    parameters: dict[str, torch.nn.Parameter],
    return_mean: bool = False,
):
    """Average per-example squared supervised gradients on public train images."""
    values = {name: torch.zeros_like(parameter, device="cpu")
              for name, parameter in parameters.items()}
    means = {name: torch.zeros_like(parameter, device="cpu")
             for name, parameter in parameters.items()}
    count = 0
    for sample in samples:
        loss = loss_fn(model, sample)
        if loss.ndim or not torch.isfinite(loss):
            raise ValueError("task loss must be a finite scalar")
        gradients = torch.autograd.grad(
            loss, tuple(parameters.values()), allow_unused=True,
        )
        for (name, _), gradient in zip(parameters.items(), gradients):
            if gradient is not None:
                gradient_cpu = gradient.detach().float().cpu()
                values[name].add_(gradient_cpu.square())
                means[name].add_(gradient_cpu)
        count += 1
    if count == 0:
        raise ValueError("Fisher estimation needs at least one sample")
    for value in values.values():
        value.div_(count)
    if return_mean:
        for value in means.values():
            value.div_(count)
        return values, means
    return values


def damped_precision(fisher: dict[str, torch.Tensor], damping_ratio: float = 0.01):
    if damping_ratio <= 0:
        raise ValueError("positive damping is required")
    total = sum(value.sum().item() for value in fisher.values())
    size = sum(value.numel() for value in fisher.values())
    damping = max(damping_ratio * total / size, 1e-12)
    return {name: value + damping for name, value in fisher.items()}, damping


def sample_delta(parameters, precision, relative_norm: float, seed: int,
                 geometry: str = "fisher"):
    """Draw equal-L2-norm isotropic or inverse-precision PEFT directions."""
    if geometry not in ("fisher", "isotropic") or relative_norm <= 0:
        raise ValueError("invalid geometry or relative norm")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    delta = {}
    parameter_norm = math.sqrt(sum(
        parameter.detach().float().square().sum().item()
        for parameter in parameters.values()
    ))
    for name, parameter in parameters.items():
        noise = torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
        if geometry == "fisher":
            noise = noise / precision[name].sqrt()
        delta[name] = noise
    noise_norm = math.sqrt(sum(value.square().sum().item() for value in delta.values()))
    if parameter_norm == 0 or noise_norm == 0:
        raise ValueError("zero-norm PEFT parameter or perturbation")
    scale = relative_norm * parameter_norm / noise_norm
    return {name: value * scale for name, value in delta.items()}


def apply_delta(parameters, baseline, delta):
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(baseline[name] + delta[name].to(
                device=parameter.device, dtype=parameter.dtype
            ))


def restore_parameters(parameters, baseline):
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(baseline[name])


def basin_linear_lower_bound(loss, loss_gradient, precision, radius):
    """Exact minimum of a linearized loss over a positive-definite ellipsoid."""
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    dual_norm_sq = sum(
        (loss_gradient[name].square() / precision[name]).sum()
        for name in loss_gradient
    )
    return loss - radius * dual_norm_sq.sqrt()
