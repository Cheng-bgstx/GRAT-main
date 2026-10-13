"""Core metrics for Adaptation-Coupled Response Stability analysis."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


BANDS = ("low", "mid", "high")


def feature_bchw(feature: torch.Tensor, layout: str) -> torch.Tensor:
    """Convert an explicitly declared feature layout to BCHW."""
    if feature.ndim != 4:
        raise ValueError(f"expected a 4-D feature tensor, got {tuple(feature.shape)}")
    if layout == "bhwc":
        return feature.permute(0, 3, 1, 2).contiguous()
    if layout == "bchw":
        return feature.contiguous()
    raise ValueError(f"unknown feature layout {layout!r}")


def spatial_band_masks(height: int, width: int, device=None) -> Dict[str, torch.Tensor]:
    """Return non-overlapping radial masks on the true spatial H/W axes."""
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / np.sqrt(0.5)
    return {
        "low": radius < 0.15,
        "mid": (radius >= 0.15) & (radius < 0.4),
        "high": radius >= 0.4,
    }


def _energy(value: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(value):
        return value.abs().square().sum()
    return value.float().square().sum()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item())


def response_metrics(
    source_response: torch.Tensor,
    adapted_response: torch.Tensor,
    clean_source: torch.Tensor,
    eta: float = 1e-12,
) -> Dict[str, object]:
    """Measure C = d_m - d_0 and its spatial-frequency organization."""
    if source_response.shape != adapted_response.shape:
        raise ValueError(
            f"response shapes differ: {tuple(source_response.shape)} and "
            f"{tuple(adapted_response.shape)}"
        )
    coupled_error = adapted_response - source_response
    source_norm = source_response.float().norm()
    adapted_norm = adapted_response.float().norm()
    error_norm = coupled_error.float().norm()
    clean_norm = clean_source.float().norm().clamp_min(eta)
    source_spectrum = torch.fft.fft2(
        source_response.float(), dim=(-2, -1), norm="ortho"
    )
    error_spectrum = torch.fft.fft2(
        coupled_error.float(), dim=(-2, -1), norm="ortho"
    )
    masks = spatial_band_masks(
        source_response.shape[-2], source_response.shape[-1], source_response.device
    )
    source_band_energy = {}
    error_band_energy = {}
    relative_band_mismatch = {}
    for name, mask in masks.items():
        expanded = mask[None, None]
        source_energy = _energy(source_spectrum * expanded)
        error_energy = _energy(error_spectrum * expanded)
        source_band_energy[name] = float(source_energy)
        error_band_energy[name] = float(error_energy)
        relative_band_mismatch[name] = float(
            error_energy.sqrt() / source_energy.sqrt().clamp_min(eta)
        )
    total_source_spectral_energy = sum(source_band_energy.values()) + eta
    total_error_spectral_energy = sum(error_band_energy.values()) + eta
    return {
        "source_response_norm": float(source_norm),
        "source_relative_response": float(source_norm / clean_norm),
        "adapted_response_norm": float(adapted_norm),
        "coupled_error_norm": float(error_norm),
        "relative_mismatch": float(error_norm / source_norm.clamp_min(eta)),
        "response_cosine": _cosine(source_response, adapted_response),
        "adapted_to_source_norm": float(adapted_norm / source_norm.clamp_min(eta)),
        "reverse_triangle_bound_ratio": float(
            (source_norm - error_norm).clamp_min(0) / source_norm.clamp_min(eta)
        ),
        "source_band_fraction": {
            name: value / total_source_spectral_energy
            for name, value in source_band_energy.items()
        },
        "error_band_fraction": {
            name: value / total_error_spectral_energy
            for name, value in error_band_energy.items()
        },
        "relative_band_mismatch": relative_band_mismatch,
    }


def normalized_entropy(values: Sequence[float], eta: float = 1e-12) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 1 or array.sum() <= eta:
        return 0.0
    probabilities = array / array.sum()
    return float(-(probabilities * np.log(probabilities + eta)).sum() / np.log(array.size))


def aggregate_source_observables(layer_metrics: Sequence[Dict[str, object]]) -> Dict[str, float]:
    energies = [float(row["source_response_norm"]) ** 2 for row in layer_metrics]
    relative = [float(row["source_relative_response"]) for row in layer_metrics]
    band_energy = {
        band: sum(
            energy * float(row["source_band_fraction"][band])
            for energy, row in zip(energies, layer_metrics)
        )
        for band in BANDS
    }
    total = sum(band_energy.values()) + 1e-12
    band_fractions = {band: band_energy[band] / total for band in BANDS}
    return {
        "source_log_response": float(np.log(np.mean(relative) + 1e-12)),
        "source_layer_entropy": normalized_entropy(energies),
        "source_spectral_entropy": normalized_entropy(list(band_fractions.values())),
        **{f"source_{band}_fraction": value for band, value in band_fractions.items()},
    }


def aggregate_acrs_observables(layer_metrics: Sequence[Dict[str, object]]) -> Dict[str, float]:
    weights = np.asarray(
        [float(row["source_response_norm"]) ** 2 for row in layer_metrics],
        dtype=np.float64,
    )
    weights = weights / max(weights.sum(), 1e-12)
    result = {
        "acrs_relative_mismatch": float(np.sum([
            weight * float(row["relative_mismatch"])
            for weight, row in zip(weights, layer_metrics)
        ])),
        "acrs_response_cosine": float(np.sum([
            weight * float(row["response_cosine"])
            for weight, row in zip(weights, layer_metrics)
        ])),
        "acrs_bound_ratio": float(np.sum([
            weight * float(row["reverse_triangle_bound_ratio"])
            for weight, row in zip(weights, layer_metrics)
        ])),
    }
    for band in BANDS:
        result[f"acrs_{band}_mismatch"] = float(np.sum([
            weight * float(row["relative_band_mismatch"][band])
            for weight, row in zip(weights, layer_metrics)
        ]))
    return result


def pairwise_structure(vectors: Dict[str, torch.Tensor]) -> List[Dict[str, object]]:
    rows = []
    for (left_name, left), (right_name, right) in combinations(vectors.items(), 2):
        rows.append({
            "left": left_name,
            "right": right_name,
            "cosine": _cosine(left, right),
        })
    return rows


def pearson(left: Iterable[float], right: Iterable[float]) -> float:
    left = np.asarray(list(left), dtype=np.float64)
    right = np.asarray(list(right), dtype=np.float64)
    if left.size < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _ridge_predict(train_x, train_y, test_x, alpha: float = 1.0):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    train = np.concatenate([np.ones((len(train), 1)), train], axis=1)
    test = np.concatenate([np.ones((len(test), 1)), test], axis=1)
    penalty = np.eye(train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ train_y)
    return test @ weights


def leave_one_group_out_ridge(
    rows: Sequence[Dict[str, object]],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Evaluate grouped prediction without sharing a case across train/test."""
    target_index = {name: index for index, name in enumerate(target_names)}
    x = []
    y = []
    groups = []
    for row in rows:
        one_hot = [0.0] * max(0, len(target_names) - 1)
        index = target_index[str(row["target"])]
        if index > 0:
            one_hot[index - 1] = 1.0
        x.append(one_hot + [float(row[name]) for name in feature_names])
        y.append(float(row["dice_drop"]))
        groups.append(str(row["case"]))
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    predictions = np.zeros_like(y)
    for group in sorted(set(groups)):
        test = groups == group
        train = ~test
        predictions[test] = _ridge_predict(x[train], y[train], x[test], alpha)
    return {
        "mae": float(np.mean(np.abs(predictions - y))),
        "rmse": float(np.sqrt(np.mean((predictions - y) ** 2))),
        "prediction_correlation": pearson(predictions, y),
    }
