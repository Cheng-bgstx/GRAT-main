"""Small-sample grouped statistics for mechanism gates."""
from __future__ import annotations

import numpy as np


def spearman(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size != right.size or left.size < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    left_rank = np.argsort(np.argsort(left, kind="mergesort"), kind="mergesort")
    right_rank = np.argsort(np.argsort(right, kind="mergesort"), kind="mergesort")
    if left_rank.std() == 0 or right_rank.std() == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def relative_mae_improvement(reference_mae: float, candidate_mae: float) -> float:
    if reference_mae <= 0:
        raise ValueError("reference_mae must be positive")
    return float((reference_mae - candidate_mae) / reference_mae)


def h1_gate(
    mae_improvement: float,
    residual_spearman_by_family: dict[str, float],
    negative_controls_pass: bool,
) -> dict[str, object]:
    family_signs_pass = bool(residual_spearman_by_family) and all(
        value > 0 for value in residual_spearman_by_family.values()
    )
    rules = {
        "mae_improvement_ge_0_05": mae_improvement >= 0.05,
        "positive_residual_spearman_each_family": family_signs_pass,
        "negative_controls_pass": bool(negative_controls_pass),
    }
    return {"passed": all(rules.values()), "rules": rules}

