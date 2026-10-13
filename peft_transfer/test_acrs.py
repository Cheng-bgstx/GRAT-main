import numpy as np
import torch

from peft_transfer.acrs import (
    feature_bchw, leave_one_group_out_ridge, response_metrics, spatial_band_masks,
)


def test_feature_bchw_uses_declared_spatial_axes():
    feature = torch.arange(1 * 3 * 5 * 2).reshape(1, 3, 5, 2)
    converted = feature_bchw(feature, "bhwc")
    assert converted.shape == (1, 2, 3, 5)
    assert torch.equal(converted[:, :, 1, 4], feature[:, 1, 4, :])


def test_spatial_bands_are_disjoint_and_complete():
    masks = spatial_band_masks(16, 12)
    total = sum(mask.int() for mask in masks.values())
    assert torch.equal(total, torch.ones_like(total))


def test_zero_coupled_error_has_perfect_stability():
    response = torch.randn(1, 4, 8, 8)
    metrics = response_metrics(response, response.clone(), torch.randn_like(response))
    assert metrics["relative_mismatch"] == 0.0
    assert abs(metrics["response_cosine"] - 1.0) < 1e-6
    assert metrics["reverse_triangle_bound_ratio"] == 1.0


def test_grouped_ridge_detects_incremental_signal():
    rows = []
    for case in range(6):
        for target in ("a", "b"):
            strength = case / 5
            mechanism = float((case % 2) * 2 - 1)
            rows.append({
                "case": str(case), "target": target,
                "strength": strength, "mechanism": mechanism,
                "dice_drop": strength + 0.5 * mechanism + (target == "b") * 0.1,
            })
    baseline = leave_one_group_out_ridge(rows, ["strength"], ["a", "b"], alpha=0.01)
    extended = leave_one_group_out_ridge(
        rows, ["strength", "mechanism"], ["a", "b"], alpha=0.01
    )
    assert extended["mae"] < baseline["mae"]
    assert np.isfinite(extended["prediction_correlation"])
