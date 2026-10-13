import torch

from peft_transfer.common_ascent import (
    common_ascent_direction,
    minimum_norm_convex_combination,
)


def test_orthogonal_gradients_have_positive_common_margin():
    gradients = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    direction, weights, margins = common_ascent_direction(gradients)
    assert torch.allclose(weights, torch.tensor([0.5, 0.5]), atol=1e-5)
    assert torch.equal(direction, torch.ones(2))
    assert bool((margins > 0).all())


def test_opposite_gradients_report_empty_common_cone():
    gradients = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    direction, _weights, margins = common_ascent_direction(gradients)
    assert torch.equal(direction, torch.zeros(2))
    assert torch.equal(margins, torch.zeros(2))


def test_feasibility_can_destroy_ascent_and_is_reported():
    gradients = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    direction, _weights, margins = common_ascent_direction(
        gradients,
        feasible_low=torch.tensor([True, True]),
        feasible_high=torch.tensor([True, False]),
    )
    assert direction[1] == 0
    assert margins[1] == 0


def test_minimum_norm_convex_combination_is_finite_and_simplex():
    mixture, weights = minimum_norm_convex_combination([
        torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])
    ])
    assert torch.isfinite(mixture).all()
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
