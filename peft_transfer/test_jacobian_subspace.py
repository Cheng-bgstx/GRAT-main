import torch

from peft_transfer.jacobian_subspace import (
    affine_tangent_projection,
    carrier_component_coverage,
    riesz_pair,
)


def test_affine_tangent_projection_recovers_known_span():
    source = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    view_a = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    view_b = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    target = view_a.clone()
    result = affine_tangent_projection(source, target, [view_a, view_b], ridge=1e-8)
    assert result.relative_residual < 1e-5
    assert result.cosine_reconstruction > 0.99999
    assert result.effective_rank == 2


def test_empty_tangent_family_is_identity_reconstruction():
    source = torch.randn(1, 3, 8, 8)
    target = torch.randn(1, 3, 8, 8)
    result = affine_tangent_projection(source, target, [])
    assert result.relative_residual == 1.0
    assert result.effective_rank == 0


def test_carrier_coverage_is_finite_for_exact_reconstruction():
    source = torch.randn(1, 3, 8, 8)
    target = torch.randn(1, 3, 8, 8)
    exact = target / target.norm()
    metrics = carrier_component_coverage(source, target, exact.reshape(-1))
    assert metrics["radial_relative_residual"] < 1e-5
    assert metrics["tangential_relative_residual"] < 1e-5


def test_riesz_pair_is_real_finite_and_annihilates_constant():
    constant = torch.ones(1, 3, 8, 8)
    horizontal, vertical = riesz_pair(constant)
    assert horizontal.shape == constant.shape and vertical.shape == constant.shape
    assert torch.isfinite(horizontal).all() and torch.isfinite(vertical).all()
    assert horizontal.abs().max() < 1e-6 and vertical.abs().max() < 1e-6
