import torch

from peft_transfer.dual_architecture_pasd import (
    pareto_preserving_fusion,
    sign_cone_fusion,
    source_mass_budget_fusion,
)


def test_pareto_fusion_keeps_positive_anchor_projection():
    generator = torch.Generator().manual_seed(11)
    anchor = torch.randn(1, 3, 8, 8, generator=generator)
    auxiliary = -anchor + 0.2 * torch.randn(1, 3, 8, 8, generator=generator)
    fused, raw_cosine, fused_cosine = pareto_preserving_fusion(anchor, auxiliary, 0.5)
    assert raw_cosine < 0
    assert fused_cosine > 0
    normalized_anchor = anchor / anchor.abs().mean()
    assert (normalized_anchor * fused).sum() > 0


def test_sign_cone_fusion_preserves_every_anchor_sign():
    generator = torch.Generator().manual_seed(17)
    anchor = torch.randn(1, 3, 8, 8, generator=generator)
    auxiliary = torch.randn(1, 3, 8, 8, generator=generator)
    fused, _raw, _cosine, preserved = sign_cone_fusion(anchor, auxiliary, 2.0)
    assert preserved == 1.0
    assert torch.all(fused * anchor >= 0)


def test_mass_budget_obeys_source_gain_bound():
    generator = torch.Generator().manual_seed(19)
    anchor = torch.randn(1, 3, 8, 8, generator=generator)
    auxiliary = -torch.randn(1, 3, 8, 8, generator=generator)
    fused, _raw, _cosine, _preserved, ratio = source_mass_budget_fusion(
        anchor, auxiliary, 0.02
    )
    normalized = anchor / anchor.abs().mean()
    measured = (normalized * fused.sign()).sum() / normalized.abs().sum()
    assert ratio >= 0.96 - 1e-6
    assert abs(float(measured) - ratio) < 1e-5
