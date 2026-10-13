import torch

from peft_transfer.core import AttackConfig
from peft_transfer.optimize_umi_phase_prior_pasd import pasd_risk


def test_pasd_risk_is_zero_feature_plus_finite_probe_term():
    generator = torch.Generator().manual_seed(5)
    clean = torch.randn(1, 4, 3, 3, generator=generator)
    layers = [torch.randn(1, 3, 3, 4, generator=generator) for _ in range(4)]
    cfg = AttackConfig(pasd_region="late", pasd_layer_count=2, pasd_samples=1)
    risk, feature, probe = pasd_risk(clean, clean, layers, layers, cfg)
    assert torch.isfinite(risk)
    assert feature < 1e-3
    assert torch.allclose(risk, feature + probe)
