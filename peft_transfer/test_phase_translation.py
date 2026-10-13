import torch

from peft_transfer.compose_phase_translation import compose, phase_translation_carrier


def test_translation_carrier_has_exact_budget_or_zero():
    clean = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 15
    carrier = phase_translation_carrier(clean, 0.25, 10 / 255)
    values = torch.unique(carrier.abs())
    assert all(
        torch.isclose(value, torch.tensor(0.0))
        or torch.isclose(value, torch.tensor(10 / 255))
        for value in values
    )


def test_composition_respects_linf_budget():
    clean = torch.linspace(0, 1, 16).reshape(4, 4)
    attack = (clean + 10 / 255).clamp(0, 1)
    result = compose(clean, attack, 0.25, 0.25, 10 / 255)
    assert float((result - clean).abs().max()) <= 10 / 255 + 1e-7
