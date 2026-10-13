import torch

from peft_transfer.architecture_rank_attack import _nonconflicting, compatible_embedding_objective


def test_nonconflicting_removes_negative_projection():
    anchor = torch.tensor([[[[1.0, 0.0]]]])
    auxiliary = torch.tensor([[[[-1.0, 1.0]]]])
    safe, cosine = _nonconflicting(anchor, auxiliary)
    assert cosine < 0
    assert float((safe * anchor).sum()) >= -1e-6


def test_phase_amplitude_embedding_objective_is_zero_at_clean():
    clean = torch.randn(1, 4, 8, 8)
    value, parts = compatible_embedding_objective(clean, clean, "phase_amp")
    assert abs(float(value)) < 1e-7
    assert all(abs(float(component)) < 1e-7 for component in parts.values())
