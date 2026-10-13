import torch

from peft_transfer.core import patch_boundary_disruption


def test_patch_boundary_is_finite_and_increases_for_checkerboard_displacement():
    generator = torch.Generator().manual_seed(23)
    clean = torch.randn(1, 8, 8, 16, generator=generator)
    smooth = clean + 0.1
    checker = torch.tensor([[(-1.0) ** (i + j) for j in range(8)] for i in range(8)])
    checker = clean + 0.1 * checker[None, ..., None]
    smooth_loss = patch_boundary_disruption(smooth, clean)
    checker_loss = patch_boundary_disruption(checker, clean)
    assert torch.isfinite(checker_loss)
    assert checker_loss > smooth_loss
