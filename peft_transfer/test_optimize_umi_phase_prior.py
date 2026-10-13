import torch

from peft_transfer.optimize_umi_phase_prior import synthesize_phase_prior


def test_phase_synthesis_obeys_radius_and_peak():
    prior = torch.randn(1, 3, 16, 16) * 0.01
    spectrum = torch.fft.rfft2(prior, norm="ortho")
    parameter = torch.randn(1, 3, 4, 3)
    value, correction = synthesize_phase_prior(
        spectrum.abs(), torch.angle(spectrum), parameter, 0.2,
        prior.shape[-2:], prior.abs().max(),
    )
    assert correction.abs().max() <= 0.2 + 1e-6
    assert value.abs().max() <= prior.abs().max() + 1e-6
    assert torch.isfinite(value).all()
