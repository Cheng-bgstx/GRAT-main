import torch

from peft_transfer.umi_phase_amplitude import match_rms_with_peak, phase_amplitude_priors


def test_matched_priors_obey_peak_and_rms():
    generator = torch.Generator().manual_seed(4)
    prior = torch.randn(1, 3, 32, 32, generator=generator).clamp(-2, 2) * 0.01
    variants = phase_amplitude_priors(prior, seed=7)
    target_rms = prior.square().mean().sqrt()
    target_peak = prior.abs().max()
    for value in variants.values():
        assert value.abs().max() <= target_peak + 1e-6
        assert torch.allclose(value.square().mean().sqrt(), target_rms, atol=1e-6)


def test_match_rms_with_peak_handles_zero():
    reference = torch.ones(1, 1, 4, 4) * 0.1
    assert torch.equal(match_rms_with_peak(torch.zeros_like(reference), reference), torch.zeros_like(reference))
