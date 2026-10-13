import torch

from peft_transfer.latent_collision_attack import collision_distance


def test_collision_distance_is_zero_at_target():
    value = torch.randn(1, 4, 3, 3)
    assert collision_distance(value, value) < 1e-6


def test_horizontal_flip_preserves_fourier_amplitude_up_to_frequency_reflection():
    value = torch.randn(1, 1, 16, 16)
    original = torch.fft.fft2(value, norm="ortho").abs()
    flipped = torch.fft.fft2(torch.flip(value, (-1,)), norm="ortho").abs()
    # Reflection reverses horizontal frequency indices modulo the DC bin.
    reflected_indices = torch.remainder(-torch.arange(16), 16)
    assert torch.allclose(flipped, original[..., reflected_indices], atol=1e-5)
