import torch
import torch.nn as nn

from peft_transfer.medsa_surrogate import IntermediateEncoderWrapper


def test_intermediate_wrapper_preserves_final_and_converts_layout():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])

        def forward(self, value):
            value = value.permute(0, 2, 3, 1)
            for block in self.blocks:
                value = block(value)
            return value.permute(0, 3, 1, 2)

    image = torch.randn(1, 3, 8, 8)
    encoder = Encoder()
    expected = encoder(image)
    final, layers = IntermediateEncoderWrapper(encoder)(image)
    assert torch.equal(final, expected)
    assert len(layers) == 2 and layers[0].shape == image.shape
