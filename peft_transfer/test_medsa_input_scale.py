import torch
from torch import nn

from peft_transfer.generate_2d_dual_peft import medsa_feature


class _IdentityMedSA(nn.Module):
    def preprocess(self, value):
        return value

    def image_encoder(self, value):
        return value


def test_medsa_feature_uses_explicit_dataset_scale():
    image = torch.full((1, 3, 2, 2), 0.25)
    model = _IdentityMedSA()
    assert torch.equal(medsa_feature(model, image, input_scale=1.0), image)
    assert torch.equal(medsa_feature(model, image), image * 255.0)
