import torch

from peft_transfer.generate_synapse_readout_attack import balanced_pseudo_ce


def test_balanced_pseudo_ce_increases_when_classes_swap():
    targets = torch.tensor([[[0, 0], [1, 1]]])
    correct = torch.tensor([[[[4.0, 4.0], [-4.0, -4.0]],
                             [[-4.0, -4.0], [4.0, 4.0]]]])
    assert balanced_pseudo_ce(-correct, targets) > balanced_pseudo_ce(correct, targets)
