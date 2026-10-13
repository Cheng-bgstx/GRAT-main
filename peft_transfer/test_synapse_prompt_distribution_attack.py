import torch

from peft_transfer.generate_synapse_prompt_distribution_attack import (
    balanced_label_ce,
    sample_multiclass_points,
    spatial_risk_weights,
)


def test_prompt_sampler_stays_inside_each_present_class():
    label = torch.tensor([[[[0, 0], [1, 1]]]])
    coordinates, flags = sample_multiclass_points(label, 3, seed=7)
    for class_id in (0, 1):
        x, y = coordinates[0, class_id].long()
        assert label[0, 0, y, x] == class_id
        assert flags[0, class_id] == 1
    assert flags[0, 2] == 0


def test_balanced_label_ce_penalizes_swapped_predictions():
    label = torch.tensor([[[[0, 0], [1, 1]]]])
    correct = torch.tensor([[[[4.0, 4.0], [-4.0, -4.0]],
                             [[-4.0, -4.0], [4.0, 4.0]]]])
    assert balanced_label_ce(-correct, label) > balanced_label_ce(correct, label)


def test_spatial_risk_weights_far_background_more():
    label = torch.zeros((1, 1, 7, 7), dtype=torch.long)
    label[0, 0, 3, 3] = 1
    weights = spatial_risk_weights(label, strength=4.0)
    assert weights[0, 0, 0, 0] > weights[0, 0, 3, 2]
    assert weights[0, 0, 3, 3] == 1
