import torch

from peft_transfer.interpolate_checkpoints import interpolate_state_dict


def test_interpolate_state_dict_floating_and_integer_entries():
    left = {"weight": torch.tensor([0.0, 2.0]), "counter": torch.tensor(3)}
    right = {"weight": torch.tensor([3.0, 5.0]), "counter": torch.tensor(3)}
    result = interpolate_state_dict(left, right, 1.0 / 3.0)
    assert torch.allclose(result["weight"], torch.tensor([1.0, 3.0]))
    assert result["counter"].item() == 3


def test_interpolate_state_dict_rejects_mismatched_keys():
    try:
        interpolate_state_dict({"a": torch.ones(1)}, {"b": torch.ones(1)}, 0.5)
    except KeyError:
        pass
    else:
        raise AssertionError("mismatched state dictionaries were accepted")
