import torch

from peft_transfer.generate_synapse_antithetic import save_quantized


def test_synapse_quantization_respects_budget(tmp_path):
    clean = torch.full((1, 3, 4, 4), 0.5)
    adv = clean + 20.0 / 255.0
    linf = save_quantized(tmp_path / "attack.png", adv, clean)
    assert linf <= 10.0001
