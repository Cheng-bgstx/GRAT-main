import torch

from peft_transfer.generate_synapse_dual_peft import load_grayscale_prior


def test_load_grayscale_prior_selects_legacy_payload_channel(tmp_path, monkeypatch):
    prior = torch.stack([
        torch.full((2, 2), -0.03),
        torch.zeros((2, 2)),
        torch.full((2, 2), 0.03),
    ])[None]
    path = tmp_path / "prior.pth"
    torch.save(prior, path)
    monkeypatch.setattr(
        "peft_transfer.generate_synapse_dual_peft.ROOT", tmp_path
    )
    loaded = load_grayscale_prior("prior.pth", torch.device("cpu"))
    assert loaded.shape == prior.shape
    assert torch.equal(loaded[:, 0], loaded[:, 1])
    assert torch.equal(loaded[:, 1], loaded[:, 2])
    assert torch.allclose(loaded, torch.full_like(loaded, -0.03))
