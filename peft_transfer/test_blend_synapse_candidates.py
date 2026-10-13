import torch

from peft_transfer.blend_synapse_candidates import fuse


def test_fusion_respects_budget():
    clean = torch.full((1, 3, 2, 2), 0.5)
    source = clean + 10 / 255
    semantic = clean - 10 / 255
    for mode in ("linear", "steepest"):
        fused = fuse(clean, source, semantic, 0.25, mode)
        assert float((fused - clean).abs().max()) <= 10 / 255 + 1e-7
