"""Architecture-compatible Medical-SAM-Adapter surrogate using only SAM weights."""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class IntermediateEncoderWrapper(nn.Module):
    """Expose BHWC block outputs as BCHW features without changing forward values."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor):
        layers, handles = [], []
        for block in self.encoder.blocks:
            def hook(_module, _inputs, output):
                value = output if torch.is_tensor(output) else output[0]
                layers.append(value.permute(0, 3, 1, 2))
            handles.append(block.register_forward_hook(hook))
        try:
            final = self.encoder(image)
        finally:
            for handle in handles:
                handle.remove()
        return final, layers


def load_medsa_base_surrogate(
    vit_name: str,
    sam_checkpoint: str,
    image_size: int = 512,
    multimask_output: int = 9,
    device: torch.device = torch.device("cuda"),
    dataset: str = "Synapse",
):
    """Build MedSA's public architecture with every adapter residual set to zero.

    No fine-tuned checkpoint is read.  The non-adapter weights come exclusively
    from the corresponding official SAM checkpoint.
    """
    msa_root = str(ROOT / "Medical-SAM-Adapter-main")
    saved_path = list(sys.path)
    if msa_root in sys.path:
        sys.path.remove(msa_root)
    sys.path.insert(0, msa_root)
    try:
        from models.sam import sam_model_registry as msa_registry

        args = Namespace(
            dataset=dataset, image_size=image_size,
            multimask_output=multimask_output, sam_ckpt=sam_checkpoint,
            encoder=vit_name, mod="sam_adpt", mid_dim=None,
            thd=False, chunk=None,
        )
        model = msa_registry[vit_name](args, checkpoint=sam_checkpoint).to(device)
    finally:
        sys.path = saved_path

    for block in model.image_encoder.blocks:
        for adapter_name in ("Space_Adapter", "MLP_Adapter", "Depth_Adapter"):
            adapter = getattr(block, adapter_name)
            torch.nn.init.zeros_(adapter.D_fc1.weight)
            torch.nn.init.zeros_(adapter.D_fc1.bias)
            torch.nn.init.zeros_(adapter.D_fc2.weight)
            torch.nn.init.zeros_(adapter.D_fc2.bias)
    model.pixel_mean = model.pixel_mean.to(device)
    model.pixel_std = model.pixel_std.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
