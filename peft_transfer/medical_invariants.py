"""Source-only medical environment views and encoder feature utilities."""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


TRAIN_DIR = Path("data/Synapse_preprocess/train_npz")
TEST_DIR = Path("data/Synapse_preprocess/test_vol_h5")


def training_slices(case: str, count: int = 3) -> list[Path]:
    paths = sorted(TRAIN_DIR.glob(f"{case}_slice*.npz"),
                   key=lambda path: int(re.search(r"slice(\d+)", path.stem).group(1)))
    if len(paths) < count:
        raise ValueError(f"Expected at least {count} slices for {case}, found {len(paths)}")
    return [paths[round((i + 1) * (len(paths) - 1) / (count + 1))]
            for i in range(count)]


def test_slices(case: str, count: int = 3) -> list[tuple[str, int]]:
    with h5py.File(TEST_DIR / f"{case}.npy.h5") as handle:
        depth = len(handle["image"])
    return [(case, round((i + 1) * (depth - 1) / (count + 1)))
            for i in range(count)]


def image_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(image.astype(np.float32))[None, None].to(device)
    if value.shape[-2:] != (512, 512):
        value = F.interpolate(value, (512, 512), mode="bilinear", align_corners=False)
    return value.repeat(1, 3, 1, 1)


def load_train_image(path: Path, device: torch.device) -> torch.Tensor:
    with np.load(path) as handle:
        return image_tensor(handle["image"], device)


def load_test_image(case: str, index: int, device: torch.device) -> torch.Tensor:
    with h5py.File(TEST_DIR / f"{case}.npy.h5") as handle:
        return image_tensor(handle["image"][index], device)


def environment_view(image: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return image
    center = image.median()
    if name in {"contrast_08", "contrast_12"}:
        factor = 0.8 if name == "contrast_08" else 1.2
        return (center + factor * (image - center)).clamp(0, 1)
    if name in {"gamma_09", "gamma_11"}:
        power = 0.9 if name == "gamma_09" else 1.1
        return image.clamp(0, 1).pow(power)
    if name == "bias_field":
        width = image.shape[-1]
        horizontal = torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype)
        field = 1 + 0.06 * torch.cos(torch.pi * horizontal)
        return (image * field.view(1, 1, 1, -1)).clamp(0, 1)
    if name == "blur":
        weights = image.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        kernel = torch.outer(weights, weights)[None, None].repeat(image.shape[1], 1, 1, 1)
        return F.conv2d(F.pad(image, (2, 2, 2, 2), mode="reflect"),
                        kernel, groups=image.shape[1])
    raise ValueError(f"Unknown medical environment {name}")


def capture_features(encoder, image: torch.Tensor, layers: tuple[str, ...]):
    captured = {}
    handles = []
    for name in layers:
        if name == "neck":
            continue
        index = int(name)

        def hook(_module, _inputs, output, key=name):
            value = output if torch.is_tensor(output) else output[0]
            captured[key] = value.permute(0, 3, 1, 2)

        handles.append(encoder.blocks[index].register_forward_hook(hook))
    try:
        result = encoder(image)
        captured["neck"] = result if torch.is_tensor(result) else result[0]
    finally:
        for handle in handles:
            handle.remove()
    return {name: captured[name].float().flatten(2).transpose(1, 2)[0]
            for name in layers}


def centered_scaled(tokens: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (tokens - tokens.mean(0, keepdim=True)) / scale.to(tokens.device).clamp_min(1e-6)
