#!/usr/bin/env python3
"""Blend two folders of 2D adversarial perturbations inside an L-infinity budget."""
from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.isic import ISIC2016
from datasets.refuge import REFUGE


DATASETS = {
    "ISIC": (ISIC2016, ROOT / "data/ISIC"),
    "REFUGE": (REFUGE, ROOT / "data/REFUGE-Multirater"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--indices", default="0,1,2,3")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dataset_cls, data_path = DATASETS[args.dataset]
    image_tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    mask_tf = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = dataset_cls(
        args=Namespace(img_size=512), data_path=str(data_path), mode="Test",
        transform=image_tf, transform_msk=mask_tf,
    )
    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        sample = dataset[index]
        case = str(sample["image_meta_dict"]["filename_or_obj"])
        clean = np.transpose(sample["image"].numpy(), (1, 2, 0)).astype(np.float32)
        left = np.load(ROOT / args.left / f"{case}.npy").astype(np.float32)
        right = np.load(ROOT / args.right / f"{case}.npy").astype(np.float32)
        for weight in args.weights:
            value = clean + (1 - weight) * (left - clean) + weight * (right - clean)
            value = np.clip(value, np.maximum(0, clean - 10 / 255), np.minimum(1, clean + 10 / 255))
            folder = ROOT / args.out / f"W{int(round(weight * 100)):03d}"
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / f"{case}.npy", value.astype(np.float32))
            lower = np.ceil(np.maximum(0, clean - 10 / 255) * 255).astype(np.int16)
            upper = np.floor(np.minimum(1, clean + 10 / 255) * 255).astype(np.int16)
            quantized = np.clip(np.rint(value * 255).astype(np.int16), lower, upper).astype(np.uint8)
            Image.fromarray(quantized).save(folder / f"{case}.png")


if __name__ == "__main__":
    main()
