#!/usr/bin/env python3
"""Partition two 2D perturbations across channels or spatial blocks."""
from __future__ import annotations

import argparse
from argparse import Namespace

import numpy as np
import torch
from torchvision import transforms

from datasets.isic import ISIC2016
from datasets.refuge import REFUGE
from peft_transfer.evaluate_official_slice import ROOT
from peft_transfer.generate_2d_checkpoint_proxy import save_candidate


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
    parser.add_argument("--blocks", type=int, nargs="+", default=(16, 32))
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
        clean = sample["image"].float()[None]
        left = torch.from_numpy(
            np.transpose(np.load(ROOT / args.left / f"{case}.npy"), (2, 0, 1))
        )[None].float()
        right = torch.from_numpy(
            np.transpose(np.load(ROOT / args.right / f"{case}.npy"), (2, 0, 1))
        )[None].float()
        for channel in range(3):
            left_mask = torch.zeros_like(clean)
            left_mask[:, channel] = 1.0
            left_one = left_mask * left + (1.0 - left_mask) * right
            right_one = left_mask * right + (1.0 - left_mask) * left
            save_candidate(ROOT / args.out / f"LEFT_C{channel}", case, left_one, clean)
            save_candidate(ROOT / args.out / f"RIGHT_C{channel}", case, right_one, clean)
        height, width = clean.shape[-2:]
        yy = torch.arange(height)[None, None, :, None]
        xx = torch.arange(width)[None, None, None, :]
        for block in args.blocks:
            mask = (((yy // block) + (xx // block)) % 2 == 0).float().expand_as(clean)
            save_candidate(ROOT / args.out / f"BLOCK{block}_A", case,
                           mask * left + (1.0 - mask) * right, clean)
            save_candidate(ROOT / args.out / f"BLOCK{block}_B", case,
                           mask * right + (1.0 - mask) * left, clean)


if __name__ == "__main__":
    main()
