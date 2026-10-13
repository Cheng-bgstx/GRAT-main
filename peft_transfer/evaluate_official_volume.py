#!/usr/bin/env python3
"""Evaluate complete Synapse adversarial volumes with repository functions."""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from peft_transfer.evaluate_official_slice import (
    ROOT,
    SAM,
    load_medsa_function,
    load_medsa_model,
    load_samed_model,
    reset_seed,
    test_single_volume,
)


class VolumeSliceDataset(Dataset):
    def __init__(self, image: np.ndarray, label: np.ndarray, case_name: str):
        if image.ndim != 3 or image.shape != label.shape:
            raise ValueError(f"expected matching DHW volumes, got {image.shape}, {label.shape}")
        self.image = image
        self.label = label
        self.case_name = case_name

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        return {
            "image": torch.from_numpy(self.image[index])[None, ..., None].float(),
            "label": torch.from_numpy(self.label[index])[None, ..., None].float(),
            "case_name": self.case_name,
        }


def load_volume(folder: Path, depth: int) -> np.ndarray:
    paths = [folder / f"{index}.png" for index in range(depth)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} slices, first: {missing[0]}")
    return np.stack([
        np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        for path in paths
    ])


def parse_candidate(spec: str):
    name, folder = spec.split("=", 1)
    return name, ROOT / folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-seed", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean = handle["image"][:].astype(np.float32)
        label = handle["label"][:].astype(np.float32)
    images = {"clean": clean}
    images.update({name: load_volume(folder, len(clean)) for name, folder in map(parse_candidate, args.candidates)})
    for name, image in images.items():
        if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            raise ValueError(
                f"{name} must use the clean Synapse [0,1] scale; "
                f"observed range [{image.min()}, {image.max()}]"
            )
    audit = {
        name: {
            "minimum": float(image.min()),
            "maximum": float(image.max()),
            "linf_255": float(np.max(np.abs(image - clean)) * 255.0),
        }
        for name, image in images.items()
    }
    result = {
        "protocol": {**vars(args), "input_scale": "float32_[0,1]"},
        "audit": audit, "SAMed": {}, "MedSA": {},
    }

    samed = load_samed_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/SAMed_Synapse_{args.vit}.pth"),
        512, 8, device=device,
    )
    clean_tensor = torch.from_numpy(clean)[None]
    label_tensor = torch.from_numpy(label)[None]
    for name, image in images.items():
        metrics = test_single_volume(
            torch.from_numpy(image)[None], clean_tensor, label_tensor,
            samed, classes=8, multimask_output=True, patch_size=[512, 512],
            test_save_path=None, case=args.case, z_spacing=1, is_3d=True,
        )
        result["SAMed"][name] = {
            "dice": float(np.mean(metrics, axis=0)[0]),
            "hd95": float(np.mean(metrics, axis=0)[1]),
            "per_class": np.asarray(metrics, dtype=float).tolist(),
        }
    del samed
    torch.cuda.empty_cache()

    medsa = load_medsa_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
        512, 9, device=device,
    )
    validation_sam = load_medsa_function().validation_sam
    medsa_args = Namespace(
        dataset="Synapse", multimask_output=9, thd=False, evl_chunk=1,
        gpu_device=0, image_size=512, out_size=512, net="sam", vis=0,
        path_helper={"sample_path": "/tmp"},
    )
    for name, image in images.items():
        reset_seed(args.prompt_seed)
        loader = DataLoader(
            VolumeSliceDataset(image, label, args.case), batch_size=1,
            shuffle=False, num_workers=0,
        )
        _loss, metrics = validation_sam(medsa_args, loader, 0, medsa)
        result["MedSA"][name] = {"dice": float(metrics[0]), "hd95": float(metrics[1])}

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
