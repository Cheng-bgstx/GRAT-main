#!/usr/bin/env python3
"""Evaluate fixed Synapse slices from several cases with the official functions."""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from peft_transfer.evaluate_official_slice import (
    ROOT,
    SAM,
    SliceDataset,
    load_medsa_function,
    load_medsa_model,
    load_samed_model,
    reset_seed,
    test_single_volume,
)


def parse_item(spec: str):
    """Parse CASE:SLICE:NAME=PATH[,NAME=PATH...] without guessing paths."""
    case, index, attacks = spec.split(":", 2)
    parsed = {}
    for attack in attacks.split(","):
        name, path = attack.split("=", 1)
        parsed[name] = ROOT / path
    return case, int(index), parsed


def mean_metrics(rows, key):
    return {
        metric: float(np.mean([row[key][metric] for row in rows]))
        for metric in ("dice", "hd95")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-seed", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    items = []
    for spec in args.items:
        case, index, paths = parse_item(spec)
        with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{case}.npy.h5", "r") as handle:
            clean = handle["image"][index].astype(np.float32)
            label = handle["label"][index].astype(np.float32)
        images = {"clean": clean}
        images.update({
            name: np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            for name, path in paths.items()
        })
        for name, image in images.items():
            if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
                raise ValueError(f"{case}/{index}/{name} violates the [0,1] input scale")
        items.append({"case": case, "slice": index, "images": images, "label": label})

    names = list(items[0]["images"])
    if any(list(item["images"]) != names for item in items):
        raise ValueError("every item must contain the same named attacks in the same order")
    result = {"protocol": vars(args), "SAMed": [], "MedSA": []}

    samed = load_samed_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/SAMed_Synapse_{args.vit}.pth"),
        512, 8, device=device,
    )
    for item in items:
        clean_tensor = torch.from_numpy(item["images"]["clean"])[None, None]
        label_tensor = torch.from_numpy(item["label"])[None, None]
        row = {"case": item["case"], "slice": item["slice"]}
        for name, image in item["images"].items():
            metric = test_single_volume(
                torch.from_numpy(image)[None, None], clean_tensor, label_tensor,
                samed, classes=8, multimask_output=True, patch_size=[512, 512],
                test_save_path=None, case=item["case"], z_spacing=1, is_3d=True,
            )
            row[name] = {
                "dice": float(np.mean(metric, axis=0)[0]),
                "hd95": float(np.mean(metric, axis=0)[1]),
            }
        result["SAMed"].append(row)
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
    for item in items:
        row = {"case": item["case"], "slice": item["slice"]}
        for name, image in item["images"].items():
            reset_seed(args.prompt_seed)
            loader = DataLoader(
                SliceDataset(image, item["label"], item["case"]), batch_size=1,
                shuffle=False, num_workers=0,
            )
            _loss, metrics = validation_sam(medsa_args, loader, 0, medsa)
            row[name] = {"dice": float(metrics[0]), "hd95": float(metrics[1])}
        result["MedSA"].append(row)

    result["mean"] = {
        model: {name: mean_metrics(result[model], name) for name in names}
        for model in ("SAMed", "MedSA")
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
