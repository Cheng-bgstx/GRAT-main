#!/usr/bin/env python3
"""Evaluate named attack templates on several slices with official functions."""
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


def load_attack(spec: str):
    name, template = spec.split("=", 1)
    return name, template


def summarize(rows):
    return {
        name: {
            "dice": float(np.mean([row[name]["dice"] for row in rows])),
            "hd95": float(np.mean([row[name]["hd95"] for row in rows])),
        }
        for name in rows[0]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--attacks", nargs="+", required=True,
        help="NAME=path/template/{slice}.png",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-seed", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    attacks = dict(load_attack(spec) for spec in args.attacks)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean = {index: handle["image"][index].astype(np.float32) for index in args.slices}
        labels = {index: handle["label"][index].astype(np.float32) for index in args.slices}
    images = {
        index: {
            "clean": clean[index],
            **{
                name: np.asarray(
                    Image.open(ROOT / template.format(slice=index)).convert("L"), dtype=np.float32
                ) / 255.0
                for name, template in attacks.items()
            },
        }
        for index in args.slices
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"slices": args.slices, "SAMed": [], "MedSA": []}
    samed = load_samed_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/SAMed_Synapse_{args.vit}.pth"),
        512, 8, device=device,
    )
    for index in args.slices:
        clean_tensor = torch.from_numpy(clean[index])[None, None]
        label_tensor = torch.from_numpy(labels[index])[None, None]
        row = {}
        for name, image in images[index].items():
            metric = test_single_volume(
                torch.from_numpy(image)[None, None], clean_tensor, label_tensor,
                samed, classes=8, multimask_output=True, patch_size=[512, 512],
                test_save_path=None, case=args.case, z_spacing=1, is_3d=True,
            )
            row[name] = {
                "dice": float(np.mean(metric, axis=0)[0]),
                "hd95": float(np.mean(metric, axis=0)[1]),
            }
        result["SAMed"].append(row)
    result["SAMed_mean"] = summarize(result["SAMed"])
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"SAMed_mean": result["SAMed_mean"]}, indent=2), flush=True)
    del samed
    torch.cuda.empty_cache()

    medsa = load_medsa_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
        512, 9, device=device,
    )
    medsa_function = load_medsa_function()
    medsa_args = Namespace(
        dataset="Synapse", multimask_output=9, thd=False, evl_chunk=1,
        gpu_device=0, image_size=512, out_size=512, net="sam", vis=0,
        path_helper={"sample_path": "/tmp"},
    )
    for index in args.slices:
        row = {}
        for name, image in images[index].items():
            reset_seed(args.prompt_seed)
            loader = DataLoader(
                SliceDataset(image, labels[index], args.case), batch_size=1,
                shuffle=False, num_workers=0,
            )
            _loss, metrics = medsa_function.validation_sam(medsa_args, loader, 0, medsa)
            row[name] = {"dice": float(metrics[0]), "hd95": float(metrics[1])}
        result["MedSA"].append(row)
        output_path.write_text(json.dumps(result, indent=2))
    result["MedSA_mean"] = summarize(result["MedSA"])
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"SAMed_mean": result["SAMed_mean"], "MedSA_mean": result["MedSA_mean"]}, indent=2))


if __name__ == "__main__":
    main()
