#!/usr/bin/env python3
"""Evaluate several post-generation Synapse candidates with one model load."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from peft_transfer.evaluate_synapse_slice import ROOT, SAM, msa_predict, samed_predict, score

import sys
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_medsa_model, load_samed_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean = handle["image"][args.slice].astype(np.float32)
        label = handle["label"][args.slice].astype(np.int64)
    images = {
        path: np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        for path in args.candidates
    }
    samed = load_samed_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/SAMed_Synapse_{args.vit}.pth"),
        512, 8, device=device,
    )
    clean_s = score(samed_predict(samed, clean, device), label)
    results = {
        path: {"SAMed": score(samed_predict(samed, image, device), label)}
        for path, image in images.items()
    }
    del samed
    torch.cuda.empty_cache()
    medsa = load_medsa_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
        512, 9, device=device,
    )
    clean_m = score(msa_predict(medsa, clean, label, device), label)
    for path, image in images.items():
        results[path]["MedSA"] = score(msa_predict(medsa, image, label, device), label)
    print(json.dumps({"clean": {"SAMed": clean_s, "MedSA": clean_m}, "results": results}, indent=2))


if __name__ == "__main__":
    main()
