#!/usr/bin/env python3
"""Quick clean/adv transfer diagnostic on one Synapse slice.

This is intentionally a smoke evaluator.  Final results must use volume-level
metrics over the complete test split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_medsa_model, load_samed_model

SAM = {
    "vit_b": ROOT / "checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": ROOT / "checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": ROOT / "checkpoints/sam_vit_h_4b8939.pth",
}


def dice(pred, target):
    pred, target = pred.astype(bool), target.astype(bool)
    if not pred.any() and not target.any():
        return 1.0
    return float(2 * (pred & target).sum() / max(1, pred.sum() + target.sum()))


def samed_predict(model, image, device):
    x = torch.from_numpy(image).float().to(device)[None, None].repeat(1, 3, 1, 1)
    with torch.no_grad():
        output = model(x, True, image.shape[0])
        return output["masks"].argmax(1).squeeze().cpu().numpy()


def msa_predict(model, image, label, device):
    x = torch.from_numpy(image).float().to(device)[None, None].repeat(1, 3, 1, 1)
    points, point_labels = [], []
    for class_id in range(1, 9):
        locations = np.argwhere(label == class_id)
        if len(locations):
            row, col = locations[len(locations) // 2]
            points.append([col, row])
            point_labels.append(1)
        else:
            points.append([0, 0])
            point_labels.append(0)
    coords = torch.tensor([points], dtype=torch.float32, device=device)
    labels = torch.tensor([point_labels], dtype=torch.int64, device=device)
    with torch.no_grad():
        prepared = model.preprocess(x)
        embedding = model.image_encoder(prepared)
        sparse, dense = model.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        logits, _ = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        logits = F.interpolate(logits, image.shape, mode="bilinear", align_corners=False)
    return logits.argmax(1).squeeze().cpu().numpy()


def score(pred, label):
    present = [class_id for class_id in range(1, 9) if (label == class_id).any()]
    return float(np.mean([dice(pred == class_id, label == class_id) for class_id in present]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--adv", required=True, help="generated PNG")
    parser.add_argument("--samed", default=None)
    parser.add_argument("--msa", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean = handle["image"][args.slice].astype(np.float32)
        label = handle["label"][args.slice].astype(np.int64)
    adv = mpimg.imread(args.adv)
    if adv.ndim == 3:
        adv = adv[..., 0]
    adv = adv.astype(np.float32)
    samed_ckpt = args.samed or str(ROOT / f"checkpoints/best_weights/SAMed_Synapse_{args.vit}.pth")
    msa_ckpt = args.msa or str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth")

    samed = load_samed_model(args.vit, str(SAM[args.vit]), samed_ckpt, 512, 8, device=device)
    clean_s = score(samed_predict(samed, clean, device), label)
    adv_s = score(samed_predict(samed, adv, device), label)
    del samed
    torch.cuda.empty_cache()
    msa = load_medsa_model(args.vit, str(SAM[args.vit]), msa_ckpt, 512, 9, device=device)
    clean_m = score(msa_predict(msa, clean, label, device), label)
    adv_m = score(msa_predict(msa, adv, label, device), label)
    result = {
        "case": args.case, "slice": args.slice,
        "linf_255": float(np.max(np.abs(adv - clean)) * 255.0),
        "SAMed": {"clean_dice": clean_s, "adv_dice": adv_s, "delta_dice": clean_s - adv_s},
        "MSA": {"clean_dice": clean_m, "adv_dice": adv_m, "delta_dice": clean_m - adv_m},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
