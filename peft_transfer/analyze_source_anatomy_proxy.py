#!/usr/bin/env python3
"""Score frozen attacks with an image-only anatomical source-SAM proxy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic

from peft_transfer.evaluate_synapse_slice import ROOT, SAM

sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_sam_encoder


def region_labels(image: np.ndarray, size: tuple[int, int], segments: int) -> torch.Tensor:
    regions = slic(image, n_segments=segments, compactness=0.05, sigma=1.0,
                   multichannel=False, start_label=0)
    tensor = torch.from_numpy(regions.astype(np.float32))[None, None]
    labels = F.interpolate(tensor, size=size, mode="nearest")[0, 0].long()
    return labels


def region_prototypes(clean_feature: torch.Tensor, labels: torch.Tensor):
    clean_tokens = F.normalize(clean_feature[0].flatten(1).T, dim=1)
    flat_labels = labels.flatten().to(clean_tokens.device)
    unique, inverse = torch.unique(flat_labels, sorted=True, return_inverse=True)
    centroids = []
    for index in range(len(unique)):
        centroids.append(F.normalize(clean_tokens[inverse == index].mean(0), dim=0))
    return torch.stack(centroids), inverse


def region_loss(feature: torch.Tensor, prototypes: torch.Tensor,
                labels: torch.Tensor, temperature: float) -> torch.Tensor:
    tokens = F.normalize(feature[0].flatten(1).T, dim=1)
    logits = tokens @ prototypes.T / temperature
    losses = F.cross_entropy(logits, labels, reduction="none")
    # Each region gets equal weight despite unequal area.
    return torch.stack([losses[labels == i].mean()
                        for i in range(len(prototypes))]).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5") as handle:
        image = handle["image"][args.slice].astype(np.float32)
    clean = torch.from_numpy(image)[None, None].repeat(1, 3, 1, 1).to(device)
    model = load_sam_encoder(args.vit, str(SAM[args.vit]), 512, 8, device)

    def encode(value):
        result = model.image_encoder(value)
        return result[0] if isinstance(result, tuple) else result

    with torch.no_grad():
        clean_feature = encode(clean)
        labels = region_labels(image, clean_feature.shape[-2:], args.segments).to(device)
        prototypes, labels = region_prototypes(clean_feature, labels)
        clean_loss = region_loss(clean_feature, prototypes, labels, args.temperature)
        rows = {}
        for path in args.candidates:
            adversarial = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(adversarial)[None, None].repeat(1, 3, 1, 1).to(device)
            changed_feature = encode(tensor)
            attacked_loss = region_loss(changed_feature, prototypes, labels, args.temperature)
            rows[path] = {
                "clean_region_loss": float(clean_loss),
                "attacked_region_loss": float(attacked_loss),
                "region_loss_increase": float(attacked_loss - clean_loss),
                "source_feature_displacement_norm": float((changed_feature - clean_feature).norm()),
            }
    payload = {"source_only": True, "trained_proxy": False,
               "case": args.case, "slice": args.slice, "vit": args.vit,
               "segments_requested": args.segments,
               "segments_at_feature_grid": int(len(prototypes)), "results": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
