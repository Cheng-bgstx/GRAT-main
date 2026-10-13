#!/usr/bin/env python3
"""Score frozen attacks with clean-image base-SAM token clusters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from peft_transfer.evaluate_synapse_slice import ROOT, SAM
from peft_transfer.score_medical_invariant_candidates import official_candidates
from peft_transfer.score_source_label_prototypes import (
    encode, prototype_bank, prototype_loss,
)

import sys
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_sam_encoder


def clean_clusters(feature: torch.Tensor, count: int) -> torch.Tensor:
    smoothed = F.avg_pool2d(feature.float(), kernel_size=3, stride=1, padding=1)
    tokens = F.normalize(smoothed[0].flatten(1).T, dim=1)
    assignments = KMeans(n_clusters=count, n_init=1, random_state=42).fit_predict(
        tokens.cpu().numpy()
    )
    return torch.from_numpy(assignments.astype(np.int64)).view(
        1, 1, *feature.shape[-2:]
    ).to(feature.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    rows = official_candidates(args.official)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_sam_encoder("vit_b", str(SAM["vit_b"]), 512, 8, device)
    encoder = model.image_encoder
    cache = {}
    with torch.no_grad():
        for row in rows:
            key = (row["case"], row["slice"])
            if key not in cache:
                with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{key[0]}.npy.h5") as handle:
                    image = handle["image"][key[1]].astype(np.float32)
                clean = torch.from_numpy(image)[None, None].repeat(1, 3, 1, 1).to(device)
                clean_feature = encode(encoder, clean)
                labels = clean_clusters(clean_feature, args.clusters)
                bank = prototype_bank(clean_feature, labels)
                if bank is None:
                    raise RuntimeError(f"Cluster bank has fewer than two classes for {key}")
                prototypes, targets, classes = bank
                clean_loss = float(prototype_loss(
                    clean_feature, prototypes, targets, args.temperature
                ))
                cache[key] = (prototypes, targets, classes, clean_loss)
            prototypes, targets, classes, clean_loss = cache[key]
            array = np.asarray(Image.open(ROOT / row["candidate"]).convert("L"),
                               dtype=np.float32) / 255.0
            attacked = torch.from_numpy(array)[None, None].repeat(1, 3, 1, 1).to(device)
            attacked_loss = float(prototype_loss(
                encode(encoder, attacked), prototypes, targets, args.temperature
            ))
            row["source_cluster"] = {
                "clean_loss": clean_loss, "attacked_loss": attacked_loss,
                "loss_increase": attacked_loss - clean_loss,
                "n_clusters": len(classes),
            }
    rankings = {}
    for key in cache:
        subset = [row for row in rows if (row["case"], row["slice"]) == key]
        if len(subset) < 2:
            continue
        values = [row["source_cluster"]["loss_increase"] for row in subset]
        rankings[f"{key[0]}/slice{key[1]}"] = {
            "predicted_best": subset[int(np.argmax(values))]["candidate"],
            "SAMed_best": max(subset, key=lambda row: row["SAMed_dice_drop"])["candidate"],
            "MedSA_best": max(subset, key=lambda row: row["MedSA_dice_drop"])["candidate"],
            "SAMed_rho": float(spearmanr(values, [row["SAMed_dice_drop"] for row in subset])[0]),
            "MedSA_rho": float(spearmanr(values, [row["MedSA_dice_drop"] for row in subset])[0]),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "source_only_score": True,
        "ground_truth_used_for_score": False,
        "target_metrics_used_for_post_hoc_ranking_only": True,
        "clusters": args.clusters,
        "rows": rows, "within_case_rankings": rankings,
    }, indent=2))
    print(json.dumps(rankings, indent=2))


if __name__ == "__main__":
    main()
