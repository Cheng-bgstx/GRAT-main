#!/usr/bin/env python3
"""Post-hoc GT prototype diagnostic for frozen base-SAM attack candidates."""
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

from peft_transfer.evaluate_synapse_slice import ROOT, SAM
from peft_transfer.score_medical_invariant_candidates import official_candidates

import sys
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_sam_encoder


def prototype_loss(embedding, prototypes, target, temperature):
    tokens = F.normalize(embedding[0].flatten(1).T, dim=1)
    logits = tokens @ prototypes.T / temperature
    losses = []
    for index in range(len(prototypes)):
        selected = target == index
        losses.append(F.cross_entropy(logits[selected], target[selected]))
    return torch.stack(losses).mean()


def encode(encoder, image):
    output = encoder(image)
    return output[0] if isinstance(output, tuple) else output


def prototype_bank(embedding, label):
    grid = F.interpolate(label.float(), size=embedding.shape[-2:], mode="nearest")[0, 0]
    tokens = F.normalize(embedding[0].flatten(1).T, dim=1)
    classes = torch.unique(grid)
    kept = [value for value in classes if int((grid == value).sum()) >= 2]
    if len(kept) < 2:
        return None
    labels = grid.flatten()
    prototypes = torch.stack([
        F.normalize(tokens[labels == value].mean(0), dim=0) for value in kept
    ])
    targets = torch.full_like(labels, -1, dtype=torch.long)
    for index, value in enumerate(kept):
        selected = labels == value
        targets[selected] = index
    return prototypes, targets, [int(value) for value in kept]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", nargs="+")
    parser.add_argument("--case")
    parser.add_argument("--slice", type=int)
    parser.add_argument("--candidates", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.official:
        rows = official_candidates(args.official)
    elif args.case and args.slice is not None and args.candidates:
        rows = [{"case": args.case, "slice": args.slice, "candidate": path}
                for path in args.candidates]
    else:
        parser.error("provide --official or --case, --slice, and --candidates")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_sam_encoder("vit_b", str(SAM["vit_b"]), 512, 8, device)
    encoder = model.image_encoder
    case_cache = {}
    with torch.no_grad():
        for row in rows:
            key = (row["case"], row["slice"])
            if key not in case_cache:
                with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{key[0]}.npy.h5") as handle:
                    image = handle["image"][key[1]].astype(np.float32)
                    label = handle["label"][key[1]].astype(np.int64)
                image = torch.from_numpy(image)[None, None].repeat(1, 3, 1, 1).to(device)
                label = torch.from_numpy(label)[None, None].to(device)
                clean_feature = encode(encoder, image)
                bank = prototype_bank(clean_feature, label)
                if bank is None:
                    case_cache[key] = None
                else:
                    prototypes, targets, classes = bank
                    # Discard classes with too few source tokens for both scores.
                    clean_loss = prototype_loss(clean_feature, prototypes, targets, args.temperature)
                    case_cache[key] = (prototypes, targets, classes, float(clean_loss))
            cached = case_cache[key]
            if cached is None:
                row["source_label_prototype"] = None
                continue
            prototypes, targets, classes, clean_loss = cached
            changed = np.asarray(Image.open(ROOT / row["candidate"]).convert("L"),
                                 dtype=np.float32) / 255.0
            image = torch.from_numpy(changed)[None, None].repeat(1, 3, 1, 1).to(device)
            attacked_loss = float(prototype_loss(
                encode(encoder, image), prototypes, targets, args.temperature
            ))
            row["source_label_prototype"] = {
                "classes": classes, "clean_loss": clean_loss,
                "attacked_loss": attacked_loss,
                "loss_increase": attacked_loss - clean_loss,
            }
    rankings = {}
    for key in case_cache:
        subset = [row for row in rows if (row["case"], row["slice"]) == key
                  and row["source_label_prototype"] is not None]
        if len(subset) < 2:
            continue
        scores = [row["source_label_prototype"]["loss_increase"] for row in subset]
        ranking = {
            "predicted_best": subset[int(np.argmax(scores))]["candidate"],
        }
        if args.official:
            ranking.update({
                "SAMed_best": max(subset, key=lambda row: row["SAMed_dice_drop"])["candidate"],
                "MedSA_best": max(subset, key=lambda row: row["MedSA_dice_drop"])["candidate"],
                "SAMed_rho": float(spearmanr(scores, [row["SAMed_dice_drop"] for row in subset])[0]),
                "MedSA_rho": float(spearmanr(scores, [row["MedSA_dice_drop"] for row in subset])[0]),
            })
        rankings[f"{key[0]}/slice{key[1]}"] = ranking
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "base_SAM_only": True, "ground_truth_used_post_hoc": True,
        "target_weights_used_for_scoring": False,
        "target_metrics_used_for_post_hoc_ranking_only": bool(args.official),
        "temperature": args.temperature,
        "rows": rows, "within_case_rankings": rankings,
    }, indent=2))
    print(json.dumps(rankings, indent=2))


if __name__ == "__main__":
    main()
