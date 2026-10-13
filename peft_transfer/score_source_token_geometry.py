#!/usr/bin/env python3
"""Test whether source-SAM token geometry ranks frozen transfer attacks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr

from peft_transfer.evaluate_synapse_slice import ROOT, SAM
from peft_transfer.medical_invariants import capture_features, load_test_image
from peft_transfer.score_medical_invariant_candidates import official_candidates

sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_sam_encoder


def token_geometry(tokens: torch.Tensor, grid: int = 16) -> torch.Tensor:
    side = int(round(tokens.shape[0] ** 0.5))
    if side * side != tokens.shape[0]:
        raise ValueError("Expected a square token grid")
    features = tokens.T.reshape(1, -1, side, side)
    features = F.adaptive_avg_pool2d(features, (grid, grid))
    matrix = features.flatten(2).transpose(1, 2)[0]
    matrix = matrix - matrix.mean(0, keepdim=True)
    matrix = F.normalize(matrix, dim=1)
    gram = matrix @ matrix.T
    return gram / gram.norm().clamp_min(1e-8)


def geometry_score(clean: torch.Tensor, attacked: torch.Tensor) -> float:
    return float((token_geometry(attacked) - token_geometry(clean)).norm())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True,
                        help="Frozen candidate index with official post-hoc metrics")
    parser.add_argument("--additional-official", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = json.loads(Path(args.prediction).read_text())["rows"]
    rows.extend(official_candidates(args.additional_official))
    encoder = load_sam_encoder("vit_b", str(SAM["vit_b"]), 512, 8, device).image_encoder
    layers = ("0", "5", "11", "neck")
    cache = {}
    with torch.no_grad():
        for row in rows:
            key = (row["case"], row["slice"])
            if key not in cache:
                image = load_test_image(*key, device)
                cache[key] = capture_features(encoder, image, layers)
            array = np.asarray(Image.open(ROOT / row["candidate"]).convert("L"),
                               dtype=np.float32) / 255.0
            changed = torch.from_numpy(array)[None, None].repeat(1, 3, 1, 1).to(device)
            attacked = capture_features(encoder, changed, layers)
            row["source_geometry"] = {
                layer: geometry_score(cache[key][layer], attacked[layer]) for layer in layers
            }

    rankings = {}
    for key in cache:
        subset = [row for row in rows if (row["case"], row["slice"]) == key]
        if len(subset) < 2:
            continue
        case_key = f"{key[0]}/slice{key[1]}"
        rankings[case_key] = {}
        for layer in layers:
            scores = [row["source_geometry"][layer] for row in subset]
            actual = [row["MedSA_dice_drop"] for row in subset]
            rankings[case_key][layer] = {
                "predicted_best": subset[int(np.argmax(scores))]["candidate"],
                "MedSA_best": subset[int(np.argmax(actual))]["candidate"],
                "MedSA_rho": float(spearmanr(scores, actual)[0]),
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "source_only_score": True,
        "target_metrics_used_for_post_hoc_ranking_only": True,
        "layers": layers,
        "rows": rows,
        "within_case_rankings": rankings,
    }, indent=2))
    print(json.dumps(rankings, indent=2))


if __name__ == "__main__":
    main()
