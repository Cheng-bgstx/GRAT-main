#!/usr/bin/env python3
"""Report per-slice MedSA metrics for diagnosis; never use for attack selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from medpy import metric

from peft_transfer.evaluate_official_slice import (
    ROOT,
    SAM,
    load_medsa_function,
    load_medsa_model,
    reset_seed,
)


def slice_metrics(prediction: np.ndarray, label: np.ndarray):
    dices = []
    distances = []
    per_class = []
    for class_id in range(1, 9):
        pred = prediction == class_id
        true = label == class_id
        union = int(pred.sum() + true.sum())
        dice = 1.0 if union == 0 else 2.0 * int((pred & true).sum()) / union
        hd95 = None
        if pred.any() and true.any():
            hd95 = float(metric.binary.hd95(pred, true))
            distances.append(hd95)
        dices.append(dice)
        per_class.append({"class": class_id, "dice": dice, "hd95": hd95})
    return float(np.mean(dices)), float(np.mean(distances)) if distances else 0.0, per_class


@torch.no_grad()
def predict(model, prompt_module, image: torch.Tensor, label: torch.Tensor):
    _unused, points, _masks, point_labels = prompt_module.generate_click_prompt(image, label)
    if points.dim() == 3 and points.shape[1] == 2:
        points = points.permute(0, 2, 1)
    prompt = (points.float().to(image.device), point_labels.int().to(image.device))
    embedding = model.image_encoder(model.preprocess(image))
    sparse, dense = model.prompt_encoder(points=prompt, boxes=None, masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
    )
    logits = F.interpolate(logits, label.shape[-2:], mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--prompt-seed", type=int, default=24)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_medsa_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
        512, 9, device=device, dataset="Synapse",
    ).eval()
    prompt_module = load_medsa_function("Synapse", 9)
    candidate = None if args.candidate == "clean" else ROOT / args.candidate
    rows = []
    predictions = []
    labels = []
    reset_seed(args.prompt_seed)
    with h5py.File(
        ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r"
    ) as handle:
        for slice_index in range(len(handle["image"])):
            image_np = (
                handle["image"][slice_index].astype(np.float32)
                if candidate is None else
                np.asarray(
                    Image.open(candidate / f"{slice_index}.png").convert("L"),
                    dtype=np.float32,
                ) / 255.0
            )
            label_np = handle["label"][slice_index].astype(np.int64)
            image = torch.from_numpy(image_np)[None, None].repeat(1, 3, 1, 1).to(device)
            label = torch.from_numpy(label_np)[None].to(device)
            prediction = predict(model, prompt_module, image, label)
            predictions.append(prediction)
            labels.append(label_np)
            dice, hd95, per_class = slice_metrics(prediction, label_np)
            row = {
                "slice": slice_index,
                "present_foreground_classes": int(len(np.unique(label_np[label_np > 0]))),
                "dice": dice,
                "hd95": hd95,
                "per_class": per_class,
            }
            rows.append(row)
            print(json.dumps({key: row[key] for key in row if key != "per_class"}), flush=True)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_volume = np.stack(predictions)
    label_volume = np.stack(labels)
    volume_per_class = []
    for class_id in range(1, 9):
        pred = prediction_volume == class_id
        true = label_volume == class_id
        union = int(pred.sum() + true.sum())
        dice = 1.0 if union == 0 else 2.0 * int((pred & true).sum()) / union
        hd95 = (
            float(metric.binary.hd95(pred, true))
            if pred.any() and true.any() else float(np.sqrt(np.sum(np.array(pred.shape) ** 2)))
        )
        volume_per_class.append({"class": class_id, "dice": dice, "hd95": hd95})
    output.write_text(json.dumps({
        "diagnostic_only": True,
        "uses_target_checkpoint": True,
        "candidate": args.candidate,
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_hd95": float(np.mean([row["hd95"] for row in rows])),
        "volume_mDSC": float(np.mean([row["dice"] for row in volume_per_class])),
        "volume_mHD95": float(np.mean([row["hd95"] for row in volume_per_class])),
        "volume_per_class": volume_per_class,
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
