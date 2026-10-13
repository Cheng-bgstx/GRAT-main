#!/usr/bin/env python3
"""Evaluate a fixed 2D ISIC/REFUGE subset with both official test functions."""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.isic import ISIC2016
from datasets.refuge import REFUGE
from peft_transfer.evaluate_official_slice import (
    SAM,
    load_medsa_function,
    load_medsa_model,
    load_samed_model,
    reset_seed,
    test_single_volume,
)


DATASETS = {
    # The local REFUGE checkpoint and dataset use two mask channels (disc/cup).
    "ISIC": (ISIC2016, ROOT / "data/ISIC", 1, "isic", 1),
    "REFUGE": (REFUGE, ROOT / "data/REFUGE-Multirater", 2, "refuge", 2),
}


class Fixed2DDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "image": torch.from_numpy(row["image"]).float(),
            "label": torch.from_numpy(row["label"]).float(),
            "image_meta_dict": {"filename_or_obj": row["case"]},
        }


def parse_candidate(spec: str):
    name, folder = spec.split("=", 1)
    return name, ROOT / folder


def load_candidate(folder: Path, case: str, suffix: str) -> np.ndarray:
    if suffix == ".npy":
        value = np.load(folder / f"{case}.npy").astype(np.float32)
        return np.transpose(value, (2, 0, 1))
    value = np.asarray(Image.open(folder / f"{case}.png").convert("RGB"), dtype=np.float32)
    return np.transpose(value / 255.0, (2, 0, 1))


def aggregate(rows):
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("dice", "hd95")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--samed-checkpoint", default=None,
                        help="optional held-out SAMed checkpoint override")
    parser.add_argument("--medsa-checkpoint", default=None,
                        help="optional held-out Med-SA checkpoint override")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--indices", default=None,
                        help="optional comma-separated dataset indices; overrides max-cases")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-seed", type=int, default=24)
    parser.add_argument("--per-case-medsa", action="store_true",
                        help="also evaluate paired per-case Med-SA metrics with a fixed seed per case")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_cls, data_path, classes, medsa_name, medsa_multimask = DATASETS[args.dataset]
    dataset_args = Namespace(img_size=512)
    image_tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    mask_tf = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = dataset_cls(
        args=dataset_args, data_path=str(data_path), mode="Test",
        transform=image_tf, transform_msk=mask_tf,
    )
    candidates = dict(map(parse_candidate, args.candidates))
    indices = (
        [int(value) for value in args.indices.split(",") if value.strip()]
        if args.indices is not None else list(range(min(args.max_cases, len(dataset))))
    )
    if not indices or min(indices) < 0 or max(indices) >= len(dataset):
        raise IndexError(f"indices {indices} outside dataset length {len(dataset)}")
    cases = []
    for index in indices:
        sample = dataset[index]
        meta = sample["image_meta_dict"]["filename_or_obj"]
        case = str(meta)
        clean = sample["image"].numpy().astype(np.float32)
        label = sample["label"].numpy().astype(np.float32)
        images = {"clean": clean}
        images.update({name: load_candidate(folder, case, ".npy") for name, folder in candidates.items()})
        png_images = {name: load_candidate(folder, case, ".png") for name, folder in candidates.items()}
        cases.append({"case": case, "label": label, "images": images, "png_images": png_images})

    audit = {}
    for name in candidates:
        if not all(np.isfinite(row["images"][name]).all() for row in cases):
            raise FloatingPointError(f"candidate {name!r} contains non-finite NPY values")
        npy_distances = [np.abs(row["images"][name] - row["images"]["clean"]) * 255.0 for row in cases]
        png_distances = [np.abs(row["png_images"][name] - row["images"]["clean"]) * 255.0 for row in cases]
        audit[name] = {
            "npy_linf_255": float(max(value.max() for value in npy_distances)),
            "png_linf_255": float(max(value.max() for value in png_distances)),
            "npy_pixels_over_budget": int(sum(np.count_nonzero(value > 10.0001) for value in npy_distances)),
            "png_pixels_over_budget": int(sum(np.count_nonzero(value > 10.0001) for value in png_distances)),
        }

    result = {"protocol": vars(args), "cases": [row["case"] for row in cases], "audit": audit}
    samed = load_samed_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / (args.samed_checkpoint or
                    f"checkpoints/best_weights/SAMed_{args.dataset}_{args.vit}.pth")),
        512, classes, device=device,
    )
    samed_rows = {name: [] for name in ("clean", *candidates)}
    for row in cases:
        clean_tensor = torch.from_numpy(row["images"]["clean"])[None]
        label_tensor = torch.from_numpy(row["label"])[None]
        for name, image in row["images"].items():
            metric = test_single_volume(
                torch.from_numpy(image)[None], clean_tensor, label_tensor,
                samed, classes=classes, multimask_output=classes > 1,
                patch_size=[512, 512], test_save_path=None,
                case=row["case"], z_spacing=1, is_3d=False,
            )
            samed_rows[name].append({
                "case": row["case"],
                "dice": float(np.mean(metric, axis=0)[0]),
                "hd95": float(np.mean(metric, axis=0)[1]),
            })
    result["SAMed"] = {
        name: {"mean": aggregate(rows), "per_case": rows}
        for name, rows in samed_rows.items()
    }
    del samed
    torch.cuda.empty_cache()

    medsa = load_medsa_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / (args.medsa_checkpoint or
                    f"checkpoints/best_weights/MSA_{args.dataset}_{args.vit}.pth")),
        512, medsa_multimask, device=device, dataset=medsa_name,
    )
    validation_sam = load_medsa_function(medsa_name, medsa_multimask).validation_sam
    medsa_args = Namespace(
        dataset=medsa_name, multimask_output=medsa_multimask, thd=False, evl_chunk=None,
        gpu_device=0, image_size=512, out_size=512, net="sam", vis=0,
        path_helper={"sample_path": "/tmp"},
    )
    medsa_results = {}
    for name in ("clean", *candidates):
        rows = []
        for row in cases:
            image = row["images"]["clean"] if name == "clean" else row["png_images"][name]
            # The official Med-SA 2D dataloader applies ToTensor() followed by
            # x255; its SAM normalization constants are also on the 0--255 scale.
            rows.append({"case": row["case"], "image": image * 255.0, "label": row["label"]})
        reset_seed(args.prompt_seed)
        loader = DataLoader(Fixed2DDataset(rows), batch_size=1, shuffle=False, num_workers=0)
        _loss, metrics = validation_sam(medsa_args, loader, 0, medsa)
        if args.dataset == "ISIC":
            iou, dice, hd95 = map(float, metrics)
            medsa_results[name] = {"dice": dice, "hd95": hd95, "iou": iou}
        else:
            iou_d, iou_c, dice_d, dice_c, hd_d, hd_c = map(float, metrics)
            medsa_results[name] = {
                "dice": (dice_d + dice_c) / 2.0, "hd95": (hd_d + hd_c) / 2.0,
                "disc_dice": dice_d, "cup_dice": dice_c,
                "disc_hd95": hd_d, "cup_hd95": hd_c,
            }
    result["MedSA"] = medsa_results
    if args.per_case_medsa:
        individual = {name: [] for name in ("clean", *candidates)}
        for index, row in enumerate(cases):
            for name in individual:
                image = row["images"]["clean"] if name == "clean" else row["png_images"][name]
                sample = [{"case": row["case"], "image": image * 255.0,
                           "label": row["label"]}]
                reset_seed(args.prompt_seed + index)
                loader = DataLoader(Fixed2DDataset(sample), batch_size=1,
                                    shuffle=False, num_workers=0)
                _loss, metrics = validation_sam(medsa_args, loader, 0, medsa)
                if args.dataset == "ISIC":
                    dice, hd95 = float(metrics[1]), float(metrics[2])
                else:
                    dice = (float(metrics[2]) + float(metrics[3])) / 2
                    hd95 = (float(metrics[4]) + float(metrics[5])) / 2
                individual[name].append({
                    "case": row["case"], "dice": dice, "hd95": hd95,
                })
        result["MedSA_per_case_paired"] = individual
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
