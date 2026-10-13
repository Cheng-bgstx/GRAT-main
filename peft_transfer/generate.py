#!/usr/bin/env python3
"""Generate source-only PRSCA examples for Synapse, ISIC, or REFUGE."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.dataset_synapse import Synapse_dataset
from datasets.isic import ISIC2016
from datasets.refuge import REFUGE
from peft_transfer.core import ablation_config, attack_image
from segment_anything import sam_model_registry


DEFAULTS = {
    "Synapse": (8, ROOT / "data/Synapse_preprocess/test_vol_h5", ROOT / "data/Synapse_preprocess/lists"),
    "ISIC": (1, ROOT / "data/ISIC", None),
    "REFUGE": (2, ROOT / "data/REFUGE-Multirater", None),
}
CHECKPOINTS = {
    "vit_b": ROOT / "checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": ROOT / "checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": ROOT / "checkpoints/sam_vit_h_4b8939.pth",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DEFAULTS, required=True)
    parser.add_argument("--vit", choices=CHECKPOINTS, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--threat-model", choices=("strict-base", "legacy"), default="strict-base",
        help="strict-base permits only the public SAM weights and per-image optimization",
    )
    parser.add_argument(
        "--role", choices=("candidate", "internal-baseline", "external-baseline"),
        default="candidate",
        help="declares whether the run is a proposed method or a comparison baseline",
    )
    parser.add_argument("--data", default=None)
    parser.add_argument("--list-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--case", default=None)
    parser.add_argument("--max-cases", type=int, default=-1)
    parser.add_argument("--indices", default=None,
                        help="optional comma-separated dataset indices for a fixed small gate")
    parser.add_argument("--max-slices", type=int, default=-1)
    parser.add_argument("--slice-index", type=int, default=None,
                        help="attack only this slice index (mainly for 3D CT smoke tests)")
    parser.add_argument("--slice-indices", default=None,
                        help="optional comma-separated slice indices for a fixed 3D gate")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=10.0, help="L_inf budget in 0-255 units")
    parser.add_argument("--step-size", type=float, default=2.0, help="step in 0-255 units")
    parser.add_argument("--virtual-scale", type=float, default=0.08)
    parser.add_argument("--ranks", default="4,8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL", "AM", "AN", "AO", "AP", "AQ", "AR", "AS", "AT", "AU", "AV", "AW", "AX", "AY", "AZ", "BA", "BB", "BC", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BK", "BL", "BM", "BN", "BO", "BP", "BQ", "BR", "BS", "BT", "BU", "BV", "BW", "BX", "BY", "BZ", "CA", "CB", "CC", "CD", "CE", "CF", "CG", "CH", "CI", "CJ", "CK", "CL", "CM", "CN", "CO", "CP", "CQ", "CR", "CS", "CT", "CU", "CV", "CW", "CX", "CY", "CZ", "DA", "DB", "DC", "DD", "DE", "DF", "DG", "DH", "DI", "DJ", "DK", "DL", "DM", "DN", "DO", "DP", "DQ", "DR", "DS", "DT", "DU", "DV", "DW", "DX", "DY", "DZ", "EA", "EB", "EC", "ED", "EE", "EF", "EG", "EH", "EI", "LPDC"], default="B",
                        help="ordered sanity-chain variant; defaults to the verified source backbone")
    parser._option_string_actions["--variant"].choices.extend(["EJ", "EK", "EL", "EM", "EN", "EO", "EP", "EQ", "ER", "ES", "ET", "EU", "EV", "EW", "EX", "EY", "EZ", "FA", "FB", "FC", "FD", "FE", "FF", "FG", "FH", "FI", "FJ", "FK", "FL", "FM", "FN", "FO", "FP", "FQ", "FR", "FS", "FT", "FU", "FV", "FW", "FX", "FY", "FZ", "GA", "GB", "GC", "GD", "GE", "GF", "GG", "GH", "GI", "GJ", "GK", "GL", "GM", "GN", "GO", "GP", "GQ", "GR", "GS", "GT", "GU", "GV", "GW", "GX", "GY", "GZ", "HA", "HB", "HC", "HD", "HE", "HF", "HG", "HH", "HI", "HJ", "HK", "CTEOT", "CTEARLY", "CTEARLY010", "CTEARLY025", "CTEARLY050", "CTEARLYSTAGE"])
    parser.add_argument("--umi-prior", default=None,
                        help="optional UMI tensor added to the first update before budget projection")
    parser.add_argument("--prior-scale", type=float, default=1.0,
                        help="scale applied to the UMI tensor before first-step projection")
    parser.add_argument("--first-step-multiplier", type=float, default=2.5,
                        help="multiplier for the online update when UMI is injected")
    parser.add_argument("--prior-guidance-weight", type=float, default=0.0,
                        help="persistent per-step weight of the normalized UMI direction")
    parser.add_argument("--save-gradients", action="store_true",
                        help="save the final source-only input gradients for spectral support analysis")
    args = parser.parse_args(argv)
    if args.threat_model == "strict-base":
        if args.checkpoint is not None:
            parser.error("strict-base does not accept --checkpoint; use the public SAM checkpoint selected by --vit")
        if args.umi_prior is not None:
            parser.error("strict-base does not accept a trained UMI prior")
        if args.prior_guidance_weight != 0:
            parser.error("strict-base does not accept prior guidance")
        if ablation_config(args.variant).use_grat and args.role != "external-baseline":
            parser.error(
                "GRAT variants are external baselines; pass --role external-baseline"
            )
        if args.role == "external-baseline" and not ablation_config(args.variant).use_grat:
            parser.error("external-baseline currently requires an explicit GRAT variant")
    return args


def make_loader(args):
    _, default_data, default_list = DEFAULTS[args.dataset]
    data = args.data or str(default_data)
    list_dir = args.list_dir or (str(default_list) if default_list else None)
    # Repository 2D datasets use the legacy ``img_size`` name, while this
    # generator exposes the clearer ``image_size`` CLI option.
    args.img_size = args.image_size
    image_tf = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    mask_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    if args.dataset == "Synapse":
        dataset = Synapse_dataset(base_dir=data, list_dir=list_dir, split="test_vol")
    elif args.dataset == "ISIC":
        dataset = ISIC2016(args=args, data_path=data, mode="Test", transform=image_tf, transform_msk=mask_tf)
    else:
        dataset = REFUGE(args=args, data_path=data, mode="Test", transform=image_tf, transform_msk=mask_tf)
    if args.indices is not None:
        indices = [int(value) for value in args.indices.split(",") if value.strip()]
        if not indices or min(indices) < 0 or max(indices) >= len(dataset):
            raise IndexError(f"indices {indices} outside dataset length {len(dataset)}")
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)


def case_name(batch, index):
    if "case_name" in batch:
        value = batch["case_name"]
        return str(value[0] if isinstance(value, (list, tuple)) else value)
    meta = batch.get("image_meta_dict", {})
    value = meta.get("filename_or_obj", [f"case_{index:04d}"])
    return str(value[0] if isinstance(value, (list, tuple)) else value)


def image_slices(batch, dataset):
    value = batch["image"].float()
    if dataset != "Synapse":
        return value
    value = value.squeeze(0)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    return value.unsqueeze(1)


def save_image(adv, clean, epsilon: float, output: Path, name: str, slice_index: int, is_3d: bool):
    if not torch.isfinite(adv).all():
        raise FloatingPointError(f"non-finite adversarial image for {name}, slice {slice_index}")
    array = adv[0].detach().cpu().clamp(0, 1).numpy()
    if is_3d:
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        clean_array = clean[0, 0].detach().cpu().clamp(0, 1).numpy()
        lower = np.ceil(np.maximum(0.0, clean_array - epsilon) * 255.0).astype(np.int16)
        upper = np.floor(np.minimum(1.0, clean_array + epsilon) * 255.0).astype(np.int16)
        quantized = np.rint(array[0] * 255.0).astype(np.int16)
        quantized = np.clip(quantized, lower, upper).astype(np.uint8)
        Image.fromarray(quantized).save(folder / f"{slice_index}.png")
        return
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    hwc = np.transpose(array, (1, 2, 0))
    np.save(output / f"{stem}.npy", hwc)
    clean_hwc = np.transpose(clean[0].detach().cpu().clamp(0, 1).numpy(), (1, 2, 0))
    lower = np.ceil(np.maximum(0.0, clean_hwc - epsilon) * 255.0).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_hwc + epsilon) * 255.0).astype(np.int16)
    quantized = np.clip(np.rint(hwc * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
    Image.fromarray(quantized).save(output / f"{stem}.png")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = DEFAULTS[args.dataset][0]
    checkpoint = args.checkpoint or str(CHECKPOINTS[args.vit])
    model, _ = sam_model_registry[args.vit](
        image_size=args.image_size, num_classes=num_classes, checkpoint=checkpoint,
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    encoder = model.image_encoder.to(device).eval()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    preprocess = (lambda x: x[:, :1].repeat(1, 3, 1, 1)) if args.dataset == "Synapse" else normalize
    config = ablation_config(
        args.variant,
        epsilon=args.epsilon / 255.0, step_size=args.step_size / 255.0, steps=args.steps,
        ranks=tuple(int(value) for value in args.ranks.split(",") if value),
        virtual_scale=args.virtual_scale, seed=args.seed,
        prior_guidance_weight=args.prior_guidance_weight,
    )
    prior = torch.load(args.umi_prior, map_location=device) * args.prior_scale if args.umi_prior else None
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    traces = []
    processed = 0
    for batch_index, batch in tqdm(enumerate(make_loader(args)), desc=f"PRSCA {args.dataset}/{args.vit}"):
        name = case_name(batch, batch_index)
        if args.case and name != args.case:
            continue
        if args.max_cases >= 0 and processed >= args.max_cases:
            break
        slices = image_slices(batch, args.dataset)
        if args.slice_indices is not None:
            slice_indices = [int(value) for value in args.slice_indices.split(",") if value.strip()]
            if not slice_indices or min(slice_indices) < 0 or max(slice_indices) >= len(slices):
                raise IndexError(f"slice-indices {slice_indices} outside [0, {len(slices)})")
        elif args.slice_index is not None:
            if not 0 <= args.slice_index < len(slices):
                raise IndexError(f"slice-index {args.slice_index} outside [0, {len(slices)})")
            slice_indices = [args.slice_index]
        else:
            limit = len(slices) if args.max_slices < 0 else min(len(slices), args.max_slices)
            slice_indices = range(limit)
        for slice_index in slice_indices:
            image = slices[slice_index:slice_index + 1].to(device)
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            if image.shape[-2:] != (args.image_size, args.image_size):
                image = F.interpolate(image, (args.image_size, args.image_size), mode="bilinear", align_corners=False)
            gradient_sink = {} if args.save_gradients else None
            adv, trace = attack_image(
                encoder, image, config, preprocess=preprocess, initial_delta=prior,
                first_step_multiplier=args.first_step_multiplier,
                gradient_sink=gradient_sink,
            )
            save_image(adv, image, config.epsilon, output, name, slice_index, args.dataset == "Synapse")
            if gradient_sink is not None:
                gradient_folder = output / name if args.dataset == "Synapse" else output
                gradient_folder.mkdir(parents=True, exist_ok=True)
                gradient_name = f"{slice_index}.grad.pt" if args.dataset == "Synapse" else f"{Path(name).stem}.grad.pt"
                torch.save(gradient_sink, gradient_folder / gradient_name)
            traces.append({"case": name, "slice": slice_index, **{key: values[-1] for key, values in trace.items()}})
        processed += 1
    with open(output / "attack_meta.json", "w", encoding="utf-8") as handle:
        json.dump({
            "args": vars(args),
            "source_only": True,
            "research_role": args.role,
            "strict_base_only": args.threat_model == "strict-base",
            "external_prior_used": args.umi_prior is not None,
            "proxy_training_used": False,
            "target_checkpoint_used": False,
            "traces": traces,
        }, handle, indent=2)


if __name__ == "__main__":
    main()
