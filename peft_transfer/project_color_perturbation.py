#!/usr/bin/env python3
"""Project saved 2D attacks onto controlled luminance/chroma mixtures."""
from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.isic import ISIC2016
from datasets.refuge import REFUGE


DATASETS = {
    "ISIC": (ISIC2016, ROOT / "data/ISIC"),
    "REFUGE": (REFUGE, ROOT / "data/REFUGE-Multirater"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--chroma-scale", type=float, required=True)
    parser.add_argument("--drop-luminance", action="store_true")
    parser.add_argument(
        "--channel-gains", default=None,
        help="optional comma-separated RGB gains; bypasses basis projection",
    )
    parser.add_argument(
        "--rg-fourier-keep", type=float, default=None,
        help="keep this fraction of largest R/G rFFT amplitudes; preserve blue",
    )
    parser.add_argument(
        "--rg-saliency-keep", type=float, default=None,
        help="keep this fraction of R/G coefficients with largest positive source contribution",
    )
    parser.add_argument(
        "--rg-spatial-saliency-keep", type=float, default=None,
        help="keep this fraction of R/G pixels with largest positive source contribution",
    )
    parser.add_argument("--gradient-dir", default=None)
    parser.add_argument(
        "--basis", choices=("chroma", "rg", "yb", "luma", "red", "green", "blue"),
        default="chroma",
    )
    parser.add_argument("--epsilon", type=float, default=10.0)
    args = parser.parse_args()

    dataset_cls, data_path = DATASETS[args.dataset]
    image_tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    mask_tf = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = dataset_cls(
        args=Namespace(img_size=512), data_path=str(data_path), mode="Test",
        transform=image_tf, transform_msk=mask_tf,
    )
    source = ROOT / args.source
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    epsilon = args.epsilon / 255.0

    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        sample = dataset[index]
        case = str(sample["image_meta_dict"]["filename_or_obj"])
        clean = sample["image"].numpy().astype(np.float32)
        adv = np.load(source / f"{case}.npy").astype(np.float32).transpose(2, 0, 1)
        delta = adv - clean
        luminance = delta.mean(axis=0, keepdims=True)
        chroma = delta - luminance
        if args.rg_spatial_saliency_keep is not None:
            keep = float(args.rg_spatial_saliency_keep)
            if not 0.0 < keep <= 1.0 or args.gradient_dir is None:
                raise ValueError("rg-spatial-saliency-keep requires a fraction in (0,1] and --gradient-dir")
            gradient_record = torch.load(
                ROOT / args.gradient_dir / f"{case}.grad.pt", map_location="cpu"
            )
            gradient = gradient_record["final"][0].numpy().astype(np.float32)
            direction = delta.copy()
            for channel in (0, 1):
                contribution = np.maximum(gradient[channel] * delta[channel], 0.0)
                count = max(1, int(np.ceil(keep * contribution.size)))
                threshold = np.partition(contribution.ravel(), -count)[-count]
                mask = (contribution >= threshold) & (contribution > 0)
                direction[channel] *= mask
            anchor = np.zeros_like(delta)
        elif args.rg_saliency_keep is not None:
            keep = float(args.rg_saliency_keep)
            if not 0.0 < keep <= 1.0 or args.gradient_dir is None:
                raise ValueError("rg-saliency-keep requires a fraction in (0,1] and --gradient-dir")
            gradient_record = torch.load(
                ROOT / args.gradient_dir / f"{case}.grad.pt", map_location="cpu"
            )
            gradient = gradient_record["final"][0].numpy().astype(np.float32)
            direction = delta.copy()
            for channel in (0, 1):
                spectrum = np.fft.rfft2(delta[channel], norm="ortho")
                gradient_spectrum = np.fft.rfft2(gradient[channel], norm="ortho")
                contribution = np.real(np.conj(gradient_spectrum) * spectrum)
                if spectrum.shape[1] > 2:
                    contribution[:, 1:-1] *= 2.0
                count = max(1, int(np.ceil(keep * spectrum.size)))
                positive = np.maximum(contribution, 0.0)
                threshold = np.partition(positive.ravel(), -count)[-count]
                mask = (positive >= threshold) & (positive > 0)
                spectrum = spectrum * mask
                direction[channel] = np.fft.irfft2(
                    spectrum, s=delta[channel].shape, norm="ortho"
                ).real.astype(np.float32)
            anchor = np.zeros_like(delta)
        elif args.rg_fourier_keep is not None:
            keep = float(args.rg_fourier_keep)
            if not 0.0 < keep <= 1.0:
                raise ValueError("rg-fourier-keep must lie in (0,1]")
            direction = delta.copy()
            for channel in (0, 1):
                spectrum = np.fft.rfft2(delta[channel], norm="ortho")
                count = max(1, int(np.ceil(keep * spectrum.size)))
                threshold = np.partition(np.abs(spectrum).ravel(), -count)[-count]
                spectrum = spectrum * (np.abs(spectrum) >= threshold)
                direction[channel] = np.fft.irfft2(
                    spectrum, s=delta[channel].shape, norm="ortho"
                ).real.astype(np.float32)
            anchor = np.zeros_like(delta)
        elif args.channel_gains is not None:
            gains = np.asarray(
                [float(value) for value in args.channel_gains.split(",")], dtype=np.float32
            )
            if gains.shape != (3,):
                raise ValueError("channel-gains must contain exactly three RGB values")
            direction = delta * gains[:, None, None]
            anchor = np.zeros_like(delta)
        elif args.basis == "chroma":
            direction = chroma
            anchor = np.zeros_like(delta) if args.drop_luminance else luminance
        elif args.basis == "luma":
            direction = luminance
            anchor = np.zeros_like(delta)
        elif args.basis in {"rg", "yb"}:
            vector = (
                np.asarray([1.0, -1.0, 0.0], dtype=np.float32) / np.sqrt(2.0)
                if args.basis == "rg"
                else np.asarray([1.0, 1.0, -2.0], dtype=np.float32) / np.sqrt(6.0)
            )
            coefficient = np.sum(delta * vector[:, None, None], axis=0, keepdims=True)
            direction = coefficient * vector[:, None, None]
            anchor = np.zeros_like(delta) if args.drop_luminance else luminance
        else:
            channel = {"red": 0, "green": 1, "blue": 2}[args.basis]
            direction = np.zeros_like(delta)
            direction[channel] = delta[channel]
            anchor = np.zeros_like(delta)
        projected = np.clip(anchor + args.chroma_scale * direction, -epsilon, epsilon)
        projected = np.clip(clean + projected, 0.0, 1.0)
        if not np.isfinite(projected).all():
            raise FloatingPointError(f"non-finite projection for {case}")

        hwc = projected.transpose(1, 2, 0)
        np.save(output / f"{case}.npy", hwc)
        clean_hwc = clean.transpose(1, 2, 0)
        lower = np.ceil(np.maximum(0.0, clean_hwc - epsilon) * 255.0).astype(np.int16)
        upper = np.floor(np.minimum(1.0, clean_hwc + epsilon) * 255.0).astype(np.int16)
        quantized = np.clip(np.rint(hwc * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
        Image.fromarray(quantized).save(output / f"{case}.png")


if __name__ == "__main__":
    main()
