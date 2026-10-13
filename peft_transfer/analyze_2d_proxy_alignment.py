#!/usr/bin/env python3
"""Measure PEFT proxy-to-target encoder-gradient alignment post hoc."""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.isic import ISIC2016
from datasets.refuge import REFUGE
from peft_transfer.evaluate_official_slice import SAM, load_medsa_model
from peft_transfer.generate_2d_dual_peft import hierarchical_gradient


DATASETS = {
    "ISIC": (ISIC2016, ROOT / "data/ISIC", "isic", 1),
    "REFUGE": (REFUGE, ROOT / "data/REFUGE-Multirater", "refuge", 2),
}


def gradient(model, clean, point):
    with torch.no_grad():
        reference = model.image_encoder(model.preprocess(clean * 255.0)).detach()
    variable = point.detach().clone().requires_grad_(True)
    feature = model.image_encoder(model.preprocess(variable * 255.0))
    loss = F.l1_loss(feature * 0.9999, reference)
    value = torch.autograd.grad(loss, variable)[0].detach()
    return value / value.abs().mean().clamp_min(1e-12), float(loss.detach())


def load_point(folder, case, clean, device):
    if folder is None:
        generator = torch.Generator(device=device).manual_seed(1701)
        jitter = torch.randn(clean.shape, generator=generator, device=device)
        return (clean + 2.0 / 255.0 * jitter.sign()).clamp(0.0, 1.0)
    value = np.load(folder / f"{case}.npy").astype(np.float32)
    return torch.from_numpy(value.transpose(2, 0, 1))[None].to(device)


def radial_components(gradient):
    height, width = gradient.shape[-2:]
    fy = torch.fft.fftfreq(height, device=gradient.device)
    fx = torch.fft.fftfreq(width, device=gradient.device)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / np.sqrt(0.5)
    spectrum = torch.fft.fft2(gradient.float(), norm="ortho")
    masks = {
        "low": radius < 0.1,
        "mid": (radius >= 0.1) & (radius < 0.3),
        "high": radius >= 0.3,
    }
    return {
        name: torch.fft.ifft2(spectrum * mask, norm="ortho").real
        for name, mask in masks.items()
    }


def polar_tangent_components(gradient, image):
    """Gradient responses to relative Fourier amplitude and phase changes."""
    image_spectrum = torch.fft.fft2(image.float(), norm="ortho")
    gradient_spectrum = torch.fft.fft2(gradient.float(), norm="ortho")
    cross = gradient_spectrum.conj() * image_spectrum
    return {"log_amplitude": cross.real, "phase": -cross.imag}


def frequency_masks(height, width, device):
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / np.sqrt(0.5)
    return {
        "low": radius < 0.1,
        "mid": (radius >= 0.1) & (radius < 0.3),
        "high": radius >= 0.3,
    }


def polar_alignment(direction, target, image):
    left = polar_tangent_components(direction, image)
    right = polar_tangent_components(target, image)
    masks = frequency_masks(image.shape[-2], image.shape[-1], image.device)
    return {
        coordinate: {
            band: alignment_metrics(
                left_value * mask[None, None], right[coordinate] * mask[None, None]
            )
            for band, mask in masks.items()
        }
        for coordinate, left_value in left.items()
    }


def alignment_metrics(direction, target):
    active = direction != 0
    active_count = active.count_nonzero().clamp_min(1)
    return {
        "cosine": float(F.cosine_similarity(
            direction.flatten(1), target.flatten(1), dim=1
        ).mean()),
        "sign_agreement": float((direction.sign() == target.sign()).float().mean()),
        "active_fraction": float(active.float().mean()),
        "active_sign_agreement": float(
            ((direction.sign() == target.sign()) & active).count_nonzero() / active_count
        ),
        "normalized_l1_error": float((direction - target).abs().mean()),
        "target_gain_of_proxy_sign": float((target * direction.sign()).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--target", required=True)
    parser.add_argument("--proxies", nargs="+", required=True,
                        help="NAME=checkpoint specifications")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--proxy-group-size", type=int, default=1)
    parser.add_argument("--seed-lambda", type=float, default=0.0)
    parser.add_argument("--trajectory-lambda", type=float, default=0.0)
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset_cls, data_path, medsa_dataset, masks = DATASETS[args.dataset]
    image_tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    mask_tf = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = dataset_cls(
        args=Namespace(img_size=512), data_path=str(data_path), mode="Test",
        transform=image_tf, transform_msk=mask_tf,
    )
    specs = [spec.split("=", 1) for spec in args.proxies]
    models = {
        name: load_medsa_model(
            args.vit, str(SAM[args.vit]), str(ROOT / checkpoint), 512, masks,
            device=device, dataset=medsa_dataset,
        )
        for name, checkpoint in [("target", args.target), *specs]
    }
    for model in models.values():
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    candidate = ROOT / args.candidate if args.candidate else None
    rows = []
    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        sample = dataset[index]
        case = str(sample["image_meta_dict"]["filename_or_obj"])
        clean = sample["image"].float()[None].to(device)
        point = load_point(candidate, case, clean, device)
        gradients = {}
        losses = {}
        for name, model in models.items():
            gradients[name], losses[name] = gradient(model, clean, point)
        target = gradients["target"]
        for name, _checkpoint in specs:
            proxy = gradients[name]
            row = {
                "index": index,
                "case": case,
                "proxy": name,
                "proxy_loss": losses[name],
                "target_loss": losses["target"],
                **alignment_metrics(proxy, target),
            }
            proxy_bands = radial_components(proxy)
            target_bands = radial_components(target)
            row["frequency"] = {
                band: alignment_metrics(proxy_bands[band], target_bands[band])
                for band in proxy_bands
            }
            row["polar_tangent"] = polar_alignment(proxy, target, point)
            rows.append(row)

        proxy_gradients = [gradients[name] for name, _checkpoint in specs]
        ensemble_directions = {
            "ensemble_mean": hierarchical_gradient(proxy_gradients)[0],
            "ensemble_hierarchical": hierarchical_gradient(
                proxy_gradients,
                group_size=args.proxy_group_size,
                seed_lambda=args.seed_lambda,
                trajectory_lambda=args.trajectory_lambda,
            )[0],
        }
        for name, direction in ensemble_directions.items():
            direction = direction / direction.abs().mean().clamp_min(1e-12)
            rows.append({
                "index": index,
                "case": case,
                "proxy": name,
                "proxy_loss": float(np.mean([losses[key] for key, _ in specs])),
                "target_loss": losses["target"],
                **alignment_metrics(direction, target),
                "frequency": {
                    band: alignment_metrics(component, radial_components(target)[band])
                    for band, component in radial_components(direction).items()
                },
                "polar_tangent": polar_alignment(direction, target, point),
            })
    summary = {
        name: {
            key: float(np.mean([row[key] for row in rows if row["proxy"] == name]))
            for key in (
                "cosine", "sign_agreement", "active_fraction", "active_sign_agreement",
                "normalized_l1_error",
                "target_gain_of_proxy_sign", "proxy_loss", "target_loss",
            )
        }
        for name in [*[name for name, _checkpoint in specs], "ensemble_mean", "ensemble_hierarchical"]
    }
    payload = {
        "protocol": vars(args),
        "victim_use": "post-hoc analysis only",
        "rows": rows,
        "summary": summary,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
