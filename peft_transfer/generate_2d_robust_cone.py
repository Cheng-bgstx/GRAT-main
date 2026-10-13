#!/usr/bin/env python3
"""Generate a two-surrogate max-min attack from a base SAM and PEFT trajectory proxy."""
from __future__ import annotations

import argparse
import json
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from datasets.isic import ISIC2016
from datasets.refuge import REFUGE
from peft_transfer.core import _asymmetric_step, _unpack_encoder
from peft_transfer.evaluate_official_slice import ROOT, SAM, load_samed_model
from segment_anything import sam_model_registry


DATASETS = {
    "ISIC": (ISIC2016, ROOT / "data/ISIC", 1),
    "REFUGE": (REFUGE, ROOT / "data/REFUGE-Multirater", 2),
}


def encoder_features(encoder, image):
    final, layers = _unpack_encoder(encoder(image))
    return final, layers[0]


def normalize_gradient(gradient):
    return gradient / gradient.abs().mean().clamp_min(1e-12)


def robust_linf_direction(left, right, grid_size=101):
    """Solve the two-gradient L-inf max-min problem through its scalar dual."""
    left = normalize_gradient(left)
    right = normalize_gradient(right)
    weights = torch.linspace(0.0, 1.0, grid_size, device=left.device, dtype=left.dtype)
    mixtures = (
        weights[:, None, None, None, None] * left[None]
        + (1.0 - weights[:, None, None, None, None]) * right[None]
    )
    dual_values = mixtures.abs().flatten(1).mean(1)
    index = int(dual_values.argmin())
    weight = weights[index]
    mixture = mixtures[index]
    direction = normalize_gradient(mixture)
    cosine = F.cosine_similarity(left.flatten(1), right.flatten(1), dim=1).mean()
    sign_agreement = (left.sign() == right.sign()).float().mean()
    return direction, {
        "dual_weight_base": float(weight),
        "dual_value": float(dual_values[index]),
        "gradient_cosine": float(cosine),
        "sign_agreement": float(sign_agreement),
    }


def attack(base, proxy, clean, normalize, prior, proxy_feature):
    with torch.no_grad():
        base_final, base_early = encoder_features(base, normalize(clean))
        proxy_final, proxy_early = encoder_features(proxy, normalize(clean))
        base_reference = base_early.detach()
        proxy_reference = (proxy_early if proxy_feature == "early" else proxy_final).detach()

    adv = clean.clone()
    momentum = torch.zeros_like(clean)
    trace = []
    for step in range(10):
        adv = adv.detach().requires_grad_(True)
        _base_final, base_early = encoder_features(base, normalize(adv))
        proxy_final, proxy_early = encoder_features(proxy, normalize(adv))
        proxy_value = proxy_early if proxy_feature == "early" else proxy_final
        base_loss = F.l1_loss(base_early * 0.9999, base_reference)
        proxy_loss = F.l1_loss(proxy_value * 0.9999, proxy_reference)
        base_gradient = torch.autograd.grad(base_loss, adv, retain_graph=True)[0]
        proxy_gradient = torch.autograd.grad(proxy_loss, adv)[0]
        gradient, diagnostics = robust_linf_direction(base_gradient, proxy_gradient)
        momentum = 0.5 * momentum + gradient
        update = _asymmetric_step(momentum, 2.0 / 255.0)
        if step == 0 and prior is not None:
            update = 2.5 * update + prior
        adv = adv.detach() + update
        adv = torch.maximum(torch.minimum(adv, clean + 10.0 / 255.0), clean - 10.0 / 255.0)
        adv = adv.clamp(0.0, 1.0)
        trace.append({
            "step": step,
            "base_loss": float(base_loss.detach()),
            "proxy_loss": float(proxy_loss.detach()),
            **diagnostics,
        })
    return adv, trace


def save_candidate(folder, case, adv, clean):
    folder.mkdir(parents=True, exist_ok=True)
    chw = adv[0].detach().cpu().numpy().astype(np.float32)
    clean_hwc = np.transpose(clean[0].detach().cpu().numpy(), (1, 2, 0))
    value_hwc = np.transpose(chw, (1, 2, 0))
    np.save(folder / f"{case}.npy", value_hwc)
    lower = np.ceil(np.maximum(0.0, clean_hwc - 10 / 255) * 255).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_hwc + 10 / 255) * 255).astype(np.int16)
    quantized = np.clip(np.rint(value_hwc * 255).astype(np.int16), lower, upper).astype(np.uint8)
    Image.fromarray(quantized).save(folder / f"{case}.png")
    return float(np.max(np.abs(value_hwc - clean_hwc)) * 255.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--proxy-checkpoint", required=True)
    parser.add_argument("--umi-prior", default=None)
    parser.add_argument("--indices", default="0,1,2,3")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset_cls, data_path, classes = DATASETS[args.dataset]
    image_tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    mask_tf = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = dataset_cls(
        args=Namespace(img_size=512), data_path=str(data_path), mode="Test",
        transform=image_tf, transform_msk=mask_tf,
    )
    proxy_model = load_samed_model(
        args.vit, str(SAM[args.vit]), str(ROOT / args.proxy_checkpoint),
        512, classes, device=device,
    )
    proxy = proxy_model.sam.image_encoder
    source, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=classes, checkpoint=str(SAM[args.vit]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    base = source.image_encoder.to(device).eval()
    for parameter in list(base.parameters()) + list(proxy.parameters()):
        parameter.requires_grad_(False)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    prior = torch.load(ROOT / args.umi_prior, map_location=device) if args.umi_prior else None
    output = ROOT / args.out
    records = []
    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        sample = dataset[index]
        case = str(sample["image_meta_dict"]["filename_or_obj"])
        clean = sample["image"].float()[None].to(device)
        for proxy_feature in ("early", "final"):
            adv, trace = attack(base, proxy, clean, normalize, prior, proxy_feature)
            name = f"ROBUST_{proxy_feature.upper()}"
            record = {
                "index": index,
                "case": case,
                "candidate": name,
                "target_checkpoint_used": False,
                "proxy_checkpoint": args.proxy_checkpoint,
                "proxy_feature": proxy_feature,
                "linf_255": save_candidate(output / name, case, adv, clean),
                "trace": trace,
            }
            records.append(record)
            print(json.dumps(record, indent=2), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "generation.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
