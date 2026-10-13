#!/usr/bin/env python3
"""Post-hoc GT/victim gradient audit; never called by attack generation."""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torchvision import transforms

from datasets.isic import ISIC2016
from peft_transfer.common_ascent import common_ascent_direction, minimum_norm_convex_combination
from peft_transfer.evaluate_official_slice import ROOT, SAM, load_medsa_model, load_samed_model
from peft_transfer.generate_isic_task_cone import medsa_logits, pseudo_task_loss, samed_logits


def gradient(loss, image):
    return torch.autograd.grad(loss, image)[0].detach().float()


def unit(value):
    return value / value.norm().clamp_min(1e-30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--lora-proxies", nargs=2, required=True)
    parser.add_argument("--adapter-proxy", required=True)
    parser.add_argument("--indices", default="0,1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoints = [ROOT / path for path in (*args.lora_proxies, args.adapter_proxy)]
    proxies = [
        load_samed_model(args.vit, str(SAM[args.vit]), str(path), 512, 1, device=device)
        for path in checkpoints[:2]
    ]
    proxies.append(load_medsa_model(
        args.vit, str(SAM[args.vit]), str(checkpoints[2]), 512, 1,
        device=device, dataset="isic",
    ))
    targets = [
        load_samed_model(
            args.vit, str(SAM[args.vit]),
            str(ROOT / f"checkpoints/best_weights/SAMed_ISIC_{args.vit}.pth"),
            512, 1, device=device,
        ),
        load_medsa_model(
            args.vit, str(SAM[args.vit]),
            str(ROOT / f"checkpoints/best_weights/MSA_ISIC_{args.vit}.pth"),
            512, 1, device=device, dataset="isic",
        ),
    ]
    for model in proxies + targets:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    image_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    mask_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = ISIC2016(
        args=argparse.Namespace(img_size=512), data_path=str(ROOT / "data/ISIC"),
        mode="Test", transform=image_transform, transform_msk=mask_transform,
    )
    rows = []
    for index in (int(value) for value in args.indices.split(",") if value.strip()):
        sample = dataset[index]
        image = sample["image"][None].to(device)
        truth = sample["label"][None, 0].to(device)
        proxy_gradients = []
        for model, function in zip(proxies, (samed_logits, samed_logits, medsa_logits)):
            with torch.no_grad():
                clean_logits = function(model, image)
                pseudo = (clean_logits[:, 0] > 0).long() if clean_logits.shape[1] == 1 else clean_logits.argmax(1)
            trial = image.detach().requires_grad_(True)
            proxy_gradients.append(unit(gradient(pseudo_task_loss(function(model, trial), pseudo), trial)))
        cone, weights, proxy_margins = common_ascent_direction(proxy_gradients)
        weighted = (
            .375 * proxy_gradients[0] + .375 * proxy_gradients[1] + .25 * proxy_gradients[2]
        ).sign()
        target_rows = {}
        for name, model, function in zip(("SAMed", "MedSA"), targets, (samed_logits, medsa_logits)):
            trial = image.detach().requires_grad_(True)
            logits = function(model, trial)
            if logits.shape[1] == 1:
                loss = F.binary_cross_entropy_with_logits(logits[:, 0], truth)
            else:
                loss = F.cross_entropy(logits, truth.long())
            target_gradient = unit(gradient(loss, trial))
            residual, mixture_weights = minimum_norm_convex_combination(
                [proxy - target_gradient for proxy in proxy_gradients]
            )
            target_rows[name] = {
                "supervised_clean_loss": float(loss.detach()),
                "cosine_with_cone": float((target_gradient * cone).sum() / cone.norm().clamp_min(1e-30)),
                "cosine_with_weighted": float((target_gradient * weighted).sum() / weighted.norm().clamp_min(1e-30)),
                "convex_hull_residual_l2": float(residual.norm()),
                "projection_weights": [float(value) for value in mixture_weights],
            }
        rows.append({
            "index": index,
            "case": sample["image_meta_dict"]["filename_or_obj"],
            "target_weights_and_gt_post_hoc_only": True,
            "proxy_common_weights": [float(value) for value in weights],
            "proxy_common_margins": [float(value) for value in proxy_margins],
            "targets": target_rows,
        })
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
