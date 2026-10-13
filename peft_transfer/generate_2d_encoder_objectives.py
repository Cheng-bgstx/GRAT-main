#!/usr/bin/env python3
"""Compare prompt-independent encoder objectives under one attack protocol."""
from __future__ import annotations

import argparse
import json
from argparse import Namespace

import torch
import torch.nn.functional as F
from torchvision import transforms

from datasets.isic import ISIC2016
from datasets.refuge import REFUGE
from peft_transfer.core import _asymmetric_step, _unpack_encoder
from peft_transfer.evaluate_official_slice import ROOT, SAM, load_samed_model
from peft_transfer.generate_2d_checkpoint_proxy import save_candidate


DATASETS = {
    "ISIC": (ISIC2016, ROOT / "data/ISIC", 1),
    "REFUGE": (REFUGE, ROOT / "data/REFUGE-Multirater", 2),
}


def final_feature(encoder, image):
    return _unpack_encoder(encoder(image))[0]


def centered_token_gram(feature):
    tokens = feature.flatten(2).transpose(1, 2).float()
    tokens = tokens - tokens.mean(dim=-1, keepdim=True)
    tokens = F.normalize(tokens, dim=-1, eps=1e-8)
    return tokens @ tokens.transpose(-2, -1)


def polar_feature_loss(adv, clean):
    adv_spectrum = torch.fft.rfft2(adv.float(), norm="ortho")
    clean_spectrum = torch.fft.rfft2(clean.float(), norm="ortho")
    phase = (1.0 - torch.cos(torch.angle(adv_spectrum) - torch.angle(clean_spectrum))).mean()
    adv_log_amplitude = torch.log1p(adv_spectrum.abs())
    clean_log_amplitude = torch.log1p(clean_spectrum.abs())
    amplitude = F.l1_loss(adv_log_amplitude, clean_log_amplitude)
    return phase + amplitude


def objective_loss(adv, clean, objective):
    if objective == "l1":
        return F.l1_loss(adv * 0.9999, clean)
    if objective == "gram":
        return F.l1_loss(centered_token_gram(adv), centered_token_gram(clean))
    if objective == "polar":
        return polar_feature_loss(adv, clean)
    raise ValueError(objective)


def attack(encoder, clean, normalize, objective, prior, optimizer):
    with torch.no_grad():
        reference = final_feature(encoder, normalize(clean)).detach()
    adv = clean.clone()
    momentum = torch.zeros_like(clean)
    trace = []
    for step in range(10):
        adv = adv.detach().requires_grad_(True)
        feature = final_feature(encoder, normalize(adv))
        loss = objective_loss(feature, reference, objective)
        gradient = torch.autograd.grad(loss, adv)[0]
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        if optimizer == "asym":
            momentum = 0.5 * momentum + gradient
            update = _asymmetric_step(momentum, 2.0 / 255.0)
        elif optimizer == "mifgsm":
            momentum = momentum + gradient
            update = (2.0 / 255.0) * momentum.sign()
        else:
            raise ValueError(optimizer)
        if step == 0 and prior is not None:
            update = 2.5 * update + prior
        adv = adv.detach() + update
        adv = torch.maximum(torch.minimum(adv, clean + 10.0 / 255.0), clean - 10.0 / 255.0)
        adv = adv.clamp(0.0, 1.0)
        trace.append(float(loss.detach()))
    return adv, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--objectives", nargs="+", choices=("l1", "gram", "polar"),
                        default=("l1", "gram", "polar"))
    parser.add_argument("--optimizers", nargs="+", choices=("asym", "mifgsm"),
                        default=("asym",))
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
    model = load_samed_model(
        args.vit, str(SAM[args.vit]), str(ROOT / args.checkpoint),
        512, classes, device=device,
    )
    encoder = model.sam.image_encoder
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    umi = torch.load(ROOT / args.umi_prior, map_location=device) if args.umi_prior else None
    output = ROOT / args.out
    records = []
    for index in [int(value) for value in args.indices.split(",") if value.strip()]:
        sample = dataset[index]
        case = str(sample["image_meta_dict"]["filename_or_obj"])
        clean = sample["image"].float()[None].to(device)
        starts = (("ZERO", None), ("UMI", umi)) if umi is not None else (("ZERO", None),)
        for objective in args.objectives:
            for start_name, prior in starts:
                for optimizer in args.optimizers:
                    adv, trace = attack(encoder, clean, normalize, objective, prior, optimizer)
                    candidate = f"{objective.upper()}_{start_name}_{optimizer.upper()}"
                    record = {
                        "index": index,
                        "case": case,
                        "candidate": candidate,
                        "checkpoint": args.checkpoint,
                        "objective": objective,
                        "initialization": start_name,
                        "optimizer": optimizer,
                        "linf_255": save_candidate(output / candidate, case, adv, clean),
                        "trace": trace,
                    }
                    records.append(record)
                    print(json.dumps(record, indent=2), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "generation.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
