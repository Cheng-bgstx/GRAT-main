#!/usr/bin/env python3
"""Run the fixed case0001/slice123 attack sanity chain.

Victim gradients are diagnostic only. They are computed after source-only
examples have been generated and never feed back into an attack update.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]

from evaluate import load_medsa_model, load_samed_model
from peft_transfer.core import _unpack_encoder, ablation_config, attack_image
from peft_transfer.evaluate_synapse_slice import msa_predict, samed_predict, score
from segment_anything import sam_model_registry


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--vit", default="vit_b", choices=("vit_b", "vit_l", "vit_h"))
    parser.add_argument("--variants", default="A,B,C,D,E,F,UMI-GRAT,UMI-SYNAPSE")
    parser.add_argument("--out", default="testset/sanity_chain_case0001_slice123")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--step-size", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def final_tensor(output):
    if torch.is_tensor(output):
        return output
    return _unpack_encoder(output)[0]


def feature_gradient(encoder, clean, adv, preprocess=lambda value: value):
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    with torch.no_grad():
        reference = final_tensor(encoder(preprocess(clean))).detach()
    value = adv.detach().clone().requires_grad_(True)
    observed = final_tensor(encoder(preprocess(value)))
    loss = F.l1_loss(observed * 0.9999, reference)
    gradient = torch.autograd.grad(loss, value)[0]
    return float(loss.detach()), gradient.detach().cpu()


def cosine(left, right):
    return float(F.cosine_similarity(left.flatten(1), right.flatten(1), dim=1).mean())


def quantize(adv, clean, epsilon):
    adv_np = adv[0, 0].detach().cpu().clamp(0, 1).numpy()
    clean_np = clean[0, 0].detach().cpu().clamp(0, 1).numpy()
    lower = np.ceil(np.maximum(0.0, clean_np - epsilon) * 255.0).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_np + epsilon) * 255.0).astype(np.int16)
    data = np.clip(np.rint(adv_np * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
    tensor = torch.from_numpy(data.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1)
    return data, tensor


def load_prior(name, device):
    if name == "UMI-SYNAPSE":
        path = ROOT / "common_perb/Synapse.pth"
    else:
        path = ROOT / "common_perb/common_perturbation_meta_0.025sp_bound2_it7batchit41upsize_ls0.0025mulmd0_10000.pth"
    return torch.load(path, map_location=device), str(path.relative_to(ROOT))


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = arguments()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sam_ckpt = ROOT / f"checkpoints/sam_{args.vit}_01ec64.pth"
    if args.vit == "vit_l":
        sam_ckpt = ROOT / "checkpoints/sam_vit_l_0b3195.pth"
    elif args.vit == "vit_h":
        sam_ckpt = ROOT / "checkpoints/sam_vit_h_4b8939.pth"
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean_np = handle["image"][args.slice].astype(np.float32)
        label = handle["label"][args.slice].astype(np.int64)
    clean = torch.from_numpy(clean_np).to(device)[None, None].repeat(1, 3, 1, 1)
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)

    source, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=8, checkpoint=str(sam_ckpt),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    encoder = source.image_encoder.to(device).eval()
    variants = [item.strip().upper() for item in args.variants.split(",") if item.strip()]
    records, images, gradients = {}, {}, {}
    for name in variants:
        base_name = name.removesuffix("-EDGE")
        attack_name = base_name.removeprefix("UMI-")
        if attack_name.startswith("K") and attack_name[1:].isdigit():
            chain_name = "K"
        elif attack_name == "GRAT":
            chain_name = "R"
        elif attack_name == "SYNAPSE":
            chain_name = "B"
        else:
            chain_name = attack_name
        config_overrides = dict(
            epsilon=args.epsilon / 255.0, step_size=args.step_size / 255.0,
            steps=args.steps, seed=args.seed,
        )
        if name.endswith("-EDGE"):
            config_overrides["use_edge"] = True
        if chain_name == "K" and attack_name[1:].isdigit():
            config_overrides["skip_gamma"] = int(attack_name[1:]) / 100.0
        cfg = ablation_config(chain_name, **config_overrides)
        prior, prior_path = (None, None)
        if name.startswith("UMI-"):
            prior, prior_path = load_prior(name, device)
        sink = {}
        adv, trace = attack_image(encoder, clean, cfg, initial_delta=prior, gradient_sink=sink)
        png, quantized_cpu = quantize(adv, clean, cfg.epsilon)
        path = output / f"{name}.png"
        Image.fromarray(png).save(path)
        quantized = quantized_cpu.to(device)
        gain, _ = feature_gradient(encoder, clean, quantized)
        images[name] = quantized_cpu
        gradients[name] = sink
        records[name] = {
            "prior": prior_path,
            "image": str(path.relative_to(ROOT)),
            "linf_255": float((quantized - clean).abs().max() * 255.0),
            "source_gain": gain,
            "trace_final": {key: values[-1] for key, values in trace.items()},
        }
        print(name, json.dumps(records[name]))
    del encoder, source
    release()

    samed = load_samed_model(
        args.vit, str(sam_ckpt), str(ROOT / f"checkpoints/best_weights/SAMed_Synapse_{args.vit}.pth"),
        512, 8, device=device,
    )
    clean_samed = score(samed_predict(samed, clean_np, device), label)
    for name in variants:
        adv = images[name].to(device)
        adv_np = adv[0, 0].cpu().numpy()
        gain, victim_gradient = feature_gradient(samed.sam.image_encoder, clean, adv)
        records[name]["SAMed"] = {
            "clean_dice": clean_samed,
            "adv_dice": score(samed_predict(samed, adv_np, device), label),
            "victim_gain": gain,
            "cos_source_victim": cosine(gradients[name]["source"], victim_gradient),
            "cos_final_victim": cosine(gradients[name]["final"], victim_gradient),
        }
    del samed
    release()

    medsa = load_medsa_model(
        args.vit, str(sam_ckpt), str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
        512, 9, device=device,
    )
    clean_medsa = score(msa_predict(medsa, clean_np, label, device), label)
    for name in variants:
        adv = images[name].to(device)
        adv_np = adv[0, 0].cpu().numpy()
        gain, victim_gradient = feature_gradient(medsa.image_encoder, clean, adv, medsa.preprocess)
        records[name]["MedSA"] = {
            "clean_dice": clean_medsa,
            "adv_dice": score(msa_predict(medsa, adv_np, label, device), label),
            "victim_gain": gain,
            "cos_source_victim": cosine(gradients[name]["source"], victim_gradient),
            "cos_final_victim": cosine(gradients[name]["final"], victim_gradient),
        }
    del medsa
    release()

    result = {
        "protocol": {"case": args.case, "slice": args.slice, "vit": args.vit,
                     "steps": args.steps, "epsilon_255": args.epsilon, "step_size_255": args.step_size,
                     "victim_gradients_are_diagnostic_only": True},
        "results": records,
    }
    with open(output / "results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
