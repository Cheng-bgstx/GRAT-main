#!/usr/bin/env python3
"""Post-hoc encoder transfer diagnostics on fixed cross-case artifacts."""
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

from peft_transfer.core import _unpack_encoder
from peft_transfer.generate import CHECKPOINTS, ROOT
from segment_anything import sam_model_registry

sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_medsa_model, load_samed_model


CASES = {
    "case0001": (123, "testset/umi_phase_mix_sweep_slice123/original.png"),
    "case0002": (115, "testset/crosscase_umi_pasd/case0002_slice115/original.png"),
    "case0003": (161, "testset/crosscase_umi_pasd/case0003_slice161/original.png"),
    "case0004": (118, "testset/crosscase_umi_pasd/case0004_slice118/original.png"),
}


def final_tensor(output):
    return output if torch.is_tensor(output) else _unpack_encoder(output)[0]


def gain_gradient(encoder, clean, adv, preprocess):
    with torch.no_grad():
        reference = final_tensor(encoder(preprocess(clean))).detach()
    value = adv.detach().clone().requires_grad_(True)
    observed = final_tensor(encoder(preprocess(value)))
    scale = reference.abs().mean().clamp_min(1e-8)
    gain = (observed * 0.9999 - reference).abs().mean() / scale
    gradient = torch.autograd.grad(gain, value)[0]
    return float(gain.detach()), gradient.detach().cpu()


def release(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="testset/crosscase_umi_pasd/encoder_transfer.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    inputs = {}
    for case, (index, candidate) in CASES.items():
        with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{case}.npy.h5", "r") as handle:
            clean_np = handle["image"][index].astype(np.float32)
        adv_np = np.asarray(Image.open(ROOT / candidate).convert("L"), dtype=np.float32) / 255.0
        inputs[case] = (
            torch.from_numpy(clean_np)[None, None].repeat(1, 3, 1, 1).to(device),
            torch.from_numpy(adv_np)[None, None].repeat(1, 3, 1, 1).to(device),
        )
    rows = {case: {"slice": CASES[case][0]} for case in CASES}
    gradients = {}

    source, _ = sam_model_registry["vit_b"](
        image_size=512, num_classes=8, checkpoint=str(CHECKPOINTS["vit_b"]),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    source = source.image_encoder.to(device).eval()
    for case, (clean, adv) in inputs.items():
        gain, gradient = gain_gradient(source, clean, adv, lambda x: x[:, :1].repeat(1, 3, 1, 1))
        rows[case]["source_gain"] = gain
        gradients[case] = gradient
    release(source)

    samed = load_samed_model(
        "vit_b", str(CHECKPOINTS["vit_b"]),
        str(ROOT / "checkpoints/best_weights/SAMed_Synapse_vit_b.pth"),
        512, 8, device=device,
    )
    for case, (clean, adv) in inputs.items():
        gain, gradient = gain_gradient(samed.sam.image_encoder, clean, adv, lambda x: x)
        cosine = F.cosine_similarity(gradients[case].flatten(1), gradient.flatten(1), dim=1).mean()
        rows[case]["samed_gain"] = gain
        rows[case]["samed_gradient_cosine"] = float(cosine)
    release(samed)

    medsa = load_medsa_model(
        "vit_b", str(CHECKPOINTS["vit_b"]),
        str(ROOT / "checkpoints/best_weights/MSA_Synapse_vit_b.pth"),
        512, 9, device=device,
    )
    for case, (clean, adv) in inputs.items():
        gain, gradient = gain_gradient(medsa.image_encoder, clean, adv, medsa.preprocess)
        cosine = F.cosine_similarity(gradients[case].flatten(1), gradient.flatten(1), dim=1).mean()
        rows[case]["medsa_gain"] = gain
        rows[case]["medsa_gradient_cosine"] = float(cosine)
    release(medsa)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
