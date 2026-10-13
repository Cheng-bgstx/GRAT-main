#!/usr/bin/env python3
"""Post-hoc task-gradient projection of frozen source-only candidates.

This diagnostic loads target checkpoints and GT after generation. Its output
must never be used to choose an online update or reported as black-box input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from peft_transfer.evaluate_official_slice import load_medsa_function, reset_seed
from peft_transfer.evaluate_synapse_slice import ROOT, SAM

import sys
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_medsa_model, load_samed_model


def present_class_dice_loss(logits, label):
    probabilities = logits.softmax(dim=1)
    classes = [int(value) for value in torch.unique(label) if 0 < value < logits.shape[1]]
    if not classes:
        raise ValueError("Diagnostic slice has no foreground classes")
    terms = []
    for class_id in classes:
        prediction = probabilities[:, class_id]
        target = (label == class_id).float()
        intersection = (prediction * target).sum()
        terms.append(1 - (2 * intersection + 1) /
                     (prediction.sum() + target.sum() + 1))
    return torch.stack(terms).mean()


def spectral_projection(gradient, displacement):
    gf = torch.fft.fft2(gradient.float(), dim=(-2, -1))
    df = torch.fft.fft2(displacement.float(), dim=(-2, -1))
    cross = gf.conj() * df
    weight = gf.abs() * df.abs()
    return {
        "amplitude_cosine": float(F.cosine_similarity(
            gf.abs().flatten()[None], df.abs().flatten()[None]).item()),
        "phase_coherence": float((cross.real.sum() / weight.sum().clamp_min(1e-12)).item()),
        "negative_cross_energy_fraction": float(
            weight[cross.real < 0].sum() / weight.sum().clamp_min(1e-12)),
    }


def analyze(model, encode, decode, clean, candidates, label):
    with torch.no_grad():
        clean_feature = encode(model, clean).detach()
    anchor = clean_feature.clone().requires_grad_(True)
    clean_loss = present_class_dice_loss(decode(model, anchor), label)
    task_gradient = torch.autograd.grad(clean_loss, anchor)[0].detach()
    rows = {}
    for name, image in candidates.items():
        with torch.no_grad():
            changed_feature = encode(model, image).detach()
            changed_loss = present_class_dice_loss(decode(model, changed_feature), label)
        displacement = changed_feature - clean_feature
        linear = (task_gradient * displacement).sum()
        norm_product = task_gradient.norm() * displacement.norm()
        path_slopes = []
        path_cosines = []
        path_amplitude_cosines = []
        path_phase_coherences = []
        for step in range(8):
            fraction = (step + 0.5) / 8
            point = (clean_feature + fraction * displacement).detach().requires_grad_(True)
            point_loss = present_class_dice_loss(decode(model, point), label)
            point_gradient = torch.autograd.grad(point_loss, point)[0].detach()
            slope = (point_gradient * displacement).sum()
            path_slopes.append(float(slope))
            path_cosines.append(float(slope /
                                      (point_gradient.norm() * displacement.norm()).clamp_min(1e-12)))
            polar = spectral_projection(point_gradient, displacement)
            path_amplitude_cosines.append(polar["amplitude_cosine"])
            path_phase_coherences.append(polar["phase_coherence"])
        rows[name] = {
            "clean_dice_surrogate_loss": float(clean_loss.detach()),
            "adv_dice_surrogate_loss": float(changed_loss.detach()),
            "actual_loss_change": float(changed_loss.detach() - clean_loss.detach()),
            "first_order_loss_change": float(linear),
            "gradient_displacement_cosine": float(linear / norm_product.clamp_min(1e-12)),
            "feature_displacement_norm": float(displacement.norm()),
            "path_midpoint_slopes": path_slopes,
            "path_midpoint_cosines": path_cosines,
            "path_amplitude_cosines": path_amplitude_cosines,
            "path_phase_coherences": path_phase_coherences,
            "path_integrated_loss_change_8pt": float(np.mean(path_slopes)),
            "path_positive_fraction": float(np.mean(np.asarray(path_slopes) > 0)),
            **spectral_projection(task_gradient, displacement),
        }
    return rows


def samed_encode(model, image):
    result = model.sam.image_encoder(model.sam.preprocess(image))
    return result[0] if isinstance(result, tuple) else result


def samed_decode(model, embedding):
    sam = model.sam
    sparse, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
    low_res, _ = sam.mask_decoder(
        image_embeddings=embedding, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=True,
    )
    return sam.postprocess_masks(low_res, input_size=(512, 512), original_size=(512, 512))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--prompt-seed", type=int, default=24)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5") as handle:
        clean_array = handle["image"][args.slice].astype(np.float32)
        label_array = handle["label"][args.slice].astype(np.int64)
    clean = torch.from_numpy(clean_array)[None, None].repeat(1, 3, 1, 1).to(device)
    label = torch.from_numpy(label_array)[None].to(device)
    candidates = {
        path: torch.from_numpy(np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255)
        [None, None].repeat(1, 3, 1, 1).to(device)
        for path in args.candidates
    }
    samed = load_samed_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/SAMed_Synapse_{args.vit}.pth"),
        512, 8, device=device,
    )
    result = {"SAMed": analyze(samed, samed_encode, samed_decode,
                               clean, candidates, label)}
    del samed
    torch.cuda.empty_cache()

    medsa = load_medsa_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
        512, 9, device=device,
    )
    official = load_medsa_function()
    reset_seed(args.prompt_seed)
    _, points, _, generated_labels = official.generate_click_prompt(
        clean, label.float()
    )
    prompts = (points.permute(0, 2, 1).float(), generated_labels.int())
    with torch.no_grad():
        sparse, dense = medsa.prompt_encoder(points=prompts, boxes=None, masks=None)

    def medsa_decode(model, embedding):
        low_res, _ = model.mask_decoder(
            image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        return F.interpolate(low_res, size=(512, 512), mode="bilinear", align_corners=False)

    result["MedSA"] = analyze(
        medsa, lambda model, image: model.image_encoder(model.preprocess(image)),
        medsa_decode, clean, candidates, label,
    )
    payload = {"post_hoc_only": True, "loss": "present-class soft Dice",
               "prompt_seed": args.prompt_seed, "case": args.case, "slice": args.slice,
               "results": result}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
