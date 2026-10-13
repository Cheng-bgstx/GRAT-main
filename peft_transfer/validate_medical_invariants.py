#!/usr/bin/env python3
"""Post-hoc held-out PEFT alignment test for a frozen source-only projector."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from peft_transfer.evaluate_synapse_slice import ROOT, SAM
from peft_transfer.medical_invariants import (
    capture_features, centered_scaled, load_test_image,
    load_train_image, test_slices, training_slices,
)

sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_medsa_model, load_sam_encoder, load_samed_model


def source_or_target_features(model, name, image, layers):
    encoder = model.sam.image_encoder if name == "SAMed" else model.image_encoder
    preprocessed = model.preprocess(image) if name == "MedSA" else image
    return capture_features(encoder, preprocessed, layers)


def basis_bank(layer_bases, device):
    bank = {name: basis.to(device) for name, basis in layer_bases.items()}
    bank["complement"] = torch.linalg.qr(bank["invariant"], mode="complete").Q[
        :, bank["invariant"].shape[1]:
    ]
    return bank


def fit_map(source, target, basis, ridge):
    x = torch.cat([value @ basis for value in source])
    y = torch.cat([value @ basis for value in target])
    gram = x.T @ x
    amount = max(float(gram.trace() / len(gram)) * ridge, 1e-6)
    matrix = torch.linalg.solve(
        gram + amount * torch.eye(len(gram), device=gram.device, dtype=gram.dtype),
        x.T @ y,
    )
    singular = torch.linalg.svdvals(matrix)
    return matrix, float(singular[-1]), float(singular[0] / singular[-1].clamp_min(1e-12))


def relative_error(source, target, basis, mapping):
    residual = sum(((x @ basis @ mapping) - (y @ basis)).square().sum()
                   for x, y in zip(source, target))
    energy = sum((y @ basis).square().sum() for y in target)
    return float((residual / energy.clamp_min(1e-12)).sqrt())


def collect(model, name, paths, scales, layers, device, test=False):
    output = {layer: [] for layer in layers}
    with torch.no_grad():
        for item in paths:
            image = (load_test_image(item[0], item[1], device) if test
                     else load_train_image(item, device))
            features = source_or_target_features(model, name, image, layers)
            for layer in layers:
                output[layer].append(centered_scaled(
                    features[layer], scales[layer].to(device)
                ))
    return output


def attack_survival(source_model, target_model, target_name, scales, bases,
                    mappings, layers, device, slices):
    records = {}
    with torch.no_grad():
        for index in slices:
            case = "case0001"
            clean = load_test_image(case, index, device)
            source_clean = source_or_target_features(source_model, "base", clean, layers)
            target_clean = source_or_target_features(target_model, target_name, clean, layers)
            for variant in ("b", "cteot"):
                candidate = ROOT / f"testset/strict_base_{variant}_{case}_slice{index}/{case}/{index}.png"
                if not candidate.exists():
                    continue
                array = np.asarray(Image.open(candidate).convert("L"), dtype=np.float32) / 255.0
                attacked = torch.from_numpy(array)[None, None].repeat(1, 3, 1, 1).to(device)
                source_adv = source_or_target_features(source_model, "base", attacked, layers)
                target_adv = source_or_target_features(target_model, target_name, attacked, layers)
                key = f"{case}/slice{index}/{variant}"
                records[key] = {}
                for layer in layers:
                    basis = bases[layer]["invariant"]
                    mapping = mappings[layer]
                    scale = scales[layer].to(device)
                    source_delta = (
                        centered_scaled(source_adv[layer], scale)
                        - centered_scaled(source_clean[layer], scale)
                    ) @ basis
                    target_delta = (
                        centered_scaled(target_adv[layer], scale)
                        - centered_scaled(target_clean[layer], scale)
                    ) @ basis
                    residual = target_delta - source_delta @ mapping
                    m = float(torch.linalg.svdvals(mapping)[-1])
                    predicted = max(0.0, m * float(source_delta.norm()) - float(residual.norm()))
                    records[key][layer] = {
                        "source_displacement_norm": float(source_delta.norm()),
                        "target_displacement_norm": float(target_delta.norm()),
                        "mapping_residual_norm": float(residual.norm()),
                        "triangle_lower_bound": predicted,
                        "target_to_source_norm": float(
                            target_delta.norm() / source_delta.norm().clamp_min(1e-12)
                        ),
                    }
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--mapping-cases", default="case0009,case0010")
    parser.add_argument("--test-cases", default="case0001,case0002")
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.artifact, map_location="cpu")
    if not artifact["source_only"] or artifact["target_checkpoint_used"]:
        raise ValueError("Projector must be fitted without target weights")
    fit_cases = {Path(path).stem.split("_slice")[0] for path in artifact["fit_paths"]}
    map_cases = tuple(args.mapping_cases.split(","))
    test_cases = tuple(args.test_cases.split(","))
    if fit_cases & set(map_cases) or fit_cases & set(test_cases) or set(map_cases) & set(test_cases):
        raise ValueError("Projector, mapping, and test cases must be disjoint")
    layers = tuple(artifact["layers"])
    scales = artifact["scales"]
    bases = {layer: basis_bank(artifact["bases"][layer], device) for layer in layers}
    mapping_paths = [path for case in map_cases for path in training_slices(case)]
    test_paths = [path for case in test_cases for path in test_slices(case)]
    vit = artifact["vit"]
    source = load_sam_encoder(vit, str(SAM[vit]), 512, 8, device)
    source_train = collect(source, "base", mapping_paths, scales, layers, device)
    source_test = collect(source, "base", test_paths, scales, layers, device, test=True)
    result = {}
    for name in ("SAMed", "MedSA"):
        if name == "SAMed":
            model = load_samed_model(
                vit, str(SAM[vit]), str(ROOT / f"checkpoints/best_weights/SAMed_Synapse_{vit}.pth"),
                512, 8, device=device,
            )
        else:
            model = load_medsa_model(
                vit, str(SAM[vit]), str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{vit}.pth"),
                512, 9, device=device,
            )
        target_train = collect(model, name, mapping_paths, scales, layers, device)
        target_test = collect(model, name, test_paths, scales, layers, device, test=True)
        result[name] = {}
        mappings = {}
        for layer in layers:
            result[name][layer] = {}
            for family, basis in bases[layer].items():
                mapping, minimum, condition = fit_map(
                    source_train[layer], target_train[layer], basis, args.ridge
                )
                result[name][layer][family] = {
                    "mapping_train_relative_error": relative_error(
                        source_train[layer], target_train[layer], basis, mapping
                    ),
                    "heldout_relative_error": relative_error(
                        source_test[layer], target_test[layer], basis, mapping
                    ),
                    "sigma_min": minimum,
                    "condition_number": condition,
                }
                if family == "invariant":
                    mappings[layer] = mapping
        result[name]["attack_survival"] = attack_survival(
            source, model, name, scales, bases, mappings, layers, device, (122, 123)
        )
        del model
        torch.cuda.empty_cache()
    payload = {
        "post_hoc_target_analysis": True,
        "frozen_source_artifact": args.artifact,
        "vit": vit, "mapping_paths": [str(path) for path in mapping_paths],
        "test_paths": [f"{case}/slice{index}" for case, index in test_paths],
        "results": result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
