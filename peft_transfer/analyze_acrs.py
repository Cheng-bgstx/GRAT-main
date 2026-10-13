#!/usr/bin/env python3
"""Post-hoc mechanism gate for Adaptation-Coupled Response Stability (ACRS).

This script may load victim checkpoints and official labels/results only after
candidate generation. Its outputs are diagnostics and must not be consumed by
an online black-box attack.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]

from datasets.isic import ISIC2016
from evaluate import load_medsa_model, load_sam_encoder, load_samed_model
from peft_transfer.acrs import (
    BANDS, aggregate_acrs_observables, aggregate_source_observables,
    feature_bchw, leave_one_group_out_ridge, pairwise_structure, pearson,
    response_metrics,
)
from peft_transfer.evaluate_official_2d_gate import load_candidate
from peft_transfer.evaluate_official_slice import SAM
from peft_transfer.medsa_surrogate import load_medsa_base_surrogate


def parse_target(spec: str):
    name, value = spec.split("=", 1)
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise ValueError(
            "target must be NAME=KIND:CHECKPOINT:OFFICIAL_JSON:RESULT_KEY"
        )
    kind, checkpoint, official, result_key = parts
    if kind not in {"lora", "adapter"}:
        raise ValueError(f"unsupported target kind {kind!r}")
    if result_key not in {"SAMed", "MedSA_per_case_paired"}:
        raise ValueError(f"unsupported official result key {result_key!r}")
    return name, {
        "kind": kind,
        "checkpoint": checkpoint,
        "official": official,
        "result_key": result_key,
    }


def parse_candidate(spec: str):
    name, folder = spec.split("=", 1)
    return name, ROOT / folder


def capture_features(encoder, image, layer_indices):
    captured = {}
    handles = []
    for index in layer_indices:
        def hook(_module, _inputs, output, index=index):
            value = output[0] if isinstance(output, tuple) else output
            captured[str(index)] = feature_bchw(value.detach(), "bhwc")

        handles.append(encoder.blocks[index].register_forward_hook(hook))
    try:
        with torch.no_grad():
            final = encoder(image)
            final = final[0] if isinstance(final, tuple) else final
            captured["neck"] = feature_bchw(final.detach(), "bchw")
    finally:
        for handle in handles:
            handle.remove()
    return captured


def subtract(second, first):
    return {name: second[name] - first[name] for name in first}


def load_official_outcomes(target, candidates):
    payload = json.loads((ROOT / target["official"]).read_text())
    result = payload[target["result_key"]]
    rows = {}
    if target["result_key"] == "SAMed":
        clean = {row["case"]: row["dice"] for row in result["clean"]["per_case"]}
        for candidate in candidates:
            attacked = {
                row["case"]: row["dice"]
                for row in result[candidate]["per_case"]
            }
            for case in clean:
                rows[(case, candidate)] = {
                    "clean_dice": float(clean[case]),
                    "adv_dice": float(attacked[case]),
                    "dice_drop": float(clean[case] - attacked[case]),
                }
    else:
        clean = {row["case"]: row["dice"] for row in result["clean"]}
        for candidate in candidates:
            attacked = {row["case"]: row["dice"] for row in result[candidate]}
            for case in clean:
                rows[(case, candidate)] = {
                    "clean_dice": float(clean[case]),
                    "adv_dice": float(attacked[case]),
                    "dice_drop": float(clean[case] - attacked[case]),
                }
    return rows


def target_model(spec, vit, device):
    checkpoint = str(ROOT / spec["checkpoint"])
    if spec["kind"] == "lora":
        model = load_samed_model(
            vit, str(SAM[vit]), checkpoint, 512, 1, device=device
        )
        return model, model.sam.image_encoder
    model = load_medsa_model(
        vit, str(SAM[vit]), checkpoint, 512, 1,
        device=device, dataset="isic",
    )
    return model, model.image_encoder


def mean_rows(rows, keys):
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--indices", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--layers", default="0,5,11")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    candidates = dict(parse_candidate(spec) for spec in args.candidates)
    targets = dict(parse_target(spec) for spec in args.targets)
    layer_indices = [int(value) for value in args.layers.split(",") if value.strip()]
    if not layer_indices:
        raise ValueError("at least one transformer layer is required")

    image_tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    dataset = ISIC2016(
        args=Namespace(img_size=512), data_path=str(ROOT / "data/ISIC"),
        mode="Test", transform=image_tf,
        transform_msk=transforms.ToTensor(),
    )
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    source = load_sam_encoder(args.vit, str(SAM[args.vit]), 512, 1, device)
    medsa_base = load_medsa_base_surrogate(
        args.vit, str(SAM[args.vit]), 512, 1, device, dataset="isic"
    )
    loaded_targets = {
        name: (*target_model(spec, args.vit, device), spec)
        for name, spec in targets.items()
    }
    for model in [source, medsa_base, *[value[0] for value in loaded_targets.values()]]:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    official = {
        name: load_official_outcomes(spec, candidates)
        for name, spec in targets.items()
    }
    layer_rows = []
    observation_rows = []
    structure_rows = []

    for index in indices:
        sample = dataset[index]
        case = str(sample["image_meta_dict"]["filename_or_obj"])
        clean = sample["image"].float()[None].to(device)
        images = {
            name: torch.from_numpy(load_candidate(folder, case, ".npy"))[None].to(device)
            for name, folder in candidates.items()
        }
        source_clean = capture_features(source.image_encoder, normalize(clean), layer_indices)
        adapter_base_clean = capture_features(
            medsa_base.image_encoder, medsa_base.preprocess(clean * 255.0), layer_indices
        )
        target_clean = {}
        for name, (model, encoder, spec) in loaded_targets.items():
            prepared = (
                normalize(clean) if spec["kind"] == "lora"
                else model.preprocess(clean * 255.0)
            )
            target_clean[name] = capture_features(encoder, prepared, layer_indices)

        for candidate_name, image in images.items():
            source_adv = capture_features(source.image_encoder, normalize(image), layer_indices)
            source_response = subtract(source_adv, source_clean)
            adapter_base_adv = capture_features(
                medsa_base.image_encoder, medsa_base.preprocess(image * 255.0), layer_indices
            )
            adapter_base_response = subtract(adapter_base_adv, adapter_base_clean)
            source_metric_layers = []
            for layer in source_response:
                source_metric_layers.append(response_metrics(
                    source_response[layer], source_response[layer], source_clean[layer]
                ))
            source_observables = aggregate_source_observables(source_metric_layers)
            coupled_vectors = defaultdict(dict)

            for target_name, (model, encoder, spec) in loaded_targets.items():
                prepared = (
                    normalize(image) if spec["kind"] == "lora"
                    else model.preprocess(image * 255.0)
                )
                target_adv = capture_features(encoder, prepared, layer_indices)
                target_response = subtract(target_adv, target_clean[target_name])
                matched_response = (
                    source_response if spec["kind"] == "lora" else adapter_base_response
                )
                matched_clean = (
                    source_clean if spec["kind"] == "lora" else adapter_base_clean
                )
                target_layer_metrics = []
                for layer in matched_response:
                    metrics = response_metrics(
                        matched_response[layer], target_response[layer], matched_clean[layer]
                    )
                    target_layer_metrics.append(metrics)
                    coupled = target_response[layer] - matched_response[layer]
                    coupled_vectors[layer][target_name] = coupled.detach().cpu()
                    layer_rows.append({
                        "index": index,
                        "case": case,
                        "candidate": candidate_name,
                        "target": target_name,
                        "kind": spec["kind"],
                        "layer": layer,
                        **metrics,
                    })
                acrs_observables = aggregate_acrs_observables(target_layer_metrics)
                outcome = official[target_name][(case, candidate_name)]
                observation_rows.append({
                    "index": index,
                    "case": case,
                    "candidate": candidate_name,
                    "target": target_name,
                    "kind": spec["kind"],
                    **source_observables,
                    **acrs_observables,
                    **outcome,
                })
            for layer, vectors in coupled_vectors.items():
                for row in pairwise_structure(vectors):
                    structure_rows.append({
                        "index": index,
                        "case": case,
                        "candidate": candidate_name,
                        "layer": layer,
                        "left_kind": targets[row["left"]]["kind"],
                        "right_kind": targets[row["right"]]["kind"],
                        **row,
                    })

    target_names = list(targets)
    baseline_features = ["source_log_response"]
    source_features = [
        *baseline_features, "source_layer_entropy", "source_spectral_entropy",
        "source_low_fraction", "source_high_fraction",
    ]
    oracle_features = [
        *baseline_features, "acrs_relative_mismatch", "acrs_response_cosine",
        "acrs_low_mismatch", "acrs_high_mismatch",
    ]
    predictive = {
        "baseline_source_strength": leave_one_group_out_ridge(
            observation_rows, baseline_features, target_names
        ),
        "extended_source_observables": leave_one_group_out_ridge(
            observation_rows, source_features, target_names
        ),
        "posthoc_true_acrs": leave_one_group_out_ridge(
            observation_rows, oracle_features, target_names
        ),
    }
    baseline_mae = predictive["baseline_source_strength"]["mae"]
    predictive["source_incremental_mae_fraction"] = float(
        (baseline_mae - predictive["extended_source_observables"]["mae"])
        / max(baseline_mae, 1e-12)
    )
    predictive["true_acrs_incremental_mae_fraction"] = float(
        (baseline_mae - predictive["posthoc_true_acrs"]["mae"])
        / max(baseline_mae, 1e-12)
    )

    correlations = {}
    for name in [
        "source_log_response", "source_low_fraction", "source_high_fraction",
        "acrs_relative_mismatch", "acrs_response_cosine",
        "acrs_low_mismatch", "acrs_mid_mismatch", "acrs_high_mismatch",
    ]:
        correlations[name] = pearson(
            [row[name] for row in observation_rows],
            [row["dice_drop"] for row in observation_rows],
        )

    cross_family = [
        row for row in structure_rows if row["left_kind"] != row["right_kind"]
    ]
    common_structure_mean = float(np.mean([row["cosine"] for row in cross_family]))
    common_structure_positive_fraction = float(np.mean([
        row["cosine"] > 0 for row in cross_family
    ]))
    decision_rules = {
        "cross_family_coupled_error_cosine_mean_gt_0.2": common_structure_mean > 0.2,
        "cross_family_positive_fraction_ge_0.6": common_structure_positive_fraction >= 0.6,
        "true_acrs_improves_grouped_mae_by_10pct": (
            predictive["true_acrs_incremental_mae_fraction"] >= 0.1
        ),
        "source_observables_improve_grouped_mae_by_5pct": (
            predictive["source_incremental_mae_fraction"] >= 0.05
        ),
    }
    payload = {
        "protocol": {
            **vars(args),
            "post_hoc_victim_analysis": True,
            "candidate_tensor": "generation NPY before PNG quantization",
            "lora_reference": "public SAM under source ImageNet normalization",
            "adapter_reference": "public SAM in Medical-SAM-Adapter architecture with zero adapter residuals",
            "fft_axes": "explicit BCHW spatial H/W only",
        },
        "theory": {
            "coupled_error": "C_delta_theta = (E_m(x+delta)-E_m(x)) - (E_0(x+delta)-E_0(x))",
            "claim_limit": "feature-response stability is not a Dice guarantee",
        },
        "summary": {
            "observations": len(observation_rows),
            "layer_observations": len(layer_rows),
            "cross_family_structure_mean_cosine": common_structure_mean,
            "cross_family_structure_positive_fraction": common_structure_positive_fraction,
            "predictive_gate": predictive,
            "pooled_correlations": correlations,
            "decision_rules": decision_rules,
            "mechanism_gate_passed": all(decision_rules.values()),
        },
        "target_means": {
            target: {
                candidate: mean_rows(
                    [row for row in observation_rows
                     if row["target"] == target and row["candidate"] == candidate],
                    [
                        "clean_dice", "adv_dice", "dice_drop",
                        "acrs_relative_mismatch", "acrs_response_cosine",
                        "acrs_low_mismatch", "acrs_mid_mismatch", "acrs_high_mismatch",
                    ],
                )
                for candidate in candidates
            }
            for target in targets
        },
        "observations": observation_rows,
        "layers": layer_rows,
        "coupled_error_structure": structure_rows,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
