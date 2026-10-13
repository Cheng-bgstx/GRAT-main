#!/usr/bin/env python3
"""Fit acquisition-stable source-SAM channel subspaces without target access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from peft_transfer.evaluate_synapse_slice import ROOT, SAM
from peft_transfer.medical_invariants import (
    capture_features, centered_scaled, environment_view,
    load_train_image, training_slices,
)

import sys
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_sam_encoder


FIT_VIEWS = ("contrast_08", "contrast_12", "gamma_09", "bias_field")
HELDOUT_VIEWS = ("gamma_11", "blur")


def spectral_subspace(shared, nuisance, rank, ridge):
    dimension = shared.shape[0]
    eye = torch.eye(dimension, dtype=shared.dtype, device=shared.device)
    regularizer = max(float(nuisance.trace() / dimension) * ridge, 1e-5)
    cholesky = torch.linalg.cholesky(nuisance + regularizer * eye)
    left = torch.linalg.solve_triangular(cholesky, shared, upper=False)
    whitened = torch.linalg.solve_triangular(cholesky, left.T, upper=False).T
    whitened = (whitened + whitened.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
    generalized = torch.linalg.solve_triangular(
        cholesky.T, eigenvectors[:, -rank:], upper=True
    )
    basis = torch.linalg.qr(generalized).Q
    return basis, eigenvalues[-rank:], regularizer


def energy_stats(clean, changed, basis):
    displacement = changed - clean
    signal = (clean @ basis).square().sum() / (clean.shape[0] * basis.shape[1])
    nuisance = (displacement @ basis).square().sum() / (clean.shape[0] * basis.shape[1])
    return float(signal), float(nuisance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--fit-cases", default="case0005,case0006,case0007")
    parser.add_argument("--holdout-cases", default="case0009,case0010")
    parser.add_argument("--layers", default="0,5,11,neck")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    layers = tuple(args.layers.split(","))
    fit_cases = tuple(args.fit_cases.split(","))
    holdout_cases = tuple(args.holdout_cases.split(","))
    if set(fit_cases) & set(holdout_cases):
        raise ValueError("Fit and holdout case IDs must be disjoint")
    model = load_sam_encoder(args.vit, str(SAM[args.vit]), 512, 8, device)
    encoder = model.image_encoder
    fit_paths = [path for case in fit_cases for path in training_slices(case)]
    holdout_paths = [path for case in holdout_cases for path in training_slices(case)]

    # Cache clean source tokens. All statistics and projections are source-only.
    clean_cache = {}
    with torch.no_grad():
        for path in fit_paths:
            clean_cache[str(path)] = {
                key: value.double() for key, value in
                capture_features(encoder, load_train_image(path, device), layers).items()
            }
    scales = {}
    for layer in layers:
        centered = [value[layer] - value[layer].mean(0, keepdim=True)
                    for value in clean_cache.values()]
        scales[layer] = torch.cat(centered).square().mean(0).sqrt().clamp_min(1e-4)

    covariance = {}
    for layer in layers:
        dimension = clean_cache[str(fit_paths[0])][layer].shape[1]
        covariance[layer] = {
            key: torch.zeros(dimension, dimension, dtype=torch.float64, device=device)
            for key in ("shared", "nuisance", "signal")
        }
    with torch.no_grad():
        for path in fit_paths:
            image = load_train_image(path, device)
            clean = {layer: centered_scaled(clean_cache[str(path)][layer], scales[layer])
                     for layer in layers}
            for layer in layers:
                covariance[layer]["signal"] += clean[layer].T @ clean[layer] / len(clean[layer])
            for view in FIT_VIEWS:
                changed = capture_features(encoder, environment_view(image, view), layers)
                for layer in layers:
                    x = clean[layer]
                    y = centered_scaled(changed[layer].double(), scales[layer])
                    cross = x.T @ y
                    difference = x - y
                    covariance[layer]["shared"] += (cross + cross.T) / (2 * len(x))
                    covariance[layer]["nuisance"] += difference.T @ difference / len(x)
    bases, eigenvalues, regularizers = {}, {}, {}
    for layer in layers:
        cov = covariance[layer]
        shared = cov["shared"] / (len(fit_paths) * len(FIT_VIEWS))
        nuisance = cov["nuisance"] / (len(fit_paths) * len(FIT_VIEWS))
        signal = cov["signal"] / len(fit_paths)
        basis, values, regularizer = spectral_subspace(shared, nuisance, args.rank, args.ridge)
        pca_values, pca_vectors = torch.linalg.eigh((signal + signal.T) / 2)
        pca = pca_vectors[:, -args.rank:]
        generator = torch.Generator(device=device).manual_seed(1701 + len(layer))
        random = torch.linalg.qr(torch.randn(
            basis.shape[0], args.rank, dtype=torch.float64, device=device,
            generator=generator,
        )).Q
        bases[layer] = {"invariant": basis.float().cpu(),
                        "pca": pca.float().cpu(), "random": random.float().cpu()}
        eigenvalues[layer] = [float(value) for value in values]
        regularizers[layer] = regularizer

    holdout = {layer: {key: [] for key in ("invariant", "pca", "random", "complement")}
               for layer in layers}
    with torch.no_grad():
        for path in holdout_paths:
            image = load_train_image(path, device)
            clean_features = capture_features(encoder, image, layers)
            for view in HELDOUT_VIEWS:
                changed_features = capture_features(encoder, environment_view(image, view), layers)
                for layer in layers:
                    clean = centered_scaled(clean_features[layer].double(), scales[layer])
                    changed = centered_scaled(changed_features[layer].double(), scales[layer])
                    for name in ("invariant", "pca", "random"):
                        basis = bases[layer][name].to(device=device, dtype=torch.float64)
                        signal, nuisance = energy_stats(clean, changed, basis)
                        holdout[layer][name].append((signal, nuisance))
                    invariant = bases[layer]["invariant"].to(device=device, dtype=torch.float64)
                    signal_all = clean.square().sum() / clean.shape[0]
                    nuisance_all = (changed - clean).square().sum() / clean.shape[0]
                    signal_part = (clean @ invariant).square().sum() / clean.shape[0]
                    nuisance_part = ((changed - clean) @ invariant).square().sum() / clean.shape[0]
                    count = clean.shape[1] - invariant.shape[1]
                    holdout[layer]["complement"].append((
                        float((signal_all - signal_part) / count),
                        float((nuisance_all - nuisance_part) / count),
                    ))
    summary = {}
    for layer in layers:
        summary[layer] = {}
        for name, values in holdout[layer].items():
            signal = sum(item[0] for item in values) / len(values)
            nuisance = sum(item[1] for item in values) / len(values)
            summary[layer][name] = {
                "signal_per_dim": signal,
                "nuisance_per_dim": nuisance,
                "nuisance_to_signal": nuisance / max(signal, 1e-12),
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "source_only": True, "target_checkpoint_used": False,
        "vit": args.vit, "layers": layers, "rank": args.rank,
        "fit_paths": [str(path) for path in fit_paths],
        "holdout_paths": [str(path) for path in holdout_paths],
        "fit_views": FIT_VIEWS, "holdout_views": HELDOUT_VIEWS,
        "scales": {key: value.float().cpu() for key, value in scales.items()},
        "bases": bases,
    }
    torch.save(artifact, output.with_suffix(".pt"))
    output.write_text(json.dumps({
        "source_only": True, "vit": args.vit, "rank": args.rank,
        "fit_paths": artifact["fit_paths"], "holdout_paths": artifact["holdout_paths"],
        "fit_views": FIT_VIEWS, "holdout_views": HELDOUT_VIEWS,
        "eigenvalues_top_rank": eigenvalues,
        "regularizers": regularizers, "A_holdout": summary,
        "artifact": str(output.with_suffix(".pt")),
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
