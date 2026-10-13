#!/usr/bin/env python3
"""Prompt-free class-prototype attack from independent Synapse PEFT proxies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from peft_transfer.evaluate_official_slice import (
    SAM,
    load_medsa_function,
    load_medsa_model,
    load_samed_model,
    reset_seed,
)
from peft_transfer.generate_2d_dual_peft import (
    antithetic_views,
    hierarchical_gradient,
    medsa_features,
    project_grayscale_gradient,
    samed_feature,
)
from peft_transfer.generate_synapse_dual_peft import load_grayscale_prior, save_grayscale
from peft_transfer.core import (
    _edge_map,
    _asymmetric_step,
    _unpack_encoder,
    ablation_config,
    phase_amplitude_stochastic_disruption,
)
from segment_anything import sam_model_registry


def feature_bchw(feature: torch.Tensor):
    """Canonicalize ViT block BHWC and neck BCHW outputs."""
    if feature.ndim != 4:
        raise ValueError("encoder feature must be four-dimensional")
    if feature.shape[-1] > max(feature.shape[1], feature.shape[2]):
        return feature.permute(0, 3, 1, 2).contiguous()
    return feature


def class_prototypes(feature: torch.Tensor, label: torch.Tensor):
    """Return normalized per-class prototypes and encoder-grid class indices."""
    if feature.ndim != 4 or label.ndim != 4:
        raise ValueError("feature and label must be BCHW")
    feature = feature_bchw(feature)
    grid = F.interpolate(label.float(), feature.shape[-2:], mode="nearest").long()[:, 0]
    tokens = F.normalize(feature, dim=1).permute(0, 2, 3, 1)
    classes = torch.unique(grid)
    prototypes = []
    kept = []
    for class_id in classes:
        mask = grid == class_id
        if int(mask.sum()) < 2:
            continue
        prototypes.append(F.normalize(tokens[mask].mean(0), dim=0))
        kept.append(class_id)
    if not prototypes:
        raise ValueError("prototype loss requires at least one represented class")
    prototypes = torch.stack(prototypes)
    class_ids = torch.stack(kept)
    targets = torch.full_like(grid, -1)
    for index, class_id in enumerate(class_ids):
        targets[grid == class_id] = index
    return prototypes.detach(), targets.detach(), class_ids.detach()


def class_prototype_bank(feature_function, samples):
    """Pool normalized encoder tokens into a full-class training prototype bank."""
    return class_prototype_bank_from_features([
        (feature_function(image), label) for image, label in samples
    ])


def class_prototype_bank_from_features(samples):
    """Build a class bank from already computed feature/label pairs."""
    sums = {}
    counts = {}
    current_grid = None
    for sample_index, (feature, label) in enumerate(samples):
        feature = feature_bchw(feature)
        grid = F.interpolate(label.float(), feature.shape[-2:], mode="nearest").long()[:, 0]
        if sample_index == 0:
            current_grid = grid
        tokens = F.normalize(feature, dim=1).permute(0, 2, 3, 1)
        for class_id in torch.unique(grid):
            key = int(class_id)
            selected = tokens[grid == class_id]
            if selected.numel() == 0:
                continue
            value = selected.sum(0)
            sums[key] = sums.get(key, torch.zeros_like(value)) + value
            counts[key] = counts.get(key, 0) + len(selected)
    class_values = sorted(sums)
    if len(class_values) < 2 or current_grid is None:
        raise ValueError("prototype bank requires at least two represented classes")
    prototypes = torch.stack([
        F.normalize(sums[class_id] / counts[class_id], dim=0)
        for class_id in class_values
    ]).detach()
    class_ids = torch.tensor(class_values, device=prototypes.device)
    targets = torch.full_like(current_grid, -1)
    for index, class_id in enumerate(class_values):
        targets[current_grid == class_id] = index
    return prototypes, targets.detach(), class_ids


def diagonal_lda_bank_from_features(samples, shrinkage: float = 0.05):
    """Estimate a channel-rescaling-invariant diagonal LDA class bank."""
    if not 0 <= shrinkage <= 1:
        raise ValueError("LDA shrinkage must be in [0, 1]")
    sums = {}
    squared_sums = {}
    counts = {}
    current_grid = None
    for sample_index, (feature, label) in enumerate(samples):
        feature = feature_bchw(feature).permute(0, 2, 3, 1)
        grid = F.interpolate(label.float(), feature.shape[1:3], mode="nearest").long()[:, 0]
        if sample_index == 0:
            current_grid = grid
        for class_id in torch.unique(grid):
            key = int(class_id)
            selected = feature[grid == class_id].float()
            if selected.numel() == 0:
                continue
            sums[key] = sums.get(key, torch.zeros_like(selected[0])) + selected.sum(0)
            squared_sums[key] = (
                squared_sums.get(key, torch.zeros_like(selected[0]))
                + selected.square().sum(0)
            )
            counts[key] = counts.get(key, 0) + len(selected)
    class_values = sorted(sums)
    if len(class_values) < 2 or current_grid is None:
        raise ValueError("LDA bank requires at least two represented classes")
    means = torch.stack([sums[class_id] / counts[class_id] for class_id in class_values])
    residual_sum = torch.stack([
        squared_sums[class_id] - sums[class_id].square() / counts[class_id]
        for class_id in class_values
    ]).sum(0).clamp_min(0.0)
    degrees = max(1, sum(counts.values()) - len(class_values))
    within_variance = residual_sum / degrees
    total_count = sum(counts.values())
    total_sum = torch.stack(list(sums.values())).sum(0)
    total_squared_sum = torch.stack(list(squared_sums.values())).sum(0)
    total_variance = (
        total_squared_sum - total_sum.square() / total_count
    ).clamp_min(0.0) / max(1, total_count - 1)
    variance = (
        (1.0 - float(shrinkage)) * within_variance
        + float(shrinkage) * total_variance
    )
    relative_floor = torch.finfo(variance.dtype).eps * (
        total_squared_sum / total_count
    ).clamp_min(torch.finfo(variance.dtype).tiny)
    precision = variance.clamp_min(relative_floor).reciprocal().detach()
    class_ids = torch.tensor(class_values, device=means.device)
    targets = torch.full_like(current_grid, -1)
    for index, class_id in enumerate(class_values):
        targets[current_grid == class_id] = index
    return means.detach(), precision, targets.detach(), class_ids


def diagonal_lda_scores(feature: torch.Tensor, means: torch.Tensor,
                        precision: torch.Tensor):
    """Return diagonal-LDA logits, normalized to avoid layer-scale domination."""
    tokens = feature_bchw(feature).permute(0, 2, 3, 1).float()
    weights = means * precision
    bias = -0.5 * means.square().mul(precision).sum(-1)
    logits = tokens @ weights.t() + bias
    return logits / logits.detach().std().clamp_min(1e-6)


def balanced_lda_loss(feature: torch.Tensor, means: torch.Tensor,
                      precision: torch.Tensor, targets: torch.Tensor):
    """Class-balanced LDA cross entropy; ascent erodes the proxy task subspace."""
    logits = diagonal_lda_scores(feature, means, precision)
    losses = []
    for target in torch.unique(targets):
        if int(target) < 0:
            continue
        mask = targets == target
        losses.append(F.cross_entropy(logits[mask], targets[mask]))
    if not losses:
        raise ValueError("no valid LDA targets")
    return torch.stack(losses).mean()


def lda_coverage_margin(feature: torch.Tensor, means: torch.Tensor,
                        precision: torch.Tensor, targets: torch.Tensor,
                        class_ids: torch.Tensor):
    """Induce every absent class in a separate region under diagonal-LDA scores."""
    scores = diagonal_lda_scores(feature, means, precision)
    background = torch.nonzero(class_ids == 0, as_tuple=False).flatten()
    if background.numel() != 1:
        raise ValueError("LDA coverage objective requires class id 0")
    background_index = int(background[0])
    present = set(int(value) for value in torch.unique(targets) if int(value) >= 0)
    absent = [index for index in range(len(class_ids)) if index not in present]
    if not absent:
        return balanced_lda_loss(feature, means, precision, targets)
    height, width = targets.shape[-2:]
    y = torch.arange(height, device=targets.device).view(1, height, 1)
    x = torch.arange(width, device=targets.device).view(1, 1, width)
    regions = ((2 * y // max(1, height)) * 2 + (2 * x // max(1, width)))
    regions = regions.remainder(len(absent)).expand_as(targets)
    absent_tensor = torch.tensor(absent, device=targets.device)
    confusers = absent_tensor[regions]
    confusers = torch.where(targets == background_index, confusers,
                            torch.full_like(confusers, background_index))
    losses = []
    for true_index in torch.unique(targets):
        true_value = int(true_index)
        if true_value < 0:
            continue
        for wrong_index in torch.unique(confusers[targets == true_index]):
            mask = (targets == true_index) & (confusers == wrong_index)
            losses.append(
                scores[..., int(wrong_index)][mask].mean()
                - scores[..., true_value][mask].mean()
            )
    return torch.stack(losses).mean()


def balanced_prototype_loss(feature: torch.Tensor, prototypes: torch.Tensor,
                            targets: torch.Tensor, temperature: float = 0.07):
    """Class-balanced token CE; ascent collapses the clean semantic margin."""
    feature = feature_bchw(feature)
    tokens = F.normalize(feature, dim=1).permute(0, 2, 3, 1)
    logits = tokens @ prototypes.t() / temperature
    losses = []
    for target in torch.unique(targets):
        if int(target) < 0:
            continue
        mask = targets == target
        losses.append(F.cross_entropy(logits[mask], targets[mask]))
    if not losses:
        raise ValueError("no valid prototype targets")
    return torch.stack(losses).mean()


def nearest_wrong_classes(prototypes: torch.Tensor):
    """Choose a fixed semantic confuser for every clean class prototype."""
    similarity = prototypes @ prototypes.t()
    similarity.fill_diagonal_(-torch.inf)
    return similarity.argmax(dim=1).detach()


def background_swap_classes(prototypes: torch.Tensor, class_ids: torch.Tensor):
    """Map organs to background and background to its nearest organ."""
    background = torch.nonzero(class_ids == 0, as_tuple=False).flatten()
    if background.numel() != 1:
        raise ValueError("background-swap objective requires class id 0")
    background_index = int(background[0])
    confusers = torch.full(
        (len(class_ids),), background_index, dtype=torch.long,
        device=prototypes.device,
    )
    similarity = prototypes[background_index] @ prototypes.t()
    similarity[background_index] = -torch.inf
    confusers[background_index] = similarity.argmax()
    return confusers.detach()


def balanced_prototype_margin(feature: torch.Tensor, prototypes: torch.Tensor,
                              targets: torch.Tensor,
                              confusers: torch.Tensor):
    """Maximize nearest-wrong minus clean-class cosine score per class."""
    feature = feature_bchw(feature)
    tokens = F.normalize(feature, dim=1).permute(0, 2, 3, 1)
    scores = tokens @ prototypes.t()
    losses = []
    for target in torch.unique(targets):
        target_index = int(target)
        if target_index < 0:
            continue
        mask = targets == target
        class_scores = scores[mask]
        losses.append(
            class_scores[:, confusers[target_index]].mean()
            - class_scores[:, target_index].mean()
        )
    if not losses:
        raise ValueError("no valid prototype targets")
    return torch.stack(losses).mean()


def coverage_margin(feature: torch.Tensor, prototypes: torch.Tensor,
                    targets: torch.Tensor, class_ids: torch.Tensor):
    """Create spatially distributed false positives for classes absent in the slice."""
    feature = feature_bchw(feature)
    tokens = F.normalize(feature, dim=1).permute(0, 2, 3, 1)
    scores = tokens @ prototypes.t()
    background = torch.nonzero(class_ids == 0, as_tuple=False).flatten()
    if background.numel() != 1:
        raise ValueError("coverage objective requires class id 0")
    background_index = int(background[0])
    present = set(int(value) for value in torch.unique(targets) if int(value) >= 0)
    absent = [index for index in range(len(class_ids)) if index not in present]
    if not absent:
        return balanced_prototype_loss(feature, prototypes, targets)

    confusers = torch.full_like(targets, background_index)
    width = targets.shape[-1]
    x = torch.arange(width, device=targets.device).view(1, 1, width)
    stripes = torch.div(x * len(absent), width, rounding_mode="floor").clamp_max(len(absent) - 1)
    absent_tensor = torch.tensor(absent, device=targets.device)
    background_confusers = absent_tensor[stripes].expand_as(targets)
    confusers = torch.where(targets == background_index, background_confusers, confusers)

    losses = []
    for true_index in torch.unique(targets):
        true_value = int(true_index)
        if true_value < 0:
            continue
        for wrong_index in torch.unique(confusers[targets == true_index]):
            mask = (targets == true_index) & (confusers == wrong_index)
            losses.append(
                scores[..., int(wrong_index)][mask].mean()
                - scores[..., true_value][mask].mean()
            )
    return torch.stack(losses).mean()


def prototype_attack_loss(feature: torch.Tensor, prototypes: torch.Tensor,
                          targets: torch.Tensor, objective: str,
                          temperature: float, class_ids=None):
    if objective == "ce":
        return balanced_prototype_loss(feature, prototypes, targets, temperature)
    if len(prototypes) < 2:
        raise ValueError(f"{objective} objective requires at least two classes")
    if objective == "margin":
        return balanced_prototype_margin(
            feature, prototypes, targets, nearest_wrong_classes(prototypes)
        )
    if objective == "background":
        if class_ids is None:
            raise ValueError("background objective requires class ids")
        return balanced_prototype_margin(
            feature, prototypes, targets,
            background_swap_classes(prototypes, class_ids),
        )
    if objective == "coverage":
        if class_ids is None:
            raise ValueError("coverage objective requires class ids")
        return coverage_margin(feature, prototypes, targets, class_ids)
    raise ValueError(f"unknown prototype objective: {objective}")


def bank_attack_loss(feature: torch.Tensor, bank, objective: str,
                     temperature: float):
    if objective == "lda":
        means, precision, targets, _classes = bank
        return balanced_lda_loss(feature, means, precision, targets)
    if objective == "lda_coverage":
        means, precision, targets, classes = bank
        return lda_coverage_margin(feature, means, precision, targets, classes)
    prototypes, targets, classes = bank
    return prototype_attack_loss(
        feature, prototypes, targets, objective, temperature, classes
    )


def spatial_absent_class_targets(label: torch.Tensor, num_classes: int = 9):
    """Assign absent foreground classes to fixed spatial regions."""
    spatial_label = label[:, 0] if label.ndim == 4 else label
    present = set(int(value) for value in torch.unique(label))
    absent = [class_id for class_id in range(1, num_classes) if class_id not in present]
    if not absent:
        return None
    height, width = spatial_label.shape[-2:]
    y = torch.arange(height, device=label.device).view(1, height, 1)
    x = torch.arange(width, device=label.device).view(1, 1, width)
    regions = ((2 * y // height) * 2 + (2 * x // width)).remainder(len(absent))
    return torch.tensor(absent, device=label.device)[regions].expand_as(spatial_label)


def prompt_ensemble(model, prompt_module, clean, label, seeds):
    prompts = []
    with torch.no_grad():
        for seed in seeds:
            reset_seed(seed)
            _unused, points, _masks, point_labels = prompt_module.generate_click_prompt(
                clean, label[:, 0]
            )
            if points.dim() == 3 and points.shape[1] == 2:
                points = points.permute(0, 2, 1)
            prompt = (points.float().to(clean.device), point_labels.int().to(clean.device))
            sparse, dense = model.prompt_encoder(points=prompt, boxes=None, masks=None)
            prompts.append((sparse.detach(), dense.detach()))
    return prompts


def prompt_marginalized_coverage_loss(model, image, label, prompts):
    """Expected absent-class coverage over prompts from an independent proxy."""
    embedding = model.image_encoder(model.preprocess(image))
    target = spatial_absent_class_targets(label)
    losses = []
    for sparse, dense in prompts:
        logits, _ = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        logits = F.interpolate(logits, label.shape[-2:], mode="bilinear", align_corners=False)
        losses.append(
            F.cross_entropy(logits, label[:, 0].long())
            if target is None else -F.cross_entropy(logits, target.long())
        )
    return torch.stack(losses).mean()


def spatial_collision_image(image: torch.Tensor, shift_fraction: float):
    if not 0 < shift_fraction <= 0.5:
        raise ValueError("collision shift fraction must be in (0, 0.5]")
    shifts = tuple(max(1, int(round(size * shift_fraction))) for size in image.shape[-2:])
    return torch.roll(image, shifts=shifts, dims=(-2, -1))


def spatial_collision_loss(features, references):
    """Attract encoder features to those of a far translated image."""
    return -torch.stack([
        F.l1_loss(feature, reference) for feature, reference in zip(features, references)
    ]).mean()


def smooth_deformation_image(image: torch.Tensor, strength: float):
    """Apply a smooth, non-periodic displacement in normalized coordinates."""
    if not 0 < strength <= 0.5:
        raise ValueError("deformation strength must be in (0, 0.5]")
    height, width = image.shape[-2:]
    y = torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype)
    x = torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dx = strength * torch.sin(torch.pi * yy) * torch.cos(torch.pi * xx / 2)
    dy = strength * torch.sin(torch.pi * xx) * torch.cos(torch.pi * yy / 2)
    grid = torch.stack((xx + dx, yy + dy), dim=-1)[None].expand(len(image), -1, -1, -1)
    return F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def deformation_fidelity_loss(features, clean_references, deformed_references):
    """Move features toward a deformed manifold point and away from the clean point."""
    losses = []
    for feature, clean, deformed in zip(
        features, clean_references, deformed_references
    ):
        feature = feature_bchw(feature).flatten(1)
        clean = feature_bchw(clean).flatten(1)
        deformed = feature_bchw(deformed).flatten(1)
        losses.append(
            F.cosine_similarity(feature, deformed, dim=1).mean()
            - F.cosine_similarity(feature, clean, dim=1).mean()
        )
    return torch.stack(losses).mean()


def normalized_gradient(loss, image, retain_graph=False, project_grayscale=True):
    gradient = torch.autograd.grad(loss, image, retain_graph=retain_graph)[0]
    if project_grayscale:
        gradient = project_grayscale_gradient(gradient)
    return gradient / gradient.abs().mean().clamp_min(1e-12)


def load_raw_prior(path, device):
    prior = torch.load(ROOT / path, map_location="cpu", weights_only=True)
    if isinstance(prior, dict):
        for key in ("perturbation", "delta", "noise"):
            if key in prior:
                prior = prior[key]
                break
    if not isinstance(prior, torch.Tensor) or prior.ndim != 4 or prior.shape[1] != 3:
        raise ValueError("joint base/PEFT attack requires a BCHW three-channel prior")
    return prior.to(device=device, dtype=torch.float32)


def attack(samed, medsa_models, clean, label, initial, weight, sigma,
           uncertainty_lambda, seed, temperature, objective="margin",
           anchor_radius=None, base_encoder=None, base_weight=0.0,
           momentum_decay=1.0, asymmetric_step=False,
           first_step_multiplier=1.0, reference_samples=(),
           base_only_steps=0, medsa_layer_indices=(),
           references_for_single_class_only=False, reference_max_classes=0,
           medsa_task_objective="encoder", prompt_module=None,
           prompt_seeds=(24, 43, 91), collision_shift_fraction=0.25,
           collision_weight=0.1, deformation_strength=0.12,
           deformation_weight=1.0):
    stochastic_seed = 42 if base_encoder is not None else seed
    torch.manual_seed(stochastic_seed)
    if clean.is_cuda:
        torch.cuda.manual_seed_all(stochastic_seed)
    canonical = (
        (lambda value: value[:, :1].repeat(1, 3, 1, 1))
        if base_encoder is not None else (lambda value: value)
    )
    feature_families = [
        lambda value: [samed_feature(samed, canonical(value), lambda item: item)],
        *[
            lambda value, model=model: medsa_features(
                model, canonical(value), medsa_layer_indices, input_scale=1.0
            )
            for model in medsa_models
        ],
    ]
    prototype_data = []
    clean_feature_references = []
    current_class_count = int(torch.unique(label).numel())
    use_references = bool(reference_samples) and (
        (reference_max_classes > 0 and current_class_count <= reference_max_classes)
        or (reference_max_classes <= 0 and not references_for_single_class_only)
        or (references_for_single_class_only and current_class_count == 1)
    )
    with torch.no_grad():
        for function in feature_families:
            clean_features = function(clean)
            clean_feature_references.append([
                feature.detach() for feature in clean_features
            ])
            if use_references:
                sample_features = [(clean_features, label)] + [
                    (function(image), reference_label)
                    for image, reference_label in reference_samples
                ]
                bank_builder = (
                    diagonal_lda_bank_from_features
                    if objective.startswith("lda") else class_prototype_bank_from_features
                )
                prototype_data.append([
                    bank_builder([
                        (features[index], sample_label)
                        for features, sample_label in sample_features
                    ])
                    for index in range(len(clean_features))
                ])
            else:
                if objective.startswith("lda"):
                    prototype_data.append([
                        diagonal_lda_bank_from_features([(feature, label)])
                        for feature in clean_features
                    ])
                else:
                    prototype_data.append([
                        class_prototypes(feature, label) for feature in clean_features
                    ])
        base_reference = (
            _unpack_encoder(base_encoder(canonical(clean)))
            if base_encoder is not None else None
        )
    prompt_data = None
    if medsa_task_objective == "prompt_marginalized_coverage":
        if prompt_module is None:
            raise ValueError("prompt-marginalized coverage requires a prompt module")
        prompt_data = [
            prompt_ensemble(model, prompt_module, clean, label, prompt_seeds)
            for model in medsa_models
        ]
    collision_references = None
    if medsa_task_objective in ("encoder_spatial_collision", "encoder_ce_collision"):
        translated = spatial_collision_image(clean, collision_shift_fraction)
        with torch.no_grad():
            collision_references = [
                [feature.detach() for feature in medsa_features(
                    model, canonical(translated), medsa_layer_indices, input_scale=1.0
                )]
                for model in medsa_models
            ]
    deformation_references = None
    if medsa_task_objective in ("encoder_deformation", "encoder_ce_deformation"):
        deformed = smooth_deformation_image(clean, deformation_strength)
        with torch.no_grad():
            deformation_references = [
                [feature.detach() for feature in function(deformed)]
                for function in feature_families
            ]
    base_cfg = ablation_config("AG") if base_encoder is not None else None
    edge = _edge_map(clean).detach() if base_encoder is not None else None
    defer_initial = initial is not None and first_step_multiplier != 1.0
    adv = clean.clone()
    if initial is not None and not defer_initial:
        adv = torch.maximum(torch.minimum(clean + initial, clean + 10 / 255), clean - 10 / 255)
        adv = adv.clamp(0, 1)
    if anchor_radius is not None and defer_initial:
        raise ValueError("trust-region anchoring cannot defer the initial perturbation")
    anchor = adv.clone() if anchor_radius is not None else None
    if anchor_radius is not None and initial is None:
        raise ValueError("anchor radius requires an initial perturbation or image")
    momentum = torch.zeros_like(clean)
    trace = []
    for step in range(10):
        adv = adv.detach().requires_grad_(True)
        neighborhood = antithetic_views(
            adv, sigma, seed + 10000 + step, boundary="centered"
        )
        losses = []
        gradients = []
        for family_index, (function, family_data) in enumerate(zip(
            feature_families, prototype_data
        )):
            if objective == "ce" and all(len(data[0]) == 1 for data in family_data):
                # A one-class softmax is constant, so its exact input gradient is zero.
                losses.append(adv.sum() * 0.0)
                gradients.append(torch.zeros_like(adv))
                continue
            if family_index > 0 and medsa_task_objective == "prompt_marginalized_coverage":
                loss = torch.stack([
                    prompt_marginalized_coverage_loss(
                        medsa_models[family_index - 1], view, label,
                        prompt_data[family_index - 1],
                    )
                    for view in neighborhood
                ]).mean()
            elif family_index > 0 and medsa_task_objective == "encoder_spatial_collision":
                loss = torch.stack([
                    spatial_collision_loss(
                        function(view), collision_references[family_index - 1]
                    )
                    for view in neighborhood
                ]).mean()
            elif family_index > 0 and medsa_task_objective == "encoder_ce_collision":
                view_losses = []
                for view in neighborhood:
                    features = function(view)
                    semantic = torch.stack([
                        bank_attack_loss(feature, bank, objective, temperature)
                        for feature, bank in zip(features, family_data)
                    ]).mean()
                    collision = spatial_collision_loss(
                        features, collision_references[family_index - 1]
                    )
                    view_losses.append(
                        semantic / semantic.detach().abs().clamp_min(1e-6)
                        + float(collision_weight) * collision
                        / collision.detach().abs().clamp_min(1e-6)
                    )
                loss = torch.stack(view_losses).mean()
            elif family_index > 0 and medsa_task_objective in (
                "encoder_deformation", "encoder_ce_deformation"
            ):
                view_losses = []
                for view in neighborhood:
                    features = function(view)
                    deformation = deformation_fidelity_loss(
                        features,
                        clean_feature_references[family_index],
                        deformation_references[family_index],
                    )
                    if medsa_task_objective == "encoder_deformation":
                        view_losses.append(deformation)
                    else:
                        semantic = torch.stack([
                            bank_attack_loss(feature, bank, objective, temperature)
                            for feature, bank in zip(features, family_data)
                        ]).mean()
                        view_losses.append(
                            semantic / semantic.detach().abs().clamp_min(1e-6)
                            + float(deformation_weight) * deformation
                            / deformation.detach().abs().clamp_min(1e-6)
                        )
                loss = torch.stack(view_losses).mean()
            else:
                loss = torch.stack([
                    torch.stack([
                        bank_attack_loss(feature, bank, objective, temperature)
                        for feature, bank in zip(function(view), family_data)
                    ]).mean()
                    for view in neighborhood
                ]).mean()
            losses.append(loss)
            gradients.append(normalized_gradient(
                loss, adv, retain_graph=True,
                project_grayscale=base_encoder is None,
            ))
        samed_gradient = gradients[0]
        medsa_gradient, retained = hierarchical_gradient(
            gradients[1:], seed_lambda=uncertainty_lambda
        )
        medsa_gradient = medsa_gradient / medsa_gradient.abs().mean().clamp_min(1e-12)
        peft_gradient = weight * samed_gradient + (1.0 - weight) * medsa_gradient
        peft_gradient = peft_gradient / peft_gradient.abs().mean().clamp_min(1e-12)
        source_cosine = None
        source_loss_value = None
        if base_encoder is not None:
            base_final, base_layers = _unpack_encoder(base_encoder(canonical(adv)))
            clean_final, clean_layers = base_reference
            source_loss = F.l1_loss(base_final * 0.9999, clean_final.detach())
            source_loss = source_loss + phase_amplitude_stochastic_disruption(
                base_layers, clean_layers, base_cfg
            )
            source_gradient = torch.autograd.grad(source_loss, adv)[0]
            source_gradient = source_gradient * (1.0 + base_cfg.edge_gamma * edge)
            source_gradient = source_gradient / source_gradient.abs().mean().clamp_min(1e-12)
            source_cosine = float(F.cosine_similarity(
                source_gradient.flatten(1), peft_gradient.flatten(1), dim=1
            ).mean())
            source_loss_value = float(source_loss.detach())
            effective_base_weight = 1.0 if step < int(base_only_steps) else base_weight
            gradient = (
                effective_base_weight * source_gradient
                + (1.0 - effective_base_weight) * peft_gradient
            )
        else:
            gradient = peft_gradient
            effective_base_weight = 0.0
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = float(momentum_decay) * momentum + gradient
        update = (
            _asymmetric_step(momentum, 2 / 255)
            if asymmetric_step else (2 / 255) * momentum.sign()
        )
        if step == 0 and defer_initial:
            update = float(first_step_multiplier) * update + initial
        adv = adv.detach() + update
        lower = (clean - 10 / 255).clamp(0, 1)
        upper = (clean + 10 / 255).clamp(0, 1)
        if anchor is not None:
            radius = float(anchor_radius) / 255.0
            lower = torch.maximum(lower, anchor - radius)
            upper = torch.minimum(upper, anchor + radius)
        adv = torch.maximum(torch.minimum(adv, upper), lower)
        trace.append({
            "step": step,
            "samed_loss": float(losses[0].detach()),
            "medsa_losses": [float(value.detach()) for value in losses[1:]],
            "family_gradient_cosine": float(F.cosine_similarity(
                samed_gradient.flatten(1), medsa_gradient.flatten(1), dim=1
            ).mean()),
            "adapter_retained_fraction": retained,
            "source_loss": source_loss_value,
            "source_peft_gradient_cosine": source_cosine,
            "effective_base_weight": effective_base_weight,
        })
    return adv, trace, [
        [layer_data[-1].tolist() for layer_data in family_data]
        for family_data in prototype_data
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slices", type=int, nargs="+", required=True)
    parser.add_argument("--samed-proxy", required=True)
    parser.add_argument("--medsa-proxies", nargs="+", required=True)
    parser.add_argument("--samed-weights", type=float, nargs="+", default=(0, 0.5, 1))
    parser.add_argument("--base-weights", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--initial-perturbation", default="common_perb/Synapse.pth")
    parser.add_argument("--no-initial-perturbation", action="store_true")
    parser.add_argument(
        "--label-source", choices=("ground_truth", "proxy"), default="proxy",
        help="proxy uses only the independent LoRA model's clean prediction",
    )
    parser.add_argument(
        "--allow-ground-truth-diagnostic", action="store_true",
        help="permit test labels only for explicitly marked retrospective diagnostics",
    )
    parser.add_argument(
        "--anchor-image", default=None,
        help="optional grayscale PNG used as the initial and trust-region anchor",
    )
    parser.add_argument("--anchor-radii", type=float, nargs="+", default=(None,))
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--momentum-decay", type=float, default=1.0)
    parser.add_argument("--asymmetric-step", action="store_true")
    parser.add_argument("--first-step-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--base-only-steps", type=int, default=0,
        help="lexicographic continuation: use only the base objective for the first N steps",
    )
    parser.add_argument(
        "--medsa-layer-indices", type=int, nargs="*", default=(),
        help="Adapter block outputs included before the final neck embedding",
    )
    parser.add_argument(
        "--prototype-references", nargs="+", default=(),
        help="training NPZ slices used to complete the semantic class bank",
    )
    parser.add_argument(
        "--references-for-single-class-only", action="store_true",
        help="use the reference bank only when the attacked slice has one class",
    )
    parser.add_argument(
        "--reference-max-classes", type=int, default=0,
        help="use the reference bank only on slices with at most this many classes",
    )
    parser.add_argument("--uncertainty-lambda", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--medsa-task-objective",
        choices=(
            "encoder", "encoder_spatial_collision", "encoder_ce_collision",
            "encoder_deformation", "encoder_ce_deformation",
            "prompt_marginalized_coverage"
        ), default="encoder",
    )
    parser.add_argument("--prompt-seeds", type=int, nargs="+", default=(24, 43, 91))
    parser.add_argument("--collision-shift-fraction", type=float, default=0.25)
    parser.add_argument("--collision-weight", type=float, default=0.1)
    parser.add_argument("--deformation-strength", type=float, default=0.12)
    parser.add_argument("--deformation-weight", type=float, default=1.0)
    parser.add_argument(
        "--objective", choices=(
            "background", "margin", "ce", "coverage", "lda", "lda_coverage"
        ),
        default="background",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0 <= args.base_only_steps <= 10:
        parser.error("--base-only-steps must be between 0 and 10")
    if args.label_source == "ground_truth" and not args.allow_ground_truth_diagnostic:
        parser.error("ground-truth generation requires --allow-ground-truth-diagnostic")
    device = torch.device(args.device)
    samed = load_samed_model(
        args.vit, str(SAM[args.vit]), str(ROOT / args.samed_proxy), 512, 8,
        device=device,
    )
    medsa_models = [load_medsa_model(
        args.vit, str(SAM[args.vit]), str(ROOT / checkpoint), 512, 9,
        device=device, dataset="Synapse",
    ) for checkpoint in args.medsa_proxies]
    for parameter in list(samed.parameters()) + [
        value for model in medsa_models for value in model.parameters()
    ]:
        parameter.requires_grad_(False)
    base_encoder = None
    use_base = any(weight > 0 for weight in args.base_weights) or args.base_only_steps > 0
    if use_base:
        base_model, _ = sam_model_registry[args.vit](
            image_size=512, num_classes=8, checkpoint=str(SAM[args.vit]),
            pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
        )
        base_encoder = base_model.image_encoder.to(device).eval()
        for parameter in base_encoder.parameters():
            parameter.requires_grad_(False)
    initial = None if args.no_initial_perturbation else (
        load_raw_prior(args.initial_perturbation, device)
        if use_base
        else load_grayscale_prior(args.initial_perturbation, device)
    )
    reference_samples = []
    for reference_path in args.prototype_references:
        with np.load(ROOT / reference_path) as reference:
            reference_image = reference["image"].astype(np.float32)
            reference_label = reference["label"].astype(np.int64)
        reference_samples.append((
            torch.from_numpy(reference_image)[None, None].repeat(1, 3, 1, 1).to(device),
            torch.from_numpy(reference_label)[None, None].to(device),
        ))
    prompt_module = (
        load_medsa_function("Synapse", 9)
        if args.medsa_task_objective == "prompt_marginalized_coverage" else None
    )
    rows = []
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        for slice_index in args.slices:
            image = handle["image"][slice_index].astype(np.float32)
            clean = torch.from_numpy(image)[None, None].repeat(1, 3, 1, 1).to(device)
            if args.label_source == "ground_truth":
                label = handle["label"][slice_index].astype(np.int64)
                label_tensor = torch.from_numpy(label)[None, None].to(device)
            else:
                with torch.no_grad():
                    logits = samed.sam(clean, multimask_output=True, image_size=512)["masks"]
                    label_tensor = logits.argmax(1, keepdim=True).detach()
            class_count = int(torch.unique(label_tensor).numel())
            slice_initial = initial
            if args.anchor_image is not None:
                anchor_path = ROOT / args.anchor_image.format(slice=slice_index)
                anchor_np = np.asarray(Image.open(anchor_path).convert("L"), dtype=np.float32) / 255.0
                anchor_tensor = torch.from_numpy(anchor_np)[None, None].repeat(1, 3, 1, 1).to(device)
                slice_initial = anchor_tensor - clean
            for anchor_radius in args.anchor_radii:
              for base_weight in args.base_weights:
               for weight in args.samed_weights:
                adv, trace, classes = attack(
                    samed, medsa_models, clean, label_tensor, slice_initial, weight,
                    args.sigma / 255.0, args.uncertainty_lambda,
                    args.seed + 100 * slice_index, args.temperature, args.objective,
                    anchor_radius,
                    base_encoder, base_weight,
                    args.momentum_decay, args.asymmetric_step,
                    args.first_step_multiplier,
                    reference_samples,
                    args.base_only_steps,
                    args.medsa_layer_indices,
                    args.references_for_single_class_only,
                    args.reference_max_classes,
                    args.medsa_task_objective,
                    prompt_module,
                    tuple(args.prompt_seeds),
                    args.collision_shift_fraction,
                    args.collision_weight,
                    args.deformation_strength,
                    args.deformation_weight,
                )
                anchor_name = "" if anchor_radius is None else f"_A{int(round(anchor_radius)):02d}"
                base_name = "" if base_weight == 0 else f"_B{int(round(100 * base_weight)):03d}"
                candidate = (
                    f"PROTO_{args.objective.upper()}_W{int(round(100 * weight)):03d}"
                    f"{base_name}{anchor_name}"
                )
                if args.label_source == "proxy":
                    candidate += "_PSEUDO"
                path = ROOT / args.out / candidate / f"{slice_index}.png"
                row = {
                    "case": args.case, "slice": slice_index, "candidate": candidate,
                    "target_checkpoints_used": False,
                    "uses_prompt_or_decoder": (
                        args.medsa_task_objective == "prompt_marginalized_coverage"
                    ),
                    "uses_ground_truth_spatial_labels": args.label_source == "ground_truth",
                    "label_source": args.label_source,
                    "samed_proxy": args.samed_proxy,
                    "medsa_proxies": args.medsa_proxies,
                    "classes": classes, "temperature": args.temperature,
                    "objective": args.objective,
                    "anchor_image": args.anchor_image,
                    "anchor_radius_255": anchor_radius,
                    "base_weight": base_weight,
                    "momentum_decay": args.momentum_decay,
                    "asymmetric_step": args.asymmetric_step,
                    "first_step_multiplier": args.first_step_multiplier,
                    "base_only_steps": args.base_only_steps,
                    "medsa_layer_indices": list(args.medsa_layer_indices),
                    "prototype_references": list(args.prototype_references),
                    "references_used": bool(args.prototype_references) and (
                        (args.reference_max_classes > 0 and class_count <= args.reference_max_classes)
                        or (args.reference_max_classes <= 0 and not args.references_for_single_class_only)
                        or (args.references_for_single_class_only and class_count == 1)
                    ),
                    "reference_max_classes": args.reference_max_classes,
                    "medsa_task_objective": args.medsa_task_objective,
                    "prompt_seeds": list(args.prompt_seeds),
                    "collision_shift_fraction": args.collision_shift_fraction,
                    "collision_weight": args.collision_weight,
                    "deformation_strength": args.deformation_strength,
                    "deformation_weight": args.deformation_weight,
                    "linf_255": save_grayscale(path, adv, clean), "trace": trace,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    output = ROOT / args.out / "generation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
