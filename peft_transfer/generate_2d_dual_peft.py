#!/usr/bin/env python3
"""Generate attacks from independent LoRA and Adapter trajectory proxies."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.isic import ISIC2016
from datasets.refuge import REFUGE
from peft_transfer.core import _unpack_encoder
from peft_transfer.evaluate_official_slice import SAM, load_medsa_model, load_samed_model
from peft_transfer.generate_2d_checkpoint_proxy import save_candidate
from peft_transfer.generate_2d_robust_cone import robust_linf_direction
from peft_transfer.task_basin import apply_delta, peft_parameters


DATASETS = {
    "ISIC": (ISIC2016, ROOT / "data/ISIC", 1, "isic", 1),
    "REFUGE": (REFUGE, ROOT / "data/REFUGE-Multirater", 2, "refuge", 2),
}


def unlabeled_image(dataset, dataset_name, index, transform):
    if dataset_name == "ISIC":
        path = Path(dataset.data_path) / dataset.name_list[index]
        case = path.stem
    elif dataset_name == "REFUGE":
        folder = Path(dataset.subfolders[index])
        case = folder.name
        path = folder / f"{case}.jpg"
    else:
        raise ValueError(dataset_name)
    with Image.open(path) as image:
        clean = transform(image.convert("RGB"))
    return case, clean[None]


def samed_feature(model, image, normalize):
    return _unpack_encoder(model.sam.image_encoder(normalize(image)))[0]


def medsa_feature(model, image, input_scale=255.0):
    return model.image_encoder(model.preprocess(image * input_scale))


def medsa_features(model, image, layer_indices, input_scale=255.0):
    if not layer_indices:
        return [medsa_feature(model, image, input_scale)]
    captured = {}
    handles = []
    for index in layer_indices:
        def capture(_module, _inputs, output, index=index):
            captured[index] = output

        handles.append(model.image_encoder.blocks[index].register_forward_hook(capture))
    try:
        final = medsa_feature(model, image, input_scale)
    finally:
        for handle in handles:
            handle.remove()
    return [captured[index] for index in layer_indices] + [final]


def feature_loss(features, references, objective="l1"):
    if objective == "l1":
        losses = [
            F.l1_loss(feature * 0.9999, reference)
            for feature, reference in zip(features, references)
        ]
    elif objective == "roll":
        losses = []
        for feature, reference in zip(features, references):
            if feature.ndim != 4 or feature.shape[1] != reference.shape[1]:
                raise ValueError("roll objective requires BCHW encoder features")
            shifts = (max(1, feature.shape[-2] // 4), max(1, feature.shape[-1] // 4))
            target = torch.roll(reference, shifts=shifts, dims=(-2, -1))
            losses.append(-F.l1_loss(feature, target))
    elif objective == "collision":
        losses = [
            -F.l1_loss(feature, reference)
            for feature, reference in zip(features, references)
        ]
    else:
        raise ValueError(objective)
    return torch.stack(losses).mean()


def spatial_collision_target(image, shift_fraction=0.25):
    """Create a real translated-image target for prompt-independent encoder collision."""
    if not 0.0 < shift_fraction <= 0.5:
        raise ValueError("shift_fraction must be in (0, 0.5]")
    shift = tuple(max(1, int(round(size * shift_fraction))) for size in image.shape[-2:])
    return torch.roll(image, shifts=shift, dims=(-2, -1))


def diverse_pair(clean, adv, step, probability, resize_rate, seed):
    if probability <= 0:
        return clean, adv
    generator = torch.Generator(device="cpu").manual_seed(seed + step)
    if float(torch.rand((), generator=generator)) >= probability:
        return clean, adv
    size = clean.shape[-1]
    lower = max(1, int(round(size * resize_rate)))
    resized = int(torch.randint(lower, size + 1, (), generator=generator))
    top = int(torch.randint(0, size - resized + 1, (), generator=generator))
    left = int(torch.randint(0, size - resized + 1, (), generator=generator))
    padding = (left, size - resized - left, top, size - resized - top)

    def transform(value):
        value = F.interpolate(value, size=(resized, resized), mode="bilinear", align_corners=False)
        return F.pad(value, padding, value=0.0)

    return transform(clean), transform(adv)


def normalized_gradient(loss, image, retain_graph=False):
    gradient = torch.autograd.grad(loss, image, retain_graph=retain_graph)[0]
    return gradient / gradient.abs().mean().clamp_min(1e-12)


def hierarchical_gradient(gradients, group_size=1, seed_lambda=0.0, trajectory_lambda=0.0):
    """Estimate a direction with separate seed and trajectory confidence penalties."""
    if not gradients:
        raise ValueError("at least one gradient is required")
    if group_size < 1 or len(gradients) % group_size:
        raise ValueError(f"{len(gradients)} gradients cannot be divided into groups of {group_size}")
    if len(gradients) == 1:
        return gradients[0], 1.0
    stack = torch.stack(gradients)
    grouped = stack.reshape(len(gradients) // group_size, group_size, *stack.shape[1:])
    seed_means = grouped.mean(1)
    mean = seed_means.mean(0)
    if seed_lambda <= 0 and trajectory_lambda <= 0:
        return mean, 1.0
    seed_error = seed_means.std(0, unbiased=False) / seed_means.shape[0] ** 0.5
    trajectory_error = grouped.std(1, unbiased=False).square().mean(0).sqrt()
    trajectory_error = trajectory_error / group_size ** 0.5
    confidence = (
        mean.abs() - seed_lambda * seed_error - trajectory_lambda * trajectory_error
    ).clamp_min(0.0)
    return mean.sign() * confidence, float((confidence > 0).float().mean())


def polar_gradient_components(gradient, image):
    """Return the relative-amplitude and phase tangent responses of a gradient."""
    image_spectrum = torch.fft.fft2(image.float(), norm="ortho")
    gradient_spectrum = torch.fft.fft2(gradient.float(), norm="ortho")
    cross = gradient_spectrum.conj() * image_spectrum
    return cross.real, -cross.imag, image_spectrum


def gradient_from_polar_components(amplitude, phase, image_spectrum, floor=1e-12):
    """Invert ``polar_gradient_components`` on nonzero image frequencies."""
    cross = torch.complex(amplitude.float(), -phase.float())
    power = image_spectrum.abs().square()
    gradient_spectrum = cross.conj() * image_spectrum / power.clamp_min(floor)
    gradient_spectrum = torch.where(power > floor, gradient_spectrum, torch.zeros_like(gradient_spectrum))
    return torch.fft.ifft2(gradient_spectrum, norm="ortho").real


def hierarchical_polar_gradient(gradients, image, group_size=1, seed_lambda=0.0,
                                trajectory_lambda=0.0):
    components = [polar_gradient_components(gradient, image) for gradient in gradients]
    amplitude, retained_amplitude = hierarchical_gradient(
        [value[0] for value in components], group_size, seed_lambda, trajectory_lambda
    )
    phase, retained_phase = hierarchical_gradient(
        [value[1] for value in components], group_size, seed_lambda, trajectory_lambda
    )
    gradient = gradient_from_polar_components(amplitude, phase, components[0][2])
    return gradient, 0.5 * (retained_amplitude + retained_phase)


def scheduled_proxy_indices(count, group_size, step, schedule):
    if group_size < 1 or count % group_size:
        raise ValueError(f"{count} proxies cannot be divided into groups of {group_size}")
    if schedule == "all":
        return list(range(count))
    if schedule not in ("forward", "reverse"):
        raise ValueError(schedule)
    trajectory_index = min(group_size - 1, step * group_size // 10)
    if schedule == "reverse":
        trajectory_index = group_size - 1 - trajectory_index
    return list(range(trajectory_index, count, group_size))


def antithetic_views(image, sigma, seed, boundary="clip"):
    if sigma <= 0:
        return [image]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, image.shape, generator=generator, dtype=torch.int8)
    radius = sigma
    if boundary == "centered":
        radius = torch.minimum(
            torch.full_like(image, sigma), torch.minimum(image, 1.0 - image)
        ).clamp_min(0.0)
    elif boundary != "clip":
        raise ValueError(boundary)
    noise = (signs.to(device=image.device, dtype=image.dtype) * 2.0 - 1.0) * radius
    return [(image + noise).clamp(0.0, 1.0), (image - noise).clamp(0.0, 1.0)]


def project_grayscale_gradient(gradient):
    """Project an RGB gradient onto the shared-channel CT input subspace."""
    if gradient.ndim != 4:
        raise ValueError(f"expected BCHW gradient, got {tuple(gradient.shape)}")
    return gradient.mean(1, keepdim=True).expand_as(gradient)


def basin_output_logits(model, image, kind, normalize):
    if kind == "lora":
        return model.sam(normalize(image), False, 512)["masks"]
    embedding = medsa_feature(model, image)
    point = image.new_tensor([[[256.0, 256.0]]])
    labels = torch.ones((image.shape[0], 1), device=image.device, dtype=torch.int)
    sparse, dense = model.prompt_encoder(points=(point, labels), boxes=None, masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return F.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False)


def basin_output_drift(logits, clean_logits):
    if logits.shape[1] == 1:
        target = clean_logits.sigmoid().detach()
        return F.binary_cross_entropy_with_logits(logits, target) - F.binary_cross_entropy_with_logits(
            clean_logits.detach(), target
        )
    target = clean_logits.softmax(1).detach()
    return F.kl_div(logits.log_softmax(1), target, reduction="sum") / target.numel()


def safeguarded_basin_gradient(carrier, basin, weight, cosine_floor):
    if weight <= 0 or basin.abs().max() == 0:
        return carrier, 0.0
    candidate = basin / basin.abs().mean().clamp_min(1e-12)
    carrier = carrier / carrier.abs().mean().clamp_min(1e-12)

    def cosine(scale):
        value = carrier + scale * candidate
        return float(F.cosine_similarity(value.flatten(1), carrier.flatten(1), dim=1).min())

    scale = weight
    if cosine(scale) < cosine_floor:
        low, high = 0.0, scale
        for _ in range(20):
            middle = (low + high) / 2
            if cosine(middle) >= cosine_floor:
                low = middle
            else:
                high = middle
        scale = low
    return carrier + scale * candidate, scale


def basin_family_gradients(models, references, clean_logits, image, normalize,
                           temperature, output_weight):
    losses = {"lora": [], "adapter": []}
    gradients = {"lora": [], "adapter": []}
    for (kind, model), reference, reference_logits in zip(models, references, clean_logits):
        feature = (
            samed_feature(model, image, normalize) if kind == "lora"
            else medsa_feature(model, image)
        )
        loss = F.l1_loss(feature * 0.9999, reference) / reference.abs().mean().clamp_min(1e-6)
        if output_weight:
            logits = basin_output_logits(model, image, kind, normalize)
            loss = loss + output_weight * basin_output_drift(logits, reference_logits)
        gradient = torch.autograd.grad(loss, image)[0].detach()
        losses[kind].append(loss.detach())
        gradients[kind].append(gradient)
    family_scores = {}
    family_gradients = {}
    for kind in ("lora", "adapter"):
        if not losses[kind]:
            raise ValueError(f"no {kind} basin models")
        values = torch.stack(losses[kind])
        weights = torch.softmax(-values / temperature, dim=0)
        family_scores[kind] = -temperature * (
            torch.logsumexp(-values / temperature, 0)
            - values.new_tensor(len(values)).log()
        )
        family_gradients[kind] = sum(
            (weight * gradient for weight, gradient in zip(weights, gradients[kind]))
        )
    return losses, gradients, family_scores, family_gradients


def basin_gradient(models, references, clean_logits, image, normalize,
                   temperature, output_weight):
    _losses, _gradients, scores, family_gradients = basin_family_gradients(
        models, references, clean_logits, image, normalize, temperature, output_weight
    )
    score_values = torch.stack([scores["lora"], scores["adapter"]])
    family_weights = torch.softmax(-score_values / temperature, dim=0)
    gradient = sum(weight * family_gradients[kind] for weight, kind in zip(
        family_weights, ("lora", "adapter")
    ))
    return gradient, [float(scores[kind]) for kind in ("lora", "adapter")]


def constrained_pareto_direction(objective, carrier, constraints, beta,
                                 iterations=80, step_size=0.1):
    """Maximize Adapter ascent while retaining explicit carrier/LoRA margins.

    The returned tensor is the actual L-infinity update direction. Candidate
    directions are generated by the low-dimensional Lagrange dual and are
    accepted only when every measured first-order constraint is satisfied.
    """
    if not 0 <= beta <= 1:
        raise ValueError("Pareto beta must be in [0, 1]")

    def unit(value):
        return value / value.abs().mean().clamp_min(1e-12)

    objective = unit(objective)
    carrier = unit(carrier)
    constraints = [carrier, *(unit(value) for value in constraints)]
    reference = carrier.sign()

    def dot(left, right):
        return (left * right).flatten(1).mean(1).min()

    reference_margins = torch.stack([dot(value, reference) for value in constraints])
    bounds = beta * reference_margins.clamp_min(0.0)
    # The carrier constraint has a known maximum of one after normalization.
    bounds[0] = beta
    multipliers = torch.zeros(
        len(constraints), device=objective.device, dtype=objective.dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([multipliers], lr=step_size)
    best_direction = reference
    best_score = dot(objective, reference)
    best_margins = torch.stack([dot(value, reference) for value in constraints])
    tolerance = 5e-4
    found_feasible = bool(torch.all(best_margins + tolerance >= bounds))

    for _iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        combined = objective + sum(
            multiplier * value for multiplier, value in zip(multipliers, constraints)
        )
        # Exact L-infinity support function dual, averaged over pixels so the
        # multipliers and bounds stay independent of image resolution.
        dual = combined.abs().mean() - (multipliers * bounds).sum()
        dual.backward()
        optimizer.step()
        with torch.no_grad():
            multipliers.clamp_(0.0, 50.0)

            combined = objective + sum(
                multiplier * value for multiplier, value in zip(multipliers, constraints)
            )
            direction = combined.sign()
            margins = torch.stack([dot(value, direction) for value in constraints])
            if bool(torch.all(margins + tolerance >= bounds)):
                score = dot(objective, direction)
                if not found_feasible or score > best_score:
                    best_direction, best_score, best_margins = direction, score, margins
                found_feasible = True

    requested_bounds = bounds.clone()
    fallback_to_reference = not found_feasible
    if fallback_to_reference:
        bounds = torch.minimum(bounds, reference_margins)
        best_direction = reference
        best_score = dot(objective, reference)
        best_margins = reference_margins

    diagnostics = {
        "objective_margin": float(best_score),
        "reference_margins": [float(value) for value in reference_margins],
        "constraint_margins": [float(value) for value in best_margins],
        "constraint_bounds": [float(value) for value in bounds],
        "requested_constraint_bounds": [float(value) for value in requested_bounds],
        "fallback_to_reference": fallback_to_reference,
        "feasible": bool(torch.all(best_margins + tolerance >= bounds)),
        "multipliers": [float(value) for value in multipliers.detach()],
    }
    return best_direction, diagnostics


def constrained_pareto_multiobjective_direction(
        objectives, carrier, constraints, beta, iterations=80, step_size=0.1):
    """Maximize the worst proxy-family gain under the Pareto constraints."""
    if len(objectives) < 2:
        raise ValueError("multiobjective Pareto needs at least two objectives")
    if not 0 <= beta <= 1:
        raise ValueError("Pareto beta must be in [0, 1]")

    def unit(value):
        return value / value.abs().mean().clamp_min(1e-12)

    objectives = [unit(value) for value in objectives]
    carrier = unit(carrier)
    constraints = [carrier, *(unit(value) for value in constraints)]
    reference = carrier.sign()

    def dot(left, right):
        return (left * right).flatten(1).mean(1).min()

    def objective_score(direction):
        return torch.stack([dot(value, direction) for value in objectives]).min()

    reference_margins = torch.stack([dot(value, reference) for value in constraints])
    bounds = beta * reference_margins.clamp_min(0.0)
    bounds[0] = beta
    objective_logits = torch.zeros(
        len(objectives), device=carrier.device, dtype=carrier.dtype,
        requires_grad=True,
    )
    multipliers = torch.zeros(
        len(constraints), device=carrier.device, dtype=carrier.dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([objective_logits, multipliers], lr=step_size)
    best_direction = reference
    best_score = objective_score(reference)
    best_margins = reference_margins
    tolerance = 5e-4
    found_feasible = bool(torch.all(best_margins + tolerance >= bounds))

    for _iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        objective_weights = torch.softmax(objective_logits, dim=0)
        combined = sum(
            weight * value for weight, value in zip(objective_weights, objectives)
        ) + sum(
            multiplier * value for multiplier, value in zip(multipliers, constraints)
        )
        dual = combined.abs().mean() - (multipliers * bounds).sum()
        dual.backward()
        optimizer.step()
        with torch.no_grad():
            multipliers.clamp_(0.0, 50.0)
            objective_weights = torch.softmax(objective_logits, dim=0)
            combined = sum(
                weight * value for weight, value in zip(objective_weights, objectives)
            ) + sum(
                multiplier * value for multiplier, value in zip(multipliers, constraints)
            )
            direction = combined.sign()
            margins = torch.stack([dot(value, direction) for value in constraints])
            if bool(torch.all(margins + tolerance >= bounds)):
                score = objective_score(direction)
                if not found_feasible or score > best_score:
                    best_direction, best_score, best_margins = direction, score, margins
                found_feasible = True

    requested_bounds = bounds.clone()
    fallback_to_reference = not found_feasible
    if fallback_to_reference:
        bounds = torch.minimum(bounds, reference_margins)
        best_direction = reference
        best_score = objective_score(reference)
        best_margins = reference_margins

    diagnostics = {
        "objective_margin": float(best_score),
        "objective_margins": [float(dot(value, best_direction)) for value in objectives],
        "objective_weights": [
            float(value) for value in torch.softmax(objective_logits.detach(), dim=0)
        ],
        "reference_margins": [float(value) for value in reference_margins],
        "constraint_margins": [float(value) for value in best_margins],
        "constraint_bounds": [float(value) for value in bounds],
        "requested_constraint_bounds": [float(value) for value in requested_bounds],
        "fallback_to_reference": fallback_to_reference,
        "feasible": bool(torch.all(best_margins + tolerance >= bounds)),
        "multipliers": [float(value) for value in multipliers.detach()],
    }
    return best_direction, diagnostics


def load_basin_models(manifests, geometry, kind, vit, device, max_variants=2):
    models = []
    for name in manifests:
        manifest = json.loads((ROOT / name).read_text())
        if manifest["kind"] != kind or manifest["vit"] != vit:
            raise ValueError(f"mismatched basin manifest: {name}")
        checkpoint = (ROOT / manifest["checkpoint"]).resolve()
        if (ROOT / "checkpoints/best_weights").resolve() in checkpoint.parents:
            raise ValueError("victim checkpoint cannot be a basin proxy")
        loader = load_samed_model if kind == "lora" else load_medsa_model
        if kind == "lora":
            anchor = loader(vit, str(SAM[vit]), str(checkpoint), 512, 1, device=device)
        else:
            anchor = loader(vit, str(SAM[vit]), str(checkpoint), 512, 1,
                            device=device, dataset="isic")
        if geometry == "anchor":
            anchor.eval()
            for parameter in anchor.parameters():
                parameter.requires_grad_(False)
            models.append((kind, anchor))
            continue
        for path in manifest["accepted"][geometry][:max_variants]:
            state = torch.load(ROOT / path, map_location="cpu", weights_only=True)
            variant = copy.deepcopy(anchor)
            parameters = peft_parameters(variant, kind)
            baseline = {key: value.detach().clone() for key, value in parameters.items()}
            apply_delta(parameters, baseline, state["delta"])
            variant.eval()
            for parameter in variant.parameters():
                parameter.requires_grad_(False)
            models.append((kind, variant))
        del anchor
    return models


def attack(samed, medsa_models, clean, normalize, mode, samed_weight=0.5,
           medsa_layer_indices=(), initial_perturbation=None,
           medsa_uncertainty_lambda=0.0, input_diversity_prob=0.0,
           input_diversity_rate=0.9, diversity_seed=42,
           medsa_objective="l1", medsa_proxy_group_size=1,
           medsa_trajectory_lambda=0.0, medsa_coordinate="pixel",
           medsa_trajectory_schedule="all", gradient_smoothing_sigma=0.0,
           smoothing_boundary="clip", collision_shift_fraction=0.25,
           grayscale_projection=False, medsa_input_scale=255.0,
           basin_models=(), basin_weight=0.0, basin_temperature=0.1,
           basin_cosine_floor=0.8, basin_output_weight=0.0,
           pareto_beta=None, pareto_joint_lora_objective=False):
    with torch.no_grad():
        samed_reference = samed_feature(samed, clean, normalize).detach()
        medsa_reference_image = (
            spatial_collision_target(clean, collision_shift_fraction)
            if medsa_objective == "collision" else clean
        )
        medsa_references = [
            [value.detach() for value in medsa_features(
                model, medsa_reference_image, medsa_layer_indices, medsa_input_scale
            )]
            for model in medsa_models
        ]
        basin_references = [
            (samed_feature(model, clean, normalize) if kind == "lora"
             else medsa_feature(model, clean)).detach()
            for kind, model in basin_models
        ]
        basin_clean_logits = [
            basin_output_logits(model, clean, kind, normalize).detach()
            if basin_output_weight else None
            for kind, model in basin_models
        ]
    adv = clean.clone()
    if initial_perturbation is not None:
        prior = initial_perturbation.to(device=clean.device, dtype=clean.dtype)
        if prior.shape[-2:] != clean.shape[-2:]:
            prior = F.interpolate(prior, size=clean.shape[-2:], mode="bilinear", align_corners=False)
        if prior.shape[1] == 1 and clean.shape[1] == 3:
            prior = prior.repeat(1, 3, 1, 1)
        adv = (clean + prior).clamp(0.0, 1.0)
        adv = torch.maximum(torch.minimum(adv, clean + 10.0 / 255.0), clean - 10.0 / 255.0)
    momentum = torch.zeros_like(clean)
    trace = []
    for step in range(10):
        adv = adv.detach().requires_grad_(True)
        clean_view, adv_view = diverse_pair(
            clean, adv, step, input_diversity_prob, input_diversity_rate, diversity_seed
        )
        if input_diversity_prob > 0:
            with torch.no_grad():
                step_samed_reference = samed_feature(samed, clean_view, normalize).detach()
                step_medsa_reference_image = (
                    spatial_collision_target(clean_view, collision_shift_fraction)
                    if medsa_objective == "collision" else clean_view
                )
                step_medsa_references = [
                    [value.detach() for value in medsa_features(
                        model, step_medsa_reference_image, medsa_layer_indices,
                        medsa_input_scale,
                    )]
                    for model in medsa_models
                ]
        else:
            step_samed_reference = samed_reference
            step_medsa_references = medsa_references
        samed_loss = F.l1_loss(
            samed_feature(samed, adv_view, normalize) * 0.9999, step_samed_reference
        )
        active_indices = scheduled_proxy_indices(
            len(medsa_models), medsa_proxy_group_size, step, medsa_trajectory_schedule
        )
        effective_group_size = medsa_proxy_group_size
        if medsa_trajectory_schedule != "all":
            effective_group_size = 1
        neighborhood = antithetic_views(
            adv_view, gradient_smoothing_sigma, diversity_seed + 10000 + step,
            boundary=smoothing_boundary,
        )
        medsa_losses = [
            torch.stack([
                feature_loss(
                    medsa_features(
                        medsa_models[index], view, medsa_layer_indices,
                        medsa_input_scale,
                    ),
                    step_medsa_references[index],
                    objective=medsa_objective,
                )
                for view in neighborhood
            ]).mean()
            for index in active_indices
        ]
        medsa_loss = medsa_losses[0] if len(medsa_losses) == 1 else torch.stack(medsa_losses).mean()
        samed_gradient = normalized_gradient(samed_loss, adv, retain_graph=True)
        medsa_gradients = [
            normalized_gradient(loss, adv, retain_graph=index + 1 < len(medsa_losses))
            for index, loss in enumerate(medsa_losses)
        ]
        combine = hierarchical_polar_gradient if medsa_coordinate == "polar" else hierarchical_gradient
        combine_kwargs = {
            "group_size": effective_group_size,
            "seed_lambda": medsa_uncertainty_lambda,
            "trajectory_lambda": (
                medsa_trajectory_lambda if medsa_trajectory_schedule == "all" else 0.0
            ),
        }
        if medsa_coordinate == "polar":
            medsa_gradient, retained_fraction = combine(medsa_gradients, adv_view, **combine_kwargs)
        else:
            medsa_gradient, retained_fraction = combine(medsa_gradients, **combine_kwargs)
        medsa_gradient = medsa_gradient / medsa_gradient.abs().mean().clamp_min(1e-12)
        if mode == "mean":
            gradient = 0.5 * (samed_gradient + medsa_gradient)
            weight = 0.5
            dual_value = float(gradient.abs().mean())
        elif mode == "weighted":
            weight = float(samed_weight)
            gradient = weight * samed_gradient + (1.0 - weight) * medsa_gradient
            dual_value = float(gradient.abs().mean())
        elif mode == "maxmin":
            gradient, diagnostics = robust_linf_direction(samed_gradient, medsa_gradient)
            weight = diagnostics["dual_weight_base"]
            dual_value = diagnostics["dual_value"]
        else:
            raise ValueError(mode)
        if grayscale_projection:
            gradient = project_grayscale_gradient(gradient)
        basin_scores = None
        basin_scale = 0.0
        pareto_diagnostics = None
        direct_direction = None
        if basin_models and pareto_beta is not None:
            _family_losses, family_model_gradients, family_scores, family_gradients = (
                basin_family_gradients(
                    basin_models, basin_references, basin_clean_logits, adv,
                    normalize, basin_temperature, basin_output_weight,
                )
            )
            normalized_carrier = gradient / gradient.abs().mean().clamp_min(1e-12)
            carrier_momentum = momentum + normalized_carrier
            pareto_constraints = [samed_gradient, *family_model_gradients["lora"]]
            if pareto_joint_lora_objective:
                direct_direction, pareto_diagnostics = (
                    constrained_pareto_multiobjective_direction(
                        [family_gradients["adapter"], family_gradients["lora"]],
                        carrier_momentum, pareto_constraints, pareto_beta,
                    )
                )
            else:
                direct_direction, pareto_diagnostics = constrained_pareto_direction(
                    family_gradients["adapter"], carrier_momentum,
                    pareto_constraints, pareto_beta,
                )
            basin_scores = [float(family_scores[kind]) for kind in ("lora", "adapter")]
            momentum = carrier_momentum.detach()
        elif basin_models and basin_weight > 0:
            robust_gradient, basin_scores = basin_gradient(
                basin_models, basin_references, basin_clean_logits, adv,
                normalize, basin_temperature, basin_output_weight,
            )
            gradient, basin_scale = safeguarded_basin_gradient(
                gradient, robust_gradient, basin_weight, basin_cosine_floor,
            )
        if direct_direction is None:
            gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
            momentum = momentum + gradient
            direct_direction = momentum.sign()
        adv = adv.detach() + (2.0 / 255.0) * direct_direction
        adv = torch.maximum(torch.minimum(adv, clean + 10.0 / 255.0), clean - 10.0 / 255.0)
        adv = adv.clamp(0.0, 1.0)
        trace.append({
            "step": step,
            "samed_loss": float(samed_loss.detach()),
            "medsa_loss": float(medsa_loss.detach()),
            "medsa_proxy_losses": [float(value.detach()) for value in medsa_losses],
            "medsa_active_proxy_indices": active_indices,
            "medsa_confidence_retained_fraction": retained_fraction,
            "gradient_cosine": float(F.cosine_similarity(
                samed_gradient.flatten(1), medsa_gradient.flatten(1), dim=1
            ).mean()),
            "dual_weight_samed": weight,
            "dual_value": dual_value,
            "basin_family_scores": basin_scores,
            "basin_scale": basin_scale,
            "pareto": pareto_diagnostics,
        })
    return adv, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--samed-proxy", required=True)
    parser.add_argument("--medsa-proxy", default=None)
    parser.add_argument("--medsa-proxies", nargs="+", default=None)
    parser.add_argument("--modes", nargs="*", choices=("mean", "maxmin"),
                        default=("mean", "maxmin"))
    parser.add_argument("--samed-weights", type=float, nargs="+", default=None)
    parser.add_argument("--indices", default="0,1,2,3")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--medsa-layers", default="",
                        help="comma-separated zero-based Adapter block indices; final is always included")
    parser.add_argument("--candidate-suffix", default="")
    parser.add_argument("--initial-perturbation", default=None)
    parser.add_argument("--medsa-uncertainty-lambda", type=float, default=0.0)
    parser.add_argument(
        "--medsa-proxy-group-size", type=int, default=1,
        help="number of chronological checkpoints per seed; proxies must be ordered seed-major",
    )
    parser.add_argument(
        "--medsa-trajectory-lambda", type=float, default=0.0,
        help="coordinate confidence penalty for within-seed checkpoint drift",
    )
    parser.add_argument("--input-diversity-prob", type=float, default=0.0)
    parser.add_argument("--input-diversity-rate", type=float, default=0.9)
    parser.add_argument("--diversity-seed", type=int, default=42)
    parser.add_argument("--medsa-objective", choices=("l1", "roll", "collision"), default="l1")
    parser.add_argument(
        "--collision-shift-fraction", type=float, default=0.25,
        help="relative diagonal translation used by the encoder-collision objective",
    )
    parser.add_argument("--medsa-coordinate", choices=("pixel", "polar"), default="pixel")
    parser.add_argument(
        "--medsa-trajectory-schedule", choices=("all", "forward", "reverse"), default="all",
        help="map attack iterations to chronological proxy checkpoints within each seed group",
    )
    parser.add_argument(
        "--gradient-smoothing-sigma", type=float, default=0.0,
        help="antithetic input-neighborhood radius in 0--255 units",
    )
    parser.add_argument(
        "--smoothing-boundary", choices=("clip", "centered"), default="clip",
        help="clip projected pairs or shrink their radius to remain exactly centered",
    )
    parser.add_argument("--grayscale-projection", action="store_true")
    parser.add_argument("--basin-lora-manifests", nargs="+", default=())
    parser.add_argument("--basin-adapter-manifests", nargs="+", default=())
    parser.add_argument("--basin-geometry", choices=("fisher", "isotropic", "anchor"), default="fisher")
    parser.add_argument("--basin-weight", type=float, default=0.0)
    parser.add_argument("--basin-temperature", type=float, default=0.1)
    parser.add_argument("--basin-cosine-floor", type=float, default=0.8)
    parser.add_argument("--basin-output-weight", type=float, default=0.0)
    parser.add_argument(
        "--pareto-beta", type=float, default=None,
        help="retain this fraction of the carrier/LoRA local ascent margins",
    )
    parser.add_argument(
        "--pareto-joint-lora-objective", action="store_true",
        help="maximize the worst aggregate LoRA/Adapter ascent margin",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    uses_medical_family = args.basin_weight or args.pareto_beta is not None
    if uses_medical_family and args.dataset != "ISIC":
        parser.error("task-basin calibration is currently implemented for ISIC only")
    if uses_medical_family and (
        not args.basin_lora_manifests or not args.basin_adapter_manifests
    ):
        parser.error("both LoRA and Adapter basin manifests are required")
    if args.basin_temperature <= 0 or not 0 <= args.basin_cosine_floor <= 1:
        parser.error("basin temperature must be positive and cosine floor in [0, 1]")
    if args.pareto_beta is not None and not 0 <= args.pareto_beta <= 1:
        parser.error("pareto beta must be in [0, 1]")
    if args.pareto_beta is not None and args.basin_weight:
        parser.error("choose either fixed basin weight or Pareto constraints")
    if args.pareto_joint_lora_objective and args.pareto_beta is None:
        parser.error("joint LoRA objective requires --pareto-beta")
    device = torch.device(args.device)
    dataset_cls, data_path, classes, medsa_dataset, medsa_masks = DATASETS[args.dataset]
    image_tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    mask_tf = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    dataset = dataset_cls(
        args=Namespace(img_size=512), data_path=str(data_path), mode="Test",
        transform=image_tf, transform_msk=mask_tf,
    )
    samed = load_samed_model(
        args.vit, str(SAM[args.vit]), str(ROOT / args.samed_proxy),
        512, classes, device=device,
    )
    medsa_proxy_paths = args.medsa_proxies or ([args.medsa_proxy] if args.medsa_proxy else [])
    if not medsa_proxy_paths:
        raise ValueError("--medsa-proxy or --medsa-proxies is required")
    victim_root = (ROOT / "checkpoints/best_weights").resolve()
    if any(victim_root in (ROOT / path).resolve().parents for path in
           [args.samed_proxy, *medsa_proxy_paths]):
        raise ValueError("victim checkpoint cannot be an attack proxy")
    if args.medsa_proxy_group_size < 1 or len(medsa_proxy_paths) % args.medsa_proxy_group_size:
        raise ValueError(
            f"{len(medsa_proxy_paths)} proxies cannot be divided into groups "
            f"of {args.medsa_proxy_group_size}"
        )
    medsa_models = [
        load_medsa_model(
            args.vit, str(SAM[args.vit]), str(ROOT / checkpoint),
            512, medsa_masks, device=device, dataset=medsa_dataset,
        )
        for checkpoint in medsa_proxy_paths
    ]
    for parameter in list(samed.parameters()) + [
        parameter for model in medsa_models for parameter in model.parameters()
    ]:
        parameter.requires_grad_(False)
    basin_models = []
    if args.basin_weight or args.pareto_beta is not None:
        basin_models.extend(load_basin_models(
            args.basin_lora_manifests, args.basin_geometry, "lora", args.vit, device
        ))
        basin_models.extend(load_basin_models(
            args.basin_adapter_manifests, args.basin_geometry, "adapter", args.vit, device
        ))
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    output = ROOT / args.out
    metadata_path = output / "generation.json"
    records = json.loads(metadata_path.read_text()) if args.resume and metadata_path.exists() else []
    existing = {(row["index"], row["candidate"]) for row in records}
    indices = (
        list(range(len(dataset))) if args.all_cases
        else [int(value) for value in args.indices.split(",") if value.strip()]
    )
    if not indices or min(indices) < 0 or max(indices) >= len(dataset):
        raise IndexError(f"indices outside dataset length {len(dataset)}")
    configurations = [(mode, 0.5, mode.upper()) for mode in args.modes]
    configurations.extend(
        ("weighted", weight, f"W{int(round(weight * 100)):03d}")
        for weight in (args.samed_weights or [])
    )
    if not configurations:
        raise ValueError("at least one mode or SAMed weight is required")
    medsa_layer_indices = tuple(
        int(value) for value in args.medsa_layers.split(",") if value.strip()
    )
    depths = {len(model.image_encoder.blocks) for model in medsa_models}
    if len(depths) != 1:
        raise ValueError(f"inconsistent MedSA encoder depths: {depths}")
    depth = depths.pop()
    if medsa_layer_indices and (
        min(medsa_layer_indices) < 0 or max(medsa_layer_indices) >= depth
    ):
        raise IndexError(f"MedSA layers {medsa_layer_indices} outside encoder depth {depth}")
    initial_perturbation = (
        torch.load(ROOT / args.initial_perturbation, map_location="cpu")
        if args.initial_perturbation else None
    )
    if initial_perturbation is not None and not torch.is_tensor(initial_perturbation):
        raise TypeError("initial perturbation checkpoint must contain a Tensor")
    for index in indices:
        case, clean = unlabeled_image(dataset, args.dataset, index, image_tf)
        clean = clean.float().to(device)
        for mode, samed_weight, candidate in configurations:
            candidate += args.candidate_suffix
            npy_path = output / candidate / f"{case}.npy"
            png_path = output / candidate / f"{case}.png"
            if args.resume and npy_path.exists() and png_path.exists():
                print(json.dumps({
                    "index": index, "case": case, "candidate": candidate,
                    "status": "skipped_existing",
                }), flush=True)
                continue
            adv, trace = attack(
                samed, medsa_models, clean, normalize, mode, samed_weight,
                medsa_layer_indices=medsa_layer_indices,
                initial_perturbation=initial_perturbation,
                medsa_uncertainty_lambda=args.medsa_uncertainty_lambda,
                input_diversity_prob=args.input_diversity_prob,
                input_diversity_rate=args.input_diversity_rate,
                diversity_seed=args.diversity_seed + index * 100,
                medsa_objective=args.medsa_objective,
                medsa_proxy_group_size=args.medsa_proxy_group_size,
                medsa_trajectory_lambda=args.medsa_trajectory_lambda,
                medsa_coordinate=args.medsa_coordinate,
                medsa_trajectory_schedule=args.medsa_trajectory_schedule,
                gradient_smoothing_sigma=args.gradient_smoothing_sigma / 255.0,
                smoothing_boundary=args.smoothing_boundary,
                collision_shift_fraction=args.collision_shift_fraction,
                grayscale_projection=args.grayscale_projection,
                basin_models=basin_models,
                basin_weight=args.basin_weight,
                basin_temperature=args.basin_temperature,
                basin_cosine_floor=args.basin_cosine_floor,
                basin_output_weight=args.basin_output_weight,
                pareto_beta=args.pareto_beta,
                pareto_joint_lora_objective=args.pareto_joint_lora_objective,
            )
            record = {
                "index": index, "case": case, "candidate": candidate,
                "target_checkpoints_used": False,
                "uses_test_ground_truth": False,
                "samed_proxy": args.samed_proxy,
                "medsa_proxy": medsa_proxy_paths[0],
                "medsa_proxies": medsa_proxy_paths,
                "samed_weight": samed_weight,
                "medsa_layers": medsa_layer_indices,
                "initial_perturbation": args.initial_perturbation,
                "medsa_uncertainty_lambda": args.medsa_uncertainty_lambda,
                "medsa_proxy_group_size": args.medsa_proxy_group_size,
                "medsa_trajectory_lambda": args.medsa_trajectory_lambda,
                "input_diversity_prob": args.input_diversity_prob,
                "input_diversity_rate": args.input_diversity_rate,
                "medsa_objective": args.medsa_objective,
                "medsa_coordinate": args.medsa_coordinate,
                "medsa_trajectory_schedule": args.medsa_trajectory_schedule,
                "gradient_smoothing_sigma_255": args.gradient_smoothing_sigma,
                "smoothing_boundary": args.smoothing_boundary,
                "grayscale_projection": args.grayscale_projection,
                "collision_shift_fraction": args.collision_shift_fraction,
                "basin_lora_manifests": args.basin_lora_manifests,
                "basin_adapter_manifests": args.basin_adapter_manifests,
                "basin_geometry": args.basin_geometry if uses_medical_family else None,
                "basin_weight": args.basin_weight,
                "basin_temperature": args.basin_temperature,
                "basin_cosine_floor": args.basin_cosine_floor,
                "basin_output_weight": args.basin_output_weight,
                "pareto_beta": args.pareto_beta,
                "pareto_joint_lora_objective": args.pareto_joint_lora_objective,
                "linf_255": save_candidate(output / candidate, case, adv, clean),
                "trace": trace,
            }
            if (index, candidate) in existing:
                records = [
                    row for row in records
                    if (row["index"], row["candidate"]) != (index, candidate)
                ]
            records.append(record)
            existing.add((index, candidate))
            print(json.dumps(record, indent=2), flush=True)
            output.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
