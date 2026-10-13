#!/usr/bin/env python3
"""Post-hoc coverage of MedSA task gradient by source-only Jacobian views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

from peft_transfer.analyze_jacobian_tangent_coverage import source_gradients, summarize
from peft_transfer.jacobian_subspace import riesz_pair
from peft_transfer.evaluate_official_slice import (
    ROOT, SAM, load_medsa_function, load_medsa_model, reset_seed,
)

import sys
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_sam_encoder


def span_coverage(target: torch.Tensor, views: list[torch.Tensor]) -> float:
    """Maximum cosine attainable in the linear span of supplied source views."""
    columns = torch.stack([view.float().flatten() for view in views], dim=1)
    columns = F.normalize(columns, dim=0)
    direction = F.normalize(target.float().flatten(), dim=0)
    gram = columns.T @ columns
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    kept = eigenvalues > eigenvalues.max().clamp_min(1e-12) * 1e-6
    if not bool(kept.any()):
        return 0.0
    coefficients = eigenvectors[:, kept].T @ (columns.T @ direction)
    return float((coefficients / eigenvalues[kept].sqrt()).norm().clamp(max=1.0))


def medsa_task_gradient(model, clean: torch.Tensor, point: torch.Tensor,
                        label: torch.Tensor, seed: int) -> torch.Tensor:
    module = load_medsa_function("Synapse", 9)
    reset_seed(seed)
    _, points, _, point_labels = module.generate_click_prompt(clean, label)
    if points.dim() == 3 and points.shape[1] == 2:
        points = points.permute(0, 2, 1)
    prompt = (points.float(), point_labels.int())
    variable = point.detach().clone().requires_grad_(True)
    embedding = model.image_encoder(model.preprocess(variable))
    sparse, dense = model.prompt_encoder(points=prompt, boxes=None, masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
    )
    logits = F.interpolate(logits, label.shape[-2:], mode="bilinear", align_corners=False)
    loss = F.cross_entropy(logits, label.long())
    return torch.autograd.grad(loss, variable)[0].detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-seed", type=int, default=24)
    args = parser.parse_args()
    device = torch.device(args.device)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5") as handle:
        image = torch.from_numpy(handle["image"][args.slice].astype("float32"))[None, None].to(device)
        label = torch.from_numpy(handle["label"][args.slice].astype("int64"))[None].to(device)
    from PIL import Image
    import numpy as np
    candidate = np.asarray(Image.open(ROOT / args.candidate).convert("L"), dtype=np.float32) / 255.0
    point = torch.from_numpy(candidate)[None, None].to(device)
    source_model = load_sam_encoder("vit_b", str(SAM["vit_b"]), 512, 8, device)
    source, families, indices, _probes, _loss = source_gradients(
        source_model.image_encoder, image.repeat(1, 3, 1, 1),
        point.repeat(1, 3, 1, 1), gain=0.0, seed=1701 + args.slice,
    )
    del source_model
    torch.cuda.empty_cache()
    victim = load_medsa_model(
        "vit_b", str(SAM["vit_b"]),
        str(ROOT / "checkpoints/best_weights/MSA_Synapse_vit_b.pth"),
        512, 9, device=device,
    )
    for parameter in victim.parameters():
        parameter.requires_grad_(False)
    target = medsa_task_gradient(victim, image.repeat(1, 3, 1, 1),
                                 point.repeat(1, 3, 1, 1), label, args.prompt_seed)
    source_gray = source.sum(dim=1, keepdim=True)
    family_gray = {name: [value.sum(dim=1, keepdim=True) for value in views]
                   for name, views in families.items()}
    target_gray = target.sum(dim=1, keepdim=True).cpu()
    result = summarize(source_gray, target_gray, family_gray, ridge=1e-4)
    result["linear_span_ceiling"] = {
        name: span_coverage(target_gray, [source_gray, *views])
        for name, views in family_gray.items()
    }
    views = family_gray["attn_mlp"]
    quadratures = [component for view in [source_gray, *views]
                   for component in riesz_pair(view)]
    result["linear_span_ceiling"]["attn_mlp_riesz"] = span_coverage(
        target_gray, [source_gray, *views, *quadratures]
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "diagnostic_only": True,
        "victim_weights_and_labels_used": True,
        "case": args.case,
        "slice": args.slice,
        "candidate": args.candidate,
        "source_objective": "encoder feature disruption with fixed random probes",
        "victim_objective": "cross entropy with official fixed GT-derived prompt",
        "source_layers": indices,
        "coverage": result,
    }, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
