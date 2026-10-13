#!/usr/bin/env python3
"""Post-hoc encoder displacement diagnostics; never used by attack generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from peft_transfer.evaluate_synapse_slice import ROOT, SAM

import sys
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_medsa_model, load_sam_encoder, load_samed_model
from peft_transfer.medsa_surrogate import load_medsa_base_surrogate


def features(encoder, preprocess, image):
    values = []
    handles = [block.register_forward_hook(
        lambda _module, _inputs, output: values.append(output.detach())
    ) for block in encoder.blocks]
    try:
        with torch.no_grad():
            final = encoder(preprocess(image))
            final = (final if torch.is_tensor(final) else final[0]).detach()
    finally:
        for handle in handles:
            handle.remove()
    return [*values, final]


def displacement(model, encoder, preprocess, clean, candidate):
    first = features(encoder, preprocess, clean)
    second = features(encoder, preprocess, candidate)
    return [(b - a).cpu().float() for a, b in zip(first, second)]


def compare(source, victim):
    if source.shape != victim.shape:
        raise ValueError(f"Feature shapes differ: {source.shape} and {victim.shape}")
    if source.ndim != 4:
        raise ValueError(f"Expected 4D features, got {source.shape}")
    # ViT blocks are BHWC; the neck is BCHW.
    if source.shape[-1] > source.shape[1]:
        source, victim = source.permute(0, 3, 1, 2), victim.permute(0, 3, 1, 2)
    s = source.flatten()
    v = victim.flatten()
    cosine = F.cosine_similarity(s[None], v[None]).item()
    norm_ratio = (v.norm() / s.norm().clamp_min(1e-12)).item()
    sf = torch.fft.fft2(source, dim=(-2, -1))
    vf = torch.fft.fft2(victim, dim=(-2, -1))
    cross = sf.conj() * vf
    weights = sf.abs() * vf.abs()
    coherence = (cross.real.sum() / weights.sum().clamp_min(1e-12)).item()
    phase_consistency = (cross.real / weights.clamp_min(1e-12))
    reliable = weights > weights.flatten().quantile(0.75)
    strong_phase = phase_consistency[reliable].mean().item() if reliable.any() else 0.0
    amplitude_cosine = F.cosine_similarity(sf.abs().flatten()[None], vf.abs().flatten()[None]).item()
    return {
        "spatial_cosine": cosine,
        "victim_to_source_norm": norm_ratio,
        "cross_spectral_coherence": coherence,
        "strong_bin_phase_cosine": strong_phase,
        "amplitude_cosine": amplitude_cosine,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5") as handle:
        clean_np = handle["image"][args.slice].astype(np.float32)
    clean = torch.from_numpy(clean_np)[None, None].repeat(1, 3, 1, 1).to(device)
    images = {
        path: torch.from_numpy(np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0)
        [None, None].repeat(1, 3, 1, 1).to(device)
        for path in args.candidates
    }
    factories = {
        "base": lambda: load_sam_encoder(args.vit, str(SAM[args.vit]), 512, 8, device),
        "SAMed": lambda: load_samed_model(
            args.vit, str(SAM[args.vit]),
            str(ROOT / f"checkpoints/best_weights/SAMed_Synapse_{args.vit}.pth"),
            512, 8, device=device,
        ),
        "MedSA": lambda: load_medsa_model(
            args.vit, str(SAM[args.vit]),
            str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
            512, 9, device=device,
        ),
        "MedSA-zero": lambda: load_medsa_base_surrogate(
            args.vit, str(SAM[args.vit]), 512, 9, device,
        ),
    }
    all_deltas = {}
    for name, factory in factories.items():
        model = factory()
        encoder = model.sam.image_encoder if name == "SAMed" else model.image_encoder
        preprocess = model.preprocess if name.startswith("MedSA") else lambda value: value
        all_deltas[name] = {
            path: displacement(model, encoder, preprocess, clean, image)
            for path, image in images.items()
        }
        del model
        torch.cuda.empty_cache()
    results = {}
    for path in args.candidates:
        results[path] = {}
        for name in ("SAMed", "MedSA", "MedSA-zero"):
            source_layers = all_deltas["base"][path]
            results[path][name] = [
                {"layer": index if index < len(source_layers) - 1 else "neck", **compare(a, b)}
                for index, (a, b) in enumerate(zip(source_layers, all_deltas[name][path]))
            ]
    payload = {"case": args.case, "slice": args.slice, "vit": args.vit,
               "post_hoc_victim_analysis": True, "results": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
