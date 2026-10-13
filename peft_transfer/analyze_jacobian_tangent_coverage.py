#!/usr/bin/env python3
"""Test whether source structural tangents cover PEFT victim gradients.

Victim gradients are used only for post-hoc measurement.  Every tangent basis
vector is generated from the frozen source SAM while preserving its forward
activations exactly.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]

from evaluate import load_medsa_model, load_sam_encoder, load_samed_model
from peft_transfer.jacobian_subspace import (
    affine_basis_projection,
    affine_tangent_projection,
    carrier_component_coverage,
    riesz_pair,
    unit_l2,
)


SAM = {
    "vit_b": ROOT / "checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": ROOT / "checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": ROOT / "checkpoints/sam_vit_h_4b8939.pth",
}


def tensor_output(value):
    return value if torch.is_tensor(value) else value[0]


class SelectiveBackwardGain:
    """Scale chosen branch Jacobians without changing their forward values."""

    def __init__(self, encoder, sites: Sequence[Tuple[int, str]], gain: float):
        self.encoder = encoder
        self.sites = tuple(sites)
        self.gain = float(gain)
        self.handles = []

    @contextlib.contextmanager
    def active(self):
        try:
            for layer, site in self.sites:
                module = getattr(self.encoder.blocks[layer], site)

                def hook(_module, _inputs, output, gain=self.gain):
                    value = tensor_output(output)
                    scaled = value.detach() + gain * (value - value.detach())
                    return scaled if torch.is_tensor(output) else (scaled, *output[1:])

                self.handles.append(module.register_forward_hook(hook))
            yield
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()


def capture_blocks(encoder, image):
    outputs, handles = [], []
    for block in encoder.blocks:
        handles.append(block.register_forward_hook(
            lambda _module, _inputs, output: outputs.append(tensor_output(output))
        ))
    try:
        final = encoder(image)
    finally:
        for handle in handles:
            handle.remove()
    return tensor_output(final), outputs


def late_indices(depth: int, count: int = 6) -> List[int]:
    stop = max(1, int(depth * 0.9))
    return list(range(max(0, stop - count), stop))


def make_probes(clean_layers, indices, seed):
    generator = torch.Generator(device=clean_layers[0].device).manual_seed(seed)
    return [torch.randn(clean_layers[i].shape, generator=generator, device=clean_layers[i].device) for i in indices]


def common_loss(final, layers, clean_final, clean_layers, indices, probes):
    terms = []
    for index, probe in zip(indices, probes):
        scale = clean_layers[index].abs().mean().clamp_min(1e-6)
        residual = ((layers[index] * 0.999 - clean_layers[index]) - 2.0 * probe) / scale
        terms.append(residual.abs().mean())
    output = F.l1_loss(final * 0.9999, clean_final)
    return torch.stack(terms).mean() + output


def gradient_for(encoder, clean, point, indices, probes, preprocess, context=None):
    clean_input = preprocess(clean)
    point_var = point.detach().clone().requires_grad_(True)
    with torch.no_grad():
        clean_final, clean_layers = capture_blocks(encoder, clean_input)
        clean_final = clean_final.detach()
        clean_layers = [value.detach() for value in clean_layers]
    manager = context if context is not None else contextlib.nullcontext()
    with manager:
        final, layers = capture_blocks(encoder, preprocess(point_var))
        loss = common_loss(final, layers, clean_final, clean_layers, indices, probes)
    gradient = torch.autograd.grad(loss, point_var)[0].detach()
    return unit_l2(gradient), float(loss.detach())


def source_gradients(encoder, clean, point, gain, seed):
    preprocess = lambda value: value
    with torch.no_grad():
        _final, clean_layers = capture_blocks(encoder, clean)
    indices = late_indices(len(encoder.blocks))
    probes = make_probes(clean_layers, indices, seed)
    source, loss = gradient_for(encoder, clean, point, indices, probes, preprocess)
    depth = len(encoder.blocks)
    attn, mlp, blocks = [], [], []
    for layer in range(depth):
        attn.append(gradient_for(
            encoder, clean, point, indices, probes, preprocess,
            SelectiveBackwardGain(encoder, [(layer, "attn")], gain).active(),
        )[0].cpu())
        mlp.append(gradient_for(
            encoder, clean, point, indices, probes, preprocess,
            SelectiveBackwardGain(encoder, [(layer, "mlp")], gain).active(),
        )[0].cpu())
        blocks.append(gradient_for(
            encoder, clean, point, indices, probes, preprocess,
            SelectiveBackwardGain(encoder, [(layer, "attn"), (layer, "mlp")], gain).active(),
        )[0].cpu())
    global_view = gradient_for(
        encoder, clean, point, indices, probes, preprocess,
        SelectiveBackwardGain(
            encoder, [(layer, site) for layer in range(depth) for site in ("attn", "mlp")], gain
        ).active(),
    )[0].cpu()
    return source.cpu(), {"global": [global_view], "block": blocks, "attn_mlp": [*attn, *mlp]}, indices, probes, loss


def summarize(source, victim, families, ridge):
    source_cos = float(torch.dot(unit_l2(source.reshape(-1)), unit_l2(victim.reshape(-1))))
    result: Dict[str, object] = {"source_cosine": source_cos}
    for name, views in families.items():
        projection = affine_tangent_projection(source, victim, views, ridge=ridge)
        result[name] = {
            "relative_residual": projection.relative_residual,
            "cosine_reconstruction": projection.cosine_reconstruction,
            "coefficient_norm": projection.coefficient_norm,
            "effective_rank": projection.effective_rank,
            **carrier_component_coverage(source, victim, projection.reconstruction),
        }
        source_flat = unit_l2(source.float().reshape(-1)).reshape_as(source)
        tangents = [unit_l2(view.float().reshape(-1)).reshape_as(view) - source_flat for view in views]
        quadratures = []
        for tangent in [source_flat, *tangents]:
            quadratures.extend(riesz_pair(tangent))
        phase_projection = affine_basis_projection(
            source, victim, [*tangents, *quadratures], ridge=ridge
        )
        result[name + "_riesz"] = {
            "relative_residual": phase_projection.relative_residual,
            "cosine_reconstruction": phase_projection.cosine_reconstruction,
            "coefficient_norm": phase_projection.coefficient_norm,
            "effective_rank": phase_projection.effective_rank,
            **carrier_component_coverage(source, victim, phase_projection.reconstruction),
        }
    return result


def load_point(case, index, template, device):
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{case}.npy.h5", "r") as handle:
        clean_np = handle["image"][index].astype(np.float32)
    clean = torch.from_numpy(clean_np)[None, None].repeat(1, 3, 1, 1).to(device)
    if template:
        path = ROOT / template.format(slice=index)
        point_np = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        point = torch.from_numpy(point_np)[None, None].repeat(1, 3, 1, 1).to(device)
    else:
        generator = torch.Generator(device=device).manual_seed(991 + index)
        point = (clean + 1e-4 * torch.randn(clean.shape, generator=generator, device=device)).clamp(0, 1)
    return clean, point


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slices", type=int, nargs="+", default=[78, 95, 112, 129])
    parser.add_argument("--candidate-template", default="testset/ck_multislice/slice{slice}/CK.png")
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="testset/jacobian_tangent_coverage.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    records = []

    source_model = load_sam_encoder(args.vit, str(SAM[args.vit]), 512, num_classes=8, device=device)
    cached = {}
    for index in args.slices:
        clean, point = load_point(args.case, index, args.candidate_template, device)
        source, families, indices, probes, source_loss = source_gradients(
            source_model.image_encoder, clean, point, args.gain, 1701 + index
        )
        cached[index] = {
            "clean": clean.cpu(), "point": point.cpu(), "source": source,
            "families": families, "indices": indices,
            "probes": [probe.cpu() for probe in probes], "source_loss": source_loss,
        }
        print(f"[source] slice={index} tangents={sum(len(v) for v in families.values())}", flush=True)
    del source_model
    torch.cuda.empty_cache()

    victim_specs = [
        ("SAMed", lambda: load_samed_model(
            args.vit, str(SAM[args.vit]),
            str(ROOT / f"checkpoints/best_weights/SAMed_Synapse_{args.vit}.pth"),
            512, 8, device=device,
        ), lambda model: model.sam.image_encoder, lambda value: value),
        ("MedSA", lambda: load_medsa_model(
            args.vit, str(SAM[args.vit]),
            str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
            512, 9, device=device,
        ), lambda model: model.image_encoder, lambda model: model.preprocess),
    ]
    for victim_name, loader, encoder_getter, preprocess_spec in victim_specs:
        model = loader()
        encoder = encoder_getter(model)
        preprocess = preprocess_spec(model) if victim_name == "MedSA" else preprocess_spec
        for index in args.slices:
            item = cached[index]
            clean, point = item["clean"].to(device), item["point"].to(device)
            probes = [probe.to(device) for probe in item["probes"]]
            victim, victim_loss = gradient_for(
                encoder, clean, point, item["indices"], probes, preprocess
            )
            measurement = summarize(item["source"], victim.cpu(), item["families"], args.ridge)
            records.append({
                "slice": index, "victim": victim_name,
                "source_loss": item["source_loss"], "victim_loss": victim_loss,
                **measurement,
            })
            print(json.dumps(records[-1]), flush=True)
        del model
        torch.cuda.empty_cache()

    aggregate = {}
    for victim_name, *_ in victim_specs:
        rows = [row for row in records if row["victim"] == victim_name]
        aggregate[victim_name] = {
            "source_cosine": float(np.mean([row["source_cosine"] for row in rows])),
            **{
                family: {
                    key: float(np.mean([row[family][key] for row in rows]))
                    for key in ("relative_residual", "cosine_reconstruction", "coefficient_norm",
                                "victim_radial_fraction", "radial_relative_residual",
                                "tangential_relative_residual")
                }
                for family in (
                    "global", "global_riesz", "block", "block_riesz",
                    "attn_mlp", "attn_mlp_riesz",
                )
            },
        }
    payload = {
        "protocol": {
            "victim_use": "post-hoc measurement only", "gain": args.gain,
            "ridge": args.ridge, "candidate_template": args.candidate_template,
        },
        "records": records,
        "aggregate": aggregate,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
