#!/usr/bin/env python3
"""Validate the exact first-adapter rank-tail transfer lower bound."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "newest")]

from evaluate import load_medsa_model, load_sam_encoder
from peft_transfer.medsa_surrogate import load_medsa_base_surrogate


SAM = {"vit_b": ROOT / "checkpoints/sam_vit_b_01ec64.pth"}


class Capture:
    def __init__(self, module):
        self.value = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.value = output if torch.is_tensor(output) else output[0]

    def close(self):
        self.handle.remove()


def tail_energy(matrix: torch.Tensor, rank: int):
    singular = torch.linalg.svdvals(matrix.float())
    squared = singular.square()
    cutoff = min(rank, len(squared))
    return squared[cutoff:].sum(), singular


def matrix(value):
    return value.reshape(-1, value.shape[-1])


def effective_rank(singular, tolerance=1e-5):
    return int((singular > singular.max().clamp_min(1e-12) * tolerance).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slices", type=int, nargs="+", default=[78, 95, 112, 123, 129])
    parser.add_argument("--candidate-template", default="testset/ck_multislice/slice{slice}/CK.png")
    parser.add_argument("--slice123-candidate", default="testset/deterministic_schedule_gate/CK.png")
    parser.add_argument("--rank", type=int, default=192)
    parser.add_argument("--source", choices=("canonical", "medsa_base"), default="medsa_base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="testset/first_adapter_bound.json")
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.source == "canonical":
        source = load_sam_encoder("vit_b", str(SAM["vit_b"]), 512, 8, device)
        source_encoder = source.image_encoder
        source_preprocess = lambda value: value
    else:
        source = load_medsa_base_surrogate("vit_b", str(SAM["vit_b"]), 512, 9, device)
        source_encoder = source.image_encoder
        source_preprocess = source.preprocess
    medsa = load_medsa_model(
        "vit_b", str(SAM["vit_b"]),
        str(ROOT / "checkpoints/best_weights/MSA_Synapse_vit_b.pth"),
        512, 9, device=device,
    )
    source_capture = Capture(source_encoder.blocks[0].attn)
    victim_attn_capture = Capture(medsa.image_encoder.blocks[0].attn)
    victim_adapter_capture = Capture(medsa.image_encoder.blocks[0].Space_Adapter)
    rows = []
    try:
        with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
            for index in args.slices:
                clean_np = handle["image"][index].astype(np.float32)
                template = args.slice123_candidate if index == 123 else args.candidate_template
                candidate_np = np.asarray(
                    Image.open(ROOT / template.format(slice=index)).convert("L"), dtype=np.float32
                ) / 255.0
                clean = torch.from_numpy(clean_np)[None, None].repeat(1, 3, 1, 1).to(device)
                candidate = torch.from_numpy(candidate_np)[None, None].repeat(1, 3, 1, 1).to(device)

                captured = {}
                for state, image in (("clean", clean), ("adv", candidate)):
                    with torch.no_grad():
                        source_encoder(source_preprocess(image))
                        captured[f"source_{state}"] = source_capture.value.detach().clone()
                        medsa.image_encoder(medsa.preprocess(image))
                        captured[f"victim_attn_{state}"] = victim_attn_capture.value.detach().clone()
                        captured[f"victim_adapter_{state}"] = victim_adapter_capture.value.detach().clone()

                source_displacement = matrix(captured["source_adv"] - captured["source_clean"])
                victim_base_displacement = matrix(
                    captured["victim_attn_adv"] - captured["victim_attn_clean"]
                )
                victim_displacement = matrix(
                    captured["victim_adapter_adv"] - captured["victim_adapter_clean"]
                )
                correction = victim_displacement - victim_base_displacement
                tail, source_singular = tail_energy(source_displacement, args.rank)
                correction_tail, correction_singular = tail_energy(correction, args.rank)
                source_energy = source_displacement.square().sum()
                victim_energy = victim_displacement.square().sum()
                base_gap = (victim_base_displacement - source_displacement).norm()
                row = {
                    "slice": index,
                    "source_tail_ratio": float(tail / source_energy.clamp_min(1e-12)),
                    "certified_victim_energy_ratio": float(tail / victim_energy.clamp_min(1e-12)),
                    "bound_holds": bool(victim_energy + 1e-5 * source_energy >= tail),
                    "source_victim_base_relative_gap": float(
                        base_gap / source_displacement.norm().clamp_min(1e-12)
                    ),
                    "correction_effective_rank": effective_rank(correction_singular),
                    "correction_tail_ratio": float(
                        correction_tail / correction.square().sum().clamp_min(1e-12)
                    ),
                    "source_effective_rank": effective_rank(source_singular),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        source_capture.close()
        victim_attn_capture.close()
        victim_adapter_capture.close()

    summary = {
        "mean_source_tail_ratio": float(np.mean([row["source_tail_ratio"] for row in rows])),
        "mean_certified_victim_energy_ratio": float(np.mean([
            row["certified_victim_energy_ratio"] for row in rows
        ])),
        "max_base_relative_gap": float(max(row["source_victim_base_relative_gap"] for row in rows)),
        "max_correction_effective_rank": int(max(row["correction_effective_rank"] for row in rows)),
        "all_bounds_hold": all(row["bound_holds"] for row in rows),
    }
    payload = {"rank": args.rank, "source": args.source, "rows": rows, "summary": summary}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
