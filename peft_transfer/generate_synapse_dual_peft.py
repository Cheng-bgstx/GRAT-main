#!/usr/bin/env python3
"""Generate Synapse slice attacks from independent LoRA and Adapter proxies."""
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
sys.path.insert(0, str(ROOT))

from peft_transfer.evaluate_official_slice import SAM, load_medsa_model, load_samed_model
from peft_transfer.generate_2d_dual_peft import attack


def load_grayscale_prior(path: str | None, device: torch.device):
    """Load a UMI gray field, including legacy tensors stored in one RGB slot."""
    if path is None:
        return None
    prior = torch.load(ROOT / path, map_location="cpu", weights_only=True)
    if isinstance(prior, dict):
        for key in ("perturbation", "delta", "noise"):
            if key in prior:
                prior = prior[key]
                break
    if not isinstance(prior, torch.Tensor) or prior.ndim != 4:
        raise ValueError(f"unsupported perturbation in {path}")
    if prior.shape[1] == 3:
        # Legacy Synapse UMI files store the grayscale field in channel zero
        # and pad the other channels with zeros.  Select the payload channel;
        # averaging would silently reduce its epsilon by a factor of three.
        channel_energy = prior.square().flatten(2).mean(2).mean(0)
        prior = prior[:, int(channel_energy.argmax()), None]
    if prior.shape[1] != 1:
        raise ValueError(f"expected one or three channels, got {tuple(prior.shape)}")
    return prior.repeat(1, 3, 1, 1).to(device=device, dtype=torch.float32)


def save_grayscale(path: Path, adv: torch.Tensor, clean: torch.Tensor):
    clean_np = clean[0, 0].detach().cpu().numpy()
    adv_np = adv[0, 0].detach().cpu().numpy()
    lower = np.ceil(np.maximum(0.0, clean_np - 10.0 / 255.0) * 255.0).astype(np.int16)
    upper = np.floor(np.minimum(1.0, clean_np + 10.0 / 255.0) * 255.0).astype(np.int16)
    value = np.clip(np.rint(adv_np * 255.0).astype(np.int16), lower, upper).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)
    return float(np.abs(value.astype(np.float32) - clean_np * 255.0).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slices", type=int, nargs="+", required=True)
    parser.add_argument("--samed-proxy", required=True)
    parser.add_argument("--medsa-proxies", nargs="+", required=True)
    parser.add_argument("--samed-weights", type=float, nargs="+", default=(0.65, 0.75))
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--uncertainty-lambda", type=float, default=0.5)
    parser.add_argument("--medsa-objective", choices=("l1", "roll", "collision"),
                        default="l1")
    parser.add_argument("--collision-shift-fraction", type=float, default=0.25)
    parser.add_argument("--boundary", choices=("clip", "centered"), default="centered")
    parser.add_argument("--initial-perturbation", default=None,
                        help="optional UMI tensor, projected to shared CT gray channels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    samed = load_samed_model(
        args.vit, str(SAM[args.vit]), str(ROOT / args.samed_proxy),
        512, 8, device=device,
    )
    medsa_models = [
        load_medsa_model(
            args.vit, str(SAM[args.vit]), str(ROOT / checkpoint),
            512, 9, device=device, dataset="Synapse",
        )
        for checkpoint in args.medsa_proxies
    ]
    for parameter in list(samed.parameters()) + [
        parameter for model in medsa_models for parameter in model.parameters()
    ]:
        parameter.requires_grad_(False)
    initial_perturbation = load_grayscale_prior(args.initial_perturbation, device)
    records = []
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        for slice_index in args.slices:
            image_np = handle["image"][slice_index].astype(np.float32)
            clean = torch.from_numpy(image_np)[None, None].repeat(1, 3, 1, 1).to(device)
            for weight in args.samed_weights:
                adv, trace = attack(
                    samed, medsa_models, clean, lambda value: value,
                    mode="weighted", samed_weight=weight,
                    medsa_uncertainty_lambda=args.uncertainty_lambda,
                    medsa_objective=args.medsa_objective,
                    initial_perturbation=initial_perturbation,
                    diversity_seed=args.seed + slice_index * 100,
                    gradient_smoothing_sigma=args.sigma / 255.0,
                    smoothing_boundary=args.boundary,
                    grayscale_projection=True,
                    medsa_input_scale=1.0,
                    collision_shift_fraction=args.collision_shift_fraction,
                )
                initialization = "UMI" if initial_perturbation is not None else "ZERO"
                objective = args.medsa_objective.upper()
                candidate = (
                    f"{initialization}_W{int(round(weight * 100)):03d}_"
                    f"{objective}_LCB_SMOOTH{args.sigma:g}_{args.boundary}"
                )
                path = ROOT / args.out / candidate / f"{slice_index}.png"
                record = {
                    "case": args.case, "slice": slice_index, "candidate": candidate,
                    "target_checkpoints_used": False,
                    "samed_proxy": args.samed_proxy,
                    "medsa_proxies": args.medsa_proxies,
                    "initial_perturbation": args.initial_perturbation,
                    "samed_weight": weight,
                    "sigma_255": args.sigma,
                    "boundary": args.boundary,
                    "uncertainty_lambda": args.uncertainty_lambda,
                    "medsa_objective": args.medsa_objective,
                    "linf_255": save_grayscale(path, adv, clean),
                    "trace": trace,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    output = ROOT / args.out / "generation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
