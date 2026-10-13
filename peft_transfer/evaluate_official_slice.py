#!/usr/bin/env python3
"""Evaluate slice candidates through the repository's two official functions."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from peft_transfer.evaluate_synapse_slice import ROOT, SAM

sys.path[:0] = [str(ROOT), str(ROOT / "newest")]
from evaluate import load_medsa_model, load_samed_model
from test_samed import test_single_volume


class SliceDataset(Dataset):
    def __init__(self, image: np.ndarray, label: np.ndarray, case_name: str):
        if image.ndim != 2 or label.ndim != 2 or image.shape != label.shape:
            raise ValueError(f"Expected matching 2D arrays, got {image.shape=} and {label.shape=}")
        labels = np.unique(label)
        if labels.min() < 0 or labels.max() > 8:
            raise ValueError(f"Synapse labels must be in [0, 8], got {labels.tolist()}")
        self.image = torch.from_numpy(image)[None, ..., None].float()
        self.label = torch.from_numpy(label)[None, ..., None].float()
        self.case_name = case_name

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return {"image": self.image, "label": self.label, "case_name": self.case_name}


def load_medsa_function(dataset: str = "Synapse", multimask_output: int = 9):
    msa_root = ROOT / "Medical-SAM-Adapter-main"
    saved_path = list(sys.path)
    saved_argv = list(sys.argv)
    sys.path.insert(0, str(msa_root))
    sys.argv = [saved_argv[0]]
    try:
        spec = importlib.util.spec_from_file_location("_official_medsa_function", msa_root / "function.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # function.py passes its local args to validation_sam, while the imported
        # prompt helper reads utils.args. val.py configures both through the same
        # CLI; synchronize the helper when importing it programmatically.
        module.generate_click_prompt.__globals__["args"].dataset = dataset
        module.generate_click_prompt.__globals__["args"].multimask_output = multimask_output
        return module
    finally:
        sys.path = saved_path
        sys.argv = saved_argv


def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--case", default="case0001")
    parser.add_argument("--slice", type=int, default=123)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-seed", type=int, default=24)
    parser.add_argument("--output", default="testset/official_slice_results.json")
    parser.add_argument("--skip-medsa", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5", "r") as handle:
        clean = handle["image"][args.slice].astype(np.float32)
        label = handle["label"][args.slice].astype(np.float32)
    images = {"clean": clean}
    images.update({
        path: np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        for path in args.candidates
    })

    samed = load_samed_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/SAMed_Synapse_{args.vit}.pth"),
        512, 8, device=device,
    )
    samed_results = {}
    clean_tensor = torch.from_numpy(clean)[None, None]
    label_tensor = torch.from_numpy(label)[None, None]
    for name, image in images.items():
        metric = test_single_volume(
            torch.from_numpy(image)[None, None], clean_tensor, label_tensor,
            samed, classes=8, multimask_output=True, patch_size=[512, 512],
            test_save_path=None, case=args.case, z_spacing=1, is_3d=True,
        )
        samed_results[name] = {
            "dice": float(np.mean(metric, axis=0)[0]),
            "hd95": float(np.mean(metric, axis=0)[1]),
        }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"SAMed": samed_results}, indent=2))
    print(json.dumps({"SAMed": samed_results}, indent=2), flush=True)
    del samed
    torch.cuda.empty_cache()

    if args.skip_medsa:
        return

    medsa = load_medsa_model(
        args.vit, str(SAM[args.vit]),
        str(ROOT / f"checkpoints/best_weights/MSA_Synapse_{args.vit}.pth"),
        512, 9, device=device,
    )
    medsa_function = load_medsa_function()
    medsa_args = Namespace(
        dataset="Synapse", multimask_output=9, thd=False, evl_chunk=1,
        gpu_device=0, image_size=512, out_size=512, net="sam", vis=0,
        path_helper={"sample_path": "/tmp"},
    )
    medsa_results = {}
    for name, image in images.items():
        reset_seed(args.prompt_seed)
        loader = DataLoader(SliceDataset(image, label, args.case), batch_size=1, shuffle=False, num_workers=0)
        _loss, metrics = medsa_function.validation_sam(medsa_args, loader, 0, medsa)
        dice, hd95 = float(metrics[0]), float(metrics[1])
        medsa_results[name] = {"dice": dice, "hd95": hd95}

        output_path.write_text(json.dumps({"SAMed": samed_results, "MedSA": medsa_results}, indent=2))

    print(json.dumps({"SAMed": samed_results, "MedSA": medsa_results}, indent=2))


if __name__ == "__main__":
    main()
