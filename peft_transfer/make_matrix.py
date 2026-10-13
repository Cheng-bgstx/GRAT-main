#!/usr/bin/env python3
"""Print reproducible generation commands for the 3 dataset x 3 ViT matrix."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/root/anaconda3/envs/adattack/bin/python"
for dataset in ("Synapse", "ISIC", "REFUGE"):
    for vit in ("vit_b", "vit_l", "vit_h"):
        output = ROOT / "testset" / "prsca" / dataset / vit
        print(f"CUDA_VISIBLE_DEVICES=0 {PYTHON} -m peft_transfer.generate --dataset {dataset} --vit {vit} --out {output}")
