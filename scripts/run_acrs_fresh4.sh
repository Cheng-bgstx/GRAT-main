#!/bin/bash
set -euo pipefail

ROOT=/root/data2/fyc/fyc/GRAT-main
PY=/root/anaconda3/envs/adattack/bin/python
GPU=${GPU:-4}

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" peft_transfer/analyze_acrs.py \
  --vit vit_b \
  --indices 220,221,222,223 \
  --candidates \
    carrier=testset/isic_carrier_fresh4/W075 \
    joint=testset/isic_pareto_joint_fresh4/W075 \
  --targets \
    rank16_samed=lora:checkpoints/samed_isic/isic_512_pretrain_vit_b_epo200_bs12_lr0.0001/epoch_99.pth:testset/isic_pareto_joint_fresh4_rank16_official.json:SAMed \
    seed45_samed=lora:output_runs_correct/SAMed_pareto_dev/isic_512_lrexp0.9_pretrain_vit_b_bs12_lr0.005_s45/epoch_29.pth:testset/isic_pareto_joint_fresh4_seed45_official.json:SAMed \
    default_medsa=adapter:checkpoints/best_weights/MSA_ISIC_vit_b.pth:testset/isic_pareto_joint_fresh4_rank16_official.json:MedSA_per_case_paired \
    seed45_medsa=adapter:output_runs_correct/MedicalSAMAdapter_pareto_dev/isic_512_pretrain_vit_b_epo10_bs24_lr0.0001_s45/Model/epoch_10.pth:testset/isic_pareto_joint_fresh4_seed45_official.json:MedSA_per_case_paired \
  --layers 0,5,11 \
  --device cuda:0 \
  --output testset/acrs_fresh4/acrs_gate.json
