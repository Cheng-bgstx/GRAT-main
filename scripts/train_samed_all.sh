#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SAM_CKPT_VIT_B="${SAM_CKPT_VIT_B:-$REPO_ROOT/checkpoints/sam_vit_b_01ec64.pth}"
SAM_CKPT_VIT_L="${SAM_CKPT_VIT_L:-$REPO_ROOT/checkpoints/sam_vit_l_0b3195.pth}"
SAM_CKPT_VIT_H="${SAM_CKPT_VIT_H:-$REPO_ROOT/checkpoints/sam_vit_h_4b8939.pth}"
SYNAPSE_TRAIN_NPZ="${SYNAPSE_TRAIN_NPZ:-$REPO_ROOT/data/Synapse_preprocess/train_npz}"
SYNAPSE_LIST_DIR="${SYNAPSE_LIST_DIR:-$REPO_ROOT/data/Synapse_preprocess/lists}"
ISIC_ROOT="${ISIC_ROOT:-$REPO_ROOT/data/ISIC}"
REFUGE_ROOT="${REFUGE_ROOT:-$REPO_ROOT/data/REFUGE-Multirater}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/OUT_ROOT}"

run_one () {
  local dataset="$1"
  local vit="$2"
  local ckpt="$3"
  local root_path="$4"
  local list_dir="${5:-}"
  local bs="$6"
  local img_size="$7"

  echo
  echo "=== SAMed: dataset=$dataset vit=$vit bs=$bs img_size=$img_size ==="

  if [[ -n "$list_dir" ]]; then
    python SAMed_h/train.py \
      --dataset "$dataset" \
      --root_path "$root_path" \
      --list_dir "$list_dir" \
      --img_size "$img_size" --batch_size "$bs" --n_gpu 1 \
      --output "$OUT_ROOT/SAMed" \
      --vit_name "$vit" --ckpt "$ckpt" \
      --use_amp
  else
    python SAMed_h/train.py \
      --dataset "$dataset" \
      --root_path "$root_path" \
      --img_size "$img_size" --batch_size "$bs" --n_gpu 1 \
      --output "$OUT_ROOT/SAMed" \
      --vit_name "$vit" --ckpt "$ckpt" \
      --use_amp
  fi
}

# Synapse
run_one "Synapse" "vit_b" "$SAM_CKPT_VIT_B" "$SYNAPSE_TRAIN_NPZ" "$SYNAPSE_LIST_DIR" 12 512
run_one "Synapse" "vit_l" "$SAM_CKPT_VIT_L" "$SYNAPSE_TRAIN_NPZ" "$SYNAPSE_LIST_DIR" 12 512
run_one "Synapse" "vit_h" "$SAM_CKPT_VIT_H" "$SYNAPSE_TRAIN_NPZ" "$SYNAPSE_LIST_DIR" 6 512

# ISIC
run_one "isic" "vit_b" "$SAM_CKPT_VIT_B" "$ISIC_ROOT" "" 24 512
run_one "isic" "vit_l" "$SAM_CKPT_VIT_L" "$ISIC_ROOT" "" 16 512
run_one "isic" "vit_h" "$SAM_CKPT_VIT_H" "$ISIC_ROOT" "" 8 512

# REFUGE
run_one "REFUGE" "vit_b" "$SAM_CKPT_VIT_B" "$REFUGE_ROOT" "" 24 512
run_one "REFUGE" "vit_l" "$SAM_CKPT_VIT_L" "$REFUGE_ROOT" "" 16 512
run_one "REFUGE" "vit_h" "$SAM_CKPT_VIT_H" "$REFUGE_ROOT" "" 8 512

