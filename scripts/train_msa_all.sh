#!/usr/bin/env bash
# Order matches scripts/train_samed_all.sh: Synapse -> isic -> REFUGE; each vit_b -> vit_l -> vit_h.
# Image size / batch sizes match train_samed_all.sh (512 + 12/12/6, 24/16/8, 24/16/8). Lower -b if OOM.
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
MSA_OUT="${MSA_OUT:-$OUT_ROOT/MedicalSAMAdapter}"

MOD="${MOD:-sam_adpt}"  # set MOD=sam_lora to train LoRA instead
GPU="${GPU:-0}"

run_one () {
  local dataset="$1"
  local vit="$2"
  local ckpt="$3"
  local data_path="$4"
  local list_dir="${5:-}"
  local image_size="$6"
  local out_size="$7"
  local bs="$8"
  local extra="${9:-}"

  echo
  echo "=== MSA: dataset=$dataset vit=$vit mod=$MOD bs=$bs image_size=$image_size (out_root=$MSA_OUT) ==="

  if [[ -n "$list_dir" ]]; then
    # shellcheck disable=SC2086
    python Medical-SAM-Adapter-main/train.py \
      -net sam -mod "$MOD" -dataset "$dataset" \
      -encoder "$vit" -sam_ckpt "$ckpt" \
      -data_path "$data_path" -list_dir "$list_dir" \
      -image_size "$image_size" -out_size "$out_size" -b "$bs" -gpu_device "$GPU" \
      -out_root "$MSA_OUT" \
      -exp_name "unused_when_out_root" \
      $extra
  else
    # shellcheck disable=SC2086
    python Medical-SAM-Adapter-main/train.py \
      -net sam -mod "$MOD" -dataset "$dataset" \
      -encoder "$vit" -sam_ckpt "$ckpt" \
      -data_path "$data_path" \
      -image_size "$image_size" -out_size "$out_size" -b "$bs" -gpu_device "$GPU" \
      -out_root "$MSA_OUT" \
      -exp_name "unused_when_out_root" \
      $extra
  fi
}

# Synapse (512) — same vit/batch pattern as train_samed_all.sh
run_one "Synapse" "vit_b" "$SAM_CKPT_VIT_B" "$SYNAPSE_TRAIN_NPZ" "$SYNAPSE_LIST_DIR" 512 512 12 ""
run_one "Synapse" "vit_l" "$SAM_CKPT_VIT_L" "$SYNAPSE_TRAIN_NPZ" "$SYNAPSE_LIST_DIR" 512 512 12 ""
run_one "Synapse" "vit_h" "$SAM_CKPT_VIT_H" "$SYNAPSE_TRAIN_NPZ" "$SYNAPSE_LIST_DIR" 512 512 6 ""

# ISIC (512)
run_one "isic" "vit_b" "$SAM_CKPT_VIT_B" "$ISIC_ROOT" "" 512 512 24 ""
run_one "isic" "vit_l" "$SAM_CKPT_VIT_L" "$ISIC_ROOT" "" 512 512 16 ""
run_one "isic" "vit_h" "$SAM_CKPT_VIT_H" "$ISIC_ROOT" "" 512 512 8 ""

# REFUGE (512) + multimask_output=2
run_one "REFUGE" "vit_b" "$SAM_CKPT_VIT_B" "$REFUGE_ROOT" "" 512 512 24 "-multimask_output 2"
run_one "REFUGE" "vit_l" "$SAM_CKPT_VIT_L" "$REFUGE_ROOT" "" 512 512 16 "-multimask_output 2"
run_one "REFUGE" "vit_h" "$SAM_CKPT_VIT_H" "$REFUGE_ROOT" "" 512 512 8 "-multimask_output 2"

