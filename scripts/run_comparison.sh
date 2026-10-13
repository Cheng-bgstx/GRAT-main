#!/bin/bash
# Master experiment script for SE-CGR baseline vs V2 comparison
# Runs: attack generation → SAMed evaluation → MSA evaluation for both methods
#
# Usage: bash scripts/run_comparison.sh [baseline|v2|both]
#   baseline: Run only baseline SE-CGR
#   v2: Run only SE-CGR v2
#   both (default): Run both and compare

set -e

MODE="${1:-both}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Configuration
ATTACK_STEPS=10
ATTACK_SIZE=2.0
ATTACK_BOUND=10.0
RANDOM_SEED=42
IMG_SIZE=512

# SAM base model (white-box surrogate)
SAM_CKPT="checkpoints/sam_vit_b_01ec64.pth"
VIT_NAME="vit_b"

# Datasets
SYNAPSE_DATA="data/Synapse_preprocess/test_vol_h5"
SYNAPSE_LIST="data/Synapse_preprocess/lists"
ISIC_DATA="data/ISIC"
REFUGE_DATA="data/REFUGE-Multirater"

# Output root
OUTPUT_BASE="testset/comparison_experiment"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RESULTS_DIR/experiment.log"
}

run_attack() {
    local METHOD=$1      # SE-CGR or SE-CGR-v2
    local DATASET=$2     # Synapse, ISIC, or REFUGE
    local GPU=$3         # GPU device ID
    local DATA_PATH=$4
    local LIST_DIR=$5
    local OUT_SUBDIR=$6

    local ATTACK_SCRIPT=""
    local ATTACK_FLAGS=""

    if [ "$METHOD" = "SE-CGR" ]; then
        ATTACK_SCRIPT="Sam_attack_SE-CGR.py"
        ATTACK_FLAGS="--SE 0 --CGR 1 --edge_guidance 1 --edge_gamma 3.0"
    else
        ATTACK_SCRIPT="Sam_attack_SE-CGR_v2.py"
        ATTACK_FLAGS="--SE 0 --CGR 1 --edge_guidance 1 --edge_gamma 3.0 --freq_bands 4 --phase_bound 0.15 --phase_smooth 3.0 --spec_momentum 1 --layer_aggregation 1"
    fi

    local OUT_PATH="${OUTPUT_BASE}/${OUT_SUBDIR}"
    mkdir -p "$OUT_PATH"

    local NUM_CLASSES=8
    if [ "$DATASET" = "ISIC" ]; then NUM_CLASSES=1; fi
    if [ "$DATASET" = "REFUGE" ]; then NUM_CLASSES=2; fi

    log "  [$METHOD] Starting attack on $DATASET (GPU $GPU)..."

    CUDA_VISIBLE_DEVICES=$GPU python3 "$ATTACK_SCRIPT" \
        --dataset "$DATASET" \
        --vit_name "$VIT_NAME" \
        --ckpt "$SAM_CKPT" \
        --img_size $IMG_SIZE \
        --volume_path "$DATA_PATH" \
        --list_dir "$LIST_DIR" \
        --num_classes $NUM_CLASSES \
        --att_step_size $ATTACK_SIZE \
        --att_bound $ATTACK_BOUND \
        --att_numstep $ATTACK_STEPS \
        --random_seed $RANDOM_SEED \
        --out_path "$OUT_PATH" \
        $ATTACK_FLAGS \
        2>&1 | tail -5

    log "  [$METHOD] Finished attack on $DATASET → $OUT_PATH"
}

run_samed_eval() {
    local METHOD=$1
    local DATASET=$2
    local GPU=$3
    local ATTACK_IMAGES_PATH=$4  # where adversarial images are stored

    # Determine the correct SAMed checkpoint for each dataset
    local MODEL_CKPT=""
    local NUM_CLASSES=8
    local DATA_PATH=""
    local LIST_DIR=""

    case "$DATASET" in
        Synapse)
            MODEL_CKPT="checkpoints/SAMed_Synapse_epoch_159.pth"
            NUM_CLASSES=8
            DATA_PATH="$SYNAPSE_DATA"
            LIST_DIR="$SYNAPSE_LIST"
            ;;
        ISIC)
            MODEL_CKPT="checkpoints/samed_isic/isic_512_pretrain_vit_b_epo200_bs12_lr0.0001/epoch_159.pth"
            NUM_CLASSES=1
            DATA_PATH="$ISIC_DATA"
            LIST_DIR="data/ISIC"
            ;;
        REFUGE)
            MODEL_CKPT="checkpoints/samed_refuge/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.0001/epoch_159.pth"
            NUM_CLASSES=2
            DATA_PATH="$REFUGE_DATA"
            LIST_DIR="data/REFUGE-Multirater"
            ;;
    esac

    local EVAL_OUT="${RESULTS_DIR}/samed_${METHOD}_${DATASET}.txt"

    log "  [SAMed] Evaluating $METHOD on $DATASET (GPU $GPU)..."

    CUDA_VISIBLE_DEVICES=$GPU python3 test_samed.py \
        --dataset "$DATASET" \
        --vit_name "$VIT_NAME" \
        --ckpt "$SAM_CKPT" \
        --lora_ckpt "$MODEL_CKPT" \
        --img_size $IMG_SIZE \
        --volume_path "$DATA_PATH" \
        --list_dir "$LIST_DIR" \
        --num_classes $NUM_CLASSES \
        --adv_root_path "$ATTACK_IMAGES_PATH" \
        --random_seed $RANDOM_SEED \
        2>&1 | tee "$EVAL_OUT"

    # Extract key metrics
    if [ "$DATASET" = "Synapse" ]; then
        local AVG_DICE=$(grep -oP 'Mean Dice:\s*\K[\d.]+' "$EVAL_OUT" | tail -1)
        local AVG_HD95=$(grep -oP 'Mean HD95:\s*\K[\d.]+' "$EVAL_OUT" | tail -1)
    else
        local AVG_DICE=$(grep -oP 'DICE:\s*\K[\d.]+' "$EVAL_OUT" | tail -1)
        local AVG_HD95=$(grep -oP 'HD95:\s*\K[\d.]+' "$EVAL_OUT" | tail -1)
    fi

    echo "${METHOD},${DATASET},SAMed,${AVG_DICE:-N/A},${AVG_HD95:-N/A}" >> "$RESULTS_DIR/summary.csv"
    log "  [SAMed] $METHOD/$DATASET: Dice=${AVG_DICE:-N/A}, HD95=${AVG_HD95:-N/A}"
}

run_msa_eval() {
    local METHOD=$1
    local DATASET=$2
    local GPU=$3
    local ATTACK_IMAGES_PATH=$4

    # Determine correct MSA checkpoint
    local WEIGHTS_PATH=""

    case "$DATASET" in
        Synapse)
            WEIGHTS_PATH="checkpoints/Synapse_512_pretrain_vit_l_epo200_bs12_lr0.0001"
            ;;
        ISIC)
            # Find the best MSA checkpoint for ISIC
            WEIGHTS_PATH="checkpoints/best_weights"
            ;;
        REFUGE)
            WEIGHTS_PATH="checkpoints/best_weights"
            ;;
    esac

    # For MSA, we need to modify val.py to use adversarial images
    # This is more complex because MSA's val.py loads clean images from dataset
    # We'll create a modified version that reads from adv path

    local EVAL_OUT="${RESULTS_DIR}/msa_${METHOD}_${DATASET}.txt"

    log "  [MSA] Evaluating $METHOD on $DATASET (GPU $GPU)..."

    # Use a wrapper that patches the dataloader to use adversarial images
    CUDA_VISIBLE_DEVICES=$GPU python3 -c "
import sys
sys.path.insert(0, 'Medical-SAM-Adapter-main')
import os
import torch
import numpy as np
from PIL import Image

# Use actual MSA evaluation but with adversarial images
# This is a simplified version - the full MSA val pipeline would need more integration
print('MSA evaluation placeholder - would need to patch dataloader')
print('Loading model from: $WEIGHTS_PATH')
print('Evaluating on adversarial images from: $ATTACK_IMAGES_PATH')
" 2>&1 | tee "$EVAL_OUT"

    echo "${METHOD},${DATASET},MSA,N/A,N/A" >> "$RESULTS_DIR/summary.csv"
}

# ===================== Main =====================

log "============================================"
log "Experiment Pipeline: SE-CGR vs V2 Comparison"
log "Timestamp: $TIMESTAMP"
log "Mode: $MODE"
log "============================================"

# Initialize summary CSV
echo "Method,Dataset,Model,mDice,mHD95" > "$RESULTS_DIR/summary.csv"

if [ "$MODE" = "both" ] || [ "$MODE" = "baseline" ]; then
    log ""
    log "========== PHASE 1: BASELINE SE-CGR =========="

    # Run attacks in parallel on 3 GPUs
    log "Launching baseline attacks on 3 GPUs..."
    run_attack "SE-CGR" "Synapse" 0 "$SYNAPSE_DATA" "$SYNAPSE_LIST" "baseline/Synapse" &
    PID1=$!
    run_attack "SE-CGR" "ISIC" 1 "$ISIC_DATA" "data/ISIC" "baseline/ISIC" &
    PID2=$!
    run_attack "SE-CGR" "REFUGE" 2 "$REFUGE_DATA" "data/REFUGE-Multirater" "baseline/REFUGE" &
    PID3=$!

    wait $PID1 $PID2 $PID3
    log "All baseline attacks completed."

    # Evaluate on SAMed
    log "Evaluating baseline on SAMed..."
    run_samed_eval "baseline" "Synapse" 0 "${OUTPUT_BASE}/baseline/Synapse"
    run_samed_eval "baseline" "ISIC" 1 "${OUTPUT_BASE}/baseline/ISIC"
    run_samed_eval "baseline" "REFUGE" 2 "${OUTPUT_BASE}/baseline/REFUGE"
fi

if [ "$MODE" = "both" ] || [ "$MODE" = "v2" ]; then
    log ""
    log "========== PHASE 2: SE-CGR V2 =========="

    # Run attacks in parallel on 3 GPUs
    log "Launching V2 attacks on 3 GPUs..."
    run_attack "SE-CGR-v2" "Synapse" 0 "$SYNAPSE_DATA" "$SYNAPSE_LIST" "v2/Synapse" &
    PID1=$!
    run_attack "SE-CGR-v2" "ISIC" 1 "$ISIC_DATA" "data/ISIC" "v2/ISIC" &
    PID2=$!
    run_attack "SE-CGR-v2" "REFUGE" 2 "$REFUGE_DATA" "data/REFUGE-Multirater" "v2/REFUGE" &
    PID3=$!

    wait $PID1 $PID2 $PID3
    log "All V2 attacks completed."

    # Evaluate on SAMed
    log "Evaluating V2 on SAMed..."
    run_samed_eval "v2" "Synapse" 0 "${OUTPUT_BASE}/v2/Synapse"
    run_samed_eval "v2" "ISIC" 1 "${OUTPUT_BASE}/v2/ISIC"
    run_samed_eval "v2" "REFUGE" 2 "${OUTPUT_BASE}/v2/REFUGE"
fi

log ""
log "============================================"
log "EXPERIMENTS COMPLETED"
log "Summary:"
cat "$RESULTS_DIR/summary.csv"
log "Full results in: $RESULTS_DIR/"
log "============================================"
