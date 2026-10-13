#!/bin/bash
# ============================================================
# AutoSAM Training Script (default sam_size=512, matching SAMed/MSA)
# Supports datasets: isic, refuge, synapse, monu, glas, polyp, lung
# ============================================================
#
# Prerequisites:
#   1. SAM checkpoint: download sam_vit_h.pth to ./cp/sam_vit_h.pth
#      wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
#   2. Conda environment with required packages (torch, cv2, sklearn, etc.)
#
# Usage:
#   bash run_train.sh <task> [epochs] [batch_size] [lr] [nW] [nW_eval] [sam_size]
#
# Examples:
#   bash run_train.sh isic 100 3 0.01 4 2 512
#   bash run_train.sh refuge 100 2 0.01 4 2 512
#   bash run_train.sh synapse 100 4 0.01 4 2 512
# ============================================================

set -e

TASK=${1:-isic}
EPOCHS=${2:-100}
BATCH_SIZE=${3:-3}
LR=${4:-0.01}
NW=${5:-4}
NW_EVAL=${6:-2}
SAM_SIZE=${7:-512}
OUTPUT_DIR=${8:-./output}

echo "=============================================="
echo "AutoSAM Training"
echo "  Task:       ${TASK}"
echo "  Epochs:     ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  LR:         ${LR}"
echo "  SAM size:   ${SAM_SIZE}"
echo "  Output:     ${OUTPUT_DIR}"
echo "=============================================="

# Activate conda environment
source /root/anaconda3/etc/profile.d/conda.sh
conda activate sam3

# Ensure SAM checkpoint directory
mkdir -p ./cp

# Check SAM checkpoint
SAM_CKPT="./cp/sam_vit_h.pth"
if [ ! -f "$SAM_CKPT" ]; then
    echo "WARNING: SAM checkpoint not found at $SAM_CKPT"
    echo "  Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    echo "  Or use vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    echo ""
    # Check for alternative checkpoints
    if [ -f "./cp/sam_vit_b.pth" ]; then
        echo "  Found vit_b checkpoint, will use that instead."
        SAM_CKPT="./cp/sam_vit_b.pth"
        MODEL_TYPE="vit_b"
    else
        echo "  ERROR: No SAM checkpoint found. Please download one."
        exit 1
    fi
else
    MODEL_TYPE="vit_h"
fi

# Run training
cd /root/data2/fyc/fyc/GRAT-main/AutoSAM-main

python3 train.py \
    --task "${TASK}" \
    --epoches "${EPOCHS}" \
    --Batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --nW "${NW}" \
    --nW_eval "${NW_EVAL}" \
    --sam_size "${SAM_SIZE}" \
    --output_dir "${OUTPUT_DIR}" \
    --rotate 15 \
    --scale1 0.8 \
    --scale2 1.2

echo "Training complete!"
echo "Results saved to: ${OUTPUT_DIR}"
