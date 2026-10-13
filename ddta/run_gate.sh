#!/usr/bin/env bash
# D2TA first-round gate: Synapse/BTCV x SAM ViT-B x {SAMed, Med-SA}.
# 5 groups: base / +offline / +online / full / random-basis.
#
# Conservative online params (alpha=0.3 beta=0.3 rho=0.5 tau=0.85) are the
# script defaults in online_ddta_attack.py.
set -euo pipefail

ROOT=/root/data2/fyc/fyc/GRAT-main
PY=/root/anaconda3/envs/adattack/bin/python
GPU=0
cd "$ROOT"

PRIOR=ddta/drift_prior_Synapse_vit_b.pth
OUT=testset/gate_Synapse_vitb
GROUPS_JSON='{"base":"'$OUT'/base","offline":"'$OUT'/offline","online":"'$OUT'/online","full":"'$OUT'/full","random":"'$OUT'/random"}'

echo "==== [1/4] offline drift prior (num_samples=128 rank=16) ===="
if [ -f "$PRIOR" ]; then
  echo "prior already exists, skipping -> $PRIOR"
else
  CUDA_VISIBLE_DEVICES=$GPU $PY ddta/offline_drift_prior.py \
    --dataset Synapse --vit_name vit_b --num_classes 8 \
    --ckpt checkpoints/sam_vit_b_01ec64.pth --img_size 512 \
    --volume_path data/Synapse_preprocess/train_npz \
    --list_dir data/Synapse_preprocess/lists \
    --num_samples 128 --rank 16 --out_path "$PRIOR"
fi

echo "==== [2/4] 5 online attack groups ===="
COMMON="--dataset Synapse --vit_name vit_b --num_classes 8 \
  --ckpt checkpoints/sam_vit_b_01ec64.pth --img_size 512 \
  --volume_path data/Synapse_preprocess/test_vol_h5 \
  --list_dir data/Synapse_preprocess/lists"

echo "--- base ---"
CUDA_VISIBLE_DEVICES=$GPU $PY ddta/online_ddta_attack.py $COMMON \
  --out_path "$OUT/base"

echo "--- +offline ---"
CUDA_VISIBLE_DEVICES=$GPU $PY ddta/online_ddta_attack.py $COMMON \
  --drift_prior "$PRIOR" --use_offline_decouple --out_path "$OUT/offline"

echo "--- +online ---"
CUDA_VISIBLE_DEVICES=$GPU $PY ddta/online_ddta_attack.py $COMMON \
  --use_online_decouple --out_path "$OUT/online"

echo "--- full ---"
CUDA_VISIBLE_DEVICES=$GPU $PY ddta/online_ddta_attack.py $COMMON \
  --drift_prior "$PRIOR" --use_offline_decouple --use_online_decouple --out_path "$OUT/full"

echo "--- random basis ---"
CUDA_VISIBLE_DEVICES=$GPU $PY ddta/online_ddta_attack.py $COMMON \
  --use_offline_decouple --random_basis --basis_rank 16 --out_path "$OUT/random"

echo "==== [3/4] target eval (SAMed + Med-SA) ===="
CUDA_VISIBLE_DEVICES=$GPU $PY ddta/eval_gate.py --vit vit_b \
  --samed_ckpt checkpoints/best_weights/SAMed_Synapse_vit_b.pth \
  --medsa_ckpt checkpoints/best_weights/MSA_Synapse_vit_b.pth \
  --groups "$GROUPS_JSON" --out ddta/gate_results.json

echo "==== [4/4] summary table ===="
$PY ddta/summarize_gate_eval.py --results ddta/gate_results.json --groups "$GROUPS_JSON"
