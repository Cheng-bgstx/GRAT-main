#!/bin/bash
# 用法: sweep_samed.sh <dataset:Synapse|ISIC|REFUGE> <vit:b|l|h> <gpu> <lora_ckpt1> <label1> [<lora_ckpt2> <label2> ...]
# 对多个 SAMed LoRA 权重依次跑 test_samed.py --use_clean_image，打印一行结果
set -u
PY=/root/anaconda3/envs/adattack/bin/python
ROOT=/root/data2/fyc/fyc/GRAT-main
ds=$1; vit=$2; gpu=$3; shift 3
export CUDA_VISIBLE_DEVICES=$gpu

case $vit in
  b) CKPT="$ROOT/checkpoints/sam_vit_b_01ec64.pth";;
  l) CKPT="$ROOT/checkpoints/sam_vit_l_0b3195.pth";;
  h) CKPT="$ROOT/checkpoints/sam_vit_h_4b8939.pth";;
esac

case $ds in
  Synapse) NC=8; VP="$ROOT/data/Synapse_preprocess/test_vol_h5"; LD="$ROOT/SAMed_h/lists/lists_Synapse";;
  ISIC)    NC=1; VP="$ROOT/data/ISIC"; LD="$ROOT/SAMed_h/lists/lists_Synapse";;
  REFUGE)  NC=2; VP="$ROOT/data/REFUGE-Multirater"; LD="$ROOT/SAMed_h/lists/lists_Synapse";;
esac

while [ $# -ge 2 ]; do
  lora="$1"; label="$2"; shift 2
  out=$($PY "$ROOT/test_samed.py" --dataset "$ds" --vit_name "vit_$vit" --num_classes $NC \
        --volume_path "$VP" --list_dir "$LD" --lora_ckpt "$lora" --ckpt "$CKPT" \
        --img_size 512 --module sam_lora_image_encoder --consider_lora --use_clean_image 2>&1 \
        | /bin/grep -aE "mean_dice" | /usr/bin/tail -1)
  echo "[$ds vit_$vit] $label  ->  $out"
done
