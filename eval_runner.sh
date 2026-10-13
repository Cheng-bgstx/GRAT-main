#!/bin/bash
# Usage: eval_runner.sh <method:samed|msa> <dataset:Synapse|ISIC|REFUGE> <vit:b|l|h> <gpu>
set -u
PY=/root/anaconda3/envs/adattack/bin/python
ROOT=/root/data2/fyc/fyc/GRAT-main
method=$1; ds=$2; vit=$3; gpu=$4
export CUDA_VISIBLE_DEVICES=$gpu

case $vit in
  b) CKPT="$ROOT/checkpoints/sam_vit_b_01ec64.pth";;
  l) CKPT="$ROOT/checkpoints/sam_vit_l_0b3195.pth";;
  h) CKPT="$ROOT/checkpoints/sam_vit_h_4b8939.pth";;
esac

if [ "$method" = "samed" ]; then
  case $ds in
    Synapse) NC=8; VP="$ROOT/data/Synapse_preprocess/test_vol_h5"; LD="$ROOT/SAMed_h/lists/lists_Synapse";;
    ISIC)    NC=1; VP="$ROOT/data/ISIC"; LD="$ROOT/SAMed_h/lists/lists_Synapse";;
    REFUGE)  NC=2; VP="$ROOT/data/REFUGE-Multirater"; LD="$ROOT/SAMed_h/lists/lists_Synapse";;
  esac
  LORA="$ROOT/checkpoints/best_weights/SAMed_${ds}_vit_${vit}.pth"
  out=$($PY "$ROOT/test_samed.py" --dataset "$ds" --vit_name "vit_$vit" --num_classes $NC \
        --volume_path "$VP" --list_dir "$LD" --lora_ckpt "$LORA" --ckpt "$CKPT" \
        --img_size 512 --module sam_lora_image_encoder --consider_lora --use_clean_image 2>&1 \
        | /bin/grep -E "mean_dice|Testing performance|Error|Traceback|out of memory" | /usr/bin/tail -1)
  echo "[SAMed $ds vit_$vit] $out"

else  # msa
  cd "$ROOT/Medical-SAM-Adapter-main"
  case $ds in
    Synapse) mds="Synapse"; DP="../data/Synapse_preprocess/test_vol_h5"; LD="../data/Synapse_preprocess/lists"; MM="";;
    ISIC)    mds="isic";    DP="../data/ISIC"; LD="../dataset/Synapse/lists"; MM="";;
    REFUGE)  mds="REFUGE";  DP="../data/REFUGE-Multirater"; LD="../dataset/Synapse/lists"; MM="-multimask_output 2";;
  esac
  W="$ROOT/checkpoints/best_weights/MSA_${ds}_vit_${vit}.pth"
  out=$($PY val.py -dataset "$mds" -net sam -mod sam_adpt -encoder "vit_$vit" \
        -image_size 512 -out_size 512 $MM -data_path "$DP" -list_dir "$LD" \
        -weights "$W" -sam_ckpt "$CKPT" -gpu_device 0 2>&1 | tr '\r' '\n' \
        | /bin/grep -E "Mean Dice|DICE|HD95|Total score|Error|Traceback|out of memory" | /usr/bin/tail -4)
  echo "[MSA $ds vit_$vit]"
  echo "$out"
fi
