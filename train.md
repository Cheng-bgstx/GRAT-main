# 训练命令

```bash
export REPO_ROOT="/root/data2/fyc/fyc/GRAT-main"
export SAM_CKPT_VIT_B="$REPO_ROOT/checkpoints/sam_vit_b_01ec64.pth"
export SAM_CKPT_VIT_L="$REPO_ROOT/checkpoints/sam_vit_l_0b3195.pth"
export SAM_CKPT_VIT_H="$REPO_ROOT/checkpoints/sam_vit_h_4b8939.pth"

export SYNAPSE_ROOT="$REPO_ROOT/data/Synapse_preprocess/train_npz"
export SYNAPSE_LIST_DIR="$REPO_ROOT/SAMed_h/lists/lists_Synapse"
export ISIC_ROOT="$REPO_ROOT/data/ISIC"
export REFUGE_ROOT="$REPO_ROOT/data/REFUGE-Multirater"

export OUT_SAMED_SYNAPSE="$REPO_ROOT/checkpoints"
export OUT_SAMED_ISIC="$REPO_ROOT/checkpoints/samed_isic"
export OUT_SAMED_REFUGE="$REPO_ROOT/checkpoints/samed_refuge"
export OUT_MSA="$REPO_ROOT/OFFICIAL_RETRAIN/MedicalSAMAdapter"
```

先执行上面的 `export`，再执行下面命令；否则像 `"$SYNAPSE_ROOT"` 这类变量会展开成空字符串，最终就会出现只去找 `case0040_slice184.npz` 这种错误路径。

# SAMed

说明：
- `vit_b` 尽量对齐你现有可信 `checkpoints` 的官方训练风格。
- `ISIC/REFUGE` 的 `vit_l/vit_h` 仍然走 `SAMed_h/train.py`，但训练流程保持同一套，只改 `vit_name` 和对应 SAM 权重，不改与 ViT 无关的训练逻辑。
- 统一保持：`base_lr=0.0001`、`batch_size=12`、`rank=16`、`--warmup`、`--AdamW`、不开 `--use_amp`。

## Synapse vit_l

```bash
CUDA_VISIBLE_DEVICES=5 python train.py \
  --dataset Synapse \
  --root_path "$SYNAPSE_ROOT" \
  --list_dir "$SYNAPSE_LIST_DIR" \
  --output "$OUT_SAMED_SYNAPSE" \
  --vit_name vit_l \
  --ckpt "$SAM_CKPT_VIT_L" \
  --img_size 512 \
  --batch_size 12 \
  --n_gpu 1 \
  --base_lr 0.0001 \
  --rank 16 \
  --max_epochs 200 \
  --stop_epoch 160 \
  --seed 1234 \
  --warmup \
  --AdamW
```

## ISIC vit_b

```bash
CUDA_VISIBLE_DEVICES=0 python SAMed_h/train.py \
  --dataset isic \
  --root_path "$ISIC_ROOT" \
  --output "$OUT_SAMED_ISIC" \
  --vit_name vit_b \
  --ckpt "$SAM_CKPT_VIT_B" \
  --img_size 512 \
  --batch_size 12 \
  --n_gpu 1 \
  --base_lr 0.0001 \
  --rank 16 \
  --max_epochs 200 \
  --stop_epoch 160 \
  --seed 1234 \
  --lr_exp 0.9 \
  --warmup \
  --AdamW
```

## ISIC vit_l

```bash
CUDA_VISIBLE_DEVICES=0 python SAMed_h/train.py \
  --dataset isic \
  --root_path "$ISIC_ROOT" \
  --output "$OUT_SAMED_ISIC" \
  --vit_name vit_l \
  --ckpt "$SAM_CKPT_VIT_L" \
  --img_size 512 \
  --batch_size 12 \
  --n_gpu 1 \
  --base_lr 0.0001 \
  --rank 16 \
  --max_epochs 200 \
  --stop_epoch 160 \
  --seed 1234 \
  --lr_exp 0.9 \
  --warmup \
  --AdamW
```

## ISIC vit_h

```bash
CUDA_VISIBLE_DEVICES=0 python SAMed_h/train.py \
  --dataset isic \
  --root_path "$ISIC_ROOT" \
  --output "$OUT_SAMED_ISIC" \
  --vit_name vit_h \
  --ckpt "$SAM_CKPT_VIT_H" \
  --img_size 512 \
  --batch_size 12 \
  --n_gpu 1 \
  --base_lr 0.0001 \
  --rank 16 \
  --max_epochs 200 \
  --stop_epoch 160 \
  --seed 1234 \
  --lr_exp 0.9 \
  --warmup \
  --AdamW
```

## REFUGE vit_b

```bash
CUDA_VISIBLE_DEVICES=0 python SAMed_h/train.py \
  --dataset REFUGE \
  --root_path "$REFUGE_ROOT" \
  --output "$OUT_SAMED_REFUGE" \
  --vit_name vit_b \
  --ckpt "$SAM_CKPT_VIT_B" \
  --img_size 512 \
  --batch_size 12 \
  --n_gpu 1 \
  --base_lr 0.0001 \
  --rank 16 \
  --max_epochs 200 \
  --stop_epoch 160 \
  --seed 1234 \
  --lr_exp 0.9 \
  --warmup \
  --AdamW
```

## REFUGE vit_l

```bash
CUDA_VISIBLE_DEVICES=0 python SAMed_h/train.py \
  --dataset REFUGE \
  --root_path "$REFUGE_ROOT" \
  --output "$OUT_SAMED_REFUGE" \
  --vit_name vit_l \
  --ckpt "$SAM_CKPT_VIT_L" \
  --img_size 512 \
  --batch_size 12 \
  --n_gpu 1 \
  --base_lr 0.0001 \
  --rank 16 \
  --max_epochs 200 \
  --stop_epoch 160 \
  --seed 1234 \
  --lr_exp 0.9 \
  --warmup \
  --AdamW
```

## REFUGE vit_h

```bash
CUDA_VISIBLE_DEVICES=0 python SAMed_h/train.py \
  --dataset REFUGE \
  --root_path "$REFUGE_ROOT" \
  --output "$OUT_SAMED_REFUGE" \
  --vit_name vit_h \
  --ckpt "$SAM_CKPT_VIT_H" \
  --img_size 512 \
  --batch_size 12 \
  --n_gpu 1 \
  --base_lr 0.0001 \
  --rank 16 \
  --max_epochs 200 \
  --stop_epoch 160 \
  --seed 1234 \
  --lr_exp 0.9 \
  --warmup \
  --AdamW
```

# Medical-SAM-Adapter

说明：
- 训练入口固定为 `Medical-SAM-Adapter-main/train.py`。
- 训练流程、优化器、验证频率、保存逻辑都保留当前仓库原实现。
- 扩展到 `vit_l/vit_h` 时，只改 `-encoder` 和 `-sam_ckpt`，其余与 ViT 架构无关的流程保持不变。
- 下面是 `3 个数据集 x 3 个 ViT` 的 9 条命令。

## Synapse vit_b

```bash
CUDA_VISIBLE_DEVICES=0 python Medical-SAM-Adapter-main/train.py \
  -net sam \
  -mod sam_adpt \
  -dataset Synapse \
  -out_root "$OUT_MSA" \
  -encoder vit_b \
  -sam_ckpt "$SAM_CKPT_VIT_B" \
  -data_path "$SYNAPSE_ROOT" \
  -list_dir "$SYNAPSE_LIST_DIR" \
  -image_size 512 \
  -out_size 512 \
  -b 12 \
  -gpu_device 0 \
  -exp_name x
```

## Synapse vit_l

```bash
CUDA_VISIBLE_DEVICES=0 python Medical-SAM-Adapter-main/train.py \
  -net sam \
  -mod sam_adpt \
  -dataset Synapse \
  -out_root "$OUT_MSA" \
  -encoder vit_l \
  -sam_ckpt "$SAM_CKPT_VIT_L" \
  -data_path "$SYNAPSE_ROOT" \
  -list_dir "$SYNAPSE_LIST_DIR" \
  -image_size 512 \
  -out_size 512 \
  -b 12 \
  -gpu_device 0 \
  -exp_name x
```

## Synapse vit_h

```bash
CUDA_VISIBLE_DEVICES=0 python Medical-SAM-Adapter-main/train.py \
  -net sam \
  -mod sam_adpt \
  -dataset Synapse \
  -out_root "$OUT_MSA" \
  -encoder vit_h \
  -sam_ckpt "$SAM_CKPT_VIT_H" \
  -data_path "$SYNAPSE_ROOT" \
  -list_dir "$SYNAPSE_LIST_DIR" \
  -image_size 512 \
  -out_size 512 \
  -b 6 \
  -gpu_device 0 \
  -exp_name x
```

## ISIC vit_b

```bash
CUDA_VISIBLE_DEVICES=0 python Medical-SAM-Adapter-main/train.py \
  -net sam \
  -mod sam_adpt \
  -dataset isic \
  -out_root "$OUT_MSA" \
  -encoder vit_b \
  -sam_ckpt "$SAM_CKPT_VIT_B" \
  -data_path "$ISIC_ROOT" \
  -image_size 512 \
  -out_size 512 \
  -b 24 \
  -gpu_device 0 \
  -exp_name x
```

## ISIC vit_l

```bash
CUDA_VISIBLE_DEVICES=0 python Medical-SAM-Adapter-main/train.py \
  -net sam \
  -mod sam_adpt \
  -dataset isic \
  -out_root "$OUT_MSA" \
  -encoder vit_l \
  -sam_ckpt "$SAM_CKPT_VIT_L" \
  -data_path "$ISIC_ROOT" \
  -image_size 512 \
  -out_size 512 \
  -b 16 \
  -gpu_device 0 \
  -exp_name x
```

## ISIC vit_h

```bash
CUDA_VISIBLE_DEVICES=0 python Medical-SAM-Adapter-main/train.py \
  -net sam \
  -mod sam_adpt \
  -dataset isic \
  -out_root "$OUT_MSA" \
  -encoder vit_h \
  -sam_ckpt "$SAM_CKPT_VIT_H" \
  -data_path "$ISIC_ROOT" \
  -image_size 512 \
  -out_size 512 \
  -b 8 \
  -gpu_device 0 \
  -exp_name x
```

## REFUGE vit_b

```bash
CUDA_VISIBLE_DEVICES=0 python Medical-SAM-Adapter-main/train.py \
  -net sam \
  -mod sam_adpt \
  -dataset REFUGE \
  -out_root "$OUT_MSA" \
  -encoder vit_b \
  -sam_ckpt "$SAM_CKPT_VIT_B" \
  -data_path "$REFUGE_ROOT" \
  -image_size 512 \
  -out_size 512 \
  -b 24 \
  -gpu_device 0 \
  -multimask_output 2 \
  -exp_name x
```

## REFUGE vit_l

```bash
CUDA_VISIBLE_DEVICES=0 python Medical-SAM-Adapter-main/train.py \
  -net sam \
  -mod sam_adpt \
  -dataset REFUGE \
  -out_root "$OUT_MSA" \
  -encoder vit_l \
  -sam_ckpt "$SAM_CKPT_VIT_L" \
  -data_path "$REFUGE_ROOT" \
  -image_size 512 \
  -out_size 512 \
  -b 16 \
  -gpu_device 0 \
  -multimask_output 2 \
  -exp_name x
```

## REFUGE vit_h

```bash
CUDA_VISIBLE_DEVICES=0 python Medical-SAM-Adapter-main/train.py \
  -net sam \
  -mod sam_adpt \
  -dataset REFUGE \
  -out_root "$OUT_MSA" \
  -encoder vit_h \
  -sam_ckpt "$SAM_CKPT_VIT_H" \
  -data_path "$REFUGE_ROOT" \
  -image_size 512 \
  -out_size 512 \
  -b 8 \
  -gpu_device 0 \
  -multimask_output 2 \
  -exp_name x
```
