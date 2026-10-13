# 训练命令（完全符合官方要求）

> **关键修正**（相比之前的版本）：
> - SAMed: `base_lr=0.005`（之前错误写成 0.0001，差了 50 倍）
> - SAMed: `rank=4`（之前错误写成 16）
> - SAMed: `--module sam_lora_image_encoder`（mask decoder 全参数训练，与官方 Synapse checkpoint 一致）
> - SAMed: `--warmup --AdamW`（必须使用 AdamW + warmup）

---

## 环境变量（先执行）

```bash
export REPO_ROOT="/root/data2/fyc/fyc/GRAT-main"

export SAM_CKPT_VIT_B="$REPO_ROOT/checkpoints/sam_vit_b_01ec64.pth"
export SAM_CKPT_VIT_L="$REPO_ROOT/checkpoints/sam_vit_l_0b3195.pth"
export SAM_CKPT_VIT_H="$REPO_ROOT/checkpoints/sam_vit_h_4b8939.pth"

export SYNAPSE_ROOT="$REPO_ROOT/data/Synapse_preprocess/train_npz"
export SYNAPSE_LIST_DIR="$REPO_ROOT/SAMed_h/lists/lists_Synapse"
export ISIC_ROOT="$REPO_ROOT/data/ISIC"
export REFUGE_ROOT="$REPO_ROOT/data/REFUGE-Multirater"

export OUT_SAMED="$REPO_ROOT/output_runs_correct/SAMed"
export OUT_MSA="$REPO_ROOT/output_runs_correct/MedicalSAMAdapter"

conda activate adattack
```

---

## 一、SAMed（LoRA-SAM）— 官方参数

**官方论文超参**：`base_lr=0.005 --rank 4 --warmup --AdamW --module sam_lora_image_encoder`

- 训练 image encoder LoRA（Q/V projection）+ mask decoder 全参数 + prompt encoder 全参数
- `--seed 1234 --img_size 512`
- **vit_b**: `max_epochs=200 --stop_epoch 160`（官方默认值）
- **vit_l**: `max_epochs=300 --stop_epoch 250`（24 blocks，按比例 300×160/200=240，取 250）
- **vit_h**: `max_epochs=300 --stop_epoch 250`（32 blocks，同上按比例缩放；SAMed_h 官方对 224×224 用 `--max_epochs 400 --stop_epoch 300`）
- 优化器：AdamW（betas=(0.9, 0.999), weight_decay=0.1）
- 学习率调度：polynomial decay `lr = base_lr * (1 - iter/max_iter) ** 0.9`（warmup 250 iters）
- 损失：CE + Dice（dice_param=0.8）

入口：`SAMed_h/train.py`

### 1.1 Synapse（512×512，8 类）

```bash
# vit_b — bs=12
CUDA_VISIBLE_DEVICES=0 python SAMed_h/train.py \
  --dataset Synapse --root_path "$SYNAPSE_ROOT" --list_dir "$SYNAPSE_LIST_DIR" \
  --output "$OUT_SAMED" --vit_name vit_b --ckpt "$SAM_CKPT_VIT_B" \
  --module sam_lora_image_encoder \
  --img_size 512 --batch_size 12 --n_gpu 1 --base_lr 0.005 --rank 4 \
  --max_epochs 200 --stop_epoch 160 --seed 1234 --warmup --AdamW

# vit_l — bs=12
CUDA_VISIBLE_DEVICES=4 python SAMed_h/train.py \
  --dataset Synapse --root_path "$SYNAPSE_ROOT" --list_dir "$SYNAPSE_LIST_DIR" \
  --output "$OUT_SAMED" --vit_name vit_l --ckpt "$SAM_CKPT_VIT_L" \
  --module sam_lora_image_encoder \
  --img_size 512 --batch_size 12 --n_gpu 1 --base_lr 0.005 --rank 4 \
  --max_epochs 300 --stop_epoch 250 --seed 1234 --warmup --AdamW

# vit_h — bs=6（A100 40GB 显存限制）
CUDA_VISIBLE_DEVICES=2 python SAMed_h/train.py \
  --dataset Synapse --root_path "$SYNAPSE_ROOT" --list_dir "$SYNAPSE_LIST_DIR" \
  --output "$OUT_SAMED" --vit_name vit_h --ckpt "$SAM_CKPT_VIT_H" \
  --module sam_lora_image_encoder \
  --img_size 512 --batch_size 6 --n_gpu 1 --base_lr 0.005 --rank 4 \
  --max_epochs 300 --stop_epoch 250 --seed 1234 --warmup --AdamW
```

### 1.2 ISIC（~1024×768，1 类）

```bash
# vit_b — bs=12
CUDA_VISIBLE_DEVICES=0 python SAMed_h/train.py \
  --dataset isic --root_path "$ISIC_ROOT" \
  --output "$OUT_SAMED" --vit_name vit_b --ckpt "$SAM_CKPT_VIT_B" \
  --module sam_lora_image_encoder \
  --img_size 512 --batch_size 12 --n_gpu 1 --base_lr 0.005 --rank 4 \
  --max_epochs 200 --stop_epoch 160 --seed 1234 --lr_exp 0.9 --warmup --AdamW

# vit_l — bs=12
CUDA_VISIBLE_DEVICES=1 python SAMed_h/train.py \
  --dataset isic --root_path "$ISIC_ROOT" \
  --output "$OUT_SAMED" --vit_name vit_l --ckpt "$SAM_CKPT_VIT_L" \
  --module sam_lora_image_encoder \
  --img_size 512 --batch_size 12 --n_gpu 1 --base_lr 0.005 --rank 4 \
  --max_epochs 300 --stop_epoch 250 --seed 1234 --lr_exp 0.9 --warmup --AdamW

# vit_h — bs=8
CUDA_VISIBLE_DEVICES=3 python SAMed_h/train.py \
  --dataset isic --root_path "$ISIC_ROOT" \
  --output "$OUT_SAMED" --vit_name vit_h --ckpt "$SAM_CKPT_VIT_H" \
  --module sam_lora_image_encoder \
  --img_size 512 --batch_size 8 --n_gpu 1 --base_lr 0.005 --rank 4 \
  --max_epochs 300 --stop_epoch 250 --seed 1234 --lr_exp 0.9 --warmup --AdamW
```

### 1.3 REFUGE（~2124×2056，2 类：Cup + Disc）

```bash
# vit_b — bs=12
CUDA_VISIBLE_DEVICES=5 python SAMed_h/train.py \
  --dataset REFUGE --root_path "$REFUGE_ROOT" \
  --output "$OUT_SAMED" --vit_name vit_b --ckpt "$SAM_CKPT_VIT_B" \
  --module sam_lora_image_encoder \
  --img_size 512 --batch_size 12 --n_gpu 1 --base_lr 0.005 --rank 4 \
  --max_epochs 200 --stop_epoch 160 --seed 1234 --lr_exp 0.9 --warmup --AdamW

# vit_l — bs=12
CUDA_VISIBLE_DEVICES=4 python SAMed_h/train.py \
  --dataset REFUGE --root_path "$REFUGE_ROOT" \
  --output "$OUT_SAMED" --vit_name vit_l --ckpt "$SAM_CKPT_VIT_L" \
  --module sam_lora_image_encoder \
  --img_size 512 --batch_size 12 --n_gpu 1 --base_lr 0.005 --rank 4 \
  --max_epochs 300 --stop_epoch 250 --seed 1234 --lr_exp 0.9 --warmup --AdamW

# vit_h — bs=8
CUDA_VISIBLE_DEVICES=0 python SAMed_h/train.py \
  --dataset REFUGE --root_path "$REFUGE_ROOT" \
  --output "$OUT_SAMED" --vit_name vit_h --ckpt "$SAM_CKPT_VIT_H" \
  --module sam_lora_image_encoder \
  --img_size 512 --batch_size 8 --n_gpu 1 --base_lr 0.005 --rank 4 \
  --max_epochs 300 --stop_epoch 250 --seed 1234 --lr_exp 0.9 --warmup --AdamW
```

---

## 二、Medical-SAM-Adapter（MSA）— 官方参数

**官方超参**：`-lr 1e-4 -mod sam_adpt -seed 24`

- 优化器：Adam（betas=(0.9, 0.999), weight_decay=0）
- 调度器：StepLR（step=10, gamma=0.5）
- EPOCH=100（`conf/global_settings.py`）
- val_freq=5
- 数据归一化：SAM preprocess（pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]）
- ISIC/REFUGE：`-image_size 1024 -out_size 1024`（官方推荐）
- Synapse：`-image_size 512 -out_size 512`
- REFUGE：`-multimask_output 2`

入口：`Medical-SAM-Adapter-main/train.py`

### 2.1 Synapse（512×512，8 类）

```bash
# vit_b — bs=12
CUDA_VISIBLE_DEVICES=0 python Medical-SAM-Adapter-main/train.py \
  -net sam -mod sam_adpt -dataset Synapse -out_root "$OUT_MSA" \
  -encoder vit_b -sam_ckpt "$SAM_CKPT_VIT_B" \
  -data_path "$SYNAPSE_ROOT" -list_dir "$SYNAPSE_LIST_DIR" \
  -image_size 512 -out_size 512 -b 12 -gpu_device 0 -exp_name x

# vit_l — bs=12
CUDA_VISIBLE_DEVICES=1 python Medical-SAM-Adapter-main/train.py \
  -net sam -mod sam_adpt -dataset Synapse -out_root "$OUT_MSA" \
  -encoder vit_l -sam_ckpt "$SAM_CKPT_VIT_L" \
  -data_path "$SYNAPSE_ROOT" -list_dir "$SYNAPSE_LIST_DIR" \
  -image_size 512 -out_size 512 -b 12 -gpu_device 0 -exp_name x

# vit_h — bs=6
CUDA_VISIBLE_DEVICES=3 python Medical-SAM-Adapter-main/train.py \
  -net sam -mod sam_adpt -dataset Synapse -out_root "$OUT_MSA" \
  -encoder vit_h -sam_ckpt "$SAM_CKPT_VIT_H" \
  -data_path "$SYNAPSE_ROOT" -list_dir "$SYNAPSE_LIST_DIR" \
  -image_size 512 -out_size 512 -b 6 -gpu_device 0 -exp_name x
```

### 2.2 ISIC（512×512，1 类）

```bash
# vit_b — bs=24
CUDA_VISIBLE_DEVICES=1 python Medical-SAM-Adapter-main/train.py \
  -net sam -mod sam_adpt -dataset isic -out_root "$OUT_MSA" \
  -encoder vit_b -sam_ckpt "$SAM_CKPT_VIT_B" -data_path "$ISIC_ROOT" \
  -image_size 512 -out_size 512 -b 24 -gpu_device 0 -exp_name x

# vit_l — bs=16
CUDA_VISIBLE_DEVICES=3 python Medical-SAM-Adapter-main/train.py \
  -net sam -mod sam_adpt -dataset isic -out_root "$OUT_MSA" \
  -encoder vit_l -sam_ckpt "$SAM_CKPT_VIT_L" -data_path "$ISIC_ROOT" \
  -image_size 512 -out_size 512 -b 16 -gpu_device 0 -exp_name x

# vit_h — bs=8
CUDA_VISIBLE_DEVICES=4 python Medical-SAM-Adapter-main/train.py \
  -net sam -mod sam_adpt -dataset isic -out_root "$OUT_MSA" \
  -encoder vit_h -sam_ckpt "$SAM_CKPT_VIT_H" -data_path "$ISIC_ROOT" \
  -image_size 512 -out_size 512 -b 8 -gpu_device 0 -exp_name x
```

### 2.3 REFUGE（512×512，2 类）— 降分辨率防 OOM

> REFUGE 原始 1634×1634，dataloader 每 sample 加载 14 张 mask（7 cup + 7 disc），
> 1024×1024 + multimask_output=2 导致 vit_l/vit_h 显存溢出，统一降为 512×512。

```bash
# vit_b — bs=24
CUDA_VISIBLE_DEVICES=4 python Medical-SAM-Adapter-main/train.py \
  -net sam -mod sam_adpt -dataset REFUGE -out_root "$OUT_MSA" \
  -encoder vit_b -sam_ckpt "$SAM_CKPT_VIT_B" -data_path "$REFUGE_ROOT" \
  -image_size 512 -out_size 512 -b 24 -gpu_device 0 \
  -multimask_output 2 -exp_name x

# vit_l — bs=12
CUDA_VISIBLE_DEVICES=5 python Medical-SAM-Adapter-main/train.py \
  -net sam -mod sam_adpt -dataset REFUGE -out_root "$OUT_MSA" \
  -encoder vit_l -sam_ckpt "$SAM_CKPT_VIT_L" -data_path "$REFUGE_ROOT" \
  -image_size 512 -out_size 512 -b 12 -gpu_device 0 \
  -multimask_output 2 -exp_name x

# vit_h — bs=6
CUDA_VISIBLE_DEVICES=5 python Medical-SAM-Adapter-main/train.py \
  -net sam -mod sam_adpt -dataset REFUGE -out_root "$OUT_MSA" \
  -encoder vit_h -sam_ckpt "$SAM_CKPT_VIT_H" -data_path "$REFUGE_ROOT" \
  -image_size 512 -out_size 512 -b 6 -gpu_device 0 \
  -multimask_output 2 -exp_name x
```

---

## 三、参数汇总

### SAMed（与论文 arXiv:2304.13785 完全一致）

| 参数 | 值 | 说明 |
|------|-----|------|
| base_lr | **0.005** | 论文初始学习率 |
| rank | **4** | LoRA rank（论文值） |
| 优化器 | **AdamW** | betas=(0.9,0.999), weight_decay=0.1 |
| warmup | **True** | 250 iterations linear warmup |
| module | **sam_lora_image_encoder** | Image Encoder LoRA + Mask Decoder 全参数 + Prompt Encoder 全参数 |
| seed | 1234 | |
| max_epochs | vit_b=200, vit_l=300, vit_h=300 | SAMed/SAMed_h 官方设定 |
| stop_epoch | vit_b=160, vit_l=250, vit_h=250 | 在 stop_epoch 处停止优化（早停） |
| img_size | 512 | 所有数据集 |
| lr_schedule | polynomial decay | `lr * (1 - iter/max_iter)^0.9` |
| lr_exp | 0.9 | ISIC/REFUGE 需要显式指定 |
| dice_param | 0.8 | CE:Dice = 0.2:0.8 |
| batch_size | vit_b=12, vit_l=12, vit_h=6~8 | 视 GPU 显存调整 |

### Medical-SAM-Adapter（与官方 GitHub 完全一致）

| 参数 | 值 | 说明 |
|------|-----|------|
| lr | **1e-4** | |
| 优化器 | **Adam** | betas=(0.9,0.999), weight_decay=0 |
| 调度器 | **StepLR** | step=10, gamma=0.5 |
| mod | sam_adpt | |
| seed | 24 | MSA 默认值 |
| EPOCH | 100 | conf/global_settings.py |
| val_freq | 5 | |
| image_size | 512（Synapse, REFUGE）/ 1024（ISIC） | REFUGE 降 512 防 OOM |
| multimask_output | 2（REFUGE）/ 1（其他） | |

### GPU 分配建议（6×A100 40GB）

| GPU | SAMed | Medical-SAM-Adapter |
|-----|-------|---------------------|
| 0 | Synapse vit_b (bs12) | Synapse vit_b (bs12) |
| 1 | Synapse vit_l (bs12) | Synapse vit_l (bs12) |
| 2 | Synapse vit_h (bs6) | Synapse vit_h (bs6) |
| 3 | ISIC vit_b (bs12) | ISIC vit_b (bs24) |
| 4 | ISIC vit_l (bs12) | REFUGE vit_b (bs24) |
| 5 | ISIC vit_h (bs8) | REFUGE vit_l (bs12) |

> REFUGE vit_h 复用 GPU 4（vit_b 完成后）。ISIC MSA 使用 GPU 3/4/5（REFUGE 完成后）。

---

## 四、输出目录结构

### SAMed
```
$OUT_SAMED/
  Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/
    config.txt, log.txt, log/
    epoch_19.pth, epoch_39.pth, ..., epoch_159.pth
  Synapse_512_pretrain_vit_l_epo300_bs12_lr0.005/
    ...
  isic_512_pretrain_vit_b_epo200_bs12_lr0.005/
    ...
```

### Medical-SAM-Adapter
```
$OUT_MSA/
  Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/
    Model/    → best_dice_checkpoint.pth, checkpoint_best.pth
    Log/      → 训练日志
    Samples/  → 可视化样本
    log/      → TensorBoard
  isic_1024_pretrain_vit_b_epo100_bs24_lr0.0001_s24/
    ...
```

---

## 五、已有资产 & 待训练清单

训练输出路径：`$OUT_SAMED` / `$OUT_MSA` → `output_runs_correct/`

### ✅ SAMed 可用（module=sam_lora_image_encoder，decoder 全参数）

| 数据集 | 变体 | 路径 | Dice | 备注 |
|--------|------|------|------|------|
| Synapse | **vit_b** | `checkpoints/SAMed_Synapse_epoch_159.pth` | 0.763 | 官方权重 |

### ✅ MSA 可用

| 数据集 | vit_b | vit_l | vit_h |
|--------|-------|-------|-------|
| ISIC | 0.949 ✅ | 0.957 ✅ | 0.959 ✅ |
| REFUGE | 0.885 ✅ | 0.875 ✅ | 0.896 ✅ |
| Synapse | ⚠️ NaN loss | ⚠️ NaN loss | ⚠️ NaN loss |

### ❌ SAMed 需要重训（之前用错 module=sam_lora_image_encoder_mask_decoder）

| # | 数据集 | 变体 | batch_size | 原因 |
|---|--------|------|------------|------|
| 1 | ISIC | vit_b | 12 | module 错 |
| 2 | ISIC | vit_l | 12 | module 错 |
| 3 | ISIC | vit_h | 8 | module 错 |
| 4 | REFUGE | vit_b | 12 | module 错 |
| 5 | REFUGE | vit_l | 12 | module 错 |
| 6 | REFUGE | vit_h | 8 | module 错 |
| 7 | Synapse | vit_l | 12 | module 错 |
| 8 | Synapse | vit_h | 6 | 过训练 (epoch 299→stop_epoch=250) |

### ❌ MSA Synapse 需要重训（NaN loss）

| # | 数据集 | 变体 | batch_size | 原因 |
|---|--------|------|------------|------|
| 9 | Synapse | vit_b | 12 | train loss NaN from epoch 0 |
| 10 | Synapse | vit_l | 12 | train loss NaN from epoch 0 |
| 11 | Synapse | vit_h | 6 | train loss NaN from epoch 0 |

> 共 **11 项**需要重训。
> 旧路径 `output_runs/SAMed/`、`OUT_ROOT/`、`checkpoints/samed_isic/`、`checkpoints/samed_refuge/` 均为错误参数产物，忽略。
> `output_runs_correct/SAMed/` 中现有的 ISIC/REFUGE/Synapse_vit_l checkpoint（module=mask_decoder）也是错误产物，重训后会覆盖。
