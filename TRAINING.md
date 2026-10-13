# 训练指南：SAMed（LoRA）与 Medical-SAM-Adapter（Adapter/LoRA）

本仓库里有两条独立训练线：

- **SAMed（LoRA-SAM）**：使用 `SAMed_h/train.py`
- **Medical-SAM-Adapter**：使用 `Medical-SAM-Adapter-main/train.py`

下面提供**可直接复制运行**的命令模板（3 个 ViT × 3 数据集），并提供数据自检脚本。

---

## 0. 统一配置：先设置环境变量（建议）

SAM 预训练权重在 `checkpoints/`，数据在 `data/`。可直接复制下面配置（或保持默认，脚本已内置这些路径）：

```bash
# SAM 预训练权重（checkpoints/）
export SAM_CKPT_VIT_B="/root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth"
export SAM_CKPT_VIT_L="/root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_l_0b3195.pth"
export SAM_CKPT_VIT_H="/root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_h_4b8939.pth"

# Synapse：train_npz + lists（train.txt/test_vol.txt）
export SYNAPSE_TRAIN_NPZ="/root/data2/fyc/fyc/GRAT-main/data/Synapse_preprocess/train_npz"
export SYNAPSE_LIST_DIR="/root/data2/fyc/fyc/GRAT-main/data/Synapse_preprocess/lists"

# ISIC / REFUGE：根目录
export ISIC_ROOT="/root/data2/fyc/fyc/GRAT-main/data/ISIC"
export REFUGE_ROOT="/root/data2/fyc/fyc/GRAT-main/data/REFUGE-Multirater"

# 输出根目录
export OUT_ROOT="/root/data2/fyc/fyc/GRAT-main/output_runs"
```

---

## 1. 数据最小结构要求（快速核对）

### 1.1 Synapse（SAMed / Medical-SAM-Adapter 都会用到）

- `SYNAPSE_LIST_DIR/train.txt`：每行一个 slice_id（不带后缀），训练时拼成 `${SYNAPSE_TRAIN_NPZ}/${slice_id}.npz`
- `SYNAPSE_LIST_DIR/test_vol.txt`：每行一个 volume_id，测试时拼成 `${test_vol_h5}/${volume_id}.npy.h5`
- `SYNAPSE_TRAIN_NPZ/<slice_id>.npz`：必须包含 `image` 与 `label` 两个 key
- Medical-SAM-Adapter 会从 `train_npz` 推断 `test_vol_h5`（见 `Medical-SAM-Adapter-main/dataset/__init__.py`），因此建议保持目录命名包含 `train_npz`。

### 1.2 ISIC / REFUGE

这两者的目录结构由对应 dataset 文件决定（SAMed 用 `datasets/isic.py`、`datasets/refuge.py`；Medical-SAM-Adapter 用其子仓库下的 dataset 实现）。如果你“已经能跑 test/val”，通常训练也就能读。

---

## 2. 训练方案 A：SAMed（LoRA-SAM）

入口脚本：`SAMed_h/train.py`

### 2.1 通用参数要点

- `--vit_name vit_b|vit_l|vit_h`
- `--ckpt <对应SAM预训练权重>`
- `--dataset Synapse|isic|REFUGE`
- `--root_path`：
  - Synapse：指向 `train_npz` 目录
  - isic/REFUGE：指向数据集根目录
- `--list_dir`：Synapse 需要（ISIC/REFUGE 可不传或给个默认）
- 资源/速度：
  - 显存紧：先用 vit_b + 减小 `--batch_size`
  - 想提速：加 `--use_amp`、可选 `--tf32`、`--compile`

### 2.2 Synapse：3×vit 命令

```bash
python SAMed_h/train.py \
  --dataset Synapse \
  --root_path "$SYNAPSE_TRAIN_NPZ" \
  --list_dir "$SYNAPSE_LIST_DIR" \
  --img_size 512 --batch_size 12 --n_gpu 1 \
  --output "$OUT_ROOT/SAMed" \
  --vit_name vit_b --ckpt "$SAM_CKPT_VIT_B" \
  --use_amp

python SAMed_h/train.py \
  --dataset Synapse \
  --root_path "$SYNAPSE_TRAIN_NPZ" \
  --list_dir "$SYNAPSE_LIST_DIR" \
  --img_size 512 --batch_size 12 --n_gpu 1 \
  --output "$OUT_ROOT/SAMed" \
  --vit_name vit_l --ckpt "$SAM_CKPT_VIT_L" \
  --use_amp

python SAMed_h/train.py \
  --dataset Synapse \
  --root_path "$SYNAPSE_TRAIN_NPZ" \
  --list_dir "$SYNAPSE_LIST_DIR" \
  --img_size 512 --batch_size 6 --n_gpu 1 \
  --output "$OUT_ROOT/SAMed" \
  --vit_name vit_h --ckpt "$SAM_CKPT_VIT_H" \
  --use_amp
```

### 2.3 ISIC：3×vit 命令

```bash
python SAMed_h/train.py \
  --dataset isic \
  --root_path "$ISIC_ROOT" \
  --img_size 512 --batch_size 24 --n_gpu 1 \
  --output "$OUT_ROOT/SAMed" \
  --vit_name vit_b --ckpt "$SAM_CKPT_VIT_B" \
  --use_amp

python SAMed_h/train.py \
  --dataset isic \
  --root_path "$ISIC_ROOT" \
  --img_size 512 --batch_size 16 --n_gpu 1 \
  --output "$OUT_ROOT/SAMed" \
  --vit_name vit_l --ckpt "$SAM_CKPT_VIT_L" \
  --use_amp

python SAMed_h/train.py \
  --dataset isic \
  --root_path "$ISIC_ROOT" \
  --img_size 512 --batch_size 8 --n_gpu 1 \
  --output "$OUT_ROOT/SAMed" \
  --vit_name vit_h --ckpt "$SAM_CKPT_VIT_H" \
  --use_amp
```

### 2.4 REFUGE：3×vit 命令

```bash
python SAMed_h/train.py \
  --dataset REFUGE \
  --root_path "$REFUGE_ROOT" \
  --img_size 512 --batch_size 24 --n_gpu 1 \
  --output "$OUT_ROOT/SAMed" \
  --vit_name vit_b --ckpt "$SAM_CKPT_VIT_B" \
  --use_amp

python SAMed_h/train.py \
  --dataset REFUGE \
  --root_path "$REFUGE_ROOT" \
  --img_size 512 --batch_size 16 --n_gpu 1 \
  --output "$OUT_ROOT/SAMed" \
  --vit_name vit_l --ckpt "$SAM_CKPT_VIT_L" \
  --use_amp

python SAMed_h/train.py \
  --dataset REFUGE \
  --root_path "$REFUGE_ROOT" \
  --img_size 512 --batch_size 8 --n_gpu 1 \
  --output "$OUT_ROOT/SAMed" \
  --vit_name vit_h --ckpt "$SAM_CKPT_VIT_H" \
  --use_amp
```

> 说明：vit_h 往往显存吃紧，上面把 batch_size 设得更保守；你可按实际显存调整。

---

## 3. 训练方案 B：Medical-SAM-Adapter（Adapter/LoRA）

入口脚本：`Medical-SAM-Adapter-main/train.py`

### 3.1 关键参数（强相关）

- `-net sam`：使用 SAM 主干
- `-encoder vit_b|vit_l|vit_h`：选择 ViT 版本（见 `Medical-SAM-Adapter-main/utils.py`）
- `-sam_ckpt <pth>`：对应的 SAM 权重
- `-dataset Synapse|isic|REFUGE`
- `-mod` 决定训练策略（见 `Medical-SAM-Adapter-main/function.py`）：
  - `-mod sam_adpt`：只训练 Adapter（推荐先用它跑通）
  - `-mod sam_lora`：只训练 LoRA
  - 其他值会更接近全量训练（不推荐一开始就开）

### 3.2 与 SAMed 对齐（排列顺序 + 目录名）

- **跑批顺序**与 `scripts/train_samed_all.sh` 相同：**Synapse → isic → REFUGE**，每组内 **vit_b → vit_l → vit_h**。
- 使用 **`-out_root`** 时，子目录名与 `SAMed_h/train.py` 的 `snapshot_path` 规则一致：  
  `{Dataset}_{image_size}_pretrain_{encoder}[_NNk][_epoN]_bs{B}[_lrX][_sY]`  
  其中 `_epo` 默认取 `Medical-SAM-Adapter-main/conf/global_settings.py` 的 `EPOCH`（与 SAMed 默认 `max_epochs=200` 可能不同）；`_lr`、`_s` 与当前 `-lr`、`-seed` 一致（SAMed 默认 seed=1234 无 `_s` 后缀，MSA 默认 seed=24 会带 `_s24`，需要完全一致时可加 `-seed 1234`）。
- 目录布局：`Model/`（`best_dice_checkpoint.pth`）、`Log/`、`Samples/`、**TensorBoard：`log/`**（对齐 SAMed 的 `log` 子目录）、以及 `checkpoint/sam/...`。

### 3.3 Synapse：3×vit（`-out_root`，batch 与 SAMed 一致 12/12/6）

```bash
OUT_M="$OUT_ROOT/MedicalSAMAdapter"
python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset Synapse -out_root "$OUT_M" \
  -encoder vit_b -sam_ckpt "$SAM_CKPT_VIT_B" \
  -data_path "$SYNAPSE_TRAIN_NPZ" -list_dir "$SYNAPSE_LIST_DIR" \
  -image_size 512 -out_size 512 -b 12 -gpu_device 0 -exp_name x

python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset Synapse -out_root "$OUT_M" \
  -encoder vit_l -sam_ckpt "$SAM_CKPT_VIT_L" \
  -data_path "$SYNAPSE_TRAIN_NPZ" -list_dir "$SYNAPSE_LIST_DIR" \
  -image_size 512 -out_size 512 -b 12 -gpu_device 0 -exp_name x

python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset Synapse -out_root "$OUT_M" \
  -encoder vit_h -sam_ckpt "$SAM_CKPT_VIT_H" \
  -data_path "$SYNAPSE_TRAIN_NPZ" -list_dir "$SYNAPSE_LIST_DIR" \
  -image_size 512 -out_size 512 -b 6 -gpu_device 0 -exp_name x
```

> 需要 LoRA 时把 `-mod sam_adpt` 改成 `-mod sam_lora`。设置了 `-out_root` 时 `-exp_name` 只影响日志里的字符串，**实验文件夹名由上面规则自动生成**。

### 3.4 ISIC：3×vit（512，batch 24/16/8）

```bash
python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset isic -out_root "$OUT_M" \
  -encoder vit_b -sam_ckpt "$SAM_CKPT_VIT_B" -data_path "$ISIC_ROOT" \
  -image_size 512 -out_size 512 -b 24 -gpu_device 0 -exp_name x

python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset isic -out_root "$OUT_M" \
  -encoder vit_l -sam_ckpt "$SAM_CKPT_VIT_L" -data_path "$ISIC_ROOT" \
  -image_size 512 -out_size 512 -b 16 -gpu_device 0 -exp_name x

python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset isic -out_root "$OUT_M" \
  -encoder vit_h -sam_ckpt "$SAM_CKPT_VIT_H" -data_path "$ISIC_ROOT" \
  -image_size 512 -out_size 512 -b 8 -gpu_device 0 -exp_name x
```

### 3.5 REFUGE：3×vit（512，batch 24/16/8 + `-multimask_output 2`）

```bash
python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset REFUGE -out_root "$OUT_M" \
  -encoder vit_b -sam_ckpt "$SAM_CKPT_VIT_B" -data_path "$REFUGE_ROOT" \
  -image_size 512 -out_size 512 -b 24 -gpu_device 0 -multimask_output 2 -exp_name x

python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset REFUGE -out_root "$OUT_M" \
  -encoder vit_l -sam_ckpt "$SAM_CKPT_VIT_L" -data_path "$REFUGE_ROOT" \
  -image_size 512 -out_size 512 -b 16 -gpu_device 0 -multimask_output 2 -exp_name x

python Medical-SAM-Adapter-main/train.py -net sam -mod sam_adpt -dataset REFUGE -out_root "$OUT_M" \
  -encoder vit_h -sam_ckpt "$SAM_CKPT_VIT_H" -data_path "$REFUGE_ROOT" \
  -image_size 512 -out_size 512 -b 8 -gpu_device 0 -multimask_output 2 -exp_name x
```

---

## 4. 输出与检查点在哪里？

### 4.1 SAMed

`--output "$OUT_ROOT/SAMed"` 下会创建一个实验目录（含 vit_name 等后缀），每 20 epoch 保存一次 LoRA 权重：`epoch_*.pth`。

### 4.2 Medical-SAM-Adapter

- **未设置 `-out_root`**：仍在 `logs/<exp_name>_timestamp/Model/` 下保存 best 权重。
- **设置 `-out_root`**：在  
  `<out_root>/<与SAMed同规则的目录名>/Model/best_dice_checkpoint.pth`，TensorBoard 在同级 `log/`。

---

## 5. 一键化脚本（可选）

你也可以直接用（脚本已内置 `checkpoints/` 路径，在仓库根目录执行即可）：

- `./scripts/train_samed_all.sh`
- `./scripts/train_msa_all.sh`
- `./scripts/check_data_layout.sh`

