# AutoSAM 训练命令

所有命令在 `/root/data2/fyc/fyc/GRAT-main/AutoSAM-main` 目录下执行。
结果输出到 `./output/gpu*/` 和 `./results/gpu*/`。

## 通用配置

| 参数 | 值 | 说明 |
|------|------|------|
| `--sam_size` | 512 | 对齐 SAMed/MSA |
| `--learning_rate` | 0.005 | Adam |
| `--WD` | 1e-4 | weight decay |
| `--nW` | 4 | DataLoader workers |
| `--nW_eval` | 2 | eval workers |
| `--epoches` | 200 | 默认 200 epoch |
| `--output_dir` | ./output | 结果目录 |

## 环境

```bash
conda activate adattack   # 或 sam3
```

## SAM Checkpoint

| vit | 路径 |
|-----|------|
| vit_b | `/root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth` |
| vit_l | `/root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_l_0b3195.pth` |
| vit_h | `/root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_h_4b8939.pth` |

---

## ISIC 2016 — 皮肤病变 (binary, BCE + Dice)

```bash
cd /root/data2/fyc/fyc/GRAT-main/AutoSAM-main

# vit_b
CUDA_VISIBLE_DEVICES=0 python train.py --task isic --sam_type vit_b \
    --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth \
    --sam_size 512 --Batch_size 8 --learning_rate 0.005 \
    --nW 4 --nW_eval 2 --WD 1e-4

# vit_l
CUDA_VISIBLE_DEVICES=0 python train.py --task isic --sam_type vit_l \
    --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_l_0b3195.pth \
    --sam_size 512 --Batch_size 12 --learning_rate 0.005 \
    --nW 4 --nW_eval 2 --WD 1e-4

# vit_h
CUDA_VISIBLE_DEVICES=0 python train.py --task isic --sam_type vit_h \
    --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_h_4b8939.pth \
    --sam_size 512 --Batch_size 8 --learning_rate 0.005 \
    --nW 4 --nW_eval 2 --WD 1e-4
```

---

## REFUGE — 眼底视盘+视杯 (2 通道, BCE + Dice ×2)

```bash
cd /root/data2/fyc/fyc/GRAT-main/AutoSAM-main

# vit_b
CUDA_VISIBLE_DEVICES=0 python train.py --task refuge --sam_type vit_b \
    --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth \
    --sam_size 512 --Batch_size 6 --learning_rate 0.005 \
    --nW 4 --nW_eval 2 --WD 1e-4

# vit_l
CUDA_VISIBLE_DEVICES=2 python train.py --task refuge --sam_type vit_l \
    --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_l_0b3195.pth \
    --sam_size 512 --Batch_size 12 --learning_rate 0.005 \
    --nW 4 --nW_eval 2 --WD 1e-4

# vit_h
CUDA_VISIBLE_DEVICES=2 python train.py --task refuge --sam_type vit_h \
    --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_h_4b8939.pth \
    --sam_size 512 --Batch_size 8 --learning_rate 0.005 \
    --nW 4 --nW_eval 2 --WD 1e-4
```

---

## Synapse — 腹部CT多器官 (9 类, CrossEntropy + Dice)

```bash
cd /root/data2/fyc/fyc/GRAT-main/AutoSAM-main

# vit_b
CUDA_VISIBLE_DEVICES=2 python train.py --task synapse --sam_type vit_b \
    --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth \
    --sam_size 512 --Batch_size 6 --learning_rate 0.005 \
    --nW 4 --nW_eval 2 --WD 1e-4

# vit_l
CUDA_VISIBLE_DEVICES=2 python train.py --task synapse --sam_type vit_l \
    --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_l_0b3195.pth \
    --sam_size 512 --Batch_size 12 --learning_rate 0.005 \
    --nW 4 --nW_eval 2 --WD 1e-4

# vit_h
CUDA_VISIBLE_DEVICES=2 python train.py --task synapse --sam_type vit_h \
    --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_h_4b8939.pth \
    --sam_size 512 --Batch_size 6 --learning_rate 0.005 \
    --nW 4 --nW_eval 2 --WD 1e-4
```

---

## 一键并行

```bash
cd /root/data2/fyc/fyc/GRAT-main/AutoSAM-main

# 三个数据集 × vit_b
CUDA_VISIBLE_DEVICES=0 python train.py --task isic --sam_type vit_b --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth --sam_size 512 --Batch_size 8 --lr 0.005 --nW 4 --nW_eval 2 --WD 1e-4 &
CUDA_VISIBLE_DEVICES=1 python train.py --task refuge --sam_type vit_b --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth --sam_size 512 --Batch_size 6 --lr 0.005 --nW 4 --nW_eval 2 --WD 1e-4 &
CUDA_VISIBLE_DEVICES=2 python train.py --task synapse --sam_type vit_b --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth --sam_size 512 --Batch_size 6 --lr 0.005 --nW 4 --nW_eval 2 --WD 1e-4 &
wait
```

---

## 架构说明

| 数据集 | num_classes | 输出通道 | Loss | 标签格式 |
|--------|-------------|---------|------|---------|
| isic | 1 | 1 | BCEWithLogitsLoss + Dice | 二值 mask |
| refuge | 2 | 2 | BCEWithLogitsLoss + Dice(per channel) | 2ch [cup, disc] |
| synapse | 9 | 9 | CrossEntropyLoss + MultiDice(softmax, 8 organs) | class indices 0-8 |
