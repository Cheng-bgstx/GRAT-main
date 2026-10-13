# CODD 理论验证计划：相位方向稳定 + 幅度响应随机

> 验证两个核心主张：微调后特征漂移的相位方向稳定（RR >> 0.5），幅度响应随机（CV 高）

---

## 一、可用的模型和数据

### 已有模型

| 模型类型 | vit_b 模型路径 | 数据集 |
|---------|---------------|--------|
| Base SAM | `checkpoints/sam_vit_b_01ec64.pth` | —（源模型） |
| SAMed Synapse | `output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth` | Synapse |
| SAMed ISIC | `output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth` | ISIC |
| SAMed REFUGE | `output_runs_correct/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_199.pth` | REFUGE |
| MSA Synapse | `output_runs_correct/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/Model/best_dice_checkpoint.pth` | Synapse |
| MSA ISIC | `output_runs_correct/MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/best_dice_checkpoint.pth` | ISIC |
| MSA REFUGE | `output_runs_correct/MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/best_dice_checkpoint.pth` | REFUGE |

### 训练数据

| 数据集 | 训练数据位置 | 数量 |
|--------|------------|------|
| Synapse | `data/Synapse_preprocess/train_npz/*.npz` | ~1256 slices（有器官的） |
| ISIC | `data/ISIC/ISBI2016_ISIC_Part1_Training_Data/` | ~900 images |
| REFUGE | `data/REFUGE-Multirater/Training-400/` | ~400 images |

---

## 二、验证实验设计

### 2.1 总览

对 **3 数据集 × 2 微调方法 = 6 个 victim**，每张训练图：

```
1. 提取 base SAM 特征 F_base
2. 提取 victim 特征 F_victim  
3. 计算 FFT: Z_base, Z_victim
4. 计算漂移: ΔZ = Z_victim - Z_base
5. 在源载波基下分解: e^{jΦ_base}
   - r = Re(ΔZ · conj(e^{jΦ_base}))  → 径向系数
   - t = Im(ΔZ · conj(e^{jΦ_base}))  → 切向系数
6. 计算:
   - RR = Σr² / (Σr² + Σt²)          → 径向比率（接近1=方向稳定）
   - CV = std(r) / mean(|r|)          → 幅度变异系数（高=幅度随机）
```

### 2.2 对比条件

| 对比 | 条件 | 验证问题 | 预期 |
|------|------|---------|------|
| C1: 径向主导 | source carrier vs random carrier | 相位方向是否稳定？ | RR_source >> RR_random ≈ 0.5 |
| C2: 跨数据集 | Synapse vs ISIC vs REFUGE | 相位方向稳定性是否跨数据集成立？ | 所有数据集 RR > 0.7 |
| C3: 跨微调方法 | SAMed vs MSA | 是否跨微调架构成立？ | 两者 RR 均 > 0.7 |
| C4: 幅度随机性 | CV_radial | 沿稳定相位方向的响应幅度是否不可预测？ | CV >> 0（显著随机） |
| C5: 逐层分析 | last 6 encoder layers | 径向主导是否在所有层成立？ | 所有层 RR > 0.7，深层更强 |

### 2.3 统计

- 每张训练图作为一个样本
- 按数据集和微调方法分组报告 mean ± std
- 对 RR_source vs RR_random 做配对 Wilcoxon 检验

---

## 三、代码文件

### 新文件：`newest/verify_codd_theory.py`

```python
#!/usr/bin/env python3
"""
CODD Theory Core Verification:
  C1: Feature drift from base SAM → fine-tuned SAM is radially dominant
      when decomposed in the source phase carrier basis.
  C2: The radial response magnitude along this stable direction is
      highly stochastic (cannot be predicted from A_clean).
      
Verification: 3 datasets × 2 fine-tuning methods × vit_b
"""
import sys, os, gc, time, json
import numpy as np
import torch

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model, load_medsa_model

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {"vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"}

# ... (full implementation)
```

### 已有文件

| 文件 | 作用 |
|------|------|
| `evaluate.py` | `load_sam_encoder()` / `load_samed_model()` / `load_medsa_model()` |
| `theory.py` | `decompose_drift()` / `DriftStats` — 可直接复用 |

---

## 四、实验配置

```text
GPU:       cuda:1 (单 GPU)
样本数:    每数据集 200~500 张训练图（有前景的）
vit:       vit_b only（基础验证）
层:        encoder 最后 6 层（逐层报告 + 平均）
batch:     单图处理（无 batch，loop 遍历）

输出:
  results/codd_theory_verification.json  — 原始数据
  results/codd_theory_summary.txt        — 汇总表格
```

## 五、预期结果格式

```
=== CODD Theory Verification ===

C1: Radial Ratio (source carrier vs random carrier)
  Dataset    Victim     N     RR_source   RR_random   p-value
  Synapse    SAMed      500   0.83±0.08   0.50±0.02   <0.001
  Synapse    MSA        500   0.86±0.07   0.51±0.02   <0.001
  ISIC       SAMed      200   0.78±0.10   0.50±0.03   <0.001
  ISIC       MSA        200   0.81±0.09   0.50±0.03   <0.001
  REFUGE     SAMed      200   0.79±0.10   0.50±0.03   <0.001
  REFUGE     MSA        200   0.82±0.09   0.50±0.03   <0.001

C2: Radial Response Amplitude CV
  Dataset    Victim     CV_radial (↑ = more random)
  Synapse    SAMed      1.23±0.15
  Synapse    MSA        1.31±0.18
  ...

C3: Per-Layer Radial Ratio (Synapse SAMed)
  Layer  RR_source  RR_random
  6      0.89       0.51
  7      0.85       0.50
  8      0.93       0.50
  9      0.93       0.51
  10     0.94       0.50
  11     0.95       0.50
```

## 六、验证成功标准

| 标准 | 条件 |
|------|------|
| C1 通过 | 所有 6 个 victim 的 RR_source > 0.7 且 RR_source >> RR_random (p<0.001) |
| C4 通过 | 所有 6 个 victim 的 CV_radial > 0.5（显著随机） |
| C5 通过 | 所有层的 RR_source > 0.7 |
