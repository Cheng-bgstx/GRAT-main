# CODD 最终实验方案 (Full Matrix Edition)

---

## 模型矩阵现状

### 18个目标checkpoint

| Dataset | Model | vit_b | vit_l | vit_h |
|---------|-------|-------|-------|-------|
| ISIC | SAMed | ✅ Dice=0.900 | ⏳ 训练中 (ep91/300) | ⏳ 训练中 (ep119/300, Dice=0.923) |
| ISIC | MSA | ✅ Dice=0.913 | ✅ Dice=0.926 | ✅ Dice=0.922 |
| REFUGE | SAMed | ✅ Dice=0.896 | ✅ Dice=0.913 | ✅ Dice=0.913 |
| REFUGE | MSA | ✅ Dice=0.877 | ✅ Dice=0.891 | ✅ Dice=0.899 |
| Synapse | SAMed | ✅ Dice=0.816 | ⏳ 训练中 (ep79/300, Dice=0.721) | ✅ Dice=0.843 |
| Synapse | MSA | ⏳ 重训中 (s42) | ⏳ 重训中 (s42) | ❌ 待GPU空闲后重训 |

### Source models (SAM 预训练权重)

| ViT | Checkpoint |
|-----|-----------|
| vit_b | `checkpoints/sam_vit_b_01ec64.pth` |
| vit_l | `checkpoints/sam_vit_l_0b3195.pth` |
| vit_h | `checkpoints/sam_vit_h_4b8939.pth` |

---

## 实验分层策略

三个 ViT 架构有 **不同的实验深度**：

| 层级 | ViT | 实验覆盖 |
|------|-----|---------|
| **核心层** | vit_b | 全部理论验证 + 全部消融 + 主实验 |
| **架构验证层** | vit_l, vit_h | 主实验 + 关键消融 (phase carrier + amplitude) |
| **补充验证层** | vit_l, vit_h | 理论验证单点检查 + 防御鲁棒性抽检 |

**理由**：
- vit_b 是 SAM 的标准 backbone，与已有 paper 可比
- vit_l/vit_h 用于证明 CODD 的架构无关性 (scalability)
- 全矩阵跑所有实验需要 18×N 组，计算资源不可行
- 分层设计既保证核心结论可靠性，又展示跨架构泛化

---

## 一、理论验证实验 (Theory Validation)

> **统一 backbone**: 理论验证全部在 vit_b 上完成 (主文)。vit_l/vit_h 在 Synapse 上做单点验证 (appendix)。
> **跨域**: Synapse (CT 3D) + ISIC (皮肤镜 2D) + REFUGE (眼底 2D)
> **跨架构**: SAMed (LoRA end-to-end) + MSA (Adapter interactive)

### 实验 1: Layer-wise Radial Dominance

**目的**：验证 SAM → target model 的模型漂移在所有 ViT 层上满足径向能量 > 切向能量。

**模型矩阵**：

| Dataset | SAMed | MSA |
|---------|-------|-----|
| Synapse | vit_b | vit_b |
| ISIC | vit_b | vit_b |
| REFUGE | vit_b | vit_b |

(Appdx: Synapse vit_l, vit_h 两张图)

**样本数**：每条件 20 samples

**主文呈现**：

**Fig 1: Layer-wise radial ratio** (3×2 子图网格)

| | Synapse | ISIC | REFUGE |
|---|---------|------|--------|
| SAMed | (a) | (b) | (c) |
| MSA | (d) | (e) | (f) |

- 横轴: ViT layer (1-12 for vit_b)
- 纵轴: Radial ratio R = E_r / (E_r + E_t)
- 虚线 R=0.5
- 阴影 ±1 std
- **6 条线全在 0.5 以上** → CODD 理论跨域跨架构成立

**Table 1: Mean radial ratio summary**

| Dataset | Target | Mean R ± std | Layers R>0.5 | Min R (layer) |
|---------|--------|-------------|-------------|---------------|
| Synapse | SAMed | | | |
| Synapse | MSA | | | |
| ISIC | SAMed | | | |
| ISIC | MSA | | | |
| REFUGE | SAMed | | | |
| REFUGE | MSA | | | |
| _Synapse | SAMed vit_l (appdx) | | | |
| _Synapse | SAMed vit_h (appdx) | | | |

### 实验 2: Carrier Null Test

**目的**：证明源 SAM 相位 carrier 的独特性。

**3种 carrier**：
- Source carrier: `e^{j·Φ_s}` (源 SAM 相位)
- Shuffled carrier: `e^{j·shuffle(Φ_s)}` (空间打乱)
- Random carrier: `e^{j·rand(0,2π)}` (均匀随机)

**模型**：同实验 1 的 6 个 vit_b target models

**主文呈现**：

**Fig 2: Carrier null test** (分组柱状图，6 组 × 3 根柱子)

| Group | Source | Shuffled | Random |
|-------|--------|----------|--------|
| Synapse/SAMed | | | |
| Synapse/MSA | | | |
| ISIC/SAMed | | | |
| ISIC/MSA | | | |
| REFUGE/SAMed | | | |
| REFUGE/MSA | | | |

**预期**：所有组 source >> shuffled ≈ random ≈ 0.5

### 实验 3: Theory-to-Attack Consistency

**目的**：radial ratio vs attack effect 相关性。

**现有数据**：Synapse SAMed vit_b (5 conditions × n=24)

**需补充**：ISIC SAMed vit_b + REFUGE SAMed vit_b（各 5 conditions × 8-12 cases）

**主文呈现**：

**Fig 3: Radial ratio vs ΔDice** (3 子图散点图，对应 3 数据集)

---

## 二、主攻击实验 (Main Experiments)

> ⚠️ 用户自行设计执行。以下为建议框架。

### 完整受害者矩阵

| Dataset | SAMed vit_b | SAMed vit_l | SAMed vit_h | MSA vit_b | MSA vit_l | MSA vit_h |
|---------|------------|------------|------------|----------|----------|----------|
| Synapse | ✅ 核心 | ✅ 架构 | ✅ 架构 | ✅ 跨架构核心 | ✅ 架构 | ✅ 架构 |
| ISIC | ✅ 核心 | ✅ 架构 | ✅ 架构 | ✅ 跨架构核心 | ✅ 架构 | ✅ 架构 |
| REFUGE | ✅ 核心 | ✅ 架构 | ✅ 架构 | ✅ 跨架构核心 | ✅ 架构 | ✅ 架构 |

### Baseline 对比

| Method | Type |
|--------|------|
| PGD | White-box baseline |
| MI-FGSM | Momentum-based |
| NI-FGSM | Nesterov accelerated |
| DI-FGSM | Input diversity |
| TI-FGSM | Translation invariant |
| SI-FGSM | Scale invariant |
| **SE-CGR (ours)** | Frequency-domain carrier-guided |
| **SE-CGR + SRS (ours)** | + Shared radial subspace warm-start |

### 建议主表结构

**Table 2: Main results — Synapse (8-class)**

| Method | SAMed vit_b | SAMed vit_l | SAMed vit_h | MSA vit_b | MSA vit_l | MSA vit_h |
|--------|------------|------------|------------|----------|----------|----------|
| Clean (Dice) | | | | | | |
| PGD | | | | | | |
| MI-FGSM | | | | | | |
| ... | | | | | | |
| SE-CGR | | | | | | |
| SE-CGR + SRS | | | | | | |

**Table 3: Main results — ISIC (binary)** (同结构)

**Table 4: Main results — REFUGE (3-class)** (同结构)

> 每表底部加一行 "Rank (avg across targets)" 汇总方法排名。

### 跨架构/跨域一致性分析

**Fig 4: Cross-architecture consistency** (散点图)
- 横轴: ΔDice on SAMed
- 纵轴: ΔDice on MSA
- 每点一个 (dataset, method) 组合
- 预期: r > 0.8, 说明攻击在不同架构间一致迁移

---

## 三、消融实验 (Ablation Studies)

### 实验 4: Phase Carrier 消融

**目的**：验证载波相位选择的因果效应。

**5 条件**：

| 条件 | 探针 | 载波 |
|------|------|------|
| SE-CGR | `IFFT(Σ·e^{j·Φ_s})` | Source phase |
| phase-shuffled | `IFFT(Σ·e^{j·shuffle(Φ_s)})` | Shuffled source phase |
| tangential | `IFFT(Σ·j·e^{j·Φ_s})` | Orthogonal to source |
| random-phase | `IFFT(Σ·e^{j·rand})` | Uniform random |
| no-probe | None | None |

**实验矩阵**：

| Dataset | SAMed vit_b | SAMed vit_l | SAMed vit_h | MSA vit_b |
|---------|------------|------------|------------|----------|
| Synapse | ✅ 24 cases (已有, 整合) | ✅ 12 cases | ✅ 12 cases | ✅ 12 cases |
| ISIC | ✅ 12 cases | ✅ 8 cases | ✅ 8 cases | ✅ 8 cases |
| REFUGE | ✅ 12 cases | ✅ 8 cases | ✅ 8 cases | ✅ 8 cases |

> vit_l/vit_h 仅在 SAMed 上做，MSA 仅在 vit_b 上做 — 不撑爆计算但覆盖关键维度

**主文呈现**：

**Table 5: Phase carrier ablation — Synapse** (最全, n=24 for vit_b)

| Condition | SAMed v_b ΔDice | SAMed v_l ΔDice | SAMed v_h ΔDice | MSA v_b ΔDice |
|-----------|----------------|----------------|----------------|--------------|
| SE-CGR | | | | |
| phase-shuffled | | | | |
| random-phase | | | | |
| tangential | | | | |
| no-probe | | | | |

**Table 6: Phase carrier ablation — ISIC & REFUGE** (简表, vit_b only)

| Condition | ISIC SAMed | ISIC MSA | REFUGE SAMed | REFUGE MSA |
|-----------|-----------|---------|-------------|-----------|
| SE-CGR | | | | |
| ... (all 5) | | | | |

### 实验 5: Amplitude 消融

**目的**：验证振幅谱的能量分配贡献。

**5 条件**：

| 条件 | Σ (振幅) | 载波相位 |
|------|---------|---------|
| clean spectrum | `2.0 × |G_clean|` | `e^{j·Φ_s}` |
| constant | `2.0 × 1.0` | `e^{j·Φ_s}` |
| random spectrum | `2.0 × shuffle(|G_clean|)` | `e^{j·Φ_s}` |
| low-frequency | `2.0 × |G_clean| × H_low` | `e^{j·Φ_s}` |
| high-frequency | `2.0 × |G_clean| × H_high` | `e^{j·Φ_s}` |

**实验矩阵**：

| Dataset | SAMed vit_b | SAMed vit_l | SAMed vit_h |
|---------|------------|------------|------------|
| Synapse | ✅ 12 cases | ✅ 8 cases | ✅ 8 cases |
| ISIC | ✅ 12 cases | — | — |
| REFUGE | ✅ 12 cases | — | — |

> vit_l/vit_h 仅在 Synapse 上做 — 证明跨架构振幅结论一致性

**主文呈现**：

**Table 7: Amplitude ablation**

| Condition | Syn v_b | Syn v_l | Syn v_h | ISIC v_b | REF v_b | Avg |
|-----------|--------|--------|--------|---------|--------|-----|
| clean spectrum | | | | | | |
| constant | | | | | | |
| random spectrum | | | | | | |
| low-frequency | | | | | | |
| high-frequency | | | | | | |

### 实验 6: SRS 离线蒸馏消融 (mean vs PCA)

**目的**：验证 PCA 子空间 vs mean direction。

**条件**：no warm-start, UAP, mean only, mean+PCA(rank=8,16,32)

**模型**：Synapse SAMed vit_b only（SRS 消融不需要跨架构）

**Table 8: SRS decomposition ablation**

| Condition | Mean ΔDice | Pos% | vs no warm-start (p) |
|-----------|-----------|------|----------------------|
| no warm-start | | | — |
| UAP warm-start | | | |
| mean only | | | |
| mean + PCA (rank=8) | | | |
| mean + PCA (rank=16) | | | |
| mean + PCA (rank=32) | | | |

### 实验 7: 离线代理数据规模消融

**目的**：SRS 对代理数据量的敏感度。

**条件**：10%, 25%, 50%, 75%, 100% Synapse 训练集

**模型**：Synapse SAMed vit_b

**Table 9: Proxy data scale**

| Scale | Mean ΔDice | Std |
|-------|-----------|-----|
| 10% | | |
| 25% | | |
| 50% | | |
| 75% | | |
| 100% | | |
| no warm-start | | |

### 实验 8: 离线代理数据类型消融（含自然图像）

**目的**：跨域 SRS 迁移。

**代理数据类型**：Synapse, ISIC, REFUGE, ImageNet (128 imgs), SA-1B (128 imgs)

**模型**：Synapse SAMed vit_b（测试固定，变代理数据）

**Table 10: Proxy data type**

| Proxy | Domain | Mean ΔDice | vs no warm-start (p) |
|-------|--------|-----------|----------------------|
| Synapse | CT medical | | |
| ISIC | Dermoscopy | | |
| REFUGE | Fundus | | |
| ImageNet | Natural | | |
| SA-1B | Natural (SAM-trained) | | |
| no warm-start | — | | |

### 实验 9: 攻击超参数消融 (ε / steps)

**目的**：方法在不同攻击预算下的稳健性。

**实验矩阵**：

| Dataset | SAMed vit_b | SAMed vit_l | SAMed vit_h |
|---------|------------|------------|------------|
| Synapse | ✅ 全档位 | ✅ 2档位 | ✅ 2档位 |
| ISIC | ✅ 2档位 | — | — |
| REFUGE | ✅ 2档位 | — | — |

**ε 档位**：4/255, 8/255, 16/255, 32/255
**steps 档位**：5, 10, 20, 40

**Table 11: Hyperparameter ablation — Synapse vit_b**

| Setting | SE-CGR ΔDice | no-probe ΔDice | Ratio |
|---------|-------------|---------------|-------|
| ε=4, 20st | | | |
| ε=8, 20st | | | |
| ε=16, 20st | | | |
| ε=32, 20st | | | |
| ε=16, 5st | | | |
| ε=16, 10st | | | |
| ε=16, 40st | | | |

**Fig 5 (appdx): ε / steps vs ΔDice across ViT scales** (Synapse, 3 线: v_b, v_l, v_h)

---

## 四、补充实验 (Supplementary)

### 实验 10: 防御鲁棒性

**防御方法**：JPEG compression (Q=75/50), Gaussian blur (σ=1/2), Median filter (3×3/5×5), Bit-depth reduction (4bit), Resize+pad

**实验矩阵**：

| Dataset | SAMed vit_b | MSA vit_b |
|---------|------------|----------|
| Synapse | ✅ 全防御 | ✅ 全防御 |
| ISIC | ✅ 3种代表防御 | — |
| REFUGE | ✅ 3种代表防御 | — |

**对比方法**：SE-CGR, MI-FGSM, DI-FGSM, SI-FGSM

**Table 12: Defense robustness**

| Defense | MI-FGSM | DI-FGSM | SI-FGSM | SE-CGR (ours) |
|---------|---------|---------|---------|--------------|
| None (Synapse) | | | | |
| JPEG Q=75 | | | | |
| ... | | | | |

### 实验 11: 计算开销

**目的**：各方法时间/显存对比。

**测试**：Synapse 单切片 (512×512), 20 步, 10 次取平均

**Table 13: Computational cost**

| Method | Time (s) | GPU Mem (GB) | ΔDice (Syn v_b) | ΔDice/Time |
|--------|---------|-------------|----------------|-----------|
| PGD | | | | |
| MI-FGSM | | | | |
| NI-FGSM | | | | |
| DI-FGSM | | | | |
| TI-FGSM | | | | |
| SI-FGSM | | | | |
| SE-CGR (no SRS) | | | | |
| SE-CGR + SRS | | | | |

### 实验 12: 跨架构泛化热力图

**目的**：展示 CODD 在所有 18 个模型上的攻击效果一致性。

**Fig 6: Full matrix heatmap**

| | SAMed v_b | SAMed v_l | SAMed v_h | MSA v_b | MSA v_l | MSA v_h |
|---|---|---|---|---|---|---|
| Synapse | | | | | | |
| ISIC | | | | | | |
| REFUGE | | | | | | |

每个格子 = SE-CGR 20-steps vs no-probe 的 ΔDice improvement。绿色=正迁移，红色=负迁移。

### 实验 13: 可视化

**选择**：3 数据集 × 2 case 各 (1 success + 1 failure/edge)

**每例一行**：Clean image | Adv image | Pert (×5) | Clean pred | Adv pred | GT

**Fig 7: Qualitative results** (6 行, 2 cases × 3 datasets)

---

## 五、实验总表 (按执行优先级)

| 序号 | 实验 | 覆盖面 | 主文位置 | 预计GPU时 | 优先级 |
|------|------|--------|---------|----------|--------|
| 1 | Layer-wise radial dominance | 6 targets (3ds×2m, vit_b) | Fig 1, Table 1 | 2-3h | ⭐⭐⭐ |
| 2 | Carrier null test | 6 targets (3ds×2m, vit_b) | Fig 2, Table 1 | 1-2h | ⭐⭐⭐ |
| 3 | Theory-to-attack consistency | 3 targets (vit_b), 5 conds | Fig 3 | 已有+1-2h | ⭐⭐ |
| 4 | Phase carrier ablation | 16 targets (全矩阵核) | Table 5-6 | 8-12h | ⭐⭐⭐ |
| 5 | Amplitude ablation | 7 targets (3ds v_b + Syn v_l/h) | Table 7 | 4-6h | ⭐⭐⭐ |
| 6 | SRS mean vs PCA | Syn SAMed v_b | Table 8 | 已有+2h | ⭐⭐ |
| 7 | Proxy data scale | Syn SAMed v_b | Table 9 | 3-4h | ⭐⭐ |
| 8 | Proxy data type | Syn SAMed v_b | Table 10 | 2-3h | ⭐⭐ |
| 9 | ε / steps | Syn 全ViT + ISIC/REF v_b | Table 11, Fig 5 | 3-4h | ⭐⭐ |
| 10 | Defense robustness | Syn(2m) + ISIC/REF(SAMed) | Table 12 | 3-4h | ⭐ |
| 11 | Computational cost | Syn SAMed v_b | Table 13 | <1h | ⭐ |
| 12 | Cross-arch heatmap | 18 targets (单条件) | Fig 6 | 3-4h | ⭐⭐ |
| 13 | Qualitative | 6 cases | Fig 7 | <1h | ⭐ |

**预计总GPU时**：30-45 小时（可并行的独立实验能同时跑）

---

## 六、执行顺序

```
Phase A（理论先行，2-3h）:
├─ 实验 1: Radial dominance (6 targets)
├─ 实验 2: Carrier null (6 targets)  
└─ 实验 3: Theory-attack consistency (补充ISIC/REFUGE)

Phase B（核心消融，12-18h，可并行）:
├─ 实验 4: Phase carrier ablation (全矩阵)
├─ 实验 5: Amplitude ablation
└─ 实验 12: Cross-arch heatmap (单条件快速扫)

Phase C（SRS消融，6-8h）:
├─ 实验 6: mean vs PCA
├─ 实验 7: Proxy scale
└─ 实验 8: Proxy type

Phase D（补充，6-8h）:
├─ 实验 9: ε/steps
├─ 实验 10: Defense robustness
├─ 实验 11: Computational cost
└─ 实验 13: Qualitative

Phase E（主实验，用户执行）:
└─ 18 targets × 8 methods baseline comparison
```

---

## 七、关键设计决策

### 7.1 为什么理论验证只用 vit_b？

理论验证需要提取 ViT 内部特征并做 FFT 分解。vit_b 有 12 层，vit_l 有 24 层，vit_h 有 32 层。全层 FFT 对 vit_h 的内存开销极大（单张 512×512 图像在 vit_h 深层的特征图可能 OOM）。

vit_b 是 SAM 原论文的标准 backbone，也是 most widely used variant。vit_l/vit_h 的理论验证仅需在 Synapse 上做单点确认（appendix 一张图），证明 "radial dominance holds across ViT scales"。

### 7.2 为什么 MSA 消融仅在 vit_b 上做？

MSA 的评估需要 click prompt 交互，比 SAMed 的端到端评估慢 10×+（每张图需多次 prompt 调用）。在 vit_b 上完成跨架构验证即可证明 "CODD works across fine-tuning paradigms (LoRA vs Adapter)"。

### 7.3 为什么 SRS 消融仅 Synapse？

SRS 的离线蒸馏需要代理训练集。Synapse 有最大的训练集（~36 volumes），最适合做 SRS 分析。ISIC 仅 900 张训练图，REFUGE 仅 400 张，代理数据量不足。

### 7.4 ISIC 跨域 reversal 风险

之前 ISIC 跨域实验出现过 result reversal（no-probe > SE-CGR）。Phase A 的理论实验会重新验证：
- 如果 radial dominance 在 ISIC 上不成立 → 发现理论边界，作为 limitation
- 如果 radial dominance 成立但攻击 reversal → 可能是实现问题，需 debug
- Phase B 的 phase carrier ablation 在 ISIC 上的结果给出最终答案

---

## 八、论文结构（更新）

```
1. Introduction
2. Related Work
3. Method
   3.1 CODD Theory
   3.2 Frequency-Domain SE-CGR Attack
   3.3 Shared Radial Subspace (SRS)
4. Theory Validation
   4.1 Layer-wise Radial Dominance (Fig 1, Table 1) [6 targets]
   4.2 Carrier Null Test (Fig 2) [6 targets]
   4.3 Theory-to-Attack Consistency (Fig 3) [3 datasets]
5. Experiments
   5.1 Main Results (Tables 2-4) [18 targets × 8 methods]
   5.2 Cross-Architecture Analysis (Fig 4, Fig 6)
   5.3 Phase Carrier Ablation (Tables 5-6)
   5.4 Amplitude Ablation (Table 7)
   5.5 SRS Design Analysis (Tables 8-10)
   5.6 Hyperparameter & Defense (Tables 11-12, Fig 5)
   5.7 Computational Cost (Table 13)
   5.8 Qualitative Results (Fig 7)
6. Discussion
   6.1 Cross-Architecture Generalization
   6.2 Limitations (ISIC reversal, MSA interactivity)
7. Conclusion
```
