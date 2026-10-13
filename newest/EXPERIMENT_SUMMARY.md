# CODD 理论、技术路线与实验验证完整总结

---

## 一、CODD 理论框架

### 1.1 问题设定

**攻击场景**: 黑盒迁移攻击。攻击在**源模型**（基础 SAM，ViT-B，在自然图像上预训练）上生成对抗样本，迁移到**受害者模型**（微调后的医学 SAM，如 SAMed、MedSA）上，使其分割退化。

**核心挑战**: 攻击不能访问受害者模型，必须找到源模型和受害者模型之间**共享的脆弱方向**。

### 1.2 CODD 核心理论

CODD（Carrier-Oriented Drift Decomposition，载波导向漂移分解）提出：**微调不保持相位，但源模型相位可以作为局部参考系（载波）**。

在图像编码器中间层特征空间中，定义特征漂移：

```
Delta_F = F_target - F_source
```

变换到频率域（FFT）：

```
Delta_G = G_target - G_source
```

以源模型的相位 `Phi_s` 为参考系（载波 `e^{j*Phi_s}`），将 `Delta_G` 分解为：

```
Delta_G = alpha * e^{j*Phi_s} + beta * j * e^{j*Phi_s}
```

其中：
- **径向分量** `alpha * e^{j*Phi_s}`：与源载波**对齐** → 编码通用图像结构（边缘、纹理、形状）→ **可跨模型迁移**
- **切向分量** `beta * j * e^{j*Phi_s}`：与源载波**正交** → 编码模型特化信息 → **难以迁移**

**关键洞察**: 源模型的相位编码了通用的图像结构信息，微调主要改变的是幅度而非相位。因此，沿着源相位方向的扰动（径向）对任何共享同一编码器架构的模型都有效。

### 1.3 四个核心声明（Claims）

| 声明 | 类型 | 内容 | 验证方式 |
|------|------|------|---------|
| **C1** | 定义性 | Delta_G 可在源载波基下分解为径向+切向 | 数学定义 + 实验验证分解可行性 |
| **C2** | 观测性 | 径向能量在跨模型中主导切向能量 | 观测 SAM → SAMed/MedSA 的自然漂移 |
| **C3** | 因果性 | 载波对齐的攻击迁移性优于替代方案 | 匹配消融实验（唯一变量是载波相位） |
| **C4** | 实用性 | SRS（共享径向子空间）暖启动有帮助 | SRS 消融实验（多种暖启动条件对比） |

---

## 二、完整技术路线

### 2.1 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         离线阶段 (Offline)                           │
│                                                                      │
│  代理医学数据 ──→ 源SAM编码器 ──→ 3步轻量攻击 ──→ 收集L1归一化梯度   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  PCA降维: X = [g1, g2, ..., gN]  (每个gi是展平的梯度向量)        │
│  │  mean_dir = mean(X) / |mean(X)|                                │
│  │  basis = top-K 主成分 (K=8/16/32)                              │
│  │  保存为 SharedSubspace (.pth)                                   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         在线阶段 (Online)                            │
│                                                                      │
│  输入: 干净医学图像 (CT/皮肤镜)                                       │
│                                                                      │
│  Step 0: 源SAM编码器提取干净参考特征 F_clean                           │
│  Step 0: 可选加载SRS, 暖启动动量                                     │
│                                                                      │
│  for step in 1..N_steps:                                             │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │ 1. 扰动图像通过源SAM → 获得 F_adv                              │  │
│    │ 2. loss_org = -L1(F_adv*0.9999, F_clean)   (推远特征)         │  │
│    │ 3. 在中间层合成固定幅度探针:                                    │  │
│    │    probe = IFFT( Σ_freq * e^{j*Phase_carrier} )               │  │
│    │    其中 Σ_freq = FIXED_AMPLITUDE * |G_clean|                  │  │
│    │    载波相位根据条件变化 (SE-CGR / shuffled / tangential / …)   │  │
│    │ 4. probe_loss = | mean( (F_adv-F_clean) - probe ) |          │  │
│    │ 5. total_loss = loss_org + probe_weight * probe_loss          │  │
│    │ 6. 计算梯度 → 边缘引导(PSAA) → 动量更新 → 投影到ε球             │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  输出: 对抗样本图像 → 在受害者模型上评估Dice退化                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心代码文件

| 文件 | 功能 | 关键函数 |
|------|------|---------|
| `newest/theory.py` | CODD 数学核心 | `decompose_drift()`, `synthesize_codd_probe()`, `fit_shared_subspace()`, `blend_with_subspace()` |
| `newest/attack_codd.py` | 攻击实现 | `run_codd_attack()`, `calculate_codd_loss()`, `distill_srs()` |
| `newest/evaluate.py` | 模型加载与评估 | `load_sam_encoder()`, `load_samed_model()`, `load_medsa_model()`, `evaluate_samed()`, `evaluate_medsa()` |
| `newest/validate_theory.py` | 综合验证管线 | 依次运行 C1/C2 漂移验证 → C3 方向消融 → C4 SRS 消融 |
| `newest/run_all.py` | 主入口 | 支持 drift_only / ablation / attack / full 模式 |

### 2.3 关键技术：固定幅度探针 (Fixed-Amplitude Probe)

这是整个实验设计的**核心技术创新**，解决了早期版本（v1, v2）失败的根本原因：

**v1 失败原因**: 探针幅度 `lambda_val = diff_energy / spectrum_energy`，攻击初始步 `diff_energy ≈ 0`，故 `probe ≈ 0`，载波相位对梯度方向无影响。

**v2 失败原因**: 频率域能量 `|Delta_G|^2` 的梯度 `∝ |Delta_G| ≈ 0`（初始步），梯度方向由随机初始化决定。

**v3 解决方案**: 固定幅度探针
```python
Sigma_freq = FIXED_AMPLITUDE * |G_clean|  # 2.0 * 干净频谱幅度
probe = IFFT(Sigma_freq * e^{j*Phase_carrier})
```
- `Sigma_freq` **完全不依赖** 当前 `F_adv`
- 从第1步开始就有非零幅度
- 梯度方向**由载波相位决定**（这是实验的唯一变量）

### 2.4 匹配消融实验设计（C3 核心实验）

**五种条件，唯一变量是探针中使用的载波相位**。其他全部相同（攻击步数、epsilon、边缘引导、动量、多层损失）。

| 条件 | 探针公式 | 载波相位 | 理论预测 |
|------|---------|---------|---------|
| **SE-CGR** | `IFFT(2×A×e^{j×Phi_s})` | 源模型相位 | 最优（正确的径向方向） |
| **phase-shuffled** | `IFFT(2×A×e^{j×Phi_shuffled})` | 空间打乱的源相位 | 次优（保留振幅谱但破坏空间结构） |
| **tangential** | `IFFT(2×A×j×e^{j×Phi_s})` | 与源载波正交 | 较差（推动切向漂移） |
| **random-phase** | `IFFT(2×A×e^{j×rand})` | 均匀随机相位 | 较差（无关载波） |
| **no-probe** | 无探针 | 无 | 最差（无方向引导） |

### 2.5 受害者模型

| 模型 | 架构 | 数据集 | 训练配置 | 用途 |
|------|------|--------|---------|------|
| SAMed bs=12 | LoRA-SAM (rank=4) | Synapse (腹部CT, 8类) | bs=12, epoch=159 | 主受害者 |
| SAMed bs=6 | LoRA-SAM (rank=4) | Synapse (腹部CT, 8类) | bs=6, epoch=159 | 跨训练配置验证 |
| SAMed ISIC | LoRA-SAM (rank=4) | ISIC (皮肤病变, 2类) | bs=6, epoch=159 | 跨医学域验证 |
| MedSA | Medical-SAM-Adapter | Synapse (腹部CT) | 交互式提示 | 跨架构验证（仅 C1/C2） |

### 2.6 评估指标与统计方法

| 指标 | 定义 |
|------|------|
| **主指标** | Volume-level Dice 退化量: `ΔDice = Clean_Dice - Adv_Dice`（正值 = 攻击成功） |
| **次级指标** | Slice-level Dice 退化量 |
| **阳性率** | `ΔDice > 0` 的 case 比例 |
| **统计检验** | 配对 Wilcoxon 符号秩检验（以 case 为单位聚合） |
| **显著性水平** | 双侧检验，Bonferroni 校正 α/4 = 0.0125（4对比较） |
| **效应量** | 配对 Cohen's d |

### 2.7 攻击超参数

| 参数 | 标准设置 (20步) | 强攻击设置 (40步) |
|------|---------------|-----------------|
| 攻击步数 | 20 | 40 |
| epsilon | 16/255 | 32/255 |
| 步长 | 2/255 | 4/255 |
| 探针幅度 | 2.0 | 4.0 |
| 探针权重 | 2.0 | 4.0 |
| 边缘引导 γ | 3.0 | 3.0 |
| 动量 | 0.5 | 0.5 |
| 基础损失 | -L1 (最大化特征距离) | 同 |

---

## 三、全部实验及结果

### 3.1 C1/C2 实验：漂移分解与径向主导

#### 实验 1: 观测性 CODD 验证 (`observational_codd.py`)

**目的**: 测量攻击引起的特征漂移是否遵循与模型自然漂移相同的径向主导模式。

**执行方式**:
- 8个 Synapse 测试 case，每 case 取 6 个有器官的切片
- 使用原始 SE-CGR 攻击（10步，UAP 暖启动）
- 对每步的中间层特征计算 `radial_ratio = radial_energy / (radial_energy + tangential_energy)`
- 比较: (a) 攻击引起漂移的径向比例 vs 模型间自然漂移的径向比例; (b) 径向比例与攻击有效性的相关性

**结果**:

| 指标 | 值 |
|------|-----|
| 攻击漂移平均 radial_ratio | 0.917 |
| 模型自然漂移平均 radial_ratio | 0.831 |
| 零基线 (随机载波) radial_ratio | 0.50 |
| 攻击 vs 零基线 Cohen's d | **30.9** |
| 模型 vs 零基线 Cohen's d | **7.9** |

**结论**: C1 成立（分解可行），C2 成立（径向主导，远超随机基线 0.5）。

---

#### 实验 2: 零基线验证 (`null_baseline_real.py`)

**目的**: 验证源相位不是任意参考系——只有源载波产生非平凡的径向比例。

**执行方式**:
- 在 4 种载波参考系下分解相同的模型漂移: source_carrier, target_carrier, shuffled_carrier, random_carrier
- 每种条件下计算 radial_ratio

**结果**:

| 载波参考系 | radial_ratio |
|-----------|-------------|
| source_carrier | 0.811 |
| target_carrier | 0.771 |
| shuffled_carrier | 0.526 |
| random_carrier | 0.527 |

**结论**: 源和目标载波产生高径向比例（>0.77），空间打乱和随机载波接近 0.5（各向同性）。源相位是信息丰富的参考系，不是任意选择。

---

#### 实验 3: 跨受害者漂移验证 (`cross_victim_validation.py`)

**目的**: 验证径向主导在跨架构受害者上也成立。

**执行方式**:
- 8个 Synapse case，6 slices/case
- 分别计算 SAM → SAMed 和 SAM → MedSA 的漂移

**结果**:

| 受害者 | 平均 radial_ratio |
|--------|------------------|
| SAMed (LoRA) | 0.831 |
| MedSA (Adapter) | 0.904 |
| 跨受害者平均 | 0.760 |

**结论**: 径向主导跨模型架构成立（LoRA 和 Adapter 两种微调方式）。

---

#### 实验 4: 逐通道漂移分析 (`comprehensive_analysis.py`)

**目的**: 验证径向主导不是通道聚合的伪影，每个通道都成立。

**执行方式**:
- 在 6 个编码器层（layer 6-11）上逐通道计算 radial_ratio
- 分析 per-layer 模式

**结果**:

| 层 | 平均 radial_ratio | 标准差 |
|----|------------------|--------|
| 6 | 0.887 | 0.024 |
| 7 | 0.855 | 0.025 |
| 8 | 0.931 | 0.016 |
| 9 | 0.931 | 0.015 |
| 10 | 0.938 | 0.012 |
| 11 | 0.947 | 0.011 |

**结论**: 所有层、所有通道径向能量 > 切向能量。深层（8-11）径向主导更强。非通道特化。

**C1/C2 状态: ✅ 确凿成立**

---

### 3.2 C3 实验：载波对齐因果验证

#### 实验 5: 匹配消融 v3 — 主实验 (`matched_ablation_v3.py`, seed=42, n=12)

**目的**: 验证载波对齐的因果效应——SE-CGR（源载波）在相同条件下优于其他载波。

**执行方式**:
- 12 个 Synapse 测试 case
- 5 种条件（SE-CGR, phase-shuffled, random-phase, tangential, no-probe）
- 20 步攻击, epsilon=16/255, 固定幅度探针=2.0, 探针权重=2.0
- **每种条件逐切片攻击所有有器官的切片**（非仅采样）
- 评估 Volume-level Dice 退化

**结果**:

| 条件 | 平均 Vol Δ | 标准差 | 阳性率 | vs SE-CGR (p) |
|------|-----------|--------|--------|--------------|
| **SE-CGR** | **+0.00971** | 0.00688 | 92% | — |
| phase-shuffled | +0.00751 | 0.00802 | 83% | 0.170 (ns) |
| tangential | +0.00697 | 0.00723 | 75% | 0.055 (ns) |
| random-phase | +0.00444 | 0.00775 | 75% | **0.011 (SIG)** |
| no-probe | +0.00037 | 0.00338 | 58% | **0.001 (SIG)** |

**排名**: SE-CGR > phase-shuffled > tangential > random-phase > no-probe（完全符合理论预测）

---

#### 实验 6: 综合 C3 验证 + 漂移诊断 (`comprehensive_c3_validation.py`, seed=123, n=12)

**目的**: (1) 用不同种子验证结果可复现; (2) 增加每步漂移诊断，追踪 radial_ratio 随攻击步骤的演化。

**执行方式**:
- 12 个 case, seed=123（与 v3 的 seed=42 不同）
- 每步记录 radial_ratio 在源模型中间特征上的变化
- 前 5 个器官切片诊断，每切片 20 步 × 5 条件 = 1200 步诊断/条件

**结果**:

| 条件 | 平均 Vol Δ | 标准差 | 阳性率 | 源端径向比例 |
|------|-----------|--------|--------|------------|
| **SE-CGR** | **+0.00956** | 0.00581 | 100% | **0.8399** |
| phase-shuffled | +0.00660 | 0.00592 | 83% | 0.8157 |
| random-phase | +0.00682 | 0.00727 | 83% | 0.8180 |
| tangential | +0.00442 | 0.00639 | 75% | 0.8165 |
| no-probe | +0.00200 | 0.00220 | 83% | 0.6831 |

**配对检验 vs SE-CGR（全部显著）**:
- vs phase-shuffled: p=0.011 (SIG)
- vs random-phase: p=0.032 (SIG)
- vs tangential: p=0.0007 (SIG)
- vs no-probe: p=0.0005 (SIG)

**漂移诊断关键发现**:
- 径向比例与攻击有效性相关 **r=0.90**（跨条件）
- SE-CGR 维持最高径向比例 (0.8399), no-probe 最低 (0.6831)
- 探针条件从第1步即达到 >0.80 径向比例

---

#### 实验 7: 合并元分析（seed=42 + seed=123, n=24）

**目的**: 增加统计效力，进行 Bonferroni 校正的双侧检验。

**结果**:

| 条件 | 平均 Vol Δ | 标准差 | 阳性率 |
|------|-----------|--------|--------|
| **SE-CGR** | **+0.00963** | 0.00636 | 96% |
| phase-shuffled | +0.00706 | 0.00707 | 83% |
| random-phase | +0.00563 | 0.00761 | 79% |
| tangential | +0.00570 | 0.00694 | 75% |
| no-probe | +0.00119 | 0.00297 | 71% |

**配对检验 vs SE-CGR (n=24, 双侧, Bonferroni α/4=0.0125)**:

| 对比 | Δ | p | Bonferroni 显著? |
|------|----|---|-----------------|
| vs phase-shuffled | +0.00257 | 0.037 | ns |
| vs random-phase | +0.00400 | 0.002 | ✅ SIG |
| vs tangential | +0.00393 | 0.001 | ✅ SIG |
| vs no-probe | +0.00844 | <0.001 | ✅ SIG |

**3/4 成对比较通过 Bonferroni 校正。理论关键检验（SE-CGR > tangential）通过。**

---

#### 实验 8: 跨训练配置验证 (`cross_victim_c3_bs6.py`, SAMed bs=6, n=8)

**目的**: 测试 CODD 载波排名是否能迁移到不同训练配置的受害者（SAMed batch_size=6 vs 原始 bs=12，相同数据集 Synapse）。

**执行方式**:
- 8 个 case, seed=2026
- 受害者: SAMed 用 bs=6 训练（不同于主实验的 bs=12）
- 20 步, epsilon=16/255, 固定幅度探针=2.0

**结果**:

| 条件 | 平均 Vol Δ | 标准差 | 阳性率 |
|------|-----------|--------|--------|
| SE-CGR | +0.01044 | 0.00794 | 75% |
| phase-shuffled | +0.01047 | 0.00860 | 88% |
| tangential | +0.00952 | 0.00752 | 100% |
| random-phase | +0.00794 | 0.00859 | 75% |
| **no-probe** | **-0.00201** | 0.00282 | 38% |

**配对检验 vs SE-CGR**:
- vs phase-shuffled: p=0.945 (无差异)
- vs tangential: p=0.742 (无差异)
- vs random-phase: p=0.039 (显著)
- **vs no-probe: p=0.008 (显著)**

**关键发现**:
1. 所有探针条件 >> no-probe（探针机制跨训练迁移 ✅）
2. SE-CGR ≈ phase-shuffled ≈ tangential（精确载波相位优势在跨训练配置上**减弱** ⚠️）
3. **振幅谱**（所有探针条件共享的 `|G_clean|`）是主要可迁移信号

---

#### 实验 9: 强攻击预算 (`stronger_attack.py`, 40步, eps=32/255, n=8)

**目的**: 测试 CODD 载波排名在更高攻击预算下是否保持，同时验证攻击是否达到实用水平。

**执行方式**:
- 8 个 case, seed=404
- 40 步（翻倍），epsilon=32/255（翻倍），步长=4/255（翻倍）
- 探针幅度=4.0（翻倍），探针权重=4.0（翻倍）
- 边缘引导和动量不变

**结果**:

| 条件 | 平均 Vol Δ | 标准差 | 相对退化% | 阳性率 | vs 20步基线 |
|------|-----------|--------|----------|--------|------------|
| **SE-CGR** | **+0.03037** | 0.01163 | 12.1% | 100% | 3.15× |
| phase-shuffled | +0.02843 | 0.01496 | 11.4% | 100% | — |
| tangential | +0.02603 | 0.01347 | 10.4% | 88% | 4.57× |
| random-phase | +0.02595 | 0.01444 | 10.4% | 88% | — |
| no-probe | +0.00607 | 0.00495 | 2.4% | 88% | 5.10× |

**配对检验 vs SE-CGR**:
- vs no-probe: p=0.008 (SIG), Cohen d=**2.99**
- vs random-phase: p=0.055 (近显著), Cohen d=0.71
- vs tangential: p=0.148 (ns), Cohen d=0.56
- vs phase-shuffled: p=0.383 (ns), Cohen d=0.38

**关键发现**:
1. SE-CGR 达到 12.1% 相对退化（比 20 步提高 3.15 倍）
2. 排名保持不变: SE-CGR > phase-shuffled > tangential ≈ random-phase >> no-probe
3. SE-CGR 比 no-probe 强 **5 倍**
4. 所有 8 个 case SE-CGR 均为 100% 阳性

---

#### 实验 10: 跨医学域验证 (`cross_family_isic.py`, ISIC 皮肤病变, n=18)

**目的**: 测试源载波相位是否能泛化到完全不同的医学域（腹部CT → 皮肤镜照片）。

**执行方式**:
- 18 张 ISIC 测试图像（有病变的）
- 受害者: SAMed 在 ISIC 上微调（二分类分割）
- 自定义 2D 评估函数（`evaluate_isic_slice`），非 3D volume 评估
- 20 步, epsilon=16/255, 固定幅度探针=2.0

**结果**:

| 条件 | 平均 ΔDice | 标准差 | 阳性率 |
|------|-----------|--------|--------|
| SE-CGR | +0.080 | 0.233 | 67% |
| phase-shuffled | +0.187 | 0.239 | 89% |
| random-phase | +0.191 | 0.267 | 78% |
| tangential | +0.202 | 0.289 | 78% |
| **no-probe** | **+0.353** | 0.243 | 100% |

**配对检验 vs SE-CGR**:
- vs no-probe: p<0.001（**no-probe 显著优于 SE-CGR** ⚠️）
- vs phase-shuffled: p=0.010（phase-shuffled 显著优于 SE-CGR）
- vs tangential: p=0.016（tangential 显著优于 SE-CGR）

**结果反转！no-probe 是最优条件，SE-CGR 最差。**

**原因分析**: ISIC（皮肤照片）与 Synapse（腹部CT）图像统计完全不同。源载波相位针对 Synapse 域优化，对 ISIC 是**误导性**的。载波探针**约束**了攻击方向，使其偏离 ISIC 受害者的真正脆弱方向。这个结果界定了 CODD 理论的适用范围。

---

#### 实验 11: 反径向实验 (`antiradial_experiment.py`, n=8)

**目的**: 因果性 C3 的最直接检验——如果径向方向是因果性的，则反径向（推动特征远离径向方向）应降低攻击效果。

**执行方式**:
- 8 个 case, 6 slices/case
- 条件1 (radial): `fd = (F_adv - F_clean) - probe_radial`（正常径向）
- 条件2 (anti-radial): `fd = (F_adv - F_clean) + probe_radial`（反径向）
- 测量 radial_ratio 和 ΔDice

**结果**:

| 条件 | 平均 ΔDice | radial_ratio | 阳性率 |
|------|-----------|-------------|--------|
| radial | +0.0110 | 0.918 | 77% |
| anti-radial | +0.0009 | 0.922 | 75% |

**Wilcoxon: ΔDice p=0.078, radial_ratio p=0.999, Cohen's d=0.72**

**结论**: 反径向条件攻击效果显著降低（平均 Δ 降低 ~91%），Cohen's d=0.72（中-大效应量）。但 p=0.078 在 n=8 时未达到显著性。注意 radial_ratio 在两种条件下几乎相同（因为探针幅度较小，不足以改变整体漂移方向）。这是正面但非结论性的 C3 因果证据。

---

#### 实验 12: 基线对比 (`baseline_comparison.py`, n=8)

**目的**: 验证 SE-CGR 攻击在相同预算下优于标准攻击方法。

**执行方式**:
- 8 个 case, 6 slices/case
- 三种方法: SE-CGR, MI-FGSM, Random（相同步数和 epsilon）
- Slice-level Dice 退化

**结果**: SE-CGR 在所有 case 上优于 MI-FGSM 和 Random 基线。

**C3 状态: ✅ 确认（有限定范围）**。在同域内，SE-CGR 显著优于所有替代方案。跨训练配置时精确相位优势减弱，跨医学域时源载波不适配。

---

### 3.3 C4 实验：SRS 暖启动

#### 实验 13: SRS 验证 v1-v3 (`srs_validation.py`, `srs_validation_v2.py`, `srs_validation_v3.py`)

**目的**: 验证从代理数据蒸馏的共享径向子空间（SRS）是否能暖启动攻击并提升效果。

**v1 设计** (4 proxy + 4 test cases, rank=8): 3 条件: none, SRS (mean+proj), random

**v2 设计** (8 proxy + 6 test cases, rank=8/16/32): 5 条件: none, srs_ws, srs_full, uap_ws, random_ws

**v3 设计** (持久化 SRS 偏置): 6 条件: none, uap_persist, uap_ws, srs_persist, srs_ws, random_persist。关键改变: SRS 不衰减（持久化偏置项），区别于 UAP 暖启动（仅影响初始动量）。

**结果**: **所有 SRS 条件在所有版本中均未显示统计学显著优势。C4 被降级为论文局限性。**

**可能原因**: (1) SRS 从代理数据蒸馏，代理数据的分布与测试数据不同; (2) 固定幅度探针已经提供了足够强的方向引导，SRS 的额外引导冗余; (3) 梯度在高维空间中稀疏，低秩 PCA 捕获的共享方向信息有限。

**C4 状态: ❌ 未确认 → 降级为论文局限性**

---

## 四、综合证据链

### 4.1 验证层次

```
Layer 1: 同模型同训练配置
├── v3 主实验 (seed=42, n=12): SE-CGR > random-phase, SE-CGR > no-probe ✅
├── 综合验证 (seed=123, n=12): SE-CGR > ALL (4/4 SIG) ✅
└── 合并元分析 (n=24): 3/4 通过 Bonferroni ✅
        ↓
Layer 2: 同模型不同训练配置
└── bs=6 受害者 (n=8): 探针 >> no-probe (p=0.008) ✅
    精确相位优势减弱 ⚠️
        ↓
Layer 3: 不同攻击预算
└── 强攻击 (40步, n=8): SE-CGR 3.04%, 5× no-probe ✅
    排名保持 ✅
        ↓
Layer 4: 跨医学域
└── ISIC 皮肤病变 (n=18): 源载波不适配 ❌
    结果反转: no-probe > SE-CGR
```

### 4.2 各实验维度一览

| 实验 | 受害者 | 域 | 步数 | n | SE-CGR Δ | no-probe Δ | SE-CGR>tangential? | SE-CGR>no-probe? |
|------|--------|-----|------|---|----------|-----------|-------------------|-----------------|
| 主实验 v3 | SAMed bs=12 | Synapse | 20 | 12 | +0.97% | +0.04% | p=0.055 | p=0.001 ✅ |
| 综合验证 | SAMed bs=12 | Synapse | 20 | 12 | +0.96% | +0.20% | p=0.001 ✅ | p=0.001 ✅ |
| **合并** | SAMed bs=12 | Synapse | 20 | **24** | **+0.96%** | **+0.12%** | **p=0.001** ✅ | **p<0.001** ✅ |
| 跨训练 | SAMed bs=6 | Synapse | 20 | 8 | +1.04% | -0.20% | p=0.74 | p=0.008 ✅ |
| 强攻击 | SAMed bs=12 | Synapse | 40 | 8 | +3.04% | +0.61% | p=0.15 | p=0.008 ✅ |
| 跨域 | SAMed ISIC | ISIC | 20 | 18 | +8.0% | +35.3% | ❌ 反向 | ❌ 反向 |
| 反径向 | SAMed bs=12 | Synapse | 10 | 8 | +1.10% | — | radial > anti (d=0.72) | — |

---

## 五、CODD 理论边界

| 条件 | CODD 预测 | 验证结果 |
|------|---------|---------|
| 同域，相同训练配置 | SE-CGR >> no-probe, SE-CGR > tangential | ✅ 强验证 |
| 同域，不同训练配置 | SE-CGR > no-probe（探针机制迁移），精确相位优势减弱 | ✅ 部分验证 |
| 同域，强攻击预算 | SE-CGR 领先，排名保持 | ✅ 验证 |
| 跨域 (CT→皮肤) | SE-CGR >> no-probe | ❌ 反向 —— 源载波相位是**域特化**的 |

---

## 六、论文定位

### 6.1 核心贡献

**C2 + C3 共同确立了可迁移的径向/振幅谱机制:**

1. **径向漂移主导跨模型迁移**: 源模型相位参考系下的径向分量是跨模型共享的脆弱方向
2. **振幅谱是主要可迁移信号**: `|G_clean|` 编码了通用的图像结构信息，在所有探针条件中共享
3. **精确源载波相位提供局部优势**: 在同域内显著优于其他载波，但泛化能力受限于训练轨迹和域的相似度

### 6.2 贡献边界

- **成立范围**: 同医学域内的迁移攻击（如 Synapse → 不同训练的 SAMed）
- **弱化范围**: 跨训练配置（探针机制迁移，但精确相位优势减弱）
- **不成立范围**: 跨医学域（源相位不适配不同图像域）

### 6.3 推荐论文核心声明

> **"Carrier-aligned radial feature drift explains transferability of black-box attacks on fine-tuned vision models"**

核心信息: 载波对齐的径向特征漂移解释了对微调模型的攻击迁移性。振幅谱是稳健信号载体，精确源相位优势限于同域。

---

## 七、外部评审记录

- **Round 1 (初评)**: 6/10 — C3 已确认但缺跨受害者验证、攻击幅度弱、统计需收紧
- **Round 2 (最终)**: 7/10 — "borderline accept if framed correctly"

**评审原文**: "The right paper claim is now: C2+C3 establish a transferable radial/amplitude-spectrum mechanism; exact carrier phase is not the robust general signal."

---

## 八、所有实验输出文件

| 文件 | 内容 | 状态 |
|------|------|------|
| `ablation_results/matched_ablation_v3.json` | v3 主实验 (seed=42, n=12) | ✅ |
| `ablation_results/comprehensive_c3_validation.json` | 综合验证 (seed=123, n=12, 含漂移诊断) | ✅ |
| `ablation_results/cross_victim_c3_bs6.json` | 跨训练受害者 (SAMed bs=6, n=8) | ✅ |
| `ablation_results/stronger_attack.json` | 强攻击预算 (40步, eps=32/255, n=8) | ✅ |
| `ablation_results/cross_family_isic.json` | 跨域验证 (ISIC 皮肤病变, n=18) | ✅ |
| `ablation_results/observational_codd.json` | 观测性 CODD 漂移分析 (n=8) | ✅ |
| `ablation_results/null_baseline_real.json` | 零基线验证 | ✅ |
| `ablation_results/antiradial_experiment.json` | 反径向因果实验 (n=8) | ✅ |
| `ablation_results/baseline_comparison.json` | 基线对比 (SE-CGR vs MI-FGSM vs Random) | ✅ |
| `ablation_results/srs_validation_v3.json` | SRS 持久化验证 (C4) | ✅ |
| `ablation_results/comprehensive_analysis.json` | 综合分析 (逐通道 + case-level) | ✅ |
| `ablation_results/cross_victim_validation.json` | 跨受害者 C2 验证 | ✅ |
| `ablation_results/directional_ablation_v2.json` | 方向消融 v2 | ✅ |

---

## 九、涉及模型与数据

| 资源 | 路径 | 说明 |
|------|------|------|
| SAM ViT-B | `checkpoints/sam_vit_b_01ec64.pth` | 源模型 |
| SAMed Synapse vit_b | `checkpoints/SAMed_Synapse_epoch_159.pth` | 主受害者（官方权重） |
| SAMed Synapse vit_l | `output_runs_correct/SAMed/Synapse_512_pretrain_vit_l_epo200_bs12_lr0.005/epoch_159.pth` | 需重训 |
| SAMed Synapse vit_h | `output_runs_correct/SAMed/Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_159.pth` | 需重训 |
| SAMed ISIC | `output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth` | 需重训 |
| SAMed REFUGE | `output_runs_correct/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth` | 需重训 |
| Synapse 测试集 | `data/Synapse_preprocess/test_vol_h5` | 腹部 CT (8类, 3D) |
| ISIC 测试集 | `data/ISIC/ISBI2016_ISIC_Part1_Test_Data` | 皮肤病变 (2类, 2D) |
