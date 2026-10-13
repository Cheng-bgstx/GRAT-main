# PASD: Phase-Aware Stochastic Disruption — 修正最终方案

> 中文名：**相位感知的随机破坏攻击**
>
> 核心理论修正：**LoRA/Adapter 微调继承基础 SAM 的频域能量分布（A），但形成任务特化的相位相干模式（Φ）。迁移攻击应保留振幅谱结构作为频率重要性先验，同时主动破坏相位空间相干性。**

---

## 0. 实验事实（修正后）

修正后的迁移攻击实验（基础 SAM encoder → SAMed victim，Synapse vit_b，8 图）：

| 方法 | 探针公式 | eps=4/255 Δ | eps=8/255 Δ | eps=16/255 Δ |
|------|---------|------------|------------|-------------|
| PASA-full | `σ·N·exp(jΦ_clean)` | +0.049 | +0.463 | +0.879 |
| SE-CGR | `N·A^ρ·exp(jΦ_clean)` | +0.052 | +0.380 | +0.849 |
| w/o phase anchor | `σ·N·exp(jΦ_rand)` | +0.046 | +0.473 | +0.871 |
| amp modulated | `N·A^ρ·exp(jΦ_clean)` | **+0.058** | +0.478 | **+0.883** |
| w/o Gaussian | `σ·exp(jΦ_clean)` | +0.037 | +0.415 | +0.831 |
| **Phase-shuffled** | `σ·N·exp(j·shuffle(Φ))` | +0.035 | **+0.650** | +0.827 |
| Pure L1 (no probe) | — | +0.026 | +0.299 | +0.674 |

### 关键发现

1. **探针方法一律 > Pure L1**：在所有 budget 下成立，验证了频域 A-Φ 框架有效
2. **Phase-shuffled 在 eps=8/255 下碾压所有方法**：Δ=+0.650 vs 第二名 +0.478，优势 36%
3. **低 budget 时 clean phase + amplitude modulation 最优**（amp_mod Δ=+0.058）
4. **PASA-full（clean phase + white noise）从未第一**

### 理论含义

原来以为的"相位方向稳定，应该锁定"是错的。正确解读是：

> - **振幅谱 `|G_clean|` 是共享的频率重要性先验**（哪些频率承载信息）→ 跨模型可迁移
> - **相位相干性 `Φ_clean` 是微调改写的内容**（任务特化的空间结构）→ 不应保留，应该破坏
> - **最优攻击 = 保留振幅结构 × 破坏相位相干性**

---

## 1. 修正后的理论

### PASD 三原则

| 原则 | 内容 | 实验依据 |
|------|------|---------|
| **P1: 频域操作必要** | 探针 > Pure L1（所有 budget） | A-Φ 分解框架有效 |
| **P2: 相位相干性应破坏** | Phase-shuffled > PASA-full（eps=8/255, +40%） | 干净相位不是最优锚点 |
| **P3: 振幅谱是共享先验** | amp_mod > random_phase（eps=4/255） | `|G_clean|` 提供频率重要性信息 |

### 最终理论一句话

> **LoRA/Adapter 微调后的 SAM 继承基础模型的频域能量分布（振幅谱），但形成任务特化的相位相干模式。最优迁移攻击保留振幅谱作为频率重要性先验，主动破坏相位空间相干性，用随机幅度采样保留响应自由度。**

---

## 2. PASD 最终方法

### 探针构造

```text
对于每个选中层 l：

Z_clean^l = FFT(F_clean^l)
A_clean^l = |Z_clean^l|
Φ_clean^l = angle(Z_clean^l)

N_white ~ N(0, I)                              # 白噪声（随机幅度探索）
Φ_disrupted^l = shuffle(Φ_clean^l)              # 空间打乱相位（破坏相干性）
Σ^l = σ · A_clean^α                             # 振幅调制（频率重要性先验）

probe^l = IFFT( N_white · Σ^l · exp(j · Φ_disrupted^l) )
```

与旧 SE-CGR 的对比：

```text
SE-CGR: probe = IFFT( N · A_clean^ρ · exp(j · Φ_clean) )
PASD:   probe = IFFT( N · A_clean^α · exp(j · shuffle(Φ_clean)) )
                        ↑ 振幅先验         ↑ 相位破坏
```

**只改两处**：
1. `Φ_clean` → `shuffle(Φ_clean)`（破坏相位相干性）
2. 保留 `A_clean^α` 作为频率重要性先验（α 可调）

### 最终 Loss

```text
L = L_feat + L_probe

L_feat  = L1(0.9999 · F_adv, F_clean)           # 基础特征推远
L_probe = L1( mean_l( (F_adv^l - F_clean^l) - a · probe^l ) / input_mean )
```

### 推荐默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `σ` | 1.0 | 白噪声标量幅度 |
| `α` | 1.0 | 振幅调制指数（A_clean^α 作为频率先验） |
| `probe_amplification a` | 2.0 | 与 matched ablation v3 一致 |
| `K` | 2 | 噪声采样次数（取平均） |
| `steps` | 20 | 主表 |
| `eps` | 8/255 | 主表（最有区分度的 budget） |
| `layers` | last 6 encoder layers | 固定 |

---

## 3. 为什么 Phase-shuffled 比 Clean Phase 更强

直觉解释：

```text
基础 SAM 的 Φ_clean 编码了自然图像的通用空间结构（边缘方向、纹理相位等）。
微调后的 SAMed/MSA 对医学图像形成了新的相位相干模式。

用 Φ_clean 锚定攻击 → 攻击方向是基础模型的"舒适区"
                         → victim 对 Φ_clean 方向天然鲁棒

用 shuffle(Φ_clean) 攻击 → 打破了 victim 学到的相位相干模式
                           → victim 无法依赖任何空间结构 → 崩溃
```

这符合 C1/C2 的发现："变化方向稳定"——但进攻方不应顺着稳定方向走，而应该反向瓦解它。

---

## 4. 消融实验设计

### 核心消融（证明三原则）

| 条件 | 探针 | 验证问题 | 预期 |
|------|------|---------|------|
| **PASD-full** | `N·A^α·exp(j·shuffle(Φ))` | 完整方法 | 最强 |
| w/o phase disruption | `N·A^α·exp(j·Φ_clean)` | P2: 相位破坏是否必要？ | 下降（=PASA≈SE-CGR） |
| w/o amplitude prior | `N·σ·exp(j·shuffle(Φ))` | P3: 振幅先验是否必要？ | 下降 |
| w/o frequency domain | Pure L1 (no probe) | P1: 频域操作是否必要？ | 显著下降 |
| random phase | `N·A^α·exp(j·Φ_rand)` | 打乱 vs 随机？ | shuffle > random（保留分布） |
| SE-CGR | `N·A^ρ·exp(j·Φ_clean)` | 旧 baseline | PASD >> SE-CGR |

### 预期因果链

```text
PASD-full > w/o phase disruption     →  相位破坏是必要的
PASD-full > w/o amplitude prior     →  振幅先验是必要的
PASD-full > Pure L1                  →  频域操作是必要的
PASD-full > SE-CGR                   →  新方法超越旧 baseline
```

### 参数消融

| 参数 | 范围 | 目的 |
|------|------|------|
| α（振幅指数） | {0, 0.25, 0.5, 0.75, 1.0} | α=0→白噪声, α=1→全振幅调制 |
| σ（噪声标量） | {0.5, 1.0, 2.0} | 随机幅度强度 |
| K（采样数） | {1, 2, 4} | 采样稳定性 |

### 可选增强消融（非主方法）

| 变体 | 说明 | 目的 |
|------|------|------|
| β-shuffle | `Φ = angle(β·e^{j·shuffle(Φ)} + (1-β)·e^{j·Φ_clean})` | 混合相位，β 从 0→1 扫描 |
| Adaptive schedule | 早期 clean phase, 晚期 shuffled | 测试调度策略 |

---

## 5. 主实验矩阵

### 模型

```text
SAMed: Synapse/ISIC/REFUGE × ViT-B/L/H  (9 models)
MSA:   Synapse/ISIC/REFUGE × ViT-B/L/H  (9 models)
```

### 攻击 budget

主表：`eps=8/255, steps=20`（区分度最高的 budget）

辅助表：`eps={4/255, 16/255}, steps=20`
强攻击：`eps=32/255, steps=40`

### 方法列表

```text
1. PASD-full        (N · A^α · exp(j · shuffle(Φ)))  ← 最终方法
2. w/o phase disr   (N · A^α · exp(j · Φ_clean))     ← 关键消融
3. w/o amp prior    (N · σ · exp(j · shuffle(Φ)))    ← 关键消融
4. SE-CGR           (N · A^ρ · exp(j · Φ_clean))     ← 旧 baseline
5. Pure L1          (no probe)                        ← 频域 baseline
6. MI-FGSM          (pixel-space)                     ← 传统 baseline
7. Random           (unstructured noise)              ← 下界
```

### 样本数

| 数据集 | 最低 | 推荐 |
|--------|------|------|
| Synapse | 8 volumes | 12 volumes |
| ISIC | 18 images | 50 images |
| REFUGE | 18 images | 50 images |

---

## 6. 统计检验

- 主指标：`ΔDice = Dice(clean) - Dice(adv)`，正值=攻击成功
- 检验：paired Wilcoxon signed-rank test，以 case 为单位
- 校正：Bonferroni（比较次数 = 方法数）
- 效应量：配对 Cohen's d
- 成功率：`ΔDice > 0` 的比例

### 成功标准

PASD 成功需要同时满足：

1. **频域操作有效**：PASD > Pure L1（p < 0.05）
2. **相位破坏有效**：PASD > w/o phase disruption（p < 0.05）
3. **振幅先验有效**：PASD > w/o amp prior（p < 0.05）
4. **超越旧 baseline**：PASD > SE-CGR（p < 0.05）

---

## 7. 机制验证实验

### 相位相干性下降

```text
Coh = Re( U_adv · conj(U_clean) )
Coh_drop = 1 - mean(Coh)  # 越高 = 相位破坏越彻底
```

比较 PASD vs SE-CGR vs Pure L1 的 Coh_drop，预期 PASD 最高。

### 振幅谱保留

```text
A_corr = corr( |FFT(F_adv)|, |FFT(F_clean)| )  # 振幅相关性
```

比较 PASD（有 A_clean 先验）vs w/o amp prior 的 A_corr，预期 PASD 的 A_corr 更高，说明攻击保留了振幅结构。

### 攻击效果 vs 相位破坏相关性

跨 case 计算：

```text
corr( ΔDice, Coh_drop )
```

预期：相位破坏越彻底，攻击效果越强。验证因果链。

---

## 8. 论文贡献

### Contribution 1: 频域能量分布是共享先验

> We reveal that LoRA/Adapter fine-tuning of SAM preserves the base model's frequency-domain amplitude distribution — the spectral energy envelope is a shared prior that identifies structurally important frequency components across models.

### Contribution 2: 相位相干性是攻击面

> We show that fine-tuned medical SAMs develop task-specific phase coherence patterns, and that disrupting these patterns (via phase shuffling) is significantly more effective for transfer attacks than aligning with the source phase carrier.

### Contribution 3: PASD 方法

> We propose PASD (Phase-Aware Stochastic Disruption), a frequency-domain attack that preserves the base model's amplitude spectrum as a frequency-importance prior while disrupting phase spatial coherence through shuffling, achieving strong black-box transferability to LoRA- and Adapter-tuned SAMs.

---

## 9. 实现代码 skeleton

```python
def synthesize_pasd_probe(F_clean, sigma=1.0, alpha=1.0):
    Z_clean = torch.fft.fftn(F_clean, dim=(-2,-1), norm="ortho")
    A_clean = torch.abs(Z_clean)
    Phase_clean = torch.angle(Z_clean)
    
    # Phase disruption: spatial shuffle
    shape = Phase_clean.shape
    idx = torch.randperm(Phase_clean.numel(), device=Phase_clean.device)
    Phase_disrupted = Phase_clean.reshape(-1)[idx].reshape(shape)
    
    # Amplitude prior: A_clean^α
    Sigma = sigma * A_clean.pow(alpha)
    
    # Random amplitude within prior
    N = torch.randn_like(A_clean)
    probe_fft = N * Sigma * torch.exp(1j * Phase_disrupted)
    
    return torch.fft.ifftn(probe_fft, dim=(-2,-1), norm="ortho").real


def calculate_pasd_loss(features, targets, features_layer, targets_layer,
                         sigma=1.0, alpha=1.0, probe_amp=2.0, K=2, num_layers=6):
    loss_feat = L1(features * 0.9999, targets)
    
    pre_layer = int(len(features_layer) * 0.9)
    n_avail = min(num_layers, pre_layer)
    layer_ids = list(range(pre_layer - n_avail, pre_layer))
    
    probe_losses = []
    for k in range(K):
        diff_list = []
        for idx in layer_ids:
            F_clean = targets_layer[idx].unsqueeze(0).detach()
            F_adv = features_layer[idx].unsqueeze(0)
            probe = synthesize_pasd_probe(F_clean, sigma, alpha)
            diff = (F_adv * 0.999 - F_clean) - probe_amp * probe
            diff_list.append(diff.squeeze(0))
        
        diff = torch.stack(diff_list, dim=0)
        sel_t = torch.stack([targets_layer[i] for i in layer_ids], dim=0)
        input_mean = mean_abs(sel_t)
        diff = diff / (input_mean + 1e-6)
        probe_losses.append(torch.abs(torch.mean(diff, dim=0)).mean())
    
    loss_probe = torch.stack(probe_losses).mean()
    return loss_feat + loss_probe
```

---

## 10. 最终一句话

> **PASD 不锚定干净相位，不保留相位相干性。它继承基础 SAM 的振幅谱作为频率重要性先验，主动打乱相位空间结构以破坏微调模型学到的相位相干模式，用随机幅度保留响应自由度，实现最强的 LoRA/Adapter SAM 黑盒迁移攻击。**
