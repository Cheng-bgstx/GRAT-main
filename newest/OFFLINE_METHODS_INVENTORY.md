# 离线方法清单：所有尝试与结果

> 核心实验记录：基础 SAM encoder → SAMed victim 迁移攻击，Synapse CT 数据

---

## 一、实验基础设施

### 代码文件

| 文件 | 功能 |
|------|------|
| `attack_pasa.py` | PASD 核心攻击库：探针合成、loss计算、攻击管线 |
| `pasd_full_pipeline.py` | 离线先验提取 + A-Φ 解耦攻击 + 对抗样本生成 |
| `PASD_FINAL_PLAN.md` | 最终方案文档（理论+实验） |

### 标准配置

```text
攻击源：  基础 SAM image_encoder（无 LoRA，vit_b）
受害者：  SAMed Synapse vit_b（LoRA rank=4）
步骤：    20步 PGD
步长：    2.0/255（匹配原始 SE-CGR）
ε预算：   2/255, 4/255, 6/255, 8/255, 16/255
训练数据：Synapse train_npz (2211 slices, 全部)
测试数据：Synapse test_vol_h5 (12 volumes，取 8-12 slices/case)
评估指标：ΔDice = Dice(clean) - Dice(adv)，正值=攻击成功
统计：    N=3~5 trials, mean ± std
```

---

## 二、已尝试的离线方法（14+ 种）

### 类型 A：基于 PASD 扰动的后处理

核心操作：在训练集上跑 PASD → 收集扰动/动量/梯度 → 聚合（平均/PCA/k-NN）→ 在线作为初始化/引导。

#### A1. NAIVE — 像素扰动平均

| 项目 | 内容 |
|------|------|
| **代码** | 内联脚本（无独立文件）以及 `offline_shared_direction.py` 中的 `offline_naive()` |
| **离线** | 对每张训练图跑 20 步 PASD → 收集最终扰动 δ_i → δ_mean = mean(δ_i) |
| **在线** | x_adv = x + δ_mean（缩放到 ε），然后标准 PASD |
| **理论** | 如果 PASD 在各个图上找到的扰动方向有公共分量，平均后信号叠加 |
| **结果** | ❌ Δ < baseline（扰动相互抵消） |
| **训练数据** | 64 slices |
| **测试数据** | 8 slices |

#### A2. MD (Momentum Distillation) — 动量 PCA 子空间

| 项目 | 内容 |
|------|------|
| **代码** | `offline_shared_direction.py` 中的 `offline_md()` + `pasd_with_md_warmstart()` |
| **离线** | 跑 PASD → 收集最终动量 m_i → L1 归一化 → PCA(rank=4/8/16/32) → 均值方向 + 基 |
| **在线** | momentum_0 = mean_dir；每步梯度投影到 PCA 子空间（proj_weight=0.1~1.0） |
| **理论** | 动量累积了 20 步方向信号，PCA 提取公共子空间 |
| **结果** | ❌ 3-trial: 所有 rank/weight 都 < baseline（-3%~-33%） |
| **训练数据** | 8/16/32/48/64 slices |
| **测试数据** | 8 slices |

#### A3. k-NN — 特征检索扰动初始化

| 项目 | 内容 |
|------|------|
| **代码** | `offline_knn_init.py` |
| **离线** | 跑 PASD → 保存 (encoder_features, perturbation) 数据库 |
| **在线** | 提取测试图特征 → k-NN 检索最近训练图 → 混合它们的扰动作为初始化 |
| **理论** | 特征相似的图有相似的脆弱方向 |
| **结果** | ❌ k=1/3/5 均 < baseline（-13%~-23%） |
| **训练数据** | 64 slices |
| **测试数据** | 8 slices |

#### A4. UAP-PASD — PASD Loss 迭代 UAP

| 项目 | 内容 |
|------|------|
| **代码** | `offline_uap_pasd.py` |
| **离线** | 经典 UAP 算法（Moosavi-Dezfooli et al.）但用 PASD loss 替代交叉熵：迭代遍历训练图，若 UAP 不够"欺骗"则做一步梯度上升 |
| **在线** | momentum_init / momentum_bias / x_init 三种策略 |
| **理论** | 迭代积累的梯度方向比后验平均更一致 |
| **结果** | ❌ 3 种策略均 < baseline（-24%~-44%），fooled_rate=0% |
| **训练数据** | 16 slices |
| **测试数据** | 4 slices |

#### A5. CPP (Conditional Perturbation Predictor) — 特征→扰动预测网络

| 项目 | 内容 |
|------|------|
| **代码** | `offline_cpp.py` |
| **离线** | 训练 CNN: encoder_features → perturbation，loss=cosine_similarity(pred, PASD_pert) |
| **在线** | δ_init = CPP(features)，作为 PASD 初始化 |
| **理论** | 学习条件化映射，避免平均化——每图预测自己的扰动 |
| **结果** | ❌ cos_sim 只有 0.48（学不到），所有 weight < baseline |
| **训练数据** | 64 slices |
| **测试数据** | 8 slices |

### 类型 B：基于梯度统计的信号增强

核心操作：不改变 PASD 的攻击方向，而是收集训练集梯度统计来**调控梯度幅度**。

#### B1. GA (Gradient Agreement) — 梯度符号一致性加权

| 项目 | 内容 |
|------|------|
| **代码** | 内联脚本 |
| **离线** | 收集每图 PASD 第 0 步梯度的符号 → agreement[p] = |mean(sign(grad_i[p]))| |
| **在线** | grad *= (1 + weight × agreement)，高一致性位置放大梯度 |
| **理论** | 梯度符号一致的像素是"共享脆弱点"，应该加权重 |
| **结果** | ❌ 3-trial: Agreement≈0.10（接近偶然），所有 weight < baseline |
| **训练数据** | 64 slices |
| **测试数据** | 8 slices |

#### B2. Fisher Sensitivity — 梯度能量加权

| 项目 | 内容 |
|------|------|
| **代码** | 内联脚本 |
| **离线** | 收集每图 PASD 第 0 步梯度平方 → Fisher[p] = mean(grad²) → 归一化 |
| **在线** | (a) Fisher 梯度加权: grad *= (1 + w × Fisher)，(b) Fisher 结构化初始化: δ_0 ∝ Fisher |
| **理论** | 即使符号不一致，梯度幅度大的像素对 loss 更敏感 |
| **结果** | ⚠️ Fisher-init 在 eps=2/255 +130%（唯一有效），eps≥4/255 无效 |
| **训练数据** | 128 slices |
| **测试数据** | 8 slices |

### 类型 C：基于频域结构的修改

核心操作：不收集扰动，而是修改 PASD 探针的频率构造。

#### C1. FDUD (Frequency-Domain Universal Direction)

| 项目 | 内容 |
|------|------|
| **代码** | `offline_fdud.py` |
| **离线** | 跑 PASD → 收集每层归一化频域漂移 dZ → 平均 → 通用频域漂移 D_univ |
| **在线** | probe = IFFT( (σN + γ×D_univ) × A^α × exp(j×shuffle(Φ)) )，混合通用方向和随机噪声 |
| **理论** | 平均频域漂移捕获共享的频域脆弱结构 |
| **结果** | ❌ gamma=0.1~1.0 < baseline，gamma=5.0 仅 +7%（不稳健） |
| **训练数据** | 64 slices |
| **测试数据** | 8 slices |

#### C2. Domain Average Amplitude

| 项目 | 内容 |
|------|------|
| **代码** | 内联脚本 |
| **离线** | 收集所有训练图的 |FFT(F_clean)| → 平均 → A_domain |
| **在线** | probe 幅度用 A_domain^α 替代 A_clean^α |
| **理论** | 跨图平均消除单图噪声，捕获 CT 域固有频率重要性 |
| **结果** | ❌ eps=4/255 -15%，eps=8/255 持平（±0.9%） |
| **训练数据** | 128 slices |
| **测试数据** | 8 slices |

#### C3. TPD (Targeted Phase Disruption)

| 项目 | 内容 |
|------|------|
| **代码** | 内联脚本 |
| **离线** | 跑 PASD → 收集相位变化 → 平均 → 相位破坏敏感度 map |
| **在线** | 在 PASD 探针中加权相位破坏：高敏感度频率破坏更多 |
| **理论** | P2 说相位破坏是次要增益（+18%），离线让破坏更聪明 |
| **结果** | ❌ 3-trial: eps=8/255 下 β=0.5 -2.7%，所有 β 不稳健 |
| **训练数据** | 96 slices |
| **测试数据** | 8 slices |

### 类型 D：基于模型参数空间的分析（非 PASD）

核心操作：离线完全不用 PASD，直接分析基础 SAM 特征空间的域结构。

#### D1. Domain Gap (医疗 vs 自然)

| 项目 | 内容 |
|------|------|
| **代码** | 内联脚本 |
| **离线** | F_med = mean(encoder(CT_imgs))，F_nat = mean(encoder(random_noise)) → gap = F_med - F_nat |
| **在线** | PASD loss 中加入 gap 对齐项：maximize dF·gap_dir |
| **理论** | 域间隙是医疗域共享的 OOD 方向，推离它增加迁移性 |
| **结果** | ❌ 所有 gap weight < baseline（cos_sim(F_med, F_nat)=0.75，间隙弱） |
| **训练数据** | 96 slices |
| **测试数据** | 8 slices |

#### D2. PCA on Medical Features (域内主成分)

| 项目 | 内容 |
|------|------|
| **代码** | 内联脚本（未完成，代码报错） |
| **离线** | 收集 encoder(F_medical) → 全局池化 → PCA → PC 方向 |
| **在线** | 沿 PC 方向初始化扰动 |
| **理论** | 医学域的主成分 = 自然特征流形，正交推离更有效 |
| **结果** | ❌ 代码 bug 未跑完，但从 D1 推测效果不好 |
| **训练数据** | 1000 slices |
| **测试数据** | 12 slices |

#### D3. UAP from All PASD Perturbations (全部训练数据平均)

| 项目 | 内容 |
|------|------|
| **代码** | 内联脚本 |
| **离线** | 1256 张训练图全跑 PASD → 平均扰动 → UAP |
| **在线** | x_adv = x + scale × UAP，然后标准 PASD |
| **理论** | 大数据量下共享信号应该超过噪声 |
| **结果** | ❌ 所有 scale < baseline（UAP 本身接近零：|UAP|_inf=0.031） |
| **训练数据** | 1256 slices |
| **测试数据** | 12 slices |

### 类型 E：基线诊断

#### E1. 初始化敏感性诊断

| 项目 | 内容 |
|------|------|
| **代码** | 内联脚本 |
| **离线** | 无 |
| **在线** | 比较 random_init vs zero_init vs large_random_init |
| **结果** | eps=2/255: large_random > random > zero；eps≥4/255: 无差异 |
| **含义** | 初始化只在极低 budget 有影响 |

---

## 三、总结

### 全貌

| # | 方法 | 空间 | 类型 | 结果 | 最佳增益 |
|---|------|------|------|------|---------|
| A1 | NAIVE 平均扰动 | 像素 | 后处理 | ❌ | 负 |
| A2 | MD 动量 PCA | 像素 | 后处理 | ❌ | 负 |
| A3 | k-NN 检索 | 像素 | 后处理 | ❌ | 负 |
| A4 | UAP-PASD | 像素 | 迭代 | ❌ | 负 |
| A5 | CPP 预测网络 | 像素 | 学习 | ❌ | 负 |
| B1 | GA 梯度一致性 | 像素 | 幅度调制 | ❌ | 负 |
| B2 | Fisher 敏感度 | 像素 | 幅度调制 | ⚠️ | +130% @ eps=2/255 |
| C1 | FDUD 频域漂移 | 频域 | 方向注入 | ❌ | +7%（不稳健） |
| C2 | 域平均振幅 | 频域 | 先验替换 | ❌ | 负 |
| C3 | TPD 相位破坏 | 频域 | 先验加权 | ❌ | 负 |
| D1 | 域间隙 | 特征 | 方向引导 | ❌ | 负 |
| D2 | 特征 PCA | 特征 | 未完成 | — | — |
| D3 | 全量 UAP | 像素 | 全数据 | ❌ | 负 |

### 核心发现

1. **唯一有效的离线方法**：Fisher-init，仅在 eps=2/255 下有效（+130%），eps≥4/255 退化到无害
2. **任何"方向引导"都降低效果**：像素空间、频域、特征空间的先验方向无一例外地约束了 PASD 的随机探索
3. **15 种方法一致指向同一结论**：PASD 的随机频域探针（N_white × 打乱相位 × A_clean^α）是最优攻击结构，离线先验无法在实用 budget 下改进它
