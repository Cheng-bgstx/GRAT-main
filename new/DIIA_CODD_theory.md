# DIIA-CODD 理论重构稿

## 1. 为什么原来的“幅度变、相位不变”不能再作为主理论

原始 DIIA 的 ADS（Amplitude Deviation Simulation）将微调后的未知偏移解释为：

- 幅度响应主要变化；
- 相位响应基本保持；
- 因此在频域中只对幅度注入高斯扰动，并沿干净特征的相位载体回到空间域。

这套叙述的工程实现本身很强，但它依赖一个过强的理论前提：**target 的 phase 近似保持不变**。

当前验证结果说明，这个“普适的相位不变性”不能成立，尤其在 Adapter/Med-SA 一类目标上，深层 phase drift 很明显。因此，问题不在于“频域调制不能用”，而在于**相位在方法里扮演的理论角色必须改写**。

---

## 2. 新主理论：Carrier-Oriented Drift Decomposition (CODD)

### 2.1 共享继承 + 局部漂移

设基础 SAM 的某层干净特征为 \(F_s\)，下游微调模型对应特征为 \(F_t\)。
我们不再假设：

\[
F_t \approx \text{same-phase}(F_s) + \text{amplitude shift}.
\]

而改为更一般的局部漂移观点：

\[
F_t = F_s + \Delta F,
\]

其中 \(\Delta F\) 是围绕 foundation representation 的局部漂移。这个漂移既包含共享继承，也包含 target-specific 改变。

关键点：**黑盒可迁移攻击不需要精确拟合真实 \(\Delta F\)，只需要抓住其中“跨微调方式更稳定、更可迁移”的那一部分。**

### 2.2 在复频域中的径向-切向分解

令干净特征的频域系数为

\[
G_s = \mathcal{F}(F_s) = A_s e^{j\Phi_s}.
\]

对于任意未知目标漂移 \(\Delta G\)，都可以在以 \(e^{j\Phi_s}\) 为基底的局部复平面中分解为：

\[
\Delta G = \underbrace{\alpha e^{j\Phi_s}}_{\text{carrier-aligned radial drift}} + \underbrace{\beta j e^{j\Phi_s}}_{\text{carrier-orthogonal rotational drift}}.
\]

其中：

- \(\alpha\) 表示沿 source 载体方向的**径向漂移**；
- \(\beta\) 表示与 source 载体正交的**旋转/切向漂移**。

### 2.3 新的可迁移性假设

我们不再说“phase 不变”。

我们新的核心假设是：

> 在 foundation SAM 到 fine-tuned medical SAM 的迁移中，真正对跨模型攻击最有价值的，不是全部的 target drift，而是其中沿 source carrier 的稳定径向分量；而切向/旋转分量更依赖具体的微调策略、层位置和目标数据，因此更不稳定、更难迁移。

换句话说：

- **相位不必保持不变**；
- 但 **source phase 仍然可以作为局部参考系/载体坐标轴**；
- 黑盒攻击应重点建模“carrier-aligned radial uncertainty”，而不是拟合全部未知旋转。

这就把“phase”从 **target invariant quantity** 改成了 **source carrier / local reference frame**。

---

## 3. 为什么原来的高斯噪声还可以保留

原始代码中的核心构造是：

\[
\Sigma_{freq} = \lambda_t A_s + \gamma,
\]

\[
N \sim \mathcal{N}(0, I),
\]

\[
\Delta G_{sim} = (N \odot \Sigma_{freq}) e^{j\Phi_s},
\]

\[
\xi_{struct} = \mathcal{F}^{-1}(\Delta G_{sim}).
\]

过去它被解释为：

- target amplitude 变化；
- target phase 基本不变；
- 所以沿 clean phase 注入 amplitude noise。

现在要改写为：

- \(A_s\) 决定了 source 频谱的局部能量分布，因此可作为**各频率方向上的各向异性协方差包络**；
- \(\Phi_s\) 不是 target 必须继承的相位，而是用于定义**source carrier 上的径向方向**；
- \(\Delta G_{sim}\) 不是在“拟合真实 target 相位”，而是在 source carrier 上采样**稳定的、结构对齐的局部漂移探针**；
- 这等价于在 inherited basin 内做一种**carrier-aligned anisotropic Gaussian smoothing**。

因此：

> **你不需要删掉幅度/相位调制，也不需要删掉高斯噪声；你只需要把它从“phase invariance assumption”改写为“carrier-oriented radial uncertainty modeling”。**

这会最大化保留当前代码里已经验证有效的强攻击性。

---

## 4. 模块重命名：保留 ADS 缩写，但改变含义

为了尽量少改论文和代码，建议保留 ADS 缩写，但将其从：

- **Amplitude Deviation Simulation**

改成：

- **Anisotropic Drift Simulation**

此时 ADS 的定义是：

> ADS simulates the transferable component of fine-tuning uncertainty as carrier-aligned anisotropic radial drift around the inherited source representation, rather than assuming universal phase preservation.

这样：

- 代码名可以不改；
- 论文图、表、消融中的 ADS 缩写也可以保留；
- 但理论含义完全升级。

---

## 5. 与共享子空间的结合：Shared Radial Adversarial Subspace

你前面提到的“共享对抗子空间”是可以保留的，但要和上面的 CODD 结合起来。

### 5.1 共享子空间不再定义为“phase-shared space”

而定义为：

> 在 proxy medical data 上，由多图像、多步攻击梯度反复激活的、**carrier-aligned radial directions** 所张成的低秩子空间。

记为 \(\mathcal{U}_r\)。

它表示：

- 在不同医学图像上反复出现；
- 与 source feature carrier 对齐；
- 经高斯各向异性漂移筛选后仍然稳定；
- 因而最可能跨越不同 fine-tuned SAM 传递的共享脆弱方向。

### 5.2 离线阶段的新创新

原论文离线阶段只学一个 universal perturbation \(\delta_{univ}\)。

现在可改成两部分：

1. **universal mean direction**：\(\bar m\)
2. **shared radial subspace**：\(U=[u_1,\dots,u_k]\)

离线训练时，对 proxy medical set 的若干图像运行少步原始高斯 ADS 攻击，收集每一步的归一化动量/梯度 \(g_i\)，做 PCA/Oja 更新，得到：

\[
\bar m = \frac{1}{N}\sum_i g_i,
\]

\[
U = \text{top-k eigenvectors of } \frac{1}{N}\sum_i (g_i-\bar m)(g_i-\bar m)^\top.
\]

于是在线攻击时，初始化可写为：

\[
m_0 = \alpha \bar m + \beta \Pi_U(g_0),
\]

其中 \(\Pi_U\) 表示投影到共享子空间。

注意：

- **在线主损失完全不改**；
- 共享子空间只做 warm-start / gradient bonus；
- 这样不会破坏你当前最强的高斯主攻击逻辑。

---

## 6. 新的在线目标解释

你当前代码本质上在最大化：

\[
L_{ADS} = \left\| (F_{adv} - F_{clean}) - \xi_{struct} \right\|_1 + \lambda_{org}\|F_{adv} - F_{clean}\|_1.
\]

新的解释是：

- \(F_{adv}-F_{clean}\) 是当前攻击方向在 source encoder 上造成的真实漂移；
- \(\xi_{struct}\) 是 source carrier 上采样到的各向异性局部漂移探针；
- 最大化它们的偏离，等价于寻找**在局部 basin 扰动下仍具破坏性的稳定攻击方向**。

这不要求 target 的 phase 与 source 一致，只要求 target 仍落在 source inherited basin 周围。

---

## 7. 论文里可以直接替换的关键表述

### 原表述（建议删除）

> Fine-tuning mainly shifts amplitude responses while preserving phase responses.

### 新表述（建议替换）

> Fine-tuning does not preserve a universal spectral component across all downstream medical SAMs. Instead, it induces local representation drift around the inherited foundation response. In the complex spectral plane of each source feature coefficient, this drift can be decomposed into a carrier-aligned radial component and a carrier-orthogonal rotational component. We argue that transferable attacks should emphasize the stable radial component while marginalizing unstable rotational variations. Based on this, we propose ADS as an anisotropic drift simulation mechanism that injects Gaussian uncertainty along the source spectral carrier rather than assuming target phase invariance.

### 对方法模块的描述

> ADS uses the clean source phase only as a local carrier/reference frame to synthesize spatially coherent uncertainty probes. This does not imply that downstream phase responses remain unchanged after fine-tuning.

### 对共享子空间的描述

> The offline stage distills a shared radial adversarial subspace from proxy medical data, capturing transferable directions that consistently survive carrier-aligned stochastic drift.

---

## 8. 这条线的创新点到底在哪里

你的新创新点不再是“发现 phase 基本不变”，而是：

1. **Carrier-Oriented Drift Decomposition (CODD)**：
   第一次把 foundation-to-finetuned 的频域漂移解释为“沿 source carrier 的径向稳定分量 + 与之正交的旋转不稳定分量”。

2. **ADS 的新定义**：
   不是 amplitude-only 假设，而是 carrier-aligned anisotropic Gaussian drift simulation。

3. **Shared Radial Adversarial Subspace**：
   离线阶段不只学单个 UAP，而是蒸馏低秩共享子空间；在线阶段只把它当 warm-start/bonus，不替换原强攻击主干。

4. **工程上几乎不损失原始强度**：
   因为在线主损失仍然保留原始强攻击结构。

---

## 9. 最推荐的论文标题级表述

### 方法总名

**DIIA-CODD: Domain-aligned Initialization and Instance-specific Adaptation with Carrier-Oriented Drift Decomposition**

### 模块名

- DI：Domain-aligned Initialization
- SGA：Structural Gradient Alignment
- ADS：Anisotropic Drift Simulation
- SRS：Shared Radial Subspace

### 一句话摘要

> We replace the fragile phase-invariance hypothesis with a carrier-oriented drift decomposition view: transferable vulnerabilities are concentrated in carrier-aligned radial directions that remain stable under anisotropic Gaussian drift around the inherited SAM representation.

---

## 10. 最终建议

### 你不该做的

- 不要再把论文主理论写成“fine-tuning mainly changes amplitude while preserving phase”；
- 不要删掉你当前最强的高斯模块；
- 不要把 online 主损失大改成完全不同的东西。

### 你应该做的

- 保留原始高斯噪声与频域调制；
- 重新解释 phase 的角色：从“target invariant”改为“source carrier”；
- 把 ADS 重命名为 Anisotropic Drift Simulation；
- 增加离线 shared radial subspace 作为创新点；
- 在论文实验里重点做“carrier-aligned radial noise vs random complex-phase noise vs spatial Gaussian noise”的对照，这会直接支撑新理论。

