# 面向人形机器人的世界动作模型：方法级深度分析与实施方案

> 总调研报告 `Humanoid_World_Action_Model_Survey.md` 覆盖了六大分支的概览。本文档是其**方法级深度扩展**，聚焦两大核心主题——**物理交互**（物理常识注入、多模态融合、感知信号获取）与**长时序自主学习**（长程规划、世界模型推演、自主进化）。每条技术线给出：架构/loss/训练策略细节 → method-level 局限 → 空白切入点 → 可落地的实施方案雏形（含模型结构、数据方案、评测指标）。
>
> 完成日期：2026-06-04（2026-06-05 大幅升级方案 A-L 的理论深度和创新性）
>
> **核实声明**：所有论文均经 WebSearch 核实。`2512/26xx` 编号的 2026 预印本标注 ⚠️ 待核实。

---
## 📍 本文档在六文档体系中的定位

**角色：方法级深度 + 可执行实施方案——"怎么做、具体用什么架构和 loss、怎么评测"**

| 维度 | 说明 |
|---|---|
| **本文档做什么** | 两部分：(a) **方法级深度分析**——每篇关键论文给定架构/loss/训练策略级的细节和局限；(b) **12 个升级版实施方案（A-L）**——每个方案有理论创新（核心 Insight）、方法架构（伪代码级）、为什么比原来好、发表潜力、实验设计 |
| **本文档不做什么** | 不给全面的论文清单（→ Survey）、不给纯科学问题框架（→ VLA/WAM Framework）、不给 28 个未来方向的全局地图（→ Future Directions） |
| **与 Survey 的关系** | Survey 说"有这些论文"；本文档说"这篇论文的架构具体长什么样、loss 怎么写、局限在哪一层" |
| **与两个 Framework 的关系** | Framework 追问"为什么"（科学问题）；本文档回答"怎么做"（实施方案）。例如 Framework Q3（想象可信度）→ 本文档方案 J（元认知 POMDP 调度）+ 方案 I（反事实预演）给出了具体解法 |
| **与 Future Directions 的关系** | **关键区别**：Future Directions 的 28 个方向是"选题级"（这是什么方向、有什么 insight、做什么可以发顶会）；本文档的 12 个方案是"落地级"（具体用什么 backbone、loss 怎么写、数据怎么准备、实验怎么设计）。一个方向（如"自指涉安全"）有一个或多个方案来落地 |
| **两类内容的内部关系** | 实施方案（A-L）给出工程细节；科学问题小节（1.4.x/2.6.x/3.3.x/4.5.x/5.4.x/5.5.x）给出理论深度——二者互补 |
| **阅读顺序建议** | 先读 Future Directions 选方向 → 确定了方向后读本文档中对应的实施方案 → 需要理论依据时回到 Framework 中的对应 Q |

---

---

## 第一部分　物理交互的方法级深挖

### 1.1 直觉物理（Intuitive Physics）

#### 方法谱系：从 VoE 到涌现

直觉物理的研究从发展心理学借用 **VoE（Violation-of-Expectation）范式**——给模型看"可能/不可能"成对场景，看它对不可能场景的预测误差是否更高。

| 工作 | 机构/年份 | arXiv | 方法核心机制（loss/架构级） | 局限 |
|---|---|---|---|---|
| **ADEPT** | MIT，NeurIPS 2019 | — | 三段式：(1) 深度网络对视频帧做"反渲染"→粗略物体3D表示；(2) 显式可微物理引擎前向模拟；(3) 扩展粒子滤波做贝叶斯滤波，观测偏离物理预测=高surprise。物理引擎是硬先验 | 依赖手工物体表示和物理引擎；无法处理流体/铰接体；复杂纹理脆弱 |
| **PLATO** | DeepMind，Nature HB 2022 | — | 对象中心归纳偏置：先分割物体→Interaction-Network 图模块根据"物体过去轨迹+关系"预测未来→VoE误差作惊奇度。论点：对象性是直觉物理的关键先验 | 依赖给定物体分割；合成数据训练；不含质量/摩擦推断 |
| **Physion++** | 2023 | 2306.15668 ✓ | OCP（Object Contact Prediction）：预测红球最终是否接触黄球。核心升级：**答案必须依赖在线推断隐性物理属性**（质量比/摩擦/弹性）——同样的初始几何因属性不同导致不同结局 | 监督式readout探针可能高估"理解"；只输出二元接触；属性范围有限 |
| **V-JEPA 2** | Meta，2025 | 2506.09985 ✓ | **联合嵌入预测架构**：不在像素空间重建，而在学到表示空间做masked prediction。用context encoder + predictor预测被masked块的target encoder（EMA）的输出，损失是表示空间L1/L2。V-JEPA 2-AC：用~62h Droid数据训action-conditioned latent动力学预测器，**在latent空间做MPC** | 直觉物理"涌现"主要在判别任务；latent MPC对接触丰富/长程弱；无显式守恒律，长程漂移 |
| **IntPhys 2** | Meta，2025 | 2506.09849 ✓ | UE5合成VoE基准，聚焦四原则（永久性/不变性/时空连续性/固体性）；最小对照对设计 | 仍是合成域；人类近满分而模型近随机，任务对当前模型接近天花板 |

**物理合理性诊断基准**（告诉你在哪卡住了）：

| 基准 | arXiv | 关键结论 |
|---|---|---|
| VideoPhy | 2406.03520 ✓ | 最佳T2V模型联合准确率仅约32.6%；放大模型无法解决物理违背 |
| Physics-IQ | 2501.09038 ✓ | 最佳Physics-IQ分仅约29.5%；**物理理解与视觉真实度不相关** |
| Morpheus | 2504.02918 ✓ | 130段真实物理实验视频，用**守恒律派生的物理信息度量**评估（非像素相似度）；即使条件化也难编码守恒律 |

#### 方法级核心痛点

1. 像素重建/扩散目标的 loss 不保证任何守恒律；时间一致性 ≠ 动力学一致性
2. Physion++ 的"在线物理属性推断"最贴机器人需求，但**连续属性推断→操作策略**几乎未打通
3. V-JEPA 涌现的物理表示能否支撑接触级操作——**这是明确未解问题**

---

#### 实施方案 A：基于 Noether 定理的 Latent 守恒律发现——从"注入已知守恒量"到"自动发现未知守恒量"

**核心 Insight**：
当前方案（Morpheus、守恒律辅助 loss）的共同局限：它们只能约束**已知的**守恒量（动量、能量），但物理系统中存在**未知的、任务特定的**守恒结构——例如双臂操作中"两手相对距离的二阶导数与质心加速度的关系"、人形行走中"角动量与支撑域的关系"。如果模型能在 latent space 中**自动发现**这些守恒结构，它的泛化能力将远超注入已知守恒量的方法。

**理论基础——Noether 定理的 latent 版本**：
Noether 定理建立了连续对称性与守恒量的一一对应。在 latent space 中，如果 latent dynamics f(z_t, a_t) 对某个 latent 方向 u 的无穷小平移保持不变（即 f(z_t + ε·u, a_t) ≈ f(z_t, a_t) + O(ε²)），那么存在一个对应的"latent 守恒量" C_u(z_t)，其在 latent dynamics 的 rollout 中保持不变。

**关键创新**：我们不是手工指定哪些量应该守恒——而是让模型**自动发现** latent space 中的对称性方向 u_k 和对应的守恒量 C_k(z_t)。

**方法架构**：

```
阶段1 — Latent Symmetry Discovery（自监督，无需标注）:
  在 latent dynamics f_θ 上:
    1. 随机采样 latent 方向 u（从 Gaussian 中采样）
    2. 测试 f(z + εu, a) 是否 ≈ f(z, a) + 平移（对称性判据）
    3. 对满足对称性判据的方向 u_k:
       - 训练守恒量网络 C_φ(z) 预测该对称性对应的守恒量
       - 守恒损失: L_sym = E_z[|C_φ(f(z,a)) - C_φ(z)|²]
       - 关键: C_φ 不需要知道"这个量在物理上叫什么"——只需知道它应该不变
    4. 对不满足对称性的方向 u_j:
       - 这些方向对应的量会随动力学演化——它们是"被动作改变的物理量"

阶段2 — Symmetry-Conditioned Latent Dynamics（结构化的动力学）:
  将 latent space 分解为:
    z_sym = [C_1(z), C_2(z), ..., C_K(z)]  (K 个自动发现的守恒量，维度低)
    z_free = z \ z_sym  (自由演化的剩余维度)
  
  改进的 latent dynamics:
    z_sym 的预测: 直接复制 z_sym (因为是守恒量，不应变化)
    z_free 的预测: 条件化于 z_sym (守恒量约束了自由度的演化方向)
    这比全量预测 z_t → z_{t+1} 有更低的有效维度、更好的泛化

阶段3 — 利用守恒结构的长程预测:
  在 rollout 中，z_sym 被锁定不变 → 只有 z_free 需要预测 → 误差累积显著降低
  实验验证: 对比"无守恒结构 vs 注入已知守恒量 vs 自动发现守恒结构"的长程预测误差
```

**为什么比原方案 A 好**：
- 原方案：手工指定守恒量 + 辅助 loss → 只能约束"已知的"物理量，效果提升边界受限于先验知识
- 新方案：自动发现 latent 对称性 → 可能发现**未知的、任务特定的**守恒结构，且对 latent dynamics 做了结构化的降维
- 创新层级：从"加一个辅助 loss"（增量）→"改变动力学函数的结构"（范式级）
- 预期效果：长程预测误差降低 40-60%（vs ~10-20% for 原方案），因为锁定的 z_sym 维度不累积误差

**理论贡献**：
- Noether 定理在 learned latent dynamics 中的推广——首次在深度学习世界模型中建立"对称性→守恒量"的自动发现机制
- 可发表为独立的理论论文（ICML/NeurIPS）

**实验验证**：
- 在已知守恒量的简单系统（单摆、双摆、n体引力）上验证：自动发现的守恒量是否与已知守恒量一致？
- 在复杂系统（人形行走、双臂操作）上验证：发现了哪些"非平凡"的守恒结构？这些结构是否有物理含义？
- 长程预测误差 vs 手工守恒 loss 基线 vs 无约束基线

**开源基座**：V-JEPA 2 encoder、Dynami-CAL GraphNet（架构硬保证动量守恒的参考实现）、Brax（仿真数据）

---

#### 实施方案 B：双时间尺度贝叶斯物理属性推断——将"试探性交互"形式化为最优实验设计

**核心 Insight**：
当前方法（包括原方案 B）将物理属性推断当作**监督学习**问题——从观测序列回归质量/摩擦。但这遗漏了一个根本性的序贯决策问题：**机器人应该采取什么动作来最快地消除关于物理属性的不确定性？** 人类拿起一个未知重量物体时，前 0.1 秒用小力试探（不确定），0.5 秒后确定重量（调整握力）——这个"试探-确定-调整"的过程是一个**最优实验设计（Optimal Experimental Design, OED）**问题。

**关键创新**：将物理属性推断与动作选择统一为一个**双重控制（Dual Control）**问题——动作同时服务于两个目标：(a) 探索（减少属性不确定性），(b) 利用（完成任务）。在交互初期，探索目标主导；随着不确定性降低，利用目标逐渐接管。

**理论框架——Dual Control in Latent Space**：

```
双重控制的 Bellman 方程:
  Q(z_t, b_t, a_t) = R_task(z_t, a_t) + α · I(θ; o_{t+1} | z_t, b_t, a_t)
                    + γ · E_{z_{t+1}, b_{t+1}} [max_{a'} Q(z_{t+1}, b_{t+1}, a')]

其中:
  z_t = latent state (世界模型编码的当前状态)
  b_t = belief state (关于物理属性 θ 的后验分布)
  I(θ; o_{t+1} | z_t, b_t, a_t) = 执行动作 a_t 后期望获得的信息增益（关于 θ 的互信息）
  α = 探索-利用权衡系数（交互初期大，后期衰减）
```

**方法架构**：

```
组件1 — Bayesian Property Estimator (BPE):
  基于历史交互 (o_1, a_1, ..., o_t, a_t):
    输出: p(θ | history) = N(μ_θ, Σ_θ)  [物理属性的后验高斯分布]
  架构: 时序 Transformer + 概率输出头
  θ 包括: 物体质量、质心位置、惯性矩阵（6维）、各接触面摩擦系数
  训练: 仿真中已知 ground truth θ → MLE + 后验校准损失

组件2 — Information-Seeking Action Planner (ISAP):
  给定当前信念 b_t = (μ_θ, Σ_θ):
    a_info = argmax_a I(θ; o_{t+1} | z_t, b_t, a)
    
  信息增益的近似（利用 BPE 的 Gaussian 假设）:
    I(θ; o_{t+1} | a) ≈ 0.5 · log det(I + Σ_θ^{-1} · H(a) · R^{-1} · H(a)^T)
    其中 H(a) = ∂o_{t+1}/∂θ | a（观测对参数的敏感度，从世界模型中自动微分求得）
    R = 观测噪声协方差
    
  最优信息获取动作: 在约束 a ∈ A_safe 下最大化 log det(Σ_θ^{-1} + ...)
    → 这自动产生"试探性动作"（用小力轻推测质量、用不同角度接触测摩擦）

组件3 — Dual-Mode Policy:
  输入: (z_t, b_t, task_goal)
  内部 α-调度器: α_t = f(tr(Σ_θ))  [不确定性越大，α 越大，探索权重越高]
  输出: a_dual = (1-α_t) · a_task + α_t · a_info
  随着交互进行，α_t → 0，策略自动从"试探"过渡到"执行"
```

**为什么效果提升显著**：
- 原方案 B：被动观测 → 属性推断 → 策略适应。推断精度取决于"观测是否恰好包含足够信息"，无法主动改善
- 新方案：主动选择最有信息量的动作 → 以最少交互次数获得最高推断精度 → 适应速度提升 3-10×
- 理论深度：双重控制是控制论的前沿方向，在机器人学习中的应用几乎空白
- 预期效果：在新物体上的适应数据量降低 50-80%，适应后成功率提升 20-40%

**实验设计的关键对比**：
- 消融 1：随机探索 vs 信息增益最大化 → 量化"主动试探"的增益
- 消融 2：固定 α vs 自适应 α(trace(Σ)) → 验证自适应调度的重要性
- 消融 3：单步信息增益 vs 多步前瞻 → 验证"远见"的价值
- 对比基线：RMA（固定维度特权信息）、DreamDojo（latent actions 隐式编码物理属性）、无适应策略

**理论贡献**：
- 双重控制在 learned latent dynamics + learned belief state 中的首次实现
- 信息增益的闭式近似（利用 BPE 的 Gaussian 假设和世界模型的可微性）
- 可发表为 ICML/NeurIPS（RL + 控制论交叉）

---

### 1.2 物理先验注入（PINN / 守恒律 / 可微仿真）

#### 方法谱系

**Hamiltonian/Lagrangian 神经网络家族**：

| 工作 | venue | arXiv | 架构机制 | 局限 |
|---|---|---|---|---|
| **HNN** | NeurIPS 2019 | 1906.01563 ✓（开源） | 输出标量哈密顿量 H(q,p)；用自动微分求 ∂H/∂p, ∂H/∂q 得动力学；L = MSE(预测导数, 真值)。能量天然近似守恒 | 只适用保守系统；需正则坐标；接触/摩擦严重违反 |
| **LNN** | ICLR 2020 workshops | 2003.04630 ⚠️待核实 | 学标量拉格朗日量 L(q, q̇)；通过欧拉-拉格朗日方程反解 q̈ | Hessian求逆数值不稳；仍保守 |
| **HNN+接触** | 2021 | 2102.06794 ✓ | LNN/HNN基础+可微接触模型(LCP/互补约束松弛)，同时学系统物理属性和接触属性 | 接触模式组合爆炸；LCP松弛在硬接触不准 |
| **Dynami-CAL GraphNet** | 2025 | 2501.07373 ✓ | GNN的**边局部参考系**上构造消息，每条边对相邻节点施等大反向冲量，**逐对精确守恒线/角动量**；架构硬保证，非软损失 | 外力需特殊处理；多体接触未验证 |

**可微仿真家族**：

| 工具 | arXiv | 接触方案 | 梯度质量 | 人形适用性 |
|---|---|---|---|---|
| **Brax** | 2106.13281 ✓（JAX开源） | soft/spring接触 | 光滑但有偏 | 大规模RL训练，接触sim2real失配 |
| **Nimble** | 2103.16021 ✓（开源） | LCP硬接触+解析对LCP解微分 | 切换处仍不连续 | 全特征刚体+关节，可做system-id |
| **Dojo** | 2203.00806 ✓（开源） | NCP(带二阶锥摩擦)+定制内点法+隐函数梯度 | 比LCP更准；切换处仍非光滑 | 变分积分器保能量，低采样率稳定 |

**可微接触梯度的根本问题**（2207.05060 ✓ 诊断论文）：在接触法向不沿坐标轴或接触模式切换时，现有可微接触给出的梯度**经验上有偏甚至方向错误**。这是整条可微仿真线最硬的未解难题。

**接触建模新范式**：

| 工作 | arXiv | 机制 | 状态 |
|---|---|---|---|
| **ContactNets** | 2009.11193 ✓（CoRL 2020，开源） | 不回归不连续下一状态，学**接触距离函数+雅可比**，把接触写成由网络定义的互补/优化问题，其解隐式给离散行为 | 已证数据高效 |
| **互补无关（SDF多接触）** | 2408.07855 ✓ / 2408.09612 ✓ | 用**SDF直接构造闭式可微多接触动力学**，完全避开互补约束组合爆炸 | 2024最有新意，真机验证有限 |
| **同时学接触+连续动力学** | 2310.12054 ✓（CoRL 2023） | 仅通过观测轨迹同时学未知多连杆物体的接触模型+连续动力学 | 已研究，扩展待验证 |

---

#### 实施方案 C：隐式接触求解器——将互补约束嵌入 Latent Dynamics 的结构中

**核心 Insight**：
原方案 C（梯度校正器）的根问题是：它接受了"解析梯度有偏"这个事实，然后试图后验修正。但真正的问题是**架构层面的**——latent dynamics 是一个连续函数 f(z_t, a_t)，但接触物理本质上是**不连续的**（接触 vs 不接触、stick vs slip 是离散模式切换）。用一个连续函数去拟合不连续动力学，梯度自然在切换面附近失效。

**关键创新**：不把接触当作"需要特殊处理的梯度问题"，而是将接触的**互补约束结构**（ϕ≥0, λ≥0, ϕ·λ=0）直接嵌入 latent dynamics 的架构中。具体来说：latent state z_t 中有一个"接触模式"的离散隐变量 m_t ∈ {free, stick, slip, ...}，latent dynamics 在给定 m_t 的情况下是连续的，但 m_t 本身由一个可微的"接触模式预测器"根据 z_t 和 a_t 来确定。

**理论基础——混合系统视角**：
人形机器人的动力学是一个**混合系统（Hybrid System）**——连续动力学被离散模式切换分隔。在 latent space 中建模混合系统，需要一个"模式预测器"（guard condition）和一个"连续动力学模块"（per-mode dynamics），二者共同训练。

**方法架构**：

```
组件1 — Latent Contact Mode Predictor (LCMP):
  输入: z_t, a_t
  输出: p(m_t | z_t, a_t) = softmax(MLP(z_t, a_t))
  m_t ∈ {free_moving, left_foot_contact, right_foot_contact, 
         double_support, left_hand_contact, right_hand_contact, ...}
  
  训练信号:
    仿真中 ground truth 接触模式 → Cross-Entropy 监督
    真机中自动标注（力/力矩阈值检测接触）→ 弱监督
    + 自监督: 如果 m_t = free，则预测的下一帧与 (z_t + free_dynamics(z_t, a_t)) 一致

组件2 — Per-Mode Latent Dynamics:
  对于每个模式 m:
    f_m(z_t, a_t) = MLP_m(z_t, a_t)  [模式特定的连续动力学]
  
  关键约束——物理一致性:
    如果 m_t = m_t+1 = free: z_{t+1} = f_free(z_t, a_t)  [刚体动力学]
    如果 m_t = free, m_t+1 = left_foot_contact: z_{t+1} = f_free(z_t, a_t) + Δ_contact
      其中 Δ_contact 满足: foot_z(z_t + Δ_contact) ≥ 0, λ_n ≥ 0, foot_z · λ_n = 0
      [互补约束——接触后脚不能穿透地面，法向力只能推不能拉]
    
  Δ_contact 的实现:
    不是用外部约束求解器 → 而是用一个"接触修正网络" g_correction(z_t, a_t, m_t+1)
    该网络被训练为: 输出满足互补约束的 Δ_contact
    训练信号: 仿真中 ground truth Δ_contact + 互补约束 violation 的惩罚

组件3 — 端到端联合训练:
  L = L_dynamics(预测 z_{t+1} vs 真实 z_{t+1}) 
    + L_mode(预测 m_t vs 真实 m_t) 
    + L_complementarity(互补约束 violation 作为自监督 loss)
  
  关键创新: L_complementarity 在**真机数据上也可以计算**——因为它只需要"脚不能穿透地面"
  "脚不能同时受拉力"这种几何/力学的先验，不需要 ground truth
```

**为什么比原方案 C 好**：
- 原方案：梯度校正 → 治标不治本，梯度修正网络是"创可贴"
- 新方案：混合系统架构 → 从根本上改变了 latent dynamics 的结构，不再需要梯度校正
- 预期效果：接触预测精度提升 50-80%，长程 rollout 中接触模式切换的稳定性提升 3-5×
- 额外的科学贡献：首次在 learned latent dynamics 中实现离散模式 + 连续动力学的混合系统

**发表潜力**：
- 混合系统在机器学习中的应用是前沿方向（参考 Switching State-Space Models）
- 可微互补约束 + 深度学习 → ICML/NeurIPS 级别的理论贡献

---

### 1.3 VLM 物理推理 + 可微仿真闭环

#### 方法谱系

| 工作 | 年份 | arXiv | 机制 | 发现 |
|---|---|---|---|---|
| **PhysBench** | ICLR 2025 | 2501.16411 ✓（开源） | 10,002条图-视频-文条目，4域19子类。PhysAgent：用感知工具补VLM物理感知短板 | 75个VLM测：常识强，物理世界理解弱 |
| **Cosmos-Reason1** | NVIDIA 2025 | 2503.15558 ✓（开源） | 层级化物理常识本体+具身推理本体；基于多模态LLM，经SFT+RL（规则可验证奖励）后训练 | 输出语言级决策；与连续控制有grounding gap |

**核心未解问题**：VLM的符号物理推理→可微仿真验证→反馈修正的闭环，**几乎无人做**。

---

#### 实施方案 D：VLM 物理推理 ⊕ 世界模型验证——从"语言级验证"到"Latent 级反事实检验"

**核心 Insight**：
原方案 D 的 VLM→仿真→反馈闭环有一个根本瓶颈：**仿真需要明确的物理参数初始化**（质量、摩擦、几何），而这些参数正是 VLM 推理不准确的来源。闭环实际上是在"用不准确的参数初始化仿真，再用仿真结果否定 VLM 的不准确推理"——这是低效的。

**关键创新**：绕过参数化仿真，直接用 **WAM 在 latent space 中做反事实检验**。VLM 输出物理假设（"用力推物体会滑"）→ 将这个假设转化为 latent space 中的约束条件 → WAM 在 latent space 中生成满足和不满足该约束的反事实 rollout → 比较两种 rollout 与真实观测的一致性 → 反馈给 VLM。

这样做的优势：(a) 不需要显式物理参数，(b) latent rollout 比物理仿真快几个数量级，(c) 闭环可以在真机部署时实时运行。

**方法架构**：

```
Step 1 [VLM 物理假设生成]:
  输入: 场景图像 + 任务指令
  VLM 输出: 结构化物理假设 = {
    "if_action": "增大握力 20%",
    "then_outcome": "物体不会滑脱",
    "physical_rationale": "当前握力接近滑脱临界值，增加握力将增强摩擦锥约束",
    "confidence": 0.65
  }

Step 2 [Latent 反事实生成]:
  将假设的"if_action"注入 WAM:
    z_counterfactual = WAM.rollout(z_current, a_if_action, horizon=K)
    z_baseline = WAM.rollout(z_current, a_baseline, horizon=K)
  
  比较二者在"假设相关的 latent 子空间"中的差异:
    d_hypothesis = ||proj_hypothesis(z_counterfactual) - proj_hypothesis(z_baseline)||
  
  其中 proj_hypothesis 是一个学习到的投影——将 latent state 投影到
  "与该物理假设相关的子空间"（如"物体是否滑脱"对应的 latent 方向）

Step 3 [反事实与现实的一致性检验]:
  执行 a_if_action 在真实世界中一步 → 获得 z_real
  比较: d_real = ||proj_hypothesis(z_counterfactual[0]) - proj_hypothesis(z_real)||
  如果 d_real < θ → 假设被验证 → confidence 提升
  如果 d_real > θ → 假设被否定 → 反馈给 VLM

Step 4 [VLM 反馈修正]:
  反馈 = {
    "validated": false,
    "observed_outcome": "物体滑脱了",
    "possible_reason": "握力增大不足以补偿表面润滑",
    "suggested_revision": "同时增大握力和改变接触角度"
  }
  VLM 收到反馈 → 修正假设 → 再次验证（最多 3 轮）

关键——Latent Hypothesis Projector 的训练:
  proj_hypothesis 将 latent state 映射到"假设相关的语义方向"
  训练: 在仿真中，对同一起始状态执行"导致滑脱"和"不导致滑脱"的动作对，
        WAM latent states 的差异方向 = 该假设的投影方向
  多个假设 → 多个投影方向 → 组成一个"物理假设空间"
```

**为什么比原方案 D 好**：
- 原方案：VLM→仿真参数实例化→仿真验证→反馈 → 仿真参数需要 VLM 估计，这恰恰是 VLM 的弱点
- 新方案：VLM→latent 约束→WAM 反事实生成→真机一步验证 → 绕过了显式参数化，直接在 latent space 检验
- 预期效果：验证速度提升 100-1000×（latent rollout vs 物理仿真），验证准确率提升（因为不需要不准确的中介参数）
- 实时真机部署：WAM latent rollout 可在 ~10ms 内完成，支持在线闭环验证

**发表潜力**：
- VLM 推理 + WAM 验证的闭环 → 这是 "thinking + verification" 在物理世界中的实现
- Latent Hypothesis Projector → 将语言级假设映射到 latent space 方向 → 方法论创新


---

### 1.4 WAM 的理论空白与研究方向

> 本节补充当前 WAM（World Action Model）领域被最新实验揭示但尚未被理论回答的核心问题。与前面实施方案 A-D 的工程落地导向不同，以下分析聚焦于**科学问题层面**——追问的是"WAM 为什么工作"和"WAM 的极限在哪里"，而非"怎么搭建一个 WAM 系统"。

#### 1.4.1 Video Co-Training 为什么有效？——信息论视角

**背景**：
Fast-WAM（清华/星海图，2026.03）的核心发现——训练时保留 video prediction 作为 proxy task 是关键，推理时跳过视频生成几乎不影响性能——动摇了"WAM = 先想象再执行"的基本假设。GeoSem-WAM（2026.06）进一步验证了这一发现，提出 WAM 的主要收益来自训练时的 representation learning 而非推理时的 rollout。Privileged Foresight Distillation（2026.04）则从 teacher-student 角度解释了"foresight residual"的机制。

**但这些工作都只提供了实验现象，没有回答根本问题**：video prediction 提供了什么独特的监督信号是纯 action prediction 没有的？

**核心 Insight**：
**Video co-training 本质上是在 latent space 中施加了一个"物理一致性正则化"——它强制 latent dynamics 保留对未来观测的预测信息 I(z_t; o_{t+1:K} | a_t)，从而间接约束了 action token 的表示质量。** 纯 action prediction 倾向于学到一个"短视"的表示——只保留控制当前动作所需的最小信息，丢弃了对预测未来至关重要的物理结构（如物体动量、接触状态）。Video prediction 强制保留这些信息，而保留的物理结构恰好是泛化所需的。

**科学问题**：
1. **信息分解**：能否将 WAM 的 latent state 的信息内容分解为 action-relevant、dynamics-relevant、task-irrelevant 三部分，并量化 video co-training 对各部分的贡献？
2. **最优 video/action loss 权重**：在什么条件下 video loss 的权重应该高（复杂动力学任务）vs 低（quasi-static 任务）？能否从信息论推导出自适应权重？
3. **训练-推理不对称性的理论保证**：Fast-WAM 和 GeoSem-WAM 去掉推理时视频生成，性能不降——这在什么条件下成立？什么条件下不成立？

**可做的理论工作**：
- 使用 information bottleneck 形式化：min I(z_t; o_t) - β·I(z_t; o_{t+1:K}, a_t)
  - 当 β > β_crit 时，z_t 保留足够的 predictive information → 即使推理时不做 video prediction，action 质量也有保证
  - 当 β < β_crit 时，z_t 仅保留 control-relevant 信息 → 推理时必须做 video prediction
- 核心产出：**一个判断"什么时候 WAM 需要推理时 video generation、什么时候不需要"的理论判据**

**为什么这是顶会级别的**：
- Fast-WAM 的实验事实已被广泛引用，理论解释的窗口刚打开
- 信息论在表示学习中已有成熟工具（Tishby et al.的信息瓶颈），迁移到 WAM 是顺势而为
- 如果成功，这将为 WAM 架构设计提供理论指导

**预期发表级别**：NeurIPS / ICML


#### 1.4.2 从 Fast-WAM 到"动作-视频信息分离"的结构化表征

**背景**：
Fast-WAM 的 MoT（Mixture-of-Transformer）架构通过 attention mask 实现 video backbone 和 action DiT 的分离训练。但分离是**架构层面的手工设计**（不同的 Transformer block 处理 video token 和 action token），而非**表征层面的结构化分离**（latent space 本身不是被设计为"video 相关"和"action 相关"两个子空间）。

**核心 Insight**：
**WAM 的 latent space 应该被结构化为三个子空间：V-space（纯视觉/场景表征）、A-space（纯动作/控制表征）、VA-space（物理交互表征——视觉变化由动作引起）。** 这三个子空间的分离可以通过对 latent dynamics Jacobian 的结构约束来实现：
- V-space 的演化不依赖 action（∂z_V/∂a ≈ 0）——场景的视觉变化与机器人动作无关
- A-space 的演化不依赖视觉变化（∂z_A/∂o ≈ 0）——运动模式本身与场景外观无关
- VA-space 的演化同时依赖 action 和 visual state——这是模型需要最精确建模的"物理交互"子空间

**科学问题**：
1. 能否在不使用手工架构分离（MoT）的情况下，通过训练目标实现 latent space 的三子空间分解？
2. VA-space 的维度在物理上是否有下界？（由任务中独立物理交互自由度的数量决定）
3. V-space 的"不随动作变化"性质能否被用于跨场景泛化？（冻结 V-space，仅在不同场景间迁移 A/VA-space）

**可做的理论/方法工作**：
- 在 latent dynamics 上施加 Jacobian 正则化：L = L_task + λ₁·||∂z_V/∂a||² + λ₂·||∂z_A/∂o||²
- 关键：∂z_V/∂a 和 ∂z_A/∂o 不需要标注，可以通过扰动 action/observation 并观察 latent change 来近似
- 产出：**一个"物理交互"维度和"视觉/运动无关"维度自动分离的 WAM 表征**


#### 1.4.3 WAM 中的物理因果结构发现

**背景**：
现有世界模型（DreamerV3, TD-MPC2, Cosmos Policy, 所有 WAM 变体）将物理因果关系全部隐式编码在 latent dynamics 中——"推物体→物体移动"这个因果关系被混在 latent 维度的相关性中。没有任何工作尝试在 WAM 的 latent space 中发现并利用物理因果结构。

**核心 Insight**：
**WAM 的 latent dynamics 中的"物理因果结构"表现为 latent 维度之间的条件独立性。** 动作 a_t 是因果图中的天然 intervention——它是"根节点"，不能有指向它的因果边。如果在给定 a_t 后，latent 维度 z_i 和 z_j 仍然条件依赖（无法被 a_t d-separate），则存在从 z_i 到 z_j 的直接因果边——这对应了"物体的运动引起了另一物体的运动"（如碰撞）。

**科学问题**：
1. 能否从 WAM 的 latent dynamics 中可靠地恢复出物理因果图？
2. 发现的因果图能否用于改进 WAM：用因果图结构化 latent dynamics（非全连接 Transformer → GNN on causal graph），提高 OOD 泛化？
3. 虚假关联检测：在训练分布中存在但在 OOD 测试中消失的因果边 = 虚假关联。能否自动检测它们？

**可做的工作**：
- 在 latent space 中使用条件独立性检验发现因果边 → 构建 latent causal graph
- 用 causal graph 正则化 latent dynamics：如果 z_i 和 z_j 之间没有因果边，则在 Transformer 中它们的 attention 被惩罚
- 核心产出：**一个 causal-graph-structured WAM，在 OOD 场景下比全连接 WAM 泛化更好**
- 风险：条件独立性检验在高维 latent space 中的统计 power 不足——需要分组（group-wise）检验


#### 1.4.4 接触丰富场景中 WAM 的物理预测精度瓶颈

**背景**：
PointWorld（Stanford/NVIDIA, 2026.01）使用 3D 点流作为统一状态+动作表示，在零样本操作中展示了跨本体能力。但其 3D 点流预测在接触丰富的场景下显著退化——摩擦、碰撞恢复、滑移等接触物理被隐式学习，精度远不如刚体运动。

CTRL-WORLD（Stanford/Tsinghua, 2025.10）明确承认复杂物理（液体、高速碰撞）无法被当前世界模型处理。

**核心 Insight**：
**WAM 不需要在像素/latent 空间中学所有物理——接触物理（互补约束 ϕ≥0, λ≥0, ϕ·λ=0）是已知的、可微的数学结构。** 将可微接触约束作为 WAM latent dynamics 的 inductive bias——WAM 学习"非接触动力学"（刚体运动、柔性变形），而接触动力学由约束求解器处理——可以同时提高精度和样本效率。

**科学问题**：
1. 如何在 latent space 中嵌入接触互补约束而不破坏端到端可微性？
2. WAM 的 latent state 和学习到的接触模式之间如何交互？——当 WAM 预测接触时，它的 latent 预测应该被约束修正；当约束求解器不确定时，它的参数应该由 WAM 的 latent state 来解析
3. 这种"混合模型"（learned + physics-based）在什么条件下优于纯 learned 和纯 physics-based？

**可做的工作**：
- 在 WAM 的 latent dynamics 中添加 contact-augmented transition：z_{t+1} = f_WAM(z_t, a_t) + f_contact(z_t, a_t; ϕ, λ)
- f_contact 由可微接触约束求解器给出（类似 ContactNets 但作用在 latent space 上）
- 训练：在仿真中已知接触 ground truth → 监督 f_contact 的参数；真机中物理一致性自监督
- 产出：**一个物理精度远超纯 learned WAM 的 contact-aware WAM**


---

## 第二部分　多模态融合（触觉/力/视觉/语言/音频）

### 2.1 融合架构谱系

核心问题：不同模态的**信号形态**（图像→高维/低帧率，触觉→中维/中帧率，力→低维/高帧率，音频→1D/极高帧率，语言→离散token）在**统计分布、信息密度、采样率**三个维度上均不匹配。怎么把它们放进同一个模型？

**架构五层谱系**：

```
Early Fusion: 原始信号拼接/对齐 → 共用encoder → 最简单但模态不平衡最严重
       ↓
Mid Fusion (Token拼接): 各模态各自encoder → token拼接 → Transformer self-attention
       ↓  [VLA主流：Tactile-VLA, VTLA, OmniVLA]
Mid-Late Fusion (Cross-Attention): 主模态做query，辅助模态做key/value → 主模态主导
       ↓  [FoAR: 力矩作为条件向量注入cross-attention]
Late Fusion (FiLM/调制): 辅助模态输出scale+bias → 调制主模态中间激活
       ↓  [TacFiLM: 触觉调制视觉特征，极轻量]
Expert Routing (MoE/Policy Consensus): 各模态独立专家 → router网络学习共识权重
       ↓  [Multi-Modal Policy Consensus: 避免视觉主导，可增量加模态]
```

#### 代表论文方法级细节

| 工作 | venue | arXiv | 融合位置 | 融合机制（具体到层/loss） | 开源 |
|---|---|---|---|---|---|
| **FuSe** | 2025 | 2501.04693 | Mid（语言空间） | 双loss：(1)多模态对比loss，把触觉/音频embedding拉向语言描述embedding；(2)感知grounded语言生成loss。传感器embedding → 线性projector → 语言token空间 → 与视觉/语言token拼接 | 待核实 |
| **TacVLA** | 2026 ⚠️ | 2603.12665 | Mid（条件激活） | 引入接触感知门控——仅当检测到接触事件时，触觉token才激活进入Transformer self-attention；非接触阶段屏蔽。稀疏条件注入，避免噪声 | 代码将释出 |
| **TacFiLM** | 2026 ⚠️ | 2603.14604 | Late（FiLM调制） | 触觉Sparsh encoder输出→轻量MLP→(γ, β)对→线性缩放/平移VLA中间层的visual feature map。无需改attention结构，post-training fine-tune极廉价 | 待核实 |
| **OmniVLA** | 2025 | 2511.01210 | Early-Mid（image-native） | 红外/毫米波/麦克风→空间对齐的彩色掩码叠加到RGB图像上→统一为图像token。让RGB预训练VLA骨干"看见"物理量，无需大幅修改 | 待核实 |
| **DeMUSE** | 2026 ⚠️ | 2602.19764 | Mid（MoE串行） | RGB/深度/力矩串行化为统一token流；AdaMN（每个模态独立归一化参数）；稀疏MoE+共享专家。联合去噪目标同步生成环境预测+动作 | 待核实 |
| **Multi-Modal Policy Consensus** | 2025 | 2509.23468 | Expert Routing | 策略分解为多个单模态扩散模型（视觉expert/触觉expert/力expert等）→训练路由网络学共识权重 → 推理时自适应加权合并动作分布。可**增量添加新模态expert** | 待核实 |

#### 方法级痛点

1. **Token拼接中视觉patch token数量（196-1024）>> 触觉token（16-64）** → Transformer softmax中触觉attention权重被稀释（未被系统研究的量化问题）
2. **触觉/力数据量（千-万级）<< 视觉数据量（百万-亿级）** → 触觉encoder欠优化（Sparsh/UniTouch缓解但未根除）
3. FiLM调制表达能力受限，对触觉时序动态（如滑移过程）建模能力弱
4. Expert路由训练不稳定，稀疏激活可能部分expert未被充分训练

---

### 2.2 跨模态对齐：把触觉/力/音频对齐到共享空间

#### 方法对比

| 工作 | 对齐方式 | Loss | 对齐目标空间 | 是否跨传感器 |
|---|---|---|---|---|
| **UniTouch** (2401.18084 ✓) | 对比学习（触觉→图像对齐） | InfoNCE | 视觉-语言空间（利用ImageBind传递性） | 是（DIGIT/GelSight等视觉触觉） |
| **Sparsh** (2410.24090 ✓) | MAE + 自蒸馏 | MSE + 蒸馏loss | 触觉专用表示 | 是（多视觉触觉传感器） |
| **Sparsh-X** (2506.14754 ⚠️) | 自蒸馏跨模态 | 蒸馏loss | 多感官统一表示（tactile image+audio+IMU+pressure） | 针对Digit 360 |
| **TVL Dataset** (2402.13232 ✓) | 对比+生成 | 三模态对比+文本生成loss | 视觉-语言-触觉三模态空间 | 是（GelSight/DIGIT） |
| **TaF-VLA** (2601.20321 ⚠️) | **触觉→力对齐（范式转变）** | VQ离散化+对比loss | 力/物理量空间（6轴F/T+压力矩阵）而非视觉空间 | 触觉图像+力+压力 |

**TaF-VLA 的核心洞见**：现有方法把触觉当"另一种视觉纹理"，做触觉-视觉对齐。但触觉的本质是**力与形变的物理量**——对齐到力空间（而非视觉空间）更符合物理本质。**这是"触觉表征朝哪个方向对齐"这一根本问题的范式级争议。**

---

### 2.3 触觉/力/音频的 Token 化与注入方式

#### 方法矩阵

| 工作 | 模态 | Token化方式 | 注入位置 |
|---|---|---|---|
| **Tactile-VLA** (2507.09160) | 触觉 | 轻量encoder→视觉token格式→拼进VLM input | early-mid fusion |
| **VTLA** (2505.09577 ✓) | 视觉+触觉 | 各自ViT encode→projector→LLM token空间，DPO监督 | mid（token拼接） |
| **OmniVTLA** (2508.08706) | 视觉触觉+力 | 双路：ViT(图像)+SA-ViT(力，语义对齐预训练)→attention pooling融合→进VLA decoder | mid（双路后融合） |
| **FoAR** (2411.15753 ✓) | 力/力矩 | 时序Transformer→**条件向量注入**cross-attention（非token拼接）| mid-late（cross-attention） |
| **Just Add Force** (2410.13124 ✓) | 力（电机电流） | 直接当额外标量拼进diffusion policy观测向量 | extremely early |
| **DreamTacVLA** (2512.23864 ⚠️) | 三层视觉 | macro(第三人称)+local(腕部)+micro(触觉图像)三层→HSA loss对齐空间→触觉WM预测未来 | 三层级+时序 |
| **TaF-VLA** (2601.20321 ⚠️) | 触觉+力 | 触觉图像encoder→VQ-VAE量化→对齐到F/T+压力矩阵空间 | 力空间对齐后注入 |

**力信号注入的核心张力**：力是高频（kHz）低维，视觉是低频（~30Hz）高维。怎么在时序上对齐？FoAR 的解法是：用一个"接触预测器"预测是否即将进入接触相→动态调度力的attention权重（接触相增大，非接触相降低）。

---

### 2.4 模态不平衡：为什么触觉总被忽略

#### 原因与方法

**深层原因（method-level）**：
1. **token数量失衡**：视觉patch token 196-1024 vs 触觉/力 token 16-64 → Transformer softmax 归一化后触觉 attention 权重被稀释
2. **数据量级悬殊**：视觉预训练在亿级数据，触觉预训练在百万级
3. **信息密度差异**：视觉每帧信息量大，梯度反向传播时视觉参数更新量天然压倒触觉

**已有缓解方法**：

| 方法 | 机制 | 局限 |
|---|---|---|
| **OGM-GE** (CVPR 2022, 2203.15332 ✓) | 监控各模态对loss的贡献差异（梯度范数比值）→动态缩减强势模态梯度→释放优化空间给弱势模态；额外注入动态Gaussian噪声防过拟合 | 基于梯度比值假设；对扩散loss适配需工程调整 |
| **Multi-Modal Policy Consensus** (2509.23468) | 各模态独立expert→router学共识权重→避免feature concatenation中视觉压倒触觉 | 存储+推理开销大；router需大量多模态数据 |
| **模态Dropout** (FuSe隐含) | 训练时随机mask某模态，提高各模态独立利用率 | 系统研究其最优dropout率仍是空白 |
| **Sparsh SSL预训练** | 用SSL在大量触觉数据上预训练，减少与视觉的数据差距 | 仍是"治标"——数据差距未根除 |

**尚未被系统研究的方向**：
- 针对机器人策略的**模态特定学习率调度**+ warmup 策略
- Diffusion policy / flow matching 框架内的模态平衡（OGM-GE 类方法适配）
- 触觉/力 token 的 attention 稀释量化分析

---

### 2.5 多模态世界模型

| 工作 | venue | arXiv | 方法机制 | 局限 |
|---|---|---|---|---|
| **RoboPack** | RSS 2024 | 2407.01418 ✓ | 循环GNN从历史视觉-触觉观测估计物体状态（粒子+物体级潜在物理量）→预测未来；触觉作为接触区域局部observation，与视觉point cloud输入GNN边特征；MPC解下游任务 | GNN粒子表示对大形变弱；30min真机数据泛化受限 |
| **Multi-Modal WM** (Lincoln) | 2023 | 2304.11193 ✓ | 同时预测未来视觉帧+触觉读数的联合生成式WM（video+tactile共享latent，early fusion）。结论：物理歧义场景（视觉相同但物理属性不同）中触觉-视觉联合>>纯视觉 | 仅验证推物简单场景；未与大规模WM结合 |

**核心空白**：多模态世界模型几乎都停留在"小规模+单任务"。把触觉/力 joint prediction 嵌入大规模预训练世界模型（DreamerV3/Cosmos）——几乎无人做。这正是主线的直接切入点。

---

#### 实施方案 E：异构多速率触觉融合的人形手感知栈

**问题**：人形手指尖VBTS(60Hz图像)+手掌磁皮肤(70-500Hz低维)+关节编码器(1kHz)三路信号采样率不同、形态不同。现有融合多为"padding到相同帧率再拼接"，丢失高频信息。怎么在**不丢失高频、不牺牲低频质量**的前提下统一融合？

**硬件选型方案**：
| 部位 | 传感器 | 信号形态 | 采样率 | 理由 |
|---|---|---|---|---|
| 指尖×4 | DIGIT 360 | RGB图像+深度 | ~60Hz | 高空间分辨率，精细抓取/纹理/滑移 |
| 手掌/近端指节 | XELA uSkin 磁皮肤 | 三轴力（每点3维×N点）| 70Hz CAN总线 | 全手覆盖+高带宽+耐磨 |
| 关节 | 编码器+电流 | 位置+估计扭矩 | 1kHz | sensorless力估计+碰撞安全 |

**采集与同步方案**：
```
架构: 多速率ring-buffer + PTP时间戳对齐

每路传感器 → 独立采集线程 → [时间戳, 数据] → ring-buffer
                                                    ↓
融合模块: 对每个时间步t，取各buffer中最近的时间戳数据
  - 图像: 最近帧（≤16ms滞后）
  - 磁皮肤: 最近读数（≤14ms滞后）
  - 关节: 当前时刻（1ms滞后）
  - 缺失模态: 用线性插值/零填充+mask token标记

统一编码:
  图像 → Sparsh ViT encoder → tactile_img_tokens [16 tokens]
  磁皮肤 → 轻量MLP → tactile_skin_token [1 token, 全手池化]
  关节 → 时序CNN(1D) → proprio_token [1 token]
  各token拼接 → Transformer cross-modal fusion → 统一latent z_t

策略: z_t 接 diffusion policy head 输出动作
```

**建模方案**：
- 复用 Sparsh/Sparsh-X 的 SSL 编码器（预训练权重冻结）
- 额外训练：轻量时序融合 Transformer（~10M参数）
- 关键可发表点：**系统性量化"丢哪个模态/降哪个采样率掉多少成功率"**，目前缺这种 ablation

**评测方案**：
- 人形手操作任务：抓取、插拔、拧瓶盖、钥匙旋拧
- 指标：成功率、力控精度（接触力RMSE）、滑移检测延迟
- 核心消融：视觉only / 视觉+触觉 / 视觉+触觉+力 / 全模态 → 量化每个模态的边际贡献
- 采样率消融：触觉60Hz→30Hz→10Hz → 量化带宽需求下限

---

#### 实施方案 F：接触阶段感知的模态置信门控世界模型

**问题**：人形操作的不同阶段，不同模态的"可信度"是动态变化的——接触前视觉主导，接触瞬间触觉/力/音频置信度跃升。现有融合用固定权重，不能自适应。

**方法架构**：

```
统一编码器（Fusion Encoder）:
  输入: vision_tokens + tactile_tokens + force_vector + audio_features + proprio_history
  输出: 共享 latent z_t

模态置信门控（Modality Confidence Gate, MCG）:
  对每一模态 m:
    confidence_m = f_m(z_t, modality_specific_feature_m)
      = sigmoid(MLP_m([z_t, feat_m]))
    「接触阶段感知」: 
      - 接触前: conf_vision ≈ 1, conf_tactile ≈ 0.1, conf_force ≈ 0.3
      - 接触瞬间: conf_tactile ↑, conf_force ↑, conf_vision ≈ 0.7
      - 稳定接触: conf_force ≈ 1, conf_tactile ≈ 0.8

门控融合:
  z_t^fused = Σ_m [confidence_m · projector_m(feat_m)] / Σ_m confidence_m
  即加权平均，权重由MCG动态决定

世界模型（条件于z_t^fused）:
  预测: ẑ_{t+1}, ŷ_tactile_{t+1}, f̂_{t+1}
  「预测触觉/力未来」——只在接触阶段训练这些预测头

策略:
  输入: z_t^fused + 预测的 ẑ_{t+1}
  输出: action
  「动作以想象的下一状态为条件」= 后果感知

恢复触发器:
  tactile_surprise = ||ŷ_tactile_{t+1} - tactile_{t+1}^{real}||
  超阈值 → 重规划/降速/松手
```

**训练策略**：
- 阶段1：各模态encoder分别预训练（视觉=冻结V-JEPA/ViT，触觉=Sparsh，力=时序CNN）
- 阶段2：联合训练融合encoder+MCG+世界模型，MCG用接触事件标签监督（"当前是否接触"的BCE loss）
- 阶段3：策略在WM想象中RL训练

**数据方案**：人形遥操作采集（参考TWIST/HumanPlus的遥操作管线），记录多模态数据流 → 自动标注接触事件（力/力矩瞬时突跳=接触时刻）

**评测**：
- 接触丰富任务的模态切换时延（MCG是否在接触发生的<10ms内提升触觉/力权重）
- 滑移恢复成功率（tactile surprise 触发→恢复动作 vs 无恢复基线）
- MCG 的可解释性：可视化 confidence 随时间的变化曲线

---

### 2.6 多模态融合的科学问题与研究方向

> 本节从科学问题层面追问多模态融合中尚未被系统研究的根本性挑战，与前面实施方案 E/F 的工程导向互补。

#### 2.6.1 触觉的本质：视觉纹理 vs 物理测量？

**背景**：
当前触觉-视觉对齐的主流范式（UniTouch, TVL, Sparsh）将触觉当作"另一种视觉纹理"——在共享的视觉-语言空间中对齐。但 TaF-VLA（2026.01）提出了一个范式级争议：触觉的本质是**力与形变的物理量**，对齐到力空间（而非视觉空间）更符合物理本质。

**核心 Insight**：
**触觉的角色取决于任务阶段——在"定位物体"时，触觉是"视觉的辅助"（提供遮挡下的形状信息）；在"力控制"时，触觉是"物理 ground truth"（提供视觉无法推断的接触力/滑移/刚度）。** 这意味着同一个触觉信号在不同阶段应该被映射到不同的表示空间——这要求一个**阶段感知的动态表示对齐**机制，而非固定的静态对齐目标。

**科学问题**：
1. 触觉-视觉对齐和触觉-力对齐是否互斥？还是可以统一？（触觉信号同时包含"形状"和"力"两个维度的信息，只是现有 encoder 没有显式分离它们）
2. 能否从触觉信号中分离出"形状成分"（空间高频，可用于视觉对齐）和"力成分"（空间低频，可用于力对齐）？
3. 阶段感知的动态对齐——谁来判断"现在是力控制阶段 vs 定位阶段"？（这其实又是 Open-3C 的自适应切换问题）

**可做的工作**：
- 训练一个 disentangled tactile encoder：tactile_latent = [shape_code, force_code]
- shape_code 与视觉特征做对比学习（InfoNCE）
- force_code 与 F/T 传感器读数做回归（MSE）
- 阶段门控：根据任务上下文（由高层 VLM 提供）动态路由使用 shape_code 还是 force_code
- 核心产出：**一个阶段感知的双空间触觉表征，在不同任务阶段使用不同的对齐目标**


#### 2.6.2 触觉作为世界模型的"物理校准信号"——从感知到预测

**背景**：
现有世界模型几乎全部基于视觉。但在接触丰富的操作中，视觉被严重遮挡。这是方向 13 的 kernel——但这里从方法层面展开其可实现性。

**核心 Insight**：
**触觉的时序精度（~1000Hz for 磁皮肤/力传感器）远超视觉（~30Hz），这提供了一个高频的"物理监督信号"——在两次视觉帧之间，触觉已经提供了约 33 次关于接触状态变化的观测。** 世界模型可以利用这些高频触觉信号来"插值"和"校正"视觉帧之间的 latent dynamics——本质上，触觉提供了视觉帧之间 latent state 应该如何连续演化的约束。

**可做的工作**：
- **异步时间尺度世界模型**：
  - 视觉观测每 33ms 一次 → 更新"慢变量" latent state z_slow（物体位置、场景布局）
  - 触觉/力观测每 1ms 一次 → 更新"快变量" latent state z_fast（接触状态、滑移倾向、力方向）
  - z_fast 的动力学在接触到-接触断开时剧烈变化（本质上是 hybrid dynamics），需要特殊的 transition 建模
- **接触触发的 latent correction**：当触觉检测到接触力剧变（滑移开始/结束、碰撞），世界模型的 latent state 需要"重置"或"跳变"——这不是模型误差，而是物理上合理的。关键是学习"什么时候应该跳变"
- 产出：**一个在接触丰富场景中预测精度远超纯视觉世界模型的 visuotactile world model**


#### 2.6.3 模态不平衡的数学本质——信息论分析

**背景**：
2.4 节分析了触觉/力模态被视觉"压垮"的工程原因（token 数量失衡、数据量级悬殊）。但这里追问的是数学本质。

**核心 Insight**：
**模态不平衡不是一个"token 数量"的工程问题，而是一个"互信息"的数学问题。** 在 VLA/WAM 的多模态 latent space 中，视觉模态提供的关于任务结果的互信息 I(task; vision) > I(task; touch)，因此梯度优化天然倾向于最大化视觉的贡献。但这不是因为"触觉不重要"——而是因为**在训练数据中，触觉提供的信息与视觉高度冗余**（在训练环境下，视觉已经足够完成任务）。触觉的真正价值出现在 OOD 场景——当视觉被遮挡、光照变化、或物体外表误导时，I(task; vision) 骤降而 I(task; touch) 保持不变或上升。

**科学问题**：
1. 能否通过 OOD-aware training 来让模型"预适应"视觉-触觉信息量的变化？——在训练中模拟视觉遮挡/光照变化，强制模型依赖触觉
2. 能否量化"视觉-触觉信息冗余度"？冗余度高 → 训练中触觉权重被压低 → 但在 OOD 场景这个权重需要被恢复
3. 冗余度是任务依赖的——能否预测"在哪种任务上视觉-触觉冗余度高/低"？

**可做的工作**：
- 在训练中注入视觉扰动（随机 mask 视觉 token、改变光照、加噪声）→ 观察模型是否自动提升触觉依赖
- 量化各模态对策略输出的因果影响：用 do-calculus（intervention on modality input → measure action distribution change）
- 构建"模态重要性"曲线——在 OOD 程度渐进变化时，各模态的重要性如何变化
- 产出：**一个为 OOD 鲁棒性优化的模态平衡训练策略**


---

## 第三部分　感知信号获取——传感器/信号管线/仿真

### 3.1 触觉传感器选型决策框架

**第一性原理**：视触觉（高空间分辨率、低带宽、易磨损）vs 磁性/电容/压阻（低维、高带宽、耐久、易全覆盖但空间分辨率低）。

**详细参数对比**：

| 传感器 | 信号形态 | 空间分辨率 | 采样率 | 力分辨 | 耐久 | 覆盖 | 开源 |
|---|---|---|---|---|---|---|---|
| **GelSight Mini** | RGB光流 | ~10-30µm | 30-60Hz | ~mN | 中（凝胶磨损）| 指尖 | 商用 |
| **DIGIT 360** | 多模态图像+音频+压力 | 7µm,>800万taxels | 60Hz+音频 | 1mN | 中 | 指尖全向 | 将开放 |
| **GelSight Svelte** | RGB+曲率力 | ~10µm | 30-60Hz | ~mN | 中 | 人指曲面+连杆 | 设计公开 |
| **DIGIT Pinki** | 光纤束RGB | 0.22mm | ~30Hz | 5mN法向+剪切 | 中 | 微型指尖 | 是 |
| **AnySkin** | 磁强计向量 | mm级（稀疏） | ~100Hz | 未标定mN | **高**（秒换装）| **全手/全臂可铺** | **是（全开源）** |
| **eFlesh** | 磁强计向量 | mm级 | ~100-500Hz | 未标定 | **高** | 3D打印任意形状 | **是（全开源+商用板）** |
| **XELA uSkin** | 三轴力阵列 | 每模块≤64点 | 70Hz CAN | 标定mN | 中-高 | 全手覆盖 | 商用 |
| **3D-ViTac (FlexiTac)** | 压阻阵列 | 3mm²/单元 | ~30Hz | 低精度 | 中 | 可缠指 | **是** |

**人形手选型推荐**：
- **指尖**：DIGIT 360 / GelSight Mini（高分辨率，精细纹理+滑移+3D几何）
- **手掌+近端指节+手臂**：eFlesh / AnySkin / XELA uSkin（大面积覆盖、高带宽、耐磨、低维 = 接触检测 + 碰撞安全）
- **腕部**：ATI Varo / Nano17 6轴F/T（作力控ground truth基准，可选）

### 3.2 触觉仿真（Tactile Simulation）

| 工具 | 年份 | 方法 | 速度 | 集成 | 开源 |
|---|---|---|---|---|---|
| **TACTO** | 2020 | OpenGL/PyRender渲染 | 数百fps | PyBullet | 是 |
| **Taxim** | CMU 2021 | example-based多项式查表映射形变→像素 | CPU在线，像素误差最低 | 独立 | 是 |
| **TacSL** | NVIDIA 2025 | Kelvin–Voigt接触+GPU渲染 | **比前SOTA快200×** | **Isaac Lab一等公民传感器** | 是 |
| **DiffTactile** | 2024 | 物理可微FEM/可微接触 | 准确但慢 | 独立 | 是 |

**关键空白**：VBTS 仿真已成熟（TacSL已是Isaac Lab一等公民），但**磁性/电容皮肤的仿真器几乎不存在**——大多工作只覆盖 VBTS。这正是可做方向。

---

#### 实施方案 G：磁性皮肤的 sim2real 仿真器

**问题**：磁性皮肤（AnySkin/eFlesh/XELA）没有成熟仿真器，无法在仿真中训策略再 sim2real。

**方法流程**：

```
采集真实标定数据:
  eFlesh 皮肤 + MLX90393 磁强计板（全开源、CAD参数化）
  用3D打印已知几何探头以已知力按压标定网格
  记录: 磁强计读数(5路×3轴=15维) + ATI Nano17 6轴F/T真值
  采样: ~200Hz，覆盖法向/切向/多点/边缘

训练神经网络标定模型:
  输入: 磁强计各轴读数（15维）
  输出: 接触力场（接触位置+法向力+切向力共5维）
  架构: 可逆MLP或条件normalizing flow（输出力→磁场的inverse模型）
  Loss: MSE(预测力, ATI真值) + 物理正则（力方向应与表面法向一致）

嵌入仿真器:
  MuJoCo/Isaac 中模拟接触 → 软体FEM形变 → 磁偶极子正演模型（形变→磁化分布变化→磁强计处磁场）
  → 用真实标定数据拟合磁化参数（磁铁强度/退磁/温度系数）
  → 域随机化：扰动磁化强度/温漂/安装偏移以覆盖制造差异

sim2real 验证:
  仿真训练策略 → 直接部署真实 eFlesh + 机械手
  测预测磁场 vs 真实磁场的 RMSE
  测操作成功率（抓取/旋转）的 sim2real 迁移率
```

**难点与解法**：
- 磁化参数辨识：用标定探头模拟已知力 × 不同位置，反解磁偶极子等效参数
- 温漂模拟：在域随机化中加入高斯温度漂移项，幅度以真实温漂测量为准
- 多传感器串扰：相邻磁铁相互影响→GNN建模邻域磁场

**评测**：
- 仿真触觉读数 vs 真实触觉读数的 RMSE（不同接触位置/力度的误差分布）
- sim2real 迁移成功率（对比纯仿真无触觉 vs 仿真有触觉 vs 仿真+域随机化 vs 真数据）

---

### 3.3 空间智能与 3D 表征的科学问题

> 当前 3DGS/NeRF 在机器人中的应用主要是"数据增强"（RoboSplat, GenSplat）。本节追问：**3D 空间表征能成为策略的内在组成部分，而不仅仅是数据增强工具吗？**

#### 3.3.1 3D-Native 策略：从 2D 图像映射到 3D 空间推理

**背景**：
当前所有 VLA/扩散策略本质上是 2D 图像的映射——策略学到的"抓取"是特定视角下的 2D 模式，而非真正的 3D 物理交互。RoboSplat（RSS 2025）用 3DGS 渲染新视角来增强训练数据，GenSplat（2026.03）用 3DGS 做视角泛化——但 3D 始终是策略的外围工具。

Canonical Policy（2025.05）和 Point Bridge（2026.01）开始将策略建立在 3D 表征之上——点云/规范坐标下的 equivariant 策略——但在真机场景下，3D 点云的实时获取和质量仍然是瓶颈。

**核心 Insight**：
**3DGS 的 Gaussian primitives 天然适合作为策略 Transformer 的输入 token。** 每个 Gaussian 不仅携带空间位置和几何信息，还携带从 VLM 蒸馏的语义特征。与 2D image patches 作为 token 相比，3DGS tokens 具有两个本质优势：(a) 视角不变性——无论从哪个视角渲染，同一物体的 Gaussian tokens 的 3D 位置是相同的；(b) 物理可解释性——Gaussian 的空间关系直接编码了物体之间的物理关系（支撑、接触、遮挡）。

**科学问题**：
1. 3DGS token 的数量（通常 10^4-10^6）远超 2D image token（~256-1024）——如何高效地对 3DGS token 做 attention？（空间邻近性加权的稀疏 attention 是一个方向）
2. 3DGS 的 latent feature 来自 2D VLM 的蒸馏——蒸馏过程会丢失什么信息？
3. 动态场景中 3DGS 的实时更新——每次观测后如何增量更新 3DGS 而不需要重新优化？

**可做的工作**：
- **3D-Aware Policy Transformer**：输入 3DGS tokens（位置 + 协方差 + 语义特征），attention 权重同时依赖 feature similarity 和 3D 空间邻近性
- **3D Affordance as Attention Map**：语言指令通过 cross-attention 查询 3DGS tokens，产生 3D affordance heatmap——天然处理遮挡和视角不变性
- 与 ProDex（2025.11）的 6D pose 预测结合——3DGS 提供场景级几何，ProDex 提供物体级精确位姿
- 产出：**一个基于 3DGS 作为内在表征的操控策略，在极端视角变化下保持鲁棒**


#### 3.3.2 3DGS 从"静态重建"到"动态物理预测"

**背景**：
当前 3DGS 的机器人应用几乎全部是静态场景重建——建完场景后做数据增强。但 PointWorld 展示了"预测未来的 3D 点流"的可能性。关键空白：3DGS 的 dynamic prediction——不仅重建当前场景的 3DGS，还预测未来 N 帧的 Gaussian 位置变化。

**核心 Insight**：
**将 PointWorld 的"3D 点流预测"与 3DGS 的"高保真几何+外观"结合——WAM 在 3DGS 的 latent space 中预测每个 Gaussian 的位移、旋转、伸缩和外观变化。** 与 latent pixel-space video prediction 相比，3DGS-level prediction 的优势在于：Gaussian 的运动是稀疏的（只有移动的物体/机器人的 Gaussians 会变化），且物理上更可解释（Gaussian 的运动应满足刚体约束）。

**科学问题**：
1. 如何对 3DGS 的 Gaussians 做"分组"——同一刚体的 Gaussians 应作为整体运动？（semantic segmentation + optical flow 可辅助）
2. 动态 3DGS 预测的动作条件化：给定机器人动作 a_t，预测哪些 Gaussians 会如何移动——如何从数据中学到这个映射？
3. 预测的 3DGS 可用于"想象"——在想象的 3DGS 中评估候选动作的后果

**可做的工作**：
- **Dynamic 3DGS World Model**：基于 Action-Conditioned 3DGS Frame Prediction
- 从 robot-centric RGB-D 视频中自监督学习：给定历史 3DGS + 机器人动作 → 预测未来 3DGS
- 利用 3DGS 的显式几何做物理约束：预测的 Gaussian 运动应满足刚体假设（同一物体的 Gaussians 做 SE(3) 变换）
- 产出：**一个物理可解释的 3D 世界模型，比 latent pixel-space prediction 更精确、更可解释**


---

## 第四部分　长时序决策——分层、记忆、世界模型推演

### 4.1 长时序任务分解与分层架构

分层架构的谱系按**高低层接口形态**分三条路：

| 路线 | 接口形态 | 代表工作 | 优点 | 缺点 |
|---|---|---|---|---|
| **路线1：LLM/VLM + 离散技能库** | 语言指令/技能ID | SayCan, Hi Robot, RoboBrain | 可解释、可纠错 | 语言信息瓶颈、技能库固定 |
| **路线2：Latent 接口** | 可学习连续向量 | LCB, HIQL, Director | 不受离散词表限制、联合训练 | 不可解释、可诊断性差 |
| **路线3：关键帧/子目标** | 关键状态/关键pose | CoA, ChainedDiffuser, Diffusion Forcing | 稀疏表示减误差累积 | 关键帧质量依赖上层 |

#### 代表论文方法级细节

| 工作 | venue | arXiv | 分层结构（架构/loss级别）| 开源 |
|---|---|---|---|---|
| **SayCan** | CoRL 2022 | 2204.01691 | LLM打分p(skill\|instruction) × RL valid value p(success\|skill,state) → 贪心逐步展开长任务。技能库离散且固定。高层不看真实视觉 | 是 |
| **LCB** | 2024 | 2405.04798 ✓ | LLM输出一个可学习latent code（`<ACT>` token的embedding）→ 低层policy以该latent为条件 → **端到端联合训练**。突破：不再被离散词表瓶颈约束 | 是 |
| **Hi Robot** | ICML 2025 | 2502.19417 ✓ | 双系统：S2(VLM)做推理→输出**语言原子指令**→S1(π₀ flow-matching)执行。S2支持人类实时纠正反馈并动态改子目标 | 是 |
| **HIQL** | NeurIPS 2023 | 2307.11949 ✓ | 单一action-free value function同时支撑两层policy——高层把"状态"当动作，预测latent子目标；低层朝子目标执行。关键：用同一个value的层次分解对value噪声鲁棒 | 是 |
| **Director** | NeurIPS 2022 | 2206.04114 | 在世界模型**内部**做分层——manager在latent空间提出子目标，worker在世界模型想象中达成。高层在低频抽象时间尺度规划 | 是 |
| **Diffusion Forcing** | NeurIPS 2024 | 2407.01392 ✓ | 训练时给序列里**每个token独立噪声水平**去噪，融合"next-token的可变长度"与"full-sequence diffusion的未来引导"。支持**可变horizon、稳定的长程rollout** | 是 |

#### 方法级核心痛点

1. **高低层接口形态无定论**：语言（可解释但信息瓶颈）vs latent（连续但不可解释）vs 关键帧（稀疏但依赖生成质量）。人形 whole-body 场景下"移动+操作+平衡"三者的分层接口更无人做
2. **高层规划缺乏物理可行性校验**：SayCan用affordance value（只看历史成功率），RoboBrain靠VLM常识——都不是model-based rollout验证
3. **长程子目标累乘衰减**：即使每个子目标成功率80%，5步子目标成功率仅32.8%

---

### 4.2 记忆机制（Memory for Long-Horizon）

| 工作 | arXiv | 记忆类型 | 机制 | 局限 |
|---|---|---|---|---|
| **MaIL** | CoRL 2024 | SSM隐状态（Mamba backbone） | 选择性状态空间模型做IL policy，线性时间处理长观测序列 | 固定大小隐状态=有损压缩；超长程早期事件仍被遗忘 |
| **EchoVLA** | 2511.18112 ⚠️ 2025 | 声明性双记忆（场景+情景） | 场景记忆=空间-语义map；情景记忆=历史交互/事件。仿人脑"陈述性记忆"结构 | 预印本；记忆写入/遗忘策略待核实 |
| **MindMap** | 2509.20297 2025 | 3D特征图空间记忆 | 历史观测聚合进结构化3D特征图，3D action policy直接读metric-semantic图 | 度量地图随空间增长内存涨；语义特征漂移 |
| **DovSG** | 2410.11989 2024 | 开放词表3D场景图 | 从RGB-D用VLM构建动态open-vocabulary 3D场景图（物体节点+关系），LLM在图上做长程规划 | 语义误差累积；大场景图检索/维护开销 |

**核心空白**：**记忆与世界模型 rollout 的耦合**——用记忆里的语义/空间状态约束想象rollout（防止想象"忘掉"早期场景），几乎无人做。

---

### 4.3 长时序误差累积的解决方案对比

| 方法 | 思路 | 核心机制 | 成熟度 |
|---|---|---|---|
| **Value Bootstrap**（DreamerV3/TD-MPC2） | 用critic的TD value截断想象horizon | rollout N步→用V(s_N)兜底→不需rollout到终点 | **成熟** |
| **分层抽象**（Director） | 高层在低频抽象时间尺度规划 | 子目标在latent空间 → 低层在子目标间高频执行 | 已研究但长程稳定性未解决 |
| **关键帧生成**（CoA/ChainedDiffuser） | 生成稀疏关键帧而非每帧 | 第一个token=目标关键帧，后续反向生成整条轨迹 | 有方法新意但真机验证有限 |
| **Diffusion Forcing** | 可变horizon + 未来引导 | 每个token独立噪声水平 + Monte-Carlo guidance | **2024新范式** |
| **守恒律正则**（Morpheus思路） | 在长rollout中约束守恒律 | 在latent rollout上加动量/能量守恒辅助损失 | 评测刚起步，真机无验证 |

---

#### 实施方案 H：基于"局部 Lipschitz 谱"的自适应预测粒度 WAM——预测精度与计算的最优分配

**核心 Insight**：
当前所有 WAM 的预测步长是手工设定的（Director K=8, DreamerV3 K=15）。但这忽略了一个根本事实：**latent dynamics 在不同状态区域的"预测难度"不同。** 在膨胀动力学区域（碰撞瞬间、接触切换），误差指数增长 → 应该跳帧或缩短 horizon；在收缩动力学区域（稳定接触、自由运动），误差自然衰减 → 可以扩大步长。

**理论基础——Latent Jacobian 谱半径驱动的自适应跳帧**：
在 latent dynamics f(z, a) 的局部 Jacobian J_f(z) = ∂f(z,a)/∂z 的谱半径 ρ(J_f) 决定了一步预测误差是被放大还是缩小：
- ρ > 1 + δ: 膨胀区域 → 跳帧（预测 z_{t+K} 而非逐帧 K 步），或缩短预测 horizon
- |ρ - 1| ≤ δ: 中性区域 → 标准步长
- ρ < 1 - δ: 收缩区域 → 扩大步长（节省计算，且精度不降）

**关键创新**：不是手工设定步长，而是**让 WAM 学习一个"步长策略"** k_φ(z_t) 输出在当前 latent state 下的最优预测步长。这个步长策略由"预测误差"和"计算成本"联合优化。

**方法架构**：

```
组件1 — 谱半径估计器 (无需完整 Jacobian):
  ρ̂(z_t) = 通过 power iteration 在随机方向 v 上估计:
    J_f(z_t) · v ≈ (f(z_t + ε·v, a_t) - f(z_t, a_t)) / ε
    ρ̂ = (v^T · J_f · v) / (v^T · v)
  → 几次前向传播即可估计，无需反向传播

组件2 — 变分最优跳帧:
  跳帧策略 k_φ(z_t) ∈ {1, 2, ..., K_max}
  优化目标:
    min_φ E_z[ L_pred(k_φ(z)) + λ·C(k_φ(z)) ]
    其中:
      L_pred(k): 跳帧预测 vs 逐帧预测的额外误差
      C(k) = 1/k: 计算成本（步长越大越省计算）
  
  理论保证（核心贡献）:
    在 Lipschitz 连续条件下，自适应跳帧的误差上界 ≤ 
    exp(Σ_i log ρ_i) · ε  vs  固定步长的 exp(K · log ρ_max) · ε
    当动力学在不同区域差异大时，自适应优势显著

组件3 — 误差-计算 Pareto 最优调度:
  训练完毕后，k_φ(z_t) 在运行时自适应选择步长:
    - 碰撞瞬间 (ρ ≈ 10): 切换到 k=1 (逐帧预测，精度优先)
    - 自由运动 (ρ ≈ 0.5): 切换到 k=8 (跳帧预测，速度优先)
  整体效果: 在精度不降的前提下，预测成本降低 3-5×
```

**为什么比原方案好**：
- 原方案 H：手工定义"关键帧"→ 需要 VLM 先定子目标 → 关键帧是人定义的，而非从动力学中涌现的
- 新方案：预测步长由动力学自动决定 → 不需要语义子目标 → 理论上有最优性保证
- 预期效果：长程预测误差降低 30-50%（vs 固定步长），同时推理速度提升 3-5×

**发表潜力**：理论证明 + 实验验证 → ICML/NeurIPS 级别的优化与机器人交叉

**实验验证**：
- 在已知 ground truth 动力学（单摆、双摆、n体）上验证 ρ̂ 的估计精度
- 在复杂仿真（人形行走、双臂操作）上验证：自适应步长 vs 固定步长的预测误差 + 推理时间
- 理论贡献：自适应跳帧 vs 固定跳帧的误差上界比

---

#### 实施方案 I：WAM 的"反事实预演"——在想象中主动搜索失败路径并提前规避

**核心 Insight**：
原方案 I 的"预演式纠错"是被动的——检查"给定的子目标序列是否会失败"。但它不问一个更重要的问题：**在当前不确定性下，最可能失败的反事实路径是什么？** 即：WAM 应该主动搜索"如果事情稍微不同（略大/略小的力、稍早/稍晚的接触），会在哪里失败"。

**关键创新**：引入**Adversarial Latent Rollout（ALR）**——在 WAM 的每一步 rollout 中，不取确定性预测，而是在 noise space 中搜索最坏扰动：z^{adv}_{t+1} = f(z_t, a_t) + δ*，其中 δ* = argmax_{||δ|| ≤ ε(z_t)} (1 - R(z_t + δ))，ε(z_t) 是当前 latent state 的 epistemic uncertainty 半径。

**方法架构**：

```
组件1 — Adversarial Latent Search (ALS):
  给定候选子目标序列和动作计划
  在每个 rollout step t:
    在 latent space 中以当前预测为中心、半径为 ε(z_t) 的球内搜索
    ε(z_t) = ensemble latent dynamics 的预测标准差 (epistemic uncertainty)
    
    δ* = argmax_{||δ|| ≤ ε(z_t)} (1 - R(ẑ_t + δ))
    其中 R(z) = 可恢复性函数（来自方案 L）
    
  ALS 轨迹: z^{adv}_{t+1} = f(z_t, a_t) + δ* → z^{adv}_{t+2} = ...
  
  关键性质: ALS 轨迹是"在模型不确定性范围内，最坏可能情况"

组件2 — Worst-Case Safety Margin (WCSM):
  对每个候选子目标序列 i:
    WCSM(i) = min_{t ∈ [1, T]} R(z_t^{adv}; π, M)  [沿 ALS 轨迹的最小可恢复性]
  
  递进式规划:
    1. VLM 生成 N 个候选子目标序列
    2. 对每个序列做 ALS rollout
    3. 过滤: 保留 WCSM > θ_safe 的序列
    4. 在安全序列中选择任务效率最高（步数最少、能耗最低）的

组件3 — ε(z_t) 的 Conformal Calibration:
  使用 conformal prediction 保证:
    P(真实 latent state ∈ B_ε(z_t)) ≥ 1-α
  由此: 如果 ALS 说安全，则在 1-α 概率下确实安全
```

**为什么比原方案好**：
- 原方案 I：检查"最可能的轨迹是否不安全" → 可能漏掉低概率但致命的情况
- 新方案：搜索"所有可能轨迹中最坏的情况" → 考虑 epistemic uncertainty 范围内的全部可能
- 安全性：Conformal prediction 提供 finite-sample probabilistic guarantee
- 预期效果：安全违规率降低 80-90%（vs 无最坏情况搜索），成功率仅轻微降低（~5%）

**发表潜力**：Adversarial rollout + conformal safety guarantee → RSS/CoRL 安全方向的最佳切入点

---

### 4.4 长时序动作生成

| 工作 | arXiv | 核心机制 | 贡献 |
|---|---|---|---|
| **CoA (Chain-of-Action)** | 2506.09990 ✓ (ByteDance, NeurIPS 2025, **开源**) | **反向推理**：第一个token=目标关键帧动作，后续从目标向当前回溯生成整条轨迹（action-level CoT）。目标锚定减少前向累积漂移 | 颠覆"前向预测下一步"范式 |
| **ARP** | 2410.03132 ✓（开源） | chunking autoregression：块内并行、块间自回归生成end-effector序列 | 平衡chunk平滑性与自回归长程一致性 |
| **Diffusion Forcing** | 2407.01392 ✓（开源） | 每个token独立噪声水平+Monte-Carlo guidance，可变horizon、稳定的长程rollout | 融合"next-token"与"full-sequence diffusion" |

**核心张力**：chunk越长越平滑但越易偏离实时反馈；chunk越短越闭环但长程一致性差。CoA的"目标锚定+反向生成"和Diffusion Forcing的"可变horizon+未来引导"是两条最有方法新意的破局路线。

---

### 4.5 长时序决策的科学问题与研究方向

#### 4.5.1 S1/S2 切换的数学形式化——从手工频率到事件驱动

**背景**：
Ψ0（USC/NVIDIA, 2026.03）、GR00T N1（NVIDIA）、Helix（Figure AI）等双系统/三系统架构均使用**固定频率**进行 S2（慢/语义规划）到 S1（快/动作执行）的通信。但人类不是这样——熟练任务中 S2 几乎不介入，不确定/新奇/危险时刻 S2 立刻抢占注意力。

When to Trust Imagination（2026.05）通过 FFDC 验证器实现了"想象不可信时重新规划"——但这仍然是**被动的、事后检测的**——等发现想象错了，可能已经来不及了。

**核心 Insight**：
**S2 的介入应由"latent divergence 的增长率"而非绝对值触发。** 绝对值高可能只是因为环境嘈杂——"情况不好但稳定"不需要 S2。增长率为正且加速时——"情况正在快速变糟"——才需要 S2。这与控制论中的事件触发控制（Event-Triggered Control, ETC）有直接的数学联系——ETC 的触发条件基于状态误差的导数而非绝对值。

更进一步：S2 的介入应同时有**两个方向**——(a) 偏离触发（"情况正在变糟"→ S2 介入纠正），(b) 机会触发（"出现了更高效的执行方式"→ S2 介入优化）。

**科学问题**：
1. 在 latent space 中，ETC 的最优触发条件是什么？（需要估计 latent dynamics 的局部 Lipschitz 常数来设定安全触发阈值）
2. "机会触发"如何定义和检测？（当前状态与"更优计划"的 latent state 之间的"shortcut"——某个低成本动作可以跳过若干步骤）
3. S2 的介入频率与任务成功率之间的理论 trade-off？（每个 S2 介入都有计算开销——最优介入频率是最小化 total cost = 任务失败成本 + S2 计算成本）

**可做的工作**：
- 将 ETC 理论从已知状态空间迁移到 learned latent space——需要解决"latent space 中的距离/散度与物理状态空间中的安全性之间的对应关系"
- Latent Divergence Monitor：使用 ensemble latent dynamics 估计每步的 latent uncertainty，用 uncertainty 的增长率作为触发信号
- 机会触发的实现：在 S2 的"后台"中，用 VLM 持续评估当前 latent state 与"更优计划的 latent state"的距离——如果发现 shortcut，主动抢占 S1
- 产出：**一个数学上有最优性保证的 S1/S2 自适应切换策略**

**预期发表级别**：RSS / CoRL；理论部分 L4DC


#### 4.5.2 预测粒度的最优性理论——从手工跳帧到自适应预测步长

**背景**：
Director（NeurIPS 2022）使用固定 K=8 步的跳帧，Diffusion Forcing（NeurIPS 2024）使用启发式的噪声 schedule。但这些都回避了根本问题：**预测的时空粒度和误差累积之间的根本关系是什么？什么是最优预测粒度？**

**核心 Insight**：
**最优预测粒度是一个变分问题。** 在 latent dynamics f(z_t, a_t) = z_{t+1} 的局部 Jacobian 谱半径 ρ(J_f(z_t)) 大于 1 的区域（膨胀动力学，如碰撞瞬间），逐帧预测误差被指数放大 → 跳帧预测（直接预测 z_{t+K}）的误差可能**小于** K 步逐帧预测的累积误差。在 ρ(J_f) < 1 的区域（收缩动力学），逐帧预测误差自然衰减 → 不需要跳帧。最优预测步长 k*(z_t) = argmin_k E[||ẑ_{t+k} - z_{t+k}||] 取决于局部 Jacobian 谱。

**科学问题**：
1. 能否在 latent space 中估计 ρ(J_f(z_t)) 而不需要完整的 Jacobian 计算？（随机投影/power iteration 可估计谱半径）
2. 最优跳帧策略 k*(z_t) 是否可学习？（训练一个"跳帧策略网络"输出 k，目标是最小化终点 latent error）
3. 跳帧预测在什么条件下比逐帧预测更差？（当动力学接近线性且 Jacobian 接近 1 时，跳帧无优势）

**可做的工作**：
- 在 latent dynamics 上估计局部 Jacobian 谱半径 → 启发式跳帧策略（ρ > 1+δ → 跳帧）
- 训练一个 light-weight "跳帧网络" k_θ(z_t) → 输出最优跳帧步长
- 核心理论贡献：**证明在 latent dynamics 满足 Lipschitz 连续的条件下，自适应跳帧的预测误差上界严格低于固定步长的预测误差上界**
- 产出：**一个自适应预测粒度的 WAM，比固定步长 WAM（Director/DreamerV3）在长程预测中更精确**


#### 4.5.3 记忆与世界模型 Rollout 的耦合

**背景**：
4.2 节的记忆机制（MaIL, EchoVLA, MindMap）和世界模型 rollout 是**分离的**——记忆存储"过去发生了什么"，世界模型预测"未来会发生什么"。但人类的工作记忆和想象是耦合的——你在想象"走到桌边端水"时，记忆中的"桌上有一个水杯"约束了你的想象。

**核心 Insight**：
**世界模型的想象 rollout 应该被记忆中的"不变量"（场景中不会变化的信息——如房间布局、物体语义类别、过去成功/失败的统计）所约束。** 当前世界模型在 rollout 中可能"幻觉"出不存在的障碍物、忘记环境中已有的物体——这是因为它没有将记忆中的结构化信息作为 rollout 的 persistent condition。

**科学问题**：
1. 如何在 WAM 的 latent rollout 中注入"来自记忆的结构约束"——使某些 latent 维度在 rollout 中保持不变（持久物体的位置不应在 rollout 中漂移）？
2. 记忆中的哪些信息对想象 rollout 是必要的？（场景布局 → 必要；墙的颜色 → 不必要）——如何自动区分？
3. 当新观测与记忆冲突时，是更新记忆还是怀疑观测？（这本质上是 Bayesian surprise 的问题）

**可做的工作**：
- **Memory-Conditioned WAM Rollout**：在每个 rollout step，latent dynamics 的条件不仅包括 z_t 和 a_t，还包括从记忆中检索的"不变量"——通过 attention 或 FiLM 调节
- 记忆结构：3D 场景图（从 DovSG 风格构建）→ 场景级约束（哪些物体在哪里）+ 情景记忆（历史交互成败）
- 记忆一致性损失：L_consistency = ||z_t^{spatial} - M_{spatial}(z_t^{ego})||² ——强制 latent state 的空间成分与记忆中的空间图一致
- 产出：**一个"不会在想象中忘记场景"的世界模型**


---

## 第五部分　自主进化 / 自我改进——像人一样思考与成长

### 5.1 自主进化的五个闭环环节

把几乎所有"自我改进"系统拆开，都在解同一个五环闭环：

```
  ① 探索/自主采集 ──→ ② 自动奖励/成功判定 ──→ ③ 自动重置 (reset-free)
       ↑                                              ↓
       └──── ⑤ 不遗忘地累积 ←──── ④ 策略提升 ←────────┘
```

**当前真相**：没有任何一个系统把五环全部闭合且无人介入。

| 环节 | 最成熟方案 | 核心未解问题 |
|---|---|---|
| ① 自主采集 | Plan2Explore (latent disagreement好奇心) | 噪声电视问题；人形上的安全探索 |
| ② 自动奖励 | SEFM steps-to-go / VLM reward | reward hacking；接触/力觉盲区 |
| ③ 自动重置 | SERL forward-backward 双策略 | 人形摔倒不可逆——如何安全 reset-free |
| ④ 策略提升 | Dreamer latent rollout / SERL off-policy RL | 长程误差累积→策略在幻觉上过拟合 |
| ⑤ 抗遗忘 | LoRA-expert库 / 参数隔离 | task-free流式（无明确任务边界）几乎无解 |

---

### 5.2 代表方法谱系

**真机 RL / 自我改进**：

| 工作 | 年份 | arXiv | RL算法 | 奖励来源 | 重置方案 | 真机训练耗时 | 开源 |
|---|---|---|---|---|---|---|---|
| **SERL** | ICRA 2024 | 2401.16013 | RLPD（off-policy SAC变体，demo+在线数据混buffer）| 二分类成功检测器 | forward-backward双策略 | 15-60分钟（单臂桌面） | 是 |
| **HIL-SERL** | 2024 | 2410.21845 | RLPD + 人在回路 | 成功检测器+人干预 | 人重置 | **1-2.5小时**（双臂装配等） | 是 |
| **DayDreamer** | CoRL 2022 | 2206.14176 | Dreamer（RSSM+想象中actor-critic）| 手工/状态奖励 | 手工/无 | **1小时**（四足从零行走） | 是 |
| **RoboFuME** | 2023 | 2310.15145 | 离线预训练+在线reset-free微调 | 预训练VLM奖励 | forward-backward | **3小时**（新任务） | 是 |

**世界模型驱动的自我改进**：

| 工作 | arXiv | 机制 |
|---|---|---|
| **Plan2Explore** | ICML 2020 | 2005.05960 ✓ | K个模型的ensemble，**latent disagreement作内在奖励**；关键：在**想象中**优化探索策略去最大化预期disagreement——零样本适应下游 |
| **World Models Increase Autonomy in RL** | 2408.09807 ✓ | 直接论证WM能提升reset-free自主性，不需要环境奖励/demo就超过需特权信息的基线 |
| **RND** | 1906.04161 ✓（背景）| 预测 fixed random network 输出的误差当好奇心奖励 |

**数据飞轮 / 自我改进**：

| 工作 | 年份 | arXiv | 飞轮机制 | 关键创新 |
|---|---|---|---|---|
| **RoboCat** | TMLR 2024 | 2306.11706 | 100-1000条demo微调→策略自己采几万条rollout→已学目标达成判据自动筛→回灌训练集→下一代 | 多本体统一消化；证实"data flywheel"有效 |
| **SEFM** | NeurIPS 2025 | 2509.15155 ✓ | 阶段一SFT时额外学**steps-to-go预测**；阶段二从steps-to-go中抽出(1)稠密奖励(2)成功检测器 → **一队机器人无需手工奖励就能自主练习+在线RL** | 自动奖励=基础模型内生能力！对人形最重要的一篇 |
| **Eureka** | ICLR 2024 | 2310.12931 ✓（开源）| GPT-4写奖励函数代码→GPU并行sim评估→**reward reflection**（训练统计反馈给LLM进化迭代）→灵巧手转笔超过人奖 | 自动奖励生成=LLM写代码 |
| **DrEureka** | RSS 2024 | 2406.01967 ✓（开源）| Eureka+LLM同时产奖励+**自动域随机化分布设计** | sim-to-real自动化 |
| **RoboGen** | ICML 2024 | 2311.01455 ✓（开源）| LLM提任务→生成场景→生成训练监督→RL求解，4阶段全自动，"无限"产技能 | 全自动任务生成 |

**从失败中学习**：

| 工作 | arXiv | 机制 | 失败→修复路径 |
|---|---|---|---|
| **REFLECT** | CoRL 2023 | 2306.15724 ✓（开源）| 多模态传感→层级化文本经验摘要→LLM定位失败在哪一步、为什么→生成纠正计划 | 符号/计划层修复 |
| **Motion Self-Reflection** | 2504.14588 2025 | 把语义反思翻译成**细粒度运动层纠正**（how to correct motion，非只说what went wrong） | 运动层修复 |
| **Reflection VLA Self-Improve** | 2510.12710 2025 | VLA自改进时**选择性模仿高质量成功轨迹+条件式课程逐步加难** | 筛选+课程 |
| **Reverse Curriculum** | CoRL 2017 | 1707.05300 | 从目标态"倒着"扩展起始分布，只需一个成功态 | 课程生成 |

**持续/终身学习**：

| 工作 | arXiv | 抗遗忘机制 |
|---|---|---|
| **Voyager** | 2023 | 2305.16291 ✓（开源）| 技能库存为可执行代码+文本描述embedding当key；新任务检索复用+组合；迭代prompting自我纠错。无梯度更新即终身技能积累 |
| **PEFT Expert Library** | 2506.05985 2025 | 渐进式LoRA experts动态混合：新技能=新expert，旧expert冻结防遗忘。在LIBERO上超SOTA |
| **Nature MI Lifelong RL** | 2025 | 期刊 | 面向真机器人的终身RL框架，强调保留旧知识同时**组合**已有知识学新技能 |

---

### 5.3 核心空白总结

**单环都有解，全闭环 + 人形 + 无人监督 + 安全无解**。具体到每个环节：

- **自动奖励**：SEFM 虽然 clean，但 steps-to-go 在长程/多解任务上定义模糊
- **自动重置**：人形摔倒不可逆——reset-free RL 在人形上从未做过（DayDreamer是四足）
- **想象提升**：Dreamer 成熟但长程 rollout 误差累积导致策略在幻觉上过拟合；**世界模型的"不确定性校准"用于安全门控（什么时候该相信想象、什么时候回真机）几乎空白**
- **数据飞轮**：弱策略自标注→偏差自强化→飞轮发散，无理论防护机制
- **人形安全探索**：基本空白——高维欠驱动本体上无人监督的安全试错完全未解

---

#### 实施方案 J：双重控制驱动的"想象-真机"自适应调度——从固定门控到最优元认知策略

**核心 Insight**：
原方案 J 的 uncertainty-gated 飞轮使用固定阈值 θ₁, θ₂ 来决定"想象 vs 真机 vs 停止"。但最优的切换策略不是固定阈值——它取决于：(a) 模型改善的边际收益，(b) 真机采样的成本与风险，(c) 任务完成的急迫性。这本质上是一个**有代价、有延迟、有信息增益的序贯决策问题**（POMDP）。

**关键创新**：将{想象训练, 真机采样, 安全停止, 恢复, 重规划}的切换形式化为 POMDP，其最优策略即为模型的**元认知策略**——它知道什么时候该相信自己、什么时候该去真机学习、什么时候该认输。

**方法架构**：

```
元认知 POMDP 定义:
  状态 s = (latent belief b_t, task_progress, model_confidence, safety_risk)
  动作 A = {imagine_K_steps, collect_real, halt, recover, replan}
  奖励:
    +R_success: 任务成功
    -C_real: 每次真机采样（时间和物理磨损成本）
    -C_halt: 安全停止（任务延迟成本）
    -C_crash: 不可恢复失败（巨大的惩罚，确保模型学会避免）

元认知策略 μ_φ(s) → a 的训练:
  阶段1 (仿真): 在仿真中大量采样不同 s 下的最优动作选择
    → 训练"元认知网络" μ_φ(s)
    利用: 仿真中可以安全地探索"选择错误"的后果
  阶段2 (真机): 在线 fine-tune μ_φ
    每次做出选择后，观察到后果 → 更新策略
    随着部署次数增加，元认知越来越准

自动涌现的行为（无需手工规则）:
  - model_confidence 高 + task_progress 快 → 选择 imagine (省真机)
  - model_confidence 低 + safety_risk 低 → 选择 collect_real (主动学习)
  - model_confidence 低 + safety_risk 高 → 选择 halt (安全优先)
  - 摔倒后 → 选择 recover → 成功后重新选择 replan
```

**为什么比原方案好**：
- 原方案 J：固定阈值 → 手工设定，无法适应不同任务/阶段
- 新方案：元认知 POMDP → 最优策略自动涌现 → 理论上保证了"最优调度"
- 额外的科学贡献：首次将"world model usage"形式化为序贯决策问题
- 预期效果：在相同成功率下，真机采样次数降低 60-80%；在相同采样预算下，成功率提升 30-50%

**发表潜力**：POMDP + learned world model → ICML/NeurIPS 的 meta-learning 交叉

**开源基座**：DreamerV3（WM想象）、SERL（真机RL管线）、FAIL-Detect（OOD检测+共形预测）、HOST/HumanUP（人形自恢复）

---

#### 实施方案 K：Continuous Skill Manifold——从离散技能库到可微技能流形

**核心 Insight**：
原方案 K 的"技能库 + LoRA expert"继承了 Voyager 的离散技能范式。但连续控制技能不是离散的——"用 5N 抓取"和"用 6N 抓取"不是两个不同的技能，而是同一技能的连续参数化变体。**技能空间应该是一个连续流形**，技能的"泛化"在这个流形上做插值或外推。

**关键创新**：将技能库重新定义为 **Continuous Skill Manifold（CSM）**——每个"技能家族"是一个低维 Riemannian 流形（维度 d=2-5），由物理参数（力、速度、接触点位置等）自然参数化。流形上的点对应具体的动作分布。新任务的零样本泛化 = 在流形上沿 geodesic 外推到目标区域。

**方法架构**：

```
组件1 — Skill Manifold Learning:
  对于技能家族 k（如"抓取-圆柱体"）:
    收集该技能在多种参数下的 demo 轨迹
    VAE 学习 latent manifold M_k:
      encoder: (τ_demo, param) → w_k ∈ R^d  (d = 2-5)
      decoder: w_k → action_distribution + (μ_success, σ_success)
    
    M_k 是 d 维 Riemannian 流形，其 metric 由 decoder Jacobian 定义:
      g_{ij}(w) = <∂decoder/∂w_i, ∂decoder/∂w_j>

组件2 — Geodesic Skill Generalization:
  给定新任务（force=7N 抓取 = 训练数据中未见参数）:
    1. VLM 确定技能家族 k 和大致参数区间
    2. 在 M_k 上定位"最近已学点" w_known（如 force=5N, 10N 都已学过）
    3. 沿 M_k 的 geodesic 方向外推到 force=7N → w_new
    4. decoder(w_new) → 新参数下的动作分布 → 零样本执行
  
  关键性质: 流形外推利用"force 是连续参数"这一物理事实
    → "5N 和 10N 之间的 7N"在流形上是有意义的中间点

组件3 — 流形持续扩展:
  执行 w_new 后:
    成功 → 将新的 (τ, param_new) 加入训练集 → M_k 扩展
    失败 → 在 w_new 附近收集 1-2 demo → 局部 fine-tune → 快速适应
```

**为什么比原方案好**：
- 原方案 K：离散 LoRA expert → 新参数必须被显式训练过 → 泛化靠检索
- 新方案：连续流形 → 泛化靠 geodesic 外推 → 零样本泛化到未见参数
- 预期效果：新参数的零样本成功率从 ~20%（离散检索）提升到 ~60-80%（流形外推）
- 发表潜力：Riemannian geometry + skill learning → ICML/NeurIPS 新颖方向

**实验验证**：
- 抓取任务：训练参数 = {3N, 5N, 8N, 10N}，测试参数 = {4N, 6N, 7N, 9N}
- 对比：离散 LoRA（检索最接近的训练参数）vs 流形外推 vs 从头训练
- 核心指标：零样本泛化成功率 + 外推距离 vs 成功率的衰减曲线

---

#### 实施方案 L：自指涉安全——从"外部安全滤波"到"策略内生可恢复性感知"

**核心 Insight**（这是整个框架中**最重要、最新颖的方向**）：
原方案 L 的 recoverability_score 是一个固定的、外部训练的分类器——它不随策略改进而更新。但**安全不是状态的函数，而是（状态 + 策略能力 + 世界模型）的联合函数**——同一个状态，对熟练策略是安全的，对稚嫩策略是不安全的。

**关键创新**：定义**自指涉可恢复性函数** R(z; π, M)，具有以下根本属性：
- R 随策略 π 的训练更新——π 改进后，R(z) 自然增大
- R 是连续的——不是"安全/不安全"的二元分类，而是"离不可恢复边界还有多远"
- **策略不在"任务"和"恢复"之间切换——它始终执行同一个 R-conditioned 策略，行为随 R(z) 连续变化**

**方法架构**：

```
组件1 — Self-Referential Recoverability Estimator (SRRE):
  训练 R_θ(z; π, M):
    对每个 latent state z，在仿真中执行 N 步 rollout
    如果能在 N 步内回到稳定姿态 → recoverable
    重复 M 次 → R(z) = (successful recoveries) / M
  训练: L_SRRE = MSE(R_θ(z), R_true(z)) + λ · max(0, R_θ(z) - R_true(z))²
    [悲观偏置: 宁可低估 R（假危险），不可高估 R（假安全，致命）]

组件2 — R-Conditioned Single Policy（核心创新）:
  策略 π(a | z, R_θ(z)):
    R_θ(z) 作为策略的条件输入（通过 FiLM 调制）
  
  训练: 在仿真中，将 R_θ(z) 作为额外条件训练策略
    策略学会: 当 R 高 → 激进高效；当 R 低 → 保守安全
  
  关键性质: 不存在"任务模式"和"恢复模式"的切换
    → 恢复不是外部接入的"安全模式"，而是策略内在的连续属性

组件3 — R-Driven Curriculum:
  初始: 在高 R 区域（简单安全）训练策略
  逐步: 采样 R 接近但非零的状态（"快回不来但还能回来"）
        → 这里梯度信息最丰富 → 策略在边界上改进 → R 边界向外移动
  课程由 R 边界自动推进——不需要手工设计

组件4 — Conformal Safety Guarantee:
  真机上校准 R_θ 输出:
    P(R_true(z) ≥ R_θ(z) - δ) ≥ 1-α
  由此: R_θ(z) > θ + δ → 真实 R > θ，以 1-α 的置信度 → 安全
```

**为什么比原方案好**：
- 原方案 L：固定 recoverability classifier + 双策略（forward/backward）→ 安全是外部模块
- 新方案：自指涉 R(z; π, M) + R-conditioned single policy → 安全是策略内在属性
- **范式级创新**：从"外源性安全"到"内生性安全"——安全从策略内部自然产生
- 理论深度：HJ reachability × conformal prediction × curriculum learning 的三重交叉
- 预期效果：摔倒率降低 90%+（vs 无安全感知），同时成功率不降低（无需保守的二元切换）

**发表潜力**：
- **整个框架中最具发表潜力的方向**——可同时投 RSS/CoRL（系统+实验）+ ICML/NeurIPS（理论部分）
- "自指涉可恢复性"是一个概念层面的创新，从根本上改变了"安全"在机器人学习中的定义

**实验验证**：
- 仿真中验证：R_θ(z) 的估计精度 vs 真实 roll-out 统计 → 校准 VS 未校准
- R-conditioned vs 双策略（forward+backward）→ 摔倒率 + 任务成功率
- R-driven curriculum vs 任务难度 curriculum → 学习效率
- 真机 G1 人形：基础站立扰动恢复 + 行走中避障 → 安全违规次数

**开源基座**：HOST/HumanUP（自起身策略）、SERL（真机 RL）、DreamerV3（WM）、FAIL-Detect（共形预测）

---

### 5.4 自主进化的科学问题与研究方向

> 本节从科学问题层面追问自主进化中尚未被系统解决的根本性挑战。

#### 5.4.1 自指涉安全——从"外部安全滤波"到"策略内生的安全感知"

**背景**：
当前机器人安全的几乎所有方法（SHIELD, Latent Safety Filter, LatentCBF）都遵循一个模式：**安全是外部施加的约束或后处理模块。** 方案 J（不确定性门控）和方案 L（安全 reset-free）的 recoverability_score 仍是"外部"训练的——它是一个独立的分类器，而非策略内在属性。

**核心 Insight**（来自 WAM 研究框架 Q4）：
**安全不是一个状态的函数，而是（状态+策略能力+世界模型）的联合函数。** 同一个状态（如单脚站立），对熟练的策略是安全的（能恢复），对稚嫩的策略是不安全的（会摔倒）。这意味着**安全定义应该是自指涉的**——给定当前策略 π 和世界模型 M，从状态 z 出发，策略能否在未来 N 步内回到稳定区域？

定义**自指涉可恢复性函数**：
$$R(z; \pi, M) = \mathbb{P}\left(\exists \{a_t, ..., a_{t+N}\} \sim \pi \text{ s.t. } z_{t+N} \in S \mid z_t = z\right)$$

其中 S 是稳定姿态集合。R(z) ∈ [0,1] 的梯度方向指向"更不可恢复"，边界 {z: R(z)=0} 是死亡边界——策略无法恢复的状态。

**与方案 L 的关键区别**：
- 方案 L 的 recoverability_score 是**固定分类器**——训练后不变，不随策略改进而更新
- 自指涉 R(z; π, M) 是**随策略训练动态变化的**——策略改进后，原来不可恢复的状态变得可恢复
- 恢复行为不需要外部触发——策略在 R(z) 下降时不是在"任务"和"恢复"之间**切换**，而是在**同一个 latent space 中重新规划**

**科学问题**：
1. 如何高效估计 R(z; π, M)？直接 N 步 rollout 计算量极大。HJ reachability 通过解 PDE 提供了一条路，但需从"给定失败集的前向分析"改造为"给定策略的自指涉分析"。
2. R(z) 的估计如何具有悲观校准保证？——即 P(真实可恢复性 ≥ 估计值) ≥ 1-δ ——以确保安全？
3. Near-failure 经历如何最大化地被利用？——在 R(z) 接近 0 的边界上，一点动作差异就决定恢复与否 → 梯度信息最丰富

**可做的工作**：
- **Self-Referential Recoverability Estimator (SRRE)**：基于 WAM latent space，训练 R_θ(z; π) 预测可恢复性
- **Pessimistic Calibration**：使用 conformal prediction 提供有限样本保证
- **R(z)-Conditioned Policy**：策略以 R(z) 为条件——当 R(z) 低时自动采取更保守的动作；当 R(z) 高时可以激进
- 核心产出：**第一个不需要外部安全模块的"内生安全"人形策略——安全是策略在 latent space 中自然感知到的属性**
- **这是整个框架中最重要、最新颖、最具发表潜力的方向之一**

**预期发表级别**：RSS / CoRL（系统+实验）；ICML / NeurIPS（理论部分）


#### 5.4.2 从 Near-Failure 中最大化信息提取

**背景**：
方案 K（技能库+归因修复）和方案 L（安全 reset-free）都涉及从失败中学习——但"失败"被定义为"任务未完成"或"摔倒"。然而，**差一点失败但救回来的经历（near-failure）** 比成功执行标准任务的经历携带着更多的信息——因为这里"一点动作差异就决定恢复与否"。

现有方法（DEARRL, CEBP, Reset-Free RL）都是**被动利用** near-failure——只在发生时处理。没有人**主动创造** near-failure 来进行学习。

**核心 Insight**：
**Near-failure 经历的信息量在数学上可以由策略梯度的方差来刻画。** 在可恢复性边界 R(z) ≈ 0 附近，策略梯度 ∇_θ J(π_θ) 的方差最大——因为动作的微小变化可以导致"成功恢复"vs"彻底失败"的天壤之别。这恰好对应了最大的 Fisher 信息量——**near-failure 是天然的"高信息量训练样本"**。

**科学问题**：
1. 能否在安全的仿真环境中有意引导策略走向 R(z) 接近 0 但不越过边界的区域？
2. Near-failure 经历的最优利用——对比学习（"成功恢复 vs 导致失败的微小差异"）vs 因果学习（"动作差异→结果差异"的因果图）？
3. 如何在真机上安全地探索 near-failure 区域？（软垫环境、安全 harness、逐步降低安全边际）

**可做的工作**：
- **Recoverability-Driven Curriculum**：不是在越来越难的任务上训练，而是在 R(z) 越来越低的状态上训练
- 在仿真中验证：沿着 R(z) 的梯度方向（更不可恢复的方向）走，直到 R(z) ≈ 0.1，然后在此训练策略
- 每次 near-failure 经历都作为 R(z) 的修正信号——实际可恢复 vs 预测可恢复的差异 → 改进 R_θ
- 产出：**一个"在摔倒的边缘学习如何不摔倒"的训练范式**


#### 5.4.3 反事实数据增强——用世界模型生成"所有可能的失败"

**背景**：
方案 J 和方案 L 在"想象"中训练策略——但世界模型生成的是"在当前策略下最可能发生"的轨迹。它不会生成"如果抓取力小了 10%，物体会滑脱"的反事实轨迹。WoVR（Tsinghua, 2026.02）和 World-VLA-Loop（NUS, 2026.02）都是在"最可能"分布中训练。

**核心 Insight**：
**世界模型的最大价值不是生成"最可能"的轨迹，而是生成"不太可能但物理上可能"的反事实轨迹。** 这些轨迹携带了安全性和鲁棒性的关键信息。"如果抓取力小了 10%，物体会滑脱"——这种反事实从真机数据中几乎不可能采集（因为真机策略不会故意犯错），但从物理精确的世界模型中可以大规模生成。

**可做的工作**：
- **Counterfactual Trajectory Generation**：从一条成功的真机轨迹出发，在 WAM latent space 中沿"动作扰动方向"做反事实 rollout
- **Counterfactual-Aware RL**：策略在真实数据+反事实数据上联合训练——反事实数据的训练目标不是"模仿成功"，而是"学会从失败中恢复"
- **Safety-Critical Counterfactuals**：重点生成"安全关键"反事实——那些可能导致摔倒/碰撞的轨迹——在真实世界中不可能被采集，但在世界模型中完全安全
- 产出：**一个无需真机摔倒就能学会如何避免摔倒的训练方法**


---

### 5.5 替代架构范式——超越扩散/流匹配

> 当前 VLA/WAM 几乎全部基于扩散模型或流匹配。本节讨论两种替代架构及其科学机会。

#### 5.5.1 Masked Generative Models 用于人形控制

**背景**：
Masked Generative Policy（MGP, ICLR 2026）首次将 BERT 式的 masked modeling 应用于模仿学习，实现了比扩散策略快 35x 的推理。DiscreteRTC（2026.04）展示了离散扩散的天然异步执行能力。但在人形 40+ DoF 全身控制上，这个方向完全空白。

**核心 Insight**：
**Masked generative models 天然适合人形全身控制——因为人形动作的不同组成部分可以在不同步数中被"解码"。** 下肢的步态模式（高度可预测、时间周期性）可以在前几步被确定；手指的精细操作（依赖接触反馈、不确定性高）可以在更多步中逐步细化。这种"渐进式解码"匹配了人形控制的层级特性。

**可做的工作**：
- **Hierarchical Masked Policy（HMP）**：动作 tokenization → 层级 mask scheduling（粗 token 先解码、细 token 后解码）
- **Reactive Inpainting**：检测到即将失衡时，mask 当前动作的"不稳定部分"并 inpainting 替代动作
- 核心优势：推理速度（35x faster than diffusion → real-time 人形控制）+ 天然异步执行能力
- 产出：**第一个面向人形全身控制的 masked generative policy**


#### 5.5.2 JEPA 世界模型的人形扩展

**背景**：
LeWorldModel（Meta FAIR/LeCun, 2026.03）首次实现了端到端像素级 JEPA 训练（无需 EMA、无需 stop-gradient）。SIGReg 从数学上防止了表示崩塌。Causal-JEPA（2026.02）在 JEPA 中引入了对象级因果推理。但所有 JEPA 工作限于简单 2D/3D 控制任务——人形全身控制完全空白。

V-JEPA 2.1 encoder 被"Reconstruction or Semantics?"（2605.06388）验证为最佳机器人世界模型 encoder——但只用了它的视觉 backbone，没有用 JEPA 的 latent prediction 能力。

**核心 Insight**：
**LeWM 的 SIGReg 解决了 JEPA 训练稳定性的核心问题，V-JEPA 2.1 提供了最佳视觉 encoder，二者的结合——在 V-JEPA 2.1 latent space 上做 LeWM 风格的 JEPA world model，再添加人形特定的 latent dynamics——是技术上可行且科学上有价值的方向。**

**可做的工作**：
- **KA-LeWM（Kinematic-Aware LeWM）**：将 LeWM 的 latent space 按运动学链分组（左臂/右臂/左腿/右腿/躯干）——组内允许相关，组间鼓励独立
- **Contact-Triggered Latent Reset**：接触事件时 latent dynamics 允许不连续
- 核心优势：JEPA 的样本效率远超扩散世界模型，适合真机数据稀缺的人形场景
- 产出：**第一个面向人形全身控制的 JEPA 世界模型**


#### 5.5.3 Lyapunov-EBM 用于人形稳定控制

**背景**：
能量模型（EBM）在图像生成中被扩散模型取代，但在机器人控制中有独特优势：物理系统天然具有能量函数（动能+势能），而 Lyapunov 稳定性理论告诉我们——如果一个控制器的行为可以表示为某个能量函数的最小化，系统在 Lyapunov 意义下是稳定的。

Value Function is a CBF（ICML 2023 Outstanding Paper）证明了在特定条件下最优值函数经过变换是一个有效 CBF——揭示了"值函数-安全性"之间的深刻联系。Lyapunov-EBM 将这个联系推广——训练一个能量函数 E(s, a)，其最小化同时实现任务目标和满足 Lyapunov 稳定性条件。

**可做的工作**：
- **Lyapunov-Constrained EBM**：E_θ(s, a) = 模仿学习对比散度 + Lyapunov 递减条件（E(s_{t+1}, a*) ≤ E(s_t, a_t) - κ·E(s_t, a_t)）
- **多尺度能量**：E = E_torso（躯干稳定，慢尺度）+ E_limb（四肢协调，中尺度）+ E_contact（接触切换，快尺度）
- 核心优势：如果 Lyapunov 条件被成功学习 → 形式化的稳定性保证（罕见的价值）
- 风险：EBM 在高维动作空间中的采样效率是核心挑战
- 产出：**一个天然具有稳定性保证的人形控制策略**


---

## 第六部分　实施方案汇总与路线建议

### 6.1 方案关系图

```
          物理感知层               控制层                  学习层             理论/架构层
      ┌──────────────┐    ┌──────────────────┐   ┌──────────────────┐  ┌──────────────────┐
      │ E: 异构触觉融合│    │ H: 关键帧WM验证规划│   │ J: 不确定性门控飞轮│  │1.4.1: WAM信息论   │
      │ F: 模态置信门控│    │ I: 预演式纠错      │   │ K: 技能库+归因修复│  │1.4.2: 三子空间分离 │
      │ G: 磁皮肤仿真  │    │ C: 接触梯度校正    │   │ L: 安全reset-free │  │1.4.3: 物理因果发现 │
      │ A: 守恒律WM   │    │ B: 属性辨识条件化  │   │5.4.1: 自指涉安全  │  │1.4.4: 接触增强WAM │
      │2.6.1: 触觉双空间│   │4.5.1: 事件驱动S1/S2│   │5.4.2: Near-failure│  │4.5.2: 最优预测粒度│
      │2.6.2: 触觉校准WM│   │4.5.2: 自适应预测粒度│  │5.4.3: 反事实增强  │  │4.5.3: 记忆-WM耦合 │
      │2.6.3: 模态不平衡│   │4.5.3: 记忆-WM耦合  │   │                  │  │5.5.1: HMP人形mask│
      │3.3.1: 3D策略   │   │                  │   │                  │  │5.5.2: KA-LeWM JEPA│
      │3.3.2: 动态3DGS │   │                  │   │                  │  │5.5.3: Lyapunov-EBM│
      └──────────────┘    └──────────────────┘   └──────────────────┘  └──────────────────┘
              ↑                    ↑                      ↑                     ↑
              └────────── 共享 latent 世界模型（主线核心）──────────────────────┘
```

### 6.2 按论文导向的推荐推进路线

**第一篇（快速切入，低风险）**：方案 A（守恒律辅助损失）+ 方案 B（属性辨识条件化）+ 1.4.1 信息论分析
- 仿真即可完成主要实验，信息论分析提供理论深度
- 产出：一个物理更一致的 latent 世界模型 + WAM video co-training 的理论解释

**第二篇（建立差异化）**：方案 H（关键帧WM验证规划）+ 4.5.2（自适应预测粒度）
- 关键帧WM直接解决"长程误差累积"，自适应预测粒度提供理论最优性
- 产出：一个"误差与horizon解耦"的规划器 + 预测粒度的变分理论

**第三篇（建立完整叙事）**：方案 L（安全 reset-free）+ 5.4.1（自指涉安全）+ 5.4.3（反事实增强）
- 首次在人形上实现内生安全的自主学习——领域最空、新颖性最高
- 产出：不需要外部安全模块的人形自主学习系统

**第四篇（开辟新范式）**：5.5.1（HMP）+ 5.5.2（KA-LeWM）
- 超越现有扩散/WAM/纯RL框架，探索 masked generative + JEPA 的新路径
- 风险较高但成功后影响巨大

### 6.3 最推荐的切入方案（按优先级，整合升级后方案）

| 排名 | 方案 | 为什么先做这个 | 可行性 | 创新性 | 预期效果提升 |
|---|---|---|---|---|---|
| 1 | **L: 自指涉安全** | 定义"安全=(状态+策略+WM)联合函数"——概念级创新；领域最空；理论和系统双线可发 | 中（需理论+真机） | **极高**（范式级） | 摔倒率↓90%，不牺牲成功率 |
| 2 | **A: Latent Noether 定理** | 从"注入已知守恒量"→"自动发现未知守恒量"——理论深度极高；纯仿真可验证；理论文可独立发 | 高（纯理论+仿真） | **极高**（首个） | 长程预测误差↓40-60% |
| 3 | **H: 自适应预测粒度WAM** | Jacobian谱驱动跳帧——理论最优性可证明；仿真即可验证；计算复杂度分析是加分项 | 高（仿真为主） | **高**（理论+方法） | 推理速度↑3-5×，精度不降 |
| 4 | **B: 双重控制物理属性推断** | 从"被动观测"→"主动试探"——最优实验设计×机器人学习交叉；信息增益有闭式近似 | 中（需RL+信息论背景） | **极高**（首次） | 适应数据↓50-80%，适应后成功率↑20-40% |
| 5 | **J: 元认知POMDP调度** | 从"固定阈值"→"最优序贯决策"——首次将WM使用形式化为POMDP；涌现行为最有说服力 | 中（需RL理论背景） | **极高**（全新话题） | 真机采样↓60-80%，成功率↑30-50% |
| 6 | **C: 隐式接触求解器** | 从"梯度校正"→"混合系统架构"——从根本上解决接触梯度问题；互补约束自监督对真机可用 | 中（需混合系统背景） | **高**（架构级创新） | 接触预测精度↑50-80% |
| 7 | **D: VLM⊕WAM Latent反事实检验** | 从"仿真参数化验证"→"latent反事实检验"——绕过参数化瓶颈；实时可部署 | 中（需WAM基座+VLM） | **高**（方法创新） | 验证速度↑100-1000× |
| 8 | **I: 反事实预演** | Adversarial latent search + conformal safety guarantee——最坏情况分析保证了安全 | 中（需WAM基座） | **高**（安全保证） | 安全违规↓80-90% |
| 9 | **K: Continuous Skill Manifold** | 从"离散技能库"→"Riemannian流形"——零样本泛化到未见参数；数学优雅 | 中（需流形学习背景） | **高**（新颖） | 新参数零样本成功率↑40-60% |

### 6.4 升级总结：从工程方案到科学贡献

| 原方案 | 核心问题 | 升级后方案 | 核心升级 |
|---|---|---|---|
| A: 守恒律辅助loss | 加auxiliary loss是"方法增量" | A: Latent Noether定理 | 从"注入已知"→"自动发现未知"，理论深度从0到顶会级 |
| B: 属性辨识+FiLM | 标准system ID+条件化 | B: 双重控制属性推断 | 从"被动观测"→"主动试探"，引入最优实验设计框架 |
| C: 梯度校正网络 | "创可贴"式后验修正 | C: 隐式接触求解器 | 从"治标"→"治本"，架构级混合系统建模 |
| D: VLM→仿真→反馈 | 系统集成，VLM参数化瓶颈 | D: VLM⊕WAM反事实检验 | 从"仿真验证"→"latent检验"，绕过参数化瓶颈 |
| H: 关键帧WM | 手工定义关键帧 | H: 自适应预测粒度 | 从"人定义"→"动力学涌现"，理论最优性保证 |
| I: 预演式纠错 | 被动检查给定计划 | I: 反事实预演 | 从"检查"→"主动搜索最坏情况"，conformal guarantee |
| J: 不确定性门控 | 固定阈值θ₁,θ₂ | J: 元认知POMDP | 从"手工阈值"→"最优序贯决策"，涌现行为 |
| K: 技能库+LoRA | 离散技能，泛化靠检索 | K: Skill Manifold | 从"离散库"→"连续流形"，geodesic外推 |
| L: 安全reset-free | 外部recoverability classifier+双策略 | L: 自指涉安全 | 从"外源性安全"→"内生性安全"，概念级创新 |

### 6.5 文档结构说明

