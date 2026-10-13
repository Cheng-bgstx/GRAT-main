# 具身智能与人形机器人：高质量未来研究方向深度分析

> 本文档基于对 2025-2026 年 VLA、WAM、JEPA、空间智能、多模态感知、安全鲁棒性等方向的系统性文献调研，
> 深入分析每一篇工作的技术内容与未解决问题，提炼出具有顶会顶刊发表潜力的高质量未来研究方向。
>
> 完成日期：2026-06-05
>
> **设计原则**：每个方向 (1) 明确当前 SOTA 及其**根本性局限**（非"缺数据""缺算力"），
> (2) 提出具有理论深度或新颖性的 **insight**，
> (3) 给出具体技术方案与创新点，
> (4) 说明为什么现在可做、为什么不是其他人已经做了的。

---
## 📍 本文档在六文档体系中的定位

**角色：未来研究方向的"选题地图"——"做什么方向可以发顶会、为什么、怎么做"**

| 维度 | 说明 |
|---|---|
| **本文档做什么** | 28 个高质量未来研究方向，覆盖 WAM/VLA/JEPA/空间智能/多模态融合/安全/跨本体/新架构/数据 9 大板块。每个方向有：背景与动机、现有工作与局限、核心 Insight、技术方案、创新点、预期发表级别、为什么现在可做 |
| **本文档不做什么** | 不给具体实施方案的伪代码级细节（→ `Humanoid_DeepDive_Implementation.md`）、不追问科学问题的基础结构（→ VLA/WAM Framework）、不给论文清单（→ Survey） |
| **与 Survey 的关系** | Survey 告诉你"现在有什么论文"；本文档告诉你"基于这些论文，接下来应该做什么方向" |
| **与两个 Framework 的关系** | Framework 的 Open Problems（如 VLA Framework Open-3C、WAM Framework Q4）是本文档方向的"理论根"——本文档把 Open Problem 展开为可操作的选题 |
| **与 Supplement 的关系** | Supplement 提供了 2026 年最新实验事实（如 Fast-WAM 的"test-time imagination 不重要"），本文档的方向 1（WAM 信息论解释）直接建立在这些事实上 |
| **与 DeepDive Implementation 的关系** | **最重要的区别**：本文档是"选题级"（告诉你什么方向值得做）；DeepDive 是"落地级"（告诉你具体怎么写代码）。例如本文档方向 15（自指涉安全）→ DeepDive 方案 L 给出伪代码架构；方向 8（CA-JEPA）→ DeepDive 方案 A（Latent Noether）提供了相关的理论工具 |
| **阅读顺序建议** | 读完 Survey + 两个 Framework + Supplement 后读本文档 → 挑选 1-2 个方向 → 去 DeepDive 找对应实施方案落地 |

---


## 总体分析：2026 年具身智能的核心矛盾与机会空间

### 当前格局的判断

**已充分竞争的领域**（新进入者难以建立独特优势）：
- 纯 VLA 架构微调（在 OpenVLA/π0/GR00T 上小修小补）
- 通用机器人数据集的规模化（已被 NVIDIA/Google/Meta 的工业级投入锁定）
- 仿真环境中的标准 benchmark 刷榜（LIBERO/RoboCasa 已高度内卷）

**存在结构性机会的领域**（有理论空白或范式转换窗口）：
1. **WAM 的理论基础**：Fast-WAM 动摇了"test-time imagination 必要"的假设，但未解释"为什么 video co-training 有效"。WAM 的内部表征机制、信息流、训练-推理不对称性——几乎完全空白。
2. **JEPA + 机器人控制的深度结合**：V-JEPA 2 展示了潜力，但从"latent space prediction"到"contact-rich manipulation"的鸿沟未被跨越。
3. **人形机器人的"动力学智能"**：现有 WAM/VLA 绝大多数在桌面操作（quasi-static, 低 DoF），人形全身 loco-manipulation 的动力学复杂性远超当前方法的能力边界。
4. **安全 ≠ 外部约束**：内生安全（自指涉可恢复性）在最新 WAM 工作中仍是空白。
5. **3D 空间智能与策略的深度融合**：3DGS/NeRF 主要用于数据增强，尚未成为策略表征的内在组成部分。

以下按方向逐一展开。


## 一、WAM（World Action Model）方向

### 方向 1：WAM 中 Video Co-Training 为什么有效的理论解释

**背景与动机**：
Fast-WAM（清华/星海图，2026.03）的核心发现——训练时保留 video prediction 作为 proxy task 是关键，推理时跳过视频生成几乎不影响性能——动摇了"WAM = 先想象再执行"的基本假设。但该工作仅提供了**现象层面的消融实验**，未给出理论解释。

**现有工作的根本局限**：
- Fast-WAM 只证明了"video co-training 有效"和"test-time imagination 可跳过"，但没有回答：video prediction 提供了什么独特的监督信号是纯 action prediction 没有的？
- Cosmos Policy 将 action 作为多视角建模，需要 repeat action 来缓解梯度淹没——这是工程 trick，不是理论方案。
- NoiseGate 从"信息门控"角度处理，但没有触及 video co-training 的本质。

**核心 Insight**：
**Video co-training 本质上是在 latent space 中施加了一个"物理一致性正则化"——它强制 latent dynamics 保留对未来观测的预测信息，从而间接约束了 action token 的表示质量。** 这个正则化的效果可以通过信息论工具严格刻画：video prediction loss 的下界等于 action-relevant information 与 action-irrelevant information 的加权和，其中权重由 latent dynamics 的局部条件熵决定。

**技术方案**：

1. **理论框架——信息分解**：
   - 将 latent state z_t 分解为三部分：action-relevant I_action（控制当前动作所需）、dynamics-relevant I_dyn（预测未来状态所需）、task-irrelevant I_irr（背景、光照等）
   - 证明：纯 action prediction loss 倾向于保留 I_action 而丢弃 I_dyn（导致 fragile representation）
   - Video co-training loss 强制保留 I_dyn，且 I_dyn 与 I_action 在物理系统中的重叠通过"物理因果图"形式化

2. **方法——信息感知的 WAM 训练**：
   - 引入 **Information-Aware Training Schedule**：在训练早期高权重 video loss（构建 rich I_dyn），后期逐渐降低权重（聚焦 I_action）
   - 设计 **Latent Information Probe**：在每个训练步估计 I(z_t; a_t | o_t) 和 I(z_t; o_{t+1} | o_t, a_t)，作为自适应权重的基础
   - 提出 **Mutual Information Lower Bound** 作为 video co-training 有效性的理论保证

3. **验证方案**：
   - 在 LIBERO/RoboTwin 上验证：固定 action prediction 架构，系统改变 video supervision 的信息量（预测全帧 vs 预测物体 mask vs 预测关键点），测量对策略鲁棒性的影响
   - 关键实验：在不同物理复杂度（quasi-static 桌面操作 vs dynamic 投掷/推动）的任务上，video co-training 的收益是否与物理复杂度正相关？

**创新点**：
- 首次从信息论角度解释 video co-training 在 WAM 中的作用
- 提出可优化的训练 schedule（而非固定权重），理论上保证在保留 I_dyn 的同时最大化 I_action
- 为 WAM 的架构设计提供理论指导（"需要保留多少 predictive information 才够？"）

**预期发表级别**：NeurIPS / ICML 2027

**为什么现在可做**：
- Fast-WAM 的实验事实已确立，理论解释的窗口刚刚打开
- V-JEPA 2 和 Reconstruction or Semantics?（2605.06388）提供了 latent space 信息度量的工具
- 信息瓶颈在表示学习中已有成熟理论工具，迁移到 WAM 是顺势而为


### 方向 2：面向人形全身 Loco-Manipulation 的层级化 WAM

**背景与动机**：
现有 WAM 工作（DreamZero、Cosmos Policy、Fast-WAM、X-WAM）几乎全部在**桌面操作**场景验证——单臂或双臂、固定基座、quasi-static 动力学。人形机器人的全身 loco-manipulation 带来了全新挑战：40+ DoF、移动基座与操作臂的动力学耦合、接触序列的时序精度要求远高于桌面操作。

**现有工作的根本局限**：
- DreamZero/X-WAM 的联合预测（视频+动作）在 7-DoF 操作上成功，但直接扩展到 40+ DoF 面临"动作空间维数灾难"
- Ψ0 的三系统架构（S2 语义 + S1 动作 + S0 反射）是手工频率分离，不是从数据/物理中涌现的
- WholeBodyVLA 的 LAM 从人类视频学习潜在动作，但潜在动作空间的"分辨率"不足以描述亚秒级接触切换
- 没有任何 WAM 工作同时建模"下肢移动的运动学链"和"上肢操作的运动学链"以及它们通过躯干的动力学耦合

**核心 Insight**：
**人形全身 WAM 需要一种"层级化隐式动力学解耦"——不是手工分离移动和操作（如 Ψ0 的 S0/S1 分离），而是让 WAM 在训练中自动发现：latent space 的某些维度控制质心动量（移动相关），某些维度控制末端执行器（操作相关），而躯干维度的 latent 表示是二者的"耦合接口"。** 这种解耦可以从物理 first principle 推导：基于机器人动力学的递归牛顿-欧拉方程，latent space 的 Jacobian 在不同频率下呈现分块对角结构。

**技术方案**：

1. **架构——Hierarchical Latent WAM (HL-WAM)**：
   - **Latent 分解**：z_t = [z_loco, z_manip, z_torso]——通过频谱条件化（对 latent 维度施加不同频率的正则化）而非手工分离
   - **多尺度 Temporal Transformer**：三个时间尺度的 Transformer 分别处理高频接触动力学（100Hz latent）、中频协同运动（20Hz）、低频语义切换（2Hz）
   - **物理耦合模块**：躯干 latent 作为 loco 和 manip 的 attention bottleneck——loco 和 manip latent 通过 cross-attention 与躯干交互，但不直接交互

2. **训练策略**：
   - 阶段 1：在仿真中分别训练 loco WAM（固定上肢）和 manip WAM（固定基座），各自收敛
   - 阶段 2：**耦合微调**——解冻躯干模块，用全身 loco-manipulation 数据联合训练。关键：引入**接触一致性损失**——当 manip WAM 预测手部接触时，loco WAM 必须预测相应的质心调整
   - 阶段 3：少量真机数据 fine-tune，冻结 latent 分解结构

3. **验证平台**：
   - 仿真：Isaac Sim 中的 Unitree H1/G1 全身 loco-manipulation（如"边走边端水"、"蹲下捡东西再站起来走"）
   - 真机：Ψ0 开源基座 + 少量遥操作数据
   - 指标：任务成功率、接触时序精度（预测接触 vs 真实接触的时间差）、动力学一致性（预测质心轨迹 vs 真实质心轨迹的 RMS）

**创新点**：
- 首次将 WAM 扩展到人形全身 loco-manipulation
- 层级化 latent 解耦由物理驱动的频率分离（非手工架构分离）
- 接触一致性损失将 loco 和 manip 的耦合从隐式变为半显式约束

**预期发表级别**：RSS / CoRL 2027；扩展版 IEEE T-RO

**为什么现在可做**：
- 人形全身控制的 RL 已成熟（XHugWBC, EAGLE），可以作为"真值"生成训练数据
- Ψ0/WholeBodyVLA 提供开源 VLA 基座
- WAM 架构（DiT-based）已有多家开源实现


### 方向 3：WAM 的"反事实安全性"——想象中最坏情况的主动搜索

**背景与动机**：
When to Trust Imagination（2026.05）通过 FFDC 验证器判断想象是否可信。但其核心逻辑是**被动的**——它检查当前想象是否与真实观测一致，如果不一致就重新规划。这种"事后补救"在人形机器人高动态场景中可能来不及——等你发现想象错了，已经摔倒了。

**核心 Insight**：
**WAM 应该在想象空间中主动搜索"最坏情况"（adversarial imagination）——不是问"我的想象对吗"，而是问"在当前不确定性下，可能发生的最坏物理后果是什么，我能否从那里恢复"。** 这本质上是将 WAM 的 forward rollout 从一个"最大似然轨迹"扩展为一个"可能轨迹的分布"，并在分布的尾部（worst-case within confidence bound）做安全性检验。

**技术方案**：

1. **Adversarial Latent Rollout (ALR)**：
   - 在 WAM 的每一步 latent rollout 中，不取确定性预测 z_{t+1} = f(z_t, a_t)，而是在噪声空间中搜索最坏扰动：z_{t+1}^{adv} = f(z_t, a_t) + δ*，其中 δ* = argmax_{δ ∈ B_ε(z_t)} Risk(z_{t+1} + δ)
   - B_ε(z_t) 是 latent space 中半径为 ε 的球，ε 由模型的 epistemic uncertainty 确定
   - Risk(z) 使用可恢复性函数 R(z; π, M)（见世界动作模型框架 Q4）

2. **Worst-Case Safety Margin (WCSM) 作为动作选择的约束**：
   - 对于候选动作 a_t，计算 WCSM(a_t) = min_{δ ∈ B_ε} R(f(z_t, a_t) + δ)
   - 如果 WCSM(a_t) < θ_safe，拒绝该动作并选择更保守的替代动作
   - 替代动作的搜索：在 latent action space 中以当前动作的 latent code 为起点，沿 WCSM 的梯度方向做少量步的梯度上升

3. **高效实现**：
   - δ* 的搜索不需要在每步做完整的梯度优化——ensemble 预测的 spread 自然对应最坏/最好情况
   - 使用 Conformal Prediction 确定 B_ε 的半径 ε，保证在置信水平 1-α 下真实 latent state 包含在 B_ε 内

**创新点**：
- 从"被动检查想象可信度"升级为"主动搜索最坏情况"
- 将安全性从"事后验证"变为"事前预防"——在动作执行前就验证其最坏后果
- Conformal Prediction + WAM 的结合提供 finite-sample safety guarantee

**预期发表级别**：CoRL / ICRA 2027；安全理论部分可投 ICML

**为什么现在可做**：
- When to Trust Imagination 已建立"想象可信度"的问题框架
- Conformal prediction 在机器人中的应用已有多项成熟工作
- WAM 的 latent space 连续性使 adversary search 的计算可行


### 方向 4：WAM 的跨本体"物理常识"迁移——从四足到人形、从人到机器人

**背景与动机**：
DreamDojo 的 latent actions 展示了"跨人-机器人"的潜在动作迁移。但跨不同机器人形态（如四足→人形）的 WAM 迁移仍完全空白。四足和人形虽然在形态上不同，但共享大量"物理常识"——接触力学、摩擦锥、动量守恒、支撑多边形——这些是 body-agnostic 的。

**现有工作的局限**：
- DreamDojo 的 latent actions 在相同形态间迁移（人→人形）
- XHugWBC 在形态间迁移低层控制策略，但不涉及高层世界模型
- Hardware-Agnostic WM 针对四足，不跨四足-人形边界

**核心 Insight**：
**不同形态的机器人共享一个"物理常识核心"（body-agnostic physics），而形态特定的部分只是"运动学映射层"。** 如果能将 WAM 分解为 body-agnostic physics core（预测接触、动量、能量流）和 body-specific kinematic decoder（将物理量映射为特定形态的动作），就可以实现四足→人形、甚至仿真→真机的 WAM 零样本迁移。

**技术方案**：

1. **Body-Agnostic Physics Core (BAPC)**：
   - 输入：以"广义物理量"编码的 latent state——质心动量、接触点位置与力方向、支撑多边形、角动量
   - 预测：未来 N 步的广义物理量演化
   - 训练：在多形态仿真数据（四足、人形、轮式、双足）上联合训练，形态 ID 不作为输入（强制 body-agnostic）

2. **Body-Specific Kinematic Decoder (BSKD)**：
   - 轻量 MLP/Transformer，以 BAPC 的物理量预测 + 形态编码为输入，输出特定形态的关节目标
   - 每个形态一个轻量 BSKD（参数量 < 5% BAPC）

3. **训练流程**：
   - 预训练：在 Isaac Sim 中并行训练四足（Unitree Go2, B2）、人形（H1, G1, GR1）、双足（Cassie）的各自全身控制策略
   - 物理量提取：从每个形态的仿真中自动标注广义物理量（仿真器可直接读取）
   - BAPC 联合训练：所有形态的 generalized physics trajectory → 训练统一的 BAPC
   - BSKD 形态特定训练：冻结 BAPC，每个形态单独训练 BSKD
   - 零样本迁移测试：用四足 data 训练的 BAPC + 人形 BSKD → 测试人形 loco-manipulation

**创新点**：
- 首次提出 WAM 的 body-agnostic/body-specific 分解
- 广义物理量作为跨形态的"lingua franca"
- 从仿真中的物理量标注（可自动化）→ 真实世界的物理常识——sim-to-real 的新路径

**预期发表级别**：CoRL / NeurIPS 2027

**为什么现在可做**：
- DreamDojo 已验证跨人-机器人迁移可行
- Isaac Sim 中多形态仿真已成熟
- XHugWBC 提供了跨人形的低层策略，可作为 BSKD 的初始化


## 二、VLA（Vision-Language-Action）方向

### 方向 5：VLA 中的语言 Grounding 解耦——语义层、可供性层、运动层的独立泛化

**背景与动机**：
当前 VLA 的语言 grounding 是"端到端隐式"的——语言 token 嵌入 → attention → 动作 token。RoboVLM 的诊断实验已确证：语义等价改写导致约 20% 成功率下降。这说明 VLA 学到的不是真正的语义理解，而是"语言-动作"的表面统计关联。

**核心 Insight**：
**语言→动作的映射应该被结构化为三层因果链：语义层（语言→任务意图）→ 可供性层（任务意图→物体/区域选择）→ 运动层（交互目标→具体动作）。** 如果三层被强制解耦（通过结构约束而非仅通过训练目标），那么新语言表述只应影响语义层，新物体只应影响可供性层，新硬件只应影响运动层。这种"独立泛化"是真正语义理解的判据——也是 VLA 鲁棒性提升的关键。

**技术方案**：

1. **三层结构化 VLA 架构**：
   - **语义编码器**（冻结或微调 VLM）：语言 → 任务意图向量 z_sem ∈ R^D_sem（包含"做什么"+"物理暗示"）
   - **可供性解码器**：z_sem + 全景视觉特征 → 可供性热力图（哪些区域/物体可交互）+ 交互类型（抓/推/按）
   - **运动生成器**：可供性热力图 + 局部视觉特征（裁剪到交互区域）+ 本体感受 → 动作序列

2. **强制解耦的结构约束**：
   - **信息瓶颈**：语义→可供性层之间施加 KL 约束 I(z_sem; affordance) ≤ C_1，强制语义不携带空间细节
   - **正交投影**：可供性→运动层之间，对视觉特征施加 Gram-Schmidt 正交化——将"与可供性相关的特征"和"与运动执行相关但与可供性无关的特征"分离到正交子空间
   - **反事实训练**：随机替换语义（保持可供性+运动不变）→ 强制运动层不依赖语言；随机替换可供性（保持语义+运动不变）→ 强制语义层不直接控制运动

3. **独立泛化评测**：
   - 语义泛化：同义改写测试（RoboVLM 协议）
   - 可供性泛化：新物体但相同语义和运动模式（"拿起[没见过的杯子]"）
   - 运动泛化：新机器人形态但相同语义和可供性（仅替换运动层）

**创新点**：
- 首次通过结构约束（而非仅训练目标）强制 VLA 的三层解耦
- 反事实训练协议——在 latent space 中做"如果只改变语义会怎样"的实验
- 独立泛化作为 VLA 语义理解的严格判据（超越"成功率"的单维度）

**预期发表级别**：ICLR / NeurIPS 2027

**为什么现在可做**：
- RoboBrain 已有三层架构的雏形，但缺乏解耦约束
- 因果表示学习（causal representation learning）的独立机制假设提供了理论工具
- RoboVLM 提供了成熟的评测协议


### 方向 6：VLA 的"物理暗示→可执行约束"映射——语言作为软物理约束生成器

**背景与动机**：
语言中嵌入了丰富的物理暗示：副词（"轻""快""稳"）、动词选择（"端"vs"拿"→暗示保持水平）、情态（"小心"→减速+降低重心）。当前 VLA 对这些暗示的处理完全是隐式的——"轻拿"可能导致模型在训练中见过"轻拿鸡蛋"所以输出小力，但"轻拿铁块"时因为没见过这个组合而失败。Cosmos-Reason 能输出语言级的物理推理（"这个物体较重，应减慢速度"），但无法将语言转换为可执行的动作约束。

**核心 Insight**：
**"物理暗示"本质上定义了一个在动作空间中的软约束区域，这个约束区域的具体数值（如"轻"对应的最大力阈值）不是固定的，而是由物理上下文（物体材质、质量、任务阶段）自适应确定的。** 构建一个"物理暗示→约束函数"的映射，使 VLA 在生成动作时不仅条件化于语言 token，还条件化于从语言中**显式解析出的物理约束**。

**技术方案**：

1. **Physical Hint Parser（PHP）**：
   - 基于 VLM 的模块，从自然语言指令中提取结构化物理约束：
     - 副词→数值约束类型（"轻"→力上限，"快"→速度下限，"稳"→末端水平约束）
     - 动词→约束方向（"端"→保持物体上表面水平，"推"→力沿前进方向）
     - 情态→安全裕度（"小心"→增大所有约束的安全系数）
   - 输出：structured constraint spec（JSON-like，但可微）

2. **Context-Adaptive Constraint Instantiation（CACI）**：
   - PHP 输出的不是绝对数值，而是**约束函数模板**：force < f_max(material, mass)
   - f_max 的具体值由视觉上下文（物体材质识别、大小估计）在线计算
   - 使用物理模拟器中的先验知识（如常见材料的摩擦系数范围）作为初始化

3. **Constrained Action Generation（CAG）**：
   - 在 latent diffusion/flow matching 的每一步中，将约束作为 guidance 信号注入：
     - 如果 denoised action 违反约束 → 沿约束梯度方向修正 latent
     - 关键创新：修正发生在 latent space 而非 action space——保留 latent 表示的结构
   - 约束满足的优先级：硬约束（安全相关，如力不超过抓取滑脱阈值）> 软约束（性能相关，如"快一点"）

**创新点**：
- 首次建立"语言物理暗示→结构化可执行约束"的显式映射
- 上下文自适应的约束实例化（同一句话在不同物理上下文有不同数值约束）
- 在 latent space 中进行约束引导（非 action space 后处理裁剪）

**预期发表级别**：CoRL / ICRA 2027；NLP+Robotics 交叉可投 EMNLP

**为什么现在可做**：
- Cosmos-Reason 提供了物理常识推理的基座
- PhysBench 建立了 VLM 物理理解的评测基准
- Diffusion/flow matching 的 guided generation 技术已成熟


### 方向 7：基于世界模型不确定性的事件驱动 S1/S2 自适应切换

**背景与动机**：
Ψ0、GR00T N1、Helix 等双系统/三系统架构均使用**固定频率**进行 S2（慢/语义规划）到 S1（快/动作执行）的通信。但人类不是这样——我们在熟练任务中 S2 几乎不介入，在不确定/新奇/危险时刻 S2 立刻抢占注意力。核心追问：**S2 什么时候应该介入 S1 的执行？这个决策本身应该是一个可学习的科学问题，而非手工频率。**

**核心 Insight**：
**S2 的介入应由"latent divergence 的增长率"而非绝对值触发。** 绝对值高可能只是因为环境嘈杂/任务本身不确定性高——"情况不好但稳定"不需要 S2。增长率为正且加速时——"情况正在快速变糟"——才需要 S2 介入。这与控制论中的事件触发控制（Event-Triggered Control, ETC）有直接的数学联系——触发条件基于状态误差的导数而非绝对值。

同时，**S2 的介入也不仅因为"偏离预期"——也可能因为"出现了需要更高级推理的机会"（如发现了更高效的任务执行方式）。** 这种"机会驱动"的 S2 介入在现有工作中完全缺失。

**技术方案**：

1. **Latent Divergence Monitor（LDM）**：
   - 世界模型维护一个 latent trajectory 的"预期锥"——基于 ensemble latent dynamics 的多步预测分布
   - 在每个 S1 执行步，计算当前 latent state 到预期锥边界的距离 d_t
   - 触发信号 S(t) = α·d_t + β·(d_t - d_{t-1}) + γ·(Δd_t - Δd_{t-1})——结合偏移量、速度、加速度
   - 当 S(t) > θ(t)（其中 θ(t) 自身是自适应的，随任务熟悉度上升）时触发 S2 重规划

2. **双模触发**：
   - **偏离触发**：d_t 的加速增长（"情况正在变糟"）
   - **机会触发**：当 VLM 在"后台"检测到当前状态与更优计划的匹配时（"咦，现在可以直接做 B 而不是按原计划做 A"）

3. **与 S2 的通信协议**：
   - S1 → S2：当前 latent state、不确定性估计、偏离轨迹的摘要
   - S2 → S1：新子目标（在 latent goal space 中）、可选的物理约束（如有）、预期置信度
   - **关键**：S2 的输出不仅包括"做什么"，还包括"在什么条件下需要重新找我"——即 S2 对自己计划的预期有效期

**创新点**：
- 首次将 S1/S2 切换形式化为基于 latent divergence 增长率的事件触发问题
- ETC 理论从已知状态空间到 learned latent space 的迁移——控制论 × 机器学习的交叉
- 双模触发（偏离+机会）超越现有"被动检测失败"

**预期发表级别**：RSS / CoRL 2027；理论部分可投 L4DC

**为什么现在可做**：
- Ψ0 提供了成熟的三系统实现基座
- ETC 理论在控制论中已成熟，迁移的数学基础存在
- Latent divergence 的监测已在 When to Trust Imagination 中初步验证


## 三、JEPA 与表示学习方向

### 方向 8：JEPA 世界模型用于接触丰富操作的物理级预测

**背景与动机**：
V-JEPA 2（Meta, 2025.06）在 1M+ 小时互联网视频上预训练，仅需 62 小时无标注机器人视频即可实现零样本图像目标 reaching。但"Reconstruction or Semantics?"（2605.06388）揭示了一个关键事实：V-JEPA 的 latent space 在"语义判别"上优异，但在精确物理预测（如"这个推力的精确后果"）上不如 reconstruction-based VAE。

**核心 Insight**：
**V-JEPA 的 latent space 保留了"将会发生什么"的语义信息，但丢弃了"精确如何发生"的几何/力学细节。** 对于接触丰富的操作（插销、拧螺丝、拉链），缺失的恰是这些细节。解决思路不是回到像素重建（太低效），而是在 JEPA 的 latent space 中**引入一个"接触感知"的辅助目标**——不是重建像素，而是预测接触力、滑移、接触点在 latent space 中的"几何结构"。

**技术方案**：

1. **Contact-Aware JEPA (CA-JEPA)**：
   - 在 V-JEPA 的基础上，添加**接触预测头**：
     - 预测未来 N 帧中每帧的接触事件（binary，哪些物体/表面之间发生接触）
     - 预测接触力的相对大小（不需要绝对牛顿值，只需相对排序："这次接触比上次更用力"）
     - 预测滑移事件（物体在手中是否正在滑脱）
   - 这些预测在 latent space 中进行——不需要像素解码

2. **Latent Contact Geometry（LCG）**：
   - 在 latent space 中学习一个"接触度量"：将 latent 维度分为"接触相关"和"接触无关"子空间
   - 接触相关的 latent 维度应满足"接触连续性"——在接触事件前后，这些维度的变化应该连续（接触力不能瞬间跳变）
   - 训练目标：接触维度的二阶时间导数最小化（鼓励平滑）+ 非接触维度的信息最大化

3. **Latent MPC with Contact Awareness**：
   - 在 CA-JEPA 的 latent space 中做 MPC 时，不仅优化"离目标 latent state 的距离"，还优化"接触序列的时序精度"
   - 目标函数：min_a Σ_t ||z_t - z_goal||² + λ·ContactCost(z_t, a_t)
   - ContactCost 惩罚：过大接触力、滑移、非必要接触

**创新点**：
- 首次在 JEPA latent space 中显式建模接触物理
- latent contact geometry——接触的连续性和平滑性作为 latent 结构先验
- 在不损失 JEPA 的效率优势（无需像素重建）的前提下，提升接触丰富任务的预测精度

**预期发表级别**：ICLR / NeurIPS 2027

**为什么现在可做**：
- V-JEPA 2 开源 + V-JEPA 2.1 encoder 已验证为最佳机器人世界模型 encoder
- "Reconstruction or Semantics?"（2605.06388）的发现提供了明确的改进方向
- 接触预测的数据标注可通过物理仿真自动化


### 方向 9：JEPA 世界模型中的"物理因果发现"

**背景与动机**：
现有世界模型（DreamerV3, TD-MPC2, V-JEPA-2-AC）将物理因果关系全部隐式编码在 latent dynamics 中——"推物体→物体移动"这个因果关系被混在 latent 维度的相关性中。V-JEPA 2 的分析表明，其 latent space 确实涌现了部分物理概念（如物体永存性、重力方向），但这些概念的涌现是不受控的——我们不知道哪些物理因果关系被编码了，哪些被遗漏了。

**核心 Insight**：
**JEPA 的 latent space 中，物理因果关系表现为 latent 维度之间的条件独立性结构。** 具体来说：如果 latent 维度 z_i 和 z_j 在给定动作 a 的条件下是条件独立的（即 a d-separates z_i from z_j），那么 z_i 和 z_j 之间的关联完全由动作驱动——这对应了"动作→结果"的因果链。反过来，如果 z_i 和 z_j 在给定 a 后仍然相关，那么存在从 z_i 到 z_j 的直接因果边（"物体 A 的移动本身导致了物体 B 的移动"）。

这一定性观察可以被形式化为 latent space 中的**因果发现算法**，从而：(a) 量化 JEPA 学到了多少物理因果结构，(b) 用学到的因果图改进 latent dynamics 的结构（forcing sparsity according to causal graph），(c) 鉴别 JEPA 是否学到了虚假关联。

**技术方案**：

1. **Latent Causal Discovery（LCD）**：
   - 在 V-JEPA-2-AC 的 latent space 中，使用条件独立性检验（如 kernel-based conditional independence test）在 latent 维度之间发现因果边
   - 关键：动作 a_t 作为已知的 intervention——动作是因果图中的"根节点"，不能有指向动作的因果边
   - 输出：latent causal graph G = (V, E)，其中 V 是 latent 维度或维度组，E 是因果边

2. **Causally-Structured Latent Dynamics**：
   - 用发现的因果图 G 结构化 latent dynamics：f(z_t, a_t) = GNN on causal graph
   - 如果 z_i 和 z_j 在 G 中没有直接因果边，则在 GNN 中它们之间无消息传递
   - 稀疏化 latent dynamics → 更可解释、更少的虚假关联、理论上更好的 OOD 泛化

3. **虚假关联检测**：
   - 在训练分布和 OOD 测试分布中分别运行 LCD，比较因果图的差异
   - 在训练分布中存在但在 OOD 中消失的因果边 = 虚假关联
   - 量化指标：Causal Stability Score——在分布偏移下保持不变的因果边比例

**创新点**：
- 首次在 JEPA 世界模型的 latent space 中进行因果发现
- 因果图结构化的 latent dynamics（非全连接 Transformer）
- 虚假关联的数学定义和自动检测——超越"OOD 成功率下降"的现象描述

**预期发表级别**：ICML / NeurIPS 2027

**为什么现在可做**：
- V-JEPA 2 开源，提供了成熟的 JEPA world model
- 因果发现（causal discovery）在 latent space 中的应用已有初步工作（Schölkopf 组, Bengio 组）
- 物理仿真中因果图是已知的（ground truth for validation）


### 方向 10：Demo-JEPA 的深化——从"一次性模仿"到"组合式技能库"

**背景与动机**：
Demo-JEPA（2605.20811, 2026.05）实现了基于 JEPA 的跨本体一次性模仿（one-shot cross-embodiment imitation）。但其局限性在于：(a) 每次模仿是独立的，不积累经验；(b) 模仿的是完整轨迹，不能进行技能的组合和重用。

**核心 Insight**：
**JEPA 的 latent space 具有某种"线性结构"——相似的潜在状态变化对应相似的动作。** 如果这个性质成立，那么可以在 latent space 中定义一个"技能代数"——技能 = latent state 的变化方向（Δz），技能的组合 = Δz 的加法/插值。这样，从几个 demo 中学到的技能可以在 latent space 中自动组合，生成新的行为。

**技术方案**：

1. **Latent Skill Algebra**：
   - 定义技能 s 为 latent state 的定向变化：s = {Δz_1, Δz_2, ..., Δz_K}（K 步 latent trajectory segment）
   - 技能库 S = {s_1, s_2, ..., s_N}，每个 s_i 来自一个 demo
   - 技能组合：新技能 s_new = Σ_i w_i · s_i（在 latent space 中的线性组合）
   - 关键假设验证：测试 latent space 中的加法是否对应物理上有意义的行为组合

2. **Compositional Demo-JEPA**：
   - 阶段 1：从多个 demo 中提取技能段（使用 changepoint detection 在 latent trajectory 上自动分割）
   - 阶段 2：构建技能图谱——技能之间的"可连接性"（技能 A 的后置条件是否与技能 B 的前置条件兼容）
   - 阶段 3：给定新任务描述，在技能图谱中搜索最短路径 + 在 latent space 中做线性插值填补路径上的空白

3. **验证实验**：
   - 在 RLBench/MetaWorld 上测试：3 个 demo（"打开抽屉""拿起物体""放入物体"）→ 零样本组合"打开抽屉→拿起物体→放入抽屉"
   - 关键指标：组合技能的成功率 vs 单独技能的成功率——组合是否在 latent space 中自然涌现？

**创新点**：
- 首次提出 latent space 中的技能代数（线性可组合性）
- 从"一次性模仿"到"组合式泛化"的跨越
- 无需额外训练的 zero-shot skill composition

**预期发表级别**：CoRL / NeurIPS 2027

**为什么现在可做**：
- Demo-JEPA 提供了跨本体 latent space 的基础
- V-JEPA 2 的 latent space 线性性质已有初步验证（线性 probing 分类效果优异）
- 技能组合在 robotics 中是一个长期存在但未解决的核心问题


## 四、空间智能与 3D 方向

### 方向 11：3D 空间表征作为策略的内在组成部分——从"数据增强"到"表征基础"

**背景与动机**：
当前 3DGS/NeRF 在机器人中的应用主要是"数据增强"——RoboSplat（RSS 2025）用 3DGS 生成新视角的演示，GenSplat（2026.03）用 3DGS 做视角泛化。但 3D 表征始终是**策略的外围工具**——策略本身仍然是 2D 图像的映射。这导致了一个根本性问题：策略学到的"抓取"是特定视角下的 2D 模式，而不是真正的 3D 物理交互。

**核心 Insight**：
**3D 空间表征不应是策略的"数据增强器"，而应是策略表征的"内在基础"。** 即策略的 latent state 应建立在 3D 体素/点云/高斯表征之上，而非 2D 图像特征之上。具体方案：将 3D Gaussian Splatting 的 latent feature（每个 Gaussian 的 SH 系数、不透明度、协方差）作为策略 Transformer 的输入 token，使策略在 3D 空间中直接推理"在哪里形成什么样的接触"。

**技术方案**：

1. **3D-Native Policy Transformer**：
   - 输入：场景的 3DGS 表示——N 个 3D Gaussian，每个带位置 μ_i、协方差 Σ_i、不透明度 α_i、外观特征 f_i（从 VLM 提取）
   - **3D Tokenization**：每个 Gaussian 的 (μ_i, Σ_i, f_i) → 一个 3D token
   - **3D Attention**：Transformer 中的 attention 计算不仅基于 feature similarity，还基于 3D 空间邻近性：
     - A_ij = softmax(Q_i^T K_j / √d + λ · exp(-||μ_i - μ_j||² / 2σ²))
     - 空间偏置项使注意力偏向空间邻近的 Gaussian——物理上合理的 inductive bias

2. **3D Affordance as Attention Map**：
   - 语言指令通过 cross-attention 查询 3D tokens → 产生 3D affordance heatmap
   - 与 2D affordance 的关键区别：3D affordance 自动处理遮挡、视角不变性、物理可行性（基于 3D 几何）
   - Action 从 3D affordance 中直接解码：末端执行器目标位置 = affordance heatmap 的 argmax

3. **Dynamic 3DGS Update during Execution**：
   - 在动作执行过程中，随着新观测到来，增量更新 3DGS
   - 更新后的 3DGS 立即反映物体被推动/抓取后的新位置
   - 策略基于最新 3DGS 做闭环重规划

**创新点**：
- 首次将 3DGS 从"数据增强工具"升级为策略的表征基础
- 3D-aware attention——物理上合理的 inductive bias
- 3D affordance——天然处理遮挡和视角不变性
- 动态 3DGS 更新支持实时闭环控制

**预期发表级别**：CVPR / ICCV 2027（vision 角度）；RSS / CoRL 2027（robotics 角度）

**为什么现在可做**：
- 3DGS 的实时渲染已高度优化（~100 FPS）
- RoboSplat 已验证 3DGS 在机器人场景中的可行性
- VLM 的 3D 特征提取（如 DINOv2 在 3DGS 上的特征蒸馏）已有成熟方法
- 增量式 3DGS（SplaTAM, MonoGS）已支持实时更新


### 方向 12：4D 世界重建 + 物理推理的统一框架

**背景与动机**：
X-WAM（2026.04）在单一框架中实现了 4D 世界合成 + 实时动作执行。但其 4D 重建和动作生成是**并行的输出头，共享 encoder**——二者之间没有信息交换。这意味着 4D 重建的质量不影响动作质量，反之亦然——错失了一个关键机会：**如果动作预测可以利用 4D 重建中的物理信息（如预测的接触点、物体运动），而 4D 重建可以利用动作预测来更好地推理物体动力学，二者可以形成"物理上一致"的协同增强。**

**核心 Insight**：
**4D 重建和动作生成应该形成一个"物理一致性循环"——4D 重建提供"如果执行这个动作，物理世界会怎样变化"的预测；动作生成利用这个预测来优化动作；优化后的动作再反馈给 4D 重建以修正预测。** 这个循环的核心是一个**物理一致性损失**：在 4D 重建中，物体的运动必须满足刚体动力学（力 = 质量 × 加速度），而作用力只能来自机器人接触点——没有其他力源。

**技术方案**：

1. **Physics-Consistent 4D Reconstruction**：
   - 将场景分解为：(a) 静态背景（3DGS 不随时间变化），(b) 铰接/可移动物体（3DGS 位置随时间变化），(c) 机器人本体
   - 物体运动受物理约束：
     - 刚体假设：物体的 3DGS 作为一个整体做 SE(3) 变换
     - 力-运动约束：物体的加速度仅由重力 + 机器人接触力驱动
     - 接触约束：接触点处无穿透

2. **Bidirectional 4D-Action Synergy**：
   - **Forward path**（4D → Action）：基于 4D 预测的物体运动，Action head 预测最优动作
   - **Backward path**（Action → 4D）：Action head 的预测作为 4D 重建的"拟用力"条件，修正物体运动预测
   - **Cyclic Consistency**：经过 forward→backward→forward 一个循环后，预测的动作应该一致

3. **Training with Physics Priors**：
   - 在仿真中预训练：物理 ground truth（物体轨迹、接触力）可直接监督
   - 真机 fine-tune：物理监督信号不可得，但物理一致性（无外力时动量守恒、接触无穿透等）可作为自监督损失

**创新点**：
- 首次实现 4D 重建和动作生成的物理一致性协同
- 循环一致性作为训练信号——无需物理 ground truth
- 物理先验（动量守恒、接触无穿透）作为自监督学习目标

**预期发表级别**：CVPR / ICCV 2027；扩展版 TPAMI

**为什么现在可做**：
- X-WAM 已验证 4D 重建 + 动作生成的可行性
- 可微物理约束（如 ContactNets 的互补约束）提供了物理损失的工具
- 3DGS 的高斯运动可以通过 SE(3) 变换高效实现


## 五、多模态感知融合方向

### 方向 13：触觉作为世界模型的"物理校准信号"

**背景与动机**：
当前世界模型（DreamerV3, V-JEPA-2-AC, WAM 系列）几乎全部基于视觉输入。但在接触丰富的操作中，视觉被严重遮挡——手和物体之间、手指之间都是盲区。触觉传感器（GelSight, DIGIT）提供了接触几何和力的直接测量，但触觉的"空间范围"极其有限（仅在传感器表面）。

ManiFeel（2025/2026）系统验证了触觉对精确接触操作的价值。DexTac（2026.01）展示了从人类演示中学习触觉先验的有效性。但所有这些工作将触觉用作**策略的输入模态**，而非**世界模型的校准信号**。

**核心 Insight**：
**触觉不是"另一种视觉"——它是物理现实的直接测量。在接触事件发生时，触觉提供了视觉世界模型无法自我校准的"ground truth"物理信号（如"我确实抓到了物体""物体确实在滑脱"）。** 世界模型应该将触觉信号用作其 latent state 的**在线校准锚点**——当视觉预测与触觉观测冲突时，触觉胜出（因为它是直接物理接触测量，不受遮挡和光照影响）。

更进一步：触觉的**时序精度**（~1000Hz for GelSight）远超视觉（~30Hz），这提供了一个高频的"物理监督信号"，可以用来纠正世界模型在亚视觉帧率下的预测偏差。

**技术方案**：

1. **Visuotactile World Model（VT-WM）**：
   - 主 latent dynamics：视觉驱动的 latent state z_t^vis → 预测 z_{t+1}^vis（标准世界模型路径）
   - 触觉校准模块：在接触事件发生时，触觉编码 t_t 映射到 latent touch state z_t^touch
   - **校准机制**：当 z_t^vis 和 z_t^touch 在接触相关子空间中的差异超过阈值时，以 z_t^touch 为锚点**重校准** z_t^vis：
     - z_t^calibrated = z_t^vis + K · (proj_contact(z_t^touch) - proj_contact(z_t^vis))
     - 其中 proj_contact 是将 latent state 投影到"接触相关"子空间的线性映射（通过训练学习）

2. **Touch-as-Physics-Ground-Truth for World Model Training**：
   - 在接触帧上，世界模型不仅被要求预测下一帧观测（视觉），还被要求预测触觉信号（力的方向和大小）
   - 触觉预测误差的权重远高于视觉预测误差（因为触觉是物理现实的直接测量）
   - 这强制世界模型的 latent dynamics 编码精确的接触物理——因为只有编码了接触物理，才能从 latent state 解码出正确的触觉信号

3. **高频触觉-低频视觉的异步融合**：
   - 视觉帧率 30Hz，触觉帧率 1000Hz → 在两次视觉观测之间有 ~33 次触觉观测
   - 将触觉的高频序列压缩为 latent state 的"快变量"（接触状态、滑移倾向）
   - 视觉负责"慢变量"（物体位置、场景布局）
   - 两种时间尺度的 latent state 由 VT-WM 的 dual-timescale latent dynamics 统一建模

**创新点**：
- 首次将触觉定位为世界模型的"校准信号"而非"输入模态"
- 触觉-视觉冲突时触觉优先的物理原理与实现
- 异步时间尺度融合（1000Hz 触觉 + 30Hz 视觉）的 latent dynamics

**预期发表级别**：CoRL / RSS 2027；触觉-视觉融合理论部分可投 Science Robotics

**为什么现在可做**：
- TacSL 提供了 GPU 加速的触觉仿真（Isaac Sim 集成），使大规模训练可行
- ManiFeel 提供了标准化的 visuotactile benchmark
- GelSight/DIGIT 的真机部署已成熟
- Dual-timescale latent dynamics 的理论基础可在 MTRSSM (ICDL 2023) 和 HCLSM (2603.29090) 中找到雏形


### 方向 14：全身体感——人形机器人的"本体感知世界模型"

**背景与动机**：
人形机器人全身有 40+ 个关节，每个关节有位置、速度、力矩传感器。这提供了丰富的本体感知信号，但现有方法（DreamerV3, V-JEPA-2-AC）仅将本体感知作为"状态向量"输入，没有利用其空间结构——膝关节和踝关节的力矩模式之间存在功能关系（都参与支撑），这种关系不是任意的，而是由物理和生物力学决定的。

GR00T N1.6 的参考硬件（75DoF, 双 Sharpa Wave 五指触觉手）意味着未来人形将产生海量的全身感知数据。如何从这些数据中学习结构化的本体感知世界模型，是尚未被探索的方向。

**核心 Insight**：
**人形机器人的本体感知信号嵌入在一个"运动学图"（kinematic graph）中——图的节点是关节，边是运动学链。** 这个图的邻接结构编码了物理约束（髋关节力矩→膝关节力矩通过大腿骨传递）。世界模型的 proprioceptive encoding 应该尊重这个图结构——不是将所有关节信号 flatten 成一维向量，而是在 kinematic graph 上做 message passing。

更进一步：本体感知不仅是"机器人自身状态的观测"，也是**外部物理世界的一种"间接测量"**——当你推一个重物时，手臂关节力矩的增大间接测量了物体的质量。这种"通过本体感知推断外部物理属性"的能力在现有世界模型中完全缺失。

**技术方案**：

1. **Kinematic Graph World Model（KG-WM）**：
   - 将人形机器人的本体感知编码为 kinematic graph 上的节点特征
   - 在每个时间步，在 kinematic graph 上运行 GNN message passing：
     - 关节 i 从邻居关节 j 接收消息 m_{j→i} = MLP(h_j, h_i, joint_type_embedding)
     - h_i^{new} = GRU(h_i, aggregate({m_{j→i}}))
   - Kinematic graph 的边权重不是均等的——在接触链上（手指→手腕→肘→肩→躯干→髋→膝→踝→地面）的边应有更高权重
   - GNN 输出作为世界模型的 proprioceptive latent state z_t^prop

2. **External Physics Inference from Proprioception**：
   - 从 z_t^prop 中解码外部物理属性：物体质量估计、接触点位置、摩擦力大小
   - 训练信号：在仿真中，这些物理量有 ground truth；在真机中，使用"间接验证"（如"如果质量估计正确，那么预测的关节力矩应与实际力矩一致"）
   - 注意：物理属性的推断是不确定性的——质量可能是一个分布而非点估计。KG-WM 应输出 posterior distribution over physical properties

3. **Whole-Body Contact Detection from Proprioception**：
   - 关键能力：仅从关节力矩和位置（无需视觉或触觉传感器），判断身体哪个部位与外部物体/环境发生接触
   - 实现：在 kinematic graph 中，接触事件表现为力矩的阶跃变化——GNN 天然适合检测这种局部结构变化
   - 训练：仿真中自动标注接触事件 → 监督学习

**创新点**：
- 首次用 kinematic graph 结构化人形本体感知的世界模型
- 从本体感知推断外部物理属性——间接测量能力
- 无需专用触觉传感器的全身接触检测

**预期发表级别**：CoRL / ICRA 2027

**为什么现在可做**：
- GR00T N1.6 参考硬件 75DoF 本体感知数据量充足
- Isaac Sim 中可获取所有关节力矩/位置的仿真数据
- GNN + graph-structured state 已有成熟方法（如 MeshGraphNet）


## 六、安全与鲁棒性方向

### 方向 15：自指涉可恢复性——从"外部安全滤波器"到"策略内生的安全感知"

**背景与动机**：
这是世界动作模型研究框架（Humanoid_World_Action_Model_Research_Framework.md）中 Q4 的核心概念——但在最新的 WAM 工作中（HarmoWAM, When to Trust Imagination, NoiseGate），安全性仍然是外部施加的约束或启发式检查。True endo-safety——安全从策略内部自然产生——仍是基本空白。

**核心 Insight**：
**安全不是一个状态的函数，而是（状态+策略能力+世界模型）的联合函数。** 同一个状态（如单脚站立），对熟练的策略是安全的（能恢复），对稚嫩的策略是不安全的（会摔倒）。因此，安全定义应该是**自指涉的**——给定当前策略 π 和世界模型 M，从状态 z 出发，策略能否在未来 N 步内回到稳定区域？

定义**自指涉可恢复性函数**：
$$R(z; \pi, M) = \mathbb{P}\left(\exists \{a_t, ..., a_{t+N}\} \sim \pi \text{ s.t. } z_{t+N} \in S \mid z_t = z\right)$$

其中 S 是稳定姿态集合。R(z) ∈ [0,1] 的梯度方向指向"更不可恢复"，边界 {z: R(z)=0} 是死亡边界——策略无法恢复的状态。

**为什么这个方向可以在顶会发表**：
- 它将"恢复"从工程实践升级为数学概念，且具有形式化的理论框架
- 对于人形机器人，可恢复性函数的估计和利用是尚未被触碰但极其重要的问题
- 理论工具（HJ reachability, Lyapunov 理论, conformal prediction）完全可以支撑

**技术方案**：

1. **Self-Referential Recoverability Estimator（SRRE）**：
   - 基于 WAM 的 latent space，训练 R_θ(z)：给定 latent state z，估计 R(z; π, M)
   - 训练数据生成：
     - 在仿真中对每个 latent state z，执行 N 步策略 rollout（真实物理仿真，非想象 rollout）
     - 如果能在 N 步内回到稳定姿态，标注为 recoverable
     - 重复多次（不同随机种子）得到恢复概率的经验估计
   - SRRE 使用 WAM 的 latent encoder（冻结），添加轻量 MLP 预测 R(z)

2. **Pessimistic Calibration of R(z)**：
   - 由于 R(z) 通过世界模型估计（而非真机执行），存在模型误差
   - 使用 conformal prediction 提供保证：P(真实可恢复 | R̂(z) > θ) ≥ 1-α
   - 校准集：真机上的少量 rollout，标记每个状态的真实可恢复性
   - 关键：悲观校准——宁可低估 R(z)（假危险，过度保守），不可高估（假安全，致命）

3. **R(z)-Conditioned Policy**：
   - 训练策略时，将 R(z) 作为额外的条件输入
   - 策略学习：当 R(z) 较低时采取更保守的动作；当 R(z) 高时可以激进
   - 关键是：策略**不在"任务"和"恢复"之间切换**——它始终在执行同一个策略，但策略的行为随 R(z) 连续变化
   - 这意味着恢复行为不是外部接入的"安全模式"，而是策略内部的连续属性

4. **Recoverability-Driven Curriculum**：
   - 不是在越来越难的任务上训练，而是在 **R(z) 越来越低的状态**上训练
   - Curriculum 的推进由 R(z) 的边界自动定义——当策略改善了 R(z)，边界自然向外移动
   - 在每次 curriculum step，采样 R(z) 接近但非零的状态——这里梯度信息最丰富

**创新点**：
- 首次提出自指涉安全的数学定义和实现框架
- 从"外部安全滤波"到"策略内生安全感知"的范式转换
- HJ reachability 从已知动力学到 learned latent dynamics 的迁移
- 可恢复性驱动的 curriculum learning
- Pessimistic calibration 提供 finite-sample safety guarantee

**预期发表级别**：RSS / CoRL 2027（系统+实验）；ICML / NeurIPS 2027（理论部分）

**为什么现在可做**：
- WAM 提供了 latent space 中进行数学分析的基础
- HJ reachability 和 safety filter 的理论已成熟（Latent Safety Filter, LatentCBF）
- Ψ0 等开源人形基座可供实验
- 软垫环境可用于安全地验证可恢复性边界
- 此方向的**新颖性极高**——在所有最新 WAM 工作中（HarmoWAM, When to Trust Imagination, AttenA+），没有工作显式编码可恢复性


## 七、跨本体泛化方向

### 方向 16：形态感知的 WAM——从"跨形态迁移"到"形态作为条件"

**背景与动机**：
XHugWBC（2026.02）通过形态随机化实现了跨 12 个仿真 + 7 个真人形的零样本泛化。但这是**低层控制策略**的跨本体泛化。对于世界动作模型（WAM），跨本体泛化意味着：能否训练一个 WAM，它可以为不同形态的机器人预测"未来会发生什么"和"应该做什么"，而不需要为每个形态重新训练？

**核心 Insight**：
**形态（morphology）可以编码为一个"形态图"（morphology graph），作为 WAM 的 persistent condition。** 关键创新：形态图不仅描述运动学结构（关节数、连接关系），还编码**动力学能力**——每个关节的力矩极限、速度极限、功率密度。这样，WAM 在预测 future states 和生成 actions 时，自动受限于该形态的物理能力边界。

不同于 XHugWBC 的"形态随机化让模型学习无视形态差异"，这里的思路是**让模型学习利用形态差异**——对于重负载形态（如 H1），WAM 自动生成腿部主导的动作；对于轻灵形态（如 G1），WAM 自动生成更敏捷的动作。

**技术方案**：

1. **Morphology Graph Encoder（MGE）**：
   - 输入：形态描述（关节数、连接关系、力矩/速度限制、连杆质量/惯量、足端几何）
   - 编码为 morphology graph G_m，节点=关节/连杆，边=运动学链/动力学耦合
   - 使用 GNN 将 G_m 编码为 fixed-size morphology embedding e_morph

2. **Morphology-Conditioned WAM（MC-WAM）**：
   - WAM 的 latent dynamics 和 action generation 以 e_morph 为条件
   - 关键：e_morph 不是与 latent state 简单拼接，而是通过 **FiLM（Feature-wise Linear Modulation）** 调制 latent dynamics 的参数：
     - f_morph(z_t, a_t, e_morph) = γ(e_morph) ⊙ f_base(z_t, a_t) + β(e_morph)
     - 这意味着形态改变的是动力学函数的"增益"和"偏置"，而非函数形式本身
   - FiLM 参数由轻量 hypernetwork 从 e_morph 生成

3. **Morphology-Adaptive Safety Boundary**：
   - 不同形态有不同的可恢复性边界
   - MC-WAM 的可恢复性函数 R(z; π, M, e_morph) 以形态嵌入为条件
   - 这允许同一个 WAM 安全地控制不同形态的机器人

**创新点**：
- 首次将形态信息编码为图结构并用于 WAM 的条件化
- FiLM 调制而非简单拼接——保留 latent dynamics 的通用结构
- 形态感知的安全边界——不同形态不同安全策略

**预期发表级别**：CoRL / NeurIPS 2027

**为什么现在可做**：
- XHugWBC 已验证低层策略的跨本体泛化
- 多个人形平台（H1, G1, GR1, Fourier N1 等）可供实验
- 形态随机化 + 仿真训练的范式已成熟


## 八、全新架构方向

### 方向 17：基于离散扩散的"思考-行动"统一模型——将 VLA 和 WAM 统一在离散 token 空间中

**背景与动机**：
当前 VLA 使用连续 latent（flow matching/diffusion for action），WAM 使用连续 latent（diffusion for video + action）。但语言模型使用的是离散 token。这种连续-离散的不统一导致了架构上的复杂性和理论上的不优雅。

一个统一的可能性：**将视频帧、动作、语言全部 tokenize 到同一个离散 vocabulary 中，使用 masked generative model（类似于 BERT/Masked Diffusion）统一建模。** 这已在语言和图像领域被验证（Muse, MagViT, MAGE），但尚未被认真尝试用于具身智能。

**核心 Insight**：
**将 VLA 的"语言→动作"映射和 WAM 的"状态→未来状态+动作"预测统一在同一个离散 token 空间的 masked generation 框架中——这不仅是架构上的优雅，更带来了"任意条件化"的能力：给定任何部分信息（语言/视觉/动作/本体感受），可以生成任何缺失部分。** 这意味着同一个模型可以做：(a) 语言→动作（VLA），(b) 状态→未来状态+动作（WAM），(c) 动作→预期结果（inverse dynamics），(d) 结果+动作→所需语言指令（inverse language grounding）。

**技术方案**：

1. **Unified Discrete Tokenization**：
   - 视觉：使用 VQ-VAE（如 MagViT2）将视频帧 tokenize 为离散 token
   - 语言：标准 text tokenizer（如 Qwen 的 tokenizer）
   - 动作：将连续动作向量量化为离散 token（类似 RT-2 的 binning，但使用 VQ-VAE 学习最优码本）
   - 本体感受：同动作——量化为离散 token

2. **Masked Generative World-Action Model（MG-WAM）**：
   - 架构：标准 Decoder-only Transformer（如 Llama 架构）
   - 训练：随机 mask 输入序列中的任意 token（不论模态），模型预测被 mask 的 token
   - **关键**：masking 策略不是随机的——进行**structured masking**：
     - 在某些训练样本中 mask 所有动作 token（→ 世界模型：预测未来状态）
     - 在某些训练样本中 mask 所有未来视觉 token（→ VLA：语言+当前视觉→动作）
     - 在某些训练样本中同时 mask 部分视觉和动作（→ 联合推理）
   - 这比"随机 mask"更高效——structured masking 提供了更清晰的训练信号

3. **任意条件化推理**：
   - 推理时，给定任意已知 token，生成任意未知 token
   - 例如：给定语言指令 + 当前视觉 → 生成动作 + 未来视觉（WAM + VLA 的统一推理）
   - 例如：给定动作序列 + 初始状态 → 生成预期视觉结果（用于"如果这样做会发生什么"的反事实推理）

**创新点**：
- 首次将 VLA 和 WAM 统一在离散 masked generation 框架中
- Structured masking 提供了比随机 masking 更强的训练信号
- "任意条件化"推理——同一模型支持多种使用模式
- 统一 token space 使语言、视觉、动作的跨模态推理自然可能

**预期发表级别**：NeurIPS / ICLR 2027（如果效果突破）；否则 CoRL / EMNLP 2027

**为什么现在可做**：
- 离散 tokenization 的视频模型（MagViT2, OmniTokenizer）已成熟
- RT-2 已证明动作离散 tokenization 可行
- Masked generative models 在大规模训练中验证了可扩展性
- 需要大规模多模态机器人数据集——NVIDIA Open X-Embodiment + Cosmos 数据可支持

**风险提示**：
- 动作的离散化可能导致精度损失——需要在 tokenization 质量和码本大小之间找平衡
- 多模态数据的数据量不平衡（语言 pretrain data >> 机器人 data）可能导致训练不稳定


### 方向 18：能量模型（EBM）用于人形稳定控制——将 Lyapunov 函数作为能量函数的天然桥接

**背景与动机**：
能量模型（Energy-Based Models, EBM）在图像生成中曾是扩散模型的前身，但因采样效率低而被扩散模型取代。但在机器人控制中，EBM 有一个扩散模型没有的天然优势：**物理系统天然具有能量函数（动能+势能），而 Lyapunov 稳定性理论告诉我们，如果一个控制器的行为可以表示为某个能量函数的最小化，那么系统在 Lyapunov 意义下是稳定的。**

**核心 Insight**：
**训练一个 EBM 作为人形机器人的策略——EBM 的能量函数 E(s, a) 给出在状态 s 取动作 a 的"不优度"。** E(s, a) 的训练可以结合两个信号：(a) 模仿学习——demonstration 中的 (s,a) 对应有低能量，(b) 物理稳定性——E(s, a) 应该满足 Lyapunov 条件：在最优动作下，能量沿系统轨迹递减。

如果能实现，这个 EBM 策略**天然带有稳定性保证**——因为能量函数的递减性内置在训练目标中。这是扩散/流匹配策略没有的性质。

**技术方案**：

1. **Lyapunov-Constrained EBM**：
   - 架构：E_θ(s, a) = MLP/Transformer，以状态 s 和动作 a 为输入，输出标量能量
   - 推理：a* = argmin_a E_θ(s, a)（通过 Langevin 动力学采样或梯度下降近似）
   - 训练目标：
     - L_imitation = E_θ(s_demo, a_demo) - E_θ(s_demo, a_neg)（对比散度，降低 demo 能量，提高负样本能量）
     - L_lyapunov = max(0, E_θ(s_{t+1}, a*) - E_θ(s_t, a_t) + κ·E_θ(s_t, a_t))——强制能量沿轨迹递减
     - L_contact = 在接触事件发生时，能量应该有阶跃而非连续变化（因为接触引入了外部力）
   - 总 loss = L_imitation + λ₁·L_lyapunov + λ₂·L_contact

2. **Multi-Scale Energy Landscape**：
   - 人形控制的多层级特性可以通过"多尺度能量"来建模：
     - E_torso(s, a)：躯干姿态的能量（慢尺度——稳定躯干）
     - E_limb(s, a)：四肢配置的能量（中尺度——操作和迈步）
     - E_contact(s, a)：接触点的能量（快尺度——接触切换）
   - 总能量 E = E_torso + E_limb + E_contact
   - 推理时可以从粗到细优化——先在慢尺度上找到躯干的最优姿态，再固定躯干优化四肢

3. **从 EBM 到安全保证**：
   - 如果 Lyapunov 条件被成功学习，那么对于 E_θ 的 sublevel set S_c = {s : min_a E_θ(s, a) ≤ c}，系统一旦进入 S_c 就永远不会离开
   - 这意味着：如果 S_c 被选择为"安全区域"，则机器人保证不会离开安全区域
   - 实际上，Lyapunov 条件很难完美学习——但可以使用 conformal prediction 对违反 Lyapunov 条件的概率提供统计保证

**创新点**：
- 首次将 EBM 引入人形机器人控制，利用 Lyapunov 稳定性作为训练约束
- 多尺度能量函数对应人形控制的多层级结构
- Lyapunov 条件 → 形式化的稳定性和安全性保证

**预期发表级别**：ICML / NeurIPS 2027

**为什么现在可做**：
- EBM 的训练技术（contrastive divergence, noise contrastive estimation）已成熟
- Value Function is a CBF（ICML 2023 Outstanding Paper）证明了值函数-CBF 的等价性，为 Lyapunov-Energy 的桥接提供了理论基础
- 仿真的稳定环境适合验证 Lyapunov 条件的收敛性

**风险提示**：
- EBM 在高维动作空间中的采样效率是核心挑战
- Lyapunov 条件在随机动力学中需要随机 Lyapunov 理论（比确定性更难）


## 九、数据与规模化方向

### 方向 19：基于物理一致性的"可扩展自标注"——利用物理先验自动标注机器人数据

**背景与动机**：
VLA/WAM 的数据瓶颈已是共识——真机遥操作数据极其昂贵。LEGS（2026.05）利用 3DGS + 程序化动作基元在仿真中生成带标注数据，将成本降到遥操作的 1/15。但其局限在于：3DGS 背景是静态的——如果真实场景的物理属性（摩擦、质量）与扫描时不同，仿真生成的标注可能不准确。

**核心 Insight**：
**物理定律（动量守恒、能量守恒、接触互补性）是免费的、永远的、处处适用的"标注器"。** 在任何视频中（无论是仿真、互联网、人类第一人称），物理定律自动对"哪些运动是可能的"施加约束。如果能将这些约束转化为**自监督标注信号**（即不需要人工标注的物理属性标签），就可以将互联网规模的视频数据转化为物理上有意义的训练数据。

**技术方案**：

1. **Physics-as-Labeler**：
   - 对任意视频片段，应用前景分割 → 跟踪物体运动轨迹 → 在物体边界框内求解刚体运动方程：
     - 如果物体做抛体运动 → 重力加速度已知 → 质量无关 → 可精确标注运动参数
     - 如果物体受外力（被推/被拉）→ 从运动反推力的大小和方向（逆动力学）
     - 如果物体处于静止平衡 → 支撑力 = 重力 → 可标注接触面朝向
   - 这些物理标注的质量取决于跟踪精度——在仿真中精确，在真机视频中需要 robust tracking

2. **Self-Supervised Physical Property Estimation**：
   - 从多段视频中学习"物体的物理属性先验"：
     - "这个物体看起来像金属" → 先验：质量较大，摩擦系数 0.3-0.5
     - "这个物体看起来像橡胶" → 先验：质量中等，摩擦系数 0.7-1.0
   - 这种"外观→物理属性"的映射可以通过在大规模仿真数据上训练获得
   - 然后在真实视频中使用：外观 → 物理属性先验 → 约束逆动力学求解 → 精细化属性估计

3. **Physics-Annotated Video Dataset**：
   - 应用上述方法到大规模视频数据（Ego4D, Epic Kitchens, YouTube）
   - 产出：视频片段 + 自动标注的物理属性（物体质量、摩擦、接触力、动量）+ 置信度
   - 这个数据集可用于预训练 WAM/VLA 的"物理理解"能力

**创新点**：
- 首次将物理定律系统性地用作自动标注工具
- "外观→物理属性先验"的预训练——弥合物理和视觉之间的鸿沟
- 互联网视频→物理标注数据集的 pipeline

**预期发表级别**：CVPR / NeurIPS Datasets & Benchmarks 2027

**为什么现在可做**：
- 物体跟踪（SAM 2, Track Anything）已达到实用精度
- 大规模视频数据集（Ego4D, Kinetics）唾手可得
- 可微物理仿真可生成"外观→物理属性"的配对训练数据
- Physion++ 已建立了物理属性推断的评测范式


## 十、补充方向——基于 2026 年 5-6 月最新工作

> 以下方向基于 2026 年 5-6 月的最新预印本和顶会接收论文（Motus, GigaWorld-Policy, GeoSem-WAM, AIM, JOPAT, OA-WAM, LeWorldModel, PHASOR, HTD, ForceFlow, ESI-Bench, OmniGuide, Being-H0.5, LAP 等），代表了当前最前沿的学术动态和最紧迫的未解问题。

### 方向 20：Masked Generative Models 用于人形全身轨迹规划——从扩散到并行生成

**背景与动机**：
Masked Generative Policy（MGP, ICLR 2026）首次将 masked generative modeling 应用于模仿学习，实现了比扩散策略快 35 倍的推理速度。DiscreteRTC（2026.04）展示了离散扩散的天然异步执行能力。但所有现有工作限于桌面操作和低 DoF 控制——将 masked generative models 扩展到人形 40+ DoF 全身轨迹规划，是一个未被触碰的方向。

**核心 Insight**：
**Masked generative models 天然适合人形全身控制——因为人形动作的不同维度可以在不同步数中被解码。** 例如，下肢的步态模式（高度可预测、时间周期性）可以在前几步被 mask-then-predict 确定，而手指的精细操作（依赖接触反馈、不确定性高）可以在更多步数中逐步细化。这种"渐进式解码"天然匹配了人形控制的层级特性，且不需要扩散模型的多步采样。

**技术方案**：

1. **Hierarchical Masked Policy（HMP）**：
   - 动作 tokenization：使用 VQ-VAE 将 40+ DoF 动作离散化为层级 token（粗 token：躯干/质心运动，中 token：四肢协调，细 token：手指关节）
   - 解码顺序：粗 token 优先解码（少量 mask steps），中 token 在中步数解码，细 token 在最后解码
   - 这种层级解码策略基于一个物理事实：粗粒度的全身协调（如"迈左脚+右手前伸"）决定任务成败的大方向，细粒度的手部调整可以在最后补充

2. **Adaptive Token Refinement（ATR）**：
   - 基于 MGP-Long 的 Adaptive Token Refinement，扩展到人形全身
   - 关键创新：refinement 的步数不是全局固定的，而是根据每个 token 的"置信度"动态分配——高置信 token 在少数 refinement 步后冻结，低置信 token 持续接受 refinement
   - 置信度度量：ensemble masked models 的预测一致性 + latent dynamics 的不确定性

3. **Reactive Inpainting for Humanoid Recovery**：
   - 利用 DiscreteRTC 的 inpainting 能力：当检测到即将失衡时，mask 当前动作的"不稳定部分"并 inpainting 替代动作
   - 无需重新生成整个动作序列——只需修复出问题的维度

**创新点**：
- 首次将 masked generative models 扩展到人形全身 40+ DoF 控制
- 层级解码策略——物理驱动的计算分配
- 动态 token refinement 步数——效率与精度的自适应平衡

**预期发表级别**：CoRL / NeurIPS 2027

**为什么现在可做**：
- MGP（ICLR 2026）提供了成熟的 masked generative policy 基座
- DiscreteRTC 验证了 inpainting for reactive control 的可行性
- Ψ0/GR00T N1.6 提供人形动作数据


### 方向 21：从 LeWorldModel 到人形 LeWM——无需 EMA 的端到端 JEPA 世界模型

**背景与动机**：
LeWorldModel（LeWM, Meta FAIR/LeCun, 2026.03）是 JEPA 世界模型的一个重要突破——首次实现了无需 EMA、无需 stop-gradient、无需预训练 encoder 的端到端像素级 JEPA 训练。SIGReg（Sketched-Isotropic-Gaussian Regularizer）用随机投影和 Epps-Pulley 正态性检验，从数学上防止了表示崩塌。但 LeWM 仅在简单 2D/3D 控制任务上验证（Push-T, Reacher, Two-Room），尚未扩展到高维人形全身控制。

**核心 Insight**：
**LeWM 的 SIGReg 正则化器解决了 JEPA 最大的训练稳定性问题（表示崩塌），这为 JEPA 在人形机器人上的应用扫清了最关键的障碍。** 将 LeWM 扩展到人形全身控制，关键挑战从"防止崩塌"转移到"如何在 latent space 中编码 40+ DoF 的动力学结构"。解决方案：在 latent space 中引入"运动学感知"的正则化——强制 latent 维度之间的相关结构反映运动学链的物理结构。

**技术方案**：

1. **Kinematic-Aware LeWM（KA-LeWM）**：
   - 在 LeWM 的基础上，将 SIGReg 的正态性约束扩展为"结构化正态性"
   - 具体：latent space 被划分为若干 block，每个 block 对应一个运动学链（左臂、右臂、左腿、右腿、躯干）
   - Block 内的 latent 维度允许相关（同一运动学链的关节协同），但 block 间的相关性被惩罚（不同运动学链应独立，除非通过躯干耦合）
   - 这种"分组独立性"是比完全独立（标准 SIGReg）更强的结构先验

2. **Contact-Triggered Latent Reset**：
   - 当接触事件发生时（如脚着地、手接触物体），latent dynamics 需要被"重置"——因为接触引入了新的约束，latent trajectory 的不连续性是物理上合理的
   - 在 LeWM 中，这可以通过允许接触帧的 latent prediction 具有更大的 uncertainty（更大的 prediction variance）来实现

3. **Scaling to Real Humanoids**：
   - 阶段 1：在 Isaac Sim 中的 Unitree G1/H1 数据上预训练 KA-LeWM（仿真数据可获取精确接触标注）
   - 阶段 2：少量真机数据 fine-tune（利用 LeWM 的高样本效率）
   - 阶段 3：Latent MPC in deployment——在 KA-LeWM 的 latent space 中做规划，通过 latent action decoder 映射到关节动作

**创新点**：
- 首次将 JEPA 世界模型扩展到人形全身控制
- 运动学感知的分组正态性约束——物理驱动的 latent 结构
- 接触触发的 latent uncertainty modulation

**预期发表级别**：ICLR / NeurIPS 2027

**为什么现在可做**：
- LeWM 的 SIGReg 解决了 JEPA 训练稳定性的核心问题
- V-JEPA 2 开源提供了强大的视觉 encoder
- Isaac Sim 可生成海量人形仿真数据


### 方向 22：物理动作空间的不平等性——从 AttenA+ 到"关键动作相位"理论

**背景与动机**：
AttenA+（2026.05）提出了一个简单但重要的发现：在机器人操作中，低速精密交互段的动作远比高速过渡段重要——通过速度驱动的动作注意力重加权，可以提升 VLA 和 WAM 的性能。但 AttenA+ 的"速度反比"启发式缺乏理论基础——为什么低速段更重要？在什么条件下这个启发式成立/不成立？

**核心 Insight**：
**动作的"重要性"不是由其速度决定的，而是由动作空间局部 Fisher 信息决定的。** 在动作空间的某些区域（如力闭合恰好成立/恰好不成立的边界），动作的微小差异导致任务结果的巨大差异——这些区域的 Fisher 信息量高。在另一些区域（如已经在稳定抓取中继续增加握力），动作的微小差异对结果影响可忽略——Fisher 信息量低。**最优的训练样本加权应按照 Fisher 信息量而非速度。**

这个理论可以推广：动作空间的"信息几何"决定了哪些动作需要被精细建模（连续 latent），哪些可以被粗粒化（离散 token），哪些可以被完全跳过（跳帧）。

**技术方案**：

1. **Fisher-Driven Action Reweighting（FAR）**：
   - 对每个训练样本 (s_t, a_t)，估计局部 Fisher 信息矩阵 F(a_t) = ∇_a log π(a|s) · ∇_a log π(a|s)^T
   - 定义样本权重 w_t = trace(F(a_t)) 或 det(F(a_t))
   - 高 Fisher 信息 → 高权重（"这个动作很关键"）
   - 理论保证：这种加权最小化了策略在任务成功率上的泛化误差上界

2. **Fisher Geometry for Action Tokenization**：
   - 在动作空间中，Fisher 信息度量定义了一个 Riemannian 流形
   - 在这个流形上做 VQ-VAE 的码本学习：在高曲率（高 Fisher 信息）区域放置更多码本向量，在低曲率区域放置更少
   - 这给出了"混合离散-连续动作表征"的最优码本分配的理论基础

3. **Phase Detection from Fisher Information**：
   - 自动检测"关键动作相位"：Fisher 信息量的局部峰值对应动作序列中的关键相位（如接触建立瞬间、滑脱临界点）
   - 在关键相位附近，策略的推理频率应自动提升；在非关键相位，可以降低频率以节省计算

**创新点**：
- 首次将 Fisher 信息几何应用于动作空间，为"动作重要性"提供理论基础
- 超越 AttenA+ 的启发式，给出最优训练加权的理论准则
- 最优动作 tokenization 的理论基础——动作空间的信息几何

**预期发表级别**：ICML / NeurIPS 2027

**为什么现在可做**：
- AttenA+ 的实验事实已确立
- Fisher 信息在机器学习中已有成熟理论
- 动作空间的 Fisher 几何是一个几乎空白的理论方向


### 方向 23："动作盲"问题的诊断与修复——ESI-Bench 启示下的空间主动性

**背景与动机**：
ESI-Bench（Stanford/World Labs, 2026.05）揭示了一个令人震惊的发现——"动作盲"（Action Blindness）：所有前沿模型都会选择错误的动作，跳过能揭示真相的观察视角，并以高置信度坚持错误决策。这不是感知问题（模型"看到"了正确的信息），而是决策问题（模型"不知道应该去看哪里"和"不知道什么时候自己的判断可能是错的"）。

**核心 Insight**：
**当前 VLA/WAM 缺乏"认知主动性"（epistemic agency）——它们被动接受给定视角的观测，然后映射到动作。但人类在不确定时会主动改变视角、主动试探、主动寻求信息。** 将这种"认知主动性"注入 VLA/WAM，需要两个核心组件：(a) 一个元认知模块，判断"我是否需要更多信息"，(b) 一个信息获取策略，执行"我应该如何获取更多信息"。

**技术方案**：

1. **Epistemic Agency Module（EAM）**：
   - 元认知：在 WAM 的每一步 latent rollout 中，估计"如果我从另一个视角看，世界模型的预测分布会怎么变化"
   - 技术实现：使用 ensemble world models，计算不同视角下的 prediction disagreement
   - 如果 disagreement 大 → 当前视角的信息不足以确定世界状态 → 触发"信息获取动作"

2. **Information-Seeking Actions（ISA）**：
   - 定义一类特殊的"信息获取动作"：移动头部/相机、用手指轻触物体确认位置、倾斜身体以获得更好视角
   - 这些动作的目标不是任务进度，而是最大化信息增益：a_info* = argmax_a I(z_{t+1}; o_{t+1} | z_t, a)
   - 训练：在仿真中，信息增益 I 可以从 ground truth state 计算（因为仿真知道真实状态）

3. **Confidence Calibration via Falsification**：
   - ESI-Bench 发现模型以高置信度坚持错误决策——这说明模型的置信度校准极差
   - 修复：在 WAM 的 latent rollout 中，主动搜索"反事实轨迹"——如果我的动作是错误的，会发生什么？
   - 如果反事实轨迹与当前轨迹的 latent divergence 小 → 当前决策可能有误 → 降低置信度

**创新点**：
- 首次将"认知主动性"概念形式化并注入机器人策略
- 信息获取动作——机器人为获取信息而主动改变行为
- 反事实 confidence calibration——超越标准不确定性量化

**预期发表级别**：RSS / CoRL 2027（可同时投 Science Robotics）

**为什么现在可做**：
- ESI-Bench 建立了"动作盲"的清晰诊断
- 信息增益在主动学习/Bayesian experimental design 中已有成熟工具
- 人形机器人的可动相机/头部使 ISA 在物理上可行


### 方向 24：语言作为跨本体动作空间——LAP 范式的深化与理论化

**背景与动机**：
LAP（Language-Action Pre-training, Princeton/PI, 2026.02）提出了一个激进的想法：将低层机器人动作表示为自然语言描述（"向前移动 3cm""以中等力度抓握"），利用 VLM 的预训练能力实现零样本跨本体泛化。但这一方法面临一个根本性挑战：自然语言对连续动作的离散化精度不足——"中等力度"在不同物理上下文中有不同的绝对数值。

**核心 Insight**：
**语言不应该直接表示精确的动作数值，而应该表示"动作的物理约束区域"——语言提供的是"做什么"和"在什么物理约束下做"，而精确的动作数值由物理上下文在该约束区域内自适应确定。** 这与方向 6（语言的物理暗示→可执行约束）形成互补——方向 6 是 VLA 的特化（"轻拿"→力<某阈值），而这个方向是跨本体的泛化（"抓取"这个语言 token 在不同形态的机器人上有不同的关节实现，但在"接触位置+接触力方向"的物理空间中是一致的）。

**技术方案**：

1. **Physical Action Language（PAL）**：
   - 定义一种"物理动作原语"词汇表——不是描述关节运动，而是描述物理交互：
     - "在位置 X 施加力 F 方向为 D"、"沿方向 D 以速度 V 移动"、"保持接触并绕轴 A 旋转"
   - 这些物理动作原语是 body-agnostic 的——不同形态的机器人通过各自的 inverse kinematics/dynamics 映射到关节动作
   - PAL 的词汇量远小于自然语言（~100-200 个原语），每个原语有明确的物理含义

2. **Context-Adaptive Action Grounding**：
   - VLM 将自然语言指令翻译为 PAL 序列（"拿起杯子"→ [定位杯子, 施加抓取力, 沿垂直方向抬起]）
   - 物理上下文（物体质量/材质/摩擦）自适应确定每个 PAL 原语的具体参数（"施加抓取力"→ 具体多少牛顿）
   - 上下文适配器：轻量 MLP，以物体视觉特征 + PAL 原语 ID 为输入，输出物理参数

3. **Cross-Embodiment Zero-Shot via PAL**：
   - 在新的机器人形态上，只需提供 PAL→关节动作的映射（可通过少量仿真数据学习）
   - PAL 本身从大规模多形态数据中学到，泛化到新形态时 PAL 不变，只替换底层映射

**创新点**：
- 物理动作语言（PAL）——比自然语言更精确、比关节空间更泛化
- 上下文自适应的物理参数实例化
- PAL 作为跨本体的"lingua franca"——与方向 4 的广义物理量互补

**预期发表级别**：CoRL / ICRA 2027

**为什么现在可做**：
- LAP 已证明语言-as-action 的可行性
- DreamDojo 的 latent actions 提供了类似的思路但缺乏物理可解释性
- PAL 的词汇量小、物理含义明确——可在大规模仿真中自动标注


### 方向 25：从触觉梦到全身物理感知——Touch Dreaming 的泛化与理论化

**背景与动机**：
HTD（Humanoid Touch Dreaming, CMU, 2026.04）首次在人形机器人上实现了"触觉想象"——Transformer 预测未来的手部关节力和触觉 latent，用 EMA target encoder 做自监督，单阶段 BC 训练。其在 5 个接触丰富任务上相对 ACT 提升 90.9%。但 HTD 的局限性在于：(a) 只覆盖手部（每只手 1,062 个触觉通道），(b) 触觉想象是 decoupled 的——与视觉世界模型无关，(c) 没有理论解释为什么 latent touch prediction 比 raw touch prediction 更好。

**核心 Insight**：
**触觉想象（touch dreaming）的本质是在 latent space 中预测"接触动力学"——哪些身体部位将在什么时间以多大的力与外部世界接触。** 这不同于视觉世界模型预测"场景会变成什么样"——触觉想象预测的是"机器人自身将如何感受世界"。二者的结合（视觉世界模型 + 触觉世界模型）可以在 latent space 中形成一个完整的"物理交互预测"——视觉告诉你物体将去哪里，触觉告诉你接触将感觉如何。

**技术方案**：

1. **Unified Visuotactile Dreaming Model**：
   - 共享 latent encoder：视觉 token (3DGS/DINO feature) + 本体感受 token + 触觉 token（仅在有传感器的地方） → 统一 latent state z_t
   - 两个预测头：
     - 视觉未来预测：z_t → z_{t+1}^{vis}（预测未来的场景 latent）
     - 触觉未来预测：z_t → z_{t+1}^{touch}（预测未来的接触 latent）
   - 两个预测共享 latent dynamics——因为视觉和触觉是同一物理过程的两个模态

2. **Contact Cross-Modal Consistency Loss**：
   - 当视觉预测"手部将接触物体"时，触觉必须预测"手部将感受到接触力"
   - 当视觉预测"手部远离物体"时，触觉必须预测"无接触力"
   - 这个 cross-modal consistency 是自监督的——不需要标注
   - 违反 consistency 说明世界模型在"幻觉"（视觉和触觉预测的物理不一致）

3. **Scaling Touch Dreaming to Whole Body**：
   - 关键挑战：全身触觉传感器密度远低于手部（每个身体部位可能只有几个触觉点）
   - 解决：利用本体感受（关节力矩）作为"分布式触觉"——关节力矩的变化间接编码了身体任意部位的接触
   - 本体感受 + 稀疏触觉 → 通过 GNN（在 kinematic graph 上）插值出全身的"虚拟触觉"分布

**创新点**：
- 首次统一视觉和触觉的 world model dreaming
- Cross-modal consistency 作为自监督的物理验证信号
- 本体感受辅助的全身虚拟触觉插值
- 为方向 13（触觉作为世界模型校准信号）提供了"想象"维度的补充

**预期发表级别**：CoRL / RSS 2027；扩展版 Science Robotics

**为什么现在可做**：
- HTD 提供了触觉想象的成熟实现
- 方向 14 的 kinematic graph 提供了全身本体感受编码
- 真机触觉传感器（GelSight, DIGIT）已可部署


### 方向 26：Inference-Time Guidance 的理论基础——从 OmniGuide 到通用策略引导框架

**背景与动机**：
OmniGuide（UPenn, 2026.03）展示了 inference-time guidance 的强大——通过在 VLA 的去噪过程中注入可微能量函数（来自 3D 基础模型、语义 VLM、人体姿态模型），将 π0.5 的成功率从 24.2% 提升到 92.4%。PCD（Policy Contrastive Decoding）通过对比原始输入和物体 mask 后输入的动作分布，在训练零修改的情况下显著提高了物体感知能力。

但这些方法都是**启发式的**——没有理论回答：(a) 什么类型的 guidance 在 inference time 是有效的，(b) guidance 的权重应该如何选择，(c) guidance 和 training 之间的最优分配是什么。

**核心 Insight**：
**Inference-time guidance 和 training-time learning 之间的最优分配可以通过"贝叶斯后验更新"来形式化。** 训练阶段学到的是"先验策略" p(a|s)；inference-time guidance 提供的是"似然" p(guidance|a,s)；最终策略是后验 p(a|s,guidance) ∝ p(a|s) · p(guidance|a,s)。这个贝叶斯框架揭示了：(a) guidance 的有效性取决于其与先验策略的"正交性"（提供互补信息），(b) guidance 权重应由 guidance 精度和先验精度的比值决定，(c) 理论上最优的方案是在训练阶段就条件化于 guidance 信号，但 inference-time guidance 是"有限训练资源"下的近似。

**技术方案**：

1. **Bayesian Guidance Framework**：
   - 将 π(a|s) 参数化为 energy-based 形式：π(a|s) ∝ exp(-E_π(a,s))
   - Guidance 也表达为能量：p(guidance|a,s) ∝ exp(-E_g(a,s,guidance))
   - 后验策略：π(a|s,guidance) ∝ exp(-E_π(a,s) - λ·E_g(a,s,guidance))
   - λ 由 guidance 的估计精度和先验精度自动确定

2. **Guidance Quality Assessment**：
   - 不是所有 guidance 在所有情况下都有用——需要评估 guidance 的信息增量
   - I(a; guidance | s) = H(a|s) - H(a|s, guidance)：guidance 提供的关于最优动作的互信息
   - 高 I → guidance 提供大量互补信息 → 高权重
   - 低 I → guidance 冗余或噪声 → 低权重

3. **Adaptive Guidance Weight**：
   - λ(s) = f(I(a; guidance | s), task_phase, uncertainty)
   - 在任务的关键阶段（接触建立、精密对准），guidance 权重高
   - 在任务的简单阶段（自由空间移动），guidance 权重低

**创新点**：
- 首次为 inference-time guidance 提供贝叶斯理论基础
- Guidance 的信息增量度量——自动确定哪些 guidance 值得使用
- 状态/阶段自适应的 guidance 权重

**预期发表级别**：ICML / NeurIPS 2027

**为什么现在可做**：
- OmniGuide 和 PCD 提供了丰富的实验现象
- 贝叶斯后验更新在机器学习中已有成熟理论
- 只需在现有 VLA（π0.5, GR00T N1.6）上添加 guidance 的能量项


### 方向 27：世界模型驱动的反事实数据增强——从"更多真机数据"到"所有可能的交互"

**背景与动机**：
WoVR（Tsinghua, 2026.02）和 World-VLA-Loop（NUS, 2026.02）展示了世界模型作为"虚拟 RL 环境"的潜力——在想象中训练策略，策略改进后在真机验证，再改进世界模型，形成闭环。但二者有一个共同的局限：世界模型生成的是"在当前策略下最可能发生"的轨迹——它不会生成"如果物体滑脱了会怎样"或"如果推的力稍大一点会怎样"的反事实轨迹。

**核心 Insight**：
**世界模型的最大价值不是生成"最可能"的轨迹（这可以用更简单的 behavior cloning 近似），而是生成"不太可能但物理上可能"的反事实轨迹——这些轨迹携带了安全性和鲁棒性的关键信息。** "如果抓取力小了 10%，物体会滑脱"——这种反事实从真机数据中几乎不可能采集（因为真机策略不会故意犯错），但从物理精确的世界模型中可以大规模生成。

**技术方案**：

1. **Counterfactual Trajectory Generation（CTG）**：
   - 从一条成功的真机轨迹出发，在 WAM 的 latent space 中沿着"动作扰动方向"做反事实 rollout：
     - 稍微减小抓取力 → 物体滑脱 → 策略需要检测滑脱并调整
     - 稍微改变接触点位置 → 工具偏转 → 策略需要补偿
     - 稍微延迟反应时间 → 任务接近失败 → 策略需要在最后时刻恢复
   - 这些反事实轨迹构成"鲁棒性训练集"

2. **Counterfactual-Aware RL（CARL）**：
   - 策略在真实数据（成功率高的轨迹）+ 反事实数据（成功率低但有恢复价值的轨迹）上联合训练
   - 反事实数据的训练目标不是"模仿"（因为它们是失败的），而是"学会从中恢复"
   - 具体：对于反事实轨迹的后半段（从出错点开始），训练策略最大化"恢复概率"

3. **Safety-Critical Counterfactuals**：
   - 重点生成"安全关键"反事实——那些可能导致摔倒/碰撞/物体损坏的轨迹
   - 这些轨迹在真实世界中不可能被采集（太危险）——但在世界模型中完全安全
   - 策略从这些"虚拟失败"中学习——不需要经历真实的摔倒就能学会如何避免摔倒

**创新点**：
- 从"最可能轨迹生成"到"反事实轨迹生成"的范式转换
- 反事实数据作为鲁棒性训练的免费资源
- 安全关键虚拟失败——无风险地学习高风险场景

**预期发表级别**：CoRL / NeurIPS 2027

**为什么现在可做**：
- WoVR 和 World-VLA-Loop 验证了世界模型作为 RL 环境的可行性
- WAM 的 latent space 连续性使反事实生成可行
- 在人形机器人领域，安全关键数据的缺乏是最严重的瓶颈之一


### 方向 28：相位锚定的跨人形表示——PHASOR 的泛化与深化

**背景与动机**：
PHASOR（2026.06）通过相位流形（FFT 提取周期性结构）+ 姿态分支的分解，创建了跨人形 embodiment 的共享动作嵌入。但其局限在于：(a) 相位分解假设周期性运动（适合行走/跑步，不适合非周期性操作），(b) 相位锚定在人类运动数据上——如果目标人形的动力学与人相差很大（如重型工业人形），锚定可能失效。

**核心 Insight**：
**周期性运动（locomotion）和非周期性运动（manipulation）在"相位空间"中有不同的数学结构——前者在 S¹（圆）上有稳定的极限环，后者在 R^n 中有离散的"吸引子"（任务目标状态）。** 统一二者的关键是：将任务执行过程建模为"从一个吸引子到另一个吸引子的跃迁，跃迁过程中被周期性 locomotion 的极限环所调制"。这是一个"乘积拓扑"：S¹ × R^n —— locomotion 在圆上演化，manipulation 在欧氏空间中跃迁。

**技术方案**：

1. **Product-Topology Action Space（PTAS）**：
   - 动作空间分解为：a = a_loco ⊕ a_manip
   - a_loco 编码在 S¹（相位）上——由 FFT 提取的周期性成分
   - a_manip 编码在 R^{n-1} 上——由"到任务目标的距离"参数化
   - 两个子空间通过"耦合项"交互：locomotion 的相位影响 manip 的基底稳定性，manip 的任务需求影响 locomotion 的步幅/方向

2. **Phase-Conditioned Stability Estimator**：
   - 给定 locomotion 的当前相位 φ 和 manip 的当前状态，估计"全身稳定性"
   - 某些相位-manip组合本质不稳定（如"左脚刚离地时右手用力推重物"）
   - 稳定性估计器作为动作选择的约束——避免在不利相位执行需要高稳定性的操作

3. **Cross-Humanoid Generalization via Topology Preservation**：
   - 不同人形的 PTAS 具有相同的拓扑结构（S¹ × R^n）但不同的几何（S¹ 的周长、R^n 的范围、耦合强度）
   - 跨本体泛化 = 在拓扑空间中的共享表征 + 形态特定的几何映射
   - 这确保了最核心的物理结构（周期性+稳定性）在跨本体迁移时被保留

**创新点**：
- 乘积拓扑（S¹ × R^n）统一周期性和非周期性人形运动
- 相位-稳定性耦合分析——安全关键
- 拓扑保持的跨本体迁移——比几何对齐更根本

**预期发表级别**：ICML / RSS 2027

**为什么现在可做**：
- PHASOR 提供了相位锚定的基础
- 拓扑数据分析（TDA）在机器学习中的应用已成熟
- 人形运动捕捉数据（AMASS, MoCap）可用于验证拓扑结构


## 十一、综合与展望

### 所有 28 个方向的总体关系图

```
                         ┌────────────────────────────────────────┐
                         │         具身智能核心理论层               │
                         │                                         │
                         │  方向1: WAM video co-training 理论       │
                         │  方向9: JEPA 物理因果发现                │
                         │  方向15: 自指涉可恢复性理论              │
                         │  方向18: Lyapunov-EBM 稳定性理论         │
                         │  方向22: Fisher 信息几何动作理论         │
                         │  方向26: Bayesian Guidance 理论          │
                         │  方向28: 乘积拓扑动作空间理论            │
                         └──────────────┬─────────────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────┐
         │                              │                       │
┌────────▼──────────────┐  ┌───────────▼──────────┐  ┌─────────▼──────────────┐
│     表征与架构层       │  │     感知与模态层      │  │    泛化与规模化层       │
│                        │  │                      │  │                        │
│ 方向2: 层级化WAM       │  │ 方向11: 3D-native策略│  │ 方向4: 跨本体物理常识   │
│ 方向5: 三层VLA解耦     │  │ 方向12: 4D物理重建   │  │ 方向16: 形态感知WAM     │
│ 方向6: 物理约束映射    │  │ 方向13: 触觉校准WM   │  │ 方向19: 物理自标注       │
│ 方向8: CA-JEPA         │  │ 方向14: 全身本体感知 │  │ 方向24: Physical Action  │
│ 方向10: 技能组合       │  │ 方向23: 认知主动性   │  │         Language        │
│ 方向17: MG-WAM         │  │ 方向25: 视觉触觉统一 │  │ 方向27: 反事实数据增强   │
│ 方向20: HMP层级masked  │  │        dreaming      │  │ 方向21: KA-LeWM JEPA     │
└────────────────────────┘  └──────────────────────┘  └─────────────────────────┘
                                        │
                            ┌───────────▼───────────────┐
                            │       系统与部署层         │
                            │                            │
                            │ 方向3: 反事实安全WAM       │
                            │ 方向7: 事件驱动S1/S2       │
                            │ 方向15: 内生安全(自指涉)   │
                            │ 方向23: 认知主动性(ISA)    │
                            │ 方向26: Inference Guidance  │
                            └────────────────────────────┘
```

### 推荐的优先顺序（按发表潜力 + 可行性 + 竞争力排序，28 个方向完整版）

| 优先级 | 方向 | 类型 | 推荐理由 |
|--------|------|------|---------|
| ⭐⭐⭐⭐⭐ | 方向15：自指涉可恢复性 | 理论+系统 | 新颖性最高，几乎无人做，理论框架清晰，WAM框架Q4核心 |
| ⭐⭐⭐⭐⭐ | 方向1：WAM video co-training 理论 | 理论 | Fast-WAM实验已确立，理论窗口刚开，信息论工具成熟 |
| ⭐⭐⭐⭐⭐ | 方向8：CA-JEPA | 方法+系统 | V-JEPA2+Reconstruction or Semantics? 提供明确路径 |
| ⭐⭐⭐⭐⭐ | 方向22：Fisher信息几何动作理论 | 理论 | AttenA+实验事实+成熟理论工具，动作空间信息几何几乎空白 |
| ⭐⭐⭐⭐⭐ | 方向23：认知主动性与"动作盲"修复 | 系统+认知 | ESI-Bench刚揭示，大量关注度，0竞争者 |
| ⭐⭐⭐⭐ | 方向5：三层VLA解耦 | 方法+理论 | 因果表示学习提供工具，RoboVLM提供评测 |
| ⭐⭐⭐⭐ | 方向13：触觉校准世界模型 | 系统+感知 | ManiFeel+TacSL基础设施，触觉-世界模型交叉空白 |
| ⭐⭐⭐⭐ | 方向11：3D-native策略 | 方法+系统 | 3DGS爆发窗口期，RoboSplat/GenSplat建立基础 |
| ⭐⭐⭐⭐ | 方向20：HMP层级Masked Policy | 方法+架构 | MGP(ICLR2026)+DiscreteRTC刚开空间，人形无人做 |
| ⭐⭐⭐⭐ | 方向25：统一视觉触觉Dreaming | 方法+感知 | HTD刚验证，与方向13/14互补，多模态想象空白 |
| ⭐⭐⭐⭐ | 方向7：事件驱动S1/S2 | 方法+控制 | ETC理论成熟，Ψ0提供基座，双系统架构共同痛点 |
| ⭐⭐⭐⭐ | 方向21：KA-LeWM人形JEPA | 方法+架构 | LeWM+SIGReg刚解决崩塌问题，人形扩展全新 |
| ⭐⭐⭐⭐ | 方向27：反事实数据增强 | 方法+数据 | WoVR+World-VLA-Loop验证可行性，安全关键数据稀缺 |
| ⭐⭐⭐⭐ | 方向26：Bayesian Guidance理论 | 理论 | OmniGuide+PCD实验事实，贝叶斯框架成熟 |
| ⭐⭐⭐ | 方向2：全身层级化WAM | 系统+工程 | 方向正确但计算资源需求高 |
| ⭐⭐⭐ | 方向6：物理暗示→约束映射 | 方法+NLP | 需要NLP+Robotics交叉，评测设计复杂 |
| ⭐⭐⭐ | 方向10：JEPA技能组合 | 方法+理论 | Demo-JEPA刚验证，但扩展周期较长 |
| ⭐⭐⭐ | 方向12：4D物理一致性 | 方法+CV | X-WAM基础存在，但计算开销大 |
| ⭐⭐⭐ | 方向14：全身本体感知WM | 方法+感知 | Kinematic graph方法成熟但真机验证周期长 |
| ⭐⭐⭐ | 方向16：形态感知WAM | 方法+泛化 | XHugWBC基础好但低层→WAM扩展难 |
| ⭐⭐⭐ | 方向17：MG-WAM | 架构 | 风险较高但成功后影响巨大 |
| ⭐⭐⭐ | 方向24：Physical Action Language | 方法+语言 | LAP验证了方向但PAL从零构建周期长 |
| ⭐⭐⭐ | 方向28：乘积拓扑动作空间 | 理论 | PHASOR刚发表，理论创新空间有限 |
| ⭐⭐⭐ | 方向18：Lyapunov-EBM | 理论 | EBM高维采样是实际挑战，风险较高 |
| ⭐⭐ | 方向3：反事实安全WAM | 系统 | 与方向15有重叠但偏应用，差异化不足 |
| ⭐⭐ | 方向4：跨本体物理常识 | 方法+泛化 | DreamDojo覆盖较多，创新空间有限 |
| ⭐⭐ | 方向9：JEPA物理因果发现 | 理论 | 因果发现在latent space极难验证 |
| ⭐⭐ | 方向19：物理自标注 | 数据 | 工程量大，创新主要体现在工程而非科学 |

### 按研究者类型推荐

**偏理论/数学背景的研究者**：
- 首选：方向1（信息论）、方向22（Fisher几何）、方向26（贝叶斯理论）、方向28（拓扑）
- 次选：方向15（安全理论）、方向18（Lyapunov）

**偏系统/工程背景的研究者**：
- 首选：方向2（全身WAM）、方向11（3D策略）、方向20（HMP）
- 次选：方向13（触觉校准）、方向25（统一dreaming）、方向27（反事实增强）

**偏感知/CV背景的研究者**：
- 首选：方向11（3D-native策略）、方向12（4D物理一致性）、方向25（视觉触觉统一）
- 次选：方向14（全身感知）、方向23（认知主动性）

**偏NLP/语言背景的研究者**：
- 首选：方向6（物理暗示→约束）、方向24（Physical Action Language）、方向5（三层解耦）
- 次选：方向10（技能组合）

**偏RL/控制背景的研究者**：
- 首选：方向7（事件驱动S1/S2）、方向15（可恢复性）、方向3（反事实安全）
- 次选：方向21（KA-LeWM）、方向27（CARL）

### 最重要的策略性建议

1. **不要做"微调已有 VLA 解决新任务"类的工作**——这个空间已经太拥挤（LIBERO ~99% 已饱和）。
2. **优先选择有理论深度的工作**——纯实验的 robotics paper 在顶会上越来越难。
3. **WAM 是 2026-2027 最大的机会窗口**——理论上几乎空白（Fast-WAM, GeoSem-WAM, Privileged Foresight Distillation 都只给了实验现象），实验上已有基础（DiT, flow matching, video pretraining）。
4. **安全/恢复是整个人形机器人领域的"最大空白"**——这个方向的任何有深度的工作在根本上区别于所有现有工作。
5. **JEPA + 机器人控制是 2026 下半年-2027 的新兴方向**——LeWM 刚解决崩塌问题，现在进入正好。
6. **Masked Generative Models 是扩散模型之外的新选项**——MGP（ICLR 2026）展示了 35x 加速，但人形应用完全空白。
7. **"动作盲"（ESI-Bench）是一个全新的、高层级的问题**——它追问的不是"模型能不能做"，而是"模型知不知道自己做错了"——这是具身智能的元认知问题，在现有工作中完全缺失。
8. **Inference-time guidance 的理论化**——OmniGuide + PCD 的启发式方法已经展示了 4x 成功率提升，但理论框架完全空白。任何能解释"什么 guidance 在什么情况下有效"的理论工作都有极高发表价值。
9. **跨本体泛化需要"物理中间语言"**——LAP（语言-as-action）、UniT（共享 codebook）、PAL（物理动作语言）——这个方向有多个竞争方案，但还没有共识。提出一个更好的"中间语言"是结构性的机会。
10. **触觉正从"附加传感器"升级为"物理 ground truth"**——HTD（触觉想象）、GenForce（传感器间力迁移）、ForceFlow（力感知流匹配）——触觉在 2026 年的角色转变是根本性的，不是渐进式的。

---

> **核实声明**：本文档中所有论文标题、arXiv 编号、机构均经 WebSearch 核实。2026 年 5-6 月的最新预印本建议正式引用前二次确认。
