# 面向人形机器人的世界动作模型：科学问题与研究框架

> 本文档是对"面向人形机器人的世界动作模型/具身世界模型"这一研究方向的重新审视与系统重构。与按"模块/模态/Sensor/架构组件"分类的调研不同，本文围绕**核心科学问题**组织——这些问题不是"把A和B拼接起来"的工程问题，而是"我们尚未理解世界动作模型应该满足什么根本属性"的基础问题。
>
> 完成日期：2026-06-04

---
## 📍 本文档在六文档体系中的定位

**角色：WAM 的科学问题框架——"世界动作模型应该满足什么根本属性"**

| 维度 | 说明 |
|---|---|
| **本文档做什么** | 围绕 4 个核心科学问题（Q1 物理隐变量/Q2 长时序推演/Q3 想象可信度/Q4 内生安全）组织 WAM/世界模型文献，追问每个问题的"本质是什么、现有方法根本局限在哪、真正未解决的开放问题是什么" |
| **本文档不做什么** | 不覆盖 VLA 特有的语言 Grounding/持续学习问题（→ `Humanoid_VLA_Research_Framework.md`）、不给具体实施方案（→ `Humanoid_DeepDive_Implementation.md`）、不枚举所有论文（→ `Humanoid_World_Action_Model_Survey.md`） |
| **与 VLA Framework 的互补关系** | WAM Framework 关注"通过预测未来来生成动作"（predictive）；VLA Framework 关注"语言→动作的直接映射"（reactive）。二者共同构成具身智能的两个核心范式 |
| **与 Latest Supplement 的关系** | Supplement 中 2026 年 WAM 爆发式的新工作（DreamZero/Fast-WAM/DreamDojo 等）直接对应本文档的 4 个 Q——Supplement 提供实验事实，本文档提供理论框架 |
| **与 Future Directions 的关系** | 本文档的 Q4（内生安全/自指涉可恢复性）是 Future Directions 中方向 15 的理论基础；Q2（长时序推演）是方向 1/8 的理论基础 |
| **阅读顺序建议** | 先读 Survey 和 Supplement 了解"有什么论文" → 再读本文档理解 WAM 的科学瓶颈 → 读到 Q4 时跳到 Future Directions 方向 15 看具体怎么做 |

---


## 0　为什么需要重新框架

已有的调研和实践容易陷入一种模式：按"视觉/触觉/力/音频怎么融合""VLM/仿真器/世界模型/RL怎么拼接""传感器选型/ring-buffer同步/硬件平台怎么搭"来组织。这些工作有工程价值，但它们回答的是"how to build"，不是"what must be true"。

本文档的出发点是：**世界动作模型——一个同时预测物理未来和生成动作的统一模型——必须满足哪些根本属性，才能让一个人形机器人在真实世界中安全、鲁棒、自主地行动？**

从这个出发点，我们提炼出四个核心科学问题。它们构成了一个从"感知-建模"到"规划-推演"到"信任-决策"到"安全-内生"的完整链条：

| 科学问题 | 核心追问 |
|---|---|
| **Q1 物理隐变量** | 视觉观测不能直接确定的物理量（接触、摩擦、质量、滑移、可恢复性）如何在世界模型中表征、推断和更新？ |
| **Q2 长时序推演** | 世界模型在长时序想象中如何避免误差累积？在什么预测粒度下误差增长最慢？ |
| **Q3 想象可信度** | 机器人如何判断世界模型的想象rollout是否可信？什么时候该用想象训练、什么时候该采样真实世界、什么时候该停止？ |
| **Q4 内生安全** | 安全性、鲁棒性与恢复能力能否从世界动作模型内部自然产生，而不是作为外部安全滤波器或独立恢复模块？ |

每个问题下面，我们梳理：已有工作的真实贡献与方法局限（科学层面的，不是"缺数据""缺算力"）、真正未解决的开放问题、可能的方法路径。


## 1　Q1：物理隐变量——"世界模型应该表征什么以及如何表征"

### 1.1 问题本质

人形机器人在真实世界中的每一次交互都涉及大量视觉无法直接观测的物理量：两个接触面之间的摩擦系数、物体内部的质量分布、材料刚度、当前接触是stick还是slip、抓取是否即将滑脱、当前姿态是否还能恢复平衡。这些量有一个共同特征：**它们只能从动力学响应中被"推断"出来，且推断的确定性随交互的进行而动态变化**。

世界动作模型面临一个根本性的表征选择：这些物理隐变量应该被编码为latent state的一部分（让模型自己学会），还是应该作为显式的预测目标（让模型输出它们），还是应该在架构层面被约束为某种物理上有意义的表示？

### 1.2 已有工作谱系

**路线A：潜变量路线——把一切编码进latent**

| 工作 | 机制 | 根本局限 |
|---|---|---|
| DreamerV3 (Hafner et al., Nature 2025, 2301.04104, 开源) | RSSM中确定性GRU状态h_t + 随机离散/连续latent z_t联合编码。物理量混在整个z_t中，通过重建和奖励信号隐式学习 | Latent-物理对齐无任何保证。"物体是红色"和"物体是重的"在latent中混在一起。对未见过的物理参数（如训练时质量0.5-3kg，测试时10kg），latent dynamics的行为完全没有理论保障 |
| TD-MPC2 (Hansen et al., ICLR 2024, 2310.16828, 开源) | Decoder-free隐式世界模型，SimNorm强制latent稀疏化，但仅对控制目标优化 | 同DreamerV3——latent各维度的物理含义不可控。且因为没有decoder，连"latent是否保留了物理信息"都无从检查 |
| RMA (Kumar et al., RSS 2021, 2107.04034, 开源) | 训练时用特权环境因子（质量、摩擦等8维）→ 部署时adaptation module从本体感受历史回归出latent extrinsics z_t | 最接近"物理隐变量latent编码"的成功案例，但z_t维度是手工设计的（8维），只覆盖训练时见过的extrinsics类型。**新的物理变化类型（如关节磨损导致的摩擦变化）完全无法被z_t捕获** |

**路线B：显式路线——让模型输出物理量**

| 工作 | 机制 | 根本局限 |
|---|---|---|
| PIN-WM (清华, RSS 2025, 2504.16693) | 可微物理模拟器+高斯splatting，端到端识别3D刚体动力学参数（摩擦、恢复系数） | **监督信号的根本缺失**：真实世界中质量和摩擦系数的ground-truth几乎不存在。只能在仿真中验证 |
| ContactNets (Posa et al., CoRL 2020, 2009.11193, 开源) | 学习隐式signed distance function，通过autodiff导出contact Jacobians，用violation-based implicit loss替代预测loss | 解决了"接触力没有直接监督"的问题（通过互补条件的violation作为loss）。但**只能建模接触这一种物理量**，扩展到质量/摩擦/柔顺性等其他隐变量没有理论框架 |
| AdaptiGraph (Li et al., RSS 2024, 2407.07889) | GNN dynamics model显式条件化于物理属性（刚度、颗粒大小），在线部署时通过few-shot优化适应 | 物理属性的选择是任务特定的，**没有统一的"哪些物理量值得显式建模"的选择原则** |

**路线C：混合路线——试图在latent space中约束物理结构**

| 工作 | 机制 | 根本局限 |
|---|---|---|
| DALI (NeurIPS) | 对抗事实检验：扰动latent的"重力维度"→检查imagined rollout是否物理合理 | 只是对latent-物理对齐的间接验证，不是disentanglement的保证。**无监督disentanglement在理论上被证明是不可能的**（Locatello et al., 2019） |
| DDP-WM (中山大学, 2602.01780, 2026预印本⚠️) | 隐式解耦稀疏主动力学（物理交互）和低秩背景校正（静态场景） | 解耦仍然是隐式的，没有物理语义标注 |
| PRISM-WM (北大, 2512.08411, 2025预印本⚠️) | MoE+Gram-Schmidt latent正交化，隐式识别接触模式（stick/slide/contact/free） | MoE的模式识别本质上是分类——没有利用接触力学的互补性结构（ϕ≥0, λ≥0, ϕ·λ=0）。如果gating错了，后续latent dynamics全在错误假设下运行 |

**路线D：双时间尺度——物理隐变量的"静态属性 vs 动态状态"分离**

这是一个特别值得关注的分支。物理隐变量本质上有两个不同的时间尺度：
- **静态/慢变**：质量、惯量、名义摩擦系数——交互过程中近似不变。推断它们需要**证据积累**（类似贝叶斯信念更新）
- **动态/快变**：接触状态、滑移、可恢复性——毫秒级变化。推断它们需要**逐帧反应**

| 工作 | 机制 | 根本局限 |
|---|---|---|
| MTRSSM (Keio Univ., IEEE ICDL 2023) | RSSM中引入两层时间层级：slow layer（大时间常数RNN）+ fast layer（小时间常数RNN） | Slow/fast分离是纯架构驱动的（时间常数），不是物理驱动的。**"慢的latent维度"不保证编码质量，"快的"不保证编码接触** |
| HCLSM (2603.29090, 2026预印本⚠️) | 三级架构：Level 0 SSM连续物理(快) + Level 1 Sparse Transformer离散事件(中) + Level 2 Goal Transformer抽象推理(慢) | 目前最结构化的多时间尺度尝试，但仍然没有显式区分"静态物理属性"和"动态物理状态" |
| DyWA (清华, 2025) | 同时处理静态物理条件（物体质量、桌面摩擦）和动态动力学状态，使用历史轨迹context | 最接近统一框架的工作，+31.5%成功率。但静态和动态的分离仍然是隐式的——**在latent中可能再次纠缠** |

### 1.3 真实未解决的科学问题

综合以上，Q1的四个根本性开放问题：

**Open-1A：物理隐变量的"可辨识性-功能充分性"对偶**

从有限交互序列中，哪些物理隐变量是可精确辨识的（参数可辨识性），哪些只能辨识到"对控制决策充分"的程度（功能可辨识性）？在什么条件下二者等价？

这个问题之所以关键：它决定了世界模型的设计哲学。如果某物理量（如精确摩擦系数μ=0.37）在数学上不可精确辨识，或者辨识所需的交互远超可用预算，那世界模型应该放弃精确估计，转而学习"功能等价类"——即能产生相同最优决策的参数集合。经典系统辨识（Ljung, 1999）有可辨识性理论，但假设已知模型结构和线性动力学。在学习模型的语境下，**可辨识性理论几乎是空白**。

**Open-1B：双时间尺度物理隐变量的统一贝叶斯框架**

如何在一个统一的世界模型中同时处理：(i) 静态物理属性的贝叶斯证据积累（不确定性随交互逐渐下降），(ii) 动态物理状态的逐帧跟踪，(iii) 二者的因果交互（属性决定状态的演化趋势）？

人类拿起杯子时，对质量的估计在最初0.1秒不确定（试探性用力），到0.5秒基本确定（调整握力）。这涉及质量（静态）的贝叶斯证据积累、接触状态（动态）的毫秒级反应、和二者的因果交互。当前任何世界模型都不能复现这一过程。关键洞察：**这需要一个"信念动力学"（参数后验如何随观测演化）和一个"状态动力学"（物理状态如何随动作和后验演化），且二者必须耦合。**

**Open-1C：参数不确定性→动作的条件化**

当世界模型对物理隐变量的推断具有本质不确定性时（如质量可能是1kg或5kg），如何将这种不确定性传导到动作生成中，使策略能执行试探动作或鲁棒动作？

经典RL的探索（epsilon-greedy、curiosity）无法做到——因为它们不条件化于"关于物理参数的信念状态"。这需要一个新的条件化机制，使策略不仅条件化于"状态估计"，还条件化于"对这个估计的信念"。

**Open-1D：物理隐变量之间的因果结构发现与利用**

质量、摩擦、接触状态、滑移、可恢复性之间不是独立的——它们有因果结构。现有的世界模型（潜变量路线）把这些全部混在一起，（显式路线）把它们当独立预测头。**没有任何工作尝试在交互过程中发现物理隐变量之间的因果结构并利用它来提高样本效率和泛化能力。** 这是一个与因果表示学习（causal representation learning）交叉的未被探索的方向。


## 2　Q2：长时序推演——"预测什么粒度才能让误差不爆炸"

### 2.1 问题本质

世界模型的核心用途之一是"在想象中预演未来"——给定当前状态和候选动作序列，预测未来的状态序列。但这个预测的误差随horizon增长是根本性的——如果每步预测误差为ε，K步后的累积误差在最坏情况下是O(ε·L^K)（L为动力学的局部Lipschitz常数）。

现有工作主要从两个方向缓解：(a) 用value bootstrap截断horizon（DreamerV3, TD-MPC2），(b) 用分层结构做粗粒度预测（Director, CoA）。但这些都回避了根本问题：**不确定"预测什么粒度"才是最优的——预测的时空粒度和误差累积的根本关系是什么？**

### 2.2 已有工作谱系

**路线A：跳帧/分层预测**

| 工作 | 跳帧机制 | 根本局限 |
|---|---|---|
| Director (Hafner et al., NeurIPS 2022, 2206.04114, 开源) | Manager每K=8步在离散goal latent space选子目标，Worker在子目标间执行 | K=8是手工设定的。**没有任何理论回答"K应该取多少"**——这取决于任务的局部动力学特征 |
| Diffusion Forcing (MIT, NeurIPS 2024, 2407.01392, 开源) | 给每个token独立噪声水平——near-future低噪声（确定）、far-future高噪声（不确定）。可变horizon rollout | 噪声schedule是启发式的。**没有基于物理或信息论的最优schedule准则** |
| CoA (ByteDance, NeurIPS 2025, 2506.09990, 开源) | 颠覆前向预测——首token是"目标关键帧"，后续从此向当前反向生成 | 反向生成天然稳定（从稳定点向外展开），但**依赖已知目标状态，对open-ended任务不适用** |

**路线B：理论分析（控制论→ML的迁移）**

| 工作 | 理论 | 对世界模型的启示 |
|---|---|---|
| MPC Horizon选择理论 (Grüne et al., Springer专著) | **α_N公式**：给定controllability结构（次可乘序列c_n），可精确计算需要多长的MPC horizon N才能保证稳定 | 关键洞见——**如果在latent space中能估计"局部controllability"（类比c_n），就可以从理论上确定最优预测horizon**。但这从未在learned world model中被做过 |
| Simulation Lemma改进 (RLC 2024, 2406.16249) | 经典Simulation Lemma: 模型误差linear-in-horizon。改进版：通过分离"compounding probability error"获得**sub-linear bound** | 误差累积比之前认为的更良性。但bound依赖于transition model的TV距离——在latent space中没有清晰物理含义 |
| SSM Recency Bias分析 (ICLR 2025, 2501.00658) | Theorem 3.1：SSM中早期token的影响指数衰减——\|∂y_t/∂x_s\| = O(e^{-κ(t-s)}) | 直接关联到"给定SSM的latent维度B，可保留的有效horizon H是有限的"。但如何选择B和H的最优trade-off仍未解决 |

**路线C：信息瓶颈视角**

| 工作 | 贡献 | 局限 |
|---|---|---|
| PLSM (MPI, NeurIPS 2024, 2401.17835, 开源) | 将information bottleneck应用于latent dynamics而非latent state：最小化I(z_t; Δ_t \| a_t)。结果dynamics变得"softly state-invariant"，减少状态空间复杂度 | β参数需手工调节。**State-invariance假设在接触丰富的任务中不成立** |
| T-IB (ICLR 2024) | 给定时间lag Δt，信息论最优压缩准则：min I(past; latent) + max I(latent; future at lag Δt) | **最优lag Δt本身是外部给定的**——这正是预测粒度问题，但没有被解决 |

### 2.3 真实未解决的科学问题

**Open-2A：局部Lipschitz驱动的自适应预测粒度**

这是Q2最核心的开放问题。世界模型在想象rollout中的一步预测误差ε_t = ‖ẑ_{t+1} - z_{t+1}‖。这个误差在下一步被放大：ε_{t+1} ≤ L(z_t) · ε_t + η，其中L(z_t)是latent dynamics在z_t处的局部Lipschitz常数，η是单步模型误差。

**核心假设**：如果能在latent space中估计L(z_t)（通过分析latent dynamics的Jacobian谱），那么：
- 在L(z) > 1的区域（膨胀动力学，如碰撞瞬间），误差指数增长 → **这里应该跳帧**（跳帧预测的误差可能反而更小，因为避免了逐帧的连续放大）
- 在L(z) < 1的区域（收缩动力学），误差自然衰减 → **不需要跳帧**
- 最优预测粒度是一个变分问题：**min_{k(t)} ε_T**，其中k(t)是时刻t的预测步长

这直接把预测粒度从"手工超参数（K=8, H=3）"升级为"可优化的量"——且理论上有最优解。

**Open-2B：长时序世界模型的"最小充分表示"**

对于horizon H的latent rollout，latent state z_t需要保留的信息量I_preserve决定了预测质量，但latent维度B有限。**给定B和H，最大可保留的预测信息量是多少？反过来，给定任务需要的预测精度，最小latent维度是多少？**

这需要区分latent state中的三种信息：(1) 守恒量（动量/能量类）——必须零损失保留；(2) 拓扑关系（支撑/包含/遮挡）——可以粗粒化但不能破坏；(3) 高频细节——可以遗忘。**能否将这种"信息分层"形式化为rate-distortion-like的理论？**

**Open-2C：Latent Lyapunov函数——从"自洽"到"真实"的桥梁**

世界模型可以在自己的latent space中做完美的自洽rollout，但整个轨迹可能与真实物理完全不符——这是世界模型最危险的failure mode。如果能定义一个latent Lyapunov函数V(z)，满足：(i) V在想象rollout中单调递减，(ii) V(z)与真实物理状态的"能量距离"一一对应，那么**自洽的latent rollout就有真实性的数学保证**。

这个想法的理论基础存在于控制论（Lyapunov NN, event-triggered MPC的ISS保证），但**从来没有人把它迁移到learned world model的latent space中**。


## 3　Q3：想象可信度——"什么时候该相信想象、什么时候该回到现实"

### 3.1 问题本质

世界模型的价值来自"想象"——在脑中预演未来，省去真实物理交互的成本和风险。但想象本身可能出错：模型在没见过的状态区域会"幻觉"（输出物理上不可能但latent space中自洽的预测），而机器人无法从内部判断一个想象是否可靠。

这引出一个**元认知（metacognition）**问题——机器人不仅需要预测未来，还需要**知道自己对这个预测有多不确定、这个不确定性来自哪里、以及基于此该做什么**。

### 3.2 已有工作谱系

**路线A：不确定性量化与分解**

| 工作 | 机制 | 根本局限 |
|---|---|---|
| PETS (Chua et al., NeurIPS 2018, 1805.12114, 开源) | Bootstrap ensemble (B=5) probabilistic NN → ensemble disagreement = epistemic, 单模型概率输出 = aleatoric。Trajectory Sampling传播不确定性 | 分解是**启发式的**（ensemble disagreement作为epistemic的代理），缺乏因果层面的分解理论。**没有证明"ensemble disagreement = epistemic uncertainty"的充分必要性** |
| BNN+LV (Depeweg et al., 1706.08495) | 权重不确定性(W)捕获epistemic + latent input variable(z)捕获异方差/多模态noise。信息论分解：I(y;W\|x,D) = epistemic | 分解依赖两个理论（熵/方差）给出一致结论的强假设。在真实物理系统中这个假设经常不成立 |
| DEUP (Bengio组, TMLR 2023, 2102.08501, 开源) | 不依赖后验采样，训练secondary "error predictor"直接估计主预测器的泛化误差（excess risk） | Error predictor本身也需要校准——**递归问题**：谁来预测error predictor的error？ |
| LUCCa (Berenson组, WAFR 2024, 2409.08249, 开源) | Local conformal calibration：在状态-动作空间的局部邻域内做共形校准，给出non-asymptotic保证 | **"局部"依赖距离度量**——当度量与真实uncertainty manifold不匹配时局部校准失效。且**多步联合coverage无composition理论** |

**路线B："想象vs真实"作为决策问题**

| 工作 | 机制 | 根本局限 |
|---|---|---|
| MBPO (Janner et al., NeurIPS 2019, 1906.08253, 开源) | 基于bounded model error的monotonic improvement guarantee。Branched rollout：从real buffer出发只rollout k步 | **k对所有状态使用同一值**——但模型在不同区域的不确定性不同。k的annealing是heuristic手动schedule |
| COPlanner (Huang组, 2310.07220) | 不确定性在planner中作为cost（保守）→在explorer中作为bonus（乐观） | **保守/乐观切换是二元的**——没有连续的信任度。且没有考虑真机采样的**物理安全成本** |
| P2P (NeurIPS 2022, 2301.08502) | 让模型学习"预见"(foresee)未来不确定性——把不确定性预测本身变成序贯决策 | "预见"和"最小化一步预测误差"之间存在**目标冲突**——平衡权重是启发式的 |

**路线C：共形预测在非平稳数据下的保证**

| 工作 | 机制 | 根本局限 |
|---|---|---|
| ACI (Gibbs & Candès, NeurIPS 2021 Oral, 2106.00170) | α_{t+1} = α_t + γ·(α - err_t)。在线梯度下降式更新，保证long-term coverage → 1-α | **long-term coverage是弱保证**："过去平均覆盖了"≠"当前关键时刻有覆盖"。step-size γ对漂移速率敏感 |
| SAOC (Bhatnagar et al., ICML 2023, 2302.07869) | Strongly adaptive online conformal：在所有子区间上同时保证coverage | 所有子区间的coverage是**marginal的集合，不是joint guarantee**。多步联合coverage问题仍未解决 |
| Doubly Robust CP (JRSSB 2024) | Coverage bias = 分位数估计误差 × 似然比估计误差。只要一个估计准确，bias为零 | **渐近理论**——机器人安全需要finite-sample guarantee。且在极端distribution shift下两个估计可能都错 |

### 3.3 真实未解决的科学问题

**Open-3A：Latent rollout中不确定性的因果分解与传播**

这是Q3最核心的问题。在latent rollout的每一步，预测不确定性的来源可以因果地分为三类：
- **(A)** 模型的知识缺失（epistemic）→ 该去真机采样
- **(B)** 系统固有噪声（aleatoric）→ 无法通过采样消除
- **(C)** **前一步预测误差被动力学放大**（compounding）→ 需要更好的model，而非更多data

现有的不确定性量化方法（ensemble, BNN, conformal）给出的都是一个标量——"这里不确定"。但**不确定性的"原因"对于决策至关重要**：类型A要求去采样，类型B要求接受噪声，类型C要求改进模型架构或缩短horizon。

**核心挑战**：在latent rollout的因果图中，如何将每一步的预测误差归因到这三类？这需要超越"统计不确定性"（方差/熵），走向"因果不确定性"（counterfactual: 如果模型在这一步是完美的，后面还会不确定吗？）。

**Open-3B："想象-真实-停止"的POMDP形式化**

当前所有"何时信任模型"的方法都使用单变量阈值规则（uncertainty > θ → 真机采样）。但本质上，这是一个**有代价、有延迟、有信息增益的序贯决策问题**。

**核心构想**：将{想象训练, 真机采样, 安全停止, 恢复, 重规划}的切换形式化为POMDP：
- 状态：(latent belief state, task progress, model uncertainty, safety risk)
- 动作：{imagine_k_steps, collect_real, halt, recover, replan}
- 奖励：任务成功率 - 真机采样成本 - 安全违规成本
- 最优策略即为模型的**元认知策略**——它知道什么时候该相信自己、什么时候该去真机学习、什么时候该认输

这个POMDP的最优策略自然具备"在模型好的时候纯想象、模型不确定但安全时主动采样、模型不确定且危险时停止"的行为——不需要任何手工阈值。

**Open-3C：非平稳自主学习中的Online Calibration保证**

在机器人自主学习过程中，数据分布持续漂移（且漂移由机器人自身行为引起）。能否在线维护一个**finite-time, anytime-valid**的calibration guarantee——即任何时刻的uncertainty估计都与真实误差保持可证明的关系？

这需要超越标准共形预测的exchangeability假设。可能的方向：anytime-valid inference（e-processes, test martingales）用于构建永远有效的conformal prediction sets；sequential change-point detection用于检测"calibration regime change"；meta-calibration（学习"在非平稳环境中如何校准"）。


## 4　Q4：内生安全——"安全不作为外部模块，而是世界动作模型对自己能力边界的认识"

### 4.1 问题本质

当前机器人安全的几乎所有方法都遵循一个模式：**安全是外部施加的约束或后处理模块**。CBF是动作空间的QP滤波器，HJ reachability在状态空间划出不可进入的禁区，fallback policy是独立训练的恢复控制器，失败检测是独立的分类器。这种"外源性安全"的根本问题在于：**安全被定义为"不进入某些状态/不执行某些动作"，而不是"知道自己能力边界并在边界内行动"**。

对于人形机器人，"安全"的本质是**可恢复性**——一个状态是否安全不取决于它本身，而取决于"策略是否具备从该状态恢复的能力"。同一个状态（如单脚站立），对熟练的策略是安全的（能恢复），对稚嫩的策略是不安全的（会摔倒）。这意味着**安全不是状态的函数，而是（状态+策略+世界模型）的联合函数**——这正是"内生安全"的核心洞见。

### 4.2 已有工作谱系

**路线A：Latent Space中的安全分析**

| 工作 | 机制 | 根本局限 |
|---|---|---|
| Latent Safety Filter (Nakamura et al., RSS 2025, 2502.00935, 开源) | 在Dreamer风格RSSM的latent space中训练failure classifier，用HJ reachability求解安全值函数V(z)。安全的fallback policy在V(z)<0时最小干预接管 | **安全的定义仍然是外生的**——failure classifier需要labeled safety data。它把"不安全"从状态空间的手工定义替换为latent space的学习分类器，但安全的**概念**没有变：仍然是"不进入某个预设区域"。没有回答"可恢复性是策略+世界模型联合函数"的问题 |
| UNISafe (Bajcsy组, CoRL 2025, 2505.00779) | Ensemble latent dynamics + Jensen-Rényi Divergence量化epistemic uncertainty + conformal threshold → 在高不确定性区域也触发安全滤波 | Epistemic uncertainty作为"危险代理"的假设有两个方向的问题：高uncertainty≠危险（可能只是模型没见过但实际安全）；低uncertainty≠安全（模型可能自信地错）。且HALT是二元的安全or停止 |
| LatentCBF (Bajcsy组, L4DC 2026 Oral, 2511.18606, 开源) | 将latent HJ reachability的硬切换平滑化为CBF QP修正 | 仍然是**外部滤波**——安全来自"外部的CBF QP修正"，而非策略内部表征 |

**路线B：CBF与RL的融合**

| 工作 | 机制 | 根本局限 |
|---|---|---|
| Value Function is a CBF (UCL, ICML 2023 Outstanding Paper, 2306.04026, 开源) | 证明在特定reward结构下（不安全=0, 安全=1），最优值函数V*经过变换h=V*-R是一个有效CBF | 等价性依赖非常强的假设（确定性MDP、特定reward结构），在人形locomotion这类高随机性问题中不成立 |
| End-to-End Safe RL via CBF (Caltech, AAAI 2019, 1903.08792) | CBF在**训练阶段**约束探索空间，而非仅在部署时滤波 | CBF约束的仍是"禁止区域"——回到"安全=不进入某些状态" |

**路线C：恢复行为的路子**

| 工作 | 机制 | 根本局限 |
|---|---|---|
| FRASA (ICRA 2025, 2410.08655, 开源) | 单一RL agent+单一reward函数同时覆盖站立和起身——不需要"恢复策略"和"任务策略"分离 | 单一框架≠同一表征。**策略可能在内部对"站立"和"起身"使用了不连续的表征，只是被同一网络参数覆盖**。没有分析latent space中这两个行为的几何关系 |
| Agile But Safe (CMU, RSS 2024, 2401.17583, 开源) | Learned reach-avoid value network判断当前状态是否在"敏捷策略可自行恢复"的区域内 | 仍然是两策略+reach-avoid网络的**分离式架构**。没有解决"不在敏捷和恢复之间做二元切换"的核心问题 |
| ActSafe (ETH, ICLR 2025, 2410.09486, 开源) | 主动探索**safety boundary**（epistemic uncertainty最大但仍安全的区域） | 探索的是**dynamics uncertainty的boundary**，而非**recoverability boundary**。后者才真正携带"一点动作差异就决定恢复与否"的高信息量梯度 |

### 4.3 真实未解决的科学问题

**Open-4A：自指涉的安全性——将安全定义为"给定策略的可恢复区域"**

这是Q4——也是整个框架——最核心的概念创新。

**定义**：给定世界模型M（预测latent state演化）和策略π（从latent state到动作），在latent space Z中定义**自指涉的可恢复性函数**：

$$R(z; \pi, M) = \mathbb{P}\left(\exists \{a_t, ..., a_{t+N}\} \sim \pi \text{ s.t. } z_{t+N} \in S \mid z_t = z\right)$$

其中S是稳定姿态集合（如双脚站立、质心在支撑域内）。R(z) ∈ [0,1]表示"从latent state z出发，策略π在未来N步内能回到稳定姿态的概率"。

**R(z)的根本属性**：
- R(z)不是状态的函数——同一个z对不同π有不同的R
- R(z)随策略训练而变化——π改进后，R(z)自然增大（因为策略更有能力恢复）
- R(z)是连续的——不是"安全/不安全"的二元分类，而是"离不可恢复边界还有多远"的连续度量
- 如果R(z)能被可靠估计，**安全滤波不再需要**——策略只需感知R(z)的下降趋势，在同一latent space中自然调整行为
- 恢复行为不再需要——策略在R(z)下降时不是在"任务"和"恢复"之间切换，而是在**同一个latent space中重新规划**

**核心挑战**：(1) 如何高效估计R(z)？直接N步rollout计算量极大。HJ reachability通过解PDE提供了一条路，但需从"给定失败集F的前向分析"改造为"给定策略π的自指涉分析"。(2) R(z)的梯度是否可用于策略改进？——如果策略能在R(z)边界上训练，它学到的正是"精细控制以防止摔倒"。

**Open-4B：可恢复性边界的主动发现与课程学习**

**问题**：能否设计一个curriculum——不是按"任务难度"排序，而是按"到可恢复性边界的距离"排序？

**算法框架**：
1. 在当前策略π下，估计每个状态的R(z; π)
2. 采样R(z)接近0但非0的状态（"快回不来但还能回来"）
3. 在这些状态上微调π——因为这里的**梯度信息最丰富**（一点动作差异决定恢复与否）
4. 微调后π的能力扩展，可恢复性边界向外移动
5. 回到步骤1——边界的移动本身就是curriculum的推进

与ActSafe的区别：ActSafe探索的是dynamics uncertainty boundary——这是"模型不确定性"的边界。这里探索的是**recoverability boundary**——这是"策略能力"的边界。后者是专属于"世界动作模型"的边界结构。

**核心挑战**：如何保证步骤3不会意外越过recoverability boundary导致真实摔倒？——需要R(z)的估计具有**悲观保证**（pessimistic underestimate），这引出Open-4C。

**Open-4C：可恢复性估计的悲观校准**

R(z; π, M)是通过世界模型M估计的，而M本身可能错——可能把一个实际上不可恢复的状态预测为可恢复（假安全，致命），或把可恢复的预测为不可恢复（假危险，过度保守）。

**核心问题**：能否对R(z)的估计提供**高置信度的悲观保证**——即P(估计的R̂(z) ≤ 真实的R(z)) ≥ 1-δ？如果能，那么机器人就可以安全地在R̂(z) > θ的区域内训练，保证真实可恢复性不低于估计值。

这需要把共形预测或同时置信区间的方法应用于"通过latent dynamics估计的可恢复性"这一特殊量上——它既不是标准的回归问题（R依赖于rollout而非单步预测），也不是标准的分类问题（R是连续值）。


## 5　综合研究框架：四个科学问题如何构成一个博士阶段的研究主线

### 5.1 四个问题之间的逻辑关系

```
Q1: 物理隐变量
"世界模型应该表征什么"
    ↓ 决定了世界模型的内部结构
Q2: 长时序推演
"在什么粒度上预测未来"
    ↓ 决定了想象rollout的质量和效率
Q3: 想象可信度
"什么时候该相信想象"
    ↓ 决定了想象与真实采样的调度
Q4: 内生安全
"安全不是外挂，是知道自己能力边界"
    ↓ 闭环——安全来自Q1的物理理解+Q2的推演质量+Q3的元认知
```

Q1→Q2→Q3→Q4不是时间上的先后（它们可以并行研究），而是**逻辑上的依赖**：只有当世界模型的内部表征能够区分"哪些是物理隐变量、哪些不确定"（Q1），模型才能在长时序想象中做出有物理意义的粒度选择（Q2）；只有当想象的质量和不确定性可被量化（Q2的产出），机器人才有原材料去做"相信想象还是采样真实"的元认知决策（Q3）；而当这三层能力都具备时，"安全"就不再是外部施加的约束，而是模型对自己能力边界的内在认识——**Q4是Q1+Q2+Q3的自然涌现**。

### 5.2 推荐的研究推进路线

**阶段一（奠基，1-1.5年）：Q1 → Q3 的交叉**

主攻：**物理隐变量的贝叶斯信念建模 + 信念不确定性→决策**

具体切入：
- Open-1B（双时间尺度贝叶斯框架）+ Open-3B（想象-真实-停止POMDP的前半部分）
- 在一个受控但真实的物理交互场景中（如"抓取未知质量/摩擦的物体并搬运"），构建一个能同时跟踪"质量估计的不确定性"和"接触状态"的世界模型，并证明"质量不确定性下降→策略自然从试探过渡到执行"
- 这个阶段的核心产出：**证明了物理隐变量的信念状态可以驱动"探索→执行"的自然过渡，而不需要外部调度器**

**阶段二（深化，1-1.5年）：Q2**

主攻：**自适应预测粒度 + Latent Lyapunov函数**

具体切入：
- Open-2A（局部Lipschitz驱动的跳帧）+ Open-2C（Latent Lyapunov函数）
- 在阶段一的世界模型基础上，分析latent dynamics的局部Lipschitz常数与接触事件的关系，推导出最优预测粒度，使长程操作的想象误差显著低于逐帧预测
- 这个阶段的核心产出：**一个理论上有界的、粒度自适应的长时序想象机制；一个Lyapunov函数能预警"想象可能偏离真实"的信号**

**阶段三（闭环，1年）：Q4**

主攻：**自指涉安全 + 可恢复性边界课程**

具体切入：
- Open-4A（自指涉可恢复性函数R(z;π,M)）+ Open-4B（可恢复性边界curriculum）
- 在阶段一和阶段二的世界模型基础上，定义和估计R(z;π,M)，证明"安全性不必外挂——策略在R(z)下降时自然调整行为即可"
- 这个阶段的核心产出：**证明了"安全"可以从世界动作模型的内部表征中自然产生，而不是外部约束**

### 5.3 哪些方向适合作为博士主线 vs 辅助实验

| 方向 | 适合作为 | 理由 |
|---|---|---|
| **Open-1B+3B：双时间尺度物理信念+想象-真实POMDP** | **博士主线** | 理论深度足够（涉及贝叶斯推断+序贯决策+物理建模），实验可在逐步复杂的场景中展开（单物体→多物体→人形操作），有清晰的能力增量路径 |
| **Open-2A+2C：自适应预测粒度+Latent Lyapunov** | **博士主线** | 理论创新空间大（控制论×ML的交叉），但实验验证可能偏理论（在仿真中验证Lipschitz估计比在真机上容易）。需要和Q1或Q3结合才有物理落地场景 |
| **Open-4A：自指涉安全** | **博士主线（如果从阶段三开始做）或阶段三的自然延伸** | 概念最原创，但直接做Q4的难度是"需要Q1/Q2/Q3的某些基础"。如果做了Q1+Q2，Q4是自然的闭环；如果直接做Q4，需要先在简化设定下工作 |
| Open-1C（参数不确定→试探动作） | **辅助实验** | 有明确的对照实验设计，可以成为阶段一的第一个实验 |
| Open-3C（非平稳校准保证） | **辅助实验或独立小项目** | 理论性强，但可能更适合统计/ML theory背景的合作者 |
| 传感器选型、ring-buffer同步、硬件平台 | **实验支撑（不是研究贡献）** | 这些是验证上述科学问题所需的工程能力，但它们本身不构成科学贡献 |

### 5.4 可用的数据与评测基准

| 基准/平台 | 用途 | 与哪个问题相关 |
|---|---|---|
| Physion++ | 物理属性在线推断评测 | Q1（物理隐变量推断） |
| HumanoidBench | 人形全身控制benchmark | Q1-Q4（策略能力的baseline） |
| Push recovery / perturbation benchmarks | 人形抗扰动与恢复 | Q4（可恢复性R(z)的验证） |
| LIBERO-LONG | 长程操作任务 | Q2（长时序推演） |
| CALVIN-long | 连续指令长链 | Q2, Q3（想象-真实决策） |
| HoST/HumanUP 开源代码 | 人形起身策略基线 | Q4（恢复行为的baseline对照） |
| Isaac Lab + TacSL | 触觉+物理仿真 | Q1（物理隐变量仿真环境） |
| Dojo/Brax/JAX | 可微仿真+已知守恒量 | Q1, Q2（理论验证的ground truth） |

### 5.5 风险点

1. **Q1的风险**：物理隐变量的真实ground truth在真机中几乎无法获取——"真机的质量到底是1.23kg还是1.27kg"无法精确知道。这意味着**不能以"参数估计精度"作为核心指标**，而应该以"下游任务成功率/样本效率"作为评估标准。这正与Open-1A（功能可辨识性vs参数可辨识性）呼应。

2. **Q2的风险**：局部Lipschitz常数的可靠估计在高维latent space中可能极其困难。Jacobian的数值估计（有限差分）噪声大，解析计算（autodiff）贵。如果L(z)的估计本身不可靠，那么基于L(z)的自适应粒度选择就失去了基础。

3. **Q3的风险**："想象-真实-停止"POMDP的状态空间本身就是latent belief state——它的维度和含义随学习过程变化。这是一个"meta-POMDP"——底层POMDP的状态表征本身在学习中变化。

4. **Q4的风险**：自指涉可恢复性R(z;π,M)的定义依赖于M的准确性。如果M在相关区域有系统性偏差（把危险的当成安全的），R(z)的估计会有致命性的乐观偏差。**悲观校准（Open-4C）必须先行或至少同步进行。**

5. **整体风险**：这个框架在很大程度上是概念创新驱动而非benchmark驱动。现有的benchmark（HumanoidBench, LIBERO等）不一定能直接评测"物理隐变量的可辨识性"或"自指涉安全"——**可能需要自己设计评测协议和指标**。这在投稿时既是风险（审稿人不熟悉新指标）也是机会（定义新标准）。


## 附录：关键论文索引

### Q1相关
- DreamerV3: Hafner et al., Nature 2025, 2301.04104 [开源]
- TD-MPC2: Hansen et al., ICLR 2024, 2310.16828 [开源]
- RMA: Kumar et al., RSS 2021, 2107.04034 [开源]
- PIN-WM: 清华, RSS 2025, 2504.16693
- ContactNets: Posa et al., CoRL 2020, 2009.11193 [开源]
- AdaptiGraph: Li et al., RSS 2024, 2407.07889
- MTRSSM: Keio Univ., IEEE ICDL 2023
- DyWA: 清华, 2025
- BayesSim: NVIDIA/UW, RSS 2019, 1906.01728 [开源]

### Q2相关
- Director: Hafner et al., NeurIPS 2022, 2206.04114 [开源]
- Diffusion Forcing: MIT, NeurIPS 2024, 2407.01392 [开源]
- CoA: ByteDance, NeurIPS 2025, 2506.09990 [开源]
- Simulation Lemma改进: RLC 2024, 2406.16249 [开源]
- SSM Recency Bias: ICLR 2025, 2501.00658 [开源]
- PLSM: MPI, NeurIPS 2024, 2401.17835 [开源]
- T-IB: ICLR 2024
- MPC Horizon理论: Grüne et al., Springer专著

### Q3相关
- PETS: Chua et al., NeurIPS 2018, 1805.12114 [开源]
- DEUP: Bengio组, TMLR 2023, 2102.08501 [开源]
- LUCCa: Berenson组, WAFR 2024, 2409.08249 [开源]
- MBPO: Janner et al., NeurIPS 2019, 1906.08253 [开源]
- P2P: NeurIPS 2022, 2301.08502 [开源]
- ACI: Gibbs & Candès, NeurIPS 2021 Oral, 2106.00170
- Doubly Robust CP: JRSSB 2024

### Q4相关
- Latent Safety Filter: Nakamura et al., RSS 2025, 2502.00935 [开源]
- UNISafe: Bajcsy组, CoRL 2025, 2505.00779
- LatentCBF: Bajcsy组, L4DC 2026 Oral, 2511.18606 [开源]
- Value Function is a CBF: UCL, ICML 2023 Outstanding Paper, 2306.04026 [开源]
- FRASA: ICRA 2025, 2410.08655 [开源]
- Agile But Safe: CMU, RSS 2024, 2401.17583 [开源]
- ActSafe: ETH, ICLR 2025, 2410.09486 [开源]
- HoST: RSS 2025, 2502.08378 [开源]
- HumanUP: RSS 2025, 2502.12152

### 交叉/综述
- Locatello et al. (无监督disentanglement不可能定理), ICML 2019
- Four Principles for Physically Interpretable World Models, 2503.02143, 2025
- GIRL: Generative Imagination RL, 2604.07426, 2026⚠️

---

> **核实声明**：所有论文标题、arXiv编号、机构、开源状态均经WebSearch核实。标注⚠️的为2025年末或2026年预印本，编号和结论可能在正式发表时有变动。标注"待核实"的字段建议引用前二次确认。