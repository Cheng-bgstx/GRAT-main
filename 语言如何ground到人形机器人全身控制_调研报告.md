# 面向人形机器人VLA的科学问题调研报告（一）：语言如何Ground到全身控制

> 调研日期：2026-06-04
> 负责Agent：语言Grounding与全身控制Agent
> 核心科学问题：语义层次与运动层次之间的表征鸿沟、语言对物理约束的编码、语言引导的运动生成

---

## 一、总体述评：问题的定位

当前的VLA研究大多将"语言→动作"视为一个可学习的黑箱映射——将language token作为context输入Transformer，让模型从大规模机器人数据中隐式地建立关联。这种端到端范式在实验室内取得了令人瞩目的效果（RT-2, OpenVLA, π0），但掩盖了一系列深层科学问题。

本报告围绕四个子问题展开结构化调研，区分"工程上已经做到"与"科学上尚未理解"，并对每篇核心论文给出真实评估。

---

## 二、子问题1：语言→动作的"Grounding Gap"究竟卡在哪？

### 2.1 问题定性

语言描述是抽象的、符号的、离散的（"拿起杯子"）——它只指定了目标，没有指定手型、力度、重心转移、步态调整。现有VLA把语言作为context token输入策略，让模型**隐式学习grounding**——但学到的是"语言-动作"的表面关联还是真正的语义理解？

我们关注三个层次：
- **语义层**：任务是什么（what）
- **可供性层**：哪些物体/区域可以交互（where / what can be done）
- **运动层**：用什么动作（how）

核心科学问题：这三个层次的grounding是否可以解耦——使得每个层次的泛化是独立的、可验证的？

---

### 2.2 核心论文

#### Paper 1: RT-2——将动作token化，让VLM直接输出动作

- **英文标题**：RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- **机构**：Google DeepMind（54位作者）
- **Venue / 年份**：CoRL 2023
- **arXiv ID**：2307.15818
- **是否开源**：模型权重未公开；技术方案公开
- **核心机制**：
  - 将机器人动作表示为文本token（如`"terminate, Δx, Δy, Δz, Δrotx, Δroty, Δrotz, gripper_extension"`），每个连续值离散化为256个bin
  - 在PaLI-X / PaLM-E等VLM基础上co-fine-tune，同时保留原始VLM数据和机器人演示数据（比例1:2至2:3）
  - 语言直接作为prefix token与图像一起输入，模型自回归输出action token
- **关键结果**：未见场景泛化率从RT-1的32%提升至62%；涌现了符号理解、推理、人的识别和Chain-of-Thought能力
- **真实局限（科学层面）**：
  1. **运动技能受限于演示数据**：RT-2在语义层面泛化强（能理解"把苹果放到数字3上"），但运动层面只能复现数据集中出现的动作模式。语言→动作的grounding在运动层没有泛化。
  2. **学到了表面关联而非语义理解**：论文未做语言替换诊断实验——如果训练数据中"pick up"总是与特定抓取轨迹共现，模型学到的是这种共现，而非"pick up"的语义本质。
  3. **动作空间的粗离散化**：256-bin离散化对末端位姿尚可，但远远不足以表达人形机器人的全身关节空间（50+DoF）
- **分类**：**已有但未解决**

---

#### Paper 2: SayCan——LLM语义规划 + RL价值函数的双通道Grounding

- **英文标题**：Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
- **机构**：Google Robotics / Everyday Robots
- **Venue / 年份**：CoRL 2022
- **arXiv ID**：2204.01691
- **是否开源**：模拟环境开源（tabletop）；真实系统部分开源
- **核心机制**：
  - LLM提供候选动作（以自然语言形式）及其先验概率 P(action | instruction)
  - 学习到的价值函数（affordance model）给出该动作在当前状态下可执行的概率 P(success | state, action)
  - 最终选择：argmax P(action | instruction) * P(success | state, action)
  - 这是**显式两阶段grounding**：语言的语义贡献和环境的物理可行性被分解为两个独立的概率项
- **真实局限（科学层面）**：
  1. **语言和运动通过乘法门控耦合**：乘积形式假设语义先验和物理可行性独立——但在真实任务中，语义本身就包含物理信息（"轻拿"意味着特定的力控需求），这两个信号不独立
  2. **LLM不感知物理状态**：SayCan中的LLM只看语言指令，不看视觉观察——因此无法做"看到杯子快倒了所以减速"这类动态调整
  3. **运动层面没有语义参与**：低层策略是预训练的、固定的——语言只在高层做动作选择，不参与底层运动生成的调节
- **分类**：**已有但未解决**

---

#### Paper 3: Hi Robot——分层语言分解，System 2/System 1双速架构

- **英文标题**：Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models
- **机构**：Physical Intelligence (π), 多位作者（Lucy Xiaoyang Shi, Brian Ichter, Sergey Levine, Chelsea Finn等）
- **Venue / 年份**：2025年2月发布，arXiv preprint
- **arXiv ID**：2502.19417
- **是否开源**：项目网站公开，代码未完全开源
- **核心机制**：
  - 高层（System 2，~1Hz）：VLM接收复杂开放指令+当前图像→输出原子化简单语言指令（如"抓取杯子"）
  - 低层（System 1，~50Hz）：基于π0的VLA，接收原子指令→输出连续动作chunk
  - 高层仅在~1秒后或接收到新用户语言输入时重新推理
  - 支持中途实时语言纠正（"那不是垃圾"、"别动它"）
  - 合成数据训练：给定原子命令，让VLM生成可能的复杂前置指令
- **真实局限（科学层面）**：
  1. **语言分解是单向的**：高层将复杂语言→简单语言，但低层无法**反向请求**更细致的指导——"当VLA遇到不确定情况时，主动请求语义澄清"的能力缺失
  2. **粒度切换是固定的**：高层总是输出同样的原子化粒度，没有根据任务复杂度/当前状态自适应调整粒度的机制
  3. **中间表示（原子指令）本身是语言**：从语言到语言再到动作——中间没有物理约束层。如果低层VLA在某个场景下需要更细的物理指导（如"保持杯子水平"），高层无法提供，因为它的输出空间仍是非物理的
- **分类**：**已有但未解决**

---

#### Paper 4: RoboBrain——从抽象到具体的统一大脑，显式建模三层表征

- **英文标题**：RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete
- **机构**：北京大学、北京智源人工智能研究院(BAAI)、中科院、香港大学
- **Venue / 年份**：CVPR 2025
- **arXiv ID**：2502.21257
- **是否开源**：是（HuggingFace模型+GitHub代码）
- **核心机制**：
  - 基于LLaVA架构（SigLIP + Qwen2.5-7B），在三个层次上输出：
    - **规划层（Planning）**：复杂任务→子任务序列（QA对形式）
    - **可供性层（Affordance）**：A-LoRA模块→输出任务相关物体的可交互区域坐标
    - **轨迹层（Trajectory）**：T-LoRA模块→输出末端执行器轨迹的关键点序列
  - 三层次通过ShareRobot数据集（从Open X-Embodiment标注得到）联合训练
- **真实局限（科学层面）**：
  1. **三层是独立输出的**：A-LoRA和T-LoRA分别输出affordance和轨迹——但它们之间没有显式的物理耦合。Affordance（"握把手"）和轨迹（具体运动路径）之间缺乏力传递、接触约束等物理约束的显式建模
  2. **轨迹预测是2D关键点序列**：远不能表达人形机器人全身控制的复杂性（重心、接触力、关节限位）
  3. **语言主要在规划和可供性层起作用**：轨迹层不直接接收语言信号——因此"端稳"这类物理语言描述无法参与轨迹生成
- **分类**：**已有但未解决**

---

#### Paper 5: SpatialVLA——将3D空间位置显式编码到VLA中

- **英文标题**：SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model
- **机构**：复旦大学、上海AI Lab、中国电信AI研究院、上海交大、浙大、上海科技大学
- **Venue / 年份**：RSS 2025
- **arXiv ID**：2501.15830
- **是否开源**：是（项目页面+模型权重）
- **核心机制**：
  - **Ego3D Position Encoding**：将显式3D空间信息注入VLA的观测输入，弥合2D视觉输入与3D物理动作空间之间的鸿沟
  - **Adaptive Action Grids**：使用自适应离散化动作网格表示机器人运动，取代连续动作空间，支持跨机器人迁移
  - 在110万真实机器人episode上预训练，支持zero-shot部署
- **真实局限（科学层面）**：
  1. **空间≠语义**：SpatialVLA解决了"where"的问题，但没有解决"how semantically"的问题——"拿起"、"托起"、"推起"都涉及x,y,z位移但语义不同
  2. **3D位置编码仅在输入端**：Action head仍然是离散网格——语言的物理语义（如"轻""稳""快"）无法通过这些空间网格直接编码
  3. **Adaptive Action Grids是针对末端执行器的**：扩展到全身控制仍然面临维度灾难
- **分类**：**已有但未解决**

---

#### Paper 6: 诊断性研究——VLA的语言泛化到底有多脆弱？

**Paper 6a**："Bring the Apple, Not the Sofa: Impact of Irrelevant Context in Embodied AI Commands on VLA Models"（2025年10月，arXiv:2510.07067）
- 对VLA进行**系统性的语言扰动测试**：
  - 人工同义改写→**约20%性能下降**
  - 语义/词汇相似但无关的上下文→**约50%性能下降**
  - 随机无关上下文→下降在10%以内（相对鲁棒）
- 这提供了**直接证据**：现有VLA学到的确实是"语言-动作"的表面共现，而非真正的语义理解。如果模型真正理解了"拿起杯子"的语义，同义改写应该零影响

**Paper 6b**："From Intention to Execution: Probing the Generalization Boundaries of Vision-Language-Action Models"（2025年6月，arXiv:2506.09930）
- 构建50个仿真任务的统一诊断套件，横跨语言指令、视觉和物体三个维度
- 核心发现：VLA表现出"好的intention"（从VLM backbone继承的鲁棒感知和高层规划）但"执行失败"（OOD场景下动作泛化差）
- 这是一个**感知-执行鸿沟**（perception-to-action gap）

**Paper 6c**："Unveiling the Compositional Ability Gap in Vision-Language Reasoning Model"（NeurIPS 2025，arXiv:2505.19406）
- 构建ComPABench，发现VLM在跨模态、跨任务的组合泛化上存在显著gap
- 提出的RL-Ground策略（强制视觉到文本的grounding + 渐进奖励）能部分缩小这一gap
- 分类：**已有但未解决**

---

### 2.3 子问题1的小结

| 论文 | Grounding方式 | 语言参与的层次 | 语言→动作的中间表征 | 是否做了泛化诊断 |
|------|-------------|--------------|-----------------|---------------|
| RT-2 | 隐式（co-fine-tune） | 统一token序列 | 无 | 否 |
| SayCan | 显式（概率乘积） | 高层动作选择 | 无（LLM直接输出动作名） | 否 |
| Hi Robot | 分层（语言→语言→动作） | 高层：指令理解；低层：原子命令 | 语言（原子指令） | 否 |
| RoboBrain | 三层解耦（规划/可供性/轨迹） | 规划层+可供性层 | Affordance坐标（2D） | 否 |
| SpatialVLA | 空间显式编码 | 输入端（3D位置） | 空间网格 | 否 |
| 诊断研究 | N/A（meta分析） | N/A | N/A | **是**（同义改写↓20%，无关上下文↓50%） |

**核心结论**：语言→动作grounding的隐式学习在in-distribution场景有效，但在语义等价泛化上脆弱（证明学到的是表面关联）。三层解耦（语义/可供性/运动）是最有前途的方向，但现有工作**未在任意两层之间建立物理约束的显式耦合**。

---

## 三、子问题2：语言能否编码物理常识和约束？

### 3.1 问题定性

人类语言中隐含了大量物理常识：
- "轻拿轻放" = 力要小 + 速度减慢
- "端稳" = 保持水平 + 降低加速度
- "小心翼翼" = 减速 + 降低重心 + 增大稳定裕度

VLM在预训练中学到了某种物理常识（通过文本共现的统计关联），但这能否被**系统地提取**并**转化为动作生成的软约束/先验**？我们讨论的是：

**"语言→物理约束→约束条件化动作生成"的三段式 pipeline**，而非"语言→动作"的直接映射

---

### 3.2 核心论文

#### Paper 7: PhysBench——VLM物理世界理解的基准

- **英文标题**：PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding
- **机构**：USC, UC Berkeley, Toyota Research Institute
- **Venue / 年份**：ICLR 2025
- **arXiv ID**：2501.16411
- **是否开源**：是（GitHub + HuggingFace数据集）
- **核心机制**：
  - 10,002个条目，覆盖四大物理领域（物体性质、物体关系、场景理解、物理动力学），19个子类，8个能力维度
  - 人工标注超过4000小时
  - PhysAgent框架：VLM + 专用视觉模型（深度估计等） + 物理知识记忆库 → 在GPT-4o上实现+18.4%提升
  - 在MOKA系统上用5个真实机器人操作任务验证：affordance抓取、力控制、颜色选择、计数抓取、工具选择
- **关键发现**：
  - 最好VLM得分仅~52%（人类95.87%，随机25%）
  - VLM在**物理场景理解**和**物理动力学**方面尤其薄弱
- **真实局限（科学层面）**：
  1. **基准是MCQ格式**：能评测物理知识但不等于能将知识转化为动作约束。知道"玻璃易碎"和能生成"力小于X牛顿的抓取动作"之间有巨大鸿沟
  2. **PhysAgent的物理知识记忆库是手工构建的**：不可扩展，无法覆盖开放世界的物理场景
  3. **没有提供"语言→物理约束"的形式化框架**：PhysAgent仍然在做VQA，而非constraint generation
- **分类**：**已有但未解决**

---

#### Paper 8: Cosmos-Reason1——从物理常识到具身推理

- **英文标题**：Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning
- **机构**：NVIDIA
- **Venue / 年份**：2025年3月发布，arXiv preprint
- **arXiv ID**：2503.15558
- **是否开源**：是（模型权重在HuggingFace，NVIDIA Open Model License）
- **核心机制**：
  - 构建**物理常识层级本体论**：空间（关系/合理性/可供性/环境）、时间（动作/顺序/因果/规划）、基础物理（属性/状态/物体恒存/力学/电磁/热力学/反物理）
  - **具身推理二维本体论**：5类agent（人/动物/机械臂/人形机器人/自动驾驶车）×4种推理能力
  - 四阶段训练：Vision pretraining → General SFT → Physical AI SFT（400万视频-文本对+CoT推理，部分从DeepSeek-R1蒸馏）→ Physical AI RL（GRPO，规则可验证奖励）
  - 7B模型匹配/超过GPT-4o和Gemini 2.0 Flash；56B模型在物理常识上略超OpenAI o1
  - 直觉物理（时间箭头、空间谜题、物体恒存）上+32.4%
- **真实局限（科学层面）**：
  1. **CoT推理输出的是自然语言，不是可执行的约束**：模型能说出"这个杯子是玻璃的，所以需要轻柔处理"，但没有机制将这句话转化为力/力矩/速度的数值约束
  2. **本体论是静态的、手工设计的**：16个物理子类——但真实物理世界中的约束远不止这些，且很多是交叉的（如"在光滑表面上推动重物"涉及摩擦力、质量、接触稳定性）
  3. **RL阶段使用MCQ格式奖励**：奖励的是"选择正确的物理推理"而非"产生正确的物理行为"——这是proxy task，不是真正的物理grounding
- **分类**：**已有但未解决**

---

#### Paper 9: CoPa——VLM提取空间约束用于操作

- **英文标题**：CoPa: General Robotic Manipulation through Spatial Constraints of Parts with Foundation Models
- **机构**：多机构合作
- **Venue / 年份**：IROS 2024
- **arXiv ID**：2403.08248
- **是否开源**：部分开源
- **核心机制**：
  - VLM识别任务相关物体部件的**空间几何约束**（如"抓把手，沿铰链轴移动"）
  - 将操作分解为：任务导向的抓取 + 任务感知的运动规划
  - 从VLM中提取6-DoF抓取姿态和后抓取运动路径的几何约束
- **真实局限（科学层面）**：
  1. **约束仅限于空间几何**：没有力约束、没有时间约束、没有稳定性约束
  2. **VLM的约束提取是一次性的**：不参与执行过程中的动态约束调整
  3. **没有从语言中提取"物理暗示"的通用框架**：CoPa的prompt是任务定制的，不可泛化
- **分类**：**已有但未解决**

---

#### Paper 10: PRoC3S——LLM生成连续约束满足问题

- **英文标题**：Trust the PRoC3S: Solving Long-Horizon Robotics Problems with LLMs and Constraint Satisfaction
- **机构**：多机构
- **Venue / 年份**：2024年，arXiv preprint
- **arXiv ID**：2406.05572
- **是否开源**：是
- **核心机制**：
  - LLM输出**Continuous Constraint Satisfaction Problem (CCSP)**定义（含开放参数）
  - 通过采样/优化求解运动学、几何和物理约束
  - 不可满足的约束会触发LLM重提示
- **真实局限（科学层面）**：
  1. **约束来自LLM的常识，而非从语言中提取**：PRoC3S让LLM凭空生成约束——如果语言指令中说"小心"，这个信息无法进入约束生成过程
  2. **CCSP的求解器是符号化的**：不支持soft constraint和先验——而语言的物理暗示恰恰是"软约束"（"轻一点"不是硬边界，而是偏好）
  3. **离线的**：不支持在线replan的约束调整
- **分类**：**已有但未解决**

---

#### Paper 11: ZLATTE——语言驱动的轨迹重塑（利用几何感知约束）

- **英文标题**：ZLATTE: A Geometry-Aware, Learning-Free Framework for Language-Driven Trajectory Reshaping in Human-Robot Interaction
- **机构**：多机构
- **Venue / 年份**：2025年
- **arXiv ID**：2509.06031
- **是否开源**：待确认
- **核心机制**：
  - VLM将物体注册为几何基元；LLM将自然语言转化为**显式几何和运动学软约束**
  - 通过势场优化整合约束
  - 多agent策略处理冲突约束
- **真实局限（科学层面）**：
  1. **几何约束 ≠ 物理约束**：ZLATTE能处理"绕开障碍物"但不能处理"轻放"、"保持平衡"
  2. **势场优化的局部性**：可能陷入局部极小，不保证全局最优
- **分类**：**已有但未解决**

---

#### Paper 12: 语言+物理因子知识图谱

- **英文标题**：LLM with Physical Factors for the Robot Manipulation
- **机构**：多机构
- **Venue / 年份**：TAROS 2025
- **核心机制**：
  - 将物理因子（关节限位、质量、摩擦、弹性、重力、奇异性）编码为知识图谱
  - LLM Planner生成轨迹→物理因子验证→迭代修正
- **真实局限**：知识图谱依赖手工构建和标注；物理因子集不完整；未与语言描述中的物理暗示做关联
- **分类**：**已有但未解决**

---

### 3.3 子问题2的小结

| 论文 | 物理常识来源 | 从语言中提取物理暗示 | 约束形式 | 约束参与动作生成的方式 |
|------|------------|-------------------|---------|------------------|
| PhysBench | VLM统计共现 | 否（MCQ评测） | 选择题答案（离散） | 不参与（仅评测） |
| Cosmos-Reason1 | VLM+RL训练 | 否（本体论驱动） | 自然语言CoT | 不参与（推理输出） |
| CoPa | VLM prompt | 否（任务定制） | 空间几何约束 | 二次规划求解 |
| PRoC3S | LLM常识 | 否（LLM自主生成） | CCSP（连续） | 采样/优化求解 |
| ZLATTE | VLM+LLM | **是（NL→几何软约束）** | 势场 | 势场优化 |
| LLM+KG | 知识图谱（手工） | 否 | 物理因子布尔约束 | 迭代验证+重规划 |

**核心发现**：**"语言中的物理暗示→可执行约束→约束条件化动作生成"的三段式pattern在现有文献中几乎不存在**。现有工作或者（a）评测VLM是否"知道"物理知识（MCQ），或者（b）让LLM自主生成约束（而非从语言指令中提取），或者（c）只处理几何约束而不处理力/稳定性/时间约束。"端稳"→施加"末端保持水平"的先验——这种从语言词汇到latent动作空间约束的直接映射是**基本未被研究**的。

---

## 四、子问题3：任务指令的粒度与策略的自主性

### 4.1 问题定性

语言指令的粒度差异巨大："拿起杯子" vs "把杯子放到嘴边，倾斜约30度，让液体接触嘴唇"。现有VLA要么在固定粒度上训练，要么做固定的两层分解（Hi Robot）。但人类在运动执行中是**动态决定**"现在该想多细"的。

核心科学问题：能否让VLA学会"什么时候需要更细粒度的语义指导，什么时候只需要粗粒度的目标"——即VLA的"attention to language"是动态的、任务阶段依赖的？

---

### 4.2 核心论文

#### Paper 13: Long-VLA——Phase-Aware动态注意力掩码

- **英文标题**：Long-VLA: Unleashing Long-Horizon Capability of Vision Language Action Model for Robot Manipulation
- **机构**：西湖大学、阿里达摩院
- **Venue / 年份**：CoRL 2025
- **arXiv ID**：2508.19958
- **是否开源**：是（项目页面+代码）
- **核心机制**：
  - 每个子任务被自适应分割为**"移动阶段"**和**"交互阶段"**
  - 二元掩码机制动态调整注意力输入：
    - 移动阶段：mask腕部相机→聚焦全局/广角相机（导航）
    - 交互阶段：mask全局相机→聚焦抓手视图（精细操作）
  - 架构无关，可插入现有VLA
  - **首个专为长程任务设计的端到端VLA**
- **真实局限（科学层面）**：
  1. **阶段划分是观测空间的注意，不是语言粒度的自适应**：模型基于视觉和状态判断"移动vs.交互"——而非判断"当前任务的理解是否有歧义，需要更细的语言指导"
  2. **没有"语言不确定性检测"机制**：当粗粒度指令"收起桌子"在遇到异常物体时产生歧义——模型不会意识到"我需要更详细的指令"
  3. **两阶段划分过于粗糙**：真实操作的子阶段远比moving/interacting更精细
- **分类**：**已有但未解决**

---

#### Paper 14: DreamVLA——块状结构化注意力解耦

- **英文标题**：DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge
- **机构**：多机构
- **Venue / 年份**：NeurIPS 2025
- **arXiv ID**：2507.04447
- **是否开源**：待确认
- **核心机制**：
  - 块状结构化注意力：显式mask动态/空间/语义三种知识之间的相互注意力
  - 防止信息泄漏，保持每种表征的纯净
  - 动态区域引导的世界知识预测
  - 扩散Transformer作为action head，将动作表征从共享潜在特征中解耦
- **真实局限**：不涉及语言粒度的动态调整；解耦是特征层面的而非控制层面的
- **分类**：**已有但未解决**

---

#### Paper 15: NanoVLA——基于任务复杂度的动态路由

- **英文标题**：NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies
- **机构**：多机构
- **Venue / 年份**：2025年
- **arXiv ID**：2510.25122
- **是否开源**：待确认
- **核心机制**：
  - 基于任务复杂度自适应分配轻量/重型backbone
  - 视觉-语言解耦：早期融合→后期融合，提高效率
  - 面向资源受限设备（Jetson Orin Nano）
  - 52倍推断加速，98%参数减少
- **真实局限**：路由是基于计算资源的——"简单任务用轻量模型"，但不是基于语义需求的——"歧义任务请求更多语言指导"
- **分类**：**已有但未解决**

---

#### Paper 16: Gemini Robotics 1.5——具身思维+分层架构

- **英文标题**：Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer
- **机构**：Google DeepMind
- **Venue / 年份**：2025年10月，arXiv preprint
- **arXiv ID**：2510.03342
- **是否开源**：SDK开源，模型权重未公开
- **核心机制**：
  - 同一checkpoint原生控制ALOHA、Franka双臂、Apollo人形机器人三种形态
  - **Embodied Thinking**：VLA在执行动作前生成自然语言思维链（先思考后行动）
  - Motion Transfer实现跨形态技能迁移
  - GR-ER 1.5 (VLM planner) + GR 1.5 (VLA executor) 分层架构
- **真实局限（科学层面）**：
  1. **"Thinking"是单向的**：VLM思考→VLA执行，VLA执行中的不确定性不会反向触发"重新思考"
  2. **思维链的粒度是固定的**：每次重新规划的输出粒度一致，没有根据当前不确定性动态调整
  3. **语言和动作通过思维链耦合**：但思维链是自然语言，仍然是符号化的中间层——物理层面的不确定性（"抓得够不够稳"）可能无法被充分编码到语言中
- **分类**：**已有但未解决**

---

#### Paper 17: VINE——使用成功和失败演示的分层VLA

- **英文标题**：Hierarchical Vision Language Action Model Using Success and Failure Demonstrations
- **机构**：多机构
- **Venue / 年份**：2025年12月
- **arXiv ID**：2512.03913
- **是否开源**：待确认
- **核心机制**：
  - System 2（高层）：可行性引导的树搜索 + 失败感知评分
  - System 1（低层）：闭环动作执行
  - 使用混合质量（成功+失败）的演示进行训练
  - 失败演示提供了"不该做什么"的负样本信号
- **真实局限**：仍为两层固定架构；失败感知仅在高层的搜索评分中使用，低层不主动检测不确定性
- **分类**：**已有但未解决**

---

### 4.3 子问题3的小结

| 论文 | 粒度调整机制 | 动态性来源 | 低层能否主动请求更多指导 |
|------|-----------|----------|-------------------|
| Long-VLA | 视觉注意力掩码 | 移动vs交互阶段（观测驱动） | 否 |
| DreamVLA | 特征注意力解耦 | 动态/空间/语义三种知识 | 否 |
| NanoVLA | 架构路由 | 任务复杂度（资源驱动） | 否 |
| Gemini Robotics 1.5 | 思维链长度 | LLM自主决定（语义驱动） | 否 |
| VINE | 树搜索深度 | 可行性失败检测 | 否 |

**核心发现**：**"动态语言粒度"在现有文献中几乎没有被研究**。现有的"动态"机制本质上都是观测驱动的（看到什么→调整注意力），而非任务理解驱动的（"我有多确定我理解了这个指令？"→调整对语言信号的依赖权重）。VLA的"attention to language"是任务阶段依赖的这一假设尚未被设计并验证过。

---

## 五、子问题4：跨本体（Cross-Embodiment）的语言Grounding

### 5.1 问题定性

"拿起杯子"对人类手、五指灵巧手、二指夹爪意味着完全不同的动作序列。但如果语言只编码"意图"而非"动作"，那同一个意图应该能泛化到不同本体。

核心科学问题：能否显式地分解为：
- **语言→任务空间**（本体无关）
- **任务空间→关节空间**（本体特定）

其中"任务空间"不只是末端位姿，而是包含接触点、力方向、物体affordance等更丰富的中间表示？

---

### 5.2 核心论文

#### Paper 18: RT-X / Open X-Embodiment——跨本体联合训练的开创性工作

- **英文标题**：Open X-Embodiment: Robotic Learning Datasets and RT-X Models
- **机构**：21个机构，约280+作者（Open X-Embodiment Collaboration）
- **Venue / 年份**：ICRA 2024
- **arXiv ID**：2310.08864
- **是否开源**：是（数据集+模型）
- **核心机制**：
  - 汇编22种机器人、527种技能、160,266个任务的数据集（RLDS格式）
  - RT-1-X（FiLM-conditioned EfficientNet + Transformer）和RT-2-X（PaLI VLM backbone，最高55B参数）
  - **关键发现**：在多种机器人数据上联合训练产生**正向迁移**——RT-2-X可以在Google Robot上执行从Bridge Dataset（WidowX）学到的任务
  - 使用action history进一步提升泛化
  - 更大模型持续表现更好
- **真实局限（科学层面）**：
  1. **跨本体泛化是隐式的**：模型在训练中见过多种本体，但未显式学习"本体无关的任务空间"
  2. **无法泛化到训练中未见过的本体**：如果数据集里没有人形机器人，RT-X无法控制人形机器人——而"语言→任务空间"的显式分解应该让人形机器人也受益
  3. **语言在不同本体上共享**：但在RT-X中，语言只是输入token——同一语言指令在不同本体上的不同动作映射完全由数据驱动。没有机制确保"语言空间"与"本体空间"的解耦
- **分类**：**已有但未解决**

---

#### Paper 19: Octo——开源通用机器人策略

- **英文标题**：Octo: An Open-Source Generalist Robot Policy
- **机构**：UC Berkeley, Stanford, CMU, Google DeepMind（Octo Model Team）
- **Venue / 年份**：RSS 2024
- **arXiv ID**：2405.12213
- **是否开源**：是（完全开源：训练pipeline + checkpoint + 数据）
- **核心机制**：
  - Transformer-based，80万trajectory来自Open X-Embodiment
  - 支持**语言指令**和**目标图像**两种任务指定方式
  - 扩散动作头（diffusion action head）：建模表达性强的连续动作分布
  - 支持9种机器人平台，跨4个机构评估
  - 支持高效fine-tuning（几小时在消费级GPU上适配新机器人）
- **真实局限（科学层面）**：
  1. **语言和goal image是两种并行的输入模态**：但没有机制说明为什么同一个task可以用两种模态等价指定——这恰好说明任务空间是本体无关的，但这个性质没有被显式利用
  2. **fine-tuning到新本体需要新数据**：不是零样本的跨本体泛化
  3. **扩散头输出的是关节/末端动作**：没有本体无关的中间任务空间
- **分类**：**已有但未解决**

---

#### Paper 20: GR00T N1——人形机器人通用基础模型

- **英文标题**：GR00T N1: An Open Foundation Model for Generalist Humanoid Robots
- **机构**：NVIDIA（Johan Bjorck, Linxi "Jim" Fan, Yuke Zhu等）
- **Venue / 年份**：2025年3月发布
- **arXiv ID**：2503.14734
- **是否开源**：是（模型权重在HuggingFace：nvidia/GR00T-N1-2B）
- **核心机制**：
  - **双系统架构**：System 2（VLM，~10Hz）理解环境与任务；System 1（Diffusion Transformer + flow matching，~120Hz）生成流畅运动
  - 跨本体支持：每个本体有自己的MLP state encoder和action decoder→共享嵌入空间；通过embodiment ID切换
  - 支持：单臂Franka Panda、双臂、双臂+灵巧手、人形Fourier GR-1
  - **数据金字塔策略**：人类视频+合成数据+真实遥操作数据→三层数据
  - VQ-VAE latent actions (LAPA) + Inverse Dynamics Model：从无动作标签数据推断伪动作标签
  - GR-1人形机器人真实世界：双臂pick-and-place with handover 76.6%；新物体放入未见容器 73.3%
- **真实局限（科学层面）**：
  1. **跨本体是通过embodiment-specific encoder/decoder实现的**——不是通过"语言→任务空间"的显式抽象。语言信号以相同方式输入所有本体，本体间的差异完全由encoder/decoder和embodiment ID处理
  2. **任务空间没有显式的物理语义**：共享嵌入空间是模型隐式学习到的——没有显式的接触点、力方向、物体affordance等丰富的中间表示
  3. **人形机器人的全身控制仍有限**：GR-1实验主要是上身操作+固定基座或简单行走，没有复杂的loco-manipulation协同
- **分类**：**已有但未解决**

---

#### Paper 21: WholeBodyVLA——全身loco-manipulation的语言引导

- **英文标题**：WholeBodyVLA: Towards Unified Latent VLA for Whole-Body Loco-Manipulation Control
- **机构**：香港大学（OpenDriveLab & MMLab）、智元AGIBOT、复旦大学、上海创智学院
- **Venue / 年份**：2025年12月
- **arXiv ID**：2512.11047
- **是否开源**：待确认
- **核心机制**：
  - 统一潜在学习框架：从低成本、无动作标注的第一人称视频中学习loco-manipulation知识
  - LMO Policy：将连续运动控制目标简化为离散运动指令（前进/转向/下蹲）
  - 在智元灵犀X2双足人形机器人上验证：行走+装箱+搬运+推车
  - 比此前基线高21.3%
- **真实局限（科学层面）**：
  1. **离散运动指令类似于"行为基元"而非语义丰富的语言**："前进"、"转向"是运动指令，不是语义任务指令（"把箱子搬到那边"）
  2. **语言在这里的作用是运动选择而非任务理解**——与RT-2的语言grounding是不同的问题
  3. **首次将VLA扩展至人形全身控制**是一项工程突破，但从科学角度看，语言的作用被矮化了
- **分类**：**已有但未解决**

---

#### Paper 22: π0——跨本体VLA的流匹配范式

- **英文标题**：π0: A Vision-Language-Action Flow Model for General Robot Control
- **机构**：Physical Intelligence
- **Venue / 年份**：RSS 2025
- **arXiv ID**：2410.24164
- **是否开源**：模型权重未全面开源；技术方案公开
- **核心机制**：
  - PaliGemma VLM backbone + conditional flow matching
  - "action expert"（第二组Transformer权重）专门处理机器人动作token——类似MoE
  - 10,000+小时机器人数据，7种配置68个任务 + 全部OXE
  - 跨本体：单臂、双臂、移动操作
  - 预训练→后训练（fine-tuning），类比LLM训练范式
- **真实局限（科学层面）**：
  1. **action expert是本体共享的**——意味着不同本体的动作差异被压缩到同一个expert中隐式学习，而非显式建模本体无关/相关两层
  2. **与GR00T N1类似**：语言→动作的映射中的本体相关和本体无关部分没有被区分
- **分类**：**已有但未解决**

---

#### 补充：跨本体表征的最新前沿

- **"Scaling Cross-Embodiment World Models for Dexterous Manipulation"**（He et al., UCSD/Harvard, 2025年11月，arXiv:2511.01177）：将手表示为**3D粒子集**，动作表示为粒子位移——这是一个**共享的本体无关表征**。通过前向运动学将关节动作映射到粒子空间。
- **3DFlowAction**（Zhi et al., 2025年6月）：使用视频扩散预测**物体中心3D流**作为本体无关的动作表征；跨本体零样本迁移70%（Franka→XTrainer）。
- **UniSkill**（Kim et al., 2025年5月）：从**不成对**的人-机器人视频中学习共享技能表征（latent diffusion）。

这些工作展示了"任务空间"可以比"末端位姿"更丰富——但**语言在这一任务空间中的角色仍未被充分探索**。

---

### 5.3 子问题4的小结

| 论文 | 跨本体方式 | 本体无关层 | 本体特定层 | 语言在本体无关层的作用 |
|------|----------|----------|----------|-----------------|
| RT-X | 联合数据训练（隐式） | 无显式层 | 无显式层 | 作为共享输入token |
| Octo | 联合数据训练 + fine-tuning | 无显式层 | 无显式层 | 作为一种任务指定方式 |
| GR00T N1 | Embodiment-specific encoder/decoder | 共享嵌入（隐式） | MLP encoder/decoder | 统一输入 |
| WholeBodyVLA | 统一潜在学习 | 潜在空间（隐式） | 动力学相关策略 | 离散运动指令 |
| π0 | Action expert共享 | 无显式层 | 无显式层 | 作为prefix token |
| 粒子/流表征 | 显式本体无关空间 | 3D粒子/流 | FK/IK映射 | **未涉及** |

**核心发现**：**"语言→任务空间→关节空间"的三段式显式分解在现有文献中不存在**。现有跨本体工作要么（a）隐式学习共享表示，要么（b）用embodiment-specific的输入输出层做适配，要么（c）构建本体无关的物理空间（粒子/流）但不涉及语言。"任务空间包含接触点、力方向、物体affordance"的丰富中间表示——且语言在该空间中编码意图——是**基本未被研究**的。

---

## 六、认知科学视角：人类如何用语言指导运动学习

### 6.1 Fitts & Posner三阶段理论及启示

**Fitts & Posner (1967)** 提出了运动技能习得的三阶段模型：

| 阶段 | 过程特征 | 语言的角色 | 注意需求 |
|------|---------|----------|---------|
| **认知阶段** (Cognitive / Verbal-Motor) | 理解"做什么"——形成技能的心智画面，动作慢、不一致、粗大错误多 | **语言至关重要**：视觉演示、口头指导、慢动作训练、增强反馈 | 极高——每一步都要有意识地思考 |
| **联结阶段** (Associative) | 将环境线索与所需动作关联——问题从"做什么"变为"怎么做" | **语言转向内在**：自我对话、错误识别、动作序列的串连，外在指令需求减少 | 中等——部分自动化 |
| **自主阶段** (Autonomous) | 大量练习后自动化——精确、一致、高效，看似毫不费力 | **语言退出**：几乎不需要外在语言指导；可同时执行次要任务 | 极小 |

---

### 6.2 对VLA设计的启示

1. **VLA的训练范式隐含地假设"认知阶段"的语言角色**：所有VLA都将语言指令作为持续输入——这对应于人类在认知阶段的"教练口头指导"。但在自主动作阶段，人类几乎不依赖语言。**VLA是否也应该有一个阶段退火机制——在训练早期依赖语言，在训练后期逐渐降低语言信号的权重？**

2. **Wulf等人的"注意焦点"研究**：运动科学中有大量证据表明**外部注意焦点**（关注动作的效果，如"让球落到那个位置"）优于**内部注意焦点**（关注身体运动本身，如"手腕屈曲30度"）。这与"语言应该编码意图（目标）而非运动细节"的观点一致——为"语言→任务空间"的分解提供了认知科学依据。

3. **VLA中的"遗忘"问题对应认知阶段**：当OpenVLA/RT-2在机器人数据上fine-tune后，其VLM预训练获得的一般语义能力会退化——这在认知科学中对应于：从认知阶段进入联结阶段时，注意力从"理解任务"转移到"执行动作"，语言处理资源被重新分配。

4. **Gentile的两阶段模型**（初始阶段 vs 后期阶段）强调**环境调节的匹配**：初学者需要匹配不变的环境特征（regulatory conditions），而专家可以在变化的非调节特征下完成任务。这对跨本体泛化有直接启发：**语言应该编码与环境无关的"意图"，而本体应该处理环境相关的"执行"**。

---

### 6.3 人类语言-动作神经基础的关键发现

- **具身语义学（Embodied Semantics）**：fMRI和TMS研究反复证实在理解动作词汇（如"踢"、"抓"）时，对应的运动皮层区域被激活。这意味着人类的"语言→动作"grounding不是symbolic的翻译，而是**在神经层面共享表征**——语言理解和运动控制在运动皮层中不是分离的。
- **对VLA的启示**：当前的VLA架构中，语言token和动作token在Transformer的不同层次中交互——但物理上它们是不同的token类型。如果参考人脑的共享表征，**语言和动作应该在同一个latent空间中被编码**，而非仅通过attention交换信息。

---

## 七、综合分析与分类矩阵

### 7.1 所有论文按"科学完成度"分类

| 分类 | 论文 |
|------|------|
| **已解决** | （无）——语言→全身动作的科学grounding问题没有一篇真正解决 |
| **已有但未解决** | RT-2, SayCan, Hi Robot, RoboBrain, SpatialVLA, PhysBench, Cosmos-Reason1, CoPa, PRoC3S, ZLATTE, Long-VLA, DreamVLA, NanoVLA, Gemini Robotics 1.5, RT-X, Octo, GR00T N1, WholeBodyVLA, π0, VINE |
| **基本未被研究** | 语言物理暗示→约束提取、动态语言粒度调整、语言→丰富任务空间→关节空间的三段式分解 |

### 7.2 四个子问题的研究状态

| 子问题 | 状态 | 最近的代表性方法 | 主要瓶颈 |
|--------|------|---------------|---------|
| 1. Grounding Gap | 有大量隐式方案，缺乏诊断和分解 | RT-2, RoboBrain, SpatialVLA | 未证明学到的是语义还是一起出现的共现模式 |
| 2. 物理约束编码 | 物理常识评测已建立；约束提取和转化未起步 | PhysBench, Cosmos-Reason1, CoPa | "知道→做到"之间的鸿沟 |
| 3. 动态语言粒度 | 基本未被研究 | Long-VLA（观测驱动，非语义驱动） | 缺乏不确定性感知的语言依赖机制 |
| 4. 跨本体Grounding | 隐式跨本体泛化已展示；显式分解未实现 | RT-X, GR00T N1, 粒子/流表征 | 语言→任务空间的显式建模缺失 |

---

## 八、三个真正有研究价值的开放科学问题

### Open Question 1：语言grounding的层次化解耦——语义层、可供性层、运动层是否可以独立泛化？

**现有瓶颈**：RT-2, OpenVLA等的隐式grounding在语义等价表述下脆弱（同义改写↓20%性能）。RoboBrain等三层架构的各层之间没有物理耦合。

**科学假设**：如果语言→动作grounding被**显式分解**为三个层次：
- **语义层**（L→S）：语言被映射到"任务意图空间"（本体无关、场景无关）——这应该对语义等价改写具有鲁棒性
- **可供性层**（S→A）：任务意图+场景观测→物体/区域的交互可能性（哪里可以抓、哪里需要避碰）
- **运动层**（A→M）：可供性+当前本体状态→具体动作

那么**每一层的泛化能力可以被独立验证**——例如语义等价改写应不影响语义层的输出，只改变运动层（如果涉及不同动作）或完全不影响（如果任务相同但表述不同）。

**关键实验**：训练三个独立模块，测试（a）语义等价改写是否只在语义层产生一致的embedding，（b）新的本体是否只需要改变运动层，上游层不受影响。

**为什么被低估**：因为端到端VLA太成功了（在in-distribution场景）。但端到端的成功掩盖了泛化的脆性。层次化解耦的价值在于**可诊断性**和**可组合性**——这是科学理解的核心需求。

---

### Open Question 2：语言的"物理暗示"能否被系统地提取并转化为动作生成的条件先验？

**现有瓶颈**：PhysBench和Cosmos-Reason1能评测VLM的物理常识（MCQ形式），但没有任何工作将"轻拿轻放"→{force < τ_N, velocity < v_max}这种转化系统化。

**科学假设**：构建"语言→物理约束"的形式化映射：
- **词汇级映射**：建立语义词表→物理属性词典（"轻"→force < 阈值，"稳"→保持末端水平，"快"→velocity > 基线×2）
- **组合语义→组合约束**：利用LLM的compositional能力将复合物理暗示分解为可组合的约束（"轻而稳地拿起"→force < τ AND orientation variance < σ）
- **约束作为动作先验而非硬边界**：物理约束应作为latent动作空间的先验（如flow matching中的引导项）而非硬剪裁——因为实际环境中的精确阈值是未知的

**关键实验**：验证物理约束先验的效果——比较（a）不加约束的baseline VLA，（b）硬约束剪裁，（c）约束作为latent先验——在需要精细力控和稳定性控制的任务上的表现。

**为什么是hard problem**：（1）物理暗示是模糊的（"轻"到什么程度？），需要从上下文中推断；（2）约束可能冲突（"又快又稳"）；（3）约束是情境依赖的（"轻拿玻璃杯"vs"轻拿铁杯"需要不同的力阈值）。

---

### Open Question 3：VLA能否学会"什么时候该关注语言，什么时候该依赖自主运动"——动态语言依赖权重的机制是什么？

**现有瓶颈**：所有现有VLA将语言作为固定权重的输入——或全或无的两层架构（Hi Robot）。没有模型动态调整对语言的依赖权重。

**科学假设**：VLA应学习一个**动态语言门控（Dynamic Language Gating）**机制：
- 当VLA对当前动作的**不确定性**（entropy of action distribution）高时→增大语言信号的权重→请求更细粒度的语义指导
- 当VLA处于"自主运动阶段"（对应Fitts & Posner的自主阶段）→降低语言权重→依赖已学到的运动策略
- 门控信号本身应该是一个可学习的函数：g_t = f(observation, action_history, language_embedding, uncertainty)

**与Hi Robot的改进关系**：Hi Robot中高层VLM定期重新规划——但频率是固定的。动态语言门控意味着VLA主动决定"现在我不确定，需要VLM重新解读语言/重新规划"。这与"什么时候该向高层VLM请求重新规划"是同一个问题的两面。

**关键实验**：（a）在语言指令中注入歧义（故意使用模糊表述）→观察VLA是否增大语言依赖权重并求助；（b）在自主运动阶段（如重复性pick-and-place）→观察语言权重是否自然下降；（c）测试gate在不同任务阶段的行为模式是否与人类直觉一致。

**为什么被低估**：因为现有benchmark都是在"语言指令明确且持续相关"的假设下设计的——没有评测过"语言在某些阶段应该被忽略"的场景。

---

## 九、参考文献汇总

| 序号 | 论文 | arXiv ID | Venue | 年份 |
|------|------|----------|-------|------|
| 1 | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | 2307.15818 | CoRL | 2023 |
| 2 | Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan) | 2204.01691 | CoRL | 2022 |
| 3 | Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models | 2502.19417 | arXiv | 2025 |
| 4 | RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete | 2502.21257 | CVPR | 2025 |
| 5 | SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model | 2501.15830 | RSS | 2025 |
| 6 | PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding | 2501.16411 | ICLR | 2025 |
| 7 | Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning | 2503.15558 | arXiv | 2025 |
| 8 | CoPa: General Robotic Manipulation through Spatial Constraints of Parts with Foundation Models | 2403.08248 | IROS | 2024 |
| 9 | Trust the PRoC3S: Solving Long-Horizon Robotics Problems with LLMs and Constraint Satisfaction | 2406.05572 | arXiv | 2024 |
| 10 | ZLATTE: A Geometry-Aware, Learning-Free Framework for Language-Driven Trajectory Reshaping | 2509.06031 | arXiv | 2025 |
| 11 | Long-VLA: Unleashing Long-Horizon Capability of VLA for Robot Manipulation | 2508.19958 | CoRL | 2025 |
| 12 | DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge | 2507.04447 | NeurIPS | 2025 |
| 13 | NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies | 2510.25122 | arXiv | 2025 |
| 14 | Gemini Robotics 1.5 | 2510.03342 | arXiv | 2025 |
| 15 | Gemini Robotics: Bringing AI into the Physical World | 2503.20020 | arXiv | 2025 |
| 16 | VINE: Hierarchical VLA Using Success and Failure Demonstrations | 2512.03913 | arXiv | 2025 |
| 17 | Open X-Embodiment: Robotic Learning Datasets and RT-X Models | 2310.08864 | ICRA | 2024 |
| 18 | Octo: An Open-Source Generalist Robot Policy | 2405.12213 | RSS | 2024 |
| 19 | GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | 2503.14734 | arXiv | 2025 |
| 20 | WholeBodyVLA: Towards Unified Latent VLA for Whole-Body Loco-Manipulation Control | 2512.11047 | arXiv | 2025 |
| 21 | π0: A Vision-Language-Action Flow Model for General Robot Control | 2410.24164 | RSS | 2025 |
| 22 | OpenVLA: An Open-Source Vision-Language-Action Model | 2406.09246 | arXiv | 2024 |
| 23 | Bring the Apple, Not the Sofa (VLA语言扰动诊断) | 2510.07067 | arXiv | 2025 |
| 24 | From Intention to Execution: Probing VLA Generalization Boundaries | 2506.09930 | arXiv | 2025 |
| 25 | Unveiling the Compositional Ability Gap in VLM Reasoning | 2505.19406 | NeurIPS | 2025 |
| 26 | LLM with Physical Factors for Robot Manipulation | (TAROS) | TAROS | 2025 |
| 27 | Scaling Cross-Embodiment World Models for Dexterous Manipulation | 2511.01177 | arXiv | 2025 |
| 28 | Fitts & Posner (1967) - Human Performance (认知科学经典) | — | Brooks/Cole | 1967 |

---

*本报告为"面向人形机器人的视觉-语言-动作模型（VLA）深度研究框架"的第一部分。后续Agent将调研：动作表征的离散与连续、长程任务规划、物理可行性验证、失败检测与恢复、持续学习等方向。*
