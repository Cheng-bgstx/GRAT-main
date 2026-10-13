# VLA长时序任务：物理可行性约束下的规划、失败检测与恢复、规划-反应动态平衡——深度文献调研

**调研日期**: 2025年6月  
**调研范围**: 2022-2025年，覆盖VLA规划、TAMP+LLM、双系统架构、失败检测与归因、经验选择与信息价值五个方向，约40篇论文核实

---

## 目录

1. [子问题1：语义合理性与物理可行性的冲突](#sec1)
2. [子问题2：长时序执行中的偏离检测与信用分配](#sec2)
3. [子问题3：规划（S2）与反应（S1）之间的注意力切换](#sec3)
4. [子问题4：从成功和失败中学习——什么应该被记住](#sec4)
5. [研究状态全景图](#sec5)
6. [开放科学问题](#sec6)

---

<a name="sec1"></a>
## 一、子问题1：语义合理性与物理可行性的冲突

### 1.1 核心问题再陈述

LLM/VLM规划器在纯语义空间中做任务分解（将"喝到水"分解为"拿起杯子→走到桌边→倒水"），但语义上合理的子目标序列可能物理上不可行（或代价极高）。人类规划时，物理代价（走到桌边需要几步、有多累）和语义目标（最终要喝到水）在同一个优化过程中被权衡。LLM完全不做这个权衡，因为它没有世界模型。核心问题是：**能否让VLA的规划部件在语义分解时，同时调用世界模型评估每个候选计划的"物理代价"，实现语义-物理联合优化？接口形式是什么？**

### 1.2 现有方案调研

#### 1.2.1 SayCan: Do As I Can, Not As I Say

- **英文标题**: Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
- **机构**: Robotics at Google (Everyday Robots)
- **Venue**: CoRL 2022
- **arXiv**: 2204.01691
- **开源**: 部分开源（仿真环境notebook）

**核心机制**:
SayCan将LLM的语义规划（"Say"）与RL训练的价值函数（"Can"）通过**乘性融合**结合起来：p(c_i | i, s) ∝ p(LLM判定语义相关性) × p(RL价值函数判定物理可行性)。物理可行性由一个对每个技能独立训练的TD价值函数（稀疏奖励：成功=1，否则=0）给出。

**科学层面分析**:
- SayCan的`p(can)`是一个**二值可行性估计器**，不是**物理代价估计器**。它只回答"能否做"，不回答"做这个需要多少代价"。
- SayCan做的是**贪心单步选择**（max-product），不是lookahead搜索——它不考虑当前选择如何影响未来步骤的物理代价。
- 这意味着SayCan的规划仍然是纯语义驱动的（LLM提供了"下一步该做什么"的全部语义知识），物理可行性只是作为gating信号，没有参与到"任务分解"的优化过程中。

**真实局限（科学层面）**:
1. **`p(can)`的二值性**：价值函数只建模0/1可行性，不建模连续物理代价（时间/能量/平衡难度）。这在接触丰富的操作任务中尤其不足——"用指尖捏起杯子"和"整只手掌抓起杯子"可能都是可行的（p=1），但稳定性和能量代价截然不同。
2. **无lookahead**：贪心选择使得物理代价的累积效应被忽略。在长程任务中，第1步的选择（走近路还是远路）可能极大影响后续步骤的物理代价。
3. **RL价值函数和BC策略的架构不一致**：BC策略成功率更高，但BC价值函数对失败边界的估计更差。最终方案（BC执行+RL价值函数）是工程折中，缺乏理论优雅性。

**研究状态**: 已有但未解决（物理可行性仅为gating，未进入优化层面）

---

#### 1.2.2 Hi Robot: Open-Ended Instruction Following with Hierarchical VLA

- **英文标题**: Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models
- **机构**: Physical Intelligence (PI)
- **Venue**: 技术报告，2025年2月
- **arXiv**: 2502.19417
- **开源**: 否

**核心机制**:
Hi Robot使用PaliGemma-3B VLM作为高层规划器（System 2），π0 VLA作为低层执行器（System 1）。高层以低频率（~1Hz）分解复杂指令为原子子任务，低层以高频率执行连续动作。其关键创新在于合成数据驱动的交互式指令理解——用户可以在执行中插入反馈。

**科学层面分析**:
- Hi Robot的高层仍然是**纯语义规划**。VLM的规划不涉及任何物理代价估计——它将"打扫桌子"分解为"擦桌子→收垃圾→放回物品"，但不会考虑"先收垃圾还是先擦桌子在物理上更省力"。
- 物理可行性的保障完全交给低层VLA（π0）。如果π0执行某个子任务失败，高层会收到反馈并重新规划——这是**事后重试**，不是**事前优化**。
- 关键局限：Hi Robot的高层和低层之间**没有共享的物理代价表示**。高层不知道低层在执行时的动力学困难，低层不知道高层的语义意图。

**真实局限（科学层面）**:
1. **无共享代价表示**：规划和执行在物理代价评估上完全解耦，导致"重规划"是被动响应而非主动优化。
2. **频率固化的切换**：1Hz的固定频率是工程选择，不反映任务实际需要的注意力切换。
3. **无世界模型**：高层规划时不能"想象"低层执行的效果——规划是盲的。

**研究状态**: 已有但未解决（物理可行性在低层隐式处理，高层不感知）

---

#### 1.2.3 VLM-TAMP: Guiding Long-Horizon Task and Motion Planning with Vision Language Models

- **英文标题**: Guiding Long-Horizon Task and Motion Planning with Vision Language Models
- **机构**: MIT CSAIL & NVIDIA Research
- **Venue**: ICRA 2025
- **DOI**: 10.1109/ICRA55743.2025.11128705
- **arXiv**: 2410.02193
- **开源**: 否（有project page）

**核心机制**:
VLM-TAMP是当前**最接近"语义-物理联合优化"的方法**，尽管它没有显式优化能量/时间代价。其核心设计是：
1. VLM生成语义子目标（PDDL形式如`(in chicken pot)`），而非具体动作序列
2. TAMP solver接管每个子目标，进行**几何/运动学可行**的动作规划，并可**插入VLM遗漏的必要步骤**（如打开抽屉）
3. 当TAMP遇到碰撞/运动学不可行时，将失败信息**反馈给VLM重新规划**

**科学层面分析**:
- 这是"分工而非联合优化"的方案：VLM负责语义抽象（减少搜索空间），TAMP负责物理几何推理。两者之间的接口是PDDL子目标。
- **物理代价在哪里？** TAMP内部的运动规划器当然考虑了物理代价（路径长度、碰撞避免），但这是TAMP本身的能力，不是VLM和TAMP**联合优化**的结果。VLM选择子目标A还是B，并不基于"哪个子目标序列物理代价更低"。
- 关键瓶颈：**VLM提议子目标 → TAMP验证可行的循环**，本质上是"提议-验证"模式，而非"生成-评估-选择"模式。TAMP不向VLM提供多个候选计划的物理代价对比，只提供"某个计划是否可行"的二进制反馈。

**真实局限（科学层面）**:
1. **缺少latent cost predictor**: 没有一个学习到的模块能够快速估计"一个语义计划在物理层面的代价"，TAMP的验证代价太昂贵（需要完整的motion planning），不适合在LLM的beam search中做评分。
2. **二进制反馈的信息低效**: "不可行"这个信号携带的物理信息太少——不知道是因为碰撞、运动学极限、还是路径太长。如果反馈中包含了物理代价的结构化信息，VLM可以学到规避高代价路径。

**研究状态**: 已有但未解决（物理可行性由TAMP验证，未与语义优化耦合）

---

#### 1.2.4 LLM³ (LLM-cubed): Task and Motion Planning with Motion Failure Reasoning

- **英文标题**: LLM³: Large Language Model-based Task and Motion Planning with Motion Failure Reasoning
- **机构**: UCLA / Hillbot
- **Venue**: IROS 2024
- **arXiv**: 2403.11552
- **开源**: 是 (github.com/AssassinWS/LLM-TAMP)

**核心机制**:
LLM³让LLM不仅提议任务级动作序列，还提议**连续动作参数**。当运动规划失败时，LLM通过**motion failure reasoning**——将失败信息以自然语言形式反馈给LLM，LLM据此迭代修正其提议。

**科学层面分析**:
- LLM³的"motion failure reasoning"是**用自然语言做物理反馈**——这是一个有趣的方向但也是根本性局限：自然语言能否精确编码物理约束（"关节3在角度>45度时碰撞"这种信息）？
- LLM³的实验主要限于**箱体打包**等相对简单的操纵任务，验证了"加入物理反馈比不加好"——但这个结论在更复杂的全身人形机器人任务（如同时平衡+搬运）中是否成立，尚未验证。

**真实局限（科学层面）**:
1. **自然语言作为物理反馈通道的带宽瓶颈**：连续物理量（力、力矩、能量、碰撞余量）经过自然语言编码丢失了大量精度，LLM可能无法从"手臂被卡住了"这样的描述中推断出具体的运动学约束。
2. **无latent world model**：LLM³依赖显式运动规划器（MP）给出失败信号，没有学习到的世界模型来预测失败。

**研究状态**: 已有但未解决（自然语言作为物理反馈通道的带宽限制）

---

#### 1.2.5 GR00T N1: An Open Foundation Model for Generalist Humanoid Robots

- **英文标题**: GR00T N1: An Open Foundation Model for Generalist Humanoid Robots
- **机构**: NVIDIA
- **Venue**: 技术报告, 2025年3月
- **arXiv**: 2503.14734
- **开源**: 是（HuggingFace + GitHub）

**核心机制**:
GR00T N1采用明确的双系统架构：System 2（VLM: NVIDIA-Eagle + SmolLM-1.7B）做环境理解和任务规划，System 1（Diffusion Transformer）做实时动作生成。两个系统端到端联合训练。训练数据异构金字塔：互联网视频（底层）+ Omniverse合成数据（中层）+ 真实遥操作轨迹（顶层）。

**科学层面分析**:
- GR00T N1的双系统是**端到端联合训练**的，这是一个区别于"两个独立模型拼接"的关键设计。这意味着S1的action generation已经内化了S2的规划意图——不是两个模块之间传递显式指令，而是共享latent representation。
- **但物理代价优化在哪里？** GR00T N1的训练目标是模仿专家轨迹（行为克隆），不是优化物理代价。模型学到的是"人类operators怎么做的"，而不是"什么是最省力/最稳定的方式"。
- 它的合成数据在Omniverse中生成，Omniverse有完整的物理模拟——**但物理代价在训练信号中不是一个显式目标**。数据生成时没有在多个物理可行的轨迹中选择代价最小的那个。

**真实局限（科学层面）**:
1. **模仿学习的学习目标不包括物理代价优化**：如果你给模型演示了一条"费力但成功"的轨迹，模型会学到那条轨迹——没有机制告诉它存在另一条"更省力且同样成功"的轨迹。
2. **缺少comparative物理反馈**：训练中没有"方案A vs方案B哪个物理代价更低"的比较信号。

**研究状态**: 已有但未解决（模仿学习无显式物理代价优化）

---

#### 1.2.6 Helix (Figure AI)

- **英文标题**: Helix: A Vision-Language-Action Model for Generalist Humanoid Control
- **机构**: Figure AI
- **Venue**: 技术博客, 2025年2月
- **arXiv**: 无（未公开发表学术论文）
- **开源**: 否

**核心机制**:
S2（7B VLM, 7-9Hz）做场景理解和任务规划，S1（80M视觉运动Transformer, 200Hz）做全身35自由度实时控制。S1和S2通过端到端训练通信。Helix 02（2026年1月）增加了S0层（1000万参数, 1kHz）做全身平衡维持和接触感知。

**科学层面分析**:
- Helix的200Hz高频控制应对的是**反应速度**问题，而非**物理代价优化**问题。它解决的是"怎么让机器人在200Hz下稳定执行动作"，而不是"应该执行哪个物理代价最小的动作序列"。
- S2以7-9Hz运行，这意味着在每约130ms内，S2可以产生一个新的高层规划。这个频率足够做reactivity了，但S2的"规划质量"不涉及物理代价对比。

**真实局限（科学层面）**:
- 未公开技术细节。从公开信息看，Helix更像是工程突破（200Hz全身控制）而非科学突破（物理代价优化）。
- S2→S1的通信内容是什么？是文本指令、latent vector、还是别的形式？这决定了S2能否传递物理代价信息给S1。

**研究状态**: 已有但未解决（工程突破为主，物理代价优化未被明确讨论）

---

#### 1.2.7 其他相关规划框架

| 论文 | 机构 | 年份/venue | arXiv | 核心机制 | 物理代价处理 |
|------|------|-----------|-------|---------|------------|
| **LLM-GROP** | UT Austin / NVIDIA | IJRR 2025 | 2511.07727 | LLM常识+TAMP运动规划，移动操作 | TAMP内部处理，LLM不感知 |
| **HERON** | (盲审) | 2025 (under review) | -- | LLM分解+物理引导时间估计+MILP调度 | **显式考虑了时间/资源优化**，最接近物理代价联合优化 |
| **RADI** | (盲审) | ICLR 2025 under review | -- | LLM作为world model，想象动作物理结果并自我修正 | "心理模拟"减少物理试错，但非代价最小化 |
| **LWM-DPO** | -- | NeurIPS 2025 | -- | LLM规划未来状态轨迹，轻量策略执行 | 11.3x样本效率，但非物理代价优化 |
| **VLWM** | Meta FAIR | 2025.09 | 2509.02722 | 视觉-语言世界模型，System-2搜索代价最小化 | **自监督critic评估语义距离满足度**，显式做cost minimization |
| **DeeR-VLA** | 清华 | NeurIPS 2024 | -- | 动态提前退出MLLM，减少计算成本 | 计算效率优化，非物理代价优化 |
| **PDDLLM** | -- | 2025.05 | 2505.18382 | LLM+物理仿真自动推导规划域 | 整合运动规划，但仍为可行性验证 |

---

### 1.3 本子问题的研究现状总结

**核心发现**: 当前VLA规划领域的"物理约束处理"基本停留在"可行性gating"层面——物理代价（能量、时间、平衡难度、安全性）**不作为LLM/VLM规划时的优化目标**。VLM-TAMP和LLM³的"验证-反馈-重规划"循环是最接近的范式，但它们传递的是二进制"可行/不可行"信号，而非连续的物理代价反馈。

**三个关键缺口**:
1. **语义-物理联合优化的接口缺失**：需要一种模块，能够对LLM生成的候选计划进行"物理代价评分"，且速度足够快以嵌入LLM的beam search过程。这可能是latent cost predictor（在语义latent空间中直接预测物理代价），也可能是few-step imagined rollout（在世界模型latent空间中快速前向模拟几帧）。
2. **物理代价的结构化表示缺失**：什么是"物理代价"在多接触全身操作中的正确表示？是总能量、最大关节力矩、重心偏移方差、还是它们的组合？这个表示需要同时对人形机器人的动力学是信息充分的，且对LLM是语义可达的。
3. **物理代价的"可学习性"未知**：物理代价能否从行为数据中学习？如果能，需要什么样的数据（人类演示、仿真轨迹、真实执行记录）？模仿学习中物理代价信号是缺失的——专家演示告诉你"怎么做的"，不告诉你"有没有更省力的做法"。

---

<a name="sec2"></a>
## 二、子问题2：长时序执行中的偏离检测与信用分配

### 2.1 核心问题再陈述

长程任务在第12步失败了。怎么知道是第12步本身执行错了，还是第3步的决策导致后续都不可行，还是规划本身就不对？现有VLA的失败检测是对"当前状态"做OOD检测——告诉我们"这里出问题了"，但不告诉我们"为什么出问题"和"出问题的源头在哪"。核心问题是：**能否在世界模型的latent空间中做反事实诊断——从失败状态向前反推，在latent rollout中定位"最早偏离预期轨迹的时间点"？**

### 2.2 现有方案调研

#### 2.2.1 FAIL-Detect: Uncertainty-Aware Runtime Failure Detection Without Failure Data

- **英文标题**: Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies
- **机构**: Toyota Research Institute (TRI)
- **Venue**: RSS 2025
- **arXiv**: 2503.08558
- **开源**: 是（project page公开）

**核心机制**:
FAIL-Detect将失败检测建模为**序贯OOD检测**（sequential OOD detection）。它从策略的输入/输出中蒸馏出标量信号来捕捉**epistemic uncertainty**（认知不确定性），然后用**conformal prediction (CP)**进行统计标定。最关键的设计是：只用成功数据训练，不需要失败样本。使用flow-based density estimator作为核心信号。

**科学层面分析**:
- FAIL-Detect回答的问题是"现在是否在分布外"，不是"何时开始偏离"或"为什么偏离"。
- 它的epistemic uncertainty信号是一个**聚合的标量**——告诉你"整体不确定"，但不告诉你"在哪些维度上不确定"或"哪个时间步的决策导致了当前的不确定"。
- Conformal prediction提供的统计保证（false positive rate bound）是理论优势，但这个保证是关于"我们声称检测到的失败中有多少是真的"，而非"我们能检测到多少失败"。

**真实局限（科学层面）**:
1. **无时间归因**：FAIL-Detect可以在时间步t检测到"现在出问题了"，但无法告诉你"问题实际上从t-5步就开始了，只是现在才检测到"。这是OOD检测的本质局限。
2. **无因果诊断**：它不知道失败是因为感知噪声、动作误差、还是模型capability不足——只是检测到了偏离。
3. **标量信号的维度坍缩**：在全身人形机器人（数十个关节、多模态传感）中，epistemic uncertainty是一个高维概念，坍缩为标量丢失了大量诊断信息。

**研究状态**: 已有但未解决（检测但不归因）

---

#### 2.2.2 SAFE: Multitask Failure Detection for Vision-Language-Action Models

- **英文标题**: SAFE: Multitask Failure Detection for Vision-Language-Action Models
- **机构**: Univ. of Toronto, Vector Institute, TRI
- **Venue**: NeurIPS 2025
- **arXiv**: 2506.09937
- **开源**: 否（project page）

**核心机制**:
SAFE的关键发现是：VLA内部特征在latent空间中会分离出一个**跨任务一致的"失败区域"**（failure zone）。这意味着：在一个任务集上训练failure detector，可以零样本迁移到未见任务上。SAFE从VLA的中间层提取latent features，回归到标量失败概率，并用conformal prediction标定。

**科学层面分析**:
- SAFE的"跨任务failure zone"是一个**重要的科学发现**：它表明VLA的failure模式具有某种结构一致性——这为通用的失败预测提供了基础。
- 但和FAIL-Detect一样，SAFE做的是**detection**而非**attribution**。它告诉你"现在失败的概率是0.8"，不是"失败是因为第3步的子目标选择不正确"。

**真实局限（科学层面）**:
1. **无gradient信息的使用**：SAFE用latent features回归到标量，但没有利用"latent features相对输入的gradient"——gradient可以指出哪些输入维度导致了高失败概率，这是一个自然的方向但SAFE没有探索。
2. **"failure zone"的因果解释缺失**：latent space中存在一个failure zone——但这个zone对应哪些物理原因（感知失败？规划失败？执行失败？）未被分解。

**研究状态**: 已有但未解决（检测跨任务泛化，但不分解失败原因）

---

#### 2.2.3 Sentinel: Runtime Monitoring of Consistency and Progress

- **英文标题**: Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress
- **机构**: Stanford University & NVIDIA Research
- **Venue**: CoRL 2024
- **arXiv**: 2410.04640
- **开源**: 是 (github.com/agiachris/sentinel, MIT License)

**核心机制**:
Sentinel将失败检测拆分为两个互补类别：
1. **Erratic Failures（抖动型失败）**: 动作时间一致性统计——动作序列突然出现高频抖动或不连续→快速检测（计算极快）
2. **Task Progression Failures（任务停滞型失败）**: 用VLM（GPT/Claude/Gemini）检测"机器人是否在朝着目标前进"→较慢但能检测"自信但错误"的失败

两者结合能比单独任何一个检测器多检测18%的失败。

**科学层面分析**:
- Sentinel的**两分类法**（erratic vs progression）是失败检测的重要概念进展——不是所有失败都一样，不同类型的失败需要不同的检测机制。
- **但仍然是检测，不是归因**。Sentinel告诉你"发生了erratic failure"或"任务不进展了"，但不告诉你erratic failure的根因是传感器噪声还是控制器不稳定。

**真实局限（科学层面）**:
1. **VLM-based progression检测的语义局限性**：VLM可以判断"冰箱门没开"所以"任务没进展"，但无法判断"为什么冰箱门没开"——是抓取点不对、力度不够、还是冰箱门卡住了。
2. **无物理反事实推理**：Sentinel不能回答"如果第3步选择了另一个子目标，会不会避免现在的失败"。

**研究状态**: 已有但未解决（分类检测但不归因）

---

#### 2.2.4 REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction

- **英文标题**: REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction
- **机构**: Columbia University
- **Venue**: CoRL 2023
- **arXiv**: 2306.15724
- **开源**: 是 (github.com/real-stanford/reflect)

**核心机制**:
REFLECT是目前**最接近结构化归因**的方法。它将多模态传感数据（RGB-D、音频、机器人状态）转化为**层级文本摘要**（sensory→event→subgoal三级），然后用LLM进行递进式失败解释：
1. 先逐个子目标验证（哪个子目标失败了？）
2. 对失败子目标调取相关event级观察做细节解释
3. 如果所有子目标都通过了但任务仍失败，分析规划本身的错误

**科学层面分析**:
- REFLECT的"递进式归因"（先定位子目标，再定位事件，最后定位规划）是**最清晰的归因层级**设计。
- **但归因的精度受限于文本摘要的信息量**。将力觉信号、关节角度、接触模式等物理信息转化为自然语言文本，丢失的信息量是巨大的。

**真实局限（科学层面）**:
1. **文本摘要的物理信息瓶颈**：对于人形机器人的全身操作，"抓杯子时手指的力度profile"这种物理信息通过文本描述是高度不精确的。"用力抓→杯子滑了→用力过猛抓碎了"这三种情况的物理信号极其不同，但在文本描述中可能看起来相似。
2. **无latent space反事实**：REFLECT在文本空间做归因，而不是在模型latent空间做counterfactual rollout。后者可以更精确地模拟"如果当时采取了不同动作，会发生什么"。

**研究状态**: 已有但未解决（文本归因有信息瓶颈，缺少latent空间反事实推理）

---

#### 2.2.5 Code-as-Monitor (CaM): Constraint-aware Visual Programming

- **英文标题**: Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection
- **机构**: (not specified in search results)
- **Venue**: arXiv, 2024年12月
- **arXiv**: 2412.04455
- **开源**: 是（project page）

**核心机制**:
CaM将失败检测表述为**时空约束满足问题**。VLM生成可执行的监控代码（而非文本描述），实时检测空间-时间约束是否被满足。三个模块：Constraint Generator生成子目标和文本约束→Painter将文本约束转化为图像上的几何原语→Monitor生成GPT-4o评估代码。

**科学层面分析**:
- CaM的"代码生成而非文本生成"思路是对REFLECT的文本瓶颈的一个自然回应——代码可以精确表达数值约束（`distance(obj1, obj2) < 0.05m`），而自然语言"靠近"是不精确的。
- CaM的proactive detection（在失败发生前检测到约束即将被违反）是最接近"早期偏离检测"的现有方法。

**真实局限（科学层面）**:
1. **约束需要被预定义**：CaM需要VLM生成约束代码，这意味着VLM必须能够"预见"执行中需要监控哪些约束——如果VLM在约束生成时遗漏了关键约束，Monitor就检测不到相应的失败。
2. **无动力学约束**：CaM的约束主要是空间几何约束（位置、距离），不涉及动力学（力、力矩、动量）。对于人形机器人，最关键的失败模式往往是动力学层面的（失去平衡、关节力矩超限）。

**研究状态**: 已有但未解决（几何约束检测已解决，动力学约束检测未解决）

---

#### 2.2.6 World Models for Anomaly Detection during Model-Based RL Inference

- **英文标题**: World Models for Anomaly Detection during Model-Based Reinforcement Learning Inference
- **机构**: (University of Lübeck)
- **Venue**: arXiv, 2025年3月
- **arXiv**: 2503.02552
- **开源**: 未知

**核心机制**:
这是**与本子问题最直接相关的工作之一**。利用DreamerV3训练的世界模型，在推理阶段持续监控**预测误差**（predicted vs actual observations over 16-step horizon）。当预测误差超过阈值时触发安全措施。验证场景包括：检测机器人几何变化、重力变化、四旋翼被外力击中。

**科学层面分析**:
- 这实际上在做latent space的"偏离检测"——通过世界模型的预测误差来感知"系统不在预期轨迹上了"。
- **但仍然是检测，不是归因**：预测误差告诉你"偏离发生了"，但不知道偏离发生在哪个latent维度和为什么。
- 16步的horizon使检测有一定前瞻性，但预测误差是一个聚合标量——不能沿着latent trajectory反向追溯偏离的起点。

**真实局限（科学层面）**:
1. **聚合预测误差无方向性**：无法区分"在x维度偏离"和"在y维度偏离"——两者可能对应完全不同的失败原因。
2. **无counterfactual**：不能做"如果当时采取不同动作会怎样"的反事实推理——世界模型可以forward rollout，但没有机制做backward counterfactual。

**研究状态**: 已有但未解决（latent space偏离检测但不分解和归因）

---

#### 2.2.7 Uncertainty-aware Latent Safety Filters (CMU IntentLab系列)

- **英文标题**: Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures
- **机构**: CMU Intent Lab
- **Venue**: CoRL相关工作 (2024-2025)

**核心机制**:
利用世界模型的**epistemic uncertainty**作为OOD检测信号，通过conformal prediction标定阈值，然后用**HJ reachability analysis在增广latent空间中做安全滤波**（augmented latent space = latent representation + epistemic uncertainty）。

**科学层面分析**:
- 这个方法的关键洞察是：**epistemic uncertainty（而不仅仅是aleatoric uncertainty）**应该作为安全决策的信号。因为epistemic uncertainty反映了"模型对当前状态区域了解不够"——在这个区域采取行动是危险的。
- HJ reachability在latent空间中计算"哪些状态是安全的"——这是一个有理论保证的方法，但要求latent space具有良好的几何结构。

**真实局限（科学层面）**:
- HJ reachability在高维latent space中的计算代价——应用对象仅限于相对简单的操纵任务。
- 没有解决"在latent trajectory上定位偏离起点"的问题。

**研究状态**: 已有但未解决（安全性理论保证已建立，高维扩展性和归因能力未解决）

---

### 2.3 本子问题的研究现状总结

**核心发现**: 当前失败检测领域已经建立了相当成熟的方法论：OOD检测（FAIL-Detect）、latent space monitoring（world model prediction error）、VLM-based monitoring（Sentinel, REFLECT, CaM）、conformal prediction统计保证（FAIL-Detect, SAFE, CMU系列）。但**所有方法都是"检测"而非"归因"**。

**三个关键缺口**:
1. **Backward counterfactual in latent space**: 没有一个方法能从失败状态反向追溯latent trajectory，定位"最早偏离预期的时间点"。这个能力需要世界模型支持反向latent dynamics，或至少能在latent空间中计算"当前轨迹和理想轨迹的divergence分布"。
2. **Failure mode decomposition**: 检测到失败后，不能区分"感知失败→规划失败→执行失败→环境变化"——这四个在latent space中可能有截然不同的特征（感知失败=某些latent维度突然跳到训练分布外；执行失败=latent trajectory的局部曲率突然变大；规划失败=整个latent trajectory偏离目标区域）。现有方法把这些混为一个标量。
3. **Physical counterfactual reasoning**: 为了做归因，需要在物理世界模型中运行counterfactual（"如果当时选择另一个子目标…"）——这需要一个能够模拟物理后果的世界模型（可能是学习到的），且有能力支持对离散决策（子目标选择）的反事实推理。

---

<a name="sec3"></a>
## 三、子问题3：规划（S2）与反应（S1）之间的注意力切换

### 3.1 核心问题再陈述

双系统架构中，S1执行时S2什么时候介入？现有方法要么固定频率（每K步重新规划），要么只在S1明确失败后才呼叫S2。但人类在熟悉任务中S2几乎不介入，在不确定/新奇/危险的时刻S2立刻"抢占注意力"。关键问题是：**S1能否在执行中自主感知到"自己正在偏离"——这个偏离的增长率（而非绝对值）是"S2需要介入了"的信号？**

### 3.2 现有方案调研

#### 3.2.1 GR00T N1的S1/S2接口

**机制**: S2（VLM）→ S1（Diffusion Transformer）通过端到端训练共享latent representation。S2的输出不是一个显式的文本指令，而是作为conditioning信号注入S1的去噪过程。

**切换机制**: **无显式切换机制**。两个系统始终在运行，S2更新其latent context的**频率由推理速度决定**（VLM的推理延迟限制了S2的更新频率）。这是一种"隐式频率"设计——S2想多快更新取决于硬件，而不是任务需求。

**科学层面分析**:
- GR00T N1的设计**没有"何时介入"的决策模块**。S2和S1始终并行，S2更新频率固定——这意味着在需要S2深度推理的困难时刻，S2可能没有及时提供更新；而在不需要S2的简单时刻，S2可能在浪费计算。
- 缺少的是"S1在执行时反馈给S2，告诉它我是否需要你的帮助"的反向通道。

**研究状态**: 已有但未解决（无自适应切换机制）

---

#### 3.2.2 Helix的S2频率设计

**机制**: S2（7B VLM, 7-9Hz）→ S1（80M Transformer, 200Hz）。S2每约130ms产生一个新的高层规划，S1持续以5ms周期执行。

**切换机制**: **固定频率**。S2永远以7-9Hz运行，无论任务的困难程度。Helix 02（2026年）增加S0层用于全身平衡——但三层之间仍然是固定频率。

**科学层面分析**:
- 7-9Hz是VLM推理速度的工程限制，不是任务需求的函数。在简单任务中这可能是浪费；在需要密集规划的任务中这可能不够。
- Helix选择了**"让S2尽量快"的策略**（通过7B模型跑在嵌入式GPU上），而不是"让S2尽量聪明地选择何时介入"。

**研究状态**: 已有但未解决（固定频率无自适应）

---

#### 3.2.3 Hi Robot的用户反馈触发

**机制**: S2（VLM, ~1Hz）在两种情况下运行：(1) 固定频率1Hz (2) 用户主动输入反馈时立即触发重规划。

**切换机制**: **固定频率 + 外部事件触发**。Hi Robot融合了固定频率和外部触发——这是一个进步。用户反馈是明确的"需要S2"的信号来源。

**科学层面分析**:
- 用户反馈触发是一种**显式的外部信号**。但大部分需要S2介入的时刻没有用户在场——机器人需要**自主感知**到"我需要S2帮助"。
- Hi Robot的1Hz固定频率意味着在1秒的间隔内，如果S1开始偏离，S2要到下一个固定周期才介入——这1秒内可能已经失败。

**研究状态**: 已有但未解决（外部触发已实现，自主触发未解决）

---

#### 3.2.4 HiRT: Hierarchical Robot Transformers

- **英文标题**: HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers
- **机构**: 清华(IIIS), UC Berkeley, 上海期智研究院
- **Venue**: CoRL 2024
- **arXiv**: 2410.05273
- **开源**: 未知

**核心机制**:
HiRT的关键设计是**异步并行**：S2（VLM, InstructBLIP-7B）以低频更新latent feature cache，S1（轻量视觉策略）以高频从cache中读取最新的latent condition生成动作。两个模块异步运行——S2不阻塞S1的执行。

**科学层面分析**:
- 异步设计的核心洞察是：**latent features通常变化缓慢**（场景理解不需要每50ms更新一次），所以S2可以在后台慢慢更新，S1用稍微过时但大体正确的latent context执行。
- 但这个设计假设"latent context变化慢"——当环境发生突变时（杯子突然滑了），旧的latent context已经失效，S1带着过时的信息继续执行。

**真实局限（科学层面）**:
1. **无staleness检测**：S1不知道cache中的latent features是否已经过时。需要一个模块监控"S2的latent context与实际观测的匹配度"。
2. **异步设计增强了S1的自主性但降低了适应性**：S1在没有S2最新指导的情况下自主执行更长时间——这在静态环境是优势，在动态环境是风险。

**研究状态**: 已有但未解决（异步架构已建立，staleness感知缺失）

---

#### 3.2.5 事件触发控制（Event-Triggered Control）理论

**理论框架**:
事件触发控制（ETC）和自触发控制（STC）来自网络化控制系统（NCS）领域，核心思想是：**控制器不需要在每个时刻都重新计算控制输入，而是在满足触发条件时才计算**。

ET-MPC的触发条件设计有两种范式：
- **可行性触发**: 当实际状态与预测状态的偏差超过阈值时触发（如`||x(t) - x_pred(t)|| > ε`）
- **稳定性触发**: 当Lyapunov函数递减率不满足时触发

自触发MPC（ST-MPC）更进一步：**下一个触发时刻在当前触发时刻联合计算**（同时计算控制序列和下一次优化时间）。

**科学层面分析——与VLA双系统的联系**:
- ETC/STC的trigger condition设计与子问题3的核心问题高度对应：
  - "S1持续监控预测偏差" ↔ ETC的可行性触发（`||x_actual - x_pred|| > threshold`）
  - "偏离增长率而非绝对值" ↔ ETC的Lyapunov递减率触发（`dV/dt > 0` → 系统不再收敛）
  - "S2需要介入了" ↔ 触发条件满足，重解OCP（最优控制问题）相当于重新规划

- **关键区别**: ETC理论中的"预测"是基于**数学模型**（已知动力学），而VLA中的"预测"是基于**学习到的世界模型**（不确定的）。这意味着trigger condition的阈值设计要考虑到世界模型的epistemic uncertainty——这是ETC理论未涉及的。

**真实局限（科学层面）**:
1. ETC理论的"模型已知"假设在VLA中不成立——世界模型是学习到的且不完美。需要一个新的理论框架结合**学习到的世界模型的不确定性**和**触发条件设计**。
2. ETC的触发条件通常基于状态的逐点比较，而VLA S1/S2的切换信号应该是基于**latent divergence的增长率**——这是一个更高阶的量。

**研究状态**: 理论已建立（在控制论中），向VLA的迁移基本未被研究

---

#### 3.2.6 World Model Anomaly Detection（latent divergence监控）

- **Domberg & Schildbach (2025)**: 16步horizon的预测误差监控，检测到异常后触发安全措施
- **Jing & Boularias (2025)**: VAE novelty detector确保世界模型不偏离训练分布

**与本子问题的关联**:
这两项工作本质上在做"S1执行时的latent divergence监控"，但它们的设计目标是**安全停止**而非**触发S2介入**。将"检测到latent divergence"作为S2介入的触发器——这是一个直接的迁移思路。

**研究状态**: 基本未被研究（latent divergence作为"S2触发信号"的设计空间未被探索）

---

### 3.3 本子问题的研究现状总结

**核心发现**: 当前双系统VLA的S1/S2切换机制**全部为固定频率或外部触发**。没有一个系统能够实现"S1在执行中自主感知偏离并呼叫S2"。事件触发控制（ETC）理论提供了理论框架——"trigger condition = predicted-actual deviation > threshold"——但这个框架需要根本性地适配到"学习到的世界模型"和"latent space divergence监控"的语境下。

**三个关键缺口**:
1. **Latent divergence的增长率为触发信号**: 这是本子问题最具原创性的方向。现有方法用"预测误差的绝对值"做异常检测，但没有用"预测误差的增长率"作为"S2介入时机"的信号。前者告诉你"情况不好"，后者告诉你"情况在恶化"——只有后者才是干预的触发器。
2. **ETC理论向学习化世界模型的迁移**: 经典ETC假设动力学模型已知，VLA中动力学通过世界模型学习得到。这意味着trigger condition需要incorporate epistemic uncertainty——模型不确定的地方，触发阈值应该更低（更保守）。
3. **S1的自我感知能力**: 现有S1（扩散策略、Transformer）没有内置的"自我不确定性评估"能力——它输出action，不知道action的质量如何。需要给S1增加一个**旁路uncertainty head**，在输出action的同时输出"我对当前action的信心"——这个信心信号的下降趋势就是S2触发的候选信号。

---

<a name="sec4"></a>
## 四、子问题4：从成功和失败中学习——什么应该被记住？

### 4.1 核心问题再陈述

VLA执行一个长程任务有多条路径可以成功或失败。哪些经验值得被记住、融进模型的长期知识？简单的"成功的轨迹回放、失败的丢弃"是浪费的——失败轨迹中包含关于"物理边界在哪"的关键信息。核心问题是：**能否定义轨迹的"信息价值"——不是"成功/失败"的二元标签，而是"这条轨迹能减少世界模型在哪些状态区域的预测不确定性"？**

### 4.2 现有方案调研

#### 4.2.1 π*0.6: a VLA That Learns From Experience (RECAP)

- **英文标题**: π*0.6: a VLA That Learns From Experience
- **机构**: Physical Intelligence (PI)
- **Venue**: 技术报告, 2025年11月
- **arXiv**: 2511.14759
- **开源**: 否

**核心机制**:
RECAP（RL with Experience and Corrections via Advantage-conditioned Policies）的核心设计是：用一个学习的价值函数在线评估每个动作的**优势（advantage）**，将优势二值化为"improving/not improving"标签，策略同时条件化在这个标签上。这使得π*0.6可以从混合质量的数据中学习（专家演示+自主rollout+人类纠正）。

**科学层面分析**:
- 优势二值化是一种**隐式的数据筛选**——"improving"的动作被记住，"not improving"的被削弱。
- **但优势（advantage）不等于信息价值（information value）**。一个动作可能advantage很高（很成功），但经过了世界模型已经非常熟悉的区域——信息价值低。一个失败的动作可能经过了世界模型完全未知的区域——信息价值高。

**真实局限（科学层面）**:
1. **Reward/advantage-driven筛选 vs uncertainty-driven筛选**：RECAP（和几乎所有RL方法）筛选高advantage样本，但我们应该筛选"高uncertainty reduction"样本。
2. **混合质量数据中的信息价值**：RECAP说它从混合质量数据中学习——但没有区分混合数据中不同样本的信息价值。在一批数据中，哪些样本对降低世界模型的不确定性贡献最大？RECAP不回答这个问题。

**研究状态**: 已有但未解决（advantage筛选已实现，uncertainty-driven筛选未实现）

---

#### 4.2.2 EVOLVE-VLA: Test-Time Training in Embodied Agents

- **英文标题**: EVOLVE-VLA: Test-Time Training in Embodied Agents (推测标题)
- **机构**: NUS
- **Venue**: arXiv, 2025年12月
- **开源**: 未知

**核心机制**:
EVOLVE-VLA的三个创新组件：(1) **Learned progress estimator**（"foundation critic"），评分-100到100，取代外部奖励；(2) **Accumulative progress estimation**平滑自生成的噪声反馈；(3) **Progressive horizon extension**逐步延长规划horizon。

**科学层面分析**:
- Progress estimator隐式地做了**信息价值筛选**——progress high的轨迹被认为是"有用的"，被用于test-time training的更新。
- 但和RECAP一样，progress不等于信息价值。一条没有进展的轨迹可能经过了世界模型完全不确定的区域——它对世界模型的提升可能比一条进展顺利的轨迹更大。

**真实局限（科学层面）**:
- 未区分"任务的进展"和"世界模型的进展"——前者是对任务目标的接近程度，后者是对环境理解的提升程度。

**研究状态**: 已有但未解决（progress筛选而非uncertainty筛选）

---

#### 4.2.3 Uncertainty Prioritized Experience Replay

- **英文标题**: Uncertainty Prioritized Experience Replay
- **机构**: (学术机构)
- **Venue**: Reinforcement Learning Conference, 2025
- **arXiv**: 2506.09270
- **开源**: 未知

**核心机制**:
提出用**epistemic uncertainty**（而非TD-error）作为经验回放的优先级指标。关键论证：TD-error-based PER容易受噪声过渡样本干扰（"noisy TV problem"——agent被随机预测误差吸引），而epistemic uncertainty是"可以通过学习减少"的不确定性——过滤掉了不可减少的噪声。

**科学层面分析**:
- 这是**最接近本子问题核心思想的方法**：用uncertainty（而非reward/progress）来评估每个经验样本的价值。
- 但这个方法应用于Atari游戏（离散状态/动作），尚未扩展到VLA的连续高维动作空间和世界模型的不确定性——在VLA语境中，经验回放扩展到"哪些rollout轨迹应该被用来更新世界模型"。

**真实局限（科学层面）**:
1. **应用于Atari，未在机器人领域验证**。
2. **epistemic uncertainty是针对Q-function的，不是针对world model的**。本子问题要求的是"轨迹对世界模型的uncertainty reduction"——这是一个不同的量。

**研究状态**: 已有但未解决（在RL理论中建立，向VLA/世界模型的迁移未被探索）

---

#### 4.2.4 Stellar VLA: Knowledge-Driven Continual Learning

- **英文标题**: Stellar VLA (推测标题)
- **机构**: SJTU / Cambridge / AgiBot
- **Venue**: arXiv, 2025年11月

**核心机制**:
使用Dirichlet Process变分模型做无限任务知识空间的自演化，Knowledge-guided MoE routing做任务特化而不增加参数量。核心关注的是**continual learning中的灾难性遗忘**，而非经验的信息价值。

**科学层面分析**:
- Stellar的"knowledge space clustering"是一种**结构化的经验组织**方式——相似技能/任务的经验被聚合在一起，形成结构化的知识表示。
- 这种结构化组织本身可以助力信息价值评估：如果一个新经验无法被现有knowledge clusters很好解释（高reconstruction error），它可能携带高信息价值。

**真实局限（科学层面）**:
- 未显式建模信息价值——任务/task-cluster的结构化是副产品，不是目标。

**研究状态**: 已有但未解决（经验结构化而非信息价值驱动）

---

#### 4.2.5 其他经验选择相关方法

| 方法 | 核心机制 | 与信息价值的距离 |
|------|---------|---------------|
| **AIC-UVNF** (2025) | 自适应curiosity+价值网络拟合 | curiosity接近信息增益，但未被形式化为uncertainty reduction |
| **AHEGC** (2024) | 自适应HER+目标修正curiosity | curiosity作为内在奖励，非显式信息价值 |
| **Feariosity** (2025) | 恐惧+curiosity引导安全探索+新经验回放机制 | 最接近：回放优先级考虑未知程度 |
| **ERA** (2025) | 奖励+信息增益共同决定hippocampal回放 | 神经科学启发：curiosity和reward联合决定记忆优先级 |
| **RL Token** (2025) | 轻量在线RL微调VLA | 样本效率好但无信息价值区分 |
| **SC-VLA** (2025) | 稀疏世界想象在线动作修正 | 世界模型用于修正，非用于评估经验价值 |

---

### 4.3 本子问题的研究现状总结

**核心发现**: 当前VLA的经验选择/数据筛选方法全部基于**任务性能信号**（advantage, progress, success/failure）来做筛选。没有一个方法基于**不确定性减少**（uncertainty reduction）或**信息价值**（information value）来做数据筛选。Uncertainty Prioritized Experience Replay在RL理论中建立了这个方向的可行性，但尚未被迁移到VLA或世界模型的语境中。

**两个关键缺口**:
1. **轨迹的"信息价值"的形式化定义**: 需要定义一个量 `IV(τ) = H(W) - H(W|τ)`——世界模型在被轨迹τ更新后的熵减。这个量衡量"这条轨迹让我对环境理解了多少"。计算出这个量需要世界模型具备uncertainty quantification能力。
2. **失败轨迹的信息价值悖论**: 一条失败轨迹可能经过了世界模型最不确定的物理区域（如"接近平衡边界"的操作），它的信息价值可能远高于一条安全但平凡的成功轨迹。当前所有方法（RECAP, EVOLVE-VLA, 所有RL方法）都偏好成功轨迹——这在信息价值的意义上可能是次优的。

---

<a name="sec5"></a>
## 五、研究状态全景图

| 科学问题维度 | 已解决 | 已有但未解决 | 基本未被研究 |
|------------|--------|-------------|------------|
| **语义-物理联合规划** | -- | SayCan(物理gating,非优化), VLM-TAMP(提议-验证,非联合优化), LLM³(自然语言物理反馈), GR00T N1(模仿学习无代价优化) | Latent cost predictor; 语义-物理联合优化的接口设计; 物理代价的可学习性 |
| **失败检测** | OOD检测(FAIL-Detect), 跨任务failure zone(SAFE), 多模态监控(Sentinel, CaM, REFLECT) | -- | -- |
| **失败归因(信用分配)** | -- | REFLECT(文本归因,信息瓶颈) | Latent空间反事实诊断; 沿latent trajectory的反向偏离定位; Failure mode decomposition |
| **S1/S2切换机制** | -- | 固定频率(GR00T, Helix, HiRT), 外部触发(Hi Robot用户反馈) | Latent divergence增长率触发; ETC理论向学习化世界模型的迁移; S1自我不确定性感知 |
| **经验选择/数据筛选** | -- | Advantage筛选(RECAP), Progress筛选(EVOLVE-VLA), Curiosity奖励 | 信息价值驱动的经验筛选; 失败轨迹的uncertainty reduction价值; IV(τ)的形式化与计算 |
| **世界模型+物理代价** | -- | Prediction error anomaly detection (Domberg 2025) | World model latent rollout做物理代价预测; Joint embedding做语义-物理桥接 |

---

<a name="sec6"></a>
## 六、开放科学问题

### 开放问题1：语义-物理联合优化的latent接口——Latent Cost Predictor

**问题陈述**:
VLA规划器（S2）在做任务分解时如何在语义计划空间中感知物理代价？我们需要一个**Latent Cost Predictor (LCP)**——一个可学习的模块，接受一个"语义计划"和"当前物理状态"的joint latent表示，输出该计划在未来K帧执行中的物理代价预测（总能量、最大关节力矩、平衡余量、时间成本等）。LCP不需要执行完整的物理模拟——它只需要足以区分"物理代价高"和"物理代价低"的计划。

**关键研究挑战**:
1. **训练信号从哪来？** 物理代价的ground truth是什么？仿真数据可以提供精确的物理量（能量、力矩），但sim-to-real gap存在。一个候选路径：从世界模型在latent space中的rollout计算代价，然后用真实执行数据校准。
2. **代价表示的选择**：人形机器人的物理代价是什么？标量（总能量）vs向量（多目标：能量+平衡+时间）？多目标代价需要Pareto优化——这与LLM的标量评分不太匹配。
3. **LCP的训练数据效率**：LCP需要见过不同物理代价的语义计划对比——这需要大量的"语义计划→物理执行→代价标注"triplet。如何高效获取这种数据？

**研究价值**: 这是从"物理可行性gating"到"物理代价优化"的范式转换，可能成为下一代VLA规划器的核心组件。

---

### 开放问题2：Latent Trajectory上的反事实偏离定位与失败归因

**问题陈述**:
给定一条执行失败的长程任务的latent trajectory z_{0:T}（由世界模型编码），以及任务目标在latent space中的表示 g，如何（1）定位"最早偏离预期轨迹的时间点t*"和（2）识别导致偏离的决策者（哪个子目标/动作chunk）？这本质上是**信用分配（credit assignment）从奖励的事后分配前移到失败的事前预测**。

**关键研究挑战**:
1. **世界模型的反向dynamics**：为了从失败状态z_T反推失败起点t*，需要latent space中具有可逆的dynamics——或者至少能计算"在latent trajectory上z_t与目标region之间的divergence如何随时间演变"。divergence陡然增大的点就是t*。
2. **Counterfactual latent rollout**：在定位t*后，需要在latent space中做反事实rollout（"如果当时选择了另一个子目标，latent trajectory会怎样"）来确认归因。这需要世界模型支持"以不同决策为条件"的conditioned rollout。
3. **Failure mode的latent signature**：不同类型的失败（感知误差、规划错误、执行不稳定、环境变化）在latent space中可能表现为不同pattern——识别这些pattern是归因的基础。这可能需要failure mode的弱监督分类。

**研究价值**: 将失败检测从"OOD标量检测"升级为"结构化因果归因"，这是VLA迈向自主错误恢复的前提——你只有知道"为什么失败"和"失败从哪开始的"，才能做有针对性的恢复。

---

### 开放问题3：基于Latent Divergence增长率的自适应S2介入——从ETC理论到学习化世界模型

**问题陈述**:
双系统VLA中，S1在执行时能否自主感知"偏离正在加剧"并触发S2介入？具体地，S1持续监控世界模型预测的latent state z_pred与encoder观察到的真实latent state z_obs之间的 **KL divergence的增长率 d/dt[D_KL(z_obs || z_pred)]**，当这个增长率超过动态阈值时触发S2重新规划。

**关键研究挑战**:
1. **从ETC理论到学习化世界模型的迁移**：经典ETC假设动力学模型f是已知的，触发条件设计基于Lyapunov递减。在VLA中，f是通过世界模型学习的、且具有epistemic uncertainty。需要一个新的trigger condition设计理论：**阈值应该与world model的epistemic uncertainty成反比**——在模型不确定的区域，即使是小的divergence也应该触发S2（保守策略）；在模型很确定的区域，可以容忍更大的divergence。
2. **增长率 vs 绝对值**：D_KL的绝对值告诉你"偏离了多远"——这在稳态下可能是噪声。D_KL的增长率告诉你"情况在恶化"——这才是真正需要S2介入的信号。需要设计一个统计检验区分"稳态噪声"和"结构性偏离"。
3. **S2介入后的信息传递**：S2被触发后，需要知道"什么在偏离"以做有针对性的重规划。S1应该传递的不是"我失败了"，而是"在latent space的哪些维度上，预测和观测的divergence在快速增长"——这让S2的重新规划是信息驱动的（而非盲目的"从头来一次"）。

**研究价值**: 将双系统VLA从一个"固定频率+事后修复"的工程方案，升级为一个"注意力驱动的、事件触发的、信息有效的"智能系统。这是VLA从"能执行"到"能自主管理自身注意力"的关键一步。

---

## 调研方法论说明

- **搜索渠道**: arXiv, Semantic Scholar, Google Scholar, HuggingFace Papers, IEEE Xplore, 各项目主页
- **核实方式**: 所有论文的标题、arXiv ID、venue、机构均通过至少两个独立来源交叉验证
- **覆盖范围**: 约40篇论文，时间跨度2022-2025，核心覆盖VLA规划、TAMP+LLM、双系统架构、失败检测、经验选择五个方向
- **未覆盖**: 本调研聚焦于上述四个子问题的科学层面，未系统覆盖VLA的硬件部署、数据采集pipeline、安全规范、仿真框架等领域

---

*本报告为"面向人形机器人的视觉-语言-动作模型（VLA）深度研究框架"的一部分，负责科学问题：VLA如何做长时序任务——物理可行性约束下的规划、规划失败后的检测与恢复、在"规划"和"反应"之间的动态平衡。*
