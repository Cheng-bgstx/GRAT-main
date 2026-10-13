# 面向人形机器人的世界动作模型（Humanoid World-Action Model）深度调研

> 调研主题：以「世界模型 ⊗ 动作模型」的融合为长期主线，系统梳理面向人形机器人的六大问题分支、每个分支的细分子方向、可落地的研究项目（区分"已有研究但存在未解决问题"与"尚未被研究"），辅以经核实的高质量论文与开源信息，并给出针对性创新点。
>
> 完成日期：2026-06-03 ｜ 面向：具身智能 / 人形机器人方向的科研选题
>
> **重要核实声明**：本报告所有论文均经过网络检索核实标题、arXiv 编号与开源状态。但由于检索环境对 arXiv 正文抓取存在限制，**凡 arXiv 编号前缀为 `2512`、`2601`–`2605` 的条目（对应 2025-12 至 2026-05 的极新预印本）请在正式引用前务必逐篇二次核实**——这批条目最可能存在编号/作者偏差，详见文末「附录 B：待核实清单」。2024 及更早、以及已被会议正式收录的工作可放心引用。

---
## 📍 本文档在六文档体系中的定位

**角色：百科全书式的文献调研——"有什么、谁做的、问题在哪"**

| 维度 | 说明 |
|---|---|
| **本文档做什么** | 全面覆盖六大分支（世界模型/A、全身控制/B、VLA长时序/C、多模态触觉/D、鲁棒恢复/E、数据仿真/F）的文献，每篇论文有完整信息（标题/arXiv/机构/机制/局限），对每个子方向区分"已研究但未解决"与"尚未被研究" |
| **本文档不做什么** | 不提供具体实施方案（→ `Humanoid_DeepDive_Implementation.md`）、不提炼科学问题框架（→ VLA/WAM Framework）、不给出未来研究方向（→ `Humanoid_Embodied_AI_Future_Directions.md`） |
| **与 VLA Framework 的关系** | 本文档是"横向广度"——覆盖全领域论文；VLA Framework 是"纵向深度"——追问 4 个核心科学问题 |
| **与 WAM Framework 的关系** | 同上——WAM Framework 是本文档中"世界模型"相关论文的科学问题深化 |
| **与 Latest Supplement 的关系** | Supplement 是本文档的"时间维度补丁"——覆盖 2025-2026 年本文档成稿后新出的论文 |
| **与 DeepDive Implementation 的关系** | 本文档告诉你"有哪些工作"；DeepDive 告诉你"怎么在它们的基础上做工程实现" |
| **与 Future Directions 的关系** | 本文档告诉你"现在有什么"；Future Directions 告诉你"接下来应该做什么" |
| **阅读顺序建议** | 新读者：本文档 → VLA/WAM Framework（理解科学问题）→ Future Directions（选方向）→ DeepDive（落地） |

---

## 第 0 章　主线、问题地图与如何使用本报告

### 0.1 一句话主线

> **人形机器人之所以难，不在于"会不会动"，而在于"动之前能不能预见后果、动之中能不能纠错、动失败后能不能恢复"。**

把这件事拆开，就是两个模型的融合：

- **世界模型（World Model）**：给定历史观测 + 候选动作，预测未来（像素 / 隐状态 / 3D 状态 / 触觉力觉），即一个**可微、可采样的神经仿真器**；
- **动作模型（Action Model）**：给定观测 + 任务/语言，输出动作的策略。

二者融合后，机器人获得一条闭环能力链——**想象（Imagine）→ 规划（Plan）→ 行动（Act）→ 恢复（Recover）**。这条链就是本报告反复出现的统一框架，也是评判每一个子方向"是否触及核心"的标尺。

```
                ┌─────────────────────────────────────────────┐
                │            Humanoid World-Action Model        │
                │   共享 latent：本体感受+视觉+触觉/力+语言       │
                └─────────────────────────────────────────────┘
                          │                         │
                ┌─────────▼─────────┐     ┌─────────▼─────────┐
                │  World-Model Head │     │   Action Head     │
                │  预测未来 latent / │     │  策略/流匹配/扩散   │
                │  接触力 / 稳定性    │     │  (动作基座)        │
                └─────────┬─────────┘     └─────────┬─────────┘
                          │                         │
   ┌──────────┬──────────┼─────────────┬───────────┼───────────┐
   ▼          ▼          ▼             ▼           ▼           ▼
[想象]     [规划]      [行动]        [失败预测]   [恢复]     [sim2real对齐]
imagine   plan        act          predict-fail  recover    delta-action
 rollout   MPC/搜索    高频控制      OOD/共形预测   护倒+起身   残差动力学
```

### 0.2 六大问题分支（问题地图）

| 分支 | 大方向 | 在主线中的角色 | 本报告章节 |
|---|---|---|---|
| **A** | 世界模型作为主干 / 神经仿真器 | 「想象」的引擎本身——主线核心 | 第 1 章 |
| **B** | 人形全身控制与 loco-manipulation | 「行动」的执行底座 | 第 2 章 |
| **C** | 语言引导 & 长时序生成（VLA / 分层规划） | 「规划」的大脑 | 第 3 章 |
| **D** | 物理交互的多模态感知（触觉/力/多模态） | 让「想象」覆盖看不见的物理量 | 第 4 章 |
| **E** | 物理环境鲁棒恢复与安全性 | 「失败预测 + 恢复」闭环 | 第 5 章 |
| **F** | 数据引擎、仿真平台与评测基准 | 验证全链路的基础设施 | 第 6 章 |

> 遵循你的要求：**数据层面（F.1）刻意压缩**，重心放在"世界模型如何同时充当数据引擎、仿真器与评测器"。

### 0.3 怎么读

每个子方向统一用四段式：**① 已有进展 → ② 未解决痛点 → ③ 可做项目（标注"已有研究但有未解决问题" / "尚未被研究"）→ ④ 创新点 idea（紧扣主线）**。
读完六章后，**第 7 章给出跨分支的旗舰路线图 + 项目优先级矩阵 + 我对你选题的具体建议**。文末两个附录：A. 全报告论文速查表；B. 待核实清单。

---

# 第 1 章　分支 A：世界模型作为主干 / 动作条件世界模型 / 神经仿真器

> 这是整条主线最核心的分支。把世界模型放在主干位置，本质是给机器人一个**可微/可采样的环境模拟器**，服务三类用途：(a) 作为**策略本身**（预测未来帧→反推动作）；(b) 作为**神经仿真器/数据引擎**（生成训练轨迹、做 sim2real）；(c) 作为**规划器内核**（在想象中做 MPC/搜索）。人形机器人因高自由度、富接触、长时序、第一视角（egocentric）观测，恰是该分支最难也最有价值的落地场景。
>
> 一条贯穿性趋势：**世界模型与动作模型正从"两个模块"走向"一个统一的自回归/扩散主干"**（WHAM、WorldVLA、RynnVLA-002 是最直接信号）。

## 1.1 动作条件视频生成世界模型作为策略/仿真器

### ① 已有进展

把"动作"作为条件信号注入视频生成模型，使其成为可交互、可 rollout 的环境模型：

- **Genie: Generative Interactive Environments** — Google DeepMind，ICML 2024（Best Paper），arXiv:2402.15391。11B 参数，从无标注互联网视频中**无监督学出潜在动作空间（latent actions）**，实现可控的 2D 可玩环境。官方未开源权重（社区有复现）。
- **Genie 2: A Large-Scale Foundation World Model** — DeepMind，2024-12（技术博客，无 arXiv）。单图生成可玩 3D 环境，长时一致性达分钟级，面向具身 agent 训练。
- **Genie 3** — DeepMind，2025-08（博客）。720p/24fps 实时交互、多分钟一致性与可提示"世界事件"，目前最强的通用实时世界模型。
- **Oasis** — Decart & Etched，2024，项目页 oasis-model.github.io（**开源**）。首个实时、可玩、AI 生成的开放世界（Minecraft-like），逐帧扩散 + 动作条件。
- **GameNGen: Diffusion Models Are Real-Time Game Engines** — Google，arXiv:2408.14837。单 TPU >20fps 交互式模拟 DOOM，证明扩散模型可作实时神经游戏引擎。
- **IRASim: A Fine-Grained World Model for Robot Manipulation** — 字节跳动，arXiv:2406.14540（**开源**）。轨迹条件的细粒度机器人操作视频生成，逐帧对齐机械臂动作。
- **WorldVLA: Towards Autoregressive Action World Model** — 阿里达摩院，arXiv:2506.21539（HuggingFace 开源 Alibaba-DAMO-Academy/WorldVLA）。在单一自回归模型中**统一 VLA（动作模型）与世界模型**，动作生成与图像生成互相增强——本分支与主线最直接的"融合"范本。

### ② 尚未解决的痛点

1. **动作可控性精度不足**：游戏类世界模型动作空间离散且小（WASD/视角），而人形是数十维连续关节/末端位姿，精确注入并保持响应一致仍是开放问题。
2. **接触/受力不可见**：视频只编码视觉表观，缺力、力矩、摩擦等本体感受，导致富接触操作"看起来对、物理错"。
3. **闭环漂移（compounding error）**：自回归 rollout 超过几秒误差累积，物体凭空出现/消失。
4. **推理延迟**：扩散式世界模型难满足人形 ~50–200Hz 控制频率。

### ③ 可做项目

- **（已有研究但有未解决问题）egocentric 人形第一视角的"动作 + 本体感受联合条件"世界模型**：在 1X / GR00T 类数据上，把关节角、末端力/扭矩、IMU 与 RGB 一起作为条件与预测目标。WorldVLA/IRASim 已做视觉，但**多模态本体感受联合预测基本空白**。
- **（尚未充分研究）实时蒸馏的人形世界模型**：把扩散世界模型蒸馏成 few-step / consistency model，达到控制环可用延迟。

### ④ 创新点 idea

> **「Proprioceptive World Model (PWM)」——视觉+力觉双流动作条件世界模型**。在视频生成主干上并行一条"力觉/触觉 token 流"，预测下一帧 RGB 的同时预测末端 6 轴力/触觉阵列读数；训练时用对比损失对齐"视觉接触事件"与"力觉峰值"。价值：人形在富接触操作（开门、拧瓶盖）中可在想象中预判"碰到了/打滑了"，填补纯视频世界模型的物理盲区，且力觉数据在遥操作中天然可采。

## 1.2 隐空间世界模型（Dreamer 系）用于人形 RL 与规划

### ① 已有进展

隐空间动力学（latent dynamics）在紧凑潜空间预测未来，计算高效，适合在线 RL 与高频控制：

- **DreamerV3: Mastering Diverse Domains through World Models** — Hafner 等（DeepMind/多伦多），arXiv:2301.04104，**Nature 2025 正刊**。单一超参跨 150+ 任务（含机器人 loco/manipulation、从零挖钻），开源 github.com/danijar/dreamerv3。通过 symlog/two-hot/free-bits 实现"免调参"通用 MBRL。
- **DayDreamer: World Models for Physical Robot Learning** — Wu, Escontrela, Hafner 等，CoRL 2022（经典背景）。在真实四足/机械臂上**直接现实世界训练**，1 小时学会四足行走。
- **TD-MPC2: Scalable, Robust World Models for Continuous Control** — Hansen, Su, Wang（UCSD），ICLR 2024，arXiv:2310.16828（**开源**）。decoder-free 隐式世界模型 + 隐空间局部轨迹优化（MPC），单一 317M 模型跨 80 任务（含 HumanoidBench 高维人形）。
- 衍生人形工作：**TD Learning with Group Relative Policy Constraint for Humanoid Locomotion**（arXiv:2505.13549）、**DoublyAware: Dual Planning and Policy Awareness**（arXiv:2506.12095），把 TD-MPC 与不确定性/策略约束结合用于人形步态。

### ② 尚未解决的痛点

1. **隐空间 WM"视觉抽象太狠"**：丢失精细几何与外观，难支撑需细粒度视觉的操作（与 1.1 的像素级 WM 互补但未融合）。
2. **高维人形探索与长时信用分配**：几十自由度 + 稀疏奖励下，想象 rollout 误差让 actor-critic 不稳定。
3. **富接触动力学建模差**：RSSM 把接触的离散切换（脚触地、手抓握）平滑化，导致接触瞬间预测失真。
4. **缺乏跨形态迁移**：几乎每个机器人从零训。

### ③ 可做项目

- **（已有研究但有未解决问题）分层世界模型：像素 WM 做"想象数据" + 隐空间 WM 做"高频控制"**。用 1.1 的视频 WM 生成长时想象轨迹，蒸馏进 Dreamer 类隐空间 WM 做策略优化。
- **（尚未充分研究）接触感知的隐空间动力学**：在 RSSM 中显式加入"接触模式"离散隐变量（混合连续-离散潜状态）。

### ④ 创新点 idea

> **「Contact-Mode RSSM for Humanoids」**。潜状态拆成连续姿态隐变量 + 离散接触图（哪些末端/足底接触），用 Gumbel-Softmax 学接触切换，切换处插入物理一致性损失（动量守恒近似）。把传统接触动力学（hybrid system）的归纳偏置注入数据驱动 WM，解决"脚触地瞬间预测糊掉"，天然适配人形"行走+操作"的离散接触本质。在 HumanoidBench 上验证。

## 1.3 视频生成基座作为神经仿真器 / 数据引擎

> 当前**最受工业界重视**的方向：用大规模视频生成模型生成机器人训练数据，绕开真机采集瓶颈。

### ① 已有进展

- **NVIDIA Cosmos: World Foundation Model Platform for Physical AI** — NVIDIA，arXiv:2501.03575（**开源** github.com/NVIDIA/Cosmos，含 Cosmos-Predict / Cosmos-Transfer）。预训练+后训练范式的通用世界基座。后续：**Cosmos-Transfer1**（arXiv:2503.14492）、**World Simulation with Video Foundation Models for Physical AI**（arXiv:2511.00062，Cosmos-Predict2.5 / Transfer2.5）。
- **DreamGen: Unlocking Generalization in Robot Learning through Video World Models** — NVIDIA GEAR Lab（Joel Jang 等），arXiv:2505.12705，CoRL 2025。工程化为 **GR00T-Dreams**（**开源** github.com/NVIDIA/GR00T-Dreams）。四阶段管线：后训练视频 WM → 从单图生成"神经轨迹" → 抽取潜在/伪动作 → 训练策略。**首次实现零样本行为泛化 + 零样本环境泛化**——人形仅靠 pick-and-place 遥操数据学会 22 个新行为于未见环境。
- **Humanoid World Models (HWM): Open World Foundation Models for Humanoid Robotics** — arXiv:2506.01182（**开源**）。轻量开源，预测人形控制 token 条件下的第一视角未来视频，提供 MaskGIT 与 diffusion 两类。
- **1X World Model** — 1X Technologies，2024（博客+技术报告）。在 EVE/Neo 数千小时真实数据上训练，能预测刚体、可变形物（窗帘/衣物）、铰接物（门/抽屉）交互，作为评测策略的神经仿真器。配套 **1X World Model Challenge**（技术报告 arXiv:2510.07092，**开源基准**）。
- **UniSim: Learning Interactive Real-World Simulators** — UC Berkeley/Google/MIT，**ICLR 2024 Outstanding Paper**，arXiv:2310.06114。统一动作接口的真实世界交互模拟器。
- **GR-1 / GR-2** — 字节跳动。GR-1: *Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation*，arXiv:2312.13139，ICLR 2024；GR-2: arXiv:2410.06158。先在大规模视频上做视频生成预训练，再微调到"预测未来帧 + 预测动作"。

### ② 尚未解决的痛点

1. **生成数据的"伪动作"质量**：用逆动力学/潜在动作从生成视频反推动作，反推误差直接污染策略训练，缺乏置信度估计。
2. **物理不可执行的幻觉轨迹**：生成视频可能违反运动学约束（关节超限、自碰撞、穿模），却被当训练数据。
3. **分布覆盖 vs 真实**：生成器倾向"平均/安全"轨迹，稀有但关键的恢复/失败场景生成不足。
4. **评估闭环缺失**："生成数据质量"与"策略提升"之间缺乏因果度量。

### ③ 可做项目

- **（已有研究但有未解决问题）带运动学/自碰撞过滤器的神经轨迹引擎**：在 DreamGen/GR00T-Dreams 管线后接"可执行性裁判"——用机器人 URDF 做 IK + 自碰撞 + 关节限位检查，丢弃/重投影物理不可行轨迹，并给每条神经轨迹打可执行性置信分加权到策略损失。
- **（尚未充分研究）主动数据生成（active dream）**：让世界模型针对当前策略失败模式**定向生成**最有信息量的想象轨迹。

### ④ 创新点 idea

> **「Executable Neural Trajectories with Confidence-Weighted Distillation」**。三件套：(1) 生成第一视角视频后，用轻量逆动力学模型预测动作**及其方差**；(2) 把动作经运动学层投影到可行流形（关节限位/自碰撞约束），记录投影残差；(3) 策略蒸馏时用 (1)(2) 的不确定性反向加权——低置信、高残差的神经轨迹降权。把"生成器幻觉"从隐患转为可量化训练信号。对人形尤其关键，因其自碰撞与平衡约束远比机械臂严苛。

## 1.4 世界模型用于 model-based planning / MPC / 想象中的规划

### ① 已有进展

- **TD-MPC2**（见 1.2）：隐空间 MPC 的代表，在高维人形上做规划-控制。
- **DreamerV3**（见 1.2）：本质是在想象 rollout 中做策略优化（imagined RL），"想象中规划"的范式定义者。
- **iVideoGPT: Interactive VideoGPTs are Scalable World Models** — 清华，NeurIPS 2024，arXiv:2405.15223（**开源**）。自回归 Transformer 把视觉/动作/奖励统一成 token 序列，支持交互式 rollout 与 model-based 控制。
- **Navigation World Models (NWM)** — Meta FAIR（Bar, LeCun 等），CVPR 2025，arXiv:2412.03572（HuggingFace facebook/nwm）。可控视频生成 WM，在导航中**通过想象未来观测来规划轨迹**、评估目标可达性，跨机器人泛化。
- **MuZero**（Schrittwieser 等，Nature 2020，背景）：隐空间做 MCTS 规划，是 model-based planning 的另一支柱。

### ② 尚未解决的痛点

1. **规划视野 vs 误差累积的根本张力**：WM 越往远处预测越不可信，但人形长程任务需要长视野。
2. **想象规划的实时性**：像素级 WM 上做 MPC，每步采样多条未来视频，算力爆炸。
3. **不确定性感知规划缺失**：多数 WM 给点估计，规划器无法区分"模型自信"与"模型瞎猜"，在幻觉区域过度乐观。
4. **目标/奖励指定难**：像素空间里如何定义"任务完成"。

### ③ 可做项目

- **（已有研究但有未解决问题）分层想象规划**：高层用慢速像素 WM 想象"子目标关键帧"，低层用快速隐空间 WM（TD-MPC2）在子目标间做高频 MPC。缺人形端到端系统。
- **（尚未充分研究）不确定性门控的想象长度**：让 WM 输出预测不确定性，规划器据此**自适应截断想象视野**——自信就想得远，不确定就缩短并切回真机观测。

### ④ 创新点 idea

> **「Uncertainty-Gated Hierarchical Dreaming for Humanoids」**。(1) 训练带集成/dropout 不确定性头的人形世界模型；(2) 高层在像素/隐空间生成候选子目标，用不确定性做 risk-aware 评分；(3) 低层 TD-MPC2 风格隐空间 MPC 执行，滚动不确定性超阈值则触发"重新观测/重规划"。再配一个"想象-行动-恢复"闭环：当真机观测与 WM 预测显著背离（预测残差大），判定"计划失败"，自动回退安全姿态并重规划——**恢复行为不再硬编码，而是 WM 预测失配触发**。

## 1.5 世界模型评测：长时一致性、3D 一致性、物理合理性

### ① 已有进展

- **WorldModelBench: Judging Video Generation Models As World Models** — arXiv:2502.20694，NeurIPS 2025。加入指令遵循 + 物理遵循维度，覆盖 7 域。
- **Physics-IQ** — Google DeepMind 等，2025，arXiv:2501.09038（项目页 physics-iq.github.io，**开源**）。用真实物理实验视频考察流体/光学/固体，结论：当前视频模型**有视觉真实感但缺物理理解**。
- **WorldSimBench** — 2024，arXiv:2410.18072，ICML 2025。双轨评测：显式感知评测（人类偏好）+ **隐式操控评测**（生成视频能否转成正确动作），是少数把"世界模型"与"动作可执行性"绑定的工作。
- **VBench** — 南洋理工 S-Lab，CVPR 2024，arXiv:2311.17982（**开源**）。视频生成质量多维度分解；扩展 VBench++/VBench-2.0。
- **A Comprehensive Evaluation of Physical Realism in Text-to-Video Models** — arXiv:2507.13428。1050 条 prompt 覆盖基础/复合/反物理场景，系统揭示 SOTA 物理违背。

### ② 尚未解决的痛点

1. **缺乏"动作-效果"因果评测**：多评单条视频质量，少评"给定动作 a，WM 预测效果是否与真实物理因果一致"。
2. **3D/几何一致性度量薄弱**：逐帧好看但 3D 结构漂移（物体尺寸/位置跨帧不守恒）。
3. **长时序记忆评测缺失**：离开视野的物体再回来是否还在。
4. **人形/具身专属基准空白**：缺针对"人形操作/接触/平衡"的诊断集。

### ③ 可做项目

- **（尚未被充分研究）人形操作的反事实因果评测集**：构造"同一初始状态 + 不同动作"配对真机视频，测 WM 能否正确区分不同动作导致的不同结果（反事实一致性），而非记住表观。
- **（已有研究但有未解决问题）基于 3D 重建的几何一致性自动指标**：对生成视频跑单目重建/跟踪，度量物体尺度/位置的跨帧方差作为"3D 漂移分"。

### ④ 创新点 idea

> **「HumanoidWM-Bench：接触-平衡-因果三轴诊断基准」**。三个针对人形的诊断轴：(1) **接触因果**——给定"手是否接触物体"的反事实动作对，看 WM 预测的物体位移是否符合接触逻辑；(2) **平衡可信度**——评 WM 生成的人形姿态是否满足 ZMP/质心在支撑域内（用 off-the-shelf 姿态估计 + 简化力学检查）；(3) **长时物体永久性**——遮挡再现测试。配套自动评分管线（姿态估计 + 3D 跟踪 + 力学约束检查）。**首个把"平衡/接触"两个人形独有物理约束量化进世界模型评测的基准**。

## 1.6 交互式 / 可玩世界模型

### ① 已有进展

- **WHAM (Muse): World and Human Action Models towards gameplay ideation** — Microsoft Research + Ninja Theory，**Nature 2025 正刊**（s41586-025-08600-3）。开源权重 huggingface.co/microsoft/wham。自回归模型联合预测（tokenized）游戏画面与控制器动作，可只生成画面、只生成动作或两者皆生成。后续 **WHAMM!** 实现实时。**"世界+人类动作"联合建模——正是主线"世界⊗动作融合"的直接范本。**
- **Pandora: Towards General World Model with Natural Language Actions and Video States** — arXiv:2406.09455。自然语言为动作、视频为状态的通用交互世界模型。
- **DIAMOND**（NeurIPS 2024，背景）：扩散世界模型在 Atari 上训练 RL agent。
- 新近统一 VLA+WM：**RynnVLA-002: A Unified Vision-Language-Action and World Model**（arXiv:2511.17502）。

### ② 尚未解决的痛点

1. **"可玩"≠"可控到机器人精度"**：游戏交互低维离散，人形需连续高维、对延迟敏感。
2. **多智能体/人-机共存交互**：现有可玩 WM 多为单 agent，人形要预测他人行为。
3. **可控性与一致性权衡**：越强控制越易打破长时一致性。
4. **从"可玩 demo"到"可信训练环境"的鸿沟**：好玩 ≠ 物理可信到能 sim2real。

### ③ 可做项目

- **（尚未被充分研究）人在环可交互人形世界模型**：WM 同时建模"机器人动作"与"场景中人的动作"两套 action，支持遥操作者实时介入（human-in-the-loop dreaming），用于人形 HRI 场景的想象排练。
- **（已有研究但有未解决问题）连续动作的实时可控世界模型**：借鉴 WHAMM!/Genie 3 的实时化，把动作接口从离散键位换成连续末端位姿 + 关节流。

### ④ 创新点 idea

> **「Co-Player Humanoid World Model：双 agent 可交互神经孪生」**。训练一个同时接受"机器人控制流"与"人类动作/语言介入"两路 action 条件的实时世界模型。应用闭环：遥操作者可在 WM 里"先排练"人机协作（如递物），WM 预测人形与人的联合未来，确认安全后再下发真机；真机执行时若人的真实行为偏离 WM 预测，触发"想象-恢复"。把可玩世界模型从"单 agent 游戏"推向"人机共存的协作排练器"。

## 1.7 分支 A 小结

跨子方向**三个共性未解难题**，也是主线"想象-规划-行动-恢复"最该攻的点：

1. **物理可信度**（尤其接触、平衡、力觉）——纯视频 WM 的系统性盲区（贯穿 1.1/1.3/1.5）；
2. **不确定性感知**——规划与恢复行为的触发依据，目前几乎缺位（1.4）；
3. **人形/具身专属评测与约束注入**——缺基准就无法判断"WM 能否安全训练人形"（1.5）。

四个创新 idea（PWM → Contact-Mode RSSM → Executable Neural Trajectories → Uncertainty-Gated Hierarchical Dreaming）串成一条可落地链路：**力觉增强的物理可信 WM → 接触感知的高频控制 → 可执行的想象数据引擎 → 不确定性驱动的想象-规划-恢复闭环**。

---

# 第 2 章　分支 B：人形全身控制与 Loco-Manipulation

> 「行动」的执行底座。2024–2026 经历了从"单技能 RL 控制器"到"统一运动追踪（motion tracking）基座模型"的范式迁移。清晰主线：**用大规模人类动作数据（AMASS 等）训练一个通用 motion-tracking 策略作为"动作基座"，再用蒸馏/残差学习/世界模型适配下游任务**。motion-tracking 策略本质上是"以参考动作为条件的逆动力学动作模型"，而 sim-to-real 对齐、力自适应、物体交互恰是"动力学想象"缺失的环节——这正是世界模型的切入口。

## 2.1 基于 RL 的全身控制（WBC）与 Locomotion

### ① 已有进展

- **Real-World Humanoid Locomotion with Reinforcement Learning** — UC Berkeley，2023，arXiv:2303.03381。因果 Transformer 把历史本体感知作为上下文，开创"in-context adaptation"式人形行走。
- **Learning Humanoid Locomotion over Challenging Terrain** — UC Berkeley，2024，arXiv:2410.03654。Transformer 做"预测下一动作"的盲走，复杂地形鲁棒。
- **Berkeley Humanoid: A Research Platform for Learning-based Control** — UC Berkeley（Sreenath/Abbeel 组），2024，arXiv:2407.21781，项目页 berkeley-humanoid.com。中等规模低成本平台 + 轻量域随机化 RL，学术界自研硬件标杆。**Berkeley Humanoid Lite**（arXiv:2504.17249）进一步开源、可 3D 打印。
- **Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies** — 2024，arXiv:2410.11825。用 Lipschitz 约束替代繁琐 reward 调参，得平滑可部署动作。

### ② 尚未解决的痛点

- **Reward 工程与可迁移性差**：每个新技能/新机器人需重调 reward 与域随机化，缺跨形态零样本迁移。
- **盲走 vs 感知**：多数鲁棒 locomotion 仍是 blind/proprioceptive，外感受与动力学预测耦合进闭环不成熟。
- **长时程稳定性**：对累积扰动、地面属性突变（摩擦/软硬）缺乏在线动力学辨识与"预想"能力。

### ③ 可做项目

1. **（已有研究但有未解决问题）在线世界模型驱动的地形-动力学自适应行走**：现有 WMR（arXiv:2502.16230）仅做隐式地形重建，未把"未来 N 步动力学 rollout"用于实时步态调整。可做轻量隐式世界模型，onboard 实时预测落脚点接触力并修正动作。
2. **（尚未充分研究）跨形态统一 locomotion 基座**：用世界模型把不同机器人（H1/G1/Berkeley）的动力学差异编码为 latent context，实现一策略多机零样本迁移。

### ④ 创新点 idea

> **「Dreamer-WBC」**：把 locomotion 策略训练为 latent world model 上的规划器——本体感知编码进 latent state，世界模型预测接触/质心轨迹，策略在"想象"中做短时 MPC-style rollout 再输出动作。落地路径：在 ASAP 的 delta-action 思路上，用学到的残差动力学模型替代手工域随机化，实现"想象式 sim-to-real"。

## 2.2 从人类动作模仿 / 重定向学习（本分支最密集的方向）

### ① 已有进展

- **H2O: Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation** — CMU（LeCAR Lab），2024，arXiv:2403.04436。仅用 RGB 相机实现实时全身遥操作，RL sim-to-real，奠定"kinematic pose 作为通用控制接口"范式。**OmniH2O**（arXiv:2406.08858）加入灵巧手，支持 VR/视频/语言多模态控制及自主策略学习。
- **HumanPlus: Humanoid Shadowing and Imitation from Humans** — Stanford（Fu, Cheng, Finn 等），CoRL 2024，arXiv:2406.10454，**开源** github.com/MarkFzp/humanplus。40 小时人类动作训低层 shadowing 策略 + egocentric 视觉自主模仿。
- **ExBody / ExBody2: Expressive Whole-Body Control** — UCSD（Xiaolong Wang 组）等，arXiv:2402.16796 / 2412.13196，项目页 exbody2.github.io。上身严格模仿、下身放松约束，得自然表现性动作；ExBody2 做任意参考动作的通用追踪，解耦速度与姿态。
- **TWIST: Teleoperated Whole-Body Imitation System** — Stanford（Yanjie Ze, Xue Bin Peng 等），2025，arXiv:2505.02833，**开源** github.com/YanjieZe/TWIST。单一神经网络控制器实现协调全身遥操作。**TWIST2**（arXiv:2511.02832，**开源**）做免 mocap 便携遥操作 + 数据采集。
- **VideoMimic: Visual Imitation Enables Contextual Humanoid Control** — UC Berkeley，2025，arXiv:2505.03729，**开源** github.com/hongsukchoi/VideoMimic。real-to-sim-to-real 管线，把单目视频转为情境感知技能（上下楼梯、坐下）。
- **GMR: General Motion Retargeting for Humanoid Motion Tracking** — 2025，arXiv:2510.02252。通用重定向工具，为 motion tracking 提供高质量参考。

### ② 尚未解决的痛点

- **重定向的物理可行性 / 接触保持**：纯运动学重定向忽略接触关系，导致脚滑、穿模、手-物接触丢失。
- **形态差异（embodiment gap）**：人类自由度 ≠ 机器人，极限动作重定向后不可行。
- **遥操作延迟与因果性**：从感知到动作的链路延迟限制高动态任务响应。

### ③ 可做项目

1. **（已有研究但有未解决问题）接触感知重定向**：OmniRetarget（见 2.3）用 interaction mesh 保持接触，但主要面向离线数据生成；可做**在线实时接触保持重定向**用于遥操作闭环。
2. **（尚未充分研究）用世界模型做"动作可行性预筛"**：重定向输出送入控制器前，用学到的动力学模型 rollout 判断该参考是否物理可达，不可达则投影到可行流形。

### ④ 创新点 idea

> **「Imaginative Retargeting」**：把重定向建模为世界模型条件下的优化——给定人类 keypoint，世界模型预测若干候选机器人轨迹的未来稳定性（预测质心/接触力），选"想象中最稳"的重定向结果。把动作模型（motion tracker）与世界模型（dynamics predictor）在重定向阶段融合，解决物理可行性痛点。

## 2.3 移动操作 / 全身操作（Loco-Manipulation）

### ① 已有进展

- **FALCON: Learning Force-Adaptive Humanoid Loco-Manipulation** — 2025，arXiv:2505.06776。双 agent 解耦：下身抗外力维稳 + 上身精确末端追踪，跨平台泛化。
- **HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots** — NVIDIA，2024，arXiv:2410.21229，项目页 hover-versatile-humanoid.github.io。统一神经控制器，在导航/loco-manip/桌面操作多控制模式间无缝切换。
- **OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation** — 2025，arXiv:2509.26633，项目页 omniretarget.github.io。基于 interaction mesh 的接触保持数据生成引擎，仅 5 个 reward 即支持 30 秒长时程 parkour 与 loco-manip。
- **ResMimic: From General Motion Tracking to Loco-Manipulation via Residual Learning** — 2025，arXiv:2510.05070，项目页 resmimic.github.io。两阶段：GMT 预训 + 残差后训，给通用追踪补上"物体感知与精度"。
- **VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation** — 2025，arXiv:2509.20322，**开源** github.com/visualmimic/VisualMimic。低层 keypoint tracker + 高层 keypoint 生成器，视觉零样本 sim-to-real。
- **HumanoidVerse: Unified Policy for Long-Horizon Whole-Body Loco-Manipulation** — 2025，arXiv:2508.16943。视觉-语言引导的多物体重排，双教师蒸馏，子任务间无需 reset。

### ② 尚未解决的痛点

- **物体动力学未知**：质量、惯量、摩擦在线未知，纯前馈控制崩溃；多数工作仍假设物体属性已知或轻载。
- **基座-手臂动力学耦合**：行走引入的动量扰动破坏操作精度，FALCON 等用解耦回避而非真正建模耦合。
- **长时程任务组合**：子技能切换处易失稳，缺乏统一的"任务级想象-规划"。

### ③ 可做项目

1. **（已有研究但有未解决问题）在线物体参数辨识 + 控制**：腿式人形负载下的在线惯量辨识 + 敏捷动作仍开放（参考用可微仿真闭合负载 sim-to-real gap 的思路）。
2. **（尚未充分研究）loco-manipulation 的"操作-移动联合世界模型"**：把物体状态 + 机器人状态 + 接触统一进一个 latent 世界模型，做整体规划而非上下身解耦。

### ④ 创新点 idea

> **「World-Model Loco-Manipulation (WM-LM)」**：训练一个把"机器人本体 + 被操作物体 + 接触关系"统一编码的 latent 世界模型，在其中做 model-based 规划——机器人先"想象"搬起未知箱子后的质心偏移与所需步态，再执行。借鉴 HAIC（见 2.6）的 dynamics-aware world model 思想扩展到移动场景。

## 2.4 敏捷与高动态运动（Agile / Dynamic Skills）

### ① 已有进展

- **ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills** — CMU + NVIDIA（Tairan He 等），RSS 2025，arXiv:2502.01143，**开源** github.com/LeCAR-Lab/ASAP。**学一个 delta-action 模型补偿 sim-real 动力学差异，冻结后注入仿真器再微调追踪策略**，实现 C 罗式踢腿/跳跃等敏捷动作。**"用学到的残差动力学对齐物理"的代表作，与世界模型主线最契合。**
- **Humanoid Parkour Learning** — CMU/上交等，2024，arXiv:2406.10759。端到端视觉 WBC parkour 策略，无需 motion prior。
- **WoCoCo: Learning Whole-Body Humanoid Control with Sequential Contacts** — 2024，arXiv:2406.06005。以接触序列为约束，实现 parkour 跳跃、箱子 loco-manip、动态攀爬，无 motion prior。

### ② 尚未解决的痛点

- **sim-to-real 动力学差异在高动态下被放大**：接触瞬间、高速摆动时，执行器延迟/扭矩饱和/接触模型误差被放大，ASAP 缓解但未根除。
- **缺乏失败恢复的"想象"**：敏捷动作失败即摔倒，策略不具备在动作中途预测失稳并切换到恢复的能力。
- **数据稀缺**：极限动作的高质量参考动作少，重定向后多不可行。

### ③ 可做项目

1. **（已有研究但有未解决问题）可微/学习动力学驱动的敏捷 sim-to-real**：ASAP 的 delta action 是无记忆残差；可升级为**带记忆的循环世界模型**，显式建模接触瞬态与执行器动态。
2. **（尚未充分研究）敏捷动作中的实时失稳预测与中途恢复**：训练世界模型预测"未来 0.5s 内是否摔倒"，触发恢复子策略，把"行动-恢复"闭环嵌入高动态技能。

### ④ 创新点 idea

> **「Predict-then-Recover Agile Controller」**：在 ASAP/parkour 策略上挂载轻量世界模型，实时 rollout 预测落地稳定性；若预测失稳概率超阈值，无缝切到 HumanUP/HoST 式恢复策略。**这是主线"想象-行动-恢复"在高动态场景的具体实例，且每个组件都有现成开源基座可复用**（ASAP、HumanUP、HoST 均开源/有项目页）。

## 2.5 统一 / 多技能控制器与 Sim-to-Real

### ① 已有进展

- **GMT: General Motion Tracking for Humanoid Whole-Body Control** — 2025，arXiv:2506.14770。单一统一策略追踪多样真实动作，是"动作基座"的代表。
- **BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion** — UC Berkeley（Qiayuan Liao 等），2025，arXiv:2508.08241，项目页 beyondmimic.github.io，**开源** github.com/HybridRobotics/whole_body_tracking。**用 guided diffusion 在测试时合成新行为**（航点导航、摇杆遥操作、避障），把 motion tracking 与灵活生成桥接。"动作模型即生成模型"的关键作。
- **R2S2: Unleashing Humanoid Reaching Potential via Real-world-Ready Skill Space** — 2025，arXiv:2505.10918。构建"真实可用技能空间"，统一大范围伸展可达性。
- **Supersizing Motion Tracking for Natural Humanoid Whole-Body Control** — 2025，arXiv:2511.07820。**scaling law 视角**：扩大模型容量/数据/算力获得更自然的统一控制器，呼应"控制基座模型"趋势。
- **MuJoCo Playground** — DeepMind/Berkeley 等，2025，arXiv:2502.08844（**开源**）。GPU 仿真，支持人形零样本 sim-to-real，是统一控制器训练基础设施。

### ② 尚未解决的痛点

- **统一 vs 专精的张力**：通用 tracker 精度不足（故需 ResMimic 残差）；专精策略不可迁移。
- **生成式控制器的物理可行性**：diffusion 生成的动作未必动力学可行，需 guidance 但仍可能违反约束。
- **scaling 的边际**：Supersizing 显示 scaling 有效，但数据多样性与硬件多样性仍是瓶颈。

### ③ 可做项目

1. **（已有研究但有未解决问题）物理约束的生成式控制基座**：BeyondMimic 用 guided diffusion，但 guidance 是软约束；可用世界模型做硬可行性投影。
2. **（尚未充分研究）多机器人统一控制基座模型**：把 H1/G1/Berkeley 等异构形态统一进一个 conditioned policy，世界模型编码形态-动力学差异。

### ④ 创新点 idea

> **「World-Model-Guided Diffusion Controller」**：在 BeyondMimic 的 guided-diffusion 框架里，把 guidance 项换成**学到的世界模型给出的可行性/稳定性梯度**——生成动作时不仅追踪参考，还被世界模型"拉回"物理可行流形。把"动作生成模型 + 世界动力学模型"在统一控制器里深度融合。

## 2.6 世界模型 / 可微仿真在控制中的应用（主线核心交叉点）

### ① 已有进展

- **WMR: Learning Humanoid Locomotion with World Model Reconstruction** — 2025，arXiv:2502.16230。训练估计器显式重建世界状态，实现盲走跨复杂地形。
- **Mastering Challenging Terrains with Denoising World Model Learning (DWL)** — 上海 AI Lab 等，2024，arXiv:2408.14472。去噪世界模型学习，人形跨复杂地形，被广泛引为 world-model-based 人形控制代表。
- **ASAP**（arXiv:2502.01143）：delta-action 残差动力学模型本质是"学到的世界模型修正项"，model-based 对齐的经典案例。
- **Current Progress and Challenges in ... Control, Planning, and Learning（综述）** — 2025，arXiv:2501.02116。系统梳理 model-based 规划与学习方法，适合作背景引用。

> ⚠️ 注：以下两条为极新预印本，**编号待核实**：HAIC（Humanoid Agile Object Interaction Control via Dynamics-Aware World Model，arXiv:2602.11758）用世界模型补全动态交互中的全状态物体观测。引用前请二次确认。

### ② 尚未解决的痛点

- **世界模型多用于状态估计/重建，少用于真正的 model-based 规划**：WMR/DWL 主要做隐式状态补全，未在 latent 空间做长时程 rollout 规划。
- **可微仿真的接触不可微 / 梯度爆炸**：高动态接触下梯度质量差。
- **世界模型与动作基座未统一**：motion tracker（动作模型）与 dynamics predictor（世界模型）通常是两套系统，缺乏端到端联合训练。

### ③ 可做项目

1. **（尚未充分研究）人形版 Dreamer/TD-MPC 全身控制**：latent world model + 想象 rollout 规划已成功用于四足/操作，但**全身高维人形 + 接触丰富**场景下的 latent 世界模型规划基本空白。
2. **（已有研究但有未解决问题）统一"世界-动作"基座**：联合训练 motion-tracking 动作头 + dynamics 预测头，共享 latent。

### ④ 创新点 idea（主线汇聚点）

> **「Humanoid World-Action Model (HWAM)」**——本报告主线的可落地实例：
> - **统一 latent**：本体感知 + 视觉 + 物体/接触编码进共享 latent state。
> - **双头**：① 世界模型头预测未来 latent 与稳定性/接触力；② 动作头以参考动作或任务为条件输出动作（继承 GMT/BeyondMimic 动作基座）。
> - **闭环**：latent 空间做短时 imagination rollout 选动作（行动）；预测失稳则触发恢复头（恢复）；用 ASAP-style delta action 对齐真实物理（sim-to-real）。
> - **复用现成开源**：ASAP、TWIST、VisualMimic、BeyondMimic、OmniRetarget 均开源，可作各模块基座。

这一设计把六个子方向串成"想象-规划-行动-恢复"统一闭环，每一环都有 2024–2026 真实工作支撑。

---

# 第 3 章　分支 C：语言引导 & 长时序生成（VLA 与分层规划）

> 「规划」的大脑：语言如何把抽象意图翻译成可执行的长时序动作。VLA 演化主线三步：(1) 把机器人控制 token 化、接到 Transformer（RT-1）；(2) 把互联网级 VLM 语义直接迁移到控制（RT-2/OpenVLA）；(3) 在 VLA 之上叠加"慢思考/世界模型/具身思维链"，让模型在动作前先推理与想象（GR00T、Helix、Gemini Robotics、ECoT、WorldVLA）。**主线关心的"想象-规划-行动-恢复"，本质是把世界模型嵌入 VLA 的规划回路。**

## 3.1 VLA 大模型基座

### ① 已有进展

| 工作 | 机构 | Venue/年份 | arXiv | 开源 | 一句话贡献 |
|---|---|---|---|---|---|
| **RT-1** | Google | 2022 | 2212.06817 | 是 | 图像+指令+动作统一 token 化，"机器人即序列建模" |
| **RT-2** | Google DeepMind | 2023 | 2307.15818 | 否 | VLM 与机器人数据共训，动作表示为文本 token，涌现式泛化 |
| **Octo** | UC Berkeley 等 | RSS 2024 | 2405.12213 | 是 (octo-models/octo) | 轻量 Transformer + 扩散头，80 万 OXE 轨迹预训练 |
| **OpenVLA** | Stanford/Berkeley 等 | CoRL 2024 | 2406.09246 | 是 (openvla/openvla) | 7B 全开源 VLA，超闭源 RT-2-X(55B)，开源基线事实标准 |
| **π0** | Physical Intelligence | 2024 | 2410.24164 | 部分(openpi) | VLM + flow-matching 动作专家，50Hz 高频灵巧控制 |
| **RoboVLM / What Matters in Building VLA** | 字节等 | 2024 | 2412.14058 | 是 | 系统消融 VLA 设计空间，给出构建指南 |
| **SpatialVLA** | 上海AILab/上科大 | 2025 | 2501.15830 | 是 | Ego3D 位置编码 + 自适应动作网格，注入 3D 空间结构 |
| **CogACT** | 微软研究院 | 2024 | 2411.19650 | 是 (cogact.github.io) | "认知-动作"解耦，VLM 认知特征驱动专门扩散动作模块 |

### ② 尚未解决的痛点

- **推理频率 vs 语义深度矛盾**：大 VLM backbone 推理慢（几 Hz），难满足人形全身高频闭环。
- **离散 binning 丢失高频灵巧动作信息**（FAST 即针对此）。
- **跨本体异构性**：不同自由度、夹爪 vs 灵巧手、人形全身关节的动作空间难统一。
- **评测碎片化**：LIBERO/SimplerEnv 与真实人形差距大，缺"长时序+恢复"标准基准。

### ③ 可做项目

1. **（已有研究但有未解决问题）人形全身 VLA 基座**：现有 VLA 多为桌面机械臂，人形"上肢操作+下肢移动+躯干平衡"耦合的统一动作表征尚不成熟。
2. **（已有研究但有未解决问题）统一动作分词器跨本体迁移**：FAST 是黑盒分词器雏形，但面向人形高自由度+触觉的通用 tokenizer 仍待打磨。
3. **（尚未充分研究）面向"恢复"的 VLA 基座评测集**：专门评测失败后重规划/重抓取能力的人形基准基本空白。

### ④ 创新点 idea

> **世界模型蒸馏到 VLA 的"想象先验"**：在 VLA 预训练阶段加辅助 head 预测未来若干步的潜在视觉状态（latent rollout），把世界模型的动力学先验蒸馏进基座，使 base model 天然具备"预演"能力，无需推理时额外大模型。可在 OpenVLA/π0 上做对照（有无 imagination head）。

## 3.2 双系统 / 分层架构（System1/System2, fast-slow）

### ① 已有进展

| 工作 | 机构 | 年份 | arXiv/链接 | 开源 | 一句话贡献 |
|---|---|---|---|---|---|
| **GR00T N1** | NVIDIA | 2025 | 2503.14734 | 是 (HF nvidia/GR00T-N1.5-3B) | 显式双系统：S2(Eagle VLM)慢推理 + S1(DiT 流匹配)高频动作，跨本体人形基座 |
| **GR00T N1.5** | NVIDIA | 2025 | 研究页 | 是 | 改进 VLM 冻结策略与数据配比，提升语言可控性 |
| **Helix** | Figure AI | 2025 | figure.ai/helix | 否 | S2(7-9Hz)+S1(200Hz)双系统，单套权重控制双人形协作 |
| **Gemini Robotics + ER** | Google DeepMind | 2025 | 2503.20020 | 否 | Gemini 2 派生 VLA，ER 版强化"具身推理"作高层指挥 |
| **Gemini Robotics 1.5** | Google DeepMind | 2025 | 2510.03342 | 否 | "Thinking before acting" + 跨本体运动迁移，显式 think-then-act |
| **HiRT: Hierarchical Robot Transformers** | 清华等 | 2024 | 2410.05273 | 待核实 | VLM 低频提时不变特征，高频视觉策略实时控制 |

### ② 尚未解决的痛点

- **S1-S2 接口表征**：慢系统传给快系统的应是潜向量（Helix/GR00T）还是语言子目标（Hi Robot）？缺乏统一理论。
- **异步频率下的时延与抖动**：慢系统更新时快系统如何平滑过渡（参见 RTC 实时分块）。
- **双系统训练梯度耦合**：端到端易让快系统"绕过"慢系统；分阶段又损失协同。
- **世界模型尚未进入双系统**：当前 S2 多为 VLM 推理，极少把"预测未来状态的世界模型"作为第三系统。

### ③ 可做项目

1. **（已有研究但有未解决问题）S1/S2 接口表征系统对比**：潜向量 vs 语言子目标 vs 轨迹草图，同一人形平台受控对比，目前无统一结论。
2. **（尚未被充分研究）三系统架构（S3=世界模型）**：在 S2（语义）与 S1（控制）之间插入 S2.5"想象系统"，做短时 latent rollout 验证 S2 子目标可行性后再下发 S1。
3. **（已有研究但有未解决问题）异步切换的平滑性**：RTC 解决流策略实时分块，但慢系统目标切换瞬间的全身平衡恢复尚未系统研究。

### ④ 创新点 idea

> **「想象-门控双系统（Imagination-Gated Dual System）」**：S2 产生候选子目标后，用轻量 latent world model 对每个子目标做 K 步想象 rollout，预测成功概率与碰撞/失衡风险，只把"通过想象验证"的子目标 token 送入 S1。把主线"想象-规划"直接落到双系统接口上，推理开销远低于像素级视频预测。可在 GR00T N1.5（开源）的 DiT 前加 latent dynamics 模块。

## 3.3 长时序任务规划与分解（LLM-as-planner）

### ① 已有进展

- **Hi Robot: Open-Ended Instruction Following with Hierarchical VLA** — Physical Intelligence，2025，arXiv:2502.19417。高层 VLM 把复杂/纠正性指令分解为原子语言子任务，低层 π0 执行；支持实时纠正与口头反馈。
- **RoboBrain: Unified Brain Model from Abstract to Concrete** — 北京智源（BAAI），CVPR 2025，arXiv:2502.21257（**开源**）。MLLM 统一"任务规划 + 可供性感知 + 轨迹预测"。**RoboBrain 2.0**（arXiv:2507.02029）强化空间/闭环交互推理。
- **AgiBot World + GO-1 (Genie Operator-1)** — 智元机器人，2025，arXiv:2503.06669（**开源**数据/模型）。大规模真机平台 + ViLLA（Vision-Language-Latent-Action）策略，用潜动作最大化数据利用。

### ② 尚未解决的痛点

- **规划与执行的语义鸿沟**：LLM 规划的子目标常"语义合理但物理不可行"（够不着、会失衡、顺序错误），缺物理可行性校验。
- **缺乏闭环重规划**：多数是开环分解，执行失败不更新计划。
- **长时序信用分配**：几十步任务失败溯源困难。
- **人形特有约束**：loco-manipulation 中"先走过去"还是"先伸手"的顺序决策极少被建模。

### ③ 可做项目

1. **（已有研究但有未解决问题）物理可行性校验的 LLM-planner**：LLM 分解的子目标普遍缺基于动力学的可行性过滤，公认痛点但解法不成熟。
2. **（尚未被充分研究）人形 loco-manipulation 联合时序规划**：移动+操作的顺序/并行调度规划工作极少。
3. **（已有研究但有未解决问题）失败驱动的在线重规划**：Hi Robot 支持纠正但主要靠人类口头反馈，自主失败检测→重规划闭环仍弱。

### ④ 创新点 idea

> **「World-Model-in-the-Loop Planner（想象校验式分层规划）」**：LLM 输出子目标序列后，用世界模型对整条计划做"链式想象 rollout"，在潜空间逐步推演每个子目标的预期终局状态，过滤物理不可行/失衡子目标，并对失败步触发局部重规划。**这正是主线"用世界模型做长时序 rollout 验证规划"的直接实现。** 可在 RoboBrain/GO-1 的规划层叠加 latent world model 做对照（有/无想象校验的长时序成功率）。

## 3.4 世界模型作为规划器 / 想象中的长时序推演（与主线最直接相关）

### ① 已有进展

- **WorldVLA: Towards Autoregressive Action World Model** — 阿里达摩院，2025，arXiv:2506.21539（有代码）。把 VLA 与世界模型统一进单一自回归模型，动作生成与图像预测互相增强，**"世界-动作模型融合"的标志性工作**。
- **Genie Envisioner: Unified World Foundation Platform** — 智元/AgiBot 等，2025，arXiv:2508.05635（有项目页）。在单一视频生成框架内统一策略学习、仿真评测与世界建模（GE-Base）。
- **Chain-of-Action: Trajectory Autoregressive Modeling** — ByteDance Seed，2025，arXiv:2506.09990，**开源** github.com/ByteDance-Seed/Chain-of-Action。从目标关键帧出发做"反向"动作级 CoT，自回归生成整条轨迹。
- **VLMPC: Vision-Language Model Predictive Control** — 2025，arXiv:2504.05225（待核实）。VLM 采样候选动作序列，用视频预测世界模型在 MPC 框架下评估择优——典型"世界模型当 cost 评估器"。

> ⚠️ 极新预印本，**编号待核实**：MCTS+MPC with Action-Conditioned Visual World Models（arXiv:2511.03077）、An LLM-Assisted Multi-View World Model for Long-Horizon Manipulation（arXiv:2501.06605）、DreamZero（arXiv:2602.15922）。

> 注：RT-2/Octo/OpenVLA 是纯反应式策略；本子方向核心区别在于"先在脑内推演未来，再决定动作"。NVIDIA DreamGen（用视频世界模型生成训练数据）属"世界模型反哺策略"路线。

### ② 尚未解决的痛点

- **想象保真度 vs 速度**：像素级视频 WM 推演慢且易累积误差，难做实时 MPC；潜空间 WM 快但物理一致性弱。
- **长时序误差累积**：rollout 越长漂移越大。
- **想象与真实的分布偏移（world-model exploitation）**：策略在"想象"里最优，真机上失败。
- **缺乏"恢复"的想象**：几乎没有工作专门让机器人想象"如果失败了怎么办"并预演恢复策略。

### ③ 可做项目

1. **（已有研究但有未解决问题）潜空间世界模型的长时序漂移抑制**：误差累积是公认难题，RL 物理对齐是新尝试但远未解决。
2. **（尚未被充分研究）"恢复想象"（recovery imagination）**：让机器人在世界模型里主动 rollout 失败分支并预演恢复动作，这一"想象-恢复"闭环几乎空白，**与主线高度契合**。
3. **（已有研究但有未解决问题）想象保真度自适应**：根据任务阶段动态在"快潜空间推演"与"慢像素级推演"间切换。

### ④ 创新点 idea —— 本分支与主线最核心的落点

> **「Imagine-Act-Recover 三阶段世界动作模型」**：统一一个 latent world-action 模型（借鉴 WorldVLA 的动作-图像互增强），执行前做 K 步想象选最优计划；执行中持续用世界模型做单步预测与真实观测对比，**预测误差超阈值即判定异常**；异常时在世界模型里 rollout 多条恢复轨迹并择优执行。把"想象-规划-行动-恢复"四步压进一个模型，是整份报告主线的旗舰原型。可在 GR00T/π0 + 轻量 latent dynamics head 上搭建。
>
> 附加：**想象一致性奖励抑制 world-model exploitation**——训练时加约束，使策略在想象中的预期回报与真机回报对齐（domain-aligned imagination）。

## 3.5 动作表征与生成（tokenization, flow matching/diffusion, chunking）

### ① 已有进展

| 工作 | 机构 | Venue/年份 | arXiv | 开源 | 一句话贡献 |
|---|---|---|---|---|---|
| **Diffusion Policy** | Columbia/Toyota/MIT | RSS 2023 | 2303.04137 | 是 (real-stanford/diffusion_policy) | 条件去噪扩散建模多模态动作分布 + action chunking，奠基范式 |
| **RDT-1B** | 清华 TSAIL | ICLR 2025 | 2410.07864 | 是 (thu-ml/RoboticsDiffusionTransformer) | 1B 扩散 Transformer，双臂操作最大开源扩散基座之一 |
| **FAST / π0-FAST** | Physical Intelligence | 2025 | 2501.09747 | 是 (HF physical-intelligence/fast) | 频域(DCT)动作分词，自回归 VLA 训练快至扩散法 5 倍 |
| **Real-Time Chunking (RTC)** | Physical Intelligence | 2025 | 2506.07339 | 待核实 | 执行当前块时即生成下一块并冻结重叠部分，实时无缝控制 |
| **HybridVLA** | 北大等 | 2025 | 2503.10631 | 待核实 | 单 LLM 内融合自回归与扩散两种动作生成 |
| **UniVLA** | 待核实 | 2025 | 2505.06111 | 待核实 | 从无标注视频无监督学"任务中心潜动作"，跨环境统一动作表征 |

### ② 尚未解决的痛点

- **离散（自回归）vs 连续（扩散/流）之争**：前者与语言推理统一但损失精度，后者精度高但与文本推理割裂。
- **高频 + 高自由度**：人形全身（40+ DoF）的动作分词/生成仍是开放问题。
- **chunk 长度与反应性矛盾**：长 chunk 平滑但反应慢，短 chunk 反应快但抖动。
- **动作表征与世界模型脱节**：动作 token 通常不携带"预测后果"的语义。

### ③ 可做项目

1. **（已有研究但有未解决问题）人形全身动作分词器**：FAST 主要面向机械臂，人形全身高自由度+触觉的频域/潜动作分词仍待研究。
2. **（已有研究但有未解决问题）自回归与扩散的统一**：HybridVLA 是早期尝试，何时用哪种生成的自适应路由尚无定论。
3. **（尚未被充分研究）"后果感知"动作 token**：让每个动作 token 关联其在世界模型中的预测状态变化。

### ④ 创新点 idea

> **「后果条件化的动作分块（Consequence-Conditioned Chunking）」**：用世界模型预测每个候选动作 chunk 执行后的潜状态，把"预测状态"作为额外条件反馈给流匹配动作头，生成"已被想象验证安全"的 chunk。结合 RTC 的实时分块，可在生成下一块时就用世界模型预演其后果，实现"边想边动"。可在 π0/RDT-1B 上做插件式实验。

## 3.6 推理与具身思维链（Embodied Chain-of-Thought）

### ① 已有进展

- **ECoT: Robotic Control via Embodied Chain-of-Thought Reasoning** — Berkeley/Stanford 等，2024，arXiv:2407.08693，**开源** embodied-cot.github.io。训练 VLA 在动作前显式推理计划/子任务/运动/可视化特征（grounded），提升泛化与可解释性。
- **MolmoAct: Action Reasoning Models that can Reason in Space** — Allen AI (Ai2)，2025，arXiv:2508.07917（Ai2 开源传统）。把观测编码为深度感知 token，生成可编辑的中层空间轨迹 trace 再出低层动作。
- **ChatVLA: Unified Multimodal Understanding and Robot Control** — 美的/华东师大等，EMNLP 2025，arXiv:2502.14420。混合专家+分阶段训练，缓解"理解"与"控制"互相干扰。
- **Emma-X: Embodied Multimodal Action Model w/ Grounded CoT + Look-ahead Spatial Reasoning** — SUTD 等，2024，arXiv:2412.11974。把 grounded CoT 与"前瞻空间推理"结合。

### ② 尚未解决的痛点

- **推理慢**：逐 token 文本 CoT 严重拖慢控制频率（ECoT 核心瓶颈）。
- **推理是否"真用上了"**：CoT 可能只是事后合理化，与实际动作因果关联弱。
- **文本 CoT 不适合连续空间**：语言难精确表达连续运动（MolmoAct 用轨迹 trace 部分缓解）。
- **缺乏"预测性"推理**：多数 ECoT 推理当前状态，极少推理"未来会发生什么"。

### ③ 可做项目

1. **（已有研究但有未解决问题）推理-动作因果对齐**：验证并强制 CoT 真正驱动动作（而非事后解释），缺乏因果度量。
2. **（已有研究但有未解决问题）实时具身推理**：Thoughts reuse / 异步推理是热点但远未解决人形实时性。
3. **（尚未被充分研究）"想象型 CoT"（latent/visual CoT）**：用世界模型生成的未来潜状态/未来帧作为推理步，而非纯文本。

### ④ 创新点 idea

> **「World-Model Chain-of-Thought（想象式具身思维链）」**：把 ECoT 的"文本推理步"替换/增强为"世界模型潜空间想象步"——模型在动作前于潜空间想象未来 N 步状态序列作为 reasoning trace，再据此出动作；失败时想象的偏差直接定位错误步，实现"可解释的恢复"。相比纯文本 CoT，潜空间推理更快、更贴合连续控制，且天然支持长时序预演。可在 ECoT（开源）框架上做消融。

## 3.7 分支 C 小结

1. **一条贯穿主线的统一原型**：3.2 的"三系统"、3.3 的"想象校验式规划"、3.4 的"Imagine-Act-Recover"、3.5 的"后果条件化分块"、3.6 的"想象式思维链"，本质是同一思想在不同抽象层的投影——**把世界模型的前向推演插入 VLA 的不同环节**。
2. **最具差异化的空白点**：**"恢复（recovery）"维度** 在现有 VLA/规划/世界模型工作中几乎系统性缺失，是分支 C 相对全领域最有原创性、且最直接命中报告标题的切入点。
3. **可立即落地的开源底座**：GR00T N1.5、OpenVLA、π0(openpi)、RDT-1B、Diffusion Policy、ECoT、Chain-of-Action、AgiBot World/GO-1。

---

# 第 4 章　分支 D：物理交互的多模态感知（触觉 / 力 / 多模态）

> 让「想象」覆盖看不见的物理量。在世界动作模型框架下，触觉/力/接触不仅是观测，更应成为世界模型**预测的对象**——让机器人在 latent space 中"想象"接触发生时的触觉/力未来。**全章贯穿一条判断：触觉/力表征绝大多数仍停留在"判别/对齐"，真正"预测触觉未来"的世界模型极少且多为 2025-2026 新作——这是最空、最契合主线、最易出原创成果的缝隙。**

## 4.1 触觉感知与视触觉传感

### ① 已有进展

**硬件/传感器**
- **GelSight: High-Resolution Robot Tactile Sensors** — Yuan, Dong, Adelson（MIT），*Sensors* 2017。视触觉奠基，用弹性体+相机测接触几何。
- **DIGIT: A Low-Cost Compact High-Resolution Tactile Sensor** — Lambeta 等（Meta FAIR），RA-L/ICRA 2020，arXiv:2005.14679，**开源** github.com/facebookresearch/digit-design。把视触觉小型化到指尖。**DIGIT 360** 模块化平台集成 tactile image + audio + IMU + pressure（**开源**）。
- **AnySkin: Plug-and-play Skin Sensing for Robotic Touch** — Bhirangi 等（NYU/CMU），arXiv:2409.08276，2024。磁性可替换"皮肤"，解决 GelSight 系易损、需重标定。
- **eFlesh: Magnetic Touch Sensing using Cut-Cell Microstructures** — arXiv:2506.09994，2025。低成本、易制造、高度可定制的磁性触觉传感器。

**统一/跨传感器表征**
- **T3: Transferable Tactile Transformers** — arXiv:2406.13640，2024。跨多传感器、多任务触觉表征，亚毫米插装成功率较 baseline 高约 25%。
- **UniTouch: Learning Unified Multimodal Tactile Representations** — arXiv:2401.18084，2024。把视触觉 embedding 对齐到 vision/language/sound，零样本触觉理解。
- **AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors** — arXiv:2502.12191，ICLR 2025。融合触觉图像与视频，统一静态-动态表征。
- **Sensor-Invariant Tactile Representation (SITR)** — arXiv:2502.19638，ICLR 2025，**开源** github.com/hgupt3/gsrl。仿真多传感器训练 transformer，最小标定泛化到新传感器。
- **Touch2Touch / Cross-Sensor Touch Generation** — arXiv:2409.08269 / 2510.09817。给定一种传感器的触觉信号，生成另一种传感器在相同接触下的读数。

### ② 尚未解决的痛点
- 跨传感器仍以"读数翻译/对齐"为主，缺乏对接触物理量（法向/切向力场、滑移、形变）的因果可解释表征，力幅值误差仍大。
- 视触觉仅覆盖指尖、易磨损；磁性皮肤耐用但低维、未标定。无统一表征同时享有"高几何分辨率 + 全手覆盖 + 耐久"。
- 触觉表征**几乎都是判别式/对齐式，极少做"预测式"**——不能预测"下一时刻接触会变成什么样"，与世界模型主线脱节。

### ③ 可做项目
- **（已有研究但有未解决问题）把 SITR/AnyTouch 的"传感器不变表征"从判别式升级为生成-预测式**：训练跨传感器、可预测未来 N 帧触觉的 latent 模型。
- **（已有研究但有未解决问题）视触觉（高分辨指尖）+ 磁性皮肤（全手覆盖）异构融合的统一 token 化**。
- **（尚未充分研究）把"滑移检测/切向力"作为显式可预测物理量嵌入跨传感器表征。**

### ④ 创新点 idea
> **「Tactile Future Tokens」——传感器无关的触觉未来预测器**。基于 SITR/AnyTouch 的传感器不变 latent，训练条件于（当前触觉 + 拟执行动作微元）的 masked/自回归预测头，输出"未来触觉 token 序列"。使任意视触觉/磁性传感器都能接入世界模型，机器人执行前先"想象触觉"，真实触觉偏离预测即触发恢复——直接服务主线"想象-恢复"环。

## 4.2 力/力矩感知与接触丰富操作

### ① 已有进展
- **FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation** — arXiv:2411.15753，2024。高频 F/T 与视觉融合的反应式策略。
- **Just Add Force for Contact-Rich Robot Policies** — arXiv:2410.13124，2024。把力直接加入策略，对易碎物精细抓取优于纯位置策略。
- **ManipForce / Force-Guided Policy Learning with Frequency-Aware Representation** — arXiv:2509.19047，2025。频域力表征，在齿轮装配、电池插装等接触任务鲁棒。
- **UMI-FT: In-the-Wild Compliant Manipulation** — arXiv:2601.09988，2026（⚠️编号待核实）。在 UMI 手持夹爪加 F/T，把"野外柔顺操作数据采集"补齐力维度。

### ② 尚未解决的痛点
- **力是高频（kHz）、视觉是低频（~30Hz）**，时序对齐与"何时让力主导"靠启发式相位调度，缺学习化的模态仲裁。
- F/T 多为腕部/夹爪单点，缺分布式接触力场；接触模式切换（粘-滑、make/break contact）的离散非光滑性难被平滑策略捕捉。
- 力数据难在仿真高保真生成，sim2real 缺口大；野外力数据稀缺。

### ③ 可做项目
- **（已有研究但有未解决问题）学习化的"模态仲裁门控"**：由世界模型预测"接下来是否进入接触/滑移"，据此动态调度视觉慢规划与力快矫正（替代手工相位）。
- **（已有研究但有未解决问题）把腕部单点 F/T 反演为指尖分布式接触力场**（结合 4.1 触觉）。
- **（部分未研究）野外力数据规模化**：大规模 force-conditioned 预训练。

### ④ 创新点 idea
> **「Contact-Event World Model」：离散接触事件 + 连续力场的混合预测世界模型**。把 make-contact/break-contact/stick/slip 建模为离散隐变量，连续分支预测未来力/力矩轨迹。策略以"预测的接触事件时刻"为锚点切换 slow-fast——既解决模态仲裁，又让"恢复"成为"预测接触事件未按计划发生→重规划"的自然结果。

## 4.3 灵巧手操作与多模态融合

### ① 已有进展
- **3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing** — Columbia/UIUC/MIT，CoRL 2024，arXiv:2410.24091，项目页 binghao-huang.github.io/3D-ViTac。首个把视觉与触觉融入**统一 3D 点云空间**做模仿学习的双手灵巧系统。
- **NeuralFeels with neural fields** — Suresh 等（Meta/CMU），**Science Robotics 2024**，arXiv:2312.13469。多指手在线学习物体几何 neural field 并联合优化位姿图，做视触觉在手感知（旗舰工作）。
- **Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids** — arXiv:2502.20396，2025。面向人形的视觉灵巧操作 sim2real。
- **Sparsh-skin: Self-supervised perception for tactile skin covered dexterous hands** — arXiv:2505.11420，2025。磁性触觉皮肤全手覆盖 + 自监督 latent。
- **DexUMI: Human Hand as Universal Manipulation Interface** — arXiv:2505.21864，2025。用人手作自然接口迁移灵巧技能到不同机器手。

### ② 尚未解决的痛点
- 灵巧手**多指 × 多模态 token 数巨大**，实时性与融合架构受限；触觉常被视觉主导而"被忽略"。
- 在手操作**部分可观测 + 遮挡严重**，纯视觉失效；触觉跨物体/跨手泛化差。
- 示范数据采集昂贵，人手到机器手的形态差仍大。

### ③ 可做项目
- **（已有研究但有未解决问题）把 3D-ViTac 的"统一 3D 点云"扩展为含触觉的 4D 时空 + 可预测未来接触点云**，直接对接世界模型。
- **（已有研究但有未解决问题）NeuralFeels 的在线 neural field 升级为可前向 rollout 的 neural field 世界模型**（预测旋转/插入后物体几何与接触）。
- **（部分未研究）全手磁性皮肤 + 指尖视触觉异构融合的灵巧策略。**

### ④ 创新点 idea
> **「Imagine-then-Manipulate」灵巧在手世界模型**：在 NeuralFeels 在线 neural field 之上加动作条件前向动力学头，机器人每步重定向前先"想象"若执行某 finger gait 后的物体姿态 + 各指尖预测触觉；选"想象触觉最稳定"的动作。落空（真实触觉与想象不符，如滑落）即触发恢复抓取。把 4.3 旗舰感知工作接入主线。

## 4.4 触觉/力信息融入世界模型与策略（与主线最直接耦合）

### ① 已有进展
- **RoboPack: Learning Tactile-Informed Dynamics Models for Dense Packing** — Ai 等，**RSS 2024**，arXiv:2407.01418。用 recurrent GNN 从历史视触觉观测估计物体状态（粒子 + 物体级 latent physics），做密集装箱的**触觉信息动力学模型 + 规划**——"触觉世界模型用于规划"的代表。
- **Multi-Modal World Model for Physical Robot Interactions** — arXiv:2304.11193，2023。视触觉预测在"视觉上模糊的交互"中收益最大。

> ⚠️ 极新预印本，**编号待核实**（这批最需复核，但方向高度契合主线）：Visuo-Tactile World Models（arXiv:2602.06001，称物体永久性提升约 33%、合规性约 29%）、Contact-Aware Neural Dynamics（arXiv:2601.12796）、DreamTacVLA（arXiv:2512.23864）、Video-Tactile-Action Models（arXiv:2603.23481）。

### ② 尚未解决的痛点
- **触觉/力的"未来"极难预测**：接触瞬变高频、强非线性、对微小动作敏感，latent rollout 容易发散。
- 现有触觉世界模型多为物体级粒子/状态预测（RoboPack）或像素级视触觉，缺乏"接触物理语义层"（力闭合、摩擦锥、滑移裕度）的可预测中间表征。
- **想象的触觉如何驱动恢复**：多数工作止于"预测更准"，未闭环到"预测失配→主动重规划/恢复"。

### ③ 可做项目
- **（已有研究但有未解决问题）在 RoboPack 类触觉世界模型上加"触觉预测不确定性"估计**，作为风险信号驱动 MPC 何时减速/重抓。
- **（基本未研究）"接触物理语义"中间层世界模型**：预测摩擦锥/滑移裕度/力闭合质量的未来，而非原始触觉像素。
- **（已有研究但有未解决问题）触觉世界模型 + 离线 RL 做长程接触任务的"想象内规划"。**

### ④ 创新点 idea
> **「Tactile Dream-and-Recover」：以触觉预测失配为恢复触发器的世界-动作模型**。世界模型联合预测（视觉 latent + 触觉 latent + 力轨迹 + 滑移裕度）的未来；策略在想象空间内 MPC/扩散规划。关键创新：定义**"触觉惊奇度"（tactile surprise = 预测触觉与真实触觉的 latent 距离）**作为在线信号，超阈值即从当前真实状态重新"想象-规划"，实现毫秒级恢复。**把"想象-规划-行动-恢复"四环用触觉显式闭合，是主线的核心可发表点。**

## 4.5 3D 场景理解、Affordance 与空间表征

### ① 已有进展
- **DP3: 3D Diffusion Policy** — Ze 等，**RSS 2024**，arXiv:2403.03954，**开源** github.com/YanjieZe/3D-Diffusion-Policy。稀疏点云 + 高效点编码器 + 扩散策略，72 个仿真任务显著优于 2D。
- **iDP3: Generalizable Humanoid Manipulation with Improved 3D Diffusion Policies** — IROS 2025，arXiv:2410.10803，**开源** github.com/YanjieZe/Improved-3D-Diffusion-Policy（含 Humanoid-Teleoperation）。用 egocentric 3D 表征去除对相机标定/分割的依赖，**全尺寸人形**单场景数据即可泛化部署——直接服务人形主线。
- **M2T2: Multi-Task Masked Transformer for Object-centric Pick and Place** — CoRL 2023，arXiv:2311.00926。基于原始点云推理接触点并预测多动作模式的 6-DoF 夹爪位姿。
- **A Unified Representation of Affordances for Tool Usage and Articulation with VLMs** — arXiv:2409.20551，2024。
- **3D-ADLLM: Open-Vocabulary Affordance Detection in 3D Worlds** — arXiv:2502.20041，2025。

### ② 尚未解决的痛点
- 3D 表征与 affordance 多为纯视觉/几何，几乎不含接触/力先验——预测的"可操作区域"不保证物理可抓/可施力。
- DP3/iDP3 等点云策略**不预测未来**（非世界模型）；affordance 是静态标注，不随交互更新。
- affordance 缺乏"接触后验证"：点了"这里能抓"但抓下去打滑无反馈修正。

### ③ 可做项目
- **（已有研究但有未解决问题）"Tactile-grounded affordance"**：用真实接触/触觉结果反向监督/修正 3D affordance（从视觉先验→接触验证后验）。
- **（已有研究但有未解决问题）把 iDP3 的 egocentric 3D 表征升级为可预测的 3D 接触世界模型**（预测交互后点云 + 接触点）。
- **（部分未研究）M2T2 接触点预测 + 触觉确认的闭环抓取。**

### ④ 创新点 idea
> **「Affordance-as-Predicted-Contact」：把 affordance 从静态分割重定义为"世界模型预测的成功接触分布"**。给定 3D 场景点云 + 候选动作，世界模型预测每个接触候选的未来触觉/力结果与成功概率，affordance 热图 = 预测成功接触的边缘分布。执行后用真实触觉更新该后验（贝叶斯式 affordance），实现"看-想象接触-验证-更新"的空间表征闭环，天然接 iDP3 人形栈。

## 4.6 多模态统一表征（视觉+触觉+力+本体+音频）

### ① 已有进展
- **Sparsh: Self-supervised touch representations** — Higuera 等（Meta FAIR/CMU），CoRL 2024，arXiv:2410.24090，**开源** github.com/facebookresearch/sparsh（HF facebook/sparsh-mae-base）。46 万+触觉图像上 masking + 自蒸馏，支持 DIGIT/GelSight 多种传感器，配 TacBench 基准。
- **Sparsh-X: Multisensory Touch Representations** — arXiv:2506.14754，2025，**开源**。自监督融合 tactile image + audio + IMU + pressure（面向 Digit 360）为统一表征。
- **FuSe: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding** — arXiv:2501.04693，2025。用语言 grounding 把 vision+touch+audio 异构传感器接入通用策略，支持零样本跨模态推理。
- **Multi-Modal Manipulation via Multi-Modal Policy Consensus** — arXiv:2509.23468，2025。把策略分解为各模态专属扩散模型 + 路由网络学习共识权重，自适应平衡视觉/触觉。

**触觉-VLA 多模态策略（与 4.1/4.3 交叉）**
- **Tactile-VLA: Unlocking VLA Model's Physical Knowledge for Tactile Generalization** — 清华，arXiv:2507.09160，2025。激活 VLA 内蕴物理知识做零样本接触操作，提出 Tactile Chain-of-Thought。
- **VTLA: Vision-Tactile-Language-Action with Preference Learning for Insertion** — arXiv:2505.09577，2025。

### ② 尚未解决的痛点
- **模态不平衡**：视觉信息密度高，触觉/力/音频常被淹没；现有融合多在 late-fusion 或加权，缺物理因果约束。
- **缺乏真正统一、可预测未来的多模态 latent**：Sparsh-X 已统一编码但仍是表征，非世界模型；FuSe 靠语言对齐但不 rollout。
- 音频（接触声、滑移声）与本体的利用仍非常初步；跨 embodiment 多模态对齐数据稀缺。

### ③ 可做项目
- **（已有研究但有未解决问题）在 Sparsh-X 统一编码之上加多模态联合未来预测头**（预测 vision+tactile+force+audio 的下一步 latent），把统一表征升级为统一世界模型。
- **（已有研究但有未解决问题）用 Policy Consensus 的路由思想做"模态可信度随接触阶段动态变化"的世界模型仲裁。**
- **（部分未研究）接触声（audio）作为"接触发生/滑移"的廉价高频信号，补 F/T 之不足。**

### ④ 创新点 idea
> **「Unified Multimodal World-Action Model (UM-WAM)」：一个 backbone、五模态联合预测 + 动作生成**。以 Sparsh-X/FuSe 风格统一编码 vision+tactile+force+proprioception+audio，训练目标 = 联合预测各模态未来 latent（世界模型） + 动作扩散（策略）。引入**"接触阶段感知的模态置信门"**：接触前视觉主导，接触瞬间触觉/力/音频置信度跃升驱动预测与恢复。这是把分支 D 全部子方向汇聚到主线的统一收口。

## 4.7 分支 D 小结

1. **从"判别/对齐"到"预测"**：触觉/力/多模态表征绝大多数仍是判别式或跨传感器对齐式（Sparsh、SITR、AnyTouch、UniTouch）。真正"预测触觉/接触未来"的世界模型只有极少数（RoboPack、Multi-Modal WM），且新作集中在 2026——**最空、最契合主线、最易出原创成果的缝隙**。
2. **"恢复"环几乎无人显式做**：现有工作止步"预测更准/操作更稳"，鲜有把"预测触觉失配"显式作为恢复/重规划触发器。4.4 的 idea（Tactile Surprise / Contact-Event WM）是主线"恢复"的直接落点。
3. **人形栈已具备**：iDP3 + Humanoid-Teleoperation（IROS 2025，开源）提供人形 3D 策略与遥操数据底座；Sparsh/Sparsh-X（开源）提供触觉表征底座。**把"可预测的触觉/接触世界模型"嫁接到 iDP3 人形栈，是工程上最现成、最能产出 demo 的路径。**

---

# 第 5 章　分支 E：物理环境鲁棒的恢复与安全性

> 「失败预测 + 恢复」闭环。当人形机器人在真实物理世界遭遇扰动、失衡、摔倒乃至策略失效时，如何"想象危险后果 → 预测失败 → 安全规划 → 行动恢复"。世界模型在这里既是"危险想象器"（imagine-then-avoid），也是"失败预测器"（predict-then-recover），把恢复与安全从被动反应升级为主动预判。

## 5.1 摔倒恢复 / 自主起身（Fall Recovery & Getting-Up）

### ① 已有进展（2025 年人形领域进展最快的子方向之一）

- **HoST: Learning Humanoid Standing-up Control across Diverse Postures** — Shanghai AI Lab / InternRobotics 等，**RSS 2025（Best Systems Paper Finalist）**，arXiv:2502.08378，**开源** github.com/InternRobotics/HoST。多 critic 架构 + 课程式力控约束，从零（无示教数据）学出可跨多种躺姿/地形直接部署到 Unitree G1 的起身策略。
- **HumanUP: Learning Getting-Up Policies for Real-World Humanoid Robots** — UIUC/UCSD（Xialin He, Runpei Dong 等），**RSS 2025**，arXiv:2502.12152，项目页 humanoid-getup.github.io。两阶段（发现 → 平滑/强鲁棒化）RL，显式建模接触几何与稀疏起身轨迹，首次实现真机人形从任意躺姿、多地形自主起身。
- **FRASA: An End-to-End RL Agent for Fall Recovery and Stand Up** — arXiv:2410.08655。用 CrossQ 把"摔倒恢复"与"起身"统一进单一端到端策略。
- **Multi-Stage Curriculum Learning for High-Dynamics Humanoid Fall Recovery** — arXiv:2502.20061。分阶段课程，逐步引入更高维、更高动态的恢复任务。

> ⚠️ 极新预印本，**编号待核实**：Unified Humanoid Fall-Safety Policy from a Few Demonstrations（arXiv:2511.07407，把少量人类示教 + 扩散"安全反应记忆"统一了摔倒预防、冲击缓解与恢复）。

### ② 尚未解决的痛点

- 起身策略多为"开环 scripted 终点"：缺乏对**周围障碍/狭小空间**的语义感知（撞到桌腿、悬崖边起身会二次摔倒）。HumanUP/HoST 主要靠本体感受，视觉融入仍弱。
- **冲击缓解（impact mitigation）与起身解耦**：多在"已经躺平"后才起身，对"正在摔倒的几百毫秒"如何主动选择最优倒地姿态以减少损伤、并使后续起身更易，研究稀少。
- 硬件损伤约束几乎缺席：真机反复摔学习成本极高。
- 缺乏"恢复可行性预判"：机器人不知道**当前姿态是否还可恢复**，何时该呼救/锁定关节。

### ③ 可做项目

1. **（已有研究但有未解决问题）视觉-语义条件的安全起身**：在 HoST/HumanUP 基础上加入第一人称深度+语义，使起身落点避开障碍/边缘。
2. **（基本未被研究）摔倒全过程"protective fall → recovery"联合优化**：把倒地姿态选择作为后续起身难度的决策变量联合训练，类似柔道受身。
3. **（已有零星研究）恢复可行性分类器 + 安全停机**：训练判别"是否可自起身/需外部介入"的运行时模块。

### ④ 创新点 idea（结合世界模型主线）

> **「Imagined Recovery Tree」——世界模型驱动的起身可行性想象**：用学到的人形世界模型，在摔倒瞬间对若干候选倒地/起身动作序列做 latent rollout，预测每条路径的 (a) 二次摔倒概率、(b) 硬件冲击峰值、(c) 起身成功率，选择风险最低的"protect-then-getup"路径。把 HumanUP 的两阶段拓展为"想象阶段（WM rollout）→ 执行阶段"，使起身从被动 scripted 变为预判式规划。

## 5.2 扰动抵抗 / 推力恢复 / 失衡恢复

### ① 已有进展

- **Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking** — arXiv:2404.19173。系统重审奖励设计，给出抗自然扰动的站立/行走 sim-to-real 训练与评测。
- **Reactive Stepping for Humanoid Robots using RL** — arXiv:2203.01148。仅用瞬时本体感受学出可平滑迁移真机的站立推力恢复（反应式迈步）。
- **Enhancing Model-Based Step Adaptation for Push Recovery through RL of Step Timing and Region** — arXiv:2411.01000。用 RL 学步态时序与落脚区域，增强强推下双足恢复。
- **Robust Humanoid Push Recovery and Locomotion with Reduced Order Models** — Caltech AMBER Lab，arXiv:2505.11495。结合 HLIP 降阶模型，仿真中可恢复达 100N 量级推力。

### ② 尚未解决的痛点

- 大多假设扰动是"未知力脉冲"，**不利用视觉提前预判**（看到有人推过来/物体飞来）。预判式恢复几乎空白。
- 推力恢复与**手中负载/操作任务**的耦合弱：手持重物时恢复策略需重新平衡。
- 多接触环境恢复（扶墙、抓栏杆）——人类失衡第一反应是抓附近物体，机器人极少用环境做"接触辅助恢复"。

### ③ 可做项目

1. **（已有研究但有未解决问题）视觉-预判式推力恢复**：引入视觉，让机器人在受力前就预紧/调整支撑多边形。
2. **（基本未被研究）接触辅助恢复（contact-aided balance recovery）**：把"抓墙/扶栏"作为可用动作，扩展恢复动作空间。
3. **（已有研究）负载自适应推力恢复**：手持未知质量物体时的在线惯量辨识 + 恢复。

### ④ 创新点 idea

> **「Anticipatory Push Recovery via World-Model Forewarning」**：世界模型从视觉流预测"即将到来的扰动"（逼近的人/物体、地面打滑前兆），输出未来 0.3–1s 的失衡风险图，提前调整重心与步态裕度。把恢复从"受力后 reactive"提升为"受力前 proactive"。可与 5.6 的不确定性结合：当世界模型对扰动预测不确定性高时，自动收敛到更保守的宽支撑步态。

## 5.3 Sim-to-Real 鲁棒性与域随机化

### ① 已有进展

- **RMA: Rapid Motor Adaptation for Legged Robots** — Kumar, Fu, Pathak, Malik（Berkeley/CMU），arXiv:2107.04034，项目页 ashish-kmr.github.io/rma-legged-robots。teacher-student + 在线 latent 适应模块，纯仿真训练零微调部署，奠定"特权信息蒸馏 + 快速适应"范式。**A-RMA**（arXiv:2205.15299）扩展到双足 Cassie。
- **LoopSR: Looping Sim-and-Real for Lifelong Policy Adaptation of Legged Robots** — arXiv:2409.17992。Transformer 把真实轨迹映射回 latent 并重建"数字孪生"，闭环回灌仿真持续改进。

### ② 尚未解决的痛点

- 域随机化"宁宽勿窄"导致保守、动作僵硬，牺牲敏捷换鲁棒。
- RMA 类适应是"隐式 latent"，**不可解释、不可验证**——无法保证适应后仍满足安全约束。
- 人形高维 + 高重心，sim-to-real gap 比四足更大；摔倒/恢复这种接触富集任务的 gap 尤其难随机化覆盖。
- 适应模块在 OOD 下会**静默失效**，缺乏自检。

### ③ 可做项目

1. **（已有研究但有未解决问题）安全感知的在线适应**：把 RMA 的适应 latent 与安全约束耦合，适应时保证 CBF/可行集不被破坏。
2. **（部分研究）接触富集任务的自动域随机化**：针对起身/恢复，专门随机化碰撞几何与接触参数。
3. **（基本未被研究）适应模块的 OOD 自检**：让 RMA encoder 输出"当前真实域是否在训练分布内"的置信度。

### ④ 创新点 idea

> **「World-Model-Calibrated Adaptation」**：用世界模型在真机部署时在线对比"想象的下一状态 vs 实际观测"，其预测残差既作为 system-ID 信号驱动 RMA 式适应，又作为 sim-to-real gap 的可量化度量。当残差超阈值（世界模型也没见过的真实动力学），触发保守安全模式而非盲目适应。把 LoopSR 的"重建数字孪生"与运行时安全监控（5.5）统一在一个世界模型里。

## 5.4 安全强化学习与安全约束（Safe RL, CBF & Safety Filters）

### ① 已有进展

**约束优化路线（软约束/Lagrangian）**
- **CPO: Constrained Policy Optimization** — Achiam, Held, Tamar, Abbeel，ICML 2017（经典奠基；arXiv 编号待二次核实）。带单调改进保证的约束 RL。
- **Responsive Safety in RL by PID Lagrangian Methods** — Stooke, Achiam, Abbeel，ICML 2020（PMLR v119）。用 PID 控制器调拉格朗日乘子，抑制约束震荡/过冲。
- 工程库：OmniSafe / Safe-Policy-Optimization。

**硬约束/安全滤波路线（CBF & 可达性）**
- **Safe Legged Locomotion using Collision Cone Control Barrier Functions (C3BF)** — arXiv:2309.01898。碰撞锥 CBF 保证腿足机器人在静/动态障碍中安全。
- **Learning for Layered Safety-Critical Control with Predictive CBFs** — arXiv:2412.04658。用学习增强预测式 CBF 的分层安全控制。

> ⚠️ 极新预印本，**编号待核实**：A Survey of Safe RL and Constrained MDPs（arXiv:2505.17342）、Safety Filtering RL in Training with CBFs（arXiv:2510.14959）。

### ② 尚未解决的痛点

- CBF/可达性需要**已知或可解析的动力学与失败集**；人形高维、接触切换、失败集难手写。
- Lagrangian/CPO 的约束是**期望意义**的软约束，不能保证"每一步"安全，对摔倒这种灾难性单次事件不够。
- 安全滤波器与 RL 策略目标冲突导致性能塌缩或抖动；"安全 vs 敏捷"的 Pareto 前沿在人形上几乎没系统刻画。
- 几乎所有 Safe-RL 在低维/操作任务验证，**人形全身恢复场景的安全 RL 极度空白**。

### ③ 可做项目

1. **（基本未被研究）人形摔倒-恢复的安全 RL 基准**：定义全身硬件冲击/不可恢复态为 cost，系统比较 CPO/PID-Lagrangian/CBF-filter。空白明显。
2. **（已有研究但有未解决问题）学习型 CBF over 人形失败集**：从摔倒数据学出神经 CBF，替代手写失败集。
3. **（已有研究）训练期安全滤波迁移到人形全身控制。**

### ④ 创新点 idea

> **「Latent World-Model CBF (WM-CBF)」**：在世界模型 latent 空间里学一个神经控制屏障函数，把"不可恢复摔倒"定义为 latent 失败集，用世界模型的前向预测计算 CBF 的时间导数约束。这样无需解析动力学即可对高维人形施加**每步硬安全保证**。安全滤波器作用在世界模型想象的轨迹上，实现"想象中先撞墙，现实中不撞墙"。

## 5.5 失败检测 / 运行时监控 / OOD 检测（2024–2026 增长最猛）

### ① 已有进展（已形成"无失败数据 + 共形预测 + OOD"主流范式）

- **FAIL-Detect: Can We Detect Failures Without Failure Data?** — Toyota Research Institute 等，**RSS 2025**，arXiv:2503.08558，**开源** github.com/CXU-TRI/FAIL-Detect。把失败检测建模为序贯 OOD，仅用成功数据 + 共形预测给出统计保证的运行时报警。
- **SAFE: Multitask Failure Detection for Vision-Language-Action Models** — **NeurIPS 2025**，arXiv:2506.09937，项目页 vla-safe.github.io。利用 VLA 内部高层特征学一个可泛化到**未见任务**的失败检测器 + 共形预测阈值。
- **Sentinel: Unpacking Failure Modes of Generative Policies** — arXiv:2410.04640。把时间一致性检测与 VLM 进度监控融合，比单一检测器多抓 18% 失败。
- **Aha: A VLM for Detecting and Reasoning Over Failures in Robotic Manipulation** — arXiv:2410.00371（开源 VLM）。把失败检测框成自由形式语言推理，可解释说明"哪里错了"。

> ⚠️ 极新预印本，**编号待核实**：FIPER: Failure Prediction at Runtime for Generative Robot Policies（arXiv:2510.09459）。

### ② 尚未解决的痛点

- 绝大多数针对**桌面机械臂操作**，几乎没有面向**人形全身/locomotion 的运行时失败检测**（摔倒前兆、步态发散）。
- "检测到失败 → 然后呢"：恢复闭环弱，且多在仿真。
- 共形预测保证是**边际（marginal）**的，在分布漂移下失效；时间相关数据的有效共形仍是开放问题。
- 检测延迟 vs 误报权衡：人形上"误报急停"本身可能引发摔倒，代价非对称。

### ③ 可做项目

1. **（基本未被研究）人形 locomotion 的预测式摔倒检测**：把 FAIL-Detect/FIPER 的 OOD+共形思路从机械臂迁移到人形步态，预测"N 步后摔倒概率"。明确空白。
2. **（已有研究但有未解决问题）检测-恢复闭环**：把 5.1 起身策略接到失败检测器后端，形成"监测→预判→主动护倒→起身"全栈。
3. **（已有研究）面向时序的有效共形（adaptive conformal）用于机器人。**

### ④ 创新点 idea

> **「Predict-the-Fall Monitor」——世界模型作为失败预测器**：用人形世界模型对当前策略做短程 latent imagination rollout，若想象轨迹进入 latent 失败集（复用 5.4 的 WM-CBF 失败集），则在真机尚未摔倒前触发预警与护倒策略。预测分数用共形预测校准给统计保证。把 FIPER 的"OOD 指标"升级为"基于世界模型的因果失败预测"，并直接驱动 5.1 的恢复——形成主线"预测失败→安全恢复"闭环。

## 5.6 世界模型用于安全与不确定性估计（与主线最直接交汇）

### ① 已有进展（2025–2026 已涌现专门工作）

- **Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis** — arXiv:2502.00935。在 latent 空间做 HJ 可达性，使安全滤波超越简单避障，处理学到的复杂失败语义。
- **The Safety Challenge of World Models for Embodied AI Agents** — arXiv:2510.05865。系统讨论用世界模型预判未来状态以增强具身安全的机遇与隐患。

> ⚠️ 极新预印本，**编号待核实**（这批最贴主线但最需复核）：UNISafe: Uncertainty-aware Latent Safety Filters for Avoiding OOD Failures（arXiv:2505.00779，用生成式世界模型的认知不确定性把机器人推离失败）、Designing Latent Safety Filters using Pre-Trained Vision Models（arXiv:2509.14758）。

### ② 尚未解决的痛点

- latent 安全滤波/可达性目前几乎都在**操作/导航**验证，**没有用于人形全身动态恢复**（高维、强接触、快时变）。
- 世界模型的**认知不确定性估计本身不可靠**：模型可能"自信地想象错误的安全未来"（hallucinated safe rollout），反而误导安全滤波。
- 长程想象的**误差累积**使远期风险预测失真；安全 horizon 与预测可信 horizon 不匹配。
- 缺乏把"世界模型预测的风险"与"实际硬件后果（冲击力、损伤）"对齐的物理-语义桥梁。

### ③ 可做项目

1. **（基本未被研究）人形恢复的 latent HJ 安全滤波**：把 latent reachability 首次落到人形起身/推力恢复，失败集=不可恢复摔倒。空白且高价值。
2. **（已有研究但有未解决问题）世界模型不确定性的可信化**：用 ensemble/扩散熵区分"该保守的真不确定"与"幻觉自信"，做选择性安全干预。
3. **（已有研究）想象 horizon 与共形预测的自适应耦合**：根据预测可信度动态裁剪安全规划深度。

### ④ 创新点 idea（统领主线的旗舰 idea）

> **「Imagine-Danger-then-Recover (IDR)：不确定性感知的世界-动作安全栈」** —— 把 5.1–5.6 在单一世界-动作模型中闭环：
> 1. **想象**：世界模型对候选动作做 latent rollout，预测未来失衡/摔倒/碰撞；
> 2. **量化**：用认知不确定性 + 共形预测给每条想象轨迹一个**校准风险值**（解决幻觉问题）；
> 3. **过滤**：WM-CBF / latent HJ 安全滤波（5.4/5.6）在想象空间剔除进入失败集的动作；
> 4. **恢复**：一旦预测式监控（5.5）判定不可避免摔倒，切换到护倒+起身策略（5.1），并把这次真实摔倒回灌世界模型（5.3 LoopSR 式闭环）持续降低未来风险。
>
> 这条栈把"想象-规划-行动-恢复"具象为一个可训练、可验证、带统计保证的人形安全系统，且每个环节都对应已核实代表性工作的自然延伸——既有明确空白（人形 latent 安全滤波、人形 locomotion 失败预测），又有可立即复用的开源骨架（HoST、HumanUP、FAIL-Detect、SAFE）。

## 5.7 分支 E 小结

- 5.1/5.2（起身、推力恢复）真机已成熟但缺语义与预判；
- 5.3（sim2real）范式稳定但缺安全耦合与自检；
- 5.4（安全 RL/CBF）理论成熟但**人形恢复场景几乎空白**；
- 5.5（失败检测）操作领域爆发但**未触及人形 locomotion**；
- 5.6（世界模型安全）是最新且最贴主线的前沿，人形落地是清晰蓝海。

**最具差异化的切入点**：把"世界模型预测失败 + latent 安全滤波"首次系统性引入**人形全身恢复**（即 IDR 旗舰 idea）。

---

# 第 6 章　分支 F：数据引擎、仿真平台与评测基准

> 验证全链路的基础设施。世界动作模型要实现"想象-规划-行动-恢复"，必须依赖三类基础设施——**数据**（喂养感知与动作先验）、**仿真**（提供可交互、可重置、可大规模并行的"想象沙盒"）、**评测**（验证想象是否物理可信、动作是否真实可迁移）。
>
> 按你的要求：**数据部分（6.1）刻意压缩**（约占本章 1/5），重心放在仿真（6.2）与评测（6.3–6.5），并贯穿一条主线创新——**"世界模型同时充当数据引擎、仿真器与评测器"**。

## 6.1（精简）数据来源：从遥操作到跨本体

### ① 已有进展

| 数据集 | 机构 | venue/年份 | arXiv | 开源 | 一句话贡献 |
|---|---|---|---|---|---|
| **Open X-Embodiment / RT-X** | 60+ 机构联盟 | ICRA 2024 | 2310.08864 | 是 | 聚合 60+ 数据集的最大开源真机数据集，证明跨本体正迁移 |
| **DROID** | Stanford/Berkeley 等 | RSS 2024 | 2403.12945 | 是 | 76k 轨迹、564 场景的"in-the-wild"遥操作数据集 |
| **AgiBot World / GO-1** | 智元 / OpenDriveLab | 2025 | 2503.06669 | 是 (GitHub OpenDriveLab/AgiBot-World) | 100+ 真机、100万+轨迹、2976 小时平台 + GO-1 通用策略 |
| **BridgeData V2** | UC Berkeley | CoRL 2023 | 2308.12952 | 是 | 6万轨迹，跨任务泛化基线 |

其余供给线：人类视频（预训练视觉表征/动作先验）、动作重定向（human→humanoid，是人形数据的主要放大手段）、仿真合成数据（见 6.2）。

### ② 痛点

- **OXE 高度不均衡**：跨 60+ 数据集分布极不平衡，朴素混合训练收益受限；数据配比比数据量更关键。
- **人形全身数据稀缺**：遥操作全身演示成本高、retargeting 引入动力学失真。
- **真仿域差与标注异构**：动作空间、本体形态、坐标系不统一，跨本体迁移靠暴力对齐。

### ③ 可做项目

1. **世界模型驱动的数据增广器**：用视频世界模型对现有 OXE/DROID 轨迹做"反事实增广"（改变光照、物体、视角、干扰物），针对性补齐长尾。
2. **跨本体潜动作（latent action）对齐库**：复现扩展 GO-1 的 latent action，建把人类视频、单臂、人形统一到同一潜动作空间的开源工具。
3. **人形 retargeting 质量评测集**：构建带物理一致性标签的 human→humanoid 重定向数据，量化失真。

### ④ 创新点 idea

> **「想象式数据飞轮」**：让世界动作模型在 latent 空间想象大量"未见但物理可信"的交互轨迹，用 6.5 的物理合理性评分器（Physics-IQ/WorldModelBench 思路）在线过滤，只保留高物理可信度样本回灌策略训练。把"数据采集"转化为"世界模型采样 + 物理评分筛选"。

## 6.2 仿真平台与神经/可微仿真

### ① 已有进展

| 平台 | 机构 | 年份 | 链接 | 开源 | 核心贡献 |
|---|---|---|---|---|---|
| **Isaac Lab** | NVIDIA | 2023→ | 前身 Orbit arXiv:2301.04195；GitHub isaac-sim/IsaacLab | 是 | GPU 并行物理 + 光真实渲染 + 模块化学习框架，事实标准 |
| **MuJoCo / MJX** | DeepMind | 持续 | mujoco.org | 是 | 接触动力学精度高；MJX 提供 JAX 端 GPU 并行 |
| **Genesis** | CMU 等联盟 | 2024.12 | GitHub Genesis-Embodied-AI/Genesis | 是 | 通用物理引擎（多材料/多求解器）+ 生成式数据引擎（极端加速比为营销口径，**待核实**） |
| **ManiSkill3** | UCSD (Hao Su 组) | 2024 | arXiv:2410.00425，GitHub haosulab/ManiSkill | 是 | GPU 并行物理+渲染，宣称 30,000+ FPS |
| **RoboCasa** | UT Austin/NVIDIA (Yuke Zhu 组) | RSS 2024 | arXiv:2406.02523，robocasa.ai | 是 | 厨房等日常场景大规模仿真，生成式合成数据呈清晰 scaling |
| **BEHAVIOR-1K** | Stanford (基于 OmniGibson) | 2024 | arXiv:2403.09227，behavior.stanford.edu | 是 | 基于真实人类需求调研的 1000 项日常活动，长程移动操作 |

**神经/可微仿真（World-Action Model 最相关）**
- **NeRD (Neural Robot Dynamics)** — NVIDIA，2025，arXiv:2508.15755。用神经网络**替换解析仿真器中的低层动力学与接触求解器**，机器人中心、空间不变表示，可嵌入 Newton 引擎——"可学习的世界模型当仿真器用"的代表。

### ② 痛点

- **接触/形变/流体仍是软肋**：刚体接触求解的真仿差是 sim-to-real 主因；柔体、颗粒、布料昂贵。
- **吞吐宣传 vs 可复现**：Genesis 等的极端加速比缺乏中立第三方基准，**待核实**。
- **神经仿真的长程漂移与泛化**：NeRD 类方法在分布外接触模式下稳定性未知。
- **生成式场景的物理一致性**："看起来对"不等于动力学自洽。

### ③ 可做项目

1. **中立仿真吞吐与保真度基准**：在统一硬件、统一任务（如人形行走+抓取）上横测 Isaac Lab / MJX / Genesis / ManiSkill3 的 FPS、接触精度、sim-to-real 迁移率。
2. **神经仿真器即世界模型 plug-in**：把 NeRD 接进 Isaac Lab，做"解析+神经"混合 rollout，比较纯解析 vs 混合在人形 loco-manipulation 上的训练效率。
3. **可微接触 + 恢复行为**：用可微仿真训练"跌倒-恢复"策略。

### ④ 创新点 idea

> **分层仿真器：解析引擎管粗粒度刚体动力学、神经世界模型管细粒度接触与视觉**。世界动作模型在"想象"阶段调用神经世界模型快速 rollout 候选动作序列，只把 top-k 候选交给解析引擎做物理校验。把 UniSim/IRASim 的"视频级世界模型"与 NeRD 的"状态级神经动力学"缝合成统一可规划仿真器。

## 6.3 人形/操作评测基准

### ① 已有进展

| 基准 | 机构 | venue/年份 | arXiv | 开源 | 贡献 |
|---|---|---|---|---|---|
| **HumanoidBench** | UC Berkeley (Sferrazza, Abbeel 等) | RSS 2024 | 2403.10506 | 是 | 首个高维**人形全身**仿真基准，H1+灵巧手，27 项 loco+manipulation 任务 |
| **LIBERO** | UT Austin (Yuke Zhu 等) | NeurIPS 2023 | 2306.03310 | 是 | 终身学习/知识迁移操作基准，130 任务，已成 VLA 标准测试集 |
| **CALVIN** | Univ. Freiburg | RA-L 2022 | 2112.03227 | 是 | 长程语言条件操作基准，ABCD 划分测组合泛化 |
| **Meta-World** | Stanford 等 | CoRL 2019 | 1910.10897 | 是 | 50 任务多任务/元 RL 经典基准 |
| **robomimic** | Stanford (Mandlekar 等) | CoRL 2021 | 2108.03298 | 是 | 离线人类演示模仿学习的系统性研究与标准套件 |

### ② 痛点

- **人形评测碎片化**：HumanoidBench 偏 RL/控制，缺少与 VLA/世界模型对接的语言条件、长程任务标准；各全身工作各用各的任务集，不可比。
- **稀疏奖励诱导不真实行为**：HumanoidBench 稀疏奖励会产出不自然步态，需正则化。
- **操作基准饱和**：LIBERO/CALVIN 成功率已高，对扰动/改写指令脆弱，鲁棒性虚高。

### ③ 可做项目

1. **Humanoid-WorldBench（统一人形评测套件）**：在 Isaac Lab/MJX 上整合 loco + manipulation + 语言条件 + 恢复（跌倒/扰动）四类任务，统一指标，桥接 RL 与 VLA/世界模型两个社区。
2. **鲁棒性压力测试层**：给 LIBERO/CALVIN 加程序化扰动（光照、干扰物、指令改写），出 robustness leaderboard。
3. **长程"想象-规划"评测**：专门考核策略是否能在执行前用世界模型预演并纠错。

### ④ 创新点 idea

> 设计**"恢复力（recoverability）"指标**：在任务中途注入扰动（推、滑、抓空），用世界模型预测的"想象恢复轨迹"与真实恢复结果对比，量化世界动作模型的"恢复"能力——这是现有人形基准完全缺失的维度，直击主线。

## 6.4 Sim-to-real 评测与真机评测协议

### ① 已有进展

| 工作 | 机构 | venue/年份 | arXiv | 开源 | 贡献 |
|---|---|---|---|---|---|
| **SIMPLER** | Google DeepMind/Berkeley 等 | CoRL 2024 | 2405.05941 | 是 (simpler-env) | 配对 sim-real 评测，证明仿真成功率与真机强相关 |
| **AutoEval** | UC Berkeley | 2025 | 2503.24278 | 项目页待核实 | 自主真机评测系统，最小人工干预 7×24 评测 |
| **RoboArena** | 多机构（7+ 院校） | CoRL 2025 | 2506.18123 | 是 (项目页待核实) | **分布式真机评测**：跨机构众包 + 相对偏好排名 |

> 另：SplatSim（arXiv:2409.10161）用 3D Gaussian Splatting 做高视觉保真 real-to-sim。⚠️ 极新预印本待核实：Imperfect Simulators 评测（arXiv:2510.04354，用 prediction-powered inference 融合少量真实 + 大量仿真给无偏估计）。

### ② 痛点

- **真机评测不可复现/不可比**：不同实验室硬件、布置、人工判定差异巨大——RoboArena 正为此而生，但偏好众包仍引入主观性。
- **统计严谨性不足**：真机试验次数少，置信区间宽。
- **sim 代理的"过拟合代理"风险**：策略可能专门"刷"SIMPLER 而非真机能力。

### ③ 可做项目

1. **PPI 评测协议落地**：在自有机器人上复现 prediction-powered inference，给出"N 次真机 + M 次仿真"的最优配比公式与开源评测脚本。
2. **3DGS 数字孪生评测管线**：把真实工位扫成 3DGS 场景，配 ManiSkill3/Isaac Lab 物理，做 real-to-sim-to-real 闭环评测。
3. **接入 RoboArena 的人形分赛道**：把分布式协议从单臂扩展到人形。

### ④ 创新点 idea

> **世界模型作为"评测仿真器"**：用条件视频世界模型（给定初始帧 + 策略动作序列）预测 rollout，结合物理评分器自动判定成败，构成"无需真机、无需物理引擎"的第三类评测。关键创新是用 PPI 把世界模型评测结果与少量真机锚定校准，既便宜又无偏——把 6.5 的世界模型评测与 6.4 的真机协议打通。

## 6.5 世界模型本身的评测基准（与主线最直接相关）

### ① 已有进展

**(a) 视频生成通用质量**：VBench（南洋理工，CVPR 2024，arXiv:2311.17982，**开源**）把视频生成质量分解为多维度（时序一致性、运动平滑、主体一致等）。

**(b) 物理合理性（最关键）**
- **Physics-IQ** — Google DeepMind 等，2025，arXiv:2501.09038（项目页 physics-iq.github.io，**开源**）。结论：当前视频模型有视觉真实感但缺物理理解。
- **WorldModelBench: Judging Video Generation Models As World Models** — 2025，NeurIPS 2025，arXiv:2502.20694。加入指令遵循 + 物理遵循维度，覆盖 7 域。
- **WorldSimBench** — 2024，arXiv:2410.18072，ICML 2025。双轨评测：显式感知评测（人类偏好）+ **隐式操控评测**（生成视频能否转成正确动作）。

**(c) 世界模型即交互仿真器**
- **UniSim** — Berkeley/Google 等，**ICLR 2024 Outstanding Paper**，arXiv:2310.06114。
- **IRASim** — 2024，arXiv:2406.14540。交互式真机动作仿真器，输入动作轨迹"控制"初始帧里的机器人。

### ② 痛点

- **物理合理性评分仍依赖代理指标**：Physics-IQ 用预测帧与真实帧对比，但"物理对"≠"像素对"，缺可解释的因果物理分解。
- **动作可执行性评测稀缺**：只有 WorldSimBench 把"生成视频→动作"闭环，但动作落地仍依赖额外 IDM，评测链条长。
- **长程一致性与累积漂移**：长 rollout 下物体永久性、守恒律崩溃，现有 benchmark 多为短片段。
- **交互式世界模型基准割裂**：Genie/Matrix-Game 等各用私有场景，不可比。

### ③ 可做项目

1. **Action-Conditioned Physical Plausibility Bench**：构建给定机器人动作序列、考核世界模型预测的"动作-物理一致性"基准，融合 WorldSimBench 隐式评测 + Physics-IQ 物理分解，专门面向人形。
2. **长程漂移诊断套件**：标准化测物体永久性、接触守恒、能量守恒随 rollout 长度的衰减曲线。
3. **统一交互式世界模型 leaderboard**：建公共场景+轨迹，横测可交互世界模型。

### ④ 创新点 idea

> **「可执行物理分（Executable Physics Score）」**：不只看生成视频像不像，而是把世界模型 rollout 喂给一个固定的逆动力学+物理引擎校验器，测"想象出的未来是否能被真实动作复现"。把 6.5 的世界模型评测、6.2 的神经仿真、6.4 的 sim-to-real 三者统一到同一指标下，是"想象-规划-行动-恢复"主线最自然的评测落点。

## 6.6 分支 F 小结

贯穿数据/仿真/评测的一条创新主线是 **"世界模型同时充当数据引擎、仿真器与评测器"**：
- **生成数据**（6.1 想象式数据飞轮）→ **充当仿真器**（6.2 分层解析+神经仿真、UniSim/IRASim）→ **充当评测器**（6.4 世界模型评测仿真器 + 6.5 可执行物理分）。
- 三者用同一套**物理合理性评分**（Physics-IQ/WorldModelBench 思路）+ **PPI 真机锚定**串起来，形成低成本、可复现、面向人形的闭环验证基础设施。

---

# 第 7 章　跨分支综合：旗舰路线图、优先级与选题建议

> 前六章是"分支地图"，本章是"行军路线"。这里把六个分支收拢成 1 条主线判断、3 个旗舰项目、1 张优先级矩阵，并给出我对你选题的具体建议。

## 7.1 一个核心判断：缺的不是"动作"，是"恢复"

把六章的"未解决痛点"叠在一起看，会发现它们指向同一个空洞——

> **整个领域在"想象（A/F）、规划（C）、行动（B/D）"上都已百花齐放，唯独"恢复（Recover）"维度系统性缺失。**

- 分支 A 的世界模型几乎只做"正常未来"预测，不预测"失败分支"；
- 分支 C 的 VLA/规划绝大多数是开环，失败后不重规划；
- 分支 D 的触觉/力感知止步"预测更准"，不闭环到"预测失配→恢复"；
- 分支 E 的失败检测在机械臂上爆发，却**没人做人形 locomotion 的预测式摔倒检测**，更没人把"检测→护倒→起身"接成闭环；
- 分支 B 的起身/推力恢复真机已成熟，但是**被动 reactive**，没有"预判式"恢复。

这意味着：**"世界模型驱动的预测式失败检测 + 主动恢复"是当前全领域相对最空、最契合主线标题、最容易做出差异化原创成果的方向。** 它也最能体现"世界模型 ⊗ 动作模型"的融合价值——因为只有当机器人能"想象失败"，恢复才能从被动变主动。

## 7.2 三个旗舰项目（按主线"想象-规划-行动-恢复"组织）

下面三个项目分别对应**控制层、感知层、安全层**，各自独立可发表，又能拼成一个完整系统。每个都标注了可复用的开源基座，降低落地门槛。

### 旗舰一 ｜ HWAM：人形世界-动作统一模型（控制层，分支 A+B+C 交叉）

- **一句话**：一个共享 latent，双头（世界模型头预测未来 latent+稳定性+接触力；动作头继承 GMT/BeyondMimic 动作基座），在 latent 空间做想象 rollout 选动作，预测失稳即触发恢复头。
- **解决的痛点**：世界模型与动作基座长期是两套系统（B.6）、想象规划不实时（A.4）、恢复行为硬编码（C.4）。
- **开源基座**：BeyondMimic / ASAP / TWIST（控制）+ DreamerV3 / TD-MPC2（latent WM）+ HoST / HumanUP（恢复头）。
- **第一步实验**：在 ASAP 的 delta-action 框架上，用循环世界模型替代手工域随机化，做"想象式 sim-to-real"对照。

### 旗舰二 ｜ Tactile Dream-and-Recover：触觉预测失配驱动的恢复（感知层，分支 D+E 交叉）

- **一句话**：世界模型联合预测（视觉+触觉+力+滑移裕度）的未来；定义"触觉惊奇度"（预测触觉与真实触觉的 latent 距离）为在线信号，超阈值即从当前真实状态重新想象-规划，实现毫秒级恢复。
- **解决的痛点**：触觉表征停留在"判别/对齐"而非"预测"（D.4）、富接触操作没有恢复触发器（E）。
- **开源基座**：Sparsh / Sparsh-X（触觉表征）+ iDP3 / Humanoid-Teleoperation（人形 3D 策略与遥操数据）+ RoboPack（触觉动力学）。
- **第一步实验**：在 Sparsh 编码上加一个未来触觉预测头，在插拔/拧瓶盖任务上测"触觉惊奇度"能否提前 N 步预报打滑。

### 旗舰三 ｜ IDR：不确定性感知的世界-动作安全栈（安全层，分支 E+A 交叉）

- **一句话**：世界模型对候选动作做 latent rollout → 用认知不确定性+共形预测给每条想象轨迹校准风险值 → WM-CBF / latent HJ 安全滤波在想象空间剔除危险动作 → 不可避免摔倒时切换护倒+起身，并把真实摔倒回灌世界模型（LoopSR 式闭环）。
- **解决的痛点**：latent 安全滤波从未用于人形全身恢复（E.6）、世界模型不确定性会"自信地幻觉安全未来"（E.6）、人形 locomotion 失败预测空白（E.5）。
- **开源基座**：HoST / HumanUP（恢复）+ FAIL-Detect / SAFE（失败检测）+ latent reachability 系列（安全滤波）。
- **第一步实验**：把 FAIL-Detect 的"无失败数据 + 共形预测"范式从机械臂迁移到人形步态，做"N 步后摔倒概率"预测器（这是明确空白，单独即可成文）。

> 三者关系：**旗舰一是骨架（统一模型），旗舰二补感知（让想象覆盖触觉），旗舰三补安全（让恢复可验证）。** 可任选其一切入，也可按"旗舰三 → 旗舰一 → 旗舰二"的难度递增顺序推进——旗舰三的"人形预测式摔倒检测"门槛最低、空白最明确，适合作为第一篇。

## 7.3 优先级矩阵（创新性 × 可行性 × 主线契合度）

> 评分为我的主观判断（5 分制，分越高越好）。"可行性"综合考虑开源基座成熟度与算力门槛。

| 候选项目 | 所属分支 | 创新性 | 可行性 | 主线契合 | 综合建议 |
|---|---|:---:|:---:|:---:|---|
| 人形 locomotion 预测式摔倒检测 | E.5 | 4 | **5** | 5 | ⭐ **首选切入**：空白明确、开源基座全、单点可成文 |
| Tactile Dream-and-Recover | D.4+E | **5** | 4 | 5 | ⭐ **最具原创性**：触觉"预测+恢复"几乎无人做 |
| HWAM 统一世界-动作模型 | A+B+C | **5** | 3 | 5 | 旗舰但工程量大，适合作为长期主攻/博士主线 |
| Imagined Recovery Tree（起身可行性想象） | E.1+A | 4 | 4 | 5 | 在 HoST/HumanUP 上加 WM rollout，增量清晰 |
| WM-CBF（latent 安全屏障） | E.4+A | **5** | 3 | 5 | 理论漂亮，难点在 latent 失败集的学习与验证 |
| 想象-门控双系统（GR00T 上加 latent dynamics） | C.2 | 4 | 4 | 5 | 开源 GR00T N1.5 直接可改，工程友好 |
| Executable Neural Trajectories（数据引擎过滤器） | A.3+F | 4 | **5** | 4 | 蹭 GR00T-Dreams 开源，快速出结果 |
| HumanoidWM-Bench（评测基准） | A.5+F | 3 | 4 | 4 | 偏基础设施，引用价值高、新颖性中等 |
| Contact-Mode RSSM（接触隐变量 Dreamer） | A.2 | 4 | 3 | 4 | 理论新颖，需较强 MBRL 工程能力 |
| Anticipatory Push Recovery（视觉预判推力恢复） | E.2 | 3 | 4 | 4 | 增量明确，真机验证有说服力 |

**读法**：若你要**快速出一篇高质量论文**，从矩阵顶部两行选（E.5 预测式摔倒检测 / D.4 触觉恢复）；若你要**立一个能撑 2-3 年的主攻方向**，选 HWAM（A+B+C）并以 E.5 作为它的第一个可发表里程碑。

## 7.4 给你的具体建议

1. **主线一句话定位**：把课题命名为「**面向人形机器人的世界-动作模型：以"想象式失败预测与主动恢复"为核心**」。这样既扣住"世界动作模型"的大主线，又锁定了全领域最空的"恢复"维度，差异化清晰。

2. **第一篇怎么选**：建议从 **7.2 旗舰三的第一步实验**起步——把 FAIL-Detect/SAFE 的"无失败数据 + 共形预测"范式迁移到**人形 locomotion 的预测式摔倒检测**。理由：①明确空白（现有失败检测全在机械臂）；②开源基座齐全（FAIL-Detect 开源 + HoST/HumanUP 提供人形仿真与恢复）；③单点即可成文，且天然是后续 IDR/HWAM 的组件。

3. **如何体现"世界模型"而非普通分类器**：关键差异是让检测器**基于世界模型的前向 latent rollout** 给出"N 步后摔倒概率"（因果预测），而非对当前帧做判别（事后检测）。这一点是审稿人区分"又一个失败检测器"与"世界模型驱动的预测式安全"的分水岭。

4. **务必先做的一件事**：本报告标注 ⚠️ 的 `2512/2601–2605` 条目（尤其 D.4 触觉世界模型、E.6 latent 安全滤波这两批最贴你主线的）请**逐篇 WebFetch 核实 arXiv abstract 与 GitHub README**——这批新预印本既是你最大的"竞品/前置工作"，也是编号最可能有偏差的部分。详见附录 B。

---

# 附录 A：全报告论文速查表

> 按分支组织。"开源"列：✅=已确认开源/有权重；🔗=有项目页；❓=未确认；闭源标注。arXiv 编号带 ⚠️ 者见附录 B。

### A 世界模型 / 神经仿真器
| 工作 | venue/年份 | arXiv | 开源 |
|---|---|---|---|
| Genie | ICML 2024 (Best Paper) | 2402.15391 | 闭源(社区复现) |
| Genie 2 / Genie 3 | DeepMind 博客 2024/2025 | — | 闭源 |
| Oasis | 2024 | — | ✅ oasis-model.github.io |
| GameNGen | 2024 | 2408.14837 | ❓ |
| IRASim | 2024 | 2406.14540 | ✅ |
| WorldVLA | 2025 | 2506.21539 | ✅ HF Alibaba-DAMO-Academy/WorldVLA |
| DreamerV3 | Nature 2025 | 2301.04104 | ✅ danijar/dreamerv3 |
| TD-MPC2 | ICLR 2024 | 2310.16828 | ✅ |
| NVIDIA Cosmos | 2025 | 2501.03575 | ✅ NVIDIA/Cosmos |
| DreamGen / GR00T-Dreams | CoRL 2025 | 2505.12705 | ✅ NVIDIA/GR00T-Dreams |
| Humanoid World Models | 2025 | 2506.01182 | ✅ |
| 1X World Model (Challenge) | 2024/2025 | 2510.07092 | 🔗 1x.tech |
| UniSim | ICLR 2024 (Outstanding) | 2310.06114 | 🔗 |
| GR-1 / GR-2 | ICLR 2024 / 2024 | 2312.13139 / 2410.06158 | 🔗 |
| iVideoGPT | NeurIPS 2024 | 2405.15223 | ✅ |
| Navigation World Models | CVPR 2025 | 2412.03572 | 🔗 HF facebook/nwm |
| WHAM/Muse | Nature 2025 | s41586-025-08600-3 | ✅ HF microsoft/wham |
| Pandora | 2024 | 2406.09455 | ❓ |
| RynnVLA-002 | 2025 | 2511.17502 | ❓ |

### B 人形全身控制
| 工作 | venue/年份 | arXiv | 开源 |
|---|---|---|---|
| Real-World Humanoid Locomotion (RL) | 2023 | 2303.03381 | 🔗 |
| Berkeley Humanoid / Lite | 2024 / 2025 | 2407.21781 / 2504.17249 | 🔗 |
| H2O / OmniH2O | 2024 | 2403.04436 / 2406.08858 | ✅ |
| HumanPlus | CoRL 2024 | 2406.10454 | ✅ MarkFzp/humanplus |
| ExBody / ExBody2 | 2024 | 2402.16796 / 2412.13196 | ✅ |
| TWIST / TWIST2 | 2025 | 2505.02833 / 2511.02832 | ✅ YanjieZe/TWIST |
| VideoMimic | 2025 | 2505.03729 | ✅ hongsukchoi/VideoMimic |
| GMR / GMT | 2025 | 2510.02252 / 2506.14770 | 🔗 |
| FALCON | 2025 | 2505.06776 | ❓ |
| HOVER | 2024 | 2410.21229 | 🔗 |
| OmniRetarget | 2025 | 2509.26633 | 🔗 |
| ResMimic | 2025 | 2510.05070 | 🔗 |
| VisualMimic | 2025 | 2509.20322 | ✅ visualmimic/VisualMimic |
| HumanoidVerse | 2025 | 2508.16943 | ❓ |
| ASAP | RSS 2025 | 2502.01143 | ✅ LeCAR-Lab/ASAP |
| Humanoid Parkour Learning | 2024 | 2406.10759 | 🔗 |
| WoCoCo | 2024 | 2406.06005 | 🔗 |
| BeyondMimic | 2025 | 2508.08241 | ✅ HybridRobotics/whole_body_tracking |
| R2S2 | 2025 | 2505.10918 | ❓ |
| Supersizing Motion Tracking | 2025 | 2511.07820 | ❓ |
| MuJoCo Playground | 2025 | 2502.08844 | ✅ |
| WMR | 2025 | 2502.16230 | ❓ |
| DWL (Denoising World Model) | 2024 | 2408.14472 | ❓ |

### C VLA / 长时序规划
| 工作 | venue/年份 | arXiv | 开源 |
|---|---|---|---|
| RT-1 / RT-2 | 2022 / 2023 | 2212.06817 / 2307.15818 | ✅ / 闭源 |
| Octo | RSS 2024 | 2405.12213 | ✅ octo-models/octo |
| OpenVLA | CoRL 2024 | 2406.09246 | ✅ openvla/openvla |
| π0 / π0-FAST | 2024 / 2025 | 2410.24164 / 2501.09747 | 部分(openpi) / ✅ HF |
| RoboVLM | 2024 | 2412.14058 | ✅ |
| SpatialVLA | 2025 | 2501.15830 | ✅ |
| CogACT | 2024 | 2411.19650 | ✅ |
| GR00T N1 / N1.5 | 2025 | 2503.14734 | ✅ HF nvidia/GR00T-N1.5-3B |
| Helix | Figure AI 2025 | — | 闭源 |
| Gemini Robotics / 1.5 | DeepMind 2025 | 2503.20020 / 2510.03342 | 闭源(API) |
| Hi Robot | 2025 | 2502.19417 | 🔗 |
| RoboBrain / 2.0 | CVPR 2025 / 2025 | 2502.21257 / 2507.02029 | ✅ |
| AgiBot World / GO-1 | 2025 | 2503.06669 | ✅ |
| WorldVLA | 2025 | 2506.21539 | ✅ |
| Genie Envisioner | 2025 | 2508.05635 | 🔗 |
| Chain-of-Action | 2025 | 2506.09990 | ✅ ByteDance-Seed/Chain-of-Action |
| Diffusion Policy | RSS 2023 | 2303.04137 | ✅ real-stanford/diffusion_policy |
| RDT-1B | ICLR 2025 | 2410.07864 | ✅ thu-ml/RoboticsDiffusionTransformer |
| ECoT | 2024 | 2407.08693 | ✅ embodied-cot.github.io |
| MolmoAct | 2025 | 2508.07917 | ✅(Ai2) |
| ChatVLA | EMNLP 2025 | 2502.14420 | ❓ |
| Emma-X | 2024 | 2412.11974 | ❓ |

### D 多模态感知
| 工作 | venue/年份 | arXiv | 开源 |
|---|---|---|---|
| GelSight | Sensors 2017 | — | 🔗 |
| DIGIT / DIGIT 360 | RA-L 2020 / 2024 | 2005.14679 | ✅ facebookresearch/digit-* |
| AnySkin | 2024 | 2409.08276 | 🔗 |
| eFlesh | 2025 | 2506.09994 | ❓ |
| T3 (Transferable Tactile Transformers) | 2024 | 2406.13640 | 🔗 |
| UniTouch | 2024 | 2401.18084 | 🔗 |
| AnyTouch | ICLR 2025 | 2502.12191 | ❓ |
| SITR | ICLR 2025 | 2502.19638 | ✅ hgupt3/gsrl |
| FoAR | 2024 | 2411.15753 | 🔗 |
| Just Add Force | 2024 | 2410.13124 | ❓ |
| 3D-ViTac | CoRL 2024 | 2410.24091 | 🔗 binghao-huang.github.io/3D-ViTac |
| NeuralFeels | Science Robotics 2024 | 2312.13469 | 🔗 |
| Sparsh-skin | 2025 | 2505.11420 | ❓ |
| DexUMI | 2025 | 2505.21864 | ❓ |
| RoboPack | RSS 2024 | 2407.01418 | 🔗 |
| Multi-Modal World Model (Physical Robot) | 2023 | 2304.11193 | ❓ |
| DP3 | RSS 2024 | 2403.03954 | ✅ YanjieZe/3D-Diffusion-Policy |
| iDP3 | IROS 2025 | 2410.10803 | ✅ YanjieZe/Improved-3D-Diffusion-Policy |
| M2T2 | CoRL 2023 | 2311.00926 | 🔗 |
| Sparsh / Sparsh-X | CoRL 2024 / 2025 | 2410.24090 / 2506.14754 | ✅ facebookresearch/sparsh |
| FuSe | 2025 | 2501.04693 | 🔗 |
| Tactile-VLA | 2025 | 2507.09160 | ❓ |

### E 鲁棒恢复与安全
| 工作 | venue/年份 | arXiv | 开源 |
|---|---|---|---|
| HoST | RSS 2025 (Best Sys Finalist) | 2502.08378 | ✅ InternRobotics/HoST |
| HumanUP | RSS 2025 | 2502.12152 | 🔗 humanoid-getup.github.io |
| FRASA | 2024 | 2410.08655 | ❓ |
| Reactive Stepping (RL) | 2022 | 2203.01148 | ❓ |
| Push Recovery w/ Reduced Order Models | 2025 | 2505.11495 | ❓ |
| RMA / A-RMA | 2021 / 2022 | 2107.04034 / 2205.15299 | 🔗 ashish-kmr.github.io/rma-legged-robots |
| LoopSR | 2024 | 2409.17992 | ❓ |
| CPO | ICML 2017 | ⚠️编号待核实 | (OmniSafe 实现) |
| PID Lagrangian | ICML 2020 | PMLR v119 | ✅(库) |
| C3BF (Safe Legged Locomotion) | 2023 | 2309.01898 | ❓ |
| FAIL-Detect | RSS 2025 | 2503.08558 | ✅ CXU-TRI/FAIL-Detect |
| SAFE (VLA failure detection) | NeurIPS 2025 | 2506.09937 | 🔗 vla-safe.github.io |
| Sentinel | 2024 | 2410.04640 | ❓ |
| Aha | 2024 | 2410.00371 | ✅ |
| Latent-Space Reachability (beyond collision) | 2025 | 2502.00935 | ❓ |

### F 数据 / 仿真 / 评测
| 工作 | venue/年份 | arXiv | 开源 |
|---|---|---|---|
| Open X-Embodiment | ICRA 2024 | 2310.08864 | ✅ |
| DROID | RSS 2024 | 2403.12945 | ✅ |
| AgiBot World | 2025 | 2503.06669 | ✅ |
| Isaac Lab (Orbit) | 2023 | 2301.04195 | ✅ isaac-sim/IsaacLab |
| ManiSkill3 | 2024 | 2410.00425 | ✅ haosulab/ManiSkill |
| RoboCasa | RSS 2024 | 2406.02523 | ✅ robocasa.ai |
| BEHAVIOR-1K | 2024 | 2403.09227 | ✅ |
| Genesis | 2024 | — | ✅ (加速比待核实) |
| NeRD | 2025 | 2508.15755 | ❓ |
| HumanoidBench | RSS 2024 | 2403.10506 | ✅ |
| LIBERO | NeurIPS 2023 | 2306.03310 | ✅ |
| CALVIN | RA-L 2022 | 2112.03227 | ✅ |
| Meta-World | CoRL 2019 | 1910.10897 | ✅ |
| robomimic | CoRL 2021 | 2108.03298 | ✅ |
| SIMPLER | CoRL 2024 | 2405.05941 | ✅ |
| AutoEval | 2025 | 2503.24278 | 🔗 |
| RoboArena | CoRL 2025 | 2506.18123 | 🔗 |
| VBench | CVPR 2024 | 2311.17982 | ✅ |
| Physics-IQ | 2025 | 2501.09038 | ✅ |
| WorldModelBench | NeurIPS 2025 | 2502.20694 | 🔗 |
| WorldSimBench | ICML 2025 | 2410.18072 | 🔗 |

---

# 附录 B：待核实清单（引用前务必逐篇确认）

调研在受限网络环境下完成，arXiv 正文抓取被拦截，标题/编号以多源检索快照（项目页、HuggingFace、GitHub、二级引用）交叉核对。以下三类条目**请在正式写入论文前逐篇 WebFetch 核实 arXiv abstract + GitHub README**，确认编号、作者、机构、是否真实存在、开源状态：

### B.1 高风险（2025-12 至 2026-05 极新预印本，编号最可能有偏差）

> 这批同时也是**最贴近你主线、最可能成为竞品/前置工作**的条目，核实优先级最高。

| 条目 | 报告中给的 arXiv | 所在章节 | 备注 |
|---|---|---|---|
| Visuo-Tactile World Models | 2602.06001 | D.4 | 触觉世界模型，称物体永久性 +33%——若属实是你旗舰二的直接竞品 |
| Contact-Aware Neural Dynamics | 2601.12796 | D.4 | 接触非光滑动力学建模 |
| DreamTacVLA | 2512.23864 | D.4 | "梦境"+触觉 VLA |
| Video-Tactile-Action Models | 2603.23481 | D.4 | 视频+触觉隐式世界动力学 |
| UNISafe (latent safety filter) | 2505.00779 | E.6 | 不确定性感知 latent 安全滤波——你旗舰三的直接竞品 |
| Designing Latent Safety Filters w/ Pretrained Vision | 2509.14758 | E.6 | latent 失败集分类器 |
| FIPER (failure prediction at runtime) | 2510.09459 | E.5 | 运行时失败预测，RND+共形 |
| Safety Filtering RL in Training w/ CBFs | 2510.14959 | E.4 | 训练期安全滤波 |
| A Survey of Safe RL and Constrained MDPs | 2505.17342 | E.4 | 综述 |
| Unified Humanoid Fall-Safety Policy | 2511.07407 | E.1 | 摔倒预防+冲击缓解+恢复一体 |
| HAIC (dynamics-aware world model) | 2602.11758 | B.6 | 世界模型补全物体观测 |
| UMI-FT | 2601.09988 | D.2 | 野外柔顺操作+力 |
| DreamZero | 2602.15922 | C.4 | "世界动作模型即零样本策略" |
| MCTS+MPC w/ Visual World Models | 2511.03077 | C.4 | 想象中树搜索 |
| Imperfect Simulators 评测 (PPI) | 2510.04354 | F.4 | prediction-powered inference |

### B.2 中风险（机构署名 / 开源状态 / 最终 venue 未完全确认）

- **ARIO** 数据集（2408.10899）的具体署名机构；
- **Genesis** 物理引擎宣称的"430,000× 实时"加速比——**几乎确定是营销口径**，论文中引用务必改为中性措辞或注明"厂商数据，未经第三方复现"；
- **FALCON / HumanoidVerse / R2S2 / WMR / DWL** 等是否正式开源（报告标 ❓）；
- **HiRT / RTC / HybridVLA / UniVLA / VLMPC** 的开源与最终 venue；
- **CPO**（Achiam et al., ICML 2017）的确切 arXiv 编号（报告未给，凭领域知识应为 1705.10528，**待核实**）；
- 厂商闭源系统（**Helix / Gemini Robotics 系列 / Genie 2/3**）以官方博客/技术报告为准，无 arXiv，引用时注明来源类型。

### B.3 核实方法建议

1. 对每个 ⚠️ 条目，直接 `WebFetch https://arxiv.org/abs/<编号>`，确认标题与报告一致；
2. 若编号对应不上，用标题在 Google Scholar / arxiv-sanity 反查正确编号；
3. 开源状态点开对应 GitHub，确认 README 与 License 真实存在（不只是占位仓库）；
4. 对 B.1 的"竞品"条目，重点读其 method 与 limitation，确认你的创新点与之有实质差异。

---

# 结语

这份报告的组织遵循一条贯穿始终的主线——**世界模型 ⊗ 动作模型 = "想象-规划-行动-恢复"闭环**。六个分支不是平行罗列，而是这条闭环的不同环节：A 是想象引擎、C 是规划大脑、B 是行动底座、D 让想象覆盖触觉、E 补上失败预测与恢复、F 提供验证基础设施。

最值得你投入的判断只有一句：**全领域在"想象/规划/行动"上已经拥挤，唯独"恢复"是蓝海，而恰恰是"恢复"最能证明世界模型的价值——因为只有能想象失败的机器人，才谈得上主动恢复。** 建议以「人形 locomotion 的预测式摔倒检测」作为低门槛、空白明确的第一篇，再逐步搭建 HWAM / Tactile Dream-and-Recover / IDR 三个旗舰。

> 再次提醒：附录 B 的 ⚠️ 条目（尤其 B.1 的触觉世界模型与 latent 安全滤波两批）是你最可能的竞品与编号风险所在，动笔前务必逐篇核实。

---

# 第 8 章　深度扩展：物理交互与长时序的方法级分析 + 具体实施方案

> 本章是对第 1–7 章的深度补充，从"知道有这个方向"升级到"知道怎么做、用什么架构、loss 怎么设计、数据怎么来、评测怎么跑"。重点覆盖五大技术线：
> 1. **物理常识 / 物理知识注入**（直觉物理、守恒律、可微仿真、接触动力学）
> 2. **多模态融合架构**（怎么 token 化、怎么对齐、怎么解决视觉主导）
> 3. **感知信号获取**（传感器选型、信号管线、仿真 sim2real）
> 4. **长时序决策**（分层规划、记忆、世界模型长程推演、纠错重规划）
> 5. **自主进化 / 自我改进**（真机 RL 飞轮、世界模型自改进、从失败学习、持续学习）
>
> 每条技术线给出：**方法核心机制（架构/loss/训练策略）→ method-level 局限 → 空白切入点 → 可直接落地的实施方案雏形**。

---

## 8.1 物理常识与物理知识注入

### 8.1.1 直觉物理（Intuitive Physics）

#### 方法谱系

**VoE 范式（Violation-of-Expectation）**：测"模型对不可能场景是否给更高惊讶度"。本身是评测协议，但也可反向作训练信号。

| 工作 | venue/年份 | arXiv | 方法核心机制 | 局限 |
|---|---|---|---|---|
| **ADEPT** | MIT，NeurIPS 2019 | — | 三段式：深度网络"反渲染"→粗略物体 3D 表示→显式可微物理引擎前向模拟→扩展粒子滤波做贝叶斯滤波，观测偏离物理预测=high surprise。**物理引擎是硬先验，非端到端**。 | 依赖手工物体表示和物理引擎；无法处理流体/可变形体；粗略几何在精细接触失效 |
| **PLATO** | DeepMind，Nature Human Behaviour 2022 | — | 对象中心归纳偏置：先分割物体→Interaction-Network 图式模块用物体过去轨迹+关系预测未来；**VoE 误差作惊奇度**。论点：对象性是直觉物理的关键先验 | 依赖给定物体分割；合成数据训练，真实视频迁移未验证；不含动力学量（质量/摩擦）推断 |
| **Physion++** | 2023 | 2306.15668 ✓ | OCP（Object Contact Prediction）：预测红色物体最终是否接触黄色物体。关键：**答案必须依赖在线推断隐性物理属性**（质量比/摩擦/弹性），同样初始几何因属性不同导致不同结局。逼模型从动态线索做连续属性推断 | 监督式 readout 探针可能高估"理解"；只输出二元接触，不评完整轨迹/力；属性范围有限 |
| **V-JEPA 2** | Meta，2025 | 2506.09985 ✓ | 联合嵌入预测架构（JEPA）：**不在像素空间重建，而在学到的表示空间做掩码预测**。用 context encoder + predictor 预测被掩码块的 target encoder（EMA）输出表示，损失是表示空间 L1/L2。**V-JEPA 2-AC** 额外用 ~62h Droid 数据训动作条件化 latent 动力学预测器，**在 latent 空间做 MPC** 规划（候选动作序列→选使预测 latent 最接近目标的动作）实现 zero-shot manipulation | 直觉物理"涌现"主要在判别任务；latent MPC 对接触丰富/长程任务弱；表示空间预测无显式守恒律，长程漂移 |
| **IntPhys 2** | Meta，2025 | 2506.09849 ✓ | UE5 级别复杂合成 VoE 基准，聚焦四原则（永久性/不变性/时空连续性/固体性）；最小对照对设计 | 仍是合成域；人类近满分而模型近随机，说明任务对当前模型接近天花板 |

**物理合理性诊断基准**（告诉你"世界模型物理卡在哪"）：

| 基准 | arXiv | 关键结论 |
|---|---|---|
| VideoPhy | 2406.03520 ✓ | 最佳模型联合准确率约 32.6%；放大模型无法解决 |
| Physics-IQ | 2501.09038 ✓ | 最佳 Physics-IQ 分仅约 29.5%；**物理理解与视觉真实度不相关** |
| Morpheus | 2504.02918 ✓ | 用守恒律派生的物理信息度量（非像素相似度）；即使条件化也难编码守恒律 |

#### 方法-level 核心痛点

1. **像素重建/扩散目标没有任何约束保证守恒**；时间一致性 ≠ 动力学一致性。
2. Physion++ 的"在线物理属性推断"是最贴机器人需求的评测，但**连续属性推断 + 用于操作**几乎未打通。
3. V-JEPA 涌现的物理表示**能否支撑接触级操作**是明确未解问题。

#### ✅ 可实施方案 A：守恒律辅助损失提升 Latent 世界模型的物理一致性

**问题**：V-JEPA-2-AC 类 latent 世界模型长程预测漂移、违背动量守恒。

**方法架构**：
```
输入: 观测序列 o_{1..t} + 动作 a_t
主干: V-JEPA 式 context encoder (frozen) → z_t
动力学头: MLP/Transformer → ẑ_{t+1}  [主 latent 预测 loss: L2]
物理辅助头: 轻量 readout → (质心位置, 接触事件标志, 动量代理量)
                                  ↓
物理正则 loss:
  L_conservation = ||p̂_{t+1} - p̂_t - Δp_external|| (动量守恒残差)
  L_contact      = BCE(接触预测, 接触真值)  [Physion++ 风格标签]
  L_cal          = disagreement_ensemble与真实rollout误差的校准损失
总 loss = L_latent + λ₁·L_conservation + λ₂·L_contact + λ₃·L_cal
```

**数据**：Physion++（OCP + 物理属性标签）+ Droid（真机接触数据）+ Brax 生成的已知守恒量轨迹作监督。

**评测**：Physion++ OCP 准确率、Morpheus 守恒律指标、真机 reach/grasp 的 k-step latent 预测误差 vs 无守恒正则的基线。

**开源基座**：V-JEPA 2（Meta 开源）、Physion++（开源数据集）、Brax（Google JAX 开源）。

---

### 8.1.2 物理先验注入（守恒律架构 + 可微仿真）

#### Hamiltonian / Lagrangian 神经网络谱系

| 工作 | venue | arXiv | 方法机制（架构/loss 级） | 局限 |
|---|---|---|---|---|
| **HNN** | NeurIPS 2019 | 1906.01563 ✓（开源） | 网络不直接输出 $\dot{q},\dot{p}$，而是输出**标量哈密顿量 H(q,p)**；再用哈密顿方程 $\dot q=\partial H/\partial p,\ \dot p=-\partial H/\partial q$（对网络输出求自动微分）给出动力学；损失是 $(\dot q,\dot p)$ 与真值的 MSE。能量天然近似守恒 | 只适用保守/无耗散系统；需正则坐标 $(q,p)$；接触/摩擦严重违反 |
| **LNN** | ICLR 2020 workshop | 2003.04630 ⚠️待核实 | 学标量拉格朗日量 $L(q,\dot q)$，通过欧拉-拉格朗日方程反解 $\ddot q$；可用任意广义坐标，不需正则动量，更贴机器人关节空间 | Hessian 求逆数值不稳；仍假设保守系统 |
| **HNN+接触扩展** | 2021 | 2102.06794 ✓ | 在 LNN/HNN 的连续动力学之外加**可微接触模型**（基于 LCP/互补约束的松弛），同时学系统物理属性和接触属性，建模含碰撞/stiction 的混合动力学 | 接触模式数量随接触点组合爆炸；LCP 松弛在硬接触极限不准 |
| **Dynami-CAL GraphNet** | 2025 | 2501.07373 ✓ | GNN 的**边局部参考系**上构造消息，使每条边对相邻节点施加的冲量等大反向，**逐对精确守恒线/角动量**（架构硬保证，非软损失）；对旋转等变、对平移不变 | 守恒只对"封闭交互"成立；外力（重力/驱动/地面接触）需特殊处理 |

#### 可微仿真谱系

| 工具 | arXiv | 接触方案 | 梯度质量 | 人形适用性 |
|---|---|---|---|---|
| **Brax** | 2106.13281 ✓（JAX 开源） | soft/spring（光滑可微但物理不精确） | 光滑但有偏 | 大规模 RL 训练，sim-to-real 接触失配 |
| **Nimble** | 2103.16021 ✓（开源） | LCP 硬接触，**解析对 LCP 解微分**（隐函数梯度） | 接触模式切换处仍不连续 | 全特征刚体+关节，可做 system-id |
| **Dojo** | 2203.00806 ✓（Dojo.jl 开源） | NCP（带二阶锥摩擦）+ **定制内点法 + 隐函数梯度** | 比 LCP 更准；切换处仍非光滑 | 变分积分器保能量，低采样率稳定 |

**可微接触梯度的根本问题**（2207.05060 ✓）：在接触法向不沿坐标轴或接触模式切换时，现有可微接触给出的梯度**经验上有偏甚至方向错误**。这是整条可微仿真线最硬的未解难题。

#### 接触动力学建模新范式

| 工作 | arXiv | 方法机制 | 状态 |
|---|---|---|---|
| **ContactNets** | CoRL 2020 | 2009.11193 ✓ | 不直接回归不连续下一状态，而是学**接触距离函数与接触雅可比**，把接触动力学写成由网络定义的**互补/优化问题**，其解隐式给出离散接触行为。用光滑可学组件表达不连续物理 | 已研究，已证数据高效 |
| **互补无关（SDF 多接触）** | 2408.07855 ✓ / 2408.09612 ✓ | 用**SDF 直接构造闭式可微多接触动力学**，**完全避开互补约束**，使多接触规划可解、梯度更好 | 2024 最有新意，真机验证有限 |
| **同时学接触+连续动力学** | CoRL 2023 | 2310.12054 ✓ | 仅通过观测接触丰富轨迹同时学未知多连杆物体的接触模型和连续动力学，无需先验几何/惯量 | 已研究但扩展性待验证 |

#### ✅ 可实施方案 B：硬接触梯度"学习型校正器"

**问题**：可微仿真在接触模式切换处梯度有偏，导致一阶策略梯度在接触丰富任务上方差大、方向错。

**方法架构**：
```
基础: Dojo/Nimble 给出解析梯度 g_analytic = ∂V/∂θ
校正器 f_φ(s, contact_state, g_analytic) → Δg
  输入: 接触状态(SDF值/法向/相对速度), 解析梯度
  输出: 梯度修正量
训练: 用零阶/有限差分梯度 g_fd 作"近真值"监督
  L_corrector = ||f_φ(·) - (g_fd - g_analytic)||²
策略梯度: g_corrected = g_analytic + f_φ(·)  [比纯零阶快，比纯解析准]
```

**数据**：Brax/Isaac 仿真中大量接触事件的轨迹（已知精确有限差分梯度作监督，GPU 并行采集）。

**评测**：locomotion（接触丰富的 ANYmal/人形）策略收益曲线、梯度方向余弦（vs 有限差分真值）、sim-to-real 迁移率。对照：纯解析梯度 vs 纯零阶 vs 校正后。

**开源基座**：Dojo（Julia，有 Python binding）、Brax（JAX）、Isaac Lab（PyTorch）。

---

### 8.1.3 物理推理大模型（VLM 物理常识）

| 工作 | venue/年份 | arXiv | 方法机制 | 关键发现 |
|---|---|---|---|---|
| **PhysBench** | ICLR 2025 | 2501.16411 ✓（USC，开源） | 10,002 条图-视频-文交错条目，4 大域 19 子类。提出 **PhysAgent**：用感知工具（检测/分割/深度）补 VLM 物理感知短板，再由 VLM 综合推理 | 75 个 VLM 上评测：常识强但物理世界理解弱 |
| **Cosmos-Reason1** | NVIDIA，2025 | 2503.15558 ✓（开源） | 定义两套本体：层级化物理常识本体 + 二维具身推理本体；基于多模态 LLM，经视觉-物理 SFT + **强化学习（规则可验证奖励）**后训练；构建对应物理 AI 推理基准 | 输出语言级决策/推理，与连续控制仍有 grounding gap |

**尚未被研究的空白**：把 VLM 的**符号物理推理与可微物理/学习接触模型闭环**（用 VLM 提物理假设 → 可微仿真验证 → 反馈给 VLM 修正规划）目前几乎空白。

#### ✅ 可实施方案 C：VLM 物理假设 ⊕ 可微仿真验证闭环

**方法**：
```
输入: 场景图像 + 任务指令
Step 1 [VLM 推理]: Cosmos-Reason1/开源 VLM → 结构化物理假设
  e.g. {"mass_ratio": "heavy_obj > light_obj", "friction": "low", "grasp_stable": true}
Step 2 [仿真验证]: 假设 → Brax/ContactNets 快速 rollout → 验证假设与物理结果是否一致
  L_verify = consistency(rollout_outcome, assumed_outcome)
Step 3 [反馈修正]: 验证不一致 → 生成结构化反馈 → VLM 修正物理假设 → 循环
Step 4 [输出]: 验证通过的假设 + 规划 → 低层策略执行
奖励信号: 验证一致性作 RL 可验证奖励（呼应 Cosmos 的"可验证奖励"思路）
```

**数据**：PhysBench 动力学子集（有标准答案，可做批量验证对比）、Physion++（连续属性推断对照）。

**评测**：VLM 物理判断在验证后的准确率提升；错误自检率；机器人操作任务成功率（对比无验证基线）。


---

## 附录 C：六文档体系导航

> 本文档是六文档体系的**入口**——覆盖最全面、信息量最大。当你需要更深入的信息时，按以下指南跳转到对应文档。

### 文档关系图

```
                   ┌─────────────────────────────────┐
                   │  Humanoid_World_Action_Model     │
                   │  _Survey.md  (1538行)            │
                   │  "百科全书：有什么论文"           │
                   │  ← 入口文档，覆盖六大分支        │
                   └──────────┬──────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
  ┌───────▼──────────┐ ┌─────▼─────────┐ ┌───────▼──────────────┐
  │ VLA Framework    │ │ WAM Framework │ │ Latest Supplement    │
  │ (438行)          │ │ (439行)       │ │ (333行)              │
  │ "VLA的科学瓶颈"  │ │ "WAM的根本属性"│ │ "2026最新论文+        │
  │ 4Q: 表征/语言/   │ │ 4Q: 隐变量/   │ │  对Framework的验证"  │
  │   时序/进化      │ │   推演/信任/  │ │                      │
  │                  │ │   内生安全    │ │                      │
  └───────┬──────────┘ └─────┬─────────┘ └───────┬──────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
  ┌───────▼──────────────┐          ┌─────────────▼──────────┐
  │ Future Directions    │          │ DeepDive Implementation │
  │ (1346行)             │          │ (1523行)                │
  │ "做什么方向能发顶会" │          │ "具体怎么做、用什么     │
  │ 28个方向, 每方向有:  │          │  架构/loss/数据"        │
  │ Insight + 方案 +     │          │ 12个升级方案(A-L) +     │
  │ 发表级别 + 可行性    │          │ 科学问题小节            │
  └──────────────────────┘          └─────────────────────────┘
```

### 按需求导航

| 你的需求 | 应该读哪个文档 | 阅读路径 |
|---|---|---|
| "我想了解这个领域有哪些论文" | Survey | Survey → 按六大分支查阅感兴趣的子方向 |
| "我想理解 VLA 的根本科学瓶颈" | VLA Framework | Survey → VLA Framework 四个 Q |
| "我想理解 WAM 应该满足什么根本属性" | WAM Framework | Survey → Supplement（了解最新 WAM）→ WAM Framework |
| "我想找一个能发顶会的方向" | Future Directions | Survey → 两个 Framework → Future Directions → 按优先级表挑选 |
| "我已经有了方向，需要落地实现" | DeepDive Implementation | Future Directions 选中方向 → DeepDive 对应方案 → 按伪代码实施 |
| "我想知道 2026 年有什么最新进展" | Latest Supplement | Supplement → 关注"关键认知更新"和"与 Framework 的对应表" |
| "我想写 Related Work 章节" | Survey + Supplement | Survey 获取全面覆盖 → Supplement 补最新工作 |
| "我想写 Method 章节" | DeepDive Implementation | DeepDive 对应方案 → 按伪代码+loss 设计 method |
| "我想写 Introduction/Motivation" | VLA/WAM Framework | Framework 的 Q 提供"根本性问题"的 motivation |

### 关键交叉引用速查

| 概念 | Survey | VLA Framework | WAM Framework | Supplement | Future Dir | DeepDive |
|---|---|---|---|---|---|---|
| 自指涉可恢复性 R(z;π,M) | 分支 E 恢复 | — | Q4 内生安全 | 对应表 Q4 空白 | **方向 15** | **方案 L** |
| Fast-WAM video co-training | — | — | Q2 长时序推演 | §1.4 核心 | **方向 1** | §1.4.1 |
| S1/S2 自适应切换 | 分支 C 分层 | Q3 长时序 | — | §2.1 Ψ0 | **方向 7** | §4.5.1 |
| 触觉的"物理校准"角色 | 分支 D 触觉 | — | — | — | **方向 13** | **方案 F**, §2.6.2 |
| 3DGS 作为策略内在表征 | — | — | — | §1.11 PointWorld | **方向 11** | §3.3.1 |
| 跨本体物理常识迁移 | 分支 B 控制 | — | Q1 物理隐变量 | §1.2 DreamDojo | **方向 4** | — |
| 语言→物理约束映射 | 分支 C VLA | Q2 语言Grounding | — | — | **方向 6** | — |
| JEPA+接触物理 | — | — | — | §3.4 V-JEPA2 | **方向 8** | §5.5.2 |
| Near-failure 主动学习 | 分支 E 恢复 | Q4 持续进化 | Q3 想象可信度 | — | — | §5.4.2 |
| Fisher 信息动作几何 | — | Q1 动作表征 | — | §1.8 AttenA+ | **方向 22** | — |
| 反事实数据增强 | — | — | — | — | **方向 27** | §5.4.3 |
| Lyapunov-EBM 稳定控制 | — | — | Q4 内生安全 | — | **方向 18** | §5.5.3 |

### 文档维护原则

1. **Survey 和 Supplement 是"数据层"**——只加新论文、更新核实状态，不改变分析框架
2. **VLA/WAM Framework 是"理论层"**——科学问题框架相对稳定，新论文通过 Supplement 的"对应表"接入
3. **Future Directions 是"选题层"**——随着新论文出现，方向可以被验证/修正/新增
4. **DeepDive Implementation 是"落地层"**——方案随技术进步迭代（如新的 backbone、新的 loss 设计）















