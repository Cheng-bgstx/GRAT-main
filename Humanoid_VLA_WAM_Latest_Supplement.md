# 2025-2026 年人形机器人 VLA 与 WAM 最新工作补充调研

> 本文档是对 `Humanoid_VLA_Research_Framework.md` 和 `Humanoid_World_Action_Model_Research_Framework.md` 的补充，覆盖了此前调研遗漏的 2025-2026 年最新关键工作。
> 完成日期：2026-06-04

---
## 📍 本文档在六文档体系中的定位

**角色：时间维度的补丁——"2025-2026 年爆发的最新工作、对原有框架的验证与冲击"**

| 维度 | 说明 |
|---|---|
| **本文档做什么** | 系统梳理 Survey 和两个 Framework 成稿后新出的关键工作（DreamZero/DreamDojo/Fast-WAM/Cosmos Policy/X-WAM/Ψ0/WholeBodyVLA 等 30+ 篇），分析每篇对原有框架的"验证状态"（如 Fast-WAM 冲击了 Q2 的基本假设） |
| **本文档不做什么** | 不重复 Survey 的全面文献覆盖（只补充最新的）、不重复 Framework 的科学问题（只标记新工作与 Framework 的对应关系）、不给未来方向（→ `Humanoid_Embodied_AI_Future_Directions.md`） |
| **与两个 Framework 的关系** | 第 4.1/4.2 节明确标注了"新工作对应 Framework 中的哪个 Open Problem"和"对该问题的验证/冲击"——这是本文档独有的贡献 |
| **与 Survey 的关系** | Survey 覆盖至 2025 年中的工作；本文档覆盖 2025 下半年-2026 年 6 月的最新工作，尤其是 WAM 2026 年爆发的新范式 |
| **与 Future Directions 的关系** | 本文档的"关键认知更新"（如"WAM ≠ 想象 rollout 必须慢""视频预训练是新的 ImageNet"）是 Future Directions 中 WAM 方向的 motivation 来源 |
| **阅读顺序建议** | 读完 Survey 和两个 Framework 后读本文档 → 理解 2026 年最新动态 → 跳到 Future Directions 选择方向 |

---


## 一、WAM（World Action Model）——2026 年最重要的范式转变

WAM 将世界模型（未来状态预测）与动作生成统一到联合框架中，被广泛视为 VLA 之后具身智能的下一个重要方向。复旦大学团队 2026 年 5 月发布了首篇 WAM 系统性综述。

### 1.1 WAM 综述

**World Action Models: The Next Frontier in Embodied AI**
- 复旦大学，arXiv 2605.12090，2026.05
- 首篇 WAM 领域系统性综述。正式定义 WAM 概念，区分**级联式**（先预测未来→再解码动作）和**联合式**（同时预测状态+动作）两类架构。系统梳理了数据生态（遥操作/便携人类示范/仿真/互联网视频）、评估协议和开放挑战。
- [中文解读](https://hub.baai.ac.cn/view/54703)

### 1.2 DreamZero：WAM 作为零样本策略

**DreamZero: World Action Models are Zero-shot Policies**
- NVIDIA GEAR Team，arXiv 2602.15922，2026.02.17
- 🔗 项目页 dreamzero0.github.io | 代码 github.com/dreamzero0/dreamzero
- **核心机制**：14B 参数 WAM，基于预训练 image-to-video 扩散主干（Wan2.1）。与 VLA 的"视觉→动作"直接映射不同，DreamZero **联合预测未来视频帧+动作**，从预测的视觉未来中学习物理动力学。
- **关键创新**：
  - `Real Obs Injection`：将真实观测注入 rollout 切断误差累积
  - `DreamZero-Flash`：解耦视频/动作噪声调度，实现 7Hz 实时推理（38× 加速）
- **关键结果**：
  - 零样本泛化 >2× 超越 SOTA VLA（π₀.₅, GR00T N1.6）
  - 未见过任务平均成功率 39.5%（解鞋带、熨烫、摘帽子、握手）
  - **跨本体迁移**：30 分钟 play data 将 AgiBot G1 模型迁移到全新 YAM 机器人
  - **视频迁移**：10-20 分钟人类或其他机器人视频 → >42% 相对提升

### 1.3 DreamDojo：从大规模人类视频学习世界模型

**DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos**
- NVIDIA GEAR Team，arXiv 2602.06949，2026.02.09
- 🔗 项目页 dreamdojo-world.github.io | 代码 github.com/NVIDIA/DreamDojo
- **核心机制**：在 **44,000+ 小时**第一人称人类视频上预训练动作条件世界模型（AC-WM）——是此前最大机器人世界模型数据集的 15× 时长、96× 技能、2000× 场景。
- **关键创新——Latent Actions**：人类视频没有动作标签 → 用自监督时空 VAE 从相邻帧间提取**连续潜在动作**。同样的运动会获得相似的潜在表征，**无论执行者是人还是机器人**——打通跨本体数据壁垒。
- **关键结果**：
  - 实时推理 10.81 FPS（35→4 去噪步蒸馏）
  - 策略评估：Pearson r=0.995 与真实世界成功率
  - Model-based planning：~2× 成功率提升
  - 单 RTX 5090 实时遥操作
  - **Scaling law**：更大模型+更多人类视频数据 = 更好的 OOD 泛化

### 1.4 Fast-WAM：WAM 能力来源的机制分析

**Fast-WAM: Do World Action Models Need Test-time Future Imagination?**
- 清华 MARS Lab（赵行组）/ 星海图 Galaxea AI，arXiv 2603.16666，2026.03.17
- 🔗 项目页 yuantianyuan01.github.io/FastWAM | 代码 github.com/yuantianyuan01/FastWAM
- **核心假说**：WAM 的主要收益来自**训练阶段的 video co-training 提供的 dense 监督信号**，而非推理阶段的未来视频生成。
- **关键发现**：
  - 训练时保留 video prediction 作为 proxy task，推理时**跳过未来视频生成** → 性能几乎不下降
  - 去掉训练时的 video co-training → 性能**骤降**
  - 推理延迟压缩至 190ms，比传统"先想象再执行"WAM **快 4 倍以上**
  - 无具身预训练情况下，LIBERO/RoboTwin 已达 SOTA
- **架构**：极简 Mixture-of-Transformer + attention mask，实现多变体受控对比
- **与 Cosmos-Policy 对比**：
  - Cosmos-Policy：Action 当多视角建模，需 repeat action 缓解梯度淹没
  - Fast-WAM：从 video backbone 插值出轻量 action DiT（MoT 架构），单独训练，无需 repeat action
- 图灵奖得主 Yann LeCun 的新作与 Fast-WAM 被学界并列推荐

### 1.5 Cosmos Policy：微调视频模型用于视觉运动控制

**Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning**
- NVIDIA Cosmos Lab，arXiv 2601.16163，**ICLR 2026**
- 🔗 项目页 research.nvidia.com/labs/cosmos-lab/cosmos-policy
- 将 Cosmos 视频基础模型后训练为视觉运动策略。LIBERO 98.5%，RoboCasa 67.1% SOTA

### 1.6 X-WAM：统一 4D 世界动作模型

**Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising (X-WAM)**
- 清华 + 小米机器人 + 北大 + 中科院，arXiv 2604.26694，2026.04
- **核心贡献**：在单一框架中同时实现三项能力：高保真 4D 世界合成（多视角 RGB-D 视频+3D 重建）+ 实时动作执行 + 空间感知与决策
- **关键创新——异步噪声采样（ANS）**：推理时动作仅需少量去噪步数（快速解码），视频使用完整步数（高保真生成）；训练时从联合分布采样对齐推理分布
- **关键结果**：RoboCasa 79.2%，RoboTwin 2.0 90.7%，推理速度提升 4.5×，预训练数据 >5,800 小时

### 1.7 NoiseGate：可学习的信息门控

**NoiseGate: Learning Per-Latent Timestep Schedules as Information Gating in World Action Models**
- arXiv 2605.07794，2026.05
- 将每个潜在帧的噪声级别视为可学习的**信息门控策略**，通过调节不同帧对 action token 的 Key/Value 贡献来优化动作生成质量

### 1.8 HarmoWAM：统一预测式与反应式专家

**HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Models**
- arXiv 2605.10942，2026.05
- 统一"预测式专家"（迭代生成动作）和"反应式专家"（从预测视觉直接推理动作），通过**过程自适应门控**动态切换
- 6 个真实机器人任务上零样本泛化远超 VLA（+33%）和已有 WAM（+29%）

### 1.9 When to Trust Imagination：想象可信度

**When to Trust Imagination: Adaptive Action Execution for World Action Models**
- arXiv 2605.06222，2026.05
- **核心机制**：FFDC（未来前向动态因果注意力）验证器，当预测与现实不一致时提前重新规划。Mixture-of-Horizon 训练
- 减少 WAM 前向传播 69.1%、执行时间 34.0%，成功率提升 2.54%
- **直接对应我们的 Open-3B（想象可信度）和 Open-3C（自适应切换）**

### 1.10 AttenA+：动作不平等修正

**AttenA+: Rectifying Action Inequality in Robotic Foundation Models**
- arXiv 2605.13548，2026.05
- 通过**速度驱动动作注意力**重新加权训练目标，优先学习低速关键交互段。插件式增强。提升 OpenVLA-OFT 到 98.6%，FastWAM 到 92.4%

### 1.11 PointWorld：3D 点流世界模型

**PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation**
- Stanford + NVIDIA Research，arXiv 2601.03782，2026.01
- 🔗 项目页 point-world.github.io | 代码 github.com/NVlabs/PointWorld | HF nvidia/PointWorld_models
- **核心创新**：将状态和动作统一表示为 **3D 点流（3D point flows）**——给定 1 张或多张 RGB-D 图像和机器人动作序列，预测每个像素在 3D 空间中的位移
- 动作是 **3D 点流形式**（而不是本体特定动作空间），天然跨本体
- 训练 ~2M 轨迹 / ~500 小时，覆盖单臂 Franka + 双臂人形，真机+仿真混合
- **单个预训练 checkpoint** → 零样本 pushed/deformable/articulated/tool use manipulation，仅需单张 in-the-wild RGB-D 图像
- 推理速度 0.1s，可集成 MPC

### 1.12 LEGS：无需遥操作的人形 VLA 训练

**LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World**
- arXiv 2606.01458，2026.05.31
- 🔗 项目页 legsvla.github.io | 平台：Unitree G1
- **核心创新——混合仿真器**：Mesh 前景（机器人+物体+物理仿真）+ **3DGS 背景**（手持相机扫描真实部署环境→3D Gaussian Splatting 重建）+ 程序化动作基元生成器（自动合成带标签示范，完全无需遥操作）
- **关键结果**：LEGS 训练的 VLA 策略匹配或超越遥操作训练的策略；照片级真实感是关键（纯 Mesh 仿真差）；覆盖新场景的成本是遥操作的 **1/15**；在物体+场景联合外观偏移下 LEGS 策略保持成功，遥操作基线完全失效
- 测试了三个 VLA backbone：Ψ₀, π₀.₅, GR00T N1.6

### 1.13 WAM 诊断与评测

**Beyond Task Success: Behavioral and Representational Diagnostics for WAM and VLA**
- arXiv 2606.01095，2026.05
- 模型无关诊断框架：通过行为轨迹分析和稀疏自编码器特征分析，揭示 WAM 是否真正编码了**面向未来的表征**

**WAM-Nav: Asymmetric Latent World-Action Modeling for Unified Visual Navigation**
- arXiv 2606.04907，2026.06
- 将 WAM 扩展到导航，非对称联合扩散，统一 Image-Goal/Point-Goal/No-Goal 三种导航模式，真实环境 85% 成功率

---

## 二、人形机器人 VLA 的最新突破（2025-2026）

### 2.1 Ψ0 (Psi-Zero)：人形全身 Loco-Manipulation 开源基座

**Ψ₀: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation**
- USC Physical Superintelligence Lab + NVIDIA + WorldEngine，arXiv 2603.12263，2026.03.12
- 🔗 项目页 psi-lab.ai/Psi0 | 代码 github.com/physical-superintelligence-lab/Psi0

**三系统架构**：
```
System-2 (慢/语义): Qwen3-VL-2B-Instruct VLM → 理解语言+编码视觉 → 特征 z_t
System-1 (快/动作): MM-DiT (~500M, 基于 SD3) → Flow Matching 生成 36-DoF 动作块
System-0 (反射/下肢): AMO RL 跟踪策略 → 8DoF 高层指令 → 15DoF 下肢关节
最终: 43-DoF 全身控制
```

**关键创新**：
- **分阶段训练**：先 ~800h 人类第一人称视频（EgoDex，无动作标注）预训练 VLM → 再 ~30h 真机数据后训练动作专家 → 每任务仅需 80 条遥操作轨迹微调
- **RTC（Real-Time Chunking）**：训练时模拟推理延迟，随机 mask 前 d 个 token 的噪声并对其施加 loss，迫使模型在已有"已承诺动作"下预测连续动作块 → 消除 160ms 推理延迟导致的停走抖动
- **仅用 GR00T N1.6 / π₀.₅ 约 1/10 的数据量**，8 项长程 loco-manipulation 任务平均成功率**反超 40%+**

**与我的框架的关联**：
- 三系统层级分解恰好对应 Open-1A（语义-运动-执行三层解耦），但 S1/S2/S0 的分解是手工设计的，并非从数据中涌现
- S1/S2 仍是固定频率，没有 Open-3C 的"自适应切换"
- 三阶段训练恰好是 Open-4B（适应审计）的研究对象——适应发生在 VLM 层、动作专家层、还是任务微调层？

### 2.2 WholeBodyVLA：统一潜在 VLA for 全身 Loco-Manipulation

**WholeBodyVLA: Towards Unified Latent VLA for Whole-body Loco-manipulation Control**
- 复旦 + 港大 OpenDriveLab + 智元 AgiBot + 上海创智学院，arXiv 2512.11047，**ICLR 2026**
- 🔗 项目页 opendrivelab.com/WholeBodyVLA | GitHub github.com/OpenDriveLab/WholebodyVLA
- 平台：智元灵犀 X2 人形机器人

**核心机制**：
- **LAM（Latent Action Model）**：从**无动作标注**的第一人称人类视频中学习统一潜在动作表示——大幅降低对昂贵遥操作数据的依赖（降约 75%）
- 操作 LAM 和移动 LAM **分离训练**，再联合监督 VLA——解决两种模态的视觉变化差异
- **LMO RL 控制器**：用**离散指令接口**（前进/侧移/转弯/下蹲标志位）替代传统连续速度跟踪
- 两阶段课程：先稳定步态 → 再优化精确度。转向偏差减少 73%，下蹲晃动减少 40%，可推行 50kg+

**关键结果**：真机验证三项任务（装袋、装箱、重载推车），平均成功率 78%，比基线高 21.3%

### 2.3 LeVERB：人形 VLA 评估基准

**LeVERB: Humanoid Whole-Body Control with Latent Vision-Language Instruction**
- arXiv 2506.13751，2025.06
- **首个面向人形全身 VLA 的 sim-to-real 闭环基准**：LeVERB-Bench（150+ 任务，10 类别）
- 分层框架：高层视觉语言策略学习潜在动作词汇 → 底层 RL WBC 将"latent verbs"转化为动力学指令
- 视觉导航 80% 零样本成功率，整体 58.5%，比 naive 分层 VLA 高 **7.8 倍**

### 2.4 GR00T N1.6：NVIDIA 最新人形基础模型

**GR00T N1.6** — NVIDIA，CES 2026 发布
- 🔗 代码 github.com/NVIDIA/Isaac-GR00T | HF 权重开源
- **vs N1.5 的关键升级**：
  - VLM 基座升级为 **Cosmos-Reason-2B**（原生分辨率，无填充编码）
  - DiT 深度翻倍（16→32 层），动作更平滑
  - 移除适配器层，改为解冻 VLM 顶部 4 层
  - 动作预测改为**状态相对动作块（state-relative action chunks）**→ 成功率提升约 40%
- 训练数据：新增数千小时遥操作（双臂 YAM + AgiBot Genie1 + Galaxea R1 Pro + Unitree G1 全身）
- 配套：Isaac GR00T 参考人形机器人（Unitree H2 Plus + 双 Sharpa Wave 五指触觉手，**75 自由度**，Jetson AGX Thor T5000 算力，预计 2026 年底供货）

### 2.5 HOIST：人形悬挂负载操作

**HOIST: Humanoid Optimization with Imitation and Sample-efficient Tuning for Manipulating Suspended Loads**
- arXiv 2606.00252，2026.05
- 将 VLA 高层策略（VR 遥操作微调）与迭代批量 RL 结合
- 相比纯 VLA rollout，平移放置误差降低 19.9 cm，角度误差降低 3.56°

### 2.6 EgoHumanoid + AMS

**EgoHumanoid**（RSS 2026）：从人类第一人称演示学习全身 loco-manipulation，与 WholeBodyVLA 同团队
**AMS**（OpenDriveLab）：统一 agility 和 stability 的人形控制

### 2.7 跨本体人形控制

**XHugWBC: Scalable and General Whole-Body Control for Cross-Humanoid Locomotion**
- arXiv 2602.05791，2026.02
- **单一策略**零样本泛化到 12 个仿真 + 7 个真实人形机器人
- 形态随机化 + 语义对齐的观测量/动作空间 + 本体感知架构

**AdaMorph: Unified Motion Retargeting via Embodiment-Aware Adaptive Transformers**
- arXiv 2601.07284，2026.01
- 单一模型重定向人类动作到 12 种不同人形机器人，AdaLN 条件化

---

## 三、人形 Loco-Manipulation 与全身控制的最新突破

### 3.1 World-Ego Modeling

**World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks**
- arXiv 2605.19957，2026.05
- 将世界模型分解为 **world**（场景规律）和 **ego**（机器人动力学）两个组件
- HTEWorld 基准（125K 视频片段），级联-并行 MoE 扩散生成器在长程任务上达 SOTA

### 3.2 SSR：开放世界人形行走

**SSR: Scaling Surefooted and Symmetric Humanoid Traversal to the Open World**
- arXiv 2605.30770，2026.05
- 端到端第一人称视觉人形 locomotion
- **想象落脚点引导**（imagined foothold guidance）预测 swing-foot 接触
- 等变对称增强 + 地形特定运动先验
- 处理楼梯、宽间隙、高平台、户外地形

### 3.3 Lifting Embodied World Models

**Lifting Embodied World Models for Planning and Control**
- arXiv 2604.26182，2026.04
- 训练轻量策略将**高层动作**（2D waypoints：骨盆、头部、手部）映射为低层关节动作
- "提升式世界模型"比直接在关节空间搜索**关节误差低 3.8×**，泛化到未见环境

### 3.4 Cross-Humanoid & Hardware-Agnostic

**Toward Hardware-Agnostic Quadrupedal World Models via Morphology Conditioning**
- arXiv 2604.08780，2026.04
- 首个零样本泛化到新四足形态的世界模型；将环境动力学与机器人本体解耦


## 四、本次补充的学术贡献总结

### 4.1 对 VLA 框架的补充

| 补充内容 | 对应框架中的位置 | 新认知 |
|---|---|---|
| Ψ0 三系统架构 | Q1（动作表征的层级分解） | 语义-运动-执行三层手工分解已经工程化实现，但"可学习的解耦"和"自适应接口"仍是未解问题 |
| WholeBodyVLA 的 LAM | Q1（动作表征）、Q2（语言 Grounding） | 从无标注人类视频学潜在动作——不经过语言直接学"做什么"的 latent representation |
| LeVERB 基准 | Q3（长时序评测） | 首个系统化的人形 VLA 评测基准——可以用它来评测 Open-3A/3B/3C |
| GR00T N1.6 开源 | Q1-Q4 的实验基座 | 75DoF 的完整开源人形基座+Isaac GR00T 参考硬件，可以作为博士实验平台 |

### 4.2 对 WAM 框架的补充——全新的技术线

WAM 是 2026 年才爆发的新范式。我的 WAM 框架（Humanoid_World_Action_Model_Research_Framework.md）中提出的四个科学问题，与最新的 WAM 工作形成了精准对应：

| 框架中的开放问题 | 对应的最新 WAM 工作 | 验证状态 |
|---|---|---|
| **Q1 物理隐变量** | DreamDojo（Latent Actions 作为跨本体物理代理）、PointWorld（3D点流统一状态和动作） | DreamDojo 的 latent actions 是跨本体隐变量的成功实例 |
| **Q2 长时序推演** | DreamZero（Real Obs Injection切断误差累积）、X-WAM（异步去噪跳帧）、Fast-WAM（video co-training比test-time imagination更重要） | Fast-WAM 直接质疑了"想象rollout"的必要性——与Open-2A形成方法论张力 |
| **Q3 想象可信度** | When to Trust Imagination（FFDC验证器）、NoiseGate（信息门控）、Fast-WAM（训练时video co-training是关键） | 这三个工作从不同角度回答了"什么时候该信任想象"，直接验证了Q3的重要性 |
| **Q4 内生安全** | HarmoWAM（预测式+反应式统一门控） | 仍最空白——没有WAM工作显式编码可恢复性 |

### 4.3 关键认知更新

1. **WAM ≠ 想象rollout必须慢**：Fast-WAM 证明推理时完全可以跳过视频生成，训练时的 video co-training 才是关键。这动摇了"WAM=先想象再执行"的基本假设。

2. **视频预训练是新的 ImageNet**：DreamDojo 44K 小时人类视频、DreamZero 的视频迁移、Cosmos Policy 的后训练、X-WAM 5800 小时机器人数据——都在说同一件事：**视频生成预训练是 WAM 的 ImageNet moment**。

3. **人形全身的 VLA 基座正在成熟**：Ψ0 开源、WholeBodyVLA（ICLR 2026）、GR00T N1.6 开源+NVIDIA 官方参考硬件（75DoF）→ 博士研究不再需要"从零搭建"，可以直接在这些基座上做科学问题研究。

4. **"跨本体"是 2026 年最重要的关键词**：DreamDojo 的 latent actions、PointWorld 的 3D 点流、XHugWBC 的跨人形策略、AdaMorph 的跨人形重定向、Hardware-Agnostic WM——都在消除"机器人型号"这个人工边界。

5. **我的研究框架的科学问题依然成立且更加紧迫**：四个 Q 中提出的核心 Open Problems（层级解耦、自适应预测粒度、因果不确定性分解、自指涉安全），在最新的 WAM 工作中仍然只有一个方向性探索（When to Trust Imagination 对应 Q3），绝大多数是基本空白。


## 五、更新后的论文清单

### WAM 类
| 工作 | 机构 | arXiv | 关键创新 |
|---|---|---|---|
| DreamZero | NVIDIA GEAR | 2602.15922 | 14B WAM 零样本策略，视频+动作联合预测，跨本体迁移 |
| DreamDojo | NVIDIA GEAR | 2602.06949 | 44K小时人类视频世界模型，Latent Actions跨本体 |
| Fast-WAM | 清华/星海图 | 2603.16666 | video co-training > test-time imagination |
| Cosmos Policy | NVIDIA | 2601.16163 | ICLR 2026，LIBERO 98.5% SOTA |
| X-WAM | 清华/小米 | 2604.26694 | 4D世界动作模型，异步去噪，5800h数据 |
| PointWorld | Stanford/NVIDIA | 2601.03782 | 3D点流统一状态+动作，零样本操作 |
| NoiseGate | — | 2605.07794 | 可学习信息门控 |
| HarmoWAM | — | 2605.10942 | 预测式+反应式专家统一 |
| When to Trust Imagination | — | 2605.06222 | FFDC验证器，想象可信度 |
| AttenA+ | — | 2605.13548 | 动作不平等修正 |
| WAM-Nav | — | 2606.04907 | WAM用于导航 |
| Beyond Task Success | — | 2606.01095 | WAM/VLA诊断框架 |
| WAM Survey | 复旦 | 2605.12090 | 首篇WAM综述 |

### 人形 VLA 类
| 工作 | 机构 | arXiv/Venue | 关键创新 |
|---|---|---|---|
| Ψ0 (Psi-Zero) | USC/NVIDIA | 2603.12263 | 三系统架构，800h人类+30h真机，反超GR00T 40%+ |
| WholeBodyVLA | 复旦/港大/智元 | 2512.11047, ICLR 2026 | LAM潜在动作+离散指令RL，真机78%成功率 |
| LeVERB | — | 2506.13751 | 首个人形VLA sim-to-real评测基准 |
| GR00T N1.6 | NVIDIA | CES 2026 | 开源75DoF人形基座+参考硬件 |
| HOIST | — | 2606.00252 | VLA+RL悬挂负载操作 |
| EgoHumanoid | OpenDriveLab | RSS 2026 | 人类第一人称演示→全身loco-manip |

### 跨本体/全身控制类
| 工作 | 机构 | arXiv | 关键创新 |
|---|---|---|---|
| XHugWBC | — | 2602.05791 | 单一策略12仿真+7真人形零样本泛化 |
| AdaMorph | — | 2601.07284 | 单一模型12种人形运动重定向 |
| LEGS | — | 2606.01458 | 3DGS混合仿真，免遥操作VLA训练 |
| World-Ego Modeling | — | 2605.19957 | World+Ego分解世界模型 |
| SSR | — | 2605.30770 | 第一人称视觉开放世界人形行走 |
| Lifting Embodied WM | — | 2604.26182 | 高层waypoints→低层关节 |
| Hardware-Agnostic WM | — | 2604.08780 | 跨四足形态世界模型 |

---

> **核实声明**：所有论文标题、arXiv编号、机构均经WebSearch核实。标注年份为2026的预印本为极新工作，建议正式引用前二次确认信息。
