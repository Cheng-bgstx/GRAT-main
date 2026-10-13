# 人形机器人接触感知与物理交互：全流程深度调研

> **调研日期：** 2026年6月11日
> **覆盖范围：** 经典基础→2026最新，300+篇论文/数据集/系统
> **调研逻辑：** 硬件→数据→算法→训练→推理→评测 全链路

---

## 目录

1. [接触感知硬件层](#1-硬件层)
2. [数据采集与数据集](#2-数据层)
3. [基础算法：力控制与接触建模](#3-算法基础层)
4. [现代方法：学习驱动的接触感知与物理交互](#4-现代方法层) 
5. [世界模型与接触预测](#5-世界模型层)
6. [训练范式：从BC到RL到Sim-to-Real](#6-训练范式层)
7. [推理与部署](#7-推理层)
8. [评测体系与Benchmark](#8-评测层)
9. [当前不足与开放问题](#9-不足与问题)

---

## 1. 硬件层：接触感知传感器技术全景

### 1.1 触觉传感器的技术谱系

#### 1.1.1 视觉式触觉传感器（VBTS）— 当前主流

**原理：** 弹性体+内置相机，通过拍摄弹性体变形来推断接触力/几何/纹理。

| 传感器 | 机构 | 核心技术参数 | 优势 | 局限 | 年份 |
|--------|------|------------|------|------|------|
| **GelSight** | MIT → GelSight Inc. | 30µm空间分辨率，<5µm深度分辨率，30-60fps | 微米级纹理重建（可读盲文），最高空间分辨率 | 厚(15-30mm)，弹性体寿命有限 | 2009- |
| **GelSight Mini** | GelSight Inc. | 252mm² 感知面积，商业量产 | 标准化产品，社区使用最广 | 面积小，价格高 | 2023- |
| **DIGIT** | Meta AI → GelSight | 304mm²，模块化可更换弹性体 | 紧凑指尖型，多指集成，开源设计 | 面积小 | 2020- |
| **Digit 360** | Meta+ GelSight | 18+模态，1mN力分辨率，全向感知 | 最强单传感器性能 | 新发布，生态未成熟 | 2024.10 |
| **TacTip** | Bristol Robotics Lab | 2500mm²大面积，针脚位移追踪 | 大面积覆盖，3D打印自制，力向量估计 | 空间分辨率较低(0.2mm) | 2018- |
| **ViTacTip** | Bristol | 视觉+触觉+力多模态，MAE 0.01-0.016N | 多模态融合，高精度力估计 | 体积较大 | 2025 |
| **GelStereo** | 中科院自动化所 | 双目立体视觉，高精度3D重建 | 3D几何精度高 | 计算开销大 | 2021- |
| **MagicTac** | — | 透明层式VBTS，平面定位0.145mm | 定位精度高 | 新出，对标测试少 | 2025 |

**关键技术细节（GelSight原理示例）：**
- 透明弹性体+内表面反射涂层
- RGB三色LED从不同角度(0°/120°/240°)打光
- 弹性体变形→颜色变化编码表面法向量
- 光度立体(Photometric Stereo)算法→重建3D高度图
- 力估计：接触面积×接触压力模型；最新benchmark显示MAE 0.024-0.036N

#### 1.1.2 非视觉式触觉传感器

| 类型 | 原理 | 代表产品/论文 | 特点 | 适用场景 |
|------|------|------------|------|---------|
| **压阻式** | 压力→电阻变化 | 猿声科技HexSkin | 低成本、抗干扰、可大面积 | 全身皮肤覆盖 |
| **电容式** | 压力→电容变化 | RoboSkin (iCub) | 灵敏度高、近距感知 | 精细操作 |
| **压电式** | 压力→电荷 | PVDF薄膜 | 微秒级动态响应、自供电 | 滑移检测、碰撞 |
| **摩擦电式** | 接触→静电 | TENG传感器 | 结构简单、低成本、自供电 | 大面积低精度 |
| **霍尔效应** | 磁场变化 | PaXini PX-6AX | 6轴力/力矩测量 | 关节/腕部 |
| **光导式** | 光强变化 | 光纤传感器 | 抗电磁干扰、可分布式 | MRI兼容场景 |

#### 1.1.3 人形全身触觉皮肤：2025-2026商业化拐点tmu

| 公司/机构 | 产品 | 技术 | 覆盖范围 | 关键指标 | 成本 |
|----------|------|------|---------|---------|------|
| **猿声科技** | HexSkin + Tacta | 空间编码多维压阻 | 全身(六边形拼接) + 手指(1956感测点/手) | 361感测点/cm²，1000Hz | ~$100-150/全身 |
| **PaXini** | PX-6AX-GEN3 | 多维触觉(15维) | 指尖→手腕→全身(TORA-ONE) | 0.01N分辨率，<0.5%FS | 传感器$49起 |
| **KAIST** | 织物式多模态皮肤 | 超分辨率+层析成像 | 机械臂全身 | 多模态触觉分类 | 研究阶段 |
| **iCub** | RoboSkin | 电容式阵列 | 全身(手臂+躯干+手掌) | ~4000个电容单元 | 研究平台 |

**2025-2026关键突破：** 人形全身触觉皮肤从"实验室奢侈品"→"千元级商品"。关键是分布式模块化架构（非集中式MUX）+ 边缘计算（片上处理非原始数据传输）。

### 1.2 力/力矩传感器

| 传感器类型 | 代表产品 | 原理 | 精度 | 部署位置 |
|-----------|---------|------|------|---------|
| **6轴F/T传感器** | ATI Mini45/Nano17, Robotiq FT300 | 应变片 | 0.01-0.1N/0.001Nm | 腕部 |
| **关节力矩** | KUKA iiwa, Franka Panda | 应变片+模型 | ~0.1Nm | 各关节 |
| **无传感器力估计** | UniTac, SixthSense | 本体感觉→ML→力估计 | ~7-8cm定位, ~数N力精度 | 全身 |

### 1.3 接触感知硬件全景对比与选择指南

| 需求场景 | 推荐方案 | 理由 |
|---------|---------|------|
| 精密指尖操作(插USB) | GelSight Mini / Digit 360 | 微米级空间分辨率 |
| 大面积接触(搬运箱子) | TacTip / HexSkin | 大面积+力向量 |
| 低成本全身覆盖 | 本体感知力估计(UniTac) + 稀疏压阻皮肤 | 成本/性能折中 |
| 动态碰撞检测 | 压电薄膜 + 关节力矩突变检测 | 微秒级响应 |
| 人机安全交互 | 全身电容式皮肤 | 近距感知+接触感知双模 |

---

## 2. 数据层：数据集全景

### 2.1 触觉感知数据集

| 数据集 | 年份 | 规模 | 传感器 | 任务 | 模态 | 关键特点 |
|--------|------|------|--------|------|------|---------|
| **FoTa** (T3) | 2024 | 3M+数据点, 13传感器, 11任务 | 多源异构 | 通用 | 触觉 | 首个大规模多传感器统一数据集；零样本跨传感器迁移 |
| **ObjTac** (OmniVTLA) | 2025.08 | 135K三模态样本, 56物体 | 力式触觉 | 物体识别 | 视觉+触觉+文本 | 含粗糙度/硬度标注；10类材质 |
| **TLA Dataset** | 2025.03 | 24K触觉-动作-指令对 | GelStereo | 指尖轴孔装配 | 触觉+语言+动作 | 专为语言条件触觉策略设计 |
| **FreeTacMan** | 2025.06 | 3M+视触觉对, 10K+轨迹, 50任务 | 自研(~$30) | 多样化 | 视觉+触觉+动作 | **无机器人采集**；可穿戴式 |
| **OmniViTac** | 2026.03 | 21,879轨迹, 126物体, 86任务 | Xense+GelSight+DM-Tac+Tac3D | 6类物理交互 | RGB-D+触觉+动作 | **最大视触觉对齐数据集**；EAI-2025 Top10 |
| **HapTile** | 2026.06 | — | 指尖触觉+力反馈遥操 | 日常操作 | 视觉+触觉+语言+动作 | 操作者触觉反馈 |
| **ManiFeel** | 2025 | 标准化benchmark | GelSight Mini | 抓取+插入 | 视觉+触觉 | 标准化的触觉表示对比框架 |

### 2.2 力/接触数据集

| 数据集 | 年份 | 规模 | 机器人 | 任务 | 关键特点 |
|--------|------|------|--------|------|---------|
| **REASSEMBLE** | 2025(RSS) | 4551次演示, 781分钟 | Franka FR3 | NIST Task Board装配 | 多模态(事件相机+F/T+音频+RGB)；层级动作标注 |
| **ForceVLA-Data** | 2025(NeurIPS) | 244轨迹, 140K同步时间步 | Flexiv Rizon 7-DOF | 5类接触丰富任务 | 6轴F/T+VR遥操 |
| **DexCanvas** | 2025.10 | 7000h物理验证rollout, 70h真机, 30物体, 21操作类型 | 灵巧手 | 灵巧操作全类型 | **含连续力向量标注**(非二元接触)；20TB |
| **AGIBOT WORLD 2026** | 2026.06 | — | AgiBot G1 | 接触交互(含失败) | **收录不完美交互**(掉落/碰撞/飞溅) |

### 2.3 人形机器人全身操作数据集

| 数据集 | 年份 | 规模 | 机器人 | 模态 | 特点 |
|--------|------|------|--------|------|------|
| **Humanoid Everyday** | 2025.10 | 10.3K轨迹, 3M+帧, 260任务 | — | RGB+深度+LiDAR+触觉 | 7类任务+人机交互 |
| **SIMPLE** | 2026.06 | 60全身操作任务, 50场景, 1000+物体 | Unitree G1等 | 仿真+真实 | 自动(规划)+遥操(V R)双模式采集 |
| **Humanoid Visual-Tactile-Action** | 2025.10 | — | 灵巧手人形(2124taxel/手) | RGB+触觉+关节动作 | 软体/变形物体；强/弱压力条件 |
| **Punyo EGRL** | 2025.08 | CC0开放 | 软体皮肤人形上半身 | 压力感知皮肤 | 全身接触操作；域随机化sim-to-real |

### 2.4 经典操作Benchmark（可作为机械臂起点）

| Benchmark | 任务数 | 特点 | 适用性 |
|-----------|--------|------|--------|
| **NIST Task Board 1** | 17物体, 4基本动作 | 工业装配标准 | 接触丰富装配验证 |
| **RLBench** | 100任务 | 程序化生成+域随机化 | 大规contact-rich policy训练 |
| **LIBERO** | 130任务, 5 suite | 系统性泛化测试 | VLA/WAM baseline |
| **RoboTwin 2.0** | 50双臂任务 | Easy+Hard设置 | 双臂协调操作 |
| **SimplerEnv** | Google OOD基准 | 真实场景泛化 | sim-to-real桥梁 |
| **MimicGen** | 程序化演示生成 | 自动数据扩增 | 大规模数据生成 |

### 2.5 数据采集方案对比

| 方案 | 代表系统 | 成本 | 传感器丰富度 | 数据效率 | 适用规模 |
|------|---------|------|------------|---------|---------|
| **VR遥操作** | Touch Dreaming, TAMEn, SIMPLE | PICO头显~$400 | 高(视觉+力反馈可选) | 1:1人机时间 | 千条轨迹 |
| **外骨骼遥操作** | ALOHA 2, GELLO | 硬件~$500-5000 | 中(本体感觉为主) | 1:1 | 百-千条 |
| **手持采集(无机器人)** | FreeTacMan, UMI | ~$30-500 | 高(触觉+视觉) | 高于1:1(易操作) | 万条轨迹 |
| **动捕+仿真rollout** | DexCanvas | 动捕~$10K, 仿真算力 | 最高(含物理验证力) | >1:1(仿真扩增) | 7000h仿真 |
| **程序化自动生成** | MimicGen, RLBench | 算力 | 低(仿真) | 远>1:1 | 百万轨迹 |
| **人-机器人现场采集** | AGIBOT探索式遥操 | 真机+人力 | 高(全模态) | 1:1+失败案例 | 面向世界模型 |

---

## 3. 算法基础层：力控制与经典接触建模

### 3.1 力控制经典三部曲（必读基础）

| 方法 | 提出者 | 核心公式/思想 | 适用场景 | 局限 |
|------|--------|-------------|---------|------|
| **刚度控制** | Salisbury 1980 | $F = K(x_d - x)$ | 简单位置-力解耦 | 不能处理动态接触 |
| **阻抗控制** | **Hogan 1985** | $F = M\ddot{x} + B\dot{x} + K(x - x_0)$ | 与环境交互的主流框架 | 需要知道环境位置 |
| **导纳控制** | — | 阻抗的对偶：测力→输出位置 | 高刚度环境、高精度力跟踪 | 力传感器噪声敏感 |
| **混合位置/力控制** | **Raibert & Craig 1981** | 选择矩阵$S$和$I-S$在任务空间解耦 | 约束面操作（擦拭/打磨） | 需要精确的约束几何 |
| **并行力/位置控制** | Chiaverini 1993 | 力/位置控制器并行输出求和 | 避免混合控制的不连续 | 优先级冲突解决 |

**Hogan阻抗控制的物理直觉：**
- 机器人不直接命令"位置"或"力"，而是命令**"质量-弹簧-阻尼"的物理行为**
- 这使机器人在自由空间中表现为位置控制（低阻抗），在接触时表现为力控制（高阻抗）
- 可以理解为"指定机器人'感觉起来'有多硬/多软"

### 3.2 经典接触状态识别与力观测

```
接触状态分类：
├── 自由空间(Free Motion)：F=0, 无约束
├── 过渡(Transition)：从自由→接触的瞬间(碰撞/冲击)
├── 约束运动(Constrained Motion)：与环境接触中
│   ├── 滑动(Sliding)：切向运动+法向约束
│   ├── 滚动(Rolling)：无滑动的转动接触
│   └── 固定(Fixed/Grasped)：无相对运动
└── 卡阻/楔入(Jamming/Wedging)：装配失败模式
```

**力观测/估计方法演化：**
1. 直接测量(ATI F/T传感器) — 最准确但昂贵
2. 关节力矩+动力学模型 — Franka/KUKA的默认方式($\tau_{ext} = \tau_{meas} - M(q)\ddot{q} - C(q,\dot{q})\dot{q} - g(q)$)
3. 广义动量观测器(Haddadin 2017) — 避免加速度噪声
4. 学习式力估计(UniTac, SixthSense) — 不用F/T传感器，只用关节传感器+ML(2025-2026)

---

## 4. 现代方法层：学习驱动的接触感知与物理交互

### 4.1 力/触觉增强的VLA模型

#### ForceVLA (NeurIPS 2025) — 力觉融入VLA的里程碑

```
论文: ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
arXiv: 2505.22159 | NeurIPS 2025
团队: 复旦/上交/新国立等
```

**技术路线：**
- **FVLMoE (Force-aware Mixture-of-Experts):**
  - 6轴F/T信号作为一等模态（与视觉、语言并列）
  - **Late-fusion策略**(关键设计)：VLM先处理视觉-语言→产生中间特征→力token通过4个expert MLP注入(**top-k=1动态路由**)
  - 早期融合(early fusion)导致了0%成功率——因为力信号的时序模式破坏了VLM的语义特征
- 融合特征→流匹配(Flow Matching)动作头→生成力感知的动作chunk

**关键结果：**
- 平均成功率: ForceVLA 60.5% vs π₀ 37.3% (+23.2%)
- USB插拔: 80%, 黄瓜削皮: 100%
- **视觉遮挡下: 90%** 成功率（纯视觉baseline 30-60%）
- 涌现行为：遇到异常阻力→自动回退→调整姿态→重试("retry"机制)

**局限：**
- F/T传感器需要改装（标准VLA平台不配F/T）
- MoE的late-fusion设计依赖任务特化
- 仅Flexiv Rizon单臂5任务验证

**ForceVLA2 (arXiv:2603.15169, 2026.03) — 后续工作：**
- 引入**显式混合力-位置控制**(Hybrid Force-Position)
- 将力控制回路从隐式学习→显式结构
- 提升力跟踪精度和接触稳定性

#### OmniVTLA (2025.08) — 语义对齐触觉

```
论文: OmniVTLA: Vision-Tactile-Language-Action Model with Semantic-Aligned Tactile Sensing
arXiv: 2508.08706
```

**技术路线：**
- 双路径触觉编码器（空间+时序）+ 语义对齐对比学习
- 让触觉信号在语义空间中与视觉/语言对齐
- VLM融合→动作生成

**关键结果：**
- 夹爪: 96.9%, 灵巧手: 100% pick-and-place
- 比纯VLA基线 +21.9%
- 利用ObjTac数据集(135K样本)

#### TLA / VTLA (2025.03 / 2025.05) — 语言条件触觉策略

```
TLA: arXiv:2503.08548 | VTLA: arXiv:2505.09577
团队: 中科院自动化所/三星/北京智源
```

**技术路线：**
- TLA: 触觉+语言→动作（无视觉），24K装配数据
- VTLA: 加回视觉，引入DPO(Direct Preference Optimization)作为回归式监督
- 触觉→跨模态语言对齐→语言条件策略生成

**关键结果：**
- 未见装配间隙和销钉形状: 85%+ (TLA), 90%+ (VTLA)
- Sim2Real: 真机UR3 + GelStereo 2.0 + 腕部相机

### 4.2 触觉感知的表示学习

#### T3 (CoRL 2024) — 可迁移触觉Transformer

```
论文: Transferable Tactile Transformers for Representation Learning Across Diverse Sensors and Tasks
arXiv: 2406.13640 | CoRL 2024 | MIT CSAIL
```

**技术路线：**
- **共享主干 + 传感器特定编码器 + 任务特定解码器**
- 将异构触觉传感器的输出映射到统一潜在空间
- 预训练→特定传感器/任务微调

**关键结果：**
- 零样本跨传感器迁移（部分配对成功）
- 亚毫米多针电子插入: +25% vs 从头训练，+53% vs 无触觉
- 规模定律：更大模型→更强性能

**局限：**
- 需要为每种新传感器设计编码器
- 训练需要大量多传感器数据
- 仅覆盖VBTS类传感器

#### Sparsh (Meta FAIR, 2024) — 通用触觉编码器

- **自监督预训练** on 460K GelSight图像
- 在所有下游任务上**平均>95%超越**任务特化模型
- 首个证明"触觉ImageNet"可行的工作

#### UniTacHand (北大+BeingBeyond, 2025.12)

- 人类触觉手套+机器人触觉手→统一到MANO UV图
- **对比学习对齐** → 仅需**10分钟配对数据**即可零样本迁移
- 混合人+机器人数据协训 > 仅机器人数据

### 4.3 触觉伺服与力反馈闭环控制

**触觉伺服(Tactile Servoing)是将触觉/力信号直接用于实时控制闭环的方法——不同于"感知→规划→执行"的开环范式。**

#### 经典与近期工作

| 论文 | 核心方法 | 控制频率 | 关键结果 |
|------|---------|---------|---------|
| **LeTac-MPC** (Xu & She, IEEE T-RO 2024) | 学习式MPC→GelSight触觉嵌入→可微分MPC层→夹爪调整 | **25Hz** | 从简单块→日常物体泛化 |
| **TacDiffusion** (TUM, 2024) | 扩散策略在力域操作→6D力矩命令→阻抗控制+前馈力→动态系统滤波器 | ~10Hz策略+高频控制 | 频率桥接(扩散慢vs控制快) |
| **FARM** (2025) | 预测目标夹爪宽度+夹持力+机器人姿态→双模控制器(位置↔力) | — | 触觉条件力域策略 |
| **Reactive Diffusion Policy** (2024) | 慢速视触扩散策略→子目标 | 慢策略+快触觉反应 | 频率不匹配的解决方案 |
| | 快速触觉反应控制器→实时接触调整 | | |
| **Fingertip Force Direction Control** (IEEE CASE 2024) | Xela触觉传感器→Rodrigues旋转公式→力矩级任务空间伺服 | 关节力矩级 | 指尖力方向精确控制 |
| **Energy Flow** (IEEE RA-L 2025) | 物理信息能量抽象→虚拟能量容器→滑动感知抓取力优化 | 实时 | 从零分钟级学习 |

**触觉伺服的关键挑战：**
1. **频率不匹配：** 触觉信号(100-1000Hz) vs 策略推理(10-50Hz) vs 视觉(30-60Hz)
2. **信号稀疏性：** 95%时间无接触→触觉信号为噪声
3. **传感器异构性：** 不同触觉传感器的输出格式完全不同

**当前解决方案趋势：**
- 快慢双通道(OmniVTA, Reactive Diffusion Policy)
- 力域策略(TacDiffusion, FARM)——直接在力空间优化
- 学习式MPC(LeTac-MPC)——用可微分MPC桥接学习与控制

### 4.4 从机械臂→人形的接触感知迁移

#### TACT (IEEE RA-L 2025) — 人形全身接触操作

```
论文: TACT: Humanoid Whole-Body Contact Manipulation through Deep Imitation Learning with Tactile Modality
arXiv: 2506.15146 | RHP7 Kaleido人形机器人
```

**技术路线：**
- **ACT (Action Chunking Transformer) + 触觉扩展**
- 关节位置 + 视觉 + **触觉**(上半身触觉传感器) → 动作
- 全身接触操作：**边走边用身体推箱子**

**关键结果：**
- 视觉+触觉 > 纯视觉，尤其在遮挡/接触密集阶段
- 首次在人形机器人上展示"边走边推"的全身接触操作

**局限：**
- 触觉传感器覆盖有限（仅上半身）
- 没有力/力矩传感器

#### Touch Dreaming (CMU+UTA+Bosch, 2026.04)

```
论文: Learning Versatile Humanoid Manipulation with Touch Dreaming
arXiv: 2604.13015
```

**技术路线（详细）：**
- **HTD (Humanoid Transformer with Touch Dreaming)**
- 多模态编码器-解码器Transformer
- **"触觉做梦"机制：**
  - 训练时：EMA Teacher预测**未来**手部关节力 + 未来触觉潜在表示
  - 学生模型学习这些预测作为辅助任务
  - 推理时：不额外计算，但表征已具备"接触预测"能力
- RL全身控制器(WBC) + VR遥操 + IK + 灵巧手 + 触觉

**关键结果：**
- 5个真机任务：+90.9% **相对**提升 vs ACT基线
- 潜在触觉"梦境" → +30% vs 原始触觉预测

**局限：**
- 需要RL控制器+VR遥操堆栈（系统工程复杂）
- 仅CMU一台机器人验证

#### HMC (UC San Diego, ICRA 2026) — 异构元控制：力矩空间的自适应混合

```
论文: HMC: Learning Heterogeneous Meta-Control for Contact-Rich Loco-Manipulation
arXiv: 2511.14756 | ICRA 2026
团队: UC San Diego (Xiaolong Wang组) — Lai Wei, Xuanbin Peng等
```

**核心问题：** 纯位置控制器无法处理变化的接触动力学(不同表面/负载/接触模式)。但是应该切换到哪种控制模态？阻抗？力控？混合？

**技术路线：**
- **HMC-Controller（力矩空间控制层）：**
  - 在**关节力矩空间**自适应混合三种控制模态：
    - 位置控制(PD跟踪)：适用于自由空间运动
    - 阻抗控制(关节空间+笛卡尔空间)：适用于与环境的柔性接触
    - 混合力-位置控制(自由轴位置+约束轴力)：适用于有明确约束的接触
  - 混合权重由高层策略动态输出
- **HMC-Policy（MoE架构）：**
  - 两阶段训练：大规模位置数据预训练(freeze非位置expert头)→力感知数据微调所有头
  - 软路由机制→不同任务阶段自动激活不同控制模态
- 真机：**Unitree G1人形机器人**

**关键结果：** 接触丰富任务（擦桌子/开门/抽屉）>**50%相对提升** vs 纯位置baseline

**为什么重要：** 与ForceVLA的"力增强VLA"不同——HMC关注的是"**在什么物理条件下应该使用什么控制模态**"，是更底层的控制架构创新。这是接触感知的下游——感知到什么类型的接触→自适应选择控制策略。

#### GentleHumanoid (2025.11) — 全身顺应：把顺从性写进RL奖励

```
论文: GentleHumanoid: Learning Upper-body Compliance for Contact-rich Human and Object Interaction
arXiv: 2511.04679
```

**技术路线：**
- **统一弹簧公式：** 同时建模**抵抗接触**(恢复力，如靠墙)和**引导接触**(推/拉，如人引导机器人)
- 将阻抗控制**整合进全身RL运动跟踪策略**(不是事后叠加)
- **安全力阈值**作为策略观测输入→ISO/TS 15066兼容
- Sim-to-real: teacher-student PPO (Isaac Gym/MuJoCo)→**Unitree G1**

**关键结果：**
- 峰值接触力 **降低50-60%** vs 刚性RL跟踪
- 验证：轻柔拥抱、起坐辅助、脆弱气球操作
- 控制率: 50Hz关节位置目标→底层PD

#### Embrace Contacts (CoRL 2025) — 非标准全身接触

- 处理**随机、不可预测的全身地面接触**(坐/翻滚/起身)——不仅是脚和手
- GPU加速碰撞检测→sim-to-real RL；极端躯干旋转下的实时控制

### 4.5 人形基础模型：从VLA到灵巧操作

#### GR00T N1系列 (NVIDIA, 2025-2026) — 行业基础设施

```
GR00T N1 (GTC 2025.03) → N1.5 (2025中) → N1.6 (CES 2026.01) → N1.7 (2026.04)
核心: 双系统认知 = VLM(System 2, 高层推理) + DiT(System 1, 实时动作30Hz)
```

**版本关键演进：**

| 版本 | VLM骨干 | 关键升级 | 接触相关能力 | 跨具身 |
|------|---------|---------|------------|--------|
| N1 | Eagle+SmolLM-1.7B | 首个开源人形VLA | 取放、双臂 | Fourier GR-1, 1X Neo |
| N1.5 | Eagle 2.5 | GR00T-Dreams合成数据(36h替代3月) | 工厂操作(零件运输/放置) | Unitree H1(吉利工厂) |
| N1.6 | Cosmos-2B | DiT×2(32层), 状态相对动作块, Jetson Thor | 跨具身Body Dynamics Atlas | Figure 03(BMW/Amazon) |
| **N1.7** | Cosmos-Reason2-2B/Qwen3-VL | **22-DoF灵巧手**, 相对EEF空间, **首个灵巧操作Scaling Law** | **小零件装配等接触丰富任务** | G1, YAM, AGIBot |

**关键披露（N1.7 — 与我们方向最相关）：**
- **EgoScale数据集：** 20,854小时人类第一人称视频，20+任务类别
- **Scaling Law：** 1K→20K小时人类视频→平均任务完成率**翻倍**
- **灵巧手控制：** 22-DoF手指级别→小零件装配

**GR00T与接触感知的关系：**
- GR00T是通用VLA——做"通用任务"而非"优化接触"
- N1.7开始涉足灵巧手接触操作——信号：接触感知从"可选"→"必需"
- 但GR00T用端到端学习(视觉→动作)，**没有显式力/触觉世界模型**
- **窗口：** GR00T + 接触世界模型 > GR00T alone

#### MotionWAM (2026.06) — 首个统一人形WAM
- 统一运动token替代上下半身分治，>30% vs VLA，Unitree G1实时

#### MotionDisco (2026.06) — 零遥操发现接触行为
- LLM引导进化+运动动力学优化→发现接触丰富行为→RL真机迁移

### 4.6 经典基础：装配力控制与Peg-in-Hole（必读）

**Peg-in-Hole为什么是接触感知的"ImageNet"：**
- 所有接触物理的缩影：自由→接近→接触→对齐→插入→卡阻/楔入
- 33%工业自动化装配操作→高工业相关性
- 小角度/轴向偏差→插入力剧增→零件损坏→需力感知和顺应控制

**经典综述(Xu et al., arXiv:1904.05240)：**
- **接触模型法(模型式)：** 接触状态识别(FSM/HMM)+顺应控制→可解释但需精确建模
- **无模型学习法(学习式)：** LfD(BC/IRL)或RL→数据驱动但无安全保证

**关键近期工作：**

| 论文 | 年份 | 核心创新 | 与我们方向的关系 |
|------|------|---------|----------------|
| **Divide et Impera** (MIT, 2024) | 2024(arXiv:2410.01054) | 6561阻抗参数/peg测试→**不存在唯一最优阻抗族**→多种都可成功 | 证明接触控制需要自适应(非固定参数) |
| **AugInsert** (2024, arXiv:2410.14968) | 2024 | 视觉+F/T多感官策略+在线数据增强→6类任务变体 | F/T数据对鲁棒性至关重要 |
| **Stability-Guaranteed RL** (2023) | 2023 | 首个All-Time稳定性保证RL→变阻抗控制(VIC) | 接触中的安全保证 |
| **POMDP Force Search** (IROS 2023) | 2023 | POMDP→不确定下主动力搜索孔 | 接触中的不确定性推理 |
| **Contact Model Survey** (Xu et al.) | 2019+ | 模型vs学习全分类 | 该领域的地图 |

---

## 5. 世界模型层：预测式接触推理

### 5.1 视觉-触觉世界模型

#### Dream-Tac (北大/港科大, 2026.06) — 统一触觉世界动作模型

```
论文: Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation
arXiv: 2606.08737
```

**技术路线（详细剖析）：**
- **基座：** Cosmos Policy（视频扩散WAM）→ 扩展触觉通道
- **Contact-Gated Visuotactile Fusion（接触门控融合）：**
  - 当接触事件概率低→视觉主导（触觉被降权）
  - 当接触事件概率高→触觉信号大幅增强（门控打开）
  - 门控信号：从触觉潜在特征中学习
- **CASA (Contact-Aware Self-Attention)：**
  - 在扩散Transformer的注意力层引入**接触感知偏置**
  - 偏置 = 触觉帧差异的强度 → softmax($QK^T/\sqrt{d} + Bias_{contact}$)
  - 实质：在接触时刻，让触觉token的注意力权重大幅增加
- **FlashBias加速：** 将接触偏置融合到FlashAttention kernel中 → 训练2.9× 推理1.8×

**关键结果（最详细的实验对比）：**
| 方案 | 平均成功率 |
|------|-----------|
| 纯视觉WAM (Cosmos Policy) | 51.7% |
| 视觉+触觉 Naive Fusion | 74.2% |
| **Dream-Tac (Visuotactile + CASA)** | **83.3%** |
| → 提升 | +31.6% vs 纯视觉, +9.1% vs Naive |

**关键消融：**
- CASA bias > 接触门控 > 触觉通道的简单添加
- 在接触转换瞬间（自由→接触、滑动→卡住），CASA的效果最大

**代码：** 已开源 github.com/LYFCLOUDFAN/Dream-Tac

#### DreamTacVLA (西北大学, 2025.12) — 层次化感知+触觉世界模型

```
论文: Learning to Feel the Future: DreamTacVLA for Contact-Rich Manipulation
arXiv: 2512.23864
```

**技术路线（与Dream-Tac不同）：**
- **三级层次化感知：**
  - Macro: 第三人称RGB
  - Local: 腕部相机
  - Micro: 高分辨率触觉图像
- **HSA (Hierarchical Spatial Alignment)：**
  - 将触觉token与视觉token在空间上对齐
  - 利用运动学+相机标定→InfoNCE对比损失
- **Think-Dream-Act闭环：**
  - Think: VLA生成草稿动作
  - Dream: 触觉世界模型预测未来触觉状态
  - Act: 真实观测+"梦境"→精化最终动作

**关键结果：**
- 四项精密任务（Peg-in-Hole/USB Insert/Gear Assembly/Tool Stabilize）: **平均95%**
- **仅用20%训练数据**→超越全量训练baseline

**与Dream-Tac的关键区别：**
| | Dream-Tac (北大) | DreamTacVLA (西北) |
|---|---|---|
| 范式 | 扩散(基于Cosmos Policy) | VLA(基于LLM) |
| 触觉使用 | CASA注意力偏置+门控 | 空间对齐+未来触觉预测 |
| 预测目标 | 联合去噪未来帧+触觉+动作 | 触觉嵌入预测→动作精化 |
| 加速 | FlashBias+缓存 | Think-Dream-Act两次前传 |
| 开源 | ✅ | ❌ |

#### OmniVTA (TARS+6机构, 2026.03)

```
论文: OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation
arXiv: 2603.19201
```

**技术路线（核心架构）：**

```
慢通道(~4Hz):
  视觉观测 → TactileVAE(触觉编码) → 视觉-触觉世界模型(双流扩散)
  → 预测未来接触状态 → 自适应融合策略(LTD编码器+门控)

快通道(60Hz):
  触觉观测 → 反射式触觉控制器(RLTC)
  → 比较预测vs实际触觉 → 实时修正
```

**关键创新点：**
1. **TactileVAE：** 自监督时空编码器→压缩3D触觉变形到低维潜在表示
2. **Visuo-Tactile World Model：** 双流扩散→**联合建模视觉+触觉时序演化**
3. **LTD (Latent Tactile Differential)：** 触觉潜在差分的编码器→动态门控→权重自适应视觉vs触觉
4. **RLTC (Reflexive Tactile Controller)：** 受人类"触觉反射弧"启发→60Hz闭环→预测vs观察的偏差检测→实时动作修正

**为什么这是范式级贡献：**
- 之前的工作：触觉 = 被动感知（"摸到了告诉我"）
- OmniVTA: 触觉 = 主动预测+闭环（"我预测到即将接触→预先调整→实际接触后验证→修正"）
- 类比：这是触觉领域的"前向模型+卡尔曼滤波"融合

**关键结果：**
- 6类交互模式全部SOTA
- 强泛化到未见物体/几何/工具/位置
- 学习到了"可迁移的接触动力学"

**数据集OmniViTac:** 21,879轨迹, 126物体, 获评EAI-2025 Top10数据集

#### TacForeSight (TARS, ICRA 2026)

```
论文: TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation
arXiv: 2606.11184 | ICRA 2026
```

**核心升级（vs OmniVTA）：**
- 输入从纯触觉→**腕部F/T+双指触觉**的多尺度力信号
- **TacForceWM：** 力条件触觉世界模型
  - 条件变量：腕部F/T（低频全局力）
  - 预测目标：指端触觉潜在动力学（高频局部接触）
  - 核心insight：腕部力和指尖触觉有**互补的时空角色**——一个告诉你"力有多大"，一个告诉你"力在哪里"
- 预测性触觉条件策略→交叉注意力→当前→未来触觉演变
- **20Hz实时推理** → 进入高频控制闭环

**关键结果：**
- 5个标准接触任务平均79.0%完成率
- 扰动鲁棒性：高度/角度/姿态扰动下85-90%
- 部署于TARS DexHand灵巧手

---

### 5.2 WAM与力/触觉的融合趋势总结

```
演进路径：

2024前: 触觉=独立模态(分类/重建)
  ↓
2024-2025: 触觉+VLA融合(ForceVLA, TLA, OmniVTLA)
  ├── 融合方式: Late-fusion MoE / 语义对齐 / 对比学习
  └── 局限: 触觉仍是"感知输入"，不是"预测对象"
  ↓
2025底-2026: 触觉+世界模型(Dream-Tac, OmniVTA, TacForeSight)
  ├── 融合方式: 双向扩散 / 快慢通道 / 力条件预测
  └── 创新: 触觉成为"预测对象"——预测未来的接触状态
  ↓
2026+: 缺失项——人形全身接触世界模型
  ├── 现有局限: 所有工作在单臂/指尖/夹爪上
  └── 未解决: 全身多接触点的时空协调预测
```

---

## 6. 训练范式层：从BC到RL到Sim-to-Real

### 6.1 行为克隆(BC)及其变体

| 方法 | 核心思想 | 代表工作 | 优势 | 局限 |
|------|---------|---------|------|------|
| **BC (朴素)** | 监督学习模仿专家动作 | — | 简单 | 分布漂移、累积误差 |
| **ACT** | Transformer分块预测 | TACT, Touch Dreaming | 时序一致性好 | 需要chunk size调参 |
| **Diffusion Policy** | 扩散模型生成动作 | OmniVTA, Dream-Tac | 多模态动作分布 | 推理慢 |
| **Flow Matching** | 连续归一化流 | ForceVLA, DreamZero | 更快采样、更稳定 | 模型设计复杂 |

### 6.2 强化学习(RL)在接触操作中的应用

| 方法 | 场景 | 代表工作 | 关键设计 |
|------|------|---------|---------|
| **PPO+触觉** | 抓取+灵巧操作 | DexSim2Real | 触觉-视觉交叉注意力策略(78.2%真机) |
| **SAC+域随机化** | 指尖操作 | Kasolowsky 2024(IROS) | 关节偏移+摩擦+弹性+寄生刚度域随机化 |
| **多Agent RL** | 力交换协助 | AssistMimic(CVPR 2026) | 伙伴感知策略+接触促进奖励 |
| **Adversarial RL** | 抓取合成 | Chen 2026(ICRA) | 零和博弈：手 vs 物体(75.78%仿真, 85-87.5%真机) |

### 6.3 Sim-to-Real for Contact

| 方法 | 核心策略 | 代表工作 | 关键结果 |
|------|---------|---------|---------|
| **经典域随机化** | 物理参数随机化 | Kasolowsky 2024 | 子taxel精度(<1mm) sim-to-real |
| **FM-DR** | VLM作为域随机化"视觉真实性Critic" | DexSim2Real 2026 | 仅8.3% sim-to-real gap |
| **并行FK触觉模拟** | 快速触觉仿真 | Zhao 2026 | 五指灵巧手零样本sim-to-real(首次) |
| **DrEureka** | LLM自动设计域随机化 | — | 自动化 → 减少人工tuning |
| **混合仿真** | MuJoCo(物理)+Isaac Sim(渲染) | SIMPLE 2026 | 人形全身操作sim-to-real |

### 6.4 辅助训练目标

| 技术 | 如何工作 | 代表工作 | 效果 |
|------|---------|---------|------|
| **触觉做梦(Touch Dreaming)** | 预测未来触觉潜伏/力作为辅助任务 | Touch Dreaming, TacForeSight | +30% vs 原始触觉预测 |
| **接触门控(Contact Gating)** | 根据接触状态动态调节触觉权重 | Dream-Tac | +9.1% vs Naive融合 |
| **自蒸馏RL** | 冻结世界模型→仅优化策略头 | AIM | 防止RL破坏世界模型 |
| **Teacher-Student触觉对齐** | EMA Teacher预测未来→学生蒸馏 | Touch Dreaming, DreamTacVLA | 时序一致性增强 |

---

## 7. 推理层：部署与实时性

### 7.1 推理架构对比

| 架构 | 频率 | 延迟 | 代表工作 | 适用场景 |
|------|------|------|---------|---------|
| **单次前传** | 10-50Hz | 20-100ms | ForceVLA, TACT | 标准VLA |
| **快慢双通道** | 慢4Hz+快60Hz | 慢250ms+快16ms | OmniVTA | 接触丰富操作 |
| **Think-Dream-Act** | ~3Hz+~3Hz+~3Hz | ~300ms总 | DreamTacVLA | 精密操作 |
| **单步去噪** | 7Hz | 150ms | DreamZero-Flash | WAM实时控制 |
| **边缘推理** | ~285Hz | 3.5ms | Being-H0.7(Jetson Orin) | 端侧部署 |
| **本体推断接触** | **2000Hz** | 0.5ms | UniTac(RTX 3090) | 实时碰撞检测 |

### 7.2 部署挑战

| 挑战 | 当前状态 | 代表方案 |
|------|---------|---------|
| 触觉+WAM联合推理慢 | Dream-Tac用FlashBias加速2.9× | 专用kernel |
| 人形多传感器带宽 | 猿声HexSkin→片上处理→数字总线 | 分布式边缘计算 |
| 真机实时控制闭环 | TacForeSight 20Hz, RLTC 60Hz | 分级频率 |
| 模型压缩 | Being-H0.7 3B → Jetson Orin | 知识蒸馏+潜在空间 |

---

## 8. 评测层：Benchmark体系

### 8.1 触觉传感器Benchmark

| Benchmark | 发布 | 评测维度 | 传感器覆盖 |
|-----------|------|---------|-----------|
| **TacEva** | 2025.09 | 力精度/定位精度/空间分辨率/灵敏度/重复性 | ViTacTip, MagicTac, GelSight, GelSightWM |
| **ManiFeel** | 2025 | 触觉表示对策略的影响 | GelSight Mini |
| **Tactile Gym 2.0** | 2024 | 多种VBTS的sim-to-real对比 | TacTip, DIGIT, DigiTac |

### 8.2 接触丰富操作Benchmark

| Benchmark | 任务 | 主要评测 | 备注 |
|-----------|------|---------|------|
| **NIST Task Board 1** | 17物体装配 | 成功率, 完成时间, 力峰值 | 工业标准 |
| **ManiSkill-ViTac 2025** | Track1纯触觉/Track2融合/Track3硬件 | 仿真+真机 | 社区比赛 |

### 8.3 评测维度缺失

当前benchmark主要评测**任务成功率**，以下维度严重缺失：
- **接触稳定性**(力/力矩的平滑度、振荡)
- **接触安全性**(碰撞力、安全裕度)
- **接触效率**(最小接触力完成任务)
- **泛化的接触适应性**(新材质/新几何下的力调节)
- **物理可解释性**(为什么选择这个接触策略)

---

## 9. 当前不足与开放问题：系统总结

### 9.1 "七层不足"全景

```
L7: 接触因果理解       ❌ 完全空白（"为什么这个接触策略有效？"）
L6: 多Agent物理接触协调  ❌ 完全空白（两人/人-机协作时力如何分配？）
L5: 人形全身多接触协调   ❌ 几乎空白（TACT仅上半身推箱子）
L4: 接触预测+实时修正    ⚠️ 刚起步（OmniVTA/TacForeSight，仅单臂/指尖）
L3: 力/触觉增强WAM      ⚠️ 少量工作（Dream-Tac/DreamTacVLA，仅夹爪/指尖）
L2: 力/触觉增强VLA      ⚠️ 数个工作（ForceVLA/TLA，数据/传感器仍是瓶颈）
L1: 触觉硬件             ✅ 成熟（GelSight/DIGIT/全身皮肤已商业化）
```

### 9.2 针对每一层的具体不足

**L1 硬件层不足：**
- 全身触觉密度远低于人手（人手~17,000个机械感受器，现有最好~2,000/手）
- 触觉+力传感器+本体感觉的**时空同步精度**不足（微秒级→目前毫秒级）
- 灵巧手+触觉+腕部F/T+视觉的统一硬件平台稀缺

**L2-L3 算法层不足：**
- ForceVLA的F/T数据难以扩展到大规模（需要改装机器人）
- 触觉表示没有统一标准（GelSight/DIGIT/TacTip各自格式）
- 力/触觉信号的**时序建模**远落后于视觉（大多数工作只做空间建模）
- **HMC揭示了控制模态需要自适应混合**——但GR00T/DreamZero等基础模型仍使用单一控制模态
- GR00T N1.7虽有灵巧手但**缺乏显式的接触世界模型**——端到端学习无法解释"为什么这个力是合适的"

**L4 预测层不足：**
- 当前世界模型预测的是"视频帧"——不包含力/力矩/接触状态的直接预测
- Dream-Tac是唯一联合预测视觉+触觉的工作，但只预测触觉帧（而非接触力物理量）
- 没有人预测"关节力矩"或"接触压力场"的时空演化

**L5 人形层不足：**
- 所有触觉WAM工作在**固定基座机械臂**上——人形的浮动基座+平衡约束完全未涉及
- 人形全身操作时，手臂的接触力会影响**下肢平衡**——当前无工作建模这个耦合
- 人形"边走边做"时的接触感知完全空白（TACT只走了几步+推箱子）

**L6 多Agent层不足：**
- 当两个Agent操作同一物体时，力通过物体传导→如何"读"出对方的意图？
- 没有多Agent的接触感知数据集
- 力博弈/冲突解决在接触层面完全空白

**L7 因果层不足：**
- 无人能回答："为什么用这个力？为什么选这个接触点？"
- 从WAM预测到动作选择的因果链不透明
- 接触失败后无法归因（是力不够？方向不对？摩擦不足？）

---

## 10. 参考文献汇总

### 经典基础（必读）
- Hogan, "Impedance Control: An Approach to Manipulation," ASME JDSMC, 1985
- Raibert & Craig, "Hybrid Position/Force Control of Manipulators," ASME JDSMC, 1981
- Mason, "Compliance and Force Control for Computer Controlled Manipulators," IEEE SMC, 1981
- Haddadin et al., "Robot Collisions: A Survey on Detection, Isolation, and Identification," IEEE TRO, 2017

### 触觉传感器与数据集
- GelSight系列 (Johnson & Adelson, 2009 → GelSight Mini 2023)
- DIGIT (Lambeta et al., 2020) → Digit 360 (2024)
- TacTip系列 (Lepora et al., 2018 → ViTacTip 2025)
- T3/FoTa (Zhao et al., CoRL 2024)
- OmniViTac (Zheng et al., 2026.03)
- REASSEMBLE (Sliwowski et al., RSS 2025)
- FreeTacMan (Wu et al., 2025.06)

### VLA + 力/触觉
- ForceVLA (Yu et al., NeurIPS 2025, 2505.22159)
- ForceVLA2 (2603.15169)
- OmniVTLA (2508.08706)
- TLA (2503.08548) / VTLA (2505.09577)
- TACT (2506.15146)

### 世界模型 + 触觉
- Dream-Tac (Lou et al., 2606.08737)
- DreamTacVLA (Ye et al., 2512.23864)
- OmniVTA (Zheng et al., 2603.19201)
- TacForeSight (Zang et al., ICRA 2026, 2606.11184)

### 触觉表示与迁移
- T3 (Zhao et al., CoRL 2024)
- Sparsh (Meta FAIR, 2024)
- UniTacHand (2512.21233)

### 触觉伺服与力控制闭环
- LeTac-MPC (Xu & She, IEEE T-RO 2024)
- TacDiffusion (TUM, 2024)
- FARM (2025)
- Reactive Diffusion Policy (2024)
- Fingertip Force Direction Control (IEEE CASE 2024)
- Energy Flow Grasping (IEEE RA-L 2025)

### 人形全身操作
- Touch Dreaming (CMU, 2604.13015)
- TACT (2506.15146)
- SIMPLE (2606.08278)
- Humanoid Everyday (2510.08807)
- HMC (Wei et al., ICRA 2026, 2511.14756)
- GentleHumanoid (Lu et al., 2511.04679)
- Embrace Contacts (Zhuang & Zhao, CoRL 2025)

### 人形基础模型
- GR00T N1/N1.5/N1.6/N1.7 (NVIDIA, 2025-2026)
- MotionWAM (2606.09215)
- MotionDisco (2606.06139)

### 经典装配力控制
- Divide et Impera (Lachner et al., MIT, 2024, 2410.01054)
- AugInsert (Diaz et al., 2024, 2410.14968)
- Stability-Guaranteed RL (Khader et al., 2023)
- Contact Model Survey (Xu et al., 2019, 1904.05240)

### Sim-to-Real
- DexSim2Real (2605.05241)
- Closing the Reality Gap (2601.02778)
- Kasolowsky & Bäuml (IROS 2024)

### 本体感觉力估计（无需力传感器）
- UniTac (Brown, 2507.07980)
- SixthSense (2605.01427)
- PRIME (2605.17681)

### 多Agent与博弈
- LatentToM (He et al., CoRL 2025, 2505.09144)
- AssistMimic (CVPR 2026, 2603.11346)
- Adversarial Grasp (ICRA 2026, 2511.05809)
