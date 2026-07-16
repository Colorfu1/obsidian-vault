---
title: DreamZero 技术报告
type: paper_note
topic: world_action_model
status: mature
importance: high
updated: 2026-07-16
tags:
  - dreamzero
  - world-action-model
  - video-diffusion
  - flow-matching
  - action-chunking
  - robotics
---

# DreamZero 技术报告

> 基于论文 **World Action Models are Zero-shot Policies**（NVIDIA，arXiv:2602.15922v1，2026-02-17）及围绕论文架构、训练目标、注意力掩码、实时执行和跨机器人泛化的技术讨论整理。
>
> 报告版本：1.0
>
> 术语约定：本文保留 `video latent`、`action chunk`、`embodiment`、`flow matching`、`KV cache` 等英文术语，以避免中文翻译造成歧义。

---

## 摘要

DreamZero 是一个建立在 14B 预训练图生视频扩散模型之上的 **World Action Model（WAM）**。它不再只学习从视觉和语言直接预测机器人动作，而是在同一个扩散 Transformer 中联合预测：

1. 当前观测之后的视觉未来；
2. 与该视觉未来相对应的连续机器人动作。

其核心假设是：大规模视频预训练已经学习了相当一部分物体运动、接触、遮挡和场景变化规律；机器人训练只需进一步学习如何把这些视觉未来与特定机器人的动作空间对齐。DreamZero 因而可以被理解为：

> **以视频生成模型为主体、通过联合 video-action flow matching 实现的端到端闭环机器人策略。**

但需要准确限定它的能力范围：DreamZero 不是“一套冻结参数直接控制任意机器人”的通用控制器。论文对 AgiBot G1 和 Franka 分别训练模型，并明确把 multi-embodiment joint training 留给未来工作。其“zero-shot”主要指在固定目标机器人上，对未见任务、未见动作语义、未见物体和未见环境进行零样本执行，而不是不使用机器人数据，也不是零样本切换机器人。

论文的主要结果包括：

- AgiBot 已见任务、未见环境/物体：平均任务进度 62.2%，高于最佳预训练 VLA 基线的 27.4%；
- AgiBot 未见任务：平均任务进度 39.5%，高于最佳预训练 VLA 基线的 16.3%；
- DROID-Franka 未见动词：49% 任务进度、22.5% 完整成功率；
- 仅加入 12 分钟人类视频或 20 分钟其他机器人视频，未见任务进度由 38.3% 提升到 54.3%/55.4%；
- 使用约 30 分钟目标机器人 play data，可从 AgiBot checkpoint 适配到 YAM；
- 通过系统优化和 DreamZero-Flash，将推理从约 5.7 秒降到约 150 毫秒，实现约 7 Hz 的策略更新。

---

## 1. 研究问题与方法定位

### 1.1 VLA 的局限

传统 Vision-Language-Action（VLA）模型通常从视觉语言模型初始化，并学习：

$$
\pi(a_{t:t+H}\mid o_t,c,q_t),
$$

其中：

- $o_t$：当前视觉观测；
- $c$：语言指令；
- $q_t$：机器人 proprioceptive state；
- $a_{t:t+H}$：未来动作块。

这类模型善于继承图文预训练中的语义知识，例如识别物体、理解类别和解析语言，但静态图文预训练不直接提供“物体接下来怎样运动、接触后怎样变化、工具怎样作用于对象”等连续动力学先验。

DreamZero 的出发点是：视频模型在预训练中得到的时空先验，比静态 VLM 更接近机器人所需的物理运动表示。

### 1.2 WAM 的目标

DreamZero 联合建模：

$$
\pi(o_{t:t+H},a_{t:t+H}\mid o_{0:t},c,q_t).
$$

论文把它概念性地分解为：

$$
\underbrace{\pi(o_{t:t+H}\mid o_{0:t},c,q_t)}_{\text{video prediction}}
\cdot
\underbrace{\pi(a_{t:t+H}\mid o_{0:t+H},q_t)}_{\text{implicit inverse dynamics model}}.
$$

该分解表达的是一种解释：

1. 先在表示层面形成“世界应该怎样变化”的视觉未来；
2. 再把视觉未来与当前机器人的动作对应起来。

不过，**实际模型不是“独立视频生成器 + 独立 IDM”的两阶段 pipeline**。视频 latent 和 action latent 被送入同一个 DiT，并在同一次去噪过程中联合预测。所谓隐式 IDM 是对共享模型内部作用机制的解释，而不是一个可独立调用或单独评测的模块。

### 1.3 DreamZero 是什么，不是什么

| 判断 | 结论 |
|---|---|
| 基于预训练视频生成模型的机器人策略 | 是 |
| 联合预测视频和动作 | 是 |
| 端到端闭环控制策略 | 是 |
| 显式生成多个候选未来并用 reward/value 选择 | 否 |
| 使用 MPC、搜索或测试时优化动作 | 否 |
| 只靠互联网视频、不使用机器人动作数据 | 否 |
| 一套冻结参数直接控制所有机器人 | 没有证明 |
| 固定 embodiment 上的任务/环境 generalist | 是 |
| 少量目标机器人数据适配新 embodiment | 是 |

因此，DreamZero 更接近“带视觉未来生成目标的端到端 policy”，而不是 Dreamer、PlaNet 或 visual foresight 中那种显式 model-based planner。

---

## 2. 模型架构

### 2.1 输入与编码器

模型接收三类条件：

- 多视角视觉观测；
- 自然语言指令；
- proprioceptive state。

主要组件为：

$$
\text{Image}\rightarrow\text{VAE latent},
$$

$$
\text{Language}\rightarrow\text{Text Encoder},
$$

$$
q_t\rightarrow\text{State Encoder},
$$

$$
a_t\rightarrow\text{Action Encoder}.
$$

随后，视频 token、动作 token、语言条件和 state condition 一起进入一个 joint video-action diffusion Transformer。模型分别通过 video decoder 和 action decoder 输出两种模态的 flow velocity。

DreamZero 从 **Wan2.1-I2V-14B-480P** 初始化。训练时：

- 更新全部 DiT blocks；
- 更新 state encoder、action encoder 和 action decoder；
- 冻结 text encoder、image encoder 和 VAE。

对于多摄像头数据，论文没有设计专门的多视角几何融合模块，而是把不同视角在空间上拼接成一张图像，再交给原视频 backbone。

### 2.2 单 embodiment 参数化

论文对 AgiBot G1 和 Franka **分别进行预训练**：

$$
\theta_{\text{Wan}}
\xrightarrow{\text{AgiBot data}}
\theta_{\text{DreamZero-AgiBot}},
$$

$$
\theta_{\text{Wan}}
\xrightarrow{\text{DROID/Franka data}}
\theta_{\text{DreamZero-Franka}}.
$$

这意味着：

- action/state encoder 和 decoder 需要与目标机器人的动作空间、关节定义、控制频率和归一化规则匹配；
- 同一个视觉未来可能对应不同机器人的不同控制命令；
- 视频世界知识具有一定跨 embodiment 可迁移性，但隐式 inverse dynamics 显著依赖 embodiment。

论文中的跨 embodiment 结果分为两类：

1. **video-only skill transfer**：用人类或 YAM 视频增强 AgiBot 的视觉世界知识，最终仍输出 AgiBot 动作；
2. **new embodiment adaptation**：用约 30 分钟带目标动作的 YAM play data 对 AgiBot checkpoint 做 post-training，得到适配后的 YAM policy。

二者都不等于“一套冻结权重同时控制多个机器人”。

---

## 3. 时间组织：Diffusion、Bidirectional 与 Autoregressive

这是理解 DreamZero 最关键的部分。

### 3.1 两个相互独立的“方向”

需要区分：

1. **扩散时间方向**：从噪声状态逐步积分到 clean sample；
2. **物理时间 token 的 attention 方向**：不同帧、不同动作步之间谁可以看谁。

论文讨论 bidirectional 与 autoregressive 时，主要指第二种，而不是说 diffusion 本身是双向或单向。

### 3.2 DreamZero 不是逐帧生成

DreamZero 不是一次只预测一帧。它采用 **chunk-wise generation**：

- 一个 chunk 内同时生成若干视频 latent frame；
- 同时生成覆盖相同物理时间的 action chunk；
- 不同 chunk 之间按照时间因果关系 autoregressive 地推进。

默认配置：

| 参数 | AgiBot | DROID-Franka |
|---|---:|---:|
| 视频采样率 | 5 FPS | 5 FPS |
| 动作控制频率 | 30 Hz | 15 Hz |
| 每个 chunk 的 action horizon | 48 | 24 |
| 每个 chunk 覆盖时间 | 1.6 s | 1.6 s |
| 每个 chunk 的视频 latent frame 数 $K$ | 2 | 2 |
| 最大 chunk 数 $M$ | 4 | 4 |
| 最大视觉历史 | 约 6.6 s | 约 6.6 s |

在 AgiBot 中，一个 1.6 秒 chunk 在 5 FPS 下约覆盖 8 帧原始视频，同时对应 48 个 30 Hz 动作。论文还指出 8 个 latent frame 的最大上下文约对应 33 个 raw frame。

### 3.3 “Chunk 内 bidirectional，chunk 间 autoregressive”

使用论文 Figure 14 的记号：

- $C_i$：clean visual context；
- $Z_i$：第 $i$ 个待去噪视频 chunk；
- $Y_i$：第 $i$ 个待去噪动作 chunk。

模型依赖关系可写成：

```text
C0             -> [ Z1 <-> Y1 ]
C0, C1         -> [ Z2 <-> Y2 ]
C0, C1, C2     -> [ Z3 <-> Y3 ]
```

方括号内部表示当前 chunk 的 noisy video/action token 可以互相 attention：

- 当前 chunk 的早期视频 token 可以看到晚期视频 token；
- action token 可以看到同一 chunk 中其他 action token；
- 视频和动作 token 可以双向交换信息。

这就是 **chunk 内 bidirectional self-attention**。它们看到的是当前扩散步的 noisy target token，而不是 clean future label，因此不构成标签泄漏。

跨 chunk 则是 causal 的：$Z_2,Y_2$ 只能使用 $C_0,C_1$，不能使用未来 $C_2$，也不会把上一 chunk 的 noisy prediction 作为可靠历史。

### 3.4 Context 不需要预测，但需要产生 KV

$C_0,C_1,C_2$ 是条件 token：

- 不加噪；
- 不预测 flow；
- 不承担当前 target loss；
- 但要经过 Transformer 生成每层 Key/Value，供当前 noisy chunk 查询。

例如：

$$
Q_{Z_2,Y_2}K_{C_0,C_1}^{\top}.
$$

Figure 14 中，clean context 不反向依赖当前 noisy target。这样做有两个作用：

1. 保持训练和推理一致：推理中的新 context 来自真实摄像头，而不是上一 chunk 的预测视频；
2. 允许 KV cache：context 表示不随每个 diffusion step 的 noisy target 改变，可以只计算一次。

### 3.5 论文批评的“bidirectional diffusion”究竟是什么

论文所批评的不是 diffusion 数学目标天然要求固定长度，而是常见 **fixed-clip bidirectional video diffusion architecture**：

$$
C_0\rightarrow[Z_1,Y_1,Z_2,Y_2,\ldots,Z_n,Y_n],
$$

整个固定长度未来 clip 在一次扩散过程中联合去噪，并在目标时间轴上全局双向 attention。

如果语言指令描述一个较长任务，而模型一次只能生成固定数量的视频帧，就面临两个选择：

1. 保持原始 FPS，只覆盖任务中的一个短片段，造成语言与视频覆盖范围不一致；
2. 用固定帧数覆盖完整任务，只能强时间下采样，导致原始 FPS 被扭曲，接触和精细运动与高频动作难以局部对齐。

DreamZero 并没有消除“每次扩散生成固定长度”这件事，而是把它局部化：

> 仍然每次生成固定长度 chunk，但通过跨 chunk 的 autoregressive 闭环反复生成，从而保持局部 native FPS。

因此，这一问题不是 diffusion 本身不可避免的性质，也可以通过 variable-length mask、padding、hierarchical generation 等其他设计缓解；论文的论述更准确地说是对常见固定 clip 双向架构的工程批评。

---

## 4. 联合 Flow Matching 训练

### 4.1 加噪与目标速度

对于第 $k$ 个 chunk，视频 latent 和动作分别写为：

- clean target：$z_1^k,a_1^k$；
- Gaussian noise：$z_0^k,a_0^k$；
- flow timestep：$t_k\in[0,1]$。

线性插值：

$$
z_{t_k}^k=t_kz_1^k+(1-t_k)z_0^k,
$$

$$
a_{t_k}^k=t_ka_1^k+(1-t_k)a_0^k.
$$

其中：

- $t=0$ 表示纯噪声；
- $t=1$ 表示 clean sample。

目标 velocity 为：

$$
v^k=[z_1^k,a_1^k]-[z_0^k,a_0^k].
$$

模型输出：

$$
u_\theta([z_{t_k}^k,a_{t_k}^k];\mathcal C_k,c,q_k,t_k).
$$

### 4.2 多 chunk 平均损失

使用更一致的记号，若一条训练轨迹包含 $M$ 个 chunk，则损失可写为：

$$
\mathcal L(\theta)
=
\mathbb E
\left[
\frac{1}{M}
\sum_{k=1}^{M}
 w(t_k)
\left\|
 u_\theta([z_{t_k}^k,a_{t_k}^k];\mathcal C_k,c,q_k,t_k)-v^k
\right\|_2^2
\right].
$$

对多个 chunk 求平均的原因是：

- 一条长轨迹中的每个阶段都构成一个训练目标；
- 每个阶段具有不同 clean context 和不同 noise timestep；
- 通过 block-causal mask，可以在一次前向中并行计算多个 chunk 的 loss，同时保持概率分解是 causal 的。

论文正文公式使用 $K$ 作为求和上限，但附录又把 $K=2$ 定义为“每个 chunk 的 latent frame 数”，把 $M=4$ 定义为“chunk 数”。因此公式中的求和上限更可能应当是 $M$，属于符号复用或笔误。

### 4.3 视频与动作不要求相同帧率

联合 loss 不要求：

$$
N_{\text{video}}=N_{\text{action}}.
$$

它要求的是：

$$
\text{video chunk 覆盖的物理时间区间}
=
\text{action chunk 覆盖的物理时间区间}.
$$

AgiBot 的一个 chunk 例如同时包含：

- 约 1.6 秒、5 FPS 的视觉数据，编码成 2 个 video latent frame；
- 同一 1.6 秒内的 48 个 30 Hz 动作。

因此对齐是时间区间对齐，而不是一帧图像对应一个动作。共享 attention 负责学习一对多、局部连续的跨模态关系。

论文没有充分交代以下实现细节：

- 相机 timestamp 与控制 timestamp 的硬件同步方式；
- 视频采样使用最近邻、插值还是其他重采样；
- video/action position encoding 是否共享真实物理时间；
- 同一 chunk 内是否有更细粒度的局部时间 mask；
- video loss 和 action loss 是否分别归一化或加权。

最后一点尤其重要：video latent 维数远大于 action。如果直接拼接后对所有元素求和，视频项可能主导总 loss。公式没有明确说明模态级归一化或权重，复现时需要检查代码。

### 4.4 Teacher forcing

第 $k$ 个 chunk 的历史条件使用 clean ground truth context，而不是模型之前生成的预测视频。普通 autoregressive video generation 中，这会引入 exposure bias；DreamZero 的闭环系统则有一个特殊优势：每执行完一个 action chunk，就能获得新的真实相机观测，并把真实观测写回 KV cache。

因此训练时使用 clean context 与推理时使用 real observation 在结构上比较一致。

不过，论文公式把历史写成：

$$
\mathcal C_k=\{(z_1^j,a_1^j)\}_{j<k},
$$

似乎包括历史 clean action；Figure 14 和推理伪代码则主要把历史建模为 clean visual context，更新 KV cache 时也只注入真实视频 latent。这一训练/推理的 action-history 细节需要结合代码确认。

---

## 5. 闭环推理

### 5.1 推理流程

每次控制循环大致执行：

1. 将当前真实图像通过 VAE 编码为 clean visual latent；
2. 把 clean context 的 K/V 写入缓存；
3. 从 Gaussian noise 初始化当前 chunk 的 video latent 和 action latent；
4. 用 flow solver 联合去噪视频和动作；
5. 取出 clean action chunk；
6. 对 action chunk 进行平滑；
7. 异步发送给机器人；
8. 获得新的真实图像和 proprioceptive state；
9. 用新的真实 visual latent 更新 KV cache；
10. 丢弃当前 chunk 生成的视频 latent。

简化表示：

```text
real observation
      |
      v
VAE -> clean context KV
      |
noise video + noise action
      |
      v
joint video-action denoising
      |
      +------> predicted video latent（用于当前联合推理/可视化）
      |
      +------> action chunk -> smoothing -> robot execution
                                      |
                                      v
                              new real observation
```

### 5.2 预测视频是否需要 decode 后再 encode

不需要。DreamZero 不执行：

$$
z^{\text{pred}}
\rightarrow\text{VAE decoder}
\rightarrow\hat o
\rightarrow\text{VAE encoder}
\rightarrow\hat z.
$$

算法 2 明确写出：动作执行后，使用真实观测编码得到 $z_{\text{real}}$，并 **discard predicted video latent**。

因此，预测视频 latent 的作用主要是：

1. 在当前 chunk 的联合去噪中作为隐式视觉计划，与 action token 深度交互；
2. 在训练中提供密集时空监督，保持视频 backbone 的物理先验；
3. 用于可视化和分析 video-action alignment。

执行完成后，真实世界已经可能偏离预测未来；用真实观测替换预测视频可以避免跨 chunk 的视觉误差累积。

### 5.3 它不是显式 planner

DreamZero 没有：

- 采样多个候选视频；
- 用 reward/value 对候选视频打分；
- 搜索最优轨迹；
- 在执行前通过 MPC 反复优化动作。

它一次条件生成一组视频和动作，并直接执行动作。因此“visual planning”更准确的含义是：

> 单次联合生成中的隐式视觉未来表示。

---

## 6. 实时执行与系统优化

### 6.1 Reactivity gap

未经优化的 14B video diffusion policy 每个 action chunk 约需 5.7 秒，主要瓶颈包括：

- 16 个 diffusion/flow solver step；
- 14B DiT 的计算量；
- 推理和动作执行串行进行。

AgiBot 的 action chunk 长度为 1.6 秒。如果采用同步串行 pipeline：

```text
Observe -> Infer -> Execute -> Observe -> Infer -> Execute
```

机器人在推理期间没有新的动作轨迹，只能等待、保持上一目标姿态，或在旧 chunk 耗尽后停住。这就是论文所谓：

> sequential execution blocks robot motion during inference。

底层 servo 可能仍以 30 Hz 运行，但它只是在保持控制目标，并没有持续得到新的任务动作。

### 6.2 异步闭环

DreamZero 改为：

```text
Execute chunk n    ||    Infer chunk n+1
```

控制器持续执行当前 action buffer，推理线程根据最新可用观测生成新的 action chunk。约束由：

$$
\text{推理必须在机器人开始运动前完成}
$$

变成：

$$
T_{\text{inference}}
<
\text{当前 action chunk 剩余可执行时间}.
$$

论文将目标 latency 设为约 200 ms，最终 DreamZero-Flash 在 GB200 上达到约 150 ms，即约 6.7 个新 action chunk/s，通常表述为 7 Hz。

需要区分：

- **7 Hz**：高层策略/action chunk 更新频率；
- **30 Hz**：AgiBot 底层动作控制频率。

### 6.3 系统和实现优化

| 优化 | 核心思想 |
|---|---|
| CFG parallelism | 条件与无条件 forward 分布到两张 GPU |
| DiT caching | flow velocity 方向稳定时跳过部分昂贵 DiT forward |
| `torch.compile` + CUDA Graphs | 消除 Python/CPU launch overhead，融合算子 |
| Kernel/scheduler tuning | 使用更高效 attention kernel，把 scheduler 运算迁移到 GPU |
| NVFP4 quantization | Blackwell 上量化权重和激活，敏感算子保留较高精度 |
| DreamZero-Flash | 通过训练目标改变，实现单步 action denoising |

论文报告：

- H100 上系统与实现优化约 9.6×；
- GB200 上加入量化约 16.6×；
- 加入 DreamZero-Flash 后总计约 38×。

### 6.4 DiT velocity caching

Flow matching 推理近似求解：

$$
\frac{dx_t}{dt}=v_\theta(x_t,t).
$$

离散更新：

$$
x_{t_{i+1}}=x_{t_i}+\Delta t_i v_i.
$$

若相邻真实计算得到的 velocity 方向相似：

$$
\cos(v_{i-1},v_i)>\epsilon,
$$

则认为局部 flow trajectory 近似线性，可以复用缓存 velocity：

$$
v_{i+1}\approx v_i,
$$

从而跳过部分 14B DiT forward。所谓“从 16 步降到 4 步”更准确地说是：仍可进行多次 solver integration，但昂贵的 DiT velocity evaluation 减少到约 4 次。

需要注意：

- cosine similarity 只检查方向，不检查幅值；
- 如果对 joint video-action velocity 整体计算 cosine，video latent 的高维度可能主导指标；
- Algorithm 2 中的 `vlast` 没有清楚初始化和更新，伪代码不是可直接运行的严格实现。

### 6.5 异步执行仍未完全解释的问题

假设在时刻 $t$ 读取观测，模型在 $t+150$ ms 才输出 action；此时机器人状态已经变化。系统需要处理：

- observation latency；
- action timestamp offset；
- 已经 committed 的动作步；
- 新旧 chunk 的覆盖和拼接；
- chunk 边界的速度/位置连续性。

论文只说明控制器执行“当前时间戳最新的动作”，没有充分描述 action buffer、延迟补偿、chunk blending 或 temporal ensemble。因此论文证明了实时性的大方向，但稳定部署的 scheduling 细节仍不完整。

---

## 7. DreamZero-Flash

### 7.1 标准训练与单步推理的不匹配

标准 DreamZero 对 video 和 action 使用同一个 flow timestep：

$$
t_k^{\text{video}}=t_k^{\text{action}}\sim\mathcal U(0,1).
$$

在少步甚至单步推理时，动作需要快速变得接近 clean，但视频 latent 仍可能十分嘈杂。标准训练却主要让模型在“视频与动作噪声水平相似”的条件下学习，造成 train-test mismatch。

### 7.2 解耦噪声 schedule

DreamZero-Flash 使用：

$$
t_k^{\text{video}}=1-\eta,
\qquad
\eta\sim\operatorname{Beta}(7,1),
$$

$$
t_k^{\text{action}}\sim\mathcal U(0,1).
$$

因为：

$$
\mathbb E[t_k^{\text{video}}]=0.125,
$$

训练中的视频通常处于高噪声状态，而动作噪声覆盖整个范围。模型因此被迫学习：

> 即使当前生成视频还很不准确，也要能从 noisy visual context 中恢复出较干净动作。

表 3 结果：

| 方法 | Denoising steps | Table bussing task progress | 延迟 |
|---|---:|---:|---:|
| DreamZero | 4 | 83% | 350 ms |
| DreamZero | 1 | 52% | 150 ms |
| DreamZero-Flash | 1 | 74% | 150 ms |

DreamZero-Flash 通过训练适配，恢复了大部分四步性能。

### 7.3 论文中的伪代码疑点

正文和附录公式都写成：

$$
t^{\text{video}}=1-\operatorname{Beta}(7,1),
$$

但 Algorithm 1 第 13 行直接写：

$$
t_{\text{vid}}\sim\operatorname{Beta}(7,1).
$$

后者会把 $t$ 推向 1，即低噪声，与正文目标相反。这很可能是伪代码笔误。

---

## 8. Action Chunk Smoothing

论文对生成动作做：

1. 三次插值，上采样到 2× 时间分辨率；
2. Savitzky–Golay filter，窗口 21，三阶多项式；
3. 下采样回原始控制频率。

形式上：

$$
a'
=
\operatorname{Downsample}
\left(
\operatorname{SGFilter}
\left(
\operatorname{CubicInterpolate}(a)
\right)
\right).
$$

### 8.1 为什么先上采样再下采样

假设原 action chunk 为 30 Hz、48 个点。先插值到约 60 Hz、96 个点，再做局部多项式拟合，最后恢复到 30 Hz、48 个点。

上采样不产生新信息，它的作用是构造更稠密的数值网格。相同的 21 点窗口：

- 直接用于 30 Hz，覆盖约 $21/30=0.7$ 秒；
- 用于 60 Hz，覆盖约 $21/60=0.35$ 秒。

因此可以使用足够多的拟合点，同时把平滑限制在更局部的物理时间范围内，减少对抓取、松手和真实转折的过度抹平。

最终下采样是为了恢复：

- 原动作数量；
- 原控制频率；
- 原 chunk 持续时间；
- 机器人接口期望的 action representation。

它只能降低局部高频抖动和 jerk，不能修复错误任务计划。若动作方向本身错误，滤波只会让机器人更平滑地执行错误动作。

对于相对或增量动作表示，复现时还应检查插值和滤波是否改变累计位移语义。

---

## 9. 数据策略

### 9.1 AgiBot 预训练数据

论文采集约 500 小时 AgiBot G1 teleoperation data：

- 22 个真实环境；
- 7193 个 episode；
- episode 平均约 4.4 分钟；
- 平均约 42.4 个 subtask；
- 场景包括家庭、餐厅、超市、咖啡店、办公室、仓库、实验室和酒店。

其数据理念是 **diversity over repetition**：

- 每个 episode 连续执行三个粗粒度任务；
- 某任务收集到约 50 个 episode 后从任务列表下线；
- 鼓励 teleoperator 持续提出新任务；
- 以任务覆盖和真实实用性为优先，而不是在固定配置中反复采集同一任务。

这使 DreamZero 的贡献不只在模型，也包括一种长尾、多环境、非重复的数据构造方式。

### 9.2 DROID-Franka

论文还在公开 DROID 数据上训练 Franka 版本，以验证 WAM 是否能利用公开、异质、in-the-wild robot dataset。

### 9.3 多样性消融

在相同约 500 小时规模下：

| 数据类型 | PnP Easy task progress |
|---|---:|
| Repetitive | 33% |
| Diverse | 50% |

这支持“多样状态—动作对应关系有助于学习鲁棒 implicit IDM”的解释。但消融只在缩小训练配置和 PnP Easy 上进行，不能推出所有机器人任务都应放弃重复演示。高精度插入、装配和毫米级控制仍可能需要大量密集重复数据。

---

## 10. 实验结果

### 10.1 已见任务、未见环境和物体

AgiBot 平均 task progress：

| 方法 | 平均任务进度 |
|---|---:|
| 最佳 pretrained VLA baseline | 27.4% |
| DreamZero | 62.2% |

DreamZero 在简单 PnP、困难 PnP 和 contact-rich manipulation 上均显著优于从 scratch VLA。论文认为 VLA 难以从高度异质、非重复数据中直接学习稳定 observation-to-action mapping，而视频世界建模目标提供了更密集的结构约束。

### 10.2 未见任务

AgiBot 的未见任务包括解鞋带、摘帽子、画画、取吸管、堆方块、刷漆、熨衣服、握手、折地图和拉车。

| 方法 | AgiBot 未见任务平均进度 |
|---|---:|
| 最佳 pretrained VLA baseline | 16.3% |
| DreamZero | 39.5% |

DROID-Franka：

- DreamZero：49% task progress；
- DreamZero：22.5% complete success rate。

需要区分 task progress 与 success rate。较高的部分进度不表示完整完成任务。

### 10.3 Task-specific post-training

三个 downstream task：

- shirt folding：33 小时；
- fruit packing：12 小时；
- table bussing：40 小时。

DreamZero 平均 task progress 为 90.5%，显示其环境泛化能力在 task-specific post-training 后仍能部分保留。

### 10.4 Cross-embodiment video transfer

在 9 个未见任务上：

| 方法 | 平均任务进度 |
|---|---:|
| DreamZero | 38.3% |
| + 12 分钟 human video-only | 54.3% |
| + 20 分钟 YAM video-only | 55.4% |

这些 source embodiment 数据没有动作标签，只训练视频预测相关能力。它们帮助目标 AgiBot 理解任务视觉动态，但不会使模型直接输出 YAM 或人类动作。

### 10.5 Few-shot new embodiment adaptation

从 DreamZero-AgiBot 出发，使用 YAM 上约 55 条轨迹、11 个任务、约 30 分钟 play data 做 post-training。模型可在 YAM 上执行一些新物体和新语言组合。

该实验说明视觉世界模型可能使新机器人的 inverse dynamics 学习更省数据；它不说明原 checkpoint 无需训练即可控制 YAM，也没有报告适配后是否仍保持 AgiBot 控制能力。

### 10.6 模型和架构消融

| 消融 | 结果 |
|---|---:|
| 5B WAM + diverse data | 21% |
| 14B WAM + diverse data | 50% |
| 14B AR WAM | 50% |
| 14B bidirectional WAM | 50% |

结论：

- 更大视频 backbone 明显提升性能；
- AR 与 bidirectional 在该消融上的任务进度相同；
- AR 的主要优势是动作更平滑、KV cache 和 3–4× 更快推理，而不是已证明的最终任务准确率优势。

---

## 11. DreamZero 与其他方法的区别

### 11.1 与 VLA

| 维度 | VLA | DreamZero/WAM |
|---|---|---|
| 主要互联网预训练 | 静态图像-文本 | 视频生成 |
| 机器人目标 | 直接预测动作 | 联合预测视觉未来和动作 |
| 动力学监督 | 隐式、稀疏 | 连续视频变化提供密集监督 |
| 视觉未来 | 通常不显式生成 | 作为联合生成变量 |
| 测试时规划搜索 | 通常无 | DreamZero 也无 |

需要谨慎解释基线：DreamZero 论文中的 `π0.5 pretrained` 或 `GR00T pretrained` 并不是官方 checkpoint 直接零样本部署到 AgiBot。论文先从官方跨 embodiment 预训练 checkpoint 初始化，再在与 DreamZero 相同的 AgiBot/DROID target data 上继续训练。因此对比主要是在目标机器人数据相同后，不同预训练和训练目标谁更能利用异质数据。

### 11.2 与显式视频规划器 + IDM

一些方法先生成视频计划，再通过独立 IDM 或 point tracking 提取动作：

```text
video generator -> decoded future video -> IDM/controller -> action
```

DreamZero 则是：

```text
shared DiT jointly denoises video latent and action latent
```

视频和动作不是串联的两个独立模型，而是共享 attention 的联合随机变量。

### 11.3 与 Dreamer/latent world model

Dreamer 类方法通常学习：

$$
p(s_{t+1}\mid s_t,a_t),
$$

并在 latent imagination 中借助 reward、value 或 actor 进行规划/优化。

DreamZero 直接生成：

$$
p(o_{t:t+H},a_{t:t+H}\mid o_{0:t},c,q_t),
$$

无需测试时搜索，但也缺少显式 reward-based candidate selection。

### 11.4 与完全 bidirectional WAM

完全 bidirectional WAM 通常一次生成固定未来窗口，所有 target time token 彼此可见。DreamZero 将长序列拆成局部固定 chunk：

- chunk 内双向；
- chunk 间 causal；
- 每执行一个 chunk，用真实观测替换预测未来。

这种设计的主要价值是保持局部 FPS、使用 KV cache、避免跨 chunk 视觉误差积累。

---

## 12. 关键技术判断与阅读注意事项

### 12.1 “Zero-shot”不等于零机器人训练数据

DreamZero 使用约 500 小时目标机器人数据。“Zero-shot”指评测任务或环境未在训练中出现，而不是模型从未看过该机器人或动作标签。

### 12.2 “Generalist”主要是固定 embodiment 内的 generalist

论文明确分别训练 AgiBot 和 Franka，并把 multi-embodiment training 留给未来。因此更准确的说法是：

> 在固定机器人上的任务、物体和环境 generalization。

### 12.3 IDM 是概念性解释，不是独立模块

公式分解有助于理解，但没有直接证明共享 DiT 内部形成了可分离、可评测的 IDM。更稳妥的表述是：模型学习了 joint video-action distribution 及其统计对应关系。

### 12.4 预测视频不是显式闭环规划器

模型没有候选生成、打分和搜索。它的视觉未来是当前动作生成的联合隐变量，而不是独立可优化的计划。

### 12.5 固定长度问题不是 diffusion 的数学必然

论文所说的 fixed-length/subsampling 问题来自常见 fixed-clip bidirectional architecture。DreamZero 仍然在每个 chunk 内预测固定长度视频和动作，只是通过跨 chunk autoregression 保持局部原始 FPS。

### 12.6 AR 消融没有显示更高任务进度

AR 与 bidirectional 都是 50%。AR 的证据主要是推理速度、平滑性和模态时间对齐，而不是最终任务准确率显著提高。

### 12.7 “失败主要来自视频生成”证据偏定性

论文展示了视频计划错误、机器人忠实执行错误计划的案例，但没有系统统计：

- 视频正确而动作失败的比例；
- video-action alignment 的定量指标；
- 视频质量与控制成功率之间的相关性；
- 隐式 IDM 的独立精度。

因此这是有支持的假设，但还不是充分证明的因果结论。

### 12.8 基线比较不是完全控制变量

WAM 从视频生成模型初始化，VLA 从 VLM/VLA 初始化。性能差异可能同时来自：

- 预训练数据分布；
- 预训练任务；
- 模型架构；
- 模型尺度；
- 联合视频目标；
- 数据处理和动作表示。

论文证明了完整 WAM recipe 有优势，但尚未完全隔离每一项贡献。

### 12.9 评测规模和指标限制

- AgiBot 每个 checkpoint 的 seen/unseen structured evaluation 各约 80 rollouts；
- DROID 每任务约 2 个 rollout；
- 大量结论依赖 task progress，而非完整成功率；
- free-form 100+ 任务主要是定性展示；
- 没有专门评测必须使用记忆才能完成的任务；
- 对亚厘米精度、高精度装配和长时程规划仍有限。

---

## 13. 论文中的符号和伪代码疑点

### 13.1 公式 3 的 $K/M$ 混用

正文用 $K$ 对 chunk 求和；附录把 $K$ 定义为每 chunk latent frame 数，把 $M$ 定义为 chunk 数。求和上限更可能应为 $M$。

### 13.2 DreamZero-Flash 的 Beta 变换

正文：

$$
t_{\text{video}}=1-\eta,\quad\eta\sim\operatorname{Beta}(7,1).
$$

Algorithm 1：

$$
t_{\text{vid}}\sim\operatorname{Beta}(7,1).
$$

二者噪声方向相反，伪代码很可能遗漏了 $1-$。

### 13.3 Clean context 的 timestep 表示

Flow matching 定义中 $t=1$ 是 clean，Algorithm 2 却以 `t=0` 注入 clean context。这里可能是“context token 使用特殊无噪声/条件编码”，而非普通 diffusion sample，但论文没有清晰解释。

### 13.4 历史 action context 不一致

训练公式似乎把以前的 clean action 放进 $\mathcal C_k$；Figure 14 和推理算法则主要缓存真实视觉 context。需要检查代码中是否显式缓存已执行动作，或者公式只是简化表示。

### 13.5 DiT caching 变量不完整

Algorithm 2 使用 `CosSim(vprev, vlast)`，但 `vlast` 的初始化和更新未明确。真实实现应维护最近两次实际 DiT evaluation 的 velocity，并设置最多连续复用多少 solver step。

---

## 14. 复现与工程实现检查表

### 14.1 数据与时间同步

- 相机和机器人控制时钟是否统一；
- 视频 5 FPS 是硬件采样还是从更高 FPS 下采样；
- action timestamp 如何切成精确 1.6 秒 chunk；
- 丢帧、控制延迟和 teleoperation lag 如何处理；
- 多视角帧是否严格同步。

### 14.2 Action representation

- relative joint position 的精确定义；
- 各机器人 action dimension 和 joint order；
- normalization 是否按数据集、机器人或关节分别统计；
- gripper、base、torso 与 arm 的尺度是否单独加权；
- action smoothing 是否保持累计位移和边界条件。

### 14.3 联合 loss

- video/action loss 是否分别求均值；
- 两种模态是否有独立权重；
- video latent 空间的 channel/spatial token normalization；
- action decoder 的噪声尺度；
- Flash 阶段是否从标准 DreamZero checkpoint 继续训练。

### 14.4 Attention mask 与 cache

- context token 不得依赖当前 noisy target；
- 当前 chunk video/action token 是否完全双向；
- 不同 chunk target 是否严格隔离；
- 推理时 context K/V 更新顺序；
- 真实 observation 替换 prediction 后是否正确处理 position index。

### 14.5 实时控制

- inference thread 与 control thread 的同步；
- action chunk 的 timestamp；
- latency compensation；
- committed action 前缀如何处理；
- 新旧 chunk 边界是否做 blend；
- 失去新预测时的 hold/fallback 行为；
- safety limits、速度限制和碰撞保护。

### 14.6 DiT caching

- cosine 是对 video、action 分别算，还是联合算；
- 是否同时检查 velocity magnitude；
- 最多连续复用次数；
- 不同 solver 时间段是否使用不同阈值；
- caching 对动作质量和视频质量的独立影响。

---

## 15. 常见问题速查

### Q1：DreamZero 是否每次 diffusion 只预测一帧？

不是。每次预测一个固定时间 chunk；默认每 chunk 有 2 个 video latent frame，并同时预测 1.6 秒动作。

### Q2：它为什么还叫 autoregressive？

Autoregressive 发生在 chunk 之间。当前 chunk 执行后，真实观测成为下一 chunk 的 causal context。

### Q3：当前 chunk 内是不是 bidirectional？

是。当前 noisy video/action token 在同一 chunk 中可以双向 attention，联合恢复整个 chunk。

### Q4：历史 context 要不要预测？

不要。它只作为 clean condition 产生 K/V，并被缓存。

### Q5：生成的视频 latent 要 decode 再 encode 吗？

不要。执行后直接丢弃预测视频 latent，用新的真实图像经过 VAE 编码后更新 cache。

### Q6：视频与动作必须同频吗？

不必。它们只需覆盖同一物理时间区间。AgiBot 是 5 FPS 视频对 30 Hz 动作。

### Q7：为什么对多个 chunk 的 flow loss 求平均？

一条长轨迹中的每个 chunk 都是一个训练样本阶段；平均后可在一次 block-causal forward 中训练所有阶段。

### Q8：为什么 velocity cosine 高就复用缓存？

相邻 flow step 的方向近似相同意味着局部轨迹近似直线，可跳过昂贵 DiT forward，用旧 velocity 做近似积分。

### Q9：为什么异步推理能防止机器人停住？

因为机器人执行当前 action chunk 时，后台同时计算下一 chunk；不再要求推理完成后机器人才能开始运动。

### Q10：为什么 action 要先 2× upsample 再 downsample？

在更稠密时间网格上用固定点数的 SG filter，可以缩短滤波窗口对应的物理时间，局部去抖后再恢复原控制频率。

### Q11：DreamZero 是不是同一套参数控制所有机器人？

不是。论文分别训练 AgiBot 和 Franka；YAM 也需要约 30 分钟目标机器人数据 post-training。

### Q12：论文中的 pretrained $\pi_0.5$ 是否直接零样本控制 AgiBot？

不是。论文从官方 pretrained checkpoint 初始化后，还在与 DreamZero 相同的 AgiBot target data 上继续训练。

---

## 16. 综合评价

DreamZero 的真正价值不在于简单地“给 VLA 增加一个视频预测头”，而在于把以下几个要素组合成一个完整系统：

1. 从大规模视频 diffusion backbone 继承时空和物理变化先验；
2. 在共享 DiT 中联合去噪 video latent 和 action latent；
3. 使用 chunk 内 bidirectional、chunk 间 causal 的混合 attention mask；
4. 通过真实观测回灌避免 autoregressive 视频误差跨 chunk 累积；
5. 通过异步 action execution、KV cache、DiT cache、量化和 Flash 训练满足实时控制；
6. 用多样、长尾、非重复机器人数据强化新任务和新环境泛化；
7. 利用 video-only 数据实现较低成本的跨 embodiment 技能迁移。

其最有说服力的结论是：

> **视频生成预训练和联合未来建模，可能比静态视觉语言预训练更适合作为开放世界机器人策略的基础表示。**

但目前仍不能据此得出：

- WAM 已成为跨任意机器人统一控制模型；
- 生成视频已经构成可解释、可优化的显式 planner；
- 所有失败都来自视频生成而非 action extraction；
- 多样数据在所有机器人任务中都优于重复演示；
- DreamZero 已解决长时程记忆、高精度操作和低算力部署。

因此，DreamZero 更合理的定位是：

> **一个以视频世界模型为核心、面向固定目标 embodiment、具备强任务和环境泛化能力的实时端到端策略原型。**

---

## 17. 论文阅读索引

| 内容 | 论文位置 |
|---|---|
| 总体主张和贡献 | p.1–3 |
| WAM 与 VLA/其他方法的关系 | p.4–5 |
| 模型架构 Figure 4 | p.6 |
| Flow matching 与 autoregressive chunk | p.6–8 |
| 实时系统和 DreamZero-Flash | p.8–10 |
| 数据和评测协议 | p.10–13 |
| 主要泛化实验 | p.13–18 |
| WAM 与其他 world model 的区别 | p.20 |
| Bidirectional vs. autoregressive | p.20–21 |
| Attention mask Figure 14 | p.21 |
| Algorithms 1–2 | p.22 |
| 系统优化细节 | p.22–24 |
| 数据采集策略 | p.24–25 |
| 详细任务列表 | p.26–28 |
| 失败案例 | p.28–29 |

---

## 参考来源

1. Seonghyeon Ye et al., **World Action Models are Zero-shot Policies**, NVIDIA, arXiv:2602.15922v1, 2026.
2. 本报告中的批判性分析、符号一致性检查和工程推断，基于论文正文、附录、算法伪代码及 Figure 13–16 的交叉阅读。

---

## 18. 相关笔记

- [[OA_WAM|OA-WAM]]：同属 World Action Model 路线，但以对象地址和 slot routing 代替视频 latent 作为结构化世界状态。
- [[WorldVLA 论文综述(不建议读)|WorldVLA]]：同样联合视觉与动作建模；WorldVLA 的图像预测只服务于辅助训练，DreamZero 则在推理时联合生成视频和动作。
- [[UniPi_技术总结|UniPi]]：对比“先生成视频计划、再由 inverse dynamics 恢复动作”和端到端 video-action flow matching。
- [[DreamerV3_技术报告|DreamerV3]]：对比依赖 reward/value 与 latent imagination 的 world-model RL 路线。
- [[Pi0_7_technical_report|π0.7]]：对比异步 subgoal world model 加 flow-matching action expert 与统一视频—动作生成。
