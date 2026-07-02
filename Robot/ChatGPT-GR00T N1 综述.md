---
title: GR00T N1 综述
type: paper_note
topic: humanoid_robot_foundation_model
status: mature
importance: high
updated: 2026-07-02
tags:
  - gr00t-n1
  - humanoid-robot
  - vla
  - diffusion-transformer
  - flow-matching
  - robot-foundation-model
  - embodiment-adaptation
  - synthetic-data
  - robotics
---
# GR00T N1 综述

**User:** Anonymous  
**Created:** 6/29/2026 1:02:32  
**Updated:** 7/2/2026 15:44:02  
**Exported:** 7/2/2026 23:18:52  
**Link:** [https://chatgpt.com/c/6a41535c-e280-83e8-bee5-1192d35c88ba?mweb_fallback=1](https://chatgpt.com/c/6a41535c-e280-83e8-bee5-1192d35c88ba?mweb_fallback=1)  

下面是一份**结合标记重点 + 文章内容 + 我们前面讨论**整理出来的 Markdown 技术报告。

# GR00T N1 技术报告

论文：**GR00T N1: An Open Foundation Model for Generalist Humanoid Robots**  
机构：NVIDIA  
主题：面向通用类人机器人的 VLA foundation model

---

## 1. 核心概览

GR00T N1 是 NVIDIA 提出的一个面向 humanoid robot 的 **Vision-Language-Action, VLA** 模型。它的目标不是只解决某个单一 manipulation task，而是训练一个可以跨任务、跨 embodiment、并能用少量下游示教快速适配的机器人基础模型。

结合你标记的重点，这篇文章的核心主线可以概括为：

```text
GR00T N1 =
    dual-system VLA architecture
    + action flow-matching / DiT policy
    + real data / neural trajectories / simulation trajectories 的数据金字塔
    + pre-training 到 post-training 的统一训练范式
```

论文强调它是一个 **dual-system architecture**：System 2 是 VLM，负责视觉语言理解；System 1 是 Diffusion Transformer，负责实时动作生成。两个模块不是松散拼接，而是在统一学习框架里 jointly trained end-to-end。你标记的 “unified learning framework”、“mixture”、“rapid adaptation” 其实正好对应这篇文章的三个重点：统一模型、多源数据混合、少量数据快速适配。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

## 2. 论文要解决的问题

机器人 foundation model 的核心难点是：**真实机器人数据太少、太贵、太分散**。

文本和图像领域有互联网规模数据，但 humanoid robot 没有类似规模的数据。即便有 Open X-Embodiment 这类跨机器人数据，也会因为不同机器人之间的 embodiment、传感器、控制方式、自由度、action space 差异很大，形成很多 “data islands”。GR00T N1 的策略不是简单把所有数据混起来，而是构建一个 **data pyramid**：

```text
底部：web data + human videos
中部：synthetic data = simulation trajectories + neural trajectories
顶部：real-world robot data
```

底层数据规模大，但不够 embodiment-specific；顶层真实机器人数据规模小，但最贴近真实部署。论文明确说，低层数据提供 broad visual and behavioral priors，高层真实数据负责 embodied real-robot grounding。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

## 3. 模型架构

### 3.1 总体结构

GR00T N1 是一个 VLA 模型，输入是：

```text
image observation
language instruction
robot state
```

输出是：

```text
motor action chunk
```

模型由两部分组成：

```text
System 2: Vision-Language Model, VLM
    输入图像和语言，输出视觉语言 token embeddings

System 1: Diffusion Transformer, DiT
    输入 noised action、robot state、VLM tokens
    输出 denoised motor action
```

论文图 2 和图 3 对应了这个结构。你标记的重点包括 “System 2 / System 1”、“chunk of 16 actions”、“embodiment-specific module”、“cross-attention”，这些都是理解模型 pipeline 的关键。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

### 3.2 System 2：Vision-Language Module

System 2 使用 **Eagle-2 VLM**。图像先被编码为 image tokens，语言被 tokenize 成 text tokens，然后一起输入 VLM。论文中图像输入分辨率为：

$$
224 \times 224
$$

经过 pixel shuffle 后，每帧得到：

$$
64
$$

个 image token embeddings。一个重要标记点是：作者没有使用 VLM 最后一层 hidden states，而是使用中间层 representation。对于 GR00T-N1-2B，他们使用第 12 层，因为实验发现 **middle-layer embeddings 同时带来更快推理速度和更高 downstream policy success rate**。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

在公式中：

$$
\phi_t
$$

表示的就是 **VLM 输出的 vision-language token embeddings**。它不是原始 image patch tokens，也不是原始 text token ids，而是 image + text 经过 VLM 融合后的 hidden representations。

---

### 3.3 System 1：Diffusion Transformer Action Module

System 1 是一个 DiT 风格的 action generator。它输入：

```text
noised action chunk
robot state embedding
VLM output tokens
flow-matching timestep
```

然后输出 denoised action chunk。

为了支持不同机器人 embodiment，GR00T N1 使用 embodiment-specific modules：

```text
State Encoder
Action Encoder
Action Decoder
```

这些模块用 MLP 把不同维度的 state/action 投影到统一 hidden dimension。这样主干 DiT 可以共享，而不同机器人只需要不同的输入输出适配器。论文还特别提到 Action Encoder MLP 会把 **diffusion timestep** 和 noised action vector 一起编码，这也是你标记的重点之一。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

DiT 内部采用 alternating blocks：

```text
Self-Attention:
    处理 noised action token embeddings 和 state embeddings

Cross-Attention:
    condition on VLM 输出的 vision-language token embeddings φ_t
```

这点很重要：System 1 不是单纯根据 proprioception 生成动作，而是通过 cross-attention 使用 VLM 对图像和语言的理解。

### 3.4 如何融合不同模态的机器人
**GR00T N1 并不是把所有机器人强行变成同一个固定自由度的 action vector**。它的做法更像是：

> **共享一个 VLM + DiT 主干，但每种机器人 / embodiment 有自己的 state encoder、action encoder 和 action decoder。**

所以它所谓“一个模型适配不同机器人”，不是指完全没有差异地用同一个输入输出头，而是：

```text
共享主体参数：
    Eagle-2 VLM
    Diffusion Transformer / DiT action core

机器人相关参数：
    embodiment-specific State Encoder
    embodiment-specific Action Encoder
    embodiment-specific Action Decoder
```

论文在模型结构里明确说，为了处理不同机器人 embodiment 的 state/action 维度差异，他们为每个 embodiment 使用 MLP，把 state 和 action 投影到共享 embedding dimension；图 3 里也把 State Encoder、Action Encoder、Action Decoder 标成了 embodiment-specific module。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

#### 3.4.1. 最核心机制：不同机器人先投影到同一个 latent 空间

假设有三个机器人：

```text
Robot A：单臂 Franka，action_dim = 7
Robot B：双臂 Panda，action_dim = 14
Robot C：GR-1 humanoid，action_dim = 50+
```

如果直接让 DiT 处理原始 action vector，就会有问题：

```text
Robot A 的 action 是 7 维
Robot B 的 action 是 14 维
Robot C 的 action 是 50+ 维
```

维度不同，根本没法直接共享同一个 Transformer 输入输出。

GR00T N1 的做法是给每个 embodiment 一个自己的 action encoder：

$$
\mathrm{ActionEncoder}_e: \mathbb{R}^{d_a^e} \rightarrow \mathbb{R}^{D}
$$

其中：

- $$e$$ 是 embodiment id；
- $$d_a^e$$ 是这个机器人自己的 action 维度；
- $$D$$ 是所有机器人共享的 hidden dimension。

所以对于不同机器人：

$$
A_t^{\text{Franka}} \in \mathbb{R}^{7}
\rightarrow
\mathrm{ActionEncoder}_{\text{Franka}}
\rightarrow
\text{shared action token}
$$

$$
A_t^{\text{GR-1}} \in \mathbb{R}^{50+}
\rightarrow
\mathrm{ActionEncoder}_{\text{GR-1}}
\rightarrow
\text{shared action token}
$$

进入 DiT 之后，它们都已经变成统一 hidden dimension 的 token，所以 DiT 可以共享。

---

#### 3.4.2. State 也是一样处理

不同机器人的 state 差异也很大：

```text
单臂机器人：
    EEF pose
    gripper state
    joint positions

双臂机器人：
    left EEF pose
    right EEF pose
    left gripper
    right gripper

humanoid：
    双臂 joint
    双手 joint
    waist
    neck
    可能还有 base state
```

GR00T N1 不要求这些 state 原始维度一致，而是使用每个 embodiment 自己的 state encoder：

$$
\mathrm{StateEncoder}_e: \mathbb{R}^{d_q^e} \rightarrow \mathbb{R}^{D}
$$

所以：

```text
Robot A 的 proprioception/state
        ↓
StateEncoder_A
        ↓
统一 hidden state token

Robot B 的 proprioception/state
        ↓
StateEncoder_B
        ↓
统一 hidden state token
```

然后这些统一维度的 state token 和 action token 一起进入共享 DiT。

---

#### 3.4.3. 输出时再用各自的 Action Decoder 转回自己的动作空间

DiT 输出的是共享 hidden tokens，不是直接输出某个机器人的原始动作。

最后一步是：

$$
\mathrm{ActionDecoder}_e: \mathbb{R}^{D} \rightarrow \mathbb{R}^{d_a^e}
$$

也就是说，当前 batch 里的样本来自哪个机器人，就调用哪个机器人的 decoder。

比如：

```text
共享 DiT 输出 hidden action tokens
        ↓
ActionDecoder_Franka
        ↓
7 维 Franka action
```

或者：

```text
共享 DiT 输出 hidden action tokens
        ↓
ActionDecoder_GR1
        ↓
GR-1 的双臂 + 双手 + waist + neck action
```

论文也说，DiT 最后会对 final H tokens 应用 embodiment-specific Action Decoder 来预测动作。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

所以真正的结构是：

```text
image + language
        ↓
Eagle-2 VLM
        ↓
φ_t 视觉语言 tokens
        ↓
                   ┌─────────────────────┐
robot state q_e →  │ StateEncoder_e       │
noised action A_e →│ ActionEncoder_e      │
                   └─────────────────────┘
        ↓
shared DiT blocks
        ↓
ActionDecoder_e
        ↓
robot-specific action
```

---

#### 3.4.4. 它不是靠 padding 解决吗？

论文没有把重点放在 padding 上，而是放在 **embodiment-specific projector / encoder / decoder** 上。

你可以想象两种方案：

##### 方案 A：全部 padding 到最大维度

```text
Franka action = 7 维，后面补 0 到 64 维
GR-1 action = 50 维，也补到 64 维
```

这个做法简单，但语义很乱。第 8 维对 Franka 可能不存在，对 GR-1 可能是某个手指关节。

##### 方案 B：每个 embodiment 自己编码到共享 latent 空间

```text
Franka action 7 维 → ActionEncoder_Franka → shared token
GR-1 action 50 维 → ActionEncoder_GR1 → shared token
```

GR00T N1 更接近方案 B。

也就是说，**共享的不是原始 action dimension，而是投影后的 action token representation**。

---

#### 3.4.5. 那单臂、双臂、有无移动底盘怎么处理？

核心原则是：**机器人有什么，就在它自己的 state/action schema 里定义什么；没有的字段就不属于这个 embodiment 的输入输出。**

比如：

##### 单臂机器人

```text
state:
    one arm EEF pose
    gripper state
    maybe joint positions

action:
    delta EEF position
    delta EEF rotation
    gripper command
```

##### 双臂机器人

```text
state:
    left arm EEF pose
    right arm EEF pose
    left gripper
    right gripper

action:
    left arm delta pose
    right arm delta pose
    left gripper command
    right gripper command
```

##### GR-1 humanoid

```text
state:
    both arms
    both dexterous hands
    waist
    neck
    possibly base-related state

action:
    both arm / hand / waist / neck commands
```

论文在 benchmark 描述中也给了类似例子：RoboCasa 的 action 是相对 EEF position/rotation 加 gripper；DexMimicGen 里包含双臂 Panda、双臂 dexterous hands、GR-1 humanoid，不同 embodiment 的 state/action space 分别包含 EEF、gripper、hands、waist、neck 等不同部分。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

所以有无移动底盘也是一样的逻辑：

```text
有移动底盘：
    state/action schema 里包含 base pose / base velocity / base command

没有移动底盘：
    schema 里没有这些字段
```

然后由对应 embodiment 的 encoder/decoder 负责把它映射到共享 hidden space。

---

#### 3.4.6. 数据格式层面也做了标准化

仅有 encoder/decoder 还不够。因为如果每个数据集里的 action vector 都是黑盒数组，模型不知道第几维是什么含义，训练会很混乱。

所以论文附录里说，他们扩展了 LeRobot 数据格式，要求有 `modality.json`，明确 state/action vector 每个维度的语义，并把 state/action 从一个 monolithic vector 拆成更细的字段，比如 EEF position、orientation、gripper state 等。这样可以减少 preprocessing 和训练中的语义错误。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

此外，论文还做了 best-effort action/state space unification：

```text
end-effector rotation state → 6D rotation representation
end-effector rotation action → axis-angle
joint states / joint actions / EEF positions 等做 min-max normalization
state/action ordering 统一：left arm 到 right arm
```

这些标准化不是让所有机器人维度相同，而是让**相同语义的字段尽量用相同表示方式**。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

#### 3.4.7. 训练时一个 batch 里怎么处理不同机器人？

概念上可以这样理解：

```text
sample_i 有一个 embodiment_id = e

根据 e 选择：
    StateEncoder_e
    ActionEncoder_e
    ActionDecoder_e

共享：
    VLM
    DiT
```

训练时，对于来自 Robot A 的样本：

```text
q_A, A_A^τ
    ↓
StateEncoder_A, ActionEncoder_A
    ↓
shared DiT
    ↓
ActionDecoder_A
    ↓
loss with A_A
```

对于来自 Robot B 的样本：

```text
q_B, A_B^τ
    ↓
StateEncoder_B, ActionEncoder_B
    ↓
shared DiT
    ↓
ActionDecoder_B
    ↓
loss with A_B
```

因此，loss 永远是在当前机器人自己的 action space 里算的。DiT 学到的是更通用的“视觉语言条件 + 状态 + 动作去噪”的共享动态模式，而输入输出的具体维度交给 embodiment-specific modules 处理。

---

#### 3.4.8. 这到底是不是“一个模型”？

严格说，它是：

> **一个共享主干 + 多个 embodiment-specific adapter/head 的统一模型。**

不是：

> 所有机器人完全共用同一个输入层和输出层。

这点要分清楚。

论文确实说 GR00T N1 以 unified model / single set of weights 支持 single-arm、bimanual、humanoid embodiments；但从架构图和文字看，这个 single model 内部包含了 embodiment-specific state/action encoders and decoders。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

所以更准确的说法是：

```text
共享：
    视觉语言理解能力
    action generation DiT 主体
    flow-matching 训练框架
    跨任务/跨 embodiment 表示

不共享：
    每个机器人的原始 state/action 输入输出维度
    每个机器人的 encoder/decoder adapter
```

---

#### 3.4.9. 如果来一个全新的机器人怎么办？

论文没有声称“任意新机器人零配置直接可用”。更合理的推断是：

1. 你需要定义这个新机器人的 `modality.json` / state-action schema；
2. 需要给它新的 State Encoder / Action Encoder / Action Decoder；
3. 需要用这个机器人的数据做 post-training；
4. 共享的 VLM + DiT 主干可以继承已有知识。

也就是说，新机器人适配不是完全免费，而是比从零训练更省数据。论文的 post-training 部分也说，它会 fine-tune pretrained model 到 each single embodiment 的数据集上。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

#### 3.4.10. 最后用一句话总结

GR00T N1 适配不同机器人靠的是：

> **把不同机器人的 state/action 先通过 embodiment-specific encoder 投影到统一 latent token 空间，在共享 DiT 中做动作生成，再通过 embodiment-specific decoder 映射回各自机器人的原始动作空间；同时用标准化的数据 schema、旋转表示、normalization 和字段顺序减少不同机器人之间的语义错位。**

所以它不是“所有机器人天然同一个自由度”，而是：

```text
不同自由度 / 单臂 / 双臂 / 有无底盘
        ↓
各自的 state/action encoder-decoder 处理差异
        ↓
共享 VLM + DiT 学通用决策和动作生成
        ↓
各自 decoder 输出各自机器人能执行的动作
```

---

## 4. Action Chunking 与 Flow Matching

### 4.1 Action Chunking

GR00T N1 不是一步一步预测 action，而是每次预测一个 action chunk：

$$
A_t = [a_t, a_{t+1}, \dots, a_{t+H-1}]
$$

论文设置：

$$
H = 16
$$

这也是你标记的重点。动作 chunking 的好处是可以让策略输出更平滑，也减少逐步预测时的误差累积。推理时模型仍然可以闭环执行：每隔一段时间根据最新观测重新生成 action chunk。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

### 4.2 Flow Matching 目标

给定真实 action chunk：

$$
A_t
$$

采样高斯噪声：

$$
\epsilon \sim \mathcal{N}(0, I)
$$

采样 timestep：

$$
\tau \in [0, 1]
$$

构造 noised action：

$$
A_t^\tau = \tau A_t + (1-\tau)\epsilon
$$

模型预测：

$$
V_\theta(\phi_t, A_t^\tau, q_t)
$$

其中：

- $$\phi_t$$ 是 VLM 输出的 vision-language tokens；
- $$q_t$$ 是 robot proprioceptive state embedding；
- $$A_t^\tau$$ 是当前噪声水平下的 action chunk；
- $$V_\theta$$ 是 DiT 预测的 vector field。

论文写的 loss 是：

$$
\mathcal{L}_{fm}(\theta)
=
\mathbb{E}_{\tau}
\left[
\left\|
V_\theta(\phi_t, A_t^\tau, q_t)
-
(\epsilon - A_t)
\right\|^2
\right]
$$

这里有一个我们前面讨论过的细节：从公式直觉上看，如果从 noise 走向 clean action，vector field 更自然地像是 $$A_t-\epsilon$$；但论文写成 $$\epsilon-A_t$$，并且推理公式又是加上 $$V_\theta$$。所以这里可能存在符号方向的简写或笔误。理解上可以把它当成：**模型学习从 noisy action chunk 到 clean action chunk 的 denoising vector field**。

---

### 4.3 Timestep Sampling

GR00T N1 没有使用最简单的 uniform timestep sampling，而是参考 Black et al. 2024，也就是 π₀ 的设置：

$$
p(\tau)
=
\mathrm{Beta}
\left(
\frac{s-\tau}{s}; 1.5, 1
\right),
\quad s = 0.999
$$

它的作用是在训练时决定采样哪些噪声强度。根据论文定义：

$$
A_t^\tau = \tau A_t + (1-\tau)\epsilon
$$

当：

$$
\tau = 0
$$

是纯噪声；当：

$$
\tau = 1
$$

是真实 action。这个 shifted beta 分布更偏向小 $$\tau$$，也就是让模型更多在 noisy side 上训练。

推理时不再采样这个 $$p(\tau)$$，而是先从噪声初始化：

$$
A_t^0 \sim \mathcal{N}(0,I)
$$

然后用 forward Euler 做 K-step denoising。论文中设置：

$$
K = 4
$$

并称这个步数在各个 embodiments 上效果都不错。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

## 5. 数据体系：这篇文章的重点

你这份 PDF 的标记非常集中地落在数据部分：**real data、neural trajectories、simulation trajectories、pre-training、post-training**。这也是这篇文章真正的核心。

GR00T N1 的训练数据可以分为四类：

| 数据类型 | 是否有 action 真值 | 主要用途 | 主要问题 |
|---|---:|---|---|
| Real robot trajectories | 有 | 真实控制 grounding | 贵、少 |
| Human videos | 没有 | 行为、affordance、motion prior | 没有机器人 action |
| Neural trajectories | 原始没有 | 扩充 counterfactual 场景 | 需要 pseudo-label |
| Simulation trajectories | 有 | 大规模低成本动作监督 | sim-to-real gap |

论文将 pre-training corpus 分成 real-robot datasets、synthetic datasets、human video datasets，其中 synthetic datasets 又包括 **simulation trajectories** 和 **neural trajectories**。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

## 6. Real Robot Trajectories

真实机器人数据主要包括：

1. **GR00T N1 Humanoid Pre-Training Dataset**  
   内部收集的 Fourier GR-1 teleoperation 数据。使用 VIVE Ultimate Tracker 捕捉 wrist pose，用 Xsens Metagloves 捕捉 finger movements，然后通过 inverse kinematics retarget 到 humanoid action。数据包含 head-mounted camera 图像、机器人动作、低维 proprioception 和语言标注，控制频率为 20Hz。

2. **Open X-Embodiment 子集**  
   包括 RT-1、Bridge-v2、Language Table、DROID、MUTEX、RoboSet、Plex 等。

3. **AgiBot-Alpha**  
   使用训练启动时可用的 140,000 trajectories。

真实数据在训练中承担两个作用：

```text
1. 直接用 ground-truth robot action 训练可执行策略
2. 同时抽取 latent action，用于跨 embodiment latent action pretraining
```

也就是说，real robot trajectory 不只是教模型“怎么动”，还参与构建跨 embodiment 的 latent action space。论文在 pre-training 细节中明确说，robot datasets 会同时使用 ground-truth robot actions 和 learned latent actions 作为 flow-matching targets。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

## 7. LAPA / Latent Action

### 7.1 LAPA 是什么

LAPA 可以理解为：

```text
Latent Action Pretraining from Videos
```

在 GR00T N1 中，LAPA 被当成一个 **虚拟 embodiment**。它不是一个真实机器人，也不是一个真实 motor action space，而是：

```text
LAPA embodiment:
    action = 从视频帧对中抽取的 latent action z_t
```

对于没有真实 action 的人类视频或 neural trajectories，模型没法直接训练：

```text
observation → robot action
```

所以作者训练一个 VQ-VAE 风格的 latent action model。

---

### 7.2 Latent Action 如何生成

输入当前帧和未来帧：

$$
x_t,\ x_{t+H}
$$

encoder 输出 latent action：

$$
z_t
$$

decoder 接收：

$$
x_t,\ z_t
$$

并重建：

$$
x_{t+H}
$$

训练好后，作者取 encoder 的 continuous pre-quantized embedding 作为 latent action label。这个 label 会作为 flow-matching target，用于训练 GR00T N1。论文还指出，在 heterogeneous data 上共同训练 VQ-VAE，有助于让不同 robot embodiment 和 human embodiment 共享一个 latent action space。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

直觉上，latent action 学到的不是：

```text
GR-1 第 3 个关节增加 0.02 rad
```

而更像：

```text
右手向左移动
手靠近物体
物体被抓起
物体从 A 移动到 B
```

所以 LAPA 更偏向“视觉变化的抽象动作语义”。

---

## 8. Neural Trajectories

### 8.1 什么叫 Neural Trajectory

“Neural trajectory” 这个词确实容易误解。它不是传统机器人轨迹，也不是仿真器 rollout，而是：

> **由 image-to-video / video generation model 生成出来的一段机器人操作视频。**

流程是：

```text
真实机器人初始帧
    + 新语言指令
        ↓
image-to-video generation model
        ↓
生成一段看起来像机器人执行任务的视频
```

论文使用 88 小时 in-house teleoperation 数据 fine-tune image-to-video model，然后从已有 initial frames 出发，用 novel language prompts 生成 827 小时视频数据，相当于把真实数据扩充约 10 倍。你标记的 “88 hours”、“827 hours”、“novel language prompts”、“counterfactual trajectories” 是理解这部分的关键。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

### 8.2 Neural Trajectory 的生成与过滤

为了增加 diversity，作者先用 commercial-grade multimodal LLM 检测 initial frame 中的物体，然后生成更多类似：

```text
pick up {object} from {location A} to {location B}
```

的可行组合。生成视频后，再用 multimodal LLM 作为 judge，输入下采样的 8 帧，过滤不符合语言指令的视频；不符合的还会被 re-caption。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

这说明 neural trajectories 不是任意生成的视频，而是经过：

```text
prompt generation
video generation
filtering
re-captioning
pseudo-action labeling
```

的一套数据增强流程。

---

### 8.3 Neural Trajectory 为什么有用

它的核心价值是生成 counterfactual 数据。例如同一张初始帧，可以生成：

```text
用右手拿黄瓜放进篮子
用左手拿喷雾瓶放进篮子
把苹果从砧板放到锅里
把土豆放进微波炉并关门
```

图 5 展示了这类由同一 initial frame 出发、换 prompt 后生成的不同轨迹。标记里也强调了 “same initial frames but with different prompts”、“replace the object to pick up”、“challenging to generate in simulation”等内容。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

所以 neural trajectories 主要补的是：

```text
真实机器人数据中没有覆盖到的视觉变化和任务变化
```

但它原始没有 action，因此需要 LAPA 或 IDM 补标签。

---

## 9. IDM：Inverse Dynamics Model

### 9.1 为什么需要 IDM

虽然 LAPA 能把无动作视频变成 latent action label，但 latent action 不是具体机器人能执行的 motor action。为了让 generated videos 更接近真实机器人控制数据，作者训练了 IDM：

```text
IDM:
    输入当前帧和未来帧
    输出中间 action chunk
```

形式上是：

$$
\mathrm{IDM}(x_t, x_{t+H})
\rightarrow
\hat{A}_{t:t+H-1}
$$

训练 IDM 时使用真实机器人数据，因为真实数据有 action 真值：

$$
A_{t:t+H-1}
=
[a_t, a_{t+1}, \dots, a_{t+H-1}]
$$

训练好后，IDM 可以给 neural trajectories 打 pseudo-action labels。论文附录 F 说明，IDM condition on current and future frames，训练目标是生成两帧之间的 action chunks；架构使用 System 1 Diffusion Transformer + SigLIP-2 vision embeddings，并用 flow-matching objective 训练。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

### 9.2 Latent Action 和 IDM 的区别

| 项目 | LAPA / Latent Action | IDM |
|---|---|---|
| 输入 | 当前帧 + 未来帧 | 当前帧 + 未来帧 |
| 监督 | 重建未来帧 | 真实 action chunk |
| 输出 | latent action code | robot action / pseudo-action |
| 是否可直接控制机器人 | 否 | 更接近可以 |
| 作用 | 跨 embodiment 表示学习 | 给 neural trajectory 打 action 伪标签 |
| 依赖真实 action 数据 | 弱 | 强 |

前面我们讨论过，IDM 隐式要求模型理解 3D 空间、手-物体-目标之间的关系、接触关系和常见运动路径。它不是显式 3D world model，但它要从 2D 图像变化中反推 3D 机器人动作，因此问题本身很难。GR00T N1 的做法是把问题限制在固定 embodiment、固定任务分布、固定 horizon 和真实 action 监督下，使 IDM 成为一个可用的 pseudo-label 工具。

---

## 10. Simulation Trajectories

### 10.1 为什么 Simulation Trajectories 有动作真值

Simulation trajectories 来自物理模拟器和自动数据生成系统，所以每一步 action command 是已知的。机器人在模拟器里执行控制命令，系统能记录：

```text
observation
robot state
action command
success / failure
```

这些 action command 就是 simulation trajectory 的 action ground truth。论文表 1 明确说，simulation trajectories 依赖 physics simulator，并使用 DexMimicGen-based automated data generation system。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

### 10.2 DexMimicGen 的机制

论文使用 DexMimicGen 从少量 human demonstrations 生成大量仿真轨迹。它的核心流程是：

```text
少量人类示教
    ↓
分解为 object-centric subtasks
    ↓
切成多个 manipulation segments
    ↓
根据新环境中的物体位置变换这些 segments
    ↓
保持 end-effector 与 object 的相对 pose
    ↓
插值连接当前状态与 transformed segment
    ↓
在模拟器中 replay
    ↓
只保留成功轨迹
```

你标记的 “small set of human demonstrations”、“decomposed”、“subtask involving a single object”、“adapted to new environments”、“interpolates movements” 都是在强调这套机制。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

“根据新环境中物体位置变换和拼接轨迹”的意思是：原始示教不是被当成绝对坐标路径死记，而是按“手相对于物体怎么接近、怎么抓、怎么放”来切分和迁移。比如原来苹果在桌子左边，新的仿真场景中苹果在右边，那么抓取片段会被平移/旋转到新苹果附近，同时保持 end-effector 和苹果之间的相对姿态。

---

### 10.3 Simulation Data 的规模

论文说，考虑 pre-training 和 post-training 数据，他们生成了：

```text
780,000 simulation trajectories
≈ 6,500 hours human demonstration equivalent
生成用时 11 hours
```

这就是 simulation trajectories 的核心优势：**大规模、有动作标签、成本低**。但缺点也很明显：它们有 sim-to-real gap，尤其在摩擦、接触、柔性物体、液体等场景上可能不真实。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

## 11. Pre-training 阶段

Pre-training 目标是训练一个 generalist checkpoint，覆盖多机器人、多任务、多数据来源。论文明确说，GR00T N1 在 pre-training 阶段使用 flow-matching loss，在 diverse embodiments 和 data sources 上训练。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

不同数据的训练 target 不一样：

| 数据来源 | Pre-training target |
|---|---|
| Human videos | learned latent actions |
| Real robot datasets | ground-truth robot actions + learned latent actions |
| Neural trajectories | latent actions + IDM predicted actions |
| Simulation trajectories | simulation ground-truth actions，也可参与 latent/action 统一 |

这张表是理解全文的关键。它说明 GR00T N1 不是简单“把数据混起来”，而是根据每类数据是否有 action、是否真实、是否可执行，设计不同的监督信号。

---

## 12. Post-training 阶段

Post-training 是把 pre-trained model 适配到单个 embodiment / 下游任务。

论文说，在 post-training 中，模型会在对应 single embodiment 的数据集上 fine-tune；和 pre-training 类似，language component of VL backbone 保持 frozen，其他部分继续 fine-tune。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

### 12.1 Low-data Regime

Low-data regime 指下游任务只有很少真实示教数据。真实机器人实验里，论文比较了：

```text
10% data
full data
```

仿真实验里，则比较：

```text
30 / 100 / 300 demonstrations per task
```

你标记的 “only 10%” 和 “low-data regime” 指的就是这种设定。注意论文还强调：在 low-data regime 下，IDM 也只用 low-data 训练，避免用 full data 训练 IDM 导致信息泄漏。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

### 12.2 Post-training with Neural Trajectories

为了解决 post-training 数据不足，作者会针对每个 downstream task 继续生成 neural trajectories。

流程是：

```text
少量真实下游示教
    ↓
fine-tune video generation model
    ↓
生成任务相关 neural trajectories
    ↓
用 LAPA 或 IDM 标注 pseudo-actions
    ↓
真实轨迹与 neural trajectories 按 1:1 ratio co-train
```

论文中明确写到，post-training 时 real-world trajectories 和 neural trajectories 使用 1:1 sampling ratio co-train。对于真实机器人任务，他们只用 10% post-training 数据训练视频生成模型，以模拟真实 low-data 场景。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

## 13. 数据格式与 Action Space 统一

标记 PDF 中后半部分还强调了系统设计。GR00T N1 的数据格式基于 LeRobot，但为 cross-embodiment VLA 增加了额外约束：

```text
modality.json
fine-grained modality specification
multiple annotation support
rotation type specification
```

这些设计的作用是：让不同机器人的 state/action 不再是难以解释的 monolithic vector，而是被拆分成有语义的字段，比如 end-effector position、orientation、gripper state 等。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

Action space 标准化包括：

```text
end-effector rotation state → 6D rotation representation
end-effector rotation action → axis-angle
joint/action/EEF position 等做 min-max normalization
state/action ordering 统一为 left arm 到 right arm
```

这部分虽然不是模型结构，但对跨 embodiment 训练非常关键。没有这种数据规范化，所谓 “unified model” 很难成立。

---

## 14. Auxiliary Object Detection Loss

论文还加入了一个辅助 object detection loss，用于增强 spatial understanding。具体做法是：

1. 用 OWL-v2 给每帧中的 target object 标注 bounding box；
2. 取 bbox center，归一化到图像宽高；
3. 在 final vision-language embedding tokens 上加一个 linear layer；
4. 预测目标物体中心坐标；
5. 使用 squared loss。

最终 loss 是：

$$
L = L_{fm} + L_{det}
$$

这个辅助 loss 的意义是：让 VLM/Policy 不只是学 action，还被迫学会根据语言定位目标物体。对 language-conditioned manipulation 很有帮助。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

## 15. 实验结果

### 15.1 Simulation Results

仿真 benchmark 包括：

```text
RoboCasa Kitchen: 24 tasks
DexMimicGen Cross-Embodiment Suite: 9 tasks
GR-1 Tabletop: 24 tasks
```

使用 100 demonstrations per task 时，结果如下：

| Method | RoboCasa | DexMG | GR-1 | Average |
|---|---:|---:|---:|---:|
| BC Transformer | 26.3% | 53.9% | 16.1% | 26.4% |
| Diffusion Policy | 25.6% | 56.1% | 32.7% | 33.4% |
| GR00T-N1-2B | 32.1% | 66.5% | 50.0% | 45.0% |

GR00T N1 在三个 benchmark 上都超过 baseline，尤其在 GR-1 任务上提升明显。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

### 15.2 Real-world Results

真实 GR-1 benchmark 上，平均成功率如下：

| Method | 10% Data | Full Data |
|---|---:|---:|
| Diffusion Policy | 10.2% | 46.4% |
| GR00T-N1-2B | 42.6% | 76.8% |

这个结果说明 GR00T N1 的 pre-training 显著提升了 post-training 的样本效率。特别是 GR00T-N1-2B 只用 10% data 就达到 42.6%，接近 Diffusion Policy 用 full data 的 46.4%。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

### 15.3 Neural Trajectories Ablation

在 post-training 中加入 neural trajectories 后，RoboCasa 三个数据 regime 的平均提升分别为：

```text
30 demos/task: +4.2%
100 demos/task: +8.8%
300 demos/task: +6.8%
```

真实 GR-1 的 8 个任务平均提升：

```text
+5.8%
```

论文还观察到：低数据时 LAPA 略优于 IDM；随着数据增加，IDM pseudo-action label 越来越接近真实 action，因此 IDM 的效果更强。真实 GR-1 对作者来说属于相对 high-data regime，所以 real-world neural trajectory co-training 只用了 IDM actions。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

## 16. 一个重要现象：Post-training 也可能遗忘 Pre-training 能力

论文 qualitative result 中有一个很有意思的例子：pretrained checkpoint 在 “pick up the red apple and place it in the basket” 任务中，虽然苹果放在两只手左侧，它仍能用左手抓取，handover 给右手，再放入篮子。但 post-trained checkpoint 反而失败，因为 post-training 数据全是右手操作，没有 inter-hand transfer。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

这说明：

```text
pre-training 学到的泛化能力
    可能在 narrow post-training 数据上被覆盖
```

这对 VLA 实践很重要。下游 fine-tuning 不只是提升任务性能，也可能损失一些 generalist behaviors。

---

## 17. 论文贡献总结

### 17.1 模型贡献

GR00T N1 把 VLM reasoning 和 continuous action generation 结合起来：

```text
Eagle-2 VLM:
    理解图像和语言

DiT action model:
    通过 flow matching 输出连续动作 chunk
```

它不是 RT-2 那种把 action 离散成 text token 的路线，而是更接近 π₀ / Diffusion Policy 的连续动作生成路线。

---

### 17.2 数据贡献

这是文章最大的贡献。GR00T N1 的数据体系可以总结为：

```text
real robot trajectories:
    提供真实控制 grounding

human videos:
    提供大规模 hand-object interaction / affordance prior

simulation trajectories:
    提供大规模、有动作真值的低成本训练数据

neural trajectories:
    用视频生成模型扩充 counterfactual 场景，再用 LAPA / IDM 补动作标签
```

核心不是哪一种数据最强，而是把不同可靠性、不同规模、不同 embodiment-specific 程度的数据统一进 VLA 训练。

---

### 17.3 训练范式贡献

GR00T N1 给出了一套比较完整的机器人 foundation model 训练范式：

```text
Pre-training:
    多数据源、多 embodiment、统一 flow-matching policy

Post-training:
    单 embodiment / 下游任务少量数据适配

Data augmentation:
    用 video generation model 生成 neural trajectories
    用 LAPA / IDM 为无 action 视频补标签
```

---

## 18. 局限性

论文自己也承认，GR00T N1 当前主要集中在：

```text
short-horizon tabletop manipulation
```

还没有解决长时序 loco-manipulation，也没有真正覆盖开放世界的复杂任务。合成数据也有明显限制：

```text
simulation trajectories:
    有 sim-to-real gap

neural trajectories:
    可能不符合物理规律
    需要过滤
    原始没有 action label

IDM pseudo-label:
    依赖真实 action 数据质量
    从两帧图像反推动作本身很难
```

作者也指出，未来需要更强的 VLM backbone、更好的 synthetic data generation、更强的模型架构和 pre-training 策略。`GR00T N1 An Open Foundation Model for Generalist Humanoid Robots-with-annotations.pdf`

---

## 19. 最终理解

这篇文章可以这样理解：

> **GR00T N1 不是单纯提出了一个更大的 VLA 模型，而是提出了一套围绕 humanoid robot 的数据金字塔和统一训练系统。**

它的关键不只是：

```text
VLM + DiT
```

而是：

```text
real data
+ simulation trajectories
+ neural trajectories
+ human videos
+ LAPA
+ IDM
+ pre-training/post-training
```

共同构成的机器人 foundation model pipeline。

最值得记住的一句话是：

> **真实数据负责 grounding，仿真数据负责低成本动作监督，neural trajectories 负责 counterfactual 场景扩增，人类视频负责行为先验；GR00T N1 的核心就是把这些数据源统一成一个可训练、可适配、可部署的 VLA 系统。**


---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)

## 相关笔记
- [[Robot/ChatGPT-RDT-1B|RDT-1B]]
- [[Robot/ChatGPT-Diffusion Policy 概述|Diffusion Policy 概述]]
- [[Robot/PI/Pi0_7_technical_report|π0.7 技术报告]]
- [[Robot/ChatGPT-RT-2 论文综述|RT-2 论文综述]]
- [[Robot/ChatGPT-ALOHA硬件与ACT算法|ALOHA 硬件与 ACT 算法]]
