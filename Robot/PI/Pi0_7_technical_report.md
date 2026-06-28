---
title: π0.7 技术报告
type: paper_note
topic: robotics_foundation_model
status: mature
importance: high
updated: 2026-06-26
tags:
  - pi0.7
  - physical-intelligence
  - vla
  - robot-foundation-model
  - steerable-policy
  - subgoal
  - subtask
  - action-chunking
  - flow-matching
  - mem
  - recap
  - robotics
---
# π0.7 技术报告：Steerable Generalist VLA 的模型结构、训练机制与推理系统

> 论文：**π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities**  
> 核心主题：通过 rich context / prompt conditioning，把多源、混合质量、不同策略的数据组织成一个可 steer 的机器人动作生成模型。

---

## 目录

1. [一句话总结](#1-一句话总结)
2. [核心问题：为什么普通 VLA 不够](#2-核心问题为什么普通-vla-不够)
3. [π0.7 的整体建模形式](#3-π07-的整体建模形式)
4. [模型结构](#4-模型结构)
5. [KI：Knowledge Insulation 训练方式](#5-kiknowledge-insulation-训练方式)
6. [Action expert 与 50-step action chunk](#6-action-expert-与-50-step-action-chunk)
7. [MEM-style video history encoder 与长时序问题](#7-mem-style-video-history-encoder-与长时序问题)
8. [Proprioception history 与 history dropout](#8-proprioception-history-与-history-dropout)
9. [Prompt / Context 设计](#9-prompt--context-设计)
10. [Subtask instruction：长任务的语义分解](#10-subtask-instruction长任务的语义分解)
11. [Subgoal images 与 world model](#11-subgoal-images-与-world-model)
12. [Episode metadata：mixed-quality data 的关键](#12-episode-metadatamixed-quality-data-的关键)
13. [π0.7 与 π\*0.6 / advantage indicator 的关系](#13-π07-与-π06--advantage-indicator-的关系)
14. [Prompt dropout 与 test-time flexibility](#14-prompt-dropout-与-test-time-flexibility)
15. [RTC：Training-time Real-Time Action Chunking](#15-rtctraining-time-real-time-action-chunking)
16. [Inference-time CFG：具体执行方式](#16-inference-time-cfg具体执行方式)
17. [Positive / Negative branch packing 与 attention tree](#17-positive--negative-branch-packing-与-attention-tree)
18. [异步推理系统：subtask / subgoal / VLA 如何切换](#18-异步推理系统subtask--subgoal--vla-如何切换)
19. [训练数据与 π\*0.6 行为蒸馏](#19-训练数据与-π06-行为蒸馏)
20. [实验结论与 ablation](#20-实验结论与-ablation)
21. [技术贡献总结](#21-技术贡献总结)
22. [局限性与需要谨慎理解的点](#22-局限性与需要谨慎理解的点)
23. [最终总结](#23-最终总结)

---

## 1. 一句话总结

**π0.7 是一个 steerable generalist robotic foundation model。它不是只根据语言指令预测动作，而是把 task instruction、subtask instruction、subgoal images、episode metadata、control mode、多帧历史观测等都作为 context，让模型从大规模、多源、混合质量的数据中学习一个可控的条件行为分布。**

传统 VLA 更像：

$$
\pi(a_{t:t+H} \mid o_{t-T:t}, \ell)
$$

π0.7 更像：

$$
\pi(a_{t:t+H} \mid o_{t-T:t}, C_t)
$$

其中：

$$
C_t = \{\ell, \hat{\ell}_t, g_t, m, c\}
$$

含义为：

- $\ell$：全局 task instruction
- $\hat{\ell}_t$：当前 subtask instruction
- $g_t$：当前 subtask 对应的 subgoal images
- $m$：episode metadata，包括 speed、quality、mistake
- $c$：control mode，例如 joint 或 end-effector

因此，π0.7 的核心不是单纯扩大模型或数据，而是：

> **用 rich prompt/context 将异构机器人经验变成一个可被 steering 的条件行为模型。**

---

## 2. 核心问题：为什么普通 VLA 不够

传统 VLA 通常输入当前观测和任务语言，然后输出动作：

```text
observation + "clean the kitchen" -> action chunk
```

这种方式在高质量、干净、策略单一的数据上可以工作，但机器人基础模型面对的数据往往非常复杂：

- 高质量 teleoperation demonstrations
- 低质量 demonstrations
- failure episodes
- autonomous rollout
- RL specialist rollout
- human intervention
- 不同机器人平台
- 不同 control mode
- egocentric human video
- web multimodal data

同一个任务、相似状态下，数据中可能同时存在：

- 快速成功动作
- 慢速成功动作
- 绕路动作
- 抓空动作
- 中途失败后恢复动作
- 执行错误 subtask 的动作

如果直接做普通 behavior cloning，模型容易把这些行为平均掉，学出一个低质量的动作分布。

π0.7 的解决方法是：**保留这些多样化数据，但给每条数据加 context，让模型知道这条数据的质量、速度、是否有错误、当前子任务、目标状态和控制方式。**

训练时模型学习：

$$
\pi(a \mid o, \text{quality}, \text{speed}, \text{mistake}, \text{subtask}, \text{subgoal})
$$

推理时固定：

$$
\text{quality}=5,\quad \text{mistake}=\text{false},\quad \text{speed}=\text{fast}
$$

这样模型不是平均所有轨迹，而是被 steer 到训练数据中的高质量行为模式。

---

## 3. π0.7 的整体建模形式

论文给出的 VLA 训练目标可以写成：

$$
\max_\theta \mathbb{E}_{D}
\left[
\log \pi_\theta(a_{t:t+H} \mid o_{t-T:t}, C_t)
\right]
$$

其中：

- $o_{t-T:t}$：最近一段观测历史
- $a_{t:t+H}$：未来 action chunk
- $C_t$：rich context

由于 action expert 使用 flow matching，严格来说它优化的是 approximate lower bound，而不是显式 closed-form log-likelihood。

从系统角度看，π0.7 是：

```text
observation history
+ task instruction
+ dynamic subtask instruction
+ generated subgoal images
+ episode metadata
+ control mode
    ↓
VLM backbone + action expert
    ↓
50-step continuous action chunk
```

---

## 4. 模型结构

π0.7 是一个约 5B 参数的 VLA，主要由三部分组成：

1. **VLM backbone**
   - 约 4B 参数
   - 初始化自 Gemma3
   - 负责整合视觉、语言、metadata、subgoal 等上下文

2. **MEM-style video history encoder**
   - 处理多帧历史视觉输入
   - 对历史观测做 temporal 和 spatial compression
   - 输出固定数量 visual tokens，避免 token 数随历史帧数量线性增长

3. **Action expert**
   - 约 860M 参数
   - 一个较轻量的 transformer
   - 使用 flow matching objective 预测连续 action chunk
   - 固定处理 50 个 action tokens，即 50-step action chunk

整体结构：

```text
                         task instruction ℓ
                                  │
                                  ▼
                    high-level semantic policy / human
                                  │
                                  ▼
                    current subtask instruction ˆℓ_t
                                  │
                                  ▼
 current observation ──► world model ──► subgoal images
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────┐
│                         π0.7 VLA                            │
│                                                            │
│  Inputs:                                                   │
│    - multi-view observation history                         │
│    - proprioception history                                 │
│    - task instruction                                       │
│    - current subtask instruction                            │
│    - subgoal images                                         │
│    - episode metadata: speed / quality / mistake            │
│    - control mode: joint / ee                               │
│                                                            │
│  Backbone:                                                  │
│    Gemma3 VLM + MEM-style video history encoder             │
│                                                            │
│  Action expert:                                             │
│    860M transformer + flow matching                         │
│                                                            │
│  Output:                                                    │
│    50-step continuous action chunk                          │
└────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    execute 15 or 25 steps, then replan
```

---

## 5. KI：Knowledge Insulation 训练方式

π0.7 使用 **Knowledge Insulation, KI**。它的核心是：

> **VLM backbone 和 action expert 都参与 action 学习，但 flow matching loss 的梯度不反传进 VLM backbone。**

训练中有两条动作学习路径。

### 5.1 VLM backbone 路径：FAST token CE loss

VLM backbone 使用 FAST tokens 做监督。FAST tokens 可以理解为将连续动作序列 tokenization 后得到的离散 action tokens。VLM 用离散 cross-entropy loss 学习 action-aware representation：

$$
\mathcal{L}_{\text{FAST}}
=
\text{CE}(\text{FAST action tokens})
$$

因此，VLM 不是完全不学 action。它学的是离散 action token 表示。

### 5.2 Action expert 路径：flow matching loss

Action expert 使用 flow matching 学习连续动作：

$$
\mathcal{L}_{\text{FM}}
=
\left\|
v_\theta(x_\tau,\tau,C)
-
u^\star
\right\|^2
$$

其中：

- $x_\tau$：从噪声到真实 action chunk 的中间状态
- $v_\theta$：模型预测的 flow direction
- $u^\star$：目标速度场

### 5.3 梯度隔离

关键是：

$$
\nabla_{\text{VLM}} \mathcal{L}_{\text{FM}} = 0
$$

也就是说，action expert 可以 attend 到 VLM backbone 的 activations，但 flow matching loss 的梯度不会更新 VLM backbone。

训练图：

```text
FAST token CE loss
        ↓
更新 VLM backbone

flow matching loss
        ↓
更新 action expert
        × 不更新 VLM backbone
```

这样做的原因是：连续动作生成的 flow matching loss 相对不稳定，直接反传进 VLM 可能破坏视觉语言 backbone 的通用表示。KI 让 VLM 通过更稳定的离散 CE loss 训练，而 action expert 单独学习连续控制。

---

## 6. Action expert 与 50-step action chunk

π0.7 的 action expert 固定处理 50 个 action tokens，输出一个 50-step continuous action chunk：

$$
A_t = [a_t, a_{t+1}, \dots, a_{t+49}]
$$

需要注意：**50-step action chunk 不是整个任务长度，而是局部控制窗口。**

推理时，模型每次生成 50 步，但只执行其中 15 或 25 步，然后重新根据最新状态规划下一段：

```text
t = 0:
    生成 a_0 ... a_49
    执行 a_0 ... a_14 或 a_0 ... a_24

t = 15 / 25:
    重新观察
    更新 subtask / subgoal / metadata
    重新生成下一段 50-step chunk
```

因此，不同任务的总时长由 rolling inference 的次数决定，而不是由 action expert 的输出长度决定。

短任务可能几个 chunk 完成；长任务可能需要几十个甚至上百个 chunk。

---

## 7. MEM-style video history encoder 与长时序问题

π0.7 使用 **MEM-style video history encoder**，主要解决短期视觉历史问题。

输入包括：

- front camera
- two wrist cameras
- optional rear camera
- 每个 view 最多 6 个 history frames
- history frame stride 为 1 秒
- 图像 resize 到 448×448
- subgoal images 也通过同一个 vision encoder 处理

MEM-style encoder 的作用是：

```text
多帧历史图像
    ↓
temporal + spatial compression
    ↓
固定数量 visual tokens
```

这让模型能看到最近几秒发生了什么，例如：

- 门是否已经打开
- 物体是否已经被抓住
- 上一次抓取是否失败
- 手臂是否已经移动到目标附近
- 物体被遮挡前在哪里

但是，π0.7 论文中明确使用的是 MEM-style video history encoder，并没有明确说完整采用 MEM 的 long-horizon language memory summarization 模块。

因此，对长时序任务和避免重复动作，π0.7 主要依赖：

1. 最近几秒的 video history
2. 当前 observation
3. high-level policy 或 human coaching 生成动态 subtask
4. high-level policy 可使用 past subtask instruction history
5. subgoal image 随 subtask 或时间刷新

这意味着 π0.7 没有一个硬性的“保证不重复”的模块。如果 high-level policy 误判任务阶段，或者 observation 不足以判断某一步是否完成，模型仍可能重复做同一个 subtask。

---

## 8. Proprioception history 与 history dropout

π0.7 follow MEM 的方式，用 linear projection 将 proprioceptive state 映射到 backbone embedding dimension：

$$
e^q_t = W_q q_t
$$

每个 history state 都作为一个 token 输入模型。

如果某个 history frame 被 dropout，对应的 state token 也会被 mask 掉。

原因是：图像历史和 proprioception 历史是同一时刻 observation 的两个模态。如果丢掉 $I_{t-k}$，但保留 $q_{t-k}$，模型会看到一个不完整的历史时刻：

```text
image_{t-k}: missing
state_{t-k}: visible
```

这会破坏多模态历史的一致性。因此，当某个历史帧被 dropout 时，对应 state token 也要 mask，表示这个历史时刻整体不可见。

为什么用 mask，而不是直接删除 token？合理原因包括：

- 固定 token layout 更便于 batching
- 保留 temporal slot，模型知道哪个时间点缺失
- attention mask 更容易实现
- 不同样本的 history dropout 不会导致动态 sequence layout
- 多 view、多 history、多 subgoal 的 block-causal attention 更容易维护

---

## 9. Prompt / Context 设计

π0.7 的最大创新之一是将 prompt 扩展为多模态、多属性 context。

完整 prompt 示例：

```text
<Multi-view observation>
<Multi-view subgoals>

Task: peel vegetables.
Subtask: pick up the peeler.
Speed: 8000.
Quality: 5.
Mistake: false.
Control Mode: joint.
<Proprioception>
```

主要包括五类信息：

1. task instruction
2. subtask instruction
3. subgoal images
4. episode metadata
5. control mode

这种 prompt 不只是告诉模型“做什么”，还告诉模型“怎么做”“做得好不好”“目标状态长什么样”。

---

## 10. Subtask instruction：长任务的语义分解

π0.7 不只输入整体 task instruction，还输入当前阶段的 semantic subtask。

例如：

```text
Task: put food on table

Subtasks:
    push the open button on the microwave
    pick up the plate of food in the microwave
    move to the dining table
    put the plate with food on the dining table
    close the microwave
```

这里 task instruction $\ell$ 可以固定，但 subtask instruction $\hat{\ell}_t$ 会随时间变化。

也就是说：

```text
全局任务：toast a bagel
当前子任务 1：open toaster oven
当前子任务 2：pick up the white plate
当前子任务 3：put bagel on the plate
```

subtask 可以由 high-level policy 生成，也可以由 human coaching 给出。high-level policy 根据当前 observation、task specification 和 past subtask instruction history 预测下一步 subtask。

这就是 π0.7 处理长任务的重要机制：

```text
长任务 = 动态 subtask sequence
低层控制 = 每个 subtask 下滚动生成 50-step action chunks
```

---

## 11. Subgoal images 与 world model

语言有时无法精确描述低层执行细节。例如：

```text
open the fridge door
```

它没有说明：

- 手应该抓哪里
- 门应该开到什么角度
- wrist view 下手和门把手应该是什么相对位置
- 当前 subtask 完成后的视觉状态是什么样

π0.7 引入 **multi-view subgoal images**，表示当前 subtask 成功推进后的 near-future visual state。

可以理解为：

```text
subtask instruction:
    告诉模型要做什么

subgoal image:
    告诉模型做到之后应该长什么样
```

### 11.1 World model 的输入输出

π0.7 使用 lightweight world model 生成 subgoal images。world model 输入：

$$
(o_t, \hat{\ell}_t, m)
$$

输出：

$$
g_t^\star
$$

也就是：

```text
current observation
+ current subtask instruction
+ episode metadata
    ↓
future subgoal image
```

world model 输入的是当前 subtask，而不是完整 subtask list。因此不会出现“输入全部 subtask 导致不知道生成哪一步未来图像”的问题。

### 11.2 World model 训练

训练时，subgoal ground truth 来自 segment 末尾图像：

$$
g_t^\star = o_{t_{\text{end}}}
$$

真实 subgoal image 的采样策略：

- 25% 概率采样 segment end image
- 75% 概率从当前时刻未来 0–4 秒内均匀采样 future image

为了减轻 train-test mismatch，π0.7 训练时也会使用 world model 生成的 subgoal images 作为 context，而不是只使用真实未来图像。

### 11.3 推理时刷新 subgoal

推理时，如果满足以下任一条件，就异步刷新 subgoal：

```text
1. subtask changed
2. 距离上次生成 subgoal 超过 4 秒
```

新 subgoal 返回后，会替换缓存中的 latest_subgoal。主 VLA 在下一次 chunk inference 时使用最新可用的 subgoal。

---

## 12. Episode metadata：mixed-quality data 的关键

π0.7 的 episode metadata 包括：

```text
Overall speed
Overall quality
Mistake
```

### 12.1 Overall speed

speed 实际上是 episode length in timesteps。episode 越短，通常表示执行越快。

它会按 500 steps 离散化，例如：

```text
1750–2250 steps -> 2000 steps
```

推理时，speed 设置为每个任务 episode length 分布的第 15 百分位。也就是说，对某个任务的训练 episodes 按长度从短到长排序，取前 15% 位置的长度，作为“比较快但不极端”的速度目标。

### 12.2 Overall quality

quality 是 1 到 5 分，5 最高。推理时固定为：

```text
Quality: 5
```

### 12.3 Mistake

mistake 表示当前 action segment 是否发生错误，例如抓取失败或执行错误 subtask。它是人工粗标。推理时固定为：

```text
Mistake: false
```

### 12.4 metadata 的作用

metadata 让模型可以同时学习：

- 高质量数据
- 低质量数据
- 失败数据
- autonomous rollout
- RL specialist rollout

但推理时选择：

```text
fast + high quality + no mistake
```

因此，metadata 是 π0.7 能够利用 mixed-quality data 的核心机制。

---

## 13. π0.7 与 π\*0.6 / advantage indicator 的关系

π0.7 的 mistake / quality / speed 与 π\*0.6 的 advantage indicator 在形式上相似，都是 conditional policy learning：

```text
训练时给数据加质量相关条件
推理时把条件设成“好”
```

但本质不同。

π0.7 的 mistake 是人工标注的错误描述：

```text
这段有没有明显犯错？
```

π\*0.6 的 advantage indicator 是 value / reward 驱动的策略改进信号：

```text
这个动作是否相对更有利于任务成功？
```

因此：

$$
\text{mistake} \neq \text{advantage}
$$

一个没有 mistake 的动作也可能很慢、低效，因此 advantage 不一定高。反过来，一个动作可能看起来不标准，但如果能显著推进任务，advantage 可能为正。

所以，把 π0.7 的 mistake 换成 advantage indicator，只能吸收 π\*0.6 的一部分思想；不能覆盖完整 π\*0.6 RL pipeline，因为 π\*0.6 还包括：

- value function training
- advantage estimation
- thresholding / binarization
- on-robot rollout
- expert intervention
- iterative policy improvement

---

## 14. Prompt dropout 与 test-time flexibility

π0.7 训练时随机 dropout prompt 组件，让模型推理时可以使用不同 context 组合。

具体策略：

- subgoal images 只加到 batch 中 25% examples
- 有 subgoal image 的 examples 中，subtask instruction 30% dropout
- episode metadata 整体 15% dropout
- speed / quality / mistake 各自额外 5% dropout
- control mode 不 dropout

为什么要 dropout？

第一，增加 test-time flexibility。模型可以在没有 subgoal 或没有 metadata 的情况下运行。

第二，避免模型过度依赖某个 prompt 组件。例如 subgoal image 会让 action prediction 近似 inverse dynamics，训练很容易，但可能削弱语言理解和自主泛化。

第三，为 inference-time CFG 提供基础。因为模型训练过 conditional 和 unconditional / partially dropped context，所以推理时可以构造 positive 和 negative branch。

---

## 15. RTC：Training-time Real-Time Action Chunking

π0.7 使用 training-time RTC 来解决异步推理下的动作连续性问题。

### 15.1 问题：异步推理导致 chunk 边界不连续

VLA 推理需要时间，但机器人控制不能停。

假设模型开始生成新 chunk 时，推理需要 $r$ 个控制周期。那在这 $r$ 个周期内，机器人只能继续执行旧 chunk 中已经计划好的动作。

如果新 chunk 返回后直接从第一步硬切，可能出现：

```text
旧 chunk 当前动作
    ↓
新 chunk 第一帧动作
```

二者差异很大，导致机械臂抖动、不连续或 jerky motion。

### 15.2 RTC 的核心思想

训练时随机模拟 0 到 12 timesteps 的 delay。设 delay 为 $r$，则 action chunk 分成：

$$
A = [A^{\text{pre}}, A^{\text{post}}]
$$

其中：

- $A^{\text{pre}}$：前 $r$ 个动作，表示已经 committed 的动作 prefix
- $A^{\text{post}}$：后续还可以重新规划的动作 postfix

训练时，prefix 作为条件输入，postfix 才是要生成的部分。因此 RTC 更合理地建模为：

$$
p(A^{\text{post}} \mid o, C, A^{\text{pre}})
$$

而不是：

$$
p(A \mid o, C)
$$

### 15.3 RTC 与 flow matching 的关系

普通 flow matching 通常对整个 action chunk 使用同一个 flow time $\tau$：

$$
X_\tau = (1-\tau)Z + \tau A
$$

但 RTC 中，prefix 已经是 clean action，相当于 $\tau=1$；postfix 仍处于当前 denoising time $\tau$。从完整 chunk 角度看，不同位置不是同一个 $\tau$。

这并不是破坏 flow matching，因为 prefix 不再被看作要生成的变量，而是条件。真正做 flow matching 的是 postfix：

$$
X_\tau^{\text{post}}
=
(1-\tau)Z^{\text{post}}
+
\tau A^{\text{post}}
$$

loss 只作用在 postfix 上：

$$
\mathcal{L}_{\text{RTC}}
=
\left\|
v_\theta^{\text{post}}(P,X_\tau^{\text{post}},\tau,C)
-
(A^{\text{post}}-Z^{\text{post}})
\right\|^2
$$

其中：

$$
P = A^{\text{pre}}
$$

所以 RTC 可以理解为 **conditional / masked flow matching**，类似图像 inpainting：

```text
image inpainting:
    已知区域固定，只生成 masked 区域

RTC:
    已 committed 动作固定，只生成后续动作
```

### 15.4 π0.7 中 RTC 的作用

π0.7 训练时模拟 0–12 timesteps delay。在 50Hz 机器人上，12 steps 对应：

$$
12 \times 20\text{ms} = 240\text{ms}
$$

这样模型训练时就见过：

```text
前几步动作已经确定
后面动作需要接着生成
```

推理时，在旧 chunk 继续执行、新 chunk 异步生成的情况下，模型更容易生成与旧动作平滑衔接的新 chunk。

---

## 16. Inference-time CFG：具体执行方式

π0.7 的 CFG 发生在 **每一次 action denoising / flow sampling step** 中，而不是生成完动作之后再后处理。

假设当前 noisy action chunk 是 $x_i$，当前 flow time 是 $\tau_i$。

π0.7 构造两个 context。

### 16.1 Positive context

带完整目标条件，例如：

```text
Task: fold the shirt
Subtask: fold the left sleeve
Subgoal image: ...
Speed: 2000
Quality: 5
Mistake: false
Control Mode: joint
```

记为：

$$
C_{\text{pos}}
$$

### 16.2 Negative / unconditional context

去掉被 CFG 的条件。π0.7 主要对 episode metadata 做 CFG，所以 negative context 可以理解为：

```text
Task: fold the shirt
Subtask: fold the left sleeve
Subgoal image: ...
metadata dropped
Control Mode: joint
```

记为：

$$
C_{\text{neg}}
$$

### 16.3 分别预测两个方向

对同一个 noisy action $x_i$ 和同一个 flow time $\tau_i$，分别预测：

$$
v_{\text{pos}}
=
v_\theta(x_i,\tau_i,o,C_{\text{pos}})
$$

$$
v_{\text{neg}}
=
v_\theta(x_i,\tau_i,o,C_{\text{neg}})
$$

然后合成 CFG direction：

$$
v_{\text{cfg}}
=
v_{\text{pos}}
+
\beta
(
v_{\text{pos}}-v_{\text{neg}}
)
$$

即：

$$
v_{\text{cfg}}
=
(1+\beta)v_{\text{pos}}
-
\beta v_{\text{neg}}
$$

最后使用 $v_{\text{cfg}}$ 更新 action latent：

$$
x_{i+1}
=
x_i + \Delta\tau \cdot v_{\text{cfg}}
$$

实际 sampler 可能不是最简单的 Euler 更新，但核心逻辑是：

> **每一步 denoising 都分别计算 positive 和 negative 方向，再合成 guided direction。**

完整伪代码：

```text
x = GaussianNoise([50, action_dim])

for i in range(5):
    τ = schedule[i]

    C_pos = full_context_with_metadata
    C_neg = context_with_metadata_dropped

    v_pos = model(x, τ, obs_history, C_pos)
    v_neg = model(x, τ, obs_history, C_neg)

    v = v_pos + β * (v_pos - v_neg)

    x = flow_update(x, v, τ)

action_chunk = x
```

CFG 的作用不是让 denoising 计算更快，而是让动作更符合条件。图像 diffusion 的 CFG 是让图像更符合 text prompt；π0.7 的 CFG 是让动作更符合 metadata，例如 high quality、no mistake、fast speed。

---

## 17. Positive / Negative branch packing 与 attention tree

朴素 CFG 需要两次 forward：

```text
forward 1: positive context
forward 2: negative context
```

π0.7 为了高效推理，把 positive 和 negative example pack 到同一个 sequence 中，并构造 attention tree。

这里的 branch 不是两个不同网络分支，而是同一个 transformer sequence 中，由 attention mask 划分出的两个互不通信的 token 子图。

结构可以理解为：

```text
shared image / memory tokens
        │
        ├── negative branch
        │       text prompt without metadata
        │       flow actions (-)
        │
        └── positive branch
                text prompt with metadata
                flow actions (+)
```

两个 branch 共享 observation / image memory，但互相不能 attend：

```text
negative branch:
    可以看 shared image/memory
    可以看 negative prompt/action
    不能看 positive branch

positive branch:
    可以看 shared image/memory
    可以看 positive prompt/action
    不能看 negative branch
```

逻辑上等价于 batch 中放两个样本：

```text
sample 1 = obs + C_neg + x_i
sample 2 = obs + C_pos + x_i
```

但实现上通过 attention mask 和 sequence packing 复用共享 tokens，提高 inference efficiency。

---

## 18. 异步推理系统：subtask / subgoal / VLA 如何切换

π0.7 推理不是一个严格同步系统，而是多个异步模块共同运行：

1. high-level policy / human coaching 生成当前 subtask
2. world model 异步生成当前 subtask 的 subgoal image
3. VLA 根据最新可用 context 生成 action chunk
4. controller 持续执行当前 action chunk

系统中可以维护几个缓存变量：

```text
latest_subtask
latest_subgoal
latest_context
current_action_chunk
```

### 18.1 subtask 如何变化

task instruction $\ell$ 是固定的，例如：

```text
toast a bagel
```

但 subtask $\hat{\ell}_t$ 会随任务阶段变化：

```text
open toaster oven
pick up the white plate
put bagel on the plate
turn the knob
```

high-level policy 根据当前 observation、task specification 和 past subtask instruction history 生成当前 subtask。人类也可以通过 verbal coaching 直接给出 subtask。

### 18.2 subgoal 如何刷新

当满足以下任一条件时，world model 重新生成 subgoal：

```text
1. subtask changed
2. 距离上次生成 subgoal 超过 4 秒
```

world model 生成是 non-blocking async。机器人不会停下来等 subgoal image 返回。VLA 总是使用 latest available subgoal。新 subgoal 返回后，替换缓存中的 latest_subgoal，下一次 VLA chunk inference 使用新的 context。

### 18.3 VLA 如何切换

“切换”不是指立即中断当前 motor command，而是：

```text
subtask 更新 latest_subtask
subgoal 更新 latest_subgoal
下一次 VLA inference 读取新的 latest_context
生成新 action chunk
```

当前 action chunk 会继续执行，直到到了 $\hat{H}$ steps 或新 chunk 准备好。RTC 负责让新旧 chunk 动作衔接更平滑。

整体流程：

```text
当前 chunk 正在执行
        │
        ├── high-level policy 异步更新 subtask
        ├── world model 异步更新 subgoal
        │
每隔 15 / 25 steps:
        ↓
VLA 读取 latest_subtask + latest_subgoal + metadata
        ↓
生成新 50-step action chunk
        ↓
继续执行
```

---

## 19. 训练数据与 π\*0.6 行为蒸馏

π0.7 的训练数据包括：

- 多机器人、多任务 demonstrations
- static / mobile robots
- single-arm / bimanual robots
- lab-like / home-like / in-the-wild environments
- autonomous policy evaluation data
- human interventions within rollouts
- open-source robot datasets
- egocentric human video data
- web multimodal data
- object localization、VQA、text-only prediction、video-language tasks

一个重要点是，π0.7 使用了 π\*0.6 在 RL training 过程中收集的数据作为 additional examples。这相当于 distill π\*0.6 的 behavior，使 generalist π0.7 继承 RL-trained specialists 的能力。

因此，π0.7 能达到甚至超过 π\*0.6，不是因为“普通 BC 凭空超过 RL”，而是因为：

```text
π*0.6:
    用 RL 在单任务上搜索/优化出高性能行为

π0.7:
    把 π*0.6 RL rollout 作为监督数据吸收
    同时学习更大规模的 demo、failure、recovery、human video、web data
    用 metadata 区分好坏行为
    推理时 prompt 到 high-quality / fast / no-mistake 模式
```

---

## 20. 实验结论与 ablation

π0.7 主要展示了四类能力。

### 20.1 Out-of-the-box dexterity

π0.7 不做 task-specific post-training，也能执行 espresso、laundry folding、box building、peeling vegetables、take out trash 等任务，并匹配或超过 π\*0.6 / π0.6 specialists。

### 20.2 Instruction following

π0.7 能在 unseen kitchens / bedrooms 中 follow 多阶段语言指令，并处理复杂 referential instructions 和反 dataset bias 的指令。

### 20.3 Cross-embodiment transfer

π0.7 能将技能迁移到不同 morphology 的机器人平台，例如 UR5e 双臂系统上的 laundry folding。

### 20.4 Compositional generalization 与 coaching

π0.7 可以通过 step-by-step verbal coaching 执行新任务，如 air fryer、toasting bagel 等。之后可以用 coaching data 训练 high-level language policy，使模型自主完成这些任务。

### 20.5 Metadata / eval data ablation

论文比较了：

```text
π0.7
π0.7 no metadata
π0.7 no eval data
```

结果显示，π0.7 明显优于 no metadata 和 no eval data，尤其 throughput 差距明显。这说明：

1. autonomous / evaluation / RL rollout data 有用
2. metadata 对利用 mixed-quality data 至关重要
3. 没有 metadata，模型可能被低质量数据拖累
4. metadata 让模型可以在训练时吸收多种行为质量，在测试时选择高质量模式

---

## 21. 技术贡献总结

π0.7 的核心贡献可以总结为以下几点。

### 21.1 从 language-conditioned VLA 到 context-steerable VLA

它不只是用语言控制动作，而是用 rich context 控制动作分布。

### 21.2 让 mixed-quality robot data 可用

通过 episode metadata，π0.7 可以吸收低质量、失败和 autonomous data，而不是简单过滤掉。

### 21.3 将 specialist RL 能力蒸馏进 generalist model

π0.7 不直接对每个任务跑 RL，而是把 π\*0.6 的 RL rollout 作为 supervised data，使 generalist 继承 specialist 能力。

### 21.4 引入 subgoal image 作为低层执行目标

subgoal image 弥补语言在空间和执行细节上的不足。

### 21.5 组合 high-level subtask policy 与 low-level rolling action chunks

长任务不是通过一次性输出完整动作解决，而是通过动态 subtask + subgoal + rolling action chunk 实现。

### 21.6 使用 KI 稳定训练

VLM backbone 由 FAST token CE loss 训练，flow matching loss 只训练 action expert，避免破坏 VLM 表示。

### 21.7 使用 RTC 和 inference-time CFG 改善实时执行和条件控制

RTC 解决推理延迟下的动作连续性；CFG 放大 metadata 对动作生成的影响。

---

## 22. 局限性与需要谨慎理解的点

π0.7 很强，但不能过度简化。

### 22.1 不是单一端到端模型解决全部问题

π0.7 是一个系统，包括：

- high-level policy
- world model
- VLA backbone
- action expert
- metadata annotation
- prompt dropout
- CFG
- RTC
- asynchronous inference

### 22.2 没有明确完整使用 MEM long-horizon memory

π0.7 明确使用的是 MEM-style short-term video history encoder。长时序任务不重复主要依赖 high-level subtask policy 和视觉状态，而不是硬性 memory guarantee。

### 22.3 metadata 不等价于 advantage

mistake 是人工错误标签，quality 是人工质量评分，speed 是 episode length。它们可以 steer 行为，但不等价于 value-function-driven policy improvement。

### 22.4 RTC 不是全局最优保证

RTC 解决的是实时异步推理下的动作连续性，但已经 committed 的错误动作无法撤回。它只能让后续动作接上 prefix，而不是保证全局最优。

### 22.5 CFG 是条件放大器，不是万能控制器

CFG 可以让动作更符合 high-quality / fast / no-mistake 条件，但过强可能导致动作过于激进、偏离数据分布或降低稳定性。

### 22.6 π0.7 的强性能来自系统性叠加

π0.7 不是“BC 神奇超过 RL”。它依赖：

- 大模型容量
- 多源数据
- specialist rollout distillation
- metadata conditioning
- subgoal image grounding
- MEM-style history
- runtime CFG steering
- RTC 实时动作衔接

---

## 23. 最终总结

π0.7 是一个 steerable generalist robotic foundation model。它建立在 Gemma3 VLM、MEM-style video history encoder 和 flow matching action expert 之上，但真正的核心贡献是 **diverse context conditioning**。

它通过在 prompt 中加入：

- task instruction
- subtask instruction
- subgoal images
- episode metadata
- control mode
- history observations

将高质量 demo、低质量 demo、失败轨迹、autonomous rollout、π\*0.6 RL rollout、人类视频和 web data 组织成一个可控的条件行为分布。

它的策略可以写成：

$$
\pi(a_{t:t+H}
\mid
 o_{t-T:t},
 \ell,
 \hat{\ell}_t,
 g_t,
 m,
 c)
$$

其中：

- $\ell$ 控制全局任务
- $\hat{\ell}_t$ 控制当前语义阶段
- $g_t$ 提供 near-future visual goal
- $m$ 控制行为质量、速度和错误模式
- $c$ 控制动作接口
- $o_{t-T:t}$ 提供最近历史观测

π0.7 的推理不是一次性规划完整任务，而是：

```text
动态 subtask
+ 异步 subgoal generation
+ rolling 50-step action chunk
+ RTC 平滑衔接
+ CFG metadata guidance
```

它和 π\*0.6 的关系不是“替代 RL”，而是“把 RL specialist 的数据蒸馏进 generalist”。π\*0.6 用 RL 在单任务上获得强行为，π0.7 用 supervised learning 和 rich context 吸收这些行为，并结合更广泛的数据实现更强泛化。

因此，π0.7 的技术本质可以概括为：

> **用 rich prompt/context 将异构机器人经验变成可被 steering 的条件行为模型；用 subtask 和 subgoal 解决长任务阶段控制；用 metadata 解决 mixed-quality data；用 KI 稳定 VLM-action 训练；用 RTC 和 CFG 改善实时推理和高质量行为控制。**

## 相关笔记

- [[Robot/PI/ChatGPT-Pi_star0.6论文问题解答|pi*0.6 / RECAP]]：pi0.7 吸收 specialist/RL 数据的上游来源之一。
- [[Robot/PI/ChatGPT-MEM 文章分析|MEM]]：长程任务中的 language/video memory 机制。
- [[Robot/PI/ChatGPT-Pi_0.6论文问题解答|pi0.6]]：Knowledge Insulation、continuous action chunk 与 subtask-conditioned action generation。
- [[Robot/PI/ChatGPT-Pi_0.5综述|pi0.5]]：high-level language intermediate outputs 和长程任务分解。
- [[Robot/ChatGPT-RDT-1B|RDT-1B]]：另一条 large-scale continuous action policy 路线。
- [[Robot/PI/FAST_知识总结|FAST]]：离散 action tokenization 路线。
- [[Robot/ChatGPT-Diffusion Policy 概述|Diffusion Policy]]：diffusion action chunk 与 receding-horizon control 路线。
