---
title: MolmoAct2 论文框架分析
type: paper_note
topic: action_reasoning_model
status: mature
importance: high
updated: 2026-07-08
tags:
  - molmoact2
  - action-reasoning-model
  - vla
  - embodied-reasoning
  - fast-action-tokenizer
  - flow-matching
  - adaptive-depth
  - robot-deployment
  - robotics
---


# MolmoAct2 技术报告：Action Reasoning Models for Real-World Deployment

## 0. 报告摘要

MolmoAct2 是 Ai2 提出的一个面向真实机器人部署的开源 VLA 系统。它的目标不是只在 benchmark 上提高成功率，而是解决现实部署中的几个核心问题：现有 frontier VLA 封闭、open-weight 模型依赖昂贵或特定硬件、reasoning 型策略推理太慢、fine-tuning 后的真实任务成功率仍不够稳定。论文高亮部分反复强调了这些痛点：**real-world deployment、spatial and embodied reasoning、open data、per-layer KV conditioning、adaptive depth tokens**。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

从技术上看，MolmoAct2 是一个“系统型”工作。它不是单独提出一个 action head，而是把 **VLM backbone、机器人数据、action tokenizer、离散 action 预训练、连续 action expert、adaptive depth reasoning、推理优化** 串成完整 pipeline。整体主线可以概括为：

```text
Molmo2
→ Molmo2-ER：增强 embodied / spatial reasoning 的 VLM backbone
→ MolmoAct2-Pretrain：用 FAST action tokens 做离散动作预训练
→ MolmoAct2-Post：接入 flow-matching continuous action expert
→ MolmoAct2-Finetune：适配具体 embodiment / benchmark
→ MolmoAct2-Think：加入 adaptive depth-token reasoning
```

---

## 1. 论文解决的问题

论文认为今天的 VLA 模型距离“可靠真实部署”还有明显差距。高亮部分主要对应四类问题：

第一，**frontier robot policies 多数封闭**。训练数据、训练 recipe、权重、代码往往不可复现，限制了学术和工程使用。

第二，**open-weight VLA 常常绑定昂贵或特定机器人平台**，不利于普通实验室或开发者使用。

第三，**reasoning-augmented policy 虽然更可解释、更强，但通常推理延迟高**。例如先生成大量 depth、point trajectory、future image 或 world model rollout，再输出动作，会拖慢 closed-loop control。

第四，**zero-shot 和 fine-tuned success rate 仍不够稳**，尤其在真实世界、OOD 相机位姿、未见物体、空间扰动下，模型容易失败。论文把 MolmoAct2 定位为一个“fully open action reasoning model built for practical deployment”。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

---

## 2. MolmoAct2 的五个核心贡献

论文在引言中把 MolmoAct2 相比 MolmoAct 的提升总结为五条：

1. **Molmo2-ER backbone**：一个专门面向 spatial / embodied reasoning 的 VLM backbone，在 3.3M embodied reasoning corpus 上训练。
2. **三类开源机器人数据**：MolmoAct2-BimanualYAM、MolmoAct2-DROID、MolmoAct2-SO100/101。
3. **MolmoAct2-FAST Tokenizer**：开放数据与权重的 action tokenizer，支持多 embodiment、多控制模式。
4. **per-layer KV conditioning 的连续 action expert**：把离散-token VLM 的 reasoning state 接到 flow-matching continuous controller 上。
5. **MolmoAct2-Think**：用 adaptive depth tokens 做几何 reasoning，只重新生成变化区域的 depth tokens，从而降低 reasoning latency。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

这五条里，第 3、4、5 是我们之前讨论最多的部分：FAST tokenizer 不是神经网络训练，而是 DCT-BPE 式的 action token fitting；per-layer KV conditioning 是结构核心；Think 的 adaptive depth 是 reasoning 与 latency 的折中。

---

## 3. 系统总体架构

MolmoAct2 的架构可以分成四个层级：

```text
视觉输入层：
    RGB image / video frame
    → SigLIP2 ViT
    → Molmo2 connector
    → visual tokens

VLM reasoning 层：
    visual tokens + language instruction + setup/control descriptors + state tokens
    → Molmo2-ER / MolmoAct2 backbone

离散 action 层：
    continuous action chunk
    → MolmoAct2-FAST Tokenizer
    → <action_0> ... <action_2047>
    → next-token prediction

连续 action 层：
    noisy action chunk + flow time
    → DiT-style action expert
    → per-layer KV conditioning from VLM
    → continuous action chunk
```

文中 Figure 4 是核心结构图：图像、语言、机器人状态进入 pre-trained VLA backbone；离散 action tokens 继续用于 next-token prediction；连续 action expert 通过每一层 cross-attention 接收 VLM 对应层的 K/V；训练时 target action-token span 被 mask，防止 continuous action expert 看到 ground-truth discrete action。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

---

## 4. Molmo2-ER：面向机器人空间推理的 VLM backbone

MolmoAct2 不是直接拿普通 VLM 做机器人控制，而是先训练了 **Molmo2-ER**。原因是普通 VLM 主要优化语义理解，但机器人控制需要更具体的空间能力：

```text
metric distance
free space
object localization
cross-view correspondence
egocentric / exocentric reasoning
video temporal reasoning
pointing / affordance
scene geometry
```

Molmo2-ER 在约 3.3M embodied reasoning samples 上训练，数据覆盖 image embodied QA、image pointing、detection、video embodied QA、multi-image / ego-exo reasoning、abstract embodied reasoning 等。训练 recipe 是 **specialize-then-rehearse**：

```text
Stage 1: embodied specialization
    用 embodied corpus 强化空间/具身能力

Stage 2: joint refinement / rehearse
    把 embodied corpus 与原 Molmo2 multimodal data 混合
    避免损失通用 VLM 能力
```

实验上，Molmo2-ER 在 13 个 embodied reasoning benchmark 上平均 63.8%，比 Molmo2 提升约 17 点，并超过文中对比的多个开源和闭源 VLM。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

技术意义是：**action policy 的上限很大程度受 backbone 的空间表征能力限制**。MolmoAct2 并不是只靠 action head 提升，而是先把 VLM 变成更适合机器人控制的 embodied reasoning backbone。

---

## 5. 数据体系：真实部署导向的数据混合

MolmoAct2 的数据设计是论文很重要的一部分。它不只依赖 OXE / DROID / 社区数据，而是重新收集、过滤和重标注。

### 5.1 MolmoAct2-BimanualYAM

这是作者新收集的双臂 YAM 数据集，包含约 **34.5k demonstrations、720 小时**，覆盖 28 个以上真实任务，如折衣服、解缆线、清桌、扫描商品、打包药品等。论文强调这是最大的 open bimanual dataset 之一，且硬件成本低于 6000 美元。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 5.2 MolmoAct2-SO100/101

SO-100/101 是 Hugging Face 社区低成本机器人平台。作者从社区 LeRobot 数据中筛选，最终得到 **1,222 个 datasets、377 contributors、38,059 episodes、19.8M frames、约 184 小时**。过滤包括结构合法性、去除 eval-style datasets、license/codebase eligibility、TOPReward quality gate。高亮处特别标注了这些 filtering steps，说明作者非常重视数据质量。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 5.3 MolmoAct2-DROID

DROID 原始数据很大，但有 idle segments、失败尝试、重复任务说明等问题。MolmoAct2-DROID 使用 supplementary annotations、extended language labels 和 idle-frame filter，得到 **74,604 valid episodes、17,758,044 frames**，每个 episode 都成功、至少有一个有效语言 instruction、且没有明显 pause。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 5.4 语言重标注

论文高亮了语言 annotation 的问题：一些数据集中任务说明重复、无意义，甚至出现 `lerobot_test`、`Test run`。作者用 Qwen3.5-27B 根据 sampled frames 和原始 instruction 重新生成任务描述，使整体 unique labels 从 **71,121 / 22%** 提升到 **146,485 / 46%**。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

技术意义是：VLA 的语言条件不仅是“用户 prompt”，也是训练时任务区分和 embodiment/action semantics 的载体。如果 instruction 低质量，模型会更难学会语言到动作的对应关系。

---

## 6. MolmoAct2-FAST Tokenizer：多机器人动作的离散化

这是我们前面讨论最多的部分之一。

### 6.1 它不是神经网络训练，而是 tokenizer fitting

论文说 “train MolmoAct2-FAST Tokenizer”，但这里的 train 更接近：

```text
fit tokenizer on action corpus
```

而不是通过 backprop 训练一个 neural network。它基于 FAST 的 DCT-BPE 思路：

```text
1 秒 continuous action trajectory
→ pad 到 32 维
→ 归一化
→ frequency-domain transform / DCT
→ quantize coefficients
→ flatten
→ BPE
→ 2048 action vocabulary
```

所以它本质上是 **数据统计驱动的 action compression vocabulary**。社区习惯把 BPE vocabulary / merge rules 的拟合也叫 “training tokenizer”，和文本 BPE tokenizer training 类似。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 6.2 多 embodiment 的统一表示

Tokenizer 使用一个统一的 32D action vector。Appendix 里给出格式：

```text
Single-arm:
    [A1, ..., An, G1, 0, ..., 0]

Bimanual:
    [AL1, ..., ALn, GL, AR1, ..., ARn, GR, 0, ..., 0]
```

其中 `A` 是 arm joints，`G` 是 gripper state。不同机器人和控制模式都 pad 到 32 维。FAST tokenizer 不需要统一坐标系，也不强制把所有数据转为 delta end-effector，而是在 heterogeneous “dialects” 上训练，例如 absolute joint 和 delta end-effector velocity。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 6.3 gripper 的特殊处理

论文明确说 gripper commands 不和普通连续维度一起做 1-99 percentile normalization，因为 gripper 往往是 binary 或 narrow-range open/close signal。更准确地说：

```text
continuous joint / ee dims:
    用 1-99 percentile statistics 做 normalization

gripper dims:
    作为 open/close 或 narrow-range signal 单独处理
    不用普通连续关节那套 percentile scaling
```

进入 DCT-BPE 后，gripper 仍然作为 action vector 的一维或两维参与 tokenization。它不是被变成 `<open>` / `<close>` 的语义 token，而是作为时间序列的一维被 DCT、量化、BPE 压缩。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 6.4 tokenizer 不输入机器人配置

MolmoAct2-FAST Tokenizer 本身只看 action chunk，不看 image、text、robot_id 或 URDF。真正区分机器人和控制模式的是后面的 VLA prompt，包括：

```text
<setup_start> bimanual yam robotic arms ...
<control_start> absolute joint pose ...
<control_start> delta end-effector pose ...
```

所以 tokenizer 是 context-free compression，policy 是 context-conditioned action generation。Appendix 明确说，VLM backbone 会学习把不同 action dialect 和 task prompt / visual context 中的 embodiment 对应起来。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

---

## 7. 视觉输入：SigLIP2 ViT 与 single resized crop

MolmoAct2 使用 SigLIP2 ViT 作为视觉编码器。它不是文本意义上的 tokenizer，而是 visual encoder：把图像切成 patch，经 ViT 得到连续 visual embeddings，再通过 connector 投影到 LLM embedding space。

论文的架构表中，image encoder 为约 380M 参数、27 层、hidden dim 1152、image size 384×384、patch size 14；connector 读取 ViT 第三倒数层和第九倒数层特征，并做 2×2 image pooling / 3×3 video pooling。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

文中高亮的 “single resized crop rather than high-resolution tiled crops” 很关键。含义是：

```text
single resized crop:
    每个 camera observation 只取一个 resize 到固定尺寸的图像输入 ViT

high-resolution tiled crops:
    把高清图切成多个 tile
    每个 tile 单独 resize / encode
    拼接大量 visual tokens
```

MolmoAct2 选择 single resized crop，是因为机器人 policy 的 sequence length 和 latency 压力很大。一个 robot example 可能有多相机图像、语言、状态、setup/control、action targets。如果每个相机都切成很多 tile，visual token 数会爆炸，影响训练和闭环控制速度。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

---

## 8. Pre-training：把 Molmo2-ER 变成 action-aware VLA

MolmoAct2-Pretrain 的目标是：**先不训练 continuous action expert，而是用离散 action tokens 让 VLM 学会机器人状态和动作接口**。

输入包括：

```text
visual observations
language instruction
setup/control descriptors
discrete state tokens
<action_output>
```

输出是：

```text
FAST action tokens
```

关键细节：

- continuous action 和 state 先做 normalization；
- action pad 到 32 维，用 2048 action vocab 编码；
- proprioceptive state 单独离散化，每个 state scalar 变成 256 个 state tokens 之一；
- robot target 是 1 秒 action chunk；
- pre-training 最大 sequence length 是 4200；
- 用 on-the-fly packing 把多个短样本打包进一个 4200-token sequence，但用 attention mask 隔离，避免样本之间互相 attend。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

我们之前讨论的 packing 本质是训练效率优化：物理上多个样本拼成一个长序列，共用一次 forward；语义上用 block-diagonal attention mask 保证每个样本只能看到自己的 text、visual tokens、state tokens、action targets。

---

## 9. Post-training：加入 flow-matching continuous action expert

部署时不能只靠 autoregressive action tokens，因为离散 action decoding 太慢，也不适合高频连续控制。因此 post-training 加入 **DiT-style action expert**。

### 9.1 Flow matching 目标

给定 normalized action chunk $a$、Gaussian noise $\epsilon$ 和时间 $t$：

$$
x_t = (1 - t)\epsilon + t a
$$

目标速度为：

$$
u^\* = a - \epsilon
$$

action expert 学习：

$$
f_\theta(x_t, t, c) \rightarrow u^\*
$$

其中 $c$ 是 VLM context，包含任务、视觉、setup/control、state tokens。训练时每个 action chunk 会采样多个 $(\epsilon_i, t_i)$，post-training 用 $K=4$，fine-tuning 用 $K=8$。推理时则从一次 Gaussian noise 初始化开始，做固定步数 Euler integration，逐步得到 continuous action trajectory。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

我们之前讨论过：**训练时 noise 每次重新采样；推理时每个 action chunk 开始采一次 pure noise，Euler loop 内不再每步重新采 noise**。

### 9.2 Per-layer KV conditioning

这是 MolmoAct2 的核心结构创新。传统 VLA action expert 常用 final hidden states 作为 context。MolmoAct2 改为每一层 action expert cross-attend 到对应 VLM layer 的 K/V：

$$
\tilde K_\ell = reshape(P_K K^{vlm}_\ell)
$$

$$
\tilde V_\ell = reshape(P_V V^{vlm}_\ell)
$$

然后 action expert 第 $\ell$ 层做 cross-attention：

$$
CA(Q_\ell, \tilde K_\ell, \tilde V_\ell)
$$

直观上，action expert 不只是读取 VLM 最后一层压缩后的 residual state，而是读取 VLM 每层 self-attention 真正使用的 attention state。实验也显示 per-layer KV conditioning 比 hidden-state conditioning 更好。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

---

## 10. 离散 action 与连续 action 的联合训练

Post-training 和 fine-tuning 中，一个 robot example 同时有：

```text
离散 action target:
    FAST(action chunk)

连续 action target:
    original normalized action chunk
```

训练目标是：

$$
L_{post} = L_{LM} + L_{flow}
$$

其中：

- $L_{LM}$ 继续训练 VLM 预测文本 token / 离散 action token；
- $L_{flow}$ 训练 action expert 生成连续 action chunk。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

但关键是：**continuous action expert 不能看到 ground-truth discrete action-token span**。论文 Figure 4 和正文都强调 target action-token span 被 mask 掉。原因是 discrete action tokens 是同一个 ground-truth action chunk 的 FAST 压缩版本，如果 continuous expert 能看见它，就相当于看见了答案，会造成 label leakage。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

这和“用高层 plan 条件 continuous controller”不同。FAST action tokens 不是高层粗粒度 plan，而是低层 action trajectory 的离散编码。因此 MolmoAct2 选择：离散 action 作为 VLM 的训练监督，但不作为 continuous expert 的条件输入。

---

## 11. Knowledge Insulation：post-training 与 fine-tuning 的区别

论文在 post-training 使用 **Knowledge Insulation (KI)**：action expert condition on VLM K/V，但这些 K/V 进入 expert 前被 detach，因此 $L_{flow}$ 不反传到 VLM backbone，只更新 action expert 和 adapter projections；VLM 仍由 $L_{LM}$ 更新。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

fine-tuning 阶段则不同：论文明确说不使用 KI，允许 flow loss 通过 action-expert conditioning path 更新 VLM。这是因为 fine-tuning 已经面向具体 embodiment / task，允许 backbone 适配目标控制分布。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

因此更准确的总结是：

```text
MolmoAct2 post-training:
    离散 + 连续 action 同训
    expert 不看 target action tokens
    flow loss 不回传 VLM backbone

MolmoAct2 fine-tuning:
    离散 + 连续 action 同训
    expert 仍不看 target action tokens
    flow loss 可以回传 VLM backbone
```

MolmoAct2-Think 也遵循类似逻辑，只是多了 depth prefix。

---

## 12. MolmoAct2-Think：adaptive depth-token reasoning

MolmoAct2-Think 的动机是：机器人动作依赖几何信息，但普通 imitation loss 不显式要求模型先理解 depth、free space、occlusion、surface layout。Think 在 action 前加入一个中间 depth reasoning step。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 12.1 Depth token 的来源

流程是：

```text
RGB frame
→ Depth Anything V2 估计 dense monocular depth
→ trained depth VQ-VAE 量化 depth
→ 10×10 grid
→ 每个 cell 是 {0,...,127} 的 codebook index
→ 映射成 <depth_0> ... <depth_127> token
```

这里 VQ-VAE 可以理解为固定的 depth tokenizer。它提供 code id，但其 codebook embedding 不直接进入 VLA。VLA 词表中单独添加 `<depth_0>` 到 `<depth_127>`，这些 token 有自己的 VLM embedding。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

你之前的理解很准确：这类似给 depth map 做“视觉离散词表”编码，但真正建模 `<depth_i>` 序列概率和使用这些 token 的，是 VLA transformer，而不是 VQ-VAE。

### 12.2 Adaptive depth perception data

每帧有完整 depth code：

$$
d_t \in \{0,\ldots,127\}^{100}
$$

同时维护 depth buffer：

$$
b_t
$$

和 update mask：

$$
m_t \in \{0,1\}^{100}
$$

对于第一帧，全部更新。之后把 RGB resize 到 320×320，分成 10×10 个 32×32 patches，比较当前 patch 和上一帧 patch 的 cosine similarity。若低于 0.996，则该 cell 更新；否则沿用上一帧 buffer：

$$
m_{t,i} = 1[\cos(x_{t,i}, x_{t-1,i}) < 0.996]
$$

$$
b_{t,i} =
\begin{cases}
d_{t,i}, & m_{t,i}=1 \\
b_{t-1,i}, & m_{t,i}=0
\end{cases}
$$

这里的 update mask 不是 Transformer attention mask，而是 depth cache 的更新标志。它省的是 autoregressive depth decoding 的步数，而不是 action expert 的 cross-attention 计算。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 12.3 Adaptive depth inference

推理时：

```text
没有 depth cache:
    生成完整 100 个 depth tokens

有 depth cache:
    changed cells:
        重新 autoregressive decode
    unchanged cells:
        从上一帧 predicted depth buffer replay
```

最后得到完整 100-code depth prefix，再让 action expert 使用 depth-conditioned VLM K/V 生成连续 action。也就是说，adaptive 只改变 depth prefix 的产生方式，action interface 不变。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 12.4 Depth gate

Appendix A.3 中的 learned depth gate 控制 action expert 每层多大程度使用 depth token K/V。公式：

$$
c_\ell =
\frac{\sum_t A_t(1-M_t)V^{vlm}_{\ell,t}}
{\sum_t A_t(1-M_t)}
$$

其中 $M_t=1$ 表示 depth 相关 token，$A_t$ 表示有效 token。这个 $c_\ell$ 是“有效非 depth context”的平均 value 表示。然后：

$$
g_\ell = \sigma(w_\ell^\top c_\ell + b_\ell)
$$

再只作用到 depth-token 的 K/V：

$$
\bar K^{vlm}_{\ell,t} = (1-M_t+M_tg_\ell)K^{vlm}_{\ell,t}
$$

$$
\bar V^{vlm}_{\ell,t} = (1-M_t+M_tg_\ell)V^{vlm}_{\ell,t}
$$

非 depth token 不变；depth token 被乘以 $g_\ell$。bias 初始化为 -4，使训练开始时 depth pathway 近似关闭，再逐渐学会哪些层需要使用 depth。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 12.5 Teacher-forced depth prefix noise

Think fine-tuning 时，训练中 teacher forcing 使用 oracle depth prefix，但推理时使用模型自己预测的 depth prefix。为减小 mismatch，作者把 teacher-forced depth prefix 中 10% depth-code input tokens 替换成 uniformly sampled depth codes，但 target 不变。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

这不是改变标签，而是做 robustness regularization：

```text
训练输入:
    corrupted depth prefix

训练目标:
    clean depth target / correct action
```

目的是让模型和 action expert 不要过度依赖完美 depth prefix，而能容忍预测 depth token 的局部错误。

---

## 13. 推理优化

### 13.1 标准 MolmoAct2

在一个 action chunk 的 flow loop 内，VLM context 不变，变化的只有 noisy action state 和 flow time。论文高亮了这一点。因此可以缓存：

```text
context-dependent cross-attention states:
    projected VLM K/V
    layout / reshape 后的 cross-attn context

fixed position-dependent terms:
    action horizon 位置编码
    attention mask
    fixed-shape buffer
```

再用 CUDA Graph 捕获 fixed-shape flow loop，减少 Python 和 kernel-launch overhead。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 13.2 MolmoAct2-Think

Think 的 adaptive depth decoding 是 data-dependent：每帧 changed / unchanged cells 不同，KV length 也随着 autoregressive decoding 改变。因此 full adaptive loop 不适合整体 CUDA Graph capture。论文采用：

```text
attention:
    eager 执行
    因为 effective KV length 变化

post-attention 到下一层 pre-attention:
    fixed-shape transformer work
    用 CUDA Graph stages capture
```

这能减少 one-token decode 的 launch bubbles，但加速幅度低于标准 MolmoAct2。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

---

## 14. 实验结果总结

### 14.1 Molmo2-ER embodied reasoning

Molmo2-ER 在 13 个 embodied reasoning benchmark 上整体平均 63.8%，比 Molmo2 的 46.8% 有明显提升，并在多个任务上超过 GPT-5、Gemini Robotics ER-1.5 Thinking 和 Qwen3-VL 等对比模型。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 14.2 Out-of-the-box deployment

DROID setup：

| Benchmark | MolmoAct2-DROID | 对比 |
|---|---:|---:|
| MolmoSpaces 平均 | 37.7 | π0.5-DROID 34.5 |
| Simulation held-out 平均 | 20.6 | π0.5-DROID 10.0 |
| Real-world DROID 平均 | 87.1 | MolmoBot 48.4 / π0.5 45.2 |

SO-100 setup：

| Model | 平均成功率 |
|---|---:|
| MolmoAct2-SO100/101 | 56.7 |
| π0-SO100/101 | 45.3 |
| SmolVLA | 2.3 |

但也有弱点：MolmoSpaces 的 Open 类任务中 MolmoAct2-DROID 不如 π0.5-DROID，说明 articulated-object interaction 仍然有改进空间。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 14.3 Fine-tuning

LIBERO：

| Model | Average |
|---|---:|
| MolmoAct2 | 97.2% |
| MolmoAct2-Think | 98.1% |
| GR00T N1.7 | 97.0% |
| π0.5 | 96.9% |
| MolmoAct-7B-D | 86.6% |

RoboEval：MolmoAct2 达到 44.3%，比 π0.5 高 3.8 点。

真实 bimanual YAM：MolmoAct2 在 8 个任务中赢 7 个，平均 50.1% / 附录表中约 50.6%，比 OpenVLA-OFT 等 baseline 高。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 14.4 OOD robustness

在 spatial variation、lighting、language rephrasing、distractors 四类扰动下，MolmoAct2-Think 平均 50.69%，高于 OpenVLA-OFT 的 39.89%、π0.5 的 27.01%。不过 spatial variance 是最低项，仅 26.25%，说明精细空间泛化仍然困难。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 14.5 推理速度

LIBERO 单 H100、horizon 10：

| Model | Original | Caching | Caching + CUDA Graph |
|---|---:|---:|---:|
| MolmoAct2 | 23.02 Hz | 27.39 Hz | 55.79 Hz |
| MolmoAct2-Think | 8.04 Hz | 9.72 Hz | 12.71 Hz |

标准 MolmoAct2 的 continuous path 很快；Think 因为 autoregressive depth decoding 明显更慢。论文还说 discrete action path 比 continuous path 慢，因此默认部署使用 continuous action expert。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

---

## 15. Ablation 结论

几个最关键的消融结果：

| 组件 | 结果 / 结论 |
|---|---|
| Molmo2-ER backbone | LIBERO Long 从 77.6% 提到 83.6% |
| Per-layer KV conditioning | 平均 95.9%，优于 hidden-state conditioning 94.0% |
| Flow samples K | K=8 最好，平均 95.90% |
| Fine-tuning recipe | full fine-tuning + discrete co-training + no KI 最好 |
| Think depth recipe | mixed training + noise injection + depth gate 最好，98.10% |

这说明文章的主要设计不是孤立堆叠，而是都能在 LIBERO 消融中看到一定贡献。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

---

## 16. 关键技术澄清：结合我们前面讨论的内容

### 16.1 FAST Tokenizer 的“训练”不是神经网络训练

它是对 action corpus 进行 DCT-BPE vocabulary fitting。BPE merge rules / vocabulary 由数据统计得到，因此社区仍称作 tokenizer training。

### 16.2 同一个 action token 不一定有跨机器人语义一致性

`<action_134>` 不是“向左移动”或“闭合夹爪”的语义 token，而是 32D normalized action trajectory 的频域/BPE pattern。其物理意义依赖于当前 embodiment、action slot、control mode、normalization stats 和 prompt context。

### 16.3 不让 continuous expert 看 discrete action target 是为了防 label leakage

FAST(action) 是 ground-truth continuous action 的离散压缩。如果 continuous expert 看到它，训练会变成从答案编码还原答案，而不是从图像/语言/状态决策动作。

### 16.4 VQ-VAE depth codebook id 和 VLA depth token embedding 不同

VQ-VAE 提供固定 depth code id；VLA 词表中添加 `<depth_0>` 到 `<depth_127>`，这些 token 在 VLA embedding table 里有自己的 embedding。VLA 学的是预测这些 token，并学习如何通过 K/V 和 action expert 使用它们。

### 16.5 Adaptive depth mask 不是 attention mask

`m_t` 是 10×10 depth grid 的更新 mask，用于决定 changed cells 重新 decode，unchanged cells replay cache。它省的是 depth autoregressive generation 的 sequential steps，而不是 action expert 的 cross-attention 长度。

### 16.6 Depth gate 是“强度门控”，不是选择哪些 depth token 进入 K/V

depth token 仍然在 context 里；gate 是 per-layer scalar，控制该层 depth-token K/V 被乘以多大系数。

### 16.7 Teacher-forced depth noise 是为了解决中间变量的 train-test mismatch

普通 teacher forcing 使用 oracle prefix；但 Think 推理时 action expert 使用模型预测的 depth prefix。给 10% depth input tokens 加噪声，是为了让模型和 action expert 适应 imperfect predicted depth。

---

## 17. 技术优势

MolmoAct2 的优势主要体现在五个方面：

第一，**完整开源和可复现**：模型、数据、训练代码、tokenizer 训练混合都开放，区别于只开权重不开放数据/recipe 的模型。

第二，**空间 reasoning backbone 与 robot action learning 结合紧密**：Molmo2-ER 不只是 VLM benchmark 提升，也在 action-token prediction ablation 中提升 LIBERO Long。

第三，**离散预训练 + 连续后训练的工程路线合理**：离散 action token 让 VLM 继续用 next-token prediction 学机器人动作；连续 action expert 解决部署时高频控制问题。

第四，**per-layer KV conditioning 比 final hidden-state conditioning 更充分地利用 VLM 内部 attention state**。

第五，**Think 版本把 depth reasoning 做成 adaptive cache，而不是每步固定生成全部 depth tokens**，降低 reasoning latency，同时保留可解释几何中间表示。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

---

## 18. 局限与风险

论文 Appendix E 也承认了两个重要限制。

### 18.1 Fixed action chunk 与 open-loop execution

MolmoAct2 预测固定 horizon 的 action chunk：

```text
YAM / SO-100/101: 30 steps at 30 Hz
DROID: 15 steps at 15 Hz
LIBERO: 10 steps at 10 Hz
```

然后 open-loop 执行这个 chunk，再重新 query。问题是：

- chunk boundary 之间没有 continuity loss，可能产生速度/加速度不连续；
- chunk 内无法实时响应扰动、接触事件或 tracking error；
- 55.79 Hz 是 amortized chunk throughput，不等于真正每个控制步都闭环 replan。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 18.2 Zero-shot deployment 仍是 embodiment-specific

MolmoAct2 的 out-of-the-box deployment 主要限于三个有大量训练数据的平台：

```text
Bimanual YAM
SO-100/101
DROID Franka
```

它不是可以零样本迁移到任意机器人结构的 universal controller。新 embodiment 仍需要 fine-tuning demonstrations。`MolmoAct2 Action Reasoning Models for Real-world Deployment-with-annotations.pdf`

### 18.3 Think 的性能-速度折中

MolmoAct2-Think 有更好的 spatial reasoning 和 robustness，但推理速度显著低于标准 MolmoAct2。它适合需要几何解释和更强空间鲁棒性的任务，但不一定适合所有高频闭环控制场景。

---

## 19. 结论

MolmoAct2 的核心价值在于：它把 VLM 的 embodied reasoning、机器人数据规模、action tokenization、flow-matching continuous control 和 depth reasoning 以一个可部署的系统方式连接起来。

如果用一句话概括：

**MolmoAct2 先用 Molmo2-ER 提供空间/具身 reasoning backbone，再用 FAST action tokens 让 VLM 具备 action awareness，随后用 per-layer KV conditioning 把 VLM 的 attention state 接到 flow-matching continuous action expert 上；MolmoAct2-Think 进一步把 depth VQ-VAE code id 变成 VLA depth tokens，并通过 adaptive cache、depth gate 和 noise-robust fine-tuning，让几何 reasoning 更适合真实机器人部署。**

从实验上看，它在 embodied reasoning、DROID / SO-100 out-of-the-box deployment、LIBERO fine-tuning、RoboEval trajectory quality、真实 YAM 任务和 OOD robustness 上都优于主要 baseline；从系统上看，它最大的贡献不是某一个公式，而是把 **数据、tokenization、architecture、training recipe、inference optimization** 组合成一个完整、开放、可复现的 VLA deployment stack。

## 相关笔记

- [[FAST_知识总结|FAST 知识总结]]
- [[Pi0_7_technical_report|π0.7 技术报告]]
- [[Gemini Robotics 1.5 综述|Gemini Robotics 1.5 综述]]
- [[GR00T N1 综述|GR00T N1 综述]]
- [[RDT-1B|RDT-1B]]
- [[Diffusion Policy 概述|Diffusion Policy 概述]]
- [[VQVAE_综述|VQ-VAE 综述]]



---
