---
title: RDT-1B 论文综述
type: paper_note
topic: diffusion_robot_policy
status: mature
importance: high
updated: 2026-06-28
tags:
  - rdt-1b
  - diffusion-policy
  - bimanual-manipulation
  - robot-foundation-model
  - vla
  - action-chunking
  - unified-action-space
  - dit
  - robotics
---

# RDT-1B 技术报告：Diffusion Policy 如何扩展为双臂机器人基础模型

## 1. 论文定位

RDT-1B，全称 **Robotics Diffusion Transformer**，是一篇面向双臂机器人操作的 diffusion-based foundation policy 论文。

它的核心目标不是做显式 world model，也不是预测未来图像或环境状态，而是学习一个语言条件下的视觉运动策略：

$$
p(a_{t:t+T_a} \mid \ell, o_t)
$$

其中：

$$
\ell
$$

是语言指令，

$$
o_t
$$

是当前观测，

$$
a_{t:t+T_a}
$$

是未来一段 action chunk。

因此，RDT 更准确的定位是：

> **一个大规模 diffusion action generator / continuous-action VLA policy，而不是 AWM/WAM。**

它的主要贡献是把原本偏 task-level 的 Diffusion Policy 思路扩展到：

- 1.2B 参数规模；
- 多机器人数据预训练；
- 双臂机器人微调；
- 语言 + 图像 + 本体状态输入；
- 连续动作 chunk 生成；
- 真实机器人部署。

---

## 2. 论文要解决的问题

### 2.1 双臂动作更强多模态

双臂机器人执行同一个任务时，可能有多种合理动作模式。

例如抓取一个物体时，可能：

- 左手先动；
- 右手先动；
- 双手同时接近目标；
- 两只手从不同方向接近目标。

如果用确定性回归：

$$
(\ell, o_t) \rightarrow a_t
$$

模型容易学到多个动作模式的平均值。但多个成功动作的平均值不一定是成功动作，甚至可能是不可执行的动作。

因此，RDT 选择建模条件动作分布：

$$
p(a_{t:t+T_a} \mid \ell, o_t)
$$

而不是只预测一个确定性动作。

---

### 2.2 双臂数据稀缺

双臂机器人硬件昂贵，遥操作采集成本高，所以单一双臂机器人上的数据通常不足以训练 foundation model。

RDT 采用两阶段路线：

$$
\text{multi-robot pretraining}
\rightarrow
\text{target bimanual fine-tuning}
$$

即先用大量多机器人数据学习可迁移的物理先验，再用目标双臂机器人的数据进行微调。

---

### 2.3 多机器人 action space 异构

不同机器人动作空间不同：

- 有的输出 joint position；
- 有的输出 end-effector pose；
- 有的有 gripper；
- 有的有 mobile base；
- 有的是单臂；
- 有的是双臂；
- 控制频率也不同。

如果直接混合训练，会造成 action 语义混乱。

RDT 因此设计了 **Physically Interpretable Unified Action Space**，把不同机器人的动作按物理含义映射到统一空间。

---

## 3. RDT 整体模型 Pipeline

RDT 的完整 pipeline 可以拆成以下几个阶段。

---

### Stage 1：输入数据准备

每个训练样本包括：

$$
(\ell, o_t, a_{t:t+T_a})
$$

其中：

- 语言指令：

$$
\ell
$$

- 图像历史：

$$
X_{t-1:t}
$$

默认使用 2 帧，每帧 3 个相机视角：

- exterior camera；
- right-wrist camera；
- left-wrist camera。

- 当前 proprioception：

$$
z_t
$$

表示机器人当前本体状态。

- action chunk：

$$
a_{t:t+T_a}
$$

论文中：

$$
T_a = 64
$$

也就是一次预测未来 64 个动作。

- control frequency：

$$
c
$$

用于告诉模型当前数据来自什么控制频率的机器人系统。

---

### Stage 2：动作加噪，构造 diffusion 输入

训练时，RDT 不直接输入真实 action，而是先对真实 action chunk 加噪。

前向加噪过程是：

$$
\tilde a_{t:t+T_a}
=
\sqrt{\bar\alpha_k}a_{t:t+T_a}
+
\sqrt{1-\bar\alpha_k}\epsilon
$$

其中：

$$
\epsilon \sim \mathcal N(0, I)
$$

$$
k \sim \text{Uniform}(\{1,\ldots,K\})
$$

这里的：

$$
\tilde a_{t:t+T_a}
$$

就是 noisy action chunk。

模型训练目标是从 noisy action chunk 中恢复 clean action chunk：

$$
f_\theta(\ell, o_t, \tilde a_{t:t+T_a}, k)
\approx
a_{t:t+T_a}
$$

需要注意的是，RDT 采用的是 **x0 prediction / clean-action prediction**，而不是经典 DDPM 中常见的 noise prediction。

也就是说，模型直接输出：

$$
\hat a^0_{t:t+T_a}
$$

而不是：

$$
\hat\epsilon
$$

训练 loss 是：

$$
\mathcal L(\theta)
=
\mathrm{MSE}
\left(
a_{t:t+T_a},
f_\theta(\ell, o_t, \tilde a_{t:t+T_a}, k)
\right)
$$

---

### Stage 3：Unified Action Space 映射

RDT 需要同时处理不同机器人数据，所以它定义了一个 128 维的 physically interpretable unified action space。

这个空间中的不同维度对应明确物理含义，例如：

- 右臂 joint positions；
- 右 gripper joint positions；
- 右臂 joint velocities；
- 右 end-effector position；
- 右 end-effector 6D pose；
- 左臂 joint positions；
- 左 gripper joint positions；
- 左 end-effector pose；
- base linear velocity；
- base angular velocity；
- reserved dimensions。

对于单臂机器人，默认映射到右臂部分。

对于缺失的维度，使用 padding。但 padding 不能简单填 0，因为 0 可能表示真实物理值，例如速度为 0 表示静止。

为了解决这个歧义，RDT 会额外拼接一个 availability mask：

$$
[u, m]
$$

其中：

- $u$：128 维统一物理向量；
- $m$：128 维 0/1 mask，表示每个维度是否真实存在。

因此，进入低维 MLP encoder 的输入实际类似：

$$
[u, m] \in \mathbb R^{256}
$$

这个设计的核心意义是：

> **跨机器人训练时，不只是把动作 padding 到同一维度，而是按物理语义对齐。**

---

### Stage 4：低维输入 tokenization

低维输入包括：

$$
z_t,\ \tilde a_{t:t+T_a},\ c,\ k
$$

其中：

- 当前 proprioception $z_t$；
- noisy action chunk $\tilde a_{t:t+T_a}$；
- control frequency $c$；
- diffusion timestep $k$。

RDT 会先把 $z_t$ 和 $\tilde a_{t:t+T_a}$ 映射到 unified action space，然后用 shared MLP 编码到 token space。

注意，这里不是把 proprioception 和 noisy action 直接拼成一个大向量，而是分别编码成 token：

$$
z_t \rightarrow h_z
$$

$$
\tilde a_t \rightarrow h_{\tilde a_t}
$$

$$
\tilde a_{t+1} \rightarrow h_{\tilde a_{t+1}}
$$

一直到：

$$
\tilde a_{t+T_a-1} \rightarrow h_{\tilde a_{t+T_a-1}}
$$

然后在 sequence length 方向拼接：

$$
H_0 =
[
h_z,\ 
h_{\tilde a_t},\ 
h_{\tilde a_{t+1}},\ 
\ldots,\ 
h_{\tilde a_{t+T_a-1}},\ 
h_c,\ 
h_k
]
$$

如果：

$$
T_a = 64
$$

那么主序列长度是：

$$
1 + 64 + 1 + 1 = 67
$$

这些 token 构成 DiT 主干真正要更新的 denoising stream。

---

### Stage 5：语言和图像编码

语言和图像不是直接拼进主序列，而是作为 condition tokens 通过 cross-attention 注入。

#### 语言编码

语言指令：

$$
\ell
$$

经过 frozen T5-XXL 编码，再通过 MLP adaptor 投影到 RDT token space：

$$
C_{\text{text}}
=
\mathrm{MLP}(\mathrm{T5}(\ell))
$$

#### 图像编码

图像历史：

$$
X_{t-1:t}
$$

经过 frozen SigLIP 编码，再通过 MLP adaptor 投影：

$$
C_{\text{img}}
=
\mathrm{MLP}(\mathrm{SigLIP}(X_{t-1:t}))
$$

图像 token 还会加入多维位置编码，用于区分：

- 时间；
- 相机视角；
- 图像 patch 位置。

T5 和 SigLIP 都被冻结，主要训练 adaptor 和 RDT 主体。

---

### Stage 6：DiT 主干融合信息

RDT 主干是 Diffusion Transformer。它处理的是低维 denoising token sequence：

$$
H_0 =
[
h_z,\ 
h_{\tilde a_t},\ldots,
h_{\tilde a_{t+T_a-1}},
h_c,
h_k
]
$$

在每个 DiT block 中，主序列先通过 self-attention 进行内部融合。

这样 noisy action token 可以读取：

- proprioception token；
- diffusion timestep token；
- control frequency token；
- 其他 action chunk token。

例如，每个 noisy action token 都可以通过 self-attention 获取当前机器人状态：

$$
h_{\tilde a_i}
\leftarrow
\mathrm{SelfAttn}
(
h_{\tilde a_i},
[h_z, h_{\tilde a_t}, \ldots, h_c, h_k]
)
$$

这一步实现了 proprioception 和 noisy action chunk 的融合。

---

### Stage 7：Cross-attention 注入图像和语言条件

RDT 的 image/language condition injection 不是通过 adaptive RMSNorm / adaptive LayerNorm，而是通过 cross-attention。

在某一层中，主序列 hidden state 是：

$$
H^{(l)}
$$

condition tokens 是：

$$
C
$$

其中 $C$ 可以是 image tokens，也可以是 text tokens。

cross-attention 计算形式是：

$$
Q = H^{(l)}W_Q
$$

$$
K = CW_K
$$

$$
V = CW_V
$$

$$
\mathrm{CrossAttn}(H^{(l)}, C)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right)V
$$

也就是说：

- action/proprio/noisy-action tokens 作为 Query；
- image 或 language tokens 作为 Key 和 Value；
- 主序列通过 attention 读取外部条件信息。

这和 adaptive norm 的区别在于：

- adaptive norm 通常把 condition 压成一个向量，然后生成 scale/shift/gate；
- RDT 的图像和语言条件是变长 token sequence，直接压成一个向量会损失信息，所以用 cross-attention。

---

### Stage 8：Alternating Condition Injection

RDT 没有每层同时注入 image 和 language，而是交替注入。

例如可以理解为：

$$
C^{(l)}
=
\begin{cases}
C_{\text{img}}, & l \text{ 为某些层} \\
C_{\text{text}}, & l \text{ 为另一些层}
\end{cases}
$$

这个设计叫 **Alternating Condition Injection**，简称 **ACI**。

动机是：image tokens 通常远多于 language tokens。如果把 image tokens 和 text tokens 每层都拼在一起做 cross-attention，语言信息可能被图像信息淹没，从而影响 instruction following。

交替注入可以让模型在一部分层专门读取图像，在另一部分层专门读取语言，从而缓解 image-token dominance。

---

### Stage 9：输出 clean action chunk

经过多层 DiT 后，得到最终 hidden states：

$$
H_L =
[
h_z^L,\ 
h_{\tilde a_t}^L,\ldots,
h_{\tilde a_{t+T_a-1}}^L,
h_c^L,
h_k^L
]
$$

模型主要取 action token 对应的 hidden states，通过 nonlinear MLP decoder 投影回 action space：

$$
\hat a_{t:t+T_a}
=
\mathrm{MLPDecoder}
(
h_{\tilde a_t}^L,\ldots,h_{\tilde a_{t+T_a-1}}^L
)
$$

这里的 nonlinear MLP decoder 指的是：

$$
a = W_2\sigma(W_1h+b_1)+b_2
$$

而不是简单 linear decoder：

$$
a = Wh+b
$$

真正让它 nonlinear 的是中间激活函数 $\sigma$。如果 MLP 没有激活函数，多层 linear 仍然可以合并成一层 linear。

即使 action 只有 2 维，MLP decoder 和 linear decoder 也不一样，因为 decoder 的输入是高维 latent，而不是 2 维 action 本身。MLP decoder 可以对高维 latent 做非线性组合，再输出动作。

---

### Stage 10：推理时 iterative denoising

推理时，模型从纯噪声 action chunk 开始：

$$
a^K_{t:t+T_a} \sim \mathcal N(0, I)
$$

然后经过多步 denoising：

$$
a^K
\rightarrow
a^{K-1}
\rightarrow
\cdots
\rightarrow
a^0
$$

每一步都调用 RDT denoiser：

$$
\hat a^0
=
f_\theta(\ell, o_t, a^k, k)
$$

再根据 diffusion scheduler 更新到下一步。

论文部署时使用 DPM-Solver++ 把采样步数从 100 步降到 5 步，使得 action chunk 生成可以实时运行。

最终得到：

$$
\hat a_{t:t+T_a}
$$

机器人再执行这个 action chunk 中的动作。

论文没有特别详细展开 chunk 执行策略，但可以理解为：模型以一定频率生成未来动作序列，控制器按时间顺序执行，并在后续观测更新后再次生成新的 action chunk。

---

## 4. RDT 的关键结构设计

### 4.1 MLP with Fourier Features

RDT 低维物理量编码中使用 MLP with Fourier features。

普通 MLP 是：

$$
y = \mathrm{MLP}(x)
$$

Fourier-feature MLP 是：

$$
y = \mathrm{MLP}(\gamma(x))
$$

其中：

$$
\gamma(x)
=
[
x,
\sin(2\pi Bx),
\cos(2\pi Bx)
]
$$

它的作用是帮助 MLP 更容易表达高频变化。

机器人低维物理量中可能存在高频变化，例如接触、碰撞、夹爪闭合、摇杆推动等。因此 RDT 用 Fourier features 增强低维输入表达能力。

不过，这个设计不是当前 VLA/AWM 的主流标准模块。它更像是一个合理的工程 trick，而不是论文最核心贡献。

---

### 4.2 QKNorm + RMSNorm

RDT 使用 QKNorm 稳定 attention 计算，避免大模型训练时：

$$
QK^\top
$$

数值不稳定。

同时使用 RMSNorm 替代 LayerNorm。理由是 LayerNorm 的均值中心化可能引入 token shift / attention shift，对时间序列建模不利；RMSNorm 不做 centering，更适合保留时间序列结构。

这些设计主要服务于训练稳定性。

---

### 4.3 Nonlinear MLP Decoder

RDT 将 final linear decoder 替换为 nonlinear MLP decoder，用于增强从 Transformer latent 到物理动作空间的非线性映射能力。

这个设计对 dexterous manipulation 有帮助，但它不是特别新的方法，更像常见的表达能力增强手段。

---

### 4.4 Alternating Condition Injection

ACI 是 RDT 中相对有意思的结构设计。它通过交替注入 image/text tokens，缓解图像 token 过多导致语言信息被淹没的问题。

但它也不是通用 VLA/AWM 标配，更多是针对 RDT 这种 DiT + cross-attention condition injection 架构的局部优化。

---

## 5. 数据和训练

### 5.1 预训练数据

RDT 使用 46 个机器人数据集进行预训练，总规模约：

- 1M+ trajectories；
- 21TB；
- 多机器人；
- 多任务；
- 多动作空间。

其中包括：

- RT-1；
- DROID；
- RH20T；
- Mobile ALOHA；
- BridgeData V2；
- RoboSet；
- Open X-Embodiment 等数据。

---

### 5.2 微调数据

作者自采了一个 Mobile ALOHA 双臂数据集，包括：

- 300+ tasks；
- 6K+ trajectories；
- 3M+ frames；
- 100+ objects；
- 15+ scenes；
- 三视角 RGB；
- 双臂 joint positions / velocities；
- 人工语言标注；
- 使用 GPT-4-Turbo 对指令进行扩写和简化。

---

### 5.3 训练规模

RDT-1B 规模为：

- 28 层；
- hidden size 2048；
- 32 attention heads；
- 1.2B 参数。

训练代价很高：

- 预训练：48 张 H100 80GB，约 1 个月；
- 微调：同样 48 张 H100，约 3 天。

这说明 RDT 是一个典型的大规模系统工程，而不是轻量方法。

---

## 6. 实验和结论

论文测试了 7 类真实机器人任务：

| 任务 | 测试能力 |
|---|---|
| Wash Cup | unseen object |
| Pour Water | unseen scene |
| Pour Water-L-1/3 | instruction following |
| Pour Water-R-2/3 | instruction following |
| Handover | 5-shot learning |
| Fold Shorts | 1-shot learning |
| Robot Dog | dexterity |

对比方法包括：

- ACT；
- OpenVLA；
- Octo；
- RDT scratch；
- RDT ours。

实验结论是 RDT 在多数任务上显著优于 baseline。尤其在：

- unseen object；
- unseen scene；
- instruction following；
- few-shot skill；
- dexterous control；

这些方面表现更好。

Ablation 也说明：

- 去掉 diffusion，变成 regression，性能下降；
- 模型变小，性能下降；
- 不做预训练，泛化性能下降；
- diffusion + large model + large-scale pretraining 三者都重要。

因此，论文真正想证明的是：

$$
\text{strong bimanual policy}
=
\text{diffusion action modeling}
+
\text{large model}
+
\text{multi-robot pretraining}
+
\text{target bimanual fine-tuning}
$$

---

## 7. 主要贡献

### 7.1 将 Diffusion Policy 扩展为 foundation-scale policy

RDT 最大的贡献是证明 diffusion action model 可以 scale 到 foundation model 级别。

它不是只在单任务小数据上训练 diffusion policy，而是构建了：

$$
\text{Diffusion Policy}
+
\text{DiT}
+
\text{large-scale pretraining}
+
\text{bimanual fine-tuning}
$$

的完整系统。

---

### 7.2 提出物理可解释 Unified Action Space

这是方法上最值得关注的贡献。

它解决了多机器人预训练中的 action/proprioception 异构问题，让不同机器人的物理量可以按语义对齐，而不是变成无意义的统一 vector。

---

### 7.3 系统验证双臂机器人 foundation policy

RDT 聚焦双臂操作，覆盖：

- unseen object；
- unseen scene；
- few-shot；
- dexterity；
- instruction following。

它的价值很大程度来自真实机器人系统验证。

---

### 7.4 证明 diffusion + scale + data 的组合有效

论文不是证明某个小 trick 决定性能，而是证明大模型、大数据、diffusion action modeling 的组合对双臂操作泛化很重要。

---

## 8. 局限性

第一，RDT 不是显式 world model。它不预测未来 observation，也不做 world rollout，因此它不是 AWM/WAM。

第二，它不是真正 zero-shot cross-embodiment policy。它需要在目标双臂机器人上 fine-tune。

第三，训练成本极高。48 张 H100 训练一个月，这对大多数团队都不现实。

第四，baseline 对比需要谨慎。OpenVLA 和 Octo 原本不一定针对双臂高精度连续控制优化，它们在 RDT 任务中失败，不完全说明路线本身弱。

第五，很多结构 trick 没有充分独立 ablation。例如 Fourier features 没有单独消融，无法判断其关键性。

第六，Unified Action Space 是人工设计的物理槽位，适合 gripper-arm 类机器人，但对 dexterous hand、soft robot、legged manipulation 等更复杂 embodiment 是否适用，还需要进一步验证。

---

## 9. 和 VLA/AWM 的关系

RDT 和典型 VLA 的区别在于，很多 VLA 走的是：

$$
\text{VLM}
\rightarrow
\text{discrete action tokens}
$$

而 RDT 走的是：

$$
\text{language/image/state}
\rightarrow
\text{DiT denoiser}
\rightarrow
\text{continuous action chunk}
$$

所以它更适合高精度、连续控制、多模态动作分布强的任务。

但 RDT 和 AWM/WAM 的关系较弱。它没有显式建模：

$$
p(o_{t+1} \mid o_t, a_t)
$$

也没有显式 object-centric world state、planning rollout、memory 或 future prediction。

因此，RDT 对 AWM/WAM 的启发主要在 action-side：

- 如何表示连续动作；
- 如何处理多模态动作分布；
- 如何做 action chunk；
- 如何进行跨机器人 action space 对齐；
- 如何利用多机器人数据预训练 action generator。

---

## 10. 总结

RDT-1B 不是未来 VLA/AWM 的完整答案，但它是 continuous-action diffusion foundation policy 这条路线中的重要系统论文。

它的核心不是 Fourier features、MLP decoder、ACI 这些局部 trick，而是证明了：

> **Diffusion Policy 可以通过 DiT、大规模多机器人预训练、物理可解释统一动作空间和目标双臂微调，扩展成一个真实可部署的双臂机器人基础策略模型。**

最值得记住的四点是：

1. 双臂动作多模态更强，所以 diffusion 比 deterministic regression 更自然。
2. RDT 输入 noisy action chunk，输出 clean action chunk，采用的是 x0 prediction。
3. Unified Action Space 是跨机器人预训练中最有价值的设计之一。
4. RDT 是 action generator，不是 world model；它对 AWM/WAM 的直接贡献有限，但对连续动作建模非常有参考价值。

## 相关笔记

- [[ChatGPT-ALOHA硬件与ACT算法|ALOHA / ACT]]：双臂硬件、连续 action chunk、ACT baseline。
- [[FAST_知识总结|FAST]]：离散 action tokenization 路线，可与 RDT 的 continuous diffusion action 对比。
- [[ChatGPT-Pi_0机器人文章分析|pi0]]：flow matching action expert 路线。
- [[ChatGPT-Pi_0.6论文问题解答|pi0.6]]：continuous action chunk 与 FAST joint likelihood。
- [[Pi0_7_technical_report|pi0.7]]：steerable generalist VLA 与 rich context conditioning。
- [[ChatGPT-MolmoAct2论文框架分析|MolmoAct2]]：离散 action-token pretraining 与 flow-matching continuous expert 结合的部署路线。
- [[ChatGPT-MEM 文章分析|MEM]]：长程记忆与 action policy 的关系。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
- [[ChatGPT-Diffusion Policy 概述|Diffusion Policy 概述]]
- [[ChatGPT-RT-1 论文综述|RT-1 论文综述]]
- [[ChatGPT-RT-2 论文综述|RT-2 论文综述]]
- [[ChatGPT-GR00T N1 综述|GR00T N1 综述]]
