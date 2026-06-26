---
title: RDT-1B 论文综述
type: paper_note
topic: diffusion_robot_policy
status: mature
importance: high
updated: 2026-06-25
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

# RDT-1B

**User:** Anonymous  
**Created:** 6/25/2026 20:00:19  
**Updated:** 6/25/2026 21:10:01  
**Exported:** 6/26/2026 13:48:20  
**Link:** [https://chatgpt.com/c/6a3d184f-ca14-83ec-b8e5-e067232a58a4?mweb_fallback=1](https://chatgpt.com/c/6a3d184f-ca14-83ec-b8e5-e067232a58a4?mweb_fallback=1)  

# RDT-1B 论文综述：Diffusion Policy 如何 scale 到双臂机器人基础模型

论文：**RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation**。核心目标是做一个面向双臂操作的 **language-conditioned visuomotor policy**，即输入语言、图像、机器人状态，输出双臂机器人未来一段连续动作。论文声称 RDT-1B 是一个 1.2B 参数规模的 diffusion-based 双臂机器人基础模型，先在 46 个多机器人数据集、1M+ trajectories 上预训练，再在作者自采的 6K+ 双臂数据上微调。`RDT-1b.pdf`

---

# 1. 这篇文章的核心定位

这篇文章不是一个 World Model / Action-World Model，也不是那种显式预测未来 observation、显式做 planning rollout 的模型。它本质上还是一个 **policy model / action generator**：

$$
p(a_{t:t+T_a} \mid \ell, o_t)
$$

也就是给定语言指令 $\ell$、当前观测 $o_t$，生成未来一段 action chunk。

所以它在 VLA/AWM 脉络里的位置可以这样理解：

> RDT 是一条 **continuous-action diffusion VLA / visuomotor policy** 路线，而不是 WAM/AWM 路线。  
> 它的重点在 action-side：如何生成高维、连续、多模态、双臂动作。

它真正想证明的是：**Diffusion Policy 不只是小规模 imitation learning trick，也可以 scale 成一个大模型、多机器人预训练、真实双臂部署的 foundation policy。**

---

# 2. 论文要解决的问题

作者认为双臂 manipulation foundation model 面临两个主要挑战。

## 2.1 双臂动作分布更强多模态

单臂操作中，同一个任务通常也可能有多种执行方式，但双臂会更明显。比如抓一个物体，可以左手先动、右手先动、双手同时夹取、从不同方向接近目标。对于同一个语言和视觉观测，demo 里可能出现多个合理 action mode。

如果直接做确定性回归：

$$
(\ell, o_t) \rightarrow a_t
$$

模型容易学到多个 action mode 的“平均动作”。而多个成功动作的平均值不一定是成功动作，甚至可能是完全不可执行的 out-of-distribution action。论文正是用这个理由说明为什么需要 diffusion 来建模连续条件动作分布。`RDT-1b.pdf`

## 2.2 双臂数据稀缺，需要多机器人预训练

双臂机器人贵，遥操作采集也贵，所以特定双臂机器人的数据很少。论文里明确说，目标双臂机器人的可用数据通常远不到 foundation model 所需规模，因此采用：

$$
\text{multi-robot pre-training}
\rightarrow
\text{target bimanual fine-tuning}
$$

也就是先用大量其他机器人数据学 transferable physical priors，再用目标双臂机器人数据适配部署。论文也强调，它的目标不是做一个可以直接跨 embodiment 部署到所有机器人的模型，而是利用多机器人数据提升目标双臂机器人的泛化能力。`RDT-1b.pdf`

## 2.3 多机器人 action space 异构

不同机器人动作空间差异很大：

- 有的输出 joint position；
- 有的输出 EEF pose；
- 有的有 gripper；
- 有的有 mobile base；
- 有的是单臂；
- 有的是双臂；
- 控制频率也不一样。

如果只保留所有机器人共有的 action 维度，会丢很多信息；如果直接拼起来训练，又会造成语义混乱。这就是 RDT 提出 **Physically Interpretable Unified Action Space** 的动机。`RDT-1b.pdf`

---

# 3. RDT 的整体框架

RDT 的输入可以分成三类。

第一类是 **denoising inputs**：

$$
z_t,\ \tilde a_{t:t+T_a},\ c,\ k
$$

其中：

- $z_t$：当前 proprioception；
- $\tilde a_{t:t+T_a}$：当前 diffusion step 下的 noisy action chunk；
- $c$：control frequency；
- $k$：diffusion timestep。

第二类是 **image condition**：

$$
X_{t-1:t}
$$

论文默认使用 2 帧图像历史，每帧有 3 个相机视角：exterior camera、right-wrist camera、left-wrist camera。

第三类是 **language condition**：

$$
\ell
$$

模型输出是 denoised action chunk：

$$
\hat a_{t:t+T_a}
$$

论文图 3 里画得比较清楚：低维输入进入 DiT 主干，语言和图像作为 condition 通过 cross-attention 注入，最后输出 denoised action chunk。`RDT-1b.pdf`

---

# 4. Diffusion 建模：它输出的是 clean action，不是 noise

这是我们前面讨论过的重点。

经典 DDPM 里最常见的是预测 noise：

$$
\epsilon_\theta(x_k, k)
$$

但 RDT 这里采用的是 **x0 prediction / clean-action prediction**。它不是让模型预测噪声，而是让模型从 noisy action chunk 直接预测 clean action chunk：

$$
\hat a^0_{t:t+T_a}
=
f_\theta(\ell, o_t, \tilde a_{t:t+T_a}, k)
$$

训练时，先对真实 action chunk 加噪：

$$
\tilde a_{t:t+T_a}
=
\sqrt{\bar\alpha_k}a_{t:t+T_a}
+
\sqrt{1-\bar\alpha_k}\epsilon
$$

然后用 MSE 让模型预测原始 clean action：

$$
\mathcal L(\theta)
=
\mathrm{MSE}
\left(
a_{t:t+T_a},
f_\theta(\ell, o_t, \tilde a_{t:t+T_a}, k)
\right)
$$

这和传统 noise prediction 并不矛盾。因为给定 $x_k$、$\bar\alpha_k$，预测 noise 和预测 $x_0$ 是可以互相换算的不同 parameterization：

$$
\hat x_0 =
\frac{x_k - \sqrt{1-\bar\alpha_k}\hat\epsilon}{\sqrt{\bar\alpha_k}}
$$

$$
\hat\epsilon =
\frac{x_k - \sqrt{\bar\alpha_k}\hat x_0}{\sqrt{1-\bar\alpha_k}}
$$

RDT 选择直接预测 clean action，直觉上更符合机器人策略：最终真正要执行的是 action，而不是 noise。论文的反向采样公式里也直接使用了 clean action estimate 来得到上一步 noisy action。`RDT-1b.pdf`

---

# 5. 为什么要 action chunk？

RDT 不是每次只预测一个动作，而是一次预测一段未来动作：

$$
a_{t:t+T_a}
=
(a_t, a_{t+1}, \ldots, a_{t+T_a-1})
$$

论文中：

$$
T_a = 64
$$

action chunk 的作用主要有两个。

第一，减少决策次数。如果每一步都重新决策，策略误差会不断累积，机器人可能逐渐走出训练分布。一次预测多个动作，可以降低这种 compounding error。

第二，提高动作时间连续性。diffusion 一次生成整段 action，模型可以同时考虑前后动作之间的协调关系，而不是单步贪心输出。

这和 ACT、Diffusion Policy 里的 action chunking 思想是一致的。RDT 的区别是把 action chunk diffusion scale 到了 1.2B 参数、多机器人预训练和双臂部署。

---

# 6. 低维输入如何融合：proprioception 和 noisy action 不是简单拼成一个大向量

这是你前面问得很关键的一点。

RDT 里 $z_t$ 和 $\tilde a_{t:t+T_a}$ 都属于 low-dimensional inputs。它们会先进入 unified action space，再用 shared MLP 编码到 token space。之后不是在数值维度上直接拼接，而是在 **sequence length 方向**拼成 token sequence：

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

如果 $T_a=64$，那么主序列长度大致是：

$$
1 + 64 + 1 + 1 = 67
$$

这些 token 进入 DiT 主干后，通过 self-attention 融合。也就是说，当前 proprioception 对未来 noisy action 的影响不是手工规则，而是通过 Transformer attention 学出来的。论文附录明确说，proprioception 和 noisy action chunk 会先嵌入 unified action space，再用 shared MLP 编码；随后和 control frequency、diffusion timestep 在长度方向 concat。`RDT-1b.pdf`

这个设计可以理解成：

> $z_t$ 告诉模型“机器人现在在哪里”；  
> $\tilde a_{t:t+T_a}$ 是当前 diffusion step 下“待去噪的未来动作草稿”；  
> DiT 通过 self-attention 让未来动作 token 读取当前状态 token，再结合视觉和语言条件去 denoise。

---

# 7. 统一动作空间：这篇论文最值得关注的设计之一

RDT 提出的 **Physically Interpretable Unified Action Space** 是 128 维。它不是无意义的 128 维 latent，而是每段维度都有物理含义，例如：

- 右臂 joint positions；
- 右 gripper joint positions；
- 右臂 joint velocities；
- 右 EEF position；
- 右 EEF 6D pose；
- 左臂对应的一套量；
- base linear velocity；
- base angular velocity；
- reserved dimensions。

单臂机器人默认映射到“右臂”部分；缺失维度 padding。论文表 4 给出了这 128 维的详细物理含义。`RDT-1b.pdf`

这个设计背后的核心思想是：

> 跨机器人预训练时，不要把 action 当成没有语义的 vector，而应该把相同物理含义的量对齐到相同槽位。

例如，一个机器人输出 7-DoF joint positions，另一个机器人输出 6-DoF joint positions，只要它们都是“右臂 joint position”，就填到同一段物理槽位的前几维。

还有一个非常重要的工程细节：padding 不能简单填 0。因为 0 本身也有物理意义，比如速度为 0 表示静止。为了避免模型分不清“真实 0”和“padding 0”，RDT 给 action/proprioception 额外拼接一个 0/1 availability vector，表示每一维是否真实存在，所以编码前会从 128 维变成类似 256 维。`RDT-1b.pdf`

我认为这是 RDT 最值得记住的贡献之一。它不一定是跨 embodiment 的最终答案，但它非常清楚地指出了一个关键问题：**action representation 的物理语义对齐是跨机器人预训练的核心难点。**

---

# 8. 多模态编码方式

RDT 的编码器大致如下。

## 8.1 低维物理量编码

低维输入包括：

$$
z_t,\ \tilde a_{t:t+T_a},\ c,\ k
$$

其中 $z_t$ 和 $\tilde a$ 通过 unified action space 对齐，再通过 shared MLP 编成 token。control frequency 和 diffusion timestep 分别用 MLP 编码。

论文提到这些 MLP 使用 Fourier features。这个意思是：不是直接把原始标量/向量输入 MLP，而是先做类似正弦余弦的频率映射：

$$
\gamma(x)
=
[
x,\ 
\sin(2\pi Bx),\ 
\cos(2\pi Bx)
]
$$

然后：

$$
h = \mathrm{MLP}(\gamma(x))
$$

这样做的动机是机器人低维物理量可能有高频变化，比如接触、碰撞、夹爪闭合、摇杆推动等。Fourier features 可以帮助 MLP 更容易表示高频函数。

但从 VLA/AWM 主流方法角度看，**MLP 常见，Fourier-feature MLP 不是主流标配**。这更像一个合理的工程 encoding trick，而不是这篇论文最核心的方法贡献。

## 8.2 图像编码

图像使用 frozen SigLIP 编码，然后接 MLP adaptor 投影到 RDT token space。RDT 使用固定的三视角格式：

$$
\text{exterior},\ \text{right-wrist},\ \text{left-wrist}
$$

单臂数据或者缺失相机的数据会用 background color padding。图像还使用多维 positional embedding，以区分时间、相机视角和 patch 位置。`RDT-1b.pdf`

## 8.3 语言编码

语言使用 frozen T5-XXL 编码，再通过 MLP adaptor 投影到 token space。T5 和 SigLIP 都被冻结，主要训练 adaptor 和 RDT 主干。这能降低训练显存压力，同时利用已有视觉语言表征能力。`RDT-1b.pdf`

---

# 9. Cross-attention condition injection：不是 adaptive RMSNorm

RDT 的 image/language 条件不是通过 adaptive LayerNorm / adaptive RMSNorm 注入的，而是通过 DiT block 里的 cross-attention 注入。

在某一层里，主序列 hidden state 是：

$$
H^{(l)}
$$

condition tokens 是：

$$
C_{\text{img}}
\quad \text{或} \quad
C_{\text{text}}
$$

cross-attention 大致是：

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

所以更准确地说：

- action/proprio/noisy-action tokens 作为 Query；
- image 或 language tokens 作为 Key 和 Value；
- 通过 attention 把 condition 信息读入主序列。

这和 adaptive norm 的区别很大。adaptive norm 通常把 condition 压成一个向量，然后生成 scale/shift/gate 去调制 hidden state。而 RDT 认为图像和语言都是高维、变长 token sequence，压缩成一个 token 会丢信息，因此采用 cross-attention。论文也明确说，image/language 条件高维且长度可变，原始 DiT 的 adaptive layer norm approach 不适合，因此用 cross-attention。`RDT-1b.pdf`

---

# 10. ACI：Alternating Condition Injection

RDT 没有每一层都同时注入 image 和 language，而是交替注入：

$$
C^{(l)}
=
\begin{cases}
C_{\text{img}}, & l \text{ 为某些层} \\
C_{\text{text}}, & l \text{ 为另一些层}
\end{cases}
$$

动机是：image tokens 通常远多于 language tokens。如果每层把 image tokens 和 language tokens 拼在一起 cross-attend，语言信息可能被图像 token 淹没，导致 instruction following 变差。

所以 RDT 让连续层交替读图像和语言。这个设计叫 **Alternating Condition Injection, ACI**。论文的 ablation 显示，不使用 ACI 时，倒水到指定水位这类 instruction-following 任务会明显变差。`RDT-1b.pdf`

我的评价是：ACI 是一个有意思的局部设计，但不一定是未来 VLA/AWM 的标准范式。它更多是针对 RDT 这种 “DiT denoising tokens + image/text cross-attention” 结构的 token imbalance 问题。

---

# 11. DiT 主干的几个结构改造

RDT 基于 Diffusion Transformer，但作者认为机器人动作数据和图像数据不同：低维物理量有非线性动力学、高频变化、数值范围不稳定。因此做了几处改造。`RDT-1b.pdf`

## 11.1 QKNorm + RMSNorm

QKNorm 用于稳定 attention 的 $QK^\top$ 计算，避免大模型训练时数值不稳定。

RMSNorm 替代 LayerNorm。作者的理由是，LayerNorm 的 centering 操作可能引入 token shift / attention shift，不适合时间序列预测；RMSNorm 不做均值中心化，更适合保留时间序列结构。

论文图 4 显示，不使用 QKNorm 和 RMSNorm 时，训练 loss 会不稳定甚至爆炸。`RDT-1b.pdf`

## 11.2 Nonlinear MLP decoder

标准 DiT 常用 final linear decoder：

$$
a = Wh + b
$$

RDT 改成 nonlinear MLP decoder：

$$
a = W_2\sigma(W_1h+b_1)+b_2
$$

这里“nonlinear”的关键就是中间有激活函数。如果没有激活函数，多层 linear 可以合并成一层 linear，本质没有区别。

需要注意：即使 action 只有 2 维，MLP decoder 和 linear decoder 也不一样。因为 decoder 的输入不是 2 维 action，而是高维 latent：

$$
h \in \mathbb{R}^{2048}
$$

输出才是：

$$
a \in \mathbb{R}^{2}
$$

linear decoder 只能做高维 latent 到 action 的线性读出，而 MLP decoder 可以对 latent features 做非线性组合。论文认为这对非线性机器人动作和 dexterous task 有帮助。`RDT-1b.pdf`

不过这也不是特别新的方法。MLP decoder 是常见增强表达能力的工程手段。RDT 里的核心贡献不应该被理解成“提出 MLP decoder”。

---

# 12. 数据和训练规模

RDT 的预训练数据包括 46 个机器人数据集，总规模 1M+ trajectories、21TB。主要包括 RT-1、DROID、RH20T、Mobile ALOHA、BridgeData V2、RoboSet、Open X-Embodiment 相关数据等。论文还对不同数据集设置 sampling weights，大体思路是避免大数据集过度主导，同时保证小数据集也有足够采样。`RDT-1b.pdf`

微调数据是作者自采的 Mobile ALOHA 双臂数据：

- 300+ tasks；
- 6K+ trajectories；
- 3M+ frames；
- 100+ objects；
- 15+ scenes；
- 三视角 RGB、双臂 joint 信息、人工语言标注；
- 使用 GPT-4-Turbo 扩写和简化指令，提升语言多样性。`RDT-1b.pdf`

训练规模很重：RDT-1B 是 1.2B 参数，论文称预训练使用 48 张 H100 80GB 训练一个月，共 1M steps；微调同样使用 48 张 H100，训练 130K steps，大约 3 天。推理时使用 DPM-Solver++，把 action chunk 采样从 100 steps 降到 5 steps，在 onboard RTX 4090 上达到 action chunk 6Hz、平均 action 381Hz。`RDT-1b.pdf`

这也意味着它是一个非常重的系统工程，训练门槛很高。

---

# 13. 实验设计和结论

论文设计了 7 类真实机器人任务：

| 任务 | 测试维度 |
|---|---|
| Wash Cup | unseen object |
| Pour Water | unseen scene |
| Pour Water-L-1/3 | instruction following |
| Pour Water-R-2/3 | instruction following |
| Handover | 5-shot learning |
| Fold Shorts | 1-shot learning |
| Robot Dog | dexterity |

baselines 包括 ACT、OpenVLA、Octo、RDT scratch。ACT 是双臂领域强 baseline，使用 VAE 建模动作分布；OpenVLA 是 discretized action token 路线；Octo 是 diffusion-based generalist policy，但规模较小。`RDT-1b.pdf`

实验结果总体显示：

- RDT 在 unseen cups / unseen rooms 上明显优于 baseline；
- 在 Pour Water-L-1/3 / R-2/3 中，RDT 能 follow 左/右手和指定水位；
- Handover 只用 5 demos，Fold Shorts 只用 1 demo，RDT 仍有一定成功率；
- Robot Dog 任务要求精细推摇杆角度，RDT 比其他方法更稳。`RDT-1b.pdf`

论文的 ablation 更能说明主线：

| 变体 | 含义 |
|---|---|
| RDT ours | large model + pretraining + diffusion |
| RDT regress | 不用 diffusion，做 deterministic regression |
| RDT small | 小模型，166M |
| RDT scratch | 不做预训练 |

结果显示，没有 diffusion、没有大模型、没有预训练都会造成性能下降。尤其 RDT scratch 在 unseen object / scene 上明显变差，说明大规模多机器人预训练对泛化很重要。`RDT-1b.pdf`

所以论文真正想证明的是：

$$
\text{bimanual generalization}
\approx
\text{diffusion action modeling}
+
\text{large model}
+
\text{large multi-robot pretraining}
+
\text{target robot fine-tuning}
$$

而不是某个单点 trick 决定了一切。

---

# 14. 这篇文章的主要贡献

结合我们上面的讨论，我认为 RDT 的贡献分成四层。

## 14.1 第一贡献：把 Diffusion Policy scale 到 foundation model 级别

这是最核心的贡献。

Diffusion Policy 原本更多是 task-level policy。RDT 证明了 diffusion action model 可以变成：

- 1.2B 参数；
- Transformer/DiT 主干；
- 多模态输入；
- 多机器人预训练；
- 目标双臂机器人微调；
- 真实机器人部署。

这在 VLA 发展脉络中很重要，因为它提供了一条不同于 “VLM + discrete action token” 的路线：

$$
\text{language/image/state}
\rightarrow
\text{diffusion denoiser}
\rightarrow
\text{continuous action chunk}
$$

对于双臂、高精度、接触丰富、动作多模态的场景，continuous diffusion action modeling 比离散 action token 更自然。

## 14.2 第二贡献：Physically Interpretable Unified Action Space

这是方法上最值得关注的地方。

跨机器人训练最大难点之一就是 action/proprioception 表示不统一。RDT 的统一动作空间虽然是人工设计的，但它明确保留物理语义：

$$
\text{raw action}
\rightarrow
\text{physical slots}
$$

这比简单归一化成无意义 vector 更合理。对于后续 AWM/WAM 或跨 embodiment 模型来说，action representation 仍然是核心问题。RDT 在这点上提供了一个非常清晰的工程方案。

## 14.3 第三贡献：双臂 manipulation foundation policy 的系统验证

RDT 的实验聚焦双臂任务，而不是简单单臂 pick-and-place。它强调 bimanual coordination、多模态 action、few-shot skill、dexterity、instruction following。对于双臂机器人学习来说，这篇论文的系统意义大于单点结构创新。

## 14.4 第四贡献：证明 diffusion + scale + data 都重要

Ablation 表明：

- deterministic regression 不够；
- 小模型不够；
- 不预训练泛化差；
- diffusion 更适合多模态动作分布。

这比 Fourier features、MLP decoder、ACI 等 trick 更重要。

---

# 15. 哪些部分不要过度解读

我们前面也讨论过，这篇文章里很多设计不是当前 VLA/AWM 的“主流核心模块”。

## 15.1 Fourier-feature MLP 不是主流 VLA 标配

它合理，但不是核心贡献。论文也没有单独 ablate 它，所以不能证明它对最终效果有决定性作用。

## 15.2 Nonlinear MLP decoder 是常见工程增强

它有用，但本质是把最后 linear projection 换成带激活的 MLP。这个思路很常见，不是新范式。

## 15.3 QKNorm / RMSNorm 是训练稳定性配置

这对 RDT 训练很重要，但也不是机器人 foundation model 独有创新。

## 15.4 ACI 有趣，但结构依赖强

ACI 针对的是 RDT 这种 image/text cross-attention 注入方式里的 token imbalance 问题。它不一定会成为所有 VLA/AWM 的标准设计。

---

# 16. 和 OpenVLA、Octo、π 系列的关系

## 16.1 和 OpenVLA

OpenVLA 是典型的：

$$
\text{VLM}
\rightarrow
\text{discrete action tokens}
$$

它的优势是继承 LLM/VLM 的语义能力，路线更接近大语言模型范式。RDT 则是：

$$
\text{DiT denoiser}
\rightarrow
\text{continuous action chunk}
$$

它更适合连续高精度动作生成，尤其是双臂动作多模态场景。

## 16.2 和 Octo

Octo 也是 diffusion-based generalist robot policy，但模型规模更小，论文里提到 Octo 最大版本约 93M 参数。RDT 的差异在于：规模更大、双臂 fine-tuning 数据更多、统一动作空间设计更强、目标任务更聚焦双臂真实部署。`RDT-1b.pdf`

## 16.3 和 π₀ / π₀.₅

π 系列更接近新的 VLA/action expert 路线，尤其 π₀ 使用 flow matching 做连续动作生成，并和 VLM backbone 结合得更紧。RDT 则更像一个 DiT-based diffusion policy foundation model。

可以粗略理解为：

$$
\text{RDT}
=
\text{Diffusion Policy}
+
\text{DiT scale-up}
+
\text{Unified Action Space}
+
\text{Bimanual finetuning}
$$

$$
\pi_0
=
\text{VLM}
+
\text{Action Expert}
+
\text{Flow Matching}
$$

如果你关注下一代 VLA/AWM 主线，π 系列可能更接近当前趋势；但如果你关注连续 action chunk、多模态动作分布、双臂操作，RDT 仍然非常值得读。

---

# 17. 和 AWM / WAM 的关系

RDT 对 AWM/WAM 的直接贡献有限，因为它没有显式 world model。

它没有建模：

$$
p(o_{t+1} \mid o_t, a_t)
$$

也没有显式预测未来图像、未来 object state、contact state、success/failure、scene dynamics。

它学的是：

$$
p(a_{t:t+T_a} \mid \ell, o_t)
$$

所以 RDT 更像一个强 low-level action generator，而不是 world-action model。

但它对 AWM/WAM 有 action-side 启发：

1. action representation 需要物理语义对齐；
2. continuous action 可能比离散 action token 更适合双臂精细控制；
3. action chunk 可以作为低层控制接口；
4. diffusion/flow 类模型适合表达多模态动作分布；
5. 多机器人预训练需要认真处理 embodiment/action heterogeneity。

---

# 18. 主要局限

第一，RDT 不是真正的 zero-shot cross-embodiment policy。它最终还是需要在目标双臂机器人上 fine-tune。论文也明确说，目标是用多机器人数据提升目标双臂机器人的泛化能力，而不是训练一个直接适配所有机器人的跨 embodiment 模型。`RDT-1b.pdf`

第二，训练成本非常高。48 张 H100 80GB 训练一个月，这使得它更像大团队系统工程，而不是普通实验室容易复现的方法。

第三，baseline 对比需要谨慎。OpenVLA、Octo 原本不一定针对双臂高精度连续控制优化，它们在 RDT 的任务设定里失败，不一定完全说明模型路线本身弱，也可能是 action representation、fine-tuning recipe、deployment setup 不匹配。

第四，实验 trial 数不大。很多 real-robot 任务是 8 次或 25 次测试，能证明方向和趋势，但具体成功率数字不要过度解读。`RDT-1b.pdf`

第五，很多结构 trick 没有充分独立 ablation。比如 Fourier features 没有单独实验，不能判断其关键性。

第六，unified action space 是人工设计的物理槽位，适合 gripper-arm 类机器人，但对 dexterous hand、soft robot、legged manipulation、tool-use heavy setting 是否足够，还不清楚。

---

# 19. 最终评价

我会把这篇论文总结为：

> **RDT-1B 不是未来 VLA/AWM 的完整答案，但它是 continuous-action diffusion foundation policy 这条路线里非常重要的一篇系统论文。**

它的价值不在于提出了很多新的小模块，而在于系统性证明：

$$
\text{Diffusion Policy}
\quad
\text{可以被 scale 到}
\quad
\text{large model + large data + bimanual real robot deployment}
$$

这篇文章你需要重点记住四个东西：

1. **为什么双臂动作更适合 diffusion**：动作多模态强，回归会平均多个模式。
2. **RDT 如何做 action diffusion**：输入 noisy action chunk，直接预测 clean action chunk。
3. **Unified Action Space 为什么重要**：跨机器人训练必须对齐 action/proprioception 的物理语义。
4. **真正贡献是系统 scale-up，而不是 Fourier features / MLP decoder / ACI 这些 trick**。

一句话版：

**RDT 是把 Diffusion Policy 扩展成双臂机器人基础模型的一次大规模系统验证；它对 AWM/WAM 的直接贡献有限，但对连续动作建模、双臂多模态动作生成、跨机器人 action representation 很有参考价值。**

## 相关笔记

- [[Robot/ChatGPT-ALOHA硬件与ACT算法|ALOHA / ACT]]：双臂硬件、连续 action chunk、ACT baseline。
- [[Robot/PI/FAST_知识总结|FAST]]：离散 action tokenization 路线，可与 RDT 的 continuous diffusion action 对比。
- [[Robot/PI/ChatGPT-Pi_0机器人文章分析|pi0]]：flow matching action expert 路线。
- [[Robot/PI/ChatGPT-Pi_0.6论文问题解答|pi0.6]]：continuous action chunk 与 FAST joint likelihood。
- [[Robot/PI/Pi0_7_technical_report|pi0.7]]：steerable generalist VLA 与 rich context conditioning。
- [[Robot/PI/ChatGPT-MEM 文章分析|MEM]]：长程记忆与 action policy 的关系。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
