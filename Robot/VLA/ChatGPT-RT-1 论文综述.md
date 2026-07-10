---
title: RT-1 论文综述
type: paper_note
topic: robotics_foundation_model
status: mature
importance: high
updated: 2026-06-28
tags:
  - rt-1
  - robotics-transformer
  - vla
  - behavior-cloning
  - action-tokenization
  - robot-foundation-model
  - robotics
---
# RT-1 论文综述

**User:** Anonymous  
**Created:** 6/26/2026 13:57:01  
**Updated:** 6/28/2026 17:06:42  
**Exported:** 6/28/2026 17:06:55  
**Link:** [https://chatgpt.com/c/6a3e148e-b430-83ec-8d09-780389cb20f5](https://chatgpt.com/c/6a3e148e-b430-83ec-8d09-780389cb20f5)  

# RT-1: Robotics Transformer for Real-World Control at Scale 技术报告

## 1. 论文概览

RT-1，全称 **Robotics Transformer 1**，是 Google Robotics / Everyday Robots 提出的一种面向真实机器人控制的大规模 language-conditioned visuomotor policy。

这篇论文的核心目标不是提出一个非常复杂的新训练算法，而是验证一个机器人学习领域非常重要的问题：

> 能否像 NLP / CV 一样，通过大规模、多任务、多样化真实机器人数据，训练一个统一的机器人 policy，使其具备较强的任务泛化、物体泛化、环境鲁棒性和数据吸收能力？

RT-1 的方法可以概括为：

> 输入自然语言指令和最近若干帧图像，经过 language-conditioned EfficientNet、TokenLearner 和 Transformer，输出离散化的机器人动作，并以闭环方式控制真实机器人执行任务。

它训练在：

- 13 台真实机器人；
- 17 个月采集周期；
- 约 130k 条 robot demonstrations；
- 744 个自然语言指令；
- 多种厨房桌面、抽屉、容器、物体操作任务。

最终 RT-1 在 seen tasks、unseen instruction combinations、distractor robustness、background robustness 和 long-horizon SayCan 任务上都明显优于 Gato、BC-Z 等 baseline。

---

## 2. RT-1 的核心定位

RT-1 可以被看作早期 VLA 系统的重要代表，但它和今天的 OpenVLA、Octo、π₀ 系列仍然有明显不同。

RT-1 的定位更接近：

> 大规模真实机器人数据驱动的 language-conditioned behavior cloning policy。

它不是一个大 VLM，也不是一个具备复杂语言推理能力的 LLM-based agent。语言部分使用的是 Universal Sentence Encoder，视觉部分使用 EfficientNet-B3，动作输出是离散 action bin。Transformer 在这里主要承担时序 token 建模和多任务行为建模，而不是语言生成。

它的历史意义主要在于证明：

1. 真实机器人数据规模化有用；
2. 任务多样性比单纯数据数量更关键；
3. Transformer-style policy 可以吸收多任务机器人数据；
4. 离散动作 tokenization 在大规模 BC 中优于简单 continuous Gaussian action；
5. 模型可以一定程度吸收仿真数据和其他机器人形态的数据。

---

## 4. 输入：语言指令和图像历史

RT-1 的输入包括两部分。

第一部分是自然语言指令，例如：

```text
pick apple from top drawer and place on counter
```

第二部分是最近 6 帧 RGB 图像，每帧分辨率为：

```text
300 × 300
```

所以输入可以抽象为：

$$
i = \text{language instruction}
$$

$$
x_{t-5:t} = \{x_{t-5}, x_{t-4}, ..., x_t\}
$$

其中每个 $x_\tau$ 是一帧 RGB 图像。

RT-1 在每个控制 timestep 都根据当前历史观测和语言指令输出一个动作，然后机器人执行动作，再进入下一个闭环 timestep。

---

## 5. Universal Sentence Encoder：变长语言到固定向量

RT-1 没有使用 LLM 或 VLM backbone 来处理语言，而是使用 **Universal Sentence Encoder**，简称 USE。

USE 的作用是把任意长度的自然语言指令编码成一个固定长度的 sentence embedding。

单条指令时可以理解为：

$$
\text{instruction string}
\rightarrow
e_{\text{text}} \in \mathbb{R}^{1 \times 512}
$$

batch 训练时：

$$
e_{\text{text}} \in \mathbb{R}^{B \times 512}
$$

这里的 512 是语言 embedding 的维度。

USE 内部会先 tokenize 输入文本，再通过 sentence encoder 对变长 token 序列做上下文建模或 pooling，最后得到固定维度的句向量。RT-1 论文没有展开 USE 的内部实现，所以关键不在 USE 本身，而在它提供了一个稳定的、固定维度的语言条件向量。

这个语言向量后面会被送入 FiLM 层，用于调制 EfficientNet 的视觉特征。

---

## 6. FiLM-conditioned EfficientNet：语言如何注入视觉编码器

RT-1 使用 ImageNet pretrained EfficientNet-B3 作为视觉编码器。

和简单地把语言 embedding 与图像 token concat 不同，RT-1 采用 **FiLM** 方式在 EfficientNet 内部进行 early language fusion。

FiLM 的核心形式是：

$$
\tilde h_l = (1 + \gamma_l) \odot h_l + \beta_l
$$

其中：

- $h_l$ 是 EfficientNet 第 $l$ 层的中间视觉特征；
- $\gamma_l$ 和 $\beta_l$ 是由语言 embedding 生成的调制参数；
- $\odot$ 表示 channel-wise multiplication；
- $1 + \gamma_l$ 的写法方便 identity initialization。

如果 EfficientNet 某一层输出是：

$$
h_l \in \mathbb{R}^{B \times H_l \times W_l \times C_l}
$$

语言 embedding 是：

$$
e_{\text{text}} \in \mathbb{R}^{B \times 512}
$$

那么对应 FiLM 层会用一个 Linear / MLP 生成：

$$
\gamma_l, \beta_l \in \mathbb{R}^{B \times C_l}
$$

然后 broadcast 到空间维：

$$
\gamma_l, \beta_l \rightarrow \mathbb{R}^{B \times 1 \times 1 \times C_l}
$$

最后调制视觉特征：

$$
\tilde h_l[b, u, v, c]
=
(1 + \gamma_l[b,c])h_l[b,u,v,c] + \beta_l[b,c]
$$

---

## 7. FiLM 的直觉

FiLM 可以理解为给视觉编码器加了一个“任务滤镜”。

同一张图像，在不同指令下应该关注不同区域：

- 指令是 `pick coke can`，视觉编码器应该更关注 coke can 和夹爪；
- 指令是 `open drawer`，视觉编码器应该更关注 drawer handle；
- 指令是 `place object into bowl`，视觉编码器应该更关注 object、bowl 和它们之间的位置关系。

FiLM 不是直接产生空间 attention map，而是对 CNN 的 channel 特征做语言条件化调制。也就是说，它更像是：

> 根据语言指令增强或抑制某些视觉语义 channel。

后面的 TokenLearner 再从这些已经 language-conditioned 的视觉特征中选择重要空间信息。

---

## 8. FiLM 的 identity initialization

RT-1 使用的是 pretrained EfficientNet。如果直接在中间层插入 FiLM，很可能会破坏预训练网络原本的 feature distribution。

所以论文采用 identity-initialized FiLM。

具体来说，产生 $\gamma$ 和 $\beta$ 的 dense layer 初始化为 0，于是训练初期：

$$
\gamma = 0
$$

$$
\beta = 0
$$

此时：

$$
\tilde h = (1 + 0)h + 0 = h
$$

也就是说，FiLM 一开始不改变 EfficientNet 的行为。模型训练过程中再逐渐学习如何根据语言调制视觉特征。

这个设计很重要，因为 RT-1 的泛化能力很大程度来自 ImageNet pretrained EfficientNet 的视觉先验。

---

## 9. FiLM 和 Adaptive RMSNorm 的区别

FiLM 和 Adaptive RMSNorm 都属于 condition-based modulation。它们共同点是：

$$
\text{condition} \rightarrow \gamma, \beta
$$

然后用 $\gamma,\beta$ 调制主网络 hidden states。

但二者区别在于调制位置和是否先归一化。

### 9.1 FiLM

FiLM 通常直接作用在 feature 上：

$$
\tilde h = (1 + \gamma(c)) \odot h + \beta(c)
$$

它不一定包含 normalization。

RT-1 中的 FiLM 用于 EfficientNet 的 CNN feature map，是一种视觉编码阶段的 early language fusion。

---

### 9.2 Adaptive RMSNorm

Adaptive RMSNorm 通常用于 Transformer block 内部，尤其常见于 diffusion / flow / modern VLA action expert。

形式大概是：

$$
\tilde x = (1 + \gamma(c)) \odot \operatorname{RMSNorm}(x) + \beta(c)
$$

它先对 hidden state 做 RMSNorm，然后再用 condition 生成的 scale / shift 进行调制。

需要注意的是：

> RMSNorm 本身并不负责“控制 attention / MLP”。真正起调制作用的是 condition 生成的 $\gamma,\beta$。由于调制后的 hidden state 会进入 QKV 计算或 MLP，所以它会间接改变 attention pattern 和 MLP activation。

更严谨地说：

```text
AdaRMSNorm 是 condition injection point，
不是 attention / MLP 本身。
```

如果模型还引入 residual gate，例如：

$$
x' = x + g_{\text{attn}}(c) \cdot \operatorname{Attention}(\tilde x)
$$

那么 gate 才是在更直接地控制 attention / MLP 分支输出强度。

---

## 10. EfficientNet 输出：9 × 9 × 512 feature map

经过 FiLM-conditioned EfficientNet-B3 后，每一帧图像输出一个空间特征图：

$$
F \in \mathbb{R}^{9 \times 9 \times 512}
$$

可以理解为图像被压缩成一个 $9 \times 9$ 的低分辨率空间网格，每个空间格子有一个 512 维视觉-语言特征。

将空间维 flatten 后得到：

$$
X \in \mathbb{R}^{81 \times 512}
$$

其中每个 token 是：

$$
X_i \in \mathbb{R}^{512}, \quad i = 1,2,\dots,81
$$

---

## 11. TokenLearner：从 81 个 token 压缩到 8 个 token

RT-1 的 TokenLearner 是模型实时性的关键。

它不是把 $9 \times 9 \times 512$ 硬切成 8 块，也不是从 81 个位置里 top-k 选择 8 个位置，而是学习 8 个 soft attention map。

输入是：

$$
X \in \mathbb{R}^{81 \times 512}
$$

TokenLearner 预测：

$$
A \in \mathbb{R}^{81 \times 8}
$$

其中第 $k$ 个 attention map 是：

$$
A_{:,k} \in \mathbb{R}^{81}
$$

表示第 $k$ 个输出 token 对 81 个空间位置的关注权重。

第 $k$ 个输出 token 通过加权求和得到：

$$
z_k = \sum_{i=1}^{81} A_{i,k}X_i
$$

其中：

$$
z_k \in \mathbb{R}^{512}
$$

对 $k=1,\dots,8$ 都做一次，得到：

$$
Z \in \mathbb{R}^{8 \times 512}
$$

所以 TokenLearner 的本质是：

> 学习 8 个 task-relevant soft pooling，把 81 个空间 token 压缩成 8 个 compact tokens。

---

## 12. TokenLearner 的伪代码

单帧图像的 TokenLearner 可以理解为：

```python
# x: [B, 9, 9, 512]

# 1. 预测 8 张 attention map
attn_logits = small_conv_net(x)          # [B, 9, 9, 8]

# 2. flatten 空间维度
x_flat = x.reshape(B, 81, 512)           # [B, 81, 512]
attn = attn_logits.reshape(B, 81, 8)     # [B, 81, 8]

# 3. 对空间位置做 softmax
attn = softmax(attn, dim=1)              # [B, 81, 8]

# 4. 8 个 attention map 分别对 81 个位置做加权汇聚
tokens = einsum("bpc,bpk->bkc", x_flat, attn)

# tokens: [B, 8, 512]
```

其中最关键的是：

```python
tokens = einsum("bpc,bpk->bkc", x_flat, attn)
```

对应公式：

$$
z_k = \sum_i A_{i,k}X_i
$$

---

## 13. 为什么 TokenLearner 很重要？

不用 TokenLearner 时，每帧图像有 81 个 token，6 帧就是：

$$
6 \times 81 = 486
$$

个 token。

使用 TokenLearner 后，每帧只保留 8 个 compact token，6 帧就是：

$$
6 \times 8 = 48
$$

个 token。

这将 token 数量减少了约 10 倍，对 Transformer 推理速度非常重要。RT-1 需要真实机器人闭环控制，因此不能使用过大的 token 序列。

---

## 14. Transformer 输入：6 帧 × 8 token = 48 token

每帧图像经过 TokenLearner 后得到：

$$
Z_\tau \in \mathbb{R}^{8 \times 512}
$$

6 帧历史图像得到：

$$
Z_{t-5:t} \in \mathbb{R}^{6 \times 8 \times 512}
$$

flatten 时间和 token 维后：

$$
Z \in \mathbb{R}^{48 \times 512}
$$

然后加入 positional encoding，送入 decoder-only Transformer。

RT-1 的 Transformer 有 8 层 self-attention，总模型参数约 35M，其中 EfficientNet + FiLM 约 16M，Transformer 部分约 19M。

---

## 15. Transformer 输出如何变成 11 维动作？

Transformer 输入 48 个 token，输出通常也是 48 个 hidden states：

$$
H \in \mathbb{R}^{B \times 48 \times d}
$$

但最终动作不是 48 个，而是 11 个 action dimensions。

因此 Transformer 后面一定有一个 action prediction head。论文没有给出特别细的 48-to-11 代码级实现，但合理流程是：

1. 从 48 个 hidden states 里取一个 readout representation；
2. readout 可以是最后一个 token、pooling 后的全局表示，或者某个 action query；
3. 将 readout 输入 linear head；
4. 输出 11 个动作维度的分类 logits。

可以抽象为：

$$
h_{\text{readout}} = \operatorname{Readout}(H)
$$

其中：

$$
h_{\text{readout}} \in \mathbb{R}^{B \times d}
$$

再通过 action head：

$$
\text{logits} = W h_{\text{readout}} + b
$$

得到：

$$
\text{logits} \in \mathbb{R}^{B \times 11 \times 256}
$$

每个 action dimension 对应一个 256 类分类。

所以不要把 RT-1 理解成：

```text
48 tokens in → 48 tokens out → 每个 token 对应一个动作
```

更合理的理解是：

```text
48 observation tokens
      ↓
Transformer 建模历史和视觉语言上下文
      ↓
readout representation
      ↓
action head
      ↓
11 action dimensions × 256 bins
```

---

## 16. Action Space：11 维动作

RT-1 的动作空间包括：

### 16.1 Arm action：7 维

```text
x, y, z, roll, pitch, yaw, gripper
```

### 16.2 Base action：3 维

```text
x, y, yaw
```

### 16.3 Mode action：1 维

```text
arm / base / terminate
```

所以总共是：

$$
7 + 3 + 1 = 11
$$

个 action dimensions。

RT-1 是闭环控制模型，每个 timestep 输出当前动作。机器人执行该动作后，再采集新的观测，继续输出下一个动作，直到模型输出 terminate 或达到最大时间步限制。

---

## 17. Action Tokenization：每个动作维度离散成 256 个 bin

RT-1 不直接回归连续动作，而是把每个 action dimension 离散成 256 个 bin。

对于第 $j$ 个动作维度：

$$
a^j \rightarrow \operatorname{bin}(a^j) \in \{0,1,\dots,255\}
$$

模型输出：

$$
p(a^j \mid o, i)
$$

它是一个 256 类 categorical distribution。

最终 logits 是：

$$
\text{logits} \in \mathbb{R}^{B \times 11 \times 256}
$$

训练时对每个 action dimension 做 cross entropy：

$$
\mathcal{L}
=
-\sum_{j=1}^{11}
\log p_\theta(\operatorname{bin}(a^j) \mid o, i)
$$

---

## 18. Loss 和 Causal Masking 的理解

论文写到：

```text
We use a standard categorical cross-entropy objective and causal masking.
```

这里容易误解。

如果 RT-1 只是用 6 帧图像历史预测当前一个 action vector，那么这个 action vector 本身并不需要 causal masking。

一个动作向量：

$$
a_t = (a_t^1, a_t^2, ..., a_t^{11})
$$

如果 11 个维度并行输出，就不需要 action-level causal mask。

更合理的理解是：

> causal masking 是 Transformer sequence modeling 的实现选择，主要用于 decoder-only temporal token modeling，而不是因为一个 action vector 必须自回归生成。

如果训练时在一个 temporal window 中预测多个时间位置的动作，那么 causal mask 可以防止早期位置看到未来图像。例如预测 $a_{t-3}$ 时不应该看到 $x_{t-2}, x_{t-1}, x_t$。

对应 mask 是：

$$
M_{\tau,\tau'} =
\begin{cases}
0, & \tau' \le \tau \\
-\infty, & \tau' > \tau
\end{cases}
$$

然后 attention 是：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}} + M
\right)V
$$

如果只预测最后一个 action，那么最后位置本来就只包含过去和当前观测，causal mask 不是数学上必需的。

论文后面还做了 autoregressive action 的 ablation，发现 auto-regressive action 没有明显提升性能，但推理速度显著变慢。因此 RT-1 最终不依赖 action-level autoregressive generation。

---

## 19. 连续 Gaussian BC 是什么？

连续 Gaussian BC 是传统 continuous action policy 的常见形式。

假设动作是：

$$
a \in \mathbb{R}^{D}
$$

最简单的 deterministic BC 输出均值：

$$
\mu_\theta(s) \in \mathbb{R}^{D}
$$

训练用 MSE：

$$
\mathcal{L}
=
\|a - \mu_\theta(s)\|_2^2
$$

这等价于假设动作分布是固定方差的 Gaussian：

$$
p_\theta(a \mid s)
=
\mathcal{N}(a; \mu_\theta(s), \sigma^2 I)
$$

更完整的 continuous Gaussian policy 会同时输出均值和方差：

$$
\mu_\theta(s) \in \mathbb{R}^{D}
$$

$$
\log \sigma_\theta(s) \in \mathbb{R}^{D}
$$

然后：

$$
p_\theta(a \mid s)
=
\mathcal{N}
\left(
a;
\mu_\theta(s),
\operatorname{diag}(\sigma_\theta^2(s))
\right)
$$

训练 loss 是 negative log-likelihood：

$$
\mathcal{L}
=
-\log p_\theta(a \mid s)
$$

展开后大致是：

$$
\mathcal{L}
=
\sum_{d=1}^{D}
\left[
\frac{(a_d - \mu_d)^2}{2\sigma_d^2}
+
\log \sigma_d
\right]
+
\text{const}
$$

---

## 20. 为什么 RT-1 里离散动作优于 continuous Gaussian？

论文的 ablation 里，RT-1 将离散动作换成 continuous action 后性能显著下降。原因主要是 continuous Gaussian BC 的单峰假设太弱。

在真实机器人 demonstration 数据中，同一个状态可能存在多个合理动作模式：

```text
模式 A：从左边接近目标物体
模式 B：从右边接近目标物体
```

真实动作分布可能是多峰的：

$$
p(a \mid s)
=
0.5p_{\text{left}}(a)
+
0.5p_{\text{right}}(a)
$$

但单个 Gaussian：

$$
p_\theta(a \mid s)
=
\mathcal{N}(a; \mu, \Sigma)
$$

通常只能表达一个主峰。它可能把 $\mu$ 学到两个模式中间，导致输出“平均动作”。

这个平均动作可能不是任何一种有效策略。例如两个可行抓取路径中间的位置可能会撞到物体，或者夹爪对不准目标。

这就是 behavior cloning 中经典的 action averaging 问题。

RT-1 的离散动作输出是 categorical distribution：

$$
p(a^j \mid s)
=
\operatorname{Categorical}(256)
$$

它可以在一个维度上表达多峰分布，例如：

```text
bin 40: 0.45
bin 180: 0.40
others: small
```

因此它比单个 Gaussian 更适合复杂 demonstration 数据。

---

## 21. 那么多个 Gaussian 可以替代离散 action 吗？

理论上可以。

如果一个 Gaussian 不够，可以使用 Gaussian Mixture Model，也就是 Mixture Density Network。

### 21.1 每个维度独立使用多个 Gaussian

对于每个动作维度：

$$
p(a_d \mid s)
=
\sum_{k=1}^{K}
\pi_{d,k}(s)
\mathcal{N}
\left(
a_d;
\mu_{d,k}(s),
\sigma_{d,k}^2(s)
\right)
$$

这种方式能表达每个动作维度上的多峰分布。

但它的问题是不能很好表达动作维度之间的相关性。

例如真实动作只有两个模式：

$$
(-1,-1)
$$

和：

$$
(1,1)
$$

如果每个维度独立 mixture，可能采样出：

$$
(-1,1)
$$

或者：

$$
(1,-1)
$$

而这些组合在真实数据中并不存在。

---

### 21.2 Joint Gaussian Mixture 更合理

更好的方式是对整个 action vector 建模：

$$
p(a \mid s)
=
\sum_{k=1}^{K}
\pi_k(s)
\mathcal{N}
\left(
a;
\mu_k(s),
\Sigma_k(s)
\right)
$$

其中每个 component 都对应一个完整动作模式：

$$
\mu_k(s) \in \mathbb{R}^{D}
$$

这种方式可以理解为：

```text
component 1: 从左边抓的一整套动作
component 2: 从右边抓的一整套动作
component 3: 先调整 base 再伸手的一整套动作
```

它比 per-dimension GMM 更能保持动作维度之间的一致性。

不过 joint GMM 训练比离散 CE 更复杂，容易遇到：

- component collapse；
- variance collapse；
- log-sum-exp 数值问题；
- mode selection 不稳定；
- sampling 导致控制抖动；
- bounded action 处理复杂。

所以在 RT-1 的场景下，离散 action 是一个简单、稳定、实时友好的选择。

---

## 22. 离散 Action 的优点和缺点

### 22.1 优点

第一，训练稳定。

分类交叉熵比 Gaussian mixture NLL 更容易训练。

第二，天然支持 bounded action。

机器人动作通常有固定范围，离散 bin 可以直接覆盖动作范围。

第三，可以表达非高斯、多峰分布。

一个 categorical distribution 不需要预设 Gaussian component 数量。

第四，推理简单。

只需要输出：

$$
11 \times 256
$$

个 logits，然后对每个维度 argmax 或采样。

---

### 22.2 缺点

第一，有 quantization error。

连续动作被映射到有限 bin，精度受 bin 宽影响。

第二，per-dimension discretization 不能完整表达 joint action correlation。

第三，动作维度或 action horizon 很大时，离散输出会变重。

例如一次输出 50 个 action steps，每步 10 维，就会有 500 个离散动作变量。

第四，它不是 continuous action 的最终答案。

后来的 Diffusion Policy、flow matching action expert 等方法，保留了 continuous action，同时通过生成模型表达多峰动作分布。

---

## 23. RT-1 的数据集

RT-1 的主数据集包括：

- 约 130k 条 demonstrations；
- 13 台真实机器人；
- 17 个月采集；
- 744 个 language instructions；
- 多种厨房环境；
- 多种物体和桌面配置。

任务包括：

| Skill | 示例 |
|---|---|
| Pick Object | pick iced tea can |
| Move Object Near Object | move pepsi can near rxbar blueberry |
| Place Object Upright | place water bottle upright |
| Knock Object Over | knock redbull can over |
| Open Drawer | open the top drawer |
| Close Drawer | close the middle drawer |
| Place Object into Receptacle | place brown chip bag into white bowl |
| Pick Object from Receptacle and Place on Counter | pick green jalapeno chip bag from paper bowl and place on counter |
| Additional realistic tasks | pull napkin out of dispenser |

这些 instruction 大多由一个 verb 和一个或多个 noun 构成。论文把 instruction 按动词技能分组，例如 pick、move、open、close 等。

---

## 24. 数据采集方式

RT-1 的 demonstrations 来自人类 teleoperation。

机器人会自动到达数据采集 station，并告诉示教者需要示教哪个 instruction，以及如何随机化场景。示教者使用 VR controllers 控制机器人。

这种数据采集方式保证了：

1. 任务种类多；
2. 物体组合多；
3. 初始状态有随机化；
4. 数据分布更适合真实机器人操作；
5. 每条 episode 都带有自然语言 instruction。

这也是 RT-1 成功的核心之一：不是只有模型变大，而是数据收集系统也足够规模化。

---

## 25. 主要实验结果

RT-1 和 Gato、BC-Z、BC-Z XL 在同一 RT-1 数据集上训练，比较不同模型架构。

结果大致如下：

| Model | Seen Tasks | Unseen Tasks | Distractors | Backgrounds |
|---|---:|---:|---:|---:|
| Gato | 65 | 52 | 43 | 35 |
| BC-Z | 72 | 19 | 47 | 41 |
| BC-Z XL | 56 | 43 | 23 | 35 |
| RT-1 | 97 | 76 | 83 | 59 |

RT-1 在所有指标上都最好。

这里的 unseen tasks 需要注意：它不是完全没见过的新技能，而是训练中见过的 object 和 skill 的新组合。例如训练中见过 apple，也见过 pick，但没有见过某个具体的 `pick apple from ...` 指令组合。

所以 RT-1 的泛化主要是组合泛化，而不是完全新动作泛化。

---

## 26. 数据多样性比数据数量更重要

论文做了数据 ablation，结论是：

> 数据多样性对泛化的影响比单纯数据数量更大。

减少数据量会降低性能，但减少任务多样性带来的泛化下降更严重。

尤其是，当保留 97% 数据量但移除 25% task diversity 时，泛化性能下降接近于把数据量减少 49%。

这说明在机器人 foundation model 中，不能只追求更多同质数据。更重要的是收集：

- 更多任务；
- 更多物体；
- 更多组合；
- 更多场景；
- 更多背景；
- 更多干扰物；
- 更多初始状态分布。

---

## 27. 仿真数据吸收实验

RT-1 还测试了是否能吸收 simulation data。

实验设置是：在真实 demonstrations 基础上，加入一些仿真数据，其中包含真实世界没有见过的 sim objects。

结果是：

| Setting | Real Only | Real + Sim |
|---|---:|---:|
| Real Objects, Seen Skill | 92 | 90 |
| Sim Objects, Seen Skill | 23 | 87 |
| Sim Objects, Unseen Skill | 7 | 33 |

这个结果说明：

1. 加入仿真数据后，真实任务性能几乎不掉；
2. 对只在仿真中见过的物体，真实环境成功率显著提升；
3. 对仿真物体和真实技能的新组合，也有一定泛化提升。

这体现了 RT-1 的 data absorption property。

---

## 28. Everyday Robots + Kuka IIWA 数据混合实验

这是论文里非常值得注意的实验。

它不是测试同一机器人上的多数据集训练，而是测试：

> RT-1 能否吸收来自不同机器人形态的数据？

### 28.1 两类数据

第一类是 RT-1 主数据集，也就是 Everyday Robots mobile manipulator 数据。

第二类是 Kuka IIWA bin-picking 数据，来自 QT-Opt，包括 209k 个成功 bin-picking episodes。

这两类数据差异很大：

| 维度 | Everyday Robots | Kuka IIWA |
|---|---|---|
| 机器人形态 | 移动机械臂 | 工业机械臂 |
| 场景 | 厨房、桌面、抽屉 | bin-picking |
| 数据来源 | 人类 teleoperation | RL agent |
| action space | arm/base/mode | Kuka 4-DOF grasp action |
| 语言标签 | 具体 instruction | 原始数据没有物体名 |

---

### 28.2 Kuka 数据如何对齐到 RT-1？

论文做了几步处理：

1. 将 Kuka 原始 4-DOF action 转换到 RT-1 action space；
2. roll 和 pitch 设为 0；
3. 保留 Kuka 原始 yaw；
4. binary gripper-close command 转换成 RT-1 的 continuous gripper-closedness command；
5. 因为 Kuka 数据没有具体物体名，所有 Kuka 数据统一 relabel 为：

```text
pick anything
```

6. 按照如下比例混合训练：

```text
RT-1 data : Kuka data = 2 : 1
```

---

### 28.3 评估方式

所有评估都在 Everyday Robots 上做，而不是在 Kuka 上做。

评估包括两类：

#### Classroom eval

原始 RT-1 任务，用于检查加入 Kuka 数据后原本任务是否掉点。

#### Bin-picking eval

新构造的 bin-picking 场景，尽量接近 Kuka 数据分布，但仍然使用 Everyday Robots 执行。

为了更接近 Kuka 数据，论文还在 EDR 机器人上加入一些外观修改，例如加线缆、将 gripper 变灰。

---

### 28.4 实验结果

| Training Data | Classroom eval | Bin-picking eval |
|---|---:|---:|
| EDR only | 92 | 22 |
| Kuka only | 0 | 0 |
| EDR + Kuka | 90 | 39 |

结果说明：

1. 加 Kuka 数据后，原始 classroom eval 从 92 降到 90，只掉 2 点；
2. bin-picking eval 从 22 提升到 39，接近翻倍；
3. Kuka-only 在 EDR 上是 0，说明不能纯靠 Kuka 数据零样本迁移到 EDR。

因此，这个实验不能被解释为“任意机器人数据都能直接迁移”。更准确的解释是：

```text
EDR 数据提供 embodiment grounding：
    模型知道如何控制 Everyday Robots。

Kuka 数据提供 bin-picking task / scene prior：
    模型吸收了更多 bin-picking 相关经验。

混合训练使模型把二者部分结合起来：
    在 EDR 上提升 bin-picking 类任务表现。
```

这体现了 RT-1 的异构数据吸收能力，但还没有真正解决跨机器人 embodiment 泛化问题。

---

## 29. Long-horizon：RT-1 依赖 SayCan

RT-1 本身不是长任务 planner。

它每次输出的是一个低层 action，用于执行一个 instruction-level skill。

对于长任务，例如：

```text
How would you bring me two different sodas?
```

需要先被分解成多个低层步骤：

```text
find coke
pick coke
bring coke
put down coke
find pepsi
pick pepsi
bring pepsi
put down pepsi
done
```

论文中 long-horizon 任务使用 SayCan 框架。SayCan 做高层任务分解和 affordance grounding，RT-1 作为低层 manipulation policy 执行具体 instruction。

结果显示：

| Method | Kitchen1 Execution | Kitchen2 Execution |
|---|---:|---:|
| SayCan + Gato | 33 | 0 |
| SayCan + BC-Z | 53 | 13 |
| SayCan + RT-1 | 67 | 67 |

这说明 RT-1 的低层技能成功率和泛化能力更强，因此接入 SayCan 后长任务成功率也更高。

但要注意：

> RT-1 自己没有解决长时序 reasoning 和 planning。长任务能力来自 SayCan + RT-1 的组合。

---

## 30. 模型消融实验

RT-1 做了多个关键 ablation。

### 30.1 去掉大模型

减小模型规模会降低 seen task 和 unseen task 性能，但不是最致命的因素。

### 30.2 去掉 ImageNet pretraining

去掉预训练后，泛化明显下降，尤其是 unseen task 和 robustness。

这说明视觉预训练对 RT-1 很重要。

### 30.3 使用 continuous actions

将离散动作改成 continuous action 后，性能显著下降。

这支持了论文的判断：在大规模复杂 demonstration 数据中，离散动作更适合表达多峰动作分布。

### 30.4 使用 autoregressive actions

自回归 action 没有明显提升性能，但推理速度显著变慢。

因此 RT-1 最终不使用 action-level autoregressive generation。

### 30.5 去掉 history

去掉 6 帧历史，只用单帧图像，会明显降低 distractor robustness。

说明历史帧对判断夹爪状态、物体交互过程、运动趋势有帮助。

### 30.6 去掉 Transformer

用非 Transformer 结构替代会带来整体下降，说明 Transformer 对多任务、时序、语言条件建模有帮助。

---

## 31. RT-1 为什么比 Gato 更适合真实机器人？

Gato 是更通用的 generalist agent，但在 RT-1 的真实机器人 setting 中有几个问题：

1. Gato 更偏 autoregressive token sequence modeling；
2. 图像 tokenization 没有 RT-1 这种 early language fusion；
3. 缺少 TokenLearner 这种为实时控制设计的 token 压缩；
4. 推理速度较慢；
5. 对 distractor 和背景鲁棒性不如 RT-1。

RT-1 的几个设计更贴近真实机器人闭环控制：

- FiLM EfficientNet：早期语言-视觉融合；
- TokenLearner：减少 token 数，提升推理速度；
- 非 autoregressive action output：避免慢速逐 token 生成；
- per-dimension discrete action：表达多峰动作分布；
- 6-frame history：增强时序感知；
- 3Hz closed-loop control：适配真实机器人执行。

---

## 32. RT-1 的主要贡献

RT-1 的贡献可以总结为五点。

### 32.1 大规模真实机器人数据训练

它展示了真实机器人 policy 也可以从大规模多任务数据中获益。

### 32.2 Efficient VLA Policy 架构

RT-1 通过 FiLM EfficientNet、TokenLearner 和 Transformer 组合，在保持模型容量的同时满足真实机器人实时控制需求。

### 32.3 数据多样性的重要性

论文明确展示了任务多样性对泛化至关重要，甚至比单纯增加数据量更关键。

### 32.4 异构数据吸收能力

RT-1 可以吸收仿真数据和其他机器人数据，并在不明显损害原任务性能的情况下提升新场景表现。

### 32.5 离散动作对复杂 BC 的价值

论文通过 ablation 说明，离散 action tokenization 在多任务 demonstration 数据上明显优于简单 continuous Gaussian action。

---

## 33. RT-1 的局限性

### 33.1 本质仍然是 Behavior Cloning

RT-1 主要通过 imitation learning 训练，继承了 BC 的限制：

- 难以超过 demonstrator；
- 可能出现 covariate shift；
- 失败恢复能力有限；
- 没有显式探索和自我改进机制。

### 33.2 语言理解能力有限

RT-1 的语言编码来自 USE，不是 LLM / VLM。它能处理 instruction embedding，但不具备复杂语言推理能力。

### 33.3 泛化主要是组合泛化

RT-1 可以泛化到见过 skill 和 object 的新组合，但不能很好泛化到完全没见过的新 motion。

### 33.4 长任务依赖外部 planner

RT-1 本身不是 planner。长任务依赖 SayCan 做任务分解。

### 33.5 动作表示仍然有局限

离散 action 解决了部分多峰问题，但也带来 quantization error，而且 per-dimension discretization 不能完整表达 joint action correlation。

### 33.6 机器人和场景范围有限

RT-1 主要在厨房桌面、抽屉、物体操作任务上验证，距离开放世界通用机器人还有很大差距。

---

## 34. 放在 VLA 发展脉络中的位置

RT-1 可以看作第一代 scalable VLA policy 的代表。

它解决的问题是：

```text
大规模真实机器人 demonstration + 语言条件 + Transformer policy
是否可以带来泛化和鲁棒性？
```

它没有充分解决的问题是：

```text
大规模 VLM backbone
跨 embodiment 统一 action space
长时序 memory
高层 reasoning
复杂 continuous action generation
大规模 action chunk
实时低延迟部署
```

可以粗略放在如下脉络中：

| 阶段 | 代表方法 | 特点 |
|---|---|---|
| 早期 language-conditioned BC | BC-Z, SayCan low-level policy | 语言条件策略，但模型较浅 |
| 第一代 scalable VLA policy | RT-1 | 真实机器人大数据 + Transformer + 离散动作 |
| 跨 embodiment 数据扩展 | RT-X / Open X-Embodiment | 多机器人、多数据集统一训练 |
| VLM-based VLA | OpenVLA, Octo, π₀ | 更强视觉语言 backbone，action chunk / diffusion / flow |
| Real-world deployment 强化 | π₀.₅ / π₀.₆ / π₀.₇ 等 | latency、robustness、reasoning、feedback、deployment |

RT-1 的价值不在于今天仍然是最强架构，而在于它提出并验证了几个后来 VLA 都绕不开的问题：

- 数据规模；
- 任务多样性；
- 动作表达；
- 语言-视觉融合；
- 实时推理；
- 跨数据源吸收；
- 长任务和低层技能的组合。

---

## 35. 总结

RT-1 是一篇机器人基础模型发展史上非常关键的论文。

它证明了机器人 policy 可以通过大规模、多任务、多样化真实数据获得较强泛化能力，并提出了一个在真实机器人上可实时运行的 VLA-style policy 架构。

完整 pipeline 可以总结为：

```text
Instruction
  → Universal Sentence Encoder
  → fixed language embedding
  → FiLM parameters

Image history
  → FiLM-conditioned EfficientNet
  → 9×9×512 feature map per frame
  → flatten to 81 visual-language tokens
  → TokenLearner compresses 81 tokens to 8 tokens
  → 6-frame history gives 48 tokens
  → decoder-only Transformer
  → action readout
  → 11 action dimensions × 256 bins
  → categorical CE training
  → closed-loop 3Hz real robot control
```

从方法上看，RT-1 是一个大规模 language-conditioned behavior cloning system；从贡献上看，它是早期机器人 foundation model 的重要验证；从局限上看，它还没有现代 VLA 的强语言推理、长时序 memory、continuous generative action modeling 和跨 embodiment 统一能力。

因此，读 RT-1 的重点不是把它当成最新 SOTA，而是理解它提出的主线：

> 机器人策略模型要走向 foundation model，核心瓶颈不只是模型结构，而是数据规模、任务多样性、动作表达、实时控制和跨环境泛化的系统性组合。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)

## 相关笔记
- [[ChatGPT-RT-2 论文综述|RT-2 论文综述]]
- [[FAST_知识总结|FAST 知识总结]]
- [[ChatGPT-Pi_0机器人文章分析|pi0 机器人文章分析]]
- [[ChatGPT-RDT-1B|RDT-1B]]
