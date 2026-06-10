---
title: VQ-VAE 与自回归先验模型综述
type: concept_note
topic: generative_model
status: mature
importance: high
updated: 2026-06-10
tags:
  - vqvae
  - autoregressive
  - codebook
  - quantizer
  - generative-model
---

# VQ-VAE 与自回归先验模型综述

> 整理自 ChatGPT 讲解 + 问答讨论，覆盖 VQ-VAE 基础、自回归先验结构、训练细节、Weight Tying 等核心知识点。

---

## 1. VQ-VAE 基础

### 1.1 整体架构

VQ-VAE（Vector Quantized Variational Autoencoder）由三部分组成：

```
图像 → Encoder → 连续特征 z_e → Quantizer → 离散 token z_q → Decoder → 重建图像
```

- **Encoder**：将原始图像压缩成连续的特征向量。
- **Quantizer**：将连续特征映射到最近的 codebook 向量，输出离散的 token id（codebook 索引）。
- **Decoder**：根据量化后的特征重建图像。

### 1.2 Codebook

Codebook 是 VQ-VAE 的核心，本质是一个可学习的向量查找表：

```
codebook: (codebook_size, embedding_dim)
```

- 每一行是一个"码字"（code vector）。
- Quantizer 对每个空间位置的特征，找到 codebook 中与其欧氏距离最近的码字，输出其索引 id。
- 一张图片经过 Encoder + Quantizer 后，变成一个整数序列：

```
z = [z1, z2, ..., zN],  zi ∈ [0, codebook_size - 1]
```

### 1.3 训练目标

VQ-VAE 的训练 loss 由三部分组成：

$$
\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{重建 loss}} + \underbrace{\|sg(z_e) - z_q\|^2}_{\text{codebook loss}} + \underbrace{\beta \|z_e - sg(z_q)\|^2}_{\text{commitment loss}}
$$

- **重建 loss**：Decoder 重建质量。
- **Codebook loss**：让 codebook 向量靠近 encoder 输出。
- **Commitment loss**：让 encoder 输出靠近 codebook 向量，防止 encoder 输出"乱跑"。
- $sg(\cdot)$：stop-gradient，阻止梯度回传。

### 1.4 为什么同时需要 Codebook Loss 和 Commitment Loss？

#### 根本原因：梯度无法穿过 Quantizer

Quantizer 的操作是：

```
z_q = codebook[argmin_k ||z_e - e_k||]
```

`argmin` 是不可微的，梯度在这里断掉：

```
loss → Decoder → z_q → ✗ argmin 不可微 ✗ → z_e → Encoder
```

因此 Encoder 和 Codebook 都收不到来自重建 loss 的梯度，必须用额外的 loss 单独驱动它们更新。

#### 两个 Loss 各自驱动谁

| Loss | stop-gradient 位置 | 驱动谁更新 |
|---|---|---|
| Codebook loss $\|sg(z_e) - z_q\|^2$ | sg 套在 $z_e$ 上 | 只更新 **Codebook**，Encoder 不动 |
| Commitment loss $\|z_e - sg(z_q)\|^2$ | sg 套在 $z_q$ 上 | 只更新 **Encoder**，Codebook 不动 |

stop-gradient **人为隔离**了两者的更新，让它们各自负责靠近对方，互不干扰。

直觉上：

```
Codebook loss：让 codebook 追着 encoder 跑
Commitment loss：让 encoder 别跑太快，等着 codebook
```

#### 为什么不能合并成一个 $\|z_e - z_q\|^2$？

如果去掉所有 stop-gradient，直接用 $\|z_e - z_q\|^2$，梯度会同时流向 Encoder 和 Codebook，理论上也能收敛，但实际训练中有以下问题：

- **Codebook 通常用 EMA 更新**：不走普通梯度下降，所以需要单独的 codebook loss 来驱动。
- **Encoder 容易"乱跑"**：Encoder 表达能力强，输出可能频繁大幅跳动，导致 codebook 来不及跟上，出现 **codebook collapse**（大量 codebook 向量从不被使用）。
- **β 系数提供独立控制**：commitment loss 的系数 $\beta$ 让你单独调节 Encoder 被约束的力度，而不影响 Codebook 的更新节奏。

---

## 2. 自回归先验模型（Autoregressive Prior）

VQ-VAE 只解决了图像的**压缩与重建**问题。要做**图像生成**（特别是文本条件生成），还需要在 image token 空间上训练一个先验模型，建模：

$$
P(z_1, z_2, \ldots, z_N \mid \text{text})
$$

这个先验模型通常是一个**自回归 Transformer**。

### 2.1 序列组成

输入序列由文本 token 和图像 token 拼接而成：

```
[text tokens] + [<image_start>] + [image tokens]
```

- **text tokens**：文本 prompt 经 tokenizer 转成 token id 序列。
- **image tokens**：VQ-VAE encoder + quantizer 输出的 codebook 索引序列。

### 2.2 Embedding Table 设计

文本和图像使用**两个独立的 Embedding Table**，最终维度对齐到同一个 `d_model`：

| Embedding Table | Shape |
|---|---|
| 文本 token embedding | `(vocab_size_text, d_model)` |
| 图像 token embedding | `(codebook_size, d_model)` |

两者 embedding 都送入 Transformer，在同一个 hidden dimension 空间内处理。

> **注意**：DALL·E 1 的实际实现中，text 和 image token 共用同一个 embedding table，通过 offset 区分。"两个独立 table"是更通用的实现方式，并非唯一方案。

### 2.3 Embedding Lookup 的本质

Token → Embedding 的过程**只是一个查表操作**，没有矩阵乘法发生：

```python
embedding_vector = embedding_table[token_id]  # 直接索引，无计算
```

反向传播时，梯度只更新**本 batch 中被查到的那几行**，其余行梯度为零（稀疏更新）。

---

## 3. 自回归训练

### 3.1 Teacher Forcing

训练时采用 Teacher Forcing：不使用模型自己生成的 token，而是将真实历史 token 直接作为输入，让模型预测下一个真实 image token。

```
输入:  [t1, t2, ..., tM, <image_start>, z1, z2, ..., zN-1]
目标:  [z1, z2, z3, ..., zN]
```

### 3.2 Loss 计算

对每个 image token 位置，做分类预测（cross entropy），**不是预测 embedding vector 的 MSE**：

```
logits_i.shape = [codebook_size]
target_i = z_i  （真实 codebook id）
loss_i = cross_entropy(logits_i, target_i)
```

总 loss：

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P_\theta(z_i \mid \text{text}, z_{<i})
$$

### 3.3 Loss Mask

通常只在 image token 部分计算 loss，text token 部分 ignore：

```
[text tokens]        → ignore（只作为条件）
<image_start>        → ignore
[image token logits] → compute cross entropy loss
```

因为训练目标是 $P(\text{image tokens} \mid \text{text})$，text token 只作为条件输入。

### 3.4 与 LLM 的类比

| | LLM | VQ-VAE Prior |
|---|---|---|
| 输入 | 文本 token | text token + image token |
| 预测目标 | 下一个文本 token id | 下一个 image token id |
| Vocab | 文本词表 | 图像 codebook id 集合 |
| Loss | cross entropy on vocab | cross entropy on codebook |

---

## 4. 自回归生成循环

推理时，按如下步骤逐步生成 image token：

1. 将文本 token embedding 固定在序列前端作为条件。
2. 预测第一个 image token：$P(z_1 \mid \text{text})$，采样得到 $z_1$。
3. 查 image embedding table 得到 $e_{z_1}$，追加到序列。
4. 预测下一个：$P(z_2 \mid \text{text}, z_1)$，采样得到 $z_2$。
5. 重复直到生成完整 image token 序列 $[z_1, z_2, \ldots, z_N]$。
6. 将 image token 序列送入 VQ-VAE Decoder，重建出图像。

---

## 5. Weight Tying（输入输出权重共享）

### 5.1 为什么可以共享

Logit head 和 image embedding table 的权重 shape 完全相同：

```python
image_embedding = nn.Embedding(codebook_size, d_model)
# weight shape: (codebook_size, d_model)

logit_head = nn.Linear(d_model, codebook_size, bias=False)
# weight shape: (codebook_size, d_model)
```

数学上，logit 计算等价于：

$$
\text{logits} = h \cdot W^\top
$$

其中 $W$ 正是 image embedding table 的权重矩阵。

### 5.2 实现方式

```python
self.logit_head.weight = self.image_embedding.weight
```

两者指向**同一块内存**，前向传播、反向传播、参数更新都只操作这一份权重，参数量减少一半。

### 5.3 两份梯度的来源

共享权重 $W$ 在反向传播时接受**两份梯度的累加**：

**第一份**（来自 logit head，dense 梯度）：

```
h_image @ W.T → logits → loss
∂L/∂W  ←  直接从 loss 回传，W 每一行都有梯度
```

**第二份**（来自 image embedding lookup，sparse 梯度）：

```
image_ids → W[image_ids] → image_emb → transformer → loss
∂L/∂W[image_ids]  ←  只有本 batch 出现的 token id 对应行有梯度
```

PyTorch 自动累加两份梯度：

```python
W.grad = ∂L/∂W_logit_head + ∂L/∂W_image_embedding
```

### 5.4 对 Embedding 学习的影响

**会影响，但通常是正向影响。**

- Embedding 自身的梯度：学习"这个 token 应该被表示成什么向量，让 Transformer 读进去好用"。
- Logit head 传回的梯度：学习"这个 token 的向量，应该和'该出现它的位置的 hidden state'更相似"。

两者优化的是同一件事：

> **让 token 的向量表示，和语义上应该出现它的上下文表示，在同一空间里对齐。**

这也是 NLP 中 GPT、BERT 等模型普遍采用 weight tying 的原因。

### 5.5 潜在的冲突场景

当 input 和 output 的语义需求差异较大时，weight tying 可能产生轻微的梯度拉扯。Image token 的 weight tying 比 text token 更自然，因为 image codebook 中每个 id 对应一个视觉 patch 特征，input 与 output 的语义需求差异较小。

### 5.6 对比总结

| | 独立权重 | Weight Tying |
|---|---|---|
| 参数量 | `2 × codebook_size × d_model` | `1 × codebook_size × d_model` |
| 实现 | 默认 | 加一行赋值 |
| 梯度来源 | 各自独立 | 两份累加（dense + sparse）|
| 效果 | 灵活 | 参数更少，通常效果相当甚至更好 |

---
## 6. txt 和 image的vocabulary问题
### 方案一：两个独立的 logit head（职责分离）

```
hidden state → logit_head_text:  Linear(d_model, vocab_size_text)   → text logits
hidden state → logit_head_image: Linear(d_model, codebook_size)     → image logits
```

推理时根据当前生成阶段**手动切换**用哪个 head：

python

```python
if current_position < text_length:
    logits = logit_head_text(h)    # 从文本词表采样
else:
    logits = logit_head_image(h)   # 从 codebook 采样
```

- 优点：两个空间完全隔离，互不干扰。
- 缺点：模型本身不能"自己决定"什么时候切换，需要外部控制。

---

### 方案二：单一 logit head，vocabulary 合并（DALL·E 1 的做法）

```
unified vocab = [text tokens (0 ~ vocab_text-1)] + [image tokens (vocab_text ~ vocab_text+codebook_size-1)]
```

只有一个 logit head：

```
hidden state → Linear(d_model, vocab_text + codebook_size) → unified logits
```

推理时从统一的分布里采样，模型**自己学会**在合适位置生成 `<image_start>` token，之后自然切换到 image token 区间。

- 优点：更像真正的统一自回归模型，`<image_start>` 也是可学习的边界。
- 缺点：text 和 image token 在同一个概率分布里竞争，训练时需要仔细处理。

---

### 两种方案的本质区别

||方案一|方案二|
|---|---|---|
|logit head 数量|2 个|1 个|
|vocab 大小|各自独立|`vocab_text + codebook_size`|
|切换时机|外部硬控制|模型自己学|
|weight tying|image head ↔ image embedding|统一 head ↔ 统一 embedding|
|代表模型|很多定制实现|DALL·E 1|
## 7. 整体流程总结

```
训练阶段：

  图像 ──→ VQ-VAE Encoder ──→ Quantizer ──→ image tokens [z1..zN]
                                                      │
  文本 ──→ Tokenizer ──→ text tokens                  │
                │                                     │
                └──────────── Autoregressive Prior ←──┘
                              (Transformer)
                              Teacher Forcing + CE Loss on image tokens


生成阶段：

  文本 ──→ Tokenizer ──→ text tokens
                │
                └──→ Autoregressive Prior ──→ 逐步采样 image tokens [z1..zN]
                                                      │
                                             VQ-VAE Decoder
                                                      │
                                                   生成图像
```

---

## 8. 参考概念索引

| 概念 | 说明 |
|---|---|
| VQ-VAE | 用离散 codebook 做图像压缩的变分自编码器 |
| Codebook | 可学习的离散向量查找表 |
| Image Token | VQ-VAE 输出的 codebook 索引整数序列 |
| Autoregressive Prior | 在 image token 空间建模生成分布的自回归模型 |
| Teacher Forcing | 训练时用真实历史 token 而非模型生成 token 作为输入 |
| Weight Tying | 输入 embedding table 与输出 logit head 共享同一份权重 |
| Loss Mask | 训练时只在 image token 位置计算 loss，忽略 text token |
| Sparse Gradient | Embedding lookup 时只有被查到的行有梯度更新 |
