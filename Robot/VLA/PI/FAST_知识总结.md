---
title: FAST 核心知识总结
type: paper_note
topic: action_tokenization
status: mature
importance: high
updated: 2026-06-10
tags:
  - fast
  - action-tokenization
  - vla
  - dct
  - bpe
  - robotics
---

# FAST 核心知识总结（校订版）

> 基于论文：*FAST: Efficient Action Tokenization for Vision-Language-Action Models*
> Pertsch et al., Physical Intelligence / UC Berkeley / Stanford, arXiv 2501.09747

---

## 1. FAST 要解决什么问题？

FAST 论文关注的是 **VLA 模型中的 action tokenization 问题**。

在自回归式 Vision-Language-Action 模型里，模型本质上和语言模型类似：

$$
p(T_1, T_2, ..., T_n \mid \text{image, language, state})
$$

只不过这里的 $T_i$ 不一定是文本 token，也可以是 action token。

问题在于，机器人 action 是连续值：

$$
a_t \in \mathbb{R}^D
$$

而自回归 Transformer 只能预测离散 token。因此需要把连续 action 序列变成离散 token 序列。

传统方法是 **naive binning**：每个 timestep、每个 action dimension 单独分箱（通常 256 bins）。

例如一个 1 秒 action chunk（50Hz，14 维双臂）：

```
50 × 14 = 700 个 action tokens
```

这会带来两个问题：

**第一**，token 太长，训练和推理都慢。

**第二**，高频 action 相邻 timestep 极其相似，next-token prediction 的学习信号很弱。模型可能只学会"下一个 token 和上一个差不多"，而不是学会真正的动作规划。论文在 toy case 中指出，高采样率下 naive tokenization 的预测误差会显著恶化。

FAST 的核心目标就是：

> 把高频、冗余、连续的 action chunk 压缩成更短、更高信息密度的离散 action tokens。

---

## 2. FAST 的整体流程

FAST 的流程可以概括为：

```
连续 action chunk
    → quantile normalization
    → DCT 变换到频域
    → scale-and-round 量化成整数
    → low-frequency-first flatten
    → BPE 压缩
    → action tokens
```

也就是：

$$
a_{1:H}
\rightarrow C
\rightarrow \bar{C}
\rightarrow [T_1, ..., T_k]
\rightarrow [\bar{T}_1, ..., \bar{T}_m]
$$

其中：

- $a_{1:H}$：未来 $H$ 步连续动作；
- $C$：DCT 频域系数；
- $\bar{C}$：量化后的整数频域系数；
- $[T_1, ..., T_k]$：flatten 后的整数序列；
- $[\bar{T}_1, ..., \bar{T}_m]$：BPE 压缩后的最终 action tokens。

论文 Figure 4 和 Algorithm 1 展示的就是这个过程：DCT → Quantize → Flatten → BPE。

---

## 3. Action chunk 是什么？

FAST 不是对单个 timestep 的 action 做 tokenization，而是对 **一整段未来动作** 做 tokenization。

例如：
```
H = 50，D = 14
→ action chunk ∈ ℝ^{50×14}（未来 1 秒的双臂动作）
```

论文实验中，作者通常 tokenize **1-second action chunks**：

```
15Hz, 7维  → shape = (15, 7)
20Hz, 7维  → shape = (20, 7)
50Hz, 14维 → shape = (50, 14)
```

这和 naive binning 的区别很大：naive binning 是逐点离散化；FAST 是先把整段动作压缩成频域表示，再离散化。

---

## 4. 第一步：Quantile Normalization

FAST 先对每个 action dimension 做分位数归一化。

论文中使用训练集每个 action 维度的 **1% 分位数和 99% 分位数**，把数值映射到 $[-1, 1]$。

这样做有两个原因：

**第一**，不同机器人、不同 action space 的数值尺度不同，归一化可以统一尺度。

**第二**，用 1%/99% 分位数而不是 min/max，可以避免 outlier action 影响整体尺度。

$$
\tilde{a}_{t,d}
= 2 \cdot
\frac{
\mathrm{clip}(a_{t,d}, q_{1\%,d}, q_{99\%,d}) - q_{1\%,d}
}{
q_{99\%,d} - q_{1\%,d}
}
- 1
$$

---

## 5. 第二步：DCT 编码

DCT 是 FAST 的核心。

对于 action chunk $A \in \mathbb{R}^{H \times D}$，FAST **对每个 action dimension 沿时间轴分别做 DCT**：

$$
C^i_j = \mathrm{DCT}(a^i_{1:H})
$$

其中 $i$ 是 action dimension，$j$ 是 frequency index。

做完 DCT 后，矩阵形状仍然是 $D \times H$（每个维度对应一行，每列对应一个频率分量），但含义变了：

```
原来每列是 timestep → 现在每列是 frequency index
col 0 → freq0，最低频（整体均值/直流分量）
col 1 → freq1，低频变化趋势
col 2 → freq2，中频
...
col H-1 → 高频细节
```

DCT 的作用是把时间域 action 曲线变成频域系数。对于平滑动作，大部分信息会集中在低频系数里，高频项通常很小。论文指出，低频捕捉整体形状，高频反映 sharp jumps；平滑信号可以用少量 DCT coefficients 表示。

---

## 6. 为什么 DCT 适合机器人 action？

机器人动作通常是平滑的，尤其是高频控制数据。

例如 x 方向位置：`0.10, 0.12, 0.15, 0.18, 0.20, ...`，相邻 timestep 变化很小。

如果直接逐 timestep tokenization，会得到大量相似 token。但 DCT 会把这段平滑曲线表示成：

```
少量低频系数 + 大量接近 0 的高频系数
```

这是 FAST 的关键直觉：

> 不要让模型逐 timestep 预测高度相关的 action token，而是让模型预测整段 action chunk 的频域形状。

---

## 7. 第三步：Scale-and-Round 量化

DCT 输出的是连续值，还不能直接作为 token。因此 FAST 对 DCT 系数做量化：

$$
\bar{C}^i_j = \mathrm{round}(\gamma \cdot C^i_j)
$$

其中 $\gamma$ 是 scale 超参数。

论文单数据集实验中的默认值：

```
γ = 10
BPE vocab size = 1024
```

例如：
```
C  = [0.990, -0.460, 0.048, 0.003]
γC = [9.90,  -4.60,  0.48,  0.03]
round(γC) = [10, -5, 0, 0]
```

经过 round 后，大量小的高频系数会变成 0。

这一步是 FAST 中主要的 **有损压缩** 环节。BPE 是无损的，但 scale-and-round 会带来量化误差。论文明确说明，FAST 的压缩不是完全无损，compression ratio 和 reconstruction accuracy 由 scale 参数 $\gamma$ 控制。

---

## 8. 第四步：得到稀疏频域整数矩阵

scale-and-round 后，原来的 DCT 系数矩阵变成稀疏整数矩阵：

```
dim0: [124,  12,  -3,   0,   0,   0,  12, ...]
dim1: [-86,   0,   0,   0,   0,   0,   0, ...]
dim2: [344,   3,   1,   0,   0,   1,   5, ...]
...
```

这个矩阵通常非常稀疏。原因是：

```
平滑 action → 高频 DCT 系数很小 → round 后变成 0
```

---

## 9. 第五步：Low-Frequency-First Flatten

FAST 把矩阵展开为一维整数序列，供 BPE 使用。

展开顺序选择 **低频优先（column-first flattening）**：先把所有维度的最低频分量排在一起，再是次低频，以此类推。

例如矩阵：
```
dim0: [10, -3,  3]
dim1: [-5,  0,  4]
dim2: [ 0, -2,  0]
```

低频优先 flatten 后：
```
[10, -5, 0,   ← freq0 of all dims
 -3,  0, -2,  ← freq1 of all dims
  3,  4,  0,  ← freq2 of all dims
 ...]
```

（对应 Algorithm 1 中的 $[\bar{C}^1_1, \bar{C}^2_1, \ldots, \bar{C}^1_2, \bar{C}^2_2, \ldots]$）

论文强调 flatten 顺序很重要，选择低频优先的原因是：

> 低频系数决定整段 action chunk 的整体形状，让自回归模型先预测低频项，可以让 rollout 更稳定。

也就是说，模型先预测"这 1 秒动作整体往哪里走"，再预测"动作细节和高频修正"，这比先预测某个 action dimension 的所有频率更合理。

---

## 10. 第六步：BPE 压缩

BPE 是 FAST 中**唯一需要训练的部分**。

BPE 的输入是 scale-and-round + flatten 后的一维整数序列，在大量 action chunk 数据上训练，反复寻找最频繁的相邻 pair 并合并成新 token。

论文说 BPE 的作用是：
- **squash zero-valued components**（压缩大量 0）
- **merge frequently-occurring coefficient combinations across action dimensions**（合并常见频域系数组合）

DCT + quantize 之后会产生很多 0；这些 0 仍然保留在 flatten 后的整数序列里；然后 BPE 在训练/编码过程中通过 merge 频繁出现的相邻模式，把大量连续的 0 合并进更大的 token 里。

最终把长整数序列压缩成较短 token 序列。

### 为什么 BPE 适合 FAST？

因为 DCT + round 之后的整数序列具有两个特点：

**第一**，大量连续 0：BPE 很容易把它们合并成短 token。

**第二**，低频系数组合会重复出现（机器人执行相似动作时，频域低频模式类似）：BPE 可以把这些常见组合变成一个 token。

---

## 11. FAST 的解码过程

推理时，VLA 生成 action token 后，需要 decode 回连续 action。

解码流程是编码的反过程：

```
action tokens
    → BPE decode（无损）
    → flattened integer sequence
    → unflatten 成整数矩阵（无损）
    → ÷ γ 得到近似 DCT 系数
    → inverse DCT
    → inverse normalization
    → continuous action chunk
```

主要误差来自量化步骤 $\bar{C} = \mathrm{round}(\gamma C)$，BPE 本身不引入误差。

---

## 12. FAST 如何接入 VLM/VLA 词表？

论文原文：

> "During training, we tokenize 1-second action chunks and overwrite the **least used tokens** in the VLM vocabulary with the resulting action tokens."

这句话的意思是：**训练时，把 FAST action tokens 映射到 VLM 词表中使用频率最低的那批 token id 上。**

注意，这不是扩展词表，而是 **overwrite / repurpose least-used tokens**。

FAST BPE vocab size 默认为 1024，就复用词表中 1024 个最少使用的 token id。

这样做的好处是：

```
不改变 vocabulary size
不改变 embedding matrix shape
不改变 LM head output dimension
不新增 action regression head
```

VLA 仍然用原来的 next-token prediction 方式训练，无需修改模型架构。

> ⚠️ 原始总结中错误地编造了"257k"这一具体词表大小数字，原论文并未提及。实际词表大小取决于所用 VLM backbone（π0 基于 PaliGemma-3B，OpenVLA 基于 Prismatic 7B），各有不同。

---
## 13. decode的全过程
### 1. 先明确：VLA 推理出来的 token 不是直接等于动作

假设训练时作者把 FAST action token 覆盖到了 VLM 词表里一些 least-used token 上。

例如：

```text
VLM token id 30976 -> FAST action token 0
VLM token id 30977 -> FAST action token 1
VLM token id 30978 -> FAST action token 2
...
```

推理时，VLA 看到图像、语言指令、proprio state 后，会自回归生成一串 VLM token ids：

```text
[30993, 31209, 30980, 31788]
```

第一步不是直接 decode 动作，而是先把这些 VLM token ids 映射回 FAST action token ids：

```text
[30993, 31209, 30980, 31788]
        ↓ rare-token-id inverse map
[17, 233, 4, 812]
```

这串 `[17, 233, 4, 812]` 才是 FAST/BPE action token 序列。

---

### 2. BPE token 到底怎么 decode？

你说的“推理出这个 token 的位置，找到对应的 BPE 压缩后的结果”，基本就是这个意思，但更准确地说：

> 每个 BPE token id 对应 BPE 词表中的一个符号，这个符号可能是一个原始整数，也可能是多个整数合并出来的组合。BPE decode 就是把这些合并符号递归展开，恢复成原始的整数序列。

论文 Figure 4 里展示了一个例子：flatten 后的 DCT 整数序列类似：

```text
124, -86, 344, -45, 178, 12, 0, 3, 0, 15, ...
```

经过 BPE 后变成压缩 action tokens：

```text
978, 233, 19, 1022, 1
```

论文说明 BPE 会 squash 大量 0，并合并跨 action 维度频繁出现的系数组合。`FAST.pdf`

所以 decode 时就是反过来：

```text
[978, 233, 19, 1022, 1]
    ↓ BPE decode
[124, -86, 344, -45, 178, 12, 0, 3, 0, 15, ...]
```

这里的关键是：**BPE decode 是无损的**。如果 token 序列合法，它可以精确恢复 scale-and-round 后的整数序列。

---

### 3. 一个具体的 BPE decode 例子

假设 BPE 词表里有这些 merge rule：

```text
token 1000 = [0, 0]
token 1001 = [1000, 1000]       # 等价于 [0, 0, 0, 0]
token 1002 = [10, -3]
token 1003 = [3, 1]
token 1004 = [1002, 1003]       # 等价于 [10, -3, 3, 1]
token 1005 = [-5, 0]
```

VLA 推理得到 FAST action tokens：

```text
[1004, -1, 3, 1005, 4, 0, 1001]
```

BPE decode 时递归展开：

```text
1004 -> [1002, 1003]
     -> [10, -3, 3, 1]

1005 -> [-5, 0]

1001 -> [1000, 1000]
     -> [0, 0, 0, 0]
```

最终恢复成整数序列：

```text
[10, -3, 3, 1, -1, 3, -5, 0, 4, 0, 0, 0, 0, 0]
```

这串整数不是最终动作，而是 **量化后的 DCT 系数 flatten 序列**。

---

### 4. BPE decode 后要 unflatten

FAST 在 encoding 时做过 low-frequency-first flatten。论文说他们将 DCT coefficient matrix flatten 成一维整数序列，交错 action dimensions，并且先放低频 components；这样自回归模型先预测整体 shape 相关的低频项，rollout 更稳定。`FAST.pdf`

所以 decode 时要按同样规则反过来。

假设 action chunk 的 shape 是：

$$
H = 8,\quad D = 6
$$

那么 BPE decode 后应该恢复出：

$$
H \times D = 48
$$

个整数。

例如 BPE decode 得到：

```text
seq = [
  10, -3, 3, 1, -1, 3,
  -5, 0, 4, 0, -1, -2,
  0, -2, 0, 0, 0, 0,
  ...
]
```

因为 FAST 是 low-frequency-first flatten，所以前 6 个数是 `freq0` 的 6 个 action 维度：

```text
freq0: [10, -3, 3, 1, -1, 3]
```

接下来 6 个数是 `freq1`：

```text
freq1: [-5, 0, 4, 0, -1, -2]
```

再接下来是 `freq2`：

```text
freq2: [0, -2, 0, 0, 0, 0]
```

最终 unflatten 成矩阵：

```text
Q shape = (H, D) = (8, 6)

freq0: [10, -3,  3,  1, -1,  3]
freq1: [-5,  0,  4,  0, -1, -2]
freq2: [ 0, -2,  0,  0,  0,  0]
freq3: [ 0,  0,  0,  0,  0,  0]
freq4: [ 0,  0,  0,  0,  0,  0]
freq5: [ 0,  0,  0,  0,  0,  0]
freq6: [ 0,  0,  0,  0,  0,  0]
freq7: [ 0,  0,  0,  0,  0,  0]
```

这个 $Q$ 就是量化后的 DCT coefficient matrix。

---

### 5. 然后 inverse quantization

encoding 时做的是：

$$
\bar C^i_j = \mathrm{round}(\gamma C^i_j)
$$

所以 decode 时做近似反量化：

$$
\hat C^i_j = \frac{\bar C^i_j}{\gamma}
$$

如果论文常用：

$$
\gamma = 10
$$

那么：

```text
Q:
freq0: [10, -3,  3,  1, -1,  3]
freq1: [-5,  0,  4,  0, -1, -2]
...

C_hat = Q / 10:

freq0: [ 1.0, -0.3, 0.3, 0.1, -0.1,  0.3]
freq1: [-0.5,  0.0, 0.4, 0.0, -0.1, -0.2]
...
```

注意这里已经不可能完全恢复原来的 DCT 系数，因为 `round` 已经丢掉了小数精度。论文也明确说 FAST 的压缩不是完全无损，重建精度和压缩率由 scale 参数 $\gamma$ 控制。`FAST.pdf`

所以：

```text
BPE decode: 无损
unflatten: 无损
Q / γ: 近似恢复
```

主要误差来自 scale-and-round。

---

### 6. 对每个 action 维度做 inverse DCT

现在得到的是近似 DCT 系数矩阵：

$$
\hat C \in \mathbb{R}^{H \times D}
$$

其中：

```text
row = frequency index
col = action dimension
```

要恢复时间域动作，就对每个 action dimension 独立做 inverse DCT：

$$
\hat a^i_{1:H} = \mathrm{IDCT}(\hat C^i_{1:H})
$$

如果矩阵是：

```text
C_hat shape = (H, D)
```

那么代码概念上就是：

```python
A_norm_hat = idct(C_hat, axis=0)
```

这里 `axis=0` 表示沿频率/时间轴反变换，每一列单独 IDCT。

得到：

```text
A_norm_hat shape = (H, D)
```

这就是归一化空间里的连续 action chunk。

例如：

```text
A_norm_hat:

t0: [0.108, -0.198,  0.302, 0.035, -0.084, 0.008]
t1: [0.146, -0.144,  0.272, 0.035, -0.077, 0.023]
...
t7: [0.599, -0.198, -0.090, 0.035,  0.014, 0.204]
```

---

### 7. 最后 inverse normalization

encoding 前，FAST 把每个 action dimension 根据训练集的 1% / 99% quantile 映射到了：

$$
[-1, 1]
$$

decode 时要反过来，把 normalized action 还原回机器人 action 的真实尺度。

如果 encoding 近似是：

$$
\tilde a_{t,d}
=
2 \cdot
\frac{
a_{t,d} - q_{1\%,d}
}{
q_{99\%,d} - q_{1\%,d}
}
- 1
$$

那么 decoding 就是：

$$
\hat a_{t,d}
=
\frac{\hat{\tilde a}_{t,d} + 1}{2}
\cdot
(q_{99\%,d} - q_{1\%,d})
+
q_{1\%,d}
$$

得到最终机器人动作：

```text
A_hat shape = (H, D)
```

比如：

```text
H = 50
D = 14
```

最后就是未来 1 秒、双臂 14 维连续 action chunk：

```text
A_hat ∈ R^(50×14)
```

机器人可以按控制频率逐步执行这段 action chunk。

## 14. 训练时 VLA 学什么？

训练样本包括：image observation、language instruction、robot proprio state、future action chunk。

首先用 FAST 把 future action chunk 编码成 action tokens，并映射到 VLM 词表 id。VLA 的训练目标就是：

$$
p(T_1, T_2, ..., T_n \mid \text{image, instruction, state})
$$

用 cross entropy / next-token prediction 训练。

VLA 不是天然懂 FAST token，而是在 fine-tuning 中学会：看到这种图像 + 指令 + 当前状态，应该输出哪串 FAST action tokens。

---

## 15. FAST+ 是什么？

FAST 是一套 tokenizer 流程。对于每个数据集，可以单独训练 BPE vocabulary。

FAST+ 是作者训练好的 **universal robot action tokenizer**，在约 **1M 个 1-second robot action chunks** 上训练，覆盖：

```
单臂机器人 / 双臂机器人 / 移动操作机器人
joint-space action / end-effector action
多种控制频率
```

（详见论文附录 A 的数据表，包含 ARX、UR5、Franka、DROID、Bridge V2、OpenX 等多个数据集。）

FAST+ 的重点是：

```
tokenizer 先独立训练好（仅使用 action 数据，无需图像和语言）
之后可以作为 black-box tokenizer 用于新的 robot setup
```

但 VLA 仍然需要 fine-tuning 才能学会预测 FAST+ token。

---

## 16. "Backbone independent" 怎么理解？

论文说 FAST tokenization approach is independent of the underlying model backbone，意思是：

> FAST/FAST+ 不绑定某个特定 VLA 架构；只要 backbone 是 autoregressive token prediction 模型，就可以把 FAST action tokens 接入其 vocabulary，并进行 fine-tuning。

论文中作者测试了：

```
π0 + FAST
OpenVLA + FAST+
```

OpenVLA 原本用 naive tokenization，在高频 T-shirt Folding（50Hz）上效果很差；接入 FAST+ 后性能显著提升。因此 FAST 的收益不属于某一个特定 backbone，可以迁移到不同自回归 VLA 上。

---

## 17. FSQ 在论文中的角色

FSQ（Finite Scalar Quantization）是论文中作为 **learned compression baseline** 出现的。

它是一种神经网络编码器-解码器方案（VQ-VAE 的更简单替代），流程是：

```
action chunk → neural encoder → scalar quantization → discrete latent tokens
             → neural decoder → reconstructed action
```

论文中比较了：Naive binning / FSQ / FAST / FAST+

结论：
- FSQ 和 FAST 都是 compression-based，因此都明显好于 naive binning；
- FAST 通常和 FSQ 相当或更好，尤其在**高频灵巧任务**上；
- FSQ 在粗粒度、低保真任务上表现尚可，但在**高频、精细控制**场景下会失败；
- FAST 更简单，无需训练复杂 encoder-decoder；
- FSQ 证明了"先压缩 action target"方向有效，FAST 是更简单、白盒的实现。

---

## 18. 方法对比

| 方法 | 核心思想 | 是否训练神经网络 | 是否可解释 | 主要问题 |
|---|---|:---:|:---:|---|
| Naive binning | 每个 timestep 每维分箱 | 否 | 高 | token 多、相关性强 |
| FSQ | encoder-decoder + scalar quantization | 是 | 中 | 需要训练和调参，高频场景失败 |
| FAST | DCT + round + BPE | 仅 BPE vocabulary | 高 | scale 有损量化 |
| FAST+ | 通用 FAST tokenizer | 仅 BPE vocabulary（预训练） | 高 | 泛化依赖预训练数据覆盖 |

---

## 19. 实验结论：Token 压缩率

论文 Table I 显示，FAST 能显著减少 1 秒 action chunk 的 token 数：

| 数据集 | 控制频率 | Action 维度 | Naive token 数 | FAST token 数 | 压缩比 |
|---|:---:|:---:|:---:|:---:|:---:|
| BridgeV2 | 5 Hz | 7 | 35 | 20 | 1.75× |
| DROID | 15 Hz | 7 | 105 | 29 | 3.6× |
| Table Bussing | 20 Hz | 7 | 140 | 28 | 5.0× |
| T-Shirt Folding | 50 Hz | 14 | 700 | 53 | 13.2× |

尤其是高频任务，压缩效果更明显。这说明 FAST 不是简单减少一点 token，而是大幅消除了高频动作序列的冗余。

论文还观察到：FAST 在各数据集上大约生成**每个机械臂约 30 个 action token**，这说明 FAST 找到了一种与控制频率大致无关的、近似反映底层动作复杂度的表示。

---

## 20. FAST 相比 diffusion VLA 的优缺点

论文把 π₀-FAST 和 diffusion π₀ 做了比较。

**优点：**
- 训练更快，在大数据集上收敛更快
- 语言跟随能力可能更强（DROID 评测中 diffusion π₀ 常常忽略语言指令）
- 可用 next-token prediction 训练，不需要复杂 diffusion action head

论文报告，在 generalist training 中，π₀-FAST 可以用约 **5× 更少的 GPU hours** 达到和 diffusion π₀ 相当的性能。

**缺点：推理更慢。**

因为 FAST 是自回归生成 action tokens，通常需要生成 **30–60 个 action tokens**；而 diffusion π₀ 使用约 300M 参数的 action expert 做约 10 步 diffusion。

在 NVIDIA 4090 上：
```
diffusion π₀：预测 1 秒 action chunk ≈ 100ms
π₀-FAST：预测 1 秒 action chunk ≈ 750ms
```

（数字来自原文 Section VI-E）

---

## 21. FAST 的局限

1. **不是完全无损**：scale-and-round 是有损的，$\gamma$ 控制重建精度和压缩率的 trade-off。

2. **推理速度慢**：自回归生成 30–60 个 action tokens，比 diffusion action expert 慢约 7.5×。

3. **需要 VLA fine-tuning**：FAST+ 是独立 tokenizer，但 VLA 不是天然会预测 FAST+ tokens，仍然需要监督训练。

4. **BPE token 序列长度可变**：工程上需要处理 decode mismatch、长度不匹配等问题。

5. **主要验证在静态操作任务**：论文主要验证了桌面操作、折衣服、DROID tabletop 等任务。更动态的平台（humanoid、legged robot、高速闭环控制）还需要进一步验证。

---

## 22. FAST 的优点

1. **解决高频 action token 冗余**：DCT 把时间冗余转成频域稀疏性，再用 BPE 压缩。

2. **简单、白盒、可解释**：相比 FSQ/VQ-VAE，不需要训练复杂 encoder-decoder，超参数少且不敏感。

3. **易接入已有 VLM**：通过 overwrite least-used VLM tokens，无需修改模型架构。

4. **训练效率高**：让自回归 VLA 在高频、灵巧任务上可训练，大规模训练比 diffusion VLA 收敛更快。

5. **可训练通用 tokenizer**：FAST+ 可作为 universal action tokenizer 用于多种 robot setup。

---

## 23. 一句话理解 FAST

> FAST 把连续高频机器人动作从"逐 timestep 的数值序列"变成"频域低频形状参数 + BPE 压缩 token"，从而让自回归 VLA 能像预测语言 token 一样高效预测未来 action chunk。

```
naive binning：让模型预测每一帧每一维动作
FAST：        让模型预测整段动作的频域形状
```

FAST 的核心不是"换了一个 tokenizer 小技巧"，而是把 VLA 的 action prediction 目标从低信息密度的逐点预测，改成高信息密度的压缩动作表示。

---

## 24. 最值得记住的几个点

1. **FAST 解决的是 action tokenization，而不是 VLA backbone 本身。**

2. **DCT 是核心：把时间域 action 沿每个维度变成频域系数，利用动作的时间平滑性做压缩。**

3. **scale-and-round 把连续 DCT 系数变成整数，并带来主要有损压缩。**

4. **BPE 是无损压缩，用来合并大量 0 和常见频域系数组合。**

5. **FAST action tokens overwrite VLM 词表中 least-used tokens，而不是扩展词表；具体词表大小取决于所用 VLM backbone，论文未给出统一数字。**

6. **FAST+ 是独立训练的 universal tokenizer，但 VLA 仍需 fine-tuning 才能预测正确 token。**

7. **FAST 比 naive binning 更适合高频灵巧动作，因为它减少了 token 冗余，提高了每个 token 的信息量。**

8. **FAST 训练效率高（5× 更少 GPU hours），但推理速度慢是当前主要限制。**

---

## 25. 最终总结

FAST 的贡献可以分成三层：

**第一层：问题洞察。**
高频机器人 action 用 naive binning 会产生大量高度相关 token，导致自回归 VLA 的 next-token prediction 学习信号弱，模型陷入"复制上一个 token"的局部最优。

**第二层：方法设计。**
用 DCT 把 action chunk 变到频域，用 scale-and-round 量化成稀疏整数矩阵，低频优先 flatten，再用 BPE 压缩成短 action token 序列。

**第三层：系统接入。**
把 FAST action tokens 覆盖到 VLM vocabulary 中最少使用的 token id 上，让已有 VLM/VLA 可以直接用 next-token prediction 学习机器人动作，无需修改模型结构。

## 相关笔记

- [[Pi_0机器人文章分析|pi0]]：pi0-FAST 与 flow matching action expert 的关系。
- [[Pi_0.6论文问题解答|pi0.6]]：FAST discrete action tokens 与 continuous action chunks 的 joint likelihood。
- [[RDT-1B|RDT-1B]]：continuous diffusion action chunk 路线，可与 FAST 离散 token 路线对比。
- [[Diffusion Policy 概述|Diffusion Policy]]：连续 action diffusion 路线。
- [[Pi0_7_technical_report|pi0.7]]：FAST token CE loss 与 flow matching action expert 的后续结合。
- [[MolmoAct2论文框架分析|MolmoAct2]]：FAST action tokens、per-layer KV conditioning 与 continuous action expert 的部署型组合。
- [[VQVAE_综述|VQ-VAE 综述]]：离散 token、codebook、autoregressive prior 的基础概念。
- [[RT-1 论文综述|RT-1 论文综述]]
- [[RT-2 论文综述|RT-2 论文综述]]
