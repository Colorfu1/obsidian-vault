---
title: OPD On-Policy Distillation 知识笔记
type: concept_note
topic: llm_training
status: mature
importance: high
updated: 2026-08-10
tags:
  - opd
  - on-policy-distillation
  - kl
  - reverse-kl
  - llm-training
---

# OPD（On-Policy Distillation）知识笔记

## 0. 一句话理解 OPD

**OPD = 让 student 自己 rollout，然后 teacher 在 student 实际走到的 prefix/state 上给 token-level 指导。**

它不是简单地让 student 模仿 teacher 预先生成好的完整答案，而是：

```text
student 先自己生成一条 response；
teacher 在 student 的每一个 prefix 上预测 next-token distribution；
student 学 teacher 在这些 prefix 上的判断。
```

所以 OPD 结合了两类方法的优点：

```text
像 RL：数据来自 student 自己的 on-policy rollout。
像 KD：监督来自 teacher 的 dense token-level signal。
```

---

## 1. 为什么需要 OPD？

普通 SFT / KD 的训练状态通常来自 teacher 或人工数据：

```text
prompt + teacher_prefix
```

但推理时 student 看到的是自己生成的 prefix：

```text
prompt + student_prefix
```

这会导致 **train-test mismatch / exposure bias**。

举例：

```text
teacher 原本会走几何证明路线；
student 推理时可能一开始走了代数路线。
```

普通 KD 更多是在教 student：

```text
在 teacher 的 prefix 下应该怎么继续。
```

但它没有充分教 student：

```text
如果你自己已经走到了 student 的 prefix，应该怎么继续。
```

OPD 的核心价值就在于：

```text
student 自己走；
teacher 在 student 实际走到的地方指导它。
```

---

## 2. KL 散度为什么可以衡量两个分布的差？

KL 散度定义为：

$$
D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$

其中：

```text
P = 真实分布 / teacher 分布 / 目标分布
Q = 近似分布 / student 分布
```

KL 的直觉是：

> 如果真实数据来自 P，但你用 Q 去解释这些数据，平均要多付出多少 surprise / 编码代价。

单个事件 i 的 surprise 是：

$$
-\log Q(i)
$$

如果 Q 认为某个 token 概率很高，那么这个 token 真的出现时，模型不惊讶；如果 Q 认为概率很低，但它真的出现了，模型就很惊讶。

用 Q 解释 P 的平均 surprise 是 cross entropy：

$$
H(P, Q) = -\sum_i P(i)\log Q(i)
$$

用 P 自己解释 P 的平均 surprise 是 entropy：

$$
H(P) = -\sum_i P(i)\log P(i)
$$

二者相减就是 KL：

$$
D_{KL}(P\|Q) = H(P,Q) - H(P)
$$

所以 KL 表示：

```text
真实分布是 P，
但你用 Q 来近似它时，
平均多付出的解释成本。
```

注意：KL 不是严格意义上的距离，因为它有方向：

$$
D_{KL}(P\|Q) \neq D_{KL}(Q\|P)
$$

---

## 3. Forward KL 和 Reverse KL

### 3.1 Forward KL

Forward KL 通常指：

$$
D_{KL}(\pi_T \| \pi_\theta)
$$

其中：

```text
π_T = teacher
π_θ = student
```

展开是：

$$
D_{KL}(\pi_T \| \pi_\theta)
= \sum_a \pi_T(a|s) \log \frac{\pi_T(a|s)}{\pi_\theta(a|s)}
$$

它的权重来自 teacher。

直觉：

```text
teacher 认为重要的 token，student 要覆盖。
```

所以 Forward KL 更偏 **mode-covering**。

---

### 3.2 Reverse KL

Reverse KL 是：

$$
D_{KL}(\pi_\theta \| \pi_T)
$$

展开是：

$$
D_{KL}(\pi_\theta \| \pi_T)
= \sum_a \pi_\theta(a|s) \log \frac{\pi_\theta(a|s)}{\pi_T(a|s)}
$$

它的权重来自 student。

直觉：

```text
student 自己想选的 token，teacher 是否认可。
```

所以 Reverse KL 更偏 **mode-seeking**。

可以写成期望形式：

$$
D_{KL}(\pi_\theta \| \pi_T)
= \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}
\left[
\log \pi_\theta(a|s) - \log \pi_T(a|s)
\right]
$$

这个形式很重要，因为 sampled-token reverse KL 就来自这里。

---

## 4. OPD 的基本流程

一个典型 OPD iteration 是：

```text
1. 从数据集中采样 prompt x。

2. student πθ 根据当前策略生成回答 y：
   y ~ πθ(. | x)

3. 对于 y 的每个 token 位置 t：
   state s_t = prompt + y_<t

4. teacher πT 在同一个 student prefix 上计算 next-token distribution：
   πT(. | s_t)

5. student 在这些 state 上匹配 teacher。
```

核心区别：

```text
普通 KD：teacher 写答案，student 模仿 teacher 的答案。
OPD：student 自己写答案，teacher 在 student 的 prefix 上指导 student。
```

---

## 5. OPD 的算法主线：到底在优化什么？

前面已经分别介绍了 student rollout、student prefix、teacher next-token distribution
以及 Forward KL / Reverse KL。但真正理解 OPD，需要把它们连接成一条完整的优化链路。

OPD 的定义不在于某一种特定 KL，也不在于某一个特殊 loss。它真正要解决的问题是：

> **让 student 在自己实际会访问到的状态上，学习 teacher 的行为。**

因此 OPD 的优化可以拆成两个先后相连的选择：

```text
先决定训练状态来自哪里
        ↓
再决定 teacher 如何在这些状态上指导 student
```

第一步决定它为什么是 **on-policy**；第二步决定它具体采用哪种
**distillation objective**。

### 5.1 普通蒸馏的问题：训练状态不是 student 自己产生的

普通 SFT / KD 可以抽象为：

```text
dataset 或 teacher 提供 prefix/state
        ↓
teacher 给出目标
        ↓
student 在这个 state 上学习
```

形式上，训练状态可能来自数据分布或 teacher 的状态分布：

$$
s\sim d_{\mathrm{data}}
\quad\text{或}\quad
s\sim d_{\pi_T}.
$$

student 学习在这些状态上逼近 teacher：

$$
\pi_\theta(\cdot\mid s)
\approx
\pi_T(\cdot\mid s).
$$

但自回归推理时，student 的状态不是 teacher 产生的。student 会自己采样下一个 token：

$$
y_t\sim\pi_\theta(\cdot\mid s_t),
$$

然后把它追加到 prefix，得到下一个状态：

$$
s_{t+1}=\operatorname{append}(s_t,y_t).
$$

随着生成继续，student 实际访问的是自己的状态分布：

$$
s_t\sim d_{\pi_\theta}.
$$

这就是 exposure bias 的核心：

```text
训练时：dataset / teacher 的 prefix
推理时：student 自己生成的 prefix
```

所以仅在 teacher 的轨迹上做得很好，并不保证 student 在自己走偏之后仍然知道下一步
该怎么做。

### 5.2 OPD 的第一步：让 student 自己决定训练状态

OPD 首先改变的不是 loss，而是训练 state 的来源。本轮使用当前 rollout policy
$\pi_{\mathrm{old}}$ 让 student 自己生成：

$$
x\sim\mathcal D,
\qquad
y\sim\pi_{\mathrm{old}}(\cdot\mid x).
$$

由此得到 student 实际走过的 prefix：

$$
s_t=(x,y_{<t}),
\qquad
s_t\sim d_{\pi_{\mathrm{old}}}.
$$

例如 student 生成：

```text
prompt:
求解 2x + 3 = 7

student response:
First, subtract 3 from both sides...
```

那么训练状态依次是：

```text
s1 = prompt
s2 = prompt + "First"
s3 = prompt + "First,"
s4 = prompt + "First, subtract"
...
```

这些 state 全部来自 student 自己实际生成的 trajectory。因此：

$$
s_t\sim d_{\pi_{\mathrm{old}}}
$$

就是 OPD 中 “On-Policy” 的来源。它表达的是：

```text
student 决定自己走到哪里；
teacher 只负责告诉它：
“既然你已经走到这里，下一步应该怎样做。”
```

### 5.3 OPD 的第二步：teacher 给 student state 标注目标

student rollout 得到：

$$
s_t=(x,y_{<t}).
$$

teacher 不需要重新生成一条完整答案，而是在这条 student prefix 上计算 next-token
distribution：

$$
\pi_T(\cdot\mid s_t).
$$

于是每个 student state 上都有两套分布：

$$
\pi_{\mathrm{old}}(\cdot\mid s_t)
\quad\text{和}\quad
\pi_T(\cdot\mid s_t),
$$

训练时当前 student $\pi_\theta$ 重新在同一个固定 prefix 上 forward，并学习：

$$
\pi_\theta(\cdot\mid s_t)
\rightarrow
\pi_T(\cdot\mid s_t).
$$

这里要区分两个角色：

- student 的 rollout token 和 prefix 决定训练数据，在本轮更新中固定；
- teacher 是 frozen 的，只提供监督；
- 当前 student 的 logits/log-probability 参与反向传播并更新参数。

因此，OPD 的主干是：

```text
student rollout
      ↓
student 实际访问的 states
      ↓
teacher 在这些 states 上给 next-token target
      ↓
current student 匹配 teacher
```

### 5.4 OPD 的核心优化目标

概念上，可以把 OPD 写成：

$$
\mathcal L_{\mathrm{OPD}}(\theta)
=
\mathbb E_{
\substack{x\sim\mathcal D,\\
y\sim\pi_\theta(\cdot\mid x)}
}
\left[
\frac{1}{T}
\sum_{t=0}^{T-1}
D\left(
\pi_T(\cdot\mid s_t),
\pi_\theta(\cdot\mid s_t)
\right)
\right],
\qquad
s_t=(x,y_{<t}).
$$

这个目标包含两个不同层次：

1. **状态分布层**：$y\sim\pi_\theta$，决定训练状态来自 student 的 rollout，体现
   on-policy 性质。
2. **局部匹配层**：$D(\pi_T,\pi_\theta)$，决定在这些 state 上如何让 student
   靠近 teacher，体现 distillation 性质。

所以可以把 OPD 记成：

```text
OPD
= student 决定去哪里
+ teacher 告诉 student 到那里以后应该怎么做
```

不过，上式是理想化目标。实际训练通常不会对离散生成链路
`token sample → next state → token sample` 整体直接反向传播，而是使用旧 student
采集固定 rollout，再在这些 state 上优化当前 student；这一点在第 5.8 节展开。

### 5.5 Forward KL、Reverse KL、Top-K KL 只是“怎么教”的不同选择

拿到 student state $s_t$ 后，才需要决定在这个 state 上采用什么分布匹配方式。它们共享
同一条 OPD 主干，区别只是 teacher 以什么形式给 student 提供指导。

#### 方法 A：Forward KL

$$
D_{\mathrm{KL}}
\left(
\pi_T\|\pi_\theta
\right).
$$

权重来自 teacher，直觉是：

```text
teacher 认为重要的 token，student 尽量覆盖。
```

这对应后面的 `sample rollout + Full-KL OPD`，可以利用完整 vocabulary distribution
提供较稳定的 dense signal。

#### 方法 B：Reverse KL

$$
D_{\mathrm{KL}}
\left(
\pi_\theta\|\pi_T
\right)
=
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}
\left[
\log\pi_\theta(a\mid s)-\log\pi_T(a\mid s)
\right].
$$

权重来自 student，直觉是：

```text
student 自己想选的 token，teacher 是否认可。
```

可以完整计算整个 vocabulary 的 reverse KL，也可以从 student 分布中采样 token，对
上述期望做 Monte Carlo 估计，这就得到 `sampled-token reverse KL`。

#### 方法 C：Top-K / Top-P KL

只在 teacher 或 student 的局部 support 上做分布匹配，处在：

```text
Full KL
   ↕
Top-K / Top-P KL
   ↕
单个 sampled token
```

之间，是计算量与估计稳定性的折中。因此，Full-KL、sampled-token 和 Top-K/Top-P KL
不是三个互不相关的算法，而是同一主干下不同的 supervision granularity。

### 5.6 一个完整的 OPD iteration

一次实际迭代可以写成如下闭环。

#### Step 1：固定本轮 rollout policy

记本轮开始时的 student 为：

$$
\pi_{\mathrm{old}}.
$$

本轮先用它采集数据；在后续优化 epoch 中，rollout 得到的 token、prefix 和
`old_student_logp` 都视为固定量。

#### Step 2：student rollout

对 prompt：

$$
x\sim\mathcal D
$$

student 生成：

$$
y\sim\pi_{\mathrm{old}}(\cdot\mid x),
$$

并记录：

```text
prompt
student response
每个 response token 对应的 prefix/state
response loss mask
必要时记录 old student chosen-token logprob
```

#### Step 3：teacher evaluate

对每一个：

$$
s_t=(x,y_{<t}),
$$

teacher 计算：

$$
\pi_T(\cdot\mid s_t),
$$

或只计算实际 variant 需要的信息：

```text
teacher logits / full log-probability
teacher top-k 或 top-p support
teacher chosen-token logprob
```

teacher 不改变这条 trajectory，只是在 student 已经访问到的 state 上提供监督。

#### Step 4：构造 distillation loss

选择具体的 OPD variant：

```text
Full Forward KL
Full Reverse KL
Top-K / Top-P KL
Sampled-token Reverse KL
```

它们都在优化同一个局部目标：

$$
\pi_\theta(\cdot\mid s_t)
\rightarrow
\pi_T(\cdot\mid s_t).
$$

#### Step 5：更新 current student

轨迹本身作为已采集的训练数据，不需要对 token sampling 操作反向传播；teacher 也保持
frozen。真正更新的是当前 student：

$$
\theta
\leftarrow
\theta-\eta\nabla_\theta\mathcal L.
$$

因此：

```text
rollout / sampled tokens：固定
teacher：固定
current student parameters：更新
```

#### Step 6：使用新 student 重新 rollout

更新后：

$$
\pi_{\theta_{\mathrm{new}}}
\neq
\pi_{\mathrm{old}},
$$

它未来访问的 state distribution 也会变化：

$$
d_{\pi_{\theta_{\mathrm{new}}}}
\neq
d_{\pi_{\mathrm{old}}}.
$$

所以不能把同一批旧 trajectory 永久当作 on-policy 数据。下一轮应重新 rollout：

```text
new student
     ↓
new rollout
     ↓
new student states
     ↓
teacher 再指导
     ↓
student 再更新
```

完整闭环是：

```text
current student
      ↓
student rollout
      ↓
student prefix / state
      ↓
teacher evaluation
      ↓
distillation loss
      ↓
update student ───────────┐
      ↑                   │
      └───────────────────┘
```

### 5.7 为什么 On-Policy 不一定意味着 Policy Gradient

这是 OPD 中最容易混淆的概念之一。

**On-policy 首先描述数据或 state 从哪里来，而不是 loss 长什么样。**

OPD 的 state 来自：

$$
s_t\sim d_{\pi_{\mathrm{old}}},
$$

所以它是 on-policy distillation。但在拿到这些固定 state 后，可以直接计算：

$$
D_{\mathrm{KL}}
\left(
\pi_T(\cdot\mid s_t)
\|\pi_\theta(\cdot\mid s_t)
\right)
$$

然后用普通 backprop 更新 student。

因此：

```text
On-policy
    = 训练 state 来自当前 student 的 rollout

Policy Gradient
    = 如何估计涉及 policy sampling 的目标梯度
```

Full-KL OPD 的数据可以是 on-policy，但它在固定 prefix 上做的是普通分布匹配，不要求
使用 PPO-style policy gradient。Sampled-token Reverse KL 则可以进一步写成 PG-style
estimator，所以它看起来更像 RL/PPO；但：

> **OPD 本身不等于 Policy Gradient。**

### 5.8 理想目标和实际训练循环的区别

理想化地写，生成 trajectory 的策略和正在优化的策略都是 $\pi_\theta$：

$$
\mathcal L_{\mathrm{OPD}}(\theta)
=
\mathbb E_{y\sim\pi_\theta}
\left[
\text{distillation loss}
\right].
$$

但实际训练通常是交替进行的：

```text
① 用 πold rollout
② 固定 rollout 得到的 trajectory 和 prefix
③ 在这些 state 上训练 current πθ
④ 更新 student
⑤ 用新的 student 再 rollout
```

所以在一轮更新内部，更准确的目标是：

$$
s_t\sim d_{\pi_{\mathrm{old}}},
$$

$$
\mathcal L_{\mathrm{iteration}}(\theta)
=
\mathbb E_{s_t\sim d_{\pi_{\mathrm{old}}}}
\left[
D\left(
\pi_T(\cdot\mid s_t),
\pi_\theta(\cdot\mid s_t)
\right)
\right].
$$

这一轮里 $d_{\pi_{\mathrm{old}}}$ 被当成固定的数据分布。student 更新之后，下一轮
再采集新的 $d_{\pi_{\mathrm{new}}}$。因此 OPD 的“on-policy”不是：

```text
对 rollout token 一路穿过离散 sample 操作反向传播
```

而是：

```text
student 变了
→ rollout 数据也刷新
→ 训练尽量跟着 student 当前的 state distribution 走
```

### 5.9 把整篇 OPD 压缩成一条主线

```text
普通 KD 的 state 来自 teacher / dataset，推理 state 来自 student
        ↓
让 student 自己 rollout
        ↓
得到 student 真正访问的 prefix/state distribution
        ↓
teacher 在这些 state 上给 next-token guidance
        ↓
选择 Forward KL / Reverse KL / Top-K KL / sampled-token objective
        ↓
更新 student
        ↓
student policy 改变，state distribution 也改变
        ↓
重新 rollout，继续训练
```

因此最核心的因果链是：

$$
\boxed{
\text{student rollout}
\rightarrow
\text{student state distribution}
\rightarrow
\text{teacher supervision}
\rightarrow
\text{student update}
\rightarrow
\text{new state distribution}
}
$$

后面的 Full-KL、sampled-token reverse KL、PG-style loss 和 PPO 对比，都是这条主线上的
具体实现或对照，而不是 OPD 定义本身。

---

## 6. Full-KL OPD

Full-KL OPD 的做法是：

```text
1. student 生成 response。
2. 得到 input_ids = prompt + student_response。
3. teacher 和 student 都在这条 input_ids 上 forward。
4. 对每个 response token 位置，比较完整 vocab distribution。
```

Forward KL loss：

$$
D_{KL}(\pi_T(\cdot|s_t) \| \pi_\theta(\cdot|s_t))
= \sum_{v \in V}
\pi_T(v|s_t)
\log
\frac{\pi_T(v|s_t)}{\pi_\theta(v|s_t)}
$$

伪代码：

```python
# student rollout
with torch.no_grad():
    response_ids = student.generate(
        prompt_ids,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=1024,
    )

input_ids = concat_prompt_response(prompt_ids, response_ids)
response_mask = build_response_mask(input_ids, prompt_ids)

# teacher forward
with torch.no_grad():
    teacher_logits = teacher(input_ids).logits
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

# student forward
student_logits = student(input_ids).logits
student_log_probs = torch.log_softmax(student_logits, dim=-1)

# shift
teacher_log_probs = teacher_log_probs[:, :-1, :]
student_log_probs = student_log_probs[:, :-1, :]
loss_mask = response_mask[:, 1:].float()

# forward KL: KL(teacher || student)
teacher_probs = teacher_log_probs.exp()
kl_per_pos = torch.sum(
    teacher_probs * (teacher_log_probs - student_log_probs),
    dim=-1,
)

loss = (kl_per_pos * loss_mask).sum() / loss_mask.sum()
loss.backward()
optimizer.step()
```

特点：

```text
优点：信号更稳定，因为每个位置都比较完整分布。
缺点：成本高，因为需要 [B, T, V] 的完整 vocab KL。
```

---

## 7. Sampled-token Reverse KL

Reverse KL 是：

$$
D_{KL}(\pi_\theta \| \pi_T)
= \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}
\left[
\log \pi_\theta(a|s) - \log \pi_T(a|s)
\right]
$$

因为它是对 student 分布取期望，所以可以从 student 里 sample token 来估计。

流程：

```text
1. student 在当前 prefix s 上 sample 一个 token a。
2. teacher 计算 log πT(a | s)。
3. student 计算 log πθ(a | s)。
4. 根据 teacher 是否认可这个 token 来更新 student。
```

直觉：

```text
student 自己选 token；
teacher 评价这个 token；
teacher 认可 -> 提高这个 token 概率；
teacher 不认可 -> 降低这个 token 概率。
```

---

## 8. 为什么 sample，而不是取 max prob？

Reverse KL 的数学形式本身是：

$$
a \sim \pi_\theta(\cdot|s)
$$

也就是从 student 分布里采样。

如果用 argmax：

```python
a = argmax(student_probs)
```

那只会训练 student 当前最自信的 token，而不是估计整个 student 分布下的期望。

问题包括：

```text
1. 只看 top-1 token，覆盖不到其他可能 token。
2. 探索少，容易过早 collapse 到某一种模式。
3. 不是真正的 reverse KL 期望估计。
```

例子：

```text
student:
A: 0.40
B: 0.35
C: 0.25

teacher:
A: 0.10
B: 0.80
C: 0.10
```

如果 argmax，student 永远选 A，只知道 A 不好。

如果 sample，有 35% 概率采到 B，teacher 会给 B 高 reward，student 就能更快提高 B 的概率。

当然，OPD 中也不能毫无限制地 sample，否则 student 可能采到坏 prefix，teacher 被带偏。

实践中常用：

```text
top-p sampling
top-k sampling
较低 temperature
max_new_tokens 限制
bad rollout 过滤
```

---

## 9. 正常 LLM 生成时也会 sample 吗？

会。很多 LLM 生成回答时确实会 sample，但通常不是从完整词表毫无限制地采样，而是加约束。

常见参数：

```python
do_sample=True
temperature=0.7
top_p=0.9
top_k=50
```

### top-k

只在概率最高的 k 个 token 中采样。

### top-p

只保留累计概率达到 p 的 token，再重新归一化采样。

### temperature

控制分布尖锐程度：

```text
低 temperature：更稳定、更保守。
高 temperature：更多样，但更容易发散。
```

所以采样确实可能带来坏 token，但通过 top-k/top-p/temperature 可以大幅降低风险。

---

## 10. Sampled-token Reverse KL 的 PG-style loss

从 reverse KL 的单样本形式看：

$$
\log \pi_\theta(a|s) - \log \pi_T(a|s)
$$

但如果直接把它当普通 loss 反传，会有问题。

因为 teacher logprob 是常数，直接最小化：

```python
student_logp - teacher_logp.detach()
```

会倾向于降低 student 对 sampled token 的 logprob，不管 teacher 是否喜欢它。
我们本来希望的是：

```
teacher 喜欢这个 token
→ student 提高它的概率

teacher 不喜欢这个 token
→ student 降低它的概率
```

但是 naive loss 变成了：

```
不管 teacher 喜不喜欢
↓
梯度永远都是 ∇ log πθ(a|s)
↓
gradient descent 永远试图降低 log πθ(a|s)
```

所以实际更合理的 PG-style 写法是把 teacher signal 放进 reward：

$$
r(s,a) = \log \pi_T(a|s) - \log \pi_{old}(a|s)
$$

然后：

$$
\mathcal{L} = -r(s,a) \log \pi_\theta(a|s)
$$

伪代码：

```python
# 1. rollout，用 old student 采样
with torch.no_grad():
    response_ids, old_student_chosen_logp = student.generate(
        prompt_ids,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_logprobs=True,
    )

input_ids = concat_prompt_response(prompt_ids, response_ids)
labels = input_ids[:, 1:]
loss_mask = build_response_loss_mask(input_ids, prompt_ids)[:, 1:]

# 2. teacher 给 sampled tokens 打分
with torch.no_grad():
    teacher_logits = teacher(input_ids).logits[:, :-1, :]
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

    teacher_chosen_logp = torch.gather(
        teacher_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

# 3. current student 重新 forward
student_logits = student(input_ids).logits[:, :-1, :]
student_log_probs = torch.log_softmax(student_logits, dim=-1)

current_student_chosen_logp = torch.gather(
    student_log_probs,
    dim=-1,
    index=labels.unsqueeze(-1),
).squeeze(-1)

# 4. reward 不回传梯度
with torch.no_grad():
    reward = teacher_chosen_logp - old_student_chosen_logp

# 5. PG-style OPD loss
loss_per_token = -reward.detach() * current_student_chosen_logp
loss = (loss_per_token * loss_mask).sum() / loss_mask.sum()

loss.backward()
optimizer.step()
```

---

## 11. 哪些量回传梯度？

在 PG-style sampled reverse KL 里，有三个 logprob：

```text
teacher_chosen_logp
old_student_chosen_logp
current_student_chosen_logp
```

### teacher_chosen_logp

teacher 对 student sampled token 的 logprob。

```text
不回传梯度。
```

原因：teacher 是 frozen teacher。

---

### old_student_chosen_logp

rollout 时 old student 对 sampled token 的 logprob。

```text
不回传梯度。
```

原因：它表示当初这个 token 是如何被旧策略 sample 出来的，类似 PPO 中的 old_logprob。

---

### current_student_chosen_logp

当前正在优化的 student 对同一个 token 的 logprob。

```text
回传梯度。
```

原因：我们要更新 student。

总结：

```text
teacher_chosen_logp: no grad
old_student_chosen_logp: no grad
reward: no grad
current_student_chosen_logp: has grad
```

---

## 12. loss = -reward * logp 的符号和梯度方向

loss 是：

$$
\mathcal{L} = -r \log \pi_\theta(a|s)
$$

其中：

```text
log πθ(a|s) <= 0
reward r 可以是正，也可以是负。
```

所以 loss 不一定永远是负的。

### reward > 0

说明 teacher 比 student 更认可这个 token。

最小化：

$$
-r \log \pi_\theta(a|s)
$$

会让：

```text
log πθ(a|s) 数值变大（即更接近 0）
πθ(a|s) 变大
```

也就是提高该 token 概率。

### reward < 0

说明 teacher 不认可 student sampled token。

最小化 loss 会让：

```text
log πθ(a|s) 变小
πθ(a|s) 变小
```

也就是降低该 token 概率。

所以这个 loss 的直觉是：

```text
teacher 认可 -> 提高概率；
teacher 不认可 -> 降低概率。
```

---

## 13. OPD 和 PPO 的关系

OPD 看起来像 PPO，是因为它也有：

```text
old policy rollout
current policy update
old_logprob / current_logprob
```

但 OPD 不是标准 PPO。

PPO：

```text
1. old policy rollout
2. reward model / environment 给 reward
3. 算 advantage
4. 用 ratio + clip objective 更新
```

OPD sampled reverse KL：

```text
1. student rollout
2. teacher 给 token-level logprob
3. 构造 token-level reward
4. 用 -reward * current_logprob 更新
```

PPO 有：

$$
ratio = \frac{\pi_{new}(a|s)}{\pi_{old}(a|s)}
$$

并使用 clipped objective。

OPD 通常没有 PPO 的 clipped ratio，更多是：

```text
on-policy data collection + teacher distillation。
```

---

## 14. Teacher 被 student prefix 带偏怎么办？

OPD 的风险是：

```text
teacher condition on student prefix。
```

如果 student 前缀很差，teacher 被迫在一个不自然、错误、混乱的上下文里预测 next token。

例如：

```text
题目：2x + 3 = 7

teacher 原本会写：
2x = 4, x = 2

student prefix：
We assume x = 100 because ...
```

teacher 看到这个 prefix 后，可能开始纠错、犹豫或者被带偏。

常见解决办法：

### 14.1 SFT warmup student

先让 student 通过 SFT / offline KD 学会基本格式和基本能力，再做 OPD。

```text
阶段 1：SFT / KD
阶段 2：OPD
```

---

### 14.2 限制 rollout 采样

使用：

```text
top-p sampling
low temperature
max_new_tokens
bad response filtering
```

避免 student 生成太离谱的 prefix。

---

### 14.3 Top-K / Top-P local support matching

不比较完整 vocab，也不只看 sampled token，而是在 teacher 的 top-k/top-p token 集合中做局部分布匹配。

这样比 sampled-token 更稳定，比 full KL 更省。

---

### 14.4 Drift detection

检测 teacher 和 student 的 top-k token 是否重叠。

如果重叠高：

```text
说明 prefix 还在 teacher 可指导区域。
```

如果重叠低：

```text
说明 student prefix 已经 drift，teacher signal 可能不可靠。
```

可以降低后续 loss 权重，或者直接截断。

---

### 14.5 Rollout truncation

如果后半段 prefix 已经明显坏掉，不要继续在后面训练。

```text
前 200 token 还可靠；
后面 1000 token 已经 drift；
那就只训练前 200 token。
```

---

### 14.6 混合 teacher forcing 和 student forcing

训练早期可以多用 teacher prefix，后期逐步增加 student prefix。

类似 scheduled sampling / DAgger。

---

## 15. Teacher 和 student 生成长度不一致怎么办？

OPD 通常不需要 teacher 生成完整答案。

训练长度由 student rollout 决定。

student 生成：

```text
y_student = [y1, y2, ..., yT]
```

teacher 只需要在每个 prefix 上预测 next token：

```text
s_t = prompt + y_student_<t
πT(. | s_t)
```

所以不存在必须让 teacher response 和 student response 对齐的问题。

### student 太短

如果 student 很早 EOS，训练信号会很短。

解决方法：

```text
min_new_tokens
EOS penalty
过滤过短 response
混合 SFT traces
```

### student 太长

teacher 仍然可以继续给 next-token distribution，但后面 prefix 可能越来越 drift。

解决方法：

```text
max_new_tokens
length penalty
drift detection
loss down-weight
rollout truncation
```

---

## 16. OPD 的几种常见形式

### 16.1 sample rollout + full KL

```text
student sample response；
teacher/student 在 student prefix 上比较完整 vocab distribution。
```

优点：稳定。

缺点：贵。

---

### 16.2 sample rollout + sampled-token reverse KL

```text
student sample response；
teacher 只评价 student 实际生成的 token；
用 PG-style loss 更新。
```

优点：省。

缺点：噪声大，不稳定。

---

### 16.3 sample rollout + top-k/top-p KL

```text
student sample response；
teacher 提供 top-k/top-p local support；
student 在局部 token 集合上匹配 teacher。
```

优点：比 full KL 省，比 sampled-token 稳。

这是实践上比较有吸引力的折中。

---

### 16.4 greedy rollout + full KL

```text
student greedy decode；
teacher/student 在 greedy prefix 上做 KL。
```

优点：稳定。

缺点：探索少，不是真正的 stochastic on-policy distribution。

---

## 17. OPD、SFT、KD、PPO 的对比

| 方法 | rollout 来自谁 | 监督来自谁 | 信号密度 | 主要问题 |
|---|---|---|---|---|
| SFT | dataset | label response | token-level | off-policy / exposure bias |
| KD | teacher | teacher logits / response | token-level | student prefix 不充分 |
| PPO/RLVR | student | reward / verifier | sparse 或 sequence-level | reward 稀疏、方差大 |
| OPD | student | teacher logits / logprobs | token-level | teacher 可能被 student prefix 带偏 |

---

## 18. 最核心的理解

OPD 不是：

```text
teacher 写答案，student 模仿答案。
```

而是：

```text
student 自己尝试；
teacher 看 student 走到了哪里；
teacher 在这个位置告诉 student 下一步应该更像什么；
student 在自己真实会访问的状态分布上变强。
```

一句话总结：

> OPD = on-policy rollout + teacher token-level distillation。

它的优点：

```text
减少 train-test mismatch；
提供 dense teacher signal；
能在 student 自己犯错后的状态上训练。
```

它的风险：

```text
student prefix 太差会带偏 teacher；
sampled-token loss 噪声大；
长链任务 teacher forward 成本高；
teacher-student reasoning pattern 不兼容时可能失败。
```

实践上较稳的 recipe：

```text
1. SFT / KD warmup student。
2. 用受限采样生成 student rollout。
3. 使用 full KL 或 top-k/top-p local KL，而不是一上来只用 sampled-token。
4. 检测 prefix drift，必要时截断或降权。
5. 对过短、重复、明显错误 response 做过滤。
```

---

## 19. 复习用超短版

```text
KL：
真实分布 P，用 Q 解释时多付出的平均 surprise。

Forward KL：
KL(teacher || student)，teacher 认为重要的 token，student 要覆盖。

Reverse KL：
KL(student || teacher)，student 自己想选的 token，teacher 是否认可。

OPD：
student 先 rollout，teacher 在 student prefix 上给 next-token 指导。

Full-KL OPD：
每个位置比较完整 vocab distribution。

Sampled-token OPD：
student sample token，teacher 给这个 token logprob，构造 reward 更新 student。

PG-style loss：
reward = teacher_logp - old_student_logp
loss = -reward.detach() * current_student_logp

梯度：
teacher_logp 不回传；
old_student_logp 不回传；
reward 不回传；
current_student_logp 回传。

teacher 被带偏：
用 SFT warmup、受限采样、top-k KL、drift detection、rollout truncation 缓解。
```

## 相关笔记

- [[PPO_逻辑重构版|PPO]]：policy-gradient style loss 的基础。
- [[SAC_PPO_compare|SAC vs PPO]]：on-policy / off-policy 策略优化对比。
- [[Pi_star0.6论文问题解答|pi*0.6 / RECAP]]：reward/value/advantage 信号进入 VLA 的相关机器人案例。
