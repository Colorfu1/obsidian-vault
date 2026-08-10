---
title: PPO：从策略梯度到裁剪目标
type: concept_note
topic: reinforcement_learning
status: draft
importance: high
updated: 2026-08-10
tags:
  - ppo
  - reinforcement-learning
  - actor-critic
  - policy-gradient
  - gae
---

# PPO：从策略梯度到裁剪目标

这篇笔记沿着一条主线解释 PPO：

> 环境和 reward 通常不可导，为什么只凭采样到的动作与回报，仍然可以更新一个完整的策略分布？

理解这条主线后，Policy Gradient、log probability、Advantage、Critic、GAE、importance
ratio 和 clipped objective 就不再是彼此独立的公式，而是为了解决同一个问题依次引入的
组件。

本文将原始 PPO 讨论中的关键问题与 Policy Gradient 推导合并为一次完整推导，重点解释
环境不可导时，策略如何通过采样、log probability 和 Advantage 获得更新信号。

## 1. PPO 的定位与完整流程

PPO（Proximal Policy Optimization）是一种基于 Policy Gradient 的
Actor-Critic 算法。按强化学习的标准分类，它属于 **on-policy** 方法：训练数据由当前
策略收集，只在策略仍与采样策略足够接近时进行有限次数复用。因为一批 rollout 会被
优化多个 epoch，所以也常被非正式地描述为 “near on-policy”。

PPO 同时训练两个网络：

- **Actor** 表示策略 $\pi_\theta(a\mid s)$，决定在状态 $s$ 下如何选择动作。
- **Critic** 估计状态价值 $V_\phi(s)$，判断从状态 $s$ 出发，按照当前策略继续行动，
  预期能获得多少未来回报。

一次训练迭代可以概括为：

```text
Actor 输出动作分布
→ 从分布中采样动作
→ env.step(action)
→ 收集 state、action、reward、done、old_log_prob、value
→ 用 Critic 对 rollout 末端做 bootstrap
→ 用 reward、value、done 计算 GAE Advantage 与 return
→ 用 Advantage 更新 Actor
→ 用 return 更新 Critic
→ 用 ratio 和 clip 限制策略更新带来的收益
→ 丢弃当前 rollout，使用新策略重新采样
```

这里有三条彼此不同的学习信号：

| 对象 | 学习信号 | 回答的问题 |
|---|---|---|
| Actor | Advantage | 这个已执行动作应该更常出现，还是更少出现？ |
| Critic | Return target | 这个状态的长期价值应该是多少？ |
| PPO clip | 新旧策略概率比 | 对同一批旧数据，策略是否已经改得过多？ |

## 2. 为什么不能直接对 reward 反向传播

PPO 要优化 Actor 参数 $\theta$。如果像监督学习一样写：

```python
loss = -reward
loss.backward()
```

那么计算图中必须存在下面这条可导路径：

```text
Actor 参数 θ
→ 动作 a
→ 环境转移
→ reward
```

对应的链式法则是：

$$
\frac{\partial r}{\partial \theta}
=
\frac{\partial r}{\partial a}
\frac{\partial a}{\partial \theta}.
$$

但典型强化学习环境中，这条路径会断开：

1. 离散动作通常由 `Categorical.sample()` 采样，采样结果不是普通的可导函数。
2. `env.step(action)` 可能包含物理仿真、碰撞检测、离散规则或真实机器人，不在
   PyTorch 计算图中。
3. reward 往往只是环境返回的一个数，并没有连接到 Actor 参数。

连续动作能够使用重参数化采样，也不代表环境就变得可导。只要
`action → env.step → reward` 这段仍在计算图之外，就不能沿动作路径直接把 reward
反传给 Actor。

### 2.1 与监督学习的关键区别

监督学习和强化学习都在优化一个期望，但它们的采样分布与模型参数之间存在关键差异。

监督学习通常写为：

$$
L(\theta)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
\left[
\ell_\theta(x,y)
\right].
$$

数据分布 $p_{\mathrm{data}}$ 通常不依赖模型参数 $\theta$。改变模型不会改变训练集中
哪些 $(x,y)$ 被采样，因此可以把梯度直接作用在样本 loss 上：

$$
\nabla_\theta L(\theta)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
\left[
\nabla_\theta\ell_\theta(x,y)
\right].
$$

强化学习的目标则是：

$$
J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta(\tau)}
\left[
R(\tau)
\right].
$$

这里轨迹分布 $p_\theta(\tau)$ 本身依赖 $\theta$：策略改变会改变动作分布，动作改变会
改变后续状态、reward 和整条轨迹。因此即使 $R(\tau)$ 不能直接对 $\theta$ 求导，
$J(\theta)$ 仍会通过 **轨迹出现概率的变化** 而改变：

$$
\nabla_\theta J(\theta)
=
\int
\nabla_\theta p_\theta(\tau)
R(\tau)\,d\tau.
$$

这就是 RL 不能简单照搬普通监督学习反向传播、而需要 Policy Gradient 的更深层原因：
监督学习通常对固定数据分布下的可导 loss 求导；RL 则需要对由策略参数决定的采样分布
求导。

Policy Gradient 绕开了这条不可导路径。它不计算
$\partial r/\partial a$，而是计算：

$$
\nabla_\theta \log \pi_\theta(a_t\mid s_t)\,\hat A_t.
$$

其中：

- $\log \pi_\theta(a_t\mid s_t)$ 在 Actor 的计算图中，可以对 $\theta$ 求导；
- 已采样的动作 $a_t$ 只用于选出对应的 log probability，不需要对动作本身求导；
- $\hat A_t$ 是这次动作好坏的评分，在 Actor 更新时作为常数权重。

直觉上，这个更新表达的是：

```text
结果比预期好  → 提高这次动作在该状态下的概率
结果比预期差  → 降低这次动作在该状态下的概率
```

下面从期望回报目标推导出这个形式。

## 3. 从期望回报到 Policy Gradient

### 3.1 为什么目标是期望回报

一条轨迹记为：

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots).
$$

轨迹的折扣回报可以写为：

$$
R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t.
$$

策略不是直接输出“最优轨迹”，而是定义了轨迹出现的概率分布
$p_\theta(\tau)$。因此 Actor 的目标是：

$$
J(\theta)
=
\mathbb{E}_{\tau\sim p_\theta(\tau)}[R(\tau)].
$$

它表示：

> 如果反复按照当前策略与环境交互，平均能获得多少回报？

为什么不是直接求当前状态下 reward 最高的动作？

- 训练时通常只执行了一个动作，不知道其他未执行动作的反事实结果。
- 动作影响后续状态，当前 reward 最高的动作不一定带来最高长期回报。
- 策略需要在未知环境中探索，不能一开始就把当前偶然表现最好的动作当作最优动作。
- 连续动作空间无法枚举所有动作再逐一比较。

所以强化学习优化的是整个策略诱导出的 **长期回报期望**，而不是一个已知 reward
函数上的单步 `argmax`。

### 3.2 Monte Carlo 如何估计期望

“把目标写成积分”并不等于已经得到了这个积分的解析解。比如
$\int_0^1x^2\,dx$ 可以直接算出结果，但强化学习中的积分变量是一整条轨迹：

$$
\tau=(s_0,a_0,s_1,a_1,\ldots,s_T).
$$

因此：

$$
\int p_\theta(\tau)R(\tau)\,d\tau
$$

本质上是在高维、通常无法枚举的轨迹空间上积分或求和。如果这个积分确实存在可计算的
解析解，当然不需要 Monte Carlo；Monte Carlo 是在解析计算不可行时，用样本近似积分的
方法。

这里还要区分“知道当前参数”和“知道整个积分”。当前优化步中 $\theta$ 确实是已知的，
但它主要让我们能够计算策略在某个已访问状态下的动作概率：

$$
\pi_\theta(a_t\mid s_t).
$$

它并不意味着我们已经知道所有可能状态、动作和后续转移组成的完整轨迹分布，也不意味着
高维积分可以解析求出。更准确地说，$p_\theta(\tau)$ 与 $R(\tau)$ 都可以写成轨迹的
函数，例如：

$$
R(\tau)=\sum_{t=0}^{T-1}\gamma^t r(s_t,a_t),
$$

$$
p_\theta(\tau)
=\rho_0(s_0)\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t),
$$

但在 model-free RL 中，环境转移 $P(s'\mid s,a)$ 通常未知；即使形式上能写出这些
函数，也通常没有能直接求解整个高维积分的闭式表达式。

一般地，如果：

$$
x\sim p(x),
$$

那么：

$$
\mathbb{E}_{x\sim p}[f(x)]
=
\int p(x)f(x)\,dx
\approx
\frac{1}{N}\sum_{i=1}^{N}f(x_i),
\qquad x_i\sim p.
$$

这就是 Monte Carlo 估计。对应到强化学习：

```text
从当前策略运行一次 rollout  = 从 pθ(τ) 采样一条轨迹
运行很多次 rollout           = 获得 τ₁, τ₂, ..., τₙ
平均这些轨迹的回报            = 估计当前策略的期望回报
```

一次采样不能给出精确期望，只能给出一个随机样本。随着轨迹或时间步样本增多，样本平均
才逐渐接近期望。PPO 的 rollout buffer 就是在收集用于这种估计的一批样本。

这里 Monte Carlo 首先估计的是目标 $J(\theta)$；但训练真正需要的是它对参数的梯度：

$$
\nabla_\theta J(\theta),
$$

也就是参数应该朝哪个方向变化，才能提高期望回报。

### 3.3 为什么原始梯度形式不能直接用当前策略采样

把期望写成积分并求导：

$$
J(\theta)
=
\int p_\theta(\tau)R(\tau)\,d\tau,
$$

$$
\nabla_\theta J(\theta)
=
\int \nabla_\theta p_\theta(\tau)R(\tau)\,d\tau.
$$

这里不能直接从 $p_\theta(\tau)$ 采样轨迹，然后平均
$\nabla_\theta p_\theta(\tau)R(\tau)$。

Monte Carlo 从 $p_\theta$ 采样时，能够直接估计的形式必须是：

$$
\int p_\theta(\tau)f(\tau)\,d\tau
=
\mathbb{E}_{\tau\sim p_\theta}[f(\tau)].
$$

如果从 $p_\theta$ 采样后直接平均
$\nabla_\theta p_\theta(\tau)R(\tau)$，其期望实际是：

$$
\int
p_\theta(\tau)
\nabla_\theta p_\theta(\tau)
R(\tau)\,d\tau.
$$

与目标相比，它额外多乘了一个 $p_\theta(\tau)$，估计的不是同一个量。

#### 一个两轨迹例子

假设轨迹空间中只有 $\tau_1$ 和 $\tau_2$：

$$
p_\theta(\tau_1)=0.8,
\qquad
p_\theta(\tau_2)=0.2.
$$

目标梯度是：

$$
\nabla_\theta J
=
\nabla_\theta p_\theta(\tau_1)R(\tau_1)
+
\nabla_\theta p_\theta(\tau_2)R(\tau_2).
$$

如果从 $p_\theta$ 采样后直接平均
$\nabla_\theta p_\theta(\tau)R(\tau)$，其期望却是：

$$
0.8\,
\nabla_\theta p_\theta(\tau_1)R(\tau_1)
+
0.2\,
\nabla_\theta p_\theta(\tau_2)R(\tau_2).
$$

这两个式子显然不同：直接采样平均又乘了一次轨迹的采样概率。

因此需要把：

$$
\int \nabla_\theta p_\theta(\tau)R(\tau)\,d\tau
$$

改写成：

$$
\int p_\theta(\tau)f(\tau)\,d\tau.
$$

log-derivative trick 正是完成这一步的工具。

更一般地，如果只能从 $p(\tau)$ 采样，而想计算：

$$
\int g(\tau)\,d\tau,
$$

可以在 $p(\tau)>0$ 的区域改写为：

$$
\begin{aligned}
\int g(\tau)\,d\tau
&=
\int
p(\tau)
\frac{g(\tau)}{p(\tau)}
\,d\tau \\
&=
\mathbb{E}_{\tau\sim p}
\left[
\frac{g(\tau)}{p(\tau)}
\right].
\end{aligned}
$$

在 Policy Gradient 中取：

$$
g(\tau)
=
\nabla_\theta p_\theta(\tau)R(\tau),
$$

就会自然出现：

$$
\frac{g(\tau)}{p_\theta(\tau)}
=
\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}
R(\tau)
=
\nabla_\theta\log p_\theta(\tau)R(\tau).
$$

这说明 log trick 并不是凭空引入的技巧，而是“只能从 $p_\theta$ 采样”这一条件下进行
期望改写的直接结果。

### 3.4 Log-derivative trick

由：

$$
\nabla_\theta\log p_\theta(x)
=
\frac{\nabla_\theta p_\theta(x)}{p_\theta(x)}
$$

可得：

$$
\nabla_\theta p_\theta(x)
=
p_\theta(x)\nabla_\theta\log p_\theta(x).
$$

将其代回目标梯度：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=
\int \nabla_\theta p_\theta(\tau)R(\tau)\,d\tau \\
&=
\int p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau)
R(\tau)\,d\tau \\
&=
\mathbb{E}_{\tau\sim p_\theta}
\left[
\nabla_\theta\log p_\theta(\tau)R(\tau)
\right].
\end{aligned}
$$

现在它已经是当前轨迹分布下的期望，因此可以用 rollout 样本估计：

$$
\nabla_\theta J(\theta)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
\nabla_\theta\log p_\theta(\tau_i)R(\tau_i),
\qquad
\tau_i\sim p_\theta.
$$

log trick 没有让 reward 或环境变得可导。它做的是另一件事：

> 把“轨迹分布的概率如何随参数变化”改写成“采样轨迹的 log probability 如何随参数
> 变化”，从而得到可由样本估计的梯度。

### 3.5 环境转移概率未知，为什么仍然能计算梯度

轨迹概率可以分解为：

$$
p_\theta(\tau)
=
\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t),
$$

其中：

- $\rho_0(s_0)$ 是初始状态分布；
- $\pi_\theta(a_t\mid s_t)$ 是 Actor 的动作概率；
- $P(s_{t+1}\mid s_t,a_t)$ 是环境转移概率。

取对数后，连乘变为求和：

$$
\log p_\theta(\tau)
=
\log \rho_0(s_0)
+
\sum_t\log\pi_\theta(a_t\mid s_t)
+
\sum_t\log P(s_{t+1}\mid s_t,a_t).
$$

初始状态分布与环境转移不依赖 Actor 参数 $\theta$，所以：

$$
\nabla_\theta\log p_\theta(\tau)
=
\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

这一步解释了为什么标准 model-free Policy Gradient 不需要知道环境转移概率的显式形式：

- $P(s_{t+1}\mid s_t,a_t)$ 当然会影响实际经过哪些状态、得到哪些 reward，但它不依赖
  Actor 参数，因此对 $\theta$ 的 log-derivative 为零；
- Policy Gradient 也不需要先算出完整的 $p_\theta(\tau)$ 或遍历所有可能轨迹，只需在
  已采样的轨迹上计算 Actor 给出的动作 `log_prob`；
- 环境仍然负责产生下一状态和 reward，但不必把 `env.step` 放进 PyTorch 的反向传播图。

于是，3.4 节得到的轨迹级 Monte Carlo 估计可以落到实际可计算的形式：

$$
\nabla_\theta J(\theta)
\approx
\frac{1}{N}\sum_{i=1}^{N}
R(\tau_i)
\sum_t
\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)}),
\qquad
\tau_i\sim p_\theta.
$$

这里 Monte Carlo 估计的是梯度期望，而不是在对环境求导；log-derivative trick 负责把
梯度改写成可由轨迹样本估计的形式。

因此，Policy Gradient 与环境之间的接口只需要是“执行动作并返回下一状态与 reward”；
真正进入 Actor 反向传播图的，是对已采样动作计算出的 `log_prob`。

进一步使用 reward-to-go，并用 Advantage 替代原始 return 后，常见形式为：

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}_t
\left[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat A_t
\right].
$$

### 3.6 为什么 `log_prob * advantage` 可以反向传播

以离散动作为例：

```python
logits = actor(obs)
dist = torch.distributions.Categorical(logits=logits)

action = dist.sample()
log_prob = dist.log_prob(action)

policy_loss = -(log_prob * advantage.detach()).mean()
policy_loss.backward()
```

计算图中的关键关系是：

```text
θ → logits → log_prob → policy_loss
```

`action` 是一个已经采样出的索引。它不需要梯度，只负责告诉
`dist.log_prob(action)` 取分布中的哪一项。`advantage.detach()` 也被当作固定权重，
所以：

$$
\nabla_\theta
\left[
\log\pi_\theta(a_t\mid s_t)\hat A_t
\right]
=
\hat A_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

如果 $\hat A_t>0$，最小化负号后的 loss 会提高该动作的 log probability；如果
$\hat A_t<0$，则会降低它。

### 3.7 为什么用 log probability，而不是普通 probability

根本原因是梯度恒等式：

$$
\nabla p=p\nabla\log p.
$$

它把原本无法直接用当前策略采样估计的梯度，改写成了当前策略分布下的期望。数值稳定
和计算方便是额外收益：

- 轨迹概率是许多小概率的连乘，容易下溢；取对数后变成求和。
- 新旧策略概率比可以稳定地写成：

  $$
  \frac{\pi_\theta(a_t\mid s_t)}
  {\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
  =
  \exp\left(
  \log\pi_\theta(a_t\mid s_t)
  -
  \log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
  \right).
  $$

例如：

```text
π(a | s) = 1e-20
log π(a | s) ≈ -46.05
```

极小概率直接参与长序列连乘时很容易下溢为零，而有限的 log probability 可以继续
稳定地进行求和与求差。

因此，“log 更稳定”是正确的工程解释，但 **log trick 让梯度变成可采样期望** 才是它
在 Policy Gradient 中出现的理论原因。

## 4. Actor、动作分布与探索

Actor 输出的是动作分布，而不是动作价值。

### 4.1 离散动作

假设动作集合是：

```text
0：直行
1：左转
2：右转
3：刹车
```

Actor 输出 logits，并据此构造分类分布：

```python
logits = actor(obs)
dist = torch.distributions.Categorical(logits=logits)

action = dist.sample()
log_prob = dist.log_prob(action)
entropy = dist.entropy()
```

### 4.2 连续动作

机器人控制中，动作可能是多维连续向量。Actor 常输出高斯分布的均值与标准差：

$$
\pi_\theta(a\mid s)
=
\mathcal N\bigl(\mu_\theta(s),\sigma_\theta(s)\bigr).
$$

例如二维动作表示 `[速度, 转向角]`，Actor 可能输出：

```text
mean = [0.60, 0.10]
std  = [0.20, 0.05]
```

从该分布进行两次采样，可能分别得到：

```text
[0.53, 0.13]
[0.71, 0.08]
```

同一状态下动作并不固定为均值；标准差决定采样的分散程度，也影响探索强度。

```python
mean, log_std = actor(obs)
std = log_std.exp()

base_dist = torch.distributions.Normal(mean, std)
dist = torch.distributions.Independent(base_dist, 1)

action = dist.sample()
log_prob = dist.log_prob(action)
```

如果动作有界，实际实现还可能使用 `tanh` 变换或截断，并相应修正 log probability。
训练时通常采样以保持探索；评测时常使用离散分布的 `argmax` 或连续分布的均值，但具体
做法取决于策略分布和环境接口。

### 4.3 为什么 PPO 通常不使用 epsilon-greedy

DQN 常使用 epsilon-greedy：

```text
以 1 - ε 的概率选择 Q 最大的动作
以 ε 的概率随机选择动作
```

PPO 的 Actor 本身就是随机策略：

$$
a\sim\pi_\theta(\cdot\mid s).
$$

从分布中采样已经提供了探索，Entropy bonus 还可以防止策略过早变得过于确定。因此
PPO 通常不额外叠加 epsilon-greedy。

## 5. Advantage、Critic 与 baseline

### 5.1 Return、Value、Q 与 Advantage

从时刻 $t$ 开始的折扣回报是：

$$
G_t
=
\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}.
$$

状态价值定义为：

$$
V^\pi(s)
=
\mathbb{E}_\pi[G_t\mid s_t=s].
$$

动作价值定义为：

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a].
$$

Advantage 是二者之差：

$$
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s).
$$

含义是：

> 在状态 $s$ 下，这个动作相对当前策略通常会做出的动作，究竟好多少或差多少？

因此：

```text
A(s, a) > 0  → 该动作比当前策略在此状态下的平均水平好
A(s, a) < 0  → 该动作比平均水平差
```

Advantage 的正负决定“应该提高还是降低动作概率”。它与稍后出现的 ratio 大于或小于
1 是两件不同的事：

- Advantage 表示动作结果的相对好坏；
- ratio 表示新策略相对旧策略如何改变了该动作的概率。

### 5.2 为什么不直接使用原始 return

假设某些初始状态天然容易获得高回报，另一些状态天然很难。只使用 $G_t$ 时，即使一个
动作在困难状态下已经做得很好，它的绝对回报仍可能低于容易状态中的普通动作，梯度
方差会很大。

$V(s_t)$ 充当与状态相关的 baseline：

$$
\hat A_t\approx G_t-V(s_t).
$$

这样 Actor 关注的是“相对该状态预期水平的提升”，而不是不同状态间不可直接比较的
绝对回报。

从理论上看，只要 baseline 不依赖当前采样动作，从 Policy Gradient 中减去它不会改变
梯度期望：

$$
\mathbb{E}_{a\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a\mid s)b(s)
\right]
=0.
$$

它主要用于降低方差，让训练更稳定。

### 5.3 Critic 为什么输出 $V(s)$

经典 PPO 的 Critic 通常输出：

$$
V_\phi(s),
$$

而不是直接输出 Advantage。原因是环境只给出 reward、next state 和 episode
终止信息，Advantage 不是一个直接观测到的标签。它需要结合：

```text
reward + V(s_t) + V(s_{t+1}) + done
→ TD residual
→ GAE
→ Advantage
```

Actor 已经负责建模动作分布，因此 PPO 的 Critic 只需提供状态 baseline。相比拟合
$(s,a)\mapsto Q(s,a)$，拟合 $s\mapsto V(s)$ 不需要额外覆盖动作维度，也更符合 PPO
中 Actor 与 Critic 的分工。

这并不意味着 $Q(s,a)$ 本身不好。SAC、TD3 和 DDPG 都使用 Q Critic，因为它们的 Actor
更新直接依赖 $Q(s,a)$。算法选择哪种 Critic，取决于它如何更新策略以及如何使用数据。

### 5.4 为什么 $V(s)$ 通常比 $Q(s,a)$ 更平滑

在同一个策略 $\pi$ 下：

$$
V^\pi(s)
=
\mathbb{E}_{a\sim\pi(\cdot\mid s)}
\left[
Q^\pi(s,a)
\right].
$$

离散动作情况下：

$$
V^\pi(s)
=
\sum_a
\pi(a\mid s)Q^\pi(s,a).
$$

因此 $V^\pi(s)$ 是 $Q^\pi(s,a)$ 在动作维度上的加权平均。假设：

```text
Q(s, a1) = 0
Q(s, a2) = 10
Q(s, a3) = 20

π(a | s) = [0.3, 0.4, 0.3]
```

那么：

$$
V^\pi(s)
=
0\times0.3
+
10\times0.4
+
20\times0.3
=10.
$$

如果策略小幅变为：

```text
π(a | s) = [0.25, 0.45, 0.30]
```

则：

$$
V^\pi(s)
=
0\times0.25
+
10\times0.45
+
20\times0.30
=10.5.
$$

这个加权平均会平滑一部分动作之间的尖锐差异。相比之下，$Q(s,a)$ 必须分别刻画每个
具体动作的长期后果。连续动作下，它可能需要区分：

```text
方向盘角度 0.10 与 0.11
速度 0.50 与 0.55
```

对应价值的细微或突变差异，因此要拟合的是一个更复杂的动作价值曲面。

“$V$ 更平滑”不是绝对定理。若环境本身高度不连续，或者策略发生剧烈变化，
$V^\pi(s)$ 也可能明显波动。但在 PPO 通过 ratio、clip 或 KL 监控进行小步策略更新的
条件下，状态价值作为动作加权平均通常更稳定，也更适合作为 baseline。

## 6. Return、TD error 与 GAE

### 6.1 一段 rollout 中有哪些 value

假设 rollout 长度为 $T=5$：

```text
s0 --a0--> r0, s1
s1 --a1--> r1, s2
s2 --a2--> r2, s3
s3 --a3--> r3, s4
s4 --a4--> r4, s5
```

buffer 保存：

```text
states  = [s0, s1, s2, s3, s4]
actions = [a0, a1, a2, a3, a4]
rewards = [r0, r1, r2, r3, r4]
values  = [V(s0), V(s1), V(s2), V(s3), V(s4)]
```

rollout 收集完后，还要对末端状态计算：

```text
next_value = V(s5)
```

为了向量化计算，也可以构造：

```text
next_values = [V(s1), V(s2), V(s3), V(s4), V(s5)]
```

所以：

- `values` 是长度为 $T$ 的序列；
- `next_value` 通常特指末端状态 $s_T$ 的一个价值估计；
- `next_values` 是将 value 向后平移并补上末端 value 后得到的长度为 $T$ 的序列。

### 6.2 TD residual

单步 TD residual 为：

$$
\delta_t
=
r_t
+
\gamma(1-d_t)V_\phi(s_{t+1})
-
V_\phi(s_t),
$$

其中 $d_t=1$ 表示该 transition 后 episode 真正终止。终止后没有可继续获得的未来
回报，因此 bootstrap 项被置零。

$\delta_t$ 可以理解为：

> 实际看到的“一步 reward + 下一状态估值”，比 Critic 原本对当前状态的预测高多少？

### 6.3 GAE

Generalized Advantage Estimation（GAE）从 rollout 末端向前递推：

$$
\hat A_t
=
\delta_t
+
\gamma\lambda(1-d_t)\hat A_{t+1}.
$$

展开后：

$$
\hat A_t
=
\delta_t
+
(\gamma\lambda)\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+\cdots.
$$

$\lambda$ 控制 bias 与 variance 的折中：

```text
λ 接近 0  → 更接近短步 TD，方差较低，但更依赖 Critic，bias 较高
λ 接近 1  → 更接近 Monte Carlo return，bias 较低，但方差较高
```

常见设置是 $\lambda=0.95$，但它是超参数，并非固定规则。

### 6.4 为什么 GAE 不一定等 episode 结束

如果 rollout 在 episode 中途结束，末端状态 $s_T$ 后面仍可能有未来回报。PPO 使用：

$$
V_\phi(s_T)
$$

近似剩余未来回报，这就是 bootstrap。因此 PPO 可以收集固定长度的 rollout，而不必
每次都等完整 episode 结束。

如果 episode 在 rollout 内真正终止，则对应位置的 $(1-d_t)$ 为零，不再跨越 episode
传播下一状态价值或后续 Advantage。

### 6.5 Critic 的 return target

PPO + GAE 中常构造：

$$
\hat R_t
=
\hat A_t+V_{\phi_{\mathrm{old}}}(s_t).
$$

然后训练 Critic：

$$
L_V(\phi)
=
\mathbb{E}_t
\left[
\left(
V_\phi(s_t)-\hat R_t
\right)^2
\right].
$$

这里的 $\hat R_t$ 不是单步 reward，而是由真实 rollout reward 和 bootstrap value
共同构造的长期回报 target。bootstrap 会引入 Critic 估计带来的 bias，但可以降低
方差并允许截断 rollout；GAE 的 $\lambda$ 用来调节这项折中。

## 7. PPO 的 clipped surrogate objective

### 7.1 为什么要比较新旧策略

rollout 是由采样时的旧策略
$\pi_{\theta_{\mathrm{old}}}$ 产生的。进入优化阶段后，Actor 参数变为 $\theta$，但动作
仍是旧策略采样出来的。

PPO 对每个已采样动作计算 importance ratio：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
}.
$$

代码中通常使用 log probability：

```python
ratio = torch.exp(new_log_prob - old_log_prob)
```

它的含义是：

```text
ratio > 1  → 新策略提高了这次动作的概率
ratio < 1  → 新策略降低了这次动作的概率
ratio = 1  → 新旧策略对这次动作给出相同概率
```

如果不约束更新，同一批 Advantage 可能被优化很多次，导致新策略迅速远离实际产生这批
数据的旧策略。这样数据会越来越失配，训练容易不稳定。

### 7.2 Clipped objective

PPO-Clip 最大化：

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min\left(
r_t(\theta)\hat A_t,\,
\operatorname{clip}
\left(r_t(\theta),1-\epsilon,1+\epsilon\right)
\hat A_t
\right)
\right].
$$

PyTorch 优化器默认最小化 loss，所以实现中取负号：

```python
ratio = torch.exp(new_log_prob - old_log_prob)

unclipped = ratio * advantages
clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

policy_loss = -torch.min(unclipped, clipped).mean()
```

可以按 Advantage 的正负理解 `min`：

#### 当 $\hat A_t>0$

这个动作比预期好，应该提高它的概率。随着 ratio 增大，目标先增大；当 ratio 超过
$1+\epsilon$ 后，裁剪项不再提供额外收益。

```text
好动作：允许提高概率，但提高得太多以后不再奖励
```

#### 当 $\hat A_t<0$

这个动作比预期差，应该降低它的概率。当 ratio 低于 $1-\epsilon$ 后，裁剪项不再因为
继续降低概率而提供额外收益。

```text
差动作：允许降低概率，但降低得太多以后不再奖励
```

### 7.3 Clip 不是 ratio 的硬约束

若 $\epsilon=0.2$，常说裁剪区间是 $[0.8,1.2]$。准确含义是：

> 超过该区间且更新方向有利于当前 Advantage 时，surrogate objective 不再给予额外
> 收益。

它并不把 Actor 参数或实际 ratio 强行投影回 $[0.8,1.2]$。由于共享网络参数、多个样本
共同优化以及更新方向不同，训练后的某些 ratio 完全可能越界。

因此 clip 是目标函数中的保守机制，不是严格的概率变化上限。实际实现还可能监控
approximate KL，并在 KL 过大时提前停止当前优化轮次。

## 8. Actor loss、Critic loss 与 Entropy bonus

PPO 常见总 loss 为：

$$
L
=
L_{\mathrm{policy}}
+
c_v L_V
-
c_e H(\pi_\theta),
$$

其中：

- $L_{\mathrm{policy}}=-L^{\mathrm{CLIP}}$；
- $L_V$ 是 Critic 的 value regression loss；
- $H(\pi_\theta)$ 是策略熵；
- $c_v$ 与 $c_e$ 控制各项权重。

### 8.1 Entropy 不是 cross entropy

离散策略的 entropy 为：

$$
H(\pi(\cdot\mid s))
=
-
\sum_a
\pi(a\mid s)\log\pi(a\mid s).
$$

分布越平均，entropy 越高；分布越集中，entropy 越低。因为总 loss 中使用
$-c_eH$，最小化 loss 会鼓励策略保留一定随机性，避免过早失去探索。

例如四个离散动作的策略为：

```text
直行：0.7
左转：0.1
右转：0.1
刹车：0.1
```

那么：

$$
H
=
-
\left(
0.7\log0.7
+
3\times0.1\log0.1
\right).
$$

对比下面两个策略：

```text
几乎确定：
[0.990, 0.003, 0.003, 0.004]

较为分散：
[0.35, 0.25, 0.20, 0.20]
```

前者的 entropy 很低，探索较少；后者的 entropy 更高，仍会较频繁地尝试多种动作。

一维高斯 $\mathcal N(\mu,\sigma^2)$ 的 differential entropy 为：

$$
H
=
\frac{1}{2}
\log\left(2\pi e\sigma^2\right).
$$

独立多维高斯的 entropy 是各维之和。标准差越大，entropy 通常越大，探索也更强：

```python
base_dist = torch.distributions.Normal(mean, std)
dist = torch.distributions.Independent(base_dist, 1)
entropy = dist.entropy().mean()
```

### 8.2 Value loss clipping（可选）

最简单的 Critic loss 是：

$$
L_V
=
\mathbb{E}_t
\left[
\left(
V_\phi(s_t)-\hat R_t
\right)^2
\right].
$$

有些 PPO 实现还会限制新 value 相对 rollout 时旧 value 的变化：

$$
V_t^{\mathrm{clip}}
=
V_{\mathrm{old}}(s_t)
+
\operatorname{clip}
\left(
V_\phi(s_t)-V_{\mathrm{old}}(s_t),
-\epsilon_V,
+\epsilon_V
\right).
$$

然后采用较保守的一项：

$$
L_V^{\mathrm{clip}}
=
\mathbb{E}_t
\left[
\max
\left(
\left(V_\phi(s_t)-\hat R_t\right)^2,\,
\left(V_t^{\mathrm{clip}}-\hat R_t\right)^2
\right)
\right].
$$

它的目的与 policy clip 类似：避免在同一批 rollout 上训练多个 epoch 时，Critic 的
预测一次变化过大。不过 value clipping 是 **可选实现细节**，并非所有 PPO 代码都会
启用，$\epsilon_V$ 也不一定与 policy 的 $\epsilon$ 相同。

### 8.3 哪些量应该固定

对一批已经收集好的 rollout，通常将以下量视为固定训练目标：

```text
actions
old_log_probs
advantages
returns
```

每个 minibatch 中重新计算：

```text
new_log_probs
entropy
new_values
```

这样梯度只更新当前 Actor 与 Critic，不会反向穿过环境、rollout 采样过程或 GAE
递推过程。

### 8.4 常见稳定化手段如何分工

原版笔记提到的稳定化手段作用在不同位置：

| 手段 | 主要作用 |
|---|---|
| Advantage normalization | 调整 Actor 学习信号的尺度，改善优化条件 |
| GAE 的 $\lambda$ | 调节 Advantage 估计的 bias-variance 折中 |
| Policy ratio clipping | 取消过度有利的策略更新所带来的额外收益 |
| Value loss clipping | 可选地限制 Critic 在旧 rollout 上的剧烈变化 |
| Approximate KL early stopping | 新旧策略差异过大时提前停止当前更新 |
| Entropy bonus | 防止策略过早变得过于确定 |
| Gradient norm clipping | 限制单次反向传播产生的过大梯度 |

这些机制不能互相替代。例如 policy clip 不会修复错误的 return target，entropy bonus
也不会保证新旧策略的 KL 一定足够小。

## 9. Rollout、bootstrap 与多轮更新

PPO 的一次迭代分为两个阶段。

### 9.1 数据收集阶段

1. 固定当前 Actor 作为采样策略。
2. 与一个或多个环境交互 $T$ 步。
3. 每一步保存：

   ```text
   observation
   action
   reward
   done
   old_log_prob
   old_value
   ```

4. 对最后的 next observation 计算 `next_value`。
5. 从后向前计算 GAE 和 return。

这一步完成后，rollout buffer 相当于旧策略的一个固定快照。

### 9.2 优化阶段

1. 将 rollout 样本打乱并划分为 minibatch。
2. 用当前 Actor 重新计算同一动作的 `new_log_prob`。
3. 计算 ratio、clipped policy loss、value loss 与 entropy。
4. 对同一批 rollout 重复若干 epoch。
5. 完成后丢弃这批 rollout。
6. 使用更新后的 Actor 重新与环境交互。

所以 PPO 是：

```text
rollout 一次 → 对固定 rollout 优化多次 → 丢弃 → 重新 rollout
```

不是：

```text
每做一次梯度更新就重跑同一个 batch
```

也不是：

```text
把历史数据长期保存在 replay buffer 中反复训练
```

## 10. On-policy 以及与 DQN、SAC 的区别

PPO 的 ratio 允许有限地复用刚刚由旧参数收集的数据，但这不使它变成典型 off-policy
算法。标准分类中，PPO 仍是 on-policy：策略更新若干轮后，旧 rollout 很快失效并被
丢弃。

| 算法 | 策略表示 | Critic | 数据使用 | Actor 如何更新 | 主要探索方式 |
|---|---|---|---|---|---|
| DQN | 没有独立 Actor | $Q(s,a)$ | Replay buffer，off-policy | 通过 $\arg\max_a Q(s,a)$ 隐式决策 | epsilon-greedy |
| PPO | 随机 Actor | 通常为 $V(s)$ | 新鲜 rollout，on-policy | Advantage 加权的 clipped Policy Gradient | 策略采样与 entropy |
| SAC | 随机 Actor | 通常为双 $Q(s,a)$ | Replay buffer，off-policy | 最大化 Q 与 entropy 正则化目标 | 随机策略与 entropy |

### 10.1 为什么 DQN 必须学习 Q

DQN 学习动作价值函数，其常见 Bellman target 为：

$$
y_t
=
r_t
+
\gamma(1-d_t)
\max_{a'}
Q_{\theta^-}(s_{t+1},a'),
$$

对应 loss 为：

$$
L_{\mathrm{DQN}}(\theta)
=
\mathbb{E}_t
\left[
\left(
Q_\theta(s_t,a_t)-y_t
\right)^2
\right],
$$

其中 $\theta^-$ 表示用于构造 target 的目标网络参数。DQN 没有显式 Actor，通过：

$$
a=\arg\max_a Q(s,a)
$$

选择动作。因此 Critic 必须给出每个动作的价值。

### 10.2 为什么 SAC 使用 Q Critic

SAC 的 Actor 通过 Q 值判断一个动作是否值得提高概率，因此需要
$Q(s,a)$。它还使用 replay buffer 和 off-policy Bellman target，与 PPO 的 rollout +
Advantage 更新路径不同。

### 10.3 Q 高估与 PPO value 误差不是同一种问题

DQN 类 target 中的：

$$
\max_{a'}Q(s',a')
$$

会倾向于选择估计噪声中偏高的动作，形成系统性的 overestimation bias。

例如三个动作的真实价值都为 10，但当前估计是：

```text
Q(s, a1) = 9
Q(s, a2) = 10
Q(s, a3) = 13
```

`max` 会选择被高估的 13，并把它写入 Bellman target。这个偏高的 target 又会参与后续
更新，因此误差可能沿 Bellman backup 传播。问题不在于某次估计恰好有噪声，而在于
`max` 会系统性偏向噪声中较高的一侧。

PPO 的 $V(s)$ 也可能估错。若 $V(s_t)$ 被高估，本来较好的动作可能得到过低甚至为负的
Advantage，Actor 更新方向会受到影响。但这里没有对动作执行 `max`，误差主要通过
Advantage 的权重进入 Policy Gradient。

两者都会损害训练，但误差产生和传播的机制不同。

## 11. 常见误解与最小例子

### 11.1 Reward、return、value 和 Advantage 是同一个量吗

不是：

```text
reward r_t
    环境在单个 transition 后返回的即时反馈

return G_t
    从当前时刻起累积的折扣 reward

value V(s_t)
    Critic 对当前策略下期望 return 的预测

advantage A(s_t, a_t)
    当前动作相对该状态平均动作的好坏
```

### 11.2 一次只采样一个动作，怎么得到期望

一次动作只是一个随机样本，不会给出精确期望。PPO 在多个环境、多个时间步和多次
iteration 中不断采样，再用 minibatch 平均估计期望梯度。

关键不是“一个动作包含整个期望”，而是：

> 每个动作提供一个随机梯度样本，许多样本的平均近似期望梯度。

### 11.3 已采样动作不可导，为什么 Actor 仍能更新

因为反向传播不穿过采样动作本身，而是穿过：

$$
\log\pi_\theta(a_t\mid s_t).
$$

动作只是告诉分布计算哪一个 log probability；Advantage 只是给这个 log probability
的梯度加权。

### 11.4 Critic 是否直接输出 Advantage

经典 PPO 中通常不是。Critic 输出 $V(s_t)$；GAE 再结合 reward、done、
$V(s_t)$ 与 $V(s_{t+1})$ 构造 $\hat A_t$。

GAE 是一个数值递推过程，不是另一个神经网络。

### 11.5 Advantage 为正与 ratio 大于 1 是一回事吗

不是：

```text
Advantage > 0  → 这个动作应该被提高概率
ratio > 1      → 当前新策略实际上已经提高了它的概率
```

PPO objective 同时观察“应该往哪改”和“已经改了多少”。

### 11.6 一个两动作最小例子

假设只有一个状态和两个动作：

```text
R(A) = 10
R(B) = 0
π(A) = 0.5
π(B) = 0.5
```

当前期望回报为：

$$
J
=
0.5\times 10
+
0.5\times 0
=5.
$$

训练时每次只能采样一个动作：

- 采到 $A$ 且其 Advantage 为正时，梯度提高 $\log\pi(A)$；
- 采到 $B$ 且其 Advantage 为负时，梯度降低 $\log\pi(B)$。

随着采样次数增加，$A$ 的概率逐渐上升，期望回报接近 10。算法不需要事先枚举并知道
两个动作的真实 reward；它通过不断执行、评价和调整采样概率逐渐发现更好的动作。

### 11.7 为什么不能从 $p$ 采样后直接平均 $\nabla p\,R$

第 3.3 节已经用一般公式和 `0.8/0.2` 两轨迹例子完整推导了这一点。这里仅保留判断
规则：从 $p$ 采样会自动带上一份 $p(\tau)$ 权重，因此不能直接平均
$\nabla p(\tau)R(\tau)$；必须先除以采样密度，而
$\nabla p(\tau)/p(\tau)=\nabla\log p(\tau)$。

> Actor–Critic 的 actor 和 critic 都依赖 reward 与当前 value 预测之间形成的 TD/return 误差。critic 用这个误差修正价值预测，actor 用由它构造出的 advantage 调整固定 rollout 动作的概率。
> bootstrap 不是优化信号的原始来源，而是 reward 信息的传播器
> 真实 reward 决定“哪里是好结果”，critic 把这种信息泛化到大量没有即时 reward 的状态，advantage 再告诉 actor：
> 这次选到的动作，是否把系统带到了一个比原先预期更好的未来

## 12. 完整伪代码与总结

下面的伪代码把各部分连接起来：

```python
initialize actor parameters theta
initialize critic parameters phi

for iteration in range(num_iterations):
    buffer = []
    obs = current_observation

    # 1. 使用当前策略收集新鲜 rollout
    for t in range(rollout_length):
        with torch.no_grad():
            dist = actor.distribution(obs)
            action = dist.sample()
            old_log_prob = dist.log_prob(action)
            old_value = critic(obs)

        next_obs, reward, done, info = env.step(action)

        buffer.append(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            old_log_prob=old_log_prob,
            old_value=old_value,
        )

        obs = reset_if_done(next_obs, done)

    # 2. 给 rollout 末端做 bootstrap
    with torch.no_grad():
        next_value = critic(obs)

    # 3. 从后向前计算 GAE，并构造 Critic target
    advantages = compute_gae(
        rewards=buffer.rewards,
        dones=buffer.dones,
        values=buffer.old_values,
        next_value=next_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    returns = advantages + buffer.old_values
    advantages = normalize(advantages)

    # 4. 对同一批 rollout 做有限次数 minibatch 更新
    for epoch in range(update_epochs):
        for batch in shuffled_minibatches(buffer, advantages, returns):
            dist = actor.distribution(batch.obs)
            new_log_prob = dist.log_prob(batch.action)
            entropy = dist.entropy()
            new_value = critic(batch.obs)

            ratio = torch.exp(new_log_prob - batch.old_log_prob)

            surrogate_1 = ratio * batch.advantage
            surrogate_2 = torch.clamp(
                ratio,
                1.0 - clip_eps,
                1.0 + clip_eps,
            ) * batch.advantage

            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

            value_error = (new_value - batch.return_target) ** 2
            if value_clip_eps is None:
                value_loss = value_error.mean()
            else:
                value_clipped = batch.old_value + torch.clamp(
                    new_value - batch.old_value,
                    -value_clip_eps,
                    value_clip_eps,
                )
                value_error_clipped = (
                    value_clipped - batch.return_target
                ) ** 2
                value_loss = torch.maximum(
                    value_error,
                    value_error_clipped,
                ).mean()

            entropy_bonus = entropy.mean()

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy_bonus
            )

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
            optimizer.step()

    # 5. buffer 不进入长期 replay；下一轮用更新后的策略重新采样
```

最终可以把 PPO 记成：

```text
不可导环境
→ 用 log-derivative trick 得到可采样的 Policy Gradient
→ 用 Critic baseline 和 GAE 构造低方差 Advantage
→ 用新旧策略 ratio 在旧 rollout 上计算更新
→ 用 clipped objective 取消过度有利更新的额外收益
→ 对新鲜 rollout 有限复用后丢弃，再重新采样
```

最核心的 Actor 学习信号是：

$$
\nabla_\theta
\log\pi_\theta(a_t\mid s_t)\hat A_t.
$$

其中：

```text
Advantage 告诉模型：这次动作相对预期是好还是坏；
log probability 告诉优化器：怎样改参数才能让该动作以后更常或更少出现；
PPO ratio 与 clip 告诉优化器：不要持续从同一批旧数据中获取过大的有利更新。
```

## 相关笔记

- [[SAC_PPO_compare|SAC vs PPO]]：比较 on-policy 与 off-policy Actor-Critic。
- [[RL/opd_on_policy_distillation_知识笔记|OPD / On-Policy Distillation]]：理解 token
  级 PG-style loss 与 KL distillation。
- [[Pi_star0.6论文问题解答|pi*0.6 / RECAP]]：机器人 VLA 中基于 Advantage 的策略改进。
