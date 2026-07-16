---
title: Dreamer 潜空间想象技术报告
type: paper_note
topic: latent_imagination_model_based_rl
status: mature
importance: high
updated: 2026-07-16
tags:
  - dreamer
  - world-model
  - model-based-rl
  - latent-imagination
  - rssm
  - actor-critic
  - pathwise-gradient
  - robotics
---

# Dreamer 技术报告：基于潜空间想象的行为学习

> **论文**：Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi.  
> **标题**：*Dream to Control: Learning Behaviors by Latent Imagination*  
> **发表**：ICLR 2020  
> **报告范围**：结合论文正文、附录，以及围绕世界模型训练、REINFORCE/PPO 对比、Reward Model、潜空间想象、连续机器人动作和 LLM 离散动作的讨论，对 Dreamer 的关键机制做系统化技术整理。

---

## 摘要

Dreamer 是一种从图像输入学习连续控制策略的模型式强化学习算法。它首先利用真实环境经验训练一个可微分的潜空间世界模型，再从真实轨迹对应的潜状态出发，在世界模型内部生成反事实的未来轨迹，并在这些“想象轨迹”上训练 Actor 和 Value Model。

其核心思想可以概括为：

1. 用真实观测推断当前潜状态；
2. 用潜空间 Transition Model 预测不同动作可能导致的未来；
3. 用 Reward Model 预测想象轨迹中的即时奖励；
4. 用 Value Model 估计有限想象区间以外的长期回报；
5. 将多步回报的解析梯度穿过 Value、Reward、Transition 和动作采样过程，反向传播到 Actor。

Dreamer 和 A3C、PPO 都使用 Value Model，但用法本质不同：在典型 REINFORCE 类算法中，Value 主要作为降低方差的 baseline，Actor 不通过 Value 网络求导；在 Dreamer 中，Value 是 Actor 的可微目标之一，Actor 的梯度能够穿过 Value Model 和学习到的 Dynamics Model。

Dreamer 之所以需要 Reward Model，并不是因为普通策略梯度“避免奖励”，而是因为想象轨迹没有被真实环境执行，环境不会返回对应奖励。Reward Model 负责给出想象状态下的即时奖励，而 Value Model 负责估计按照当前策略继续行动时的长期累计奖励。二者互补而非重复。

---

## 1. 问题定义与符号

论文把视觉控制建模为部分可观测马尔可夫决策过程（POMDP）：

- 观测：$o_t$，通常为高维图像；
- 动作：$a_t$，连续控制任务中为实值向量；
- 奖励：$r_t$，由环境返回；
- 潜状态：$s_t$，世界模型内部用于概括历史信息的紧凑状态；
- 折扣因子：$\gamma$；
- 想象长度：$H$。

目标是最大化期望累计奖励：

$$
\max_\pi\;\mathbb E\left[\sum_{t=1}^{T}r_t\right].
$$

论文用以下模型构成世界模型：

### 1.1 Representation Model

$$
p_\theta(s_t\mid s_{t-1},a_{t-1},o_t)
$$

它结合前一潜状态、前一动作和当前真实观测，推断当前潜状态。由于使用了当前观测，它可理解为过滤后验或 posterior state。

### 1.2 Transition Model

$$
q_\theta(s_t\mid s_{t-1},a_{t-1})
$$

它不读取当前真实观测，仅根据历史潜状态和动作预测下一潜状态，因此是想象过程中使用的 latent dynamics 或 prior。

### 1.3 Reward Model

$$
q_\theta(r_t\mid s_t)
$$

它预测潜状态对应的即时奖励。

### 1.4 Observation Model

在像素重建版本中还包含：

$$
q_\theta(o_t\mid s_t),
$$

用于从潜状态重建图像，并为潜表示提供学习信号。

### 1.5 Actor 与 Value Model

$$
a_t\sim q_\phi(a_t\mid s_t),
$$

$$
v_\psi(s_t)\approx V^{\pi_\phi}(s_t).
$$

其中 $\theta$、$\phi$、$\psi$ 分别表示世界模型、Actor 和 Value Model 的参数。

> **符号注意**：该论文中的 $p$ 和 $q$ 用法与许多 VAE 文献的习惯不同。不要只根据字母判断 prior/posterior，应看条件变量中是否包含真实观测 $o_t$。

---

## 2. Dreamer 的三阶段工作流

> [!figure] 论文原始模型结构图
> ![[attachments/paper-figures/dreamer-architecture-overview.png]]
> Dreamer 从经验数据学习潜空间动力学，并在潜空间想象中学习价值与动作。原图来自 [Dream to Control: Learning Behaviors by Latent Imagination（arXiv:1912.01603）](https://arxiv.org/abs/1912.01603)，由论文源文件高分辨率导出。

论文第 3 页的图 3 和 Algorithm 1 将算法分为三个反复交替的过程。

### 2.1 从真实经验学习世界模型

从 replay dataset 中抽取真实序列：

$$
\{o_t,a_t,r_t\}_{t=k}^{k+L} \sim \mathcal D.
$$

用真实观测推断 posterior states：

$$
s_t\sim p_\theta(s_t\mid s_{t-1},a_{t-1},o_t).
$$

随后联合训练表示、潜空间动力学、图像重建和奖励预测。

### 2.2 在潜空间中学习行为

从真实数据对应的 posterior state $s_t$ 出发，在不访问真实环境的情况下展开长度为 $H$ 的想象轨迹：

$$
a_\tau\sim q_\phi(a_\tau\mid s_\tau),
$$

$$
s_{\tau+1}\sim q_\theta(s_{\tau+1}\mid s_\tau,a_\tau),
$$

$$
\hat r_\tau\sim q_\theta(r_\tau\mid s_\tau).
$$

然后在这些轨迹上训练 Actor 和 Value Model。

### 2.3 在真实环境中执行 Actor

真实部署时，每一步都读取新的环境观测：

$$
s_t\sim p_\theta(s_t\mid s_{t-1},a_{t-1},o_t^{\mathrm{real}}),
$$

$$
a_t\sim q_\phi(a_t\mid s_t).
$$

执行动作后，环境返回新的真实观测和奖励，再用 Representation Model 修正潜状态。

因此，Dreamer 的行为学习主要发生在想象中，但世界模型训练和数据收集仍依赖真实环境。它不是“完全不使用真实数据”的算法。

---

## 3. Algorithm 1 中 “Update $\theta$ using representation learning” 的含义

这句话是一个抽象接口，而不是某个单独的更新公式。作者有意把 Dreamer 的行为学习算法与具体表示学习目标解耦，因此原则上可以接入多种世界模型目标，例如：

- 仅预测 Reward；
- 图像重建；
- 对比学习。

论文主实验主要使用图像重建目标。此时，世界模型最大化：

$$
\mathcal J_{\mathrm{REC}}
=
\mathbb E\left[
\sum_t
\left(
\log q_\theta(o_t\mid s_t)
+
\log q_\theta(r_t\mid s_t)
-
\beta D_{\mathrm{KL}}\left[
 p_\theta(s_t\mid s_{t-1},a_{t-1},o_t)
 \Vert
 q_\theta(s_t\mid s_{t-1},a_{t-1})
\right]
\right)
\right].
$$

工程中通常最小化其负值：

$$
\mathcal L_{\mathrm{model}}
=
-\log q_\theta(o_t^{\mathrm{gt}}\mid s_t)
-\log q_\theta(r_t^{\mathrm{gt}}\mid s_t)
+
\beta D_{\mathrm{KL}}(\mathrm{posterior}\Vert\mathrm{prior}).
$$

随后执行：

$$
\theta
\leftarrow
\theta-\alpha_\theta\nabla_\theta\mathcal L_{\mathrm{model}}.
$$

### 3.1 哪些参数属于 $\theta$？

在主实验设置中，$\theta$ 通常包括：

- 图像 Encoder；
- RSSM 的确定性循环部分；
- 随机潜变量 posterior；
- 随机潜变量 prior/transition；
- 图像 Decoder；
- Reward Head；
- 若环境允许提前终止，还可能包括 discount/continuation head。

所以“Update $\theta$”不是只更新 Encoder，而是联合更新整个世界模型。

### 3.2 为什么叫 Representation Learning？

目标不是单纯让 Decoder 画出一张好看的图，而是让潜状态 $s_t$ 同时满足：

1. 能保留重建观测所需的信息；
2. 能预测任务奖励；
3. 能够被前一状态和动作较准确地预测；
4. 适合在潜空间中向前滚动并用于控制。

KL 项限制 posterior 不得随意把当前图像的所有细节塞进潜状态，而应尽量与仅根据历史和动作得到的 prior 一致。这促使潜状态形成可预测的时序表示。

---

## 4. 为什么图像重建和 Reward 预测写成概率形式？

论文中的：

$$
\log q_\theta(o_t\mid s_t),
\qquad
\log q_\theta(r_t\mid s_t)
$$

都隐含了真值。更完整的写法是：

$$
\log q_\theta(o_t^{\mathrm{gt}}\mid s_t),
\qquad
\log q_\theta(r_t^{\mathrm{gt}}\mid s_t).
$$

其中 $o_t^{\mathrm{gt}}$ 和 $r_t^{\mathrm{gt}}$ 来自 replay dataset。

若图像模型采用固定方差高斯分布：

$$
q_\theta(o_t\mid s_t)
=
\mathcal N\bigl(o_t;\mu_o(s_t),\sigma_o^2I\bigr),
$$

则：

$$
-\log q_\theta(o_t^{\mathrm{gt}}\mid s_t)
=
\frac{1}{2\sigma_o^2}
\left\|o_t^{\mathrm{gt}}-\mu_o(s_t)\right\|^2+C,
$$

本质上就是带权 MSE。

同理，若 Reward 采用高斯分布：

$$
q_\theta(r_t\mid s_t)
=
\mathcal N\bigl(r_t;\mu_r(s_t),\sigma_r^2\bigr),
$$

则负对数似然等价于 Reward 回归误差：

$$
-\log q_\theta(r_t^{\mathrm{gt}}\mid s_t)
\propto
\left(r_t^{\mathrm{gt}}-\mu_r(s_t)\right)^2.
$$

概率写法更通用：改变输出分布即可得到 MSE、二元交叉熵、分类交叉熵或其他负对数似然损失。

---

## 5. Dreamer 的关键步骤：从每个真实潜状态展开想象轨迹

Algorithm 1 中最关键的一行之一是：

> Imagine trajectories $\{(s_\tau,a_\tau)\}_{\tau=t}^{t+H}$ from each $s_t$.

这里的 $s_t$ 来自真实经验序列的 posterior，而不是任意随机采样的潜状态。随后执行：

```text
posterior state s_t
        │
        ▼
Actor: a_t ~ q_φ(a_t | s_t)
        │
        ▼
Transition: s_{t+1} ~ q_θ(s_{t+1} | s_t, a_t)
        │
        ├── Reward: r̂_{t+1} ~ q_θ(r_{t+1} | s_{t+1})
        ├── Value:  v_ψ(s_{t+1})
        │
        ▼
Actor: a_{t+1} ~ q_φ(a_{t+1} | s_{t+1})
        │
       ...
```

严格区分三个阶段非常重要：

| 阶段 | 是否使用真实 $o_t$ | 状态更新方式 |
|---|---:|---|
| 世界模型训练 | 每一步都使用 | $p_\theta(s_t\mid s_{t-1},a_{t-1},o_t)$ |
| 潜空间想象 | 仅起点来自真实 posterior | 之后反复使用 $q_\theta(s_{\tau+1}\mid s_\tau,a_\tau)$ |
| 真实环境执行 | 每一步都使用 | 环境返回新观测后再次使用 $p_\theta$ |

因此，想象阶段不是不断执行

$$
p_\theta\rightarrow q_\phi\rightarrow q_\theta\rightarrow p_\theta,
$$

而是先用 $p_\theta$ 获得起点，然后不断交替：

$$
q_\phi(a_\tau\mid s_\tau)
\quad\text{与}\quad
q_\theta(s_{\tau+1}\mid s_\tau,a_\tau).
$$

---

## 6. 为什么想象过程中不继续采样或使用观测 $o$？

这不是对观测的“执着排斥”，而是由反事实预测的逻辑决定的。

### 6.1 反事实动作对应的真实未来观测不存在

假设 replay 中记录的是：

- 状态 $s_t$；
- 数据动作 $a_t^{\mathrm{data}}$：向左；
- 下一观测 $o_{t+1}^{\mathrm{data}}$：机械臂出现在左侧。

在想象中，Actor 可能选择：

$$
a_t^{\mathrm{policy}}=\text{向右}.
$$

此时需要的未来观测应来自：

$$
p(o_{t+1}\mid s_t,a_t^{\mathrm{policy}}),
$$

而 replay 中的图像来自：

$$
p(o_{t+1}\mid s_t,a_t^{\mathrm{data}}).
$$

二者对应不同的动作，不能混用。继续输入 replay 中的 $o_{t+1}$ 会制造因果矛盾。

### 6.2 先解码预测图像再编码回来通常是冗余的

理论上可以执行：

$$
s_{t+1}
\rightarrow \hat o_{t+1}
\rightarrow \hat s_{t+1},
$$

但 $\hat o_{t+1}$ 本身就是由同一个预测状态产生的，并没有加入来自真实环境的新信息。它不能纠正世界模型误差，还会增加：

- 图像 Decoder 的计算；
- 图像采样噪声；
- 再编码的信息损失；
- 显存占用；
- 长期像素预测误差。

潜空间直接滚动：

$$
s_t\rightarrow s_{t+1}\rightarrow s_{t+2}
$$

更紧凑，也便于并行生成大量轨迹。

### 6.3 真实执行时仍然不断使用新观测

当动作真正发给机器人或模拟器后，环境会返回真实 $o_{t+1}$。此时 Representation Model 会融合真实证据，对 transition prior 进行修正：

$$
s_{t+1}^{\mathrm{post}}
\sim
p_\theta(s_{t+1}\mid s_t,a_t,o_{t+1}^{\mathrm{real}}).
$$

因此：

- 想象时没有真实未来观测，只能用 prior；
- 执行时有真实未来观测，每一步都用 posterior 修正状态。

---

## 7. Reward Model 与 Value Model 的区别

普通 model-free policy gradient 并不是“避免直接使用 Reward”。相反，Value 是从 Reward 学出来的：

$$
V(s_t)\approx r_t+\gamma V(s_{t+1}).
$$

A3C、PPO 等方法通常不需要 Reward Model，是因为真实环境已经在每一步直接返回 $r_t$，没有必要再训练一个模型去预测它。

Dreamer 在想象轨迹中没有调用真实环境，因此环境不会返回反事实动作对应的奖励。它必须用：

$$
q_\theta(r_\tau\mid s_\tau)
$$

给想象状态提供即时奖励。

### 7.1 二者预测的对象不同

| 模型 | 预测量 | 是否依赖当前策略 | 主要作用 |
|---|---|---:|---|
| Reward Model | 当前一步即时奖励 $r_t$ | 通常不依赖策略 | 评价想象轨迹的局部结果 |
| Value Model | 未来累计回报 $V^{\pi}(s_t)$ | 依赖当前 Actor | 估计想象区间以外的长期价值 |

Reward Model 表达的是：

> “处于这个状态，当前大约得到多少奖励？”

Value Model 表达的是：

> “从这个状态开始，按照当前策略继续行动，未来总共能得到多少奖励？”

当 Actor 改变时，$V^{\pi}$ 也会改变；而在固定任务中，Reward 函数相对稳定。

### 7.2 为什么不只使用 Value Model？

只依赖 Value 会让所有未来效果都压缩到一个 Critic 预测中。Dreamer 使用多步模型预测：

$$
\hat r_t
+
\gamma\hat r_{t+1}
+
\cdots
+
\gamma^k v_\psi(s_{t+k}),
$$

其中近期效果由 Transition 和 Reward Model 显式展开，远期尾部由 Value bootstrap。这样可以在模型误差和 Value 偏差之间进行折中。

---

## 8. $V_\lambda$：在有限想象长度下考虑长期回报

如果只最大化想象区间 $H$ 内的 Reward：

$$
\sum_{\tau=t}^{t+H}\gamma^{\tau-t}\hat r_\tau,
$$

策略会忽略 $H$ 步以后的后果，容易形成短视行为。

Dreamer 为每个想象状态构造 $k$ 步回报：

$$
V_N^k(s_\tau)
=
\sum_{n=\tau}^{h-1}
\gamma^{n-\tau}\hat r_n
+
\gamma^{h-\tau}v_\psi(s_h),
$$

其中：

$$
h=\min(\tau+k,t+H).
$$

再对不同 $k$ 的估计做指数加权：

$$
V_\lambda(s_\tau)
=
(1-\lambda)
\sum_{k=1}^{H-1}
\lambda^{k-1}V_N^k(s_\tau)
+
\lambda^{H-1}V_N^H(s_\tau).
$$

直观上：

- 较短的 $k$：更依赖 Value，方差较低，但偏差可能较大；
- 较长的 $k$：更依赖模型展开的 Reward，使用更多真实结构，但会累积模型误差；
- $\lambda$ return：混合不同长度，平衡偏差与方差。

论文主设置使用：

$$
H=15,\qquad \gamma=0.99,\qquad \lambda=0.95.
$$

---

## 9. Actor 与 Value Model 的更新

### 9.1 Value Model

Value Model 回归 $V_\lambda$：

$$
\min_\psi
\mathbb E\left[
\sum_\tau
\frac12
\left(
 v_\psi(s_\tau)
 -
 \operatorname{sg}[V_\lambda(s_\tau)]
\right)^2
\right],
$$

其中 $\operatorname{sg}$ 表示 stop-gradient。对 Critic 来说，$V_\lambda$ 是固定监督目标。

### 9.2 Actor

Actor 最大化想象状态的价值估计：

$$
\max_\phi
\mathbb E\left[
\sum_{\tau=t}^{t+H}
V_\lambda(s_\tau)
\right].
$$

由于动作、潜状态、Reward 和 Value 都由可微网络产生，Actor 可以获得路径导数：

$$
\phi
\rightarrow a_\tau
\rightarrow s_{\tau+1}
\rightarrow
\{\hat r_{\tau+1},v_\psi(s_{\tau+1})\}
\rightarrow V_\lambda.
$$

世界模型参数 $\theta$ 和 Value 参数 $\psi$ 在 Actor 更新时保持固定，但计算图不能被切断。即：

- 不更新 $\theta$、$\psi$；
- 仍然需要它们对输入的导数；
- 最终只更新 $\phi$。

这与将整个世界模型放入 `no_grad` 不同。若使用 `no_grad`，Actor 将无法获得 $\partial s_{\tau+1}/\partial a_\tau$。

---

## 10. “while Dreamer backpropagates through the value model” 的准确含义

原文中的 `while` 是表示对比的连词，可译为“而”“相比之下”，不是程序循环中的“当……时”。

原句的逻辑是：

> A3C、PPO 等使用 REINFORCE/score-function 梯度的方法，把 Value 当作 baseline 来降低方差；而 Dreamer 会让 Actor 的梯度穿过 Value Model。

### 10.1 A3C/PPO：Value 作为 baseline

典型 policy gradient：

$$
\nabla_\phi J
=
\mathbb E\left[
\nabla_\phi\log\pi_\phi(a_t\mid s_t)
A_t
\right],
$$

其中：

$$
A_t=G_t-V_\psi(s_t).
$$

Actor loss 常写为：

$$
\mathcal L_{\mathrm{actor}}
=-\log\pi_\phi(a_t\mid s_t)
\operatorname{sg}[A_t].
$$

Value 决定这个动作的回报相对平均水平是好还是坏，但 Actor 梯度主要经过：

$$
\log\pi_\phi(a_t\mid s_t).
$$

标准实现不会让 Actor 沿着 $V_\psi$ 的输入导数反传。即使 Actor 和 Critic 共享部分网络，也不改变这一区别：Critic loss 可以更新共享表示，但 policy loss 中的 advantage 通常被视为常数。

### 10.2 Dreamer：Value 是可微目标

Dreamer 使用重参数化动作：

$$
a_\tau=g_\phi(s_\tau,\epsilon),
$$

再经过可微 Transition：

$$
s_{\tau+1}=f_\theta(s_\tau,a_\tau,\xi).
$$

因此：

$$
\frac{\partial v_\psi(s_{\tau+1})}{\partial\phi}
=
\frac{\partial v_\psi}{\partial s_{\tau+1}}
\frac{\partial s_{\tau+1}}{\partial a_\tau}
\frac{\partial a_\tau}{\partial\phi}.
$$

这就是“backpropagates through the value model”。

### 10.3 对比总结

| 方法 | Actor 梯度类型 | Value 的作用 | 是否穿过 Dynamics |
|---|---|---|---:|
| A3C/PPO | score-function / REINFORCE | baseline、advantage 估计 | 否 |
| DDPG/SAC | 通过 $Q(s,a)$ 的动作梯度 | 可微动作价值目标 | 通常不穿过多步环境动态 |
| Dreamer | pathwise / reparameterized gradient | 可微长期价值目标 | 是，多步穿过学习到的 Dynamics |

---

## 11. 连续机器人动作如何由高斯分布产生？

Dreamer 的连续 Actor 使用 tanh 变换高斯分布：

$$
a_\tau
=
\tanh\left(
\mu_\phi(s_\tau)
+
\sigma_\phi(s_\tau)\odot\epsilon
\right),
\qquad
\epsilon\sim\mathcal N(0,I).
$$

### 11.1 7-DoF 机械臂示例

若动作定义为 7 个关节速度：

$$
a_t=[\dot q_1,\ldots,\dot q_7]\in\mathbb R^7,
$$

则 Actor 输出：

$$
\mu_\phi(s_t)\in\mathbb R^7,
\qquad
\sigma_\phi(s_t)\in\mathbb R_+^7,
$$

采样噪声：

$$
\epsilon\sim\mathcal N(0,I_7),
$$

最终：

$$
a_t=	anh(\mu+\sigma\odot\epsilon)\in[-1,1]^7.
$$

随后映射到物理动作范围。例如第 $i$ 个关节速度上限为 $v_i^{\max}$：

$$
\dot q_i^{\mathrm{cmd}}=v_i^{\max}a_{t,i}.
$$

若动作接口是末端执行器 6D 位姿增量加 1D 夹爪，则动作仍可能是 7 维；若是 7 个关节加夹爪，则可能是 8 维。高斯维度应与所定义的连续动作维度一致，而不是机械臂“自由度”这个名称自动决定。

论文采用的是对角高斯，相当于在给定潜状态下让每个动作维度的噪声条件独立：

$$
\Sigma=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_D^2).
$$

它计算简单、便于重参数化，但不能直接表达复杂多峰或强相关动作分布。高斯不是连续控制的唯一选择，也可使用 Beta、混合分布、扩散模型或流模型。

### 11.2 为什么使用 tanh？

未经变换的高斯取值范围为 $(-\infty,+\infty)$，而机器人动作通常有界。tanh 将样本压缩到 $[-1,1]$，再按关节速度、力矩或位姿增量的安全范围进行缩放。

实际机器人还应额外处理：

- 速度、加速度和力矩限制；
- 关节位置边界；
- 控制频率与 action repeat；
- 低层伺服控制器；
- 碰撞和工作空间约束；
- 紧急停止及安全过滤。

---

## 12. LLM 中的 RL 是否需要高斯分布？

通常不需要，因为标准 LLM 的动作是离散 token id：

$$
a_t\in\{1,2,\ldots,|\mathcal V|\}.
$$

策略输出 vocabulary logits：

$$
\pi_\phi(a_t=i\mid s_t)
=
\operatorname{softmax}(z_t)_i,
$$

并从 Categorical 分布中采样 token，或者选择最大概率 token。

在 RL 中，离散 token 的 policy gradient 形式是：

$$
\nabla_\phi J
\approx
\sum_t
\nabla_\phi\log\pi_\phi(a_t\mid s_t)A_t.
$$

因此：

| 动作空间 | 常见策略分布 |
|---|---|
| 连续机器人控制 | Gaussian、Beta、Diffusion、Flow 等 |
| LLM token | Categorical / Softmax |
| 离散化机器人 action token | Categorical / Softmax |
| 连续动作头的 VLA | 连续分布或连续生成模型 |

原则不是“RL 必须用高斯”，而是：

> 策略分布应与动作空间的数据类型和结构匹配。

若把连续机器人动作离散成 bins 或 action tokens，也可以像 LLM 一样做分类；代价是量化误差、动作维数增大后的组合爆炸，以及精细控制能力可能下降。

---

## 13. Dreamer 与 PlaNet 的区别

Dreamer 沿用了 PlaNet 风格的 RSSM 世界模型，但行为生成方式不同。

| 项目 | PlaNet | Dreamer |
|---|---|---|
| 世界模型 | RSSM | RSSM |
| 动作选择 | 每个真实时间步在线规划，如 CEM | Actor 一次前向推理 |
| Actor | 无 | 有 |
| Value Model | 无 | 有 |
| 长期回报 | 受有限规划 horizon 影响 | 使用 $V_\lambda$ 和 bootstrap |
| 训练策略 | 在线搜索，不保存为参数化策略 | 在潜空间想象中训练策略 |
| 部署计算 | 较高 | 较低 |

Dreamer 可以理解为把反复在线规划的计算“摊销”到 Actor 中。训练完成后，执行动作不再需要每一步运行大量候选动作序列搜索。

---

## 14. 概念性训练伪代码

下面的伪代码强调计算图关系，不代表作者代码的逐行复现。

```python
# A. 世界模型更新
obs, actions, rewards = replay.sample(batch_size=B, length=L)

posterior_states, prior_states = rssm.observe(obs, actions)
obs_dist = observation_model(posterior_states)
reward_dist = reward_model(posterior_states)

loss_obs = -obs_dist.log_prob(obs).mean()
loss_reward = -reward_dist.log_prob(rewards).mean()
loss_kl = kl_divergence(posterior_states, prior_states).mean()
loss_model = loss_obs + loss_reward + beta * loss_kl

optimizer_theta.zero_grad()
loss_model.backward()
optimizer_theta.step()


# B. 在 posterior states 上启动潜空间想象
s = sample_start_states(posterior_states)
imagined_states = []
imagined_rewards = []
imagined_values = []

for _ in range(H):
    a = actor.rsample(s)             # 重参数化连续动作
    s = transition.rsample(s, a)     # 不读取真实未来观测
    r = reward_model.mean(s)
    v = value_model(s)

    imagined_states.append(s)
    imagined_rewards.append(r)
    imagined_values.append(v)

returns = lambda_return(
    imagined_rewards,
    imagined_values,
    gamma=0.99,
    lambda_=0.95,
)

# C. Actor 更新：允许梯度穿过 action、transition、reward 和 value
loss_actor = -returns.mean()
optimizer_phi.zero_grad()
loss_actor.backward()
optimizer_phi.step()

# D. Value 更新：目标停止梯度
predicted_values = value_model(stop_gradient(imagined_states))
loss_value = 0.5 * ((predicted_values - returns.detach()) ** 2).mean()
optimizer_psi.zero_grad()
loss_value.backward()
optimizer_psi.step()
```

实现时需要特别注意：

- Actor 更新期间不应更新世界模型参数；
- 但不能把世界模型前向过程完全放入 `no_grad`；
- 否则 Actor 无法得到动作通过 Dynamics 影响未来状态的梯度。

---

## 15. 论文实验结果

### 15.1 连续视觉控制

论文在 DeepMind Control Suite 的 20 个视觉控制任务上评估，输入为 $64\times64\times3$ 图像，任务包含：

- 稀疏奖励；
- 接触动力学；
- 多自由度控制；
- 三维场景；
- 1 至 12 维连续动作。

论文报告：

- Dreamer 在 $5\times10^6$ 个环境步后平均得分约为 **823**；
- PlaNet 在相同环境步数下约为 **333**；
- D4PG 在 $10^8$ 个环境步后约为 **786**。

该比较表明，在论文的实验协议下，Dreamer兼顾了 PlaNet 的数据效率与更高的最终表现。

### 15.2 长时间信用分配

论文第 4 页图 4 和第 7 页图 7 表明：

- 只最大化 imagination horizon 内 Reward 的 Actor 容易短视；
- PlaNet 的有限 horizon 在线规划也会受到类似限制；
- 加入 Value bootstrap 后，Dreamer 对想象长度更稳健，并能解决 Acrobot、Hopper 等需要较长时间信用分配的任务。

### 15.3 表示学习目标

论文第 8 页图 8 和附录图 11 比较了：

1. 图像重建；
2. 对比估计；
3. 仅 Reward 预测。

结果显示：

- 图像重建在大多数任务上最好；
- 对比学习能解决约一半任务；
- 仅预测 Reward 在这些实验中不充分。

这不意味着所有后续世界模型都必须重建像素，而是说明在该论文的数据规模、网络结构和任务设置下，稠密的图像重建信号更有利于学习可控潜状态。

### 15.4 计算开销

论文报告在单张 Nvidia V100 和 10 个 CPU 核上：

- Dreamer 每 $10^6$ 环境步约 3 小时；
- PlaNet 在线规划约 11 小时；
- D4PG 达到相近表现所需训练约 24 小时。

这些数字依赖当时的实现、硬件和实验协议，主要用于说明参数化 Actor 相比每步在线搜索的计算优势。

---

## 16. 主要超参数与实现细节

根据论文附录 A：

| 项目 | 设置 |
|---|---:|
| Batch size | 50 条序列 |
| Sequence length | 50 |
| 随机潜变量维度 | 30 维对角高斯 |
| Imagination horizon | 15 |
| 折扣因子 $\gamma$ | 0.99 |
| $\lambda$ | 0.95 |
| 世界模型学习率 | $6\times10^{-4}$ |
| Actor 学习率 | $8\times10^{-5}$ |
| Value 学习率 | $8\times10^{-5}$ |
| Optimizer | Adam |
| 初始随机 episode | 5 |
| Action repeat | 2 |
| 连续动作探索噪声 | $\mathcal N(0,0.3)$ |
| KL free nats | 3 |

这些参数在论文的连续控制任务间保持统一，是其泛化性论证的一部分。

---

## 17. 阅读和实现时需要特别注意的事项

### 17.1 “纯粹通过想象学习行为”不等于不使用真实环境

Actor 和 Value 的主要训练数据来自潜空间想象，但世界模型必须用真实轨迹训练，Agent 也会不断回到真实环境收集新数据。

### 17.2 固定世界模型参数不等于切断梯度

Actor 更新期间，世界模型不执行 optimizer step，但梯度仍需穿过其前向计算。否则无法形成 analytic value gradient。

### 17.3 Value 解决 horizon 截断，不自动解决模型误差

Value bootstrap 可以估计 $H$ 步之后的回报，但不能自动消除：

- Transition 预测错误；
- Reward Model 误差；
- Actor 利用模型漏洞；
- 想象状态偏离真实数据分布。

### 17.4 想象越长不一定越好

较长 rollout 提供更长的显式 Reward 链，但也积累更多模型误差。Dreamer 选择较短 $H$，再用 Value 估计远期尾部。

### 17.5 图像重建好不等于控制一定好

Decoder 可能准确重建背景，却遗漏小型任务相关物体。评估世界模型时应关注：

- 奖励相关状态是否被保留；
- 动作条件动力学是否准确；
- 长时间潜状态预测是否稳定；
- Actor 是否会利用模型偏差。

### 17.6 Reward Model 误差会直接误导策略

如果 Reward Model 对某些虚假潜状态给出过高奖励，Actor 可能主动寻找这些模型漏洞。实际应用可考虑：

- 模型集成与不确定性惩罚；
- 限制想象状态远离数据分布；
- 更短 rollout；
- 保守策略优化；
- 周期性真实环境校正。

### 17.7 原始 Dreamer 不是通用 VLA

该论文主要研究：

- 单任务控制；
- 手工定义 Reward；
- 固定动作空间；
- 无语言条件；
- 低分辨率视觉输入。

它对机器人和 VLA 的重要启发是“如何在学习到的潜空间动力学中训练连续策略”，而不是直接提供一个语言驱动通用机器人模型。

### 17.8 对角高斯动作存在表达能力限制

对于多峰动作、强耦合关节或接触丰富的任务，单峰对角高斯可能过于简单。可考虑动作 chunk、混合密度、扩散或流模型，但这些替代方案需要重新设计 Actor 梯度和世界模型训练方式。

---

## 18. 核心问题的压缩回答

### 18.1 Algorithm 1 中如何更新 $\theta$？

从 replay 中采样真实序列，联合最小化：

$$
\text{观测负对数似然}
+
\text{Reward 负对数似然}
+
\beta\,\text{posterior-prior KL},
$$

并对整个世界模型参数做梯度下降。

### 18.2 论文中的 `while` 表达什么？

表达对比：“A3C/PPO 使用 Value 作为 baseline，而 Dreamer 让 Actor 梯度穿过 Value Model。”两边都有 Value，但 Value 在 Actor 更新中的计算图角色不同。

### 18.3 为什么需要 Reward Model？

真实环境会给真实轨迹 Reward，但不会给反事实想象轨迹 Reward。Dreamer若要在想象中评价动作，就必须预测即时奖励。

### 18.4 Reward 与 Value 谁由谁产生？

Value 是未来 Reward 的折扣累计期望；不是 Reward 由 Value 转化而来。

### 18.5 为什么想象时不一直使用 $o_t$？

Actor 选择的新动作对应的真实未来观测尚不存在。Replay 中的观测属于旧动作，不能用于反事实轨迹。预测图像再编码也没有新增真实信息，且成本更高。

### 18.6 7-DoF 机械臂的高斯是几维？

若动作向量是 7 维，则均值、标准差和采样噪声通常都是 7 维；若动作定义包含夹爪或采用其他控制接口，维度随动作定义改变。

### 18.7 LLM 的 RL 是否需要高斯？

标准 token action 是离散的，使用 Categorical/Softmax，不需要高斯。只有策略输出连续动作时才需要连续分布或连续生成模型。

---

## 19. 结论

Dreamer 的技术贡献不只是“有一个世界模型”，而是把以下几件事连成一个可训练的计算图：

$$
\text{Actor}
\rightarrow
\text{Action}
\rightarrow
\text{Latent Dynamics}
\rightarrow
\text{Reward/Value}
\rightarrow
\text{Multi-step Return}.
$$

其最关键的三点是：

1. **从真实 posterior states 出发，在潜空间生成反事实轨迹；**
2. **用 Reward Model 提供想象中的即时奖励，用 Value Model补足有限 horizon 之外的长期价值；**
3. **通过重参数化和可微世界模型，将多步价值梯度直接反传到 Actor。**

因此，Dreamer 与 A3C/PPO 的本质差异不在于“有没有 Critic”，而在于 Actor 是否利用可微模型获得 pathwise gradient；与 PlaNet 的本质差异不在于“有没有世界模型”，而在于是否将在线规划转化为一个在想象中训练、部署时直接执行的参数化策略。

## 相关笔记

- [[Visual Foresight|Visual Foresight]]
- [[PlaNet 论文概述|PlaNet 论文概述]]
- [[DayDreamer论文综述与阅读重点|DayDreamer]]
- [[DreamerV3_技术报告|DreamerV3]]
- [[UniPi_技术总结|UniPi]]
- [[PPO|PPO]]
- [[SAC_PPO_compare|SAC vs PPO]]
- [[RL/opd_on_policy_distillation_知识笔记|OPD / On-Policy Distillation]]
- [[Pi0_7_technical_report|π0.7 技术报告]]
- [[RDT-1B|RDT-1B]]

---

## 参考文献

Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). *Dream to Control: Learning Behaviors by Latent Imagination*. International Conference on Learning Representations (ICLR 2020).
