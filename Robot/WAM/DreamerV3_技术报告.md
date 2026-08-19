---
title: DreamerV3 技术报告
type: paper_note
topic: model_based_reinforcement_learning
status: mature
importance: high
updated: 2026-08-13
tags:
  - dreamerv3
  - world-model
  - model-based-rl
  - rssm
  - actor-critic
  - distributional-critic
  - reinforcement-learning
---

# DreamerV3 技术报告：基于世界模型的通用强化学习

> **报告范围**
>
> 基于 Hafner 等人的《Mastering Diverse Domains through World Models》批注版，重点吸收第 1、3-7 页的高亮内容。
>
> 综合对话中对 RSSM、KL balancing、reward/value、policy gradient、return normalization、straight-through estimator 与 critic replay loss 的逐步推导。
>
> 本报告是技术解读与实现分析，不包含新的实验结果。


版本 1.0

2026 年 7 月 14 日

原论文：Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap

## 摘要

DreamerV3 是一个基于离散潜变量世界模型的 model-based actor-critic。它从真实交互中学习 Recurrent State-Space Model（RSSM），显式预测潜状态转移、即时奖励、episode continuation 与观测重建，再从 replay state 出发在潜空间中生成 imagined trajectories，以 bootstrapped lambda-return 训练 distributional critic，并用统一的 REINFORCE 目标训练离散或连续 actor。论文真正的核心并非单一新模块，而是一组面向跨领域稳定性的组合设计：双向 stop-gradient 的 KL balancing、1 nat free bits、1% uniform mixture、symlog/symexp twohot、分位数 return normalization、critic EMA 与零初始化、AGC 和 LaProp。

本报告重点澄清五类常见误区：第一，reward model 与 value model 并非替代关系，前者描述环境的局部反馈，后者描述当前策略的长期回报；第二，h_t 预测的是 z_t 的分布而非具体 realization，z_t 表示当前观测带来的随机创新；第三，straight-through estimator 与 REINFORCE 都可处理离散采样，但前者是有偏低方差的 surrogate pathwise gradient，后者是无偏高方差的 score-function estimator；第四，critic replay loss 不要求 imagination 与 replay 轨迹逐步对齐，它仅用每个 replay state 的 imagination return 作为 bootstrap annotation；第五，Dreamer 的 return normalization 只缩小大信号而不放大小信号，从而在稀疏奖励尚未“触手可及”时让 entropy 保持相对主导。

> **一句话结论**
>
> DreamerV3 保留了 policy gradient 的基本优化逻辑，但把 actor-critic 的训练数据生成器从真实环境扩展为一个经过鲁棒训练的潜空间模拟器；其成功主要来自“稳定地学习和使用世界模型”，而不是把在线规划做得更复杂。


## 目录与阅读路线

> [[#1. 论文定位与核心贡献|1. 论文定位与高亮主线]]
>
> [[#2. DreamerV3 全局模型与算法流程|2. 全局模型与真实/想象闭环]]
>
> [[#3. RSSM：确定性记忆与随机状态|3. RSSM：h、z、prior 与 posterior]]
>
> [[#4. 世界模型损失与离散潜变量梯度|4. 世界模型损失与离散潜变量梯度]]
>
> [[#5. Reward、Continuation 与 Distributional Critic|5. Reward、continuation 与 critic]]
>
> [[#6. Actor：离散与连续动作的 Policy Gradient|6. Actor：离散/连续 Policy Gradient]]
>
> [[#7. Return normalization、Entropy 与探索|7. Return normalization、entropy 与探索]]
>
> [[#8. 鲁棒预测、网络与优化器|8. 鲁棒预测、网络与优化器]]
>
> [[#9. 实验结果、消融与扩展性|9. 实验结果、消融与扩展性]]
>
> [[#10. 技术评价、局限与适用边界|10. 技术评价、局限与适用边界]]
>
> [[#11. 复现与调试检查表|11. 复现与调试检查表]]
>
> [[#附录 A：符号表|附录 A-C：符号、伪代码、常见误解]]

### 批注版高亮的技术映射

| **高亮页** | **高亮关键词**                                                                        | **本报告中的技术解释**                                             |
|------------|---------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| 页 1       | general algorithm；model of the environment；imagining future scenarios               | 统一配置、世界模型与想象训练是论文主命题。                         |
| 页 3       | discrete/stochastic representations；world model / critic / actor；signal magnitudes  | RSSM 结构及跨尺度鲁棒损失是算法核心。                              |
| 页 4       | reward、continuation；prediction / dynamics / representation loss                     | 世界模型不仅预测状态，还要给 imagined rollout 提供任务与终止语义。 |
| 页 5       | free bits；1% uniform mixture；abstract trajectories                                  | KL 约束必须避免信息坍塌和 categorical 数值尖峰。                   |
| 页 6       | imagined/replay critic loss；EMA；zero initialization；reward scale/frequency；P5-P95 | critic 与 actor 的稳定性依赖目标平滑和 return 尺度控制。           |
| 页 7       | symlog；symexp twohot；reward predictor and critic                                    | 用变换和分类式回归解耦目标数值大小与梯度大小。                     |

> *高亮内容来自用户提供的论文批注；为避免逐字重复，表中采用主题化概括。*

## 1. 论文定位与核心贡献

DreamerV3 面向的不是某一个 benchmark，而是强化学习算法在跨领域迁移时的“配置脆弱性”：视觉与向量输入、离散与连续动作、稀疏与密集奖励、2D 与 3D 环境，以及相差多个数量级的 reward/return，往往迫使研究者重新调节 loss scale、entropy coefficient、regularization 和 optimizer。论文试图用一组固定的核心超参数覆盖超过 150 个任务。[1, pp. 1–2]

其贡献可以分为三层。第一层是既有 Dreamer 范式：学习 latent world model，在模型中想象未来，再训练 actor 和 critic。第二层是世界模型表征：离散 stochastic latent、RSSM 和记忆状态。第三层，也是 V3 最关键的贡献，是一整套鲁棒性工程，使上述范式能够跨任务工作，而不依赖逐领域调参。


## 2. DreamerV3 全局模型与算法流程

> [!figure] 论文原始模型结构图
> ![[attachments/paper-figures/dreamerv3-training-architecture.png]]
> DreamerV3 的完整训练结构，左侧为世界模型学习，右侧为想象轨迹中的 Actor-Critic 学习。原图来自 [Mastering Diverse Domains through World Models（arXiv:2301.04104）](https://arxiv.org/abs/2301.04104)，由论文源文件的两个原始面板按论文布局合成。

DreamerV3 可以先拆成两部分：

```text
DreamerV3
│
├── World Model
│   ├── Encoder / Posterior
│   ├── Deterministic recurrent dynamics
│   ├── Prior
│   ├── Observation decoder
│   ├── Reward predictor
│   └── Continuation predictor
│
└── Behavior Learning
    ├── Actor
    └── Distributional Critic
```

World Model 负责回答：

> 给定历史、动作和当前观测，当前 latent state 是什么？如果接下来采取某个动作，未来
> latent、reward 和 episode continuation 会怎样？

Behavior Learning 负责回答：

> 在这个已经学到的 latent environment 中，哪些 action distribution 能带来更高的长期回报？

因此，DreamerV3 的整体闭环是：

```text
真实环境提供数据
        ↓
World Model 学习 latent dynamics / reward / continuation
        ↓
从 replay latent state 开始做 latent imagination
        ↓
用 imagined return 训练 Critic 和 Actor
        ↓
更新后的 Actor 继续与真实环境交互
```

### 2.1 RSSM state：$h_t$、$z_t$ 与 $s_t$

DreamerV3 的完整 model state 写成：

$$
s_t=(h_t,z_t).
$$

其中确定性 recurrent state 累积历史、上一个 latent 和上一个动作：

$$
h_t
=
f_\phi
\left(
h_{t-1},
z_{t-1},
a_{t-1}
\right).
$$

真实观测 $x_t$ 到来时，posterior 产生当前 stochastic latent：

$$
z_t
\sim
q_\phi
\left(
z_t\mid h_t,x_t
\right).
$$

而在没有未来观测的 imagination 中，只能使用 prior：

$$
\hat z_t
\sim
p_\phi
\left(
z_t\mid h_t
\right).
$$

可以把它们压缩成：

$$
\boxed{h_t=\text{由过去推出来的现在}}
\qquad
\boxed{z_t=\text{由当前观测修正后的具体现在}}.
$$

后文第 3 节会详细解释为什么需要同时保留这两种状态，以及为什么 prior 和 posterior
不能混用。

### 2.2 Phase A：真实环境交互

真实交互阶段是真正获得新数据的地方。每一步的因果顺序是：

```text
真实 observation x_t
        ↓
Posterior q(z_t | h_t, x_t)
        ↓
model state s_t=(h_t,z_t)
        ↓
Actor πθ(a_t | s_t)
        ↓
sample action a_t
        ↓
真实 environment
        ↓
x_{t+1}, r_t, terminal d_t
        ↓
Replay Buffer
```

动作由当前 Actor 分布采样：

$$
a_t\sim\pi_\theta(\cdot\mid s_t).
$$

环境返回：

$$
(x_{t+1},r_t,d_t)
\leftarrow
\operatorname{env.step}(a_t).
$$

训练中常用 continuation 表示“下一步是否仍然有效”：

$$
c_t=1-d_t.
$$

实际实现还需要区分真正的 terminal 和 time-limit truncation，不能简单把所有 episode
结束都当成同一种 $d_t$。这批 $(x_t,a_t,r_t,c_t)$ 序列进入 replay，成为后续两个训练阶段
共同的真实数据来源。

### 2.3 Phase B：World Model Learning

从 replay 中采样长度为 $T$ 的真实序列：

$$
(x_{t_0:t_0+T-1},
a_{t_0-1:t_0+T-2},
r_{t_0:t_0+T-1},
c_{t_0:t_0+T-1}).
$$

这里特意把动作区间向前写一位，因为要根据 $a_{t-1}$ 才能更新 $h_t$。

#### 2.3.1 Replay chunk 的 recurrent state 缓存

随机 chunk 若从长 episode 的中间位置开始，就需要前一位置的 RSSM state：

$$
(h_{t_0-1},z_{t_0-1}).
$$

为避免每次都从 episode 起点重算，replay 可以缓存各时间步的 recurrent entry。缓存状态
即使由旧参数得到，也只作为 chunk 的 boundary；从 $t_0$ 开始，chunk 内部始终用当前
World Model 参数和真实 observation/action 重新 rollout：

```text
读取 cached (h_{t0-1}, z_{t0-1})
        ↓
当前模型重新 rollout chunk
        ↓
计算 world model loss，并刷新本 chunk 的缓存
```

因此不需要每次更新后重算整个 replay buffer，而是“采样到哪个 chunk，就顺便刷新哪个
chunk”。核心是：**缓存只负责初始化，chunk 内部状态由当前模型重算。**

之后的每一步才是：

```text
h_{t-1}, z_{t-1}, a_{t-1}
              ↓
        deterministic RSSM update
              ↓
             h_t
              │
       x_t ───┤
              ↓
      qφ(z_t | h_t, x_t)
              ↓
             z_t
              ↓
        s_t=(h_t,z_t)
```

随后，$s_t$ 同时送入多个 prediction head：

```text
s_t ──→ observation decoder     → x̂_t
  │
  ├──→ reward predictor         → r̂_t
  │
  └──→ continuation predictor   → ĉ_t
```

World Model 的目标可以统一写成：

$$
\mathcal L_{\mathrm{world}}
=
\mathbb E_q
\left[
\sum_t
\left(
\mathcal L_{\mathrm{pred},t}
+
\mathcal L_{\mathrm{dyn},t}
+
0.1\mathcal L_{\mathrm{rep},t}
\right)
\right].
$$

其中 prediction loss 让 latent 能解释真实数据：

$$
\mathcal L_{\mathrm{pred},t}
=
-\log p_\phi(x_t\mid s_t)
-\log p_\phi(r_t\mid s_t)
-\log p_\phi(c_t\mid s_t).
$$

另外两项用 stop-gradient 控制 posterior 与 prior 的学习方向：

$$
\mathcal L_{\mathrm{dyn},t}
=
\max
\left(
1,
D_{\mathrm{KL}}
\left[
\operatorname{sg}\left(q_\phi(z_t\mid h_t,x_t)\right)
\middle\|
p_\phi(z_t\mid h_t)
\right]
\right),
$$

$$
\mathcal L_{\mathrm{rep},t}
=
\max
\left(
1,
D_{\mathrm{KL}}
\left[
q_\phi(z_t\mid h_t,x_t)
\middle\|
\operatorname{sg}\left(p_\phi(z_t\mid h_t)\right)
\right]
\right).
$$

这一步的核心不是训练 Actor，而是让 World Model 学会一个可以在 latent space 中运行的
环境近似：它既要能解释真实 observation，也要能预测 reward、continuation 和未来
stochastic state。

### 2.4 Phase C：Latent Imagination 与 Behavior Learning

World Model 更新后，从 replay 序列中得到的 posterior states 作为 imagination 的起点。
这些起点是“真实数据锚定的 latent states”，但起点之后不再读取未来真实 observation。

对每个 imagination step：

$$
a_\tau\sim\pi_\theta(\cdot\mid s_\tau),
$$

$$
h_{\tau+1}
=
f_\phi
\left(
h_\tau,z_\tau,a_\tau
\right),
\qquad
\hat z_{\tau+1}
\sim
p_\phi
\left(
z_{\tau+1}\mid h_{\tau+1}
\right).
$$

得到 imagined state 后，由 prediction heads 提供训练所需的环境信号：

$$
\hat r_\tau
=
r_\phi(s_\tau),
\qquad
\hat c_\tau
=
c_\phi(s_\tau).
$$

重复 $H$ 步，形成：

```text
s_t, a_t, r̂_t, ĉ_t
s_{t+1}, a_{t+1}, r̂_{t+1}, ĉ_{t+1}
...
s_{t+H}
```

再用 continuation 加权的 lambda-return 连接短期 reward 与 horizon 外的 critic：

$$
R_\tau^\lambda
=
\hat r_\tau
+
\gamma\hat c_\tau
\left[
(1-\lambda)\bar v_\psi(s_{\tau+1})
+
\lambda R_{\tau+1}^\lambda
\right].
$$

其中 $\bar v_\psi$ 是慢速 target critic；terminal 或 predicted continuation 为零时，
后续 bootstrap 被切断。

这条 imagined trajectory 同时服务于两个 learner：

```text
imagined trajectory
        ↓
   lambda-return Rλ
      /          \
     ↓            ↓
Distributional  Advantage
 Critic target       ↓
                Actor: -A log π
```

因此 Phase B 和 Phase C 的分工是：

> Phase B 学会 world model；Phase C 把这个已经学到的 world model 当作 latent simulator，
> 用它生成 Actor 和 Critic 的训练数据。

### 2.5 Real trajectory 与 imagined trajectory

| 项目 | Real trajectory | Imagined trajectory |
|---|---|---|
| observation $x_t$ | 真实环境提供 | 没有未来真实 observation |
| stochastic state | posterior $q_\phi(z_t\mid h_t,x_t)$ | prior $p_\phi(z_t\mid h_t)$ |
| action | Actor sample | Actor sample |
| reward | environment 返回 $r_t$ | reward head 预测 $\hat r_t$ |
| continuation | terminal signal $c_t$ | continuation head 预测 $\hat c_t$ |
| next state | 新 observation 会重新修正 posterior | 完全由 RSSM prior rollout |
| 主要用途 | 提供真实锚点并训练 World Model | 训练 Actor 与 Critic |

最容易混淆的两条路径是：

真实阶段：

$$
h_t
\rightarrow
q_\phi(z_t\mid h_t,x_t)
\rightarrow
z_t.
$$

想象阶段：

$$
h_t
\rightarrow
p_\phi(z_t\mid h_t)
\rightarrow
\hat z_t.
$$

因此，“想象未来”不是先生成未来像素，再把像素重新编码成 state；它直接在
$s_t=(h_t,z_t)$ 的 latent space 中 rollout。Decoder 主要为 World Model 提供 observation
监督，通常不需要参与 imagination 的每一步。

### 2.6 各模块的输入、输出与监督目标

| 模块 | 输入 | 输出 | 训练目标 / supervision | imagination 中是否使用 |
|---|---|---|---|---|
| Encoder / Posterior | $h_t,x_t$ | $q_\phi(z_t\mid h_t,x_t)$ | observation、representation loss | 只用于真实起点 |
| Sequence Model | $h_{t-1},z_{t-1},a_{t-1}$ | $h_t$ | World Model loss | 是 |
| Prior | $h_t$ | $p_\phi(z_t\mid h_t)$ | 与 posterior 的 KL 对齐 | 是 |
| Observation Decoder | $s_t$ | $p_\phi(x_t\mid s_t)$ | 真实 observation | 通常不需要 |
| Reward Head | $s_t$ | $\hat r_t$ | 真实 reward | 是 |
| Continuation Head | $s_t$ | $\hat c_t$ | terminal / continuation | 是 |
| Actor | $s_t$ | $\pi_\theta(a_t\mid s_t)$ | REINFORCE / policy gradient | 是 |
| Distributional Critic | $s_t$ | return distribution | imagined $R_t^\lambda$ 与辅助 replay target | 是 |

这张表也说明了几个边界：Reward Head 不能被 Critic 替代，因为前者预测环境局部反馈，
后者估计当前策略下的长期回报；Decoder 训练时重要，但不是 imagination 中产生下一状态
所必需的模块；Prior 是未来无 observation 时真正执行 latent rollout 的模型。

### 2.7 参数更新与 Gradient Flow

DreamerV3 中至少要区分三种不同的梯度处理：

#### World Model update

$$
\mathcal L_{\mathrm{world}}
\rightarrow
\phi.
$$

其中：

```text
L_dyn：stop posterior → prior / dynamics 追 posterior
L_rep：stop prior     → posterior 适度靠近 prior
ST latent：前向使用 hard one-hot，反向沿 soft probability 传梯度
```

#### Critic update

Critic 用 $R_t^\lambda$ 作为 target，target 本身不应把梯度反传回 Actor 或 World Model：

$$
\mathcal L_{\mathrm{critic}}
=
\mathcal L_{\mathrm{dist}}
\left(
v_\psi(s_t),
\operatorname{sg}\left(R_t^\lambda\right)
\right)
+
\text{辅助 replay critic loss}.
$$

慢速 critic、EMA 和 zero initialization 用来减少 bootstrap target 的漂移。

#### Actor update

DreamerV3 使用 score-function estimator，而不是让 Actor 沿着 imagined dynamics 做
dynamics gradient：

$$
A_t
=
R_t^\lambda-v_\psi(s_t),
$$

$$
\mathcal L_{\mathrm{actor}}
=
-\operatorname{sg}
\left(
\frac{A_t}{\max(1,S)}
\right)
\log\pi_\theta(a_t\mid s_t)
-
\eta H
\left[
\pi_\theta(\cdot\mid s_t)
\right].
$$

因此：

```text
World Model φ  ← L_world
Critic ψ       ← distributional critic loss
Actor θ        ← -stop_gradient(Advantage) · log π - entropy
```

World Model 是 Actor 的 imagined sample generator，而不是 Actor loss 的可微优化路径。
Actor 使用的是“这次 sampled action 比当前策略平均水平好还是差”的信号；它不需要对
$a_t\rightarrow s_{t+1}\rightarrow\hat r\rightarrow R^\lambda$ 做 pathwise 反传。

### 2.8 一张完整的 DreamerV3 主线图

```text
                         REAL ENVIRONMENT
                                │
                     x_t, a_t, r_t, terminal
                                │
                                ▼
                         Replay Buffer
                                │
                       sample real sequences
                                │
                                ▼
┌──────────────────── WORLD MODEL LEARNING ───────────────────┐
│                                                            │
│ x_t → Posterior q(z_t|h_t,x_t) → z_t                       │
│                 ▲                       │                  │
│                 │                       ▼                  │
│ h_t=f(h_{t-1},z_{t-1},a_{t-1})      s_t=(h_t,z_t)           │
│                 │                    /    |     \           │
│            Prior p(z_t|h_t)      decoder reward continue   │
│                                                            │
│             L_pred + L_dyn + 0.1 L_rep                     │
└────────────────────────────┬───────────────────────────────┘
                             │
                    posterior latent states
                             │
                             ▼
┌──────────────────── LATENT IMAGINATION ────────────────────┐
│                                                            │
│ s_t → Actor → a_t                                          │
│            ↓                                               │
│       RSSM + Prior                                         │
│            ↓                                               │
│         s_{t+1}                                             │
│        /       \                                            │
│    reward     continuation                                  │
│                                                            │
│                 repeat H steps                              │
└────────────────────────────┬───────────────────────────────┘
                             │
                       lambda-return
                       /              \
                      ▼                ▼
              Distributional Critic   Actor
                 target / value       REINFORCE
```

后面的第 3–8 节，分别是对这张总图中 RSSM、World Model loss、reward/continuation、
Critic、Actor、return normalization 和鲁棒预测组件的局部放大。这样阅读时，任何一个公式
都可以先回答两个问题：它属于 World Model 还是 Behavior Learning？它使用真实 observation
还是 latent imagination？

> **模型偏差的代价**
>
> 想象轨迹可能并不存在于真实环境中，Actor 还可能利用 World Model 的系统性错误。Dreamer
> 用 replay state 作为真实锚点、采用较短 imagination horizon、持续加入新数据并依赖
> Critic bootstrap 缓解风险，但并未从理论上消除 model bias。


## 3. RSSM：确定性记忆与随机状态

### 3.1 时间顺序与动作依赖

$$
h_t
=
f_\phi
\left(
h_{t-1},
z_{t-1},
a_{t-1}
\right).
$$

真实观测可用时，posterior 根据当前观测采样：

$$
z_t
\sim
q_\phi
\left(
z_t\mid h_t,x_t
\right)
\qquad
\text{（真实观测 / posterior）}.
$$

想象阶段没有未来观测，prior 只根据 recurrent state 采样：

$$
\hat z_t
\sim
p_\phi
\left(
z_t\mid h_t
\right)
\qquad
\text{（想象 / prior）}.
$$

Dynamics predictor 表面上只以 $h_t$ 为条件，但动作并未缺失：$a_{t-1}$ 已经通过
sequence model 写入 $h_t$。时间索引上，$a_{t-1}$ 造成从时刻 $t-1$ 到 $t$ 的转移；
当前动作 $a_t$ 影响的是 $h_{t+1}$ 和 $z_{t+1}$。

### 3.2 为什么既需要 h_t，又需要 z_t

$h_t$ 是确定性的历史摘要，可理解为“在看到当前 $x_t$ 之前，根据过去对现在形成的预测上下文”；$z_t$ 是当前时刻的 stochastic innovation，表示当前观测揭示的具体随机分支、隐藏变量或不可由历史唯一决定的信息。$h_t$ 能输出 $p(z_t\mid h_t)$，但只能给出一个分布，不能确定本次 realization。

> **例子：开门后的随机分支**
>
> 历史和动作只告诉模型“门已打开”，prior 可能给出 60% 有敌人、40% 无敌人。
>
> 当前图像 x_t 显示实际有敌人，posterior 因此把 z_t 更新到“有敌人”分支。
>
> 随后 z_t 与动作 a_t 一起进入下一次 recurrent update，使新信息写入 h_{t+1}。


### 3.3 z_t 是否真的是采样结果

是。真实观测阶段，$z_t$ 从 posterior 分布中采样：

$$
z_t
\sim
q_\phi
\left(
z_t\mid h_t,x_t
\right),
$$

而 imagination 阶段从 prior 分布中采样：

$$
\hat z_t
\sim
p_\phi
\left(
z_t\mid h_t
\right).
$$

DreamerV3 的 stochastic state 不是一个单独的 categorical，而是由多个 categorical
变量组成：

$$
z_t
=
\left(
z_t^{(1)},
\ldots,
z_t^{(N)}
\right),
\qquad
z_t^{(i)}
\sim
\operatorname{Categorical}
\left(
p_\phi^{(i)}(\cdot\mid h_t)
\right).
$$

每个变量采样一个类别并表示为 one-hot，最后拼接成 stochastic state。因此，$z_t$ 是
一次具体采样得到的离散 realization，而不是概率向量本身；这种表示既保留多模态能力，
又能在 imagination 中快速采样。

## 4. 世界模型损失与离散潜变量梯度

$$
\mathcal L_{\mathrm{world}}
=
\mathbb E_q
\left[
\sum_t
\left(
\mathcal L_{\mathrm{pred},t}
+
\mathcal L_{\mathrm{dyn},t}
+
0.1\mathcal L_{\mathrm{rep},t}
\right)
\right].
$$

### 4.1 Prediction loss：重建、奖励与 continuation

$$
\mathcal L_{\mathrm{pred},t}
=
-\log p_\phi(x_t\mid s_t)
-\log p_\phi(r_t\mid s_t)
-\log p_\phi(c_t\mid s_t).
$$

这三个负对数似然共享“最大化条件概率”的形式，但实际分布与 loss 不必相同：向量观测可在 symlog 空间使用 squared error；reward 与 critic 最终使用 symexp twohot categorical loss；continuation 使用 Bernoulli logistic loss。统一写成 $-\log p$ 的好处，是把不同输出头都解释为条件概率模型。

### 4.2 为什么 Gaussian NLL 等价于 MSE

$$
p(x\mid s)
=
\mathcal N
\left(
x;
\mu_\phi(s),
\sigma^2 I
\right).
$$

$$
-\log p(x\mid s)
=
\frac{D}{2}\log\left(2\pi\sigma^2\right)
+
\frac{1}{2\sigma^2}
\left\|x-\mu_\phi(s)\right\|_2^2.
$$

当方差 $\sigma^2$ 固定时，第一项与参数无关，第二项只是平方误差的固定倍数，因此最小化 Gaussian negative log-likelihood 与最小化 MSE 有相同最优解。若 $\sigma$ 也由网络预测，则还会出现 $\log\sigma$ 项，loss 不再等价于普通 MSE。

### 4.3 Dynamics loss 与 Representation loss 为何拆开

$$
\mathcal L_{\mathrm{dyn},t}
=
\max
\left(
1,
D_{\mathrm{KL}}
\left[
\operatorname{sg}
\left(q_\phi(z_t\mid h_t,x_t)\right)
\middle\|
p_\phi(z_t\mid h_t)
\right]
\right).
$$

$$
\mathcal L_{\mathrm{rep},t}
=
\max
\left(
1,
D_{\mathrm{KL}}
\left[
q_\phi(z_t\mid h_t,x_t)
\middle\|
\operatorname{sg}
\left(p_\phi(z_t\mid h_t)\right)
\right]
\right).
$$

两项实际上在同一个总 loss 中一起优化；区别在于梯度流向。$\mathcal L_{\mathrm{dyn}}$ 固定 posterior，把它当作较有信息的 teacher，训练 prior/sequence model 去预测真实表示。$\mathcal L_{\mathrm{rep}}$ 固定 prior，只以较小权重要求 encoder 产生更可预测的表示。若只用一个普通 KL 同时更新 $q$ 和 $p$，两边可能一起向一个“很容易一致但几乎不含观测信息”的退化分布移动。

> **为什么权重不对称**
>
> $\beta_{\mathrm{dyn}}=1$：主要责任在 dynamics，要求它追上包含真实观测的 posterior。
>
> $\beta_{\mathrm{rep}}=0.1$：posterior 只需适度迁就 dynamics，避免为了可预测性而丢失关键细节。


### 4.4 Free bits 与 $1\%$ Unimix：分别防止什么

4.3 的 KL balancing 解决的是“**谁向谁学习**”：
$\mathcal L_{\mathrm{dyn}}$ 主要更新 prior，$\mathcal L_{\mathrm{rep}}$ 以较小权重约束 posterior。
4.4 再解决两个不同的稳定性问题：

这里的关键是区分“KL 的数值”和“KL 的单位”。DreamerV3 使用自然对数计算 KL：

$$
D_{\mathrm{KL}}^{(\mathrm{nat})}(q\|p)
=
\mathbb E_q
\left[
\ln\frac{q}{p}
\right].
$$

因此，Free bits 中的 $1\,\mathrm{nat}$ 就是 **KL 原始数值达到 $1.0$**；这里的 nat 不是额外乘上的系数。如果把同一个 KL 改用以 $2$ 为底的对数表示，则：

$$
1\,\mathrm{nat}
=
\frac{1}{\ln 2}\,\mathrm{bit}
\approx
1.443\,\mathrm{bit},
\qquad
1\,\mathrm{bit}
=
\ln 2\,\mathrm{nat}
\approx
0.693\,\mathrm{nat}.
$$

也就是说：在自然对数版本中，数值 $1.0$ 表示 $1$ nat；在二进制对数版本中，数值 $1.0$ 表示 $1$ bit。但对于同一个 KL 差异，$1$ nat 等价于约 $1.443$ bit，而不是两个不同的阈值。

- **Free bits 防止 posterior 被过度压缩。** posterior $q_\phi(z_t\mid h_t,x_t)$ 看到了真实观测，可能包含 prior 暂时预测不了但对重建有用的信息。如果 KL 从一开始就强迫 $q$ 与 $p$ 完全一致，最简单的结果是 posterior 主动丢掉这些信息，导致 latent collapse。

  对 KL 使用 free-bits 形式：

  $$
  \widetilde{K}_t
  =
  \max
  \left(
  1\,\mathrm{nat},
  D_{\mathrm{KL}}
  \left[
  q_\phi(z_t\mid h_t,x_t)
  \middle\|
  p_\phi(z_t\mid h_t)
  \right]
  \right).
  $$

  当 KL 小于 $1\,\mathrm{nat}$ 时，$\widetilde K_t$ 是常数，梯度为零；只有差异超过这个阈值，KL 才继续施加对齐压力。因此 free bits **不是把 KL 最大限制为 1 nat**，而是允许 posterior 和 prior 保留至少一小段暂时的不一致空间。

- **Unimix 防止 categorical 概率变成精确的 0 或 1。** 设网络输出的 softmax 概率为 $p$，类别数为 $C$，则实际使用的分布为：

  $$
  p_{\mathrm{mix}}
  =
  0.99p
  +
  0.01\frac{\mathbf 1}{C}.
  $$

  这样每个类别至少保留 $0.01/C$ 的概率，计算 $\log p$ 和 KL 时不会出现 $\log 0$，采样也不会过早锁死在某一个类别。Unimix 是对 categorical 分布的数值和支持集做保护，不是额外的 entropy loss。

两者的分工可以记成：

> **Free bits 控制 KL 约束“什么时候开始施压”；Unimix 保证 categorical 分布“不要失去概率支持”。**

它们并不重复：前者保留 latent 的信息容量，后者避免概率尖峰和数值异常。[1, pp. 4–5]

### 4.5 离散采样后的梯度：ST 与 REINFORCE 各自负责什么

$z_t$ 是从 categorical 分布采出的 one-hot 离散变量。采样操作本身不可微，因此需要专门处理梯度。但 DreamerV3 不是在同一位置二选一：

- World Model 需要把 observation/reward 等预测损失传回 latent encoder，使用 **straight-through estimator（ST）**；
- Actor 需要根据 sampled action 带来的 return 调整动作概率，使用 **REINFORCE / score-function estimator**。

#### 4.5.1 Straight-through：前向使用 hard sample，反向借用 soft probability

设 $p=\operatorname{softmax}(\ell)$ 是 categorical 概率，$z_{\mathrm{hard}}$ 是从 $p$ 采出的 one-hot sample，$\operatorname{sg}$ 表示 stop-gradient。ST 写成：

$$
z_{\mathrm{ST}}
=
\operatorname{sg}(z_{\mathrm{hard}})
+p
-\operatorname{sg}(p).
$$

这个公式要分前向和反向两次看：

1. **前向数值：** $\operatorname{sg}(p)$ 的数值等于 $p$，所以 $z_{\mathrm{ST}}=z_{\mathrm{hard}}$，下游真正看到的是离散 one-hot 状态。
2. **反向梯度：** $\operatorname{sg}(p)$ 不传梯度，因此 $z_{\mathrm{ST}}$ 对 $p$ 的梯度被当作恒等映射，预测损失可以沿着 $p\rightarrow\operatorname{softmax}(\ell)$ 回到 logits。

因此 ST 的含义不是“离散采样突然变得可微”，而是：**前向保持离散，反向人为使用连续概率的梯度近似。** 它通常方差较低，但这种梯度不是严格的离散采样梯度，因而有偏。DreamerV3 主要用它训练经过 $z_t$ 的 World Model 路径。

##### 4.5.1.1 $z_{\mathrm{hard}}$ 不是可训练的 codebook

DreamerV3 的离散 latent 不像 VQ-VAE 那样维护一组可训练的 codebook 向量。给定 categorical 概率 $p$，先采样类别 $k$，再直接构造固定的 one-hot：

$$
z_{\mathrm{hard}}
=
\operatorname{onehot}(k),
\qquad
k\sim\operatorname{Categorical}(p).
$$

例如 $e_1=[1,0,\ldots]$、$e_2=[0,1,\ldots]$ 等只是固定的基向量，不是需要优化的参数。因此当前采出的 $z_{\mathrm{hard}}$ 不会被“更新”，也不存在一个可学习的 $Z_{\mathrm{all}}$ 候选向量集合。

在 ST 公式中，前向传播仍有 $z_{\mathrm{ST}}=z_{\mathrm{hard}}$；反向传播则有：

$$
\frac{\partial z_{\mathrm{ST}}}{\partial z_{\mathrm{hard}}}=0,
\qquad
\frac{\partial z_{\mathrm{ST}}}{\partial p}=I.
$$

所以真正被训练的是两侧的网络：

1. **采样之前：** Encoder / Prior 更新 logits，改变各类别未来被选中的概率；
2. **采样之后：** RSSM、observation decoder、reward head 等更新参数，学习如何解释不同的 one-hot category。

这与 $z=Z_{\mathrm{all}}^\top\operatorname{onehot}(k)$ 不同：若 $Z_{\mathrm{all}}$ 是可训练 embedding，梯度确实可以更新候选向量，但这仍不能直接解决类别采样的不可微问题。DreamerV3 使用的是“**固定 one-hot category + 可训练选择概率 + 可训练 downstream interpretation**”，因此 ST 解决的是如何训练 category 的选择概率，而不是如何训练 latent 候选向量。

#### 4.5.2 REINFORCE：用 return 直接调整 action 的 log-probability

Actor 的 action 采样不沿着“action → imagined dynamics → reward”做可微反传，而是使用 score-function identity：

$$
\nabla_\theta
\mathbb E_{a\sim\pi_\theta}
\left[F(a)\right]
=
\mathbb E
\left[
F(a)\nabla_\theta\log\pi_\theta(a\mid s)
\right].
$$

在 DreamerV3 中，$F(a)$ 由 imagined return 或 advantage 提供，实际 Actor loss 写成：

$$
\mathcal L_{\mathrm{actor}}
=
-\operatorname{sg}(A_t)\log\pi_\theta(a_t\mid s_t).
$$

如果 $A_t>0$，梯度提高本次 action 的概率；如果 $A_t<0$，梯度降低它的概率。这里不需要对 sampled action 本身求导，因此既适用于离散 categorical action，也适用于连续 Gaussian action。它在理论上无偏，但通常方差更高，所以需要 critic baseline、return normalization 和较大的 batch 来稳定训练。

#### 4.5.3 两者不要混为同一种梯度

| **比较项** | **Straight-through estimator** | **REINFORCE / score function** |
|---|---|---|
| 处理的随机变量 | World Model 的离散 latent $z_t$ | Actor 采样的 action $a_t$ |
| 前向使用的值 | hard one-hot sample | 实际采出的 action |
| 反向依据 | 把 soft probability $p$ 当作可微近似 | $\log\pi_\theta(a_t\mid s_t)$ 与 advantage |
| 梯度性质 | 有偏、通常低方差 | 理论上无偏、通常高方差 |
| DreamerV3 中的用途 | 让 prediction loss 训练离散 latent | 让 return 训练 Actor 的动作分布 |

所以，ST 和 REINFORCE 不是“同一个公式的两种写法”：**ST 解决 World Model 如何穿过离散 latent 传梯度，REINFORCE 解决 Actor 如何根据 sampled action 的好坏更新概率。**

## 5. Reward、Continuation 与 Distributional Critic

### 5.1 为什么必须显式预测 reward，而不能只预测 value

传统 model-free policy gradient 通常不训练 reward network，不是因为 reward 难学，而是因为真实环境已经直接返回 $r_t$；长期 value 无法被立即观测，才需要 critic 估计。Dreamer 的 imagined trajectory 没有真实环境反馈，因此若不显式建模 $r_t$，就无法判断一条潜空间轨迹好坏。

Reward 与 value 的职责也不同。reward 是环境的局部、相对策略无关的任务反馈；$V^\pi(s)$ 是在当前策略下未来累计回报的期望，随着 Actor 更新而变化。一个 state value 只给出“按当前策略平均有多好”，不能替代动作条件下逐步发生的转移与即时反馈。Dreamer 因而采用“短期 reward model + horizon 末端 value bootstrap”的组合。

### 5.2 为什么还要预测 continuation c_t

在真实环境中，done/terminal 直接可见；在 imagination 中，模型必须自己判断 episode 是否仍然有效。continuation $c_t$ 进入有效折扣 $\gamma c_t$：若角色死亡、任务失败或成功终止，$c_t=0$ 会切断后续 bootstrap，防止模型想象“死亡后继续挖矿”或“物体掉出桌面后仍有未来价值”。

$$
R_t^\lambda
=
r_t
+
\gamma c_t
\left[
(1-\lambda)\bar v_\psi(s_{t+1})
+
\lambda R_{t+1}^\lambda
\right],
\qquad
R_T^\lambda=\bar v_\psi(s_T).
$$

### 5.3 Return、return distribution 与 value

单条轨迹给出一个具体 return realization；在同一状态下多次 rollout，会形成条件 return distribution。通常意义的 value 是该分布的期望：$V^\pi(s)=\mathbb E[R\mid s]$。因此论文说 Critic “学习 return distribution”并不意味着它不学习 value；它学习的是更完整的 $v_\psi(R\mid s)$，再用分布期望作为标量 $v_t$。

$$
v_t
=
\mathbb E_{R\sim v_\psi(\cdot\mid s_t)}[R].
$$

Critic 并不是直接回归一个标量，而是输出固定 support 上的分类分布。设 Critic 输出 $K$ 个 logits，经 softmax 得到：

$$
p_\psi(s_t)
=
\operatorname{softmax}(\ell_\psi(s_t))
=
(p_1,\ldots,p_K).
$$

每个类别对应一个固定的 return bin。DreamerV3 先在 symlog 坐标中取等距位置 $u_i$，再用 symexp 映射回原始 return 尺度：

$$
u_i\in[-20,20],
\qquad
b_i=\operatorname{symexp}(u_i),
\qquad
B=(b_1,\ldots,b_K).
$$

因此，$u_i$ 在变换空间中等距，但 $b_i$ 在原始数值空间中呈指数间隔：零附近的 bins 更密，大正负 return 处的 bins 更稀。这使同一套 support 可以覆盖从小到大的 return，而不让大数值直接产生同比例更大的回归梯度。

训练 target $y$ 时，先计算 $u=\operatorname{symlog}(y)$，找到相邻的 $u_i\leq u\leq u_{i+1}$，然后只在这两个 bin 上分配权重：

$$
w_i
=
\frac{u_{i+1}-u}{u_{i+1}-u_i},
\qquad
w_{i+1}
=
\frac{u-u_i}{u_{i+1}-u_i},
\qquad
\widetilde p(y)
=
w_i e_i+w_{i+1}e_{i+1}.
$$

$\widetilde p(y)$ 就是 twohot target。它和 RSSM 中的 one-hot latent 不是一回事：这里的 twohot 只是把一个连续的 return 标量编码成相邻两个 bin 的软分类标签。Critic 使用 categorical cross-entropy 训练：

$$
\mathcal L_{\mathrm{twohot}}
=
-\sum_{i=1}^{K}
\widetilde p_i(y)\log p_i.
$$

预测时再把分类分布映射回标量 value：

$$
v_t
=
\sum_{i=1}^{K}p_i b_i
=
\operatorname{softmax}(\ell_\psi(s_t))^\top B.
$$

这样，Critic 学习的是完整的 return distribution，Actor 和 lambda-return bootstrap 主要使用它的期望 $v_t$。分布表达可以保留多峰结构，但算法本身并不是显式风险敏感。使用 twohot 而不是单一 one-hot，还能减少连续 target 被硬边界量化带来的误差。

### 5.3.1 为什么 DreamerV3 的 Critic 要做 distributional discrete regression？

标准 Critic 完全可以直接回归连续标量：

$$
v_\psi(s)\in\mathbb R.
$$

例如用 MSE 拟合 $\lambda$-return。DreamerV3 并不是因为 value “必须是分布”才使用离散 support，而是**主动把标量回归改造成 categorical distribution prediction**。

Critic 先预测：

$$
p_\psi(R\mid s),
$$

即 return 在固定 bins 上的概率分布，再取期望得到最终用于 Actor 和 bootstrap 的标量 value：

$$
v(s)=\sum_i p_i b_i.
$$

这样做主要有三个原因：

1. **稳定梯度尺度**

   普通 MSE 的梯度

   $$
   \frac{\partial L}{\partial v}=2(v-y)
   $$

   会随 return 数值大小增长；而 categorical cross-entropy 对 logits 的梯度为

   $$
   \frac{\partial L}{\partial \ell_i}=p_i-q_i,
   $$

   不直接与 $y$ 的绝对数值成比例，因此更适合不同任务中跨度巨大的 reward / return。

2. **能够表示随机甚至多峰的 return**

   同一个 state 可能有多种未来结果，例如成功和失败形成双峰 return distribution。普通 scalar critic 只能直接表示其均值，而 distributional critic 可以先保留这种结构。

3. **Twohot 保留连续值精度**

   虽然 support 是离散 bins，但连续 target $y$ 会被编码到相邻两个 bin：

   $$
   q=w_i e_i+w_{i+1}e_{i+1}.
   $$

   因此最终期望

   $$
   v=\sum_i p_i b_i
   $$

   仍然可以得到连续值，不会被限制为某一个离散 bin。

所以这里的因果关系是：

$$
\boxed{
\text{不是 value 天生需要离散化}
\;\rightarrow\;
\text{而是为了稳定优化和表示 return 分布}
\;\rightarrow\;
\text{主动采用 categorical critic}
\;\rightarrow\;
\text{再用 twohot 处理连续 target}
}
$$

最终 Actor 实际使用的仍然是一个标量 value，而 distributional representation 主要是 Critic 的训练方式。

### 5.4 Critic EMA、零初始化与主/辅损失

- 主要 imagined critic loss：在当前 actor 的 imagined trajectories 上学习，权重 $\beta_{\mathrm{val}}=1$。

- critic replay loss：把 replay 的真实 reward 直接传播给 critic，权重 $\beta_{\mathrm{repval}}=0.3$。

- EMA critic regularizer：让当前 critic 靠近慢速参数副本，降低自举目标漂移。

- reward head 和 critic 输出层零初始化：避免训练初期凭空“幻觉”出很大的 reward/value。

### 5.5 Critic replay loss 的精确语义

从 replay trajectory 的每个状态 $s_t^R$ 分别启动一条由当前 Actor 驱动的 imagination rollout，并取该 rollout 起点的 imagined lambda-return $U_t$，作为这个 replay state 的“当前策略价值注释”。随后沿原 replay trajectory 使用真实环境奖励 $r_t^R$，再计算一个辅助 lambda-return。两条轨迹只共享起始状态，imagination 的中间状态、动作与 reward 不需要也不能和 replay 逐步对齐。

$$
G_t^{\mathrm{rep}}
=
r_t^R
+
\gamma(1-d_t)
\left[
(1-\lambda)U_{t+1}
+
\lambda G_{t+1}^{\mathrm{rep}}
\right].
$$

> **严格性说明**
>
> replay 的前若干动作来自历史 behavior policy，而 $U_t$ 来自当前 actor；因此整个 replay target 不是严格 on-policy value target。
>
> 论文/实现没有使用 importance sampling、Retrace 或 V-trace 修正，故该辅助项具有 off-policy bias。
>
> 其收益是当 reward model 漏掉稀疏奖励时，真实 replay reward 仍能直接训练 critic；这是一个低权重的偏差-监督质量权衡。


## 6. Actor：离散与连续动作的 Policy Gradient

$$
\mathcal L_{\mathrm{actor}}
=
-\sum_t
\operatorname{sg}
\left(
\frac{A_t}{\max(1,S)}
\right)
\log\pi_\theta(a_t\mid s_t)
-
\eta H
\left[
\pi_\theta(\cdot\mid s_t)
\right].
$$

$$
A_t
=
R_t^\lambda-v_\psi(s_t).
$$

DreamerV3 对离散与连续动作统一采用 REINFORCE。动作样本被视为常量，梯度不穿过“采样结果”，而是通过 $\log\pi_\theta(a\mid s)$ 回到 policy 参数。Critic 足够准确并不会让所有 advantage 都为零：$V(s)$ 是按当前策略对动作取平均的期望，不同动作仍分别具有正负 $A(s,a)$；只有策略接近局部最优时，常选最优动作的 advantage 才自然趋小。

### 6.1 离散动作：categorical policy 的具体例子

假设 Actor 输出 4 个 logits，经 softmax 得到 $p=(0.1,0.2,0.3,0.4)$，本次采到第 2 个动作，advantage $A=-2$。Loss 不是 logits 与 one-hot advantage 的乘积，也不是 $-2p_2$，而是：

$$
\mathcal L
=
-A\log p_2
=
2\log(0.2).
$$

$$
\frac{\partial\mathcal L}{\partial\ell_j}
=
-A\left(\mathbf 1[j=a]-p_j\right)
=
(-0.2,1.6,-0.6,-0.8)_j.
$$

梯度下降会降低被选中的第 2 个 logit，并相对提高其他动作概率，符合负 advantage 的含义。若 $A>0$，方向完全相反。

### 6.2 连续动作：对角 Gaussian policy

$$
\pi_\theta(a\mid s)
=
\mathcal N
\left(
a;
\mu_\theta(s),
\operatorname{diag}(\sigma_\theta(s)^2)
\right),
\qquad
\tau_i=\log\sigma_i.
$$

$$
\mathcal L
=
-A\sum_i
\log\mathcal N
\left(
a_i;\mu_i,\sigma_i^2
\right).
$$

$$
\frac{\partial\mathcal L}{\partial\mu_i}
=
-A\frac{a_i-\mu_i}{\sigma_i^2}.
$$

$$
\frac{\partial\mathcal L}{\partial\tau_i}
=
-A
\left[
\frac{(a_i-\mu_i)^2}{\sigma_i^2}-1
\right].
$$

连续变量中的 $p(a)$ 是概率密度。负 advantage 会通过移动均值和调整方差降低本次 sampled action 的密度；正 advantage 则提高其密度。整条多维动作向量共享一个 advantage，联合 log probability 等于各维 log density 之和。

### 6.3 为什么不通过 dynamics model 对 actor 反向传播

另一种做法是让梯度沿
$a_t\rightarrow\text{predicted state}\rightarrow\text{predicted reward}\rightarrow\text{return}$
反传，即 dynamics gradient。DreamerV3 选择 score-function estimator，论文没有对所有原因做完整消融；从算法目标和数值性质看，主要优势包括：

- 统一处理 categorical 与 Gaussian action，不要求动作可微。

- 不依赖 world model 对动作的局部导数是否真实，减少 actor 利用模型错误梯度的风险。

- 避免通过多步 stochastic dynamics 的长链反传，降低梯度爆炸/消失与实现复杂度。

- 代价是 REINFORCE 方差通常更高，需要 critic baseline、return normalization 和足够的 batch。

> **逻辑上的正确理解**
>
> actor 更新时把 imagined reward/return 当作近似模拟器给出的样本结果，而不是宣称这些结果等同于真实环境。梯度 stop 在 advantage 上，只根据“这次动作样本比平均好还是差”更新概率。


## 7. Return normalization、Entropy 与探索

### 7.1 P5-P95 范围与 EMA

$$
\Delta_k
=
P_{95}\left(R^\lambda_{\mathrm{batch}}\right)
-
P_{5}\left(R^\lambda_{\mathrm{batch}}\right).
$$

$$
S_k
=
0.99S_{k-1}+0.01\Delta_k.
$$

单个 $R_t^\lambda$ 是标量，但一次 imagination batch 会产生 $B\times H$ 个 return，分位数是在这组数上计算。用 5% 到 95% 而非 min/max，是为了忽略随机环境中的极端 episode；EMA 使 $S$ 随 batch 缓慢更新，避免 Actor 梯度尺度因单批数据突变。$S$ 是运行统计量，不通过梯度学习。

### 7.2 为什么分母必须有下限 1

标准 advantage normalization 把每个 batch 的 advantage 强制到近似零均值、单位标准差。即使原始 $A$ 只有 $10^{-3}$、主要来自 Critic 或 World Model 噪声，它也会被放大到 $O(1)$，从而给“最大化当前估计回报”固定强度的权重。此时奖励实际上还没有被当前策略或其邻域行为发现，即论文所说 rewards are not within reach。

Dreamer 使用 $A/\max(1,S)$：当 return range 小于 1 时，不把小信号放大，entropy regularizer 因而相对更重要，继续推动广泛探索；当奖励已经可达、return range 变大时，再缩小大梯度以稳定训练。若环境把所有 reward 乘以 1000，$A$ 与 $S$ 同时放大约 1000 倍，二者比值基本不变，因此固定 entropy coefficient $\eta=3\times10^{-4}$ 仍可跨 reward scale 使用。

> **“奖励尺度”与“奖励频率”是两件事**
>
> 尺度：同一任务把 reward 从 1 改成 1000，不应改变探索/利用平衡。
>
> 频率：稀疏奖励早期几乎没有可靠 policy signal，此时不应把微小估计噪声标准化为单位强度。
>
> Dreamer 的 denominator limit 同时处理二者：大信号归一化，小信号保持小。


## 8. 鲁棒预测、网络与优化器

### 8.1 Symlog 与 Symexp

$$
\operatorname{symlog}(x)
=
\operatorname{sign}(x)\log\left(|x|+1\right).
$$

$$
\operatorname{symexp}(x)
=
\operatorname{sign}(x)\left(\exp(|x|)-1\right).
$$

symlog 在零附近近似恒等映射，对大正数和大负数进行对称对数压缩。它避免普通 log 无法处理负值，也避免运行均值/方差 normalization 给优化目标引入额外非平稳性。Dreamer 对向量 observation 的 encoder input 与 decoder target 使用 symlog。

### 8.2 Symexp twohot 的实现补充

第 5.3 节已经给出了 symexp bins、twohot target 和 value expectation 的完整训练主线。实现层面需要记住：reward head 与 Critic 都可以把目标表示为固定 support 上的 categorical distribution，而不是直接回归原始标量。这样 cross-entropy 的梯度主要由预测概率与 target probability 的差异决定，不会随着 return 从 $1$ 增长到 $100000$ 而线性放大。

其中，$\operatorname{symlog}$ 负责把正负且跨度很大的 target 压到较规整的坐标，$\operatorname{symexp}$ 再把固定 support 映射回原始 return 尺度；twohot 则避免把连续 target 粗暴地舍入到单个 bin。具体 support、相邻 bin 插值和期望值计算见第 5.3 节；这里强调它们服务于跨 reward scale 的稳定训练，而不是另一个独立的 critic 目标。

### 8.3 主要实现配置

| **类别**     | **参数**               | **论文值**       | **作用**                                   |
|--------------|------------------------|------------------|--------------------------------------------|
| 通用         | Replay capacity        | $5\times10^6$           | 增加数据覆盖；配合 online queue            |
| 通用         | Batch / length         | 16 / 64          | 序列训练 world model                       |
| 通用         | Learning rate          | $4\times10^{-5}$          | 跨任务固定                                 |
| 优化         | AGC                    | 0.3              | 按参数张量范数自适应裁剪                   |
| 优化         | LaProp                 | $\epsilon=10^{-20}$ | 先 RMS 归一化再 momentum                 |
| 世界模型     | $\beta_{\mathrm{pred}}/\beta_{\mathrm{dyn}}/\beta_{\mathrm{rep}}$ | $1/1/0.1$ | 重建与可预测性平衡 |
| 世界模型     | Free nats / Unimix     | $1\,\mathrm{nat} / 1\%$           | 避免过压缩与 KL spike                      |
| Actor-Critic | Imagination horizon    | 15               | 控制模型误差累积                           |
| Actor-Critic | Discount / lambda      | $0.997 / 0.95$   | 有效 horizon 约 333                        |
| Critic       | imag / replay loss     | 1 / 0.3          | 主 on-policy imagination + 辅助真实 reward |
| Actor        | Entropy coefficient    | $3\times10^{-4}$ | 结合 RetNorm 跨领域使用                    |
| RetNorm      | P95-P5 / decay / limit | range / 0.99 / 1 | 稳健尺度与稀疏奖励探索                     |

> *主要数值来自 [1, Table 4, p. 21]。模型大小和 replay ratio 会按 benchmark 调整。*

### 8.4 网络与数据工程

- 图像使用 stride-2 CNN 编码与反卷积解码；向量输入使用 3 层 MLP。

- sequence model 为 8-block block-diagonal GRU，在增加记忆单元时避免完全二次参数增长。

- actor/critic 为 3 层 MLP；reward/continue 为较浅预测头；RMSNorm + SiLU。

- uniform replay + online queue；存储并更新 latent state，以便从 replay context 初始化。

- replay ratio 定义每收集一个环境步训练多少时间步；提高 replay ratio 用更多计算换更高数据效率。

## 9. 实验结果、消融与扩展性

### 9.1 跨领域结果摘要

| **领域**       | **规模/预算**    | **DreamerV3**                         | **强比较项**                    | **解读与限制**                             |
|----------------|------------------|---------------------------------------|---------------------------------|--------------------------------------------|
| Atari          | 57 tasks / 200M  | Gamer mean 3381%，median 830%         | MuZero 3054% / 693%             | 同数据预算；个别游戏仍落后                 |
| ProcGen        | 16 / 50M         | Normalized mean 66.01                 | PPG 64.89；PPO 42.80            | 平均领先很小，任务间差异大                 |
| DMLab          | 30 / 100M        | Human mean capped 71.4%               | IMPALA 1B: 66.3%；PPO 35.9%     | 强基线多用 10× 数据                        |
| Minecraft      | 1 / 100M         | Return 9.1；所有训练 agent 曾找到钻石 | IMPALA 7.1                      | 最终钻石 episode 成功率仅约 0.4%           |
| Atari100k      | 26 / 400K        | Mean 125%，median 49%                 | IRIS 105% / 29%；TWM median 51% | EfficientMuZero 改变协议，不能直接等价比较 |
| Proprio        | 18 / 500K        | Task mean 871                         | DMPO 801                        | 12M 模型即可达到强结果                     |
| Visual control | 20 / 1M          | Task mean 861                         | DrQ-v2 770                      | 无专门数据增强仍领先平均分                 |
| BSuite         | 23 task families | Task mean 66%，category mean 63%      | Boot DQN 60% / 57%              | 探索类别仅 0.01，Deep Sea 仍失败           |

> *数据汇总自 [1, pp. 24–38]；不同 benchmark 指标不可横向直接比较。*

### 9.2 消融：真正支撑性能的是什么

平均结果显示，每个鲁棒性技术只在部分任务上“决定生死”，但组合后形成跨领域稳定性。其中 KL balance + free bits 的平均影响最大，return normalization 与 symexp twohot 也明显重要。更关键的是 learning signal 消融：去掉 reward/value 对 representation 的梯度，整体性能只温和下降；去掉 reconstruction gradients 则大幅崩溃。这表明 Dreamer 的 latent 主要由 task-agnostic reconstruction 学成，而非只靠任务奖励塑形。

### 9.3 Scaling

模型参数从约 12M 扩展到 400M 时，性能总体单调提高，且更大的模型往往需要更少环境交互达到同等分数。提高 replay ratio 同样提升数据效率。这为实践提供了较清晰的 compute-data trade-off：当真实交互昂贵时，可以增加模型容量和每条数据的更新次数。

### 9.4 Minecraft 结果应如何准确表述

> **结果很强，但不能简化成“纯像素、只有钻石奖励、稳定通关”**
>
> 观测不仅有 64×64 RGB，还包含 400+ 物品库存向量、历史最大库存、装备 one-hot，以及 health/hunger/breath。
>
> 环境提供 abstract crafting action 和 flat categorical action space。
>
> 奖励包含通往钻石的 12 个 milestone，每项每 episode 奖励一次，并非只有最终钻石奖励。
>
> 100% 的训练 agent 在整个训练期间至少获得过一次钻石；在 100M 步附近，单 episode 钻石成功率约 0.4%。


## 10. 技术评价、局限与适用边界

### 10.1 Dreamer 是对 Policy Gradient 的什么改进

Dreamer 的 actor 仍然使用 policy gradient；真正变化的是 state representation、训练数据来源和长期目标构造。Model-free PG 的样本必须来自真实环境，而 Dreamer 允许从真实 replay state 出发，在 learned dynamics 中生成反事实动作与未来结果。由此它更准确地属于“model-based actor-critic with latent imagination”，而不是一种完全不同于 PG 的优化原理。

### 10.2 主要优势

- 数据效率：重复利用真实轨迹，并以想象生成大量 actor/critic 训练样本。

- 统一性：同一 actor objective 覆盖离散与连续动作；symlog/twohot/RetNorm 覆盖多种数值尺度。

- 表征能力：RSSM 同时具有历史记忆与多模态随机 latent。

- 可扩展性：模型规模和 replay ratio 增长带来较可预测的性能收益。

### 10.3 主要局限

- Model bias：policy 可能利用 dynamics/reward predictor 的错误。

- 计算成本：高性能实验常使用 200M 参数模型和单张 A100 数日。

- 重建目标可能学习任务无关细节；尽管消融证明其重要，但不等于它在所有环境中最优。

- critic replay loss 有 off-policy bias；论文将其作为低权重辅助项而非严格策略评估。

- 探索并未普遍解决：BSuite Deep Sea/Deep Sea Stochastic 仍为 0。

- “固定超参数”主要指核心算法配置，benchmark 级计算预算与模型大小仍不同。

### 10.4 适用性判断

| **场景**                         | **DreamerV3 适用性** | **原因**                             |
|----------------------------------|----------------------|--------------------------------------|
| 真实交互昂贵、可离线反复训练     | 高                   | 用计算换真实样本                     |
| 高维视觉、部分可观测、需要记忆   | 高                   | RSSM 与重建表征有优势                |
| 奖励尺度差异大、任务族多样       | 高                   | symlog/twohot/RetNorm 针对该问题设计 |
| 环境步极便宜、状态低维、任务简单 | 中                   | SAC/PPO 等可能更简单、开发成本更低   |
| 模型误差不可接受的安全关键控制   | 谨慎                 | 需要 uncertainty、约束或真实验证机制 |
| 纯探索难题、无任何中间信号       | 不保证               | 论文自身在 BSuite Deep Sea 上失败    |

## 11. 复现与调试检查表

| **模块**            | **检查项**                                                                                                | **常见错误**                      |
|---------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------|
| 时间对齐            | 确认 $a_{t-1}$ 进入 $h_t$；$a_t$ 影响下一状态；reward/terminal 的索引与 transition 一致。                | 错一位会让模型学习错误动力学      |
| Reset 处理          | episode 首状态正确重置 recurrent carry；terminal 与 time-limit truncation 区分。                          | 把 time-limit 当真正 terminal     |
| Posterior/Prior     | 真实序列用 $q(z\mid h,x)$，imagination 只能用 $p(z\mid h)$。                                                | 想象时偷看真实 $x$               |
| ST 采样             | 前向是 hard one-hot，反向沿 soft probability；不要误用 argmax 后完全断梯度。                              | 离散 latent 无法端到端训练        |
| KL 方向             | $\mathcal L_{\mathrm{dyn}}$ stop $q$；$\mathcal L_{\mathrm{rep}}$ stop $p$；权重 1 与 0.1；free bits 小于阈值时梯度为零。 | 两个 KL 梯度方向写反              |
| Unimix              | encoder、dynamics categorical 与离散 actor 都避免精确零概率。                                             | KL/日志出现 inf 或尖峰            |
| Reward/Value 初始化 | twohot 输出层零初始化，检查初期预测均值接近 0。                                                           | 初期 hallucinated reward          |
| Continuation        | imagined return 使用 $\gamma c$，终止后不 bootstrap。                                                       | 死亡后仍累计价值                  |
| Twohot              | 目标落在相邻两 bin；期望值计算注意正负大数求和顺序。                                                      | 大 target 导致梯度尺度异常        |
| Critic target       | lambda-return 的 horizon 末端用 value bootstrap；target 对 actor/critic 参数 stop-gradient。              | 把 sampled return 当可反传路径    |
| Slow critic         | EMA 更新顺序稳定；不要把慢网络误当成唯一当前 value。                                                      | 目标网络与当前网络混淆            |
| RetNorm             | P5-P95 在整个 return batch 上算；S 用 EMA；分母至少 1。                                                   | 稀疏奖励噪声被放大                |
| Actor loss          | 使用 $-A\log\pi-\eta H$；离散和连续动作都检查 joint log-prob 形状。                                        | 把 $A$ 乘动作或概率而非 log-prob  |
| Replay critic       | 每个 replay state 独立取得 imagination $U_t$；不要试图对齐两条未来轨迹；承认其 off-policy bias。            | 错误拼接 imagined/replay reward   |
| Replay ratio        | 区分“训练时间步/环境时间步”和“梯度步/环境步”，考虑 action repeat 与 batch length。                        | 误估实际更新强度                  |
| 诊断指标            | 监控 KL、free-bit 命中率、reward/value 预测、continuation、return range $S$、entropy、model rollout drift。 | 只看最终 return，错过模型崩溃先兆 |

## 附录 A：符号表

| **符号**            | **含义**                               |
|---------------------|----------------------------------------|
| $x_t$               | 真实观测；图像或向量                   |
| $h_t$               | 确定性 recurrent state，历史预测上下文 |
| $z_t$               | 离散 stochastic latent，当前随机创新   |
| $s_t=(h_t,z_t)$     | Actor/Critic 使用的 model state        |
| $q_\phi(z_t\mid h_t,x_t)$ | posterior / encoder distribution |
| $p_\phi(z_t\mid h_t)$ | prior / dynamics predictor          |
| $r_t$               | 即时 reward；真实或模型预测            |
| $c_t$               | episode continuation，0 表示终止       |
| $R_t^\lambda$       | bootstrapped lambda-return sample      |
| $v_\psi(R\mid s)$  | Critic 输出的 return distribution      |
| $v_t$               | Critic 分布的期望，即 scalar value     |
| $S$                 | P95-P5 return range 的 EMA             |
| $\operatorname{sg}(\cdot)$ | stop-gradient                          |
| $H$                 | imagination horizon，论文默认 15       |

## 附录 B：训练伪代码

```text
初始化 world model phi、actor theta、critic psi、slow critic、Replay Buffer
循环：
1. 用 actor 与真实环境交互，存储 (x, a, r, terminal)
2. 从 replay 采样 B 条、长度 T 的序列
3. Posterior rollout：
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
z_t ~ q_phi(z_t | h_t, x_t)
4. 更新 world model：
L_pred + L_dyn + 0.1 L_rep
5. 从 replay latent states 选择 imagination starts
6. 对每个 start，用当前 actor 和 prior rollout H 步：
a_t ~ pi_theta(. | s_t)
z_{t+1} ~ p_phi(. | h_{t+1})
预测 r_t, c_t, value distribution
7. 计算 imagined lambda-return R_t^lambda
8. 更新 critic：distributional loss + slow critic regularization
9. 更新 actor：-sg((R^lambda-v)/max(1,S))*log pi - eta*entropy
10. 可选 replay critic loss：
每个 replay state 取 imagination 起点 return U_t
沿 replay 真实 reward 计算辅助 lambda-return
11. 更新 EMA critic 与 return-range statistic S
```


## 附录 C：常见误解速查

> **“传统 PG 不学 reward，所以 reward 比 value 更难。”**
>
> 错误。传统 PG 已从环境直接观察 reward；value 才是需要估计的长期期望。


> **“critic 足够准，$A=R-V$ 就总是接近 0。”**
>
> 错误。$V$ 是动作平均；不同动作的 $Q-V$ 仍可显著正负。


> **“logits 乘 one-hot advantage 就是离散 PG。”**
>
> 错误。核心是 $-A\log\left[\operatorname{softmax}(\ell)\right]_a$。


> **“连续 PG 的 loss 是 $-A\,p(a)$。”**
>
> 错误。核心是 $-A\log p_\theta(a\mid s)$；梯度由 Gaussian log-likelihood 回到 $\mu$ 和 $\log\sigma$。


> **“$h_t$ 已能预测 $z_t$，所以 $z_t$ 多余。”**
>
> $h_t$ 只预测分布；$z_t$ 表示本次具体随机分支和当前观测创新。


> **“Dynamics predictor 没有 action 输入。”**
>
> action 已通过 recurrent transition 吸收进 h_t。


> **“Free bits 把 $D_{\mathrm{KL}}$ 最大限制为 $1\,\mathrm{nat}$。”**
>
> 相反，它在 $D_{\mathrm{KL}}$ 小于 $1\,\mathrm{nat}$ 时停止继续压缩。


> **“Dreamer 执行时每一步做规划。”**
>
> 论文设置中执行直接采样 actor；想象主要用于训练。


> **“Replay critic 的 imagination 与 replay 轨迹应逐步对齐。”**
>
> 不需要；每个 replay state 只取得一个当前策略 bootstrap annotation。


> **“Minecraft 已稳定学会获取钻石。”**
>
> 所有 agent 曾成功，但最终单 episode 钻石率约 0.4%，且有结构化观测、里程碑奖励和抽象 crafting action。


## 结论

DreamerV3 的技术意义在于，它把“学习一个可想象未来的世界模型”从容易失稳的研究原型推进为跨领域可扩展的 actor-critic 框架。RSSM 负责把历史和当前随机创新分开表示；prediction、dynamics 与 representation losses 共同塑造既有信息又可预测的 latent；reward 与 continuation 使 imagined trajectory 具有任务和终止语义；distributional critic、twohot 和 lambda-return 提供长期目标；REINFORCE、分位数 return normalization 与 entropy 共同处理离散/连续动作及稀疏奖励探索。

同时，正确理解其边界同样重要：Dreamer 没有消除 model bias，critic replay loss 不是严格 on-policy，统一配置仍伴随 benchmark 级计算选择，Minecraft 成果也依赖具体环境接口和中间奖励。将这些机制与限制同时纳入，才能把论文从“结果展示”转化为可复现、可迁移的技术方法。

## 相关笔记

- [[Dreamer技术报告|Dreamer]]：DreamerV3 的潜空间想象与 actor-critic 基础。
- [[DayDreamer论文综述与阅读重点|DayDreamer]]：DreamerV2 在真实机器人在线学习中的部署路线。
- [[PlaNet 论文概述|PlaNet]]：RSSM 与 latent planning 的前序工作。
- [[PPO_逻辑重构版|PPO]]：理解 DreamerV3 中 REINFORCE、advantage 与 entropy 的策略梯度背景。
- [[DreamerV4_技术报告|Dreamer 4]]：将 Dreamer 系列扩展到高容量生成式视频世界模型与离线 imagination training。

## 参考资料

[1] Hafner, D.; Pasukonis, J.; Ba, J.; Lillicrap, T. *Mastering Diverse Domains through World Models*. arXiv:2301.04104v2, 2024.

[2] DreamerV3 official implementation. `danijar/dreamerv3`, `dreamerv3/agent.py`, commit `e3f02248693a79dc8b0ebd62c93683888ddaccfe`（用于核对 critic replay loss 的代码语义）。

[3] Sutton, R. S.; Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018（lambda-return 与 policy gradient 背景）。

> *本报告依据《Mastering Diverse Domains through World Models》的论文正文与批注内容整理。*
