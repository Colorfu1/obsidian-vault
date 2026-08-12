---
title: Dreamer 技术综述：从真实经验到潜空间想象，再到可微行为学习
type: paper_note
topic: latent_imagination_model_based_rl
status: mature
importance: high
updated: 2026-08-12
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

# Dreamer 技术综述：从真实经验到潜空间想象，再到可微行为学习

> **论文**：Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi
> **标题**：*Dream to Control: Learning Behaviors by Latent Imagination*
> **发表**：ICLR 2020
> **核心问题**：如何从高维图像观测中学习一个世界模型，并让策略主要在这个世界模型的潜空间中通过“想象未来”来学习长时程行为。
> **资料边界**：本文主体严格围绕原始 Dreamer 论文展开；训练主循环、Equation 1–10、超参数来自论文正文与附录；单独标记为“官方实现”的 eval 行为来自作者公开 TensorFlow 2 实现。

---

## 0. 先给结论：Dreamer 到底做了什么？

Dreamer 最容易被误解成：

> “训练一个 world model，然后拿这个 world model 做 RL。”

这句话没有错，但太粗糙，因此几乎不能帮助理解后面的 Representation、Transition、Reward、Value、Actor、$V_\lambda$ 和梯度到底是什么关系。

Dreamer 真正的主线是：

$$
\boxed{
\text{真实环境收集经验}
\rightarrow
\text{用经验学习潜空间世界模型}
\rightarrow
\text{在世界模型中想象未来}
\rightarrow
\text{用想象回报训练 Actor / Value}
\rightarrow
\text{Actor 回到真实环境收集新经验}
}
$$

也就是：

```text
真实世界
  │
  │  observation / action / reward
  ▼
Replay Dataset
  │
  ▼
训练 World Model
  │
  │ posterior latent states
  ▼
Latent Imagination
  │
  ├── Actor 产生动作
  ├── Transition 预测未来 state
  ├── Reward 预测未来 reward
  └── Value 估计 horizon 外的长期价值
          │
          ▼
         Vλ
       ┌──┴──┐
       ▼     ▼
     Actor  Value
       │
       ▼
回真实环境执行
```

因此 Dreamer 不是“为了预测而预测”。

> **世界模型的最终用途，是成为一个可以让 Actor 在里面大量试错、并且可以反向传播梯度的内部训练环境。**

Dreamer 的名字也可以直接从这里理解：

- **Dream**：在 learned latent dynamics 中想象反事实未来；
- **Control**：把 imagined return 的信息转化成一个可以直接执行动作的 Actor。

---

## 1. 整篇论文只需要先分清两个“世界”

### 1.1 真实世界

真实环境中存在真正的：

$$
o_t^{\text{real}},\qquad
r_t^{\text{real}}.
$$

Agent 执行动作以后：

```text
当前真实 observation o_t
        ↓
Representation Model
        ↓
posterior latent state s_t
        ↓
Actor
        ↓
action a_t
        ↓
真实 Environment
        ↓
真实 o_{t+1}, r_t
```

这些真实轨迹会被写入 replay dataset。

真实世界主要承担三个职责：

1. 提供真实 observation；
2. 提供真实 reward；
3. 持续产生新的数据，用来修正 world model。

---

### 1.2 想象世界

Behavior Learning 阶段则不同。

想象从真实 replay 对应的 posterior state 开始：

$$
s_t^{\text{post}}.
$$

但从这里以后，不再调用真实环境：

```text
posterior s_t
    ↓
Actor
    ↓
a_t
    ↓
Transition Model
    ↓
imagined s_{t+1}
    ├── Reward Model → r̂_{t+1}
    └── Value Model  → v(s_{t+1})
    ↓
Actor
    ↓
a_{t+1}
    ↓
Transition
    ↓
...
```

这里不存在真实的未来：

$$
o_{t+1}^{\text{real}},\qquad
r_{t+1}^{\text{real}},
$$

因为 imagined action 根本没有真的发送给环境。

因此 Dreamer 必须自己回答：

```text
做这个动作以后会到哪里？
            ↓
        Transition

到了那里即时 reward 是多少？
            ↓
          Reward

从那里继续行动，长期总体有多好？
            ↓
           Value
```

这就是 Transition、Reward 和 Value 同时存在的根本原因。

---

## 2. 五个核心模块在整条链路中的位置

原始 Dreamer 可以先压缩成五个模块：

| 模块 | 形式 | 回答的问题 |
|---|---|---|
| Representation Model | $p_\theta(s_t\mid s_{t-1},a_{t-1},o_t)$ | **现实中我现在在哪？** |
| Transition Model | $q_\theta(s_t\mid s_{t-1},a_{t-1})$ | **如果只根据过去预测，我现在/未来会在哪？** |
| Reward Model | $q_\theta(r_t\mid s_t)$ | **这个 latent state 对应多少即时 reward？** |
| Actor | $q_\phi(a_t\mid s_t)$ | **当前应该做什么？** |
| Value Model | $v_\psi(s_t)$ | **从这里继续按当前 Actor 行动，长期回报是多少？** |

其中：

$$
\theta=\text{world model parameters},
$$

$$
\phi=\text{Actor parameters},
$$

$$
\psi=\text{Value parameters}.
$$

最核心的计算图可以写成：

$$
o_t
\rightarrow
s_t
\rightarrow
a_t
\rightarrow
s_{t+1}
\rightarrow
\{\hat r_{t+1},v(s_{t+1})\}.
$$

---

## 3. 论文 Algorithm 1：完整训练生命周期

论文把 Dreamer 的生命周期明确拆成：

1. **Dynamics Learning**
2. **Behavior Learning**
3. **Environment Interaction**

这三部分不断循环。

论文 Algorithm 1 的逻辑可以忠实转写为：

```text
初始化 replay dataset D
初始化 θ, φ, ψ

while not converged:

    for update step c = 1 ... C:

        # Dynamics Learning
        从 D 采样真实 sequence
        用真实 observation 计算 posterior model states
        更新 world model θ

        # Behavior Learning
        从每一个真实 posterior state 出发 imagination
        预测 imagined reward 和 value
        计算 Vλ
        更新 Actor φ
        更新 Value ψ

    # Environment Interaction
    reset 真实环境
    Actor 与环境交互
    加 exploration noise
    将新 episode 加入 D
```

因此整个训练节奏不是：

```text
world model 一次性训练完
        ↓
永久冻结
        ↓
再训练 Actor
```

而是：

$$
\boxed{
\text{World Model Update}
\leftrightarrow
\text{Behavior Update}
\leftrightarrow
\text{New Real Experience}
}
$$

后面逐段展开 Algorithm 1。

---

## 4. 第一阶段：Dynamics Learning

### 4.1 论文 Algorithm 1 在做什么？

每次 update 首先从 replay dataset 中采样：

$$
\{(a_t,o_t,r_t)\}_{t=k}^{k+L}\sim\mathcal D.
$$

然后用真实 observation 推断 model state：

$$
s_t
\sim
p_\theta(
s_t\mid s_{t-1},a_{t-1},o_t
).
$$

最后论文用一句高度抽象的话表示：

```text
Update θ using representation learning.
```

这句话不是说只更新 Encoder。

它的意思是：

> 使用选定的世界模型 representation-learning objective，联合更新整个 world model。

原论文允许 Dreamer 搭配多种表示学习目标。主实验最重要的一种是 **pixel reconstruction objective**。

---

## 5. Reconstruction world model：Equation 9 和 Equation 10

### 5.1 Equation 9：世界模型的四个组件

在 reconstruction 版本中，论文列出：

$$
\text{Representation:}\quad
p_\theta(s_t\mid s_{t-1},a_{t-1},o_t),
$$

$$
\text{Observation:}\quad
q_\theta(o_t\mid s_t),
$$

$$
\text{Reward:}\quad
q_\theta(r_t\mid s_t),
$$

$$
\text{Transition:}\quad
q_\theta(s_t\mid s_{t-1},a_{t-1}).
$$

其中 Observation Model 主要用于给 latent representation 提供学习信号。

---

### 5.2 Equation 10：论文原始世界模型目标

论文把 reconstruction objective 写成：

$$
\mathcal J_{\mathrm{REC}}
\doteq
\mathbb E_p
\left[
\sum_t
\left(
\mathcal J_O^t+
\mathcal J_R^t+
\mathcal J_D^t
\right)
\right]
+\mathrm{const},
$$

其中：

$$
\boxed{
\mathcal J_O^t
=
\ln q_\theta(o_t\mid s_t)
}
$$

$$
\boxed{
\mathcal J_R^t
=
\ln q_\theta(r_t\mid s_t)
}
$$

$$
\boxed{
\mathcal J_D^t
=
-\beta
D_{\mathrm{KL}}
\left[
p_\theta(s_t\mid s_{t-1},a_{t-1},o_t)
\;\Vert\;
q_\theta(s_t\mid s_{t-1},a_{t-1})
\right]
}
$$

论文写的是**最大化**这个 variational lower bound。

工程实现通常改写成最小化：

$$
\boxed{
\mathcal L_{\mathrm{model}}
=
\mathcal L_{\mathrm{obs}}
+
\mathcal L_{\mathrm{reward}}
+
\mathcal L_{\mathrm{KL}}
}
$$

下面分别看这三个 loss。

---

## 6. 第一项：Observation Reconstruction Loss

真实 observation 首先进入 Representation Model：

$$
s_t
\sim
p_\theta(
s_t\mid s_{t-1},a_{t-1},o_t
).
$$

得到 posterior state 后，再通过 Observation Model：

$$
q_\theta(o_t\mid s_t)
$$

尝试重建原始 observation。

对应最小化：

$$
\boxed{
\mathcal L_{\mathrm{obs}}
=
-
\mathbb E
\left[
\log q_\theta(o_t^{\mathrm{gt}}\mid s_t)
\right]
}
$$

计算链路：

```text
真实 o_t
  ↓
Representation
  ↓
posterior s_t
  ↓
Observation Model
  ↓
预测 ô_t

目标：
ô_t 能解释 / 重建真实 o_t
```

如果 Observation Model 使用固定方差 Gaussian：

$$
q_\theta(o_t\mid s_t)
=
\mathcal N(o_t;\mu_o(s_t),\sigma_o^2I),
$$

则：

$$
-\log q_\theta(o_t^{\mathrm{gt}}\mid s_t)
=
\frac{1}{2\sigma_o^2}
\left\|
o_t^{\mathrm{gt}}-\mu_o(s_t)
\right\|^2+C.
$$

所以它在形式上等价于带尺度的 pixel MSE。

但需要注意：

> Observation reconstruction 的目的不是“生成好看的图片”本身，而是迫使 $s_t$ 保留足以解释 observation 的信息。

---

## 7. 第二项：Reward Prediction Loss

同一个 posterior state 同时进入 Reward Model：

$$
q_\theta(r_t\mid s_t).
$$

监督来自 replay 中真实环境给出的：

$$
r_t^{\mathrm{gt}}.
$$

因此：

$$
\boxed{
\mathcal L_{\mathrm{reward}}
=
-
\mathbb E
\left[
\log q_\theta(r_t^{\mathrm{gt}}\mid s_t)
\right]
}
$$

链路为：

```text
posterior s_t
     ↓
Reward Model
     ↓
预测 r̂_t

目标：
r̂_t ≈ replay 中真实 r_t
```

若 Reward Model 使用固定方差 Gaussian：

$$
q_\theta(r_t\mid s_t)
=
\mathcal N(r_t;\mu_r(s_t),\sigma_r^2),
$$

则：

$$
-\log q_\theta(r_t^{\mathrm{gt}}\mid s_t)
\propto
\left(
r_t^{\mathrm{gt}}-\hat r_t
\right)^2.
$$

这里非常重要：

### Dynamics Learning 时

Reward Model 用：

$$
r_t^{\mathrm{real}}
$$

作为监督。

### Behavior Learning / imagination 时

真实环境没有执行 imagined action，因此没有真实 reward。

此时才使用已经训练好的 Reward Model 产生：

$$
\hat r_\tau.
$$

所以 Reward Model 的逻辑是：

$$
\boxed{
\text{真实 reward 训练 Reward Model}
\rightarrow
\text{Reward Model 为 imagined future 提供 reward}
}
$$

### 7.1 Dreamer 是否需要 dense reward？

不需要。原始 Dreamer 可以处理 sparse reward：真实环境中绝大多数时间步的 reward 都
可以是 $0$，只有任务成功或达到关键事件时才出现非零信号。例如：

$$
r_t=
\begin{cases}
1, & \text{任务成功},\\
0, & \text{其他情况}.
\end{cases}
$$

一条真实 trajectory 可能长时间都是 $0$，直到末尾才出现一次成功奖励。此时需要区分：

> **Reward 本身不做 bootstrap；发生 bootstrap 的是 return、Value 和 $V_\lambda$。**

真实环境只提供即时奖励 $r_t^{\mathrm{real}}$。Reward Model 用这些真实 reward 训练，
而 return/value target 再把较晚出现的奖励传播给较早的 imagined state。例如，如果：

$$
\hat r_0=0,\qquad
\hat r_1=0,\qquad
\hat r_2=0,\qquad
\hat r_3=1,
$$

则一个三步目标可以包含：

$$
V_N^3(s_1)
=
0+\gamma 0+\gamma^2\cdot1+\gamma^3v_\psi(s_4).
$$

因此 credit assignment 的链路是：

```text
真实 sparse reward
        ↓
训练 Reward Model
        ↓
imagined trajectory 中预测未来 reward
        ↓
return / Value / Vλ
        ↓
把较晚 reward 的影响传回较早 state
        ↓
Actor
```

但 bootstrap 不能凭空创造任务信号。如果 replay 中所有 reward 都为零：

$$
r_t=0,\qquad \forall t,
$$

那么 Reward Model 最自然地会预测 $\hat r\approx0$，$V_\lambda$ 也会缺少任务导向的
梯度。Dreamer 能处理 sparse reward，不等于它能从完全没有成功信号的数据中自动推出任务
目标。

---

## 8. 第三项：Posterior-Prior KL Loss

这是理解 Dreamer 世界模型最关键的一项。

### 8.1 Posterior

有真实 observation 时：

$$
\boxed{
p_\theta(
s_t\mid
s_{t-1},a_{t-1},o_t
)
}
$$

能够利用当前真实 $o_t$。

它回答：

> 结合现实证据以后，我认为当前 latent state 到底是什么？

---

### 8.2 Prior / Transition

Transition Model 不看当前 observation：

$$
\boxed{
q_\theta(
s_t\mid
s_{t-1},a_{t-1}
)
}
$$

它回答：

> 如果只根据上一个 latent state 和 action，我预测下一 latent state 应该是什么？

---

### 8.3 KL 约束

论文使用：

$$
\boxed{
\mathcal L_{\mathrm{KL}}
=
\beta
D_{\mathrm{KL}}
\left[
p_\theta(s_t\mid s_{t-1},a_{t-1},o_t)
\Vert
q_\theta(s_t\mid s_{t-1},a_{t-1})
\right]
}
$$

训练结构：

```text
                       真实 o_t
                          │
                          ▼
s_{t-1}, a_{t-1} ──→ posterior
       │                  │
       │                  │ KL
       │                  ▼
       └──────────────→ prior
```

posterior 知道“现实最终发生了什么”，prior 则必须学会：

> 不看未来 observation，仅根据 state + action，也尽可能预测到正确的 latent future。

这项约束为什么至关重要？

因为 imagination 阶段根本没有：

$$
o_{\tau+1}^{\mathrm{real}}.
$$

未来只能靠 Transition prior：

$$
q_\theta(s_{\tau+1}\mid s_\tau,a_\tau).
$$

如果 prior 无法逼近真实 posterior，那么 imagination 很快就会漂离现实。

---

## 9. 世界模型三个 loss 放在一起

因此第一阶段可以完整写成：

```text
真实 replay:
(o_t, a_t, r_t)
       │
       ▼
Representation
       │
       ▼
posterior s_t
       │
       ├── Observation Model
       │       ↓
       │   L_obs
       │   = -log q(o_t | s_t)
       │
       ├── Reward Model
       │       ↓
       │   L_reward
       │   = -log q(r_t | s_t)
       │
       └── 与 Transition prior 比较
               ↓
           L_KL
           =
           β KL(
             posterior
             ||
             prior
           )

L_model
=
L_obs + L_reward + L_KL
       ↓
更新 world model θ
```

因此 Algorithm 1 的：

```text
Update θ using representation learning
```

在 reconstruction 主线下，实际指的是**整个世界模型联合训练**。

---

## 10. RSSM：Transition 不是普通的单步 MLP

Dreamer 沿用 PlaNet 风格的 RSSM（Recurrent State Space Model）。

RSSM 的核心思想是同时维护：

- deterministic recurrent information；
- stochastic latent variable。

直观上可写成：

```text
上一时刻 hidden/state
        +
上一动作 a_{t-1}
        ↓
recurrent dynamics
        ↓
deterministic history
        │
        ├── prior stochastic state
        │
真实 o_t ─→ encoder
        │
        └── posterior stochastic state
```

因此 Dreamer 的 $s_t$ 不应简单理解成“某张图像经过 Encoder 后的一根向量”。

它是一个对历史：

$$
o_{\le t},a_{<t}
$$

进行压缩后、适合做 Markovian prediction 的 latent model state。

这一点直接解释了真实执行时为什么写：

$$
p_\theta(s_t\mid s_{t-1},a_{t-1},o_t),
$$

而不是简单：

$$
s_t=Encoder(o_t).
$$

---

## 11. 第二阶段：Behavior Learning

世界模型 update 完以后，Algorithm 1 紧接着进入 Behavior Learning。

论文的关键动作是：

$$
\boxed{
\text{从每一个真实 posterior state }s_t
\text{ 出发想象长度 }H\text{ 的 trajectory}
}
$$

也就是说：

```text
真实 sequence:
s_1, s_2, s_3, ..., s_L

每一个都可以成为 imagination 起点：

s_1 → imagined rollout
s_2 → imagined rollout
s_3 → imagined rollout
...
s_L → imagined rollout
```

这比“只从真实 trajectory 的最后一个 state 开始 rollout”高效得多。

官方 TensorFlow 2 实现也直接把 posterior sequence flatten 成大量 start states，再向前 rollout。

---

## 12. 想象轨迹具体怎样生成？

想象阶段使用：

$$
a_\tau
\sim
q_\phi(a_\tau\mid s_\tau),
$$

然后：

$$
s_{\tau+1}
\sim
q_\theta(
s_{\tau+1}\mid s_\tau,a_\tau
).
$$

再预测：

$$
\hat r_\tau
=
\mathbb E[q_\theta(r_\tau\mid s_\tau)],
$$

以及：

$$
v_\psi(s_\tau).
$$

整个链路：

```text
real posterior s_t
        ↓
Actor
        ↓
a_t
        ↓
Transition
        ↓
imagined s_{t+1}
        ├── Reward → r̂_{t+1}
        └── Value  → v(s_{t+1})
        ↓
Actor
        ↓
a_{t+1}
        ↓
Transition
        ↓
...
```

注意：

> 想象阶段只有起点是由真实 observation 校正出来的 posterior。之后全部使用 Transition prior 向前滚动。

### 12.1 固定 learned model 参数，不等于固定 reward 数值

Behavior Learning 时暂时固定的是 World Model、Reward Model 和 Transition Model 的
参数 $\theta$，而不是把 reward 预先固定成一个常数。Reward Model 仍然是状态的函数：

$$
q_\theta(r_\tau\mid s_\tau).
$$

不同 imagined state 会得到不同的预测 reward：

```text
同一个 posterior 起点 s_t
        ↓
Actor 选择不同 action
        ↓
Transition 得到不同 imagined state
        ↓
固定参数的 Reward Model
        ↓
得到不同的 r̂τ
```

所以 Reward Model 更像一个已经学好的 latent-space reward function。Actor 优化的是：

$$
\text{如何修改当前 policy，才能提高 predicted long-term return}.
$$

### 12.2 Behavior Learning 不是枚举所有 action

Dreamer 也不是枚举全部 action、覆盖所有 latent state 后再挑最高 reward。它按照当前
Actor 的 action distribution 生成 imagined trajectories：

$$
a_\tau\sim q_\phi(a_\tau\mid s_\tau).
$$

因此这些 trajectory 是“当前 policy 周围的反事实未来”，目标不是单纯增加场景多样性，而是
利用它们计算 predicted return，并寻找能提高长期回报的 policy 更新方向。未来 observation
不需要被渲染出来再编码；Section 13–14 会说明，直接在 latent dynamics 中 rollout 才能
保持反事实 action 的因果一致性并降低计算成本。

---

## 13. 为什么 imagination 中不继续读取 $o_{t+1}$？

这是 Dreamer 最重要的因果逻辑之一。

假设 replay 中真实发生的是：

```text
state s_t
   ↓
data action = 向左
   ↓
真实 observation o_{t+1}^{data}
```

但 Actor 在 imagination 中选择：

```text
policy action = 向右
```

那么 replay 中的：

$$
o_{t+1}^{data}
$$

对应的是“向左”的后果。

Actor 现在需要的是：

$$
p(
o_{t+1}
\mid
s_t,
a_t=\text{向右}
).
$$

这个真实 observation 根本不存在。

所以不能：

```text
Actor 选 imagined action
       ↓
却继续读取 replay 的下一张图
```

这会造成因果矛盾。

因此 imagination 必须变成：

$$
s_t
\rightarrow
a_t
\rightarrow
q_\theta(s_{t+1}\mid s_t,a_t).
$$

---

## 14. 为什么不先预测图像，再重新编码？

理论上可以：

$$
s_t
\rightarrow
\hat o_{t+1}
\rightarrow
Encoder
\rightarrow
s_{t+1}.
$$

但 $\hat o_{t+1}$ 本身就是由同一个 learned model 生成的。

它没有加入任何新的真实信息，因此：

- 不能纠正 model error；
- 增加 decoder 计算；
- 增加图像生成误差；
- 再编码又可能损失信息；
- 大幅增加显存和计算量。

所以 Dreamer 的关键优势之一就是直接：

$$
\boxed{
s_t\rightarrow s_{t+1}\rightarrow s_{t+2}\rightarrow\cdots
}
$$

在 compact latent space 里 rollout。

---

## 15. Reward Model 和 Value Model 为什么不能只留一个？

Reward Model：

$$
q_\theta(r_\tau\mid s_\tau)
$$

预测**当前/即时 reward**。

Value Model：

$$
v_\psi(s_\tau)
$$

预测**从当前 state 开始，按当前策略继续行动的长期累计价值**。

所以：

| 模型 | 预测对象 | 时间范围 |
|---|---|---|
| Reward | $r_\tau$ | 一步 |
| Value | $V^\pi(s_\tau)$ | 长期 |

直观上：

```text
Reward:
“现在这一刻值多少分？”

Value:
“从这里继续按照当前 Actor 行动，
以后总共大约还能拿多少分？”
```

Actor 改变时：

$$
V^{\pi_\phi}
$$

也会改变。

而环境 reward function 在固定任务中相对稳定。

---

## 16. 为什么需要 Value：有限 horizon 的根本问题

假设 imagination 只 rollout $H$ 步。

如果 Actor 只最大化：

$$
\hat r_t
+
\gamma\hat r_{t+1}
+\cdots+
\gamma^{H-1}\hat r_{t+H-1},
$$

那么 $H$ 之后发生什么完全不进入目标。

这会导致：

> **策略只关心 imagination horizon 内的结果。**

如果某个动作短期 reward 高、长期灾难性，它仍可能被认为是好动作。

Dreamer 因此引入 Value Model，为 horizon 外的 future bootstrap。

---

## 17. Equation 5：$k$-step value estimate

论文 Equation 5：

$$
\boxed{
V_N^k(s_\tau)
\doteq
\mathbb E_{q_\theta,q_\phi}
\left[
\sum_{n=\tau}^{h-1}
\gamma^{n-\tau}r_n
+
\gamma^{h-\tau}
v_\psi(s_h)
\right]
}
$$

其中：

$$
\boxed{
h=\min(\tau+k,t+H)
}
$$

例如：

### 1-step

$$
V_N^1(s_\tau)
=
r_\tau
+
\gamma v_\psi(s_{\tau+1})
$$

### 2-step

$$
V_N^2(s_\tau)
=
r_\tau
+
\gamma r_{\tau+1}
+
\gamma^2v_\psi(s_{\tau+2})
$$

### 3-step

$$
V_N^3(s_\tau)
=
r_\tau
+
\gamma r_{\tau+1}
+
\gamma^2r_{\tau+2}
+
\gamma^3v_\psi(s_{\tau+3})
$$

所以每个 $k$ 都在做：

$$
\boxed{
\text{前 }k\text{ 步相信 model rollout}
+
\text{剩余未来相信 Value}
}
$$

---

## 18. Equation 6：Dreamer 的 $V_\lambda$

问题是：

> 应该选哪个 $k$？

短 $k$：

- 更早 bootstrap；
- 更依赖 Value；
- model error 累积少。

长 $k$：

- 使用更多 imagined reward；
- 更少依赖 Value；
- 但 dynamics error 会积累。

Dreamer 不固定一个 $k$，而是用 Equation 6：

$$
\boxed{
V_\lambda(s_\tau)
\doteq
(1-\lambda)
\sum_{n=1}^{H-1}
\lambda^{n-1}
V_N^n(s_\tau)
+
\lambda^{H-1}
V_N^H(s_\tau)
}
\tag{6}
$$

展开就是：

$$
\begin{aligned}
V_\lambda(s_\tau)
={}&
(1-\lambda)V_N^1(s_\tau)
\\
&+
(1-\lambda)\lambda V_N^2(s_\tau)
\\
&+
(1-\lambda)\lambda^2 V_N^3(s_\tau)
\\
&+\cdots
\\
&+
\lambda^{H-1}V_N^H(s_\tau).
\end{aligned}
$$

直观上：

```text
1-step estimate ── weight (1-λ)
2-step estimate ── weight (1-λ)λ
3-step estimate ── weight (1-λ)λ²
...
H-step estimate ── weight λ^(H-1)
                 │
                 ▼
              Vλ(sτ)
```

因此 $V_\lambda$ 是整个 Behavior Learning 中连接：

```text
imagined reward
+
Value bootstrap
```

的关键目标。

---

## 19. 第二阶段完整数据流

现在可以把 Algorithm 1 中：

```text
Predict rewards
Predict values
Compute Vλ
```

展开成：

```text
s_τ
 │
 ├── Reward → r̂_τ
 │
 ▼
s_{τ+1}
 │
 ├── Reward → r̂_{τ+1}
 ├── Value  → v(s_{τ+1})
 │
 ▼
s_{τ+2}
 │
 ├── Reward → r̂_{τ+2}
 ├── Value  → v(s_{τ+2})
 │
 ...

      ↓

V_N^1
V_N^2
V_N^3
...
V_N^H

      ↓

Equation 6

      ↓

Vλ(s_τ)

   ┌────┴────┐
   ▼         ▼
 Actor     Value
```

这一步之后，Reward Model 和 Value Model 的关系就不再是两个孤立模块：

> Reward 负责显式展开近未来，Value 负责 bootstrap 更远未来，$V_\lambda$ 把两者组合起来。

---

## 20. Actor 和 Value Model 的目标：Equation 7 和 8

### 20.1 Actor

论文 Equation 7：

$$
\boxed{
\max_\phi
\mathbb E_{q_\theta,q_\phi}
\left[
\sum_{\tau=t}^{t+H}
V_\lambda(s_\tau)
\right]
}
$$

Actor 的目标很直接：

> 让自己产生的 action 导致 imagined trajectory 具有更高的 $V_\lambda$。

如果写成 minimization loss：

$$
\mathcal L_{\mathrm{actor}}
=
-
\mathbb E
\left[
\sum_\tau
V_\lambda(s_\tau)
\right].
$$

---

### 20.2 Value

论文 Equation 8：

$$
\boxed{
\min_\psi
\mathbb E_{q_\theta,q_\phi}
\left[
\sum_{\tau=t}^{t+H}
\frac12
\left\|
v_\psi(s_\tau)
-
V_\lambda(s_\tau)
\right\|^2
\right]
}
$$

对 Value update 而言：

$$
V_\lambda
$$

是监督 target。

论文明确说明，Value target 处停止梯度。

因此工程上理解成：

$$
\mathcal L_{\mathrm{value}}
=
\frac12
\left(
v_\psi(s_\tau)
-
\operatorname{sg}[V_\lambda(s_\tau)]
\right)^2.
$$

---

## 21. Dreamer 最核心的梯度：为什么 Actor 能穿过世界模型？

如果这里只记：

```text
Actor 最大化 Vλ
```

仍然没有理解 Dreamer 和普通 PPO/A3C 的关键差异。

Dreamer 的动作来自 tanh Gaussian。

论文 Equation 3：

$$
\boxed{
a_\tau
=
\tanh
\left(
\mu_\phi(s_\tau)
+
\sigma_\phi(s_\tau)\epsilon
\right),
\qquad
\epsilon\sim\mathcal N(0,I).
}
$$

这是 reparameterization。

动作因此可以看成：

$$
a_\tau=g_\phi(s_\tau,\epsilon).
$$

### 21.1 Action 仍然是 sample，但 sampling 不一定切断梯度

这里的 action 不是把 Actor 的输出改成一个 deterministic action。对每个固定的
$\epsilon$，Dreamer 仍然从当前策略分布中采样：

$$
\epsilon\sim\mathcal N(0,I),
\qquad
a_\tau=g_\phi(s_\tau,\epsilon).
$$

区别在于，随机性被显式写成了与 $\phi$ 无关的噪声 $\epsilon$。因此在固定一次噪声
realization 时，可以对 $a_\tau$ 关于 $\phi$ 求导：

$$
\frac{\partial a_\tau}{\partial\phi}
=
\frac{\partial g_\phi(s_\tau,\epsilon)}{\partial\phi}.
$$

这就是 reparameterization 的作用：它允许“采样动作”仍然参与 pathwise gradient。

### 21.2 Equation 5 中的两类梯度来源

先忽略 terminal 和 horizon 截断，Equation 5 可以直观写成：

$$
V_N^k(s_\tau)
\approx
\underbrace{
\sum_{i=0}^{k-1}\gamma^i\hat r_{\tau+i}
}_{\text{imagined reward 路径}}
+
\underbrace{
\gamma^k v_\psi(s_{\tau+k})
}_{\text{Value bootstrap 路径}}.
$$

论文中的精确边界仍然是：

$$
h=\min(\tau+k,t+H),
$$

所以当 rollout 提前结束时，最后一项应在 $s_h$ 处 bootstrap。这里的 $\hat r$ 是
Behavior Learning 阶段由 Reward Model 预测出的 imagined reward，而不是对 imagined
action 重新访问真实环境得到的 reward。

因此，一个 action 对 $V_N^k$ 的影响至少有两条路径：

```text
action → imagined state → predicted reward
action → imagined state → Value bootstrap
```

第一条是 reward path。例如：

$$
\phi
\rightarrow a_\tau
\rightarrow s_{\tau+1}
\rightarrow \hat r_{\tau+1}.
$$

第二条是 value path。例如：

$$
\phi
\rightarrow a_\tau
\rightarrow s_{\tau+1}
\rightarrow v_\psi(s_{\tau+1}).
$$

而 $V_\lambda$ 又是多个不同 $k$ 的 $V_N^k$ 的加权和，因此其中会同时出现：

- 经过一层或多层 dynamics 后到达 predicted reward 的路径；
- 经过 reward，再到达 Value bootstrap 的路径；
- 经过更长 imagined reward 序列，最后到达 Value 的路径。

所以 $V_\lambda$ 不只包含一条“action 到 reward”的梯度，而是把不同深度的
future-consequence paths 按 $\lambda$ 加权组合起来。

### 21.3 梯度如何穿过多步 latent dynamics

Transition 也通过 reparameterized latent dynamics 产生未来 state。

于是存在完整可微路径：

$$
\boxed{
\phi
\rightarrow
a_\tau
\rightarrow
s_{\tau+1}
\rightarrow
\hat r_{\tau+1}
\rightarrow
V_\lambda
}
$$

以及：

$$
\boxed{
\phi
\rightarrow
a_\tau
\rightarrow
s_{\tau+1}
\rightarrow
v_\psi(s_{\tau+1})
\rightarrow
V_\lambda
}
$$

因此可以计算类似：

$$
\frac{\partial V_\lambda}{\partial\phi}
=
\frac{\partial V_\lambda}{\partial s_{\tau+1}}
\frac{\partial s_{\tau+1}}{\partial a_\tau}
\frac{\partial a_\tau}{\partial\phi}
+\cdots
$$

这就是 Dreamer 所强调的：

> analytic / pathwise value gradients through imagined latent dynamics。

---

## 22. “冻结 world model”不等于 `no_grad`

论文明确说：

> Behavior Learning 时 world model 固定。

这里的“固定”指：

$$
\theta
$$

不被 Actor optimizer 更新。

但这不等于：

```python
with no_grad():
    imagined_state = transition(state, action)
```

如果把 transition 从计算图里切掉，就无法得到：

$$
\frac{\partial s_{\tau+1}}{\partial a_\tau}.
$$

Actor 也就无法知道：

> action 改一点，会如何通过 dynamics 改变未来 state、reward 和 value？

正确理解是：

```text
Actor update:
θ 不 optimizer.step()
ψ 不由 Actor optimizer.step()

但是：
Transition / Reward / Value 的前向计算
仍然参与 Actor 的计算图
```

最终只取：

$$
\nabla_\phi.
$$

---

## 23. Dreamer 与 PPO / A3C 的 Actor 梯度差异

### 23.1 PPO 的 score-function gradient 与 Advantage

典型 policy gradient：

$$
\nabla_\phi J
=
\mathbb E
\left[
\nabla_\phi
\log\pi_\phi(a_t\mid s_t)
A_t
\right].
$$

Actor loss：

$$
\mathcal L_{\mathrm{actor}}
=
-
\log\pi_\phi(a_t\mid s_t)
\operatorname{sg}[A_t].
$$

Value 在这里主要用于：

- baseline；
- advantage estimation；
- 降低 policy-gradient 方差。

Actor 并不通过：

$$
\frac{\partial V(s)}{\partial s}
$$

去更新策略。

Dreamer 则不同：

```text
Actor
 ↓
action
 ↓
learned dynamics
 ↓
future state
 ↓
reward / value
 ↓
Vλ
```

梯度真的沿这个 future consequence path 回到 Actor。

所以：

| 方法 | Actor 梯度 | Value 在 Actor update 中的角色 | 是否穿过多步 dynamics |
|---|---|---|---|
| A3C/PPO | score-function | baseline / advantage | 否 |
| DDPG/SAC | 对 $Q(s,a)$ 的 action gradient | 可微 action-value | 通常不穿过多步环境 dynamics |
| Dreamer | pathwise / reparameterized | imagined return 的可微组成 | 是 |

### 23.2 为什么 PPO 不能直接把 $V(s)$ 当作策略权重？

PPO 中的 $V(s)$ 与当前 action 无关，因此如果直接把它放进 score-function gradient：

$$
\mathbb E_{a\sim\pi_\phi}
\left[
\nabla_\phi\log\pi_\phi(a\mid s)\,V(s)
\right]
=
V(s)\nabla_\phi
\int\pi_\phi(a\mid s)\,da
=0.
$$

它不会提供 action preference。真正包含 action 信息的是 $Q(s,a)$；减去一个不依赖
action 的 baseline $V(s)$ 后得到：

$$
A(s,a)=Q(s,a)-V(s).
$$

因此 PPO 使用 Advantage，主要是为了在不改变期望梯度的前提下降低方差，并表达：

> 这次采样的 action 比当前 state 下的平均水平好多少。

### 23.3 Dreamer 为什么不需要显式 Advantage？

Dreamer 的 Actor 目标不是：

$$
-\log\pi_\phi(a_\tau\mid s_\tau)\,V_\lambda(s_\tau).
$$

它直接最大化 imagined trajectory 上的 $V_\lambda$：

$$
\mathcal J_{\mathrm{actor}}
=
\mathbb E
\left[
\sum_\tau V_\lambda(s_\tau)
\right].
$$

这里的 $V_\lambda$ 通过 action 依赖 imagined future state，因此通常有：

$$
\frac{\partial V_\lambda}{\partial a_\tau}\ne0.
$$

Actor 得到的是 pathwise 的局部改进方向：

> 当前 action 向哪个方向变化，会让 learned future 的 return 变大？

所以可以这样区分：

| 方法 | 主要得到的信号 |
|---|---|
| PPO / A3C | 采样 action 相对 baseline 的好坏，即 Advantage |
| Dreamer | action 通过可微 dynamics 改变 imagined future 的方向 |

Dreamer 当然仍然训练 Value Model，但 Value 在 Actor update 中是可微 imagined return 的
组成部分，而不是用来做 score-function baseline。

### 23.4 “所有 Value 都很高”并不自动说明 Dreamer 出错

在 pathwise gradient 中，给所有 Value 加上与 action 无关的常数不会改变 Actor 梯度：

$$
\tilde V_\lambda(s)=V_\lambda(s)+C
\quad\Longrightarrow\quad
\frac{\partial\tilde V_\lambda}{\partial a}
=
\frac{\partial V_\lambda}{\partial a}.
$$

真正的问题是 Value 是否对 action 产生了可区分的变化。如果模型对所有可能的
imagined action 都输出完全相同的 Value，那么：

$$
\frac{\partial V_\lambda}{\partial a}=0,
$$

Dreamer 的 Actor gradient 就会消失。此时 PPO 的 Advantage 也会趋近于零，因为
不同 action 的 $Q(s,a)$ 没有可辨别差异。关键不是 Value 的绝对数值高不高，而是它是否
保留了与 action 相关的相对结构。

### 23.5 PPO 和 Dreamer 的 action 都可以 sample

二者的共同部分其实是：

```text
state
  ↓
Actor distribution (μ, σ)
  ↓
sample noise ε
  ↓
action
```

Dreamer 的关键不是“deterministic Actor”，而是 action 之后接着一个可微的 learned
dynamics：

$$
s_{\tau+1}=f_\theta(s_\tau,a_\tau,\xi),
$$

并且后面的 predicted reward 和 Value 仍在同一张计算图中。因此梯度可以继续沿着：

$$
\phi
\rightarrow a_\tau
\rightarrow s_{\tau+1}
\rightarrow \hat r\ \text{或}\ v_\psi
\rightarrow V_\lambda
$$

回到 Actor。

PPO 对连续 action 也可以使用 reparameterization，但真实环境通常是未知、不可微且
只返回结果的黑盒：

```text
action → real environment → next observation / reward
```

训练者无法获得 $\partial s_{t+1}/\partial a_t$ 或 $\partial r_t/\partial a_t$，所以
PPO 依靠 score-function gradient 和 Advantage。真正的分界线不是“PPO sample、Dreamer
不 sample”，而是：

> **Dreamer 有一个可以沿 action 继续反向传播的 learned environment；PPO 面对的是真实
> environment black box。**

---

## 24. 一个 update step 的代码骨架

下面不是作者代码逐行复制，而是严格保持 Algorithm 1 和 Equation 5–8 的计算顺序。

```python
def dreamer_update(replay):

    # ========================================================
    # A. Dynamics Learning
    # ========================================================

    batch = replay.sample(
        batch_size=B,
        sequence_length=L,
    )

    obs = batch["obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]

    # posterior uses real observations
    posterior, prior = rssm.observe(
        obs,
        actions,
    )

    feat = rssm.get_feat(posterior)

    obs_dist = observation_model(feat)
    reward_dist = reward_model(feat)

    loss_obs = -obs_dist.log_prob(obs).mean()
    loss_reward = -reward_dist.log_prob(rewards).mean()

    post_dist = rssm.get_post_dist(posterior)
    prior_dist = rssm.get_prior_dist(prior)

    loss_kl = beta * kl(post_dist, prior_dist).mean()

    model_loss = (
        loss_obs
        + loss_reward
        + loss_kl
    )

    model_optimizer.zero_grad()
    model_loss.backward()
    model_optimizer.step()


    # ========================================================
    # B. Behavior Learning
    # ========================================================

    # each posterior state can become an imagination start
    start_states = flatten_sequence_states(posterior)

    imagined_states = [start_states]
    imagined_rewards = []
    imagined_values = []

    state = start_states

    for _ in range(H):

        # reparameterized actor action
        action = actor.rsample(
            rssm.get_feat(state)
        )

        # transition prior; no future real observation
        state = rssm.imagine_step(
            state,
            action,
        )

        feat = rssm.get_feat(state)

        reward = reward_model(feat).mean
        value = value_model(feat)

        imagined_states.append(state)
        imagined_rewards.append(reward)
        imagined_values.append(value)


    # Equation 5 + Equation 6
    returns = lambda_return(
        rewards=imagined_rewards,
        values=imagined_values,
        gamma=gamma,
        lambda_=lambda_,
    )


    # ========================================================
    # C. Actor
    # ========================================================

    actor_loss = -returns.mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()


    # ========================================================
    # D. Value
    # ========================================================

    value_target = stop_gradient(returns)

    value_loss = value_regression_on_imagined_states(
        imagined_states,
        value_target,
    )

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
```

真正实现时还需处理：

- sequence/time 对齐；
- terminal / continuation；
- distributional log-prob losses；
- free nats；
- discount weighting；
- gradient clipping；
- mixed precision；
- batch/time flatten。

但总体计算图不变。

---

## 25. 官方 TF2 实现里三种 loss 的对应关系

官方实现的 world-model loss 逻辑非常清晰：

```text
image log likelihood
reward log likelihood
KL(post || prior)
       ↓
model loss
```

形式上：

$$
\mathcal L_{\mathrm{model}}
=
\beta\,KL(post\Vert prior)
-
\log p(o\mid s)
-
\log p(r\mid s).
$$

这正好对应 Equation 10 从“最大化 objective”改写成“最小化 loss”。

Behavior 部分则：

```text
imagine ahead
    ↓
reward prediction
value prediction
    ↓
lambda return
    ↓
actor loss
value loss
```

因此论文公式和官方实现的结构是一一对应的。

---

## 26. 第三阶段：Environment Interaction

Algorithm 1 的最后一部分重新进入真实环境。

首先：

$$
o_1\leftarrow env.reset().
$$

每一步：

$$
s_t
\sim
p_\theta(
s_t\mid s_{t-1},a_{t-1},o_t
),
$$

Actor 产生：

$$
a_t
\sim
q_\phi(a_t\mid s_t),
$$

训练数据收集时加入 exploration noise，然后：

$$
r_t,o_{t+1}
\leftarrow
env.step(a_t).
$$

最终整个 episode 加入：

$$
\mathcal D.
$$

---

## 27. 真实执行和 imagination 的状态更新一定要分清

### Behavior Learning

```text
posterior s_t
      ↓
Actor
      ↓
a_t
      ↓
Transition prior
      ↓
imagined s_{t+1}
```

没有真实 observation。

---

### Environment Interaction

```text
posterior s_t
      ↓
Actor
      ↓
a_t
      ↓
真实 Environment
      ↓
真实 o_{t+1}
      ↓
Representation posterior
      ↓
s_{t+1}^{post}
```

所以一句话记忆：

$$
\boxed{
\text{梦里靠 prior，现实里靠新 observation 修正 posterior。}
}
$$

---

## 28. 论文附录给出的真实训练节奏

连续控制实验中，论文附录给出了非常具体的训练方式。

### 初始化

使用：

$$
S=5
$$

个 random-action episodes 初始化 dataset。

---

### 更新节奏

随后反复：

```text
100 gradient steps
        ↓
collect 1 real episode
        ↓
100 gradient steps
        ↓
collect 1 real episode
        ↓
...
```

因此训练主循环可以概念化为：

```python
replay = ReplayBuffer()

for _ in range(5):
    replay.add(
        collect_random_episode(env)
    )

while not_converged:

    for _ in range(100):
        dreamer_update(replay)

    episode = collect_training_episode(
        env,
        world_model,
        actor,
    )

    replay.add(episode)
```

这比只写：

```python
while training:
    dreamer_update(...)
```

更接近论文真实实验协议。

---

## 29. 论文主实验的重要超参数

连续控制设置中：

| 项目 | 设置 |
|---|---:|
| Batch size | 50 sequences |
| Sequence length | 50 |
| stochastic latent size | 30-dimensional diagonal Gaussian |
| imagination horizon | $H=15$ |
| discount | $\gamma=0.99$ |
| lambda | $\lambda=0.95$ |
| world model LR | $6\times10^{-4}$ |
| value LR | $8\times10^{-5}$ |
| actor LR | $8\times10^{-5}$ |
| optimizer | Adam |
| KL scale | $\beta=1$ |
| free nats | 3 |
| seed episodes | 5 |
| training steps between collections | 100 |
| exploration noise | $\mathcal N(0,0.3)$ |
| action repeat | 2 |

其中：

> Dreamer 选择较短的 $H=15$，不是因为只关心 15 步，而是依赖 Value bootstrap 处理 horizon 以后的未来。

---

## 30. Train collection 的代码逻辑

训练期间真实环境 collection：

```python
def collect_training_episode(env, world_model, actor):

    obs = env.reset()

    prev_state = world_model.initial_state()
    prev_action = zeros(action_dim)

    episode = []
    done = False

    while not done:

        # real posterior correction
        state = world_model.observe_step(
            prev_state,
            prev_action,
            obs,
        )

        # paper appendix: predicted mode action
        action = actor.mode(state)

        # training exploration
        action = action + normal_noise(std=0.3)

        next_obs, reward, done, info = env.step(action)

        episode.append((obs, action, reward))

        prev_state = state
        prev_action = action
        obs = next_obs

    return episode
```

这里没有在线 rollout $H=15$ 来规划动作。

Actor 已经在 training imagination 中学好了：

$$
s_t\rightarrow a_t.
$$

---

## 31. Eval：论文与官方实现必须分开说

这是容易写错的地方。

### 31.1 论文 Algorithm 1

Algorithm 1 给出的是 training-time environment interaction，其中包含 exploration noise。

论文并没有另外写一个独立的 `Evaluation Algorithm`。

因此不能把某个 eval 代码细节直接写成：

> “论文 Algorithm 1 规定 eval 这样做。”

---

### 31.2 官方 TensorFlow 2 实现

作者官方实现明确区分：

```python
training=True
```

和：

```python
training=False
```

官方实现中 evaluation：

- 使用独立 test environments；
- 周期性运行 test episode；
- `training=False`；
- Actor 使用 distribution mode；
- 默认 `eval_noise = 0`，因此不添加 exploration noise；
- evaluation episode 不写入训练 replay dataset。

因此 eval 可以理解为：

```python
def evaluate_episode(env, world_model, actor):

    obs = env.reset()

    prev_state = world_model.initial_state()
    prev_action = zeros(action_dim)

    total_reward = 0
    done = False

    while not done:

        state = world_model.observe_step(
            prev_state,
            prev_action,
            obs,
        )

        # deterministic evaluation action
        action = actor.mode(state)

        next_obs, reward, done, info = env.step(action)

        total_reward += reward

        prev_state = state
        prev_action = action
        obs = next_obs

    return total_reward
```

最重要的是：

$$
\boxed{
\text{Eval 不做在线 latent planning。}
}
$$

执行路径依然只是：

$$
o_t
\rightarrow
s_t^{post}
\rightarrow
a_t
\rightarrow
Environment.
$$

---

## 32. Train、Imagination、Eval 三条路径放在一起

### A. World Model Training

```text
真实 (o,a,r)
    ↓
posterior
    ↓
obs / reward / KL losses
    ↓
update θ
```

### B. Actor / Value Training

```text
真实 posterior start
    ↓
Actor
    ↓
Transition
    ↓
imagined future
    ↓
Reward + Value
    ↓
Vλ
   ┌┴┐
   ↓ ↓
   φ ψ
```

### C. Real Execution / Eval

```text
真实 o_t
    ↓
posterior s_t
    ↓
Actor
    ↓
a_t
    ↓
真实 environment
```

这个区分是理解 Dreamer 的核心。

---

## 33. 为什么 Dreamer 不需要像 PlaNet 一样每一步做在线规划？

PlaNet 也学习 RSSM world model。

但 PlaNet 在真实环境每一个决策点会运行类似 CEM 的在线 planning：

```text
当前 latent state
    ↓
采样大量 action sequences
    ↓
world model rollout
    ↓
比较 imagined return
    ↓
选第一步 action
```

下一个真实时间步重新规划。

Dreamer 则将这种“哪种行为好”的计算在训练阶段摊销进 Actor：

```text
training:
world model imagination
        ↓
learn Actor

deployment:
state
 ↓
Actor
 ↓
action
```

因此：

| 项目 | PlaNet | Dreamer |
|---|---|---|
| World Model | RSSM | RSSM |
| online planning | 是 | 否 |
| Actor | 无 | 有 |
| Value | 无 | 有 |
| horizon 外长期价值 | 受 planning horizon 限制 | Value bootstrap |
| deployment cost | 高 | Actor 前向推理 |

可以把 Dreamer 理解成：

> **把反复进行的 latent-space planning，摊销成一个参数化 policy。**

---

## 34. 为什么 Value Model 对 long horizon 特别关键？

论文 Figure 4 对比：

- Dreamer $V_\lambda$；
- 没有 Value prediction、只最大化 horizon 内 reward 的 Actor；
- PlaNet online planning。

结果说明：

> 仅依赖有限 horizon reward 的方法对 horizon length 很敏感，容易短视；加入 learned value 后，Dreamer 对 imagination horizon 更稳健。

所以 Value 的关键作用不是简单“Critic 是 Actor-Critic 标配”。

而是：

$$
\boxed{
\text{显式处理 imagination horizon 以后的回报}
}
$$

---

## 35. Actor 的连续动作分布

原始 Dreamer 连续 Actor 使用 tanh-transformed Gaussian：

$$
a_\tau
=
\tanh
\left(
\mu_\phi(s_\tau)
+
\sigma_\phi(s_\tau)\odot\epsilon
\right),
$$

$$
\epsilon\sim\mathcal N(0,I).
$$

如果机器人 action 是：

$$
a_t\in\mathbb R^7,
$$

那么：

$$
\mu_\phi(s_t)\in\mathbb R^7,
$$

$$
\sigma_\phi(s_t)\in\mathbb R_+^7,
$$

$$
\epsilon\in\mathbb R^7.
$$

最终：

$$
a_t\in[-1,1]^7.
$$

再映射到实际 joint velocity、torque 或 end-effector command 范围。

---

## 36. 为什么用 tanh？

Gaussian 本身范围：

$$
(-\infty,\infty).
$$

机器人动作通常有界。

tanh 把它压到：

$$
[-1,1].
$$

再根据环境 action space 缩放。

实际机器人还需要额外考虑：

- joint limits；
- velocity limits；
- acceleration limits；
- torque limits；
- collision constraints；
- low-level controller；
- emergency stop。

这些不是 Dreamer policy distribution 本身自动解决的。

---

## 37. LLM token action 为什么不需要 Gaussian？

这一点属于方法迁移时的扩展理解。

LLM token action 是离散的：

$$
a_t
\in
\{1,\dots,|\mathcal V|\}.
$$

因此通常使用：

$$
\pi_\phi(a_t=i\mid s_t)
=
\operatorname{softmax}(z_t)_i.
$$

也就是 Categorical distribution。

原则不是：

> RL 必须使用 Gaussian。

而是：

$$
\boxed{
\text{policy distribution 必须匹配 action space 的结构。}
}
$$

连续机器人动作可以 Gaussian / Beta / Flow / Diffusion。

离散 token 则通常 Categorical。

---

## 38. Dreamer 为什么不是“完全不需要真实环境”？

“learning behaviors by latent imagination”容易造成误解。

真正情况是：

### Actor / Value 的行为学习

主要使用：

$$
\text{imagined trajectories}.
$$

### World Model

必须持续使用：

$$
\text{real trajectories}.
$$

### Agent

也必须持续回到：

$$
\text{real environment}
$$

收集新数据。

所以：

$$
\boxed{
\text{Behavior learning mainly in imagination}
\neq
\text{training without real data}
}
$$

---

## 39. Dreamer 的三条“流”

如果后面公式看乱了，可以始终回到三条流。

### 39.1 Real Data Flow

```text
Environment
   ↓
(o,a,r)
   ↓
Replay
   ↓
World Model
```

目标：

> 让内部模型贴近现实。

### 39.2 Imagination Flow

```text
posterior start
     ↓
Actor
     ↓
action
     ↓
Transition
     ↓
future state
  ┌──┴──┐
Reward Value
  └──┬──┘
     ↓
    Vλ
```

目标：

> 不继续调用真实环境，也能产生行为学习信号。

### 39.3 Gradient Flow

#### World Model

$$
\mathcal L_{\mathrm{model}}
\rightarrow
\theta.
$$

#### Value

$$
\operatorname{sg}[V_\lambda]
\rightarrow
\psi.
$$

#### Actor

$$
V_\lambda
\rightarrow
\{\hat r,v\}
\rightarrow
s_{\tau+1}
\rightarrow
a_\tau
\rightarrow
\phi.
$$

目标：

> 把 imagined future consequence 直接变成 Actor gradient。

---

## 40. 三组最容易混淆的概念

### 40.1 Posterior vs Prior

**Posterior**

$$
p_\theta(s_t\mid s_{t-1},a_{t-1},o_t)
$$

看真实 observation。

**Prior**

$$
q_\theta(s_t\mid s_{t-1},a_{t-1})
$$

不看真实 observation。

记忆：

```text
posterior = reality corrected state
prior     = model predicted state
```

### 40.2 Reward vs Value

Reward：

$$
r_t
$$

一步。

Value：

$$
V^\pi(s_t)
=
\mathbb E
\left[
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}
\right].
$$

多步长期。

### 40.3 “不更新参数” vs “不经过计算图”

Behavior Learning 中：

```text
不更新 θ
```

并不意味着：

```text
不让梯度经过 dynamics
```

这两件事必须严格分开。

---

## 41. Dreamer 的主要实验结论

论文在 DeepMind Control Suite 的 20 个视觉控制任务上进行评估。

输入是：

$$
64\times64\times3
$$

图像。

任务覆盖：

- sparse reward；
- contact dynamics；
- 3D；
- 多自由度连续控制。

论文报告在其协议下：

- Dreamer 在 $5\times10^6$ environment steps 后平均约 823；
- PlaNet 同步数约 332；
- D4PG 在 $10^8$ steps 后约 786。

这些结果的意义主要是：

> 在当时的实验协议下，Dreamer 同时取得较强 data efficiency 和 final performance。

---

## 42. Representation objective 的消融

论文比较：

1. pixel reconstruction；
2. contrastive estimation；
3. reward prediction only。

结果中：

- reconstruction 在多数任务最好；
- contrastive 能解决一部分任务；
- reward-only 在这些实验中不足。

这并不意味着：

> 所有后续 world model 都必须 reconstruction。

它说明的是：

> 在原始 Dreamer 的数据规模、任务和 architecture 下，pixel reconstruction 提供了非常有效的 dense representation-learning signal。

---

## 43. Dreamer 的局限

### 43.1 Model error

Transition error 会随 rollout 累积：

$$
s_t
\rightarrow
\hat s_{t+1}
\rightarrow
\hat s_{t+2}
\rightarrow
\cdots
$$

rollout 越长，不一定越好。

### 43.2 Reward model exploitation

如果 Reward Model 在某些 unrealistic latent states 上错误给出高 reward：

$$
\hat r(s)\gg r_{\mathrm{real}},
$$

Actor 可能主动寻找这些 model loopholes。

### 43.3 Value bootstrap 不能消灭 model bias

Value 可以解决：

$$
\text{finite horizon}
$$

但不能自动修复：

- wrong transition；
- wrong reward；
- distribution shift；
- imagined OOD states。

### 43.4 Reconstruction quality 不等价于 control quality

Decoder 可以很好重建背景，却遗漏任务关键小物体。

因此 world model 是否适合 control，更应该关注：

- reward-relevant information；
- action-conditioned dynamics；
- long rollout stability；
- policy exploitation。

---

## 44. 原始 Dreamer 不是今天意义上的通用机器人/VLA

原始论文主要研究：

- single-task control；
- hand-designed reward；
- fixed action space；
- no language condition；
- low-resolution visual input。

因此它对现代 robotics / VLA 最重要的价值不是直接拿来当通用机器人模型，而是提供了一个非常清晰的思想：

$$
\boxed{
\text{如何在 learned latent dynamics 中训练一个连续控制 policy}
}
$$

---

## 45. 从实现角度重新看完整训练闭环

最终可以用下面的伪代码把整篇论文压缩起来：

```python
# ============================================================
# 0. Seed real dataset
# ============================================================

D = ReplayBuffer()

for _ in range(5):
    D.add(
        collect_random_episode(env)
    )


# ============================================================
# 1. Main loop
# ============================================================

while not_converged:

    # paper continuous-control protocol
    for _ in range(100):

        batch = D.sample(B=50, L=50)

        # ----------------------------------------------------
        # Dynamics Learning
        # ----------------------------------------------------

        post, prior = rssm.observe(
            batch.obs,
            batch.action,
        )

        feat = rssm.get_feat(post)

        L_obs = -obs_model(feat).log_prob(
            batch.obs
        ).mean()

        L_reward = -reward_model(feat).log_prob(
            batch.reward
        ).mean()

        L_kl = beta * KL(
            post_dist(post),
            prior_dist(prior),
        ).mean()

        L_model = (
            L_obs
            + L_reward
            + L_kl
        )

        update(theta, L_model)


        # ----------------------------------------------------
        # Behavior Learning
        # ----------------------------------------------------

        starts = flatten(post)

        imagined = imagine(
            starts=starts,
            actor=actor,
            transition=rssm,
            horizon=15,
        )

        r_hat = reward_model(
            imagined.features
        )

        values = value_model(
            imagined.features
        )

        V_lambda = lambda_return(
            r_hat,
            values,
            gamma=0.99,
            lambda_=0.95,
        )

        # gradients pass through imagined dynamics
        L_actor = -V_lambda.mean()
        update(phi, L_actor)

        L_value = value_regression(
            values,
            stop_gradient(V_lambda),
        )
        update(psi, L_value)


    # --------------------------------------------------------
    # Environment Interaction
    # --------------------------------------------------------

    episode = collect_real_episode(
        posterior_model=rssm,
        actor=actor,
        exploration_std=0.3,
    )

    D.add(episode)
```

这段代码真正需要理解的不是语法，而是四个边界：

```text
真实数据在哪里？
→ Dynamics Learning / collection

从哪里开始做梦？
→ replay posterior states

做梦以后还看不看 observation？
→ 不看

Actor gradient 从哪里回来？
→ imagined Vλ through dynamics
```

---

## 46. 一张最终心智模型

如果只留一张图，应该是：

```text
                 ┌──────────────────────┐
                 │      REAL WORLD      │
                 └──────────┬───────────┘
                            │
                    o_t, a_t, r_t
                            │
                            ▼
                    Replay Dataset
                            │
                            ▼
              ┌────────────────────────┐
              │   WORLD MODEL TRAIN    │
              │                        │
              │ posterior from o_t     │
              │ obs reconstruction     │
              │ reward prediction      │
              │ posterior-prior KL     │
              └───────────┬────────────┘
                          │
                   posterior states
                          │
                          ▼
              ┌────────────────────────┐
              │   LATENT IMAGINATION   │
              │                        │
              │ Actor → action         │
              │      ↓                 │
              │ Transition → state     │
              │        ↓               │
              │ Reward + Value         │
              │        ↓               │
              │       Vλ               │
              └──────────┬─────────────┘
                         │
                    ┌────┴────┐
                    ▼         ▼
                  Actor      Value
                    │
                    ▼
              learned policy
                    │
                    ▼
                 REAL WORLD
```

---

## 47. 最终一句话理解 Dreamer

Dreamer 的技术贡献不只是：

> “使用了一个 world model。”

而是把下面几件事真正连成了同一套可训练系统：

$$
\boxed{
\text{真实经验}
\rightarrow
\text{posterior latent state}
\rightarrow
\text{learned dynamics}
\rightarrow
\text{imagined trajectories}
\rightarrow
\text{Reward + Value}
\rightarrow
V_\lambda
\rightarrow
\text{pathwise Actor gradient}
}
$$

它和 PPO/A3C 的核心差异，不是“有没有 Critic”，而是：

> **Actor 是否利用一个可微 learned environment，让梯度通过多步未来后果回传。**

它和 PlaNet 的核心差异，也不是“有没有 world model”，而是：

> **是每个真实时间步重新在线规划，还是把 imagined planning 的结果摊销成一个可直接执行的 Actor。**

最终 deployment 时：

$$
\boxed{
o_t
\rightarrow
s_t^{post}
\rightarrow
a_t
}
$$

Actor 不需要在线 CEM，也不需要每一步进行 $H$-step imagination。

训练阶段在梦里学会如何行动，执行阶段直接行动。

这就是 Dreamer。

### 47.1 五个容易混淆但必须分开的结论

1. Dreamer 不要求 dense reward；sparse reward 可以通过 Reward Model、return、Value 和
   $V_\lambda$ 向前传播，但 bootstrap 不能凭空创造原本不存在的任务信号。
2. Behavior Learning 固定的是 learned model 的参数，不是把 reward 数值固定成常数；不同
   imagined state 仍会得到不同的 predicted reward。
3. Behavior Learning 不是枚举全部 action，而是从当前 Actor 分布采样反事实轨迹，并在
   latent dynamics 中评估和改进 policy。
4. $V_\lambda$ 的 Actor 梯度同时来自 imagined reward path 和 Value bootstrap path，并且
   会混合不同 rollout depth 的路径。
5. PPO 和 Dreamer 都可以 sample action；二者的关键差别是 PPO 经过不可微的真实环境，
   Dreamer 则可以让梯度穿过 learned latent environment。

---

## 48. 阅读检查表

读完以后，如果下面这些问题能明确回答，就基本掌握了原始 Dreamer 的主线：

1. 为什么需要 posterior 和 prior 两种 state distribution？
2. Equation 10 的 observation、reward、KL 三项分别训练什么？
3. 为什么 imagination 只能从真实 posterior 开始，却不能继续读取 replay future observation？
4. Reward Model 和 Value Model 分别解决什么？
5. Equation 5 的 $k$-step return 中，哪部分来自 model rollout，哪部分来自 Value bootstrap？
6. Equation 6 为什么要混合不同 $k$？
7. Actor 的 $V_\lambda$ 梯度如何穿过 action 和 Transition？
8. 为什么“固定 world model 参数”不能理解成 `no_grad`？
9. Environment Interaction 与 Behavior Learning 的 state update 有什么区别？
10. Train collection 和 eval 的 exploration 行为有什么区别？
11. 为什么 Dreamer deployment 不需要 PlaNet 式在线规划？
12. Value Model 为什么使 Dreamer 对 imagination horizon 更鲁棒？
13. sparse reward 是怎样通过 $V_N^k$ 和 $V_\lambda$ 影响更早的 state 的？
14. 为什么固定 Reward Model 参数后，predicted reward 仍然会随 imagined state 改变？
15. PPO 为什么使用 Advantage，而 Dreamer 可以直接对 $V_\lambda$ 做 pathwise update？

如果这 15 个问题形成一条连续逻辑，而不是 15 个孤立答案，那么 Dreamer 的整体框架就真正建立起来了。

---

## 资料来源

- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). *Dream to Control: Learning Behaviors by Latent Imagination*. ICLR 2020.
  Paper: https://arxiv.org/abs/1912.01603

- Dreamer official TensorFlow 2 implementation by Danijar Hafner.
  Repository: https://github.com/danijar/dreamer

---

## 相关主题

- [[PlaNet 论文概述|PlaNet / RSSM]]：理解 Dreamer 之前的潜空间动力学与在线规划。
- [[DreamerV3_技术报告|DreamerV2 / DreamerV3]]：对比离散 RSSM、稳定化技巧和后续 Dreamer 系列。
- [[PPO_逻辑重构版|PPO / Policy Gradient]]：对照 score-function gradient 与 Dreamer 的 pathwise gradient。
- [[SAC_PPO_compare|Actor-Critic 与策略优化]]：比较不同 Actor-Critic 算法的 Critic 与策略更新路径。
