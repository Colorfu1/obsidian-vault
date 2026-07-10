---
title: PPO 与 SAC 对比笔记
type: concept_note
topic: reinforcement_learning
status: mature
importance: high
updated: 2026-06-10
tags:
  - ppo
  - sac
  - reinforcement-learning
  - actor-critic
  - on-policy
  - off-policy
---

# SAC与PPO比较

**User:** Anonymous  
**Created:** 4/28/2026 16:26:31  
**Updated:** 5/5/2026 13:12:07  
**Exported:** 5/5/2026 14:32:57  
**Link:** [https://chatgpt.com/c/69f06f2a-02dc-839c-acca-6d681df78152](https://chatgpt.com/c/69f06f2a-02dc-839c-acca-6d681df78152)  

## Response:
2026/5/5 13:11:12

我会把前面零散的问题整理成一份“从概念到算法选择”的笔记，重点围绕 PPO/SAC 为什么分别用 advantage 和 Q。

# PPO 与 SAC 对比笔记：从 $V,Q,A$ 到 rollout、replay buffer、梯度路径

## 0. 总体直觉

PPO 和 SAC 都是 actor-critic 算法，都有：

$$
\text{Actor} = \pi_\theta(a|s)
$$

$$
\text{Critic} = \text{评价当前状态或动作有多好}
$$

但它们的核心训练逻辑不同：

```text
PPO:
当前策略 rollout
→ 得到真实轨迹
→ 估计 advantage
→ 提高/降低 rollout 中已执行动作的概率

SAC:
replay buffer 存历史 transition
→ 训练 Q critic
→ 当前 actor 重新采样动作
→ Q critic 给动作打分
→ actor 朝高 Q 且高 entropy 的方向更新
```

所以一句话总结：

> PPO 更像“根据真实 rollout 结果修正动作概率”；  
> SAC 更像“训练一个动作评分器 Q，然后让 actor 学会输出评分高的动作”。

---

# 1. Return、$V$、$Q$、Advantage 的关系

## 1.1 Return 是一次实际轨迹的累计收益

一条轨迹：

```text
s0, a0, r0
s1, a1, r1
s2, a2, r2
...
```

从时间 $t$ 开始的 return 是：

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...
$$

它是一次采样结果。

---

## 1.2 $Q^\pi(s,a)$：固定状态和动作后的期望 return

$$
Q^\pi(s,a)=\mathbb{E}[G_t \mid s_t=s,a_t=a]
$$

意思是：

> 在状态 $s$ 下强制执行动作 $a$，之后按照策略 $\pi$ 走，平均能获得多少 return。

所以 $Q$ 是评价 **状态-动作对** 的。

---

## 1.3 $V^\pi(s)$：固定状态后的期望 return

$$
V^\pi(s)=\mathbb{E}[G_t \mid s_t=s]
$$

意思是：

> 在状态 $s$ 下，按照当前策略自己选动作，平均能获得多少 return。

所以 $V$ 是评价 **状态本身** 的。

---

## 1.4 $V$ 和 $Q$ 的关系

$$
V^\pi(s)=\mathbb{E}_{a\sim\pi(\cdot|s)}[Q^\pi(s,a)]
$$

离散动作时：

$$
V^\pi(s)=\sum_a \pi(a|s)Q^\pi(s,a)
$$

也就是说：

> $V(s)$ 是当前策略下所有动作 $Q(s,a)$ 的加权平均。

---

## 1.5 $Q(s,a)$ 和 $V(s')$ 的关系

执行动作 $a$ 后转移到 $s'$，普通 Bellman 关系是：

$$
Q^\pi(s,a)
=
\mathbb{E}_{s'}[r(s,a,s')+\gamma V^\pi(s')]
$$

如果环境是确定性的，可以近似理解成：

$$
Q^\pi(s,a)=r+\gamma V^\pi(s')
$$

所以：

```text
Q(s,a) 不是等于 V(s')
Q(s,a) = 当前 reward + 折扣后的下一状态价值
```

---

## 1.6 Advantage

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

意思是：

> 这个动作 $a$ 比当前状态下策略的平均动作好多少。

例如：

```text
Q(s, 左) = 10
Q(s, 右) = 2
当前策略: 左 70%, 右 30%

V(s)=0.7*10+0.3*2=7.6

A(s, 左)=10-7.6=+2.4
A(s, 右)=2-7.6=-5.6
```

所以：

```text
Q 是绝对好坏
A 是相对当前状态平均水平的好坏
```

---

# 2. Baseline 到底是什么？

在 policy gradient 里，原始梯度可以写成：

$$
\nabla_\theta J
=
\mathbb{E}[
\nabla_\theta \log \pi_\theta(a|s) Q(s,a)
]
$$

但是可以从 $Q(s,a)$ 中减掉一个只依赖状态的函数 $b(s)$，而不改变期望梯度：

$$
\mathbb{E}_{a\sim\pi}
[
\nabla_\theta \log \pi_\theta(a|s)b(s)
]
=0
$$

原因是：

$$
\mathbb{E}_{a\sim\pi}
[
\nabla_\theta \log \pi_\theta(a|s)
]
=
\nabla_\theta \sum_a \pi_\theta(a|s)
=
\nabla_\theta 1
=
0
$$

所以：

$$
\nabla_\theta J
=
\mathbb{E}[
\nabla_\theta \log \pi_\theta(a|s)(Q(s,a)-b(s))
]
$$

最常用的 baseline 是：

$$
b(s)=V(s)
$$

于是：

$$
Q(s,a)-V(s)=A(s,a)
$$

---

## baseline 的直觉

baseline 表示：

> 当前状态本身的平均水平。

减去 baseline 后，训练信号变成：

> 这个动作比当前状态平均表现好还是差。

它不改变期望意义下的梯度方向，但可以显著降低方差。

---

# 3. PPO 的训练逻辑

## 3.1 PPO 是 on-policy

PPO 使用当前策略采样 rollout：

```text
当前 policy
→ 与环境交互
→ 得到 rollout
→ 计算 advantage / return
→ 更新 actor 和 critic 几轮
→ 丢掉这批 rollout
→ 再采样新 rollout
```

它通常不会长期复用旧数据，因为旧数据来自旧策略，而 PPO 的理论假设依赖 on-policy 数据。

---

## 3.2 PPO rollout buffer 存什么？

PPO 通常存每个 timestep 的信息：

$$
(s_t,a_t,r_t,d_t,\log\pi_{\text{old}}(a_t|s_t),V_{\text{old}}(s_t))
$$

rollout 结束后还会计算并存：

$$
A_t,\quad R_t
$$

所以 PPO buffer 常见内容是：

| 内容 | 作用 |
|---|---|
| $s_t$ | update 时重新喂给 actor/critic |
| $a_t$ | rollout 中旧策略实际执行过的动作 |
| $r_t$ | 计算 return / advantage |
| done | 判断 episode 是否结束 |
| old log prob | 计算 PPO ratio |
| old value | 计算 GAE |
| advantage $A_t$ | actor update 的权重 |
| return target $R_t$ | critic update 的监督目标 |

---

## 3.3 PPO actor 怎么更新？

PPO 在 update 时不重新采样动作，而是使用 rollout 中固定的旧动作：

$$
a_t
$$

计算新策略对这个旧动作的 log probability：

$$
\log\pi_\theta(a_t|s_t)
$$

然后和旧策略的 log probability 做 ratio：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\text{old}}(a_t|s_t)}
=
\exp(
\log\pi_\theta(a_t|s_t)
-
\log\pi_{\text{old}}(a_t|s_t)
)
$$

PPO clipped objective：

$$
L^{CLIP}
=
\mathbb{E}
[
\min(
r_t A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
)
]
$$

直观上：

```text
如果 A_t > 0:
    说明这个旧动作比预期好
    提高它的概率

如果 A_t < 0:
    说明这个旧动作比预期差
    降低它的概率

clip:
    限制新旧策略变化不要太大
```

---

## 3.4 PPO 里的 log probability 怎么来的？

连续动作 PPO 中，actor 通常输出高斯分布参数：

$$
\mu_\theta(s),\quad \sigma_\theta(s)
$$

这定义了：

$$
\pi_\theta(\cdot|s)=\mathcal{N}(\mu_\theta(s),\sigma_\theta(s))
$$

rollout 时采样：

$$
a_t\sim\pi_{\text{old}}(\cdot|s_t)
$$

并存：

$$
\log\pi_{\text{old}}(a_t|s_t)
$$

update 时，不重新采样，而是把旧动作 $a_t$ 放到新策略分布里算：

$$
\log\pi_\theta(a_t|s_t)
$$

所以 PPO 的 log prob 是：

> 新策略对 rollout 中旧动作的概率密度。

---

## 3.5 PPO critic 怎么更新？

PPO 的 critic 学的是：

$$
V_\phi(s_t)\approx R_t
$$

critic loss：

$$
L_V=(V_\phi(s_t)-R_t)^2
$$

这里的 $R_t$ 通常由 GAE 和 old value 构造：

$$
R_t = A_t + V_{\text{old}}(s_t)
$$

---

## 3.6 PPO 的 advantage 在一批 update 里是固定的吗？

是的。

PPO 通常流程是：

```text
rollout
→ 计算并固定 A_t / R_t
→ 用这批固定标签训练 actor 和 critic 多个 epoch
→ 丢掉这批数据
→ 重新 rollout
```

critic 在当前 batch 上更新后，主要作用体现在下一轮 rollout：下一轮会用更新后的 critic 计算更准确的 $V(s)$，进而得到更好的 advantage。

---

## 3.7 PPO 的方差来源

PPO 的主要问题是：

> 用有限 rollout 样本估计 policy gradient，因此有较大采样方差。

来源包括：

```text
1. 同一个状态下只采到了有限动作
2. 策略本身有随机性
3. 环境转移有随机性
4. 后续 return 波动大
5. advantage / value 估计有误差
```

baseline / advantage 的作用就是降低这种方差。

---

# 4. SAC 的训练逻辑

## 4.1 SAC 是 off-policy

SAC 使用 replay buffer：

$$
D=\{(s_t,a_t,r_t,s_{t+1},d_t)\}
$$

这些数据可以来自历史策略，不一定来自当前策略。

SAC 的核心流程：

```text
环境交互，存 transition 到 replay buffer
从 replay buffer 采样 batch
更新 Q critic
更新 actor
软更新 target Q 网络
```

---

## 4.2 SAC replay buffer 存什么？

SAC buffer 核心存：

| 内容 | 作用 |
|---|---|
| $s_t$ | critic update 和 actor update 都会用 |
| $a_t$ | 训练 $Q(s_t,a_t)$ |
| $r_t$ | critic target |
| $s_{t+1}$ | critic target |
| done | 判断是否 bootstrap |

即：

$$
(s_t,a_t,r_t,s_{t+1},d_t)
$$

SAC 一般不存：

```text
old log prob
advantage
return target
V(s)
```

---

## 4.3 为什么 SAC 不能只存 state？

因为 actor update 阶段确实主要只需要 $s$：

$$
s\sim D
$$

当前 actor 重新采样：

$$
a_{\text{new}}\sim\pi_\theta(\cdot|s)
$$

然后用 Q 打分：

$$
Q(s,a_{\text{new}})
$$

但是 critic update 必须知道真实环境 transition：

$$
(s,a,r,s')
$$

critic 要学：

$$
Q(s,a)\approx r+\gamma(\cdots)
$$

所以 SAC 必须存完整 transition。

---

## 4.4 SAC critic 学什么？

SAC critic 学 soft Q：

$$
Q(s,a)
$$

它评价：

> 在状态 $s$ 下执行动作 $a$，之后按照当前策略走，能获得多少 soft return。

soft return 是：

$$
reward + entropy bonus
$$

SAC critic target 是：

$$
y
=
r
+
\gamma
\left(
\min_i Q_{\bar{\phi}_i}(s',a')
-
\alpha\log\pi_\theta(a'|s')
\right)
$$

其中：

$$
a'\sim\pi_\theta(\cdot|s')
$$

---

## 4.5 公式里的 $i$ 是什么？

SAC 通常有两个 Q 网络：

$$
Q_{\phi_1},\quad Q_{\phi_2}
$$

也有两个 target Q 网络：

$$
Q_{\bar{\phi}_1},\quad Q_{\bar{\phi}_2}
$$

所以：

$$
\min_i Q_{\bar{\phi}_i}(s',a')
$$

就是：

$$
\min(
Q_{\bar{\phi}_1}(s',a'),
Q_{\bar{\phi}_2}(s',a')
)
$$

取 min 是为了缓解 Q-learning 的过估计问题。

---

## 4.6 soft Q 不是 “两个 Q 取 min”

这是一个容易混淆的点。

```text
soft Q:
    Q 里面包含 entropy bonus

twin Q / clipped double Q:
    两个 Q 网络取较小值，减少过估计
```

所以：

$$
\min(Q_1,Q_2)
$$

不是 soft 的来源。

soft 的来源是目标中包含：

$$
-\alpha\log\pi(a|s)
$$

也就是 entropy bonus。

---

## 4.7 $r+\alpha H$ 里的 $H$ 怎么执行？

理论上 SAC 最大化：

$$
\mathbb{E}\left[\sum_t \gamma^t(r_t+\alpha \mathcal{H}(\pi(\cdot|s_t)))\right]
$$

熵：

$$
\mathcal{H}(\pi(\cdot|s))
=
\mathbb{E}_{a\sim\pi}[-\log\pi(a|s)]
$$

实际实现时通常采样一个动作近似：

```python
a_next, logp_next = actor.sample(s_next)

target = r + gamma * (
    min(target_q1(s_next, a_next), target_q2(s_next, a_next))
    - alpha * logp_next
)
```

这里：

$$
-\alpha\log\pi(a'|s')
$$

就是 entropy bonus。

---

## 4.8 SAC actor 怎么更新？

SAC actor loss：

$$
L_\pi
=
\mathbb{E}_{s\sim D,a\sim\pi_\theta}
[
\alpha\log\pi_\theta(a|s)-Q_\phi(s,a)
]
$$

最小化它等价于：

```text
提高 Q(s,a)
同时保持策略 entropy
```

也就是：

```text
Q 项:
    让 actor 输出高价值动作

entropy 项:
    不要让策略太快塌缩成确定性动作
```

---

# 5. SAC 的重参数化和 PPO 的采样有什么不同？

## 5.1 两者 forward 分布一样

SAC actor 输出：

$$
\mu_\theta(s),\quad \sigma_\theta(s)
$$

可以直接写：

$$
a\sim\mathcal{N}(\mu_\theta(s),\sigma_\theta(s))
$$

也可以重参数化写成：

$$
\epsilon\sim\mathcal{N}(0,1)
$$

$$
a_\theta=\mu_\theta(s)+\sigma_\theta(s)\epsilon
$$

从采样分布看，二者一样。

---

## 5.2 区别在 backward

重参数化让动作变成 actor 参数的可微函数：

$$
a_\theta=f_\theta(s,\epsilon)
$$

于是 SAC actor loss 中的：

$$
Q(s,a_\theta)
$$

可以通过 action 反传到 actor：

$$
\nabla_\theta Q(s,a_\theta)
=
\nabla_a Q(s,a)
\nabla_\theta a_\theta
$$

所以 SAC 的 actor 更新可以理解成：

> critic Q 告诉 actor，动作往哪个方向改，Q 会更高。

---

## 5.3 PPO 也是采样动作，但梯度路径不同

PPO 也是 actor 输出分布参数并采样动作。

但是 PPO update 时通常使用 rollout 中固定旧动作：

$$
a_t
$$

梯度主要来自：

$$
\nabla_\theta \log\pi_\theta(a_t|s_t)A_t
$$

也就是说：

```text
PPO:
    提高或降低旧动作的概率

SAC:
    当前 actor 重新采样动作
    critic Q 对动作打分
    Q 的梯度通过 action 回到 actor
```

---

# 6. PPO 和 SAC 的 log probability 有什么不同？

二者都计算：

$$
\log\pi(a|s)
$$

但语义不同。

## 6.1 PPO 的 log prob

PPO update 时：

$$
a_t
$$

是 rollout 中固定的旧动作。

计算：

$$
\log\pi_\theta(a_t|s_t)
$$

用于和 old log prob 做 ratio：

$$
r_t(\theta)
=
\exp(
\log\pi_\theta(a_t|s_t)
-
\log\pi_{\text{old}}(a_t|s_t)
)
$$

所以 PPO 的 log prob 用于：

> 衡量新策略对旧动作的概率变化。

---

## 6.2 SAC 的 log prob

SAC actor update 时：

$$
a_\theta\sim\pi_\theta(\cdot|s)
$$

计算：

$$
\log\pi_\theta(a_\theta|s)
$$

用于 actor loss：

$$
\alpha\log\pi_\theta(a_\theta|s)-Q(s,a_\theta)
$$

所以 SAC 的 log prob 用于：

> entropy regularization，让策略保持随机性。

---

## 6.3 对比

| 项 | PPO | SAC |
|---|---|---|
| 动作 | rollout 中旧动作 $a_t$ | 当前 actor 新采样动作 $a_\theta$ |
| log prob 用途 | 新旧策略 ratio | entropy bonus |
| 是否需要 old log prob | 需要 | 不需要 |
| 是否重新采样动作 | update 时通常不重新采样 | actor update 时重新采样 |
| actor 更新核心 | 提高/降低旧动作概率 | 让当前动作高 Q 且高熵 |

---

# 7. 为什么 PPO 用 advantage，SAC 用 Q？

这不是因为数学上“必须”，而是因为它们的数据来源和更新方式不同。

---

## 7.1 PPO 为什么自然用 advantage？

PPO 有真实 rollout：

$$
(s_t,a_t,r_t,s_{t+1},...)
$$

可以估计这个已执行动作的 return：

$$
G_t
$$

然后用：

$$
A_t\approx G_t-V(s_t)
$$

判断：

> 这个已执行动作 $a_t$ 比当前状态的平均水平好还是差。

PPO actor update 本质是：

```text
如果 A_t > 0:
    提高这个旧动作的概率

如果 A_t < 0:
    降低这个旧动作的概率
```

所以 PPO 的锚点是：

$$
A(s,a)
$$

即“相对平均水平的好坏”。

---

## 7.2 SAC 为什么自然用 Q？

SAC 从 replay buffer 抽状态：

$$
s\sim D
$$

当前 actor 重新采样动作：

$$
a_\theta\sim\pi_\theta(\cdot|s)
$$

这个动作不一定在环境里真实执行过。

所以必须有一个 critic 能回答：

> 当前 actor 新生成的这个动作 $a_\theta$ 好不好？

这就需要：

$$
Q(s,a_\theta)
$$

如果只有 $V(s)$，同一个状态下所有动作都得到同一个分数，无法区分动作好坏。

所以 SAC 的锚点是：

$$
Q(s,a)
$$

即“具体动作的价值”。

---

# 8. PPO 的数据为什么不能无限复用？

PPO 是 on-policy。

一批数据来自旧策略：

$$
\pi_{\text{old}}
$$

PPO 可以用 ratio 和 clip 短暂复用几轮：

$$
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\text{old}}(a_t|s_t)}
$$

但随着更新进行，新策略和旧策略越来越不一样，旧数据就越来越 off-policy。

所以 PPO 通常：

```text
一批 rollout 用几个 epoch
然后丢掉
重新采样
```

---

## 能不能保留 state，然后重新采样另一个 action？

理论上可以，但你不知道这个新动作的 advantage。

原始数据里的：

$$
A_t
$$

只对应原来执行过的：

$$
a_t
$$

如果换成：

$$
a_{\text{new}}
$$

原来的 advantage 不再适用。

要评价新动作，你需要：

```text
1. 重新进环境 rollout
或者
2. 训练 Q(s,a) 来估计它
```

第 2 条就开始接近 SAC 了。

---

# 9. SAC 和 DQN 的关系

你的理解可以总结为：

> SAC 可以看作 DQN / Double Q-learning 风格的 critic，加上一个随机 actor 和 entropy 正则。

DQN critic target：

$$
y=r+\gamma\max_{a'}Q_{\text{target}}(s',a')
$$

SAC critic target：

$$
y=
r+\gamma
\left(
\min_i Q_{\bar{\phi}_i}(s',a')
-
\alpha\log\pi(a'|s')
\right)
$$

相似点：

```text
都用 replay buffer
都用 Q-learning
都用 target network
都用 Bellman backup
```

不同点：

| 项 | DQN | SAC |
|---|---|---|
| 动作空间 | 主要离散 | 主要连续 |
| 下一动作 | $\arg\max Q$ | actor 采样 |
| critic | 单/双 Q | twin Q |
| target | hard max | soft value |
| actor | 没有显式 actor | 有 stochastic actor |
| 探索 | epsilon-greedy | maximum entropy |

SAC 的 actor 可以理解成：

> 连续动作空间里可学习的 soft argmax。

因为连续动作不能枚举所有 $a$ 找最大 Q，所以 actor 学会直接产生高 Q 的动作分布。

---

# 10. PPO 和 SAC 存储内容对比

| 项目 | PPO rollout buffer | SAC replay buffer |
|---|---|---|
| 数据类型 | 短期 rollout | 长期 transition |
| 数据来源 | 当前策略 | 历史策略 / 当前策略 |
| 是否 on-policy | 是 | 否 |
| 是否长期保存 | 通常否 | 是 |
| 存 $s_t$ | 是 | 是 |
| 存 $a_t$ | 是 | 是 |
| 存 $r_t$ | 是 | 是 |
| 存 $s_{t+1}$ | 有些显式存，有些通过下一条 obs 得到 | 是 |
| 存 done | 是 | 是 |
| 存 old log prob | 是 | 否 |
| 存 $V(s_t)$ | 通常是 | 否 |
| 存 advantage | 采样后计算并存 | 否 |
| 存 return target | 是 | 否 |
| actor update 用什么 | $s_t,a_t,A_t,\log p_{\text{old}}$ | $s_t$，然后重新采样动作 |
| critic update 用什么 | $s_t,R_t$ | $s_t,a_t,r_t,s_{t+1},d_t$ |
| 数据复用 | 少量 epoch | 长期反复采样 |

---

# 11. PPO 和 SAC 梯度方向的区别

## PPO

简化 actor loss：

$$
L_\pi=-\log\pi_\theta(a_t|s_t)A_t
$$

梯度：

$$
\nabla_\theta L_\pi
=
-
A_t\nabla_\theta\log\pi_\theta(a_t|s_t)
$$

解释：

```text
∇ logπ:
    给出如何提高旧动作概率的方向

A_t:
    控制这个方向是增强还是反向，以及强度
```

如果：

$$
A_t>0
$$

提高旧动作概率。

如果：

$$
A_t<0
$$

降低旧动作概率。

---

## SAC

actor loss：

$$
L_\pi=
\alpha\log\pi_\theta(a_\theta|s)-Q_\phi(s,a_\theta)
$$

其中：

$$
a_\theta=f_\theta(s,\epsilon)
$$

梯度包含：

$$
\nabla_a Q(s,a)\nabla_\theta a_\theta
$$

解释：

```text
Q 项:
    告诉 actor 动作往哪个方向变，价值会更高

entropy 项:
    让策略不要太确定，保持探索
```

所以：

```text
PPO:
    调整旧动作的概率

SAC:
    让当前动作沿 Q 的梯度方向变好
```

---

# 12. PPO 和 SAC 的误差 / 方差来源

## PPO

主要问题：

> on-policy rollout 的采样方差较大，样本效率较低。

来源：

```text
有限 rollout 样本
动作采样随机性
环境随机性
return 波动
advantage 估计误差
```

PPO 的优势是：

```text
数据来自当前策略
训练目标比较干净
稳定性好
```

---

## SAC

主要问题：

> off-policy bootstrapped Q-learning 的估计偏差和误差传播。

来源：

```text
Q target 本身是估计值
bootstrap error 会传播
replay buffer 数据来自旧策略
actor 可能生成数据分布外动作
Q 可能过估计
function approximation error
```

SAC 的优势是：

```text
可以长期复用数据
样本效率高
适合连续控制
探索能力强
```

---

# 13. 最终核心总结

## PPO 的逻辑链条

```text
1. 当前策略 rollout
2. 存下每一步的 s, a, r, old_log_prob, V
3. 计算 return / advantage
4. 对 rollout 中固定旧动作:
       A > 0 → 提高概率
       A < 0 → 降低概率
5. 用 clip 限制新旧策略变化
6. critic 拟合 return target
7. 丢掉旧 rollout，重新采样
```

PPO 的核心锚点是：

$$
A(s,a)=Q(s,a)-V(s)
$$

因为它关心：

> 这个已执行动作比当前状态平均水平好还是差？

---

## SAC 的逻辑链条

```text
1. 与环境交互，存 transition 到 replay buffer
2. 用 (s,a,r,s') 训练 Q critic
3. target = r + gamma * soft next value
4. actor 从 replay buffer 的 s 上重新采样动作
5. Q critic 给当前动作打分
6. actor 更新: 高 Q + 高 entropy
7. target Q 软更新
```

SAC 的核心锚点是：

$$
Q(s,a)
$$

因为它关心：

> 当前 actor 新生成的这个动作本身好不好？

---

# 14. 一句话版本

```text
PPO:
    用真实 rollout 给旧动作打标签，
    根据 advantage 提高或降低旧动作概率。
    稳定，但样本效率低。

SAC:
    用 replay buffer 训练 Q 动作评分器，
    再让 actor 产生高 Q 且高 entropy 的动作。
    样本效率高，但依赖 Q 估计质量。
```

最关键的区别不是 “谁有 actor / critic”，而是：

```text
PPO 的 critic 主要是 baseline，用来得到 advantage；
SAC 的 critic 是动作评分器，用来直接指导 actor 改动作。
```

## 相关笔记

- [[RL/ChatGPT-PPO|PPO]]：policy gradient、advantage、clipped surrogate 的详细解释。
- [[ChatGPT-Pi_star0.6论文问题解答|pi*0.6 / RECAP]]：advantage-conditioned robot policy 的对照。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
