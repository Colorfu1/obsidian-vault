
---
title: PlaNet 论文概述
type: paper_note
topic: latent_world_model_planning
status: mature
importance: high
updated: 2026-07-10
tags:
  - planet
  - world-model
  - model-based-rl
  - rssm
  - latent-dynamics
  - cem-planning
  - pixel-control
  - robotics
---

# 《Learning Latent Dynamics for Planning from Pixels / PlaNet》技术报告

## 1. 论文定位

这篇论文提出 **PlaNet / Deep Planning Network**，目标是：**只从像素图像中学习环境动力学模型，然后在 latent space 里做在线规划，输出连续控制动作**。它不是训练一个 policy network 直接输出 action，而是训练一个 world model，再用 CEM planner 每一步搜索动作序列。论文明确强调 PlaNet 是一个纯 model-based agent，从图像学习 dynamics，并通过 compact latent space 中的 online planning 选择动作。

这篇论文可以看作 **Dreamer 系列的前传**：

- PlaNet：学习 latent dynamics，然后 inference 时每一步用 CEM 在线规划；
- Dreamer：沿用 RSSM 世界模型，但在 latent imagination 中训练 actor-critic，inference 时由 actor 直接输出动作。

所以 PlaNet 的核心价值不在于它最终性能多强，而在于它奠定了后续 Dreamer/world model RL 的基本结构：**RSSM + latent dynamics + imagination/planning**。

---

## 2. 要解决的问题

传统 planning 在已知动力学的系统中很强，但真实环境 dynamics 往往未知。如果要从图像中学习 dynamics，再用于 planning，会遇到几个问题：

1. 图像是高维输入，直接在 pixel space 里做 planning 计算量太大；
2. 单帧图像通常不是完整状态，所以问题是 POMDP；
3. learned model 的多步 rollout 容易累积误差；
4. 未来可能有多种可能性，纯 deterministic model 容易过度自信；
5. planner 会主动利用模型漏洞，导致 learned model 的小误差被放大。

PlaNet 的解决思路是：**训练时从图像学习 latent dynamics，planning 时只在 latent space 里预测未来 reward，不生成未来图像**。论文说 observation model 提供 rich training signal，但不用于 planning；真正 planning 时用的是 transition model 和 reward model。

---

## 3. 方法总览

PlaNet 有四个需要学习的模块：

| 模块 | 公式 | 作用 | Planning 时是否使用 |
|---|---|---|---|
| Encoder / posterior | $$q(s_t \mid o_{\le t}, a_{<t})$$ | 根据历史图像和动作推断当前 latent belief | 使用 |
| Transition model / prior | $$p(s_t \mid s_{t-1}, a_{t-1})$$ | 在 latent space 中预测下一步状态 | 使用 |
| Reward model | $$p(r_t \mid s_t)$$ | 从 latent state 预测即时 reward | 使用 |
| Observation model | $$p(o_t \mid s_t)$$ | 从 latent state 重建图像，提供训练信号 | 不用于 planning |

论文的问题设定是 POMDP：时间步为 $$t$$，隐藏状态为 $$s_t$$，图像观测为 $$o_t$$，动作为连续动作向量 $$a_t$$，奖励为标量 $$r_t$$

---

## 4. Algorithm 1：训练与数据采集闭环

Algorithm 1 是 PlaNet 的外层流程。它不是单纯 inference，也不是单纯训练，而是一个 **model learning + data collection** 的闭环。论文中 Algorithm 1 包含：随机 seed episodes 初始化数据集、训练模型、用当前模型做 planning 收集新 episode、再把新 episode 加回数据集。

整体流程可以理解成：

```text
随机动作收集 S 条 seed episodes
初始化世界模型参数 θ

while 未收敛:
    用当前数据集训练世界模型 C 步
    reset 环境
    for 当前 episode 的每个决策步:
        用 encoder 推断当前 latent belief
        调用 planner 选择动作
        加探索噪声
        将同一个动作重复执行 R 次
        记录累计 reward 和最后 observation
    把新 episode 加入数据集 D
```

几个容易误解的点：

**Action repeat 不是 reset 后重复动作。**  
如果 action repeat 为 $$R$$，planner 输出一个动作 $$a_t$$ 后，环境会连续执行：

$$
\text{env.step}(a_t), \text{env.step}(a_t), \dots, \text{env.step}(a_t)
$$

共 $$R$$ 次。每次环境状态和 observation 都会变化，但动作值保持不变。最后 reward 累加，observation 取最后一帧：

$$
r_t = \sum_{k=1}^{R} r_t^k
$$

$$
o_{t+1} = o_{t+1}^{R}
$$

论文说明 action repeat 用来缩短 planning horizon，并提供更清晰的学习信号。

**Line 11 里的 `p` 不是噪声。**  
`planner(q(...), p)` 里的 $$p$$ 指 learned model，主要包括 transition model 和 reward model。探索噪声是单独的 $$p(\epsilon)$$，在 action 选出来后再加上去。论文写到数据采集时会向动作加入 small Gaussian exploration noise。

**PlaNet 没有 action head。**  
Algorithm 1 只提供当前 latent belief；真正动作选择由 Algorithm 2 完成。

---

## 5. Algorithm 2：CEM latent planning

Algorithm 2 是 PlaNet 的动作选择器。它做的事情是：**从当前 latent belief 出发，在 learned dynamics 里搜索未来动作序列，然后只执行第一个动作**。论文明确说 PlaNet 使用 CEM 搜索最佳动作序列，使用 MPC 每一步重新规划，并且不使用 policy 或 value network。

CEM planning 的步骤是：

1. 初始化未来动作序列分布：

$$
q(a_{t:t+H}) = \mathcal{N}(0, I)
$$

2. 采样 $$J$$ 条候选动作序列：

$$
a_{t:t+H}^{(j)} \sim q(a_{t:t+H})
$$

3. 对每条动作序列，从当前 latent belief 出发，用 transition model 在 latent space 中 rollout：

$$
s_{t+1} \sim p(s_{t+1} \mid s_t, a_t)
$$

$$
s_{t+2} \sim p(s_{t+2} \mid s_{t+1}, a_{t+1})
$$

4. 用 reward model 预测未来 reward，并累加：

$$
R^{(j)} =
\sum_{\tau=t+1}^{t+H+1}
\mathbb{E}
[
p(r_\tau \mid s_\tau^{(j)})
]
$$

5. 选出 reward 最高的 top-K 条动作序列，重新拟合高斯分布的均值和方差；
6. 重复上述优化 $$I$$ 次；
7. 返回当前时刻动作均值 $$\mu_t$$。

论文附录给出的默认 planning 超参数是：horizon $$H=12$$，CEM 迭代 $$I=10$$，每轮候选数 $$J=1000$$，top-K 为 $$K=100$$。

一个关键点是：**每个环境决策步都会重新初始化动作序列分布为标准高斯**，而不是沿用上一步 CEM 的结果。论文解释说，收到下一个 observation 后，动作序列 belief 再次从 zero mean、unit variance 开始，以避免陷入局部最优。

所以 PlaNet inference 时不是：

```text
observation → network → action
```

而是：

```text
observation history
    ↓
encoder 得到当前 latent belief
    ↓
CEM 在 latent space 中搜索动作序列
    ↓
返回第一步动作
```

---

## 6. RSSM：Recurrent State-Space Model

PlaNet 最核心的模型结构是 RSSM。它把 latent state 分成两个部分：

- deterministic state：$$h_t$$
- stochastic state：$$s_t$$

RSSM 的形式是：

$$
h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
$$

$$
s_t \sim p(s_t \mid h_t)
$$

$$
o_t \sim p(o_t \mid h_t, s_t)
$$

$$
r_t \sim p(r_t \mid h_t, s_t)
$$

其中 $$f$$ 通常由 RNN/GRU 实现。论文把这个模型理解为把 state 拆成 stochastic part 和 deterministic part：$$h_t$$ 负责通过 RNN 记忆长期历史，$$s_t$$ 负责表达随机性和多种可能未来。

### 为什么需要 deterministic path？

纯 stochastic state-space model 理论上也能记忆信息，但实际优化很难让它长期稳定记忆。$$h_t$$ 的作用是把过去信息通过 RNN 确定性地传下来，帮助模型处理部分可观测问题。

### 为什么需要 stochastic path？

纯 deterministic RNN 容易只能预测单一未来，无法表达多种可能性；而且 planner 可能利用 deterministic model 的错误，产生看似高 reward、实际无效的动作。论文 Figure 2 说明：纯 deterministic transition 无法捕获 multiple futures；纯 stochastic transition 又难以长期记忆；RSSM 同时保留二者。

实验也支持这一点：论文对比 RSSM、纯 GRU、纯 SSM，发现 deterministic 和 stochastic components 对 planning 性能都很重要；deterministic part 让模型记住多步信息，stochastic component 对性能甚至更关键。

---

## 7. Encoder、prior、posterior 的关系

训练时，encoder 会一步步推断 posterior：

$$
q(s_t \mid h_t, o_t)
$$

它看到了当前真实 observation，所以可以推断一个比较准确的当前 latent state。

而 transition prior 是：

$$
p(s_t \mid h_t)
$$

它不能看当前图像，只能根据历史 latent 和 action 预测当前 state。

所以训练时本质上有两个分布：

- posterior：看过当前图像，比较“准”；
- prior：没看当前图像，只靠 dynamics 预测。

训练目标会让 prior 靠近 posterior。这样 planning 时虽然没有未来 observation，也能用 transition prior 往未来 rollout。

论文特别强调：所有 observation 信息必须经过 encoder 的 sampling step，避免从输入图像到 reconstruction 的 deterministic shortcut。原因是，如果 decoder 可以绕过 stochastic latent 直接拿 encoder feature 重建图像，那么 reconstruction loss 会很好，但 latent dynamics 可能什么有用信息都没学到。planning 时未来图像不存在，这种 shortcut 就会失效。

---

## 8. 训练目标：重建 + reward + KL 对齐

论文里的标准 variational objective 可以理解成三部分。

### 8.1 Observation reconstruction loss

要求 latent state 能重建图像：

$$
\mathbb{E}_{q(s_t)}
[
\log p(o_t \mid s_t)
]
$$

在 RSSM 中是：

$$
\mathbb{E}_{q(s_t)}
[
\log p(o_t \mid h_t, s_t)
]
$$

它的作用是让 latent state 保留观察信息。

### 8.2 Reward prediction loss

论文说 “reward losses follow by analogy”，意思是：reward 和 observation 一样，也用 likelihood loss 加到训练目标里。

也就是：

$$
\mathbb{E}_{q(s_t)}
[
\log p(r_t \mid h_t, s_t)
]
$$

reward model 被设为 scalar Gaussian，均值由 feed-forward network 输出，方差固定为 1。因为 unit variance Gaussian 的负 log-likelihood 等价于 MSE 加常数，所以 reward loss 基本就是预测 reward 和环境返回 reward 的 MSE。

### 8.3 KL alignment loss

让 prior 和 posterior 对齐：

$$
\mathrm{KL}
\left[
q(s_t \mid h_t, o_t)
\;\|\;
p(s_t \mid h_t)
\right]
$$

直观理解是：

> encoder 看到了真实图像，知道当前 latent 应该是什么；transition model 没看到当前图像，只能靠历史和动作预测。训练时让 transition prior 学着接近 encoder posterior。

所以 PlaNet 的训练目标不是单纯“图像重建”，而是：

> reconstruction 让 latent 不空；reward loss 让 latent 和任务相关；KL 对齐让 transition model 学会不用未来 observation 也能预测 latent。

---

## 9. Latent Overshooting

标准 ELBO 的问题是：它主要训练 one-step prediction。当前 KL loss 是：

$$
\mathrm{KL}
\left[
q(s_t)
\;\|\;
p(s_t \mid s_{t-1}, a_{t-1})
\right]
$$

其中 $$s_{t-1}$$ 通常来自 encoder posterior：

$$
s_{t-1} \sim q(s_{t-1})
$$

所以 transition model 每次都在“被真实 observation 校正过的上一时刻 latent”上做一步预测。论文指出，标准 objective 的梯度只通过一步 transition，然后直接进入 $$q(s_{t-1})$$，不会穿过多个 transition model 组成的 chain。

但 planning 时没有未来 observation，模型必须这样连续 rollout：

$$
s_t
\rightarrow
p(s_{t+1} \mid s_t, a_t)
\rightarrow
p(s_{t+2} \mid s_{t+1}, a_{t+1})
\rightarrow
p(s_{t+3} \mid s_{t+2}, a_{t+2})
$$

也就是说，训练时是 teacher-forced one-step prediction；planning 时是 open-loop multi-step rollout。这会造成 train-test mismatch。

Latent overshooting 的思想是：

> 从更早的 latent 出发，连续用 transition model rollout 多步，然后让多步预测出来的 latent 接近 encoder 看到真实图像后得到的 posterior latent。

比如不仅训练：

$$
q(s_{t-1}) \rightarrow p(s_t)
$$

还训练：

$$
q(s_{t-2}) \rightarrow p(s_{t-1}) \rightarrow p(s_t)
$$

以及：

$$
q(s_{t-3}) \rightarrow p(s_{t-2}) \rightarrow p(s_{t-1}) \rightarrow p(s_t)
$$

最后都要求接近：

$$
q(s_t)
$$

它和 observation overshooting 的区别是：observation overshooting 会把多步预测 decode 成图像再比对，计算很贵；latent overshooting 只在 latent space 里对齐，计算更便宜。论文将 latent overshooting 解释为一种 latent-space regularizer，用来鼓励 one-step 和 multi-step predictions 的一致性，并且对 overshooting 距离大于 1 的 posterior 停止梯度，让多步 prediction 向 informed posterior 学习。

不过这篇论文有个容易被忽略的结论：**latent overshooting 是重要思想，但不是最终 PlaNet RSSM 性能的必要条件**。附录实验显示，它对某些模型有帮助，但对最终 RSSM 可能略微降低性能；因此读这篇时更应该把 RSSM 和 latent planning 作为主线。

---

## 10. Inference 流程

PlaNet inference / evaluation 时仍然用 Algorithm 2。它没有训练一个 policy head，因此不会直接输出 action。

每个决策步流程是：

1. 根据历史 observation/action，用 encoder 推断当前 latent belief；
2. 初始化动作序列分布为标准高斯：

$$
q(a_{t:t+H}) = \mathcal{N}(0, I)
$$

3. CEM 采样大量连续动作序列；
4. 用 transition model 在 latent space 里 rollout；
5. 用 reward model 给每条动作序列打分；
6. 选 top-K，更新动作序列分布；
7. 迭代若干次后返回当前动作均值 $$\mu_t$$；
8. 执行动作，拿到新 observation；
9. 下一步重新规划，重新初始化动作序列分布。

训练采集时会加 exploration noise；正式 evaluation 通常不加这个噪声。论文的 planner 是 MPC 风格：规划未来多步，只执行第一步，然后根据新 observation 重新规划。

---

## 11. 实验设计与结果

论文在 DeepMind Control Suite 的六个 pixel-based continuous control 任务上评估 PlaNet：

- Cartpole Swing Up
- Reacher Easy
- Cheetah Run
- Finger Spin
- Cup Catch
- Walker Walk

这些任务覆盖了长 horizon、partial observability、sparse reward、contact dynamics、大动作空间等难点。所有任务只使用第三人称相机图像，大小为 $$64 \times 64 \times 3$$

主要结果是：

| Method | Modality | Episodes | Cartpole | Reacher | Cheetah | Finger | Cup | Walker |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A3C | proprioceptive | 100,000 | 558 | 285 | 214 | 129 | 105 | 311 |
| D4PG | pixels | 100,000 | 862 | 967 | 524 | 985 | 980 | 968 |
| PlaNet | pixels | 1,000 | 821 | 832 | 662 | 700 | 930 | 951 |
| CEM + true simulator | simulator state | 0 | 850 | 964 | 656 | 825 | 993 | 994 |

PlaNet 用 1,000 episodes 达到接近 D4PG 用 100,000 episodes 的性能，在 Cheetah Run 上还超过 D4PG；论文估计 PlaNet 相比 D4PG 的 data efficiency gain 在不同任务上约为 40 到 500+ 倍。

实验还验证了几个设计选择：

**RSSM 结构重要。**  
Figure 4 对比了 RSSM、纯 deterministic GRU、纯 stochastic SSM。RSSM 显著更稳定，说明 stochastic + deterministic 两条路径都重要。

**Online data collection 重要。**  
Figure 5 对比了 PlaNet 和 random collection。用当前模型规划采集数据比一直随机采集更好，尤其对 cartpole、finger、walker 很关键。

**CEM iterative refinement 重要。**  
Figure 5 还对比了 CEM 和 random shooting。只从 1000 条序列里选最好的一条，不如多轮 top-K refit 的 CEM。

这篇论文的主要贡献可以总结为三点。

第一，**从像素学习 latent dynamics，并在 latent space 中做 planning**。  
这让 model-based RL 可以处理高维图像输入，同时避免在 planning 时生成未来图像。

第二，**提出并验证 RSSM 结构**。  
RSSM 同时包含 deterministic hidden state 和 stochastic latent state，前者负责长期记忆，后者负责不确定性和多未来建模。这是后续 Dreamer 系列最重要的基础结构。

第三，**提出 latent overshooting 思想**。  
它将标准 one-step variational objective 扩展到 multi-step prediction，在 latent space 中约束多步 rollout 结果，缓解 planning 时的多步误差累积问题。论文把它称为可以改善长期预测的快速 latent regularizer。

---

## 13. 局限性

PlaNet 的局限也很清楚。

**第一，inference 计算量大。**  
每个环境 step 都要 CEM 规划，采样并评估大量动作序列。相比 actor network 一次 forward 输出动作，PlaNet 的在线规划成本高。

**第二，planning horizon 有限制。**  
horizon 太短看不到长期 reward，horizon 太长动作搜索空间会爆炸。论文附录也显示 planning horizon、候选数、迭代数都会显著影响性能。

**第三，reward model 很关键。**  
planner 完全依赖 reward model 给 imagined trajectory 打分；reward model 错了，planner 会被带偏。尤其在 sparse reward 任务里，reward prediction 更难。

**第四，representation 依赖图像重建。**  
observation reconstruction 提供训练信号，但也可能让模型浪费容量重建和控制无关的视觉细节。论文 discussion 也提到，未来可以探索不依赖 reconstruction 的 representation learning。

**第五，主要实验还是模拟环境。**  
虽然是 pixel observation，但还不是复杂真实机器人/VLA 场景。

---

## 14. 和 Dreamer 的关系

PlaNet 和 Dreamer 的关系可以这样理解：

| 项目 | PlaNet | Dreamer |
|---|---|---|
| 世界模型 | RSSM | RSSM |
| 训练信号 | reconstruction + reward + KL | reconstruction/reward/discount + KL |
| 动作选择 | CEM online planning | actor network |
| 是否有 policy | 没有显式 policy network | 有 actor |
| 是否有 value | 没有 value network | 有 critic |
| imagination 用法 | 用于 CEM 评估动作序列 | 用于训练 actor-critic |
| inference 成本 | 高，每步规划 | 低，actor forward |

所以读 Dreamer 前，PlaNet 最需要掌握的是：

1. 为什么要 latent dynamics；
2. RSSM 的 deterministic/stochastic state 分工；
3. posterior 和 prior 如何通过 KL 对齐；
4. 为什么 planning 时不需要生成图像；
5. 为什么 PlaNet inference 时没有 action head，而是 CEM 搜索动作。

---

## 15. 最终理解

这篇论文的核心不是“从图像重建未来图像”，而是：

> 训练一个能在 latent space 中预测未来 reward 的世界模型，然后用这个模型在线规划动作。

一句话总结 PlaNet：

**PlaNet = encoder 推断当前 latent belief + RSSM 预测 latent dynamics + reward model 评估未来 + CEM/MPC 在线搜索连续动作。**

你读这篇时不需要死啃 latent overshooting 的完整推导，但必须吃透三件事：

1. **PlaNet 没有 policy/action head，inference 时靠 CEM planner 输出动作。**
2. **RSSM 是核心结构，后续 Dreamer 系列继续沿用。**
3. **训练时 posterior 看 observation，planning 时 prior 不看未来 observation，因此 KL 对齐和多步 rollout 稳定性非常关键。**

## 相关笔记

- [[Visual Foresight|Visual Foresight]]
- [[Dreamer技术报告|Dreamer 技术报告]]
- [[DayDreamer论文综述与阅读重点|DayDreamer]]
- [[DreamerV3_技术报告_中文|DreamerV3]]
- [[PPO|PPO]]
- [[SAC_PPO_compare|SAC vs PPO]]
- [[RDT-1B|RDT-1B]]
- [[RT-2 论文综述|RT-2 论文综述]]
- [[Pi0_7_technical_report|π0.7 技术报告]]



---
