---
title: DayDreamer 论文综述与阅读重点
type: paper_note
topic: model_based_reinforcement_learning
status: mature
importance: high
updated: 2026-07-14
tags:
  - daydreamer
  - dreamer-v2
  - world-model
  - model-based-rl
  - online-rl
  - robotics
---

# DayDreamer：真实机器人在线世界模型学习技术报告

## 1. 论文定位

**DayDreamer: World Models for Physical Robot Learning** 研究的问题是：

> Dreamer 这种依赖 latent imagination 的世界模型算法，能否不使用模拟器、专家示范和离线预训练，直接在真实机器人上在线学习？

论文没有提出新的世界模型算法，而是将 **DreamerV2** 部署到四类真实机器人任务：

| 平台 | 任务 | 观测 | 动作 |
|---|---|---|---|
| Unitree A1 | 翻身、站立、行走 | 关节角、姿态、角速度 | 连续目标关节角 |
| UR5 | 多物体抓取放置 | RGB + proprioception | 离散末端动作 |
| XArm | 软物体抓取放置 | RGB-D + proprioception | 离散末端动作 |
| Sphero | 视觉导航 | 纯 RGB | 连续电机力矩 |

论文的核心价值不是算法创新，而是验证：

> Dreamer 式 latent world model 可以在真实机器人上持续学习，并通过想象轨迹显著提高真实数据的利用率。`daydreamer.pdf`

---

## 2. 主要实验结果

| 任务 | 训练时间 | 结果 |
|---|---:|---|
| A1 四足行走 | 约 1 小时 | 学会翻身、站立和前进 |
| A1 抗扰动 | 额外约 10 分钟 | 学会抵抗推力或摔倒后恢复 |
| UR5 抓取放置 | 约 8 小时 | 约 2.5 objects/min |
| XArm 抓取放置 | 约 10 小时 | 约 3.1 objects/min |
| Sphero 导航 | 约 2 小时 | 稳定到达固定目标 |

相同真实数据预算下：

- SAC 在 A1 上只学会翻身，未学会站立和行走；
- Rainbow、PPO 在机械臂任务上容易停留在“抓起后立即放下”的局部最优；
- Sphero 上 Dreamer 与专门面向视觉连续控制的 DrQv2 表现接近。`daydreamer.pdf`

---

## 3. 整体训练 Pipeline

DayDreamer 是一个持续在线运行的循环：

```text
真实机器人执行当前 Actor
          ↓
收集 (observation, action, reward)
          ↓
写入 Replay Buffer
          ↓
训练 RSSM World Model
          ↓
在 latent space 中展开 imagined rollouts
          ↓
训练 Actor 和 Critic
          ↓
更新后的 Actor 继续控制机器人
```

形式上：

$$
\text{Real Experience}
\rightarrow
\text{World Model}
\rightarrow
\text{Imagination}
\rightarrow
\text{Policy Improvement}
$$

世界模型把有限的真实经验转化为大量虚拟经验，因此机器人不需要亲自执行所有用于策略训练的轨迹。

---

## 4. 异步 Actor-Learner 架构

真实机器人控制具有严格延迟要求，DayDreamer 将控制和训练拆成两个并行线程。

### 4.1 Actor Thread

负责：

1. 接收当前传感器观测；
2. 更新当前 latent state；
3. 通过 actor 产生动作；
4. 控制机器人；
5. 将 transition 写入 replay buffer。

### 4.2 Learner Thread

持续执行：

- world model 更新；
- imagined rollout；
- actor 更新；
- critic 更新。

因此，机器人运动时，GPU 可以同时训练模型。

论文没有固定“每采集一步训练多少次”的频率，learner 会在算力允许时持续训练。这个设计对于 A1 的 20 Hz 控制尤其重要。`daydreamer.pdf`

---

## 5. RSSM World Model

DayDreamer 沿用 DreamerV2 的 Recurrent State-Space Model。

完整 latent state 写成：

$$
s_t=(h_t,z_t)
$$

其中：

- $h_t$：确定性的 recurrent state；
- $z_t$：随机、离散的 latent state。

这两个变量不是简单重复，而是承担不同职责。

---

## 6. $h_t$ 与 $z_t$ 的核心分工

### 6.1 $h_t$：根据历史预测当前状态

$$
h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1})
$$

$h_t$ 不读取当前观测 $x_t$，只依据：

- 历史 recurrent state；
- 上一时刻 latent state；
- 上一时刻动作。

它表达的是：

> 根据过去以及刚执行的动作，模型预测当前大概处于什么状态。

例如 A1 执行翻身动作后，$h_t$ 可能包含：

- 机器人可能已经翻正；
- 也可能仍然侧躺；
- 可能脚底发生了滑动。

因此 $h_t$ 更接近动力学模型的 **prediction state**。

---

### 6.2 $z_t$：根据当前观测形成后验状态编码

真实交互时，机器人能获得当前观测：

$$
x_t
$$

encoder 产生 posterior：

$$
q_\theta(z_t\mid h_t,x_t)
$$

随后采样：

$$
z_t\sim q_\theta(z_t\mid h_t,x_t)
$$

更准确的理解是：

> $z_t$ 是模型结合当前观测后，对当前隐状态形成的后验编码，而不是“真实物理状态本身”。

例如：

```text
h_t：根据刚才的动作，预测 A1 可能已经翻正
x_t：IMU 表明机器人实际上仍然侧躺
z_t：编码“当前仍然侧躺”这一后验判断
```

---

## 7. 为什么不直接用观测更新 $h_t$

完全可以设计一个普通 RNN：

$$
h_t=f(h_{t-1},a_{t-1},\operatorname{enc}(x_t))
$$

但 RSSM 刻意将：

$$
\text{动力学预测}
$$

和：

$$
\text{观测修正}
$$

拆开。

真实交互时：

$$
h_t=f(h_{t-1},z_{t-1},a_{t-1})
$$

$$
z_t\sim q(z_t\mid h_t,x_t)
$$

想象未来时：

$$
h_t=f(h_{t-1},z_{t-1},a_{t-1})
$$

$$
z_t\sim p(z_t\mid h_t)
$$

两种场景使用相同的 recurrent dynamics，只是 $z_t$ 的来源不同：

| 场景 | $z_t$ 来源 |
|---|---|
| 有真实观测 | posterior $q(z_t\mid h_t,x_t)$ |
| 无真实观测 | prior $p(z_t\mid h_t)$ |

如果让 $h_t$ 必须读取 $x_t$，那么 imagination 阶段没有未来观测，就无法继续使用同一个 transition。

因此这种拆分让模型同时支持：

- 真实数据上的 filtering；
- 无观测条件下的 latent imagination。

---

### 7.1 当前观测没有控制延迟

虽然 $x_t$ 没有直接进入 $h_t$，但当前动作使用完整状态：

$$
a_t\sim\pi(a_t\mid h_t,z_t)
$$

所以 $x_t$ 通过 $z_t$ 立即影响当前动作。

完整时间线是：

```text
历史 + 上一动作
      ↓
预测 h_t
      ↓
读取当前观测 x_t
      ↓
posterior z_t
      ↓
Actor(h_t, z_t) 输出当前动作 a_t
      ↓
h_{t+1}=f(h_t,z_t,a_t)
```

并不存在“观测要等到下一帧才生效”的问题。

---

## 8. 没有观测时，$z_t$ 为什么有用

在 imagination 中：

$$
z_t\sim p(z_t\mid h_t)
$$

此时 $z_t$ 不会凭空增加新的真实信息。它的作用是：

> 表达同一个历史条件下可能出现的多种未来，并采样其中一个具体分支。

假设机械臂闭合夹爪后：

$$
p(z_{t+1}\mid h_{t+1})
=
\begin{cases}
0.7 & \text{成功抓住物体}\\
0.3 & \text{物体滑落}
\end{cases}
$$

不同 imagined rollout 可以采样出不同结果：

```text
成功分支：
抓住物体 → 抬起 → 搬运 → 获得奖励

失败分支：
物体滑落 → 夹爪空着 → 需要重新接近
```

关键在于，采样出的 $z_t$ 会进入下一步 recurrent transition：

$$
h_{t+1}=f(h_t,z_t,a_t)
$$

所以随机结果能够持续影响整条未来轨迹，而不是只让某一帧图像随机变化。

---

## 9. 为什么仅使用确定性 $h_t$ 可能不够

如果只有：

$$
h_{t+1}=f(h_t,a_t)
$$

同一个输入只能得到一个确定性未来。

但是实际机器人动力学可能具有多模态性：

- 脚底可能打滑，也可能不打滑；
- 夹爪可能抓住，也可能没抓住；
- 相同动作可能因接触差异产生不同结果；
- 外部扰动和传感器噪声无法完全预测。

确定性模型通常只能输出一个“平均未来”。

但一些结果不能合理地平均，例如：

- 抓住；
- 没抓住。

“半抓住”可能是一个真实世界中不存在的状态。

随机 latent 允许模型表示条件分布：

$$
p(z_{t+1}\mid h_{t+1})
$$

而不是单个确定性点。

---

### 9.1 $z_t$ 并非一定优于单独的 $h_t$

如果环境：

- 基本确定；
- 状态完全可观测；
- 动作结果高度稳定；

那么单独的确定性 state 也可能足够。

随机 latent 不是唯一方案，还可以使用：

- dynamics ensemble；
- mixture model；
- diffusion dynamics；
- 随机 recurrent state；
- 显式 uncertainty model。

RSSM 采用 $h_t+z_t$，是在以下两种能力之间折中：

$$
\boxed{h_t:\text{稳定的长期记忆与确定性上下文}}
$$

$$
\boxed{z_t:\text{观测修正与多种未来分支}}
$$

---

## 10. Prior 与 Posterior 如何学习对齐

真实数据阶段：

$$
q_\theta(z_t\mid h_t,x_t)
$$

看到了观测，知道真实轨迹对应哪个 latent 状态。

Prior：

$$
p_\theta(z_t\mid h_t)
$$

只能根据历史进行预测。

训练通过 KL divergence 对齐：

$$
D_{\mathrm{KL}}
\left[
q_\theta(z_t\mid h_t,x_t)
\parallel
p_\theta(z_t\mid h_t)
\right]
$$

可以理解为：

```text
Posterior：
看到真实观测，判断这一次实际落在哪个 latent 分支

Prior：
根据历史，预测各个分支出现的概率

KL：
教 Prior 模仿真实数据中 Posterior 的分布
```

这样 imagination 时，即使没有未来观测，prior 仍然能生成从真实数据中学到的合理未来。

---

## 11. 随机离散 latent 的历史来源

这一设计不是 DayDreamer 提出的。

发展关系是：

```text
PlaNet / RSSM
  └─ deterministic h_t
  └─ continuous stochastic z_t

Dreamer
  └─ 继续使用连续 stochastic latent
  └─ 用 latent imagination 训练 actor

DreamerV2
  └─ 将 stochastic latent 改成离散 categorical variables

DayDreamer
  └─ 直接采用 DreamerV2
  └─ 将其部署到真实机器人
```

所以：

$$
\boxed{
\text{随机 latent 来自早期 RSSM，离散 latent 来自 DreamerV2。}
}
$$

DayDreamer 本身没有为 $z_t$ 提出新的结构。`daydreamer.pdf`

---

## 12. 离散 $z_t$ 的形式

论文配置为：

- 32 个 latent variables；
- 每个变量有 32 个 categorical classes。

即：

$$
z_t=(z_t^1,\ldots,z_t^{32})
$$

$$
z_t^i\in\{1,\ldots,32\}
$$

每个变量由一个 32 类 categorical distribution 产生。

这些类别没有人工规定的物理含义，不是：

```text
第一个 latent = 朝向
第二个 latent = 接触状态
```

而是模型为了：

- 重建观测；
- 预测未来；
- 预测奖励；

自动学习出的分布式编码。

---

## 13. World Model 的其他组成部分

### 13.1 Decoder

$$
\hat x_t\sim p_\theta(x_t\mid h_t,z_t)
$$

用于重建：

- RGB；
- depth；
- proprioception。

作用包括：

- 为 representation learning 提供密集监督；
- 让 latent 保存足够的观测信息；
- 允许人类解码 imagined trajectory 进行检查。

Actor 训练时不需要实际重建图像。

---

### 13.2 Reward Model

$$
\hat r_t=\operatorname{rew}_\theta(h_t,z_t)
$$

Imagination 中没有真实环境奖励，因此必须由 reward model 预测。

---

### 13.3 World Model Loss

整体可以概括为：

$$
\mathcal L_{\text{WM}}
=
\mathcal L_{\text{observation}}
+
\mathcal L_{\text{reward}}
+
\beta\mathcal L_{\text{KL}}
$$

其中 KL 用于连接 posterior 与 prior。

---

## 14. Latent Imagination

从 replay buffer 中的真实 posterior state 出发：

$$
a_t\sim\pi(a_t\mid h_t,z_t)
$$

$$
h_{t+1}=f(h_t,z_t,a_t)
$$

$$
z_{t+1}\sim p(z_{t+1}\mid h_{t+1})
$$

$$
\hat r_t=\operatorname{rew}(h_{t+1},z_{t+1})
$$

重复展开约 $H=15$ 步。

由于整个过程在低维 latent space 中完成，不需要生成高分辨率图像，因此可以在 GPU 上并行产生大量 imagined trajectories。

---

## 15. Actor-Critic 学习

Actor：

$$
\pi_\phi(a_t\mid s_t)
$$

Critic：

$$
v_\psi(s_t)
$$

Critic 学习长期累计回报，Actor 学习最大化 imagined return。

---

### 15.1 $\lambda$-Return

$$
V_t^\lambda
=
\hat r_t+
\gamma
\left[
(1-\lambda)\bar v(s_{t+1})
+
\lambda V_{t+1}^\lambda
\right]
$$

$$
V_H^\lambda=\bar v(s_H)
$$

论文使用：

$$
\gamma=0.95,\qquad
\lambda=0.95
$$

其中 $\bar v$ 是 slowly updated target critic。

---

### 15.2 为什么使用 Target Critic

$\lambda$-return 包含 critic 自己对未来价值的 bootstrap 估计。

如果直接使用正在快速更新的在线 critic：

```text
critic 更新
   ↓
λ-return 标签变化
   ↓
critic 追赶新标签
   ↓
标签再次变化
```

会形成 moving target。

因此使用缓慢更新的 target critic：

- 在线 critic 负责学习；
- target critic 负责生成相对稳定的监督目标。

论文给出的 target update interval 是 100。`daydreamer.pdf`

---

### 15.3 连续动作

A1、Sphero 使用连续动作，通过 reparameterization gradient 训练 Actor。

梯度可以经过：

$$
\text{return}
\rightarrow
\text{reward model}
\rightarrow
\text{dynamics}
\rightarrow
\text{action}
\rightarrow
\text{actor}
$$

但 Actor loss 不更新 world model 参数。

---

### 15.4 离散动作

UR5、XArm 使用离散动作，采用 REINFORCE：

$$
\nabla_\phi
\log\pi_\phi(a_t\mid s_t)
\left[
V_t^\lambda-v(s_t)
\right]
$$

Critic 在这里作为 baseline 降低梯度方差。

---

## 16. A1 Quadruped Walking

### 16.1 控制设置

- 12 个连续目标关节角；
- 底层 PD controller 执行；
- 控制频率 20 Hz；
- Butterworth filter 过滤高频动作。

因此 DayDreamer 并不是直接输出 motor torque，而是在已有底层控制接口之上学习高层目标关节位置。

---

### 16.2 奖励结构

总奖励：

$$
r=
r_{\text{upr}}
+r_{\text{hip}}
+r_{\text{shoulder}}
+r_{\text{knee}}
+r_{\text{velocity}}
$$

最大值为：

$$
14
$$

#### 身体朝上

$$
r_{\text{upr}}
=
\frac{\hat z^\top[0,0,1]+1}{2}
$$

- 四脚朝天：约 0；
- 侧躺：约 0.5；
- 正常朝上：约 1。

#### 关节姿态

鼓励四条腿接近站立姿态。公开代码中的归一化 standing pose 为：

$$
[0,-0.2,1.0]\times4
$$

#### 前进速度

先计算沿机身 heading 的前进速度，再惩罚横向滑动和速度不足。

最终代码形式为：

$$
r_{\text{total}}
=
r_{\text{upr}}
+r_{\text{hip}}
+r_{\text{shoulder}}
+r_{\text{knee}}
+
10\frac{r_{\text{vel}}+1}{2}
$$

---

### 16.3 Reward Gating

奖励逐级解锁：

```text
身体翻正
  ↓
髋关节接近站姿
  ↓
上腿关节接近站姿
  ↓
膝关节接近站姿
  ↓
允许获得前进奖励
```

对应代码中：

$$
r_{\text{hip}}>0
\quad\text{only if}\quad
r_{\text{upr}}>0.7
$$

$$
r_{\text{shoulder}}>0
\quad\text{only if}\quad
r_{\text{hip}}>0.7
$$

$$
r_{\text{knee}}>0
\quad\text{only if}\quad
r_{\text{shoulder}}>0.7
$$

$$
r_{\text{vel}}>0
\quad\text{only if}\quad
r_{\text{knee}}>0.7
$$

这是一种很强的人工 reward curriculum。

---

## 17. 机械臂与导航实验

### 17.1 UR5

奖励：

- 抓住：$+1$
- 原 bin 放下：$-1$
- 另一侧 bin 放下：$+10$

Dreamer 能够利用 world model 和 critic 发现长期回报，而 model-free baseline 容易停留在短视的抓取-放下行为。

### 17.2 XArm

输入包含 RGB、depth 和 proprioception，需要 world model 进行多模态融合。

软物体难以用刚体模拟器准确建模，直接在真实世界训练避免了 sim-to-real dynamics gap。

### 17.3 Sphero

单张图像无法判断对称机器人的朝向，需要 recurrent state $h_t$ 利用历史图像和动作形成 belief state。

这个实验直接体现了 $h_t$ 的时序记忆作用。

---

## 18. 在线适应

论文展示了：

- A1 在外力扰动后继续训练约 10 分钟，学会更稳定地恢复；
- XArm 遇到日出光照变化后，继续训练数小时恢复性能。

这里的 adaptation 是：

$$
\text{收集新数据}
\rightarrow
\text{继续更新参数}
$$

而不是：

- zero-shot adaptation；
- in-context adaptation；
- 不更新网络的快速适应。

---

## 19. World Model 不要求像素级完美

附录中的 imagined images 存在明显错误，例如物体颜色发生变化。

这说明策略成功不要求 world model 完美预测所有像素。

它只需要保留决策相关信息：

- 物体大致位置；
- 是否抓住；
- 机器人是否站立；
- 动作是否会增加奖励。

因此：

$$
\text{高保真视频预测}
\neq
\text{好的控制模型}
$$

真正关键的是：

$$
p(\text{task-relevant state},r\mid s_t,a_t)
$$

是否足够准确。

---

## 20. 样本效率来源

DayDreamer 的样本效率来自两层复用。

第一层：

$$
\text{真实轨迹}
\rightarrow
\text{在 replay buffer 中反复训练 world model}
$$

第二层：

$$
\text{一个真实 latent state}
\rightarrow
\text{大量 imagined trajectories}
$$

因此少量真实经验可以支持大量 actor-critic 更新。

---

## 21. 局限性

### 21.1 强任务工程

虽然不使用 simulator 或 demonstration，但依赖：

- 手工奖励；
- 预定义 standing pose；
- PD controller；
- 动作滤波；
- 离散化动作空间；
- 自动夹爪规则；
- XArm 绳子辅助。

### 21.2 没有安全探索保证

真实 RL 会导致：

- 硬件磨损；
- 碰撞；
- 电机过热；
- 人工维护。

论文主要通过控制接口和滤波降低风险，而不是从算法上解决安全性。

### 21.3 没有多任务与跨机器人泛化

四个任务分别训练独立模型，不包含：

- language conditioning；
- multi-task pretraining；
- cross-embodiment transfer；
- zero-shot generalization。

### 21.4 实验统计规模有限

部分真实机器人结果主要来自单次完整训练过程，更接近强有力的可行性验证，而非稳定的工业级统计结论。

---

## 22. 与 PlaNet、Dreamer、VLA 的关系

```text
PlaNet
  └─ latent world model
  └─ inference 时使用 CEM 搜索动作

Dreamer
  └─ 在 imagination 中训练 Actor
  └─ inference 时直接运行 Actor

DreamerV2
  └─ 引入离散 stochastic latent

DayDreamer
  └─ 将 DreamerV2 部署到真实机器人
  └─ 使用异步 Actor-Learner
```

与现代 VLA 相比：

| DayDreamer | VLA |
|---|---|
| 单任务在线 RL | 大规模多任务模仿学习 |
| reward 驱动 | demonstration 驱动 |
| 学习环境动力学 | 直接生成动作 |
| 每个任务单独训练 | 追求跨任务泛化 |
| 无语言输入 | language-conditioned |

更有潜力的未来组合可能是：

```text
大规模 VLA 离线预训练
        +
真实机器人 World Model
        +
少量在线 RL / Adaptation
```

---

## 23. 最终总结

DayDreamer 证明了：

> Dreamer 式 latent world model 可以直接从真实机器人交互中学习，并通过 latent imagination 在小时级真实数据预算内完成 locomotion、manipulation 和 navigation。

其中最关键的状态设计是：

$$
s_t=(h_t,z_t)
$$

$$
h_t=
\text{由历史和动作形成的确定性预测与长期记忆}
$$

$$
q(z_t\mid h_t,x_t)=
\text{看到当前观测后形成的后验状态编码}
$$

$$
p(z_t\mid h_t)=
\text{没有观测时对可能 latent 状态的概率预测}
$$

$z_t$ 的价值不是在无观测时提供额外事实，而是：

> 让同一个历史条件可以展开出多个可能且时序一致的未来分支。

因此，RSSM 的核心并非单纯“把观测压缩成 latent”，而是建立了一个统一的状态估计与想象机制：

$$
\boxed{
\text{真实交互时用 posterior 修正状态，想象未来时用 prior 生成可能状态。}
}
$$

这正是 DayDreamer 能够同时处理真实传感器融合、部分可观测性和 latent planning-by-imagination 的基础。


## 24. 相关笔记

- [[Visual Foresight|Visual Foresight]]：早期像素空间视频预测与 Visual MPC 路线。
- [[PlaNet 论文概述|PlaNet]]：RSSM 与 latent planning 的直接前序工作。
- [[Dreamer技术报告|Dreamer]]：DayDreamer 部署到真实机器人的算法基础。
- [[DreamerV3_技术报告_中文|DreamerV3]]：Dreamer 系列面向跨领域稳定性的后续发展。
- [[PPO|PPO]]：论文中 model-free policy-gradient baseline 的概念背景。
- [[SAC_PPO_compare|SAC vs PPO]]：理解在线数据、replay buffer 与策略更新方式的差异。
