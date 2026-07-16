---
title: ALOHA 硬件与 ACT 算法
type: paper_note
topic: robot_imitation_learning
status: mature
importance: high
updated: 2026-07-16
tags:
  - aloha
  - act
  - bimanual-manipulation
  - imitation-learning
  - action-chunking
  - cvae
  - robotics
---
# ALOHA硬件与ACT算法

**User:** Anonymous  
**Created:** 6/19/2026 23:28:42  
**Updated:** 6/21/2026 17:37:15  
**Exported:** 6/21/2026 17:37:56  

# Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware 技术报告

## 1. 论文概览

论文 **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware** 提出了一个低成本双臂精细操作系统，包含两部分核心贡献：

1. **ALOHA 硬件系统**：一个低成本、开源、双臂 leader-follower 遥操作平台，用于采集高质量人类示范数据。
2. **ACT 算法**：Action Chunking with Transformers，一种基于 Transformer 和 CVAE 的模仿学习方法，用于从少量真实示范中学习高精度双臂操作策略。

论文关注的问题是：传统精细操作任务通常依赖昂贵机械臂、高精度传感器、精细标定或复杂控制模型。作者希望证明，即使使用低成本、精度有限的硬件，只要能采集高质量示范数据，并采用合适的动作序列建模方法，机器人仍然可以完成复杂、接触丰富、需要闭环视觉反馈的双臂精细操作任务。

典型任务包括：

- 打开透明 condiment cup；
- 把电池塞进遥控器；
- 打开 ziploc bag；
- 穿魔术贴线圈；
- 准备胶带并贴到纸盒边缘；
- 给假脚穿鞋。

这些任务的共同特点是：要求毫米级精度、双臂协调、丰富接触、连续视觉闭环，以及对透明、低对比度、变形物体的感知能力。

---

## 2. ALOHA 硬件系统

### 2.1 为什么需要 ALOHA？

精细双臂操作的数据采集本身非常困难。若使用 VR controller 或手部相机追踪，人类手部位姿需要被映射到机器人末端执行器位姿，再通过 inverse kinematics 转成机器人关节角。这种方式在精细操作中容易遇到几个问题：

1. 6DoF 低成本机械臂没有冗余自由度，靠近 singularity 时 IK 容易失败；
2. task-space mapping 需要解 IK，计算和控制链路更长，延迟更大；
3. VR controller 没有真实机械结构约束，操作者容易做出 follower 机械臂难以执行的动作；
4. 手部微小抖动可能直接传递到机器人末端，影响毫米级操作稳定性。

因此，论文选择了 **joint-space leader-follower teleoperation**。

---

### 2.2 为什么有两种机械臂？

ALOHA 使用两种不同机械臂：

- **WidowX**：较小的 leader robot，由人手直接拖动；
- **ViperX**：较大的 follower robot，真正操作环境中的物体。

由于是双臂系统，实际结构是：

$$
\text{left WidowX leader} \rightarrow \text{left ViperX follower}
$$

$$
\text{right WidowX leader} \rightarrow \text{right ViperX follower}
$$

也就是说，系统中有两只小机械臂作为人的输入设备，两只大机械臂作为真实执行器。

小机械臂 WidowX 的作用类似“实体遥控器”。人通过 backdriving 它来输入动作；系统读取它的关节位置，并将其映射到 ViperX follower 上。大机械臂 ViperX 则负责真实接触、抓取、推、撬、插入等任务。

---

### 2.3 task-space mapping 和 joint-space mapping 的区别

#### Task-space mapping

task-space mapping 的流程是：

$$
\text{human hand pose}
\rightarrow
\text{robot end-effector pose}
\rightarrow
\text{IK}
\rightarrow
\text{robot joint command}
$$

也就是先得到人的手部位姿：

$$
T_{\text{hand}} = (x, y, z, R)
$$

再映射成机器人末端执行器目标位姿：

$$
T_{\text{ee}}^{\text{target}}
$$

然后通过 IK 解出目标关节角：

$$
q_{\text{target}} = \operatorname{IK}(T_{\text{ee}}^{\text{target}})
$$

问题是，在 6DoF 无冗余机械臂上，精细操作常常接近奇异位姿或关节边界，IK 容易失败或产生跳变。

#### Direct joint-space mapping

ALOHA 使用的是 direct joint-space mapping：

$$
q_{\text{follower}}^{\text{target}}
=
f(q_{\text{leader}})
$$

最简单情况下是对应关节一一映射：

$$
q_{\text{follower}, i}^{\text{target}}
=
q_{\text{leader}, i}
$$

实际系统中可能还包含 scale、offset、符号方向修正：

$$
q_{\text{follower}, i}^{\text{target}}
=
s_i q_{\text{leader}, i} + b_i
$$

这种方式的核心是：人拖动小机械臂的关节，大机械臂对应关节同步运动。它避免了末端位姿 IK，延迟更低，控制带宽更高，也更稳定。

---

### 2.4 ALOHA 数据采集的关键设计

ALOHA 的 observation 和 action 定义非常重要。

#### Observation

ACT 的 observation 包括：

$$
o_t =
(I_t^1, I_t^2, I_t^3, I_t^4, q_t^{\text{follower}})
$$

其中：

- $I_t^1, I_t^2, I_t^3, I_t^4$：四路 RGB 图像；
- $q_t^{\text{follower}}$：follower robots 当前关节位置。

四个摄像头包括：

- top camera；
- front camera；
- left wrist camera；
- right wrist camera。

数据记录和遥操作频率均为 50Hz。

#### Action

ACT 的 action 是 **leader robots 的关节位置**，不是 follower 的实际关节位置。

每个 action 是两个机械臂的目标关节位置：

$$
a_t \in \mathbb{R}^{14}
$$

因为左右两只机械臂各 7DoF：

$$
a_t =
[q_1^L, q_2^L, \dots, q_7^L,\ q_1^R, q_2^R, \dots, q_7^R]
$$

论文强调使用 leader joint positions 作为 action，而不是 follower joint positions。原因是 follower 通过底层 PID 控制器追踪 leader 的目标关节位置。leader 和 follower 之间的差值隐式包含了操作者的力控意图。如果只记录 follower 的实际关节位置，可能会丢失这部分接触控制信息。

---

## 3. ACT：Action Chunking with Transformers

> [!figure] 论文原始模型结构图
> ![[attachments/paper-figures/act-model-architecture.png]]
> ACT 的 Conditional VAE 结构：训练时用编码器得到 style variable，策略端以多视角图像、关节状态和 latent 预测 action chunk。原图来自 [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware（arXiv:2304.13705）](https://arxiv.org/abs/2304.13705)，由论文源文件高分辨率导出。

### 3.1 单步 BC 的问题

普通 behavioral cloning 学习的是单步策略：

$$
\pi_\theta(a_t \mid o_t)
$$

对于 50Hz 的机器人任务，如果一个 episode 持续 10 秒，就有：

$$
50 \times 10 = 500
$$

个决策步。

单步策略在精细操作中容易遇到 **compounding error**。前几步的小误差会导致后续状态偏离训练分布，模型在未见过的状态上继续犯错，最终任务失败。

---

### 3.2 Action Chunking 的基本思想

ACT 不预测单步动作，而是预测未来一段动作序列：

$$
\pi_\theta(a_{t:t+k} \mid o_t)
$$

其中：

$$
a_{t:t+k}
=
[a_t, a_{t+1}, \dots, a_{t+k-1}]
$$

每个动作是 14 维连续关节目标位置，因此 action chunk 是：

$$
a_{t:t+k} \in \mathbb{R}^{k \times 14}
$$

如果 $k = 100$，那么一次预测覆盖未来 100 个控制步。由于控制频率是 50Hz，这大约对应 2 秒动作。

action chunking 的意义是：把原本很长的单步决策 horizon 缩短为 chunk-level decision horizon。论文认为这可以缓解 compounding error。

单步策略：

$$
\pi_\theta(a_t \mid o_t)
$$

ACT 策略：

$$
\pi_\theta(a_{t:t+k} \mid o_t)
$$

---

### 3.3 Action 是不是离散 token？

不是。

ACT 的 action 是连续关节位置，不是离散 token。虽然论文图中会出现 “embedded action sequence”，但这里的 token 只是 Transformer 输入序列中的一个连续 embedding，不是 NLP 里的离散 token id。

每个真实动作：

$$
a_t \in \mathbb{R}^{14}
$$

进入 Transformer encoder 前，会通过 linear layer 投影到 embedding dimension：

$$
e_t = W_a a_t + b_a
$$

其中：

$$
a_t \in \mathbb{R}^{14}
$$

$$
e_t \in \mathbb{R}^{512}
$$

所以这里的 action token 更准确地说是：

> 连续 action vector 经过 linear projection 后得到的 Transformer sequence element。

ACT 最终也是直接输出连续动作：

$$
\hat{a}_{t:t+k} \in \mathbb{R}^{k \times 14}
$$

这与 BeT、RT-1 等离散化 action space 的方法不同。论文选择 continuous action prediction，是因为精细操作对毫米级精度敏感，离散化可能损失控制精度。

---

## 4. Temporal Ensemble

### 4.1 为什么需要 Temporal Ensemble？

最朴素的 action chunking 是：每 $k$ 步看一次 observation，预测一个 chunk，然后连续执行 $k$ 步。

这会带来两个问题：

1. chunk 内部接近 open-loop，不能及时利用新视觉反馈；
2. 每次切换新 chunk 时，动作可能突变，导致机器人运动不平滑。

因此，ACT 推理时不是每 $k$ 步 query 一次 policy，而是 **每个 timestep 都 query policy**。

---

### 4.2 Temporal Ensemble 的具体做法

假设 chunk size 为 4。

在 $t=0$ 时，policy 预测：

$$
[\hat{a}_0^{(0)}, \hat{a}_1^{(0)}, \hat{a}_2^{(0)}, \hat{a}_3^{(0)}]
$$

在 $t=1$ 时，policy 预测：

$$
[\hat{a}_1^{(1)}, \hat{a}_2^{(1)}, \hat{a}_3^{(1)}, \hat{a}_4^{(1)}]
$$

在 $t=2$ 时，policy 预测：

$$
[\hat{a}_2^{(2)}, \hat{a}_3^{(2)}, \hat{a}_4^{(2)}, \hat{a}_5^{(2)}]
$$

因此，对于同一个绝对时间 $t=3$，可能有多个预测：

$$
\hat{a}_3^{(0)},\quad
\hat{a}_3^{(1)},\quad
\hat{a}_3^{(2)},\quad
\hat{a}_3^{(3)}
$$

Temporal Ensemble 就是把这些“针对同一个 timestep 的动作预测”做加权平均：

$$
a_t
=
\frac{
\sum_i w_i A_t[i]
}{
\sum_i w_i
}
$$

其中：

$$
w_i = \exp(-m i)
$$

这里 $m$ 是 inference-time hyperparameter，用来控制不同 chunk 预测的权重衰减。

---

### 4.3 Temporal Ensemble 不是普通 smoothing

普通 smoothing 可能是：

$$
a_t \leftarrow \alpha a_t + (1-\alpha)a_{t-1}
$$

这种方法会把不同时间的动作混合，容易引入 lag。

ACT 的 temporal ensemble 不混合相邻时间动作，而是混合同一绝对 timestep 的多个预测：

$$
\hat{a}_t^{(t-k+1)},\quad
\hat{a}_t^{(t-k+2)},\quad
\dots,\quad
\hat{a}_t^{(t)}
$$

这些预测都表示“时间 $t$ 应该执行的动作”。因此，它更像 ensemble，而不是时间滤波。

---

### 4.4 $m$ 的含义

权重定义为：

$$
w_i = \exp(-m i)
$$

论文中 $w_0$ 是 oldest action 的权重。$m$ 控制新 observation 被纳入当前动作的速度。

直观理解：

- $m$ 较大：新预测权重衰减更快，更信任旧 chunk，动作更平滑但反应更慢；
- $m$ 较小：不同预测权重更接近，新 observation 更快影响当前动作，闭环响应更快。

论文没有在主超参表中明确列出 $m$ 的固定值，因此复现时应将其视作 inference-time hyperparameter。

---

## 5. CVAE：为什么需要 latent variable $z$？

### 5.1 人类示范的多模态问题

人类示范不是确定性的。同一个 observation 下，人可能用不同方式完成任务。

例如在胶带 handover 任务中，左右夹爪交接胶带的位置每次都可能不同，但这些轨迹都可能成功。如果直接用 L1/L2 回归，模型可能学到“平均轨迹”。平均轨迹在连续控制中往往不是合理动作，可能导致夹爪错过目标或双臂相撞。

因此，ACT 把 action chunk policy 训练成一个条件生成模型：

$$
p_\theta(a_{t:t+k} \mid o_t)
$$

通过 CVAE，引入 latent variable $z$，用于表示人类动作示范中的 style 或 mode。

---

### 5.2 CVAE 的结构

ACT 的 CVAE 包括两个部分：

1. **CVAE encoder**：

$$
q_\phi(z \mid a_{t:t+k}, \bar{o}_t)
$$

2. **CVAE decoder / policy**：

$$
\pi_\theta(\hat{a}_{t:t+k} \mid o_t, z)
$$

其中：

- $a_{t:t+k}$：真实未来 action chunk；
- $\bar{o}_t$：去掉图像后的 observation，主要是 proprioception；
- $o_t$：完整当前 observation，包括图像和关节状态；
- $z$：style variable。

训练时，encoder 看得到真实 action chunk，因此可以推断这段动作背后的 latent style。推理时，没有真实未来 action chunk，因此 encoder 被丢弃，只保留 decoder/policy。

---

## 6. Diagonal Gaussian 和 $z$ 的采样

### 6.1 什么是 diagonal Gaussian？

CVAE encoder 不直接输出一个确定的 $z$，而是输出 $z$ 的分布参数。

假设：

$$
z \in \mathbb{R}^d
$$

encoder 输出：

$$
\mu \in \mathbb{R}^d
$$

$$
\sigma \in \mathbb{R}^d
$$

然后定义：

$$
q_\phi(z \mid a_{t:t+k}, \bar{o}_t)
=
\mathcal{N}
\left(
\mu,
\operatorname{diag}(\sigma^2)
\right)
$$

其中：

$$
\operatorname{diag}(\sigma^2)
=
\begin{bmatrix}
\sigma_1^2 & 0 & 0 & \cdots \\
0 & \sigma_2^2 & 0 & \cdots \\
0 & 0 & \sigma_3^2 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$

这表示每个 latent 维度都有自己的方差，但不同维度之间没有协方差：

$$
\operatorname{Cov}(z_i, z_j)=0,\quad i \neq j
$$

因此：

$$
q_\phi(z \mid x)
=
\prod_i \mathcal{N}(z_i; \mu_i, \sigma_i^2)
$$

其中：

$$
x = (a_{t:t+k}, \bar{o}_t)
$$

---

### 6.2 $z$ 是怎么采样的？

训练时，$z$ 通过 reparameterization trick 采样：

$$
\epsilon \sim \mathcal{N}(0, I)
$$

$$
z = \mu + \sigma \odot \epsilon
$$

这里 $\odot$ 表示逐元素乘法。

代码上通常不会显式构建完整协方差矩阵，而是直接用向量形式：

```python
stats = linear(h_cls)
mu = stats[:, :z_dim]
logvar = stats[:, z_dim:]

std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)

z = mu + std * eps
```

如果真的要构建协方差矩阵，可以写成：

```python
cov = torch.diag_embed(std ** 2)
```

但训练中一般不需要这样做，因为采样和 KL loss 都可以直接用 $\mu$ 和 $\sigma$ 的向量形式完成。

---

## 7. 训练 $z$ 时是否会采错 mode？

这是理解 CVAE 的关键问题。

训练时，$z$ 不是从全局标准高斯中随便采样，而是从当前样本对应的 posterior 分布中采样：

$$
z \sim q_\phi(z \mid a_{t:t+k}, \bar{o}_t)
$$

这里的 $\mu$ 和 $\sigma$ 是根据当前这条真实 action chunk 预测出来的。

例如当前 chunk 是“从左到右执行”的动作序列，那么 encoder 看到的就是这段从左到右的真值动作，因此它应该输出适合这条轨迹的 posterior 分布。

理想情况下：

$$
q_{\text{left}\rightarrow\text{right}}(z)
=
\mathcal{N}(\mu_L, \sigma_L^2)
$$

$$
q_{\text{right}\rightarrow\text{left}}(z)
=
\mathcal{N}(\mu_R, \sigma_R^2)
$$

如果这两种动作模式差异很大，reconstruction loss 会推动它们在 latent space 中形成可区分的区域，或者推动对应 posterior 的方差变小，避免频繁采到错误区域。

---

### 7.1 采样错误会不会破坏训练？

理论上，如果 posterior 方差很大，确实可能采到与当前动作模式不匹配的 $z$。例如当前是真值“从左到右”，但采样到了更像“从右到左”的 latent 区域。

但这种情况会带来较大的 reconstruction loss：

$$
\mathcal{L}_{\text{reconst}}
=
\left\|
\hat{a}_{t:t+k}
-
a_{t:t+k}
\right\|
$$

这个 loss 会反向推动模型调整：

1. encoder 调整 $\mu$，让当前样本的 posterior 中心靠近正确 mode；
2. encoder 调整 $\sigma$，降低采到错误区域的概率；
3. decoder 学会更好地根据 $z$ 和 observation 解码正确动作；
4. 如果 observation 本身已经足够决定方向，decoder 可以减少对 $z$ 的依赖。

所以，采样带来的不一致不是无约束噪声，而是会通过 reconstruction loss 参与训练。

---

### 7.2 KL loss 的作用

训练目标是：

$$
\mathcal{L}
=
\mathcal{L}_{\text{reconst}}
+
\beta
D_{\mathrm{KL}}
\left(
q_\phi(z \mid a_{t:t+k}, \bar{o}_t)
\;\|\;
\mathcal{N}(0, I)
\right)
$$

其中 reconstruction loss 希望 $z$ 尽可能包含有用信息，能帮助 decoder 重构动作；KL loss 则希望 posterior 不要离标准高斯太远。

这二者形成 trade-off：

- 如果 $\beta$ 太小，$z$ 可以携带太多信息，posterior 可能远离 prior，测试时 $z=0$ 会失配；
- 如果 $\beta$ 太大，$z$ 被压得几乎没有信息，模型可能退化成 deterministic regression，重新遇到平均轨迹问题。

ACT 的超参表中使用：

$$
\beta = 10
$$

这说明作者希望 $z$ 有一定表达能力，但不能无限制记忆 action chunk。

---

## 8. 为什么训练时采样 $z$，推理时设 $z=0$？

训练时可以使用 encoder，因为训练数据中有真实未来 action chunk：

$$
a_{t:t+k}
$$

所以训练时：

$$
z \sim q_\phi(z \mid a_{t:t+k}, \bar{o}_t)
$$

但推理时没有真实未来动作。未来动作正是 policy 要预测的对象，因此不能再用 encoder：

$$
\text{要预测 } a_{t:t+k}
\quad \text{却需要先知道 } a_{t:t+k}
$$

所以测试时丢弃 CVAE encoder，只保留 decoder/policy，并设：

$$
z = 0
$$

原因是 prior 是标准高斯：

$$
p(z) = \mathcal{N}(0, I)
$$

它的均值是：

$$
\mathbb{E}[z] = 0
$$

因此 $z=0$ 表示选择 prior 的中心点，也就是一个 canonical / deterministic style。

推理时策略为：

$$
\hat{a}_{t:t+k}
=
\pi_\theta(o_t, z=0)
$$

这样给定同一个 observation，policy 输出是确定性的，有利于真实机器人评估的稳定性。

---

## 9. `[CLS]` token 的作用

ACT 的 CVAE encoder 使用 BERT-like Transformer encoder。输入序列包括：

$$
[\text{CLS}],\quad \text{embedded joints},\quad \text{embedded action sequence}
$$

其中 `[CLS]` 是一个可学习参数向量：

$$
e_{\text{cls}} \in \mathbb{R}^{512}
$$

它不是来自实际物理输入，也不是离散词表 token，而是随机初始化后通过训练学习的参数。

完整输入可以写成：

$$
X =
[
e_{\text{cls}},
e_{\text{joints}},
e_{\text{action},0},
e_{\text{action},1},
\dots,
e_{\text{action},k-1}
]
$$

经过 Transformer encoder 后，取 `[CLS]` 位置对应的输出：

$$
h_{\text{cls}} = H_0
$$

再通过 linear layer 预测 latent 分布参数：

$$
[\mu, \sigma] = \operatorname{Linear}(h_{\text{cls}})
$$

在论文的详细结构图中，$z$ 是 32 维，因此输出可以理解为：

$$
\mu \in \mathbb{R}^{32}
$$

$$
\sigma \in \mathbb{R}^{32}
$$

也就是总共 64 个数。

---

## 10. ACT 模型结构

### 10.1 CVAE Encoder

训练阶段的 encoder 输入包括：

- `[CLS]` token；
- 当前 joint positions；
- 真实未来 action sequence。

动作序列形状为：

$$
a_{t:t+k} \in \mathbb{R}^{k \times 14}
$$

每个 action 经过 linear layer：

$$
\mathbb{R}^{14} \rightarrow \mathbb{R}^{512}
$$

再加 position embedding，形成 action token sequence。

encoder 输出 `[CLS]` feature，用于预测：

$$
z_{\text{mean}} \in \mathbb{R}^{32}
$$

$$
z_{\text{std}} \in \mathbb{R}^{32}
$$

然后通过 reparameterization 得到 $z$。

---

### 10.2 CVAE Decoder / Policy

decoder，也就是真正的 policy，输入包括：

- 4 路 RGB 图像；
- 当前 follower joint positions；
- latent $z$。

图像先经过 ResNet18，得到 feature map，再 flatten 成视觉 token，并加 2D sinusoidal position embedding。

每路图像：

$$
480 \times 640 \times 3
\rightarrow
15 \times 20 \times 512
$$

flatten 后：

$$
15 \times 20 = 300
$$

所以每个 camera 产生：

$$
300 \times 512
$$

四个 camera 合计：

$$
1200 \times 512
$$

再加上 joint token 和 $z$ token：

$$
1202 \times 512
$$

Transformer encoder 融合多视角视觉、关节状态和 style variable。Transformer decoder 使用固定 position embedding 作为 query，通过 cross-attention 生成未来动作序列：

$$
\hat{a}_{t:t+k} \in \mathbb{R}^{k \times 14}
$$

---

## 11. ACT 训练流程

训练输入：

$$
(o_t, a_{t:t+k})
$$

其中：

$$
o_t =
(I_t^1, I_t^2, I_t^3, I_t^4, q_t^{\text{follower}})
$$

$$
a_{t:t+k}
=
[a_t, a_{t+1}, \dots, a_{t+k-1}]
$$

训练过程：

1. 从 demo dataset 中采样当前 observation 和未来 action chunk；
2. CVAE encoder 根据 $\bar{o}_t$ 和 $a_{t:t+k}$ 预测 $z$ 的 diagonal Gaussian；
3. 从该 posterior 中采样 $z$；
4. decoder/policy 根据 $o_t$ 和 $z$ 预测 action chunk；
5. 使用 reconstruction loss 和 KL loss 更新 encoder 与 decoder。

公式：

$$
q_\phi(z \mid a_{t:t+k}, \bar{o}_t)
=
\mathcal{N}
\left(
\mu_\phi,
\operatorname{diag}(\sigma_\phi^2)
\right)
$$

$$
z = \mu_\phi + \sigma_\phi \odot \epsilon,\quad
\epsilon \sim \mathcal{N}(0,I)
$$

$$
\hat{a}_{t:t+k}
=
\pi_\theta(o_t, z)
$$

$$
\mathcal{L}
=
\mathcal{L}_{\text{reconst}}
+
\beta
D_{\mathrm{KL}}
\left(
q_\phi(z \mid a_{t:t+k}, \bar{o}_t)
\;\|\;
\mathcal{N}(0,I)
\right)
$$

论文 Algorithm 1 中 reconstruction loss 写成 MSE，但正文实现部分说明实际使用 L1 loss，因为作者发现 L1 对动作序列建模更精确。

---

## 12. ACT 推理流程

推理阶段没有真实未来 action chunk，因此不能使用 CVAE encoder。流程为：

1. 输入当前 observation；
2. 设 $z=0$；
3. policy 预测未来 $k$ 步动作；
4. 每个 timestep 都重复 query policy；
5. 把多个重叠 chunk 对当前 timestep 的预测做 temporal ensemble；
6. 将最终动作发送给底层 PID 控制器执行。

公式：

$$
\hat{a}_{t:t+k}
=
\pi_\theta(o_t, z=0)
$$

$$
a_t
=
\frac{
\sum_i w_i A_t[i]
}{
\sum_i w_i
},
\quad
w_i = \exp(-m i)
$$

---

## 13. 实验结果与消融

### 13.1 真实任务结果

ACT 在多个真实任务上明显优于 baseline。代表性结果包括：

- Slide Ziploc：最终成功率 88%；
- Slot Battery：最终成功率 96%；
- Open Cup：最终成功率 84%；
- Thread Velcro：最终成功率 20%；
- Prep Tape：最终成功率 64%；
- Put On Shoe：最终成功率 92%。

Thread Velcro 最难，主要失败原因是视觉定位困难。黑色魔术贴与背景对比度低，且目标在图像中占比很小，导致右臂容易提前闭合或插入时错过小环。

---

### 13.2 Action Chunking 消融

论文测试了不同 chunk size $k$。

当：

$$
k=1
$$

相当于没有 action chunking，模型退化为单步策略。

当：

$$
k = \text{episode length}
$$

相当于完全 open-loop，根据第一帧 observation 输出整段动作。

消融结果显示，随着 $k$ 从 1 增大到 100，ACT 成功率显著提升；但继续增大到 200 或 400 后略有下降。这说明：

- 适当增大 $k$ 可以降低 effective horizon，缓解 compounding error；
- $k$ 太大时，策略过于 open-loop，缺少反应性，且长序列更难建模。

ACT 表中使用的主要 chunk size 是：

$$
k = 100
$$

---

### 13.3 Temporal Ensemble 消融

Temporal Ensemble 对 ACT 和 BC-ConvMLP 有正向提升。论文报告：

- ACT：加入 temporal ensemble 后约 +3.3%；
- BC-ConvMLP：约 +4%；
- VINN：性能下降。

作者认为 temporal ensemble 更适合参数化模型，因为它可以平滑模型预测误差；而 VINN 检索的是真实示范动作，本身不太受模型预测噪声影响，因此 temporal ensemble 反而可能破坏其动作一致性。

---

### 13.4 CVAE 消融

论文比较了带 CVAE 和不带 CVAE 的 ACT。

结果显示：

- scripted data 上，带不带 CVAE 差异很小；
- human data 上，with CVAE 明显优于 no CVAE。

这说明 CVAE 主要解决的是人类示范中的 stochasticity 和 multi-modal behavior，而不是单纯提高模型容量。

直观理解是：scripted data 的动作模式比较确定，普通回归就能学；human demonstration 中同一状态下可能存在多种合理动作轨迹，CVAE latent $z$ 可以在训练时帮助模型解释这些变化。

---

## 14. 技术主线总结

这篇论文的技术主线可以总结为：

$$
\text{low-cost leader-follower teleoperation}
\rightarrow
\text{high-quality human demonstrations}
\rightarrow
\text{continuous action chunks}
\rightarrow
\text{CVAE latent style modeling}
\rightarrow
\text{temporal ensemble closed-loop execution}
$$

其中：

- ALOHA 解决高质量示范数据采集问题；
- joint-space mapping 解决低延迟、高带宽遥操作问题；
- action chunking 解决长 horizon imitation learning 的 compounding error；
- CVAE 解决人类示范多模态和随机性问题；
- temporal ensemble 解决 chunk execution 的动作跳变和平滑闭环问题；
- continuous action prediction 保留精细操作所需的控制精度。

---

## 15. 我的理解与评价

这篇论文的核心价值不在于简单地“用了 Transformer”，而在于它重新设计了机器人模仿学习中的三个接口。

### 15.1 数据接口

ALOHA 用真实机械臂作为 leader，而不是用 VR controller。这让人类输入更符合 follower 的可执行动作空间，也避免了 IK 和 task-space retargeting 的不稳定性。

### 15.2 动作接口

ACT 不输出离散 action token，也不只输出单步动作，而是直接输出连续 action chunk：

$$
\hat{a}_{t:t+k} \in \mathbb{R}^{k \times 14}
$$

这对精细操作非常重要，因为离散化可能损失毫米级控制精度。

### 15.3 时序接口

ACT 把高频控制问题从单步决策改成 chunk-level sequence prediction，降低 effective horizon，同时又通过 temporal ensemble 保留每帧视觉闭环。

### 15.4 生成式建模接口

CVAE 并不是为了测试时随机采样多种动作，而是主要在训练时帮助模型处理人类 demonstration 的多样性。训练时用真实 action chunk 推断 $z$，推理时设 $z=0$，得到稳定的 deterministic policy。

---

## 16. 与 Diffusion Policy / VLA 方法的关系

ACT 和 Diffusion Policy、VLA 模型有相似点，也有明显区别。

### 16.1 与 Diffusion Policy 的相似点

二者都不满足于单步 action prediction，而是建模未来动作序列：

$$
a_{t:t+k}
$$

这说明在机器人控制中，action sequence representation 是非常重要的。相比单步 action，动作序列更能表达短时程协调、平滑性和非马尔可夫行为。

### 16.2 与 Diffusion Policy 的区别

ACT 使用 CVAE + Transformer decoder 一次生成动作序列；Diffusion Policy 通常通过扩散去噪过程生成动作序列。ACT 推理更简单、更快，但生成分布表达能力可能不如 diffusion。

### 16.3 与 VLA 的区别

ACT 不是语言条件策略，也不是大规模预训练 VLA。它更接近一个单任务、小数据、高质量遥操作数据驱动的 visuomotor policy。

但是 ACT 对 VLA 和机器人基础模型仍然有重要启发：

- action 不一定要离散 token；
- action chunk 比 single-step action 更适合高频控制；
- 训练时的动作分布建模和推理时的闭环执行同样重要；
- 数据采集系统本身会强烈影响 policy 上限。

---

## 17. 最终结论

ALOHA + ACT 证明了一个重要观点：

> 在机器人精细操作中，硬件不一定必须非常昂贵，模型也不一定必须非常巨大。高质量遥操作数据、合理的 action representation、稳定的闭环推理机制，往往比单纯扩大 backbone 更关键。

ACT 的核心形式可以写成：

$$
\pi_\theta(a_{t:t+k} \mid o_t, z)
$$

其中：

- $a_{t:t+k}$：连续动作序列；
- $o_t$：当前多视角图像和关节状态；
- $z$：训练时从 action chunk 中推断的 style variable；
- 推理时 $z=0$，得到确定性策略；
- temporal ensemble 将多个重叠 chunk 对同一时刻的动作预测进行融合。

因此，这篇论文最值得关注的不是某一个单独模块，而是它完整打通了：

$$
\text{teleoperation data}
+
\text{continuous action chunk}
+
\text{generative latent modeling}
+
\text{closed-loop temporal ensemble}
$$

这条技术链路对后续机器人基础模型、VLA、Diffusion Policy 以及低成本机器人数据采集系统都有很强参考价值。

## 相关笔记

- [[RDT-1B|RDT-1B]]：把 diffusion action chunk 扩展到双臂 foundation policy。
- [[Diffusion Policy 概述|Diffusion Policy]]：action chunk diffusion 与 receding-horizon control 的基础路线。
- [[FAST_知识总结|FAST]]：另一条 action chunk 表示路线，离散 tokenization 而不是连续动作生成。
- [[Pi_0机器人文章分析|pi0]]：VLA + continuous action expert 的后续路线。
- [[Pi0_7_technical_report|pi0.7]]：更大规模的双臂/通用机器人行为建模。



---
