---
title: Diffusion Policy 概述
type: paper_note
topic: robot_policy_learning
status: mature
importance: high
updated: 2026-06-28
tags:
  - diffusion-policy
  - imitation-learning
  - action-chunking
  - receding-horizon-control
  - conditional-diffusion
  - robotics
---
# Diffusion Policy 技术报告

## 1. 论文定位

**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** 的核心定位是：把机器人模仿学习里的低层动作生成问题，改写成一个 **conditional diffusion action generation** 问题。

它不是 VLA，不是 world model，也不是高层任务规划器。它主要解决的是：

$$
O_t \rightarrow A_t
$$

其中 $O_t$ 是当前/历史观测，$A_t$ 是未来一段连续动作序列。更准确地说，模型学习的是：

$$
p(A_t \mid O_t)
$$

也就是：**在给定当前观测的条件下，生成一段未来 action chunk。**

论文把这种策略称为 Diffusion Policy：机器人策略不是直接回归动作，而是从高斯噪声开始，经过多步 denoising，逐步生成动作序列。作者在 15 个仿真和真实任务上验证，报告平均性能提升 46.9%。`Diffusion_policy.pdf`

---

## 2. 它要解决什么问题？

传统 behavior cloning 通常直接学：

$$
a_t = f_\theta(o_t)
$$

但机器人动作生成有几个典型困难。

第一，**动作是多模态的**。同一个状态下，机器人可以从左边绕过去，也可以从右边绕过去。普通 MSE regression 会把两个模式平均，得到一个无效动作。

第二，**单步动作预测缺少时间一致性**。如果每一步都独立预测，模型可能这一帧选择左边模式，下一帧又选择右边模式，导致动作抖动。

第三，**高维连续动作序列难建模**。如果一次预测多个未来动作，输出维度会变高。GMM、离散化、IBC/EBM 等方法都各有问题，比如模式数难设、离散维度爆炸、训练不稳定。

Diffusion Policy 的思路是：不直接预测单步动作，而是生成一段动作序列：

$$
A_t = [a_t, a_{t+1}, \dots, a_{t+T_p-1}]
$$

并且用 diffusion 模型表达这个动作序列的条件分布。论文认为这带来三个优势：能表达多模态动作分布、适合高维动作输出、训练比 EBM/IBC 稳定。`Diffusion_policy.pdf`

---

## 3. 整体模型 Pipeline

下面是完整流程，分成 **数据构造 / 训练 / 推理执行** 三部分。

---

## 3.1 数据构造 Pipeline

原始数据来自专家 demonstration。每条数据包含：

$$
(o_0, a_0), (o_1, a_1), \dots, (o_T, a_T)
$$

其中 $o_t$ 可以是图像、机器人 proprioception、末端位姿等；$a_t$ 是连续控制命令，通常是末端 position command 或 velocity command。

训练时，从 demonstration 里随机截取一个时间点 $t$，构造：

### 观测输入

$$
O_t = [o_{t-T_o+1}, \dots, o_t]
$$

这里 $T_o$ 是 observation horizon，也就是看最近几帧观测。

### 动作监督目标

$$
A_t^0 = [a_t, a_{t+1}, \dots, a_{t+T_p-1}]
$$

这里 $T_p$ 是 prediction horizon，也就是模型要预测未来多少步动作。

这个 $A_t^0$ 就是 diffusion 训练里的 clean sample，也就是论文里说的 unmodified example $x^0$。在图像 diffusion 里 $x^0$ 是干净图片；在 Diffusion Policy 里，$x^0$ 就是专家动作序列。论文第 3 页明确写到，训练时随机抽取未修改样本 $x^0$，随机选择 denoising iteration $k$，加噪后让网络预测噪声。`Diffusion_policy.pdf`

---

## 3.2 视觉/状态编码 Pipeline

如果输入是视觉观测，模型会先把图像序列编码成 observation embedding。

论文使用的视觉编码器主要是 ResNet-18，并做了两个修改：global average pooling 换成 spatial softmax，以保留空间信息；BatchNorm 换成 GroupNorm，以提高训练稳定性。不同 camera view 使用独立 encoder，每个时间步的图像独立编码，最后拼接成 $O_t$ 的 latent representation。`Diffusion_policy.pdf`

所以视觉部分可以理解为：

$$
\text{image sequence} \rightarrow \text{ResNet / Spatial Softmax / GroupNorm} \rightarrow \text{observation embedding}
$$

如果是 state-based setting，则可以直接用 MLP 把状态输入编码成 observation embedding。

重要的是：**视觉观测不是 diffusion 的生成对象，而只是 condition。**

也就是说，模型不是生成：

$$
p(A_t, O_t)
$$

而是生成：

$$
p(A_t \mid O_t)
$$

这样做的好处是不用预测未来图像或未来状态，只需要生成动作，推理更快，也更适合实时控制。论文明确说，把 observation 从 denoising 输出中排除，可以显著提高推理速度，并支持端到端训练视觉编码器。`Diffusion_policy.pdf`

---

## 3.3 训练 Pipeline

训练目标是学习一个噪声预测网络：

$$
\epsilon_\theta(O_t, A_t^k, k)
$$

其中：

- $O_t$：当前历史观测；
- $A_t^k$：加噪后的动作序列；
- $k$：diffusion denoising step；
- $\epsilon_\theta$：预测加到动作序列上的噪声。

具体训练流程如下。

### Step 1：从数据集中抽 clean action chunk

从专家 demonstration 里截取真实动作序列：

$$
A_t^0 = [a_t, a_{t+1}, \dots, a_{t+T_p-1}]
$$

### Step 2：随机选择 diffusion step

随机采样一个 denoising step：

$$
k \sim \{1, 2, \dots, K\}
$$

### Step 3：给真实动作序列加噪声

采样噪声：

$$
\epsilon^k \sim \mathcal{N}(0, \sigma_k^2 I)
$$

构造 noisy action sequence：

$$
A_t^k = A_t^0 + \epsilon^k
$$

### Step 4：模型预测噪声

模型输入：

$$
(O_t, A_t^k, k)
$$

输出：

$$
\hat{\epsilon} = \epsilon_\theta(O_t, A_t^k, k)
$$

训练 loss 是：

$$
\mathcal{L}
=
\text{MSE}
\left(
\epsilon^k,
\epsilon_\theta(O_t, A_t^0 + \epsilon^k, k)
\right)
$$

所以它不是直接监督：

$$
\hat{A}_t = A_t^0
$$

而是监督：

$$
\hat{\epsilon} = \epsilon^k
$$

直观理解是：模型学会“看到一个 noisy action sequence 后，判断里面混入了多少噪声”，从而在推理时一步步把随机噪声去噪成专家动作序列。`Diffusion_policy.pdf`

---

## 3.4 Noise Prediction Network Pipeline

论文实现了两种 backbone：CNN-based 和 Transformer-based。

---

### 3.4.1 CNN-based Diffusion Policy

CNN 版本把动作序列看成 1D temporal signal：

$$
A_t^k \in \mathbb{R}^{T_p \times d_a}
$$

其中 $T_p$ 是动作序列长度，$d_a$ 是动作维度。

动作序列经过一系列 1D convolution block。观测条件 $O_t$ 和 diffusion step $k$ 通过 FiLM 注入到每个卷积层中：

$$
\text{FiLM}(x) = \gamma(O_t, k) \cdot x + \beta(O_t, k)
$$

也就是说，condition 不作为 token attend 进去，而是生成 scale/bias，对每层中间特征做调制。

CNN 版的优点是：大多数任务上比较稳定、调参少、作为新任务 baseline 更推荐。缺点是：temporal convolution 有低频平滑偏置，对动作变化很快的任务可能过度平滑。论文也建议，一般新任务先试 CNN，如果任务复杂或动作频率变化高，再试 Transformer。`Diffusion_policy.pdf`

---

### 3.4.2 Transformer-based Diffusion Policy

Transformer 版本把 noisy action sequence 当成 token 序列。

每个 noisy action：

$$
a_{t+i}^k
$$

先投影成 action embedding：

$$
h_i = W_a a_{t+i}^k + p_i
$$

其中 $p_i$ 是位置编码。

diffusion step $k$ 用 sinusoidal embedding 表示，并 prepend 到 action token 前面：

$$
H^0 = [e_k, h_0, h_1, \dots, h_{T_p-1}]
$$

观测 $O_t$ 被编码成 observation embedding sequence，然后作为 cross-attention 的条件输入。每个 transformer decoder block 中，action token 通过 self-attention 建模动作序列内部依赖，再通过 cross-attention 读取 observation embedding。论文 Figure 2 里也明确画出：observation embedding 进入 transformer decoder 的 multi-head cross-attention，每个 action embedding 只能 attend 自己和之前的 action embedding。`Diffusion_policy.pdf`

最后每个 action token 输出对应位置的噪声预测：

$$
\epsilon_\theta(O_t, A_t^k, k)
=
[\hat{\epsilon}_0, \hat{\epsilon}_1, \dots, \hat{\epsilon}_{T_p-1}]
$$

Transformer 版适合复杂任务、高频动作变化、velocity control 等，但对超参数更敏感。`Diffusion_policy.pdf`

---

## 3.5 推理 / 部署 Pipeline

推理时没有真实动作序列，模型需要从噪声开始生成动作。

完整推理流程如下。

### Step 1：读取当前观测

机器人在时刻 $t$ 获取最近 $T_o$ 帧观测：

$$
O_t = [o_{t-T_o+1}, \dots, o_t]
$$

包括图像、机器人状态、末端位姿等。

### Step 2：编码 observation

视觉 encoder / state encoder 把 $O_t$ 编码成 condition embedding。

### Step 3：初始化随机动作噪声

从高斯分布采样一个随机动作序列：

$$
A_t^K \sim \mathcal{N}(0, I)
$$

它的 shape 和最终动作序列一致：

$$
A_t^K \in \mathbb{R}^{T_p \times d_a}
$$

### Step 4：多步 denoising

从 $k = K$ 到 $k = 1$，反复调用噪声预测网络：

$$
\epsilon_\theta(O_t, A_t^k, k)
$$

然后根据 DDPM/DDIM 更新：

$$
A_t^{k-1}
=
\alpha
\left(
A_t^k
-
\gamma \epsilon_\theta(O_t, A_t^k, k)
+
\mathcal{N}(0, \sigma^2 I)
\right)
$$

最终得到 clean action sequence：

$$
A_t^0 = [a_t, a_{t+1}, \dots, a_{t+T_p-1}]
$$

论文真实机器人实验中用 DDIM 减少推理步数，例如训练 100 个 diffusion steps，但真实推理减少到 10 或 16 个 steps，以满足实时控制延迟要求。`Diffusion_policy.pdf`

### Step 5：只执行前 $T_a$ 个动作

虽然模型预测了 $T_p$ 步动作，但并不全部执行，而是只执行前 $T_a$ 步：

$$
[a_t, a_{t+1}, \dots, a_{t+T_a-1}]
$$

执行完这几步后，机器人重新读取新观测，再生成下一段动作。

这就是 receding horizon control。

---

## 3.6 闭环控制 Pipeline

整体执行可以写成：

```text
while task not finished:
    1. collect latest To observations O_t
    2. encode O_t
    3. initialize noisy action sequence A_t^K
    4. denoise K steps / DDIM steps to get A_t^0
    5. execute first Ta actions
    6. move time forward by Ta
    7. re-observe and re-plan
```

这点非常关键。Diffusion Policy 不是一次性生成完整任务轨迹，然后开环执行到底，而是持续 re-plan。论文强调，这个设计在保持动作时间一致性的同时，还能保持对环境变化的响应能力。`Diffusion_policy.pdf`

---

## 4. 为什么 position control 更适合 Diffusion Policy？

论文有一个很重要的发现：**Diffusion Policy 用 position control 通常比 velocity control 更好**。这和很多传统 BC 方法相反，因为很多传统方法更偏向 velocity control。

作者的解释有两层。

第一，position control 下动作多模态更明显。例如同一个状态下，机器人可以选择左侧目标位置，也可以选择右侧目标位置。普通 BC 容易把多个 position mode 平均掉。但 Diffusion Policy 擅长表达多模态分布，所以不太怕这个问题。

第二，position control 比 velocity control 更不容易产生误差累积。velocity control 是增量式的：

$$
x_{t+n} = x_t + \sum_{i=0}^{n-1} \Delta x_{t+i}
$$

每一步 velocity 误差都会被积分累积。而 position control 输出的是目标位置 / 目标位姿，后续 action chunk 更像一串 waypoint，所以更适合 sequence prediction。论文 Figure 4 显示 baseline 从 velocity 换 position 会下降，但 Diffusion Policy 能从 position control 中受益。`Diffusion_policy.pdf`

---

## 5. Diffusion Policy 为什么有效？

### 5.1 多模态动作建模

Diffusion Policy 从不同随机噪声初始化出发，可以落到不同动作模式上。例如 Push-T 中，同一个状态可以从左边或右边绕过去推。论文 Figure 3 展示了 Diffusion Policy 可以学习两个模式，并且一次 rollout 内会 commit 到其中一个模式，而 BET 会因为缺少 temporal consistency 在模式之间摇摆。`Diffusion_policy.pdf`

### 5.2 Action sequence 提升时间一致性

模型一次生成一段动作，而不是每一帧独立生成动作，因此连续动作更容易属于同一个模式。这样能减少“这一帧向左，下一帧向右”的 jitter。

### 5.3 对 idle action 更鲁棒

真实 teleoperation 数据中经常有停顿，比如倒酱时等待勺子装满、细调时暂停、任务结束前停留。单步 BC 容易过拟合这些 near-zero action，导致机器人卡住。Diffusion Policy 生成 action sequence，可以更好地区分“应该停顿等待”和“卡住不动”。论文在 4.3 节也把 robustness to idle actions 作为 action-sequence prediction 的一个优点。`Diffusion_policy.pdf`

### 5.4 训练比 EBM/IBC 稳定

IBC 这类 implicit policy 要训练 energy function：

$$
p_\theta(a \mid o)
=
\frac{\exp(-E_\theta(o,a))}{Z(o,\theta)}
$$

难点在于归一化项 $Z(o,\theta)$ 不可 tractable，所以需要 negative samples 近似，训练容易不稳定。

Diffusion Policy 直接学 score / noise prediction，不需要估计 partition function：

$$
\nabla_a \log p(a \mid o)
\approx
-\epsilon_\theta(a,o)
$$

因此训练和推理都绕开了 $Z(o,\theta)$，稳定性更好。论文 Figure 6 也展示了 IBC evaluation success rate 振荡明显，而 Diffusion Policy 更稳定。`Diffusion_policy.pdf`

---

## 6. 实验结果概括

论文在 Robomimic、Push-T、BlockPush、Kitchen 以及真实机器人任务上进行评估，包含 state-based 和 vision-based setting。整体上，Diffusion Policy 在多数任务中超过 LSTM-GMM、IBC、BET，并报告平均性能提升 46.9%。`Diffusion_policy.pdf`

几个关键结果：

- **仿真任务**：在 Robomimic 的 Lift、Can、Square、Transport、ToolHang，以及 Push-T 上，Diffusion Policy 在 state 和 vision 设置下都表现更好。
- **多阶段任务**：BlockPush 和 Kitchen 体现 long-horizon multimodality。Diffusion Policy 在 BlockPush p2 和 Kitchen p4 指标上明显优于 baseline。`Diffusion_policy.pdf`
- **真实 Push-T**：Diffusion Policy 达到 95% success rate 和 0.80 IoU，接近人类演示的 0.84 IoU；IBC 和 LSTM-GMM 的成功率明显更低。`Diffusion_policy.pdf`
- **真实复杂操作**：Mug Flip、Sauce Pouring、Sauce Spreading，以及双臂 Egg Beater、Mat Unrolling、Shirt Folding 都验证了方法可以迁移到更复杂真实任务。`Diffusion_policy.pdf`

---

## 7. 这篇论文的真正贡献

这篇论文的核心贡献不是提出新的 diffusion 数学，而是把 diffusion 变成了一个可用的机器人低层 policy 表达方式。

它证明了这套组合是有效的：

$$
\text{conditional action diffusion}
+
\text{action chunk prediction}
+
\text{receding horizon control}
+
\text{visual conditioning}
+
\text{position control}
$$

单独说“把 diffusion 用于 action 生成”没错，但不完整。它真正有价值的是：把 diffusion action generator 放进了闭环机器人控制 pipeline 里，并且系统验证了 action horizon、position/velocity control、视觉条件注入、推理加速等关键设计。论文结论也强调，receding-horizon action prediction、end-effector position control 和 efficient visual conditioning 是释放 diffusion policy 性能的关键因素。`Diffusion_policy.pdf`

---

## 8. 局限性

第一，它本质上仍然是 behavior cloning，所以依赖专家数据质量。数据不足、覆盖不够、分布外场景都会影响性能。

第二，它不是 high-level planner，也不理解语言，不做任务分解，不显式建模 world model。原始 Diffusion Policy 更像是一个 task-specific / dataset-specific 的低层动作生成器。

第三，推理成本比 LSTM-GMM 这类单步 forward policy 更高。虽然 DDIM 和 action chunk 能缓解延迟，但对于非常高频控制任务，仍可能成为瓶颈。论文也明确指出，diffusion policy 相比简单方法有更高计算成本和推理延迟。`Diffusion_policy.pdf`

---

## 9. 和 ACT / VLA / π 系列的关系

从机器人基础模型脉络看，Diffusion Policy 更像是低层 action head 的前置工作。

| 方法 | 关键点 | 与 Diffusion Policy 的关系 |
|---|---|---|
| ACT | CVAE action chunk | 同样强调 action chunk，但生成模型不同 |
| Diffusion Policy | diffusion action chunk | 重点是连续多模态动作序列生成 |
| OpenVLA / Octo | 多任务 VLA 泛化 | 更关注语言/视觉到动作的通用泛化 |
| $\pi_0$ 系列 | flow / diffusion-like action generation + foundation model | 可看作把低层生成式 action head 融入更大规模 VLA 框架 |
| WAM / world model | 建模未来状态/世界变化 | Diffusion Policy 本身不预测未来世界，只预测动作 |

所以这篇论文在你的阅读体系里应该被定位为：

> **机器人连续动作生成范式论文，而不是高层认知或世界模型论文。**

---

## 10. 最终总结

Diffusion Policy 的本质是：**给定当前观测，用 diffusion model 生成未来一段连续动作序列，再通过 receding horizon 闭环执行。**

完整 pipeline 是：

```text
专家数据
→ 截取历史观测 O_t 和未来动作序列 A_t^0
→ 给 A_t^0 加不同强度噪声
→ 训练 εθ(O_t, A_t^k, k) 预测噪声
→ 推理时从随机动作噪声 A_t^K 开始
→ 多步 denoise 得到 clean action sequence A_t^0
→ 只执行前 Ta 步
→ 重新观测并 re-plan
```

它的核心价值是：用 diffusion 的多模态分布建模能力解决机器人 BC 中的多模态、动作抖动、idle action、高维 action sequence 和训练稳定性问题。它不是 VLA，也不是 world model，但它为后续很多机器人 foundation model 中的 diffusion / flow action head 提供了非常重要的技术基础。


---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
## 相关笔记

- [[Robot/PI/FAST_知识总结|FAST 知识总结]]
- [[Robot/ChatGPT-ALOHA硬件与ACT算法|ALOHA 硬件与 ACT 算法]]
- [[Robot/PI/ChatGPT-Pi_0机器人文章分析|pi0 机器人文章分析]]
- [[Robot/PI/ChatGPT-Pi_0.6论文问题解答|pi0.6 论文问题解答]]
- [[Robot/ChatGPT-RDT-1B|RDT-1B]]
- [[Robot/ChatGPT-RT-1 论文综述|RT-1 论文综述]]
- [[Robot/ChatGPT-RT-2 论文综述|RT-2 论文综述]]
- [[Robot/ChatGPT-GR00T N1 综述|GR00T N1 综述]]
- [[Robot/ChatGPT-MolmoAct2论文框架分析|MolmoAct2 论文框架分析]]
