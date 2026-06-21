---
title: pi*0.6 RECAP 论文问题解答
type: paper_note
topic: robotics_foundation_model
status: mature
importance: high
updated: 2026-06-10
tags:
  - pi-star-0.6
  - recap
  - vla
  - advantage-conditioned-policy
  - offline-rl
  - robotics
---

# Pi_!0.6论文问题解答

**User:** Anonymous  
**Created:** 5/22/2026 18:18:28  
**Updated:** 6/3/2026 17:41:21  
**Exported:** 6/3/2026 17:45:05  
**Link:** [https://chatgpt.com/c/6a102d5a-553c-83ec-8e76-7937eb760753](https://chatgpt.com/c/6a102d5a-553c-83ec-8e76-7937eb760753)  

---

# π\*0.6 / RECAP 完整理解

这篇论文的核心不是单纯提出一个新 VLA 架构，而是提出一套让 VLA **从真实机器人经验中继续提升**的方法。模型叫 **π\*0.6**，训练方法叫 **RECAP**：

$$
\text{RL with Experience and Corrections via Advantage-conditioned Policies}
$$

它想解决的问题是：

> 普通 VLA / imitation learning 主要模仿 demonstration，但真实机器人部署时会犯自己的错误。只靠人类示范数据，很难修正 deployment-time mistakes，也很难让模型在速度和鲁棒性上超过示范数据。因此需要让 VLA 从 autonomous rollouts、失败反馈、人类 correction/intervention 中继续学习。`pi_*0.6.pdf`

RECAP 的核心做法是：

> 不直接把 PPO/TRPO 硬套到大 VLA 上，而是训练一个 value function 给动作估计 advantage，再把 advantage 二值化成 `Advantage: positive / negative`，作为文本条件输入 VLA。训练时好坏动作都学，但分 condition 学；推理时固定输入 `Advantage: positive`，从而抽取更优策略。

---

# 1. π0.6 原本是什么？

π\*0.6 是基于 π0.6 改出来的。π0.6 本身是一个 vision-language-action model。

它的 observation 输入是：

$$
o_t = [X_t^1,\dots,X_t^n,q_t]
$$

其中：

- $X_t^i$：多路相机图像；
- $q_t$：机器人 proprioceptive state，比如关节位置、夹爪状态；
- $\ell$：语言输入。

论文把语言输入写成：

$$
\ell = \ell_t + s
$$

其中：

- $\ell_t$：主任务 prompt，比如 “make me an espresso”；
- $s$：额外 language inputs / metadata，用来进一步调制任务怎么执行。

π0.6 的输出不是单一动作，而是一个联合输出：

$$
\pi_\theta(a_{t:t+H},a^\ell_{t:t+H},\hat{\ell}|o_t,\ell)
$$

这里有三类输出。

第一类是高层 subtask / text output：

$$
\hat{\ell}
$$

比如：

```text
pick up the coffee cup
```

或者：

```text
fold the left sleeve
```

第二类是离散化动作 token：

$$
a^\ell_{t:t+H}
$$

这是用 FAST tokenizer 对 action chunk 离散化得到的 token 序列。它可以像语言 token 一样用 autoregressive next-token prediction 训练。

第三类是真正的连续 robot action chunk：

$$
a_{t:t+H}
$$

比如未来一段时间内的关节角、夹爪命令等。这部分不是普通 vocabulary token，而是由 flow matching action expert 生成。

所以 π0.6 是一个混合输出模型：

```text
图像 + robot state + task prompt
→ high-level subtask text
→ FAST discrete action tokens
→ continuous action chunk
```

论文里还说，因为 $\hat{\ell}$ 是先预测的，所以 action generation 实际上 conditioned on predicted subtask。也就是说，模型先决定“现在做什么子任务”，再生成对应动作。`pi_*0.6.pdf`

---

# 2. π\*0.6 相比 π0.6 加了什么？

π\*0.6 的关键改动是：**在 π0.6 的输入序列里加入 advantage indicator**。

这个 indicator 是文本形式：

```text
Advantage: positive
```

或者：

```text
Advantage: negative
```

它的含义是：

> 当前训练样本里的动作，是 value function 判断为高 advantage 的动作，还是低 advantage 的动作。

它被放在训练序列中：

$$
\hat{\ell}
\quad \rightarrow \quad
I_t
\quad \rightarrow \quad
a^\ell_{t:t+H}, a_{t:t+H}
$$

也就是：

```text
observation + task prompt
→ predict subtask
→ insert Advantage: positive / negative
→ predict discrete action tokens
→ predict continuous action chunk
```

这个位置很重要。它意味着：

1. 高层 subtask prediction 不受 advantage indicator 影响；
2. advantage indicator 主要影响动作生成；
3. 模型先决定当前子任务，再根据 `positive / negative` condition 决定动作风格。

所以 π\*0.6 的 action distribution 可以理解成：

$$
\pi_\theta(a_{t:t+H}|o_t,\ell,\hat{\ell},I_t)
$$

推理时，通常固定：

$$
I_t=\text{positive}
$$

也就是一直给模型输入：

```text
Advantage: positive
```

这样模型会生成更像 high-advantage 数据里的动作。`pi_*0.6.pdf`

---

# 3. value function：它负责判断状态和动作好坏

advantage indicator 不是人工随便标的，而是由 value function 估出来的。

论文训练了一个 distributional value function：

$$
p_\phi(V|o_t,\ell)\in \Delta_B
$$

其中：

$$
B=201
$$

也就是说，value function 输入当前 observation 和任务语言，输出一个 201 维 value-bin distribution。

最后如果需要 scalar value，就对这个分布取期望：

$$
V(o_t,\ell)=\sum_b p_\phi(V=b|o_t,\ell)v(b)
$$

这个 value function 的语义不是“过去已经花了多少步”，而是：

> 从当前状态开始，未来到成功还剩多少成本，或者这个状态有多接近失败。

论文的 reward 设计是：

$$
r_t=
\begin{cases}
0 & t=T \text{ and success}\\
-C_{\text{fail}} & t=T \text{ and failure}\\
-1 & \text{otherwise}
\end{cases}
$$

所以对于 successful episode：

$$
V(o_t) \approx -\text{remaining steps to success}
$$

对于 failed episode，value 会是一个很大的负值。

因为不同任务的最大长度不同，比如有的任务 200 秒，有的任务 500 秒，有的任务 600 秒，所以他们会把 value normalize 到：

$$
(-1,0)
$$

其中：

- $0$：任务成功完成；
- 越接近 $-1$：离成功越远，或者失败风险越高。

比如某任务最大长度是 200 steps，如果当前状态到成功还剩 50 steps，那么 raw return 是：

$$
-50
$$

归一化后大概是：

$$
-\frac{50}{200}=-0.25
$$

这也回答了你之前那个疑问：raw value 当然可能小于 $-1$，但论文训练的是按任务长度 normalize 后的 value。`pi_*0.6.pdf`

---

# 4. VLA 形式的 value model 如何输出 201 维 value？

论文明确说了几件事：

1. value function 使用和 VLA policy 类似的 architecture design；
2. backbone 更小，是一个 670M 参数的 VLM backbone；
3. 它有一个 value head；
4. 输出是 201-bin distribution。

也就是说，最后一定是类似：

$$
\text{VLM hidden representation} \rightarrow \text{value head} \rightarrow 201 \text{ logits}
$$

但论文没有公开说明这个 value head 具体接在哪个位置。它没有明确说是：

- special value token；
- last token；
- learnable query；
- pooled representation；
- CLS-like token。

所以最严谨的说法是：

> 论文只说明 value function 是 VLM-style backbone + value head，输出 201-bin distribution；但没有披露 value head 是基于 special token、learnable query，还是 pooling hidden states 得到的。

如果按常见实现，可能有几种方式：

```text
方式 A: prompt 后加 <VALUE> token，取该 token hidden state 接 linear head
方式 B: 取最后一个文本 token hidden state 接 value head
方式 C: 用 learnable query attend 到视觉/语言 tokens 后接 value head
方式 D: 对若干 hidden states pooling 后接 value head
```

但这些是工程可能性，不是论文明确结论。

---

# 5. 为什么 value 要输出 distribution，而不是直接回归 scalar？

直接回归当然可以，比如：

$$
\min_\phi \left(V_\phi(o_t,\ell)-R_t\right)^2
$$

但论文用的是 distributional value function，也就是把 value 离散成 201 个 bins，然后用 cross entropy 训练。

这样做的好处是：

1. 机器人任务的 return 很 noisy；
2. 同一个视觉状态附近，未来可能成功，也可能失败；
3. 一个 scalar mean 会把多种可能结果平均掉；
4. distributional prediction 能表达不确定性和多峰结构；
5. 对 VLM/VLA 这类模型来说，分类式 cross entropy 通常比连续 MSE 更稳定。

举例：

某个状态可能有两种未来：

$$
70\% \rightarrow V=-0.2
$$

$$
30\% \rightarrow V=-1.0
$$

如果直接输出 mean：

$$
V=-0.44
$$

这个数不一定对应任何真实模式。  
如果输出 distribution，可以表达：

$$
p(V=-0.2)=0.7,\quad p(V=-1.0)=0.3
$$

当然，后面计算 advantage 时，还是会把分布取 expectation 得到 scalar value。

所以是：

```text
训练时：学 value distribution
使用时：取 expectation 得到 scalar value
```

---

# 6. value label 是什么时候得到的？是不是必须完整执行完任务？

是的，value function 的监督信号基本是 episode 结束后 post-hoc 得到的。

论文定义：

$$
R_t(\tau)=\sum_{t'=t}^{T}r_{t'}
$$

也就是从时间 $t$ 到 episode 结束的 empirical return。

这需要知道整条轨迹最终是 success 还是 failure，也需要知道 episode 结束位置。因此 value function 训练用的是 Monte Carlo return，通常要等 trajectory / episode 完成后才能生成 label。

这和在线执行时的视角不同。

在线执行时，机器人当然不知道未来还剩多少步。但训练时，episode 已经收集完了，可以从后往前给每个时间点打标签。

比如一条成功轨迹总共 5 步：

```text
t=0: 开始
t=1: 拿起衣服
t=2: 铺平
t=3: 折袖子
t=4: 放到堆上
t=5: success
```

中间每步 reward 是 $-1$，成功终点是 $0$。那么 episode 结束后回头算：

$$
R_0=-5
$$

$$
R_1=-4
$$

$$
R_2=-3
$$

$$
R_3=-2
$$

$$
R_4=-1
$$

$$
R_5=0
$$

所以 value 从负数逐渐接近 0。它学的是 future return / remaining cost，不是已经花掉的 past cost。

---

# 7. n-step advantage 在实际中如何计算？

论文在 preliminaries 里定义了 n-step advantage：

$$
A^\pi(o_t,a_t)
=
\mathbb{E}
\left[
\sum_{t'=t}^{t+N-1}r_{t'}
+
V^\pi(o_{t+N})
\right]
-
V^\pi(o_t)
$$

直觉是：

> 执行动作 $a_t$ 后，未来 $N$ 步实际获得的 reward，加上 $o_{t+N}$ 的 value，是否比当前 $V(o_t)$ 更好？

如果这个值大，说明动作让任务进展超出预期；如果小，说明动作没带来进展，甚至让任务变差。

实际训练中，reward 不是实时 dense reward，而是 episode 结束后根据 success/failure label 派生出来：

- 中间每步 $-1$；
- 成功终点 $0$；
- 失败终点 $-C_{\text{fail}}$。

所以 advantage 通常也是在收集完轨迹并标注完 episode 结果后，在训练 pipeline 里离线计算或 on-the-fly 计算。

这里的 on-the-fly 不是指机器人实时执行时算 reward，而是指 VLA training 时对已存在的数据样本动态跑 value function 来估计：

$$
A(o_t,a_t,\ell)
$$

再根据阈值：

$$
I_t=\mathbf{1}(A(o_t,a_t,\ell)>\epsilon_\ell)
$$

得到：

```text
Advantage: positive / negative
```

论文没有公开实际使用的 $N$ 是多少，也没有详细披露 action chunk 级 advantage 的所有工程细节。

---

# 8. chunk action 和 $I_t$：一个 indicator 对应多少步？

π0.6 / π\*0.6 输出的是 action chunk：

$$
a_{t:t+H}
$$

论文说动作以 50Hz 控制，输出的是未来一段 horizon 的 joint angles 和 gripper commands。也就是说，模型不是只输出当前一个 instantaneous action，而是输出一个 action chunk。

你问得很关键：如果 chunk 里有多个 step，不同步的 $I$ 可能不同，那怎么输入？

论文公式里是：

$$
\log \pi_\theta(a_{t:t+H},a^\ell_{t:t+H}|I_t,o_t,\ell,\hat{\ell})
$$

也就是说，一个 $I_t$ condition 对应整个 action chunk：

$$
I_t \rightarrow a_{t:t+H}
$$

而不是：

$$
I_t,I_{t+1},...,I_{t+H}
$$

所以它实际是 **chunk-level conditioning**。可以理解为：

> 在当前 observation $o_t$ 下，整个 action chunk 被作为一个动作整体，标记为 positive 或 negative。

如果训练数据是滑窗，下一条训练样本可以是：

$$
(o_{t+1},a_{t+1:t+H+1},I_{t+1})
$$

或者用某种 stride 取样。但论文没有公开 stride / overlap 的细节。

所以严谨结论是：

> 论文明确公式里是每个 action chunk 使用一个 $I_t$。它没有说明 chunk 内每个 50Hz sub-step 都有单独 indicator。

这确实是一种近似：用当前起点的 advantage 标记整个未来 chunk。

---

# 9. regularized RL 的 theoretical improved policy

RECAP 的 advantage conditioning 不是拍脑袋设计的，它来自 regularized RL 的理论动机。

普通 RL 是最大化：

$$
J(\pi)
$$

但 regularized RL 希望新策略 reward 更高，同时不要偏离 reference policy 太远。

局部来看，可以写成：

$$
\max_{\pi(\cdot|o)}
\mathbb{E}_{a\sim\pi}[A^{\pi_{\text{ref}}}(o,a)]
-
\beta\mathrm{KL}(\pi(\cdot|o)\|\pi_{\text{ref}}(\cdot|o))
$$

解这个优化问题，会得到：

$$
\hat{\pi}(a|o)
\propto
\pi_{\text{ref}}(a|o)
\exp\left(\frac{A(o,a)}{\beta}\right)
$$

这个式子的含义是：

$$
\text{new policy}
=
\text{reference policy}
\times
\text{advantage-based reweighting}
$$

高 advantage 动作概率被提高，低 advantage 动作概率被降低。

这里需要注意：

> “不偏离旧策略太远”已经通过 KL regularization 被吸收到解析解里了，不是解析解之外还有额外约束。

这个效果体现在：

- $\pi_{\text{ref}}(a|o)$：新策略以 reference policy 为底；
- $\beta$：控制 advantage reweighting 强度。

$\beta$ 越大，更新越保守；$\beta$ 越小，更新越激进。

---

# 10. 为什么不直接用 PPO/TRPO？

因为 π\*0.6 的 action model 太复杂。

它有：

1. autoregressive subtask tokens；
2. FAST discrete action tokens；
3. flow matching continuous action chunk；
4. long-horizon chunked action；
5. 混合数据来源：demonstration、autonomous rollout、human intervention、旧策略 rollout。

PPO/TRPO 通常需要：

$$
\log \pi_\theta(a|o)
$$

或者 probability ratio：

$$
\frac{\pi_\theta(a|o)}{\pi_{\text{old}}(a|o)}
$$

但 flow matching continuous action expert 不方便提供精确 tractable likelihood。它主要输出 vector field / denoising direction，而不是显式 Gaussian policy 的 $\mu,\sigma$。

所以论文选择不做标准 policy gradient，而是做：

$$
\text{policy extraction via advantage conditioning}
$$

也就是把 RL policy improvement 转成 conditional supervised learning 问题。

---

# 11. 从 advantage reweighting 到 conditional policy：贝叶斯变换

论文借鉴 CFGRL 的思想，把 improvement probability 写成：

$$
p(I|A(o,a))
$$

这里 $I$ 表示：

> 动作 $a$ 是 improvement action。

regularized RL 的另一种 improved policy 形式是：

$$
\hat{\pi}(a|o,\ell)
\propto
\pi_{\text{ref}}(a|o,\ell)
p(I|A(o,a,\ell))^\beta
$$

然后用 Bayes rule：

$$
p(I|a,o,\ell)
=
\frac{p(a|I,o,\ell)p(I|o,\ell)}{p(a|o,\ell)}
$$

忽略和动作 $a$ 无关的常数：

$$
p(I|a,o,\ell)
\propto
\frac{\pi_{\text{ref}}(a|I,o,\ell)}
{\pi_{\text{ref}}(a|o,\ell)}
$$

代回 improved policy：

$$
\hat{\pi}(a|o,\ell)
\propto
\pi_{\text{ref}}(a|o,\ell)
\left(
\frac{
\pi_{\text{ref}}(a|I,o,\ell)
}{
\pi_{\text{ref}}(a|o,\ell)
}
\right)^\beta
$$

当：

$$
\beta=1
$$

时：

$$
\hat{\pi}(a|o,\ell)
=
\pi_{\text{ref}}(a|I,o,\ell)
$$

这就是 RECAP 的理论关键：

> 如果能训练一个 conditional policy $\pi(a|I,o,\ell)$，那么在 $I=\text{positive}$ 条件下，它就可以表示 regularized RL 意义下的 improved policy。

所以 RECAP 不直接显式构造：

$$
\pi_{\text{ref}}(a|o)\exp(A/\beta)
$$

而是训练：

$$
\pi_\theta(a|I,o,\ell)
$$

推理时输入：

```text
Advantage: positive
```

---

# 12. positive / negative loss 为什么能起作用？

训练 objective 抽象写成：

$$
\min_\theta
\mathbb{E}_{D_{\pi_{\text{ref}}}}
\left[
-\log\pi_\theta(a_t|o_t,\ell)
-
\alpha\log\pi_\theta(a_t|I_t,o_t,\ell)
\right]
$$

其中：

$$
I_t=\mathbf{1}(A(o_t,a_t,\ell)>\epsilon_\ell)
$$

关键点是：positive 和 negative 样本的 loss 形式是一样的，都是：

$$
-\log\pi_\theta(a_t|I_t,o_t,\ell)
$$

它们不是一个正向 loss、一个反向 loss。

indicator 能起作用，不是因为 positive / negative loss 公式不同，而是因为：

$$
p_{\text{data}}(a|o,\ell,I=\text{positive})
\neq
p_{\text{data}}(a|o,\ell,I=\text{negative})
$$

也就是说：

- positive condition 下的数据是 high-advantage action；
- negative condition 下的数据是 low-advantage / failure action。

模型为了降低同一个 conditional generation loss，会学出：

$$
\pi_\theta(a|o,\ell,I=\text{positive})
\neq
\pi_\theta(a|o,\ell,I=\text{negative})
$$

推理时只输入：

```text
Advantage: positive
```

所以实际执行的是 positive conditional distribution。

---

# 13. 为什么还要“学坏轨迹”？这不是很奇怪吗？

这个问题非常关键。

RECAP 不是在 positive policy 下模仿坏轨迹，而是在 negative condition 下建模坏轨迹。

也就是说：

```text
Advantage: positive -> 学好动作分布
Advantage: negative -> 学坏动作分布
```

推理时只用：

```text
Advantage: positive
```

所以学坏轨迹不等于执行坏轨迹。

正常 RL 里其实也不会完全丢掉坏轨迹。比如 Q-learning / actor-critic 也需要失败轨迹来训练 critic：

```text
这个状态/动作导致低 return
```

然后 actor 避免这些动作。

RECAP 把类似思想变成 conditional generative modeling：

> 坏轨迹作为 negative condition 下的训练样本，用来帮助模型区分 high-advantage 和 low-advantage 行为分布。

它和 AWR 的差异是：

AWR：

$$
\mathcal{L}_{AWR}=w(A)[-\log\pi_\theta(a|o)]
$$

高 advantage 样本权重大，低 advantage 样本权重小。它相当于：

```text
好样本多学，坏样本少学
```

RECAP：

```text
好坏样本都学，但分 condition 学，推理只取 positive condition
```

当然，这种方法依赖一个前提：模型真的利用了 advantage indicator。如果模型忽略 condition，或者 positive / negative 分布不够可分，那坏数据可能污染执行分布。这也是这个方法的潜在风险。

论文用 task-dependent threshold $\epsilon_\ell$、把 indicator 放在 action 前、以及训练时随机 omit indicator / CFG 机制来增强这种条件化能力。`pi_*0.6.pdf`

---

# 14. human correction 怎么处理？

human intervention / correction 被强制标为：

$$
I_t=\text{positive}
$$

原因是论文假设 expert corrective action 是好的。

这很重要，因为机器人自己犯错时会进入 demonstration 中很少见的状态，比如：

- 衣服已经被抓乱；
- portafilter 快掉了；
- box 被折坏一半；
- 夹爪抓空后姿态异常。

这些状态下，人类 correction 提供了：

```text
在失败边缘如何恢复
```

RECAP 把这些 correction 作为 positive 数据训练，使模型学会从失败边缘恢复。

---

# 15. 真实 training loss：离散 likelihood + 连续 flow matching

理论上我们写：

$$
\log\pi_\theta(a|I,o,\ell)
$$

但真实 π\*0.6 同时输出 discrete 和 continuous actions，所以实际 loss 要拆开。

离散部分包括：

- high-level subtask tokens；
- FAST action tokens。

这些可以用 standard likelihood / cross entropy。

连续动作：

$$
a_{t:t+H}
$$

由 flow matching action expert 生成，不能方便地精确计算 closed-form likelihood，所以用 flow matching objective。

论文定义 noisy action：

$$
a^{\eta,\omega}_{t:t+H}
=
\eta a_{t:t+H}
+
(1-\eta)\omega
$$

其中：

- $a_{t:t+H}$：ground-truth continuous action chunk；
- $\omega\sim\mathcal{N}(0,I)$：高斯噪声；
- $\eta\in[0,1]$：flow matching time；
- $f_\theta$：模型输出的 vector field / denoising direction。

loss 里有：

$$
\left\|
\omega-a_{t:t+H}
-
f_\theta(a^{\eta,\omega}_{t:t+H},I_t,o_t,\ell,\hat{\ell})
\right\|^2
$$

你之前问过：正常 flow matching 不是应该用 $gt-noise$ 吗？

这里其实只是方向定义不同。

如果定义：

$$
v_\theta=-f_\theta
$$

那么 target 就变成：

$$
a-\omega
$$

也就是你熟悉的 ground truth minus noise。

所以不是多了奇怪的 $\omega$，而是 target vector 本来就是 noise 和 ground truth action 之间的差。

---

# 16. flow matching loss 为什么能作为 continuous likelihood surrogate？

更严谨地说：

$$
\mathcal{L}_{FM}
$$

不是精确等于：

$$
-\log p_\theta(a|o)
$$

它是 continuous action log-likelihood 的 surrogate / lower-bound style objective。

原因是 flow matching / diffusion 可以看作 latent-variable generative model。

我们真正想最大化：

$$
\log p_\theta(a|o)
$$

但生成过程中有 latent variables：

$$
\omega,\eta
$$

所以：

$$
p_\theta(a|o)=\int p_\theta(a,\omega,\eta|o)d\omega d\eta
$$

直接算这个 marginal likelihood 很难。

于是可以构造一个 denoising Gaussian likelihood。给定 noisy action $a^{\eta,\omega}$，模型预测 denoising vector。Gaussian log-likelihood 展开后就是 MSE 项。

通过 Jensen / ELBO 思路，可以得到类似：

$$
\log p_\theta(a|o)
\ge
\mathbb{E}_{\eta,\omega}
[
-\text{weighted flow matching MSE}
]
+
C
$$

所以最大化这个 lower bound，就等价于最小化 flow matching loss。

这就是为什么论文说 continuous log-likelihood 不能精确计算，但可以用 flow matching objective 作为替代训练目标。`pi_*0.6.pdf`

---

# 17. offline RL pretraining 是什么？

论文里有一个 baseline：

```text
π*0.6 offline RL + SFT
```

这里的 offline RL 指的是：

> 在大规模已有离线机器人数据上，用 RECAP-style value function + advantage conditioning 做预训练。

它不是后面真实机器人 rollout 的在线训练。

具体流程是：

1. 有一个多任务、多机器人 demonstration dataset；
2. 在这个离线数据上训练 value function；
3. value function 根据任务进度 / 成功失败预测 value；
4. 用 value function 给动作估计 advantage；
5. 根据 advantage 生成 positive / negative indicator；
6. 用 advantage-conditioned objective 训练 π\*0.6。

所以 offline RL pre-trained π\*0.6 不是普通 SFT 模型，而是一个已经在离线数据上学过“哪些动作更接近成功”的 checkpoint。

之后用目标任务 high-quality demonstration 做 SFT，论文说这种：

```text
offline RL pre-trained π*0.6 + high-quality SFT
```

比普通 standard SFT 更好。`pi_*0.6.pdf`

---

# 18. SFT 阶段为什么把 advantage 固定为 True？

在目标任务 demonstration fine-tuning 阶段，他们把 indicator 固定为 True：

```text
Advantage: positive
```

因为 high-quality demonstrations 默认是好的行为。

所以目标任务 SFT 实际上是：

$$
\text{SFT on demonstrations under positive condition}
$$

也就是：

```text
目标任务 observation + task + subtask + Advantage: positive
→ demonstration action
```

这得到初始 specialist policy。

---

# 19. RECAP 完整训练算法

论文明确给了 **Algorithm 1: RL with Experience and Corrections via Advantage-conditioned Policies (RECAP)**。核心流程是：

```text
Require: multi-task demonstration dataset D_demo

1: Train V_pre on D_demo using Eq. 1
2: Train π_pre on D_demo using Eq. 3 and V_pre
3: Initialize D_l with demonstrations for task l
4: Train V_l^0 from V_pre on D_l using Eq. 1
5: Train π_l^0 from π_pre on D_l using Eq. 3 and V_l^0
6: for k = 1 to K do
7:     Collect data with π_l^{k-1}, add it to D_l
8:     Train V_l^k from V_pre on D_l using Eq. 1
9:     Train π_l^k from π_pre on D_l using Eq. 3 and V_l^k
10: end for
```

把它翻译成更具体的流程：

## 预训练阶段

在大规模离线多任务数据上：

1. 训练 value function $V_{\text{pre}}$；
2. 用 $V_{\text{pre}}$ 给数据里的动作估计 advantage；
3. 根据 threshold 生成 `Advantage: positive / negative`；
4. 用 advantage-conditioned objective 训练 $\pi_{\text{pre}}$。

## 目标任务 SFT

对某个任务 $\ell$：

1. 初始化 $D_\ell$ 为该任务 demonstrations；
2. 从 $V_{\text{pre}}$ fine-tune value；
3. 从 $\pi_{\text{pre}}$ fine-tune policy；
4. demonstration 通常设为 positive。

得到初始策略：

$$
\pi_\ell^0
$$

## 真实机器人迭代

循环：

1. 用上一轮策略 $\pi_\ell^{k-1}$ 上机器人收集数据；
2. 数据包括 autonomous rollout、success/failure episode、human correction；
3. 新数据加入 $D_\ell$；
4. 从 $V_{\text{pre}}$ 重新 fine-tune value；
5. 重新估计 advantage 和 $I_t$；
6. 从 $\pi_{\text{pre}}$ 重新 fine-tune policy；
7. 得到 $\pi_\ell^k$。

---

# 20. 为什么每轮都从 pretrained checkpoint 重新 fine-tune？

论文说：

> Both the value function and policy are finetuned from the pre-trained checkpoint, rather than the policy and value function from the last iteration.

这句话容易让人误解成：多轮训练没用。

实际上不是。多轮 RECAP 的状态主要存在数据集里，而不是上一轮模型参数里。

第 $k$ 轮的数据集是：

$$
D_\ell
=
D_{\text{demo}}
+
D_{\text{rollout}}^0
+
D_{\text{rollout}}^1
+
...
+
D_{\text{rollout}}^{k-1}
$$

每一轮虽然都从 pre-trained checkpoint 初始化，但训练数据越来越多，越来越贴近真实部署分布。

这样做的好处是避免 drift。

如果每轮都在上一轮模型上继续 fine-tune，可能出现：

- 语言能力退化；
- 动作分布越来越窄；
- 过拟合某些失败模式；
- value function 偏差被不断放大；
- policy 越训越激进；
- generalist foundation 能力丢失。

从 pretrain 重新 fine-tune，相当于每轮都回到一个稳定基础，然后用当前累计数据重新训练 specialist。

所以多轮有效的原因是：

> 每轮新 rollout 暴露新的失败模式和恢复经验，加入数据集；虽然初始化一样，但数据集不同。

---

# 21. RECAP 和普通 RL / AWR / PPO 的关系

## PPO / TRPO

PPO / TRPO 是标准 policy gradient 风格，需要：

$$
\log\pi_\theta(a|o)
$$

或者 ratio：

$$
\frac{\pi_\theta(a|o)}{\pi_{\text{old}}(a|o)}
$$

但 π\*0.6 的 continuous action 是 flow matching action chunk，likelihood 不好精确算，所以直接套 PPO/TRPO 很困难。

## AWR

AWR 是：

$$
w(A)[-\log\pi_\theta(a|o)]
$$

高 advantage 样本权重大，低 advantage 样本权重小。它容易变成 filtered imitation：

```text
好样本多学，坏样本少学
```

问题是低 advantage / failure 数据里的信息被浪费了。

## RECAP

RECAP 是：

```text
好样本在 positive condition 下学
坏样本在 negative condition 下学
推理时只输入 positive
```

所以它既利用了失败数据，又避免在执行时直接模仿失败动作。

---

# 22. RECAP 的潜在关键假设和风险

RECAP 有几个隐含前提：

## 前提 1：value function 估计足够可靠

如果 value function 错，把坏动作标成 positive，policy 会学坏。

## 前提 2：positive / negative 数据分布有明显差异

如果两者分布差别不大，indicator 提供的信息有限。

## 前提 3：模型真的利用 indicator

如果模型忽略 `Advantage: positive/negative`，那 condition 不起作用，坏数据可能混入无条件动作分布。

## 前提 4：chunk-level indicator 足够合理

一个 $I_t$ 控制整个 action chunk，是一种近似。如果 chunk 内前半段好、后半段坏，标注会变粗糙。

## 前提 5：human correction 真的比较好

论文强制把 human correction 标成 positive，这依赖 expert correction 的质量。

这些风险并不否定方法，但说明 RECAP 成功依赖 value function、threshold、数据质量、conditioning 能力和训练工程。

---

# 23. 用一个折衣服例子串完整流程

假设任务是：

```text
fold the shirt
```

## Step 1：demonstration SFT

人类示范成功折衣服。

训练时设：

```text
Advantage: positive
```

模型学：

```text
observation + fold the shirt + subtask + Advantage: positive
→ demonstration action chunk
```

## Step 2：第一次部署

机器人自己执行，产生几类轨迹。

### 成功但慢

value function 看到它最终成功，但剩余步数较多，所以 value 进展一般。

### 失败

机器人把衣服抓乱，最后失败。

这些状态 / 动作会得到较低 value 和 negative advantage：

```text
Advantage: negative
```

### 人类 intervention

机器人快失败时，人类接管并纠正。

这类 correction 强制设为：

```text
Advantage: positive
```

## Step 3：重新训练

训练样本中会出现：

```text
状态 o + Advantage: positive → 正确折叠 / 恢复动作
```

也会出现：

```text
状态 o + Advantage: negative → 抓乱衣服的动作
```

模型学到两个 conditional distributions：

$$
\pi_\theta(a|o,\ell,I=\text{positive})
$$

和：

$$
\pi_\theta(a|o,\ell,I=\text{negative})
$$

推理时固定 positive，所以更倾向正确动作。

## Step 4：第二轮

新的 policy 修掉一部分错误后，又会暴露新的失败模式。继续收集、加入数据集、重新从 pretrain fine-tune。

---

# 24. 最后的总理解

π\*0.6 / RECAP 可以这样理解：

> π\*0.6 是在 π0.6 VLA 基础上加入 advantage conditioning 的 RL-capable VLA。它保留 π0.6 的混合输出结构：高层 subtask token、FAST 离散动作 token、flow matching 连续动作 chunk。RECAP 额外训练一个 distributional value function，用 episode success/failure 和 normalized remaining steps 预测状态价值，再用 value function 估计动作 advantage，将动作标成 `Advantage: positive / negative`。VLA 训练时把这个 indicator 作为文本条件插在 subtask 和 action 之间，使模型学习 $\pi(a|o,\ell,I)$。positive 和 negative 的 loss 形式相同，但它们对应不同动作数据分布；推理时固定输入 positive，从而抽取高 advantage 行为。整个方法避免了 PPO/TRPO 对 flow-matching action likelihood 和 ratio 的需求，同时又能利用 demonstration、失败 rollout 和 human correction 进行多轮真实机器人改进。`pi_*0.6.pdf`

更短的逻辑链条是：

```text
π0.6 VLA
  输入: 图像 + proprioception + task prompt + metadata
  输出: subtask text + FAST action tokens + continuous action chunk

        ↓

训练 value function
  success/failure + remaining steps
  → normalized value in (-1, 0)
  → 201-bin distributional value

        ↓

估计 advantage
  n-step return + future value - current value
  → threshold by ε_l
  → Advantage: positive / negative

        ↓

训练 π*0.6
  observation + task + subtask + Advantage indicator
  → discrete token likelihood + continuous flow matching loss

        ↓

推理
  固定输入 Advantage: positive
  → sample high-advantage conditional action distribution

        ↓

真实机器人迭代
  collect autonomous rollouts + failures + interventions
  → add to dataset D_l
  → retrain value and policy from pretrained checkpoint
  → repeat
```

所以这篇论文最核心的思想是：

> **RL 信号不是通过 PPO-style policy gradient 直接更新大 VLA，而是通过 `reward → value → advantage → indicator → conditional action distribution` 这条链路进入模型。**



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
