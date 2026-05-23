# PPO

**User:** Anonymous  
**Created:** 4/27/2026 12:39:28  
**Updated:** 4/29/2026 17:17:45  
**Exported:** 4/29/2026 17:29:48  
**Link:** [https://chatgpt.com/c/69eee87b-85c0-839a-b31a-3c2719195cff](https://chatgpt.com/c/69eee87b-85c0-839a-b31a-3c2719195cff)  

## Response:
4/29/2026, 3:47:48 PM

下面是一份按照我们这轮讨论整理的 **PPO 综述**。我会特别保留你一路追问过的核心疑问：为什么不用直接反传 reward、为什么用 advantage、critic 为什么输出 $V(s)$ 而不是 $Q(s,a)$、GAE 怎么算、连续动作怎么 sample、为什么 PPO 是 near on-policy、为什么不用 epsilon-greedy 等。

---

# 1. PPO 的整体定位

PPO，全称 **Proximal Policy Optimization**，是一种 **near on-policy actor-critic 强化学习算法**。

它的核心目标是学习一个策略：

$$
\pi_\theta(a|s)
$$

也就是：

```text
在状态 s 下，应该以多大概率选择动作 a。
```

PPO 不是直接学习环境，也不是学习 reward 函数，而是学习一个 actor，让它在环境中交互后获得更高的长期累计 reward。

它的主线可以概括成：

```text
actor 输出动作分布
→ sample action
→ env.step(action)
→ 得到 reward / next_obs / done
→ critic 估计 V(s)
→ GAE 计算 advantage 和 return
→ advantage 更新 actor
→ return 更新 critic
→ ratio + clip 限制策略更新幅度
→ entropy bonus 保持探索
```

---

# 2. 为什么不能像监督学习一样直接对 reward backward？

你提出的一个很重要的问题是：

> 我们又不是在学习 env 或 reward 规则，那 env.step 不可导、reward 不连续为什么有关系？

关键点是：我们虽然不学习 env/reward，但是我们要优化 actor 参数 $\theta$，所以需要知道：

$$
\theta \text{ 怎么变，期望 reward 会怎么变}
$$

如果想直接用：

```python
loss = -reward
loss.backward()
```

那么需要一条可导路径：

```text
actor 参数 θ
→ action
→ env.step(action)
→ reward
```

也就是需要：

$$
\frac{\partial reward}{\partial \theta}
=
\frac{\partial reward}{\partial action}
\frac{\partial action}{\partial \theta}
$$

但在 RL 里通常有两个断点：

第一，动作通常是 sample 出来的，特别是离散动作：

```python
action = dist.sample()
```

这个采样操作不是普通可导函数。

第二，`env.step(action)` 通常不在 PyTorch 计算图里，也可能包含物理仿真、碰撞检测、离散规则、真实世界反馈等。

所以普通反向传播不成立。

PPO 使用 policy gradient 绕开这个问题。它不需要：

$$
\frac{\partial reward}{\partial action}
$$

而是利用：

$$
\nabla_\theta \log \pi_\theta(a_t|s_t)\hat A_t
$$

意思是：

```text
如果这个 action 的结果比预期好，就提高它以后被采样到的概率；
如果这个 action 的结果比预期差，就降低它以后被采样到的概率。
```

---

# 3. PPO 为什么基于 advantage 优化？

PPO 更新 actor 时不是简单地说“这个状态 reward 高就好”，而是看：

$$
A(s,a)
$$

也就是 advantage。

Advantage 的含义是：

```text
这个 action 比当前状态下的平均动作好多少。
```

数学上：

$$
A(s,a)=Q(s,a)-V(s)
$$

其中：

```text
Q(s,a)：状态 s 下执行动作 a 的长期价值
V(s)：状态 s 在当前策略下的平均长期价值
A(s,a)：动作 a 相对平均水平的好坏
```

如果：

```text
advantage > 0
```

说明这个动作比当前状态下的平均动作好，actor 应该提高它的概率。

如果：

```text
advantage < 0
```

说明这个动作比平均动作差，actor 应该降低它的概率。

这里你之前提到“正向 >1 / 负向 <1”，需要区分：

```text
advantage 的正负决定动作概率应该升还是降；
ratio > 1 / < 1 表示新策略相比旧策略对该动作的概率变大还是变小。
```

---

# 4. Actor 在 PPO 里做什么？

Actor 是策略网络，输出：

$$
\pi_\theta(a|s)
$$

也就是动作分布。

对于离散动作，比如：

```text
0: 直行
1: 左转
2: 右转
3: 刹车
```

actor 输出 logits，然后构造：

```python
dist = Categorical(logits=logits)
action = dist.sample()
log_prob = dist.log_prob(action)
```

对于连续动作，比如机器人控制：

```text
action = [速度, 转向角]
```

actor 通常输出高斯分布参数：

```text
mean = [0.6, 0.1]
std  = [0.2, 0.05]
```

然后：

```python
dist = Normal(mean, std)
action = dist.sample()
```

一次采样可能得到：

```text
[0.53, 0.13]
```

另一次可能得到：

```text
[0.71, 0.08]
```

训练时 sample 是为了探索；部署时通常不用 sample：

```text
离散动作：取概率最大的动作
连续动作：取 mean
```

---

# 5. 为什么用 log probability？

PPO/Policy Gradient 里会用：

$$
\log \pi_\theta(a|s)
$$

而不是直接用 probability。

原因有两个。

第一，理论推导需要 log trick：

$$
\nabla_\theta p_\theta(x)
=
p_\theta(x)\nabla_\theta \log p_\theta(x)
$$

它让我们可以把“对采样分布的梯度”转成“对 log probability 的梯度”。

第二，数值更稳定。

概率可能非常小：

```text
π(a|s) = 1e-20
```

直接除法或者乘法容易数值下溢。

log probability 则是：

```text
log(1e-20) ≈ -46
```

更稳定。

PPO 的 ratio 也通常用 log_prob 算：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta old}(a_t|s_t)}
$$

代码：

```python
ratio = torch.exp(new_log_prob - old_log_prob)
```

---
## 5.1 原始目标
我们想最大化期望回报：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

也就是：

```text
用当前策略 πθ 采样轨迹 τ，
希望轨迹的 reward 总和 R(τ) 越大越好。
```

写成积分形式：

$$
J(\theta)=\int p_\theta(\tau)R(\tau)d\tau
$$

这里：

```text
pθ(τ)：当前策略采样到轨迹 τ 的概率
R(τ)：这条轨迹的总回报
```

我们要对 $\theta$ 求导：

$$
\nabla_\theta J(\theta)
=
\nabla_\theta \int p_\theta(\tau)R(\tau)d\tau
$$

因为 reward 本身通常不能直接对 $\theta$ 求导，所以主要能动的是：

$$
p_\theta(\tau)
$$

也就是“采样到这条轨迹的概率”。

**RL 的目标函数本身就是一个关于轨迹分布的期望，而这个分布通常无法枚举、无法解析积分，所以自然会想到用采样估计期望。**

RL 的目标不是让某个样本的 loss 最小，而是：
$$
J(θ)=E_{τ∼pθ(τ)}[R(τ)]
$$
意思是：

```
如果我按照当前策略 πθ 去环境里跑很多次，平均能拿到多少 reward？
```

所以它天然就是一个期望问题。

写成积分：
$$
J(θ)=∫p_θ(τ)R(τ)dτ
$$
这里 τ 是完整轨迹：

τ=(s0,a0,r0,s1,a1,r1,… )

轨迹空间极其巨大，尤其连续状态、连续动作时根本不可能枚举。

所以只要看到：
$$
∫p_θ(τ)R(τ)dτ
$$
自然就会想到：

```
这个积分算不了，那就从 pθ 里 sample 一些 τ，用样本平均近似期望。
```

**监督学习的数据分布通常不依赖模型参数 θ，而 RL 的轨迹分布依赖策略参数 θ**
监督学习：
$$
L(θ)=E_{(x,y)∼p_{data}}[ℓ_θ(x,y)]
$$
这里 $$p_{data}$$​ 不依赖 θ

所以梯度是：
$$
∇_θL(θ)=E_{(x,y)∼p_{data}}[∇_θℓ_θ(x,y)]
$$
你可以直接对 loss 求导。

---

RL：
$$
J(θ)=E_{τ∼p_θ(τ)}[R(τ)]
$$
这里 pθ(τ)p_\theta(\tau)pθ​(τ) 依赖 θ\thetaθ。

因为策略变了，采样到的 action 变了，轨迹分布也变了。

所以梯度是：

∇θJ(θ)=∇θ∫pθ(τ)R(τ)dτ\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau)R(\tau)d\tau∇θ​J(θ)=∇θ​∫pθ​(τ)R(τ)dτ

变成：

∇θJ(θ)=∫∇θpθ(τ)R(τ)dτ\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau)R(\tau)d\tau∇θ​J(θ)=∫∇θ​pθ​(τ)R(τ)dτ

问题就在这里：**梯度作用在分布 pθp_\thetapθ​ 上，而不是直接作用在一个普通 supervised loss 上。**

这就是 RL 比监督学习麻烦的地方。
---

## 5.2. 问题：怎么对采样概率 $p_\theta(\tau)$ 求导？

直接求：

$$
\nabla_\theta p_\theta(\tau)
$$

不方便。

所以使用一个恒等式：
$$
\nabla_\theta p_\theta(x)=p_\theta(x)\nabla_\theta\log p_\theta(x)
$$
这个式子来自：

$$
\nabla_\theta \log p_\theta(x)
=
\frac{1}{p_\theta(x)}\nabla_\theta p_\theta(x)
$$

所以：

$$
\nabla_\theta p_\theta(x)
=
p_\theta(x)\nabla_\theta \log p_\theta(x)
$$

这就是 **log-derivative trick**，也叫 **score function trick**。

---

## 5.3. 套回目标函数

原来：

$$
\nabla_\theta J(\theta)
=
\int \nabla_\theta p_\theta(\tau)R(\tau)d\tau
$$

用 log trick：

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)
$$

得到：

$$
\nabla_\theta J(\theta)
=
\int p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)R(\tau)d\tau
$$

这就可以写成期望：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
[
\nabla_\theta \log p_\theta(\tau)R(\tau)
]
$$

这一步非常关键。

因为它把：

```text
对整个采样分布求导
```

变成了：

```text
从当前策略采样轨迹，然后计算 log probability 的梯度，再乘 reward
```

这就可以用 Monte Carlo 采样估计了。

---

## 5.4. 为什么最后变成 $\log \pi(a|s)$？

轨迹概率是：

$$
p_\theta(\tau)
=
p(s_0)
\prod_t
\pi_\theta(a_t|s_t)
p(s_{t+1}|s_t,a_t)
$$

其中：

```text
p(s0)：初始状态分布
πθ(a_t|s_t)：策略选择动作的概率
p(s_{t+1}|s_t,a_t)：环境转移概率
```

对 $\theta$ 来说，只有策略 $\pi_\theta$ 和 $\theta$ 有关。

环境转移 $p(s_{t+1}|s_t,a_t)$ 不由 actor 参数控制。

所以：

$$
\log p_\theta(\tau)
=
\log p(s_0)
+
\sum_t \log \pi_\theta(a_t|s_t)
+
\sum_t \log p(s_{t+1}|s_t,a_t)
$$

对 $\theta$ 求导后，只剩：

$$
\nabla_\theta \log p_\theta(\tau)
=
\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

所以 policy gradient 里自然出现：

$$
\nabla_\theta \log \pi_\theta(a_t|s_t)
$$

---

## 5.5. 直觉：log probability 在干什么？

假设某一步 action 得到了很好的结果，也就是：

```text
advantage > 0
```

那么我们希望：

```text
以后在这个 state 下，更容易采样到这个 action
```

也就是增大：

$$
\pi_\theta(a_t|s_t)
$$

如果这个 action 结果很差：

```text
advantage < 0
```

就降低：

$$
\pi_\theta(a_t|s_t)
$$

所以 actor 的更新本质是：

$$
\nabla_\theta \log \pi_\theta(a_t|s_t)\hat A_t
$$

它表示：

```text
沿着让这个 action 的 log probability 增大的方向更新；
更新力度和方向由 advantage 决定。
```

---

## 5.6. 为什么不用普通 probability？

不是完全不能用，而是 **直接用 probability 的形式很难变成可采样的无偏梯度估计**。

我们想求的是：

$$
\nabla_\theta \mathbb{E}_{a\sim\pi_\theta}[R(a)]
$$

展开：

$$
\nabla_\theta \sum_a \pi_\theta(a)R(a)
=
\sum_a \nabla_\theta \pi_\theta(a)R(a)
$$

如果动作空间很大或连续，你不可能枚举所有 action。

log trick 让它变成：

$$
\sum_a \pi_\theta(a)\nabla_\theta\log\pi_\theta(a)R(a)
$$

也就是：

$$
\mathbb{E}_{a\sim\pi_\theta}
[
\nabla_\theta\log\pi_\theta(a)R(a)
]
$$

这个就可以通过 sample 来估计：

```python
action = dist.sample()
log_prob = dist.log_prob(action)
loss = -(log_prob * advantage).mean()
```

所以 log probability 的核心作用是：

```text
把“对整个分布求和/积分”变成“对采样到的 action 计算 log_prob 梯度”。
```

理论推导需要 log probability，是因为：

```text
策略参数 θ 改变的是“采样动作/轨迹的概率”；
reward 本身通常不可导；
log trick 可以把 ∇p 转成 p∇logp；
这样梯度就能写成对当前策略采样的期望；
于是可以用 sample 出来的 action 的 log_prob 来训练 actor。
```

一句话：

```text
log probability 是 policy gradient 能在“采样动作 + 不可导环境 reward”条件下仍然求梯度的关键桥梁。
```
---


# 6. PPO 的核心 loss：clipped surrogate

普通 policy gradient 可以理解成最大化：

$$
\log \pi_\theta(a_t|s_t)\hat A_t
$$

但 PPO 不是直接这么做。PPO 会比较新旧策略对同一个 action 的概率变化：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta old}(a_t|s_t)}
$$

然后使用 clipped objective：

$$
L^{CLIP}
=
\mathbb{E}
[
\min(
r_t(\theta)\hat A_t,
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
)
]
$$

代码形式：

```python
ratio = torch.exp(new_log_prob - old_log_prob)

surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages

policy_loss = -torch.min(surr1, surr2).mean()
```

注意负号是因为 PyTorch optimizer 默认最小化 loss，而 PPO 理论上是最大化 objective。

clip 的作用不是简单说“防止模型坍缩”，更准确是：

```text
防止策略一次更新太大，导致新策略和采样数据对应的旧策略差异过大。
```

如果 $\epsilon=0.2$，ratio 通常被限制在：

```text
[0.8, 1.2]
```

这表示：

```text
一个动作的概率相对旧策略不能突然变太多。
```

---

# 7. Critic 在 PPO 里做什么？

Classic PPO 里的 critic 通常输出：

$$
V(s)
$$

不是直接输出 advantage，也不是输出 $Q(s,a)$。

你之前问过一个很关键的问题：

> 为什么 critic 不直接输出 advantage？

原因是 advantage 不是环境直接给的标签。环境给的是 reward、next_obs、done。

Advantage 是后处理出来的：

```text
reward + V(s) + V(s_next) + done
→ GAE
→ advantage
```

critic 的职责是估计状态价值：

$$
V(s_t)
$$

含义是：

```text
从状态 s_t 开始，按照当前策略继续走，未来大概能拿多少累计 reward。
```

它不是说“actor 选定动作之后 rollout 多步以后的 value”，而是每个状态本身的长期价值估计。

---

# 8. GAE 怎么计算？为什么不一定等 episode 结束？

GAE，全称 Generalized Advantage Estimation。

它使用 TD error：

$$
\delta_t=
r_t+\gamma V(s_{t+1})(1-d_t)-V(s_t)
$$

其中：

```text
r_t：当前 reward
γ：折扣因子
d_t：done，episode 是否结束
V(s_t)：critic 当前状态价值
V(s_{t+1})：critic 下一状态价值
```

如果：

```text
done = 1
```

说明 episode 结束了，后面没有未来价值，所以乘：

$$
1-d_t
$$

GAE 递推公式是：

$$
\hat A_t=
\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}
$$

它从后往前算。

---

## values、next_value、next_values 的关系

你之前对这里问得很细。

假设 PPO rollout 长度为 $T=5$：

```text
s0 --a0--> r0, s1
s1 --a1--> r1, s2
s2 --a2--> r2, s3
s3 --a3--> r3, s4
s4 --a4--> r4, s5
```

rollout 里保存：

```text
obs = [s0, s1, s2, s3, s4]
rewards = [r0, r1, r2, r3, r4]
actions = [a0, a1, a2, a3, a4]
```

critic 对每个 rollout 状态输出：

```text
values = [V(s0), V(s1), V(s2), V(s3), V(s4)]
```

rollout 结束后，对最后一个 next state 再估计：

```text
next_value = V(s5)
```

这是一个值，用来给最后一步 bootstrap。

有些代码为了并行计算，会构造：

```text
next_values = [V(s1), V(s2), V(s3), V(s4), V(s5)]
```

所以：

```text
values 长度是 T；
next_value 是 rollout 最后状态 s_T 的 V(s_T)，一个值；
next_values 是为了方便计算拼出来的长度 T 序列。
```

GAE 不一定要等完整 episode 结束。它只需要 rollout 一小段，然后用 critic 的 $V(s_T)$ 估计剩下的未来，这叫 bootstrap。

---

# 9. Critic 的 loss 里的 return 是哪里来的？

critic 的 loss 是：

$$
L_V=(V(s_t)-return_t)^2
$$

其中 `return_t` 不是单步 reward，而是长期回报 target。

在 PPO + GAE 里通常：

$$
return_t = \hat A_t + V(s_t)
$$

代码：

```python
advantages, returns = compute_gae(...)
value_loss = ((value_pred - returns) ** 2).mean()
```

所以 critic 学的是：

```text
让 V(s_t) 接近由真实 rollout reward + bootstrap value 构造出来的长期回报 target。
```

这里你也问过：

> bootstrap value 不也是 critic 自己估计出来的吗？会不会影响 critic 更新？

答案是：会。

bootstrap target 确实有 bias，因为它用了 critic 自己的估计：

$$
target_t=r_t+\gamma V(s_{t+1})
$$

如果 $V(s_{t+1})$ 估错，target 也会偏。

但它的好处是：

```text
不用等完整 episode 结束；
方差更低；
样本效率更高。
```

GAE 的 $\lambda$ 就是在 bias 和 variance 之间折中：

```text
λ = 0：更接近 1-step TD，bias 高，方差低
λ = 1：更接近 Monte Carlo，bias 低，方差高
常用 λ ≈ 0.95
```

---

# 10. 为什么 PPO critic 通常用 $V(s)$，不是 $Q(s,a)$？

你一直在追问这个点：

> Q 不是更端到端吗？它同时包含 action 和 value，为什么不用 Q？

这个直觉是对的：

```text
Q(s,a) 更具体，更接近直接决策。
```

因为它回答：

```text
在状态 s 下执行动作 a，未来能拿多少回报？
```

而 $V(s)$ 只回答：

```text
当前状态整体好不好？
```

但是 $Q(s,a)$ 更难学。

原因包括：

第一，$Q(s,a)$ 绑定 action，连续动作空间下动作是无限的。critic 需要拟合一个高维动作价值曲面：

$$
(s,a)\rightarrow Q
$$

比单纯：

$$
s\rightarrow V
$$

更复杂。

第二，Q-learning 类方法常有 overestimation 问题，因为 target 中有：

$$
\max_{a'} Q(s',a')
$$

如果 Q 估计有噪声，max 会倾向于选择被高估的动作，并把这个高估值写入 target。

第三，PPO 已经有 actor 负责动作选择，critic 不需要直接选动作，只需要提供 baseline。

所以 PPO 的 classic 设计是：

```text
actor：负责动作分布 π(a|s)
critic：负责状态价值 V(s)
GAE：用 reward + V(s) 算 advantage
```

这是一种分工：

```text
actor 学如何选动作；
critic 学状态整体价值；
advantage 连接两者。
```

---

# 11. Q 的高估问题和 PPO 的 value 估计错误有什么区别？

你问过：

> Q-learning 会偏向被高估的动作，PPO 的 value 也可能估错，这两个影响程度怎么比较？

两者都会受估计误差影响，但误差进入训练目标的方式不同。

DQN/Q-learning 的 target 是：

$$
y=r+\gamma \max_{a'}Q(s',a')
$$

这里的 `max` 会系统性偏向被高估的动作。

如果真实 Q 都是 10，但估计成：

```text
Q(a1)=9
Q(a2)=10
Q(a3)=13
```

max 会选 13，target 被抬高。这个偏差会通过 Bellman backup 继续传播。

PPO 里的 value 错误主要影响 advantage：

$$
\hat A_t \approx return_t - V(s_t)
$$

如果 critic 高估 $V(s_t)$，本来好动作可能被估成负 advantage，导致 actor 更新方向错。

所以 PPO 也会受影响。

但不同点是：

```text
Q-learning 的高估会通过 max 形成系统性乐观偏差；
PPO 的 value 错误更像是给 policy gradient 的权重加噪声或偏置，没有 max over actions 这一步。
```

PPO 通过这些方式缓解：

```text
advantage normalization
GAE λ
value loss clipping
ratio clipping
KL early stopping
entropy bonus
```

---

# 12. 为什么 $V(s)$ 通常比 $Q(s,a)$ 更平滑？

你问过：

> 当前策略下动作的平均价值为什么更平滑？

因为：

$$
V^\pi(s)=\mathbb{E}_{a\sim\pi(\cdot|s)}[Q^\pi(s,a)]
$$

离散动作下：

$$
V^\pi(s)=\sum_a \pi(a|s)Q^\pi(s,a)
$$

也就是说，$V(s)$ 是对动作维度的加权平均。

假设：

```text
Q(s,a1)=0
Q(s,a2)=10
Q(s,a3)=20
```

策略：

```text
π=[0.3,0.4,0.3]
```

那么：

```text
V(s)=0*0.3 + 10*0.4 + 20*0.3 = 10
```

如果策略略微变成：

```text
π=[0.25,0.45,0.30]
```

则：

```text
V(s)=10.5
```

变化比较平滑。

而 $Q(s,a)$ 要刻画每个具体动作的价值，尤其连续动作下，需要区分：

```text
方向盘角度 0.10 和 0.11 的长期后果差异
速度 0.50 和 0.55 的长期后果差异
```

这通常更尖锐、更难拟合。

当然，这不是绝对定理。环境极端不连续、策略突变时，$V(s)$ 也会剧烈变化。但在 PPO 这种通过 clip/KL 控制策略小步更新的算法里，$V(s)$ 作为平均价值通常更稳定。

---

# 13. PPO 为什么不需要 epsilon-greedy？

epsilon-greedy 常见于 DQN：

```text
以 1-epsilon 概率选 Q 最大动作
以 epsilon 概率随机动作
```

但 PPO 是 stochastic policy，本身输出动作分布：

$$
\pi(a|s)
$$

训练时：

```python
action = dist.sample()
```

这本身就是探索。

所以 PPO 通常不需要 epsilon-greedy。

但这不代表没有 greedy 行为。部署时可以 deterministic：

```text
离散动作：取 argmax
连续动作：取 mean
```

训练时的探索还会通过 entropy bonus 来增强。

---

# 14. Entropy bonus 是什么？连续动作下怎么算？

PPO 常用总 loss：

```python
loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
```

这里是 entropy，不是 cross entropy。

Entropy 衡量策略分布有多分散。
如果策略是离散分布：

```
π(a|s) = [p1, p2, p3, ..., pn]
```

那么 entropy 是：
$$
H(π)=−a∑​π(a∣s)logπ(a∣s)
$$
比如动作有 4 个：

```
直行: 0.7左转: 0.1右转: 0.1刹车: 0.1
```

entropy 是：
$$
H=−(0.7log⁡0.7+0.1log⁡0.1+0.1log⁡0.1+0.1log⁡0.1)
$$
离散动作下，如果策略是：

```text
直行: 0.99
左转: 0.003
右转: 0.003
刹车: 0.004
```

entropy 很低，说明策略太确定，探索少。

如果策略是：

```text
直行: 0.35
左转: 0.25
右转: 0.20
刹车: 0.20
```

entropy 更高，探索更多。

连续动作下，假设策略是高斯：

$$
\pi(a|s)=\mathcal{N}(\mu,\sigma)
$$

一维高斯 entropy 是：

$$
H=\frac{1}{2}\log(2\pi e\sigma^2)
$$

多维独立高斯就是各维 entropy 相加。

PyTorch 中：

```python
base_dist = Normal(mean, std)
dist = Independent(base_dist, 1)

entropy = dist.entropy().mean()
```

std 越大，entropy 越大，探索越强；std 越小，entropy 越小，策略越确定。

---

# 15. PPO 为什么是 near on-policy？

PPO 收集 rollout 时用的是当前策略或旧策略：

$$
\pi_{\theta old}
$$

然后用这批数据训练当前策略：

$$
\pi_\theta
$$

因为训练过程中 $\theta$ 会变化，所以 PPO 用 ratio 来修正新旧策略差异：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta old}(a_t|s_t)}
$$

并用 clip 限制差异。

所以 PPO 严格说是 **near on-policy**。

它的流程是：

```text
1. 用当前 actor rollout 一批数据
2. 固定这批 rollout buffer
3. 用它训练多个 epoch
4. 丢掉这批数据
5. 用更新后的 actor 重新 rollout 新数据
```

注意不是：

```text
对同一个 batch 反复 rollout
```

而是：

```text
rollout 一次，优化多次。
```

这和 DQN/SAC/TD3 的 replay buffer 不同。

DQN/SAC/TD3 是 off-policy，可以长期复用旧数据。

PPO 的旧数据很快过期，因为它要求数据来自当前或近当前策略。

---

# 16. PPO 与 DQN / SAC 的关系

DQN 学的是：

$$
Q(s,a)
$$

loss 是：

$$
L=
\left(
Q_\theta(s,a)
-
[
r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a')
]
\right)^2
$$

DQN 没有显式 actor，它靠：

$$
a=\arg\max_a Q(s,a)
$$

来选动作。

所以 DQN 必须学 Q。

PPO 有 actor，所以 critic 不需要负责选动作，只需要估计 $V(s)$ 作为 baseline。

SAC/TD3/DDPG 这类算法也有 actor，但它们是 off-policy actor-critic，critic 通常学：

$$
Q(s,a)
$$

因为 actor 要通过最大化 Q 来更新。

所以不是说 $Q$ 不好，而是：

```text
Q 更具体、更决策相关，但更难学；
V 更粗、更平滑，但作为 PPO 的 baseline 已经足够。
```

不同算法选择不同组合：

```text
DQN：Q，没有显式 actor
PPO：actor + V critic
SAC：actor + Q critic
TD3/DDPG：actor + Q critic
```

---

# 17. PPO 的最终简明流程

可以把 PPO 记成下面这条链：

```text
初始化 actor 和 critic

for each iteration:

    1. rollout
       用当前 actor 和环境交互 T 步
       保存 obs, action, reward, done, old_log_prob, value

    2. bootstrap
       rollout 结束后，用 critic(last_obs) 得到 next_value

    3. compute GAE
       用 rewards, dones, values, next_value
       计算 advantages 和 returns

    4. update actor-critic
       多个 epoch 重复优化同一批 rollout 数据：

       new_log_prob = actor(obs).log_prob(action)
       ratio = exp(new_log_prob - old_log_prob)

       policy_loss = clipped PPO loss
       value_loss = (critic(obs) - returns)^2
       entropy_bonus = entropy(actor(obs))

       total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    5. 丢弃 rollout buffer
       用更新后的 actor 重新收集新 rollout
```

---

# 18. 你这轮问题中最重要的结论

最后把你一路问出来的关键点浓缩一下：

```text
1. PPO 不要求 env/reward 可导。
   它不通过 reward.backward() 更新 actor，而是通过 log_prob * advantage 更新动作概率。

2. advantage 是相对量。
   它不表示状态绝对好坏，而表示某个动作比该状态平均动作好多少。

3. 绝对好坏来自 reward。
   critic 学 V(s)，return target 来自真实 reward + bootstrap value。

4. GAE 不是网络。
   critic 输出 V(s)，GAE 用 rewards / values / dones / next_value 递推算 advantage。

5. critic 在 PPO 中通常输出 V(s)，不是 Q(s,a)。
   因为 actor 已经负责动作选择，critic 只需要 baseline。
   V 更平滑、更容易拟合。

6. Q 不是不好。
   Q 更端到端、更动作相关，但连续动作下更难学，也可能有 overestimation。
   SAC/TD3/DDPG 等算法仍然使用 Q critic。

7. PPO 不用 epsilon-greedy。
   训练时靠 stochastic policy sampling 和 entropy bonus 探索。
   部署时可以取 argmax 或 mean。

8. PPO 是 near on-policy。
   它用当前/近当前策略收集 rollout，优化几轮后丢掉，不像 DQN/SAC 那样长期复用 replay buffer。
```

一句话总结：

```text
PPO = stochastic actor + V critic + GAE advantage + clipped policy ratio + entropy exploration + near-on-policy rollout training。
```

## Prompt:
2026/5/1 17:22:54

1. 为啥 从当前策略采样轨迹，然后计算 log probability 的梯度，再乘 reward 的形式可以用蒙特卡洛采样估计，这样就可以求导了吗？ 没有看懂\nabla_\theta \log \pi_\theta(a_t|s_t)\hat A_t 为什么可以直接求导 2.我们要求的不是当前策略下最高 reward 的 action 吗，为什么变成\nabla_\theta \mathbb{E}_{a\sim\pi_\theta}[R(a)]了？是说按照当前策略选的 action 的 reward 期望？但是当前也只能选一个 action，如何根据这个 action 算出期望？ 是因为我们用采样 action 的概率来参与计算吗？

## Response:
2026/5/1 17:22:56

这两个问题其实是在问同一件事：

> RL 里我们只看到 sample 出来的 action 和 reward，为什么还能优化整个策略分布？

核心答案是：**我们不是用一次 action 精确算出期望，而是用很多次采样出来的 action，去估计期望梯度。**

---

# 1. 为什么这个形式可以用 Monte Carlo 估计？

先看一个普通期望。

假设有一个随机变量 $x$，它服从分布 $p(x)$。我们想算：

$$
\mathbb{E}_{x\sim p}[f(x)]
$$

如果你不能解析求这个期望，可以采样很多次：

```text
x1, x2, x3, ..., xN
```

然后用平均值近似：

$$
\mathbb{E}_{x\sim p}[f(x)]
\approx
\frac{1}{N}\sum_{i=1}^{N} f(x_i)
$$

这就是 Monte Carlo 估计。

---

Policy Gradient 推导后得到的是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
[
\nabla_\theta \log p_\theta(\tau)R(\tau)
]
$$

这也是一个期望。

所以同样可以采样很多条轨迹：

```text
τ1, τ2, τ3, ..., τN
```

然后估计：

$$
\nabla_\theta J(\theta)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
\nabla_\theta \log p_\theta(\tau_i)R(\tau_i)
$$

进一步展开到每一步 action，就是：

$$
\nabla_\theta J(\theta)
\approx
\frac{1}{N}
\sum_{i,t}
\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\hat A_{i,t}
$$

所以它可以 Monte Carlo 估计的原因是：

```text
推导以后，它变成了一个“对当前策略采样轨迹的期望”。
期望就可以用样本平均近似。
```

---

# 2. $\nabla_\theta \log \pi_\theta(a_t|s_t)\hat A_t$ 为什么可以直接求导？

因为这里真正需要反传的是：

$$
\log \pi_\theta(a_t|s_t)
$$

这个东西是 actor 网络直接输出的分布概率，所以在 PyTorch 计算图里。

例如离散动作：

```python
logits = actor(obs)
dist = Categorical(logits=logits)

action = dist.sample()
log_prob = dist.log_prob(action)

loss = -(log_prob * advantage.detach()).mean()
loss.backward()
```

这里：

```text
logits = actor(obs)
log_prob = log πθ(action | obs)
```

`log_prob` 是由 actor 网络参数 $\theta$ 计算出来的，因此可以对 $\theta$ 求导。

但注意：

```text
action 本身不需要可导
reward / advantage 也不需要可导
```

它们在 loss 里只是权重。

也就是说：

$$
\hat A_t
$$

被当成一个常数，类似监督学习里的 label 或 sample weight。

所以：

$$
\nabla_\theta[
\log \pi_\theta(a_t|s_t)\hat A_t]
=
\hat A_t
\nabla_\theta
\log \pi_\theta(a_t|s_t)
$$

只要 $\log \pi_\theta(a_t|s_t)$ 可导，就能反传。

---

## 一个很具体的离散例子

假设 actor 对某个状态输出：

```text
π(left | s) = 0.7
π(right | s) = 0.3
```

这次 sample 出来：

```text
action = left
```

环境返回结果不错：

```text
advantage = +2
```

那么 loss 是：

$$
L=-\log \pi_\theta(left|s)\cdot 2
$$

优化器最小化这个 loss。

为了让 loss 变小，需要让：

$$
\log \pi_\theta(left|s)
$$

变大。

所以：

```text
left 这个动作的概率会被提高。
```

如果 advantage 是负的：

$$
L=-\log \pi_\theta(left|s)\cdot (-2)
=
2\log \pi_\theta(left|s)
$$

最小化它，就会让：

$$
\log \pi_\theta(left|s)
$$

变小。

所以：

```text
left 这个动作的概率会被降低。
```

这就是为什么它能工作。

---

# 3. 我们要求的不是最高 reward 的 action 吗？

最终目标确实是：

```text
学到一个策略，让它倾向于选择高回报 action。
```

但训练目标不能直接写成：

$$
\arg\max_a R(a)
$$

原因有几个。

第一，RL 不知道每个 action 的 reward。你通常只执行了一个 action，只看到了这个 action 的结果。

第二，很多任务不是一步 reward，而是长期回报。一个 action 当下 reward 低，但可能带来后续高 reward。

第三，策略是概率分布，训练时需要探索。如果一开始就只选当前看起来最好的 action，很容易陷入局部最优。

所以 policy gradient 优化的是：

$$
J(\theta)=\mathbb{E}_{a\sim\pi_\theta}[R(a)]
$$

更完整是轨迹版本：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

它的意思是：

```text
如果我按照当前策略不断采样 action，
平均能拿到多少 reward？
```

我们希望这个平均值变大。

---

# 4. 当前只能选一个 action，怎么得到期望？

单次确实得不到精确期望。

一次交互只能得到：

```text
在状态 s 下，sample 了 action a，得到一个 reward/advantage。
```

这只是一个样本。

但训练会进行很多次：

```text
第 1 次：s1 采样 a1，得到 A1
第 2 次：s2 采样 a2，得到 A2
第 3 次：s3 采样 a3，得到 A3
...
```

用这些样本平均，就近似期望。

这就是 Monte Carlo。

PPO 里的 rollout buffer 正是用来收集很多这样的样本：

```text
obs_t, action_t, reward_t, value_t, log_prob_t, done_t
```

然后用一个 batch 来估计期望梯度：

$$
\frac{1}{B}
\sum_t
\nabla_\theta
\log\pi_\theta(a_t|s_t)
\hat A_t
$$

所以不是用一个 action 算出完整期望，而是：

```text
一个 action 提供一个无偏/近似无偏的梯度样本；
很多 action 的平均近似真实期望梯度。
```

---

# 5. 为什么不是直接找当前最高 reward 的 action？

因为很多时候你不知道“最高 reward 的 action”是什么。

比如状态 $s$ 下有四个动作：

```text
left
right
forward
brake
```

当前策略可能采样到了 `left`，得到了不错的结果。

但你不知道：

```text
right 会不会更好？
forward 会不会更好？
brake 会不会更好？
```

你只能逐步探索。

PPO 的做法不是立刻断言：

```text
left 就是最优动作
```

而是温和地更新：

```text
left 这次表现比预期好 → left 概率提高一点
left 这次表现比预期差 → left 概率降低一点
```

随着采样越来越多，好的 action 会越来越常被提高概率，差的 action 会越来越常被降低概率。

最终策略会逐渐集中到高回报动作附近。

---

# 6. 用采样概率参与计算是什么意思？

是的，关键就在这里。

action 是从策略分布里 sample 出来的：

$$
a\sim\pi_\theta(a|s)
$$

所以这个 action 出现的概率本身受 $\theta$ 控制。

如果某个 action 经常带来正 advantage，那么 policy gradient 会不断提高它的 log probability。

如果某个 action 经常带来负 advantage，就会降低它的 log probability。

于是整体期望 reward 变大。

直觉上：

```text
reward/advantage 告诉我们：这次 action 好不好。
log_prob 告诉优化器：怎么调整网络参数，才能让这个 action 以后更常/更少出现。
```

这两个乘起来，就是 actor 的学习信号。

---

# 7. 一个最小例子

假设只有一个状态，两个动作：

```text
A 和 B
```

真实 reward 是：

```text
R(A)=10
R(B)=0
```

当前策略：

```text
π(A)=0.5
π(B)=0.5
```

期望 reward 是：

$$
J=0.5\times10+0.5\times0=5
$$

训练时每次只能 sample 一个动作。

如果 sample 到 A：

```text
reward = 10
```

更新会提高：

$$
\log\pi(A)
$$

于是 $\pi(A)$ 变大。

如果 sample 到 B：

```text
reward = 0
```

更新对 B 的提高较少，或者如果用 baseline/advantage，B 可能是负 advantage，$\pi(B)$ 会下降。

多次采样后：

```text
π(A) 越来越大
π(B) 越来越小
```

期望 reward 从 5 提高到接近 10。

所以我们没有直接知道所有 action 的 reward，但通过采样和概率更新，逐渐逼近高 reward action。

---

# 8. 总结一句话

$$
\nabla_\theta \log \pi_\theta(a_t|s_t)\hat A_t
$$

能求导，是因为：

```text
log πθ(a|s) 是 actor 网络输出的可导量；
action/reward/advantage 不需要可导，只作为样本和权重。
```

它能优化期望 reward，是因为：

```text
这个形式是 ∇θ E[R] 经过 log trick 推出来的；
每次采样给出一个梯度估计；
很多采样平均就是 Monte Carlo 估计；
最终会提高高 advantage action 的概率，降低低 advantage action 的概率。
```

## Prompt:
2026/5/1 17:48:14

为什么
$$
\nabla_\theta J(\theta)
=
\int \nabla_\theta p_\theta(\tau)R(\tau)d\tau 
$$这样不能采样很多条轨迹同样算平均求期望？

## Response:
2026/5/1 17:51:48

这个问题问得非常关键。答案是：

**不能直接对**

$$
\int \nabla_\theta p_\theta(\tau)R(\tau)d\tau
$$

**用从 $p_\theta(\tau)$ 采样出来的轨迹做普通平均，因为它还不是一个“关于 $p_\theta$ 的期望形式”。**

---

# 1. Monte Carlo 采样估计要求什么形式？

如果你有：

$$
\mathbb{E}_{x\sim p(x)}[f(x)]
=
\int p(x)f(x)dx
$$

那么你可以从 $p(x)$ 采样：

```text
x1, x2, ..., xN
```

然后估计：

$$
\mathbb{E}_{x\sim p}[f(x)]
\approx
\frac{1}{N}\sum_{i=1}^N f(x_i)
$$

关键是积分里必须是：

$$
p(x)f(x)
$$

也就是说，采样分布 $p(x)$ 必须显式作为权重出现在积分里。

---

# 2. 但原式不是这个形式

原式是：

$$
\nabla_\theta J(\theta)
=
\int \nabla_\theta p_\theta(\tau)R(\tau)d\tau
$$

这里积分里是：

$$
\nabla_\theta p_\theta(\tau)R(\tau)
$$

不是：

$$
p_\theta(\tau) f(\tau)
$$

所以你不能直接说：

```text
从 pθ(τ) 采样很多轨迹，然后平均 ∇pθ(τ)R(τ)
```

因为那样估计的是：

$$
\mathbb{E}_{\tau\sim p_\theta}
[
\nabla_\theta p_\theta(\tau)R(\tau)
]
$$

也就是：

$$
\int p_\theta(\tau)\nabla_\theta p_\theta(\tau)R(\tau)d\tau
$$

注意多了一个：

$$
p_\theta(\tau)
$$

这就不是原来的东西了。

---

# 3. log trick 正是在补这个问题

我们希望把：

$$
\int \nabla_\theta p_\theta(\tau)R(\tau)d\tau
$$

改写成：

$$
\int p_\theta(\tau) f(\tau)d\tau
$$

这样才能从 $p_\theta$ 采样。

使用恒等式：

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)
$$

于是：

$$
\int \nabla_\theta p_\theta(\tau)R(\tau)d\tau
=
\int p_\theta(\tau)
\nabla_\theta \log p_\theta(\tau)
R(\tau)d\tau
$$

这就变成了：

$$
\mathbb{E}_{\tau\sim p_\theta}
[
\nabla_\theta \log p_\theta(\tau)R(\tau)
]
$$

现在才能用采样平均：

$$
\nabla_\theta J(\theta)
\approx
\frac{1}{N}\sum_i
\nabla_\theta \log p_\theta(\tau_i)R(\tau_i)
$$

---

# 4. 一个离散小例子

假设只有两条轨迹：

```text
τ1, τ2
```

概率分别是：

$$
p_\theta(\tau_1)=0.8
$$

$$
p_\theta(\tau_2)=0.2
$$

目标梯度原式是：

$$
\nabla J
=
\nabla p(\tau_1)R(\tau_1)
+
\nabla p(\tau_2)R(\tau_2)
$$

如果你从 $p$ 采样轨迹，然后直接平均：

$$
\nabla p(\tau)R(\tau)
$$

那么期望是：

$$
0.8\nabla p(\tau_1)R(\tau_1)
+
0.2\nabla p(\tau_2)R(\tau_2)
$$

这和原式：

$$
\nabla p(\tau_1)R(\tau_1)
+
\nabla p(\tau_2)R(\tau_2)
$$

不一样。

你看，多出来了采样概率权重。

---

# 5. 用 log trick 后就对了

因为：

$$
\nabla p(\tau)
=
p(\tau)\nabla \log p(\tau)
$$

所以原式变成：

$$
\nabla J
=
p(\tau_1)\nabla\log p(\tau_1)R(\tau_1)
+
p(\tau_2)\nabla\log p(\tau_2)R(\tau_2)
$$

也就是：

$$
\mathbb{E}_{\tau\sim p}
[
\nabla\log p(\tau)R(\tau)
]
$$

这时从 $p$ 采样再平均：

$$
\nabla\log p(\tau_i)R(\tau_i)
$$

才是对的。

---

# 6. 还有一个更实际的问题：$\nabla p_\theta(\tau)$ 很难算

轨迹概率是：

$$
p_\theta(\tau)
=
p(s_0)
\prod_t
\pi_\theta(a_t|s_t)
p(s_{t+1}|s_t,a_t)
$$

如果你直接算：

$$
\nabla_\theta p_\theta(\tau)
$$

要对一长串连乘求导，数值会很小，也不稳定。

但 log 后：

$$
\log p_\theta(\tau)
=
\log p(s_0)
+
\sum_t \log \pi_\theta(a_t|s_t)
+
\sum_t \log p(s_{t+1}|s_t,a_t)
$$

对 $\theta$ 求导时，环境项消失：

$$
\nabla_\theta \log p_\theta(\tau)
=
\sum_t
\nabla_\theta \log \pi_\theta(a_t|s_t)
$$

所以最后只需要 actor 的：

$$
\log \pi_\theta(a_t|s_t)
$$

这就是 PyTorch 里能直接算的 `log_prob`。

---

# 7. 最核心一句话

你不能直接采样估计：

$$
\int \nabla p_\theta(\tau)R(\tau)d\tau
$$

因为它不是：

$$
\int p_\theta(\tau)f(\tau)d\tau
$$

这种期望形式。

log trick 的作用就是把它改写成：

$$
\int p_\theta(\tau)
[
\nabla \log p_\theta(\tau)R(\tau)
]
d\tau
$$

这样才能从当前策略采样轨迹，然后用样本平均估计梯度。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)