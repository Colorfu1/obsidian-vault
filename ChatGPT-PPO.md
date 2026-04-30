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



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)