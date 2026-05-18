# OPD（On-Policy Distillation）知识笔记

## 0. 一句话理解 OPD

**OPD = 让 student 自己 rollout，然后 teacher 在 student 实际走到的 prefix/state 上给 token-level 指导。**

它不是简单地让 student 模仿 teacher 预先生成好的完整答案，而是：

```text
student 先自己生成一条 response；
teacher 在 student 的每一个 prefix 上预测 next-token distribution；
student 学 teacher 在这些 prefix 上的判断。
```

所以 OPD 结合了两类方法的优点：

```text
像 RL：数据来自 student 自己的 on-policy rollout。
像 KD：监督来自 teacher 的 dense token-level signal。
```

---

## 1. 为什么需要 OPD？

普通 SFT / KD 的训练状态通常来自 teacher 或人工数据：

```text
prompt + teacher_prefix
```

但推理时 student 看到的是自己生成的 prefix：

```text
prompt + student_prefix
```

这会导致 **train-test mismatch / exposure bias**。

举例：

```text
teacher 原本会走几何证明路线；
student 推理时可能一开始走了代数路线。
```

普通 KD 更多是在教 student：

```text
在 teacher 的 prefix 下应该怎么继续。
```

但它没有充分教 student：

```text
如果你自己已经走到了 student 的 prefix，应该怎么继续。
```

OPD 的核心价值就在于：

```text
student 自己走；
teacher 在 student 实际走到的地方指导它。
```

---

## 2. KL 散度为什么可以衡量两个分布的差？

KL 散度定义为：

$$
D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$```

其中：

```text
P = 真实分布 / teacher 分布 / 目标分布
Q = 近似分布 / student 分布
```

KL 的直觉是：

> 如果真实数据来自 P，但你用 Q 去解释这些数据，平均要多付出多少 surprise / 编码代价。

单个事件 i 的 surprise 是：

$$
-\log Q(i)
$$

如果 Q 认为某个 token 概率很高，那么这个 token 真的出现时，模型不惊讶；如果 Q 认为概率很低，但它真的出现了，模型就很惊讶。

用 Q 解释 P 的平均 surprise 是 cross entropy：

$$
H(P, Q) = -\sum_i P(i)\log Q(i)
$$

用 P 自己解释 P 的平均 surprise 是 entropy：

$$
H(P) = -\sum_i P(i)\log P(i)
$$

二者相减就是 KL：

$$
D_{KL}(P\|Q) = H(P,Q) - H(P)
$$

所以 KL 表示：

```text
真实分布是 P，
但你用 Q 来近似它时，
平均多付出的解释成本。
```

注意：KL 不是严格意义上的距离，因为它有方向：

$$
D_{KL}(P\|Q) \neq D_{KL}(Q\|P)
$$

---

## 3. Forward KL 和 Reverse KL

### 3.1 Forward KL

Forward KL 通常指：

$$
D_{KL}(\pi_T \| \pi_\theta)
$$

其中：

```text
π_T = teacher
π_θ = student
```

展开是：

$$
D_{KL}(\pi_T \| \pi_\theta)
= \sum_a \pi_T(a|s) \log \frac{\pi_T(a|s)}{\pi_\theta(a|s)}
$$

它的权重来自 teacher。

直觉：

```text
teacher 认为重要的 token，student 要覆盖。
```

所以 Forward KL 更偏 **mode-covering**。

---

### 3.2 Reverse KL

Reverse KL 是：

$$
D_{KL}(\pi_\theta \| \pi_T)
$$

展开是：

$$
D_{KL}(\pi_\theta \| \pi_T)
= \sum_a \pi_\theta(a|s) \log \frac{\pi_\theta(a|s)}{\pi_T(a|s)}
$$

它的权重来自 student。

直觉：

```text
student 自己想选的 token，teacher 是否认可。
```

所以 Reverse KL 更偏 **mode-seeking**。

可以写成期望形式：

$$
D_{KL}(\pi_\theta \| \pi_T)
= \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}
\left[
\log \pi_\theta(a|s) - \log \pi_T(a|s)
\right]
$$

这个形式很重要，因为 sampled-token reverse KL 就来自这里。

---

## 4. OPD 的基本流程

一个典型 OPD iteration 是：

```text
1. 从数据集中采样 prompt x。

2. student πθ 根据当前策略生成回答 y：
   y ~ πθ(. | x)

3. 对于 y 的每个 token 位置 t：
   state s_t = prompt + y_<t

4. teacher πT 在同一个 student prefix 上计算 next-token distribution：
   πT(. | s_t)

5. student 在这些 state 上匹配 teacher。
```

核心区别：

```text
普通 KD：teacher 写答案，student 模仿 teacher 的答案。
OPD：student 自己写答案，teacher 在 student 的 prefix 上指导 student。
```

---

## 5. input_ids、logits、prob 的关系

语言模型输入的是 token id，也就是词表 index。

例如：

```text
文本: "hello world"
input_ids: [15496, 995]
```

模型内部：

```text
input_ids
-> embedding lookup
-> transformer
-> logits
```

输出 logits 的 shape 通常是：

```text
[B, T, V]
```

其中：

```text
B = batch size
T = sequence length
V = vocab size
```

`logits[b, t, :]` 表示在位置 t，对下一个 token 的预测分布。

然后：

```python
log_probs = torch.log_softmax(logits, dim=-1)
```

得到每个 vocab token 的 log probability。

语言模型有 shift 关系：

```text
position t 的 logits 用来预测 token t+1。
```

所以训练时通常：

```python
log_probs = log_probs[:, :-1, :]
labels = input_ids[:, 1:]
```

---

## 6. Full-KL OPD

Full-KL OPD 的做法是：

```text
1. student 生成 response。
2. 得到 input_ids = prompt + student_response。
3. teacher 和 student 都在这条 input_ids 上 forward。
4. 对每个 response token 位置，比较完整 vocab distribution。
```

Forward KL loss：

$$
D_{KL}(\pi_T(\cdot|s_t) \| \pi_\theta(\cdot|s_t))
= \sum_{v \in V}
\pi_T(v|s_t)
\log
\frac{\pi_T(v|s_t)}{\pi_\theta(v|s_t)}
$$

伪代码：

```python
# student rollout
with torch.no_grad():
    response_ids = student.generate(
        prompt_ids,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=1024,
    )

input_ids = concat_prompt_response(prompt_ids, response_ids)
response_mask = build_response_mask(input_ids, prompt_ids)

# teacher forward
with torch.no_grad():
    teacher_logits = teacher(input_ids).logits
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

# student forward
student_logits = student(input_ids).logits
student_log_probs = torch.log_softmax(student_logits, dim=-1)

# shift
teacher_log_probs = teacher_log_probs[:, :-1, :]
student_log_probs = student_log_probs[:, :-1, :]
loss_mask = response_mask[:, 1:].float()

# forward KL: KL(teacher || student)
teacher_probs = teacher_log_probs.exp()
kl_per_pos = torch.sum(
    teacher_probs * (teacher_log_probs - student_log_probs),
    dim=-1,
)

loss = (kl_per_pos * loss_mask).sum() / loss_mask.sum()
loss.backward()
optimizer.step()
```

特点：

```text
优点：信号更稳定，因为每个位置都比较完整分布。
缺点：成本高，因为需要 [B, T, V] 的完整 vocab KL。
```

---

## 7. Sampled-token Reverse KL

Reverse KL 是：

$$
D_{KL}(\pi_\theta \| \pi_T)
= \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}
\left[
\log \pi_\theta(a|s) - \log \pi_T(a|s)
\right]
$$

因为它是对 student 分布取期望，所以可以从 student 里 sample token 来估计。

流程：

```text
1. student 在当前 prefix s 上 sample 一个 token a。
2. teacher 计算 log πT(a | s)。
3. student 计算 log πθ(a | s)。
4. 根据 teacher 是否认可这个 token 来更新 student。
```

直觉：

```text
student 自己选 token；
teacher 评价这个 token；
teacher 认可 -> 提高这个 token 概率；
teacher 不认可 -> 降低这个 token 概率。
```

---

## 8. 为什么 sample，而不是取 max prob？

Reverse KL 的数学形式本身是：

$$
a \sim \pi_\theta(\cdot|s)
$$

也就是从 student 分布里采样。

如果用 argmax：

```python
a = argmax(student_probs)
```

那只会训练 student 当前最自信的 token，而不是估计整个 student 分布下的期望。

问题包括：

```text
1. 只看 top-1 token，覆盖不到其他可能 token。
2. 探索少，容易过早 collapse 到某一种模式。
3. 不是真正的 reverse KL 期望估计。
```

例子：

```text
student:
A: 0.40
B: 0.35
C: 0.25

teacher:
A: 0.10
B: 0.80
C: 0.10
```

如果 argmax，student 永远选 A，只知道 A 不好。

如果 sample，有 35% 概率采到 B，teacher 会给 B 高 reward，student 就能更快提高 B 的概率。

当然，OPD 中也不能毫无限制地 sample，否则 student 可能采到坏 prefix，teacher 被带偏。

实践中常用：

```text
top-p sampling
top-k sampling
较低 temperature
max_new_tokens 限制
bad rollout 过滤
```

---

## 9. 正常 LLM 生成时也会 sample 吗？

会。很多 LLM 生成回答时确实会 sample，但通常不是从完整词表毫无限制地采样，而是加约束。

常见参数：

```python
do_sample=True
temperature=0.7
top_p=0.9
top_k=50
```

### top-k

只在概率最高的 k 个 token 中采样。

### top-p

只保留累计概率达到 p 的 token，再重新归一化采样。

### temperature

控制分布尖锐程度：

```text
低 temperature：更稳定、更保守。
高 temperature：更多样，但更容易发散。
```

所以采样确实可能带来坏 token，但通过 top-k/top-p/temperature 可以大幅降低风险。

---

## 10. Sampled-token Reverse KL 的 PG-style loss

从 reverse KL 的单样本形式看：

$$
\log \pi_\theta(a|s) - \log \pi_T(a|s)
$$

但如果直接把它当普通 loss 反传，会有问题。

因为 teacher logprob 是常数，直接最小化：

```python
student_logp - teacher_logp.detach()
```

会倾向于降低 student 对 sampled token 的 logprob，不管 teacher 是否喜欢它。

所以实际更合理的 PG-style 写法是把 teacher signal 放进 reward：

$$
r(s,a) = \log \pi_T(a|s) - \log \pi_{old}(a|s)
$$

然后：

$$
\mathcal{L} = -r(s,a) \log \pi_\theta(a|s)
$$

伪代码：

```python
# 1. rollout，用 old student 采样
with torch.no_grad():
    response_ids, old_student_chosen_logp = student.generate(
        prompt_ids,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_logprobs=True,
    )

input_ids = concat_prompt_response(prompt_ids, response_ids)
labels = input_ids[:, 1:]
loss_mask = build_response_loss_mask(input_ids, prompt_ids)[:, 1:]

# 2. teacher 给 sampled tokens 打分
with torch.no_grad():
    teacher_logits = teacher(input_ids).logits[:, :-1, :]
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

    teacher_chosen_logp = torch.gather(
        teacher_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

# 3. current student 重新 forward
student_logits = student(input_ids).logits[:, :-1, :]
student_log_probs = torch.log_softmax(student_logits, dim=-1)

current_student_chosen_logp = torch.gather(
    student_log_probs,
    dim=-1,
    index=labels.unsqueeze(-1),
).squeeze(-1)

# 4. reward 不回传梯度
with torch.no_grad():
    reward = teacher_chosen_logp - old_student_chosen_logp

# 5. PG-style OPD loss
loss_per_token = -reward.detach() * current_student_chosen_logp
loss = (loss_per_token * loss_mask).sum() / loss_mask.sum()

loss.backward()
optimizer.step()
```

---

## 11. 哪些量回传梯度？

在 PG-style sampled reverse KL 里，有三个 logprob：

```text
teacher_chosen_logp
old_student_chosen_logp
current_student_chosen_logp
```

### teacher_chosen_logp

teacher 对 student sampled token 的 logprob。

```text
不回传梯度。
```

原因：teacher 是 frozen teacher。

---

### old_student_chosen_logp

rollout 时 old student 对 sampled token 的 logprob。

```text
不回传梯度。
```

原因：它表示当初这个 token 是如何被旧策略 sample 出来的，类似 PPO 中的 old_logprob。

---

### current_student_chosen_logp

当前正在优化的 student 对同一个 token 的 logprob。

```text
回传梯度。
```

原因：我们要更新 student。

总结：

```text
teacher_chosen_logp: no grad
old_student_chosen_logp: no grad
reward: no grad
current_student_chosen_logp: has grad
```

---

## 12. loss = -reward * logp 的符号和梯度方向

loss 是：

$$
\mathcal{L} = -r \log \pi_\theta(a|s)
$$

其中：

```text
log πθ(a|s) <= 0
reward r 可以是正，也可以是负。
```

所以 loss 不一定永远是负的。

### reward > 0

说明 teacher 比 student 更认可这个 token。

最小化：

$$
-r \log \pi_\theta(a|s)
$$

会让：

```text
log πθ(a|s) 变大
πθ(a|s) 变大
```

也就是提高该 token 概率。

### reward < 0

说明 teacher 不认可 student sampled token。

最小化 loss 会让：

```text
log πθ(a|s) 变小
πθ(a|s) 变小
```

也就是降低该 token 概率。

所以这个 loss 的直觉是：

```text
teacher 认可 -> 提高概率；
teacher 不认可 -> 降低概率。
```

---

## 13. OPD 和 PPO 的关系

OPD 看起来像 PPO，是因为它也有：

```text
old policy rollout
current policy update
old_logprob / current_logprob
```

但 OPD 不是标准 PPO。

PPO：

```text
1. old policy rollout
2. reward model / environment 给 reward
3. 算 advantage
4. 用 ratio + clip objective 更新
```

OPD sampled reverse KL：

```text
1. student rollout
2. teacher 给 token-level logprob
3. 构造 token-level reward
4. 用 -reward * current_logprob 更新
```

PPO 有：

$$
ratio = \frac{\pi_{new}(a|s)}{\pi_{old}(a|s)}
$$

并使用 clipped objective。

OPD 通常没有 PPO 的 clipped ratio，更多是：

```text
on-policy data collection + teacher distillation。
```

---

## 14. Teacher 被 student prefix 带偏怎么办？

OPD 的风险是：

```text
teacher condition on student prefix。
```

如果 student 前缀很差，teacher 被迫在一个不自然、错误、混乱的上下文里预测 next token。

例如：

```text
题目：2x + 3 = 7

teacher 原本会写：
2x = 4, x = 2

student prefix：
We assume x = 100 because ...
```

teacher 看到这个 prefix 后，可能开始纠错、犹豫或者被带偏。

常见解决办法：

### 14.1 SFT warmup student

先让 student 通过 SFT / offline KD 学会基本格式和基本能力，再做 OPD。

```text
阶段 1：SFT / KD
阶段 2：OPD
```

---

### 14.2 限制 rollout 采样

使用：

```text
top-p sampling
low temperature
max_new_tokens
bad response filtering
```

避免 student 生成太离谱的 prefix。

---

### 14.3 Top-K / Top-P local support matching

不比较完整 vocab，也不只看 sampled token，而是在 teacher 的 top-k/top-p token 集合中做局部分布匹配。

这样比 sampled-token 更稳定，比 full KL 更省。

---

### 14.4 Drift detection

检测 teacher 和 student 的 top-k token 是否重叠。

如果重叠高：

```text
说明 prefix 还在 teacher 可指导区域。
```

如果重叠低：

```text
说明 student prefix 已经 drift，teacher signal 可能不可靠。
```

可以降低后续 loss 权重，或者直接截断。

---

### 14.5 Rollout truncation

如果后半段 prefix 已经明显坏掉，不要继续在后面训练。

```text
前 200 token 还可靠；
后面 1000 token 已经 drift；
那就只训练前 200 token。
```

---

### 14.6 混合 teacher forcing 和 student forcing

训练早期可以多用 teacher prefix，后期逐步增加 student prefix。

类似 scheduled sampling / DAgger。

---

## 15. Teacher 和 student 生成长度不一致怎么办？

OPD 通常不需要 teacher 生成完整答案。

训练长度由 student rollout 决定。

student 生成：

```text
y_student = [y1, y2, ..., yT]
```

teacher 只需要在每个 prefix 上预测 next token：

```text
s_t = prompt + y_student_<t
πT(. | s_t)
```

所以不存在必须让 teacher response 和 student response 对齐的问题。

### student 太短

如果 student 很早 EOS，训练信号会很短。

解决方法：

```text
min_new_tokens
EOS penalty
过滤过短 response
混合 SFT traces
```

### student 太长

teacher 仍然可以继续给 next-token distribution，但后面 prefix 可能越来越 drift。

解决方法：

```text
max_new_tokens
length penalty
drift detection
loss down-weight
rollout truncation
```

---

## 16. OPD 的几种常见形式

### 16.1 sample rollout + full KL

```text
student sample response；
teacher/student 在 student prefix 上比较完整 vocab distribution。
```

优点：稳定。

缺点：贵。

---

### 16.2 sample rollout + sampled-token reverse KL

```text
student sample response；
teacher 只评价 student 实际生成的 token；
用 PG-style loss 更新。
```

优点：省。

缺点：噪声大，不稳定。

---

### 16.3 sample rollout + top-k/top-p KL

```text
student sample response；
teacher 提供 top-k/top-p local support；
student 在局部 token 集合上匹配 teacher。
```

优点：比 full KL 省，比 sampled-token 稳。

这是实践上比较有吸引力的折中。

---

### 16.4 greedy rollout + full KL

```text
student greedy decode；
teacher/student 在 greedy prefix 上做 KL。
```

优点：稳定。

缺点：探索少，不是真正的 stochastic on-policy distribution。

---

## 17. OPD、SFT、KD、PPO 的对比

| 方法 | rollout 来自谁 | 监督来自谁 | 信号密度 | 主要问题 |
|---|---|---|---|---|
| SFT | dataset | label response | token-level | off-policy / exposure bias |
| KD | teacher | teacher logits / response | token-level | student prefix 不充分 |
| PPO/RLVR | student | reward / verifier | sparse 或 sequence-level | reward 稀疏、方差大 |
| OPD | student | teacher logits / logprobs | token-level | teacher 可能被 student prefix 带偏 |

---

## 18. 最核心的理解

OPD 不是：

```text
teacher 写答案，student 模仿答案。
```

而是：

```text
student 自己尝试；
teacher 看 student 走到了哪里；
teacher 在这个位置告诉 student 下一步应该更像什么；
student 在自己真实会访问的状态分布上变强。
```

一句话总结：

> OPD = on-policy rollout + teacher token-level distillation。

它的优点：

```text
减少 train-test mismatch；
提供 dense teacher signal；
能在 student 自己犯错后的状态上训练。
```

它的风险：

```text
student prefix 太差会带偏 teacher；
sampled-token loss 噪声大；
长链任务 teacher forward 成本高；
teacher-student reasoning pattern 不兼容时可能失败。
```

实践上较稳的 recipe：

```text
1. SFT / KD warmup student。
2. 用受限采样生成 student rollout。
3. 使用 full KL 或 top-k/top-p local KL，而不是一上来只用 sampled-token。
4. 检测 prefix drift，必要时截断或降权。
5. 对过短、重复、明显错误 response 做过滤。
```

---

## 19. 复习用超短版

```text
KL：
真实分布 P，用 Q 解释时多付出的平均 surprise。

Forward KL：
KL(teacher || student)，teacher 认为重要的 token，student 要覆盖。

Reverse KL：
KL(student || teacher)，student 自己想选的 token，teacher 是否认可。

OPD：
student 先 rollout，teacher 在 student prefix 上给 next-token 指导。

Full-KL OPD：
每个位置比较完整 vocab distribution。

Sampled-token OPD：
student sample token，teacher 给这个 token logprob，构造 reward 更新 student。

PG-style loss：
reward = teacher_logp - old_student_logp
loss = -reward.detach() * current_student_logp

梯度：
teacher_logp 不回传；
old_student_logp 不回传；
reward 不回传；
current_student_logp 回传。

teacher 被带偏：
用 SFT warmup、受限采样、top-k KL、drift detection、rollout truncation 缓解。
```

