---
title: ACT 补充理解：从 Action Chunking 到闭环稳定性
type: technical_note
topic: robot_imitation_learning
status: mature
updated: 2026-07-27
tags:
  - act
  - action-chunking
  - temporal-ensemble
  - cvae
  - imitation-learning
  - closed-loop-evaluation
  - libero
---

# ACT 补充理解：从 Action Chunking 到闭环稳定性

> 本文是现有 ACT 算法笔记的补充文档，不重复论文的完整模型介绍，而是结合实际实验，重点讨论：
>
> 1. Action Chunking 到底带来了什么；
> 2. Temporal Ensemble 为什么有效、但为什么不是 ACT 成功的必要条件；
> 3. `n_action_steps`、padding loss、动作差分损失和 CVAE 如何影响闭环行为；
> 4. 为什么 offline action error 很难预测 closed-loop success；
> 5. 我们对“未来事件缺少倒计时”的理解如何被后续实验修正。

---

## 1. ACT 最核心的变化：从单步动作预测转向短期轨迹预测

普通单步 Behavior Cloning 学习：

$$
\pi_\theta(o_t)=a_t
$$

ACT 学习：

$$
\pi_\theta(o_t)=
\hat A_t=
[\hat a_{t|t},\hat a_{t+1|t},\ldots,\hat a_{t+H-1|t}]
$$

其中：

- $o_t$ 是当前时刻 observation；
- $H$ 是 `chunk_size`；
- $\hat a_{t+h|t}$ 表示根据时刻 $t$ 的观测，对绝对时刻 $t+h$ 动作的预测。

ACT 的贡献并不只是使用 Transformer，而是把模仿学习目标从：

> 当前状态下，专家现在做什么？

扩展为：

> 当前状态下，专家接下来的一小段动作轨迹是什么？

这让模型能够利用 future action supervision 学习动作的时间结构，例如：

```text
接近物体
→ 闭合夹爪
→ 搬运
→ 对准目标
→ 下降
→ 松开
```

但必须注意：

> 预测出一段未来动作，不等于这段未来动作会被完整执行。

ACT 的训练目标、推理重规划方式和闭环控制策略必须一起理解。

---

## 2. ACT 输入的是当前时刻观测，不是历史视频序列

原始 ACT 使用当前时刻的多相机图像和当前机器人状态：

$$
o_t=
\{I_t^{(1)},I_t^{(2)},\ldots,I_t^{(N)},q_t\}
$$

基础 policy 不显式接收：

- 历史图像；
- 历史 proprioception；
- 上一轮预测的 action chunk；
- 上一轮计划执行到第几步；
- recurrent hidden state；
- episode 当前步数。

因此，ACT 仍然是一个 reactive policy：

$$
\hat A_t=f_\theta(o_t)
$$

只是它的输出从一个动作变成一段动作。

这意味着，当两个连续时刻的 observation 很接近时：

$$
o_{t+1}\approx o_t
$$

模型通常也会输出相似的 chunk：

$$
f_\theta(o_{t+1})\approx f_\theta(o_t)
$$

但这不代表 ACT 必然需要显式“倒计时”。只要执行动作能够持续推动真实状态前进，当前 observation 本身就可以作为任务进度的依据。

---

## 3. Action query 表示相对时间位置，不表示固定动作语义

ACT 中不同 action query 对应 chunk 内不同的相对 horizon：

```text
query 0  → 当前动作
query 1  → 未来第 1 步动作
...
query H-1 → 未来第 H-1 步动作
```

它们不是：

```text
query 0  → 接近
query 5  → 抓取
query 18 → 释放
```

训练样本来自轨迹上的滑动窗口：

$$
(o_t,\;a_t,a_{t+1},\ldots,a_{t+H-1})
$$

同一个“释放夹爪”事件，在不同训练窗口里可能位于：

- `chunk[0]`；
- `chunk[5]`；
- `chunk[18]`；
- 或不在当前 chunk 内。

因此不能把 rollout 中“release 总出现在第 17～19 步”解释为某几个 query 固定学会了释放。

更准确的理解是：

> 对某一类相似 observation，模型认为从当前状态出发，通常还需要十几步动作才进入释放阶段。

---

## 4. `chunk_size` 和 `n_action_steps` 是两个不同变量

### 4.1 `chunk_size`

决定模型每次预测多长的 future action sequence：

```text
chunk_size = 20
```

表示：

$$
o_t\rightarrow a_{t:t+19}
$$

### 4.2 `n_action_steps`

决定每次推理后，真正连续执行多少个动作：

```text
n_action_steps = 5
```

表示：

```text
预测 20 步
→ 执行前 5 步
→ 获取新 observation
→ 再预测 20 步
```

因此：

- `chunk_size` 控制训练和预测 horizon；
- `n_action_steps` 控制闭环重规划频率。

二者不能混为一谈。

`n_action_steps` 较大：

- 推理次数较少；
- 动作计划持续性更强；
- 但更开环，执行误差不能及时纠正。

`n_action_steps=1`：

- 每一步重新观察环境；
- 闭环反馈更及时；
- 但旧 chunk 会被立即覆盖；
- 如果不使用 Temporal Ensemble，相当于每次只使用新 chunk 的第一步。

---

## 5. “未来事件一直停留在第 17～19 步”的现象

实验中曾观察到：

```text
step 255: release horizon = 18
step 256: release horizon = 18
step 257: release horizon = 18
...
step 279: release horizon = 19
```

当时的解释是：

> ACT 没有保存上一轮计划，也没有显式倒计时，所以 release 永远停留在未来。

这个描述准确刻画了 rollout 表象，但后续实验说明，它不能被当作 A1 失败的根本原因。

更合理的因果链是：

```text
末端状态附近的监督或动作建模不充分
                ↓
执行前缀无法产生足够的状态进展
                ↓
机器人进入近似闭环固定点
                ↓
o[t+1] ≈ o[t]
                ↓
模型反复生成相似 action chunk
                ↓
release horizon 长期停留在 17～19
```

因此：

$$
\boxed{
\text{release horizon reset 更可能是停滞状态的症状，而不是 ACT 必然失败的根因}
}
$$

如果模型能够根据当前状态输出有效动作，使机器人持续从：

```text
靠近篮子
→ 到达篮子上方
→ 下降
→ 进入篮子
→ 松开
```

那么即使模型没有历史状态和显式计时器，release 也会自然进入 chunk 前部并最终被执行。

---

## 6. Temporal Ensemble：重要，但不是完成任务的必要条件

Temporal Ensemble 每一步重新预测一个 chunk，并将多个历史 chunk 对同一绝对时刻的动作预测进行融合。

对于绝对时刻 $\tau$，可以得到：

$$
\hat a_{\tau|\tau},
\hat a_{\tau|\tau-1},
\hat a_{\tau|\tau-2},
\ldots
$$

ensemble 后：

$$
a_\tau^{\text{exec}}
=
\frac{
\sum_i w_i\hat a_{\tau|\tau-i}
}{
\sum_i w_i
}
$$

它主要提供三种作用。

### 6.1 降低预测方差

同一绝对时刻由多个不同 observation 时刻进行预测，融合可以减小单次预测噪声。

### 6.2 缓解 chunk 切换不连续

不开 ensemble 时，不同重规划周期会直接切换到一个全新的 action chunk。Temporal Ensemble 让新旧计划平滑交接。

### 6.3 保留旧计划的影响

旧 chunk 中原本指向未来某一绝对时刻的动作，不会在下一次重规划时立即彻底消失。

因此 Temporal Ensemble 可以被理解为：

> 对多个无状态 action-chunk prediction 进行时间对齐后的在线集成。

但后续 A1-VAE 在无 Temporal Ensemble、`n_action_steps=5` 的条件下达到 19/20 成功，说明：

$$
\boxed{
\text{Temporal Ensemble 不是 ACT 正确完成 release 的必要条件}
}
$$

它是一个非常有效的闭环稳定化机制，但不能再被描述为“没有它，ACT 的未来事件就无法到达当前”。

---

## 7. CVAE 不是图像 tokenizer，而是训练期动作 latent

ACT 中的图像由 CNN backbone 编码。CVAE 处理的是机器人状态和目标 action chunk，而不是图像。

训练时，posterior encoder 建模：

$$
q_\phi(z\mid q_t,A_t)
$$

其中：

$$
A_t=a_{t:t+H-1}
$$

decoder 根据：

- 当前多视角图像；
- 当前 robot state；
- latent $z$；

预测 action chunk。

损失通常包含：

$$
L_{\text{ACT}}
=
L_{\text{action}}
+
\lambda_{\text{KL}}
D_{\text{KL}}
\left(
q_\phi(z\mid q_t,A_t)
\Vert
\mathcal N(0,I)
\right)
$$

推理时没有目标 action chunk，因此 posterior encoder 不再使用，通常令：

$$
z=0
$$

也就是使用先验均值进行确定性推理。

因此，A1-VAE 的闭环提升不能简单解释为：

> 推理时随机采样到了正确动作模式。

它的主要作用发生在训练阶段。

---

## 8. 为什么训练时使用 CVAE，推理时 `z=0` 仍可能明显变好

目前实验能确认的是：

> 打开标准 ACT CVAE 后，当前任务的闭环表现显著提高。

但不能仅凭现有结果严格证明提升来自“多模态动作建模”。可能机制包括以下几种。

### 8.1 分离相似 observation 下的不同 future chunk

确定性 ACT 学习：

$$
\hat A=f_\theta(o)
$$

当相似 observation 对应多个不同但合理的 future action chunk 时，L1 回归倾向于条件中位数：

$$
f^*(o)=\operatorname{Median}[A\mid o]
$$

这可能产生：

- 移动幅度偏小；
- 不同路径之间的折中；
- 释放时机被平均；
- 缺乏明确进展的 action prefix。

CVAE 训练时可以通过不同 latent 表示不同 action-chunk variation：

$$
A^{(1)}\leftrightarrow z_1,\qquad
A^{(2)}\leftrightarrow z_2
$$

从而减少不同轨迹模式之间的冲突梯度。

### 8.2 学出一个稳定的 canonical behavior

KL loss 将 posterior 拉向标准高斯。decoder 在 $z=0$ 附近可能形成一种稳定、典型的执行风格。

它不是简单的逐维动作平均，而可能是一条可执行的 canonical trajectory。

### 8.3 训练期 privileged information

posterior encoder 在训练时看到了目标 action chunk，因此 $z$ 可以向 decoder 提供：

- 当前 demonstration 的运动风格；
- 轨迹速度；
- 未来阶段；
- 释放时机相关信息。

这降低了训练期 action decoder 的拟合难度。

### 8.4 结构化正则化

latent sampling 和 KL 约束可能起到正则化作用，让 decoder 对小范围 latent 变化更加稳定，从而改善闭环鲁棒性。

现有实验尚不能区分上述机制，也没有必要把结果过度归因于某一种解释。

最安全的结论是：

$$
\boxed{
\text{当前任务中，标准 ACT CVAE 显著改善了 action-chunk 学习与闭环稳定性}
}
$$

---

## 9. 原始 padding loss 会降低 episode 末尾 observation 的总权重

原始固定 horizon loss 在 padding 位置清零后，对固定大小张量整体求 mean。

设每个 observation 的有效 future action 数量为 $H_b$，原始形式近似为：

$$
L_{\text{fixed}}
=
\frac{1}{BHD}
\sum_{b=1}^{B}
\sum_{h=1}^{H}
\sum_{d=1}^{D}
m_{b,h}
\left|
\hat a_{b,h,d}-a_{b,h,d}
\right|
$$

当 observation 位于 episode 中间：

$$
H_b=H
$$

当 observation 靠近 episode 末尾：

$$
H_b<H
$$

因此它对整个 batch loss 的总贡献正比于：

$$
\frac{H_b}{H}
$$

例如只剩一个有效动作时，该 observation 的总训练权重约为完整 horizon observation 的：

$$
\frac{1}{20}
$$

但 episode 末尾往往包含决定任务是否真正完成的行为：

- 进入目标区域；
- 松开夹爪；
- 松开后保持；
- 终止前稳定动作。

### ObsEqual 修改

对每个 observation 的有效动作先求 mean，再对 batch 求 mean：

$$
L_{\text{ObsEqual}}
=
\frac{1}{B}
\sum_b
\frac{
\sum_{h,d}
m_{b,h}
\left|
\hat a_{b,h,d}-a_{b,h,d}
\right|
}{
D\sum_hm_{b,h}
}
$$

这样每个 observation 的总监督权重一致。

A1-ObsEqual 的成功率从 15% 提高到 45%，而 offline 平均误差变化不大，支持：

> 末尾 observation 总监督过弱，会对关键终止行为造成明显的闭环影响。

---

## 10. 动作差分损失：优化轨迹变化，而不是简单平滑动作

原始 ACT 的 L1 loss 主要约束每个动作值：

$$
L_{\text{action}}
=
\sum_h
\left\|
\hat a_{t+h}-a_{t+h}
\right\|_1
$$

它不直接约束相邻动作的变化模式。

动作差分损失定义为：

$$
L_{\Delta a}
=
\left\|
(\hat a_{t+h+1}-\hat a_{t+h})
-
(a_{t+h+1}-a_{t+h})
\right\|_1
$$

总损失：

$$
L=
L_{\text{ACT}}
+
\beta L_{\Delta a}
$$

它不是简单惩罚：

$$
\|\hat a_{t+h+1}-\hat a_{t+h}\|
$$

因此不会把所有动作无条件压平，而是要求预测轨迹的一阶变化与 demonstration 一致。

实验中 A2 相比 A1：

- First-action MAE 降低 17.7%；
- Delta-action MAE 降低 9.0%；
- closed-loop Action Jerk 降低 20.3%；
- success rate 从 15% 提高到 70%。

这说明动作差分监督不仅改善了数值连续性，也显著改善了当前任务的闭环可执行性。

但 A2 的 teacher-forced jerk 并没有同步下降，说明：

> Offline 逐帧统计和 closed-loop 动力学结果不是同一件事。

---

## 11. 实验结果总表

### 11.1 A-Sanity

任务：

```text
LIBERO Object task 8:
pick up the chocolate pudding and place it in the basket
```

配置：

- 5 demonstrations；
- 859 frames；
- `chunk_size=20`；
- `n_action_steps=5`；
- `use_vae=false`；
- 20,000 updates。

结果：

| 指标 | 结果 |
|---|---:|
| First-action MAE | 0.02076 |
| Delta-action MAE | 0.00972 |
| Closed-loop success | 18/20（90%） |
| Closed-loop Action TV | 0.02195 |
| Closed-loop Action Jerk | 0.02210 |

这个结果说明：

- 5 条完整 demonstration 足以完成过拟合 sanity check；
- 图像 backbone 和 action decoder 能够在小数据上学习当前单任务；
- CVAE 不是完成 A-Sanity 的必要条件。

### 11.2 20-demonstration Development 实验

公共条件：

- 20 demonstrations；
- 3233 frames；
- batch size 8；
- 20,000 updates；
- seed 1000；
- `chunk_size=20`；
- Formal Clean Closed-loop 使用相同 20 个 init states。

| 配置 | 主要变化 | Success | Action TV | Action Jerk | Gripper toggles |
|---|---|---:|---:|---:|---:|
| A1 | deterministic ACT | 3/20（15%） | 0.02615 | 0.03017 | 14.80 |
| A2 | `+ 0.1 × delta loss` | 14/20（70%） | 0.02344 | 0.02403 | 14.65 |
| A2 + TE | `n_action_steps=1`, TE=0.01 | 18/20（90%） | 0.01517 | 0.00418 | 2.30 |
| A1-ObsEqual | 每个 observation 等权 | 9/20（45%） | 0.02442 | 0.02590 | 13.50 |
| A1-VAE | 标准 ACT CVAE | 19/20（95%） | 0.02307 | 0.02205 | 11.40 |

### 11.3 结果边界

这些结果必须附带以下限制：

- 当前只覆盖一个 LIBERO Object 任务；
- 每个训练配置只有一个训练 seed；
- A2 + Temporal Ensemble 同时改变了 `n_action_steps` 和 ensemble，缺少严格的同源 Replan control；
- A1-VAE 尚未完成对应 offline 指标；
- `gripper_toggles` 统计的是动作命令过零次数，不等同于物理夹爪完整开合次数；
- success episode 提前终止会影响 TV/Jerk 的序列长度分布；
- 不能把单任务实验直接外推到真实机器人或通用 ACT。

---

## 12. Offline 指标为什么不能代替 Closed-loop success

A1 与 A1-ObsEqual 的 offline 指标非常接近：

| 指标 | A1 | A1-ObsEqual |
|---|---:|---:|
| First-action MAE | 0.02564 | 0.02488 |
| Gripper MAE | 0.04787 | 0.04537 |
| Delta-action MAE | 0.01374 | 0.01392 |
| Teacher-forced Jerk | 0.01349 | 0.01353 |

但 closed-loop success 从 15% 提高到 45%。

原因是 teacher-forced offline evaluation 测量：

$$
o_t^{\text{expert}}
\rightarrow
\hat A_t
$$

模型始终在 expert trajectory observation 上预测。

Closed-loop rollout 测量：

$$
o_t^{\pi}
\rightarrow
a_t
\rightarrow
o_{t+1}^{\pi}
$$

其中 observation 分布由模型自己的历史动作决定。

小的动作误差可能产生：

- 接触位置偏移；
- 抓取失败；
- 进入未见状态；
- 释放时机偏移；
- 长时间振荡；
- 闭环固定点。

因此：

$$
\boxed{
\text{低 offline MAE 不等于高 closed-loop success}
}
$$

尤其对 release、gripper switch 等低频关键事件，整体 MAE 很容易被大量普通移动帧稀释。

---

## 13. 对我们之前几个猜想的最终修正

### 猜想一：ACT 缺少倒计时，因此未来 release 会永远停留在未来

**修正：过强。**

ACT 没有显式计划进度，但只要当前 observation 能反映状态进展，reactive policy 可以通过状态触发 release。

A1-VAE 在无 TE 条件下达到 95%，直接说明显式倒计时不是任务成功的必要条件。

### 猜想二：Temporal Ensemble 主要通过保留旧 release 计划解决失败

**修正：可能是部分机制，但不是唯一或必要机制。**

TE 确实可以保留旧预测的影响并减少方差，但 A1 的失败还受到：

- 末尾 observation 权重不足；
- action-chunk variation；
- 确定性回归；
- 动作连续性；
- 闭环状态进展不足；

等因素影响。

### 猜想三：纯 BC 无法解决“抓着不放”，必须加入 RL step penalty

**修正：不成立。**

A1-VAE 纯监督训练达到 19/20 成功，说明只要数据、loss 和模型表达足够，BC 可以学会正确 release。

RL/value-based optimization 仍然可以：

- 直接惩罚停滞；
- 优化完成时间；
- 学习 recovery；
- 优化闭环 outcome。

但不是当前任务解决 release 的必要条件。

### 猜想四：Action Difference Loss 只降低动作抖动

**修正：作用更广。**

A2 不仅降低 closed-loop jerk，也显著提高成功率。合理解释是它改善了整个 action prefix 的轨迹形状和局部可执行性，从而减少状态漂移和停滞。

### 猜想五：相似 observation 下存在多种 future chunk，确定性回归可能发生折中

**当前结果支持，但尚未严格证明。**

A1-VAE 的巨大提升与该解释一致，但还不能排除：

- 训练正则化；
- 额外参数；
- latent noise；
- privileged future-action information；
- 单 seed 波动。

---

## 14. ACT、Diffusion Policy、Flow Matching、World Model 和 RL 的正确关系

### 14.1 Diffusion Policy / Flow Matching

Diffusion Policy 和 π 系列的 Flow Matching 主要解决：

$$
p(A\mid o)
$$

的生成式建模问题。

它们比确定性 L1 回归更适合表达多峰 action distribution，但不自动带来：

- 跨控制周期记忆；
- 计划倒计时；
- previous chunk persistence；
- 任务阶段状态。

Flow Matching 中的：

$$
v_\theta(A^\tau,o,\tau)
$$

是 action generation space 中的向量场，不是机器人真实速度，也不是任务进度。

因此：

> 生成式 action policy 改善的是 action distribution 表达，不等于解决时间记忆。

### 14.2 World Model

World Model 学习：

$$
p(s_{t+1}\mid s_t,a_t)
$$

但仅预测未来状态并不自动解决 release timing。

真正有帮助的是：

$$
\text{World Model}
+
\text{Reward/Value}
+
\text{Planning}
$$

系统才能比较：

```text
继续夹紧
→ 状态停滞
→ 低长期价值
```

和：

```text
在合适位置释放
→ 任务成功
→ 高长期价值
```

同时，只有 recurrent latent state 才可能保存历史信息；一个仅根据当前图像编码状态的 world model 不必然具有时间记忆。

### 14.3 RL / Policy Gradient

Policy Gradient 可以通过：

- step penalty；
- success reward；
- failure penalty；
- progress reward；

让策略明确偏好更快完成、避免停滞。

但关键不是 PG 本身，而是 outcome-based objective。Q-learning、offline RL、value-guided decoding 和 model-based planning 也可以利用长期回报。

更准确的关系是：

```text
BC / VLA / ACT：
学习专家动作分布

Diffusion / Flow Matching：
更强地表达动作分布

Temporal Ensemble：
改善跨时刻 action prediction 的融合

World Model + Value：
评价动作未来后果

RL：
直接优化闭环长期回报
```

这些机制解决的是不同维度的问题，不能简单排列成“后一种自动解决前一种全部缺陷”。

---

## 15. 当前对 ACT 最深刻的理解

结合全部实验，ACT 应被理解为：

$$
\boxed{
\text{ACT}
=
\text{状态条件下的短期动作轨迹建模}
+
\text{闭环执行策略}
}
$$

它的表现不仅由 Transformer 决定，还由以下因素共同决定：

1. **Observation 是否足以反映任务状态；**
2. **Future action chunk 是否存在较大 variation；**
3. **Loss 是否公平覆盖关键 observation；**
4. **模型是否学到可执行的 action prefix；**
5. **`n_action_steps` 是否在闭环反馈和计划持续性之间取得平衡；**
6. **Temporal Ensemble 是否用于降低跨时刻预测不一致；**
7. **评测是否真正执行 closed-loop rollout。**

ACT 的主要优势是：

- future action supervision 提供轨迹级结构；
- action chunk 降低单步策略的短视性；
- CVAE 可以改善有变化的人类动作数据建模；
- 推理结构简单，适合高频控制。

ACT 的主要局限是：

- 基础输入仍以当前 observation 为主；
- 不显式建模环境 dynamics；
- 不直接优化任务长期回报；
- 对 loss 权重、action horizon 和执行方式敏感；
- offline 指标与 closed-loop success 可能严重脱节。

---

## 16. 最终结论

这组实验最重要的价值，不是证明某个单一模块“最好”，而是改变了对 ACT failure mode 的理解。

最初的解释是：

> ACT 没有未来事件倒计时，因此 release 可能永远停留在 future chunk 中。

现在更可信的解释是：

> 当关键末端 observation 监督不足、确定性 action-chunk 建模无法处理轨迹变化，或 action prefix 缺乏足够闭环进展时，策略可能进入状态近似不变的固定点。此时相似 observation 会重复产生相似 future chunk，于是表现为 release horizon 长期停留在 17～19。这个现象是学习和闭环控制失败的结果，而不是 ACT 必然存在、只能靠 Temporal Ensemble 或 RL 修复的结构性缺陷。

三个实验方向分别揭示了不同问题：

- **ObsEqual**：训练样本的总权重设计会决定关键末端行为是否被充分学习；
- **Delta-action loss**：轨迹变化监督会影响闭环可执行性，而不只是视觉上的平滑程度；
- **CVAE**：action-chunk variation 的建模对当前任务极其重要，即使推理使用 `z=0`，训练期 latent 仍可能显著改善最终策略。

因此，当前最完整的概括是：

$$
\boxed{
\text{Action Chunking 负责学习短期轨迹结构，}
}
$$

$$
\boxed{
\text{Loss 设计决定哪些状态和轨迹特征被真正重视，}
}
$$

$$
\boxed{
\text{CVAE 改善 future action variation 的表示，}
}
$$

$$
\boxed{
\text{Temporal Ensemble 改善跨时刻预测融合，}
}
$$

$$
\boxed{
\text{而 closed-loop rollout 才能判断这些设计是否真的形成了稳定策略。}
}
$$

---

## 17. 相关实验记录

- `2026-07-24-ACT-A-Sanity收尾与后续实验指南.md`
- `2026-07-25-ACT动作差分损失与公平评测.md`
- `2026-07-27-ACT末尾样本等权与VAE实验结论.md`
