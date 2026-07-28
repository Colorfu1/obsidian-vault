# ACT (Action Chunking with Transformers) 深度理解

## 1. ACT 的核心定位

ACT (Action Chunking with Transformers) 是一种基于 Behavior Cloning
的机器人模仿学习方法。

它的核心思想不是学习：

$$
o_t \rightarrow a_t
$$

而是学习：

$$
o_t \rightarrow (a_t, a_{t+1}, \ldots, a_{t+H-1})
$$

即根据当前观测，一次预测未来一段时间的动作序列。

ACT 的贡献并不只是使用
Transformer，而是把机器人控制从逐帧动作预测转化为短期轨迹预测。

------------------------------------------------------------------------

## 2. ACT 的输入：当前观测，而不是历史序列

ACT policy 可以表示为：

$$
\pi_\theta(a_{t:t+H} \mid s_t)
$$

其中：

$$
s_t = (I_t^1, I_t^2, \ldots, I_t^N, q_t)
$$

包含：

-   当前时刻多视角 RGB 图像；
-   当前机器人状态。

不包含：

-   历史图像；
-   历史 action；
-   previous action chunk；
-   recurrent hidden state。

因此 ACT 本质仍然是 reactive policy：

$$
a = f(o_t)
$$

只是输出从单个动作变成未来动作 chunk。

------------------------------------------------------------------------

## 3. Action Chunking 的含义

假设：

    chunk_size = 20

模型输出：

$$
\hat{A}_t =
\left[
\hat{a}_t^0, \hat{a}_t^1, \ldots, \hat{a}_t^{19}
\right]
$$

其中：

$$
\hat{a}_t^i
$$

表示从当前状态出发，第 i 步应该执行的动作。

它不是自回归生成：

    a_t -> a_{t+1} -> a_{t+2}

而是：

    同一个 observation
           |
           + query0 -> action0
           + query1 -> action1
           + query2 -> action2

所有动作并行预测。

------------------------------------------------------------------------

## 4. 为什么 Action Chunking 有效

### 4.1 学习动作轨迹结构

单步 BC：

$$
o_t \rightarrow a_t
$$

只知道当前动作。

Chunk prediction：

$$
o_t \rightarrow [a_t, \ldots, a_{t+H}]
$$

迫使模型学习：

    靠近物体
    ↓
    抓取
    ↓
    移动
    ↓
    释放

因此获得任务阶段信息。

### 4.2 降低逐帧随机误差

单步预测容易：

    +0.01
    -0.02
    +0.01

产生 jitter。

Chunk prediction 学习：

    0.01
    0.012
    0.014
    0.016

更符合连续运动。

------------------------------------------------------------------------

## 5. Action Chunking 的隐藏问题

训练：

$$
o_t \rightarrow [a_t, \ldots, a_{t+19}]
$$

但是推理：

如果：

    n_action_steps=1

只执行：

    action[0]

然后重新预测。

因此：

上一轮预测：

    chunk[18]=release

会被下一轮预测完全覆盖。

未来计划并没有真正进入执行队列。

------------------------------------------------------------------------

## 6. ACT 的时间一致性问题

ACT 输入只有：

$$
o_t
$$

没有：

-   上一次预测；
-   chunk 当前执行位置；
-   已经等待多久；
-   任务阶段记忆。

因此：

如果：

$$
o_{t+1} \approx o_t
$$

则：

$$
\pi(o_{t+1}) \approx \pi(o_t)
$$

可能出现：

    frame 255:
    18 steps later release

    frame 256:
    18 steps later release

    frame 257:
    18 steps later release

未来事件不断被推迟。

这不是模型不会释放，而是未来计划没有持续存在。

------------------------------------------------------------------------

## 7. Temporal Ensemble 的作用

Temporal Ensemble 不是简单平滑动作。

它解决的是：

> 如何让多个时刻预测的未来计划保持连续。

同一个绝对时间：

$$
\tau
$$

可能有：

$$
\hat{a}_t^{18}
$$

和：

$$
\hat{a}_{t+1}^{17}
$$

等多个预测。

Temporal Ensemble：

$$
a_\tau = \sum_i w_i \hat{a}_{\tau-i}^{\,i}
$$

融合这些预测。

效果：

1.  保留旧计划；
2.  降低单次预测噪声；
3.  减少 chunk 边界跳变。

因此 Temporal Ensemble 相当于给无状态 ACT 增加了一种弱记忆。

------------------------------------------------------------------------

## 8. 为什么 A1 chunking 可能比单步差

原因：

### 8.1 开环执行

如果：

    n_action_steps=5

一次预测后连续执行 5 步。

状态误差无法及时修正。

### 8.2 Loss 与执行不一致

训练监督：

$$
[a_t, a_{t+1}, \ldots, a_{t+19}]
$$

但是执行：

只使用第 0 个动作。

未来动作占据大量 loss，但可能被丢弃。

### 8.3 Horizon query bias

不同 action query：

    query0
    query1
    ...
    query19

可能存在不同偏差。

chunk 切换时：

    query19 -> query0

可能产生动作跳变。

------------------------------------------------------------------------

## 9. Action Difference Loss

ACT 原始 loss：

$$
L = \left\lVert \hat{a}_t - a_t \right\rVert
$$

只约束动作值。

不约束：

$$
\Delta a
$$

因此可能：

    0.1
    0.15
    0.1
    0.15

动作误差不大，但 jerk 很高。

增加：

$$
L_\Delta =
\left\lVert
(\hat{a}_{t+1} - \hat{a}_t) - (a_{t+1} - a_t)
\right\rVert
$$

要求预测动作变化趋势和 demonstration 一致。

它优化的是轨迹形状，而不是简单让动作变小。

------------------------------------------------------------------------

## 10. 实验结果理解

A2 动作差分损失实验：

-   First-action MAE 降低 17.7%
-   Delta-action MAE 降低 9.0%
-   Closed-loop jerk 降低 20.3%
-   Success rate 从 15% 提升到 70%

说明 delta loss 改善了动作轨迹质量。

来源： fileciteturn0file1

A2 + Temporal Ensemble：

-   Success rate: 70% -\> 90%
-   Action jerk: 0.02403 -\> 0.00418

说明 Temporal Ensemble 解决的不只是平滑，而是计划连续性问题。

来源： fileciteturn0file1

------------------------------------------------------------------------

## 11. ACT 的完整理解

ACT 不是：

    Transformer + action output

而是：

    Current observation
            |
            v
    Action chunk prediction
            |
            v
    Temporal aggregation
            |
            v
    Robot execution

三个关键组成：

  模块                作用
  ------------------- ------------------
  Action Chunking     学习未来动作结构
  Temporal Ensemble   保持未来计划连续
  Delta Action Loss   优化动作变化轨迹

------------------------------------------------------------------------

## 12. ACT 的局限

ACT 仍然不是 World Model。

它不知道：

    执行 action 后未来状态如何变化

它学习的是：

    当前状态附近，专家通常如何行动

因此：

-   长 horizon planning 弱；
-   分布外状态容易失败；
-   没有显式世界状态。

这也是后续方法探索：

-   Diffusion Policy
-   VLA
-   World Action Model

的原因。

------------------------------------------------------------------------

## 13. 最终总结

ACT 的核心贡献：

> 用 action chunking 将机器人控制从逐帧反应提升为短期轨迹预测。

但：

> 预测未来 ≠ 执行未来。

因此：

-   Action Chunking 负责"看未来"；
-   Temporal Ensemble 负责"记住未来"；
-   Delta loss 负责"走得平稳"。

完整 ACT 系统：

$$
\boxed{
\begin{aligned}
\mathrm{ACT} ={}&
\text{Action Chunk Prediction} \\
&+ \text{Temporal Plan Aggregation} \\
&+ \text{Trajectory Supervision}
\end{aligned}
}
$$
