---
title: ALOHA 与 ACT：从低成本双臂数据采集到闭环稳定性
type: concept_note
topic: robot_imitation_learning
status: mature
importance: high
updated: 2026-08-19
tags:
  - aloha
  - act
  - bimanual-manipulation
  - action-chunking
  - cvae
  - temporal-ensemble
  - imitation-learning
  - closed-loop-evaluation
---

# ALOHA 与 ACT：从低成本双臂数据采集到闭环稳定性

这篇笔记是仓库中 ALOHA / ACT 主题的统一主笔记，覆盖从硬件数据采集、ACT 模型结构到
闭环实验分析的完整链路：

1. ALOHA 如何以低成本 leader-follower 硬件采集高质量双臂示范，以及 ACT 论文提出的
   Action Chunking、CVAE 和 Temporal Ensemble；
2. 在本地 LIBERO 实验中，`chunk_size`、`n_action_steps`、padding loss、动作差分
   loss 和 CVAE 如何实际影响 closed-loop behavior，以及这些实验如何修正早期对 ACT
   failure mode 的理解。

原先分散的论文整理、ACT 概念理解和闭环稳定性分析已经合并到本文；`BestPractice/` 下的
文件继续保留为可追溯的实验记录，不重复展开为第二篇理论主笔记。

## 1. 阅读范围与证据层级

> [!info] 证据分层
> - **论文报告**：指 *Learning Fine-Grained Bimanual Manipulation with Low-Cost
>   Hardware* 中的系统设计、真实机器人结果和消融。
> - **本地实验**：指单个 LIBERO Object 任务上的 A-Sanity 和 20-demonstration
>   development 实验。它们主要用于理解机制，不能直接推广为 ACT 的一般结论。
> - **综合解释**：指结合论文和本地实验得到的机制判断。凡是现有证据无法区分的解释，
>   会明确保留不确定性。

ALOHA 与 ACT 解决的是一条完整链路上的不同问题：

```text
低成本、可执行的双臂人类输入
→ 高质量 joint-space demonstrations
→ 当前状态条件下的连续 action chunk
→ CVAE 建模 human action variation
→ 重规划与 Temporal Ensemble 形成闭环执行
→ Closed-loop rollout 检验策略是否真正稳定
```

因此，ACT 不能只被理解成“一个 Transformer”。硬件与数据接口决定监督信号的质量，
Action Chunking 决定预测对象，loss 决定哪些状态和轨迹特征得到重视，推理策略决定
预测出的 chunk 如何进入真实闭环。

## 2. ALOHA 解决的数据采集问题

### 2.1 为什么精细双臂示范难以采集

论文关注的是接触丰富、要求双臂协调和高精度视觉闭环的任务，例如：

- 打开透明 condiment cup；
- 把电池塞入遥控器；
- 打开 ziploc bag；
- 穿过魔术贴小环；
- 准备胶带并贴到纸盒边缘；
- 给假脚穿鞋。

这些任务同时涉及毫米级定位、遮挡、透明或低对比度物体、柔性物体、连续接触和双臂
配合。若使用 VR controller 或手部视觉追踪，通常需要：

$$
\text{human hand pose}
\rightarrow
\text{robot end-effector target}
\rightarrow
\text{inverse kinematics}
\rightarrow
\text{joint command}.
$$

对低成本 6DoF、无冗余机械臂而言，这条 task-space retargeting 链路容易遇到：

1. 靠近 singularity 或 joint limit 时 IK 失败或跳变；
2. IK、映射和控制链路增加延迟；
3. 自由空间中的人手动作未必满足 follower 的机械约束；
4. 手部追踪噪声会直接放大为末端抖动。

ALOHA 的核心选择是：不用抽象手柄描述末端位姿，而让人直接拖动一对与 follower
关节结构对应的小机械臂。

### 2.2 WidowX leader 与 ViperX follower

ALOHA 使用：

- 两只较小的 **WidowX** 作为 leader，由操作者 backdrive；
- 两只较大的 **ViperX** 作为 follower，真正与环境交互。

左右臂各自形成一条映射：

$$
\text{left WidowX}
\rightarrow
\text{left ViperX},
$$

$$
\text{right WidowX}
\rightarrow
\text{right ViperX}.
$$

WidowX 类似具有真实机械约束的实体输入设备。操作者感受到的是一套可执行的关节结构，
ViperX 则负责抓、推、撬、插入等真实接触动作。这一设计同时降低了 retargeting 难度
和遥操作延迟。

### 2.3 Task-space mapping 与 direct joint-space mapping

Task-space mapping 先获得手或 leader 的末端位姿：

$$
T_{\mathrm{hand}}=(x,y,z,R),
$$

映射为 follower 末端目标：

$$
T_{\mathrm{ee}}^{\mathrm{target}},
$$

再求：

$$
q_{\mathrm{target}}
=
\operatorname{IK}\left(T_{\mathrm{ee}}^{\mathrm{target}}\right).
$$

ALOHA 采用 direct joint-space mapping：

$$
q_{\mathrm{follower}}^{\mathrm{target}}
=
f(q_{\mathrm{leader}}).
$$

最简单时，各对应关节直接映射：

$$
q_{\mathrm{follower},i}^{\mathrm{target}}
=
q_{\mathrm{leader},i}.
$$

实际系统还可以包含 scale、offset 和符号修正：

$$
q_{\mathrm{follower},i}^{\mathrm{target}}
=
s_iq_{\mathrm{leader},i}+b_i.
$$

它避免在线求解末端 IK，让操作者输入、目标关节命令和 follower 底层控制器形成更短、
更高带宽的链路。

## 3. Observation、Action 与数据接口

### 3.1 Observation

论文中的 ACT observation 是当前时刻的四路 RGB 图像和 follower 关节状态：

$$
o_t
=
\left(
I_t^{(1)},I_t^{(2)},I_t^{(3)},I_t^{(4)},
q_t^{\mathrm{follower}}
\right).
$$

相机包括：

- top camera；
- front camera；
- left wrist camera；
- right wrist camera。

论文系统以 50 Hz 进行遥操作和数据记录。多视角图像提供物体、环境和近距离接触信息，
follower proprioception 则告诉策略机器人此刻实际处于什么关节状态。

### 3.2 Action

每个动作是左右 leader 的目标关节位置：

$$
a_t\in\mathbb R^{14},
$$

$$
a_t
=
\left[
q_1^L,\ldots,q_7^L,\,
q_1^R,\ldots,q_7^R
\right].
$$

这里使用的是 **leader joint positions**，而不是 follower 实际测得的关节位置。
follower 的底层 PID 控制器负责追踪 leader 目标。leader 与 follower 的位置差异还会
隐式包含操作者在接触阶段持续施加目标的控制意图；若只把 follower 实际位置当作动作，
可能丢失这部分监督。

数据接口可以概括为：

```text
Observation：当前多视角 RGB + follower 当前关节位置
Action：左右 leader 的连续目标关节位置，共 14 维
Controller：follower 底层 PID 追踪目标
Frequency：50 Hz
```

## 4. ACT：从单步动作到连续 Action Chunk

### 4.1 单步 Behavior Cloning 的困难

普通 Behavior Cloning 学习：

$$
\pi_\theta(a_t\mid o_t).
$$

在 50 Hz 控制下，一个 10 秒 episode 含有：

$$
50\times10=500
$$

个决策步。单步误差会改变下一时刻状态，使策略逐渐离开 demonstration
distribution；策略再在未见状态上预测，误差便可能继续累积。这就是 compounding
error。

### 4.2 Action Chunking

ACT 根据当前 observation 一次预测未来 $H$ 步动作：

$$
\pi_\theta(o_t)
\rightarrow
\hat A_t
=
\left[
\hat a_{t\mid t},
\hat a_{t+1\mid t},
\ldots,
\hat a_{t+H-1\mid t}
\right].
$$

其中 $\hat a_{t+h\mid t}$ 表示“依据 $t$ 时刻 observation，对绝对时刻 $t+h$ 的
动作预测”。若每个动作有 $D=14$ 维：

$$
\hat A_t\in\mathbb R^{H\times14}.
$$

Action Chunking 将学习目标从：

> 当前状态下，专家现在做什么？

扩展成：

> 当前状态下，专家接下来的一小段动作轨迹是什么？

这样模型可以同时学习短期动作结构：

```text
接近物体
→ 闭合夹爪
→ 搬运
→ 对准目标
→ 下降
→ 松开夹爪
```

预测未来序列缩短了 chunk-level effective horizon，也给模型提供了 future action
supervision。但“预测出一段未来动作”不等于“整段动作会原样执行”；真正执行多少步
由推理策略决定。

### 4.3 Action 是连续向量，不是离散 token

ACT 的动作仍是连续关节位置。所谓 embedded action sequence 是把每个连续动作通过
linear layer 投影成 Transformer sequence element：

$$
a_t\in\mathbb R^{14},
$$

$$
e_t=W_aa_t+b_a\in\mathbb R^{512}.
$$

因此这里的 “action token” 是连续向量的 embedding，不是 NLP token id，也没有先将
动作量化为有限词表。最终输出仍是：

$$
\hat A_t\in\mathbb R^{H\times14}.
$$

这与 BeT、RT-1 或 FAST 等离散 action representation 不同。论文选择连续预测，是因为
低成本精细操作对小幅关节变化敏感，离散化可能损失控制精度。

## 5. 当前观测、Action Query 与时间语义

### 5.1 ACT 是输出序列的 reactive policy

原始 ACT policy 主要接收当前时刻：

$$
o_t=
\left\{
I_t^{(1)},\ldots,I_t^{(N)},q_t
\right\}.
$$

基础输入不显式包含：

- 历史图像；
- 历史 proprioception；
- 上一轮 action chunk；
- 上一轮计划执行到了第几步；
- recurrent hidden state；
- episode step counter。

因此：

$$
\hat A_t=f_\theta(o_t)
$$

仍然是 reactive mapping，只是输出由一个动作变成一段动作。如果：

$$
o_{t+1}\approx o_t,
$$

那么通常也会有：

$$
f_\theta(o_{t+1})\approx f_\theta(o_t).
$$

这不意味着 ACT 必须具有显式“倒计时”。只要已执行动作能够持续推动真实状态，新的
observation 就携带了任务进度，reactive policy 可以据此进入下一阶段。

### 5.2 Action Query 表示相对 horizon

Transformer decoder 中的不同 Action Query 对应 chunk 内不同的相对时间位置：

```text
query 0     → 当前动作
query 1     → 未来第 1 步动作
...
query H - 1 → 未来第 H - 1 步动作
```

它们并不具有固定任务语义：

```text
错误理解：
query 0  永远表示“接近”
query 5  永远表示“抓取”
query 18 永远表示“释放”
```

训练样本来自轨迹滑动窗口：

$$
\left(
o_t,\,
a_t,a_{t+1},\ldots,a_{t+H-1}
\right).
$$

同一个 release 事件随窗口起点不同，可能出现在 `chunk[0]`、`chunk[5]`、
`chunk[18]`，也可能不在当前 chunk 内。某些 rollout 中 release 长期出现在未来
17～19 步，不能解释为特定 query 固定学习了 release；更准确的含义是，模型对一类
相似 observation 始终预测“还需要十几步才释放”。

## 6. `chunk_size`、`n_action_steps` 与闭环频率

### 6.1 `chunk_size`

`chunk_size=H` 决定模型每次预测的 future horizon：

```text
chunk_size = 20
```

表示：

$$
o_t\rightarrow a_{t:t+19}.
$$

在论文 50 Hz 系统中，$H=100$ 大约覆盖 2 秒。它主要影响训练 target 的长度、模型需要
表达的短期轨迹结构和 chunk-level effective horizon。

### 6.2 `n_action_steps`

`n_action_steps` 决定一次推理后真正连续执行 chunk 前缀中的多少个动作：

```text
预测 20 步
→ 执行前 5 步
→ 获取新 observation
→ 再预测 20 步
```

因此：

```text
chunk_size     → 预测多远
n_action_steps → 一次计划实际执行多久
```

`n_action_steps` 较大：

- 推理频率较低；
- 计划持续性较强；
- 但更接近 open-loop，执行误差不能及时纠正。

`n_action_steps=1`：

- 每一步都重新观察和规划；
- 闭环反馈最及时；
- 旧 chunk 会立即被新 chunk 覆盖；
- 若不开 Temporal Ensemble，每次实际上只执行最新 chunk 的第一步。

两者必须分开配置和解释。相同 `chunk_size` 下，仅改变 `n_action_steps` 就会改变闭环
动力学，因此实验比较时不能把它当作无关的推理细节。

## 7. ACT Transformer 完整结构

> [!figure] 论文原始模型结构图
> ![[attachments/paper-figures/act-model-architecture.png]]
> ACT 的 Conditional VAE：训练时由真实 action chunk 推断 latent，policy 端以多视角
> 图像、当前关节状态和 latent 预测连续 action chunk。原图来自
> [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)。

ACT 可以分为训练时的 posterior encoder 和真正用于动作预测的 policy。

### 7.1 CVAE posterior encoder

训练阶段的 encoder 输入包括：

- 一个可学习的 `[CLS]` embedding；
- 当前 joint positions；
- 真实未来 action sequence。

输入序列可写为：

$$
X=
\left[
e_{\mathrm{cls}},
e_{\mathrm{joints}},
e_{\mathrm{action},0},
\ldots,
e_{\mathrm{action},H-1}
\right].
$$

`[CLS]` 不是物理输入或离散词表 token，而是一个随机初始化、随训练学习的参数向量：

$$
e_{\mathrm{cls}}\in\mathbb R^{512}.
$$

经过 Transformer encoder 后，取 `[CLS]` 位置输出：

$$
h_{\mathrm{cls}}=H_0,
$$

再预测 latent distribution 参数。论文详细结构中 $z$ 为 32 维，因此可以理解为：

$$
\mu\in\mathbb R^{32},
\qquad
\log\sigma^2\in\mathbb R^{32}.
$$

### 7.2 图像、关节和 latent 表示

四路图像先经过 ResNet18。按论文结构图中的尺寸，每路：

$$
480\times640\times3
\rightarrow
15\times20\times512.
$$

flatten 后每路产生：

$$
300\times512
$$

个视觉 token，四路合计：

$$
1200\times512.
$$

视觉 feature 加入 2D sinusoidal position embedding；当前 joint state 和 $z$ 分别
投影为 token。按该结构示意，总输入可以理解为：

$$
1200+1+1=1202
$$

个 512 维 token。

### 7.3 Policy 与 Action Query

Transformer encoder 融合多视角视觉、当前关节状态和 latent。Transformer decoder
使用固定位置 embedding 作为 Action Query，通过 cross-attention 从融合特征中读取
信息，并为每个相对 horizon 输出一个连续动作：

$$
\hat A_t
=
\left[
\hat a_{t\mid t},
\ldots,
\hat a_{t+H-1\mid t}
\right]
\in\mathbb R^{H\times14}.
$$

Action Query 提供的是输出槽位的相对时间位置，不是预定义的抓取、搬运或释放语义。

## 8. CVAE、Diagonal Gaussian、KL 与 `z=0`

### 8.1 为什么需要 latent variable

Human demonstration 往往是多模态的。同一 observation 下，人可能采用不同但都能成功
的速度、交接位置、接近路径或释放时机。

确定性 L1/L2 regression 必须给出一个输出，可能在不同合理轨迹之间产生条件中位数或
均值。连续控制中的“平均轨迹”未必可执行，例如两条绕开障碍的路径平均后可能正好撞向
障碍。

如果同一个 observation 对应多条合法的 future action chunk，普通 deterministic
regression 只能输出一个点估计：

$$
\hat A_t=f_\theta(o_t).
$$

在 L1 或 MSE 监督下，这个点估计可能接近不同 demonstration mode 的均值或中位数：

$$
\hat A_t
\approx
\operatorname{mean/median}
\left(
A_t^{(1)},A_t^{(2)},\ldots
\right).
$$

因此，两个单独看都能成功的轨迹，其平均轨迹不一定仍然可执行：

$$
\boxed{
\text{两个成功轨迹的平均，不一定仍然是一条成功轨迹。}
}
$$

ACT 用条件生成模型表示：

$$
p_\theta(A_t\mid o_t),
$$

并引入 latent $z$ 表示 action style 或 mode：

$$
q_\phi(z\mid A_t,\bar o_t),
$$

$$
\hat A_t=\pi_\theta(o_t,z).
$$

其中 $\bar o_t$ 主要包含 proprioception，不包含图像；CVAE posterior 处理的是机器人
状态和目标 action chunk，不是图像 tokenizer。图像由 CNN backbone 编码。

### 8.2 Diagonal Gaussian

Posterior encoder 不直接输出确定的 $z$，而输出：

$$
\mu\in\mathbb R^d,
\qquad
\sigma\in\mathbb R^d,
$$

从而定义：

$$
q_\phi(z\mid A_t,\bar o_t)
=
\mathcal N
\left(
\mu,
\operatorname{diag}(\sigma^2)
\right).
$$

其协方差矩阵为对角矩阵：

$$
\operatorname{diag}(\sigma^2)
=
\begin{bmatrix}
\sigma_1^2&0&\cdots\\
0&\sigma_2^2&\cdots\\
\vdots&\vdots&\ddots
\end{bmatrix},
$$

表示在该近似 posterior 中：

$$
\operatorname{Cov}(z_i,z_j)=0,\qquad i\ne j.
$$

等价地：

$$
q_\phi(z\mid x)
=
\prod_i
\mathcal N(z_i;\mu_i,\sigma_i^2).
$$

### 8.3 Reparameterization

训练时：

$$
\epsilon\sim\mathcal N(0,I),
$$

$$
z=\mu+\sigma\odot\epsilon.
$$

代码只需保存向量形式的 `mu` 和 `logvar`：

```python
stats = linear(h_cls)
mu = stats[:, :z_dim]
logvar = stats[:, z_dim:]

std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps
```

不需要显式构造完整协方差矩阵；若仅用于观察，可以用：

```python
cov = torch.diag_embed(std ** 2)
```

### 8.4 Posterior 会不会采到错误 mode

训练时不是从全局标准高斯随意采样，而是从当前真实 action chunk 对应的 posterior
采样：

$$
z\sim q_\phi(z\mid A_t,\bar o_t).
$$

若当前 action chunk 表示从左向右执行，posterior 参数就是由这条真实轨迹推断的。理想
情况下，不同 mode 可形成不同区域：

$$
q_L(z)=\mathcal N(\mu_L,\sigma_L^2),
\qquad
q_R(z)=\mathcal N(\mu_R,\sigma_R^2).
$$

如果 posterior 方差过大，确实可能采到不适合当前 target 的区域。但错误样本会造成较大
reconstruction loss，并推动：

1. encoder 移动 $\mu$；
2. encoder 调整 $\sigma$，降低跨 mode 采样概率；
3. decoder 更准确地使用 $o_t$ 与 $z$；
4. 当 observation 已足以决定动作时，decoder 减少对 $z$ 的依赖。

因此 latent sampling 是受 reconstruction objective 约束的训练噪声，不是无约束的
随机标签。

### 8.5 Reconstruction 与 KL

标准目标为：

$$
\mathcal L_{\mathrm{ACT}}
=
\mathcal L_{\mathrm{reconst}}
+
\beta
D_{\mathrm{KL}}
\left(
q_\phi(z\mid A_t,\bar o_t)
\;\|\;
\mathcal N(0,I)
\right).
$$

两项作用相反：

- reconstruction 希望 $z$ 携带足够信息，帮助重构当前 action chunk；
- KL 希望 posterior 不要远离 prior，使推理时 prior 中心附近仍然有效。

若 $\beta$ 太小，posterior 可能携带过多信息并远离 prior，推理时 `z=0` 会产生
train-test mismatch；若 $\beta$ 太大，latent 可能失去信息，模型重新退化为近似
deterministic regression。论文超参表使用：

$$
\beta=10.
$$

### 8.6 为什么推理时设 `z=0`

训练时有真实未来 action chunk，可以用 posterior encoder：

$$
z\sim q_\phi(z\mid A_t,\bar o_t).
$$

推理时 $A_t$ 正是待预测对象，不能先知道未来动作再用它推断 $z$。因此 posterior
encoder 被丢弃，只保留 policy，并选择标准高斯 prior：

$$
p(z)=\mathcal N(0,I)
$$

的均值：

$$
z=0.
$$

推理策略为：

$$
\hat A_t=\pi_\theta(o_t,z=0).
$$

这给出一个 deterministic canonical behavior，而不是在真实机器人评测时随机切换
style。

### 8.7 为什么训练使用 CVAE、推理 `z=0` 仍可能变好

本地实验确认标准 ACT CVAE 显著提高了一个 LIBERO 任务的 closed-loop success，但现有
证据不能把提升严格归因于唯一机制。合理解释包括：

1. **分离相似 observation 下的不同 future chunk**
   确定性 L1 对多种合理 target 倾向于条件中位数。latent 可以减少不同轨迹 mode
   之间的冲突梯度。

2. **形成 canonical behavior**
   KL 将 posterior 拉向标准高斯，decoder 可能在 $z=0$ 附近学出一条典型且可执行的
   轨迹，而不是逐维动作的简单平均。

3. **训练期 privileged information**
   Posterior 看到了目标 chunk，$z$ 可以编码 demonstration style、速度、未来阶段或
   释放时机，降低 decoder 的训练难度。

4. **结构化正则化**
   Latent sampling 与 KL 可能提高 decoder 对小范围表示扰动的稳定性。

本地结果与“多模态 action-chunk 建模”解释一致，但还不能排除额外参数、latent noise、
正则化或单训练 seed 波动。稳妥结论是：

$$
\boxed{
\text{在当前本地任务中，标准 ACT CVAE 改善了 action-chunk 学习和闭环稳定性。}
}
$$

### 8.8 CVAE 的作用边界：训练期解耦，推理期不显式选 mode

ACT 的 posterior encoder 在训练时可以看到目标 action chunk：

$$
A_t^{GT}
\rightarrow
q_\phi(z\mid q_t,A_t^{GT})
\rightarrow
z.
$$

它的主要作用是把相似 observation 下的不同 demonstration mode 分开，降低 decoder
同时拟合互相冲突的 future chunk 时产生的梯度冲突。它并不是一个在部署阶段根据当前
观测显式选择左绕、右绕等 mode 的 planner。

推理时未来动作不存在，posterior encoder 不能使用，因此标准 ACT 采用：

$$
z=0,
\qquad
\hat A_t=\pi_\theta(o_t,z=0).
$$

所以 ACT CVAE 更准确的表述是：

$$
\boxed{
\text{训练期用 }z\text{ 解耦 action modes；推理期使用 prior 中心附近的 canonical behavior。}
}
$$

这也解释了它与后续直接建模 $p(A_t\mid o_t)$ 的 Diffusion、Flow Matching 或
autoregressive action-token 方法之间的差别：后者把生成自由度保留到了推理阶段，而
标准 ACT 的 $z$ 主要承担训练期的表示和正则化作用。

## 9. 训练目标：L1、Padding、ObsEqual 与 Delta-action loss

### 9.1 滑动窗口 target 与标准 ACT loss

训练样本以轨迹中的当前时刻为起点：

$$
\left(o_t,A_t\right),
$$

$$
A_t=
\left[
a_t,a_{t+1},\ldots,a_{t+H-1}
\right].
$$

每次训练：

1. 读取当前多视角图像和 follower joint state；
2. 取未来 $H$ 步 leader joint targets；
3. 靠近 episode 末尾时，对不足 $H$ 的部分 padding；
4. Posterior encoder 根据当前 joint state 和真实 chunk 推断 $z$；
5. Policy 根据当前 observation 和 $z$ 预测 chunk；
6. 有效位置计算 action reconstruction loss，并加 KL。

论文 Algorithm 1 将 reconstruction 写成 MSE；论文实现说明实际使用 L1，因为作者发现
L1 对动作序列预测更精确。因此实现层面常见：

$$
\mathcal L_{\mathrm{action}}
=
\sum_{h,d}
m_{h}
\left|
\hat a_{t+h,d}-a_{t+h,d}
\right|.
$$

总目标可概括为：

$$
\mathcal L
=
\mathcal L_{\mathrm{action}}
+
\lambda_{\mathrm{KL}}\mathcal L_{\mathrm{KL}}.
$$

### 9.2 Padding mask 会改变 observation 的总权重

固定 horizon tensor 常在 padding 位置将 loss 清零，再对完整 $B\times H\times D$
张量求 mean：

$$
\mathcal L_{\mathrm{fixed}}
=
\frac{1}{BHD}
\sum_{b=1}^{B}
\sum_{h=1}^{H}
\sum_{d=1}^{D}
m_{b,h}
\left|
\hat a_{b,h,d}-a_{b,h,d}
\right|.
$$

若第 $b$ 个 observation 只剩 $H_b$ 个有效未来动作，它对 batch loss 的总贡献正比于：

$$
\frac{H_b}{H}.
$$

例如 $H=20$，episode 末尾只剩一个有效动作时，该 observation 的总监督权重约为完整
horizon observation 的：

$$
\frac{1}{20}.
$$

但 episode 末尾往往包含：

- 进入目标区域；
- 松开夹爪；
- 松开后保持；
- 终止前的稳定动作。

所以“padding 位置不参与 loss”并不等于“每个 observation 得到相同训练权重”。

ObsEqual 的做法是先对每个 observation 的有效位置求 mean，再对 batch 求 mean：

$$
\mathcal L_{\mathrm{ObsEqual}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\frac{
\sum_{h,d}
m_{b,h}
\left|
\hat a_{b,h,d}-a_{b,h,d}
\right|
}{
D\sum_hm_{b,h}
}.
$$

这使每个 observation 的总贡献相同。本地实验中 ObsEqual 改善了闭环成功率，而整体
offline error 变化很小；具体数值集中在第 12、13 章。

### 9.3 Delta-action loss

标准 L1 主要约束每个动作值：

$$
\mathcal L_{\mathrm{action}}
=
\sum_h
\left\|
\hat a_{t+h}-a_{t+h}
\right\|_1.
$$

它不直接要求预测序列具有与 demonstration 相同的一阶变化。Delta-action loss 定义为：

$$
\mathcal L_{\Delta a}
=
\sum_h
\left\|
\left(\hat a_{t+h+1}-\hat a_{t+h}\right)
-
\left(a_{t+h+1}-a_{t+h}\right)
\right\|_1.
$$

总目标变为：

$$
\mathcal L
=
\mathcal L_{\mathrm{ACT}}
+
\beta_{\Delta}\mathcal L_{\Delta a}.
$$

它不是简单最小化：

$$
\left\|
\hat a_{t+h+1}-\hat a_{t+h}
\right\|,
$$

所以不会无条件把所有动作压平。它要求预测动作的一阶变化与 demonstration 一致，既可
保留应有运动，也可惩罚错误的轨迹形状。闭环上，它可能改善的不只是视觉平滑度，还包括
action prefix 是否能持续推动状态进展。

### 9.4 完整训练数据流

```text
从 demonstration 选择时刻 t
→ 读取当前图像与 follower joints
→ 构造未来 H 步 leader action target 与 padding mask
→ Posterior encoder(target chunk, current joints)
→ 得到 μ、logvar，并重参数化采样 z
→ Policy(current images, current joints, z)
→ 预测 H × 14 action chunk
→ 计算有效位置 L1
→ 可选 ObsEqual reduction
→ 可选 Delta-action loss
→ 加 KL loss
→ 联合更新 posterior encoder 与 policy
```

## 10. 推理、重规划与 Temporal Ensemble

### 10.1 为什么不能只看“预测了多少步”

最朴素的 chunk execution 是每隔 $H$ 步观察一次环境，预测一个 chunk，然后全部执行。
这样 chunk 内近似 open-loop：

```text
observe once
→ predict H actions
→ execute all H actions
→ observe again
```

它降低推理频率，但不能及时利用新视觉反馈；chunk 切换处还可能出现动作突变。

ACT 论文的 Temporal Ensemble 路线则每个 timestep 都重新 query policy，并把历史
chunk 对当前绝对时刻的预测对齐后融合。本地无 ensemble 配置也可以每次执行
`n_action_steps` 个动作后重新规划。两者都使用 action chunk，但闭环行为不同。

### 10.2 同一绝对时刻的重叠预测

假设 $H=4$。在 $t=0$：

$$
\left[
\hat a_0^{(0)},
\hat a_1^{(0)},
\hat a_2^{(0)},
\hat a_3^{(0)}
\right].
$$

在 $t=1$：

$$
\left[
\hat a_1^{(1)},
\hat a_2^{(1)},
\hat a_3^{(1)},
\hat a_4^{(1)}
\right].
$$

在 $t=2$：

$$
\left[
\hat a_2^{(2)},
\hat a_3^{(2)},
\hat a_4^{(2)},
\hat a_5^{(2)}
\right].
$$

因此绝对时刻 3 可以同时获得：

$$
\hat a_3^{(0)},\quad
\hat a_3^{(1)},\quad
\hat a_3^{(2)},\quad
\hat a_3^{(3)}.
$$

Temporal Ensemble 对这些表示同一绝对时刻的预测进行加权：

$$
a_t^{\mathrm{exec}}
=
\frac{
\sum_iw_iA_t[i]
}{
\sum_iw_i
},
$$

$$
w_i=\exp(-mi).
$$

按原笔记采用的索引约定，$i=0$ 对应最旧预测；$m$ 较大时新预测权重衰减更快，更依赖
旧计划，动作更平滑但响应更慢。复现时必须确认具体实现的排列顺序；如果代码把最新
预测放在索引 0，权重解释会相反。论文主超参表没有给出统一固定的 $m$，应把它视为
inference-time hyperparameter。

### 10.3 Temporal Ensemble 不是普通 smoothing

普通动作滤波可能写为：

$$
a_t
\leftarrow
\alpha a_t+(1-\alpha)a_{t-1}.
$$

它混合的是不同绝对时刻的动作，容易产生 lag。

Temporal Ensemble 混合的是：

$$
\hat a_t^{(t-H+1)},\ldots,\hat a_t^{(t)},
$$

这些预测都声称“绝对时刻 $t$ 应执行这个动作”。因此它是时间对齐后的 prediction
ensemble，而不是相邻 action smoothing。

### 10.4 它解决什么、不保证什么

Temporal Ensemble 可以：

1. 用多个 observation 时刻的预测降低单次方差；
2. 缓解重规划或 chunk 切换的不连续；
3. 让旧计划对未来绝对时刻的预测不会立即消失。

但它不能保证：

- policy 已学到正确 release；
- action prefix 能持续推动状态；
- padding loss 公平覆盖末端状态；
- 确定性 regression 能表达多种 future chunk。

本地 A1-VAE 在没有 Temporal Ensemble、`n_action_steps=5` 时达到 19/20 成功，说明：

$$
\boxed{
\text{Temporal Ensemble 是有效的闭环稳定化机制，但不是正确完成 release 的必要条件。}
}
$$

同样，A2 + TE 的本地实验还同时把 `n_action_steps` 改为 1，因此它不能单独识别
ensemble 的纯增益。

### 10.5 Temporal Ensemble 的 mode averaging 风险

Temporal Ensemble 融合的是“对同一个绝对时刻的多个预测”，但融合操作本身仍然是数值
平均。如果旧 prediction 计划从障碍物左侧绕行，而新 prediction 计划从右侧绕行，二者
分别可能是合法轨迹；直接平均后却可能得到一条从中间穿过障碍物的无效轨迹：

$$
A_t^{\mathrm{avg}}
=
\alpha A_t^{\mathrm{left}}
+
(1-\alpha)A_t^{\mathrm{right}}.
$$

因此，Temporal Ensemble 主要缓解的是跨 timestep 重规划造成的方差、动作跳变和 chunk
边界不连续，而不是从根本上解决多模态 action 的表示问题：

$$
\boxed{
\text{Temporal Ensemble 可以降低 prediction noise，但无法保证不同 mode 的平均仍然有效。}
}
$$

这也是它与 Diffusion / Flow Matching 的一个重要区别：后者在生成阶段保留采样自由度，
而 Temporal Ensemble 在已经生成的 action 之间做后处理融合。

### 10.6 Receding Horizon：一致性与反应性的折中

后续 action-generation policy 通常不把多个 overlapping chunk 做平均，而是联合生成一段
内部一致的 action trajectory，只执行其中的前缀，再根据新 observation 重新生成：

$$
A_t
=
[a_t,a_{t+1},\ldots,a_{t+H_{\mathrm{pred}}-1}],
\qquad
H_{\mathrm{exec}}<H_{\mathrm{pred}}.
$$

这就是 receding-horizon execution。它与本文第 6.2 节的 `n_action_steps` 是同一个
执行接口：`chunk_size` 决定预测多远，`n_action_steps` 决定一次真正提交多少个动作。

两种策略的差别可以概括为：

| 执行方式 | 主要机制 | 主要风险 |
| --- | --- | --- |
| Temporal Ensemble | 对齐多个 chunk 后加权平均 | 不同 mode 可能被平均成无效轨迹 |
| Receding Horizon | 生成一个 chunk，执行 prefix 后重新规划 | 新旧 chunk 交界处仍可能发生 mode switch |

因此存在不可同时消除的折中：

$$
H_{\mathrm{exec}}\uparrow
\Rightarrow
\text{更强的 temporal consistency，但反应性下降},
$$

$$
H_{\mathrm{exec}}\downarrow
\Rightarrow
\text{更强的 reactivity，但 mode jump 和 jitter 风险上升}.
$$

Receding Horizon 改变了重新规划的方式，但并没有自动解决跨 chunk 的时间一致性。

### 10.7 Trajectory-conditioned regeneration：从平均到条件续接

更进一步的生成式策略会把已经确定、正在执行或已经执行的旧 trajectory prefix 作为
条件，让新 trajectory 从这个 prefix 连续生成，而不是把旧、新 trajectory 做数值平均：

$$
A_{\mathrm{future}}
\sim
p_\theta
\left(
A_{\mathrm{future}}
\mid
o_t,A_{\mathrm{prefix}}^{\mathrm{committed}}
\right).
$$

两种操作的区别是：

$$
\text{Temporal Ensemble:}
\quad
A_{\mathrm{old}}+A_{\mathrm{new}}
\rightarrow
\text{数值平均},
$$

$$
\text{Trajectory-conditioned regeneration:}
\quad
A_{\mathrm{prefix}}^{\mathrm{committed}}
\rightarrow
\text{生成约束}
\rightarrow
A_{\mathrm{future}}.
$$

这类 RTC / conditional inpainting 思路并不是原始 ACT 的组成部分，而是后续 flow-based
action policy 为解决异步推理和 chunk 衔接提出的相关方向；π0.7 中的
[[Pi0_7_technical_report|RTC 讨论]]可作为进一步对照。它的目标是保留已承诺动作的连续性，
同时避免把来自不同 mode 的完整轨迹直接平均。

## 11. Release horizon 停滞与闭环固定点

### 11.1 观察到的现象

失败 rollout 中曾出现：

```text
step 255：release horizon = 18
step 256：release horizon = 18
step 257：release horizon = 18
...
step 279：release horizon = 19
```

早期解释是：

> ACT 没有保存旧计划和显式倒计时，所以 release 会永远停在未来。

它准确描述了表象，但后续实验表明，这不能作为根因。

### 11.2 更合理的因果链

```text
关键末端 observation 监督较弱
或 future action variation 建模不足
或 action prefix 局部不可执行
                ↓
执行动作没有产生足够状态进展
                ↓
机器人进入近似闭环固定点
                ↓
o[t + 1] ≈ o[t]
                ↓
reactive policy 重复产生相似 chunk
                ↓
release horizon 长期停留在 17～19
```

因此：

$$
\boxed{
\text{release horizon reset 更可能是闭环停滞的症状，而不是 ACT 必然失败的根因。}
}
$$

若动作能持续推动状态经历：

```text
靠近篮子
→ 到达篮子上方
→ 下降
→ 进入篮子
→ 松开
```

那么 observation 本身就会变化，release 会逐渐进入 chunk 前部并被执行，不一定需要
显式 counter 或 recurrent memory。

## 12. 论文结果与本地 LIBERO 实验

### 12.1 论文真实机器人结果

> [!note] 论文报告
> 下表来自 ALOHA + ACT 论文的真实机器人评测，不是本地 LIBERO 结果。

| 任务 | ACT 最终成功率 |
|---|---:|
| Slide Ziploc | 88% |
| Slot Battery | 96% |
| Open Cup | 84% |
| Thread Velcro | 20% |
| Prep Tape | 64% |
| Put On Shoe | 92% |

Thread Velcro 最难。黑色魔术贴与背景对比度低、目标小，右臂容易提前闭合或插入时错过
小环，说明视觉可辨识性仍会限制策略上限。

### 12.2 论文消融

#### Action Chunking

当：

$$
H=1
$$

时，ACT 退化为单步策略；当 $H$ 等于 episode length 时，则接近根据第一帧输出整段
open-loop 动作。

论文报告，从 $H=1$ 增大到 100，成功率明显提高；继续增大到 200 或 400 后略有下降。
这体现了两种力量：

- 更长 chunk 缩短 effective decision horizon，缓解 compounding error；
- 过长 chunk 更难预测，也可能损失闭环反应性。

论文主要设置为：

$$
H=100.
$$

#### Temporal Ensemble

论文报告 Temporal Ensemble 对参数化模型有小幅正向作用：

- ACT 约提升 3.3 个百分点；
- BC-ConvMLP 约提升 4 个百分点；
- VINN 性能下降。

一种解释是参数化模型的预测含噪，ensemble 可以降低噪声；VINN 检索的是真实示范动作，
混合不同检索结果反而可能破坏动作一致性。

#### CVAE

论文比较了 with CVAE 与 no CVAE：

- scripted data 上差异较小；
- human demonstrations 上 CVAE 明显更好。

这支持 CVAE 主要处理人类示范中的 stochasticity 和 multi-modal behavior，而不是
单纯增加模型容量。

### 12.3 本地 A-Sanity

> [!warning] 本地单任务实验
> 以下结果来自 LIBERO Object task 8：
> `pick up the chocolate pudding and place it in the basket`。

配置：

```text
5 demonstrations
859 frames
chunk_size = 20
n_action_steps = 5
use_vae = false
20,000 updates
```

结果：

| 指标 | 结果 |
|---|---:|
| First-action MAE | 0.02076 |
| Delta-action MAE | 0.00972 |
| Closed-loop success | 18/20（90%） |
| Closed-loop Action TV | 0.02195 |
| Closed-loop Action Jerk | 0.02210 |

它说明五条完整 demonstration 足以完成该任务的小数据 overfit sanity check，图像
backbone 与 action decoder 可以学到闭环行为；它不说明五条示范足以解决一般任务，也
不说明 CVAE 没有价值。

### 12.4 20-demonstration development 实验

公共条件：

```text
20 demonstrations
3233 frames
batch size = 8
20,000 updates
training seed = 1000
chunk_size = 20
Formal Clean Closed-loop 使用相同 20 个 init states
```

| 配置 | 主要变化 | Success | Action TV | Action Jerk | Gripper toggles |
|---|---|---:|---:|---:|---:|
| A1 | Deterministic ACT | 3/20（15%） | 0.02615 | 0.03017 | 14.80 |
| A2 | `+ 0.1 × delta loss` | 14/20（70%） | 0.02344 | 0.02403 | 14.65 |
| A2 + TE | `n_action_steps=1`，TE=0.01 | 18/20（90%） | 0.01517 | 0.00418 | 2.30 |
| A1-ObsEqual | 每个 observation 总权重相同 | 9/20（45%） | 0.02442 | 0.02590 | 13.50 |
| A1-VAE | 标准 ACT CVAE | 19/20（95%） | 0.02307 | 0.02205 | 11.40 |

相对 A1，A2 的配套统计还显示：

- First-action MAE 降低 17.7%；
- Delta-action MAE 降低 9.0%；
- Closed-loop Action Jerk 降低 20.3%；
- Success 从 15% 提高到 70%。

但 A2 的 teacher-forced jerk 没有同步下降。这说明 Delta-action supervision 的收益不能
简单归结为“让离线预测看起来更平滑”；它还改变了 rollout 中 action prefix 的局部
可执行性和策略访问到的状态分布。

这组结果分别支持：

- ObsEqual：末端 observation 总权重会影响关键终止行为；
- Delta-action loss：轨迹变化监督会影响 closed-loop executability，不只是平滑度；
- Temporal Ensemble：跨时刻融合可显著降低预测不一致；
- CVAE：action-chunk variation 的训练期建模对该任务非常重要。

### 12.5 证据边界

本地结果必须与以下限制一起阅读：

- 只覆盖一个 LIBERO Object 任务；
- 每个训练配置只有一个训练 seed；
- A2 + TE 同时改变 `n_action_steps` 与 ensemble，缺少严格同源 Replan control；
- A1-VAE 当时尚未完成对应 offline 指标；
- `gripper_toggles` 是动作命令过零次数，不等于物理夹爪完整开合次数；
- 成功 episode 提前终止，会改变 TV/Jerk 的序列长度分布；
- 不能从模拟器单任务结果直接外推到真实机器人或通用 ACT。

## 13. Offline 指标为什么不能替代 Closed-loop success

### 13.1 一个直接反例

A1 与 A1-ObsEqual 的 teacher-forced offline 指标很接近：

| 指标 | A1 | A1-ObsEqual |
|---|---:|---:|
| First-action MAE | 0.02564 | 0.02488 |
| Gripper MAE | 0.04787 | 0.04537 |
| Delta-action MAE | 0.01374 | 0.01392 |
| Teacher-forced Jerk | 0.01349 | 0.01353 |

但 closed-loop success 从：

$$
15\%
\rightarrow
45\%.
$$

### 13.2 两种评测观察的分布不同

Teacher-forced offline evaluation 测量：

$$
o_t^{\mathrm{expert}}
\rightarrow
\hat A_t.
$$

模型始终在 expert trajectory observation 上预测，不需要承担前一步错误产生的状态
偏移。

Closed-loop rollout 测量：

$$
o_t^\pi
\rightarrow
a_t
\rightarrow
o_{t+1}^\pi.
$$

Observation distribution 由策略自己的历史动作决定。小误差可能导致：

- 接触位置偏移；
- 抓取失败；
- 进入 demonstration 未覆盖的状态；
- release timing 偏移；
- 长时间振荡；
- 闭环固定点。

尤其 release 和 gripper switch 是低频关键事件，整体 MAE 很容易被大量普通移动帧
稀释。因此：

$$
\boxed{
\text{低 offline action error 不等于高 closed-loop success。}
}
$$

Offline metrics 仍然适合快速筛查拟合、比较动作值和变化误差，但不能替代正式
closed-loop rollout。

## 14. 被实验修正的五个猜想

### 14.1 “ACT 没有倒计时，所以 release 必然永远停在未来”

**修正：结论过强。**

ACT 没有显式计划进度，但若当前 observation 能反映状态进展，reactive policy 可以
通过状态触发 release。A1-VAE 在无 TE 条件下达到 19/20，说明显式倒计时不是此任务
成功的必要条件。

### 14.2 “Temporal Ensemble 主要靠保留旧 release 计划解决失败”

**修正：可能是部分机制，但不是唯一或必要机制。**

TE 确实保留旧预测影响、降低方差并平滑重规划，但 A1 失败还受到末尾 observation
权重、future chunk variation、确定性回归、动作变化和状态进展不足的影响。

### 14.3 “纯 BC 无法学会松开，必须加入 RL step penalty”

**修正：在当前任务上不成立。**

A1-VAE 使用纯监督训练达到 19/20，说明数据、loss 与模型表达充分时，BC 可以学会
release。RL 仍可用于惩罚停滞、优化完成时间、学习 recovery 和直接优化 outcome，但
不是解决当前 release 的必要条件。

### 14.4 “Delta-action loss 只会降低动作抖动”

**修正：作用更广。**

第 12.4 节的结果显示，A2 的主要变化不只体现在动作平滑度。更合理的解释是它改善了
action prefix 的轨迹形状和局部可执行性，减少状态漂移与停滞；teacher-forced 序列
统计与 closed-loop dynamics 不能被当成同一个目标。

### 14.5 “相似 observation 下存在多种 future chunk，确定性回归会折中”

**当前结果支持，但没有严格证明。**

A1-VAE 的显著提升与该解释一致，但还不能排除训练正则化、额外参数、latent noise、
privileged future-action information 或单 seed 波动。

## 15. 与 Diffusion Policy、Flow Matching、VLA、World Model 和 RL 的关系

### 15.1 Action distribution、时间融合与长期决策是不同问题

| 方法或组件 | 主要解决的问题 | 不自动提供什么 |
|---|---|---|
| ACT | 当前状态条件下的连续短期 action chunk | 环境 dynamics、长期 reward optimization |
| Diffusion / Flow Matching | 更强的多峰 action distribution 建模 | 计划倒计时、跨周期记忆 |
| Temporal Ensemble | 对同一绝对时刻的跨预测融合 | 正确任务目标与可执行 action prefix |
| World Model | 预测动作导致的未来状态 | 若无 reward/value/planning，不会自动选择好动作 |
| RL / Value learning | 依据长期 outcome 改进策略 | 不自动解决数据接口和动作表示 |

这些模块解决不同维度，不能简单排列成“后一种方法自动修复前一种方法的所有问题”。

### 15.2 ACT 与 Diffusion Policy / Flow Matching

ACT 与 [[Diffusion Policy 概述|Diffusion Policy]] 都预测未来动作序列：

$$
A_t=a_{t:t+H-1}.
$$

区别在于：

- ACT 用 Transformer policy 一次直接解码 chunk，CVAE latent 辅助训练；
- Diffusion Policy 通过迭代去噪表达条件 action distribution；
- Flow Matching 学习 action generation space 中的向量场：

  $$
  v_\theta(A^\tau,o,\tau).
  $$

这里的 $v_\theta$ 不是机器人真实速度，也不是任务进度或倒计时。生成式 action policy
改善的是 $p(A\mid o)$ 的表达能力，不会自动引入 previous chunk persistence 或历史
记忆。

[[RDT-1B|RDT-1B]] 将 diffusion action chunk 扩展为大规模双臂 foundation policy；
[[FAST_知识总结|FAST]] 则代表连续动作先经过频域压缩和离散 tokenization 的另一条
路线。

### 15.3 ACT 与 VLA

ACT 是单任务、小数据、高质量 demonstration 驱动的 visuomotor policy，不以语言指令
或大规模视觉语言预训练为核心。VLA 更关注跨任务语义条件、规模化数据和通用性。

但 ACT 对 VLA 有重要启发：

- 高频控制不一定要使用离散 action token；
- Action Chunk 比 single-step prediction 更适合表达短期协调；
- 数据采集系统会显著影响 policy 上限；
- 训练期 action distribution 建模和推理期闭环执行必须同时设计。

[[Pi_0机器人文章分析|pi0]] 等方法进一步把语言/视觉 backbone 与连续 action expert
结合起来。

### 15.4 World Model

World Model 学习：

$$
p(s_{t+1}\mid s_t,a_t).
$$

但预测未来状态本身不自动解决 release timing。真正用于决策通常需要：

$$
\text{World Model}
+
\text{Reward/Value}
+
\text{Planning}.
$$

系统才能比较：

```text
继续夹紧 → 状态停滞 → 低长期价值
合适位置释放 → 任务成功 → 高长期价值
```

同时，只有 recurrent latent state 或显式历史输入才可能保存跨时刻信息；仅编码当前
图像的 world representation 不天然具有时间记忆。

### 15.5 RL / Policy Gradient

RL 可以用 success reward、failure penalty、progress reward 或 step penalty 直接优化
长期 outcome。关键不是 Policy Gradient 这一种优化器，而是 outcome-based objective；
Q-learning、offline RL、value-guided decoding 和 model-based planning 也可以利用长期
回报。

本地结果说明纯 BC 足以解决当前单任务 release，但 RL 仍可能用于：

- 明确惩罚停滞；
- 优化完成时间；
- 学习失败恢复；
- 在 demonstration 之外优化 closed-loop success。

### 15.6 ACT 到现代 action modeling 的演化

前面的比较可以沿两条相互独立的主线整理。第一条主线解决的是多模态 action：同一个
observation 下不应被迫输出不同成功轨迹的平均值。

| 路线 | 生成自由度 | 主要作用 |
| --- | --- | --- |
| Deterministic BC | 单个回归点 | 简单，但容易 mode averaging |
| ACT + CVAE | 训练期 latent $z$ | 解耦 demonstration mode，并形成 canonical behavior |
| Diffusion Policy | noise 条件下的连续 chunk 生成 | 直接表达多峰 $p(A\mid o)$ |
| Flow Matching | 条件 vector field | 以 flow 生成连续 action chunk |
| Autoregressive action token | 离散 token 的条件概率 | 在离散空间中选择 mode，FAST 属于相关路线 |

对应的接口可以概括为：

$$
\text{ACT:}
\qquad
A^{GT}
\rightarrow
q_\phi(z\mid A^{GT},o)
\rightarrow
\pi_\theta(o,z),
$$

$$
\text{Diffusion / Flow:}
\qquad
\epsilon
\rightarrow
A,
\qquad
p(A\mid o),
$$

$$
\text{Autoregressive tokens:}
\qquad
p(\mathrm{token}_i\mid o,\mathrm{token}_{<i}).
$$

例如合法动作只有 $a=-1$ 和 $a=+1$ 时，点回归可能输出不存在的中间动作：

$$
\hat a_{\mathrm{MSE}}=0,
$$

而 categorical action model 可以保留：

$$
p(a=-1)=0.5,
\qquad
p(a=+1)=0.5,
$$

再从分布中选择一个 mode，而不是直接执行平均动作。

因此，ACT 的 CVAE 是一种间接的训练期多模态解耦机制；Diffusion、Flow Matching 和
autoregressive action tokens 则把生成自由度更直接地保留到 inference 阶段。这里的
演化不是简单的“后者替代前者”，而是从 latent-assisted regression 逐步转向显式
action-distribution modeling。

第二条主线解决的是时间一致性：

$$
\text{Action Chunk}
\rightarrow
\text{Temporal Ensemble}
\rightarrow
\text{Joint Chunk Generation + Receding Horizon}
\rightarrow
\text{Trajectory-conditioned Regeneration}.
$$

其中：

- Action Chunk 给出短期 trajectory supervision；
- Temporal Ensemble 对同一绝对时刻的重叠预测做融合；
- Joint Chunk Generation 与 Receding Horizon 生成内部一致的 chunk，并执行其 prefix；
- Trajectory-conditioned Regeneration 用已承诺 prefix 约束下一段 trajectory，减少 chunk
  boundary 的 mode switch。

所以 ACT 对后续 VLA 和生成式 action policy 的重要贡献，不是要求后续方法继续使用
CVAE 或 Temporal Ensemble，而是留下了两个更一般的原则：

$$
\boxed{
\text{Robot action 应以 trajectory / chunk 为单位建模，而不是把每个 timestep 当成独立预测。}
}
$$

$$
\boxed{
\text{当 }p(A\mid o)\text{ 多模态时，应保留 mode 选择自由度，而不是回归一个平均动作。}
}
$$

## 16. 完整流程、优势、局限与结论

### 16.1 训练流程

```text
ALOHA leader-follower teleoperation
→ 记录当前多视角图像、follower state 与 leader joint targets
→ 从轨迹滑动采样 (o_t, A_t)
→ Posterior encoder 读取 current joints + target chunk
→ 得到 diagonal Gaussian，并采样 z
→ Policy 读取 images + current joints + z
→ Transformer Action Queries 输出连续 action chunk
→ L1 reconstruction + KL
→ 可选 ObsEqual reduction 与 Delta-action loss
```

### 16.2 推理流程

```text
输入当前图像与 robot state
→ 丢弃 posterior encoder
→ 设 z = 0
→ 预测未来 H 步连续动作
```

之后可以选择：

```text
Receding-horizon：
执行前 n_action_steps
→ 获取新 observation
→ 重新预测
```

或：

```text
Temporal Ensemble：
每步重新预测
→ 对齐多个 chunk 对当前绝对时刻的动作
→ 指数加权融合后执行
```

### 16.3 ACT 的优势

- Future action supervision 提供短期轨迹结构；
- Action Chunk 缩短 effective decision horizon；
- 连续动作保留精细关节控制分辨率；
- CVAE 可减轻 human demonstration variation 带来的冲突；
- 推理结构相对直接，适合高频控制；
- 可通过重规划或 Temporal Ensemble 使用闭环视觉反馈。

### 16.4 ACT 的局限

- 基础 policy 主要依赖当前 observation，没有显式历史记忆；
- 不建模环境 dynamics；
- Behavior Cloning 不直接优化长期任务回报；
- 对 padding reduction、loss 权重、chunk horizon 和执行方式敏感；
- 长 chunk 在降低 effective horizon 的同时会损失反应性；
- Offline action metrics 与 closed-loop success 可能严重脱节。

### 16.5 最终理解

ACT 最合适的概括不是“Transformer 预测 20 或 100 个动作”，而是：

$$
\boxed{
\text{ACT}
=
\text{状态条件下的短期动作轨迹建模}
+
\text{闭环执行策略}.
}
$$

其闭环表现由以下因素共同决定：

1. observation 是否足以反映任务进度；
2. demonstration 中 future action chunk 是否具有较大 variation；
3. loss 是否公平覆盖关键 observation；
4. action prefix 是否局部可执行并推动真实状态；
5. `n_action_steps` 是否平衡反馈频率与计划持续性；
6. Temporal Ensemble 是否需要用于降低跨预测不一致；
7. 评测是否真正运行策略诱导的 closed-loop observation distribution。

论文和本地实验合起来给出的技术链是：

$$
\boxed{
\text{Action Chunking 学习短期轨迹结构；}
}
$$

$$
\boxed{
\text{Loss 设计决定哪些状态与变化模式受到重视；}
}
$$

$$
\boxed{
\text{CVAE 改善 action-chunk variation 的训练期表示；}
}
$$

$$
\boxed{
\text{Temporal Ensemble 改善跨时刻预测融合；}
}
$$

$$
\boxed{
\text{Closed-loop rollout 最终检验这些设计是否形成稳定策略。}
}
$$

## 相关笔记

- [[Diffusion Policy 概述|Diffusion Policy 概述]]：对比 action chunking、扩散策略和
  receding-horizon control。
- [[FAST_知识总结|FAST 知识总结]]：对比连续 action chunk 与离散 action tokenization。
- [[BestPractice/2026-07-24-ACT-A-Sanity收尾与后续实验指南|ACT A-Sanity 收尾与后续实验指南]]：
  五轨迹 sanity check 与后续实验设计。
- [[BestPractice/2026-07-25-ACT动作差分损失与公平评测|ACT 动作差分损失与公平评测]]：
  Delta-action loss 与同条件 closed-loop 比较。
- [[BestPractice/2026-07-27-ACT末尾样本等权与VAE实验结论|ACT 末尾样本等权与 VAE 实验结论]]：
  ObsEqual、CVAE、闭环指标和证据边界。
- [[Robot/WAM/WM-DAgger_论文阅读笔记|WM-DAgger 论文阅读笔记]]：
  用 action-conditioned world model 合成 OOD recovery data，补足普通 action-chunk BC 的闭环分布偏移。
