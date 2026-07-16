---
title: DreamerV3 技术报告
type: paper_note
topic: model_based_reinforcement_learning
status: mature
importance: high
updated: 2026-07-16
tags:
  - dreamerv3
  - world-model
  - model-based-rl
  - rssm
  - actor-critic
  - distributional-critic
  - reinforcement-learning
---

# DreamerV3 技术报告：基于世界模型的通用强化学习

> **报告范围**
>
> 基于 Hafner 等人的《Mastering Diverse Domains through World Models》批注版，重点吸收第 1、3-7 页的高亮内容。
>
> 综合对话中对 RSSM、KL balancing、reward/value、policy gradient、return normalization、straight-through estimator 与 critic replay loss 的逐步推导。
>
> 本报告是技术解读与实现分析，不包含新的实验结果。


版本 1.0

2026 年 7 月 14 日

原论文：Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap

## 摘要

DreamerV3 是一个基于离散潜变量世界模型的 model-based actor-critic。它从真实交互中学习 Recurrent State-Space Model（RSSM），显式预测潜状态转移、即时奖励、episode continuation 与观测重建，再从 replay state 出发在潜空间中生成 imagined trajectories，以 bootstrapped lambda-return 训练 distributional critic，并用统一的 REINFORCE 目标训练离散或连续 actor。论文真正的核心并非单一新模块，而是一组面向跨领域稳定性的组合设计：双向 stop-gradient 的 KL balancing、1 nat free bits、1% uniform mixture、symlog/symexp twohot、分位数 return normalization、critic EMA 与零初始化、AGC 和 LaProp。

本报告重点澄清五类常见误区：第一，reward model 与 value model 并非替代关系，前者描述环境的局部反馈，后者描述当前策略的长期回报；第二，h_t 预测的是 z_t 的分布而非具体 realization，z_t 表示当前观测带来的随机创新；第三，straight-through estimator 与 REINFORCE 都可处理离散采样，但前者是有偏低方差的 surrogate pathwise gradient，后者是无偏高方差的 score-function estimator；第四，critic replay loss 不要求 imagination 与 replay 轨迹逐步对齐，它仅用每个 replay state 的 imagination return 作为 bootstrap annotation；第五，Dreamer 的 return normalization 只缩小大信号而不放大小信号，从而在稀疏奖励尚未“触手可及”时让 entropy 保持相对主导。

> **一句话结论**
>
> DreamerV3 保留了 policy gradient 的基本优化逻辑，但把 actor-critic 的训练数据生成器从真实环境扩展为一个经过鲁棒训练的潜空间模拟器；其成功主要来自“稳定地学习和使用世界模型”，而不是把在线规划做得更复杂。


## 目录与阅读路线

> [[#1. 论文定位与核心贡献|1. 论文定位与高亮主线]]
>
> [[#2. 总体算法：真实交互与潜空间想象|2. 总体算法与真实/想象闭环]]
>
> [[#3. RSSM：确定性记忆与随机状态|3. RSSM：h、z、prior 与 posterior]]
>
> [[#4. 世界模型损失与离散潜变量梯度|4. 世界模型损失与离散潜变量梯度]]
>
> [[#5. Reward、Continuation 与 Distributional Critic|5. Reward、continuation 与 critic]]
>
> [[#6. Actor：离散与连续动作的 Policy Gradient|6. Actor：离散/连续 Policy Gradient]]
>
> [[#7. Return normalization、Entropy 与探索|7. Return normalization、entropy 与探索]]
>
> [[#8. 鲁棒预测、网络与优化器|8. 鲁棒预测、网络与优化器]]
>
> [[#9. 实验结果、消融与扩展性|9. 实验结果、消融与扩展性]]
>
> [[#10. 技术评价、局限与适用边界|10. 技术评价、局限与适用边界]]
>
> [[#11. 复现与调试检查表|11. 复现与调试检查表]]
>
> [[#附录 A：符号表|附录 A-C：符号、伪代码、常见误解]]

### 批注版高亮的技术映射

| **高亮页** | **高亮关键词**                                                                        | **本报告中的技术解释**                                             |
|------------|---------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| 页 1       | general algorithm；model of the environment；imagining future scenarios               | 统一配置、世界模型与想象训练是论文主命题。                         |
| 页 3       | discrete/stochastic representations；world model / critic / actor；signal magnitudes  | RSSM 结构及跨尺度鲁棒损失是算法核心。                              |
| 页 4       | reward、continuation；prediction / dynamics / representation loss                     | 世界模型不仅预测状态，还要给 imagined rollout 提供任务与终止语义。 |
| 页 5       | free bits；1% uniform mixture；abstract trajectories                                  | KL 约束必须避免信息坍塌和 categorical 数值尖峰。                   |
| 页 6       | imagined/replay critic loss；EMA；zero initialization；reward scale/frequency；P5-P95 | critic 与 actor 的稳定性依赖目标平滑和 return 尺度控制。           |
| 页 7       | symlog；symexp twohot；reward predictor and critic                                    | 用变换和分类式回归解耦目标数值大小与梯度大小。                     |

> *高亮内容来自用户提供的论文批注；为避免逐字重复，表中采用主题化概括。*

## 1. 论文定位与核心贡献

DreamerV3 面向的不是某一个 benchmark，而是强化学习算法在跨领域迁移时的“配置脆弱性”：视觉与向量输入、离散与连续动作、稀疏与密集奖励、2D 与 3D 环境，以及相差多个数量级的 reward/return，往往迫使研究者重新调节 loss scale、entropy coefficient、regularization 和 optimizer。论文试图用一组固定的核心超参数覆盖超过 150 个任务。/[1, pp. 1-2/]

其贡献可以分为三层。第一层是既有 Dreamer 范式：学习 latent world model，在模型中想象未来，再训练 actor 和 critic。第二层是世界模型表征：离散 stochastic latent、RSSM 和记忆状态。第三层，也是 V3 最关键的贡献，是一整套鲁棒性工程，使上述范式能够跨任务工作，而不依赖逐领域调参。

> **DreamerV3 不是什么**
>
> 不是执行时每一步都做树搜索或 MPC：环境交互时直接从 actor 采样动作。
>
> 不是只预测 value 的规划系统：world model 必须显式预测 reward 与 continuation。
>
> 不是完全脱离真实环境：真实观测仍逐步进入 posterior；想象主要用于训练而非替代感知。
>
> 不是严格意义上的“一切设置相同”：核心损失超参数固定，但模型规模、replay ratio、并行环境数和数据预算会随 benchmark 调整。


## 2. 总体算法：真实交互与潜空间想象

真实环境阶段仍遵循标准强化学习接口：actor 依据当前 latent state 选择动作，环境返回下一观测、真实奖励和终止标记，这些序列进入 replay buffer。训练阶段从 replay 中抽取长度为 T 的序列，首先更新世界模型；随后选择 replay 中的 latent state 作为 imagination 起点，由当前 actor 与世界模型生成 H 步潜空间轨迹，预测每步 reward 与 continuation，并由 critic 对 horizon 之外的未来进行 bootstrap。

因此，Dreamer 相对于经典 model-free policy gradient 的主要变化不是“PG 不再需要环境”，而是增加了一个可重复使用的模型数据源。同一条真实经验可以反复用于世界模型学习，并从多个起点、多个动作样本生成 imagined trajectories。其经济意义是用更多 GPU 计算换取更少或更高价值的真实交互；在机器人、复杂模拟器或高成本实验中，这种交换尤其重要。

> **模型偏差的代价**
>
> 想象轨迹可能并不存在于真实环境中，actor 还可能利用 world model 的系统性错误。Dreamer 用 replay state 作为真实锚点、采用较短 imagination horizon、持续加入新数据并依赖 critic bootstrap 缓解风险，但并未从理论上消除 model bias。


## 3. RSSM：确定性记忆与随机状态

### 3.1 时间顺序与动作依赖

| h_t = f_phi(h/_{t-1}, z/_{t-1}, a/_{t-1}) |
|-------------------------------------------|

| z_t ~ q_phi(z_t /| h_t, x_t) /[真实观测 / posterior/] |
|-------------------------------------------------------|

| z_hat_t ~ p_phi(z_t /| h_t) /[想象 / prior/] |
|----------------------------------------------|

Dynamics predictor 表面上只以 h_t 为条件，但动作并未缺失：a/_{t-1} 已经通过 sequence model 写入 h_t。时间索引上，a/_{t-1} 造成从时刻 t-1 到 t 的转移；当前动作 a_t 影响的是 h/_{t+1} 和 z/_{t+1}。

### 3.2 为什么既需要 h_t，又需要 z_t

h_t 是确定性的历史摘要，可理解为“在看到当前 x_t 之前，根据过去对现在形成的预测上下文”；z_t 是当前时刻的 stochastic innovation，表示当前观测揭示的具体随机分支、隐藏变量或不可由历史唯一决定的信息。h_t 能输出 p(z_t/|h_t)，但只能给出一个分布，不能确定本次 realization。

> **例子：开门后的随机分支**
>
> 历史和动作只告诉模型“门已打开”，prior 可能给出 60% 有敌人、40% 无敌人。
>
> 当前图像 x_t 显示实际有敌人，posterior 因此把 z_t 更新到“有敌人”分支。
>
> 随后 z_t 与动作 a_t 一起进入下一次 recurrent update，使新信息写入 h_{t+1}。


### 3.3 z_t 是否真的是采样结果

是。q_phi(z_t/|h_t,x_t) 与 p_phi(z_t/|h_t) 是概率分布；z_t 是从其中采样出的具体离散表示。DreamerV3 使用“多个 categorical distribution 的向量”，每个 categorical 采样一个 one-hot，最后拼接为 stochastic state。这样的表示既保留多模态能力，又能在 imagination 中快速采样。

## 4. 世界模型损失与离散潜变量梯度

| L_world = E_q /[ sum_t ( L_pred + L_dyn + 0.1 L_rep ) /] |
|----------------------------------------------------------|

### 4.1 Prediction loss：重建、奖励与 continuation

| L_pred = -log p_phi(x_t /| h_t,z_t) - log p_phi(r_t /| h_t,z_t) - log p_phi(c_t /| h_t,z_t) |
|---------------------------------------------------------------------------------------------|

这三个负对数似然共享“最大化条件概率”的形式，但实际分布与 loss 不必相同：向量观测可在 symlog 空间使用 squared error；reward 与 critic 最终使用 symexp twohot categorical loss；continuation 使用 Bernoulli logistic loss。统一写成 -log p 的好处，是把不同输出头都解释为条件概率模型。

### 4.2 为什么 Gaussian NLL 等价于 MSE

| p(x/|s) = Normal(x; mu_phi(s), sigma^2 I) |
|-------------------------------------------|

| -log p(x/|s) = D/2 log(2/*pi/*sigma^2) + 1/(2/*sigma^2) /|/|x-mu_phi(s)/|/|/_2^2 |
|----------------------------------------------------------------------------------|

当方差 sigma^2 固定时，第一项与参数无关，第二项只是平方误差的固定倍数，因此最小化 Gaussian negative log-likelihood 与最小化 MSE 有相同最优解。若 sigma 也由网络预测，则还会出现 log sigma 项，loss 不再等价于普通 MSE。

### 4.3 Dynamics loss 与 Representation loss 为何拆开

| L_dyn = max(1, KL( sg(q_phi(z_t/|h_t,x_t)) /|/| p_phi(z_t/|h_t) )) |
|--------------------------------------------------------------------|

| L_rep = max(1, KL( q_phi(z_t/|h_t,x_t) /|/| sg(p_phi(z_t/|h_t)) )) |
|--------------------------------------------------------------------|

两项实际上在同一个总 loss 中一起优化；区别在于梯度流向。L_dyn 固定 posterior，把它当作较有信息的 teacher，训练 prior/sequence model 去预测真实表示。L_rep 固定 prior，只以较小权重要求 encoder 产生更可预测的表示。若只用一个普通 KL 同时更新 q 和 p，两边可能一起向一个“很容易一致但几乎不含观测信息”的退化分布移动。

> **为什么权重不对称**
>
> beta_dyn = 1：主要责任在 dynamics，要求它追上包含真实观测的 posterior。
>
> beta_rep = 0.1：posterior 只需适度迁就 dynamics，避免为了可预测性而丢失关键细节。


### 4.4 Free bits 与 1% Unimix

Free bits 使用 max(1 nat, KL)。当 KL 已低于 1 nat 时，该项变成常数，梯度为零，模型不再被迫继续压缩表示；这为当前观测中暂时不可预测但有用的信息保留容量。1% unimix 则把 categorical 分布设为 0.99·network_softmax + 0.01·uniform，避免概率精确变为 0 或 1，从而减少无限 log probability、KL spike 与过早确定化。/[1, pp. 4-5/]

### 4.5 Straight-through estimator 与 REINFORCE 的比较

| z_ST = z_hard + p - sg(p) |
|---------------------------|

前向传播时 p-sg(p)=0，因此下游网络看到真实 one-hot sample；反向传播时 sg(p) 没有梯度，于是梯度像经过连续 probability p 一样传回 softmax logits。它不是真实离散采样梯度，而是人为定义的 surrogate。

| **维度**       | **Straight-through estimator**  | **REINFORCE / score function**      |
|----------------|---------------------------------|-------------------------------------|
| 梯度路径       | 利用下游 dL/dz，经 softmax 回传 | 用 (L-b)·grad log q(z) 更新采样概率 |
| 偏差           | 有偏                            | 理论上无偏                          |
| 方差           | 通常较低                        | 通常较高                            |
| 要求           | 下游模型必须可微                | 下游可不可微均可                    |
| 信用分配       | 逐维、局部梯度更细              | 整体 scalar loss，更粗              |
| DreamerV3 用途 | 离散 latent z                   | actor 的离散和连续动作              |

## 5. Reward、Continuation 与 Distributional Critic

### 5.1 为什么必须显式预测 reward，而不能只预测 value

传统 model-free policy gradient 通常不训练 reward network，不是因为 reward 难学，而是因为真实环境已经直接返回 r_t；长期 value 无法被立即观测，才需要 critic 估计。Dreamer 的 imagined trajectory 没有真实环境反馈，因此若不显式建模 r_t，就无法判断一条潜空间轨迹好坏。

Reward 与 value 的职责也不同。reward 是环境的局部、相对策略无关的任务反馈；V^pi(s) 是在当前策略下未来累计回报的期望，随着 actor 更新而变化。一个 state value 只给出“按当前策略平均有多好”，不能替代动作条件下逐步发生的转移与即时反馈。Dreamer 因而采用“短期 reward model + horizon 末端 value bootstrap”的组合。

### 5.2 为什么还要预测 continuation c_t

在真实环境中，done/terminal 直接可见；在 imagination 中，模型必须自己判断 episode 是否仍然有效。continuation c_t 进入有效折扣 gamma·c_t：若角色死亡、任务失败或成功终止，c_t=0 会切断后续 bootstrap，防止模型想象“死亡后继续挖矿”或“物体掉出桌面后仍有未来价值”。

| R_t^lambda = r_t + gamma c_t /[ (1-lambda) v/_{t+1} + lambda R/_{t+1}^lambda /], R_T^lambda = v_T |
|---------------------------------------------------------------------------------------------------|

### 5.3 Return、return distribution 与 value

单条轨迹给出一个具体 return realization；在同一状态下多次 rollout，会形成条件 return distribution。通常意义的 value 是该分布的期望：V^pi(s)=E/[R/|s/]。因此论文说 critic “学习 return distribution”并不意味着它不学习 value；它学习的是更完整的 p_psi(R/|s)，再用分布期望作为标量 v_t。

| v_t = E/_{R ~ v_psi(./|s_t)} /[R/] |
|------------------------------------|

DreamerV3 用指数间隔的 symexp bins 和 twohot target 训练 critic。这样梯度主要取决于分类概率误差，而不直接随 return 数值从 1 增长到 100000，适合跨领域统一超参数。分布表达还可以保留多峰结构，但 actor 和 bootstrap 主要使用其期望，因此算法并非显式风险敏感。

### 5.4 Critic EMA、零初始化与主/辅损失

- 主要 imagined critic loss：在当前 actor 的 imagined trajectories 上学习，权重 beta_val=1。

- critic replay loss：把 replay 的真实 reward 直接传播给 critic，权重 beta_repval=0.3。

- EMA critic regularizer：让当前 critic 靠近慢速参数副本，降低自举目标漂移。

- reward head 和 critic 输出层零初始化：避免训练初期凭空“幻觉”出很大的 reward/value。

### 5.5 Critic replay loss 的精确语义

从 replay trajectory 的每个状态 s_t^R 分别启动一条由当前 actor 驱动的 imagination rollout，并取该 rollout 起点的 imagined lambda-return U_t，作为这个 replay state 的“当前策略价值注释”。随后沿原 replay trajectory 使用真实环境奖励 r_t^R，再计算一个辅助 lambda-return。两条轨迹只共享起始状态，imagination 的中间状态、动作与 reward 不需要也不能和 replay 逐步对齐。

| G_t^rep = r_t^R + gamma (1-d_t) /[ (1-lambda) U/_{t+1} + lambda G/_{t+1}^rep /] |
|---------------------------------------------------------------------------------|

> **严格性说明**
>
> replay 的前若干动作来自历史 behavior policy，而 U_t 来自当前 actor；因此整个 replay target 不是严格 on-policy value target。
>
> 论文/实现没有使用 importance sampling、Retrace 或 V-trace 修正，故该辅助项具有 off-policy bias。
>
> 其收益是当 reward model 漏掉稀疏奖励时，真实 replay reward 仍能直接训练 critic；这是一个低权重的偏差-监督质量权衡。


## 6. Actor：离散与连续动作的 Policy Gradient

| L_actor = - sum_t sg(A_t / max(1,S)) log pi_theta(a_t/|s_t) - eta H(pi_theta(./|s_t)) |
|---------------------------------------------------------------------------------------|

| A_t = R_t^lambda - v_psi(s_t) |
|-------------------------------|

DreamerV3 对离散与连续动作统一采用 REINFORCE。动作样本被视为常量，梯度不穿过“采样结果”，而是通过 log pi_theta(a/|s) 回到 policy 参数。critic 足够准确并不会让所有 advantage 都为零：V(s) 是按当前策略对动作取平均的期望，不同动作仍分别具有正负 A(s,a)；只有策略接近局部最优时，常选最优动作的 advantage 才自然趋小。

### 6.1 离散动作：categorical policy 的具体例子

假设 actor 输出 4 个 logits，经 softmax 得到 p=/[0.1,0.2,0.3,0.4/]，本次采到第 2 个动作，advantage A=-2。loss 不是 logits·/[0,-2,0,0/]，也不是 -2·p_2，而是：

| L = -A log p_2 = 2 log(0.2) |
|-----------------------------|

| dL/dlogits_j = -A ( 1/[j=a/] - p_j ) = /[-0.2, 1.6, -0.6, -0.8/] |
|------------------------------------------------------------------|

梯度下降会降低被选中的第 2 个 logit，并相对提高其他动作概率，符合负 advantage 的含义。若 A/>0，方向完全相反。

### 6.2 连续动作：对角 Gaussian policy

| pi(a/|s) = Normal(a; mu(s), diag(sigma(s)^2)), tau = log sigma |
|----------------------------------------------------------------|

| L = -A sum_i log Normal(a_i; mu_i, sigma_i^2) |
|-----------------------------------------------|

| dL/dmu_i = -A (a_i-mu_i)/sigma_i^2 |
|------------------------------------|

| dL/dtau_i = -A /[ (a_i-mu_i)^2/sigma_i^2 - 1 /] |
|-------------------------------------------------|

连续变量中的 p(a) 是概率密度。负 advantage 会通过移动均值和调整方差降低本次 sampled action 的密度；正 advantage 则提高其密度。整条多维动作向量共享一个 advantage，联合 log probability 等于各维 log density 之和。

### 6.3 为什么不通过 dynamics model 对 actor 反向传播

另一种做法是让梯度沿 action -/> predicted state -/> predicted reward -/> return 反传，即 dynamics gradient。DreamerV3 选择 score-function estimator，论文没有对所有原因做完整消融；从算法目标和数值性质看，主要优势包括：

- 统一处理 categorical 与 Gaussian action，不要求动作可微。

- 不依赖 world model 对动作的局部导数是否真实，减少 actor 利用模型错误梯度的风险。

- 避免通过多步 stochastic dynamics 的长链反传，降低梯度爆炸/消失与实现复杂度。

- 代价是 REINFORCE 方差通常更高，需要 critic baseline、return normalization 和足够的 batch。

> **逻辑上的正确理解**
>
> actor 更新时把 imagined reward/return 当作近似模拟器给出的样本结果，而不是宣称这些结果等同于真实环境。梯度 stop 在 advantage 上，只根据“这次动作样本比平均好还是差”更新概率。


## 7. Return normalization、Entropy 与探索

### 7.1 P5-P95 范围与 EMA

| Delta_k = Percentile_95(R^lambda_batch) - Percentile_5(R^lambda_batch) |
|------------------------------------------------------------------------|

| S_k = 0.99 S/_{k-1} + 0.01 Delta_k |
|------------------------------------|

单个 R_t^lambda 是标量，但一次 imagination batch 会产生 B×H 个 return，分位数是在这组数上计算。用 5% 到 95% 而非 min/max，是为了忽略随机环境中的极端 episode；EMA 使 S 随 batch 缓慢更新，避免 actor 梯度尺度因单批数据突变。S 是运行统计量，不通过梯度学习。

### 7.2 为什么分母必须有下限 1

标准 advantage normalization 把每个 batch 的 advantage 强制到近似零均值、单位标准差。即使原始 A 只有 10^-3、主要来自 critic 或 world model 噪声，它也会被放大到 O(1)，从而给“最大化当前估计回报”固定强度的权重。此时奖励实际上还没有被当前策略或其邻域行为发现，即论文所说 rewards are not within reach。

Dreamer 使用 A/max(1,S)：当 return range 小于 1 时，不把小信号放大，entropy regularizer 因而相对更重要，继续推动广泛探索；当奖励已经可达、return range 变大时，再缩小大梯度以稳定训练。若环境把所有 reward 乘以 1000，A 与 S 同时放大约 1000 倍，二者比值基本不变，因此固定 entropy coefficient eta=3e-4 仍可跨 reward scale 使用。

> **“奖励尺度”与“奖励频率”是两件事**
>
> 尺度：同一任务把 reward 从 1 改成 1000，不应改变探索/利用平衡。
>
> 频率：稀疏奖励早期几乎没有可靠 policy signal，此时不应把微小估计噪声标准化为单位强度。
>
> Dreamer 的 denominator limit 同时处理二者：大信号归一化，小信号保持小。


## 8. 鲁棒预测、网络与优化器

### 8.1 Symlog 与 Symexp

| symlog(x) = sign(x) log(/|x/|+1) |
|----------------------------------|

| symexp(x) = sign(x) ( exp(/|x/|)-1 ) |
|--------------------------------------|

symlog 在零附近近似恒等映射，对大正数和大负数进行对称对数压缩。它避免普通 log 无法处理负值，也避免运行均值/方差 normalization 给优化目标引入额外非平稳性。Dreamer 对向量 observation 的 encoder input 与 decoder target 使用 symlog。

### 8.2 Symexp twohot

reward 与 return 可能随机、多峰且跨度巨大。Dreamer 在 symlog 坐标上建立等距 logits，再经 symexp 映射为指数间隔 bins。目标标量只在最近两个 bins 上分配线性权重，形成 twohot soft label；cross-entropy 梯度取决于预测概率而非 bin 的绝对数值，因而能解耦 target magnitude 与 gradient magnitude。

| y_hat = softmax(f(x))^T B, B = symexp(/[-20, ..., 20/]) |
|---------------------------------------------------------|

| L_twohot = - twohot(y)^T log softmax(f(x)) |
|--------------------------------------------|

### 8.3 主要实现配置

| **类别**     | **参数**               | **论文值**       | **作用**                                   |
|--------------|------------------------|------------------|--------------------------------------------|
| 通用         | Replay capacity        | 5×10^6           | 增加数据覆盖；配合 online queue            |
| 通用         | Batch / length         | 16 / 64          | 序列训练 world model                       |
| 通用         | Learning rate          | 4×10^-5          | 跨任务固定                                 |
| 优化         | AGC                    | 0.3              | 按参数张量范数自适应裁剪                   |
| 优化         | LaProp                 | eps=10^-20       | 先 RMS 归一化再 momentum                   |
| 世界模型     | beta_pred / dyn / rep  | 1 / 1 / 0.1      | 重建与可预测性平衡                         |
| 世界模型     | Free nats / Unimix     | 1 / 1%           | 避免过压缩与 KL spike                      |
| Actor-Critic | Imagination horizon    | 15               | 控制模型误差累积                           |
| Actor-Critic | Discount / lambda      | 0.997 / 0.95     | 有效 horizon 约 333                        |
| Critic       | imag / replay loss     | 1 / 0.3          | 主 on-policy imagination + 辅助真实 reward |
| Actor        | Entropy coefficient    | 3×10^-4          | 结合 RetNorm 跨领域使用                    |
| RetNorm      | P95-P5 / decay / limit | range / 0.99 / 1 | 稳健尺度与稀疏奖励探索                     |

> *主要数值来自 /[1, Table 4, p. 21/]。模型大小和 replay ratio 会按 benchmark 调整。*

### 8.4 网络与数据工程

- 图像使用 stride-2 CNN 编码与反卷积解码；向量输入使用 3 层 MLP。

- sequence model 为 8-block block-diagonal GRU，在增加记忆单元时避免完全二次参数增长。

- actor/critic 为 3 层 MLP；reward/continue 为较浅预测头；RMSNorm + SiLU。

- uniform replay + online queue；存储并更新 latent state，以便从 replay context 初始化。

- replay ratio 定义每收集一个环境步训练多少时间步；提高 replay ratio 用更多计算换更高数据效率。

## 9. 实验结果、消融与扩展性

### 9.1 跨领域结果摘要

| **领域**       | **规模/预算**    | **DreamerV3**                         | **强比较项**                    | **解读与限制**                             |
|----------------|------------------|---------------------------------------|---------------------------------|--------------------------------------------|
| Atari          | 57 tasks / 200M  | Gamer mean 3381%，median 830%         | MuZero 3054% / 693%             | 同数据预算；个别游戏仍落后                 |
| ProcGen        | 16 / 50M         | Normalized mean 66.01                 | PPG 64.89；PPO 42.80            | 平均领先很小，任务间差异大                 |
| DMLab          | 30 / 100M        | Human mean capped 71.4%               | IMPALA 1B: 66.3%；PPO 35.9%     | 强基线多用 10× 数据                        |
| Minecraft      | 1 / 100M         | Return 9.1；所有训练 agent 曾找到钻石 | IMPALA 7.1                      | 最终钻石 episode 成功率仅约 0.4%           |
| Atari100k      | 26 / 400K        | Mean 125%，median 49%                 | IRIS 105% / 29%；TWM median 51% | EfficientMuZero 改变协议，不能直接等价比较 |
| Proprio        | 18 / 500K        | Task mean 871                         | DMPO 801                        | 12M 模型即可达到强结果                     |
| Visual control | 20 / 1M          | Task mean 861                         | DrQ-v2 770                      | 无专门数据增强仍领先平均分                 |
| BSuite         | 23 task families | Task mean 66%，category mean 63%      | Boot DQN 60% / 57%              | 探索类别仅 0.01，Deep Sea 仍失败           |

> *数据汇总自 /[1, pp. 24-38/]；不同 benchmark 指标不可横向直接比较。*

### 9.2 消融：真正支撑性能的是什么

平均结果显示，每个鲁棒性技术只在部分任务上“决定生死”，但组合后形成跨领域稳定性。其中 KL balance + free bits 的平均影响最大，return normalization 与 symexp twohot 也明显重要。更关键的是 learning signal 消融：去掉 reward/value 对 representation 的梯度，整体性能只温和下降；去掉 reconstruction gradients 则大幅崩溃。这表明 Dreamer 的 latent 主要由 task-agnostic reconstruction 学成，而非只靠任务奖励塑形。

### 9.3 Scaling

模型参数从约 12M 扩展到 400M 时，性能总体单调提高，且更大的模型往往需要更少环境交互达到同等分数。提高 replay ratio 同样提升数据效率。这为实践提供了较清晰的 compute-data trade-off：当真实交互昂贵时，可以增加模型容量和每条数据的更新次数。

### 9.4 Minecraft 结果应如何准确表述

> **结果很强，但不能简化成“纯像素、只有钻石奖励、稳定通关”**
>
> 观测不仅有 64×64 RGB，还包含 400+ 物品库存向量、历史最大库存、装备 one-hot，以及 health/hunger/breath。
>
> 环境提供 abstract crafting action 和 flat categorical action space。
>
> 奖励包含通往钻石的 12 个 milestone，每项每 episode 奖励一次，并非只有最终钻石奖励。
>
> 100% 的训练 agent 在整个训练期间至少获得过一次钻石；在 100M 步附近，单 episode 钻石成功率约 0.4%。


## 10. 技术评价、局限与适用边界

### 10.1 Dreamer 是对 Policy Gradient 的什么改进

Dreamer 的 actor 仍然使用 policy gradient；真正变化的是 state representation、训练数据来源和长期目标构造。Model-free PG 的样本必须来自真实环境，而 Dreamer 允许从真实 replay state 出发，在 learned dynamics 中生成反事实动作与未来结果。由此它更准确地属于“model-based actor-critic with latent imagination”，而不是一种完全不同于 PG 的优化原理。

### 10.2 主要优势

- 数据效率：重复利用真实轨迹，并以想象生成大量 actor/critic 训练样本。

- 统一性：同一 actor objective 覆盖离散与连续动作；symlog/twohot/RetNorm 覆盖多种数值尺度。

- 表征能力：RSSM 同时具有历史记忆与多模态随机 latent。

- 可扩展性：模型规模和 replay ratio 增长带来较可预测的性能收益。

### 10.3 主要局限

- Model bias：policy 可能利用 dynamics/reward predictor 的错误。

- 计算成本：高性能实验常使用 200M 参数模型和单张 A100 数日。

- 重建目标可能学习任务无关细节；尽管消融证明其重要，但不等于它在所有环境中最优。

- critic replay loss 有 off-policy bias；论文将其作为低权重辅助项而非严格策略评估。

- 探索并未普遍解决：BSuite Deep Sea/Deep Sea Stochastic 仍为 0。

- “固定超参数”主要指核心算法配置，benchmark 级计算预算与模型大小仍不同。

### 10.4 适用性判断

| **场景**                         | **DreamerV3 适用性** | **原因**                             |
|----------------------------------|----------------------|--------------------------------------|
| 真实交互昂贵、可离线反复训练     | 高                   | 用计算换真实样本                     |
| 高维视觉、部分可观测、需要记忆   | 高                   | RSSM 与重建表征有优势                |
| 奖励尺度差异大、任务族多样       | 高                   | symlog/twohot/RetNorm 针对该问题设计 |
| 环境步极便宜、状态低维、任务简单 | 中                   | SAC/PPO 等可能更简单、开发成本更低   |
| 模型误差不可接受的安全关键控制   | 谨慎                 | 需要 uncertainty、约束或真实验证机制 |
| 纯探索难题、无任何中间信号       | 不保证               | 论文自身在 BSuite Deep Sea 上失败    |

## 11. 复现与调试检查表

| **模块**            | **检查项**                                                                                                | **常见错误**                      |
|---------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------|
| 时间对齐            | 确认 a/_{t-1} 进入 h_t；a_t 影响下一状态；reward/terminal 的索引与 transition 一致。                      | 错一位会让模型学习错误动力学      |
| Reset 处理          | episode 首状态正确重置 recurrent carry；terminal 与 time-limit truncation 区分。                          | 把 time-limit 当真正 terminal     |
| Posterior/Prior     | 真实序列用 q(z/|h,x)，imagination 只能用 p(z/|h)。                                                        | 想象时偷看真实 x                  |
| ST 采样             | 前向是 hard one-hot，反向沿 soft probability；不要误用 argmax 后完全断梯度。                              | 离散 latent 无法端到端训练        |
| KL 方向             | L_dyn stop q；L_rep stop p；权重 1 与 0.1；free bits 小于阈值时梯度为零。                                 | 两个 KL 梯度方向写反              |
| Unimix              | encoder、dynamics categorical 与离散 actor 都避免精确零概率。                                             | KL/日志出现 inf 或尖峰            |
| Reward/Value 初始化 | twohot 输出层零初始化，检查初期预测均值接近 0。                                                           | 初期 hallucinated reward          |
| Continuation        | imagined return 使用 gamma/*c，终止后不 bootstrap。                                                       | 死亡后仍累计价值                  |
| Twohot              | 目标落在相邻两 bin；期望值计算注意正负大数求和顺序。                                                      | 大 target 导致梯度尺度异常        |
| Critic target       | lambda-return 的 horizon 末端用 value bootstrap；target 对 actor/critic 参数 stop-gradient。              | 把 sampled return 当可反传路径    |
| Slow critic         | EMA 更新顺序稳定；不要把慢网络误当成唯一当前 value。                                                      | 目标网络与当前网络混淆            |
| RetNorm             | P5-P95 在整个 return batch 上算；S 用 EMA；分母至少 1。                                                   | 稀疏奖励噪声被放大                |
| Actor loss          | 使用 -A log pi - eta H；离散和连续动作都检查 joint log-prob 形状。                                        | 把 A 乘动作或概率而非 log-prob    |
| Replay critic       | 每个 replay state 独立取得 imagination U_t；不要试图对齐两条未来轨迹；承认其 off-policy bias。            | 错误拼接 imagined/replay reward   |
| Replay ratio        | 区分“训练时间步/环境时间步”和“梯度步/环境步”，考虑 action repeat 与 batch length。                        | 误估实际更新强度                  |
| 诊断指标            | 监控 KL、free-bit 命中率、reward/value 预测、continuation、return range S、entropy、model rollout drift。 | 只看最终 return，错过模型崩溃先兆 |

## 附录 A：符号表

| **符号**            | **含义**                               |
|---------------------|----------------------------------------|
| x_t                 | 真实观测；图像或向量                   |
| h_t                 | 确定性 recurrent state，历史预测上下文 |
| z_t                 | 离散 stochastic latent，当前随机创新   |
| s_t={h_t,z_t}       | actor/critic 使用的 model state        |
| q_phi(z_t/|h_t,x_t) | posterior / encoder distribution       |
| p_phi(z_t/|h_t)     | prior / dynamics predictor             |
| r_t                 | 即时 reward；真实或模型预测            |
| c_t                 | episode continuation，0 表示终止       |
| R_t^lambda          | bootstrapped lambda-return sample      |
| v_psi(R/|s)         | critic 输出的 return distribution      |
| v_t                 | critic 分布的期望，即 scalar value     |
| S                   | P95-P5 return range 的 EMA             |
| sg(.)               | stop-gradient                          |
| H                   | imagination horizon，论文默认 15       |

## 附录 B：训练伪代码

```text
初始化 world model phi、actor theta、critic psi、slow critic、Replay Buffer
循环：
1. 用 actor 与真实环境交互，存储 (x, a, r, terminal)
2. 从 replay 采样 B 条、长度 T 的序列
3. Posterior rollout：
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
z_t ~ q_phi(z_t | h_t, x_t)
4. 更新 world model：
L_pred + L_dyn + 0.1 L_rep
5. 从 replay latent states 选择 imagination starts
6. 对每个 start，用当前 actor 和 prior rollout H 步：
a_t ~ pi_theta(. | s_t)
z_{t+1} ~ p_phi(. | h_{t+1})
预测 r_t, c_t, value distribution
7. 计算 imagined lambda-return R_t^lambda
8. 更新 critic：distributional loss + slow critic regularization
9. 更新 actor：-sg((R^lambda-v)/max(1,S))*log pi - eta*entropy
10. 可选 replay critic loss：
每个 replay state 取 imagination 起点 return U_t
沿 replay 真实 reward 计算辅助 lambda-return
11. 更新 EMA critic 与 return-range statistic S
```


## 附录 C：常见误解速查

> **“传统 PG 不学 reward，所以 reward 比 value 更难。”**
>
> 错误。传统 PG 已从环境直接观察 reward；value 才是需要估计的长期期望。


> **“critic 足够准，A=R-V 就总是接近 0。”**
>
> 错误。V 是动作平均；不同动作的 Q-V 仍可显著正负。


> **“logits 乘 one-hot advantage 就是离散 PG。”**
>
> 错误。核心是 -A log softmax(logits)[a]。


> **“连续 PG 的 loss 是 -A·p(a)。”**
>
> 错误。核心是 -A log density；梯度由 Gaussian log-likelihood 回到 mu 和 log sigma。


> **“h_t 已能预测 z_t，所以 z_t 多余。”**
>
> h_t 只预测分布；z_t 表示本次具体随机分支和当前观测创新。


> **“Dynamics predictor 没有 action 输入。”**
>
> action 已通过 recurrent transition 吸收进 h_t。


> **“Free bits 把 KL 最大限制为 1 nat。”**
>
> 相反，它在 KL 小于 1 nat 时停止继续压缩。


> **“Dreamer 执行时每一步做规划。”**
>
> 论文设置中执行直接采样 actor；想象主要用于训练。


> **“Replay critic 的 imagination 与 replay 轨迹应逐步对齐。”**
>
> 不需要；每个 replay state 只取得一个当前策略 bootstrap annotation。


> **“Minecraft 已稳定学会获取钻石。”**
>
> 所有 agent 曾成功，但最终单 episode 钻石率约 0.4%，且有结构化观测、里程碑奖励和抽象 crafting action。


## 结论

DreamerV3 的技术意义在于，它把“学习一个可想象未来的世界模型”从容易失稳的研究原型推进为跨领域可扩展的 actor-critic 框架。RSSM 负责把历史和当前随机创新分开表示；prediction、dynamics 与 representation losses 共同塑造既有信息又可预测的 latent；reward 与 continuation 使 imagined trajectory 具有任务和终止语义；distributional critic、twohot 和 lambda-return 提供长期目标；REINFORCE、分位数 return normalization 与 entropy 共同处理离散/连续动作及稀疏奖励探索。

同时，正确理解其边界同样重要：Dreamer 没有消除 model bias，critic replay loss 不是严格 on-policy，统一配置仍伴随 benchmark 级计算选择，Minecraft 成果也依赖具体环境接口和中间奖励。将这些机制与限制同时纳入，才能把论文从“结果展示”转化为可复现、可迁移的技术方法。

## 相关笔记

- [[Dreamer技术报告|Dreamer]]：DreamerV3 的潜空间想象与 actor-critic 基础。
- [[DayDreamer论文综述与阅读重点|DayDreamer]]：DreamerV2 在真实机器人在线学习中的部署路线。
- [[PlaNet 论文概述|PlaNet]]：RSSM 与 latent planning 的前序工作。
- [[PPO|PPO]]：理解 DreamerV3 中 REINFORCE、advantage 与 entropy 的策略梯度背景。
- [[DreamerV4_技术报告|Dreamer 4]]：将 Dreamer 系列扩展到高容量生成式视频世界模型与离线 imagination training。

## 参考资料

/[1/] Hafner, D.; Pasukonis, J.; Ba, J.; Lillicrap, T. Mastering Diverse Domains through World Models. arXiv:2301.04104v2, 2024.

/[2/] DreamerV3 official implementation. danijar/dreamerv3, dreamerv3/agent.py, commit e3f02248693a79dc8b0ebd62c93683888ddaccfe（用于核对 critic replay loss 的代码语义）。

/[3/] Sutton, R. S.; Barto, A. G. Reinforcement Learning: An Introduction, 2nd ed., MIT Press, 2018（lambda-return 与 policy gradient 背景）。

> *本报告依据《Mastering Diverse Domains through World Models》的论文正文与批注内容整理。*
