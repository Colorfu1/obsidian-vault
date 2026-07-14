---
title: UniPi 技术总结
type: paper_note
topic: video_planning
status: mature
importance: high
updated: 2026-07-14
tags:
  - unipi
  - video-generation
  - video-diffusion
  - video-planning
  - inverse-dynamics
  - imitation-learning
  - robotics
---

# UniPi 技术总结：用文本引导视频生成学习通用策略

> 论文：**Learning Universal Policies via Text-Guided Video Generation**
> 作者：Yilun Du 等
> 会议：NeurIPS 2023
> 关键词：Video Diffusion、Video-as-Policy、UPDP、Inverse Dynamics、Imitation Learning、Generalist Policy

---

## 摘要

UniPi 提出了一种不同于传统强化学习和动作空间策略学习的决策框架：模型不直接从当前状态预测机器人动作，而是根据**当前视觉观察**和**文本任务描述**，先生成一段展示任务如何被完成的未来视频，再通过逆动力学模型把相邻视频帧转换成底层机器人动作。

其核心流程为：

$$
\text{当前图像 }x_0+\text{文本目标 }c
\;\longrightarrow\;
\text{未来视频轨迹 }x_{1:H}
\;\longrightarrow\;
\text{动作序列 }a_{0:H-1}
$$

这一设计把图像序列作为跨任务、跨环境和潜在跨机器人形态的中间接口，并允许模型利用大规模互联网图像—文本和视频—文本数据进行预训练。

UniPi 的核心价值不在于提出新的扩散模型，而在于提出了一种新的**策略表示方式**：

> 将策略表示为“目标条件下的未来视频”，而不是直接表示为动作映射。

从严格算法分类看，论文中的 UniPi 更接近**生成式模仿学习与视频规划**，而不是传统意义上的强化学习。

---

## 1. 问题背景

### 1.1 传统 MDP 框架面临的统一性问题

强化学习通常把任务建模为马尔可夫决策过程：

$$
\mathcal M=\langle \mathcal S,\mathcal A,T,R,\gamma\rangle
$$

其中：

- \(\mathcal S\)：状态空间；
- \(\mathcal A\)：动作空间；
- \(T(s'|s,a)\)：环境动力学；
- \(R(s,a)\)：奖励函数；
- \(\gamma\)：折扣因子。

这一抽象对单一环境非常有效，但在构建通用智能体时存在三个问题。

#### 状态接口不统一

不同环境的状态表示可能完全不同：

- Atari 使用像素或离散游戏状态；
- 机械臂使用关节角、末端位姿和相机图像；
- 自动驾驶使用车辆状态、地图和多传感器输入。

#### 动作接口不统一

不同智能体的动作空间更加难以统一：

- Atari 是离散按键；
- 机械臂是连续关节控制；
- 移动机器人是线速度和角速度；
- 无人机是推力和姿态控制。

#### 奖励函数难以统一和迁移

任务往往可以用自然语言轻松表达，但很难为每个环境手工设计数值奖励。即使两个任务语义相近，其奖励函数也可能完全不同。

---

## 2. 核心思想：Video as Policy

UniPi 的基本观察是：

> 虽然不同环境的底层状态和动作不同，但它们的行为过程通常都可以表示为视频，任务目标通常都可以用文本描述。

因此，论文使用：

- **图像**作为通用观察接口；
- **文本**作为通用任务接口；
- **视频轨迹**作为通用计划表示；
- **逆动力学模型**作为机器人或环境特定的动作适配器。

完整架构可以概括为：

```text
当前观察图像 x0
       +
文本任务描述 c
       │
       ▼
文本与首帧条件化的视频扩散模型
       │
       ▼
粗粒度未来视频计划
       │
       ▼
时间超分辨率模型
       │
       ▼
细粒度未来视频计划
       │
       ▼
逆动力学模型
       │
       ▼
机器人动作序列
```

这一结构将“任务级规划”和“机器人级控制”分离：

$$
\underbrace{\rho_\theta(\tau|x_0,c)}
_{\text{相对通用的视频规划器}}
\qquad+\qquad
\underbrace{\pi_\psi(a_{0:H-1}|\tau,c)}
_{\text{机器人特定的动作适配器}}
$$

---

## 3. UPDP：Unified Predictive Decision Process

论文提出 Unified Predictive Decision Process，简称 UPDP，用来替代传统 MDP 作为其理论抽象。

一个 UPDP 定义为：

$$
G=\langle\mathcal X,\mathcal C,H,\rho\rangle
$$

其中：

- \(\mathcal X\)：图像观察空间；
- \(\mathcal C\)：文本任务描述空间；
- \(H\)：有限规划时域；
- \(\rho(\tau|x_0,c)\)：给定首帧和任务描述的未来视频分布。

令：

$$
\tau=[x_1,x_2,\dots,x_H]
$$

则：

$$
\rho(\tau|x_0,c)
$$

表示从当前图像 \(x_0\) 出发，完成任务 \(c\) 的可能未来图像轨迹。

随后定义轨迹条件动作模型：

$$
\pi(a_{0:H-1}|x_{0:H},c)
$$

它负责寻找能够实现生成视频轨迹的底层动作。

### 3.1 UPDP 与 MDP 的本质区别

MDP 显式建模：

$$
T(s_{t+1}|s_t,a_t)
$$

即：

> 给定当前状态和动作，未来会发生什么？

而 UniPi 建模：

$$
\rho(x_{1:H}|x_0,c)
$$

即：

> 给定当前场景和任务，成功完成任务的未来通常是什么样？

因此，UniPi 并不直接学习一个可以响应任意动作查询的环境动力学模型。它学习的是一个受到任务目标和专家示范强烈偏置的“成功轨迹分布”。

---

## 4. 条件视频扩散模型

### 4.1 前向加噪过程

令干净视频轨迹为 \(\tau\)，扩散时间为 \(k\)。论文使用连续时间扩散过程：

$$
q_k(\tau_k|\tau)
=
\mathcal N(\alpha_k\tau,\sigma_k^2I)
$$

等价的采样形式为：

$$
\tau_k=\alpha_k\tau+\sigma_k\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
$$

其中：

- \(\alpha_k\tau\)：保留下来的原始视频信号；
- \(\sigma_k\epsilon\)：加入的高斯噪声；
- \(\tau_k\)：扩散时间 \(k\) 下的 noisy video。

模型训练一个条件去噪器：

$$
s_\theta(\tau_k,k|c,x_0)
$$

根据：

- 当前 noisy video；
- 扩散时间；
- 文本指令；
- 初始观察图像；

恢复干净的未来视频。

---

### 4.2 Classifier-Free Guidance

论文使用 classifier-free guidance 强化文本和首帧条件：

$$
\hat s(\tau_k,k|c,x_0)
=
(1+\omega)s(\tau_k,k|c,x_0)
-
\omega s(\tau_k,k)
$$

其中：

- \(s(\tau_k,k|c,x_0)\)：有条件预测；
- \(s(\tau_k,k)\)：无条件预测；
- \(\omega\)：条件引导强度。

较大的 \(\omega\) 会使生成视频更加服从文本任务和当前图像，但过强也可能牺牲自然性和多样性。

---

## 5. Log SNR Noise Schedule

论文附录写道：

> 使用范围为 \([-20,20]\) 的 log SNR noise schedule。

SNR 是信噪比：

$$
\operatorname{SNR}(k)
=
\frac{\alpha_k^2}{\sigma_k^2}
$$

log SNR 定义为：

$$
\lambda_k
=
\log\operatorname{SNR}(k)
=
\log\frac{\alpha_k^2}{\sigma_k^2}
$$

因此，范围 \([-20,20]\) 表示扩散训练覆盖了从“几乎纯噪声”到“几乎完全干净”的整个区间。

### 当 \(\lambda=20\)

$$
\operatorname{SNR}=e^{20}\approx4.85\times10^8
$$

信号远强于噪声，样本接近干净视频。

### 当 \(\lambda=-20\)

$$
\operatorname{SNR}=e^{-20}\approx2.06\times10^{-9}
$$

噪声远强于信号，样本接近纯高斯噪声。

在常见的 variance-preserving 参数化下：

$$
\alpha_k^2+\sigma_k^2=1
$$

有：

$$
\alpha_k^2=\operatorname{sigmoid}(\lambda_k),
\qquad
\sigma_k^2=\operatorname{sigmoid}(-\lambda_k)
$$

需要注意，论文只说明 log SNR 的取值范围，并未在正文中完整说明中间采用何种插值函数，因此不能仅凭这句话断定它一定在 \([-20,20]\) 之间线性变化。

---

## 6. Conditional Video Synthesis：首帧条件化

普通 text-to-video 模型只需要根据文本生成一段合理视频，但机器人规划要求视频必须从真实当前状态开始。

UniPi 的条件分布为：

$$
\rho_\theta(\tau|x_0,c)
$$

其中 \(x_0\) 是机器人当前看到的真实场景。

一种直观做法是在测试时强制把生成视频的第一帧替换成 \(x_0\)。论文发现这种方法效果不好：虽然第一帧正确，后续帧仍会迅速偏离原场景。

因此，UniPi 在训练阶段就把首帧作为显式条件，使模型学会：

> 如何从指定的真实初始场景继续生成未来。

这与测试时简单替换第一帧不同。前者让整个生成分布依赖初始观察，后者只对输出结果做局部修改。

---

## 7. Trajectory Consistency through Tiling

这是论文中最容易误解的技术设计之一。

原始观察图像为：

$$
x_0\in\mathbb R^{C\times H\times W}
$$

UniPi 将它沿视频时间维复制：

$$
\operatorname{Tile}(x_0)
=
[x_0,x_0,\dots,x_0]
\in
\mathbb R^{T\times C\times H\times W}
$$

扩散过程中，正在被去噪的视频为：

$$
\tau_k=[x_k^1,x_k^2,\dots,x_k^T]
$$

在每个视频时间位置 \(t\)，模型接收的局部输入近似为：

$$
[x_k^t;x_0]
$$

即把当前 noisy frame 与原始观察图像在通道维拼接。

整体上可写成：

$$
[\tau_k;\operatorname{Tile}(x_0)]
$$

### 7.1 为什么借用 temporal super-resolution 架构？

标准时间超分辨率视频模型通常执行：

$$
\text{低帧率视频}
\rightarrow
\text{高帧率视频}
$$

其网络会在每个时间位置接收一段低时间分辨率视频作为条件。

UniPi 复用了这一架构，但在第一阶段没有低帧率未来视频作为条件，只有当前观察 \(x_0\)。因此，它把原本的低帧率条件视频替换为：

$$
[x_0,x_0,\dots,x_0]
$$

即：

> 标准时间超分辨率模型在每个位置看到对应的粗视频帧；UniPi 在每个位置反复看到同一个初始观察帧。

### 7.2 Tiling 的作用

它不断提醒模型：

- 原始桌面布局是什么；
- 哪些物体原本存在；
- 背景和相机视角是什么；
- 哪些区域不应随时间无故改变。

因此，它主要抑制：

- 物体凭空出现或消失；
- 场景布局漂移；
- 背景改变；
- 非目标物体位置发生无关变化。

### 7.3 Tiling 不会把未来帧固定成初始帧

复制的 \(x_0\) 只是条件，不是生成目标。

模型仍然可以生成：

- 机械臂移动；
- 物体被抓取；
- 物体被搬运；
- 目标状态形成。

更准确的类比是：

$$
\text{原始图像}+\text{编辑指令}
\rightarrow
\text{保留场景结构的修改结果}
$$

### 7.4 First-frame conditioning 与 tiling 的区别

| 机制 | 作用 |
|---|---|
| First-frame conditioning | 告诉模型视频从哪个真实状态开始 |
| Tiling / frame consistency | 生成每个未来时间点时都重新参考初始场景 |

前者强调“起点正确”，后者强调“整个轨迹不要忘记原始环境”。

---

## 8. Hierarchical Planning：时间层级规划

对于长时间任务，直接一次生成所有细粒度帧非常困难。

UniPi 采用粗到细的时间层级：

1. 先生成时间上稀疏的粗粒度视频；
2. 再使用 temporal super-resolution 插入中间帧。

形式上可以表示为：

$$
[x_0,x_4,x_8,\dots]
\rightarrow
[x_0,x_1,x_2,\dots,x_8,\dots]
$$

粗粒度阶段负责决定：

- 先接近哪个物体；
- 何时抓取；
- 物体最终放在哪里。

细粒度阶段负责补全：

- 机械臂如何连续运动；
- 抓取和放置之间的过渡；
- 相邻帧之间的局部一致性。

论文把这种机制称作 hierarchical planning，但需要谨慎理解：

> 它不是传统规划中显式的子目标搜索、options 或 symbolic hierarchy，而是生成模型在时间维上的粗到细建模。

---

## 9. Flexible Behavioral Modulation

基础 UniPi 生成分布为：

$$
\rho_\theta(\tau|x_0,c)
$$

同一个文本任务往往对应多种合理计划。例如“把青色方块放到橙色方块上”可能存在：

- 抓取左边的青色方块；
- 抓取右边的青色方块；
- 从不同方向接近目标；
- 经过不同中间姿态。

Flexible Behavioral Modulation 的目标是在测试时加入一个额外轨迹约束：

$$
h(\tau)
$$

概念上，受引导后的分布可以写为：

$$
\rho_{\text{guided}}(\tau|x_0,c)
\propto
\rho_\theta(\tau|x_0,c)h(\tau)
$$

其中：

- \(\rho_\theta\) 保证视频自然、符合任务和数据分布；
- \(h(\tau)\) 表示额外偏好或约束。

从 score 的角度：

$$
\nabla_\tau\log\rho_{\text{guided}}(\tau)
=
\nabla_\tau\log\rho_\theta(\tau|x_0,c)
+
\nabla_\tau\log h(\tau)
$$

因此，扩散采样过程可以同时受到两种力量影响：

1. 生成一个合理的视频计划；
2. 使该计划满足新增约束。

### 9.1 分类器形式

可以训练一个分类器判断轨迹是否满足某种要求：

$$
h(\tau)=p_\phi(y=1|\tau)
$$

例如：

- 是否选择指定物体；
- 是否绕开障碍物；
- 是否经过某个区域；
- 是否达到指定中间状态。

也可以写成代价形式：

$$
h(\tau)=\exp[-\lambda C(\tau)]
$$

从而偏好低代价计划。

### 9.2 指定中间图像

论文还提出可以使用针对某个中间图像的 Dirac delta：

$$
h(\tau)=\delta(x_m-x^\star)
$$

含义是要求视频在第 \(m\) 个时间点经过指定图像 \(x^\star\)。

直观流程为：

$$
x_0
\rightarrow
\cdots
\rightarrow
x^\star
\rightarrow
\cdots
\rightarrow
x_H
$$

实际实现中通常不会真的计算理想 Dirac delta，而会使用：

- frame clamping；
- diffusion inpainting；
- 很强的图像距离约束；
- 中间帧 guidance。

论文 Figure 7 展示的正是这种中间视觉引导：在文本目标不变的情况下，通过额外中间图像控制模型操作特定方块。

### 9.3 需要注意的证据范围

这一部分更多是方法兼容性和概念展示。论文没有系统证明 UniPi 能稳定满足任意复杂测试时约束，也没有证明受引导的视频一定满足真实机器人动力学。

---

## 10. Inverse Dynamics：从视频恢复动作

视频生成器输出的是视觉计划，并不是机器人可以直接执行的关节命令。

因此，UniPi 训练逆动力学模型：

$$
\pi_\psi(a_t|x_t,x_{t+1})
$$

它回答：

> 要让图像从 \(x_t\) 变化到 \(x_{t+1}\)，机器人应该执行什么动作？

更一般地，可以写为：

$$
\pi_\psi(a_{0:H-1}|x_{0:H},c)
$$

论文模拟实验中的动作是 7 维控制量：

- 6 个机器人关节控制；
- 1 个接触或抓取控制。

逆动力学网络规模较小，主要由卷积层、残差卷积、全局均值池化和 MLP 构成，并使用均方误差训练。

### 10.1 模块化优势

视频生成器和逆动力学模型可以使用不同数据训练：

- 视频规划器可以使用大量没有动作标签的视频；
- 逆动力学模型只需要一批具体机器人上的图像—动作数据。

因此，UniPi 希望把迁移问题分解为：

$$
\text{通用行为知识}
+
\text{少量机器人适配数据}
$$

### 10.2 潜在瓶颈

从两帧图像推断动作通常不是唯一的：

- 多种关节轨迹可能产生相似视觉变化；
- 遮挡会隐藏关键运动；
- 单目图像缺少精确深度；
- MSE 可能把多个有效动作平均成无效动作；
- 换机器人后通常要重新训练适配器。

因此：

> 视频计划可以迁移，不等于底层策略可以零成本迁移。

---

## 11. 动作执行：开环与闭环

理论上，UniPi 可以使用模型预测控制：

1. 根据当前图像生成未来计划；
2. 执行一步或少量动作；
3. 获取新观察；
4. 重新生成视频；
5. 循环执行。

但论文为了降低计算成本，所有控制实验使用的是开环执行：

1. 一次生成完整视频；
2. 一次预测完整动作序列；
3. 连续执行，过程中不重新规划。

这会造成明显风险：

$$
\text{真实执行状态}\neq\text{生成视频状态}
$$

一旦早期发生抓取误差、物体滑动或机械臂偏差，后续动作仍按照原计划执行，误差可能不断累积。

---

## 12. 训练规模与架构细节

论文附录给出的主要配置如下。

### 视频扩散模型

- Video U-Net；
- 3 个 residual blocks；
- base channels：512；
- channel multiplier：\([1,2,4]\)；
- attention resolutions：\([6,12,24]\)；
- attention head dimension：64；
- conditioning embedding dimension：1024；
- 使用 temporal convolution 混合时间信息；
- 训练 200 万步；
- batch size：2048；
- learning rate：\(10^{-4}\)；
- warmup：1 万步；
- 使用 256 张 TPU-v4。

### 文本编码器

- T5-XXL；
- 约 46 亿参数。

### 模拟环境视频模型

- 首帧条件视频模型约 17 亿参数；
- 时间超分辨率模型约 17 亿参数；
- 粗粒度视频：\(10\times48\times64\)；
- 细粒度视频：\(20\times48\times64\)。

### 基线规模

论文中的 Transformer BC 基线约为 1000 万参数。

因此，实验性能差异同时混合了：

1. 视频表示的优势；
2. 大模型容量的优势；
3. 更大计算预算的优势。

论文没有提供严格的等参数、等算力比较。

---

## 13. 实验结果

### 13.1 组合语言泛化

任务包括：

- 把某种颜色的方块放入指定盒子；
- 把一个方块放在另一个方块左边、右边或上方；
- 先把白色方块染色，再进行目标摆放。

训练时使用约 70% 的语言组合，测试剩余 30%。

| 模型 | Seen Place | Seen Relation | Novel Place | Novel Relation |
|---|---:|---:|---:|---:|
| State + Transformer BC | 19.4 | 8.2 | 11.9 | 3.7 |
| Image + Transformer BC | 9.4 | 11.9 | 9.7 | 7.3 |
| Image + Trajectory Transformer | 17.4 | 12.8 | 13.2 | 9.1 |
| State-action Diffuser | 9.0 | 11.2 | 12.5 | 9.6 |
| **UniPi** | **59.1** | **53.2** | **60.1** | **46.1** |

在该受控环境中，生成视觉轨迹再恢复动作明显优于直接预测动作。

但这里的“组合泛化”主要是：

> 已知颜色、物体、关系和操作技能的未见组合。

它不是对全新概念或全新技能的开放世界泛化。

---

### 13.2 消融实验

| 首帧条件 | Frame Consistency | 时间层级 | Place | Relation |
|---|---|---|---:|---:|
| 否 | 否 | 否 | 13.2 | 12.4 |
| 是 | 否 | 否 | 52.4 | 34.7 |
| 是 | 是 | 否 | 53.2 | 39.4 |
| 是 | 是 | 是 | 59.1 | 53.2 |

结论：

- 首帧条件化贡献最大；
- Tiling 能进一步改善场景一致性；
- 时间层级对复杂关系任务尤其重要。

---

### 13.3 多任务迁移

UniPi 在 10 类训练任务上学习，在 3 类未见任务上测试。

| 模型 | Place Bowl | Pack Object | Pack Pair |
|---|---:|---:|---:|
| State + Transformer BC | 9.8 | 21.7 | 1.3 |
| Image + Transformer BC | 5.3 | 5.7 | 7.8 |
| Image + TT | 4.9 | 19.8 | 2.3 |
| Diffuser | 14.8 | 15.9 | 10.5 |
| **UniPi** | **51.6** | **75.5** | **45.7** |

结果支持视频计划表示在多任务行为建模中的潜力。

不过，这些环境依然使用相似的模拟器、机械臂和动作接口，尚不能等价为真正的跨形态机器人迁移。

---

### 13.4 互联网预训练与真实场景视频生成

预训练数据包括：

- 约 1400 万视频—文本对；
- 约 6000 万图像—文本对；
- LAION-400M；
- 之后在约 7200 个 Bridge 机器人视频—文本对上微调。

| 模型 | CLIP ↑ | FID ↓ | FVD ↓ | 末帧成功分类器 ↑ |
|---|---:|---:|---:|---:|
| 无预训练 | 24.43 | 17.75 | 288.02 | 72.6% |
| 有预训练 | 24.54 | 14.54 | 264.66 | 77.1% |

互联网预训练改善了视频生成指标，但需要注意：

- CLIP 分数提升很小；
- 论文正文中“higher FID and FVD”的表述与表格的下降数值及箭头方向不一致，应为文字错误；
- 77.1% 不是现实机器人执行成功率；
- 它是一个分类器根据生成视频最后一帧判断任务是否“看起来成功”。

论文的真实世界部分主要证明：

> 模型能够生成较合理的真实机器人未来视频。

它没有完整证明：

> 真实机器人按照这些视频和逆动力学动作执行后能够以 77.1% 的概率完成任务。

---

## 14. UniPi 与 Dreamer 的区别

UniPi 和 Dreamer 都涉及“预测未来”，但二者预测未来的目的、条件和优化方法不同。

| 维度 | Dreamer | UniPi |
|---|---|---|
| 核心模型 | 动作条件世界模型 | 目标条件成功视频生成器 |
| 典型形式 | \(p(z_{t+1},r_t|z_t,a_t)\) | \(\rho(x_{1:H}|x_0,c)\) |
| 动作是否输入未来模型 | 是 | 否 |
| 是否预测奖励 | 是 | 否 |
| 是否训练 actor | 是 | 否 |
| 是否训练 critic/value | 是 | 否 |
| 决策依据 | 最大化 imagined return | 生成类似成功示范的视频 |
| 动作产生方式 | actor 直接输出 | inverse dynamics 从视频恢复 |
| 算法类别 | Model-Based RL | Generative Imitation / Video Policy |

### 14.1 Dreamer 的逻辑

Dreamer 学习：

$$
z_t,a_t
\rightarrow
z_{t+1},r_t
$$

它可以比较不同候选动作：

- 动作 A 会导致什么；
- 动作 B 会导致什么；
- 哪个未来累计回报更高。

然后优化：

$$
\max_\pi
\mathbb E
\left[
\sum_t\gamma^tr_t
\right]
$$

因此 Dreamer 可以概括为：

$$
\boxed{
\text{预测动作后果}
\rightarrow
\text{依据奖励选择动作}
}
$$

### 14.2 UniPi 的逻辑

UniPi 学习：

$$
x_0,c
\rightarrow
\text{一个看起来能够完成任务的未来视频}
$$

随后：

$$
x_t,x_{t+1}
\rightarrow
a_t
$$

因此可以概括为：

$$
\boxed{
\text{生成成功故事板}
\rightarrow
\text{把故事板翻译成动作}
}
$$

UniPi 不会显式枚举动作，也不会使用奖励或 value function 比较候选计划。

---

## 15. UniPi 是否属于强化学习？

从严格算法定义看，论文实现的 UniPi 基本脱离了强化学习。

其训练过程没有：

- 环境试错；
- 奖励最大化；
- Bellman backup；
- Q function；
- value function；
- policy gradient；
- actor-critic；
- imagined return optimization。

视频模型优化的是扩散去噪目标，逆动力学模型优化的是监督回归目标。

因此，它更准确的分类是：

$$
\boxed{
\text{文本条件视频生成}
+
\text{生成式模仿学习}
+
\text{逆动力学}
}
$$

论文使用“policy”一词，是因为该系统最终可以产生动作，而不是因为它使用了强化学习算法。

### 15.1 Amortized Planning

UniPi 的“规划”也不是传统显式搜索。

传统规划通常是：

$$
\text{候选动作}
\rightarrow
\text{预测结果}
\rightarrow
\text{计算代价}
\rightarrow
\text{选择最优方案}
$$

UniPi 是：

$$
\text{任务条件}
\rightarrow
\text{直接生成一条成功轨迹}
$$

因此，它更接近 amortized planning：

> 训练阶段把大量示范中的规划规律压缩进生成模型，测试阶段通过一次生成过程直接输出计划，而不再进行显式搜索。

---

## 16. UniPi 如何与 RL 结合？

虽然当前论文不是 RL，但它可以自然接入 reward 或 value guidance。

例如定义：

$$
h(\tau)=\exp[\beta R(\tau)]
$$

则：

$$
\rho_{\text{guided}}(\tau|x_0,c)
\propto
\rho_\theta(\tau|x_0,c)
\exp[\beta R(\tau)]
$$

此时：

- 视频模型提供行为先验；
- reward model 提供任务评价；
- diffusion guidance 把采样推向高奖励轨迹。

另一种方式是：

1. 生成多条候选视频；
2. 使用 reward/value model 打分；
3. 选择最高分计划；
4. 用逆动力学执行；
5. 根据真实执行结果继续更新策略。

只有当系统真正利用奖励信号优化策略时，它才更接近 model-based RL。

---

## 17. 方法优势

### 17.1 通用中间表示

视频比机器人关节动作更容易跨环境共享，并且可以利用已有视觉和视频基础模型。

### 17.2 语言作为目标接口

文本避免了为每个任务单独设计奖励函数，并支持属性与关系的组合泛化。

### 17.3 可利用无动作视频

视频规划器不要求所有数据都有底层控制标签，因此理论上可以利用：

- 人类教学视频；
- 互联网操作视频；
- 无动作标注机器人视频；
- 大规模图像—文本数据。

### 17.4 计划可解释

视频计划可以被人直接检查，有利于：

- 诊断模型错误；
- 识别错误目标；
- 人工验证中间步骤；
- 添加视觉约束。

### 17.5 规划与控制解耦

高层行为知识保留在视频模型中，机器人差异集中到逆动力学适配器中。

---

## 18. 主要局限

### 18.1 “Universal Policy” 的证据有限

论文真正展示的是：

$$
\text{相对通用的视频规划器}
+
\text{机器人特定的逆动力学适配器}
$$

尚未充分证明一个规划器能够直接跨机械臂、移动机器人、无人机等完全不同形态工作。

### 18.2 视频自然不等于物理可执行

生成视频可能包含：

- 物体轻微瞬移；
- 接触关系错误；
- 机械臂穿透物体；
- 不可达姿态；
- 视觉自然但动力学不可能的变化。

系统缺少显式的：

- 物理一致性检查；
- 可达性验证；
- 碰撞检测；
- 安全约束；
- 轨迹可执行性评估。

### 18.3 逆动力学存在多解性

同一视觉变化可能对应多种动作，单纯使用 MSE 回归可能产生动作平均问题。

### 18.4 开环误差累积

论文实验使用开环控制，无法及时纠正执行偏差。

### 18.5 视频扩散速度慢

高质量视频生成可能需要约一分钟。虽然作者报告蒸馏后可获得约 16 倍加速，但仍与实时闭环控制存在距离。

### 18.6 评价指标错位

FID、FVD 和 CLIP 主要衡量视觉质量或语义一致性，不直接衡量：

- 是否可执行；
- 是否安全；
- 是否满足接触约束；
- 机器人是否真实完成任务。

### 18.7 模型与基线规模不匹配

UniPi 使用十亿级视频模型和 T5-XXL，而部分基线只有约千万参数，因此不能把全部性能提升都归因于 video-as-policy 表示。

### 18.8 真实世界控制证据不足

真实场景实验主要是视频生成评估，并没有报告完整的现实机器人动作执行成功率。

---

## 19. 阅读时应保持的概念区分

| 论文术语 | 更谨慎的解释 |
|---|---|
| Universal policy | 通用视频计划表示，底层动作仍需特定适配 |
| Planning | 条件生成成功轨迹，不一定包含显式搜索 |
| World model | 受任务和专家示范偏置的视频分布，而非任意动作动力学 |
| Real-world transfer | 真实场景视频生成迁移，不等于完整现实控制迁移 |
| Success in Table 4 | 生成视频末帧分类器结果，不是机器人实际执行成功率 |
| Offline RL scenario | 实际训练更接近离线模仿学习 |

---

## 20. 总结

UniPi 提出了一条不同于 Dreamer 等 model-based RL 的路线：

$$
\boxed{
\text{不先学习“动作会导致什么”}
}
$$

而是：

$$
\boxed{
\text{直接学习“成功完成任务的视频应该是什么样”}
}
$$

随后再通过逆动力学把视觉变化翻译成动作。

其最重要的思想是：

> 将视频视为跨任务和跨环境的策略接口，将语言视为目标接口，将底层动作差异推迟到机器人特定的逆动力学模块中处理。

该方法在模拟机器人任务中展示了较强的组合泛化和多任务迁移能力，也证明了互联网视频预训练可以改善真实机器人视频计划的生成质量。

但它仍是一项早期概念验证：

- 没有真正证明跨机器人形态的通用策略；
- 没有解决视频计划的物理可执行性；
- 没有完成高频闭环现实控制；
- 没有使用奖励信号进行策略优化；
- 与等规模动作模型的公平比较仍然不足。

因此，UniPi 最适合被定位为：

$$
\boxed{
\text{Video-as-Policy / Video Planning / Generative Imitation}
}
$$

而不是 Dreamer 式的传统 model-based reinforcement learning。

---

## 21. 相关笔记

- [[Visual Foresight|Visual Foresight]]：同样使用预测未来视觉结果进行规划，但显式以动作条件建模动力学。
- [[Dreamer技术报告|Dreamer]]：以 latent dynamics、reward 和 actor-critic 进行 model-based RL 的对照路线。
- [[DreamerV3_技术报告_中文|DreamerV3]]：更通用、稳定的潜空间 world-model RL 路线。
- [[Diffusion Policy 概述|Diffusion Policy]]：直接生成动作序列的 diffusion policy，可与 UniPi 的视频 diffusion 中间接口对比。
- [[Pi0_7_technical_report|π0.7]]：使用视觉 subgoal 与丰富上下文控制通用机器人行为的相关路线。

## 参考文献

Du, Y., Yang, M., Dai, B., Dai, H., Nachum, O., Tenenbaum, J. B., Schuurmans, D., & Abbeel, P.
**Learning Universal Policies via Text-Guided Video Generation.**
NeurIPS 2023. arXiv:2302.00111.
