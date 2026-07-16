---
title: WorldVLA 论文综述
type: paper_note
topic: vision_language_action
status: mature
importance: low
updated: 2026-07-16
tags:
  - worldvla
  - vla
  - autoregressive-model
  - action-tokenization
  - image-generation
  - world-model
  - robotics
---

# WorldVLA：统一动作生成与图像生成的自回归模型

## 1. 论文定位

WorldVLA 试图在一个统一的自回归 Transformer 中同时实现两项能力：

1. **动作生成**

$$
(\text{语言指令},\text{历史图像})
\rightarrow
\text{action chunk}
$$

2. **世界状态预测**

$$
(\text{当前图像},\text{当前动作})
\rightarrow
\text{下一帧图像}
$$

模型基于 Chameleon，将文本、图像和机器人动作全部离散成 token，并使用同一个 Transformer 进行 next-token prediction。作者希望通过动作预测和 action-conditioned 图像预测的联合训练，让策略模型与世界模型共享表示、互相增强。

---

## 2. 模型结构

> [!figure] 论文原始模型结构图
> ![[attachments/paper-figures/worldvla-model-overview.png]]
> WorldVLA 将动作模型与世界模型统一到同一个自回归框架中的整体结构。原图来自 [WorldVLA: Towards Autoregressive Action World Model（arXiv:2506.21539）](https://arxiv.org/abs/2506.21539)，由论文源文件高分辨率导出。

WorldVLA 使用三种 tokenizer：

- **文本 tokenizer**：BPE tokenizer。
- **图像 tokenizer**：VQ-GAN，将图像压缩为离散视觉 token。
- **动作 tokenizer**：将 7 维机器人动作的每个维度分别离散成 256 个区间。

一步动作表示为：

$$
[\Delta x,\Delta y,\Delta z,
\Delta\theta_x,\Delta\theta_y,\Delta\theta_z,
g]
$$

对应 7 个 action token。

训练损失为：

$$
\mathcal L
=
\mathcal L_{\text{action}}
+
\alpha\mathcal L_{\text{world}}
$$

其中动作生成和图像生成只是两种不同格式的训练样本，共享同一个模型参数。

---

## 3. 推理方式

### 动作推理

动作生成时直接执行：

```text
历史真实图像 + 任务指令
          ↓
       WorldVLA
          ↓
     action tokens
          ↓
      连续机器人动作
```

推理期间**不会先生成未来图像**，本质上仍然是传统的离散 VLA 动作生成。

### 图像推理

世界模型模式为：

```text
当前图像 + 给定动作
          ↓
       WorldVLA
          ↓
   下一帧图像 tokens
          ↓
     VQ-GAN 解码
```

如需生成多帧视频，需要在模型外部不断递归：

$$
\hat o_{t+1}=f(o_t,a_t)
$$

$$
\hat o_{t+2}=f(\hat o_{t+1},a_{t+1})
$$

未来动作仍需要由数据集或外部模块提供。

---

## 4. 主要贡献

### 4.1 统一动作和图像生成

论文证明了动作生成和 action-conditioned 图像预测可以在同一个离散自回归模型中联合训练。

### 4.2 World Model 辅助动作学习

加入下一帧图像预测任务后，LIBERO 上的动作成功率有所提升，说明环境状态预测可以作为动作策略的辅助训练任务。

### 4.3 Action Attention Mask

普通自回归 action chunk 中，后续动作依赖前面预测的动作，容易产生误差累积。

WorldVLA 将不同时间步 action block 之间的 attention 切断，使每个未来动作主要根据图像、语言以及自身序列位置进行预测，从而改善长 action chunk 的性能。

---

## 5. 主要局限

### 5.1 更像多任务学习，而不是真正的闭环世界模型

WorldVLA 并没有形成：

```text
生成候选动作
→ 预测未来状态
→ 评价未来结果
→ 修正或选择动作
```

它缺少：

- reward model；
- value model；
- success classifier；
- MPC、CEM 或 imagined rollout；
- 根据预测图像重新选择动作的机制。

因此，world model 只在训练阶段作为辅助任务参与参数共享，动作推理阶段并不显式使用图像生成能力。

更准确的定位是：

> **动作生成与 action-conditioned 图像生成的统一多任务模型。**

而不是会利用世界模型进行规划的闭环 agent。

---

### 5.2 视觉表征并不适合精细机器人感知

模型采用 VQ-GAN 图像 tokenizer。

VQ-GAN 更偏向图像压缩与重建，而不是 CLIP、SigLIP 等模型所强调的视觉语义理解和图文对齐。论文也承认其离散视觉 token 的语义能力弱于专门的视觉感知模型。

这种设计是为了同时支持图像生成，但会牺牲 VLA 所需的：

- 精细物体识别；
- 空间关系理解；
- 小物体感知；
- 视觉语言对齐能力。

因此，统一理解与生成并不一定优于分别使用强视觉编码器和生成模型。

---

### 5.3 Action tokenizer 过于简单

WorldVLA 的动作离散化只是对每一个标量维度独立分 bin：

$$
a^{(d)}
\rightarrow
\text{one of 256 bins}
$$

它没有像 FAST 一样学习动作序列中的时间结构和高频模式，也没有真正构建“动作词汇”。

这种表示存在以下问题：

- 动作维度之间的联合关系没有被 tokenizer 建模；
- 相邻动作 bin 的几何距离没有体现在交叉熵中；
- action chunk 的时序结构没有被压缩；
- 长 chunk 需要大量 action token；
- 离散化会造成精度损失。

因此它本质上仍是 OpenVLA 风格的朴素标量离散化。

---

### 5.4 Attention Mask 删除了问题，也删除了时间依赖

论文通过切断不同 action block 之间的 attention，避免早期动作错误传播到后续动作。

但这种方法近似假设：

$$
p(a_{t:t+K-1}\mid o,l)
\approx
\prod_k p(a_{t+k}\mid o,l,k)
$$

也就是不同未来时间步的动作在给定图像和语言后相互独立。

它虽然避免了误差累积，但同时失去了显式建模：

- 轨迹连续性；
- 动作平滑性；
- chunk 内动作协调；
- 动作序列的联合多模态性；
- 前后动作之间的动力学关系。

因此，该方法不是让自回归模型学会抵抗错误，而是直接禁止后续动作访问前序动作。

这更接近多 horizon 并行预测，而不是严格意义上的时间自回归动作生成。

---

### 5.5 Action horizon 只由 token 位置隐式表示

文本、图像和动作被拼接成同一个序列，共享 Chameleon 的全局 RoPE。

模型没有 action 专用、从 0 开始的位置编码，也没有明确设计：

- action timestep embedding；
- future horizon embedding；
- 控制频率 embedding；
- 时间戳或 $\Delta t$。

模型主要根据 action token 相对 `[BOA]` 的位置，判断它属于第几个 action block。

因此它知道的是：

$$
\text{第 }k\text{ 个动作}
$$

而不是：

$$
\text{未来 }\tau_k\text{ 秒的动作}
$$

---

### 5.6 难以适配不同控制频率

WorldVLA 使用相对位姿增量：

$$
\Delta x\approx v\Delta t
$$

相同物理轨迹在不同控制频率下，会产生不同的单步动作幅值。

例如：

- 10 Hz：每步移动约 1 cm；
- 50 Hz：每步移动约 2 mm。

但模型没有输入控制周期 $\Delta t$，因此相同的 action block index 在不同设备上代表不同真实时间：

$$
\tau_k=k\Delta t
$$

这会造成：

- 动作尺度冲突；
- action token 分布冲突；
- 历史图像帧间速度语义冲突；
- 不同设备间 horizon 含义不一致。

该设计更适合 LIBERO 这种机器人、动作空间和控制频率高度固定的环境，不适合直接扩展到多机器人、多频率数据。

---

### 5.7 World Model 训练只做单步预测

默认配置中，world model 只学习：

$$
(o_t,a_t)\rightarrow o_{t+1}
$$

长视频通过递归调用单步模型得到。

这种方式容易产生：

- 图像误差累积；
- 物体消失或形变；
- 长期物理不一致；
- 生成图像与真实观测之间的分布偏移。

同时，action policy 训练时看到的是真实图像，而不是 world model 生成的图像，因此不能自然地将两者递归连接成闭环。

---

## 6. 总体评价

WorldVLA 的实现方式比较直接：

```text
统一离散 token
+ 统一自回归 Transformer
+ 动作预测与图像预测联合训练
```

它证明了 action-conditioned 图像生成可以作为 VLA 的辅助任务，并提出了一个对离散 action chunk 有效的 attention mask。

但其“Action World Model”能力被标题放大了。模型实际上没有实现基于未来想象的动作规划，也没有在推理阶段让动作生成和图像生成真正形成闭环。

因此，更准确的评价是：

> **WorldVLA 是一个将动作生成和下一帧图像生成放入同一自回归模型的多任务 VLA，而不是一个能够通过世界模型进行闭环推理和规划的完整 Action World Model。**

---

## 7. 相关笔记

- [[UniPi_技术总结|UniPi]]：同样把视觉生成引入策略学习，但 UniPi 直接生成目标条件视频计划，WorldVLA 只把下一帧预测作为辅助训练任务。
- [[DreamZero_Technical_Report|DreamZero]]：同样联合视觉与动作生成，但 DreamZero 在推理阶段持续联合生成视频 latent 和动作。
- [[OA_WAM|OA-WAM]]：同样使用 world prediction 辅助动作学习，但进一步引入对象地址与 slot routing。
- [[DreamerV3_技术报告|DreamerV3]]：对比真正通过潜空间想象训练 actor-critic 的闭环 world-model RL 路线。
- [[Pi0_7_technical_report|π0.7]]：对比异步 subgoal image context 与 WorldVLA 的同步单步图像预测。
