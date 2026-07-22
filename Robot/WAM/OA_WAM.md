---
title: OA-WAM 论文综述与批判性阅读笔记
type: paper_note
topic: world_action_model
status: mature
importance: high
updated: 2026-07-16
tags:
  - oa-wam
  - world-action-model
  - object-centric-representation
  - object-addressability
  - flow-matching
  - robot-manipulation
---

# OA-WAM：Object-Addressable World Action Model 论文综述与批判性阅读笔记

> 论文：**OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation**
>
> 主题：Vision-Language-Action、World Action Model、Object-centric Representation、鲁棒机器人操作
>
> 文档性质：结合论文正文、附录与讨论整理的结构化综述

---

## 0. 一句话概括

OA-WAM 的核心不是“生成更逼真的未来”，而是把场景组织成一组**可按对象寻址的 slot**：每个 slot 带有一个 episode 内固定的对象地址 `addr`，以及随时间变化的对象状态 `cnt`；Transformer 中 slot 的 **Key 只允许依赖 `addr`**，从而降低相机、布局和机器人初始位姿变化导致的目标绑定漂移。

更准确地说，OA-WAM 提供的是一种：

$$
\boxed{\text{episode-local stable routing key} + \text{same-slot contextual value}}
$$

它不是一个显式数据库检索系统，也不是“先预测未来、再根据未来规划动作”的模型。

---

## 1. 研究问题与动机

传统 VLA/WAM 往往把场景表示成：

- 整幅图像 token；
- 视频 token 序列；
- 全局 latent；
- 混合的视觉—动作 hidden state。

这类表示能编码场景，但没有稳定的对象级接口告诉动作模块：

> 指令中提到的“红色杯子”究竟对应当前场景中的哪一个对象实例？

当训练布局发生变化时，即使目标对象仍然可见，模型也可能把目标身份与以下信息绑定：

- 固定空间位置；
- 相机视角；
- 背景纹理；
- 周围干扰物；
- 训练时常见的对象排列。

论文把这一问题称为：

$$
\textbf{lack of object addressability}
$$

OA-WAM 的基本假设是：

> 世界模型不仅要预测“场景会怎样变化”，还应提供稳定、可查询的对象级状态，使动作模块能够按对象身份读取对应信息。

---

## 2. 整体架构

> [!figure] 论文原始模型结构图
> ![[attachments/paper-figures/oa-wam-model-architecture.png]]
> OA-WAM 的多模态编码、Object Slot、Block-causal Backbone、World Head 与 Action Head。原图来自 [OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation（arXiv:2605.06481）](https://arxiv.org/abs/2605.06481)，由论文源文件高分辨率导出。

模型在时刻 \(t\) 输入：

- 最近 \(T=4\) 帧视觉观测；
- 第三人称图像和腕部图像；
- 语言指令 \(\ell\)；
- 7 维 proprioception \(q_t\)；
- 已执行的历史动作 \(a_{<t}\)。

模型输出：

1. 一个 \(16\times7\) 的连续动作 chunk；
2. 下一时刻每个对象的 content 和 pose；
3. 训练阶段的下一帧 VQ image tokens。

总体流程：

```text
语言 + 多视角图像
        │
        ├─ Qwen3-VL：提取对象名词短语和关系
        ├─ SAM 3：对象分割、候选发现、跨帧跟踪
        ├─ DINOv3：对象区域视觉特征
        ├─ VQ-GAN：整帧图像离散 token
        └─ 状态/动作离散化
                 │
                 ▼
      文本、图像、slot、状态、动作统一序列
                 │
                 ▼
       Slot-aware Chameleon-7B Transformer
                 │
        ┌────────┼─────────┐
        ▼        ▼         ▼
   World Head  Action Head  VQ Head
```

需要特别注意：**World Head 与 Action Head 是并行输出，不是串联规划。**

模型不是：

```text
预测未来对象状态 → 根据预测未来生成动作
```

而是：

```text
共享 trunk representation
├─ 辅助预测未来对象状态
└─ 直接生成动作 chunk
```

---

## 3. 为什么使用 SAM 3 + DINOv3

SAM 3 和 DINOv3 承担不同职责。

### 3.1 SAM 3：确定“哪些像素属于哪个对象”

Qwen3-VL 根据指令和初始图像提取 noun phrases，例如：

```json
["red mug", "green tray"]
```

这些短语作为 SAM 3 的文本 prompt，生成每个对象的 mask：

$$
M_k^t\in\{0,1\}^{H\times W}
$$

SAM 3 主要提供：

- 语言引导的对象实例分割；
- 自动模式发现未被语言提到的 distractors；
- 跨帧 concept tracking，使 slot \(k\) 在不同时间尽量对应同一个对象。

因此 SAM 3 提供的是：

$$
\boxed{\text{对象边界、对象实例划分和跨帧关联}}
$$

### 3.2 DINOv3：描述“这个对象区域是什么视觉内容”

在 DINOv3 dense feature map 上，对 SAM mask 内的特征做池化：

$$
f_k^t=\operatorname{MaskPool}(\operatorname{DINOv3}(I_t),M_k^t)
$$

DINOv3 原始特征为 1024 维，再投影到 256 维。它隐式编码：

- 类别与语义；
- 颜色、纹理；
- 局部外观；
- 形状信息；
- 同类实例之间的视觉差异。

DINOv3 不直接提供精确 3D pose；pose 来自另一条输入路径。

### 3.3 两者的关系

```text
SAM 3：把对象“圈出来”
DINOv3：把圈出来的对象“编码成视觉向量”
```

仅有 SAM，缺少高层视觉表征；仅有 DINO，缺少明确的对象实例边界。两者组合后才能形成对象级视觉 feature。

---

## 4. Object Slot 的构造

每个对象 slot 定义为：

$$
s_k^t=
[\underbrace{addr_k}_{32}
\Vert\underbrace{cnt_k^t}_{256}
\Vert\underbrace{\pi^t}_{16}
\Vert\underbrace{\rho_k}_{16}]
\in\mathbb R^{320}
$$

其中：

- \(addr_k\)：episode 内固定的对象地址；
- \(cnt_k^t\)：随时间变化的对象状态 latent；
- \(\pi^t\)：时间位置编码；
- \(\rho_k\)：robot/object/padding 的角色编码。

随后通过 slot adapter：

$$
f_\phi:\mathbb R^{320}\rightarrow\mathbb R^{4096}
$$

映射到 Chameleon 的 hidden dimension。

---

## 5. `addr_k` 的准确含义

论文定义：

$$
addr_k=f_{\text{addr}}([\ell_k\Vert f_k^{(0)}])
$$

逐项拆解如下。

### 5.1 \(\ell_k\)：对象语言标签 embedding

不是完整指令，而是第 \(k\) 个对象的 noun phrase，例如：

```text
"red mug"
```

论文使用 T5 编码并平均，得到约 256 维语言向量。

### 5.2 \(f_k^{(0)}\)：初始帧对象视觉特征

表示 episode 第 0 帧中对象 \(k\) 的 DINOv3 mask-pooled feature。

上标 \((0)\) 表示：只在 episode 开始时取一次。

### 5.3 拼接与 MLP

$$
[\ell_k\Vert f_k^{(0)}]\in\mathbb R^{512}
$$

通过：

$$
f_{\text{addr}}:\mathbb R^{512}\rightarrow\mathbb R^{128}\rightarrow\mathbb R^{32}
$$

得到：

$$
addr_k\in\mathbb R^{32}
$$

### 5.4 为什么同时使用语言和初始视觉

只使用语言时，同类多实例可能无法区分：

```text
red mug A
red mug B
```

它们语言标签相同。加入初始视觉 feature 后，可形成 episode 内的实例级差异。

因此：

$$
\boxed{addr_k\approx\text{语言类别提示}+\text{初始实例视觉特征}}
$$

但它不是全局离散 ID，也不是世界范围内统一的对象编号。

### 5.5 `addr` 是 episode-local，不是全局地址

在一个 episode 中：

$$
addr_k^0=addr_k^1=\cdots=addr_k^T
$$

换一个 episode 后，即使仍是“red mug”，也会根据新的初始观察重新计算地址。

因此更准确的定义是：

> 由对象语言标签和初始视觉特征生成的、episode 内固定的实例句柄。

训练过程中 \(f_{\text{addr}}\) 的参数会随 optimizer 更新；“固定”只指一次 episode 的不同时间帧之间固定，而不是整个训练过程永远不变。

---

## 6. `raw_k^t` 与 `cnt_k^t`

论文定义：

$$
cnt_k^t=f_{\text{cnt}}(raw_k^t)
$$

其中：

$$
raw_k^t=[f_k^t,p_k^t,\ell_k,shape_k^t]
$$

各部分为：

| 组成 | 含义 | 论文给出的维度 |
|---|---|---:|
| \(f_k^t\) | 当前帧 DINOv3 对象视觉特征 | 256 |
| \(p_k^t\) | 3D position + 6D rotation | 9 |
| \(\ell_k\) | 对象语言标签 embedding | 256 |
| \(shape_k^t\) | mask 形状描述 | 15 |

`shape` 包括：

- normalized centroid；
- bounding box；
- 二阶矩；
- mask area；
- convexity。

`raw` 可以理解为：

$$
\boxed{\text{当前帧中对对象的原始可观测描述}}
$$

随后：

$$
f_{\text{cnt}}:\mathbb R^{540}\rightarrow\mathbb R^{512}\rightarrow\mathbb R^{256}
$$

生成：

$$
cnt_k^t\in\mathbb R^{256}
$$

它意图表示对象当前的动态状态，例如：

- 当前外观；
- 当前位置和姿态；
- 当前 mask 几何；
- 是否发生移动或抓取；
- 与当前场景相关的对象状态。

### 6.1 一个明确的维度疑点

按论文列出的维度计算：

$$
256+9+256+15=536
$$

但论文写的是 \(raw_k^t\in\mathbb R^{540}\)。因此至少有一个地方存在笔误或遗漏的 4 维特征，需以代码为准。

### 6.2 `cnt` 并不是“纯动态内容”

因为 `raw` 中仍包含：

- 固定的语言标签 \(\ell_k\)；
- pose \(p_k^t\)；
- 视觉身份线索。

所以论文并未在表示层完全 disentangle identity 与 content。真正被严格约束的是：

$$
\boxed{\text{slot 的 Key routing 只能依赖 address}}
$$

而不是：

$$
\boxed{cnt\text{ 中绝对不允许存在身份信息}}
$$

---

## 7. Placeholder 与 LLaVA-style `masked_scatter`

### 7.1 它是什么

序列模板中先放置固定数量的 `<slot>` reserved token：

```text
... <SBOS> <slot> <slot> ... <slot> <SEOS> ...
```

整个 `input_ids` 先经过 Chameleon 的 embedding table：

$$
E=\operatorname{embed\_tokens}(input\_ids)
$$

同时，slot adapter 独立产生：

$$
e_k^t=f_\phi(s_k^t)\in\mathbb R^{4096}
$$

再找到 `<slot>` 位置并覆盖：

```python
slot_mask = input_ids == SLOT_TOKEN_ID
inputs_embeds[slot_mask] = slot_embeddings.reshape(-1, 4096)
```

最终真正进入 Transformer 的是连续 slot embedding，而不是 `<slot>` 的词表 embedding。

### 7.2 为什么不直接拼接

直接写成：

$$
E=[E_{text};E_{VQ};E_{slot};E_{action}]
$$

在数学上完全等价。

`placeholder + masked_scatter` 主要是工程模式，便于：

- 用统一 `input_ids` 表示序列布局；
- 复用 Hugging Face / Chameleon 的现有接口；
- 同步生成 position IDs、token types、attention mask 和 labels；
- 保留 frame/slot/action block 的固定位置；
- 沿用 LLaVA 注入连续视觉 feature 的实现习惯。

因此：

$$
\boxed{\texttt{masked\_scatter} \text{ 是代码组织方式，不是核心算法}}
$$

### 7.3 Slot 数量何时确定

Transformer 不负责生成 slot 数量。运行 trunk 前，感知模块已经完成对象提取。

论文的主设计是固定容量：约 16 个 object slots 加 1 个 robot slot，空余位置为 padding，并在 attention、loss 和评价中 mask。

不过论文后文又提到 24 或 27 个 active addresses，与固定 16 object slots 的描述存在明显不一致，需结合实现确认其截断或动态扩展策略。

---

## 8. OA Attention：真正被硬编码的是什么

标准 Attention：

$$
Q_i=W_Qx_i,\quad K_j=W_Kx_j,\quad V_j=W_Vx_j
$$

OA-WAM 对 slot 位置修改为：

$$
K_j^{(\ell)}=
W_K^{(\ell)}\operatorname{mask}_{\le32}(x_j^{(\ell)})
$$

其中 `mask` 保留 residual stream 的前 32 维，其他维度置零。

而：

$$
Q_j^{(\ell)}=W_Q^{(\ell)}x_j^{(\ell)}
$$

$$
V_j^{(\ell)}=W_V^{(\ell)}x_j^{(\ell)}
$$

仍读取完整 hidden state。

### 8.1 “addr 找到 cnt”并不是显式操作

OA-WAM 没有一个显式模块执行：

```text
根据 addr 查表 → 返回 cnt
```

真实计算是：

$$
\alpha_{ij}=\operatorname{softmax}_j(Q_i^\top K_j)
$$

$$
o_i=\sum_j\alpha_{ij}V_j
$$

Key 与 Value 之所以绑定，是因为 \(K_j\) 和 \(V_j\) 来自**同一个 token 位置 \(j\)**。

架构硬编码保证：

> 一旦某个 query 对 slot \(j\) 的 Key 给出高权重，读取的就是同一个 slot 行对应的 Value。

但模型仍需自己学习：

- 什么 query 应该对应 target；
- 哪个 address Key 应获得高权重；
- Value 中哪些维度对动作有用。

因此最准确的表述是：

$$
\boxed{addr\text{ 决定 slot 以什么 Key 被查询}}
$$

$$
\boxed{同一 slot 的 contextual hidden state 经 Value projection 被返回}
$$

而不是：

$$
\boxed{addr\text{ 显式索引一个独立存储的 cnt}}
$$

### 8.2 “Value 读取完整状态”的边界

从公式上，Value 确实读取完整 \(x_j^{(\ell)}\)，而非只读前 32 维。

但它读取的不是原始 \(cnt_j^t\)，而是经过：

- slot adapter；
- 多层 self-attention；
- FFN；
- 语言、图像、其他 slot 和历史信息上下文化；

之后的 hidden state。

所以 Value 可能包含该对象的 content，但也可能包含其他对象与全局上下文。

---

## 9. 为什么每层后都要 Address Reset

即使一层输入满足：

$$
x_j^{(\ell)}[1:32]=addr_j
$$

经过 Attention、output projection、residual 和 FFN 后：

$$
x_j^{(\ell+1)}[1:32]
$$

会混入上下文信息。

如果下一层只“保留前 32 维”，读到的就已经不是纯地址。

因此每个 Transformer block 后执行：

$$
x_j^{(\ell+1)}[1:32]\leftarrow addr_j
$$

剩余 4064 维保持正常更新。

流程：

```text
层输入：前 32 维为 addr
       ↓
K 只读前 32 维；Q/V 读完整 hidden
       ↓
Attention + FFN + residual
       ↓
前 32 维被上下文污染
       ↓
强制覆盖回缓存的 addr
       ↓
进入下一层
```

这里不是修改已经算完的 Key，而是修改传给下一层的 residual stream。

### 9.1 梯度含义

论文称 reset 使用 episode-cached address，并阻断深层 residual stream 对 address 的梯度。

但 \(f_{addr}\) 仍可通过最初的：

$$
e_k=f_\phi([addr_k\Vert cnt_k^t\Vert\pi^t\Vert\rho_k])
$$

获得输入层梯度。

### 9.2 第一层存在实现描述缺口

Slot adapter 是普通 MLP：

$$
f_\phi:\mathbb R^{320}\rightarrow\mathbb R^{4096}
$$

经过一般 MLP 后，其输出前 32 维并不会自然等于原始 `addr`。

但 Key mask 假设 residual stream 前 32 维就是地址。论文没有清楚说明进入第 0 层前是否也执行一次：

$$
e_k[1:32]\leftarrow addr_k
$$

因此第一层 address purity 如何保证，需要查看代码。

---

## 10. Block-Causal Sequence 与位置设计

统一序列包含：

- BPE text tokens；
- image VQ tokens；
- object-slot tokens；
- state tokens；
- past-action tokens；
- `[ACT_Q]`。

Attention 规则包括：

1. **跨帧 block-causal**：当前帧可看历史帧，不能看未来帧；
2. **同帧 slot 双向注意**：保证对象集合内部关系建模；
3. **slot/VQ → action 单向约束**：world-side hidden 不被当前 action 反向污染；
4. **`[ACT_Q]` 看全部历史**；
5. padding slots 被显式 mask。

同一帧的所有 slot 共享同一个 RoPE position index，以减弱 slot 序号本身带来的顺序偏差，并支持 permutation equivariance。

---

## 11. World Head

World head 从每个 slot 的最终 hidden state 并行预测：

$$
(\hat c_k^{t+1},\hat p_k^{t+1})
$$

### 11.1 Content branch 真值

Content branch 的目标为：

$$
c_k^{t+1}=f_{cnt}(raw_k^{t+1})
$$

即下一帧重新提取并编码得到的对象 latent。它间接包含：

- 下一帧 DINO visual feature；
- 下一帧 pose；
- 下一帧 mask shape；
- 固定对象 label。

它不是人工语义标签，而是学习得到的 latent target。

### 11.2 Pose branch 真值

$$
p_k^{t+1}\in\mathbb R^9
$$

包含：

- 3 维 workspace-normalized position；
- 6 维连续 rotation representation。

主实验中 pose 来自模拟器 ground truth quaternion 转换，因此属于 privileged object-level geometry。

### 11.3 Loss

$$
L_{world}=\frac1N\sum_k m_k^{obj}
\left(
\|\hat c_k^{t+1}-c_k^{t+1}\|_2^2
+\lambda_p\|\hat p_k^{t+1}-p_k^{t+1}\|_2^2
\right)
$$

Robot slot 不参与 world loss。

### 11.4 一个未交代清楚的问题

因为 \(c_k^{t+1}\) 由可训练的 \(f_{cnt}\) 产生，正常实现需要说明 target 端是否：

- `detach()`；
- 使用冻结 encoder；
- 使用 EMA target encoder；
- 或离线缓存。

论文只将其记作 target \(c^{t+1*}\)，没有明确说明梯度处理，存在潜在 latent collapse 复现疑点。

---

## 12. Action Head 与 Flow Matching

序列末尾有一个 `[ACT_Q]` token。其最终 hidden state：

$$
H_{ACT\_Q}\in\mathbb R^{4096}
$$

聚合：

- 指令；
- 图像；
- object slots；
- proprioception；
- past actions。

先投影为条件：

$$
c=W_cH_{ACT\_Q}\in\mathbb R^{1024}
$$

Flow MLP 的输入包括：

1. 当前 noisy action chunk \(A_t^\tau\)；
2. flow time \(\tau\) 的 128 维 sinusoidal embedding；
3. 场景条件 \(c\)。

三者拼接后进入 8-block residual MLP，输出：

$$
v_\xi\in\mathbb R^{16\times7}
$$

### 12.1 训练

$$
A_t^\tau=\tau A_t+(1-\tau)\epsilon
$$

$$
\epsilon\sim\mathcal N(0,I),\quad \tau\sim U(0,1)
$$

目标速度为：

$$
A_t-\epsilon
$$

损失：

$$
L_{act}=\|v_\xi(A_t^\tau,\tau,H_{ACT\_Q})-(A_t-\epsilon)\|_2^2
$$

### 12.2 4-step Forward Euler 推理

从噪声开始：

$$
A_t^0\sim\mathcal N(0,I)
$$

取 \(\Delta\tau=1/4\)，执行四次：

$$
A_t^{\tau+\Delta\tau}
=A_t^\tau+\Delta\tau\,v_\xi(A_t^\tau,\tau,c)
$$

最终得到完整 \(16\times7\) 动作 chunk。

“single forward pass” 的准确含义不是整个系统只运行一次，而是：

- 7B trunk 只运行一次；
- Flow action MLP 运行 4 次；
- 16 个动作不是按机器人时间步自回归生成，而是并行更新整个 chunk。

因此更准确的说法是：

$$
\boxed{\text{single trunk forward + four flow-MLP evaluations}}
$$

---

## 13. Auxiliary VQ Head

模型复用 Chameleon 原有 `lm_head` 预测下一帧 VQ image tokens。

特点：

- 只在训练时启用；
- 不增加新的 8192-way classifier 参数；
- 推理时关闭；
- 预测图像不输入动作头，不参与显式规划。

这进一步说明 OA-WAM 的 world prediction 主要是辅助 representation learning。

---

## 14. 总损失

$$
L=L_{act}+\lambda_wL_{world}+\lambda_vL_{vq}
+\lambda_cL_{compose}+\lambda_rL_{role}
$$

权重为：

$$
\{\lambda_w,\lambda_v,\lambda_c,\lambda_r\}
=\{0.5,0.04,0.1,0.05\}
$$

---

## 15. `L_compose`：Distractor Consistency

`L_compose` 由两类增强组成。

### 15.1 Distractor permutation

保持 target/reference 不变，随机打乱 distractor slots 的顺序，并一致地修改：

- slot embeddings；
- pairwise geometry tensor；
- padding mask。

要求原样本与增强样本的：

- slot assignment；
- 动作输出；

保持一致：

$$
L_{perm}
=KL(\alpha_{orig}^{detach}\|\alpha_{aug})
+\|A_{orig}^{detach}-A_{aug}\|_2^2
$$

### 15.2 Distractor insertion

从 batch 中其他样本取一个 distractor，插入当前样本的 padding slot，更新几何与 mask，再施加同样的一致性约束。

其训练目标是：

> 与任务无关的物体被重排或加入时，模型不应改变目标选择与动作。

### 15.3 边界条件

它隐含“distractor 与动作弱耦合”的假设。

若新物体：

- 挡住目标；
- 改变抓取路径；
- 引发碰撞；

正确动作本来就应变化，此时严格 invariance 不成立。

---

## 16. `L_role`：弱对象角色监督

Qwen3-VL 和简单语言规则给出：

- target；
- reference。

辅助 role attention 对 slots 产生 soft assignment，并用 one-hot 标签约束：

$$
L_{role}=KL(\alpha_{target},onehot(target))
+KL(\alpha_{reference},onehot(reference))
$$

作用：

- 训练早期帮助模型建立语言对象与 slot 的绑定；
- 不是直接动作监督；
- 只在前半段训练使用，之后权重设为 0，以避免过度依赖有噪声的弱标签。

### 16.1 Role-query 维度描述不一致

附录一处写 \(\alpha\in\mathbb R^{16\times(N+1)}\)，似乎对应 16-step action；另一处又明确展示 4 个 role queries：

- target；
- reference；
- tool；
- distractor。

更合理的主要结构应为：

$$
\alpha\in\mathbb R^{4\times(N+1)}
$$

论文没有清楚解释 16-step assignment 与 4-role queries 的关系。

---

## 17. 三阶段训练

### Stage 0：Slot-aware trunk pretraining

- 从 Chameleon-7B warm start；
- 全量训练约 7B 参数；
- 约 600k steps；
- 约 2.5T tokens；
- 数据：web image-text、Open X-Embodiment、DROID、RoboCasa、Bridge V2；
- 384×A100-80GB；
- 约 18 天，约 166k A100-hours。

OA mask 在前 5k steps 从全维逐渐收缩到 32 维，之后保持 hard mask。

### Stage I：Slot adapter alignment

冻结 trunk，训练：

- slot adapter；
- \(f_{addr}\)、\(f_{cnt}\)；
- world head。

约 23.8M 参数，50k steps。

### Stage II：Action finetuning

加入：

- flow action head；
- rank-32 LoRA；
- `L_compose`；
- `L_role`。

约 127M trainable parameters，100k steps。

Stage II 只使用标准 LIBERO demonstrations；LIBERO-Plus 作为 OOD 测试，不用于训练。

---

## 18. 主要实验结果

### 18.1 标准 benchmark

| Benchmark | OA-WAM |
|---|---:|
| LIBERO Avg | 97.8% |
| SimplerEnv WidowX Avg | 79.3% |

论文报告其在表中超过已有 VLA/WAM baseline。

### 18.2 LIBERO-Plus

| 维度 | OA-WAM |
|---|---:|
| Camera | 80.5% |
| Robot Init | 89.6% |
| Layout | 82.8% |
| Geo Avg | 84.3% |
| Light | 96.5% |
| Background | 95.9% |
| Language | 85.3% |
| Sensor Noise | 75.6% |
| Overall Avg | 83.9% |

OA-WAM 在几何扰动平均上比 \(\pi_0.5\) 高 4.8 个百分点，但七轴 overall average 低于 \(\pi_0.5\) 的 85.7%。

因此准确结论是：

> OA-WAM 的优势集中在 camera、robot initialization 和几何布局变化，并非 LIBERO-Plus overall SOTA。

Sensor Noise 明显较弱，说明系统强依赖对象分割、跟踪和视觉特征提取的稳定性。

---

## 19. 最有说服力的机制实验

### 19.1 OA 结构消融

| Variant | Key mask | Reset | LIBERO | LP Camera | LP Avg | Swap Binding |
|---|---|---|---:|---:|---:|---:|
| V2 | off | off | 95.4 | 60.5 | 76.2 | 0.06 |
| V1 | off | on | 96.3 | 67.2 | 80.8 | 0.19 |
| V3 | on | off | 96.6 | 70.8 | 83.2 | 0.32 |
| V0 | on | on | 97.8 | 80.5 | 83.9 | 0.87 |

结论：

- OA 约束对标准 LIBERO 影响较小；
- 对 camera/robot 几何 OOD 影响很大；
- mask 与 reset 均有效，且组合有明显交互作用。

这支持“结构性 OOD inductive bias”而非单纯容量增加的解释。

### 19.2 Address Swap Intervention

测试时交换 target slot 与其他 slot 的 address，但不交换各自内容。

交换前：

```text
slot A: Key(addr_A), Value(content_A)
slot B: Key(addr_B), Value(content_B)
```

交换后：

```text
slot A: Key(addr_B), Value(content_A)
slot B: Key(addr_A), Value(content_B)
```

若 action query 原本寻找 \(addr_A\)，交换后会偏向 slot B，并读取 slot B 的 Value。

OA-WAM 的 swap-binding cosine 为 0.87，而 holistic baselines 均不超过 0.09。

该实验表明：

> 训练后的模型确实较强地通过 address Key 路由对象选择。

但它不能证明：

- 上游 Qwen/SAM 初始化的 address 一定正确；
- 自然语言 grounding 已完全解决；
- 所有动作信息都只能通过 slot 通路。

---

## 20. World Head 的真实贡献

附录消融：

| 设置 | LIBERO | LP Camera | LP Avg |
|---|---:|---:|---:|
| Action only | 95.6 | 73.4 | 84.5 |
| With world | 97.8 | 80.5 | 83.9 |

World Head 提升了标准 LIBERO 和 LP Camera，但七轴 LP Avg 并没有提高。

因此论文中最强的贡献证据主要来自：

- address-only Key；
- address reset；
- object slot interface；
- distractor consistency；
- causal swap intervention。

而不是 world prediction 本身。

---

## 21. 这篇论文真正证明了什么

论文较有力地证明：

1. 在同一套感知、训练和模型规模下，限制 slot Key 只依赖固定 address，可显著提升几何 OOD；
2. address reset 对维持多层纯地址 routing 有实际作用；
3. 模型在推理时确实使用 address subspace 进行对象绑定；
4. 对象级表示可以与 Chameleon-style world/action sequence 兼容。

论文没有证明：

1. 目标选择被完全硬编码；
2. `addr` 是全局语义对象 ID；
3. `cnt` 是纯粹、可解释的动态状态；
4. 动作只能通过 object slots 产生；
5. 预测未来后进行了显式规划；
6. 方法已在真实机器人中稳定工作。

---

## 22. 关键局限与批判性阅读

### 22.1 Key-only 约束不保证选对对象

模型仍需学习：

$$
Q^\top K(addr_j)
$$

究竟应对哪个 slot 最大。

Query 本身仍受完整场景影响，因此场景变化仍可能改变目标选择。OA-WAM 只是稳定了 slot Key，不是把 grounding 变成确定规则。

### 22.2 `[ACT_Q]` 可以绕过 slots

`[ACT_Q]` 能直接 attend：

- holistic VQ image tokens；
- text；
- state；
- past actions；
- slots。

所以动作头理论上可以一部分依赖 addressable slots，一部分仍依赖全局视觉 token。OA 约束没有强制所有动作信息必须经过对象接口。

### 22.3 使用模拟器 Ground-Truth Pose

每个对象 slot 包含 9D pose，主实验中来自模拟器真值。

这属于额外的 privileged object-level geometry，跨模型 SOTA 对比并非完全同输入条件。论文只建议真实部署时使用 Depth Pro、VGGT、FoundationPose 等替代，但没有做真实系统验证。

### 22.4 仅有模拟实验

LIBERO、LIBERO-Plus 和 SimplerEnv 都是仿真环境。SimplerEnv Visual Matching 不等于真实机器人部署。

### 22.5 感知栈重且脆弱

系统依赖：

- Qwen3-VL；
- SAM 3；
- DINOv3；
- pose estimator；
- VQ-GAN。

Sensor Noise 性能下降说明上游感知失败会直接破坏 slot interface。

### 22.6 计算成本极高

Stage 0 约 166k A100-hours。最终结果不是“仅在普通 Chameleon 上加一个 32 维 mask”就能轻易复现。

### 22.7 它更像带辅助世界预测的 Object-centric VLA

World prediction 不参与显式 planning 或 rollout。更准确的定位是：

> 一个使用对象级未来监督的 object-centric VLA/WAM，而非依赖想象轨迹做决策的 model-based planner。

---

## 23. 论文内部值得核对的实现疑点

1. **`raw` 维度不一致**：列出的分量为 536 维，文中写 540 维；
2. **第一层 address purity 未说明**：普通 slot adapter MLP 后，前 32 维如何保证等于 `addr`；
3. **slot 数量矛盾**：固定约 16 object slots，但后文出现 24/27 active addresses；
4. **content target 梯度处理不清楚**：是否 detach、冻结或 EMA；
5. **role assignment 维度矛盾**：16-step assignment 与 4 role queries 的关系不清；
6. **latency 数字前后不一致**：Conclusion 与 Appendix G 的 trunk/head 时间不同；
7. **“single forward pass”表述过强**：实际为一次 trunk + 四次 flow MLP。

这些问题不一定否定核心方法，但会影响复现与严格机制解释。

---

## 24. 推荐阅读顺序

### 第一遍：理解核心假设

1. Figure 1：holistic binding failure；
2. Figure 2：整体架构；
3. Section 3.2–3.4：slot、OA attention、prediction heads；
4. Tables 2–4：几何 OOD、消融、swap intervention。

目标：先理解“为什么固定 address Key 可能改善几何变化下的对象绑定”。

### 第二遍：检查实现

1. Appendix B：Qwen/SAM/DINO/pose/raw slot；
2. Appendix C：masked scatter、RoPE、attention mask、reset；
3. Appendix D：world/action head；
4. Appendix G：三阶段训练与成本；
5. Appendix I：完整消融和失败分析。

目标：区分论文主张、实际硬约束和需要模型自己学习的部分。

---

## 25. 阅读时应持续追问的问题

1. 模型是“预测未来后做动作”，还是只把未来预测作为辅助 loss？
2. `addr` 是全局语义 ID，还是 episode 内临时实例句柄？
3. Key-only routing 是否足以保证选对 target？
4. Value 中究竟保留多少原始 `cnt`，多少是上下文化混合信息？
5. `[ACT_Q]` 是否可能绕过 object slots？
6. 几何鲁棒性有多少来自 OA attention，有多少来自 GT object pose？
7. 超过固定 slot capacity 时如何处理？
8. Content target 是否 stop-gradient？
9. 真实机器人中分割、跟踪、pose 和 latency 是否可接受？
10. Swap-binding 证明了“使用 address”，但是否证明了“address 初始化正确”？

---

## 26. 最准确的心智模型

不准确的理解：

```text
addr 显式检索数据库里的 cnt
```

准确的理解：

```text
每个 slot 位置产生一对 Key/Value
Key 只能由 episode 内固定 addr 生成
Value 来自同一 slot 位置的完整 contextual hidden state
模型学习 query 应该给哪个 Key 更高权重
```

即：

$$
\boxed{
\text{OA-WAM 稳定的是对象路由接口，而不是把对象选择变成硬编码查表}
}
$$

---

## 27. 总体评价

OA-WAM 的核心想法明确且有价值：把“对象身份用于路由”与“对象当前状态用于信息传递”在 Attention Key 路径上做结构性区分。它最强的证据不是标准 benchmark 上的小幅增益，而是：

- 几何 OOD 上的非对称提升；
- mask/reset 的系统消融；
- address swap 的因果干预。

但它的结论应限定为：

> 在依赖强对象感知与模拟器 pose 的系统中，episode-local address-only Key routing 能显著改善几何变化下的目标绑定稳定性。

它尚未证明通用真实机器人对象寻址已经解决，也没有实现显式基于未来想象的规划。作为研究方向，它最大的启发是：

$$
\boxed{
\text{不要只问世界模型能否预测未来，还要问它预测出的状态是否能被动作模块稳定地按对象查询。}
}
$$

---

## 28. 相关笔记

- [[DreamZero_Technical_Report|DreamZero]]：同属 World Action Model 路线，但使用联合视频—动作 flow matching 表示和生成未来。
- [[WorldVLA 论文综述(不建议读)|WorldVLA]]：对比共享 Transformer 的辅助图像预测与对象级 slot/world head 监督。
- [[Pi0_7_technical_report|π0.7]]：对比通过 subgoal image 提供视觉上下文和通过 object address 稳定目标绑定。
- [[RT-2 论文综述|RT-2]]：对比全局视觉语言 token 条件动作生成和对象可寻址的结构化 VLA 表示。
