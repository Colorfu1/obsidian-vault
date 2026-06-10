---
github_repo: 
title: pi0.5 文章综述
type: paper_note
topic: robotics_foundation_model
status: mature
importance: high
updated: 2026-06-10
tags:
  - pi0.5
  - vla
  - flow-matching
  - adaptive-rmsnorm
  - robotics
---
# Pi_0.5文章综述请求

**User:** Anonymous  
**Created:** 5/18/2026 16:35:46  
**Updated:** 5/22/2026 16:12:44  
**Exported:** 5/22/2026 16:13:01  
**Link:** [https://chatgpt.com/c/6a0acf53-a764-83ec-a08d-6cb77711ab35](https://chatgpt.com/c/6a0acf53-a764-83ec-a08d-6cb77711ab35)  

## Response:
5/22/2026, 4:11:35 PM

下面把 **π0.5** 这篇文章以及我们前面讨论过的关键点，整理成一份比较系统的知识笔记。

---

# 1. π0.5 想解决什么问题？

π0 / 早期 VLA 模型已经能做很多机器人操作任务，但它们大多还是：

```text
当前图像 + 当前状态 + 语言指令
        ↓
直接输出 low-level action chunk
```

这对于短任务可以，但对真实家庭环境里的长程任务很困难，比如：

```text
clean the kitchen
clean the bedroom
put the dishes in the sink
put the laundry in the basket
```

这些任务不是一步完成，而是很多子步骤组合起来：

```text
clean the kitchen
  → find plate
  → pick up plate
  → move to sink
  → put plate in sink
  → pick up cup
  → put cup in sink
  → wipe table
  → ...
```

π0.5 的核心目标就是：**让 VLA 能在新家庭、新物体、新布局里执行更长程、更开放的任务。**

论文中强调，π0.5 使用大量异构数据，包括目标 mobile manipulator 数据、其他机器人数据、高层语义任务、verbal instruction，以及 web 多模态数据；第一阶段训练中，绝大多数样本并不是目标移动机器人家庭任务数据，而是来自其他机器人或 web 等来源。`pi_0.5.pdf`

---

# 2. π0.5 的核心思想

我会把 π0.5 概括成：

```text
π0.5 = π0 的 flow matching action expert
       + high-level subtask prediction
       + heterogeneous co-training
       + web / robot / verbal instruction data
```

也就是说，它不是完全推翻 π0，而是在 π0 的基础上增强了两件事：

```text
1. 高层语义子任务预测：
   先判断当前应该做什么 subtask

2. 异构数据共训练：
   用机器人数据 + web 数据 + high-level semantic data
   提升 open-world generalization
```

所以 π0.5 的完整论文系统更像：

```text
high-level instruction + observation
        ↓
AR text generation
        ↓
subtask command
        ↓
flow matching action expert
        ↓
continuous action chunk
```

例如：

```text
"clean the kitchen"
        ↓
"pick up the plate"
        ↓
continuous robot action chunk
```

论文明确说，runtime 时模型先预测 semantic subtask，然后基于这个 subtask 预测 low-level robot action chunk。`pi_0.5.pdf`


---

# 3. 数学分解：为什么要有 `\hat{ℓ}`？

论文把模型分布写成：

```text
πθ(a_{t:t+H}, \hat{ℓ} | o_t, ℓ)
```

其中：

```text
o_t       = 当前观察，包括图像、机器人状态等
ℓ         = high-level task prompt，比如 "clean the kitchen"
\hat{ℓ}   = 模型生成的文本输出，可以是 subtask，比如 "pick up the plate"
a_{t:t+H} = action chunk
```

然后分解成：

```text
πθ(a_{t:t+H}, \hat{ℓ} | o_t, ℓ)
=
πθ(a_{t:t+H} | o_t, \hat{ℓ})
πθ(\hat{ℓ} | o_t, ℓ)
```

这句话的含义是：

```text
高层：
  根据 observation + high-level instruction
  预测当前 subtask

低层：
  根据 observation + subtask
  预测 action chunk
```

关键点是：

```text
action distribution 不直接依赖原始 ℓ，
而是依赖 \hat{ℓ}
```

也就是说，低层 action expert 不应该直接处理 `"clean the kitchen"` 这种很抽象的长程目标，而是处理 `"pick up the plate"` 这种短期、可执行的命令。论文 Section IV-A 明确写了这个分解，并说明 high-level inference 对应 `πθ(\hat{ℓ}|o_t,ℓ)`，low-level inference 对应 `πθ(a_{t:t+H}|o_t,\hat{ℓ})`。`pi_0.5.pdf`

---

# 4. Eq.1 里为什么又写成 `ℓ`？

我们之前讨论过一个 notation 问题：Eq.1 里的 action expert 写成：

```text
f^a_θ(a^{τ,ω}_{t:t+H}, o_t, ℓ)
```

但前面又说 action 只依赖 `\hat{ℓ}`。

我的理解是：**Eq.1 里的 `ℓ` 是一个泛化的 text prompt 记号，在 action loss 这一路里，它实际应该理解成 low-level command，也就是前面分解里的 `\hat{ℓ}`。**

更严格的写法应该是：

```text
f^a_θ(a^{τ,ω}_{t:t+H}, o_t, \hat{ℓ})
```

训练时通常是 ground-truth subtask：

```text
f^a_θ(..., o_t, \hat{ℓ}_{gt})
```

推理时是模型自己先生成的 subtask：

```text
\hat{ℓ} ~ πθ(\hat{ℓ}|o_t, ℓ)

a_{t:t+H} ~ πθ(a_{t:t+H}|o_t, \hat{ℓ})
```

所以这不是结构上的矛盾，而是论文公式里符号有一定复用。

---

# 5. π0.5 的模型结构

π0.5 基于 VLM 初始化，大体结构是：

```text
SigLIP vision encoder
        +
Gemma language model
        +
action expert
        +
flow matching head
```

可以理解为：

```text
PaliGemma backbone:
  image + text → text

π0 / π0.5 VLA:
  image + text + noisy action → continuous action
```

在 openpi 代码中，`Pi0` 类里会创建 `_siglip.Module(...)` 作为图像模块，同时创建 `_gemma.Module(...)` 作为 LLM 部分。`embed_prefix()` 里把图像 token 和文本 token 作为 prefix；`embed_suffix()` 里把 noisy action 通过 `action_in_proj` 投影成 action tokens，再送给 action expert。`pi0.py`

### 5.1 整体 backbone：与 π0 的继承关系

π0.5 沿用 π0 的 PaliGemma backbone：

```
SigLIP (400M vision encoder)
        +
Gemma (2.6B language model)
        +
action expert (300M)
```

两个 transformer 的维度与 π0 相同：

| 模块 | width | depth | mlp_dim | num_heads | head_dim |
|---|---|---|---|---|---|
| VLM (Gemma) | 2048 | 18 | 16384 | 18 | 256 |
| action expert | 1024 | 同上 | 4096 | 同上 | 同上 |

关键差异在于 π0.5 对 **timestep 注入方式** 和 **attention masking** 做了专门改动。

---

### 5.2 Timestep 注入：adaptive RMSNorm（π0.5 专有）

### π0 的做法

π0 直接把 flow matching timestep `τ` **与 noisy action 融合**后再送入 transformer：

```
τ (scalar)
    ↓
fuse with noisy action
    ↓
action expert
```

### π0.5 的改动

π0.5 把 timestep 的处理**独立出来**，用一个专门的 MLP 编码，再通过 **adaptive RMSNorm** 逐层注入：

```
τ (scalar)
    ↓
sinusoidal positional encoding ϕ(τ)    # τ → R^w
    ↓
W1 · ϕ(τ)  →  Swish  →  W2  →  Swish  # 两层 MLP
    ↓
timestep embedding
    ↓
adaptive RMSNorm  →  每一层 action expert
```

完整公式：

```
timestep_emb = swish(W2 · swish(W1 · ϕ(τ)))
```

其中 W1, W2 ∈ R^{w×w}，ϕ 是正弦位置编码函数。

**为什么用 adaptive RMSNorm 而不是直接相加？**

普通加法只能做全局偏移，而 adaptive RMSNorm 可以让 timestep embedding **分别控制每一层特征的 scale 和 shift**，使 action expert 在不同 denoising 阶段（τ 小 = 接近纯噪声，τ 大 = 接近真实 action）有不同的处理行为，表达能力更强。

---

### 5.3 Attention Masking 模式

π0.5 的 attention mask 需要同时处理三类 token：

```
1. prefix tokens：图像 patch、文本 prompt、proprioceptive state
2. FAST action tokens：离散化的 action token（预训练阶段使用）
3. action expert embeddings：flow matching 连续 action（后训练阶段使用）
```

这三类 token 的 attention 规则如下：

```
prefix tokens
  → 对图像、prompt、状态使用 bidirectional (full prefix) attention
  → 任何 token 都可以 attend to prefix

FAST action tokens
  → attend to prefix（双向）
  → autoregressive attend to 之前的 FAST token
  → 不 attend to action expert embeddings

action expert embeddings
  → attend to prefix（双向）
  → attend to 其他 action expert embeddings（双向）
  → 不 attend to FAST action tokens
```

示意图（简化版）：

```
              | Image | Prompt | State | FAST tokens | AE embeddings |
Image         |   ✓   |   ✓    |   ✓   |      ✓      |      ✓        |
Prompt        |   ✓   |   ✓    |   ✓   |      ✓      |      ✓        |
State         |   ✓   |   ✓    |   ✓   |      ✓      |      ✓        |
FAST tokens   |   ✓   |   ✓    |   ✓   |   causal    |      ✗        |
AE embeddings |   ✓   |   ✓    |   ✓   |      ✗      |      ✓        |
```

**为什么 FAST tokens 和 AE embeddings 互相不 attend？**

FAST tokens（离散）和 action expert embeddings（连续 flow matching）是同一个 action 的**两种并行表示**，在训练时同时优化（对应 Eq.1 里的两项 loss）。如果它们互相 attend，信息会泄漏，导致两条路径互相"抄答案"，破坏独立性。

**信息流向总结：**

```
VLM（图像 + 文本 + 状态）
        ↓  单向
action expert（连续 action 生成）
```

VLM 的表示流向 action expert，但 action expert 的输出不会反过来影响 VLM 的表示。

---

### 5.4 Flow Matching Timestep 采样分布

π0.5 沿用 π0 的 timestep 采样策略，**偏向低 timestep**（接近真实 action 的一端），而不是标准的均匀采样：

```
标准做法：τ ~ U(0, 1)

π0 / π0.5：τ ~ Beta((s - τ)/s;  α=1.5, β=1)
```

其中 s = 0.999，排除 τ > s 的区间（因为积分步长 δ > 1 - s 时这些步不需要）。

直觉：在 denoising 的后期（τ 接近 1，action 接近真实值），微小误差对最终输出影响最大，因此多采样这部分 timestep，让模型在"精细修正"阶段学得更好。

---

### 5.5 与原综述第 5 节的对应关系

原综述第 5 节描述的结构：

```
PaliGemma backbone:
  image + text → text

π0 / π0.5 VLA:
  image + text + noisy action → continuous action
```

这个描述是正确的，但过于简略。结合本节，完整的 π0.5 推理流程是：

```
输入：
  [图像×4, text prompt, proprioceptive state]
        ↓
VLM（SigLIP + Gemma）+ bidirectional prefix attention
        ↓（单向信息流）
action expert（接收 prefix + noisy action）
  ↑
  timestep τ → sinusoidal encoding → MLP → adaptive RMSNorm 注入每层
        ↓
预测 vector field → 10步 denoising → continuous action chunk [H, action_dim]
```

高层推理（subtask 生成）走的是 VLM 的 text token logits 路径，与 action expert 路径并行但独立，互不干扰。
---

# 6. text head 和 action expert 的区别

这点非常关键。

π0.5 不是把所有东西混成一个普通文本序列：

```text
"pick up the plate" → a1 → a2 → a3 → ...
```

它有两类输出路径：

```text
1. text token logits
   用来输出 subtask / caption / VQA answer / bounding box token / FAST action token

2. action output tokens
   用 action expert + flow matching 输出 continuous action
```

所以：

```text
subtask prediction:
  autoregressive text decoding

action generation:
  flow matching continuous denoising
```

这也是为什么不会出现：

```text
a1, a2, a3, ..., 你好我是xxx
```

因为 continuous action expert 输出的是：

```text
[H, action_dim]
```

也就是 float tensor，不是 vocab logits。

---

# 7. flow matching action expert 是怎么工作的？

π0.5 的 low-level action 不是一步输出 token，而是用 flow matching 生成 continuous action chunk。

训练时构造 noisy action：

```text
a^{τ,ω}_{t:t+H} = τ a_{t:t+H} + (1 - τ)ω
```

其中：

```text
a_{t:t+H} = 真实 action chunk
ω         = Gaussian noise
τ         = flow matching timestep
```

模型目标是预测 vector field：

```text
ω - a_{t:t+H}
```

也就是 Eq.1 里的 flow matching loss。

推理时则从 Gaussian noise 开始，做若干步 denoising，最后得到 action chunk。论文中说 inference 时先用 autoregressive decoding 得到 text tokens `\hat{ℓ}`，然后 conditioned on text tokens 通过 10 步 denoising 产生 action。`pi_0.5.pdf`

openpi 代码中 `sample_actions()` 也体现了这个过程：先初始化 Gaussian noise，然后循环调用 `embed_suffix()` 和 action expert 得到 `v_t`，再更新 `x_t = x_t + dt * v_t`，最后返回 `x_0`。`pi0.py` `pi0.py`

---

# 8. timestep MLP 是什么？

Appendix E 里提到：

```text
swish(W2 · swish(W1 · ϕ(τ)))
```

这就是把 scalar timestep `τ` 编码成一个高维 timestep embedding。

流程是：

```text
τ
↓
sinusoidal positional encoding ϕ(τ)
↓
Linear W1
↓
Swish
↓
Linear W2
↓
Swish
↓
timestep embedding
```

然后这个 timestep embedding 会通过 adaptive RMSNorm 注入 action expert 的每一层。

直觉是：

```text
τ 小：
  输入接近纯 noise，需要大幅 denoise

τ 大：
  输入接近真实 action，只需要微调
```

所以 action expert 必须知道当前 denoising 处在哪个时间步。

openpi 里 `pi05=True` 时，确实会创建：

```python
self.time_mlp_in
self.time_mlp_out
```

并且在 `embed_suffix()` 里对 timestep 做 sine-cosine embedding，再过两层 MLP + swish，得到 `adarms_cond`。`pi0.py`

---

# 9. joint action 和 end-effector action 的区别

论文里说：

> For all action data, we train the model to predict target joint and end-effector poses. To differentiate the two, we add `<control mode> joint/end effector <control mode>` to the text prompt.

这说明 π0.5 要兼容不同数据集里的 action 表示。

以单臂 6 DoF 机械臂为例。

## joint-space action

输出每个关节的目标：

```text
[q1, q2, q3, q4, q5, q6, gripper]
```

含义是：

```text
第 1 个关节转到多少
第 2 个关节转到多少
...
夹爪打开多少
```

## end-effector-space action

输出末端执行器位姿：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

或者：

```text
[x, y, z, qx, qy, qz, qw, gripper]
```

含义是：

```text
夹爪移动到某个空间位置和姿态
```

两者不是同一次推理同时输出，而是通过 prompt 里的 control mode 选择当前 action 的语义：

```text
"<control mode> joint <control mode>"
  → 输出 joint target

"<control mode> end effector <control mode>"
  → 输出 EE target
```

所以 control mode token 的作用是告诉模型：**当前 action 数组应该被解释成哪种控制空间。**

---

# 10. π0.5 的训练流程

论文中 π0.5 是两阶段训练。

## 第一阶段：pre-training

特点：

```text
1. 使用非常异构的数据
2. 所有任务，包括 robot action，都表示成 discrete tokens
3. 训练目标类似标准 autoregressive next-token prediction
```

数据包括：

```text
MM: mobile manipulator 数据
ME: diverse non-mobile manipulator 数据
CE: cross-embodiment lab 数据
HL: high-level subtask prediction 数据
WD: web multimodal 数据
```

论文说 pre-training 使用 broad range of robot and non-robot data，并且通过 standard autoregressive transformer 做 text、object locations、FAST encoded action tokens 的 next-token prediction。`pi_0.5.pdf`

## 第二阶段：post-training

特点：

```text
1. 专门适配 mobile manipulation
2. 加入 flow matching action expert
3. 保留 text prediction 能力
4. 加入 verbal instruction data
```

这一阶段优化的是 Eq.1：

```text
cross entropy text loss
+
flow matching action loss
```

也就是同时训练：

```text
文本输出能力
+
连续动作生成能力
```

论文说 post-training 的目标是让模型 specialize 到 home mobile manipulation，并加入 action expert 以实现连续动作和高效实时推理。`pi_0.5.pdf`

---

# 11. 为什么已经有 pretrained VLM，还要继续加 web data？

这是我们讨论里一个关键问题。

答案是：

```text
预训练 VLM 见过 web 数据
≠
robot training 后仍然保留这些能力
≠
它能把这些能力正确用于机器人 subtask / action
```

继续加 web data 有几个作用：

```text
1. 防止机器人数据微调时遗忘开放世界视觉语言知识
2. 保持 OOD 物体识别和语言 grounding 能力
3. 让 web 视觉语言能力适配 VLA 的输入输出格式
4. 改善 high-level subtask inference
```

论文 ablation 也支持这一点：去掉 web data 后，对 OOD object 的 language following 明显变差，作者认为 web data 包含广泛的物理物体知识，使模型能理解未见类别。`pi_0.5.pdf`

所以 web data 不是简单“再训练一遍 VLM”，而是让模型在机器人共训练过程中持续保留并使用开放世界语义能力。

---

# 12. π0.5 相比 π0 的主要改进

可以这样对比：

## π0

```text
observation + instruction
        ↓
flow matching action expert
        ↓
action chunk
```

主要是一个强 low-level VLA policy。

## π0.5

```text
observation + high-level instruction
        ↓
high-level subtask prediction
        ↓
observation + subtask
        ↓
flow matching action expert
        ↓
action chunk
```

同时训练数据更丰富：

```text
π0.5 = π0 low-level flow policy
       + high-level semantic supervision
       + web data
       + verbal instruction
       + more heterogeneous robot data
```

论文中也明确对比了 π0、π0-FAST+Flow 和 π0.5：π0.5 额外使用 HL 和 WD 数据，并采用 hybrid training；实验中 π0.5 在 mock home 测试环境显著优于 π0 和 π0-FAST+Flow。`pi_0.5.pdf`

---

# 13. openpi 代码和论文完整系统的区别

我们去 openpi 代码里看过一个重要细节：

**当前开源 openpi repo 里，并没有完整暴露论文中 high-level subtask AR generation → low-level flow action 的自动 pipeline。**

README 里明确写了：当前 repo 对 π0.5 主要支持 **flow matching head** 的训练和推理。`README.md`

代码里的 inference 路径是：

```text
obs + prompt
     ↓
policy.infer(...)
     ↓
model.sample_actions(...)
     ↓
continuous action chunk
```

`Policy.infer()` 直接调用 `self._sample_actions(...)` 并返回 `"actions"`，没有发现类似：

```python
subtask = model.generate_text(...)
actions = model.sample_actions(prompt=subtask)
```

这样的两阶段自动流程。`policy.py`

所以：

```text
论文完整系统:
  high-level prompt
      → AR 生成 subtask
      → flow matching action

openpi 当前公开实现:
  prompt
      → flow matching action
```

这说明开源版本更像是 π0.5 低层 flow action head 的开放实现，而不是完整家庭机器人系统 runtime。

---

# 14. π0.5 仍然是 BC 范式吗？

是的，本质上还是 **Behavior Cloning / offline supervised learning**。

训练时：

```text
给定 observation + language
监督学习 expert action / subtask
```

推理时：

```text
给定当前 observation + prompt
输出 action chunk
```

它没有显式 online RL，也没有在执行过程中通过奖励或环境反馈实时调整策略。

更准确地说：

```text
π0.5 是 receding-horizon closed-loop，
但 action chunk 内部是 open-loop。
```

也就是说，它可以周期性重新观察、重新推理，但在一个 action chunk 内，如果物体滑动、抓偏、碰撞，chunk 本身不会自动根据真实反馈修正。

---

# 15. 这是当前 VLA 的通病吗？

基本是。

很多 VLA / diffusion policy / flow policy 都是：

```text
observation + instruction
        ↓
predict action chunk
        ↓
execute several steps
        ↓
observe again
        ↓
predict next chunk
```

这种方式不是完全 open-loop，但也不是严格实时反馈控制。

它的 trade-off 是：

```text
chunk 长：
  推理快，动作平滑
  但中间偏差难修正

chunk 短：
  闭环更强
  但推理成本高，动作可能抖
```

所以当前 VLA 常见问题是：

```text
1. 依赖训练分布覆盖
2. 缺少显式世界模型
3. chunk 内无法实时修正
4. 对接触误差、物体滑动、遮挡变化敏感
```

---

# 16. 和 WAM 的关系

如果你说的 WAM 是 **World Action Model**，那它和 π0.5 / VLA 的主要区别是：

```text
普通 VLA:
  observation + instruction → action

WAM:
  observation + instruction → future world/state + action
```

也就是说，WAM 不只预测动作，还预测未来世界会怎样变化。

这个优势是：

```text
模型可以用 predicted future 和 actual future 做对比
如果真实世界偏离想象，就可以提前 replan
```

所以 WAM 提供了一个更好的闭环接口。

但 WAM 也不是天然解决问题。如果它仍然固定执行一段 action chunk，而不检查真实 observation，那么它也有 open-loop chunk 的问题。区别是：

```text
VLA:
  没有显式 future state，很难判断是否偏离

WAM:
  有 predicted future，可以做 consistency check / adaptive replanning
```

所以更准确地说：

```text
WAM 不是天然没有这个困扰，
但它提供了一个更容易解决这个困扰的机制。
```

---

# 17. 最终整体理解

我建议把 π0.5 记成下面这条主线：

```text
π0.5 不是一个在线规划/在线强化学习系统，
而是一个更强的 BC/VLA 系统。

它通过：
  1. 预训练 VLM backbone
  2. 异构机器人数据
  3. web 多模态数据
  4. high-level subtask prediction
  5. verbal instruction data
  6. flow matching action expert

把原本 “instruction → action” 的 VLA，
增强成 “high-level instruction → subtask → action” 的层次化 VLA。
```

最核心的贡献不是单点技术，而是一个组合：

```text
语义层：
  学会把长程任务拆成当前可执行 subtask

动作层：
  用 flow matching 生成连续 action chunk

数据层：
  用 web + 多机器人 + verbal instruction 增强泛化

训练层：
  pretraining 用 discrete token 高效学习
  post-training 加 action expert 做实时连续控制
```

所以一句话总结：

**π0.5 是在 π0 的 flow-based VLA 基础上，通过 high-level subtask prediction 和大规模异构共训练，把低层动作模仿扩展到更开放、更长程的家庭机器人任务；它的泛化主要来自语义分解和数据多样性，但本质上仍然是 BC/chunked action policy，缺少真正的在线世界反馈和实时纠错机制。**



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
