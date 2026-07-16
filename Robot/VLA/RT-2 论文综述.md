---
title: RT-2 论文综述
type: paper_note
topic: vision_language_action_model
status: mature
importance: high
updated: 2026-07-16
tags:
  - rt-2
  - vla
  - vlm
  - action-tokenization
  - web-knowledge-transfer
  - semantic-generalization
  - robotics
---
# RT-2 技术报告：Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

## 1. 论文基本信息

**论文标题**：RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control  
**机构**：Google DeepMind  
**核心关键词**：VLA、VLM、robot control、action tokenization、co-fine-tuning、semantic generalization  
**核心目标**：把 web-scale 视觉语言模型中的语义理解、符号理解、常识推理能力迁移到机器人闭环低层控制中。`RT2.pdf`

---

## 2. 一句话总结

**RT-2 的核心思想是：把机器人低层动作离散化成 token，让预训练 VLM 像生成文本一样生成动作序列，并通过 web-scale VLM 数据 + robot trajectory 数据共同 fine-tune，使同一个模型同时保留视觉语言理解能力和机器人控制能力。**

它不是：

- VLM 做 high-level planner；
- 再调用单独低层 policy；
- 也不是额外加一个独立 action head。

而是：

- 图像 + 指令输入 VLM；
- VLM 直接 autoregressive 输出 action token；
- action token 被 detokenize 成机器人控制量；
- 闭环重复执行。

---

## 3. 背景与动机

传统机器人 imitation learning / behavior cloning 模型，例如 RT-1，依赖大量机器人数据来学视觉到动作的映射。但机器人数据非常昂贵，规模远小于互联网图文数据。

与此同时，PaLI-X、PaLM-E 这类 VLM 已经在大规模图文数据上学到了很多能力：

- 识别新物体；
- 理解符号、数字、图标；
- 理解颜色、类别、关系；
- 做简单语言推理；
- 理解多语言指令；
- 具备一定 commonsense reasoning。

RT-2 想回答的问题是：

> 能不能让机器人低层控制也直接受益于这些 VLM 预训练知识？

论文的答案是：可以，但主要迁移的是 **语义泛化能力**，不是新的物理运动技能。模型仍然只能执行机器人数据中见过的 pick / place / move / open / close 等技能，只是能把这些技能部署到新的语义场景里。`RT2.pdf`

---

## 4. 整体 Pipeline

> [!figure] 论文原始模型结构图
> ![[attachments/paper-figures/rt-2-method-overview.png]]
> RT-2 将机器人动作表示为文本 token，与互联网视觉语言数据共同训练，并在部署时反 token 化为闭环机器人控制。原图来自 [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control（arXiv:2307.15818）](https://arxiv.org/abs/2307.15818)，由论文源文件高分辨率导出。

RT-2 的完整 pipeline 可以分为 6 步。

```text
Robot observation image
        +
Language instruction
        ↓
VQA-style prompt construction
        ↓
Pretrained VLM backbone
PaLI-X / PaLM-E
        ↓
Autoregressive action-token generation
        ↓
Detokenize action tokens into robot action
        ↓
Closed-loop robot execution
```

更具体地说：

1. 机器人获得当前相机图像；
2. 用户给出自然语言任务，例如 `pick coke can`；
3. 构造 VQA 风格输入：

```text
Q: what action should the robot take to pick coke can? A:
```

4. VLM 根据图像和文本 prompt 输出一串 action tokens；
5. 系统把 action tokens 还原成机器人动作；
6. 机器人执行一步低层动作；
7. 下一帧重新输入图像和指令，继续闭环控制。

---

## 5. Action Space 与动作离散化

RT-2 沿用 RT-1 的动作表示方式。动作空间包括：

```text
terminate
Δpos_x
Δpos_y
Δpos_z
Δrot_x
Δrot_y
Δrot_z
gripper_extension
```

也就是：

- 一个 termination command；
- 3D 末端位置增量；
- 3D 末端旋转增量；
- 夹爪开合程度。

连续维度会被均匀离散成 256 个 bin。也就是说，每个连续动作维度最终会变成一个整数：

$$
b_i \in \{0, 1, 2, ..., 255\}
$$

一个动作可以写成：

$$
a_t = [b_0, b_1, b_2, ..., b_7]
$$

例如：

```text
[1, 140, 120, 128, 132, 135, 106, 127]
```

其中每个数字不是普通数值回归目标，而是一个 **离散分类 token**。论文在 Sec. 3.2 里说明动作被表示成离散 bin 的 ordinals，并进一步关联到模型 tokenizer 里的 token。`RT2.pdf`

---

## 6. Action Tokenization：最关键部分

RT-2 的关键不是“动作离散化”本身，而是：

> 如何把离散动作 bin 接进 VLM 的文本生成接口。

假设动作离散结果是：

```text
[140, 120, 128]
```

RT-2 需要把它变成 VLM 可以预测的 token 序列：

```text
[token_for_140, token_for_120, token_for_128]
```

论文针对 PaLI-X 和 PaLM-E 用了两种不同做法。

---

## 7. PaLI-X 的做法：直接复用数字 token

论文说：

> For PaLI-X, integers up to 1000 each have a unique token, so we simply associate the action bins to the token representing the corresponding integer.

这句话的意思是：

PaLI-X 的 tokenizer 里，`0` 到 `1000` 这些整数都可以被表示成单独一个 token。

例如字符串：

```text
"140"
```

在 PaLI-X 里会被 tokenizer 编码成一个原子 token：

```text
token("140")
```

不会被拆成：

```text
token("1"), token("4"), token("0")
```

因此动作 bin 到 token 的映射可以直接写成：

$$
\phi(k) = \text{token\_id}(\text{str}(k))
$$

例如：

$$
\phi(140) = \text{token\_id}("140")
$$

所以对 PaLI-X 来说，动作 `[140, 120, 128]` 可以自然序列化成字符串：

```text
"140 120 128"
```

经过 tokenizer 后得到：

```text
[token("140"), token("120"), token("128")]
```

这个地方的空格只是为了让 tokenizer 明确数字边界。

---

## 8. PaLM-E 的做法：覆盖 256 个低频 token

PaLM-E 没有 PaLI-X 那种方便的数字 token 表示。也就是说，如果你写：

```text
"140 120 128"
```

PaLM-E 的 tokenizer 不保证它会被拆成：

```text
[token_for_140, token_for_120, token_for_128]
```

它可能会被拆成数字片段、字符片段，或者其他 subword 形式。

因此论文说：

> For the PaLM-E model, which does not provide this convenient representation of numbers, we simply overwrite the 256 least frequently used tokens to represent the action vocabulary.

更准确地说，作者选出原始词表中 256 个最少使用的 token id：

```text
rare_token_ids = [r_0, r_1, ..., r_255]
```

然后建立映射：

$$
\phi(k) = r_k,\quad k \in \{0,1,...,255\}
$$

例如：

```text
action bin 140 -> rare_token_ids[140]
action bin 120 -> rare_token_ids[120]
action bin 128 -> rare_token_ids[128]
```

所以对 PaLM-E，动作 `[140, 120, 128]` 的 GT 不应该理解成普通字符串 `"140 120 128"`，而应该理解成 token id 序列：

```text
[rare_token_ids[140], rare_token_ids[120], rare_token_ids[128]]
```

也就是：

$$
[\phi(140), \phi(120), \phi(128)]
$$

这个点非常重要：

**PaLM-E 的 action GT 本质上应该直接构造 token id 序列，而不是依赖普通数字字符串经过 tokenizer 后的结果。**

论文里的 “single string” 更像是高层描述：把 action vector 包装成一种 VLM 可以生成的序列形式。但实现上必须保证每个 action bin 对应一个确定的、原子的 token id。`RT2.pdf`

---

## 9. 为什么论文说要 concatenate into a single string？

论文中的原句大意是：把动作向量转换成一个字符串，把每个动作维度的 token 用空格连接起来，例如：

```text
"terminate Δpos_x Δpos_y Δpos_z Δrot_x Δrot_y Δrot_z gripper_extension"
```

可能实例化成：

```text
"1 128 91 241 5 101 127 255"
```

这句话容易误导，因为它看起来像是所有模型都直接用普通文本字符串。

更准确的理解是：

### 9.1 对 PaLI-X

可以真的用数字字符串：

```text
"140 120 128"
```

因为 PaLI-X 保证这些数字是单 token。

### 9.2 对 PaLM-E

不能依赖普通数字字符串。应该做的是：

```python
bins = [140, 120, 128]
target_token_ids = [rare_token_ids[b] for b in bins]
```

也就是说，**string 只是概念上的“输出序列格式”，真正进入模型的是 token ids**。

最终进入模型 embedding 的永远不是字符串，而是：

```text
input_ids -> embedding lookup -> transformer
```

所以正确关注点不是 `"140 120 128"` 这个表面字符串，而是：

$$
\text{action bins}
\rightarrow
\text{action token ids}
\rightarrow
\text{embedding lookup}
\rightarrow
\text{autoregressive prediction}
$$

---

## 10. 不同动作维度是否共享同一套 256 个 token？

是的。

如果动作有多个维度，每个维度都离散到 0 到 255，那么这些维度共享同一套 action vocabulary：

```text
A_0, A_1, ..., A_255
```

例如：

```text
Δpos_x = 128
Δpos_y = 128
Δrot_z = 128
```

它们都会使用同一个 token：

```text
A_128
```

也就是同一个 token id、同一个 embedding row。

但是它们的语义不会混淆，因为模型知道它们出现在不同的序列位置上。

例如 action token 顺序固定为：

```text
terminate, Δpos_x, Δpos_y, Δpos_z, Δrot_x, Δrot_y, Δrot_z, gripper
```

那么第二个位置上的 `A_128` 表示 `Δpos_x` 的第 128 个 bin；第三个位置上的 `A_128` 表示 `Δpos_y` 的第 128 个 bin；第七个位置上的 `A_128` 表示 `Δrot_z` 的第 128 个 bin。

同一个 token embedding 在不同位置进入 Transformer 后，会叠加不同的位置编码，并受到不同上下文影响。因此最终 hidden state 不同。

可以理解成：

$$
\text{same token id} + \text{different position/context}
\rightarrow
\text{different hidden representation}
$$

所以共享 token vocabulary 是可以成立的。

---

## 11. 训练时模型到底学什么？

RT-2 的训练目标仍然是 VLM 常见的 next-token prediction。

对一个机器人样本，输入可以理解为：

```text
<image tokens>
Q: what action should the robot take to pick coke can? A:
```

目标输出是 action token 序列：

```text
A_1 A_140 A_120 A_128 A_132 A_135 A_106 A_127
```

训练时使用 teacher forcing。

也就是说，模型预测过程是：

| 输入上下文 | 预测目标 |
|---|---|
| image + prompt | 第 1 个 action token |
| image + prompt + 第 1 个 action token | 第 2 个 action token |
| image + prompt + 前 2 个 action token | 第 3 个 action token |
| ... | ... |

loss 是标准交叉熵：

$$
\mathcal{L}
=
-\sum_{i=1}^{L}
\log p(t_i \mid \text{image}, \text{instruction}, t_{<i})
$$

其中：

- $t_i$ 是第 $i$ 个 action token；
- $L$ 是动作 token 序列长度；
- 对 RT-2 来说，next-token prediction 等价于 robot behavior cloning loss。

论文 Appendix E 明确说，RT-2 使用 next token prediction objective，对应机器人学习中的 behavior cloning loss。`RT2.pdf`

---

## 12. 为什么不直接加一个 action head？

从纯机器人策略建模角度看，当然可以加一个 action head：

$$
h_i \rightarrow W_{\text{action}} \rightarrow \text{logits over 256 bins}
$$

但是 RT-2 的目标是最大程度复用 VLM 的原有文本生成接口。因此它使用原来的 vocabulary head：

$$
h_i \rightarrow W_{\text{vocab}} \rightarrow \text{logits over vocabulary}
$$

在机器人动作任务中，只取其中 action token 对应的 logits：

$$
W_{\text{vocab}}[V_{\text{action}}]
$$

从效果上看，这近似于一个 256 类分类器；但架构上它仍然是 LLM/VLM 的文本 token 输出头。

这带来一个重要好处：

**语言任务和动作任务可以共享同一个输出空间、同一个训练目标、同一个生成机制。**

这也是 RT-2 被称为 Vision-Language-Action model 的核心原因。

---

## 13. Output Constraint：为什么推理时要限制 vocabulary？

普通 VLM 可以输出整个词表里的任意 token，例如：

```text
apple, red, person, 128, ...
```

但机器人控制时，输出必须是合法动作 token。如果模型在 action 位置输出了普通语言 token，就无法 detokenize 成机器人动作。

所以论文在 robot-action task 推理时限制输出 vocabulary：只允许采样合法 action tokens。普通 VQA / caption 等任务仍然可以使用完整自然语言词表。`RT2.pdf`

因此推理时可以理解成：

$$
p(t_i \mid x)
\quad \text{only over} \quad
t_i \in V_{\text{action}}
$$

其中：

$$
V_{\text{action}} = \{A_0, A_1, ..., A_{255}\}
$$

如果是包含 plan 的 CoT 版本，则更合理的理解是：

- `Plan:` 部分使用自然语言词表；
- `Action:` 部分限制到 action vocabulary。

论文对 CoT 部分主要是 qualitative 展示，没有把这个实现细节展开。

---

## 14. Co-Fine-Tuning：为什么不能只用机器人数据 fine-tune？

RT-2 的一个关键训练策略是 **co-fine-tuning**。

训练数据混合两类：

1. 原始 VLM web-scale 数据；
2. robot trajectory 数据。

机器人数据来自 RT-1 数据集，包括 13 台机器人、17 个月采集的 office kitchen 环境 demonstrations，每条轨迹带自然语言指令。Appendix B 中还说明了 robot dataset 包括 Pick Object、Move Object Near Object、Place Object Upright、Knock Object Over、Open Drawer、Close Drawer 等技能。`RT2.pdf`

co-fine-tuning 的目的：

> 在让模型学会输出 action token 的同时，避免模型遗忘原本 VLM 预训练中学到的视觉语言概念。

如果只用机器人数据 fine-tune，模型容易过度适应 robot dataset，丢失一部分 web-scale semantic knowledge。

论文 ablation 也支持这个结论：

- 从 scratch 训练大模型效果很差；
- 只用 robot data fine-tuning 有提升；
- co-fine-tuning 效果最好；
- 55B 比 5B 泛化更强。`RT2.pdf`

---

## 15. 模型架构

RT-2 不是单一模型，而是一组基于不同 VLM backbone 的模型。

### 15.1 RT-2-PaLI-X

PaLI-X 的大体结构是：

```text
image
  ↓
ViT image encoder
  ↓
image token projection
  ↓
UL2-like encoder-decoder language backbone
  ↓
autoregressive token generation
```

论文 Appendix D 说明 PaLI-X 使用 ViT-22B 处理图像，然后把 image tokens 经过 projection layer 输入到 encoder-decoder backbone 中，最后自回归生成输出 token。`RT2.pdf`

RT-2 里使用了：

- RT-2-PaLI-X-5B；
- RT-2-PaLI-X-55B。

### 15.2 RT-2-PaLM-E

PaLM-E 是 decoder-only LLM 风格的 embodied multimodal model。它把图像、文本、机器人相关连续变量等投影到语言 token embedding 空间，然后由 LLM 生成文本输出。

论文 Appendix D 说明，PaLM-E-12B 使用 ViT-4B 作为视觉模型，把图像投影到语言 embedding 空间。`RT2.pdf`

RT-2 里使用了：

- RT-2-PaLM-E-12B。

---

## 16. 训练配置

论文 Appendix E 给出训练细节：

| 模型 | 学习率 | Batch size | Steps |
|---|---:|---:|---:|
| RT-2-PaLI-X-55B | 1e-3 | 2048 | 80K |
| RT-2-PaLI-X-5B | 1e-3 | 2048 | 270K |
| RT-2-PaLM-E-12B | 4e-4 | 512 | 1M |
| RT-2-PaLI-3B for Language Table | 1e-3 | 128 | 300K |

训练目标统一是 next-token prediction。机器人任务里的 next-token prediction 就是 action token 的 behavior cloning。`RT2.pdf`

---

## 17. 推理 Pipeline

推理时，RT-2 的流程如下：

```text
current image
    +
language instruction
    ↓
construct VQA prompt
    ↓
VLM generates action tokens autoregressively
    ↓
restrict output to valid action tokens
    ↓
detokenize tokens into action bins
    ↓
map bins back to continuous robot command
    ↓
execute action
    ↓
repeat next control step
```

更具体地说：

1. 输入图像 $I_t$；
2. 输入语言任务 $l$；
3. 构造 prompt：

```text
Q: what action should the robot take to [instruction]? A:
```

4. 模型生成 action token：

```text
A_1 A_140 A_120 A_128 A_132 A_135 A_106 A_127
```

5. 转成 bin：

```text
[1, 140, 120, 128, 132, 135, 106, 127]
```

6. 每个 bin 根据对应动作维度的范围映射回连续控制量；
7. 发送给机器人；
8. 下一帧重新推理。

RT-2 是闭环控制，但不是高频本地模型。论文指出，55B 模型通过 multi-TPU cloud service 推理，大约 1-3 Hz；5B 模型大约 5 Hz。`RT2.pdf`

---

## 18. Chain-of-Thought 版本

RT-2 还探索了一个 CoT 版本，主要基于 PaLM-E。

普通 RT-2 输出：

```text
Action: 1 128 124 136 121 158 111 255
```

CoT 版本输出：

```text
Plan: pick rxbar chocolate.
Action: 1 128 124 136 121 158 111 255
```

也就是先生成一个自然语言 plan，再生成 action token。

论文中的例子包括：

```text
Instruction: I’m hungry.
Plan: pick rxbar chocolate.
Action: ...
```

以及：

```text
I need to hammer a nail, what object might be useful?
Prediction: Rocks.
Action: ...
```

这个部分说明 VLA 有潜力把 high-level reasoning 和 low-level action generation 合并到同一个模型里。但论文主要给的是 qualitative evidence，不是完整的大规模定量证明。`RT2.pdf`

---

## 19. 实验结果

### 19.1 Seen Tasks

RT-2 在 seen tasks 上和 RT-1 差不多。

Table 4 中：

| Model | Seen Tasks |
|---|---:|
| RT-1 | 92 |
| RT-2-PaLI-X-55B | 91 |
| RT-2-PaLM-E-12B | 93 |

这说明 RT-2 没有明显牺牲原本机器人数据分布内的能力。`RT2.pdf`

---

### 19.2 Unseen Generalization

RT-2 在 unseen objects、unseen backgrounds、unseen environments 上明显强于 baseline。

Table 4 中 unseen average：

| Model | Unseen Average |
|---|---:|
| R3M | 12 |
| VC-1 | 10 |
| RT-1 | 32 |
| MOO | 35 |
| RT-2-PaLI-X-55B | 62 |
| RT-2-PaLM-E-12B | 62 |

这说明 RT-2 的核心收益来自 VLM 的语义和视觉泛化能力。`RT2.pdf`

---

### 19.3 Emergent Capabilities

论文把 emergent capabilities 分为三类：

1. **Symbol Understanding**  
   例如 `move coke can near X`、`move apple to tree`。

2. **Reasoning**  
   例如 `move banana near the sum of two plus one`、`pick a healthy drink`、多语言颜色指令。

3. **Human Recognition**  
   例如 `move coke can to person with glasses`。

Table 5 中平均结果：

| Model | Emergent Average |
|---|---:|
| VC-1 | 11 |
| RT-1 | 17 |
| RT-2-PaLI-X-55B | 60 |
| RT-2-PaLM-E-12B | 40 |

其中 RT-2-PaLI-X-55B 的 symbol understanding 平均达到 82，而 RT-1 只有 16。说明 VLM 预训练知识确实迁移到了机器人控制任务中。`RT2.pdf`

---

### 19.4 Ablation

论文比较了：

- 从 scratch 训练；
- 只用 robot data fine-tuning；
- co-fine-tuning；
- 5B 和 55B 模型大小。

Table 6 显示：

| Model | Size | Training | Average |
|---|---:|---|---:|
| RT-2-PaLI-X | 5B | from scratch | 9 |
| RT-2-PaLI-X | 5B | fine-tuning | 42 |
| RT-2-PaLI-X | 5B | co-fine-tuning | 44 |
| RT-2-PaLI-X | 55B | fine-tuning | 52 |
| RT-2-PaLI-X | 55B | co-fine-tuning | 63 |

结论：

1. 大模型从 scratch 训练机器人策略效果很差；
2. 预训练非常关键；
3. co-fine-tuning 比单纯 robot fine-tuning 更好；
4. 模型越大，泛化性能越强。`RT2.pdf`

---

## 20. RT-2 的核心贡献

RT-2 的贡献可以总结为三点。

### 20.1 提出清晰的 VLA recipe

RT-2 给出一个非常直接的路线：

```text
pretrained VLM + action tokenization + robot data co-fine-tuning = VLA
```

它没有为机器人动作额外设计复杂结构，而是让动作成为 VLM 输出 token 的一种。

---

### 20.2 证明 web-scale VLM knowledge 可以迁移到 low-level control

RT-2 不是只让 VLM 做 high-level planner，而是让 VLM 直接参与低层闭环控制。

它证明了：

- 符号理解；
- 物体语义；
- 视觉关系；
- 简单常识；
- 多语言理解；

这些能力可以通过 action token prediction 迁移到机器人控制中。

---

### 20.3 明确指出迁移的是语义能力，不是新运动技能

RT-2 并没有因为看过 web 图文数据就学会擦桌子、折毛巾、复杂工具使用。

它只是能把已有的机器人技能用在新的语义目标上。

这点非常重要，因为它决定了 RT-2 的上限：

```text
VLM pretraining improves "what to act on"
but not necessarily "how to physically act"
```

---

## 21. 局限性

### 21.1 不能学习新的物理技能

论文明确指出，RT-2 的 physical skills 仍然受限于 robot data 的技能分布。web-scale VLM 数据不能直接让模型掌握新的运动模式。`RT2.pdf`

失败例子包括：

- 擦拭；
- 使用工具；
- 折毛巾；
- 抓取物体特定部位；
- 复杂 dexterous manipulation。

---

### 21.2 动力学泛化有限

Language Table 失败例子显示，模型能理解该操作哪个物体，但如果物体动力学和训练集差异大，例如笔会滚、香蕉质心偏移，模型仍然失败。`RT2.pdf`

这说明 RT-2 学到的主要是视觉语义泛化，而不是可靠的物理动力学预测。

---

### 21.3 推理成本高

55B 模型需要 cloud TPU service，只有 1-3 Hz；5B 也只有约 5 Hz。对于高频控制任务，这是明显瓶颈。`RT2.pdf`

---

### 21.4 复杂长程推理仍然弱

论文提到，extended reasoning requiring multiple layers of indirection 仍然是失败点。RT-2 的 CoT 能力更多是初步展示，不是完整解决长时序任务规划。

---

## 22. 和 RT-1 的关系

RT-1 和 RT-2 都使用离散动作 token 和 behavior cloning，但核心区别是：

| 维度 | RT-1 | RT-2 |
|---|---|---|
| Backbone | 机器人策略 Transformer | 预训练 VLM |
| 语言知识来源 | robot instruction | web-scale vision-language pretraining |
| 输出 | action tokens | action tokens as language tokens |
| 泛化提升来源 | 机器人数据规模 | VLM 语义知识迁移 |
| 是否具备 emergent reasoning | 弱 | 明显更强 |
| 是否学会新物理技能 | 否 | 否 |

可以把 RT-2 理解成：

```text
RT-1 action discretization
+
PaLI-X / PaLM-E VLM pretraining
+
co-fine-tuning
=
RT-2
```

---

## 23. 和后续 VLA / AWM 的关系

RT-2 是 VLA 发展中的关键节点，但从今天视角看，它还有明显早期特征：

- 动作是离散 token；
- 一步一步闭环输出；
- 没有 action chunk；
- 没有 diffusion / flow matching；
- 没有显式 world model；
- 没有 memory；
- 没有 object-centric state tracking；
- 没有真正长程任务执行机制。

后续模型，例如 OpenVLA、π₀、π₀.₅、MolmoAct、world-action model 方向，会进一步关注：

- 连续动作生成；
- action chunk；
- flow / diffusion policy；
- multi-step planning；
- object binding；
- memory；
- world model；
- 更强 real-world deployment。

因此 RT-2 的历史价值主要是：

> 它证明了 VLM 可以被直接改造成 VLA，并把 web-scale semantic knowledge 注入机器人低层控制。

---

## 24. 最重要的技术理解

RT-2 最容易误解的地方就是 action tokenization。准确理解如下：

### 24.1 对 PaLI-X

因为数字 `0` 到 `1000` 是原子 token，所以可以使用：

```text
"140 120 128"
```

并期望 tokenizer 得到：

```text
[token("140"), token("120"), token("128")]
```

### 24.2 对 PaLM-E

不能依赖普通数字字符串。应该使用 reserved token id：

```python
bins = [140, 120, 128]
target_ids = [rare_token_ids[b] for b in bins]
```

### 24.3 “single string” 的真正含义

不是说一定要把动作变成普通文本再让 tokenizer 随便分词。

而是说：

```text
action vector
→ discrete bins
→ action token sequence
→ VLM next-token prediction target
```

也就是说，动作被包装成 VLM 输出序列的一部分。

---

## 25. 伪代码理解

### 25.1 构造训练目标

```python
def build_action_target(action_continuous, model_type):
    # 1. continuous action -> discrete bins
    bins = discretize_to_256_bins(action_continuous)
    # e.g. [1, 140, 120, 128, 132, 135, 106, 127]

    if model_type == "PaLI-X":
        # integers are guaranteed to be single tokens
        target_text = " ".join(str(b) for b in bins)
        target_ids = tokenizer(target_text)

    elif model_type == "PaLM-E":
        # do not rely on numeric string tokenization
        target_ids = [rare_token_ids[b] for b in bins]

    return target_ids
```

---

### 25.2 训练

```python
image_tokens = image_encoder(image)

prompt_ids = tokenizer(
    "Q: what action should the robot take to pick coke can? A:"
)

action_target_ids = build_action_target(action, model_type)

input_ids = concat(prompt_ids, action_target_ids[:-1])
labels = action_target_ids

logits = vlm(image_tokens, input_ids)

loss = cross_entropy(
    logits_on_action_positions,
    labels
)
```

实际实现会更复杂，因为 PaLI-X 是 encoder-decoder，PaLM-E 是 decoder-only，但训练本质都是 next-token prediction。

---

### 25.3 推理

```python
image_tokens = image_encoder(current_image)

prompt_ids = tokenizer(
    "Q: what action should the robot take to pick coke can? A:"
)

generated_token_ids = autoregressive_decode(
    image_tokens=image_tokens,
    prompt_ids=prompt_ids,
    allowed_vocab=valid_action_token_ids
)

bins = action_tokens_to_bins(generated_token_ids)

robot_action = detokenize_bins_to_continuous_action(bins)

robot.execute(robot_action)
```

---

## 26. 最终总结

RT-2 的核心不是提出一个更复杂的机器人控制器，而是提出一种非常简单但影响很大的统一接口：

```text
Language token generation
=
Action token generation
```

通过把机器人动作离散化成 token，并把这些 token 接入 VLM 的输出词表，RT-2 让机器人控制可以直接复用 web-scale VLM 的视觉语言知识。

它带来的提升主要体现在：

- 新物体泛化；
- 新背景泛化；
- 新环境泛化；
- 符号理解；
- 简单推理；
- 多语言指令；
- 基础 commonsense grounding。

但它仍然不能解决：

- 新运动技能学习；
- 复杂动力学；
- 高频控制；
- 长程规划；
- 精细操作；
- 真正 world model 式预测。

因此，RT-2 最准确的定位是：

> **VLA 路线的关键开端：它证明了 pretrained VLM 可以通过 action-as-token 的方式直接变成机器人闭环控制策略，但它还不是完整的通用机器人基础模型。**



---

## 相关笔记
- [[RT-1 论文综述|RT-1 论文综述]]
- [[FAST_知识总结|FAST 知识总结]]
- [[Pi_0机器人文章分析|pi0 机器人文章分析]]
- [[Pi0_7_technical_report|π0.7 技术报告]]
- [[RDT-1B|RDT-1B]]
- [[GR00T N1 综述|GR00T N1 综述]]
- [[Gemini Robotics 1.5 综述|Gemini Robotics 1.5 综述]]
- [[MolmoAct2论文框架分析|MolmoAct2 论文框架分析]]
- [[WorldVLA 论文综述(不建议读)|WorldVLA]]：统一自回归 action/image token 建模的后续对照。
