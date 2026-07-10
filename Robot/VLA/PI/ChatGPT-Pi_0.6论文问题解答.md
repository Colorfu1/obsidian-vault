---
title: pi0.6 论文问题解答
type: paper_note
topic: robotics_foundation_model
status: mature
importance: high
updated: 2026-06-10
tags:
  - pi0.6
  - vla
  - fast
  - flow-matching
  - knowledge-insulation
  - robotics
---

# Pi_!0.6论文问题解答

**User:** Anonymous  
**Created:** 5/22/2026 18:18:28  
**Updated:** 6/3/2026 18:17:21  
**Exported:** 6/3/2026 18:18:09  
**Link:** [https://chatgpt.com/c/6a102d5a-553c-83ec-8e76-7937eb760753](https://chatgpt.com/c/6a102d5a-553c-83ec-8e76-7937eb760753)  

## Response:
6/3/2026, 6:16:32 PM

下面只总结论文 **Section V-A: The π0.6 model** 里对 **π0.6** 的描述，并把关键公式保留下来。这里先不混入 π\*0.6 / RECAP 的 advantage conditioning，重点是 π0.6 本身的输入、输出、训练形式和 action expert 结构。

---

# π0.6 模型总结：Section V-A

π0.6 是 π0.5 的后续版本，是一个可以同时处理视觉、语言和机器人动作的 **VLA model**。论文说 π0.6 继承了 π0.5 的几个核心设计：它可以通过 **flow matching** 表达 chunked continuous action distribution，也可以输出用于高层决策的 intermediate text。训练上使用 **Knowledge Insulation, KI** recipe：整个模型端到端训练 continuous actions 和 discretized tokens，但对 flow-matching action expert 使用 stop gradient，避免 action expert 反向影响模型其余部分。预训练数据同时包括 robot data 和来自 web 的 vision-language co-training data。`pi_*0.6.pdf`

---

## 1. π0.6 相比 π0.5 的改进

论文列了三点：

1. **预训练数据更大更多样**：加入了来自多个机器人平台的额外数据；
2. **基础 VLM 更强**：base VLM 使用 **Gemma 3 4B**；
3. **action expert 更大**：action expert 扩大到 **860M parameters**。`pi_*0.6.pdf`

所以 π0.6 可以理解成：

> π0.5 的升级版：更大的 VLM backbone、更大的 action expert、更多机器人平台数据、更丰富的 conditioning。

---

## 2. π0.6 的输入定义

论文把模型写成：

$$
\pi_\theta(a_{t:t+H},\hat{\ell}\mid o_t,\ell)
$$

其中 observation 是：

$$
o_t=[X_t^1,\ldots,X_t^n,q_t]
$$

这里：

- $X_t^1,\ldots,X_t^n$：多路 camera images；
- $q_t$：机器人 configuration / proprioceptive state，比如关节、夹爪状态。

语言输入是：

$$
\ell=\ell_t+s
$$

其中：

- $\ell_t$：overall task prompt，比如论文举的例子是 “make me an espresso”；
- $s$：additional language inputs，提供 metadata，用来进一步调制任务怎么执行。`pi_*0.6.pdf`

所以 π0.6 的输入可以写成：

```text
multi-camera images + robot state + task prompt + metadata
```

---

## 3. π0.6 的连续动作输出：action chunk

π0.6 输出的是 action chunk：

$$
a_{t:t+H}
$$

它不是单步 action，而是一段未来动作。论文说这个 action chunk 包含：

- joint angles；
- gripper commands；
- 控制频率是 **50 Hz**。

这部分由一个单独的 **action expert** 生成。action expert 是一组专门用于 action generation 的参数，大小是 **860M**，用 **flow matching** 训练。它可以 attend 到模型其余部分的 activations，但它本身是专门负责连续动作生成的模块。`pi_*0.6.pdf`

直观理解：

```text
VLM backbone 处理图像、语言、状态
        ↓
action expert attend 到这些 hidden activations
        ↓
生成连续 action chunk a_{t:t+H}
```

---

## 4. π0.6 的离散文本输出：$\hat{\ell}$

除了连续动作，π0.6 还会输出 tokenized discrete outputs：

$$
\hat{\ell}
$$

论文说 $\hat{\ell}$ 包括下一个 predicted sub-task 的文本表示，比如：

```text
pick up the coffee cup
```

这个 $\hat{\ell}$ 用于高层决策。

关键点是：论文说 **actions are generated after $\hat{\ell}$**，所以 action generation 实际上 conditioned on predicted sub-task。

也就是说，π0.6 的行为链路更像：

$$
o_t,\ell
\rightarrow
\hat{\ell}
\rightarrow
a_{t:t+H}
$$

先预测当前应该做什么高层子任务，再基于这个子任务生成低层连续动作。论文还说，在 inference time，sub-task prediction 的运行频率比 action generation 更低。`pi_*0.6.pdf`

直观例子：

```text
task prompt: make me an espresso
current image: portafilter on table

π0.6 predicts subtask:
  "pick up the portafilter"

then action expert generates:
  next action chunk for moving gripper to portafilter and grasping it
```

---

## 5. π0.6 训练时还预测 FAST 离散动作 token

训练时，π0.6 除了预测真实连续 action chunk，还会预测 action chunk 的 tokenized representation。

论文说：训练时模型也预测 action chunk $a_{t:t+H}$ 的 tokenized representation，使用 **FAST tokenizer**，这是 KI recipe 的一部分。论文记这些离散化动作 token 为：

$$
a^\ell_{t:t+H}
$$

注意这里的 $a^\ell_{t:t+H}$ 是离散动作 token，不是真实连续动作。

所以训练时有两种 action target：

1. 连续动作：

$$
a_{t:t+H}
$$

2. FAST 离散动作 token：

$$
a^\ell_{t:t+H}
$$

论文特别说明：**action expert does not receive these as input**。也就是说，FAST 离散动作 token 不会作为输入喂给 continuous action expert。因此 discrete action prediction 和 continuous action prediction 是 **independently predicted** 的。`pi_*0.6.pdf`

这点很重要：

> FAST action token 是一个训练辅助/离散动作建模分支；continuous action expert 不是靠这些 token 再解码出连续动作，而是独立地用 flow matching 生成连续 action chunk。

---

## 6. π0.6 的联合 log-likelihood 公式

因为 π0.6 训练时同时预测：

- $\hat{\ell}$：高层 subtask text；
- $a^\ell_{t:t+H}$：FAST 离散动作 token；
- $a_{t:t+H}$：连续 action chunk；

所以最终训练 log-likelihood 写成：

$$
\log \pi_\theta(a_{t:t+H}, a^\ell_{t:t+H}, \hat{\ell}\mid o_t,\ell)
$$

由于论文说先预测 $\hat{\ell}$，再预测动作，所以它可以按 chain rule 分解为：

$$
\log \pi_\theta(a_{t:t+H}, a^\ell_{t:t+H}, \hat{\ell}\mid o_t,\ell)
=
\log \pi_\theta(\hat{\ell}\mid o_t,\ell)
+
\log \pi_\theta(a^\ell_{t:t+H}\mid o_t,\ell,\hat{\ell})
+
\log \pi_\theta(a_{t:t+H}\mid o_t,\ell,\hat{\ell})
$$

这就是 V-A 里最关键的公式。`pi_*0.6.pdf`

---

## 7. 这个公式每一项是什么意思？

### 第一项：subtask text likelihood

$$
\log \pi_\theta(\hat{\ell}\mid o_t,\ell)
$$

表示给定当前 observation 和任务语言，预测高层 subtask 文本。

例如：

```text
input: images + robot state + "make me an espresso"
target: "pick up the coffee cup"
```

这部分通常是 autoregressive token likelihood / cross entropy。

---

### 第二项：FAST 离散动作 token likelihood

$$
\log \pi_\theta(a^\ell_{t:t+H}\mid o_t,\ell,\hat{\ell})
$$

表示给定当前 observation、任务语言和 predicted subtask，预测离散化后的 action token sequence。

例如：

```text
target: <ACT_12> <ACT_318> <ACT_45> ...
```

这部分也是离散 token prediction，可以用 cross entropy。

---

### 第三项：连续 action chunk likelihood

$$
\log \pi_\theta(a_{t:t+H}\mid o_t,\ell,\hat{\ell})
$$

表示给定 observation、task prompt 和 subtask，生成真实连续 action chunk。

例如：

```text
target: future H-step joint angles + gripper commands at 50 Hz
```

这部分由 flow matching action expert 训练。严格来说，continuous action 的 likelihood 不像离散 token 那样容易精确计算，所以后面 π\*0.6 的 B 段会进一步说明：连续部分实际用 flow matching objective。`pi_*0.6.pdf`

---

## 8. π0.6 的训练逻辑：三个任务同时训练

可以把 π0.6 的训练理解成三个 supervised / generative learning 任务同时进行：

```text
输入:
  images + robot state + task prompt + metadata

任务 1:
  predict subtask text \hat{\ell}

任务 2:
  predict FAST discrete action tokens a^\ell_{t:t+H}

任务 3:
  predict continuous action chunk a_{t:t+H} with flow matching
```

对应 loss 大致是：

$$
\mathcal{L}_{\pi0.6}
=
\mathcal{L}_{\text{subtask CE}}
+
\mathcal{L}_{\text{FAST action token CE}}
+
\mathcal{L}_{\text{continuous flow matching}}
$$

论文在 V-A 里用 log-likelihood 表达这件事；连续 action 的具体 flow matching loss 在 V-B / Appendix C 里会展开。`pi_*0.6.pdf`

---

# 9. π0.6 的核心机制总结

π0.6 可以总结成：

> π0.6 是一个多模态 VLA，它输入多路相机图像、机器人状态、任务语言和 metadata；先预测高层 subtask 文本 $\hat{\ell}$，再基于这个 subtask 生成动作。训练时，它同时预测 FAST 离散动作 token $a^\ell_{t:t+H}$ 和连续 action chunk $a_{t:t+H}$。连续动作由 860M 参数的 flow-matching action expert 生成，action expert 可以 attend 到主模型 activations，但不接收 FAST action tokens 作为输入，因此离散动作和连续动作是独立预测的。整体联合 log-likelihood 可以按 $\hat{\ell}$、离散动作 token、连续动作三部分分解。

更短的结构图是：

```text
Input:
  o_t = [X_t^1, ..., X_t^n, q_t]
  ℓ = ℓ_t + s

Backbone:
  Gemma 3 4B VLM + multimodal inputs

Discrete output:
  \hat{ℓ}: next subtask text

Discrete action auxiliary output:
  a^ℓ_{t:t+H}: FAST-tokenized action chunk

Continuous action output:
  a_{t:t+H}: joint angles + gripper commands at 50Hz
  generated by 860M flow-matching action expert

Factorization:
  log πθ(a, a^ℓ, \hat{ℓ}|o,ℓ)
  =
  log πθ(\hat{ℓ}|o,ℓ)
  +
  log πθ(a^ℓ|o,ℓ,\hat{ℓ})
  +
  log πθ(a|o,ℓ,\hat{ℓ})
```

---

# 10. 和 π\*0.6 的关系

最后补一句关系，方便和前面 RECAP 总结对齐。

π\*0.6 没有推翻 π0.6 的结构。它基本沿用上面的 π0.6 架构，只是在 $\hat{\ell}$ 和 action 之间加入：

```text
Advantage: positive / negative
```

所以 π\*0.6 的 action 部分从：

$$
\log \pi_\theta(a^\ell_{t:t+H}\mid o_t,\ell,\hat{\ell})
+
\log \pi_\theta(a_{t:t+H}\mid o_t,\ell,\hat{\ell})
$$

变成受 $I_t$ 条件控制的形式：

$$
\log \pi_\theta(a^\ell_{t:t+H}\mid I_t,o_t,\ell,\hat{\ell})
+
\log \pi_\theta(a_{t:t+H}\mid I_t,o_t,\ell,\hat{\ell})
$$

也就是说，π\*0.6 是在 π0.6 的 **subtask-conditioned action generation** 结构上，再加了一层 **advantage-conditioned action generation**。

注意在π0.6 之后, VLM不再接受flow matching部分回传的梯度.

## 相关笔记

- [[ChatGPT-Pi_0.5综述|pi0.5]]：pi0.6 继承的 high-level language/action decomposition。
- [[ChatGPT-Pi_star0.6论文问题解答|pi*0.6 / RECAP]]：在 pi0.6 上加入 advantage-conditioned policy improvement。
- [[FAST_知识总结|FAST]]：pi0.6 中 discrete action token 路线的关键基础。
- [[ChatGPT-MEM 文章分析|MEM]]：pi0.6-MEM 长程记忆扩展。
- [[Pi0_7_technical_report|pi0.7]]：后续 steerable generalist VLA。
- [[ChatGPT-RDT-1B|RDT-1B]]：continuous action chunk 的 diffusion 路线对照。

---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
