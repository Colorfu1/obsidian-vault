---
title: MEM 文章分析
type: paper_note
topic: robot_memory_model
status: mature
importance: high
updated: 2026-06-10
tags:
  - mem
  - vla
  - robot-memory
  - long-horizon
  - pi0.6
  - robotics
---


# MEM 文章分析

**User:** Anonymous  
**Created:** 6/5/2026 18:38:47  
**Updated:** 6/7/2026 19:08:48  
**Exported:** 6/7/2026 19:24:47  
**Link:** [https://chatgpt.com/c/6a22a710-6ab0-83ec-bc38-0e98fffad063](https://chatgpt.com/c/6a22a710-6ab0-83ec-bc38-0e98fffad063)  

## Response:
07/06/2026, 19:07:18

> **从文件中引用进行精确引证**
>
>
> **引用文献并评估任务适应中的MEM策略**
>
> 
>
> 思考了 19s

下面给你一份比较完整的 **MEM: Multi-Scale Embodied Memory for Vision Language Action Models** 总结。我会把论文动机、模型结构、训练/推理流程、GT 生成方式、video encoder 细节、实验结论和我认为的关键理解点都串起来。

---

# 1. 这篇文章想解决什么问题？

这篇文章的核心问题是：**机器人 VLA 模型如何在长任务中拥有有效记忆？**

普通 VLA 往往只看当前观测：

$$
o_t
$$

然后直接输出动作。但真实机器人任务经常需要记住过去发生过什么。例如：

- 做菜时要记住哪些食材已经拿过；
- 清理厨房时要记住哪些台面已经擦过；
- 洗盘子时要记住是否已经打过肥皂、正反面是否洗过；
- 抓取失败后要记住刚才怎么失败的，从而调整下一次抓取策略；
- 物体被机械臂遮挡后，要记住它刚才在哪里。

论文认为，机器人需要的 memory 不是单一形式的。**短期记忆需要保留视觉细节和动态信息，长期记忆只需要保留压缩后的语义状态。**所以 MEM 的核心设计就是：

$$
\text{short-term memory}
\rightarrow
\text{video-based dense memory}
$$

$$
\text{long-term memory}
\rightarrow
\text{language-based semantic memory}
$$

论文开头就强调，MEM 有两个关键组件：一个用于短期图像记忆的高效 video encoder，另一个用于长期语义记忆的 language memory mechanism。训练之后，MEM VLA 可以解决最长约 15 分钟记忆跨度的任务、处理 partial observability，并做 in-context manipulation adaptation。`MEM.pdf`

---

# 2. 为什么不能直接把所有历史帧都塞给 VLA？

最直接的想法是：把从任务开始到现在的所有观测都输入模型：

$$
o_{t-T:t}
$$

然后让模型预测动作：

$$
\pi(a_{t:t+H} \mid o_{t-T:t}, g)
$$

但这在机器人里很快不可行。因为观测是 dense observation，包含多相机图像、机器人 proprioceptive state 等。如果任务持续十几分钟，输入 token 数量和推理延迟会爆炸。论文明确说，直接让 policy condition on dense sequence of all previous observations 对几十分钟任务很快 infeasible。`MEM.pdf`

更重要的是，**长短期记忆需要的信息粒度不同**。短期记忆可能需要像素级信息，比如刚才物体被遮挡前的位置；长期记忆通常只需要几个语义 bit，比如 “milk 已经加过了”。所以把所有历史都用图像形式保存是浪费的。

---

# 3. MEM 的整体分解：high-level 语言记忆 + low-level 视频记忆

MEM 把原始长历史动作预测拆成两个部分：

$$
\pi(a_{t:t+H}, l_{t+1}, m_{t+1}\mid o_{t-T:t}, m_t, g)
$$

近似分解为：

$$
\pi_{\mathrm{LL}}(a_{t:t+H}\mid o_{t-K:t}, l_{t+1}, g)
\;
\pi_{\mathrm{HL}}(l_{t+1}, m_{t+1}\mid o_t, m_t, g)
$$

这里：

$$
K \ll T
$$

含义是：不要让 low-level action policy 处理完整长历史，只让它处理最近一小段 dense observation。长历史则由语言记忆 $m_t$ 负责压缩保存。

## high-level policy

high-level policy 是：

$$
\pi_{\mathrm{HL}}(l_{t+1}, m_{t+1}\mid o_t, m_t, g)
$$

它输入：

$$
o_t
$$

当前观测；

$$
m_t
$$

当前语言记忆，也就是过去语义事件的 summary；

$$
g
$$

任务目标。

它输出：

$$
l_{t+1}
$$

下一步 subtask instruction；

$$
m_{t+1}
$$

更新后的语言记忆。

也就是说 high-level policy 不直接输出连续机器人动作，而是做两件事：

$$
\text{decide next subtask}
$$

$$
\text{update long-term language memory}
$$

论文明确说，之前也有 high-level / low-level split，用 subtask instruction 作为接口；但 MEM 的关键新意是 high-level policy 还会基于自己之前预测的 $m_t$，继续预测更新后的 $m_{t+1}$。`MEM.pdf`

## low-level policy

low-level policy 是：

$$
\pi_{\mathrm{LL}}(a_{t:t+H}\mid o_{t-K:t}, l_{t+1}, g)
$$

它输入：

$$
o_{t-K:t}
$$

最近短窗口 observation；

$$
l_{t+1}
$$

high-level 给出的当前子任务；

$$
g
$$

总任务目标。

它输出：

$$
a_{t:t+H}
$$

一段连续 action chunk。

所以 MEM 的基本思想可以概括为：

$$
\text{长期任务进度}
\rightarrow
m_t
\rightarrow
\text{language memory}
$$

$$
\text{短期视觉动态}
\rightarrow
o_{t-K:t}
\rightarrow
\text{video memory}
$$

$$
(m_t,o_t,g)
\rightarrow
(l_{t+1},m_{t+1})
$$

$$
(o_{t-K:t},l_{t+1},g)
\rightarrow
a_{t:t+H}
$$

---

# 4. Language memory：长期语义记忆如何训练？

Language memory $m_t$ 是对过去语义事件的自然语言总结。它不是完整历史日志，而是压缩后的 task state。例如：

$$
m_t =
\text{“I placed a plate in the cabinet and moved to the counter.”}
$$

如果机器人刚刚成功拿起 bowl，更新后可能是：

$$
m_{t+1} =
\text{“I placed a plate in the cabinet, moved to the counter, and picked up a bowl.”}
$$

论文说，训练 high-level policy 的难点是：我们需要知道每一步正确的 memory update 是什么。这个东西不是传感器天然给出的，所以作者设计了一个离线 GT 生成 pipeline。`MEM.pdf`

## 4.1 生成 memory GT 的 teacher LLM

给定一条 robot episode，里面有 subtask language annotations：

$$
l_{0:T}
$$

同时每个 subtask 有一个 success/failure indicator。作者把这些 subtask instructions 和 indicator 一起输入给一个 **off-the-shelf pre-trained LLM**，让它总结过去 subtasks 中仍然对未来执行有用的信息。然后收集这个 LLM 的输出，作为训练 high-level policy 的 sequence label。`MEM.pdf`

注意这里有一个重要区分：

$$
\text{external off-the-shelf LLM}
\neq
\text{MEM high-level policy}
$$

外部 LLM 只是离线 teacher / label generator。它不部署在机器人推理时。真正被训练和部署的是：

$$
\pi_{\mathrm{HL}}
$$

所以流程是：

$$
(l_{0:T},\text{success/failure indicators})
\rightarrow
\text{external LLM}
\rightarrow
(m_0,m_1,\ldots,m_T)
$$

然后训练：

$$
(o_t,m_t,g)
\rightarrow
(l_{t+1},m_{t+1})
$$

## 4.2 一个具体例子

假设总任务是：

$$
g = \text{“Clean up the kitchen.”}
$$

episode 里有这些 subtask：

| step | subtask | indicator |
|---|---|---|
| 0 | Move to the counter | success |
| 1 | Pick up the plate | success |
| 2 | Place the plate in the cabinet | success |
| 3 | Pick up the bowl | failure |
| 4 | Pick up the bowl | success |
| 5 | Place the bowl in the cabinet | success |
| 6 | Wipe the counter with the sponge | success |

外部 LLM 不应该简单输出完整日志：

$$
\text{“Moved to counter, picked up plate, placed plate, failed bowl, picked bowl...”}
$$

而应该生成对未来有用的压缩 memory。

step 2 后可能是：

$$
m_3 =
\text{“I moved to the counter and placed the plate in the cabinet.”}
$$

step 3 是失败的 “pick up bowl”，LLM 很可能不更新成 “picked up bowl”，而是保持：

$$
m_4 =
\text{“I moved to the counter and placed the plate in the cabinet.”}
$$

step 4 成功拿起 bowl 后，才更新为：

$$
m_5 =
\text{“I moved to the counter, placed the plate in the cabinet, and picked up the bowl.”}
$$

step 5 放好 bowl 后，LLM 还可能进一步压缩：

$$
m_6 =
\text{“I placed the plate and bowl in the cabinet.”}
$$

这就是 language memory 的关键：**它记录任务状态，而不是记录完整动作流水账。**

## 4.3 success/failure indicator 的作用

这个 indicator 的作用主要是帮助外部 LLM 生成干净的 memory label。比如一个失败的 subtask 不应该被写成已经完成：

$$
\text{Pick up bowl: failure}
\not\Rightarrow
\text{“I picked up the bowl.”}
$$

推理时没有这个 indicator。推理时 high-level policy 只能根据：

$$
o_t,\quad m_t,\quad g
$$

自己判断是否应该更新 memory。

所以存在一个 teacher-student 式的关系：

训练 GT 生成阶段有 privileged information：

$$
\text{success/failure indicator}
$$

但推理阶段没有。模型通过监督学习学会从观测中隐式判断“任务是否完成、memory 是否该更新”。

## 4.4 为什么不直接拼接所有 subtask？

论文专门做了 “naive text + video memory” 的 ablation。naive text memory 就是把之前的 subtask instructions 直接拼接到 high-level policy 输入里，而不是让模型预测压缩 summary。结果这种方式明显更差。原因是 train-inference distribution shift。训练 demo 往往接近最优，某个 subtask 通常只出现一次：

$$
\text{pick up bowl}
\rightarrow
\text{place bowl in cabinet}
$$

但推理时模型可能连续失败，于是 high-level 会多次输出同一个 subtask：

$$
\text{pick up bowl}
\rightarrow
\text{pick up bowl}
\rightarrow
\text{pick up bowl}
\rightarrow
\text{place bowl in cabinet}
$$

如果直接拼接历史，推理时上下文会变得和训练分布很不一样。MEM 的 language memory 会在 bowl 真正成功拿起之前不更新 memory，从而丢弃失败尝试，减少分布偏移。`MEM.pdf`

---

# 5. Video encoder：短期视觉记忆怎么做？

Language memory 能记录长期语义状态，但它无法保留精确视觉细节。例如：

- 物体被手臂遮挡前的位置；
- 刚才抓取失败时的高度；
- 门刚才从哪个方向打不开；
- 盘子洗了多久；
- 三明治煎了多久。

这些需要短期 dense observation memory：

$$
o_{t-K:t}
$$

直接把 $K$ 帧图像全部分别编码，再把所有 patch tokens 传给 VLA backbone，会导致推理很慢。论文 Figure 3 显示，naive 地把多帧 observation 输入 VLA backbone，latency 随帧数快速上升；他们的 video encoder 则能在使用多帧时保持低于实时阈值。`MEM.pdf`

## 5.1 基本做法：把 ViT 改造成 video encoder

MEM 的 video encoder 基于标准 ViT。每个图像先 patchify，得到 patch tokens。普通 ViT 只在空间 patch 间做 attention。MEM 则在每 4 层加入 temporal attention，让 token 可以沿时间维度看历史帧中相同 patch 的 representation。论文说他们 interleave bidirectional spatial attention 和 causal-temporal attention。`MEM.pdf`

普通空间 attention 可以理解成：

$$
z_{p,t}
\leftarrow
\mathrm{Attn}
\left(
\{z_{p',t}\}_{p'=1}^{N}
\right)
$$

时间 attention 可以理解成：

$$
z_{p,t}
\leftarrow
z_{p,t}
+
\mathrm{TemporalAttn}
\left(
\{z_{p,t'}\}_{t'\le t}
\right)
$$

其中 $p$ 是 patch index，$t$ 是时间 index。

## 5.2 为什么不用 full space-time attention？

如果直接对所有时间、所有空间 patch 做全 attention，复杂度类似：

$$
O(n^2K^2)
$$

其中 $n$ 是空间 patch 数，$K$ 是时间帧数。

MEM 使用 space-time separable attention，把它拆成空间 attention 和时间 attention，复杂度降低为：

$$
O(Kn^2+nK^2)
$$

论文明确写了这个复杂度下降。`MEM.pdf`

## 5.3 只保留当前 timestep tokens

video encoder 的另一个关键点是：经过若干层时空 attention 后，它会丢掉过去 timestep 的 observation tokens，只把当前 timestep 的 representation 传给后面的 VLA backbone。这样传给 VLA backbone 的 token 数量接近单帧 VLA，但当前帧 token 已经通过 temporal attention 融合了历史信息。`MEM.pdf`

可以写成：

$$
\{z_{p,t-K},\ldots,z_{p,t}\}_{p=1}^{N}
\rightarrow
\text{video encoder}
\rightarrow
\{\tilde z_{p,t}\}_{p=1}^{N}
$$

然后只传：

$$
\{\tilde z_{p,t}\}_{p=1}^{N}
$$

给 VLA backbone。

## 5.4 “不引入新 learnable parameters” 到底是什么意思？

论文说 video encoder 相比标准单图 ViT 不引入新的 learnable parameters。具体是：

- 不新增 temporal-specific Q/K/V；
- 不新增新的可学习 temporal module；
- 只是改变 attention pattern；
- 加一个 fixed sinusoidal temporal position encoding。

所以它可以从原来的预训练 ViT 权重初始化。为了保证单帧输入时初始化和原 VLM 一致，他们设置 temporal position embedding 在 $t=0$ 时为 0。`MEM.pdf`

关键理解是：

$$
\text{no new learnable parameters}
\neq
\text{parameters are frozen}
$$

论文描述的是不新增参数矩阵，而不是说 video encoder 完全 freeze。模型仍然需要继续训练，让已有 ViT 参数学会利用新的 temporal attention pattern。

初始化时，单帧输入 $K=1$ 可以和原始 VLM 一致：

$$
e(0)=0
$$

$$
\hat z_{p,0}=z_{p,0}+e(0)=z_{p,0}
$$

但训练之后参数可能变化，因此不能严格保证单帧输出逐 bit 不变。更准确地说，它保证的是 architecture / initialization 层面的兼容，而不是训练后函数值完全不变。

---

# 6. π0.6-MEM 的具体实现

论文把 MEM 集成进 π0.6 VLA。π0.6-MEM 初始化自 Gemma3-4B VLM，并且训练时使用两类 action 目标：

- discrete FAST action token prediction；
- flow-matching action expert，action expert 规模是 860M。

论文还说，action expert 的梯度不回传到 VLM backbone。输入分辨率是每个 camera stream $448\times448$，最多四路相机，取决于机器人 embodiment。`MEM.pdf`

Figure 2 里画得很清楚：左边是 high-level VLM，输入 task prompt 和 language memory，输出 updated language memory 和 subtask；右边是 low-level VLM，输入 language subtask 和 video memory encoder 的输出，再接 action expert 输出 continuous actions。`MEM.pdf`

---

# 7. proprioceptive state 怎么处理？

π0.6 原本把 robot state 表示成文本。但 MEM 需要输入过去 $K$ 个 state，如果每个 state 都转成 text，会产生大量 state text tokens。所以他们改用 continuous state embedding。

假设每个 proprioceptive state 是：

$$
s_i\in\mathbb{R}^{d_s}
$$

用一个 linear projection 投到 backbone embedding space：

$$
e_i=W_s s_i+b_s
$$

其中：

$$
e_i\in\mathbb{R}^{d_{\mathrm{model}}}
$$

长度为 $K$ 的 state history 变成：

$$
E_{\mathrm{state}}
=
[e_{t-K+1},\ldots,e_t]
\in
\mathbb{R}^{K\times d_{\mathrm{model}}}
$$

带 batch 时就是：

$$
E_{\mathrm{state}}
\in
\mathbb{R}^{B\times K\times d_{\mathrm{model}}}
$$

论文明确说，这样长度为 $K$ 的 observation memory 只产生 $K$ 个 proprioceptive state tokens。`MEM.pdf`

---

# 8. 训练流程总结

训练可以拆成 high-level 和 low-level 两套监督信号。

## 8.1 high-level 训练

high-level policy 学：

$$
(o_t,m_t,g)
\rightarrow
(l_{t+1},m_{t+1})
$$

其中：

- $l_{t+1}$ 来自 subtask language annotation；
- $m_t,m_{t+1}$ 来自外部 off-the-shelf LLM 生成的 memory labels；
- success/failure indicator 只用于离线 GT 生成，不是推理输入。

一个训练样本可以理解为：

$$
\text{input}=(o_t,m_t,g)
$$

$$
\text{target}=(l_{t+1},m_{t+1}^{\mathrm{GT}})
$$

## 8.2 low-level 训练

low-level policy 学：

$$
(o_{t-K:t},l_{t+1},g)
\rightarrow
a_{t:t+H}
$$

其中 $o_{t-K:t}$ 经过 video encoder 压缩成包含短期历史的当前帧 representation，再和 subtask、goal 一起输入 VLA/action expert。

## 8.3 pre-training

π0.6-MEM 在多种数据上 pre-train，包括：

- teleoperated robot demonstrations；
- policy rollout data；
- human corrections；
- vision-language tasks；
- video-language tasks，例如 video captioning。

pre-training 时使用 6 个 observations：

$$
5\text{ past observations}+1\text{ current observation}
$$

相邻 observation stride 是 1 秒。`MEM.pdf`

## 8.4 post-training

post-training 时，他们发现可以灵活扩展 observation memory horizon，比如扩展到 18 frames 和 54 seconds。机器人实验使用 inference-time RTC 或 training-time RTC 来做 asynchronous real-time inference。`MEM.pdf`

这说明 memory 能力最好在 pre-training 阶段就学出来，而不是目标任务 post-training 才临时加。Figure 9 也显示，只在 post-training 才引入 memory 的版本明显差于完整 MEM。`MEM.pdf`

---

# 9. 推理流程总结

推理时，系统循环运行。它维护两种 memory：

$$
m_t
$$

长期 language memory；

$$
o_{t-K:t}
$$

短期 observation buffer。

每一轮大概是：

## Step 1：输入任务目标

例如：

$$
g=\text{“Clean up the kitchen.”}
$$

初始化：

$$
m_0=\emptyset
$$

或类似 “No steps have been completed yet.”

## Step 2：high-level 生成 subtask 和新 memory

$$
(o_t,m_t,g)
\rightarrow
(l_{t+1},m_{t+1})
$$

例如：

$$
m_t=
\text{“I put the plate in the sink and wiped the counter.”}
$$

当前观测看到桌上还有杯子，high-level 输出：

$$
l_{t+1}=
\text{“Pick up the cup.”}
$$

$$
m_{t+1}=
\text{“I put the plate in the sink, wiped the counter, and am picking up the cup.”}
$$

## Step 3：low-level 读取短期视频记忆

最近 $K$ 帧多相机图像和机器人 state 进入 video encoder：

$$
o_{t-K:t}
\rightarrow
\text{video encoder}
\rightarrow
\tilde o_t
$$

## Step 4：low-level 输出 action chunk

$$
(\tilde o_t,l_{t+1},g)
\rightarrow
a_{t:t+H}
$$

## Step 5：机器人执行动作，更新 memory

执行动作 chunk，同时新的 observation 进入 buffer：

$$
o_{t-K:t}\leftarrow o_{t-K+1:t+1}
$$

语言记忆更新为：

$$
m_t\leftarrow m_{t+1}
$$

然后循环。

---

# 10. 实验：MEM 到底提升了什么？

论文实验主要验证三个问题：

1. MEM 是否能做需要 15 分钟记忆的长任务？
2. MEM 是否能让 VLA 根据短期失败经验做 in-context adaptation？
3. MEM 是否比其他 memory 方法更好？

## 10.1 长任务：Recipe setup 和 Clean kitchen

论文测试了两个长任务：

**Recipe setup**：机器人根据 recipe prompt 去多个柜子、抽屉、冰箱拿食材和厨具，并放到指定位置。它需要记住哪些东西已经拿过，还要记得任务结束时关柜门、抽屉、冰箱。

**Clean up kitchen**：机器人要收纳物体、擦台面、洗盘子、放入晾碗架。它要记住是否已经打过肥皂、盘子正反面是否洗过、哪些表面擦过、哪些柜门需要关上。论文说这些任务需要最长约 15 分钟记忆，并且不仅需要 memory，还需要复杂 dexterous manipulation。`MEM.pdf`

Figure 6 的结果显示：

- 没有 memory 的 π0.6 很难完成；
- 只有 video memory 不够；
- 只有 text memory 也不够；
- naive text + video memory 因为分布偏移表现差；
- 完整 π0.6-MEM 最好。

论文明确总结：short-horizon video memory 和 long-horizon language memory 都是 essential。没有 video memory，机器人可能不知道洗盘子/擦台面持续了多久，容易卡住；没有 language memory，机器人记不住远期语义事件，比如 recipe 步骤或哪些门需要关闭。`MEM.pdf`

## 10.2 In-context adaptation

MEM 还测试了两个短期 adaptation 任务：

- pick up chopstick：桌面高度 OOD，容易抓取失败，需要调整抓取高度；
- open fridge：门打开方向不清楚，模型可能反复朝错误方向尝试。

训练方法是收集 targeted human feedback：policy 失败后，人类接管示范正确策略。对于 fridge，还收集 demonstrator 一开始也不知道开门方式的探索轨迹，因此数据自然包含失败尝试和后续纠正。然后 finetune π0.6-MEM，并在训练时保留失败尝试在 short-term memory 中。这样模型学会：看到短期记忆里的失败，就调整策略。`MEM.pdf`

结果显示，带 memory 的 MEM-VLA 更能利用 correction data。没有 memory 的 π0.6 不知道刚才尝试过什么策略，因此无法有意识地改变策略；有 memory 的模型可以根据上下文知道“之前哪种策略失败了”，从而换策略。`MEM.pdf`

## 10.3 与其他 memory 方法比较

论文比较了：

**No Memory**：普通 π0.6；

**Pool Memory**：把过去 observation 分别编码，再 average pooling 成单个 memory token；

**Proprio Memory**：只用过去低维机器人 state；

**Ours / MEM video memory**：使用他们的 video encoder；

**MEM-Posttrain-Only**：只在 post-training 阶段引入 memory，不在大规模 pre-training 里学 memory。

测试任务覆盖：

- partial observability：记住物体放在哪个抽屉；
- unpack groceries：记住袋子里还有没有东西；
- swap 3 mugs：记住多个 mug 位置；
- scoop coffee：计数；
- grilled cheese：计时；
- window cleaning：空间记忆；
- 以及多种精细操作任务。

论文认为 Pool Memory 过度压缩观察，容易丢失长期视觉信息；Proprio Memory 只能记住机器人自己的状态，不能很好记住环境状态；MEM 是唯一在多类 memory capability 上都表现强的方案。`MEM.pdf`

Figure 9 还说明，memory pre-training 很重要。只在 post-training 加 video encoder 的模型明显更差。这说明 temporal memory 不是加个结构就自然会用，而是需要在大规模 robot / non-robot video data 上提前训练出来。`MEM.pdf`

---

# 11. 论文里几个容易误解的点

## 11.1 生成 GT 的 LLM 不是 high-level policy 本身

外部 LLM 只是离线 label generator：

$$
(l_{0:T},\text{success/failure})
\rightarrow
m_{0:T}
$$

high-level policy 是被训练的 student：

$$
(o_t,m_t,g)
\rightarrow
(l_{t+1},m_{t+1})
$$

推理时不再调用这个 external LLM。

## 11.2 success/failure indicator 推理时没有

indicator 只用于训练前离线生成 memory GT。推理时模型必须从当前 observation 和已有 memory 自己判断子任务是否完成、memory 是否该更新。

所以这里不是：

$$
(o_t,m_t,g,\text{success/failure})
\rightarrow
m_{t+1}
$$

而是：

$$
(o_t,m_t,g)
\rightarrow
m_{t+1}
$$

## 11.3 “不新增 learnable parameters” 不等于 freeze

video encoder 没有新增 temporal 参数，但已有 ViT 参数仍然需要继续训练。它的保证是：

- 初始化时可以加载原单帧 ViT；
- $K=1$ 时由于 $e(0)=0$，结构上退化到原单帧 ViT；
- 训练后不保证单帧输出完全不变。

## 11.4 MEM 不是简单的长上下文 VLA

它不是把 15 分钟视频全塞进去。它是：

$$
\text{long horizon}
\rightarrow
\text{compressed language memory}
$$

$$
\text{short horizon}
\rightarrow
\text{dense video memory}
$$

这才是 “multi-scale” 的核心。

---

# 12. 我对这篇文章的整体评价

MEM 的关键贡献不是提出了一个全新的 action model，而是提出了一个比较工程化、但很合理的 **VLA memory architecture**。它把机器人记忆拆成两类：

一类是长期、语义、低带宽的记忆：

$$
m_t
$$

比如“哪些东西已经拿过”“哪些步骤已经完成”“哪些门还开着”。

另一类是短期、视觉、动态、高带宽的记忆：

$$
o_{t-K:t}
$$

比如“刚才抓取怎么失败”“物体遮挡前在哪里”“当前动作持续了多久”。

这种拆分很符合机器人任务的本质：**不是所有历史都值得用像素保存，也不是所有细节都能用语言概括。**

如果只用 language memory，会丢失细粒度空间和运动信息；如果只用 video memory，又很难扩展到 15 分钟任务。MEM 的价值就在于同时保留这两个尺度。

---

# 13. 一句话总结

MEM 是给 π0.6 这类 VLA 增加多尺度记忆的系统：它用 high-level VLM 维护长期 language memory 并生成 subtask，用 low-level VLA + video encoder 处理短期 dense observation 并输出 action chunk；训练时用外部 off-the-shelf LLM 根据 subtask annotations 和 success/failure indicators 离线生成 language memory GT，用 video encoder 学短期视觉记忆；推理时不再调用外部 LLM，也没有显式 success/failure indicator，而是由 high-level policy 自己更新 memory，由 low-level policy 根据短期视频记忆执行动作。它的实验说明，长任务需要 language memory，短期失败修正和遮挡处理需要 video memory，两者结合才能稳定完成 15 分钟级别的复杂机器人任务。

## 相关笔记

- [[Robot/PI/ChatGPT-Pi_0.6论文问题解答|pi0.6]]：MEM 主要增强的底层 VLA/action policy。
- [[Robot/PI/ChatGPT-Pi_star0.6论文问题解答|pi*0.6 / RECAP]]：policy improvement 与 correction-learning 路线。
- [[Robot/PI/Pi0_7_technical_report|pi0.7]]：后续 rich context、subtask、subgoal 机制。
- [[Robot/ChatGPT-RDT-1B|RDT-1B]]：不显式建模 memory 的 continuous diffusion action policy 对照。
- [[Robot/PI/ChatGPT-Pi_0.5综述|pi0.5]]：长程任务中的 high-level language decomposition。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
