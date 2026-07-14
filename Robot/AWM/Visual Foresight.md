---
title: Visual Foresight
type: paper_note
topic: visual_world_model_planning
status: mature
importance: high
updated: 2026-07-12
tags:
  - visual-foresight
  - visual-mpc
  - world-model
  - video-prediction
  - model-based-rl
  - cem-planning
  - robot-control
  - robotics
---

# Visual Foresight

**Visual Foresight 是一篇早期的“视觉世界模型 + MPC 控制”论文：机器人先通过自监督交互学习一个 action-conditioned video prediction model，然后在测试时用这个模型预测不同动作序列的未来图像，并选择最能达成目标的动作。** `Visual Foresight.pdf`

它的整体 pipeline 是：

```text
机器人随机交互收集数据
        ↓
训练视频预测模型：current image + actions → future images
        ↓
测试时用户指定目标
        ↓
采样多条候选动作序列
        ↓
预测每条动作序列的未来视觉结果
        ↓
用 cost function 打分
        ↓
选最优动作序列，只执行第一步
        ↓
重新观察，重新规划
```

核心方法叫 **Visual MPC**。它不是训练一个 policy network 直接输出 action，而是训练一个 forward dynamics model，然后每一步在线规划。这个思想和后来的 world model / WAM 很有历史联系。

这篇文章有几个关键技术点：

1. **Transformation-based video prediction**  
   模型不是直接生成下一帧，而是预测图像中的像素如何移动，也就是 warping / flow field。这样它既能预测未来图像，也能追踪用户指定的 designated pixel 未来会在哪里。

2. **Designated pixel planning**  
   用户可以点一个物体上的像素，再指定目标位置。模型预测这个像素未来的位置分布，然后 planner 选择能让它接近目标的动作。

3. **SNA / temporal skip connection**  
   普通 DNA 模型在物体被机械臂遮挡后容易“忘记”物体。SNA 通过 temporal skip connections 利用早期图像，缓解遮挡导致的 pixel tracking 失败。

4. **三类 cost function**  
   - **Pixel Distance Cost**：把指定像素移动到目标点。
   - **Registration-Based Cost**：通过当前图像和起始/目标图像配准，不断重新定位目标物体，适合长程闭环任务。
   - **Classifier-Based Cost**：用少量成功示例训练目标分类器，适合“物体在另一个物体左边/前面”这类抽象关系任务。

5. **CEM trajectory optimization**  
   测试时用 CEM 采样动作序列、预测未来、选低 cost 的动作序列。每次只执行第一步，然后重新规划。

实验上，它展示了同一个模型可以完成多种真实机器人任务，包括推动、避障、抓取、放置、布料折叠、多物体操作，以及扰动后的重新尝试。Registration-Based Cost 在长距离推动任务中明显优于只靠 predictor propagation 的方式，说明闭环重新定位很重要。`Visual Foresight.pdf`

不过它的限制也很明显：

- 模型很老，视频预测能力弱；
- 依赖 CEM 在线采样，推理效率不高；
- 任务 horizon 不长；
- 目标物体通常需要保持可见；
- 主要还是像素级/图像级规划，不具备现代 VLA/WAM 的语言理解和大规模预训练能力。

所以这篇论文今天不应该当 SOTA 方法读，而应该当作 **机器人视觉世界模型路线的早期代表** 来读。它最重要的价值是提出了一个非常清晰的范式：

> **先学“动作会让世界怎么变”，再基于想象出来的未来进行规划。** 🌍🤖

## 相关笔记

- [[PlaNet 论文概述|PlaNet 论文概述]]
- [[Dreamer技术报告|Dreamer 潜空间想象技术报告]]
- [[DayDreamer论文综述与阅读重点|DayDreamer]]
- [[UniPi_技术总结|UniPi]]
- [[Pi0_7_technical_report|π0.7 技术报告]]
- [[RDT-1B|RDT-1B]]
- [[RT-2 论文综述|RT-2 论文综述]]



---
