---
title: Diffusion Policy 概述
type: paper_note
topic: robot_policy_learning
status: mature
importance: high
updated: 2026-06-21
tags:
  - diffusion-policy
  - imitation-learning
  - action-chunking
  - receding-horizon-control
  - conditional-diffusion
  - robotics
---
# Diffusion Policy 概述

**User:** Anonymous  
**Created:** 6/21/2026 18:31:36  
**Updated:** 6/21/2026 21:48:37  
**Exported:** 6/21/2026 22:17:21  
**Link:** [https://chatgpt.com/c/6a37bd74-8d00-83ec-975a-59e33df41da7](https://chatgpt.com/c/6a37bd74-8d00-83ec-975a-59e33df41da7)  

**Diffusion Policy 把机器人策略建模成一个 conditional diffusion model：不是直接从观测回归动作，而是从随机噪声开始，逐步 denoise 出一段未来动作序列。** 也就是学：

$$
p(A_t \mid O_t)
$$

其中 $O_t$ 是当前/历史观测，$A_t$ 是未来一段 action chunk。`Diffusion_policy.pdf`

它主要解决传统 BC 的几个问题：同一个状态下可能有多种正确动作，普通 MSE 容易平均掉；单步动作预测容易抖动；高维连续动作序列不好建模。Diffusion 的优势是可以表达多模态动作分布、生成高维动作序列，并且训练比 EBM/IBC 这类 implicit policy 更稳定。`Diffusion_policy.pdf`

方法上，它有几个关键设计：预测一段 action sequence，而不是单步动作；用 receding horizon control，只执行前几步然后重新规划；视觉信息作为 condition，而不是一起生成；网络可以用 CNN 版，也可以用 Transformer 版。`Diffusion_policy.pdf`

实验上，它在 15 个仿真和真实机器人任务上超过 LSTM-GMM、IBC、BET 等方法，论文报告平均提升 46.9%。尤其在多模态任务、长时序任务和真实 manipulation 任务上优势明显。`Diffusion_policy.pdf`

我的理解是：**这篇论文不是 VLA、不是 world model、不是高层 reasoning；它的核心贡献是证明“diffusion action chunk + receding horizon”是一个非常强的低层连续动作生成策略。** 后续很多机器人 foundation model 里的 diffusion/flow action head，都可以看成继承了这条路线。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
## 相关笔记

- [[Robot/PI/FAST_知识总结|FAST 知识总结]]
- [[Robot/ChatGPT-ALOHA硬件与ACT算法|ALOHA 硬件与 ACT 算法]]
- [[Robot/PI/ChatGPT-Pi_0机器人文章分析|pi0 机器人文章分析]]
- [[Robot/PI/ChatGPT-Pi_0.6论文问题解答|pi0.6 论文问题解答]]
