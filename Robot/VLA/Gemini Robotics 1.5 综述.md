---
title: Gemini Robotics 1.5 综述
type: paper_note
topic: embodied_robot_agent
status: mature
importance: high
updated: 2026-07-16
tags:
  - gemini-robotics
  - embodied-reasoning
  - vla
  - robot-agent
  - multi-embodiment
  - motion-transfer
  - thinking-vla
  - robotics
---

# Gemini Robotics 1.5 综述

> [!figure] 论文原始系统结构图
> ![[attachments/paper-figures/gemini-robotics-1.5-system-overview.png]]
> Gemini Robotics 1.5 VLA、Gemini Robotics-ER 1.5 与 Agentic Framework 的整体关系。原图来自 [Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer（arXiv:2510.03342）](https://arxiv.org/abs/2510.03342)，由论文源文件高分辨率导出。

---

这篇论文可以总结成一句话：

> Gemini Robotics 1.5 不是单纯提出一个新的 VLA，而是提出一个由 **multi-embodiment VLA、embodied reasoning VLM、thinking、tool use、success detection、safety reasoning** 组成的机器人 agent 系统。

它的系统结构是：

```text
GR-ER 1.5
    高层 embodied reasoning model
    负责理解场景、规划、工具调用、进度估计、成功检测、安全判断

GR 1.5
    低层 multi-embodiment VLA
    负责把短程自然语言指令转成连续机器人动作

Physical Agent
    GR-ER 1.5 + GR 1.5
    负责长程任务执行、失败恢复、工具辅助决策
```

这篇最重要的思想不是某个具体网络结构，而是机器人 foundation model 的范式变化：

```text
旧范式：
    图像 + 语言 → 动作

新范式：
    高层 VLM 负责理解、规划、监控、工具调用
    低层 VLA 负责短程动作执行
    两者通过自然语言子任务和环境反馈闭环连接
```

它的亮点是：

```text
1. 同一个 VLA checkpoint 能控制多种机器人；
2. Motion Transfer 带来跨 embodiment 技能迁移；
3. Thinking VLA 让动作执行更可解释、更适合多步任务；
4. GR-ER 1.5 提供强 embodied reasoning；
5. Agentic system 在长程任务上明显优于单独 VLA 或普通 VLM orchestrator；
6. 论文把 progress understanding、success detection、tool use、安全推理都纳入机器人系统。
```

它的不足是：

```text
1. 模型架构和训练细节严重不足；
2. MT 是核心贡献，但实现没有公开；
3. action space 适配方式没有讲；
4. thinking trace 训练方式没有讲；
5. benchmark 多为内部任务；
6. progress score 高不代表 full success 稳定；
7. 跨 embodiment 泛化还没有被证明为通用解法。
```

所以这篇文章最适合这样读：

> 不要把它当作一篇可以复现的 VLA 方法论文，而要把它当作 Google DeepMind 对“下一代机器人 foundation agent 系统形态”的一次系统展示。重点不是公式，而是它提出的系统分工：**高层 embodied reasoning + 低层 generalist VLA + thinking + motion transfer + closed-loop orchestration**。

## 相关笔记

- [[GR00T N1 综述|GR00T N1 综述]]
- [[MolmoAct2论文框架分析|MolmoAct2 论文框架分析]]
- [[Pi0_7_technical_report|π0.7 技术报告]]
- [[RT-2 论文综述|RT-2 论文综述]]
- [[FAST_知识总结|FAST 知识总结]]
- [[RDT-1B|RDT-1B]]



---
