---
title: 2026-07-24 ACT A-Sanity 收尾与后续实验指南
type: concept_note
topic: robot_imitation_learning
status: mature
importance: medium
updated: 2026-07-24
tags:
  - lerobot
  - act
  - libero
  - imitation-learning
  - experiment-log
---

# 2026-07-24 ACT A-Sanity 收尾与后续实验指南

## 重要进展

- A-Sanity 已完成五条 demonstration、859 帧的 teacher-forced offline 诊断，并完成
  20 次正式 LIBERO closed-loop rollout；20k checkpoint 的闭环成功率为 18/20（90%）。
- 20k checkpoint 的 first-action MAE 为 0.02076，delta-action MAE 为 0.00972；这些是
  离线动作预测诊断指标，不能代替模拟器任务成功率。
- 零更新 B0、完整训练 loss 和两次失败 rollout 的分类没有保留，因此 A-Sanity 不能给出
  严格的训练前后对照或完整失败分析。
- 后续 development 实验固定扩展到同一任务的 20 条 demonstrations、3233 帧，保持
  batch size 8、20,000 updates、seed 1000 和 `use_vae=false`。
- A0/A1 Action Chunking、A2 动作差分损失和 A3 Temporal Ensemble 的设计已完成独立审查；
  对应实施指南已经写好，其中描述的新增脚本、policy 插件和 evaluator 参数尚未实现。

## 总结范围

本次总结收尾 A-Sanity，并为后续三个 ACT 小实验提供明确的实施入口。离线评测和 Rerun
可视化只作为支撑能力简要说明，详细过程见
[[BestPractice/2026-07-23-ACT离线评测与标定可视化|ACT 离线评测与标定可视化]]。

A-Sanity 的实验规划、训练入口和源码分析见
[[BestPractice/2026-07-20-ACT-A-Sanity实验与模型源码调试|ACT A-Sanity 实验与模型源码调试]]。

## A-Sanity 实测结果

A-Sanity 使用公开的 LIBERO Object task 8：
`pick up the chocolate pudding and place it in the basket`。

| 实验字段 | 配置或结果 |
|---|---|
| Task suite | `libero_object` |
| Dataset size | 5 demonstrations，859 frames |
| Observation | 两路 256×256 RGB + 8D state |
| Action space | 7D relative EEF action |
| Action horizon | `chunk_size=20`，`n_action_steps=5` |
| Control frequency | 10 Hz |
| Train loss | 未保留完整日志，无法补录 |
| First-action MAE | 0.02076 |
| Delta-action MAE | 0.00972 |
| Teacher-forced action jerk | 0.01146 |
| Closed-loop success rate | 18/20（90%） |
| Closed-loop action TV | 0.02195 |
| Closed-loop action jerk | 0.02210 |
| Model-query latency p50/p95 | 18.28/24.47 ms |
| Failure cases | 2 次失败，尚未逐帧分类 |

100-step checkpoint 的 first-action MAE 为 0.16989，delta-action MAE 为 0.02365，只能作为
近似的早期训练参考。它不是零更新 B0，不能据此声称获得了严格的 step-0 到 20k 配对结果。

在当前证据范围内，可以确认：

- 20k checkpoint 在五条训练 demonstration 上的 open-loop action error 已明显降低；
- 20 次正式 LIBERO rollout 中成功 18 次；
- teacher-forced offline 指标和 closed-loop success 回答不同问题，两者均需保留。

## 后续实验设计

后续三个实验继续使用同一个 LIBERO task。Development 数据固定为该任务按 episode ID
排序后的前 20 条 demonstrations，共 3233 帧。这样既能延续 A-Sanity，又不会直接把实验规模
扩展到全部 LIBERO tasks。

公共训练条件：

| 项目 | 固定值 |
|---|---|
| Dataset size | 20 demonstrations，3233 frames |
| Observation | 两路 256×256 RGB + 8D state |
| Action space | 7D relative EEF action |
| Control frequency | 10 Hz |
| Batch size | 8 |
| Updates | 20,000 |
| Seed | 1000 |
| VAE | false |

三组实验分别回答：

1. A0/A1：单步预测与 action chunking 相比，是否能在保持成功率的同时减少动作抖动；
2. A2：增加动作差分损失后，预测动作变化是否更接近 demonstration，闭环 jerk 是否下降；
3. A3：同一个 source checkpoint 使用 Temporal Ensemble 后，是否比每帧普通重规划更稳定。

A3 增加 source-matched Replan control。A3 若使用 A2 checkpoint，就必须和同一个 A2
checkpoint 的 `n_action_steps=1, ensemble=None` 对比，不能把 A1 与 A2 的训练差异错误归因
给 Temporal Ensemble。

## 三份实施指南

### A0/A1：单步预测与 Action Chunking

- A0 使用 `chunk_size=1, n_action_steps=1`；
- A1 使用 `chunk_size=20, n_action_steps=5`；
- 两者固定相同数据、seed、训练步数和主要网络规模；
- 统一执行 offline、clean closed-loop 和 normalized proprioception noise 评测；
- 主要判定指标为 closed-loop action jerk 和 clean/noisy success rate。

### A2：动作差分损失

- 在实践仓库内实现 `act_delta` policy，不修改安装环境中的 LeRobot；
- 使用 `delta_loss_weight=0.1`；
- 只在相邻两个 action 都有效时计算 delta loss；
- A1 与 A2 从完全相同的 step-0 model tensors 开始；
- evaluator 需要同时识别 ACT family 和 `act_delta` concrete type。

### A3：Temporal Ensemble

- 不重新训练，加载 A1 或 A2 的已有 checkpoint；
- Standard 使用 checkpoint 原始 `n_action_steps=5`；
- Replan control 使用 `n_action_steps=1`、不启用 ensemble；
- A3 使用 `n_action_steps=1, temporal_ensemble_coeff=0.01`；
- A3 与 Replan control 必须使用同一个 source checkpoint 和相同 init states。

这三组内容是实施指南，不是已经完成的 20-demo 实验结果。指南中描述的启动脚本、本地 policy
插件、运行时 inference override 和 `proprio_gaussian` 扰动仍待实现。

## 离线评测与可视化基础

现有 offline evaluator 已能在 demonstration 上执行 teacher-forced action chunk prediction，
并在 Rerun 中同步展示两路 recorded images、Panda/EEF 3D 几何、GT/Pred action、逐维曲线、
metrics 和多个可切换 recording。这些能力可以用来观察过拟合程度、动作误差和预测 chunk。

离线 Pred action 没有在模拟器执行，future chunk 也不是未来真实机器人轨迹。任务是否成功、
执行动作是否抖动以及控制器实际响应，仍必须通过 LIBERO closed-loop rollout 判断。

## 当前限制

- A-Sanity 的零更新 B0 checkpoint 没有保留，100-step 只能作为近似参考；
- 训练终端日志没有保留，无法补录完整 train loss 曲线；
- 两次 closed-loop 失败尚未分类；
- 20-demonstration 的 A0、A1 和 A2 尚未训练；
- Temporal Ensemble 和 noisy evaluation 尚未完成本地 evaluator 接口；
- 三份指南定义的是接下来的实施步骤，不能提前写成实验结论。

## 下一步

1. 实现通用 step-0 initializer、A0/A1 启动脚本和可复现 proprioception noise；
2. 使用固定 20 条 demonstrations 完成 A0/A1 的 20k 训练、offline 和 clean/noisy
   closed-loop 对比；
3. 实现本地 `act_delta` policy，并从 A1 step-0 权重生成 A2 初始化；
4. 完成 A2 训练和同条件评测；
5. 为 closed-loop evaluator 增加推理参数覆盖，使用同源 checkpoint 完成 Replan 与
   Temporal Ensemble 对比；
6. 为每次正式实验记录 task、数据量、observation/action、horizon、frequency、train loss、
   open-loop error、closed-loop success、latency 和 failure cases。

## 相关笔记

- [[Robot/VLA/ALOHA硬件与ACT算法|ALOHA 硬件与 ACT 算法]]：ACT、action chunking、CVAE
  与 Temporal Ensemble 的算法背景。
- [[BestPractice/2026-07-20-ACT-A-Sanity实验与模型源码调试|ACT A-Sanity 实验与模型源码调试]]：
  五轨迹过拟合实验的启动器、评测设计和源码分析。
- [[BestPractice/2026-07-23-ACT离线评测与标定可视化|ACT 离线评测与标定可视化]]：
  teacher-forced checkpoint 对比、LIBERO 相机标定和 GT/Pred 可视化语义。
- [[BestPractice/lerobot-libero-setup-and-smoke-test|LeRobot LIBERO 环境准备与 smoke test]]：
  数据集、无头渲染和模拟环境的准备记录。
