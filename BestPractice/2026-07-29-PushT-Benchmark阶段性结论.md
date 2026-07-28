---
title: PushT Benchmark 阶段性结论
type: concept_note
topic: robot_imitation_learning
status: draft
importance: medium
updated: 2026-07-29
tags:
  - pusht
  - diffusion-policy
  - behavior-cloning
  - checkpoint-selection
  - imitation-learning
---

# 2026-07-29 PushT Benchmark 阶段性结论

## 重要进展

- 完成 BCChunk 与 Diffusion 各 200,000 updates 的正式训练；两者使用相同
  165-episode 训练集和成对 step-zero 初始化，共享的 213 个 tensor 已验证一致。
- 在 20 个 development environment seed 上完成 checkpoint selection：BCChunk
  选择 150k，闭环成功率 55%；Diffusion 选择 175k，闭环成功率 80%。
- 逐 seed 对比中，Diffusion 比 BCChunk 高 25 个百分点，但配对检验
  `p=0.125`，当前只能描述为方向性优势。
- 完成全量 E0 数据审计：206 条 demonstration、25,650 帧、33 个自动分类的
  全局策略组；这些全局分组不能证明严格条件多模态。
- 完成 E1 双协议评测：只改变初始 noise 时差异几乎为零，改变完整 DDPM sampling
  seed 后出现可复现的 action-chunk 多样性。
- 正式 E2、E4 和最终报告尚未运行，因此还不能下独立泛化、成功多模态或
  success-latency Pareto 的最终结论。

这篇笔记承接
[[BestPractice/2026-07-27-PushT-BCChunk与Diffusion模型规划和训练|PushT BCChunk 与 Diffusion 模型规划和训练]]，
把当时的实验设计和 Smoke 结果更新为正式训练、选模和 E1 的实测结论。Diffusion
action chunk 与 receding-horizon control 的算法背景见
[[Robot/VLA/Diffusion Policy 概述|Diffusion Policy 概述]]。

## 受控实验设置

| 字段 | 固定设置 |
|---|---|
| Task suite | PushT |
| Dataset size | 206 episodes；165 train / 20 development / 21 held-out audit |
| Observation | 96×96 RGB + 2D agent position |
| Observation history | 2 帧 |
| Action space | 2D absolute target position |
| Action horizon | 内部 16 步；执行 `[1:9]`，共 8 步 |
| Control frequency | 10 Hz |
| Train seed | 1000 |
| Checkpoint interval | 25,000 updates，共比较 8 个 checkpoint |
| Diffusion selection setting | DDPM，100 inference steps，单次采样 |

BCChunk 和 Diffusion 使用同一数据、预处理、视觉编码器、temporal U-Net 容量、
optimizer 和 scheduler。成对初始化清单显示 213 个共享 tensor 一致，因此主要
受控差异是：

- BCChunk：直接回归 clean action chunk；
- Diffusion：预测 epsilon，再通过 DDPM 反向去噪生成 action chunk。

两种 loss 的语义不同，不能用绝对值判断谁更好。BCChunk 也是 action chunking
策略，它与 ACT 类 action chunk 的共同点和闭环限制可对照
[[Robot/VLA/ACT-补充理解-从Action-Chunking到闭环稳定性|ACT：从 Action Chunking 到闭环稳定性]]。

## E0：数据能证明什么

模式：`Offline / Formal`。

| 指标 | 结果 |
|---|---:|
| Demonstrations | 206 |
| Recorded frames | 25,650 |
| 自动分类的全局策略组 | 33 |
| Block geometry coverage | 100% |
| 显式正 success label | 0 |
| Terminal-only episodes | 206 |
| Mean episode max reward | 0.8923 |

E0 说明数据包含多种完整轨迹形态，而且移动块几何解析覆盖所有帧。但“全局轨迹
形态不同”不等于“相同 observation 下存在多个有效 action mode”。数据集也没有
显式正 success signal，206 条 episode 都只能结合 terminal 和 dense reward
解释。因此 E0 是数据多样性证据，不是严格条件多模态证据。

## 正式训练与 checkpoint selection

模式：训练为 `Formal`；选模为 `Closed-loop / Formal`。

两个模型均完成 200,000 updates：

| 模型 | 最后记录的 train loss | 含义 |
|---|---:|---|
| BCChunk | 0.000 | clean-action MSE；日志只有三位小数，不是精确零 |
| Diffusion | 0.004 | epsilon-prediction MSE |

在 20 个 development seed 上选出的结果是：

| 模型 | 选中 step | Success | Mean max reward | Offline denormalized MSE |
|---|---:|---:|---:|---:|
| BCChunk | 150k | 11/20，55% | 0.8652 | 582.31 |
| Diffusion | 175k | 16/20，80% | 0.8790 | 1169.87 |

选模曲线并不单调：

- BCChunk 的最低 offline MSE 在 100k（568.41），但 success 只有 35%；150k
  的 MSE 略高，却达到 55%；200k 又回落到 35%。
- Diffusion 的最低 offline MSE 在 100k（640.68），success 为 60%；175k
  的 MSE 更高，却达到 80%；200k 为 65%。

因此 checkpoint selection 必须看 closed-loop success。最低训练误差、最低
teacher-forced MSE 和最后 checkpoint 都不能替代闭环选模。

## Development 逐 seed 对照

模式：`Closed-loop / Formal`，但仍是 selection set，不是 final set。

| 结果 | 数量 |
|---|---:|
| BCChunk 成功 | 11/20，55%；Wilson 95% CI 34.2%–74.2% |
| Diffusion 成功 | 16/20，80%；Wilson 95% CI 58.4%–91.9% |
| 仅 BCChunk 成功 | 1 |
| 仅 Diffusion 成功 | 6 |
| 两者都成功 | 10 |
| 两者都失败 | 3 |

Diffusion 高 25 个百分点，并在 6 个 seed 上恢复了 BCChunk 的失败；相反方向只有
1 个 seed。对 7 个不一致 pair 做 exact two-sided McNemar 检验，得到
`p=0.125`。

这说明方向对 Diffusion 有利，但还不能称为统计显著。并且这 20 个 seed 已用于
checkpoint selection，因此 80% 对 55% 是选模集合上的表现，不能解释为独立
held-out 泛化能力。

## E1：Diffusion 的随机性来自哪里

模式：`Offline / Formal`。使用 10 个 held-out observation candidate，每个协议
产生 20 个 Diffusion action chunk，不执行环境动作。

E1 将同一批 initial noise 配对到两个协议：

- E1a：改变 initial noise，但固定 candidate-level reverse seed；
- E1b：同时改变 initial noise 和完整 DDPM reverse sampling seed。

| 指标 | E1a：initial noise only | E1b：full sampling seed |
|---|---:|---:|
| Denormalized trajectory pairwise RMS | 0.00659 px | 1.94563 px |
| Denormalized endpoint pairwise distance | 0.00750 px | 2.80300 px |
| 相同条件 repeat max error | 0 | 0 |

BCChunk 三次调用的最大误差同样为 0。E1b 相比 E1a：

- trajectory RMS 约扩大 295 倍；
- endpoint distance 约扩大 374 倍。

结论不是“Diffusion 只由初始 noise 决定”。相反，在当前 checkpoint 和 DDPM-100
下，只改变初始 noise 几乎不能改变输出；完整 reverse sampling stream 才产生了
主要 action-chunk 多样性。

但 E1 没有执行环境。约 2–3 像素的预测差异是否会形成不同且成功的闭环轨迹，必须
由 E2 回答。离线 action pattern 不等同于成功行为模式。

## 当前结论

1. **闭环选模不可省略。** Offline MSE 排名和 closed-loop success 排名不一致。
2. **Diffusion-175k 在 development 上有方向性优势。** 80% 对 55%，但
   `p=0.125` 且使用了 selection set，仍需独立评测。
3. **完整 DDPM 随机流会产生 action-chunk 多样性。** initial noise 单独变化
   几乎无效，full sampling seed 的影响大约高两个数量级。
4. **成功多模态尚未得到证明。** E0 只证明全局数据轨迹多样，E1 只证明离线
   action 多样，E2 才能验证多种成功闭环行为。
5. **部署效率尚未测量。** E4 完成前，不能判断减少 denoising steps 的速度收益
   与 success 代价。

外部预训练模型的兼容性参考是 7/10 成功、mean max reward 0.9926，但其 checkpoint
来源和 split 不同，不应与当前选模结果做公平排名。

## 下一步判断标准

### E2：闭环多 seed

对相同初始环境运行 1 条确定性 BCChunk 和 5 条不同 Diffusion policy seed，报告：

- single-sample success；
- any@5 success；
- 成功轨迹与失败轨迹各自的多样性。

只有多个 sampling seed 产生明显不同且成功的完整轨迹，才能声称存在成功闭环
多模态。

### E4：采样速度

比较 DDIM 5/10/20/50/100 与 DDPM-100，报告 p95 end-to-end action-chunk latency
和 closed-loop success。目标判断线是延迟至少加速 2 倍，同时 success drop
不超过 5 个百分点；这是标准，不是预期结果。

## 项目结果入口

以下路径相对于项目仓库，不属于 Obsidian Vault：

- E0：`outputs/pusht_diffusion/data_inspection_e0_full/metrics.md`
- Checkpoint selection：
  `outputs/pusht_diffusion/pilot/checkpoint_selection/metrics.md`
- Development comparison：
  `outputs/pusht_diffusion/pilot/development_comparison/metrics.md`
- E1：`outputs/pusht_diffusion/pilot/e1_fixed_observation/metrics.md`
- 项目日报：
  `reports/daily/2026-07-29-PushT-Benchmark阶段性结论.md`

## 相关笔记

- [[BestPractice/2026-07-27-PushT-BCChunk与Diffusion模型规划和训练|PushT BCChunk 与 Diffusion 模型规划和训练]]
- [[Robot/VLA/Diffusion Policy 概述|Diffusion Policy 概述]]
- [[Robot/VLA/ACT-补充理解-从Action-Chunking到闭环稳定性|ACT：从 Action Chunking 到闭环稳定性]]
