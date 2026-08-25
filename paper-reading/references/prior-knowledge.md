# Paper Reading V3 认知基线

本文件总结《Paper Reading V3》中跨论文反复出现的判断框架。它是后续阅读的“先验问题集”，不是外部事实库，也不代表领域共识。使用时必须由新论文的原文和实验重新验证。

## 导航

- [1. 总体认识](#1-总体认识)
- [2. 表示与架构](#2-表示与架构)
- [3. 实验与指标](#3-实验与指标)
- [4. 自动驾驶、控制与世界模型](#4-自动驾驶控制与世界模型)
- [5. 模仿学习、RL 与推理训练](#5-模仿学习rl-与推理训练)
- [6. 数据、预训练与效率](#6-数据预训练与效率)
- [7. 后续生成 Notes 的默认判断顺序](#7-后续生成-notes-的默认判断顺序)
- [8. 来源边界](#8-来源边界)

## 1. 总体认识

### 论文价值不等于模块新颖度

- 一个常见想法若找准真实瓶颈、组合干净、细节完整且证据扎实，仍可能是高价值工作。PointRend、RAFT 一类笔记强调的是问题选择、迭代式 refinement 和完整实验，而不只是全新算子。
- 相反，复杂理论包装、漂亮可视化或 SOTA 数字不能替代关键假设与公平对比。先把方法还原成最朴素的算法，再评价新意。
- 将贡献拆成四层：新问题/新观察、核心机制、实现与训练技巧、数据或评测优势。分别判断，避免总分掩盖来源。

### 学习系统经常找到捷径

- 神经网络会优先利用数据中易学而稳定的相关性，不一定学习任务希望它掌握的物理或语义规律。
- 单目深度笔记显示模型可能依赖物体接地点的纵坐标、纹理和阴影，而非预期的尺度或几何；crop、camera pose、未知物体会暴露这种依赖。
- augmentation 也可能被浅层特征轻易识别。自监督代理任务若能靠低层线索完成，不保证得到语义表示。
- 后续阅读应主动寻找反事实干预：只改位置、尺度、颜色、背景、相机姿态、时间顺序或数据来源，观察预测是否按目标机制变化。

### 结构化先验与学习应各取所长

- 几何、优化、动力学和图结构不是深度学习的对立面。它们可以作为 feature transform、aggregation、differentiable solver、cost 或约束进入网络。
- 多视角任务中，将 2D feature 按相机几何 unproject 到 voxel/BEV，再跨视角融合，通常比要求网络从数据中重新发现投影关系更可靠。
- 对应关系、二分图匹配、PnP 等结构化问题可通过迭代展开、连续松弛或隐函数/KKT 条件实现端到端求导；要检查近似是否保留原约束。
- 对物理问题，优先预测更稳定或不变量，再结合显式几何恢复最终量，往往比直接回归全部结果更容易泛化。

## 2. 表示与架构

### 先问表示是否匹配任务

- 同一 feature 被多个目标共享时可能存在根本冲突。例如 detection 需要 category 信息，ReID/association 需要 identity 细节；简单多任务合并未必互利。
- 粒度和坐标系决定难度。点云压成 pillar、图像 patchify、BEV 转换、object-level vector tokenization 都在选择保留什么、丢弃什么。
- local 与 global 信息需要明确分工。多尺度特征、context、register token、virtual/global track 等设计常用于避免局部 token 被迫承担全局存储。
- latent-space prediction 通常比 pixel reconstruction 更聚焦任务相关变化，但其质量受 target encoder 和表示坍塌控制影响。不能默认 latent 就有语义。

### 常见有效机制

- **把算力集中到不确定区域**：PointRend 只细化不确定边界，体现“稀疏处理 hard examples”比均匀提高分辨率更划算。
- **迭代 refinement**：RAFT 的 recurrent update、query/bbox 的逐层重编码、diffusion 的局部去噪都通过重复小修正完成困难预测。检查每步是否共享参数、如何监督、是否真正收敛。
- **feature-level aggregation**：在语义已形成但空间信息尚未丢失的中间层融合，常比输入级或决策级融合更有效；仍需消融证明。
- **训练时复杂、推理时简化**：辅助 assign branch、重参数化、distillation 等可把训练成本换成零或低推理开销。必须确认移除后行为一致。
- **解耦不同预测目标**：path/traj、class/identity、objectness/class conditional probability、横向/纵向控制的分解可减少学习冲突，但需要验证误差传播。

### 警惕同义改名

- 将已有的 coarse-to-fine、hard example mining、cost volume、query refinement、curriculum learning、self-paced learning 或 importance weighting 换新名字，不自动构成新机制。
- 判断是否真正不同，应比较信息流、优化目标和约束，而不是比较模块名称。

## 3. 实验与指标

### 公平比较优先于排行榜

- 固定 backbone、预训练、输入分辨率、数据量、训练 schedule、后处理和测试增强后，再比较核心方法。
- 检查收益是否只在极小模型、特殊算力预算或单一数据集出现；跨尺度、跨数据集和跨硬件趋势比一个点更有信息。
- 小幅提升需要方差、重复实验或足够强的消融支持。多个技巧叠加后的总提升不能证明每个解释都正确。

### 指标必须贴近真实目标

- MOT 的 MOTA 强受 detection FP/FN 影响；若关心 association，应同时看 IDF1、ID switches，并在可比 detector 下分析。
- 自动驾驶规划的 open-loop imitation error 不能代表 closed-loop 安全性。闭环中 compounding error、交互、恢复能力和长时域 cost 更关键。
- 模型效率不能只看 FLOPs。必须报告目标硬件、batch size、memory、吞吐、端到端 latency 和算子实现；GPU 友好结构可能只在特定 batch 下占优。
- 视觉质量、可解释性图和 attention map 只能做辅助证据，不能替代定量干预与失败案例。

### 最有价值的消融

- 核心模块 vs 等参数/等算力替代方案。
- 数据、预训练与训练预算对齐后的对比。
- 关键假设被破坏时的 robustness test。
- 不同组件单独加入及交互项，而非只报 full model。
- 跨域、跨传感器、跨相机姿态或长时域测试。
- 推理成本和训练成本分开报告。

## 4. 自动驾驶、控制与世界模型

### 闭环是硬约束

- Behavior cloning 在训练分布内可拟合得很好，但小误差会把系统带入训练集外状态并累积。强 rule-based/IDM baseline 在闭环中可能优于看似先进的 imitation model。
- 规划论文应检查数据如何收集、expert 是否可靠、是否包含恢复轨迹、闭环仿真是否交互、指标 horizon 是否覆盖真正风险。
- 多模态轨迹输出只解决“可能性”，不保证选择器能找到安全轨迹。proposal quality、cost/reward quality 和 selector 分别评估。

### World model 的关键不是“生成未来”四个字

- 判断 state 表示、action conditioning、预测 horizon、rollout 误差、uncertainty 和 planner 如何消费预测。
- 只预测 future BEV/latent feature 可能比生成像素高效，但若用过窄的 semantic label 监督，也可能丢掉规划所需信息。
- 没有 action 条件的未来视频生成更像条件生成器，不应直接等同于可用于规划的 world model。
- 长时域 reward/cost 很难建模。即使短 horizon 评价函数与 leaderboard 一致，也可能漏掉 horizon 外风险。

### 端到端与模块化不是二选一

- 结构化 perception 输出、vectorized scene、MPC、geometry 和 learned score 可以组合。原始图像输入并不天然优于 object/vector 表示。
- LLM/VLM 可提供预训练知识、解释或高层决策，但要核查 action 表示、时序建模、数据生成闭环和真实 control metric；语言输出本身不等于可解释或可控。
- 将 path 与 timing/trajectory 解耦、把感知不确定性传到 cost、对候选轨迹在线评分，都是值得验证的接口设计。

## 5. 模仿学习、RL 与推理训练

- SFT 使用 off-policy expert token，on-policy RL 使用模型自身分布；混合时存在 policy shift、importance weighting 和 entropy collapse 问题。
- 单纯把 BC/SFT 与 policy-gradient loss 相加通常是有效 baseline，但要区分它是新原理还是权重技巧。
- Expert anchor/branched rollout 是合理 curriculum：困难样本使用更多 expert prefix，能力提高后逐步缩短提示，让训练从 off-policy 强制教学过渡到 on-policy 探索。
- 评价动态加权方法时，直接推导 per-token gradient，检查概率接近 0 或 1 时是否梯度消失、是否过度强化高概率 token、是否压缩探索。
- RL 对罕见驾驶场景可补足 BC，但 reward 的覆盖范围、仿真偏差和安全约束决定实际价值。

## 6. 数据、预训练与效率

- 数据质量、采样平衡和 expert 生成策略可能比模型结构更重要。先确认不同类别、行为、天气、视角和 hard cases 是否被合理覆盖。
- 更强的领域预训练有时远胜架构改动；比较时必须把预训练来源和分辨率列为主要变量。
- 自监督目标应迫使模型捕捉下游所需信息。若代理任务可用 logo、颜色、边缘或其他低层线索解出，应设计 shortcut removal 或反事实测试。
- NAS、pruning、supernet 和 scaling 结果要检查 search cost、weight sharing 偏差、latency predictor 适用硬件及搜索空间是否人为限制结论。
- 浅层、重参数化或“极简”网络只有在端到端部署指标上更好才有实际意义；层数少不等于内存访问少或 latency 低。

## 7. 后续生成 Notes 的默认判断顺序

1. 用一句话去包装地描述核心改动。
2. 找到它依赖的最强先验、最弱假设和最可能 shortcut。
3. 把主要 claim 与直接证据逐项对齐。
4. 排除数据、预训练、算力、后处理和指标造成的混杂。
5. 比较学术新意与工程收益，不把二者混为一谈。
6. 提出一个能推翻作者解释的关键实验。
7. 留下一条可迁移到其他论文或项目的机制性认知。

## 8. 来源边界

- 本认知基线来自一份个人长期笔记，覆盖约 111 个编号条目，时间跨度和详略不一致。
- 原笔记含鲜明个人判断、部分简写和未完整展开的公式，不应当作论文原文引用。
- 新 notes 若使用这里的观点，应写成“分析框架/已有认知”，并回到新论文核对事实、数字和出处。
- 对 2020 年前后与后续论文混合出现的主题，不要据此推断当前 SOTA；需要时另行检索最新资料。
