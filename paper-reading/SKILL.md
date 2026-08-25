---
name: paper-reading
description: Systematically read, analyze, compare, and summarize research papers into two-level illustrated notes containing a concise version followed by a complete version, emphasizing only critical, unconventional, counterintuitive, or decision-relevant findings while omitting routine expected improvements. Use when Codex needs to process a paper or paper collection (especially PDF/arXiv text), produce paper reading notes, explain the core idea, extract important figures, audit claims and experiments, compare related work, identify assumptions or shortcuts, assess novelty and practicality, or connect a paper to computer vision, 3D perception, autonomous driving, world models, imitation/reinforcement learning, multimodal models, and model efficiency.
---

# Paper Reading

将论文压缩为可复用的技术判断，而不是复述摘要。优先回答：问题为何重要、方法本质是什么、证据是否成立、什么条件下会失败、是否值得采用。

## 读取资料

1. 优先读取论文正文、附录、图表和用户提供的补充材料；不要只依赖摘要。
2. 处理 PDF 时，可运行 `scripts/extract_pdf_text.sh <paper.pdf> [output.txt]` 提取带页码标记的文本。文本不完整、公式或图表是关键证据时，继续查看原始页面。
3. 读取 `references/prior-knowledge.md`，把其中内容当作分析透镜而非事实来源。涉及计算机视觉、自动驾驶、world model、模仿学习或模型效率时尤其应读取。
4. 需要完整输出结构时，使用 `assets/paper-reading-notes-template.md`；用户指定格式时服从用户格式。

## 组织双层输出

1. 在同一个 notes 文件中先写“精简版”，再写“完整版”；使用明确标题和分隔线，使读者可以只读精简版，也可以继续深入。
2. 将精简版限制为五部分：一句话结论、核心方法、关键结果、主要限制、Takeaway。保持可独立理解，优先保留 2–4 个带出处的关键数字；不要加入长推导、完整实验表或次要模块。
3. 将完整版按后续流程展开，覆盖问题、baseline、数据流、目标函数、证据审计、假设、失败模式、实践成本、关联工作和开放问题。不要因已有精简版而省略关键分析。
4. 在完成完整版后回查精简版，确保方法描述、数字、证据强度与最终判断一致。精简版只压缩完整版，不得引入完整版中没有依据的新 claim。
5. 精简版默认不重复全部图片；最多放一张能解释全局 pipeline 的 overview figure。将其余关键 figure 放在完整版对应章节。

## 筛选值得记录的结果

1. 仅记录满足至少一项条件的结果：改变对核心机制的理解；揭示失败、负结果或适用边界；呈现反直觉 trade-off；是异常大的主导因素；隔离关键因果变量；直接改变部署、研究或数据决策；暴露评测协议漏洞。
2. 默认省略符合预期的常规现象，例如“更大模型更好”“更多数据整体更好”“加入模块后小幅涨点”“ours 在多数指标略高于 baseline”，以及重复定量结果的成功案例图。
3. 若常规结果是理解非常规结论所必需的对照，只用一句话或最少数字建立基线；不要复制完整排行榜、逐指标报数或为其单独配图。
4. 主动寻找并优先保留不符合作者主线叙事的结果：小模型退化、某指标与下游效果不一致、组件交互反转、tail/跨域失败、性能与成本相抵、平均提升掩盖子项退化。
5. 将结果章节命名并组织为“关键与非常规结果”，按洞察而不是按论文表格顺序书写。每项都说明“为什么值得记录”以及它改变了哪条认知。
6. 对精简版采用更严格的门槛；完整版也不要恢复被省略的常规结果清单。省略常规结果不等于省略反例、失败或不利证据。

## 提取关键 Figure

1. 对包含有效图表的论文，默认选择 3–6 张高信息密度 figure：整体架构/数据流、核心模块、数据构成、关键消融、非常规趋势、真实失败或关键数据效率。不要为常规单调涨点或“ours 更好”的定性案例配图，不要按编号机械收集。
2. 从原始 PDF 以 150–200 DPI 渲染并按 figure 边界裁剪；优先保留 figure number、图例和可读 caption。不要用整页截图代替裁剪，不要修改图中的科学内容。
3. 将图片存入 notes 同目录下的 `<notes-stem>-assets/`，按 `figure-<number>-<short-name>.png` 命名，并在 Markdown 中使用相对路径，保证 notes 可整体移动。
4. 在每张图片下写一段简短的“读图”：指出数据流、变量映射或趋势，以及读者应关注的部分。对实验图同时写明证据边界，例如归一化坐标、缺少 error bar、挑选案例或使用 ground truth constraint。
5. 区分图片作用：架构图解释方法，实验图支持 claim，定性图只展示案例。不要把架构示意图或 cherry-picked visualization 当作有效性证据。
6. 完成后逐张检查图片非空、裁剪完整、文字可读、没有无关正文，并验证所有 Markdown 图片链接存在。若 PDF 没有有用 figure 或无法可靠渲染，明确说明原因，不为满足数量而加入低价值图片。

## 分析流程

### 1. 建立问题坐标

- 写清任务、输入输出、监督信号、部署场景和评价指标。
- 还原最强且公平的 baseline；说明论文究竟替换、增加或删除了哪个环节。
- 用一句直白的话概括核心改动。若无法做到，继续拆解 pipeline，避免沿用作者包装。

### 2. 重建方法

- 按数据流描述模块，不按论文目录复述。
- 对每个关键模块记录：输入、输出、变换、训练信号、推理时是否保留。
- 解释关键公式的变量、优化目标和直觉；只展开决定方法成立与否的推导。
- 区分核心机制、必要实现细节、best practice 和无关装饰。

### 3. 找出隐含假设

- 检查几何、数据分布、坐标系、时序、可观测性、独立性、平滑性和算力假设。
- 问模型可能利用了哪些 shortcut：绝对位置、纹理、背景、数据采样、标签泄漏、预训练先验或评价协议。
- 检查作者声称的不变性、泛化性或可解释性是否有干预实验支持。

### 4. 审核证据

- 将每个主要 claim 对应到实验、表格、消融或可视化。
- 检查 baseline 是否同数据、同预训练、同训练预算、同 backbone、同后处理和同指标。
- 分离来自方法、更多数据、更强预训练、工程调参和评价设置的收益。
- 同时报告绝对提升、相对提升、方差/重复实验、跨数据集结果和失败案例；缺失即明确写“未验证”。
- 对复合指标追问提升来自哪个子项。例如 tracking 中不要让 detection 改善掩盖 association 退化；规划中不要用 open-loop 误差替代 closed-loop 安全性。

### 5. 形成独立判断

- 分别评价问题价值、想法新颖性、技术正确性、实验充分性、工程性价比和可迁移性。
- 允许结论为“想法常见但组合完整”“结果有效但机制证据不足”“学术新颖但部署价值弱”。不要把单一分数等同于全部价值。
- 给出最可能失败的场景、一个最关键的缺失消融，以及下一步最值得做的实验。
- 将论文与已知方法建立机制层面的联系，避免只列标题或表面相似词。

## 写作规则

- 先给一句话结论，再展开依据。
- 使用短句和具体名词；保留必要的中英文术语，但不要堆术语。
- 明确标注三类信息：`论文事实`、`分析推断`、`待验证`。不要把笔记作者观点写成学界共识。
- 引用数字时附表号、图号或页码；无法定位时不要伪造定位。
- 对方法的批评必须指出因果链：哪个假设不成立、影响哪个模块、会使哪个指标失真。
- 默认产出可扫描的中文笔记；论文名称、模块名、指标和公式保留原文。用户语言不同时随用户语言。
- 根据论文密度调整长度。短文可压缩，但不要省略证据审计、限制和最终判断。

## 最低完成标准

输出前确认以下问题均有答案：

- 论文解决了什么具体问题？
- 核心改动能否用一句话说清？
- 最关键的假设和监督信号是什么？
- 哪组实验真正支持主要 claim？
- 提升是否可能来自混杂因素？
- 方法在哪些场景会失败？
- 相比 baseline，部署成本和收益是什么？
- 读完后应保留的一条认知是什么？
- 是否已嵌入真正帮助理解的关键 figure，并解释其证据边界？
- 是否先提供了可独立阅读的精简版，再提供证据完整的完整版，且两者结论一致？
- 是否删掉了符合预期的常规提升，只保留关键、非常规、反直觉或影响决策的结果？
