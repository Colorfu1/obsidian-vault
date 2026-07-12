# Obsidian Knowledge Vault

这是一个用于学习、论文阅读和技术脉络整理的 Obsidian 知识库。当前内容以中文技术笔记为主，重点覆盖机器人基础模型、VLA、AWM/world model、强化学习、动作建模和生成模型等方向。

这个仓库同时服务两类使用方式：

- 人类阅读：通过 Obsidian、GitHub 或本地 Markdown 浏览笔记。
- LLM 接入：通过 `README_for_GPT.md`、`index/` 和结构化链接帮助模型快速理解笔记脉络。

## 快速入口

- [[index/master_index|Master Index]]
  - 顶层导航，适合第一次进入知识库时使用。
- [[index/robotics_papers|Robotics Papers Index]]
  - 机器人、VLA、AWM/world model、action tokenization、diffusion/flow policy 等论文笔记入口。
- [[index/ai_fundamentals|AI Fundamentals Index]]
  - VQ-VAE、RL、PPO、SAC、OPD、model-based RL 等基础概念入口。
- [[reading_status|Reading Status]]
  - 已整理笔记和后续阅读候选。
- [[README_for_GPT|README for GPT]]
  - 面向 ChatGPT/LLM 的使用说明、冲突处理规则和隐私规则。

## 主要目录

```text
.
├── index/
│   ├── master_index.md
│   ├── robotics_papers.md
│   ├── ai_fundamentals.md
│   ├── autonomous_driving.md
│   └── personal_notes.md
├── Robot/
│   ├── AWM/
│   └── VLA/
│       └── PI/
├── RL/
├── README.md
├── README_for_GPT.md
├── reading_status.md
└── VQVAE_综述.md
```

### `Robot/AWM/`

Action/world model 与 model-based RL 相关笔记。当前重点包括：

- [[ChatGPT-Visual Foresight|Visual Foresight]]
- [[ChatGPT-PlaNet 论文概述|PlaNet]]
- [[Dreamer_潜空间想象技术报告|Dreamer]]

适合理解 visual MPC、video prediction、latent dynamics、RSSM、latent imagination、CEM planning、world model actor-critic 等内容。

### `Robot/VLA/`

Vision-Language-Action、机器人基础模型、动作生成和模仿学习相关笔记。当前重点包括：

- [[ChatGPT-RT-1 论文综述|RT-1]]
- [[ChatGPT-RT-2 论文综述|RT-2]]
- [[ChatGPT-Diffusion Policy 概述|Diffusion Policy]]
- [[ChatGPT-RDT-1B|RDT-1B]]
- [[ChatGPT-GR00T N1 综述|GR00T N1]]
- [[ChatGPT-Gemini Robotics 1.5 综述|Gemini Robotics 1.5]]
- [[ChatGPT-MolmoAct2论文框架分析|MolmoAct2]]
- [[ChatGPT-ALOHA硬件与ACT算法|ALOHA / ACT]]

### `Robot/VLA/PI/`

Physical Intelligence 系列与相关动作建模笔记。当前重点包括：

- [[ChatGPT-Pi_0机器人文章分析|pi0]]
- [[ChatGPT-Pi_0.5综述|pi0.5]]
- [[ChatGPT-Pi_0.6论文问题解答|pi0.6]]
- [[ChatGPT-Pi_star0.6论文问题解答|pi*0.6 / RECAP]]
- [[Pi0_7_technical_report|π0.7]]
- [[FAST_知识总结|FAST]]
- [[ChatGPT-MEM 文章分析|MEM]]

### `RL/`

强化学习和 LLM 训练相关基础笔记。当前重点包括：

- [[RL/ChatGPT-PPO|PPO]]
- [[RL/ChatGPT-SAC_PPO_compare|SAC vs PPO]]
- [[RL/opd_on_policy_distillation_知识笔记|OPD / On-Policy Distillation]]

## 推荐阅读路径

### 机器人基础模型 / VLA

1. [[ChatGPT-RT-1 论文综述|RT-1]]
2. [[ChatGPT-RT-2 论文综述|RT-2]]
3. [[FAST_知识总结|FAST]]
4. [[ChatGPT-Diffusion Policy 概述|Diffusion Policy]]
5. [[ChatGPT-RDT-1B|RDT-1B]]
6. [[Pi0_7_technical_report|π0.7]]
7. [[ChatGPT-GR00T N1 综述|GR00T N1]]
8. [[ChatGPT-Gemini Robotics 1.5 综述|Gemini Robotics 1.5]]
9. [[ChatGPT-MolmoAct2论文框架分析|MolmoAct2]]

### AWM / World Model / Model-Based RL

1. [[ChatGPT-PlaNet 论文概述|PlaNet]]
2. [[Dreamer_潜空间想象技术报告|Dreamer]]
3. [[Pi0_7_technical_report|π0.7]] 中的 subgoal images 与 world model 部分
4. [[ChatGPT-Visual Foresight|Visual Foresight]]
5. [[ChatGPT-RDT-1B|RDT-1B]] 中关于 “不是 world model” 的对比讨论

### RL 与策略优化基础

1. [[RL/ChatGPT-PPO|PPO]]
2. [[RL/ChatGPT-SAC_PPO_compare|SAC vs PPO]]
3. [[RL/opd_on_policy_distillation_知识笔记|OPD / On-Policy Distillation]]
4. [[ChatGPT-Pi_star0.6论文问题解答|pi*0.6 / RECAP]]

### 离散表示与动作 tokenization

1. [[VQVAE_综述|VQ-VAE 综述]]
2. [[FAST_知识总结|FAST]]
3. [[ChatGPT-RT-2 论文综述|RT-2]]
4. [[ChatGPT-MolmoAct2论文框架分析|MolmoAct2]]

## 使用建议

- 在 Obsidian 中优先从 `index/master_index.md` 或本文件进入。
- 使用关系图时，重点关注 `Robot/AWM/`、`Robot/VLA/`、`Robot/VLA/PI/` 和 `RL/` 之间的交叉链接。
- 如果多个笔记讨论同一主题，优先看更新时间较新的笔记，再回到早期笔记理解技术脉络。
- 如果问题涉及具体论文结论，优先阅读对应论文笔记；如果问题涉及概念解释，优先阅读 `index/ai_fundamentals.md` 中的基础笔记。

## 维护规则

- 新增笔记后，将其加入相关 `index/*.md` 文件。
- 新增论文笔记时，尽量补充 YAML frontmatter、`tags` 和 `## 相关笔记`。
- 机器人相关笔记优先放入 `Robot/AWM/` 或 `Robot/VLA/`，不要随意散落在根目录。
- 强化学习和 LLM 训练基础笔记放入 `RL/`。
- 面向 LLM 的全局说明更新到 `README_for_GPT.md`。
- 私人、身份、账号、合同、家庭或凭证相关内容不要进入 GPT-facing 索引，除非明确确认可以暴露。

## 同步脚本

仓库中保留了两个辅助脚本：

- `pull_obsidian.sh`
  - 从远端拉取最新 vault 内容。
- `push_obsidian.sh`
  - 将本地 vault 改动提交并推送到远端。

实际同步前建议先运行：

```bash
git status
```

确认没有意外删除、移动或未整理的新文件。
