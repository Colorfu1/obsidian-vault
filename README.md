# 🧠 Obsidian Knowledge Vault

> 一个面向 **机器人基础模型、VLA、AWM/world model、强化学习、动作建模、生成模型** 的中文技术知识库。

这里不是简单的论文堆叠，而是一张持续生长的技术地图：每篇笔记尽量放进它所属的脉络里，通过 Obsidian wikilink、索引页和阅读路径串起来。

---

## 🚪 快速入口

| 入口 | 用途 |
|---|---|
| [[index/master_index\|Master Index]] | 顶层地图，第一次进入知识库从这里开始 |
| [[index/robotics_papers\|Robotics Papers Index]] | 机器人、VLA、AWM、Diffusion Policy、FAST、PI 系列 |
| [[index/ai_fundamentals\|AI Fundamentals Index]] | VQ-VAE、RL、PPO、SAC、OPD、model-based RL |
| [[reading_status\|Reading Status]] | 已整理笔记、阅读进度、下一批候选 |
| [[README_for_GPT\|README for GPT]] | 给 ChatGPT / LLM 接入用的导航与规则 |

---

## 🌌 知识版图

```text
Obsidian Knowledge Vault
│
├── Robot/
│   ├── AWM/                    # world model / visual foresight / model-based RL
│   └── VLA/                    # vision-language-action / robot foundation models
│       └── PI/                 # Physical Intelligence 系列
│
├── RL/                         # PPO / SAC / OPD / policy optimization
│
├── VQVAE_综述.md               # 离散表示、codebook、tokenization
│
├── index/                      # 人类和 LLM 都优先读取的导航层
│
├── reading_status.md
├── README_for_GPT.md
└── README.md
```

---

## 🧭 推荐阅读路线

### 🤖 路线 A：机器人基础模型 / VLA

从早期大规模机器人 Transformer，一路走到现代 generalist robot foundation model：

```text
RT-1
  ↓
RT-2
  ↓
FAST / Diffusion Policy
  ↓
RDT-1B
  ↓
π0.7
  ↓
GR00T N1
  ↓
Gemini Robotics 1.5
  ↓
MolmoAct2
```

- [[ChatGPT-RT-1 论文综述|RT-1]]
- [[ChatGPT-RT-2 论文综述|RT-2]]
- [[FAST_知识总结|FAST]]
- [[ChatGPT-Diffusion Policy 概述|Diffusion Policy]]
- [[ChatGPT-RDT-1B|RDT-1B]]
- [[Pi0_7_technical_report|π0.7]]
- [[ChatGPT-GR00T N1 综述|GR00T N1]]
- [[ChatGPT-Gemini Robotics 1.5 综述|Gemini Robotics 1.5]]
- [[ChatGPT-MolmoAct2论文框架分析|MolmoAct2]]

### 🔮 路线 B：AWM / World Model / Model-Based RL

从“预测未来图像”到“在潜空间里想象并训练策略”：

```text
Visual Foresight
  ↓
PlaNet
  ↓
Dreamer
  ↓
π0.7 subgoal world model
  ↓
现代 VLA / AWM 对比
```

- [[ChatGPT-Visual Foresight|Visual Foresight]]
- [[ChatGPT-PlaNet 论文概述|PlaNet]]
- [[Dreamer_潜空间想象技术报告|Dreamer]]
- [[Pi0_7_technical_report|π0.7]] 中的 subgoal images 与 world model 部分
- [[ChatGPT-RDT-1B|RDT-1B]] 中关于 “不是 world model” 的对比讨论

### 🎮 路线 C：RL 与策略优化基础

适合补齐 policy gradient、actor-critic、on-policy/off-policy、distillation 相关背景：

- [[RL/ChatGPT-PPO|PPO]]
- [[RL/ChatGPT-SAC_PPO_compare|SAC vs PPO]]
- [[RL/opd_on_policy_distillation_知识笔记|OPD / On-Policy Distillation]]
- [[ChatGPT-Pi_star0.6论文问题解答|pi*0.6 / RECAP]]

### 🧩 路线 D：离散表示与动作 Tokenization

适合理解 VQ-VAE、action token、FAST、VLA 输出接口：

- [[VQVAE_综述|VQ-VAE 综述]]
- [[FAST_知识总结|FAST]]
- [[ChatGPT-RT-2 论文综述|RT-2]]
- [[ChatGPT-MolmoAct2论文框架分析|MolmoAct2]]

---

## 🗃️ 目录导览

### 🔮 `Robot/AWM/`

Action/world model 与 model-based RL 相关笔记。

| 笔记 | 关键词 |
|---|---|
| [[ChatGPT-Visual Foresight\|Visual Foresight]] | video prediction, Visual MPC, designated pixel, CEM |
| [[ChatGPT-PlaNet 论文概述\|PlaNet]] | RSSM, latent dynamics, CEM planning, MPC |
| [[Dreamer_潜空间想象技术报告\|Dreamer]] | latent imagination, actor-critic, pathwise gradient |

### 🦾 `Robot/VLA/`

Vision-Language-Action、机器人基础模型、动作生成和模仿学习相关笔记。

| 笔记 | 关键词 |
|---|---|
| [[ChatGPT-RT-1 论文综述\|RT-1]] | Robotics Transformer, large-scale robot data |
| [[ChatGPT-RT-2 论文综述\|RT-2]] | web-scale VLM, action-as-token |
| [[ChatGPT-Diffusion Policy 概述\|Diffusion Policy]] | action diffusion, receding horizon |
| [[ChatGPT-RDT-1B\|RDT-1B]] | DiT, continuous action chunk, diffusion policy |
| [[ChatGPT-GR00T N1 综述\|GR00T N1]] | humanoid foundation model, data pyramid |
| [[ChatGPT-Gemini Robotics 1.5 综述\|Gemini Robotics 1.5]] | embodied reasoning, agentic robot system |
| [[ChatGPT-MolmoAct2论文框架分析\|MolmoAct2]] | action reasoning, FAST, adaptive depth |
| [[ChatGPT-ALOHA硬件与ACT算法\|ALOHA / ACT]] | bimanual hardware, action chunking, imitation learning |

### 🧬 `Robot/VLA/PI/`

Physical Intelligence 系列与相关动作建模笔记。

| 笔记 | 关键词 |
|---|---|
| [[ChatGPT-Pi_0机器人文章分析\|pi0]] | flow matching, action expert, OpenPI |
| [[ChatGPT-Pi_0.5综述\|pi0.5]] | long-horizon, language intermediate outputs |
| [[ChatGPT-Pi_0.6论文问题解答\|pi0.6]] | FAST tokens, Knowledge Insulation |
| [[ChatGPT-Pi_star0.6论文问题解答\|pi*0.6 / RECAP]] | value model, advantage-conditioned policy |
| [[Pi0_7_technical_report\|π0.7]] | steerable VLA, subgoals, metadata, CFG |
| [[FAST_知识总结\|FAST]] | DCT, BPE, action tokenization |
| [[ChatGPT-MEM 文章分析\|MEM]] | robot memory, long-horizon control |

### 🎮 `RL/`

强化学习和 LLM 训练相关基础笔记。

| 笔记 | 关键词 |
|---|---|
| [[RL/ChatGPT-PPO\|PPO]] | policy gradient, GAE, clipped objective |
| [[RL/ChatGPT-SAC_PPO_compare\|SAC vs PPO]] | on-policy vs off-policy, actor-critic |
| [[RL/opd_on_policy_distillation_知识笔记\|OPD / On-Policy Distillation]] | KL, teacher/student, on-policy distillation |

---

## 🧠 怎么使用这个库

- **想找方向**：先打开 [[index/master_index|Master Index]]。
- **想看机器人论文**：直接进入 [[index/robotics_papers|Robotics Papers Index]]。
- **想补基础概念**：进入 [[index/ai_fundamentals|AI Fundamentals Index]]。
- **想看关系图**：重点观察 `Robot/AWM/`、`Robot/VLA/`、`Robot/VLA/PI/`、`RL/` 之间的交叉链接。
- **遇到同主题多笔记**：优先读更新时间更晚、`importance: high` 的笔记，再回到早期笔记看技术演化。
- **给 LLM 使用**：让模型先读 [[README_for_GPT|README for GPT]]，再进入 `index/`。

---

## 🛠️ 维护规则

- 新增笔记后，加入相关 `index/*.md`。
- 新增论文笔记时，尽量补充 YAML frontmatter、`tags` 和 `## 相关笔记`。
- 机器人相关笔记优先放入 `Robot/AWM/` 或 `Robot/VLA/`。
- 强化学习和 LLM 训练基础笔记放入 `RL/`。
- 面向 LLM 的全局说明更新到 `README_for_GPT.md`。
- 私人、身份、账号、合同、家庭或凭证相关内容不要进入 GPT-facing 索引，除非明确确认可以暴露。

---

## 🔄 同步脚本

仓库保留了两个辅助脚本：

| 脚本 | 用途 |
|---|---|
| `pull_obsidian.sh` | 从远端拉取最新 vault 内容 |
| `push_obsidian.sh` | 将本地 vault 改动提交并推送到远端 |

同步前建议先运行：

```bash
git status
```

确认没有意外删除、移动或未整理的新文件。

---

## 🌱 当前状态

- 已整理：VLA、AWM/world model、PI 系列、Diffusion Policy、FAST、RL 基础。
- 待扩展：RT-X、OpenVLA、Octo、Flow Matching 基础、Diffusion model 基础。
- 长期目标：把零散论文笔记整理成可导航、可复用、可被 LLM 接入的研究地图。
