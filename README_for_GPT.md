# README for GPT

This repository is my Obsidian knowledge base.

It contains technical notes, paper reading notes, and structured explanations collected during learning and research. Most existing notes are Chinese ChatGPT discussion exports that have been edited into concept or paper summaries.

## Main Topics

- AI fundamentals: VQ-VAE, autoregressive priors, reinforcement learning, PPO, SAC, OPD, KL, normalization, action/token modeling.
- Robotics: Physical Intelligence PI series, VLA, pi0.7, FAST, MEM, RDT-1B, robot foundation models, flow matching action policies, diffusion action policies.
- Robot imitation learning and hardware: ALOHA low-cost bimanual teleoperation, ACT action chunking, CVAE imitation learning, Diffusion Policy.
- Autonomous driving: LiDAR perception, TensorRT deployment, BEV visualization. This topic is planned, but there are no dedicated notes in the current index yet.
- Personal planning: life documents and planning notes. Keep private or explicitly indexed only when safe to expose.

## Main Directories

- `index/`
  - Navigation layer for GPT. Start here before reading topic notes.
- `Robot/`
  - Robotics paper notes and embodied AI notes.
  - `Robot/PI/` contains Physical Intelligence paper notes, including pi0, pi0.5, pi0.6, pi*0.6 / RECAP, pi0.7, FAST, and MEM.
  - Root-level `Robot/*.md` files contain non-PI robotics notes such as ALOHA hardware, ACT, Diffusion Policy, and RDT-1B.
- `RL/`
  - Reinforcement learning and LLM-training related notes, including PPO, SAC/PPO comparison, and OPD.
- Repository root
  - Cross-topic concept notes such as [[VQVAE_综述|VQ-VAE 综述]].

## Important Files

- [[index/master_index|Master Index]]
  - Top-level map of the vault.
- [[index/ai_fundamentals|AI Fundamentals Index]]
  - AI fundamentals, generative models, RL, and LLM training notes.
- [[index/robotics_papers|Robotics Papers Index]]
  - Robotics foundation model, Physical Intelligence, robot memory, action tokenization, diffusion policies, ALOHA hardware, and ACT notes.
- [[reading_status|Reading Status]]
  - Reading progress and next reading candidates.
- [[VQVAE_综述|VQ-VAE 综述]]
  - Core VQ-VAE, codebook, autoregressive prior, and weight tying note.

## How to Use This Vault

When answering my questions:

1. Search this repository first if the topic relates to my notes.
2. Start from [[README_for_GPT]], then [[index/master_index|Master Index]], then the most relevant topic index.
3. Prefer newer notes over older notes when multiple notes overlap.
4. Prefer files with `importance: high` for core explanations.
5. If a note is my interpretation rather than a paper claim, say so explicitly.
6. Cite file paths when possible.
7. Answer in Chinese by default.
8. Use browser-friendly block LaTeX with `$$ ... $$` for formulas.
9. If the repository does not contain enough information, say that clearly before using external knowledge.

## Conflict Handling

If two notes disagree:

1. Prefer the newer note by `updated` frontmatter or the exported update time in the note body.
2. Prefer paper notes for paper-specific claims.
3. Prefer concept notes for conceptual explanations and analogies.
4. If the conflict cannot be resolved from the vault, state the conflict and list the relevant file paths.

## Privacy Rules

Do not assume that personal, identity, account, contract, family, or credential-related files are safe to use. If such content appears, avoid summarizing sensitive details unless explicitly requested.

For GPT-connected usage, this repository should primarily contain technical, paper, and learning notes. Highly private materials should live outside this repo or remain unindexed.
