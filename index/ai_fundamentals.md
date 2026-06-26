# AI Fundamentals Index

Use this index for generative models, reinforcement learning, LLM training, KL objectives, and related mathematical concepts.

## Generative Models

- `VQVAE_综述.md`
  - Topic: VQ-VAE, codebook, quantizer, autoregressive prior, teacher forcing, loss mask, weight tying, text/image vocabulary design.
  - Importance: high
  - Notes: Core concept note. Start here for VQ-VAE, codebook loss, commitment loss, and autoregressive prior questions.

## Reinforcement Learning

- `RL/ChatGPT-PPO.md`
  - Topic: PPO, policy gradient, reward backpropagation, advantage, actor, critic, log probability, clipped surrogate, GAE, value estimation.
  - Importance: high
  - Notes: Long-form PPO concept note.

- `RL/ChatGPT-SAC_PPO_compare.md`
  - Topic: PPO vs SAC, V/Q/A relationship, rollout buffer, replay buffer, on-policy vs off-policy, actor-critic gradient paths.
  - Importance: high
  - Notes: Use when comparing algorithm design choices between PPO and SAC.

- `RL/opd_on_policy_distillation_知识笔记.md`
  - Topic: OPD, on-policy distillation, teacher/student rollout, KL, forward KL, reverse KL, sampled-token reverse KL, PG-style loss.
  - Importance: high
  - Notes: Use for LLM distillation, token-level KL, and student on-policy training questions.

## Cross-Topic Links

- `Robot/PI/ChatGPT-Pi_star0.6论文问题解答.md`
  - Topic: regularized RL, advantage-conditioned policy, value model, offline RL, RECAP.
  - Importance: high
  - Notes: Robotics paper note, but useful for RL-style policy improvement and advantage reweighting.

- `Robot/PI/FAST_知识总结.md`
  - Topic: action tokenization, quantile normalization, DCT, scale-and-round quantization, BPE compression.
  - Importance: high
  - Notes: Robotics paper note, but useful for tokenization and compression concepts.

- `Robot/ChatGPT-RDT-1B.md`
  - Topic: action diffusion, clean-action prediction, DiT denoising, continuous action chunks, unified action space.
  - Importance: high
  - Notes: Robotics paper note, but useful for diffusion policy, action distribution modeling, and x0/clean-action prediction questions.

## Not Yet Indexed

- Diffusion standalone notes: not found yet. Action diffusion appears in the RDT-1B robotics note.
- Flow matching standalone notes: not found yet. Flow matching appears inside the PI robotics notes.
- Normalization standalone notes: not found yet. Related normalization appears in FAST and pi0.5 notes.
