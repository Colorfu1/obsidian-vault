# AI Fundamentals Index

Use this index for generative models, reinforcement learning, LLM training, KL objectives, and related mathematical concepts.

## Generative Models

- [[VQVAE_综述|VQ-VAE 综述]]
  - Topic: VQ-VAE, codebook, quantizer, autoregressive prior, teacher forcing, loss mask, weight tying, text/image vocabulary design.
  - Importance: high
  - Notes: Core concept note. Start here for VQ-VAE, codebook loss, commitment loss, and autoregressive prior questions.

## Reinforcement Learning

- [[RL/ChatGPT-PPO|PPO]]
  - Topic: PPO, policy gradient, reward backpropagation, advantage, actor, critic, log probability, clipped surrogate, GAE, value estimation.
  - Importance: high
  - Notes: Long-form PPO concept note.

- [[RL/ChatGPT-SAC_PPO_compare|SAC vs PPO]]
  - Topic: PPO vs SAC, V/Q/A relationship, rollout buffer, replay buffer, on-policy vs off-policy, actor-critic gradient paths.
  - Importance: high
  - Notes: Use when comparing algorithm design choices between PPO and SAC.

- [[RL/opd_on_policy_distillation_知识笔记|OPD / On-Policy Distillation]]
  - Topic: OPD, on-policy distillation, teacher/student rollout, KL, forward KL, reverse KL, sampled-token reverse KL, PG-style loss.
  - Importance: high
  - Notes: Use for LLM distillation, token-level KL, and student on-policy training questions.

## Model-Based RL and World Models

- [[ChatGPT-Visual Foresight|Visual Foresight]]
  - Topic: action-conditioned video prediction, Visual MPC, CEM planning, designated pixel planning, robot visual foresight.
  - Importance: high
  - Notes: Useful for understanding early pixel-space world models and MPC before latent world-model RL.

- [[ChatGPT-PlaNet 论文概述|PlaNet 论文概述]]
  - Topic: latent dynamics, RSSM, model-based RL, CEM planning, MPC, reward model, observation model.
  - Importance: high
  - Notes: Useful for understanding learned world models, planning from pixels, and the precursor to Dreamer.

- [[Dreamer_潜空间想象技术报告|Dreamer 潜空间想象技术报告]]
  - Topic: latent imagination, world-model actor-critic, reward/value models, pathwise gradients, continuous action policies.
  - Importance: high
  - Notes: Useful for model-based RL, differentiable imagination, and comparing PPO-style score-function gradients with pathwise gradients.

## Cross-Topic Links

- [[ChatGPT-Pi_star0.6论文问题解答|pi*0.6 / RECAP 论文问题解答]]
  - Topic: regularized RL, advantage-conditioned policy, value model, offline RL, RECAP.
  - Importance: high
  - Notes: Robotics paper note, but useful for RL-style policy improvement and advantage reweighting.

- [[FAST_知识总结|FAST 知识总结]]
  - Topic: action tokenization, quantile normalization, DCT, scale-and-round quantization, BPE compression.
  - Importance: high
  - Notes: Robotics paper note, but useful for tokenization and compression concepts.

- [[ChatGPT-Diffusion Policy 概述|Diffusion Policy 概述]]
  - Topic: diffusion action policy, conditional diffusion, action chunks, receding-horizon control.
  - Importance: high
  - Notes: Robotics paper note, but useful for diffusion policy, multimodal action distributions, and action sequence generation questions.

- [[ChatGPT-RDT-1B|RDT-1B]]
  - Topic: action diffusion, clean-action prediction, DiT denoising, continuous action chunks, unified action space.
  - Importance: high
  - Notes: Robotics paper note, but useful for foundation-scale diffusion policy, action distribution modeling, and x0/clean-action prediction questions.

## Not Yet Indexed

- Flow matching standalone notes: not found yet. Flow matching appears inside the PI robotics notes.
- Normalization standalone notes: not found yet. Related normalization appears in FAST and pi0.5 notes.
