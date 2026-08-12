# AI Fundamentals Index

Use this index for generative models, reinforcement learning, LLM training, KL objectives, and related mathematical concepts.

## Generative Models

- [[VQVAE_综述|VQ-VAE 综述]]
  - Topic: VQ-VAE, codebook, quantizer, autoregressive prior, teacher forcing, loss mask, weight tying, text/image vocabulary design.
  - Importance: high
  - Notes: Core concept note. Start here for VQ-VAE, codebook loss, commitment loss, and autoregressive prior questions.

## Reinforcement Learning

- [[PPO_逻辑重构版|PPO：从策略梯度到裁剪目标]]
  - Topic: PPO, policy gradient derivation, Monte Carlo estimation, log-derivative trick, advantage, GAE, clipped surrogate.
  - Importance: high
  - Notes: 主 PPO 概念笔记；按“策略梯度 → Advantage/GAE → clipped objective”的连续链路组织，并补充 Monte Carlo 与 log-derivative 推导。

- [[SAC_PPO_compare|SAC vs PPO]]
  - Topic: PPO vs SAC, V/Q/A relationship, rollout buffer, replay buffer, on-policy vs off-policy, actor-critic gradient paths.
  - Importance: high
  - Notes: Use when comparing algorithm design choices between PPO and SAC.

- [[RL/opd_on_policy_distillation_知识笔记|OPD / On-Policy Distillation]]
  - Topic: OPD, on-policy distillation, teacher/student rollout, KL, forward KL, reverse KL, sampled-token reverse KL, PG-style loss.
  - Importance: high
  - Notes: OPD 主线入口；先理解 student state distribution 与 teacher supervision 的闭环，再阅读 Full-KL、sampled-token reverse KL 和 PG-style loss 的具体实现。

## Model-Based RL and World Models

- [[Visual Foresight|Visual Foresight]]
  - Topic: action-conditioned video prediction, Visual MPC, CEM planning, designated pixel planning, robot visual foresight.
  - Importance: high
  - Notes: Useful for understanding early pixel-space world models and MPC before latent world-model RL.

- [[PlaNet 论文概述|PlaNet 论文概述]]
  - Topic: latent dynamics, RSSM, model-based RL, CEM planning, MPC, reward model, observation model.
  - Importance: high
  - Notes: Useful for understanding learned world models, planning from pixels, and the precursor to Dreamer.

- [[Dreamer技术报告|Dreamer 技术综述：从真实经验到潜空间想象，再到可微行为学习]]
  - Topic: latent imagination, RSSM world model, sparse reward, $V_\lambda$, world-model actor-critic, pathwise gradients, Advantage comparison, and continuous action policies.
  - Importance: high
  - Notes: 逻辑重构完整版；用于理解 Dreamer 的完整训练链路，并对照 PPO 的 score-function gradient 与 Dreamer 的 pathwise gradient。

- [[DayDreamer论文综述与阅读重点|DreamerV2 与 DayDreamer：从潜空间想象到真实机器人在线学习]]
  - Topic: DreamerV2 improvements over Dreamer, discrete RSSM latent, KL balancing, REINFORCE/dynamics actor gradients, entropy exploration, and physical robot online RL.
  - Importance: high
  - Notes: 先理解 DreamerV2 的算法改进，再阅读 DayDreamer 如何把它部署到真实机器人在线学习。

- [[DreamerV3_技术报告|DreamerV3 技术报告]]
  - Topic: discrete RSSM, robust world-model losses, distributional critic, symlog, twohot, return normalization, policy gradient.
  - Importance: high
  - Notes: Useful for modern model-based RL, stable latent dynamics learning, and unified actor-critic training across domains.

- [[DreamerV4_技术报告|Dreamer 4 技术报告]]
  - Topic: causal video tokenization, Shortcut Model, Diffusion Forcing, x-prediction, imagined rollouts, PMPO, behavioral prior.
  - Importance: high
  - Notes: Useful for scalable generative world models, offline imagination training, and the interface between flow-based dynamics and reinforcement learning.

## Generative Policies and World-Action Models

- [[UniPi_技术总结|UniPi 技术总结]]
  - Topic: text-guided video generation, video diffusion, video planning, inverse dynamics, generative imitation learning.
  - Importance: high
  - Notes: Useful for understanding video-as-policy representations and how they differ from action-conditioned world models and direct action policies.

- [[WorldVLA 论文综述(不建议读)|WorldVLA 论文综述]]
  - Topic: autoregressive image/action token modeling, auxiliary one-step world prediction, multi-task VLA.
  - Importance: low
  - Notes: Read as a contrast case; it does not perform world-model planning at inference time.

- [[DreamZero_Technical_Report|DreamZero 技术报告]]
  - Topic: video diffusion priors, joint video-action flow matching, chunk-level autoregression, diffusion caching, asynchronous inference.
  - Importance: high
  - Notes: Useful for understanding how generative video models can become closed-loop action policies without an explicit planner or reward model.

- [[OA_WAM|OA-WAM 论文综述]]
  - Topic: object-centric representation, stable object addresses, attention routing, slot state prediction, auxiliary world modeling.
  - Importance: high
  - Notes: Useful for attention-level object binding and for distinguishing queryable state representations from explicit world-model planning.

## Cross-Topic Links

- [[Pi_star0.6论文问题解答|pi*0.6 / RECAP 论文问题解答]]
  - Topic: regularized RL, advantage-conditioned policy, value model, offline RL, RECAP.
  - Importance: high
  - Notes: Robotics paper note, but useful for RL-style policy improvement and advantage reweighting.

- [[FAST_知识总结|FAST 知识总结]]
  - Topic: action tokenization, quantile normalization, DCT, scale-and-round quantization, BPE compression.
  - Importance: high
  - Notes: Robotics paper note, but useful for tokenization and compression concepts.

- [[Diffusion Policy 概述|Diffusion Policy 概述]]
  - Topic: diffusion action policy, conditional diffusion, action chunks, receding-horizon control.
  - Importance: high
  - Notes: Robotics paper note, but useful for diffusion policy, multimodal action distributions, and action sequence generation questions.

- [[RDT-1B|RDT-1B]]
  - Topic: action diffusion, clean-action prediction, DiT denoising, continuous action chunks, unified action space.
  - Importance: high
  - Notes: Robotics paper note, but useful for foundation-scale diffusion policy, action distribution modeling, and x0/clean-action prediction questions.

## Not Yet Indexed

- Flow matching standalone notes: not found yet. Flow matching appears inside the PI robotics notes.
- Normalization standalone notes: not found yet. Related normalization appears in FAST and pi0.5 notes.
