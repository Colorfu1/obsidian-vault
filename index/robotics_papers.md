# Robotics Papers Index

Use this index for robotics foundation models, Vision-Language-Action models, Physical Intelligence papers, action tokenization, flow matching policies, and memory models.

## Physical Intelligence PI Series

- [[Robot/PI/ChatGPT-Pi_0机器人文章分析|pi0]]
  - Topic: pi0, openpi, PaliGemma, VLM backbone, flow matching action expert, prefix/suffix tokens, attention masks, KV cache.
  - Importance: high
  - Notes: Start here for pi0 architecture and openpi implementation questions.

- [[Robot/PI/ChatGPT-Pi_0.5综述|pi0.5]]
  - Topic: pi0.5, long-horizon tasks, high-level language intermediate outputs, adaptive RMSNorm, timestep conditioning, flow matching action expert.
  - Importance: high
  - Notes: Use for pi0.5 architecture, training flow, and language/action decomposition.

- [[Robot/PI/ChatGPT-Pi_0.6论文问题解答|pi0.6]]
  - Topic: pi0.6, Section V-A, continuous action chunks, intermediate text, FAST discrete action tokens, joint likelihood, Knowledge Insulation.
  - Importance: high
  - Notes: Use for pi0.6 model-specific questions.

- [[Robot/PI/ChatGPT-Pi_star0.6论文问题解答|pi*0.6 / RECAP]]
  - Topic: pi*0.6, RECAP, experience corrections, value model, advantage-conditioned policy, positive/negative losses, offline RL pretraining.
  - Importance: high
  - Notes: Use for policy improvement, advantage conditioning, and correction-learning questions.

- [[Robot/PI/Pi0_7_technical_report|pi0.7]]
  - Topic: pi0.7, steerable generalist VLA, rich context, subtask instruction, subgoal images, episode metadata, RTC, CFG, mixed-quality data, pi*0.6 distillation.
  - Importance: high
  - Notes: Use for pi0.7 architecture, steerable policy conditioning, subgoal/subtask control, and relationships with MEM and pi*0.6.

## Action Tokenization

- [[Robot/PI/FAST_知识总结|FAST]]
  - Topic: FAST, action chunk tokenization, quantile normalization, DCT, sparse frequency-domain integer matrix, low-frequency-first flattening, BPE, FSQ.
  - Importance: high
  - Notes: Start here for action tokenization and FAST vs diffusion/VLA action output questions.

## Robot Memory Models

- [[Robot/PI/ChatGPT-MEM 文章分析|MEM]]
  - Topic: MEM, long-horizon robot memory, high-level language memory, low-level video memory, pi0.6-MEM, proprioceptive state, task adaptation.
  - Importance: high
  - Notes: Start here for memory-augmented VLA and long-horizon robotic manipulation questions.

## Diffusion and Continuous-Action Policies

- [[Robot/ChatGPT-RDT-1B|RDT-1B]]
  - Topic: RDT-1B, diffusion foundation policy, DiT denoising, clean-action prediction, continuous action chunks, physically interpretable unified action space, multi-robot pretraining, bimanual manipulation, ACI cross-attention conditioning.
  - Importance: high
  - Notes: Start here for RDT-1B, diffusion-based robot foundation policies, continuous action modeling, unified action/proprioception representation, and comparisons with OpenVLA, Octo, ACT, and pi-series flow-matching policies.

## Robot Hardware and Imitation Learning

- [[Robot/ChatGPT-ALOHA硬件与ACT算法|ALOHA / ACT]]
  - Topic: ALOHA low-cost bimanual leader-follower teleoperation hardware, WidowX/ViperX setup, joint-space mapping, ACT, action chunking, Transformer policy, CVAE latent variable imitation learning.
  - Importance: high
  - Notes: Start here for low-cost bimanual manipulation, data collection hardware, teleoperation design, ACT policy learning, and comparisons between ACT action chunking and VLA/FAST action tokenization.

## Useful Cross-Topic Notes

- [[VQVAE_综述|VQ-VAE]]
  - Topic: discrete token modeling, codebook, autoregressive prior.
  - Importance: high
  - Notes: Useful when comparing image/action tokenization with VLA action token design.

- [[RL/ChatGPT-PPO|PPO]]
  - Topic: PPO, advantage, actor-critic, policy gradient.
  - Importance: high
  - Notes: Useful for understanding why pi*0.6/RECAP discusses alternatives to PPO/TRPO.
