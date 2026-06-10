# Robotics Papers Index

Use this index for robotics foundation models, Vision-Language-Action models, Physical Intelligence papers, action tokenization, flow matching policies, and memory models.

## Physical Intelligence PI Series

- `PI/ChatGPT-Pi_0机器人文章分析.md`
  - Topic: pi0, openpi, PaliGemma, VLM backbone, flow matching action expert, prefix/suffix tokens, attention masks, KV cache.
  - Importance: high
  - Notes: Start here for pi0 architecture and openpi implementation questions.

- `PI/ChatGPT-Pi_0.5综述.md`
  - Topic: pi0.5, long-horizon tasks, high-level language intermediate outputs, adaptive RMSNorm, timestep conditioning, flow matching action expert.
  - Importance: high
  - Notes: Use for pi0.5 architecture, training flow, and language/action decomposition.

- `PI/ChatGPT-Pi_0.6论文问题解答.md`
  - Topic: pi0.6, Section V-A, continuous action chunks, intermediate text, FAST discrete action tokens, joint likelihood, Knowledge Insulation.
  - Importance: high
  - Notes: Use for pi0.6 model-specific questions.

- `PI/ChatGPT-Pi_star0.6论文问题解答.md`
  - Topic: pi*0.6, RECAP, experience corrections, value model, advantage-conditioned policy, positive/negative losses, offline RL pretraining.
  - Importance: high
  - Notes: Use for policy improvement, advantage conditioning, and correction-learning questions.

## Action Tokenization

- `PI/FAST_知识总结.md`
  - Topic: FAST, action chunk tokenization, quantile normalization, DCT, sparse frequency-domain integer matrix, low-frequency-first flattening, BPE, FSQ.
  - Importance: high
  - Notes: Start here for action tokenization and FAST vs diffusion/VLA action output questions.

## Robot Memory Models

- `PI/ChatGPT-MEM 文章分析.md`
  - Topic: MEM, long-horizon robot memory, high-level language memory, low-level video memory, pi0.6-MEM, proprioceptive state, task adaptation.
  - Importance: high
  - Notes: Start here for memory-augmented VLA and long-horizon robotic manipulation questions.

## Useful Cross-Topic Notes

- `VQVAE_综述.md`
  - Topic: discrete token modeling, codebook, autoregressive prior.
  - Importance: high
  - Notes: Useful when comparing image/action tokenization with VLA action token design.

- `RL/ChatGPT-PPO.md`
  - Topic: PPO, advantage, actor-critic, policy gradient.
  - Importance: high
  - Notes: Useful for understanding why pi*0.6/RECAP discusses alternatives to PPO/TRPO.
