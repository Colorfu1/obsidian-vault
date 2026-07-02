# Robotics Papers Index

Use this index for robotics foundation models, Vision-Language-Action models, Physical Intelligence papers, action tokenization, diffusion policies, flow matching policies, and memory models.

## Physical Intelligence PI Series

- [[Robot/PI/ChatGPT-Pi_0机器人文章分析|pi0 机器人文章分析]]
  - Topic: pi0, openpi, PaliGemma, VLM backbone, flow matching action expert, prefix/suffix tokens, attention masks, KV cache.
  - Importance: high
  - Notes: Start here for pi0 architecture and openpi implementation questions.

- [[Robot/PI/ChatGPT-Pi_0.5综述|pi0.5 综述]]
  - Topic: pi0.5, long-horizon tasks, high-level language intermediate outputs, adaptive RMSNorm, timestep conditioning, flow matching action expert.
  - Importance: high
  - Notes: Use for pi0.5 architecture, training flow, and language/action decomposition.

- [[Robot/PI/ChatGPT-Pi_0.6论文问题解答|pi0.6 论文问题解答]]
  - Topic: pi0.6, Section V-A, continuous action chunks, intermediate text, FAST discrete action tokens, joint likelihood, Knowledge Insulation.
  - Importance: high
  - Notes: Use for pi0.6 model-specific questions.

- [[Robot/PI/ChatGPT-Pi_star0.6论文问题解答|pi*0.6 / RECAP 论文问题解答]]
  - Topic: pi*0.6, RECAP, experience corrections, value model, advantage-conditioned policy, positive/negative losses, offline RL pretraining.
  - Importance: high
  - Notes: Use for policy improvement, advantage conditioning, and correction-learning questions.

- [[Robot/PI/Pi0_7_technical_report|π0.7 技术报告]]
  - Topic: π0.7, steerable generalist VLA, rich context conditioning, subtask instruction, subgoal images, episode metadata, MEM-style video history encoder, RTC, CFG, mixed-quality data, π*0.6 behavior distillation.
  - Importance: high
  - Notes: Use for π0.7 architecture, prompt/context steering, mixed-quality data handling, and links between π0.6, π*0.6/RECAP, MEM, and generalist robot foundation models.

## RT Series and Web-Scale VLA

- [[Robot/ChatGPT-RT-1 论文综述|RT-1 论文综述]]
  - Topic: RT-1, Robotics Transformer, large-scale real-world robot behavior cloning, language-conditioned EfficientNet, TokenLearner, Transformer policy, discrete action bins, multi-task robot data scaling.
  - Importance: high
  - Notes: Start here for early large-scale robot transformer policies, RT-1 architecture, action discretization, and the data-scaling lineage leading to RT-2 and later VLA models.

- [[Robot/ChatGPT-RT-2 论文综述|RT-2 论文综述]]
  - Topic: RT-2, Vision-Language-Action model, PaLI-X/PaLM-E co-fine-tuning, web knowledge transfer to robot control, VQA-style action prompting, action tokens, semantic generalization.
  - Importance: high
  - Notes: Use for RT-2's VLM-to-action-token formulation, semantic generalization, web-scale co-training, and comparisons with RT-1, FAST, OpenVLA/Octo, and pi-series VLA models.
## Action Tokenization

- [[Robot/PI/FAST_知识总结|FAST 知识总结]]
  - Topic: FAST, action chunk tokenization, quantile normalization, DCT, sparse frequency-domain integer matrix, low-frequency-first flattening, BPE, FSQ.
  - Importance: high
  - Notes: Start here for action tokenization and FAST vs diffusion/VLA action output questions.

## Diffusion and Continuous-Action Policies

- [[Robot/ChatGPT-Diffusion Policy 概述|Diffusion Policy 概述]]
  - Topic: Diffusion Policy, conditional diffusion model for action chunks, denoising future action sequences, multimodal behavior cloning, receding horizon control, CNN/Transformer policy variants.
  - Importance: high
  - Notes: Start here for diffusion-based robot policy learning, comparisons with BC/IBC/BET, and links between Diffusion Policy and later diffusion/flow action heads in robot foundation models.

- [[Robot/ChatGPT-RDT-1B|RDT-1B]]
  - Topic: RDT-1B, diffusion foundation policy, DiT denoising, clean-action prediction, continuous action chunks, physically interpretable unified action space, multi-robot pretraining, bimanual manipulation, ACI cross-attention conditioning.
  - Importance: high
  - Notes: Start here for RDT-1B, diffusion-based robot foundation policies, continuous action modeling, unified action/proprioception representation, and comparisons with OpenVLA, Octo, ACT, and pi-series flow-matching policies.

## Humanoid and Generalist Robot Foundation Models

- [[Robot/ChatGPT-GR00T N1 综述|GR00T N1 综述]]
  - Topic: GR00T N1, open foundation model for generalist humanoid robots, dual-system VLA architecture, Eagle-2 VLM System 2, DiT / flow-matching System 1, embodiment-specific state/action adapters, data pyramid with real data, simulation trajectories, and neural trajectories.
  - Importance: high
  - Notes: Start here for humanoid robot foundation models, GR00T N1 architecture, VLM-conditioned action diffusion/flow matching, multi-source data mixture, rapid embodiment adaptation, and comparisons with RDT-1B, π0.7, RT-2, and Diffusion Policy.

## Robot Memory Models

- [[Robot/PI/ChatGPT-MEM 文章分析|MEM 文章分析]]
  - Topic: MEM, long-horizon robot memory, high-level language memory, low-level video memory, pi0.6-MEM, proprioceptive state, task adaptation.
  - Importance: high
  - Notes: Start here for memory-augmented VLA and long-horizon robotic manipulation questions.

## Robot Hardware and Imitation Learning

- [[Robot/ChatGPT-ALOHA硬件与ACT算法|ALOHA 硬件与 ACT 算法]]
  - Topic: ALOHA low-cost bimanual leader-follower teleoperation hardware, WidowX/ViperX setup, joint-space mapping, ACT, action chunking, Transformer policy, CVAE latent variable imitation learning.
  - Importance: high
  - Notes: Start here for low-cost bimanual manipulation, data collection hardware, teleoperation design, ACT policy learning, and comparisons between ACT action chunking and VLA/FAST action tokenization.

## Useful Cross-Topic Notes

- [[VQVAE_综述|VQ-VAE 综述]]
  - Topic: discrete token modeling, codebook, autoregressive prior.
  - Importance: high
  - Notes: Useful when comparing image/action tokenization with VLA action token design.

- [[RL/ChatGPT-PPO|PPO]]
  - Topic: PPO, advantage, actor-critic, policy gradient.
  - Importance: high
  - Notes: Useful for understanding why pi*0.6/RECAP discusses alternatives to PPO/TRPO.
