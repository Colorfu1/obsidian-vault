# Robotics Papers Index

Use this index for robotics foundation models, Vision-Language-Action models, Physical Intelligence papers, action tokenization, diffusion policies, flow matching policies, and memory models.

## Action and World Models

- [[Visual Foresight|Visual Foresight]]
  - Topic: Visual Foresight, action-conditioned video prediction, Visual MPC, designated pixel planning, SNA temporal skip connections, CEM trajectory optimization, and robot visual world models.
  - Importance: high
  - Notes: Start here for early visual world-model control, pixel-space prediction, and MPC-style robot planning before latent RSSM methods such as PlaNet and Dreamer.

- [[PlaNet 论文概述|PlaNet 论文概述]]
  - Topic: PlaNet, latent dynamics, RSSM, model-based RL from pixels, CEM planning, MPC, reward model, observation reconstruction, and latent overshooting.
  - Importance: high
  - Notes: Start here for world-model planning from pixels and the RSSM lineage that leads into Dreamer.

- [[Dreamer技术报告|Dreamer 潜空间想象技术报告]]
  - Topic: Dreamer, latent imagination, RSSM world model, actor-critic in imagined trajectories, reward/value models, pathwise gradients, and continuous-control model-based RL.
  - Importance: high
  - Notes: Use for understanding how PlaNet-style online planning becomes an amortized actor trained inside latent imagination.

- [[DayDreamer论文综述与阅读重点|DayDreamer 论文综述与阅读重点]]
  - Topic: DayDreamer, DreamerV2, real robot online learning, asynchronous actor-learner, RSSM, latent imagination, locomotion, manipulation, navigation.
  - Importance: high
  - Notes: Use for understanding how DreamerV2 is deployed for sample-efficient online learning on physical robots.

- [[DreamerV3_技术报告|DreamerV3 技术报告]]
  - Topic: DreamerV3, general world-model RL, discrete RSSM, KL balancing, symlog, twohot critic, return normalization, REINFORCE.
  - Importance: high
  - Notes: Use for DreamerV3 architecture, robust world-model training, distributional value learning, and cross-domain model-based RL.

- [[DreamerV4_技术报告|Dreamer 4 技术报告]]
  - Topic: Dreamer 4, scalable generative world model, causal tokenizer, Shortcut Forcing, Diffusion Forcing, imagination training, PMPO, offline RL.
  - Importance: high
  - Notes: Use for modern video-based world-model agents, scalable imagined environments, and policy improvement from fixed offline data.

- [[UniPi_技术总结|UniPi 技术总结]]
  - Topic: UniPi, video-as-policy, text-guided video diffusion, UPDP, hierarchical video planning, inverse dynamics, generative imitation.
  - Importance: high
  - Notes: Use for video generation as a policy interface and comparisons between visual planning, world models, and direct action generation.

- [[WorldVLA 论文综述(不建议读)|WorldVLA 论文综述]]
  - Topic: WorldVLA, Chameleon, unified action and image tokens, autoregressive VLA, next-frame prediction, action attention mask.
  - Importance: low
  - Notes: Use as a secondary comparison between auxiliary image prediction and closed-loop world-model planning.

- [[DreamZero_Technical_Report|DreamZero 技术报告]]
  - Topic: DreamZero, World Action Model, video-action flow matching, autoregressive chunks, DreamZero-Flash, asynchronous robot control, zero-shot policy.
  - Importance: high
  - Notes: Use for end-to-end policies built on video generation priors and for precise discussion of task-level versus embodiment-level zero-shot generalization.

- [[OA_WAM|OA-WAM 论文综述]]
  - Topic: OA-WAM, object-addressable world action model, object slots, address-only attention keys, address reset, world head, robust manipulation.
  - Importance: high
  - Notes: Use for object-centric world-action representations, stable target binding under geometric shift, and the limits of auxiliary world prediction.

- [[WLA_reading_notes|WLA 论文阅读笔记]]
  - Topic: WLA, WLA-0, World-Language-Action model, textual subtask reasoning, physical dynamics, World Expert, Action Expert, flow matching, and test-time scaling.
  - Importance: high
  - Notes: Use for a unified model that treats language-level planning and future-image prediction as complementary next-state representations, while retaining fast action-only inference and optional imagined-future ranking.

## Physical Intelligence PI Series

- [[Pi_0机器人文章分析|pi0 机器人文章分析]]
  - Topic: pi0, openpi, PaliGemma, VLM backbone, flow matching action expert, prefix/suffix tokens, attention masks, KV cache.
  - Importance: high
  - Notes: Start here for pi0 architecture and openpi implementation questions.

- [[Pi_0.5综述|pi0.5 综述]]
  - Topic: pi0.5, long-horizon tasks, high-level language intermediate outputs, adaptive RMSNorm, timestep conditioning, flow matching action expert.
  - Importance: high
  - Notes: Use for pi0.5 architecture, training flow, and language/action decomposition.

- [[Pi_0.6论文问题解答|pi0.6 论文问题解答]]
  - Topic: pi0.6, Section V-A, continuous action chunks, intermediate text, FAST discrete action tokens, joint likelihood, Knowledge Insulation.
  - Importance: high
  - Notes: Use for pi0.6 model-specific questions.

- [[Pi_star0.6论文问题解答|pi*0.6 / RECAP 论文问题解答]]
  - Topic: pi*0.6, RECAP, experience corrections, value model, advantage-conditioned policy, positive/negative losses, offline RL pretraining.
  - Importance: high
  - Notes: Use for policy improvement, advantage conditioning, and correction-learning questions.

- [[Pi0_7_technical_report|π0.7 技术报告]]
  - Topic: π0.7, steerable generalist VLA, rich context conditioning, subtask instruction, subgoal images, episode metadata, MEM-style video history encoder, RTC, CFG, mixed-quality data, π*0.6 behavior distillation.
  - Importance: high
  - Notes: Use for π0.7 architecture, prompt/context steering, mixed-quality data handling, and links between π0.6, π*0.6/RECAP, MEM, and generalist robot foundation models.

## RT Series and Web-Scale VLA

- [[RT-1 论文综述|RT-1 论文综述]]
  - Topic: RT-1, Robotics Transformer, large-scale real-world robot behavior cloning, language-conditioned EfficientNet, TokenLearner, Transformer policy, discrete action bins, multi-task robot data scaling.
  - Importance: high
  - Notes: Start here for early large-scale robot transformer policies, RT-1 architecture, action discretization, and the data-scaling lineage leading to RT-2 and later VLA models.

- [[RT-2 论文综述|RT-2 论文综述]]
  - Topic: RT-2, Vision-Language-Action model, PaLI-X/PaLM-E co-fine-tuning, web knowledge transfer to robot control, VQA-style action prompting, action tokens, semantic generalization.
  - Importance: high
  - Notes: Use for RT-2's VLM-to-action-token formulation, semantic generalization, web-scale co-training, and comparisons with RT-1, FAST, OpenVLA/Octo, and pi-series VLA models.

## Action Tokenization

- [[FAST_知识总结|FAST 知识总结]]
  - Topic: FAST, action chunk tokenization, quantile normalization, DCT, sparse frequency-domain integer matrix, low-frequency-first flattening, BPE, FSQ.
  - Importance: high
  - Notes: Start here for action tokenization and FAST vs diffusion/VLA action output questions.

## Diffusion and Continuous-Action Policies

- [[Diffusion Policy 概述|Diffusion Policy 概述]]
  - Topic: Diffusion Policy, conditional diffusion model for action chunks, denoising future action sequences, multimodal behavior cloning, receding horizon control, CNN/Transformer policy variants.
  - Importance: high
  - Notes: Start here for diffusion-based robot policy learning, comparisons with BC/IBC/BET, and links between Diffusion Policy and later diffusion/flow action heads in robot foundation models.

- [[RDT-1B|RDT-1B]]
  - Topic: RDT-1B, diffusion foundation policy, DiT denoising, clean-action prediction, continuous action chunks, physically interpretable unified action space, multi-robot pretraining, bimanual manipulation, ACI cross-attention conditioning.
  - Importance: high
  - Notes: Start here for RDT-1B, diffusion-based robot foundation policies, continuous action modeling, unified action/proprioception representation, and comparisons with OpenVLA, Octo, ACT, and pi-series flow-matching policies.

## Humanoid and Generalist Robot Foundation Models

- [[GR00T N1 综述|GR00T N1 综述]]
  - Topic: GR00T N1, open foundation model for generalist humanoid robots, dual-system VLA architecture, Eagle-2 VLM System 2, DiT / flow-matching System 1, embodiment-specific state/action adapters, data pyramid with real data, simulation trajectories, and neural trajectories.
  - Importance: high
  - Notes: Start here for humanoid robot foundation models, GR00T N1 architecture, VLM-conditioned action diffusion/flow matching, multi-source data mixture, rapid embodiment adaptation, and comparisons with RDT-1B, π0.7, RT-2, and Diffusion Policy.

- [[Gemini Robotics 1.5 综述|Gemini Robotics 1.5 综述]]
  - Topic: Gemini Robotics 1.5, multi-embodiment VLA, embodied reasoning VLM, thinking VLA, tool use, success detection, safety reasoning, motion transfer, and closed-loop robot agent orchestration.
  - Importance: high
  - Notes: Use for next-generation robot agent systems where high-level embodied reasoning, planning, progress monitoring, and low-level VLA action execution are separated into cooperating modules.

- [[MolmoAct2论文框架分析|MolmoAct2 论文框架分析]]
  - Topic: MolmoAct2, open action reasoning model for real-world robot deployment, Molmo2-ER embodied reasoning backbone, FAST action tokenizer, flow-matching continuous action expert, per-layer KV conditioning, adaptive depth-token reasoning, and inference optimization.
  - Importance: high
  - Notes: Use for deployment-oriented VLA systems that connect embodied/spatial reasoning, action tokenization, continuous control, depth reasoning, and open robot data/recipe reproducibility.

## Robot Memory Models

- [[MEM 文章分析|MEM 文章分析]]
  - Topic: MEM, long-horizon robot memory, high-level language memory, low-level video memory, pi0.6-MEM, proprioceptive state, task adaptation.
  - Importance: high
  - Notes: Start here for memory-augmented VLA and long-horizon robotic manipulation questions.

## Robot Hardware and Imitation Learning

- [[ALOHA硬件与ACT算法|ALOHA 硬件与 ACT 算法]]
  - Topic: ALOHA low-cost bimanual leader-follower teleoperation hardware, WidowX/ViperX setup, joint-space mapping, ACT, action chunking, Transformer policy, CVAE latent variable imitation learning.
  - Importance: high
  - Notes: Start here for low-cost bimanual manipulation, data collection hardware, teleoperation design, ACT policy learning, and comparisons between ACT action chunking and VLA/FAST action tokenization.

- [[BestPractice/2026-07-16-LeRobot-ACT调试与ALOHA数据可视化|2026-07-16 LeRobot ACT 调试与 ALOHA 数据可视化]]
  - Topic: LeRobot 0.4.4, ACT training debug, ALOHA dataset cache, Rerun synchronized cameras, joint signals, nominal bimanual forward kinematics, and frame-by-frame visualization.
  - Importance: medium
  - Notes: Use as an implementation-oriented companion to the ALOHA/ACT paper note when reproducing training entry points, VS Code debugging, dataset inspection, and 3D episode visualization.

- [[BestPractice/2026-07-20-ACT-A-Sanity实验与模型源码调试|2026-07-20 ACT A-Sanity 实验与模型源码调试]]
  - Topic: ACT A-Sanity launcher, LIBERO offline and closed-loop evaluation design, LeRobot dataset statistics, Accelerate BF16, CVAE, attention, and positional encoding.
  - Importance: medium
  - Notes: Use for the experiment plan and source-level reasoning that preceded the formal A-Sanity training run.

- [[BestPractice/2026-07-23-ACT离线评测与标定可视化|2026-07-23 ACT 离线评测与标定可视化]]
  - Topic: ACT offline checkpoint comparison, LIBERO calibrated agentview, Panda IK visualization, GT/prediction command semantics, action metrics, and Rerun tables.
  - Importance: medium
  - Notes: Use for interpreting teacher-forced ACT results and the visualization-only LIBERO image-orientation correction.

## Useful Cross-Topic Notes

- [[VQVAE_综述|VQ-VAE 综述]]
  - Topic: discrete token modeling, codebook, autoregressive prior.
  - Importance: high
  - Notes: Useful when comparing image/action tokenization with VLA action token design.

- [[PPO|PPO]]
  - Topic: PPO, advantage, actor-critic, policy gradient.
  - Importance: high
  - Notes: Useful for understanding why pi*0.6/RECAP discusses alternatives to PPO/TRPO.
