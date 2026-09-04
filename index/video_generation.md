# Video Generation Index

用于导航视频生成基础模型、视频扩散、Flow Matching、视频 VAE、视频编辑和实时视频生成笔记。

## Video Foundation Models

- [[Video/Wan2.1技术报告|Wan2.1 技术报告]]
  - Topic: Wan2.1、Wan-VAE、video diffusion transformer、Flow Matching、umT5、dense caption、视频数据清洗、Context Parallel、diffusion cache、视频编辑、实时视频和 video-to-audio。
  - Importance: high
  - Notes: 通用视频基础模型主线入口；优先理解数据 recipe、时空 latent、DiT 训练和系统级加速，再阅读 I2V、编辑、个性化和实时生成扩展。

- [[Video/Cosmos3技术报告|Cosmos 3 技术报告]]
  - Topic: Cosmos 3、omnimodal world model、Mixture-of-Transformers、统一 AR/扩散 token、clean/noisy 条件配置、6D rotation、action Flow Matching、视频/音频/action 生成、Physical AI、action mid-training、AR–DM temporal gap。
  - Importance: high
  - Notes: 视频生成 backbone 向全模态 Physical AI 扩展的主线；优先看 MoT、action representation、数据 curriculum 和 action synergy，再与 Wan2.1、DreamZero、Dreamer 4 对照。

## Related Robotics and World Models

- [[Robot/WAM/UniPi_技术总结|UniPi 技术总结]]
  - Topic: text-guided video generation、video-as-policy、video planning、inverse dynamics、generative imitation learning。
  - Importance: high
  - Notes: 对照“先生成视频计划，再由 inverse dynamics 恢复动作”的机器人路线。

- [[Robot/WAM/DreamZero_Technical_Report|DreamZero 技术报告]]
  - Topic: Wan2.1-I2V-14B、joint video-action flow matching、world action model、chunk-level autoregression、实时机器人控制。
  - Importance: high
  - Notes: 观察通用视频生成先验如何被改造成可执行的 video-action policy。

- [[Robot/WAM/DreamerV4_技术报告|Dreamer 4 技术报告]]
  - Topic: scalable generative world model、causal video tokenizer、Diffusion Forcing、imagined rollouts、offline RL。
  - Importance: high
  - Notes: 对照纯视频生成模型与具有 action/reward/value 接口的生成式 world model。

## Core Concepts

- [[VQVAE_综述|VQ-VAE 综述]]
  - Topic: latent representation、codebook、压缩、autoregressive prior。
  - Importance: high
  - Notes: 适合补充理解视频 VAE 和视觉 tokenization 的表示瓶颈。
