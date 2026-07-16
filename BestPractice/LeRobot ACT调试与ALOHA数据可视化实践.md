---
title: LeRobot ACT 调试与 ALOHA 数据可视化实践
type: concept_note
topic: robot_imitation_learning
status: draft
importance: medium
updated: 2026-07-16
tags:
  - lerobot
  - aloha
  - act
  - debugging
  - dataset-visualization
---
# LeRobot ACT 调试与 ALOHA 数据可视化实践

> 实践日期：2026-07-16
>
> 原始 workspace：`/home/mi/codes/practice`
>
> 原始日志：`reports/2026-07-16-progress.md`

## 最终结论

本次实践完成了 LeRobot ACT 训练链路的入口确认、VS Code Python Debug 配置、ALOHA
数据检查，以及一个仓库自有的同步可视化工具。现在可以在同一条 Rerun 时间轴上查看
三路相机、具名关节信号、双臂标称 3D 骨架和末端轨迹。

`lerobot/aloha_mobile_cabinet` 在 LeRobot 默认 root 中的重复下载已经完成：parquet、
六个 MP4 均存在，已无 `.incomplete` 文件。收尾复核时没有 ACT 训练进程，且 workspace
内没有本次 3-step Debug 的 checkpoint 或训练日志，因此本日志不记录 train loss、
成功率等正式实验指标。

## 环境与训练入口

已验证项目环境：

| 组件 | 当前版本或位置 |
|---|---|
| Conda 环境 | `robot-practice` |
| Python | `3.10.20` |
| PyTorch | `2.10.0+cu128` |
| LeRobot | `0.4.4` |
| Rerun | `0.26.2` |
| GPU | NVIDIA GeForce RTX 3090 |
| LeRobot 包 | `/home/mi/miniconda3/envs/robot-practice/lib/python3.10/site-packages/lerobot` |

`./scripts/doctor.sh` 和 `./scripts/quickstart_info.sh` 已在项目环境中通过。项目继续使用
`PYTHONNOUSERSITE=1`，避免用户目录下的包污染 Conda 环境。

`lerobot-train` 的 console script 位于：

```text
/home/mi/miniconda3/envs/robot-practice/bin/lerobot-train
```

它最终导入并调用：

```python
from lerobot.scripts.lerobot_train import main
```

因此训练主文件是：

```text
$CONDA_PREFIX/lib/python3.10/site-packages/lerobot/scripts/lerobot_train.py
```

## VS Code Debug

workspace 中添加了两份本地 VS Code 配置：

- `.vscode/settings.json`：固定使用 `robot-practice` Python；
- `.vscode/launch.json`：定义 `LeRobot: ACT train (3-step debug)`。

Debug 配置使用 Python module 方式启动 `lerobot.scripts.lerobot_train`，并设置：

```text
--dataset.repo_id=lerobot/aloha_mobile_cabinet
--policy.type=act
--steps=3
--batch_size=2
--num_workers=0
--policy.push_to_hub=false
--save_checkpoint=false
--eval_freq=0
--wandb.enable=false
--log_freq=1
```

`justMyCode=false` 允许进入 Conda 环境中的 LeRobot 源码。第一个推荐断点是
`lerobot_train.py:170` 的：

```python
cfg.validate()
```

此时 CLI 参数已经解析为 `TrainPipelineConfig`，但 dataset、policy 和 optimizer 尚未构造。
后续可继续观察 `make_dataset()`、`make_policy()`、optimizer 构造和
`update_policy()`。

对应设计文档位于原始 workspace：

```text
docs/superpowers/specs/2026-07-16-lerobot-vscode-debug-design.md
```

## ALOHA 数据集

本地 metadata 已验证：

| 字段 | 值 |
|---|---|
| Repo ID | `lerobot/aloha_mobile_cabinet` |
| Robot type | `aloha` |
| Episodes | 85 |
| Frames | 127,500 |
| Control frequency | 50 Hz |
| Episode 0 | 1,500 帧，约 30 秒 |
| State/action/effort | 各 14 维，左臂 7 维 + 右臂 7 维 |
| Cameras | `cam_high`、`cam_left_wrist`、`cam_right_wrist` |

Episode 0 的任务是：打开上层柜门，把锅放入柜中，然后关上柜门。

### 两套本地缓存

之前下载的完整 Hub snapshot 仍然存在：

```text
/home/mi/.cache/huggingface/lerobot/hub/
  datasets--lerobot--aloha_mobile_cabinet/
  snapshots/7a752b39f7e69de7e38aee485a6bab07528a061a
```

其中 `data/chunk-000/file-000.parquet` 约 8.7 MB，三路相机的六个 MP4 合计约
1.7 GB。snapshot 中的文件是指向同一 dataset cache 下 `blobs/` 的软链接。

VS Code Debug 使用的是 LeRobot 默认 dataset root：

```text
/home/mi/.cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet
```

这解释了为什么看起来又下载了一次：完整 snapshot 和 LeRobot 的
`snapshot_download(local_dir=...)` 默认 root 是两个物理目录，不能自动跨目录复用。
截至本次实践收尾，默认 root 也已完整，占用约 1.7 GB。后续保持相同配置会直接复用它，
不需要再次下载。

## 同步数据与 3D 可视化

原始 workspace 新增了两个自有模块，没有修改安装在 `site-packages` 中的 LeRobot：

- `src/robot_practice/aloha_kinematics.py`：NumPy 实现的双臂 ViperX 300S 标称
  正向运动学；
- `src/robot_practice/visualize_aloha.py`：dataset 校验、批量读取、派生信号和
  Rerun logging/blueprint。

可视化内容包括：

- 三路 RGB 相机；
- 14 个具名关节的 position、action 和 effort；
- 使用相邻真实 timestamp 计算的 velocity；
- `action - state`；
- 左右臂当前骨架、关节点、夹爪开合和末端坐标系；
- 左右末端执行器的完整 XYZ 轨迹；
- `frame_index` 与 elapsed `timestamp` 同步时间轴。

3D 只表示标称运动学，不包含柜子、锅、碰撞、动力学或接触力，也没有与 RGB 相机完成
外参标定或空间配准。

完整 Episode 0 已保存在原始 workspace：

```text
outputs/aloha_mobile_cabinet_episode_0.rrd  # 约 186 MB
outputs/rerun_viewer_clean.png              # 最终界面截图
```

直接打开 recording：

```bash
cd /home/mi/codes/practice
conda run -n robot-practice rerun \
  --port 9888 \
  --window-size 1600x1000 \
  --renderer vulkan \
  outputs/aloha_mobile_cabinet_episode_0.rrd
```

当前机器强制使用 OpenGL 会报 surface/adapter 不兼容，因此使用 Vulkan。Rerun 中可以用
Space 播放或暂停、Left/Right 逐帧查看，也可以拖动底部 `frame_index` 时间轴。

重新从 dataset 构建可视化：

```bash
cd /home/mi/codes/practice
conda activate robot-practice
python -m robot_practice.visualize_aloha \
  --repo-id lerobot/aloha_mobile_cabinet \
  --episode-index 0
```

只检查信号和 3D、不解码视频：

```bash
python -m robot_practice.visualize_aloha \
  --repo-id lerobot/aloha_mobile_cabinet \
  --episode-index 0 \
  --no-images \
  --max-frames 100
```

详细几何约定和数据流位于原始 workspace：

```text
docs/superpowers/specs/2026-07-16-aloha-dataset-visualizer-design.md
```

## 验证结果

- `36` 个 pytest 测试通过；
- 零位姿、非零角度独立 MuJoCo golden pose、旋转正交性和夹爪映射均有覆盖；
- 覆盖 dataset feature/shape/joint order、timestamp、velocity 和 entity path 校验；
- 覆盖 `--no-images` 不触发 dataset 视频解码；
- 覆盖每帧两个 Rerun timeline 和末端完整轨迹的 static logging；
- 使用缓存中的三帧完成离线 `.rrd` 集成测试；
- 完整 1,500 帧 recording 已通过 `rerun rrd verify`；
- 人工检查确认三路相机、3D、曲线和 50 FPS 时间轴可以同步显示。

对应测试位于：

```text
/home/mi/codes/practice/tests/test_aloha_kinematics.py
/home/mi/codes/practice/tests/test_visualize_aloha.py
```

## 主要文件

| 文件 | 作用 |
|---|---|
| `.vscode/settings.json` | 固定 Python interpreter |
| `.vscode/launch.json` | ACT 三步训练 Debug 配置 |
| `README.md` | 增加 ALOHA 可视化用法 |
| `src/robot_practice/aloha_kinematics.py` | 双臂标称 FK |
| `src/robot_practice/visualize_aloha.py` | Rerun 可视化入口 |
| `tests/test_aloha_kinematics.py` | FK 单元测试 |
| `tests/test_visualize_aloha.py` | 数据与 Rerun 测试 |

`.vscode/`、`outputs/` 和数据缓存都是本机内容，按原始 workspace 规则不提交生成物。原始
workspace 没有 `.git` 目录，因此本次实践无法在那里创建 commit 或查看 Git diff。

## 下一步

1. 在 `cfg.validate()` 打断点，重新运行 `LeRobot: ACT train (3-step debug)`；数据已完整，
   这次应直接进入 dataset/policy 构建。
2. 依次进入 `make_dataset()`、`make_policy()` 和 `update_policy()`，记录实际 batch key、
   tensor shape、loss 和单步耗时。
3. 如果三步训练成功，将结果补充到原始 workspace 的 `reports/experiment_log.md`；缺失的
   closed-loop 指标保持明确未测。
4. 后续再决定是否保留两套 1.7 GB 数据。清理前先确认训练统一使用哪一个 root。

## 相关笔记

- [[Robot/VLA/ALOHA硬件与ACT算法|ALOHA 硬件与 ACT 算法]]：理解本次 dataset 的
  双臂硬件、14 维关节空间与 ACT action chunk。
- [[Robot/VLA/Diffusion Policy 概述|Diffusion Policy]]：与 ACT 连续动作序列建模进行
  对照。
- [[BestPractice/lerobot-libero-setup-and-smoke-test|LeRobot LIBERO 环境准备与 smoke test]]：
  同一 LeRobot practice workspace 的另一套数据和环境验证记录。
