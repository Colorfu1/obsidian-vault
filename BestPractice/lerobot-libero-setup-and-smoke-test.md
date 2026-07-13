# LeRobot LIBERO 环境准备与数据/环境 smoke test

- Conda 环境：`robot-practice`
- 目标：安装 LeRobot 的 LIBERO extra，验证 EGL 无头渲染，下载并加载完整的
  `lerobot/libero` 数据集，然后对同一个语言任务完成数据与环境 smoke test。

## 1. 最终结论

本次准备工作已完成，最终状态如下：

- `lerobot[libero]` 已安装并写入 `requirements.txt`。
- Python 依赖检查、项目 doctor 和 LeRobot CLI quickstart 检查均通过。
- 在没有 `DISPLAY` 的情况下，MuJoCo 通过 EGL 成功完成双相机离屏渲染。
- LIBERO 环境可以 reset、render，并连续执行 3 个 no-op action。
- 完整的 `lerobot/libero` LeRobot v3 数据集已落盘，实际占用约 `1.9 GiB`
  （约等于页面标称的 `1.94 GB`）。
- 数据集可以通过 LeRobot 默认缓存路径完全离线加载。
- 环境任务和选中的数据 episode 使用完全相同的语言指令。

## 2. 环境和依赖

最终验证的主要版本：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| LeRobot | 0.4.4 |
| hf-libero | 0.1.4 |
| Robosuite | 1.4.0 |
| MuJoCo | 3.8.1 |
| PyTorch | 2.10.0+cu128 |
| GPU | 支持 EGL 的 NVIDIA GPU |
| egl-probe | 1.0.2 |
| hf-egl-probe | 1.0.2 |

在项目根目录进入环境，并用 `PROJECT_ROOT` 代替具体机器上的绝对路径：

```bash
cd /path/to/robot-practice
export PROJECT_ROOT="$PWD"
conda activate robot-practice
export PYTHONNOUSERSITE=1
```

这里必须使用 `robot-practice` 环境中的 Python，避免把 LIBERO 依赖装入 base 环境：

```bash
which python
python --version
```

项目依赖中的 LeRobot 条目由：

```text
lerobot
```

更新为：

```text
lerobot[libero]
```

正常安装方式：

```bash
python -m pip install --upgrade-strategy only-if-needed -r requirements.txt
```

安装后的核心依赖检查：

```bash
python -m pip check
./scripts/doctor.sh
./scripts/quickstart_info.sh
```

三项检查最终均通过，其中 `pip check` 返回 `No broken requirements found.`。

### 2.1 EGL probe 构建问题

首次安装 LIBERO extra 时，`hf-egl-probe` 的 PEP 517 build isolation 与当前环境中的
pip CMake wrapper 发生冲突。禁用 build isolation 后，又遇到 CMake 4.x 与该包旧版
`CMakeLists.txt` 的兼容问题。

最终使用系统 CMake 3.28.3 单独构建两个 probe 包，再继续安装 LIBERO extra：

```bash
export CMAKE_3_BIN=/path/to/cmake-3.x/bin

PATH="$CMAKE_3_BIN:$CONDA_PREFIX/bin:$PATH" \
  python -m pip install --no-build-isolation \
  "hf-egl-probe==1.0.2" "egl-probe==1.0.2"

python -m pip install --upgrade-strategy only-if-needed \
  "lerobot[libero]==0.4.4"
```

该 workaround 只在 probe 构建报错时需要。安装完成后再次运行
`python -m pip install -r requirements.txt` 已可正常结束。

Robosuite 的本地宏文件通过以下脚本初始化：

```bash
python -m robosuite.scripts.setup_macros
```

## 3. LIBERO 配置与仿真资产

首次导入 `libero.libero` 会创建：

```text
$HOME/.libero/config.yaml
```

如果需要非交互初始化默认配置，可以运行：

```bash
printf 'n\n' | python -c 'import libero.libero'
```

可以通过 API 查询当前环境实际使用的 BDDL、init state 和其他资源路径，无需在文档中
硬编码 Conda 的安装位置：

```bash
python - <<'PY'
from libero.libero import get_libero_path

for key in ("bddl_files", "init_states", "datasets", "assets"):
    print(f"{key}: {get_libero_path(key)}")
PY
```

`hf-libero` wheel 本身没有携带完整仿真资产。自动从 Hugging Face 下载资产时遇到 TLS
EOF，因此从官方 LIBERO GitHub 仓库的固定 commit 稀疏检出资产：

```text
repository: Lifelong-Robot-Learning/LIBERO
commit:     8f1084e3132a39270c3a13ebe37270a43ece2a01
source:     libero/libero/assets
target:     $HOME/.cache/libero/assets
```

对应的可复现命令为：

```bash
ASSET_TMP_DIR="$(mktemp -d)"
git clone --filter=blob:none --no-checkout \
  https://github.com/Lifelong-Robot-Learning/LIBERO.git \
  "$ASSET_TMP_DIR"
git -C "$ASSET_TMP_DIR" sparse-checkout init --cone
git -C "$ASSET_TMP_DIR" sparse-checkout set libero/libero/assets
git -C "$ASSET_TMP_DIR" checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01

mkdir -p "$HOME/.cache/libero/assets"
cp -a "$ASSET_TMP_DIR/libero/libero/assets/." "$HOME/.cache/libero/assets/"
rm -rf "$ASSET_TMP_DIR"
```

资产目录约占 `405 MiB`，包含：

- `articulated_objects`
- `scenes`
- `stable_hope_objects`
- `stable_scanned_objects`
- `textures`
- `turbosquid_objects`

## 4. EGL 无头渲染验证

EGL 验证使用以下环境变量。它们必须在导入 MuJoCo、Robosuite 或 LIBERO 之前设置：

```bash
unset DISPLAY
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONNOUSERSITE=1
```

验证环境为 `libero_spatial` 的 task 0，观测分辨率为 `128 x 128`，同时开启 agent
state。下面的脚本可以复现本次环境 smoke test：

```python
import numpy as np

from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action


suite_name = "libero_spatial"
suite = benchmark.get_benchmark_dict()[suite_name]()
env = LiberoEnv(
    task_suite=suite,
    task_id=0,
    task_suite_name=suite_name,
    obs_type="pixels_agent_pos",
    observation_width=128,
    observation_height=128,
)

try:
    observation, info = env.reset(seed=0)
    assert observation["pixels"]["image"].shape == (128, 128, 3)
    assert observation["pixels"]["image2"].shape == (128, 128, 3)
    assert env.render().shape == (128, 128, 3)

    action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
    for _ in range(3):
        observation, reward, terminated, truncated, info = env.step(action)

    print(suite.get_task(0).language)
    print("egl_env_smoke=ok")
finally:
    env.close()
```

实际验证结果：

- `DISPLAY=None`
- 主相机图像：`(128, 128, 3)`、`uint8`
- 腕部相机图像：`(128, 128, 3)`、`uint8`
- `render()`：`(128, 128, 3)`、`uint8`
- EEF position：`(3,)`
- 3 个 no-op step 均正常完成
- reward 为 0，`terminated=False`，`truncated=False`
- `egl_env_smoke=ok`

no-op 没有完成操作任务是预期结果；这里验证的是环境初始化、GPU EGL 渲染和 step
链路，不是策略成功率。

## 5. `lerobot/libero` 数据集准备

Hugging Face Python 客户端下载时出现 TLS EOF。Git 和 Git LFS 链路可用，因此使用数据集
仓库的 `v3.0` tag 下载：

```text
repository: https://huggingface.co/datasets/lerobot/libero
tag:        v3.0
commit:     a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
```

推荐落盘位置遵循本项目的目录约定：

```text
$PROJECT_ROOT/datasets/lerobot/lerobot/libero
```

下载与完整性检查流程：

```bash
export LIBERO_DATASET_DIR="$PROJECT_ROOT/datasets/lerobot/lerobot/libero"

mkdir -p "$(dirname "$LIBERO_DATASET_DIR")"
git clone --branch v3.0 --depth 1 \
  https://huggingface.co/datasets/lerobot/libero \
  "$LIBERO_DATASET_DIR"

git -C "$LIBERO_DATASET_DIR" lfs pull
git -C "$LIBERO_DATASET_DIR" lfs fsck
```

本次共获取 453 个 Git LFS 对象。`git lfs fsck` 通过，并确认工作区中没有残留 LFS
pointer。完整性确认后删除下载仓库的 `.git`，避免 LFS object cache 与工作文件各保存一份、
造成磁盘占用翻倍：

```bash
rm -rf "$LIBERO_DATASET_DIR/.git"
```

上述目录已被项目 `.gitignore` 忽略，不会误提交大数据文件。

为了让 `LeRobotDataset("lerobot/libero")` 使用默认 root 时直接找到项目内的数据，建立了
缓存软链接：

```bash
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

mkdir -p "$HF_LEROBOT_HOME/lerobot"
ln -sfn "$LIBERO_DATASET_DIR" "$HF_LEROBOT_HOME/lerobot/libero"
```

当前数据集统计：

| 项目 | 数值 |
| --- | ---: |
| LeRobot 数据格式 | v3 |
| 磁盘占用 | 约 1.9 GiB |
| 任务数 | 40 |
| episode 数 | 1693 |
| 总帧数 | 273465 |
| 数据文件数 | 377 |
| 视频文件数 | 74 |
| 数据频率 | 10 FPS |

## 6. 同任务数据 smoke test

环境选用的语言任务为：

```text
pick up the black bowl between the plate and the ramekin and place it on the plate
```

该任务在完整数据集中的 task index 为 34，共有 45 个 episode。本次读取第一个匹配的
episode：

```text
episode index: 1272
frame count:   84
```

离线加载验证脚本：

```bash
export HF_HUB_OFFLINE=1

python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/libero", episodes=[1272])
sample = dataset[0]

assert len(dataset) == 84
assert sample["task"] == (
    "pick up the black bowl between the plate and the ramekin "
    "and place it on the plate"
)
assert sample["observation.images.image"].shape == (3, 256, 256)
assert sample["observation.images.image2"].shape == (3, 256, 256)
assert sample["observation.state"].shape == (8,)
assert sample["action"].shape == (7,)

print(f"dataset_root={dataset.root}")
print("offline_default_root_smoke=ok")
PY
```

实际结果：

```text
dataset_root=<HF_LEROBOT_HOME>/lerobot/libero
selected_episode_frames=84
image_keys=['observation.images.image', 'observation.images.image2']
observation.images.image_shape=(3, 256, 256)
observation.images.image2_shape=(3, 256, 256)
state_shape=(8,)
action_shape=(7,)
offline_default_root_smoke=ok
```

这同时验证了：

- 默认 LeRobot cache root 软链接有效。
- 无网络模式下可以读取 metadata、Parquet 数据和视频帧。
- 两路图像可被正常解码为有限的 `float32` tensor。
- 数据中的语言任务与环境中的语言任务精确匹配。

## 7. 实验记录字段

本次属于安装和链路 smoke test，不包含训练或策略评测。为满足项目实验记录规范，字段记录
如下：

| 字段 | 本次记录 |
| --- | --- |
| task suite | `libero_spatial`，task 0；数据集全量包含 40 个任务 |
| dataset size | 全量约 1.9 GiB、1693 episodes、273465 frames；smoke episode 1272 为 84 frames |
| observation keys | 环境：`pixels.image`、`pixels.image2`、`robot_state`；数据：`observation.images.image`、`observation.images.image2`、`observation.state` |
| action space | 7 维、范围 `[-1, 1]`，环境使用 relative control |
| action horizon | `libero_spatial` 默认最大 280 steps；本次只执行 3 steps |
| control frequency | 数据集 metadata 为 10 FPS；smoke 未做实时 pacing |
| train loss | N/A，未训练 |
| open-loop action error | N/A，未做 policy inference |
| closed-loop success rate | N/A，未评测策略；no-op 的 success 为 false |
| latency | N/A，本次未做基准测试 |
| failure cases | 安装阶段遇到 EGL probe/CMake 构建兼容问题；HF Python 下载遇到 TLS EOF；均已通过固定构建工具链和 Git/LFS 路径解决 |

## 8. 目录约定

```text
requirements:  $PROJECT_ROOT/requirements.txt
dataset:       $LIBERO_DATASET_DIR
dataset link:  $HF_LEROBOT_HOME/lerobot/libero
LIBERO config: $HOME/.libero/config.yaml
LIBERO assets: $HOME/.cache/libero/assets
```

数据集、仿真资产、视频和其他生成物应由项目的 `.gitignore` 排除，不加入 Git。
