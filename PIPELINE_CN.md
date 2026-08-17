# SSR–华为 抓放流水线

端到端工作流：**UR5 + RyHand** 灵巧手抓放任务（Manus 遥操作、RealSense 相机、T265），从硬件检查到**扩散策略**训练。

**项目目录结构（相关路径）：**

| 类别 | 路径 |
|------|------|
| 项目根目录 | `Huawei/` |
| 硬件 / 遥操作 Python 代码 | `src/` |
| 数据（zarr 会话 + 合并数据集） | `data/` |
| 操作脚本 | `scripts/` |
| 系统测试 | `tests/` |
| Diffusion Policy（clone） | `external/diffusion_policy/` |

---

## 1. 环境配置

- 使用包含项目依赖的 conda 环境（如 `ssr_hw` 或团队环境）。
- 如果**仅用于训练**，需安装 PyTorch 和 diffusion-policy 相关依赖（来自 `external/diffusion_policy`）：

  ```bash
  pip install torch torchvision diffusers wandb hydra-core omegaconf threadpoolctl filelock einops robomimic
  ```

- 硬件脚本需要 `opencv-python`、`numpy`、`ur_rtde`、`pyrealsense2`、`pyzmq` 等包（详见 `scripts/system_check.py`）。

---

## 2. 系统就绪检查

### 2.1 集成启动检查

运行包检查、CAN（`ryhand`）、UR5 RTDE、RyHand 接口、T265 位姿、Manus ZMQ 以及已配置的相机（来自 `src/ssr/config`）。

```bash
cd /path/to/Huawei
python scripts/system_check.py
```

退出码 `0` 表示**系统就绪**；否则请修复故障（UR IP、CAN、RealSense 索引、Manus 数据流等）。

### 2.2 可选的单项测试（`tests/`）

从项目根目录运行（脚本会自动将 `src/` 加入 `PYTHONPATH`）：

| 脚本 | 用途 |
|------|------|
| `python tests/test_ur_readiness.py` | UR 机器人 / RTDE 就绪检查 |
| `python tests/test_realsense.py` | RealSense RGB 视频流（环境 / 腕部） |
| `python tests/test_t265.py` | T265 位姿数据流 |
| `python tests/test_manus_parsing.py` | Manus 手套数据 |
| `python tests/test_usb_health.py` | USB 稳定性 |
| `python tests/test_data_integrity.py` | 验证**所有** `data/*.zarr`（结构 + 逐集检查） |

根据需要调整硬件配置中的 IP 地址和设备 ID。

---

## 3. 数据采集

### 3.1 记录演示数据（zarr）

单次会话录制器（默认将 `collected_*.zarr` 写入 `data/` 目录）：

```bash
cd /path/to/Huawei
python scripts/collect_data.py -o data/collected_<session_name>.zarr
```

常用选项（详见 `collect_data.py --help`）：

- `--record-rate` — 记录频率（Hz），默认 `15`
- `--control-rate` — 控制循环频率（Hz），默认 `80`
- `--img-width`、`--img-height` — 默认 `320×240`（必须与训练 `shape_meta` 一致）
- `--dry-run` — 无硬件模式，用于流程测试

也有多线程变体：`scripts/collect_data_threaded.py`（同一项目；如有需要可使用）。

### 3.2 数据集结构（每个会话）

每个会话 zarr 包含（等等）：

- **观测：** `arm_eef_pose`、`hand_joint_angles`、`camera_env`、`camera_wrist`
- **动作（原始）：** `action_eef_delta`、`action_hand_joints`
- **元数据：** `meta/episode_ends`

---

## 4. 数据预处理（合并 + 统一动作张量）

默认只合并**原始会话** zarr（`data/` 下 basename 以 `collected` 开头，如 `collect_data.py` 产出的 `collected_*.zarr`）。已合并的数据集（如 `ssr_pickplace_dataset.zarr`、`ryhand_dp_dataset.zarr`）**不会**作为源，除非使用 `--all-zarr`。动作**拼接**为单一 `action` 数组**（21 = 6 EEF delta + 15 手指关节）**。

```bash
cd /path/to/Huawei
python scripts/preprocess_dataset.py
```

默认输出：

`data/ssr_pickplace_dataset.zarr`

常用覆盖参数：

```bash
python scripts/preprocess_dataset.py --data-dir data --output data/ssr_pickplace_dataset.zarr
python scripts/preprocess_dataset.py --include "collected_20260322_*.zarr"
python scripts/preprocess_dataset.py --min-episode-len 30
python scripts/preprocess_dataset.py --all-zarr   # 包含 data/ 下全部 *.zarr（高级用法）
```

脚本会使用 `diffusion_policy.common.replay_buffer.ReplayBuffer`（如果可导入）来验证结果。

### 4.1 采集数据完整性检查

```bash
python tests/test_data_integrity.py
```

### 4.2 数据集统计摘要（可选）

```bash
python scripts/analyze_datasets.py --data-dir data
python scripts/analyze_datasets.py --data-dir data --all-zarr   # 同时统计已合并的数据集
```

---

## 5. Diffusion Policy — 配置文件

配置文件位于 `configs/dp/` 目录。

| 文件 | 作用 |
|------|------|
| **`configs/dp/config/dp_config.yaml`** | 顶层一键配置，定义运行时覆盖参数如 `dataset_path` 和 `device`。 |
| **`configs/dp/task/ssr_pickplace_image.yaml`** | 任务名：`ssr_pickplace_image`。定义 `shape_meta`（观测：`camera_env`、`camera_wrist`、`arm_eef_pose`、`hand_joint_angles`；动作：21D）。将数据集类指向 `SSRPickPlaceImageDataset`。 |
| **`configs/dp/train_diffusion_unet_ssr_pickplace_fruit_workspace.yaml`** | 训练工作空间：混合 UNet + 图像策略、时域参数、DDIM 调度器、批量大小、`logging.project: ssr_pickplace`、基于 `train_loss` 的检查点保存。 |

**数据集实现：** `src/ssr/dataset/ssr_pickplace_image_dataset.py` — 类 `SSRPickPlaceImageDataset`。

**训练期间的环境运行器：** `RealPushTImageRunner`（无仿真环境 — 返回空指标；适用于纯真实数据训练）。

如果数据文件在其他位置，可在运行时覆盖数据集路径：

```text
task.dataset.dataset_path=/absolute/path/to/ssr_pickplace_dataset.zarr
```

---

## 6. 训练

训练通过自定义顶层脚本执行，动态注入 `external/diffusion_policy` 依赖以保持 fork 代码干净。

从项目根目录执行：

```bash
cd /path/to/Huawei
conda activate <your_env_with_torch>

wandb login   # 可选

# 使用 configs/dp/dp_config.yaml 中的设置一键训练
python scripts/train_dp.py
```

Hydra 将输出写入 `external/diffusion_policy/data/outputs/<date>/...`（见工作空间 YAML 中的 `hydra.run.dir`）。检查点：

- `checkpoints/latest.ckpt`
- 按 `train_loss` 排名的 Top-k 检查点（见 `checkpoint.topk`）

**提示：**

- 如果 GPU 显存不足（两个 240×320 RGB 流 + 状态），请在工作空间 YAML 中减小 `dataloader.batch_size` / `val_dataloader.batch_size`。
- 如需要，可设置 `logging.mode=offline` 或禁用 wandb。

---

## 7. 策略部署（推理）

训练产出一个**工作空间检查点**（`.ckpt`），包含策略和归一化器。

### 7.1 上游参考（Push-T 真实机器人）

原始 `external/diffusion_policy/eval_real_robot.py` 脚本面向**原版 Push-T** 系统（SpaceMouse、`RealEnv`、特定相机布局）。它**未**连接到 UR5 + RyHand + 本项目的观测/动作格式。

### 7.2 部署脚本（`scripts/deploy_policy.py`）

项目提供 **`scripts/deploy_policy.py`**：从检查点（如 `best.ckpt`）加载策略，按训练格式构建观测并调用 `policy.predict_action`，向 UR5（`servoL`）与 RyHand 下发 **6D 末端增量** 与 **15 维手部关节角**。

默认检查点路径在 **`configs/dp/dp_config.yaml`** 的 **`deployment.checkpoint_path`**（可将 `best.ckpt` 复制或软链到 `data/checkpoints/best.ckpt`）。命令行可覆盖，用法与训练脚本一致：

```bash
cd /path/to/Huawei
python scripts/deploy_policy.py
python scripts/deploy_policy.py deployment.dry_run=true
python scripts/deploy_policy.py deployment.checkpoint_path=data/outputs/<日期>/<运行>/checkpoints/best.ckpt
```

OpenCV 窗口：**空格** 启停策略，**q** 退出。将 `deployment.frequency` 与采集时的 `--record-rate` 对齐。

在尚无训练策略时，可用 **`scripts/replay_data.py`** 在硬件上回放 zarr 做健全性检查：

```bash
python scripts/replay_data.py data/ssr_pickplace_dataset.zarr --episode 0
```

详见 `replay_data.py --help`（速率、倍速、仅手臂/仅手部等选项）。

---

## 8. 快速参考 — 命令执行顺序

```text
1. python scripts/system_check.py
2. python scripts/collect_data.py -o data/collected_<name>.zarr
  # 重复 / 多次会话
3. python scripts/preprocess_dataset.py
4. python tests/test_data_integrity.py   # 可选，检查原始 zarr
5. python scripts/train_dp.py
6. python scripts/deploy_policy.py
7. （可选）python scripts/replay_data.py data/ssr_pickplace_dataset.zarr -e 0
```

---

## 9. 合作背景

**任务：** 使用 **Universal Robots UR5** 和 **RyHand** 灵巧手进行抓放操作，**SSR 实验室**与**华为**合作项目。
**遥操作 / 数据：** 由 Manus 驱动的重定向和基于 T265 的手臂增量，如 `collect_data.py` 中所记录。
**策略：** Diffusion Policy（混合图像 + 低维）在 `external/diffusion_policy` 中。
