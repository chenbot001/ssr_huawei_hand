# SSR–Huawei Pick-and-Place Pipeline

End-to-end workflow for the **UR5 + RyHand** dexterous pick-and-place task (Manus teleop, RealSense cameras, T265), from hardware checks through **diffusion policy** training.

**Repository layout (relevant paths):**

| Area | Path |
|------|------|
| Project root | `Huawei/` |
| Hardware / teleop Python | `src/` |
| Data (zarr sessions + merged dataset) | `data/` |
| Operational scripts | `scripts/` |
| System tests | `tests/` |
| Diffusion Policy (fork) | `external/diffusion_policy/` |

---

## 1. Environment

- Use a conda env with project dependencies (e.g. `ssr_hw` or your team env).
- For **training only**, install PyTorch and diffusion-policy extras (from `external/diffusion_policy`):

  ```bash
  pip install torch torchvision diffusers wandb hydra-core omegaconf threadpoolctl filelock einops robomimic
  ```

- Hardware scripts expect packages such as `opencv-python`, `numpy`, `ur_rtde`, `pyrealsense2`, `pyzmq` (see `scripts/system_check.py`).

---

## 2. System readiness check

### 2.1 Integrated startup check

Runs package checks, CAN (`ryhand`), UR5 RTDE, RyHand interface, T265 pose, Manus ZMQ, and configured cameras (from `src/ssr/config`).

```bash
cd /path/to/Huawei
python scripts/system_check.py
```

Exit code `0` means **SYSTEM READY**; otherwise fix failures (UR IP, CAN, RealSense indices, Manus stream, etc.).

### 2.2 Optional focused tests (`tests/`)

Run from project root (scripts add `src/` to `PYTHONPATH`):

| Script | Purpose |
|--------|---------|
| `python tests/test_ur_readiness.py` | UR robot / RTDE readiness |
| `python tests/test_realsense.py` | RealSense RGB streams (env / wrist) |
| `python tests/test_t265.py` | T265 pose stream |
| `python tests/test_manus_parsing.py` | Manus glove data |
| `python tests/test_usb_health.py` | USB stability |
| `python tests/test_data_integrity.py` | Validate **all** `data/*.zarr` (schema + per-episode checks) |

Adjust IPs and device IDs in your hardware config / script defaults as needed.

---

## 3. Data collection

### 3.1 Record demonstrations (zarr)

Single-session recorder (writes one `collected_*.zarr` under `data/` by default):

```bash
cd /path/to/Huawei
python scripts/collect_data.py -o data/collected_<session_name>.zarr
```

Common options (see `collect_data.py --help`):

- `--record-rate` — logging rate (Hz), default `15`
- `--control-rate` — control loop (Hz), default `80`
- `--img-width`, `--img-height` — default `320×240` (must match training `shape_meta`)
- `--dry-run` — no hardware, for plumbing tests

There is also a threaded variant: `scripts/collect_data_threaded.py` (same project; use if your setup requires it).

### 3.2 Dataset layout (per session)

Each session zarr contains (among others):

- **Observations:** `arm_eef_pose`, `hand_joint_angles`, `camera_env`, `camera_wrist`
- **Actions (raw):** `action_eef_delta`, `action_hand_joints`
- **Meta:** `meta/episode_ends`

---

## 4. Data preprocessing (merge + single action tensor)

Merge **raw session** zarr folders under `data/` (basenames starting with `collected`, e.g. `collected_*.zarr` from `collect_data.py`) into one replay-buffer-compatible store. Merged outputs like `ssr_pickplace_dataset.zarr` / `ryhand_dp_dataset.zarr` are **not** used as sources unless you pass `--all-zarr`. Actions are concatenated into a single `action` array **(21 = 6 EEF delta + 15 hand joints)**.

```bash
cd /path/to/Huawei
python scripts/preprocess_dataset.py
```

Default output:

`data/ssr_pickplace_dataset.zarr`

Useful overrides:

```bash
python scripts/preprocess_dataset.py --data-dir data --output data/ssr_pickplace_dataset.zarr
python scripts/preprocess_dataset.py --include "collected_20260322_*.zarr"
python scripts/preprocess_dataset.py --min-episode-len 30
python scripts/preprocess_dataset.py --all-zarr   # include every *.zarr (advanced)
```

The script validates the result with `diffusion_policy.common.replay_buffer.ReplayBuffer` if importable.

### 4.1 Integrity check on collected data

```bash
python tests/test_data_integrity.py
```

### 4.2 Summary statistics (optional)

```bash
python scripts/analyze_datasets.py --data-dir data
python scripts/analyze_datasets.py --data-dir data --all-zarr   # include merged datasets too
```

---

## 5. Diffusion Policy — config files

Configs live under `configs/dp/`.

| File | Role |
|------|------|
| **`configs/dp/dp_config.yaml`** | Top-level one-click configuration defining runtime overrides like `dataset_path` and `device`.
| **`configs/dp/task/ssr_pickplace_image.yaml`** | Task name: `ssr_pickplace_image`. Defines `shape_meta` (obs: `camera_env`, `camera_wrist`, `arm_eef_pose`, `hand_joint_angles`; action: 21D). Points dataset class to `SSRPickPlaceImageDataset`. |
| **`configs/dp/train_diffusion_unet_ssr_pickplace_fruit_workspace.yaml`** | Training workspace: hybrid UNet + image policy, horizons, DDIM scheduler, batch size, `logging.project: ssr_pickplace`, checkpointing on `train_loss`. |

**Dataset implementation:** `src/ssr/dataset/ssr_pickplace_image_dataset.py` — class `SSRPickPlaceImageDataset`.

**Env runner during training:** `RealPushTImageRunner` (no simulated env — returns empty metrics; appropriate for real-data-only training).

Override dataset path at runtime if the file lives elsewhere:

```text
task.dataset.dataset_path=/absolute/path/to/ssr_pickplace_dataset.zarr
```

---

## 6. Training

Training is executed via a custom top-level script, injecting the `external/diffusion_policy` dependency dynamically so the fork remains clean.

From the project root:

```bash
cd /path/to/Huawei
conda activate <your_env_with_torch>

wandb login   # optional

# One-click training using settings in configs/dp/dp_config.yaml
python scripts/train_dp.py
```

Hydra writes outputs under `external/diffusion_policy/data/outputs/<date>/...` (see `hydra.run.dir` in the workspace YAML). Checkpoints:

- `checkpoints/latest.ckpt`
- Top-k by `train_loss` per `checkpoint.topk`

**Tips:**

- Reduce `dataloader.batch_size` / `val_dataloader.batch_size` in the workspace YAML if you hit GPU OOM (two 240×320 RGB streams + state).
- Set `logging.mode=offline` or disable wandb if needed.

---

## 7. Running the policy (deployment)

Training produces a **workspace checkpoint** (`.ckpt`) containing the policy and normalizer.

### 7.1 Upstream reference (Push-T real robot)

The stock `external/diffusion_policy/eval_real_robot.py` script targets the **original Push-T** stack (SpaceMouse, `RealEnv`, specific camera layout). It is **not** wired to UR5 + RyHand + your observation/action schema.

### 7.2 Deployment script (`scripts/deploy_policy.py`)

The project includes **`scripts/deploy_policy.py`**, which loads a checkpoint (e.g. `checkpoints/best.ckpt`), builds observations like training, runs `policy.predict_action`, and sends **6D EEF deltas** + **15D hand joints** to the UR5 (`servoL`) and RyHand.

Default checkpoint path is **`deployment.checkpoint_path`** in `configs/dp/dp_config.yaml` (e.g. copy or symlink `best.ckpt` to `data/checkpoints/best.ckpt`). Override on the CLI like training:

```bash
cd /path/to/Huawei
python scripts/deploy_policy.py
python scripts/deploy_policy.py deployment.dry_run=true
python scripts/deploy_policy.py deployment.checkpoint_path=data/outputs/<date>/<run>/checkpoints/best.ckpt
```

OpenCV window: **SPACE** toggles policy on/off, **q** quits. Match `deployment.frequency` to your data collection rate (`collect_data.py` `--record-rate`).

For earlier sanity checks without a trained policy, use **`scripts/replay_data.py`** to replay recorded zarr on hardware:

```bash
python scripts/replay_data.py data/ssr_pickplace_dataset.zarr --episode 0
```

See `replay_data.py --help` for rate, speed, arm-only / hand-only, etc.

---

## 8. Quick reference — command order

```text
1. python scripts/system_check.py
2. python scripts/collect_data.py -o data/collected_<name>.zarr
  # repeat / multiple sessions
3. python scripts/preprocess_dataset.py
4. python tests/test_data_integrity.py   # optional on raw zarrs
5. python scripts/train_dp.py
6. python scripts/deploy_policy.py -c .../checkpoints/best.ckpt
7. (optional) python scripts/replay_data.py data/ssr_pickplace_dataset.zarr -e 0
```

---

## 9. Collaboration context

**Task:** Pick-and-place with **Universal Robots UR5** and **RyHand** dexterous hand, **SSR Lab** + **Huawei** collaboration.  
**Teleop / data:** Manus-driven retargeting and T265-based arm deltas as recorded in `collect_data.py`.  
**Policy:** Diffusion Policy (hybrid image + low-dim) in `external/diffusion_policy`.
