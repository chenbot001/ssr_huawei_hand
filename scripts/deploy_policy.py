#!/usr/bin/env python3
"""
Deploy a trained diffusion policy (hybrid image + low-dim) on the real UR5 + RyHand stack.

Loads a workspace checkpoint (e.g. checkpoints/best.ckpt) produced by train_diffusion_unet_hybrid,
runs inference with the same observation layout as training (see configs/dp/task/ssr_pickplace_image.yaml),
and commands the robot.

Action layout (21D), matching preprocess_dataset.py:
  [:6]  — EEF delta: body-frame relative transform from previous TCP pose to next,
          same convention as collect_data.py (delta = inv(prev) @ curr, pose-vector form).
  [6:21] — Absolute hand joint angles (15D, radians).

Usage (from project root):
  python scripts/deploy_policy.py
  python scripts/deploy_policy.py deployment.dry_run=true
  python scripts/deploy_policy.py deployment.checkpoint_path=/path/to/best.ckpt

Checkpoint path defaults to ``deployment.checkpoint_path`` in ``configs/dp/dp_config.yaml``
(copy or symlink ``best.ckpt`` there, or override on the command line).

Safety: keep the teach pendant / E-stop ready. OpenCV window **Deploy** (same three-panel
layout as ``replay_data.py``: WRIST | ENV | stats): **SPACE** run/stop policy, **q** quit,
**r** stop and reset arm+hand to the mean dataset start pose (computed from ``DATASET_PATH``).
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch
import dill
import hydra
import zarr
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Paths (match scripts/train_dp.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_PATH = str(PROJECT_ROOT / "src")
DP_PATH = str(PROJECT_ROOT / "external" / "diffusion_policy")

USE_CONFIG_NAME = " "

# Zarr dataset used to compute the mean start pose (arm EEF + hand angles)
DATASET_PATH = " "

for _p in (SRC_PATH, str(PROJECT_ROOT), DP_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OmegaConf.register_new_resolver("eval", eval, replace=True)

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from ssr.hardware.arm_ur5 import UR5Arm
from ssr.hardware.ruiyan_driver import RyHandController
from ssr.hardware.realsense_worker import RealSenseWorker
from ssr.config import get_hardware_config, get_teleop_config
from ssr.utils.camera_utils import get_video_index_by_id, find_rgb_video_index_for_usb


# ---------------------------------------------------------------------------
# Pose helpers (match collect_data.py)
# ---------------------------------------------------------------------------
def matrix_to_pose_vector(matrix: np.ndarray) -> list:
    pos = matrix[:3, 3]
    rot = R.from_matrix(matrix[:3, :3]).as_rotvec()
    return [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]]


def pose_vector_to_matrix(pose_vec: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = pose_vec[:3]
    m[:3, :3] = R.from_rotvec(pose_vec[3:]).as_matrix()
    return m


def apply_eef_delta(prev_pose_vec: np.ndarray, delta_6d: np.ndarray) -> np.ndarray:
    """Apply one training-style delta: curr = prev @ delta_mat (see collect_data recording)."""
    prev_m = pose_vector_to_matrix(prev_pose_vec.astype(np.float64))
    delta_m = pose_vector_to_matrix(delta_6d.astype(np.float64))
    target_m = prev_m @ delta_m
    return np.array(matrix_to_pose_vector(target_m), dtype=np.float32)


# ---------------------------------------------------------------------------
# Camera init (aligned with collect_data.py)
# ---------------------------------------------------------------------------
def init_realsense_cameras(hw_config: dict, img_w: int, img_h: int):
    rs_configs = hw_config.get("cameras", {}).get("realsense", [])
    rs_by_name = {cfg.get("name", ""): cfg for cfg in rs_configs}

    def _one(role: str, cfg):
        if cfg is None:
            print(f"[deploy] Missing camera config for {role}")
            return None
        cam_id = cfg.get("id", "")
        cam_offset = cfg.get("offset", 0)
        cam_zoom = cfg.get("zoom", 1.0)
        serial = (cfg.get("serial") or cfg.get("serial_number") or "").strip()

        if serial:
            try:
                worker = RealSenseWorker(
                    width=img_w, height=img_h, serial_number=serial
                )
            except (ValueError, ImportError) as e:
                print(f"[deploy] RealSense SDK init failed ({role}): {e}")
                return None
            worker.set_zoom(cam_zoom)
            worker.daemon = True
            worker.start()
            for _ in range(50):
                if worker.get_latest_frame() is not None:
                    break
                time.sleep(0.1)
            if worker.get_latest_frame() is not None:
                print(f"[deploy] {role} camera OK (serial={serial})")
                return worker
            worker.stop()
            return None

        video_idx = get_video_index_by_id(cam_id, cam_offset)
        if video_idx is None and cam_id:
            video_idx, _ = find_rgb_video_index_for_usb(cam_id, img_w, img_h)
        if video_idx is None:
            print(f"[deploy] Cannot find video device for {role}")
            return None

        worker = RealSenseWorker(camera_index=video_idx, width=img_w, height=img_h)
        worker.set_zoom(cam_zoom)
        worker.daemon = True
        worker.start()
        for _ in range(50):
            if worker.get_latest_frame() is not None:
                break
            time.sleep(0.1)
        if worker.get_latest_frame() is not None:
            print(f"[deploy] {role} camera OK (video{video_idx})")
            return worker
        worker.stop()
        return None

    env_cam = _one("Env", rs_by_name.get("rs_env"))
    wrist_cam = _one("Wrist", rs_by_name.get("rs_wrist"))
    return env_cam, wrist_cam


def grab_camera_pair(rs_env, rs_wrist, img_h: int, img_w: int):
    """Return (camera_env, camera_wrist) RGB uint8 (H,W,3), same as collect_data."""
    env_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    wrist_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    if rs_env is not None:
        frame = rs_env.get_latest_frame()
        if frame is not None:
            if frame.shape[0] != img_h or frame.shape[1] != img_w:
                frame = cv2.resize(frame, (img_w, img_h))
            env_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if rs_wrist is not None:
        frame = rs_wrist.get_latest_frame()
        if frame is not None:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            if frame.shape[0] != img_h or frame.shape[1] != img_w:
                frame = cv2.resize(frame, (img_w, img_h))
            wrist_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return env_img, wrist_img


def build_deploy_ui_canvas(
    env_rgb_hwc: np.ndarray,
    wrist_rgb_hwc: np.ndarray,
    *,
    arm_pose: np.ndarray,
    hand_angles_rad: np.ndarray,
    running: bool,
    can_reset: bool,
    max_duration: float,
    t_run_start: float,
    now: float,
    frequency_hz: float,
    dry_run: bool,
    panel_w: int = 400,
    panel_h: int = 360,
) -> np.ndarray:
    """Three-column BGR canvas: Wrist | Env | Stats (layout aligned with replay_data.py)."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    def _panel_from_rgb(rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if bgr.shape[0] != panel_h or bgr.shape[1] != panel_w:
            bgr = cv2.resize(bgr, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
        return bgr

    wrist_panel = _panel_from_rgb(wrist_rgb_hwc)
    env_panel = _panel_from_rgb(env_rgb_hwc)

    stats_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    y0 = 30
    dy = 18

    def _put(img: np.ndarray, text: str, y: int, color=(220, 220, 220), scale=0.50, thick=1):
        cv2.putText(img, text, (12, y), font, scale, color, thick)

    mode = "POLICY" if running else "idle"
    mode_color = (0, 255, 0) if running else (160, 160, 160)
    _put(stats_panel, f"Deploy: {mode.upper()}", y0, color=mode_color, scale=0.60, thick=2)
    keys = "SPACE=run/stop | q=quit"
    if can_reset:
        keys += " | r=reset"
    _put(stats_panel, keys, y0 + dy, scale=0.42, thick=1)
    _put(
        stats_panel,
        f"Rate: {frequency_hz:.1f} Hz  |  max: {max_duration:.0f}s",
        y0 + dy * 2,
        scale=0.48,
        thick=1,
    )
    if running:
        elapsed = max(0.0, now - t_run_start)
        _put(
            stats_panel,
            f"Elapsed: {elapsed:.1f} / {max_duration:.1f} s",
            y0 + dy * 3,
            color=(0, 255, 200),
            scale=0.52,
            thick=1,
        )
    else:
        _put(stats_panel, "Elapsed: —", y0 + dy * 3, color=(120, 120, 120), scale=0.50, thick=1)

    if dry_run:
        _put(stats_panel, "DRY RUN (no robot/cams)", y0 + dy * 4, color=(0, 165, 255), scale=0.48, thick=1)

    line_y = y0 + dy * (5 if dry_run else 4) + 8
    cv2.line(stats_panel, (10, line_y), (panel_w - 10, line_y), (60, 60, 60), 1)
    cy = line_y + dy

    ap = arm_pose
    _put(stats_panel, "Arm EEF Pose:", cy, color=(180, 220, 255), scale=0.52, thick=1)
    cy += dy
    _put(stats_panel, f"  pos  [{ap[0]:+.4f}, {ap[1]:+.4f}, {ap[2]:+.4f}]", cy)
    cy += dy
    _put(stats_panel, f"  rot  [{ap[3]:+.4f}, {ap[4]:+.4f}, {ap[5]:+.4f}]", cy)
    cy += dy + 4

    hj = np.rad2deg(hand_angles_rad)
    _put(stats_panel, "Hand Joints (deg):", cy, color=(180, 220, 255), scale=0.52, thick=1)
    cy += dy
    _put(stats_panel, f"  Thumb  [{hj[0]:+5.1f},{hj[1]:+5.1f},{hj[2]:+5.1f}]", cy)
    cy += dy
    _put(stats_panel, f"  Index  [{hj[3]:+5.1f},{hj[4]:+5.1f},{hj[5]:+5.1f}]", cy)
    cy += dy
    _put(stats_panel, f"  Mid    [{hj[6]:+5.1f},{hj[7]:+5.1f},{hj[8]:+5.1f}]", cy)
    cy += dy
    _put(stats_panel, f"  Ring   [{hj[9]:+5.1f},{hj[10]:+5.1f},{hj[11]:+5.1f}]", cy)
    cy += dy
    _put(stats_panel, f"  Pinky  [{hj[12]:+5.1f},{hj[13]:+5.1f},{hj[14]:+5.1f}]", cy)

    cv2.putText(wrist_panel, "WRIST", (10, 25), font, 0.65, (0, 200, 255), 2)
    cv2.putText(env_panel, "ENV", (10, 25), font, 0.65, (0, 200, 255), 2)

    canvas = np.hstack([wrist_panel, env_panel, stats_panel])

    total_w = canvas.shape[1]
    bar_y = canvas.shape[0] - 12
    progress = 0.0
    if running and max_duration > 1e-6:
        progress = min(1.0, max(0.0, (now - t_run_start) / max_duration))
    cv2.rectangle(canvas, (8, bar_y), (total_w - 8, bar_y + 6), (60, 60, 60), -1)
    cv2.rectangle(
        canvas,
        (8, bar_y),
        (8 + int((total_w - 16) * progress), bar_y + 6),
        (0, 200, 0),
        -1,
    )
    return canvas


def images_to_model_input(env_img: np.ndarray, wrist_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H,W,C) uint8 -> (C,H,W) float32 [0,1]"""
    e = np.moveaxis(env_img.astype(np.float32), -1, 0) / 255.0
    w = np.moveaxis(wrist_img.astype(np.float32), -1, 0) / 255.0
    return e, w


def load_policy(ckpt_path: str, device: torch.device) -> tuple[BaseImagePolicy, OmegaConf]:
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.eval()
    policy.to(device)

    if "diffusion" not in cfg.name:
        raise RuntimeError(
            f"Checkpoint name '{cfg.name}' is not a diffusion workspace; "
            "this script only supports hybrid diffusion policies."
        )

    return policy, cfg


def capture_observation_sample(
    ur_arm,
    hand_ctrl,
    rs_env,
    rs_wrist,
    img_h: int,
    img_w: int,
    dry_run: bool,
) -> dict:
    """One timestep: raw values matching dataset (before batching)."""
    arm_pose = np.zeros(6, dtype=np.float32)
    if ur_arm is not None and not dry_run:
        try:
            arm_pose = np.array(ur_arm.rtde_r.getActualTCPPose(), dtype=np.float32)
        except Exception:
            pass

    hand_angles = np.zeros(15, dtype=np.float32)
    if hand_ctrl is not None and not dry_run:
        try:
            hand_angles = hand_ctrl.get_angles(radians=True).astype(np.float32)
        except Exception:
            pass

    env_hwc, wrist_hwc = grab_camera_pair(rs_env, rs_wrist, img_h, img_w)
    env_chw, wrist_chw = images_to_model_input(env_hwc, wrist_hwc)

    return {
        "camera_env": env_chw,
        "camera_wrist": wrist_chw,
        "arm_eef_pose": arm_pose,
        "hand_joint_angles": hand_angles,
        "_vis_env": env_hwc,
        "_vis_wrist": wrist_hwc,
    }


def stack_obs_for_policy(
    obs_buffer: deque,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """obs_buffer entries from capture_observation_sample -> dict for predict_action."""
    stack = list(obs_buffer)

    ce = np.stack([s["camera_env"] for s in stack], axis=0)
    cw = np.stack([s["camera_wrist"] for s in stack], axis=0)
    arm = np.stack([s["arm_eef_pose"] for s in stack], axis=0)
    hand = np.stack([s["hand_joint_angles"] for s in stack], axis=0)

    obs_dict = {
        "camera_env": torch.from_numpy(ce).unsqueeze(0).to(device=device, dtype=torch.float32),
        "camera_wrist": torch.from_numpy(cw).unsqueeze(0).to(device=device, dtype=torch.float32),
        "arm_eef_pose": torch.from_numpy(arm).unsqueeze(0).to(device=device, dtype=torch.float32),
        "hand_joint_angles": torch.from_numpy(hand).unsqueeze(0).to(device=device, dtype=torch.float32),
    }
    return obs_dict


def compute_mean_start_pose(zarr_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean arm EEF pose and hand joint angles from the first frame of every episode."""
    root = zarr.open(zarr_path, mode="r")
    ends = root["meta/episode_ends"][:]
    start_indices = [0] + [int(i) for i in ends[:-1]]
    arm_data = root["data/arm_eef_pose"]
    hand_data = root["data/hand_joint_angles"]
    arm_start = np.median([arm_data[i] for i in start_indices], axis=0).astype(np.float32)
    hand_start = np.median([hand_data[i] for i in start_indices], axis=0).astype(np.float32)
    print(f"[deploy] Computed start pose from {zarr_path}")
    print(f"  Arm EEF: {arm_start}")
    print(f"  Hand:    {hand_start}")
    return arm_start, hand_start


def move_to_start_pose(ur_arm, hand_ctrl, arm_eef_pose: np.ndarray, hand_pose: np.ndarray,
                       hand_reset_speed: int,
                       servo_speed: float = 0.5, servo_accel: float = 0.5,
                       servo_dt: float = 0.002, servo_lookahead: float = 0.1,
                       servo_gain: int = 300):
    """Move arm to EEF pose (Cartesian) and hand to joint angles, both from dataset mean start pose."""
    if ur_arm is not None:
        print("[deploy] Moving arm to mean-start EEF pose via moveL ...")
        ur_arm.rtde_c.moveL(arm_eef_pose.tolist(), speed=servo_speed, acceleration=servo_accel)
        # moveL blocks until the robot reaches the target
    if hand_ctrl is not None:
        print("[deploy] Moving hand to mean-start pose ...")
        hand_ctrl.set_angles(hand_pose, speed=hand_reset_speed, radians=True)
        time.sleep(0.3)


def _resolve_checkpoint_path(path_str: str) -> str:
    p = pathlib.Path(path_str).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


@hydra.main(version_base=None, config_path="../configs/dp", config_name=USE_CONFIG_NAME)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    os.chdir(str(PROJECT_ROOT))

    dep = cfg.deployment
    ckpt_path = _resolve_checkpoint_path(dep.checkpoint_path)
    if not os.path.isfile(ckpt_path):
        print(f"[deploy] ERROR: checkpoint not found: {ckpt_path}")
        print("  Set deployment.checkpoint_path in configs/dp/dp_config.yaml or pass e.g.")
        print("  python scripts/deploy_policy.py deployment.checkpoint_path=/path/to/best.ckpt")
        sys.exit(1)

    device = torch.device(dep.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[deploy] CUDA not available, using CPU (slow).")
        device = torch.device("cpu")

    dry_run = bool(dep.dry_run)
    no_start_pose = bool(dep.no_start_pose)
    frequency = float(dep.frequency)
    steps_per_inference = int(dep.steps_per_inference)
    max_duration = float(dep.max_duration)

    print(f"[deploy] checkpoint: {ckpt_path}")

    hw_config = get_hardware_config()
    teleop_config = get_teleop_config()
    servo_cfg = teleop_config.get("servo", {})
    control_cfg = teleop_config.get("control", {})
    hand_motor_speed = control_cfg.get("hand_motor_speed", 1000)
    hand_reset_speed = control_cfg.get("hand_reset_speed", 500)

    SERVO_SPEED = servo_cfg.get("speed", 0.5)
    SERVO_ACCEL = servo_cfg.get("acceleration", 0.5)
    SERVO_DT = servo_cfg.get("dt", 0.002)
    SERVO_LOOKAHEAD = servo_cfg.get("lookahead_time", 0.1)
    SERVO_GAIN = servo_cfg.get("gain", 300)

    policy, policy_cfg = load_policy(ckpt_path, device)

    n_obs = policy_cfg.n_obs_steps
    n_action_steps = int(policy.n_action_steps)
    shape_meta = policy_cfg.task.shape_meta
    img_shape = shape_meta["obs"]["camera_env"]["shape"]
    _, img_h, img_w = img_shape

    steps_pi = max(1, min(steps_per_inference, n_action_steps))
    dt = 1.0 / max(frequency, 1e-3)

    print(f"[deploy] n_obs_steps={n_obs}, n_action_steps={n_action_steps}, "
          f"steps_per_inference={steps_pi}, img={img_h}x{img_w}")

    # Compute mean start pose from dataset — overrides teleop_config start_pose
    dataset_path = str(PROJECT_ROOT / DATASET_PATH)
    if not os.path.isdir(dataset_path):
        print(f"[deploy] ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    arm_start_eef, hand_start_pose = compute_mean_start_pose(dataset_path)

    can_reset_to_start = (not no_start_pose) and (not dry_run)
    if not can_reset_to_start and not dry_run:
        if no_start_pose:
            print("[deploy] Start-pose reset (key 'r') disabled: deployment.no_start_pose=true")

    ur_arm = None
    hand_ctrl = None
    rs_env = None
    rs_wrist = None

    if not dry_run:
        try:
            ur_arm = UR5Arm(ip=hw_config["ur_arm"]["ip"])
            print("[deploy] UR5 connected")
        except Exception as e:
            print(f"[deploy] UR5 failed: {e}")
            sys.exit(1)
        try:
            hand_ctrl = RyHandController(port=hw_config["ruiyan_hand"]["port"])
            print("[deploy] RyHand connected")
        except Exception as e:
            print(f"[deploy] RyHand failed: {e}")
            ur_arm.stop()
            sys.exit(1)
        rs_env, rs_wrist = init_realsense_cameras(hw_config, img_w, img_h)
        if rs_env is None or rs_wrist is None:
            print("[deploy] Warning: one or both RealSense cameras failed — check hardware_config.yaml")

        if not no_start_pose:
            move_to_start_pose(ur_arm, hand_ctrl, arm_start_eef, hand_start_pose, hand_reset_speed,
                               SERVO_SPEED, SERVO_ACCEL, SERVO_DT, SERVO_LOOKAHEAD, SERVO_GAIN)
    else:
        print("[deploy] Dry-run: skipping hardware")

    obs_buffer: deque = deque(maxlen=n_obs)

    def fill_buffer():
        obs_buffer.clear()
        while len(obs_buffer) < n_obs:
            s = capture_observation_sample(
                ur_arm, hand_ctrl, rs_env, rs_wrist, img_h, img_w, dry_run,
            )
            if dry_run:
                s["camera_env"][...] = np.random.rand(*s["camera_env"].shape).astype(np.float32) * 0.3
                s["camera_wrist"][...] = np.random.rand(*s["camera_wrist"].shape).astype(np.float32) * 0.3
            obs_buffer.append(s)

    fill_buffer()

    policy.reset()
    with torch.no_grad():
        warm = stack_obs_for_policy(obs_buffer, device)
        _ = policy.predict_action(warm)
    help_suffix = " | r=reset" if can_reset_to_start else ""
    print(f"[deploy] Policy warmup done. Keys: SPACE=run/stop | q=quit{help_suffix}")

    running = False
    t_run_start = 0.0

    try:
        while True:
            s = capture_observation_sample(
                ur_arm, hand_ctrl, rs_env, rs_wrist, img_h, img_w, dry_run,
            )
            obs_buffer.append(s)

            vis_env = s.get("_vis_env", np.zeros((img_h, img_w, 3), dtype=np.uint8))
            vis_wrist = s.get("_vis_wrist", np.zeros((img_h, img_w, 3), dtype=np.uint8))
            canvas = build_deploy_ui_canvas(
                vis_env,
                vis_wrist,
                arm_pose=s["arm_eef_pose"],
                hand_angles_rad=s["hand_joint_angles"],
                running=running,
                can_reset=can_reset_to_start,
                max_duration=max_duration,
                t_run_start=t_run_start,
                now=time.monotonic(),
                frequency_hz=frequency,
                dry_run=dry_run,
            )
            cv2.imshow("Deploy", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                if not can_reset_to_start:
                    continue
                running = False
                if ur_arm is not None:
                    try:
                        ur_arm.rtde_c.servoStop()
                    except Exception:
                        pass
                print("[deploy] Policy OFF — resetting to start pose...")
                move_to_start_pose(ur_arm, hand_ctrl, arm_start_eef, hand_start_pose, hand_reset_speed,
                                   SERVO_SPEED, SERVO_ACCEL, SERVO_DT, SERVO_LOOKAHEAD, SERVO_GAIN)
                policy.reset()
                fill_buffer()
                print("[deploy] Ready. SPACE to run policy again.")
                continue
            if key == ord(" "):
                running = not running
                if running:
                    policy.reset()
                    fill_buffer()
                    t_run_start = time.monotonic()
                    print("[deploy] Policy ON")
                else:
                    print("[deploy] Policy OFF")
                    if ur_arm is not None:
                        try:
                            ur_arm.rtde_c.servoStop()
                        except Exception:
                            pass

            if running:
                if time.monotonic() - t_run_start > max_duration:
                    running = False
                    print("[deploy] Max duration reached, policy OFF")
                    if ur_arm is not None:
                        try:
                            ur_arm.rtde_c.servoStop()
                        except Exception:
                            pass
                    continue

                obs_dict = stack_obs_for_policy(obs_buffer, device)
                with torch.no_grad():
                    out = policy.predict_action(obs_dict)
                actions = out["action"][0].cpu().numpy()

                prev_pose = obs_buffer[-1]["arm_eef_pose"].copy()
                
                # actions[0] corresponds to the time step of the current observation.
                # For deltas, actions[0] is the motion from T-1 to T, and we are already at T.
                # So we must start from actions[1] (motion from T to T+1).
                start_action_idx = 1
                for i in range(steps_pi):
                    if start_action_idx + i >= len(actions):
                        break
                    row = actions[start_action_idx + i]
                    d_eef = row[:6]
                    hand_j = row[6:21].copy()
                    # Consistent with RyHand_IK.py teleop data collection:
                    #   thumb side_swing locked to +10°, index side_swing locked to 0,
                    #   middle/ring/pinky fully disabled.
                    hand_j[0] = np.deg2rad(10)
                    hand_j[3] = 0.0
                    hand_j[6:15] = 0.0

                    if dry_run:
                        prev_pose = apply_eef_delta(prev_pose, d_eef)
                        continue

                    target_pose = apply_eef_delta(prev_pose, d_eef)
                    prev_pose = target_pose.copy()
                    try:
                        ur_arm.rtde_c.servoL(
                            target_pose.tolist(),
                            SERVO_SPEED,
                            SERVO_ACCEL,
                            SERVO_DT,
                            SERVO_LOOKAHEAD,
                            SERVO_GAIN,
                        )
                    except Exception as e:
                        print(f"[deploy] servoL error: {e}")

                    if hand_ctrl is not None:
                        try:
                            hand_ctrl.set_angles(hand_j, speed=hand_motor_speed, radians=True)
                        except Exception as e:
                            print(f"[deploy] hand error: {e}")

                    s2 = capture_observation_sample(
                        ur_arm, hand_ctrl, rs_env, rs_wrist, img_h, img_w, dry_run,
                    )
                    obs_buffer.append(s2)
                    time.sleep(dt)

            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[deploy] Interrupted")
    finally:
        cv2.destroyAllWindows()
        if ur_arm is not None:
            try:
                ur_arm.rtde_c.servoStop()
            except Exception:
                pass
            ur_arm.stop()
        if hand_ctrl is not None:
            try:
                hand_ctrl.set_angles(hand_start_pose, speed=hand_reset_speed, radians=True)
                time.sleep(0.3)
                hand_ctrl.close()
            except Exception:
                pass
        for w in (rs_env, rs_wrist):
            if w is not None:
                try:
                    w.stop()
                except Exception:
                    pass
        print("[deploy] Shutdown complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
