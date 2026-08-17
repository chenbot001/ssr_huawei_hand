#!/usr/bin/env python3
"""
Earbud demo: run PLACE and PICK policies in a continuous loop.

Cycle
-----
  1. Reset arm to shared start pose + hand to PLACE start pose
  2. Press SPACE → PLACE policy runs for up to PLACE_MAX_DURATION seconds
  3. Reset arm to shared start pose + hand to PICK start pose
  4. PICK policy starts automatically, runs for up to PICK_MAX_DURATION seconds
  5. Reset back to PLACE start pose and wait for SPACE  (repeat from step 2)

Keys (OpenCV "Demo" window):
  SPACE  — start the current policy (when idle) / pause while running
  r      — stop and restart the full loop from the PLACE phase
  q      — quit

Fill in checkpoint paths and hand start poses in the CONFIGURE section below,
or pass them as CLI arguments (--place-ckpt / --pick-ckpt).

Usage:
  python scripts/deploy_policy_earbud_demo.py
  python scripts/deploy_policy_earbud_demo.py --dry-run
  python scripts/deploy_policy_earbud_demo.py \\
      --place-ckpt data/outputs/.../place/checkpoints/best.ckpt \\
      --pick-ckpt  data/outputs/.../pick/checkpoints/best.ckpt
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from collections import deque
from enum import Enum

import cv2
import numpy as np
import torch
import dill
import hydra
import zarr
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_PATH = str(PROJECT_ROOT / "src")
DP_PATH  = str(PROJECT_ROOT / "external" / "diffusion_policy")

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
# *** CONFIGURE THESE BEFORE RUNNING ***
# ---------------------------------------------------------------------------

# Checkpoint paths
PLACE_CKPT_PATH = "data/outputs/2026.04.11/18.28.20_train_diffusion_unet_hybrid_earbud_ssr_pickplace_earbud_image/checkpoints/latest.ckpt"
PICK_CKPT_PATH  = "data/outputs/2026.04.02/14.39.04_train_diffusion_unet_hybrid_earbud_ssr_pickplace_earbud_image/checkpoints/latest.ckpt"

# Zarr datasets used to compute mean start poses (arm EEF + hand angles)
DATASET_PATH_PLACE = "data/pickplace_earbud/place/ryhand_place_earbud.zarr"
DATASET_PATH_PICK  = "data/pickplace_earbud/pick/ryhand_pick_earbud.zarr"

# Maximum duration per phase (seconds)
PLACE_MAX_DURATION = 25.0
PICK_MAX_DURATION  = 30.0

# ---------------------------------------------------------------------------


class Phase(Enum):
    PLACE = "PLACE"
    PICK  = "PICK"

    def next(self) -> "Phase":
        return Phase.PICK if self == Phase.PLACE else Phase.PLACE


# ---------------------------------------------------------------------------
# Pose helpers
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
    prev_m  = pose_vector_to_matrix(prev_pose_vec.astype(np.float64))
    delta_m = pose_vector_to_matrix(delta_6d.astype(np.float64))
    return np.array(matrix_to_pose_vector(prev_m @ delta_m), dtype=np.float32)


def compute_mean_start_pose(zarr_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean arm EEF pose and hand joint angles from the first frame of every episode."""
    root = zarr.open(zarr_path, mode="r")
    ends = root["meta/episode_ends"][:]
    start_indices = [0] + [int(i) for i in ends[:-1]]
    arm_data = root["data/arm_eef_pose"]
    hand_data = root["data/hand_joint_angles"]
    arm_start = np.median([arm_data[i] for i in start_indices], axis=0).astype(np.float32)
    hand_start = np.median([hand_data[i] for i in start_indices], axis=0).astype(np.float32)
    print(f"[demo] Computed start pose from {zarr_path}")
    print(f"  Arm EEF: {arm_start}")
    print(f"  Hand:    {hand_start}")
    return arm_start, hand_start


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
def init_realsense_cameras(hw_config: dict, img_w: int, img_h: int):
    rs_configs = hw_config.get("cameras", {}).get("realsense", [])
    rs_by_name = {cfg.get("name", ""): cfg for cfg in rs_configs}

    def _one(role: str, cfg):
        if cfg is None:
            print(f"[demo] Missing camera config for {role}")
            return None
        cam_id     = cfg.get("id", "")
        cam_offset = cfg.get("offset", 0)
        cam_zoom   = cfg.get("zoom", 1.0)
        serial     = (cfg.get("serial") or cfg.get("serial_number") or "").strip()

        if serial:
            try:
                worker = RealSenseWorker(width=img_w, height=img_h, serial_number=serial)
            except (ValueError, ImportError) as e:
                print(f"[demo] RealSense SDK init failed ({role}): {e}")
                return None
            worker.set_zoom(cam_zoom)
            worker.daemon = True
            worker.start()
            for _ in range(50):
                if worker.get_latest_frame() is not None:
                    break
                time.sleep(0.1)
            if worker.get_latest_frame() is not None:
                print(f"[demo] {role} camera OK (serial={serial})")
                return worker
            worker.stop()
            return None

        video_idx = get_video_index_by_id(cam_id, cam_offset)
        if video_idx is None and cam_id:
            video_idx, _ = find_rgb_video_index_for_usb(cam_id, img_w, img_h)
        if video_idx is None:
            print(f"[demo] Cannot find video device for {role}")
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
            print(f"[demo] {role} camera OK (video{video_idx})")
            return worker
        worker.stop()
        return None

    return _one("Env", rs_by_name.get("rs_env")), _one("Wrist", rs_by_name.get("rs_wrist"))


def grab_camera_pair(rs_env, rs_wrist, img_h: int, img_w: int):
    env_img   = np.zeros((img_h, img_w, 3), dtype=np.uint8)
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


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------
def images_to_model_input(env_img: np.ndarray, wrist_img: np.ndarray):
    e = np.moveaxis(env_img.astype(np.float32),   -1, 0) / 255.0
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
    return policy, cfg


def capture_observation_sample(ur_arm, hand_ctrl, rs_env, rs_wrist,
                                img_h: int, img_w: int, dry_run: bool) -> dict:
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
        "camera_env":        env_chw,
        "camera_wrist":      wrist_chw,
        "arm_eef_pose":      arm_pose,
        "hand_joint_angles": hand_angles,
        "_vis_env":          env_hwc,
        "_vis_wrist":        wrist_hwc,
    }


def stack_obs_for_policy(obs_list: list, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "camera_env":        torch.from_numpy(np.stack([s["camera_env"]        for s in obs_list])).unsqueeze(0).to(device, dtype=torch.float32),
        "camera_wrist":      torch.from_numpy(np.stack([s["camera_wrist"]      for s in obs_list])).unsqueeze(0).to(device, dtype=torch.float32),
        "arm_eef_pose":      torch.from_numpy(np.stack([s["arm_eef_pose"]      for s in obs_list])).unsqueeze(0).to(device, dtype=torch.float32),
        "hand_joint_angles": torch.from_numpy(np.stack([s["hand_joint_angles"] for s in obs_list])).unsqueeze(0).to(device, dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui_canvas(
    env_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    *,
    phase: Phase,
    arm_pose: np.ndarray,
    hand_angles_rad: np.ndarray,
    running: bool,
    max_duration: float,
    t_run_start: float,
    now: float,
    frequency_hz: float,
    dry_run: bool,
    panel_w: int = 400,
    panel_h: int = 360,
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX

    def _bgr(rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if bgr.shape[:2] != (panel_h, panel_w):
            bgr = cv2.resize(bgr, (panel_w, panel_h))
        return bgr

    def _put(img, text, y, color=(220, 220, 220), scale=0.50, thick=1):
        cv2.putText(img, text, (12, y), font, scale, color, thick)

    wrist_panel = _bgr(wrist_rgb)
    env_panel   = _bgr(env_rgb)
    stats       = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    y0, dy = 30, 18

    # Phase + run state header
    phase_color = (0, 200, 255) if phase == Phase.PLACE else (80, 200, 80)
    header = f"[{phase.value}]  {'RUNNING' if running else 'idle'}"
    _put(stats, header, y0, color=phase_color if running else (160, 160, 160), scale=0.58, thick=2)

    _put(stats, "SPACE=start/pause | r=restart | q=quit", y0 + dy, scale=0.40)
    _put(stats, f"Rate: {frequency_hz:.1f} Hz  |  max: {max_duration:.0f} s", y0 + dy * 2, scale=0.44)

    if running:
        elapsed = max(0.0, now - t_run_start)
        _put(stats, f"Elapsed: {elapsed:.1f} / {max_duration:.0f} s",
             y0 + dy * 3, color=(0, 255, 200), scale=0.50)
    else:
        _put(stats, "Elapsed: —", y0 + dy * 3, color=(120, 120, 120))
        _put(stats, f"SPACE to start {phase.value}",
             y0 + dy * 4, color=(140, 140, 140), scale=0.42)

    if dry_run:
        _put(stats, "DRY RUN (no robot/cams)", y0 + dy * 5, color=(0, 165, 255), scale=0.44)

    line_y = y0 + dy * 6 + 4
    cv2.line(stats, (10, line_y), (panel_w - 10, line_y), (60, 60, 60), 1)
    cy = line_y + dy

    ap = arm_pose
    _put(stats, "Arm EEF Pose:", cy, color=(180, 220, 255), scale=0.48)
    cy += dy
    _put(stats, f"  pos  [{ap[0]:+.3f}, {ap[1]:+.3f}, {ap[2]:+.3f}]", cy)
    cy += dy
    _put(stats, f"  rot  [{ap[3]:+.3f}, {ap[4]:+.3f}, {ap[5]:+.3f}]", cy)
    cy += dy + 4

    hj = np.rad2deg(hand_angles_rad)
    _put(stats, "Hand Joints (deg):", cy, color=(180, 220, 255), scale=0.48)
    cy += dy
    _put(stats, f"  Thumb [{hj[0]:+5.1f},{hj[1]:+5.1f},{hj[2]:+5.1f}]", cy)
    cy += dy
    _put(stats, f"  Index [{hj[3]:+5.1f},{hj[4]:+5.1f},{hj[5]:+5.1f}]", cy)

    cv2.putText(wrist_panel, "WRIST", (10, 25), font, 0.65, (0, 200, 255), 2)
    cv2.putText(env_panel,   "ENV",   (10, 25), font, 0.65, (0, 200, 255), 2)
    cv2.putText(env_panel, phase.value, (panel_w - 85, 25), font, 0.65, phase_color, 2)

    canvas = np.hstack([wrist_panel, env_panel, stats])

    # Progress bar
    total_w = canvas.shape[1]
    bar_y = canvas.shape[0] - 12
    progress = 0.0
    if running and max_duration > 1e-6:
        progress = min(1.0, max(0.0, (now - t_run_start) / max_duration))
    cv2.rectangle(canvas, (8, bar_y), (total_w - 8, bar_y + 6), (60, 60, 60), -1)
    cv2.rectangle(canvas, (8, bar_y), (8 + int((total_w - 16) * progress), bar_y + 6), (0, 200, 0), -1)
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Earbud PLACE→PICK demo loop")
    parser.add_argument("--place-ckpt", default=PLACE_CKPT_PATH, metavar="PATH",
                        help="PLACE policy checkpoint (.ckpt)")
    parser.add_argument("--pick-ckpt",  default=PICK_CKPT_PATH,  metavar="PATH",
                        help="PICK policy checkpoint (.ckpt)")
    parser.add_argument("--device",              default="cuda:0")
    parser.add_argument("--frequency",           type=float, default=15.0)
    parser.add_argument("--steps-per-inference", type=int,   default=4)
    parser.add_argument("--dry-run",             action="store_true")
    args = parser.parse_args()

    os.chdir(str(PROJECT_ROOT))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[demo] CUDA not available, using CPU.")
        device = torch.device("cpu")

    dry_run            = args.dry_run
    frequency          = args.frequency
    steps_per_inference = args.steps_per_inference
    dt                 = 1.0 / max(frequency, 1e-3)

    def _resolve(p: str) -> pathlib.Path:
        path = pathlib.Path(p).expanduser()
        return (PROJECT_ROOT / path) if not path.is_absolute() else path.resolve()

    place_ckpt_path = _resolve(args.place_ckpt)
    pick_ckpt_path  = _resolve(args.pick_ckpt)

    for label, p in [("PLACE", place_ckpt_path), ("PICK", pick_ckpt_path)]:
        if not p.is_file():
            print(f"[demo] ERROR: {label} checkpoint not found: {p}")
            print("  Set PLACE_CKPT_PATH / PICK_CKPT_PATH at the top of this script or use CLI args.")
            sys.exit(1)

    print(f"[demo] Loading PLACE policy: {place_ckpt_path}")
    place_policy, place_cfg = load_policy(str(place_ckpt_path), device)
    print(f"[demo] Loading PICK  policy: {pick_ckpt_path}")
    pick_policy,  pick_cfg  = load_policy(str(pick_ckpt_path), device)

    # Image size — both policies must have been trained with the same resolution
    shape_meta = place_cfg.task.shape_meta
    _, img_h, img_w = shape_meta["obs"]["camera_env"]["shape"]

    n_obs_place    = place_cfg.n_obs_steps
    n_obs_pick     = pick_cfg.n_obs_steps
    steps_pi_place = max(1, min(steps_per_inference, int(place_policy.n_action_steps)))
    steps_pi_pick  = max(1, min(steps_per_inference, int(pick_policy.n_action_steps)))
    buf_maxlen     = max(n_obs_place, n_obs_pick)

    print(f"[demo] PLACE: n_obs={n_obs_place}, steps_per_inference={steps_pi_place}, max={PLACE_MAX_DURATION}s")
    print(f"[demo] PICK:  n_obs={n_obs_pick},  steps_per_inference={steps_pi_pick},  max={PICK_MAX_DURATION}s")

    # Compute mean start poses from datasets
    place_path = str(PROJECT_ROOT / DATASET_PATH_PLACE)
    pick_path  = str(PROJECT_ROOT / DATASET_PATH_PICK)
    for p in (place_path, pick_path):
        if not os.path.isdir(p):
            print(f"[demo] ERROR: dataset not found: {p}", file=sys.stderr)
            sys.exit(1)
    _arm_place, _hand_place = compute_mean_start_pose(place_path)
    _arm_pick,  _hand_pick  = compute_mean_start_pose(pick_path)

    # Phase-keyed look-ups
    policies   = {Phase.PLACE: place_policy,                 Phase.PICK: pick_policy}
    n_obs_map  = {Phase.PLACE: n_obs_place,                  Phase.PICK: n_obs_pick}
    steps_map  = {Phase.PLACE: steps_pi_place,               Phase.PICK: steps_pi_pick}
    dur_map    = {Phase.PLACE: PLACE_MAX_DURATION,           Phase.PICK: PICK_MAX_DURATION}
    hand_map   = {Phase.PLACE: _hand_place,                  Phase.PICK: _hand_pick}
    arm_eef_map = {Phase.PLACE: _arm_place,                  Phase.PICK: _arm_pick}

    # Hardware init
    hw_config     = get_hardware_config()
    teleop_config = get_teleop_config()
    servo_cfg     = teleop_config.get("servo", {})
    control_cfg   = teleop_config.get("control", {})
    hand_motor_speed = control_cfg.get("hand_motor_speed", 1000)
    hand_reset_speed = control_cfg.get("hand_reset_speed", 500)

    SERVO_SPEED     = servo_cfg.get("speed", 0.5)
    SERVO_ACCEL     = servo_cfg.get("acceleration", 0.5)
    SERVO_DT        = servo_cfg.get("dt", 0.002)
    SERVO_LOOKAHEAD = servo_cfg.get("lookahead_time", 0.1)
    SERVO_GAIN      = servo_cfg.get("gain", 300)

    ur_arm    = None
    hand_ctrl = None
    rs_env    = None
    rs_wrist  = None

    if not dry_run:
        try:
            ur_arm = UR5Arm(ip=hw_config["ur_arm"]["ip"])
            print("[demo] UR5 connected")
        except Exception as e:
            print(f"[demo] UR5 failed: {e}")
            sys.exit(1)
        try:
            hand_ctrl = RyHandController(port=hw_config["ruiyan_hand"]["port"])
            print("[demo] RyHand connected")
        except Exception as e:
            print(f"[demo] RyHand failed: {e}")
            ur_arm.stop()
            sys.exit(1)
        rs_env, rs_wrist = init_realsense_cameras(hw_config, img_w, img_h)
        if rs_env is None or rs_wrist is None:
            print("[demo] Warning: one or both cameras not found — check hardware_config.yaml")
    else:
        print("[demo] Dry-run: skipping hardware init")

    obs_buffer: deque = deque(maxlen=buf_maxlen)

    # ------------------------------------------------------------------
    # Helpers that close over mutable state
    # ------------------------------------------------------------------
    def _servo_stop():
        if ur_arm is not None:
            try:
                ur_arm.rtde_c.servoStop()
            except Exception:
                pass

    def fill_buffer(n_obs: int):
        obs_buffer.clear()
        while len(obs_buffer) < n_obs:
            s = capture_observation_sample(ur_arm, hand_ctrl, rs_env, rs_wrist, img_h, img_w, dry_run)
            if dry_run:
                s["camera_env"][...]   = np.random.rand(*s["camera_env"].shape).astype(np.float32) * 0.3
                s["camera_wrist"][...] = np.random.rand(*s["camera_wrist"].shape).astype(np.float32) * 0.3
            obs_buffer.append(s)

def reset_phase(p: Phase, arm_eef: np.ndarray, hand_pose: np.ndarray,
                servo_speed=0.5, servo_accel=0.5, servo_dt=0.002,
                servo_lookahead=0.1, servo_gain=300):
    """Move arm to EEF pose and hand to phase-specific start pose (both dataset-mean)."""
    _servo_stop()
    if ur_arm is not None and not dry_run:
        print(f"[demo] Moving arm to mean-start EEF pose via moveL ...")
        ur_arm.rtde_c.moveL(arm_eef.tolist(), speed=servo_speed, acceleration=servo_accel)
        # moveL blocks until the robot reaches the target
    if hand_ctrl is not None and not dry_run:
        hand_ctrl.set_angles(hand_pose, speed=hand_reset_speed, radians=True)
        time.sleep(0.3)
    policies[p].reset()
    fill_buffer(n_obs_map[p])

    # ------------------------------------------------------------------
    # Video recording helpers
    # Frames are buffered with timestamps so the actual elapsed fps is
    # computed at save-time — avoids playback speed mismatches when the
    # main loop runs slower than `frequency` (e.g. during policy execution).
    # ------------------------------------------------------------------
    demo_dir = PROJECT_ROOT / "demo"
    demo_dir.mkdir(exist_ok=True)

    _recording:    bool            = False
    _frame_buffer: list            = []   # list of (timestamp_monotonic, bgr_frame)
    _rec_path:     str | None      = None

    def _start_recording(phase_name: str):
        nonlocal _recording, _frame_buffer, _rec_path
        _stop_recording()
        ts = time.strftime("%Y%m%d_%H%M%S")
        _rec_path     = str(demo_dir / f"{ts}_{phase_name}.mp4")
        _frame_buffer = []
        _recording    = True
        print(f"[demo] Recording started → {_rec_path}")

    def _stop_recording():
        nonlocal _recording, _frame_buffer, _rec_path
        if not _recording:
            return
        _recording = False
        if not _frame_buffer or _rec_path is None:
            _frame_buffer = []
            _rec_path     = None
            return
        # Compute actual fps from wall-clock timestamps
        timestamps = [t for t, _ in _frame_buffer]
        if len(timestamps) > 1:
            actual_fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        else:
            actual_fps = frequency
        actual_fps = float(np.clip(actual_fps, 1.0, 60.0))
        h, w = _frame_buffer[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(_rec_path, fourcc, actual_fps, (w, h))
        for _, frame in _frame_buffer:
            writer.write(frame)
        writer.release()
        print(f"[demo] Saved {_rec_path}  ({len(_frame_buffer)} frames @ {actual_fps:.1f} fps)")
        _frame_buffer = []
        _rec_path     = None

    def _write_frame(canvas: np.ndarray):
        if _recording:
            _frame_buffer.append((time.monotonic(), canvas.copy()))

    # ------------------------------------------------------------------
    # Initial setup
    # ------------------------------------------------------------------
    phase       = Phase.PLACE
    running     = False
    t_run_start = 0.0

    print(f"[demo] Resetting to {phase.value} start pose ...")
    reset_phase(phase, arm_eef_map[phase], hand_map[phase],
                SERVO_SPEED, SERVO_ACCEL, SERVO_DT, SERVO_LOOKAHEAD, SERVO_GAIN)

    # Warm up both policies so first inference is fast
    with torch.no_grad():
        for ph, pol, n in [(Phase.PLACE, place_policy, n_obs_place),
                           (Phase.PICK,  pick_policy,  n_obs_pick)]:
            obs_slice = list(obs_buffer)[:n]
            while len(obs_slice) < n:
                obs_slice.insert(0, obs_slice[0])
            _ = pol.predict_action(stack_obs_for_policy(obs_slice, device))
    print(f"[demo] Policy warmup done. Ready to run {phase.value}. SPACE=start | r=restart | q=quit")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    try:
        while True:
            s = capture_observation_sample(ur_arm, hand_ctrl, rs_env, rs_wrist, img_h, img_w, dry_run)
            obs_buffer.append(s)

            now     = time.monotonic()
            max_dur = dur_map[phase]

            canvas = build_ui_canvas(
                s.get("_vis_env",   np.zeros((img_h, img_w, 3), dtype=np.uint8)),
                s.get("_vis_wrist", np.zeros((img_h, img_w, 3), dtype=np.uint8)),
                phase=phase, arm_pose=s["arm_eef_pose"],
                hand_angles_rad=s["hand_joint_angles"],
                running=running, max_duration=max_dur,
                t_run_start=t_run_start, now=now,
                frequency_hz=frequency, dry_run=dry_run,
            )
            cv2.imshow("Demo", canvas)
            _write_frame(canvas)
            key = cv2.waitKey(1) & 0xFF

            # ---- key handling ----------------------------------------
            if key == ord("q"):
                break

            if key == ord("r"):
                # Restart the full loop from the PLACE phase
                _stop_recording()
                running = False
                _servo_stop()
                print("[demo] Restarting loop — resetting to PLACE start pose ...")
                phase = Phase.PLACE
                reset_phase(phase, arm_eef_map[phase], hand_map[phase],
                            SERVO_SPEED, SERVO_ACCEL, SERVO_DT, SERVO_LOOKAHEAD, SERVO_GAIN)
                print(f"[demo] Ready. SPACE to start {phase.value} policy.")
                continue

            if key == ord(" "):
                if running:
                    _stop_recording()
                    running = False
                    _servo_stop()
                    print(f"[demo] {phase.value} policy PAUSED")
                else:
                    # Resume: shift t_run_start forward by the paused duration so
                    # elapsed time only counts time actually running.
                    t_run_start = time.monotonic() - (t_run_start if t_run_start == 0.0 else 0.0)
                    t_run_start = time.monotonic()
                    policies[phase].reset()
                    fill_buffer(n_obs_map[phase])
                    t_run_start = time.monotonic()
                    running = True
                    _start_recording("PLACE_PICK")
                    print(f"[demo] {phase.value} policy RESUMED")

            # ---- policy execution ------------------------------------
            if running:
                if now - t_run_start > max_dur:
                    running = False
                    _servo_stop()
                    print(f"[demo] {phase.value} done (max duration reached)")

                    # PLACE→PICK: auto-start; PICK→PLACE: wait for SPACE
                    phase = phase.next()
                    print(f"[demo] Resetting to {phase.value} start pose ...")
                    reset_phase(phase, arm_eef_map[phase], hand_map[phase],
                                SERVO_SPEED, SERVO_ACCEL, SERVO_DT, SERVO_LOOKAHEAD, SERVO_GAIN)
                    if phase == Phase.PICK:
                        t_run_start = time.monotonic()
                        running = True
                        # Keep recording — PLACE and PICK are one continuous video
                        print(f"[demo] {phase.value} policy ON (auto-start)")
                    else:
                        _stop_recording()
                        print(f"[demo] Ready. SPACE to start {phase.value} policy.")
                    continue

                # Build obs slice of correct length for the current policy
                n_obs     = n_obs_map[phase]
                obs_slice = list(obs_buffer)[-n_obs:]
                while len(obs_slice) < n_obs:           # pad left if buffer not full yet
                    obs_slice.insert(0, obs_slice[0])

                obs_dict = stack_obs_for_policy(obs_slice, device)
                with torch.no_grad():
                    out = policies[phase].predict_action(obs_dict)
                actions  = out["action"][0].cpu().numpy()
                steps_pi = steps_map[phase]

                prev_pose = obs_buffer[-1]["arm_eef_pose"].copy()

                # actions[0] = motion T-1→T (already at T); start from actions[1]
                for i in range(steps_pi):
                    action_idx = 1 + i
                    if action_idx >= len(actions):
                        break
                    row    = actions[action_idx]
                    d_eef  = row[:6]
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
                    prev_pose   = target_pose.copy()
                    try:
                        ur_arm.rtde_c.servoL(
                            target_pose.tolist(),
                            SERVO_SPEED, SERVO_ACCEL,
                            SERVO_DT, SERVO_LOOKAHEAD, SERVO_GAIN,
                        )
                    except Exception as e:
                        print(f"[demo] servoL error: {e}")

                    if hand_ctrl is not None:
                        try:
                            hand_ctrl.set_angles(hand_j, speed=hand_motor_speed, radians=True)
                        except Exception as e:
                            print(f"[demo] hand error: {e}")

                    s2 = capture_observation_sample(ur_arm, hand_ctrl, rs_env, rs_wrist, img_h, img_w, dry_run)
                    obs_buffer.append(s2)
                    time.sleep(dt)

            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[demo] Interrupted")
    finally:
        _stop_recording()
        cv2.destroyAllWindows()
        _servo_stop()
        if ur_arm is not None:
            ur_arm.stop()
        if hand_ctrl is not None:
            try:
                # Return to a safe open pose on shutdown
                hand_ctrl.set_angles(hand_map[Phase.PLACE], speed=hand_reset_speed, radians=True)
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
        print("[demo] Shutdown complete")


if __name__ == "__main__":
    main()
