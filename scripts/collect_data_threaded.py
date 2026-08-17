#!/usr/bin/env python3
"""
数据采集脚本 (threaded) — 遥操作控制与数据录制在独立线程中运行,
避免录制 I/O (RTDE 读、CAN 读、图像处理) 阻塞 servoL 指令流。

Architecture:
  ┌────────────────────────────────────────────────┐
  │ Control Thread  (daemon, control_rate Hz)      │
  │  T265 poll → servoL  |  Manus IK → hand       │
  │  publishes hand_action to shared snapshot      │
  │  pauses during reset via Event                 │
  └────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────┐
  │ Main Thread                                    │
  │  Keyboard listener                             │
  │  Data recording at record_rate Hz              │
  │  Episode management (flush / discard / pop)    │
  │  Reset coordination (pauses control thread)    │
  │  Cleanup & zarr finalization                   │
  └────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────┐
  │ Background threads (unchanged from original):  │
  │  GloveDataReceiver (ZMQ)                       │
  │  RealSenseWorker × 2 (camera capture)          │
  └────────────────────────────────────────────────┘

Dataset schema identical to collect_data.py.

Usage:
    python scripts/collect_data_threaded.py [options]
    (same CLI arguments as collect_data.py)
"""

import argparse
import os
import shutil
import sys
import time
import math
import threading
from datetime import datetime

import cv2
import numpy as np
import zarr
import numcodecs
import pyrealsense2 as rs
import zmq
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

# ============================================================================
# 路径设置
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")

for _p in [src_path, project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.chdir(project_root)

# ============================================================================
# 硬件驱动导入
# ============================================================================
from ssr.hardware.arm_ur5 import UR5Arm
from ssr.hardware.ruiyan_driver import RyHandController
from ssr.hardware.realsense_worker import RealSenseWorker
from ssr.config import get_hardware_config, get_teleop_config
from ssr.utils.camera_utils import get_video_index_by_id, find_rgb_video_index_for_usb
from ssr.control.RyHand_IK import RYHandIK, ik_to_hand_angles

# ============================================================================
# 手套与 IK 配置
# ============================================================================
_hw_config = get_hardware_config()
_manus_config = _hw_config.get('manus_glove', {})
IP_ADDRESS = _manus_config.get('address', "tcp://localhost:8000")
LEFT_GLOVE_SN = _manus_config.get('left_sn', "4848debd")
RIGHT_GLOVE_SN = _manus_config.get('right_sn', "db397317")

NUM_JOINTS = 25
VALUES_PER_JOINT = 7
SHORT_IDX = [23, 24, 4, 5, 9, 10, 19, 20, 14, 15]

# ============================================================================
# 坐标系 & 参数
# ============================================================================
_teleop_config = get_teleop_config()
_servo_cfg = _teleop_config.get('servo', {})
_t265_cfg = _teleop_config.get('t265', {})
_control_cfg = _teleop_config.get('control', {})

TRANSLATION_SCALE = _t265_cfg.get('translation_scale', 1.0)
SERVO_SPEED = _servo_cfg.get('speed', 0.5)
SERVO_ACCEL = _servo_cfg.get('acceleration', 0.5)
SERVO_DT = _servo_cfg.get('dt', 0.002)
SERVO_LOOKAHEAD = _servo_cfg.get('lookahead_time', 0.1)
SERVO_GAIN = _servo_cfg.get('gain', 300)
HAND_MOTOR_SPEED = _control_cfg.get('hand_motor_speed', 1000)
HAND_RESET_SPEED = _control_cfg.get('hand_reset_speed', 500)
_profiler_cfg = _teleop_config.get('velocity_profiler', {})
PROFILER_MAX_STEP = _profiler_cfg.get('max_step', 0.15)  # max Cartesian translation step per control cycle (m)
PROFILER_MIN_STEP = _profiler_cfg.get('min_step', 0.001)  # deadband — steps smaller than this are suppressed (m)
T265_TO_UR_ALIGN = np.array([
    [ 0,  0, -1,  0],
    [-1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0,  1]
], dtype=np.float64)


# ============================================================================
# 辅助函数
# ============================================================================
def create_pose_matrix(translation, rotation_quat):
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_quat([
        rotation_quat.x, rotation_quat.y,
        rotation_quat.z, rotation_quat.w
    ]).as_matrix()
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def matrix_to_pose_vector(matrix):
    pos = matrix[:3, 3]
    rot = R.from_matrix(matrix[:3, :3]).as_rotvec()
    return [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]]


def pose_vector_to_matrix(pose_vec):
    matrix = np.eye(4)
    matrix[:3, 3] = pose_vec[:3]
    matrix[:3, :3] = R.from_rotvec(pose_vec[3:]).as_matrix()
    return matrix


# ============================================================================
# GloveDataReceiver
# ============================================================================
class GloveDataReceiver:
    def __init__(self, left_sn=LEFT_GLOVE_SN, right_sn=RIGHT_GLOVE_SN):
        self.left_sn = left_sn
        self.right_sn = right_sn
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, True)
        self.socket.connect(IP_ADDRESS)
        self.left_short = None
        self.left_wrist = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"[数据手套] 已连接至 {IP_ADDRESS}")

    def _receive_loop(self):
        while self.running:
            try:
                message = self.socket.recv(flags=zmq.NOBLOCK).decode('utf-8')
                data = message.split(",")
                if len(data) >= 176:
                    self._process_skeleton(data[0:176])
                if len(data) == 352:
                    self._process_skeleton(data[176:352])
            except zmq.Again:
                time.sleep(0.001)
            except Exception:
                pass

    def _process_skeleton(self, data):
        if len(data) < 176:
            return
        serial_number = data[0]
        short_positions = []
        for i in SHORT_IDX:
            idx = 1 + i * VALUES_PER_JOINT
            short_positions.append([float(data[idx]), -float(data[idx + 1]), float(data[idx + 2])])
        wrist_idx = 1 + 0 * VALUES_PER_JOINT
        wrist_pos = [float(data[wrist_idx]), -float(data[wrist_idx + 1]), float(data[wrist_idx + 2])]

        with self.lock:
            if serial_number == self.left_sn or serial_number not in [self.left_sn, self.right_sn]:
                self.left_short = short_positions
                self.left_wrist = wrist_pos

    def get_data(self):
        with self.lock:
            if self.left_short and self.left_wrist:
                return {"fingers": self.left_short.copy(), "wrist": self.left_wrist.copy()}
            return None

    def close(self):
        self.running = False
        self.socket.close()
        self.context.term()


# ============================================================================
# RecordingState
# ============================================================================
class RecordingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.recording = False
        self.should_quit = False
        self.should_discard = False
        self.should_reset = False
        self.clutch_active = False

    def toggle_recording(self):
        with self.lock:
            self.recording = not self.recording
            return self.recording

    def toggle_clutch(self):
        with self.lock:
            self.clutch_active = not self.clutch_active
            return self.clutch_active

    def set_clutch(self, active):
        with self.lock:
            self.clutch_active = active

    def request_quit(self):
        with self.lock:
            self.should_quit = True

    def request_discard(self):
        with self.lock:
            self.should_discard = True

    def clear_discard(self):
        with self.lock:
            self.should_discard = False

    def request_reset(self):
        with self.lock:
            self.should_reset = True

    def clear_reset(self):
        with self.lock:
            self.should_reset = False

    @property
    def is_recording(self):
        with self.lock:
            return self.recording

    @property
    def is_clutch_active(self):
        with self.lock:
            return self.clutch_active

    @property
    def quit_requested(self):
        with self.lock:
            return self.should_quit

    @property
    def discard_requested(self):
        with self.lock:
            return self.should_discard

    @property
    def reset_requested(self):
        with self.lock:
            return self.should_reset


# ============================================================================
# ControlThread — dedicated thread for T265+servoL and Manus+hand at 80 Hz
# ============================================================================
class ControlThread(threading.Thread):
    """
    Runs the real-time teleop loop independently of data recording.
    Publishes the latest IK hand angles to a shared snapshot so the
    recording path can read them without blocking.
    """

    def __init__(
        self,
        state: RecordingState,
        t265_pipeline,
        ur_arm,
        glove_receiver,
        ryhand_ik,
        hand_ctrl,
        control_dt: float,
    ):
        super().__init__(daemon=True)
        self.state = state
        self.t265_pipeline = t265_pipeline
        self.ur_arm = ur_arm
        self.glove_receiver = glove_receiver
        self.ryhand_ik = ryhand_ik
        self.hand_ctrl = hand_ctrl
        self.control_dt = control_dt

        # shared with main thread for calibration/reset
        self.base_t265_matrix = None
        self.base_ur_matrix = None

        # published snapshot (read by data thread)
        self._snapshot_lock = threading.Lock()
        self._hand_action = np.zeros(15, dtype=np.float32)

        # CAN bus lock: protects set_angles (control) vs get_angles (data)
        self.hand_hw_lock = threading.Lock()

        # pause mechanism for reset sequences
        self._resume_event = threading.Event()
        self._resume_event.set()  # starts running

        # internal
        self._was_clutch_active = False
        self._clutch_t265_matrix = None
        self._clutch_ur_matrix = None
        self._prev_tcp_target = None  # last sent TCP position for step clamping
        self._glove_connected = False

        # timing diagnostics
        self._loop_overrun_count = 0

    # ---- snapshot API ----

    def get_hand_action(self) -> np.ndarray:
        with self._snapshot_lock:
            return self._hand_action.copy()

    def set_calibration(self, base_t265, base_ur):
        self.base_t265_matrix = base_t265
        self.base_ur_matrix = base_ur

    def pause(self):
        """Pause the control loop (blocks until the loop has yielded)."""
        self._resume_event.clear()

    def resume(self):
        self._resume_event.set()

    # ---- main loop ----

    def run(self):
        last_time = time.time()

        while not self.state.quit_requested:
            # honour pause requests (e.g. during reset)
            self._resume_event.wait()

            start_time = time.time()
            dt = start_time - last_time
            if dt <= 0:
                dt = 0.001
            last_time = start_time

            # --- Hand control (Manus → IK → RyHand) ---
            if self.glove_receiver is not None and self.ryhand_ik is not None:
                skeleton_data = self.glove_receiver.get_data()

                if skeleton_data is not None and not self._glove_connected:
                    print(f"\n[信息] 成功与 MANUS 穿戴套件建立心跳反馈!")
                    self._glove_connected = True

                if skeleton_data is not None:
                    ik_positions = self.ryhand_ik.compute_ik(skeleton_data)
                    if ik_positions is not None:
                        hand_angles = ik_to_hand_angles(ik_positions)
                        with self._snapshot_lock:
                            self._hand_action = hand_angles.astype(np.float32)
                        if self.hand_ctrl is not None:
                            with self.hand_hw_lock:
                                self.hand_ctrl.set_angles(
                                    hand_angles, speed=HAND_MOTOR_SPEED, radians=True)

            # --- Arm control (T265 → servoL) ---
            if self.t265_pipeline is not None:
                frames = self.t265_pipeline.poll_for_frames()
                if frames:
                    pose_frame = frames.get_pose_frame()
                    if pose_frame:
                        pose_data = pose_frame.get_pose_data()
                        if pose_data.tracker_confidence < 2:
                            continue
                        current_t265_matrix = create_pose_matrix(
                            pose_data.translation, pose_data.rotation)

                        if self.state.is_clutch_active and self.ur_arm is not None:
                            if not self._was_clutch_active:
                                self._clutch_t265_matrix = current_t265_matrix.copy()
                                self._clutch_ur_matrix = pose_vector_to_matrix(
                                    self.ur_arm.rtde_r.getActualTCPPose())
                                self._prev_tcp_target = self._clutch_ur_matrix[:3, 3].copy()
                                self._was_clutch_active = True

                            if (self.base_t265_matrix is not None
                                    and self.base_ur_matrix is not None):
                                rot_delta = np.linalg.inv(
                                    self._clutch_t265_matrix) @ current_t265_matrix
                                rot_delta[:3, 3] = 0
                                mapped_rot = (T265_TO_UR_ALIGN
                                              @ rot_delta @ T265_TO_UR_ALIGN.T)
                                mapped_rv = R.from_matrix(
                                    mapped_rot[:3, :3]).as_rotvec()

                                adj_ry = -mapped_rv[1]
                                adj_rx = mapped_rv[2]
                                adj_rz = mapped_rv[0]
                                adj_rot_mat = R.from_rotvec(
                                    [adj_rx, adj_ry, adj_rz]).as_matrix()

                                target_rot = (self._clutch_ur_matrix[:3, :3]
                                              @ adj_rot_mat)
                                trans_delta = (current_t265_matrix[:3, 3]
                                               - self._clutch_t265_matrix[:3, 3])
                                mapped_trans = (T265_TO_UR_ALIGN[:3, :3]
                                                @ trans_delta * TRANSLATION_SCALE)

                                target = np.eye(4)
                                target[:3, :3] = target_rot
                                desired_pos = self._clutch_ur_matrix[:3, 3] + mapped_trans
                                step = desired_pos - self._prev_tcp_target
                                step_norm = np.linalg.norm(step)
                                if step_norm > PROFILER_MAX_STEP:
                                    step = step * (PROFILER_MAX_STEP / step_norm)
                                if step_norm > PROFILER_MIN_STEP:
                                    self._prev_tcp_target = self._prev_tcp_target + step
                                target[:3, 3] = self._prev_tcp_target

                                try:
                                    self.ur_arm.rtde_c.servoL(
                                        matrix_to_pose_vector(target),
                                        SERVO_SPEED, SERVO_ACCEL, SERVO_DT,
                                        SERVO_LOOKAHEAD, SERVO_GAIN)
                                except Exception:
                                    pass
                        else:
                            if self._was_clutch_active:
                                if self.ur_arm is not None:
                                    try:
                                        self.ur_arm.rtde_c.servoStop()
                                    except Exception:
                                        pass
                                self._was_clutch_active = False

            # --- timing ---
            elapsed = time.time() - start_time
            if elapsed < self.control_dt:
                time.sleep(self.control_dt - elapsed)
            else:
                self._loop_overrun_count += 1

    def force_disengage(self):
        """Disengage clutch and stop servo (called from main thread during reset)."""
        self.state.set_clutch(False)
        self._was_clutch_active = False
        if self.ur_arm is not None:
            try:
                self.ur_arm.rtde_c.servoStop()
            except Exception:
                pass


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="数据采集脚本 — threaded (diffusion_policy 格式)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出 zarr 目录路径。默认: data/collected_YYYYMMDD_HHMMSS.zarr")
    parser.add_argument("--record-rate", type=float, default=15.0,
                        help="数据记录频率 Hz (默认 15)")
    parser.add_argument("--control-rate", type=float, default=80.0,
                        help="控制循环频率 Hz (默认 80)")
    parser.add_argument("--img-width", type=int, default=320,
                        help="RealSense RGB 图像宽度 (默认 320)")
    parser.add_argument("--img-height", type=int, default=240,
                        help="RealSense RGB 图像高度 (默认 240)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅测试数据流，不连接真实硬件 (UR / RuiYan)")
    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(project_root, "data", f"collected_{timestamp}.zarr")
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    img_w, img_h = args.img_width, args.img_height
    control_dt = 1.0 / args.control_rate
    record_interval = 1.0 / args.record_rate

    print("=" * 70)
    print("     数据采集系统 — THREADED (diffusion_policy zarr 格式)")
    print("=" * 70)
    print(f"  输出路径     : {args.output}")
    print(f"  控制频率     : {args.control_rate} Hz  (dedicated thread)")
    print(f"  记录频率     : {args.record_rate} Hz   (main thread)")
    print(f"  图像分辨率   : {img_w} x {img_h}")
    print(f"  Dry-run      : {args.dry_run}")
    print("-" * 70)

    # ====================================================================
    # 1. 初始化所有硬件 (identical to collect_data.py)
    # ====================================================================
    hw_config = get_hardware_config()
    state = RecordingState()

    ur_arm = None
    if not args.dry_run:
        try:
            ur_arm = UR5Arm(ip=hw_config['ur_arm']['ip'])
            print("[✓] UR5 机械臂已连接")
        except Exception as e:
            print(f"[✗] UR5 连接失败: {e}")
    else:
        print("[~] Dry-run: 跳过 UR5 连接")

    hand_ctrl = None
    if not args.dry_run:
        try:
            hand_ctrl = RyHandController(port=hw_config['ruiyan_hand']['port'])
            print("[✓] Ruiyan 灵巧手已连接")
        except Exception as e:
            print(f"[✗] Ruiyan 手连接失败: {e}")
    else:
        print("[~] Dry-run: 跳过 Ruiyan 手连接")

    rs_configs = hw_config.get('cameras', {}).get('realsense', [])
    rs_by_name = {cfg.get('name', ''): cfg for cfg in rs_configs}

    def _init_rs_camera(role, cfg):
        if cfg is None:
            print(f"[✗] 未在配置中找到 {role} 相机设置")
            return None
        cam_id = cfg.get('id', '')
        cam_offset = cfg.get('offset', 0)
        cam_zoom = cfg.get('zoom', 1.0)
        serial = (cfg.get('serial') or cfg.get('serial_number') or '').strip()

        if serial:
            try:
                worker = RealSenseWorker(
                    width=img_w, height=img_h, serial_number=serial)
            except (ValueError, ImportError) as e:
                print(f"[✗] {role} 相机 SDK 初始化失败: {e}")
                return None
            worker.set_zoom(cam_zoom)
            worker.daemon = True
            worker.start()
            for _ in range(50):
                if worker.get_latest_frame() is not None:
                    break
                time.sleep(0.1)
            if worker.get_latest_frame() is not None:
                print(f"[✓] {role} 相机已启动 (serial={serial}, SDK)")
                return worker
            print(f"[✗] {role} 相机无画面 (serial={serial})")
            worker.stop()
            return None

        video_idx = get_video_index_by_id(cam_id, cam_offset)
        if video_idx is None and cam_id:
            print(f"[信息] {role} 按 offset 未找到节点，尝试在 USB `{cam_id}` 上自动探测 RGB 节点…")
            video_idx, used_off = find_rgb_video_index_for_usb(cam_id, img_w, img_h)
            if video_idx is not None:
                print(f"[✓] {role} 自动探测到 video{video_idx} (offset={used_off})")

        if video_idx is None:
            print(f"[✗] {role} 相机无法通过 USB ID '{cam_id}' 定位设备节点")
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
            print(f"[✓] {role} 相机已启动 (video{video_idx}, USB: {cam_id})")
            return worker

        print(f"[信息] {role} 在 video{video_idx} 无画面，尝试同 USB 下其它节点…")
        worker.stop()
        alt_idx, alt_off = find_rgb_video_index_for_usb(cam_id, img_w, img_h)
        if alt_idx is None or alt_idx == video_idx:
            print(f"[✗] {role} 相机仍无法获取画面")
            return None
        worker = RealSenseWorker(camera_index=alt_idx, width=img_w, height=img_h)
        worker.set_zoom(cam_zoom)
        worker.daemon = True
        worker.start()
        for _ in range(50):
            if worker.get_latest_frame() is not None:
                break
            time.sleep(0.1)
        if worker.get_latest_frame() is not None:
            print(f"[✓] {role} 相机已启动 (video{alt_idx}, offset={alt_off}, USB: {cam_id})")
            return worker
        print(f"[✗] {role} 相机无法获取画面 (video{alt_idx})")
        worker.stop()
        return None

    rs_env_camera = _init_rs_camera("Env (环境)", rs_by_name.get("rs_env"))
    rs_wrist_camera = _init_rs_camera("Wrist (手腕)", rs_by_name.get("rs_wrist"))

    t265_pipeline = None
    try:
        t265_pipeline = rs.pipeline()
        t265_config = rs.config()
        t265_serial = (hw_config.get("t265") or {}).get("serial") or hw_config.get(
            "t265_serial")
        if t265_serial:
            t265_serial = str(t265_serial).strip()
        if t265_serial:
            t265_config.enable_device(t265_serial)
        t265_config.enable_stream(rs.stream.pose)
        t265_pipeline.start(t265_config)
        frames = t265_pipeline.wait_for_frames(timeout_ms=3000)
        if frames.get_pose_frame():
            if t265_serial:
                print(f"[✓] T265 追踪相机已启动 (serial={t265_serial})")
            else:
                print("[✓] T265 追踪相机已启动（未配置 t265.serial 时为默认设备）")
        else:
            print("[✗] T265 无法获取位姿帧")
            t265_pipeline.stop()
            t265_pipeline = None
    except Exception as e:
        print(f"[✗] T265 连接失败: {e}")
        t265_pipeline = None

    def _t265_wait_pose(timeout_ms: int = 8000, attempts: int = 6):
        if t265_pipeline is None:
            return None
        last_err = None
        for _ in range(attempts):
            try:
                frames = t265_pipeline.wait_for_frames(timeout_ms=timeout_ms)
                if frames.get_pose_frame():
                    return frames
            except RuntimeError as e:
                last_err = e
                time.sleep(0.2)
        if last_err is not None:
            print(f"[✗] T265 取帧失败（已重试 {attempts} 次）: {last_err}")
        return None

    glove_receiver = None
    ryhand_ik = None
    try:
        glove_receiver = GloveDataReceiver()
        ryhand_ik = RYHandIK()
        print("[✓] Manus 手套接收器 + IK 引擎已启动")
    except Exception as e:
        print(f"[✗] Manus/IK 初始化失败: {e}")

    print("-" * 70)

    # ====================================================================
    # 2. T265 + UR 初始基座校准
    # ====================================================================
    base_t265_matrix = None
    base_ur_matrix = None

    if t265_pipeline is not None and ur_arm is not None:
        print("[系统] 正在执行启动校准...")
        time.sleep(1.0)
        frames = _t265_wait_pose()
        pose_frame = frames.get_pose_frame() if frames else None
        if pose_frame:
            pose_data = pose_frame.get_pose_data()
            base_t265_matrix = create_pose_matrix(
                pose_data.translation, pose_data.rotation)
            base_ur_matrix = pose_vector_to_matrix(
                ur_arm.rtde_r.getActualTCPPose())
            print("[✓] 初始校准完成: T265/UR 基座位姿已锁定")
        else:
            print("[✗] 校准失败: 无法获取 T265 初始帧")
    elif t265_pipeline is not None and args.dry_run:
        print("[~] Dry-run: 跳过 T265/UR 校准")
    else:
        print("[!] T265 或 UR 不可用，机械臂遥操作功能受限")

    # ====================================================================
    # 2b. 起始位姿
    # ====================================================================
    _start_pose_cfg = _teleop_config.get('start_pose', {})
    start_pose_joints = _start_pose_cfg.get('arm_joints')
    start_move_speed = _start_pose_cfg.get('move_speed', 0.5)
    start_move_accel = _start_pose_cfg.get('move_acceleration', 0.5)

    # ====================================================================
    # 3. Create and start the control thread
    # ====================================================================
    ctrl_thread = ControlThread(
        state=state,
        t265_pipeline=t265_pipeline,
        ur_arm=ur_arm,
        glove_receiver=glove_receiver,
        ryhand_ik=ryhand_ik,
        hand_ctrl=hand_ctrl,
        control_dt=control_dt,
    )
    ctrl_thread.set_calibration(base_t265_matrix, base_ur_matrix)

    def _move_to_start_pose():
        """Move UR to start pose, reset hand, recalibrate T265/UR base.
        Must be called while control thread is paused."""
        nonlocal base_t265_matrix, base_ur_matrix

        if ur_arm is not None and start_pose_joints is not None:
            print("[系统] 正在移动至起始位姿...")
            ur_arm.move_j(start_pose_joints,
                          speed=start_move_speed,
                          acceleration=start_move_accel)
            print("[✓] 已到达起始位姿")
        elif start_pose_joints is None:
            print("[!] teleop_config.yaml 中未定义 start_pose.arm_joints，跳过")

        if hand_ctrl is not None:
            with ctrl_thread.hand_hw_lock:
                hand_ctrl.set_angles(
                    np.zeros(15), speed=HAND_RESET_SPEED, radians=True)
            time.sleep(0.3)

        if t265_pipeline is not None and ur_arm is not None:
            time.sleep(0.5)
            frames = _t265_wait_pose()
            pose_frame = frames.get_pose_frame() if frames else None
            if pose_frame:
                pose_data = pose_frame.get_pose_data()
                base_t265_matrix = create_pose_matrix(
                    pose_data.translation, pose_data.rotation)
                base_ur_matrix = pose_vector_to_matrix(
                    ur_arm.rtde_r.getActualTCPPose())
                ctrl_thread.set_calibration(base_t265_matrix, base_ur_matrix)
                print("[✓] T265/UR 基座位姿已重新校准")
            else:
                print("[!] 复位后 T265 基座未更新")

    if not args.dry_run:
        _move_to_start_pose()
    else:
        print("[~] Dry-run: 跳过起始位姿移动")

    # Start control thread after initial calibration
    ctrl_thread.start()
    print("[✓] 控制线程已启动 (独立运行)")

    # ====================================================================
    # 4. 键盘监听
    # ====================================================================
    def on_press(key):
        try:
            if key == keyboard.Key.space:
                is_clutch = state.toggle_clutch()
                if is_clutch:
                    print("\n[状态] 离合已接合(ENGAGED) - 机械臂跟随移动")
                else:
                    print("\n[状态] 离合已解除(DISENGAGED) - 机械臂暂停")
            elif hasattr(key, 'char'):
                if key.char == 's':
                    is_rec = state.toggle_recording()
                    if is_rec:
                        print("\n>>> [REC] 开始录制 episode <<<")
                    else:
                        print("\n>>> [STOP] 结束录制 episode <<<")
                elif key.char == 'd':
                    state.request_discard()
                    print("\n>>> [DISCARD] 请求丢弃 / 删除最近 episode <<<")
                elif key.char == 'r':
                    state.request_reset()
                    print("\n>>> [RESET] 复位机械臂至起始位姿... <<<")
                elif key.char == 'q':
                    state.request_quit()
                    print("\n>>> [QUIT] 准备退出... <<<")
        except Exception:
            pass

    kb_listener = keyboard.Listener(on_press=on_press)
    kb_listener.start()

    print("\n操作指南:")
    print("  按 空格  — 接合/解除离合 (控制机械臂)")
    print("  按 's'   — 开始 / 结束录制一段 episode")
    print("  按 'd'   — 录制中丢弃本段; 空闲时删除 zarr 中最近一段 episode")
    print("  按 'r'   — 复位机械臂至起始位姿 (episode 间复位)")
    print("  按 'q'   — 保存数据集并退出")
    print("  Ctrl+C   — 紧急退出 (会尝试保存)")
    print("=" * 70)

    # ====================================================================
    # 5. zarr 存储初始化
    # ====================================================================
    store = zarr.DirectoryStore(args.output)
    root = zarr.group(store=store, overwrite=True)
    data_group = root.require_group('data')
    meta_group = root.require_group('meta')
    episode_ends_arr = meta_group.zeros('episode_ends', shape=(0,), dtype=np.int64,
                                         compressor=None)

    compressor = numcodecs.Blosc(cname='lz4', clevel=5,
                                  shuffle=numcodecs.Blosc.NOSHUFFLE)
    img_compressor = numcodecs.Blosc(cname='lz4', clevel=5,
                                      shuffle=numcodecs.Blosc.NOSHUFFLE)

    chunk_time = 256
    arr_arm_eef = data_group.zeros('arm_eef_pose', shape=(0, 6), dtype=np.float32,
                                    chunks=(chunk_time, 6), compressor=compressor)
    arr_hand_joints = data_group.zeros('hand_joint_angles', shape=(0, 15),
                                        dtype=np.float32,
                                        chunks=(chunk_time, 15), compressor=compressor)
    arr_camera_env = data_group.zeros('camera_env', shape=(0, img_h, img_w, 3),
                                      dtype=np.uint8,
                                      chunks=(1, img_h, img_w, 3),
                                      compressor=img_compressor)
    arr_camera_wrist = data_group.zeros('camera_wrist', shape=(0, img_h, img_w, 3),
                                         dtype=np.uint8,
                                         chunks=(1, img_h, img_w, 3),
                                         compressor=img_compressor)
    arr_action_hand = data_group.zeros('action_hand_joints', shape=(0, 15),
                                        dtype=np.float32,
                                        chunks=(chunk_time, 15), compressor=compressor)
    arr_action_delta = data_group.zeros('action_eef_delta', shape=(0, 6),
                                          dtype=np.float32,
                                          chunks=(chunk_time, 6), compressor=compressor)

    ep_arm_eef = []
    ep_hand_joints = []
    ep_camera_env = []
    ep_camera_wrist = []
    ep_action_hand = []
    ep_action_delta = []

    total_steps = 0
    total_episodes = 0

    # ====================================================================
    # 6. Main loop — data recording only (control runs in ControlThread)
    # ====================================================================
    print("\n系统就绪，等待按键操作...\n")

    prev_arm_pose_for_delta = None
    last_record_time = 0.0

    def _clear_episode_buffers():
        ep_arm_eef.clear()
        ep_hand_joints.clear()
        ep_camera_env.clear()
        ep_camera_wrist.clear()
        ep_action_hand.clear()
        ep_action_delta.clear()

    def _flush_episode():
        nonlocal total_steps, total_episodes
        ep_len = len(ep_arm_eef)
        if ep_len == 0:
            return

        new_total = total_steps + ep_len

        arr_arm_eef.resize(new_total, 6)
        arr_arm_eef[total_steps:new_total] = np.array(ep_arm_eef, dtype=np.float32)

        arr_hand_joints.resize(new_total, 15)
        arr_hand_joints[total_steps:new_total] = np.array(
            ep_hand_joints, dtype=np.float32)

        arr_camera_env.resize(new_total, img_h, img_w, 3)
        arr_camera_env[total_steps:new_total] = np.array(
            ep_camera_env, dtype=np.uint8)

        arr_camera_wrist.resize(new_total, img_h, img_w, 3)
        arr_camera_wrist[total_steps:new_total] = np.array(
            ep_camera_wrist, dtype=np.uint8)

        arr_action_hand.resize(new_total, 15)
        arr_action_hand[total_steps:new_total] = np.array(
            ep_action_hand, dtype=np.float32)

        arr_action_delta.resize(new_total, 6)
        arr_action_delta[total_steps:new_total] = np.array(
            ep_action_delta, dtype=np.float32)

        episode_ends_arr.resize(total_episodes + 1)
        episode_ends_arr[total_episodes] = new_total

        total_steps = new_total
        total_episodes += 1
        print(f"\n[SAVE] Episode {total_episodes} 已写入 "
              f"({ep_len} 步, 总计 {total_steps} 步)")

    def _pop_last_episode():
        nonlocal total_steps, total_episodes
        if total_episodes <= 0:
            print("\n[INFO] 没有已保存的 episode，无法删除")
            return
        ends = episode_ends_arr[:]
        last_end = int(ends[-1])
        new_total = int(ends[-2]) if total_episodes >= 2 else 0
        removed_1based = total_episodes
        last_len = last_end - new_total

        arr_arm_eef.resize(new_total, 6)
        arr_hand_joints.resize(new_total, 15)
        arr_camera_env.resize(new_total, img_h, img_w, 3)
        arr_camera_wrist.resize(new_total, img_h, img_w, 3)
        arr_action_hand.resize(new_total, 15)
        arr_action_delta.resize(new_total, 6)
        episode_ends_arr.resize(total_episodes - 1)

        total_steps = new_total
        total_episodes -= 1
        print(f"\n[DELETE] 已从 zarr 移除 Episode {removed_1based}（{last_len} 步）"
              f"；剩余 {total_episodes} 段，累计 {total_steps} 步")

    try:
        while not state.quit_requested:
            iter_start = time.time()

            # ------ Discard handling ------
            if state.discard_requested:
                if state.is_recording:
                    _clear_episode_buffers()
                    prev_arm_pose_for_delta = None
                    state.toggle_recording()
                    state.clear_discard()
                    print("[INFO] 当前录制已丢弃（未写入 zarr）")
                else:
                    _clear_episode_buffers()
                    prev_arm_pose_for_delta = None
                    _pop_last_episode()
                    state.clear_discard()

            # ------ Reset handling (pause control thread) ------
            if state.reset_requested:
                if state.is_recording:
                    state.toggle_recording()
                if ep_arm_eef:
                    _flush_episode()
                    _clear_episode_buffers()
                prev_arm_pose_for_delta = None

                ctrl_thread.force_disengage()
                ctrl_thread.pause()

                _move_to_start_pose()

                ctrl_thread.resume()
                state.clear_reset()
                print("[RESET] 复位完成。按空格接合离合，按 's' 开始录制下一段 episode")

            # ------ Episode flush on recording stop ------
            if not state.is_recording:
                if ep_arm_eef: 
                    _flush_episode()
                    _clear_episode_buffers()
                    prev_arm_pose_for_delta = None

                if int(iter_start * 2) % 2 == 0:
                    clutch_str = "ENGAGED" if state.is_clutch_active else "DISENGAGED"
                    overruns = ctrl_thread._loop_overrun_count
                    print(f"\r[IDLE] Ep: {total_episodes} | 步: {total_steps} | "
                          f"离合: {clutch_str} | "
                          f"ctrl overruns: {overruns} | "
                          f"按 's' 开始录制", end="")

            # ------ Record one data frame ------
            if state.is_recording:
                now = time.time()
                if now - last_record_time >= record_interval:
                    last_record_time = now

                    # (1) UR TCP pose (RTDE receive — separate from control's rtde_c)
                    arm_pose = np.zeros(6, dtype=np.float32)
                    if ur_arm is not None:
                        try:
                            tcp_pose = ur_arm.rtde_r.getActualTCPPose()
                            arm_pose = np.array(tcp_pose, dtype=np.float32)
                        except Exception:
                            pass

                    # (2) Hand joint angles (CAN read — locked against control's CAN write)
                    hand_angles_obs = np.zeros(15, dtype=np.float32)
                    if hand_ctrl is not None:
                        try:
                            with ctrl_thread.hand_hw_lock:
                                hand_angles_obs = hand_ctrl.get_angles(
                                    radians=True).astype(np.float32)
                        except Exception:
                            pass

                    # (3) Camera frames (from RealSenseWorker threads — non-blocking)
                    env_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                    if rs_env_camera is not None:
                        frame = rs_env_camera.get_latest_frame()
                        if frame is not None:
                            if frame.shape[0] != img_h or frame.shape[1] != img_w:
                                frame = cv2.resize(frame, (img_w, img_h))
                            env_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    wrist_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                    if rs_wrist_camera is not None:
                        frame = rs_wrist_camera.get_latest_frame()
                        if frame is not None:
                            frame = cv2.rotate(frame, cv2.ROTATE_180)
                            if frame.shape[0] != img_h or frame.shape[1] != img_w:
                                frame = cv2.resize(frame, (img_w, img_h))
                            wrist_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # (4) Hand action (from control thread snapshot)
                    action_hand = ctrl_thread.get_hand_action()

                    # (5) EEF delta
                    action_delta = np.zeros(6, dtype=np.float32)
                    curr_arm_mat = pose_vector_to_matrix(arm_pose)
                    if prev_arm_pose_for_delta is not None:
                        delta_mat = np.linalg.inv(
                            prev_arm_pose_for_delta) @ curr_arm_mat
                        action_delta = np.array(
                            matrix_to_pose_vector(delta_mat), dtype=np.float32)
                    prev_arm_pose_for_delta = curr_arm_mat.copy()

                    ep_arm_eef.append(arm_pose)
                    ep_hand_joints.append(hand_angles_obs)
                    ep_camera_env.append(env_img)
                    ep_camera_wrist.append(wrist_img)
                    ep_action_hand.append(action_hand)
                    ep_action_delta.append(action_delta)

                    step_in_ep = len(ep_arm_eef)
                    clutch_str = "ON" if state.is_clutch_active else "OFF"
                    print(
                        f"\r[REC] Ep {total_episodes + 1} | 步: {step_in_ep} | "
                        f"离合: {clutch_str} | "
                        f"arm: [{arm_pose[0]:.3f},{arm_pose[1]:.3f},"
                        f"{arm_pose[2]:.3f}] | "
                        f"hand: {np.rad2deg(hand_angles_obs[:3]).astype(int)} | "
                        f"delta: [{action_delta[0]:.4f},{action_delta[1]:.4f},"
                        f"{action_delta[2]:.4f}]",
                        end="")

            # Main thread sleeps at record_rate when recording, or a modest
            # idle rate otherwise. The control thread runs independently.
            sleep_target = record_interval if state.is_recording else 0.05
            elapsed = time.time() - iter_start
            if elapsed < sleep_target:
                time.sleep(sleep_target - elapsed)

    except KeyboardInterrupt:
        print("\n\n[!] Ctrl+C 收到，准备保存并退出...")

    # ====================================================================
    # 7. 清理与保存
    # ====================================================================
    state.request_quit()

    print("\n" + "=" * 70)

    if ur_arm is not None:
        try:
            ur_arm.rtde_c.servoStop()
        except Exception:
            pass

    kb_listener.stop()

    print("正在保存数据集...")
    if ep_arm_eef:
        _flush_episode()

    print("正在关闭硬件连接...")

    if hand_ctrl is not None:
        try:
            print("  [系统] 灵巧手归零中...")
            with ctrl_thread.hand_hw_lock:
                hand_ctrl.set_angles(
                    np.zeros(15), speed=HAND_RESET_SPEED, radians=True)
            time.sleep(0.5)
        except Exception as e:
            print(f"  [!] 灵巧手归零失败: {e}")
        try:
            hand_ctrl.close()
            print("  [✓] 灵巧手已断开")
        except Exception:
            pass

    if ur_arm is not None:
        try:
            ur_arm.stop()
            print("  [✓] UR5 已断开")
        except Exception:
            pass

    for cam_label, cam in [("Env", rs_env_camera), ("Wrist", rs_wrist_camera)]:
        if cam is not None:
            try:
                cam.stop()
                print(f"  [✓] {cam_label} 相机已停止")
            except Exception:
                pass

    if t265_pipeline is not None:
        try:
            t265_pipeline.stop()
            print("  [✓] T265 已停止")
        except Exception:
            pass

    if glove_receiver is not None:
        try:
            glove_receiver.close()
        except Exception:
            pass
    if ryhand_ik is not None:
        try:
            ryhand_ik.close()
        except Exception:
            pass

    overruns = ctrl_thread._loop_overrun_count
    print(f"\n  [统计] 控制线程循环超时次数: {overruns}")

    print("\n" + "=" * 70)
    print("数据集摘要:")
    print(f"  路径          : {args.output}")
    print(f"  总 episodes   : {total_episodes}")
    print(f"  总步数        : {total_steps}")
    if total_episodes == 0:
        print("-" * 70)
        print("  未记录任何 episode，退出后将删除空 zarr 目录。")
    else:
        print(f"  记录频率      : {args.record_rate} Hz")
        ends = episode_ends_arr[:]
        lengths = np.diff(np.concatenate([[0], ends]))
        print(f"  Episode 长度  : min={lengths.min()}, max={lengths.max()}, "
              f"mean={lengths.mean():.1f}")
        print("-" * 70)
        print("zarr 数据结构:")
        print(f"  data/arm_eef_pose        : {arr_arm_eef.shape}")
        print(f"  data/hand_joint_angles   : {arr_hand_joints.shape}")
        print(f"  data/camera_env          : {arr_camera_env.shape}")
        print(f"  data/camera_wrist        : {arr_camera_wrist.shape}")
        print(f"  data/action_hand_joints  : {arr_action_hand.shape}")
        print(f"  data/action_eef_delta    : {arr_action_delta.shape}")
        print(f"  meta/episode_ends        : {episode_ends_arr.shape}")
        print("=" * 70)
        print("数据集保存完毕!")

    if total_episodes > 0:
        try:
            sys.path.insert(0, os.path.join(
                project_root, "external", "diffusion_policy"))
            from diffusion_policy.common.replay_buffer import ReplayBuffer
            rb = ReplayBuffer.create_from_path(args.output, mode='r')
            print(f"\n[验证] ReplayBuffer 读取成功:")
            print(f"  n_episodes = {rb.n_episodes}")
            print(f"  n_steps    = {rb.n_steps}")
            print(f"  keys       = {list(rb.keys())}")
        except Exception as e:
            print(f"\n[验证] ReplayBuffer 读取测试: {e}")

    if total_episodes == 0:
        out_abs = os.path.abspath(os.path.expanduser(args.output))
        try:
            shutil.rmtree(out_abs)
            print(f"\n[清理] 已删除空数据集目录: {out_abs}")
        except OSError as e:
            print(f"\n[清理] 删除空 zarr 目录失败 ({out_abs}): {e}")
        print("=" * 70)


if __name__ == "__main__":
    main()
