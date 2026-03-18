#!/usr/bin/env python3
"""
数据采集脚本 — 用于采集符合 diffusion_policy ReplayBuffer (zarr) 格式的遥操作数据集。

控制逻辑参考 run_teleop_manus.py，在遥操作过程中同步采集数据。

数据集结构:
  observation (观测):
    - arm_eef_pose:        UR机械臂末端位姿 [x, y, z, rx, ry, rz]  (6D)
    - hand_joint_angles:   Ruiyan灵巧手关节角 (15D, 弧度)
    - camera_0:            RealSense相机画面  (H, W, 3) uint8

  action (动作):
    - action_hand_joints:  Manus动捕手套 retarget 到灵巧手的关节角 (15D, 弧度)
    - action_eef_delta:    T265相机测量的末端相对位移/旋转增量 [dx,dy,dz,drx,dry,drz] (6D)

存储格式:  zarr DirectoryStore
  root/
    data/
      arm_eef_pose          (N, 6)    float32
      hand_joint_angles     (N, 15)   float32
      camera_0              (N, H, W, 3) uint8
      action_hand_joints    (N, 15)   float32
      action_eef_delta      (N, 6)    float32
    meta/
      episode_ends          (E,)      int64

用法:
    1. 确保 UR 机械臂已上电、Ruiyan 手已通过 CAN 连接、
       RealSense RGB 相机已接入、T265 追踪相机已接入、MANUS SDK 已运行
    2. python scripts/collect_data.py [选项]
    3. 按 空格键 — 接合/解除离合 (控制机械臂跟随)
    4. 按 's'   — 开始 / 结束录制一段 episode
    5. 按 'd'   — 丢弃当前正在录制的 episode
    6. 按 'q'   — 保存数据集并退出
    7. Ctrl+C   — 紧急退出 (会尝试保存)

依赖: pip install zarr numcodecs pyrealsense2 numpy opencv-python pybullet pyzmq pynput python-can scipy
"""

import argparse
import os
import sys
import time
import json
import math
import threading
from datetime import datetime

import cv2
import numpy as np
import zarr
import numcodecs
import pyrealsense2 as rs
import pybullet as p
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

# ============================================================================
# 手套与 IK 配置 (与 run_teleop_manus.py 保持一致)
# ============================================================================
IP_ADDRESS = "tcp://localhost:8000"
LEFT_GLOVE_SN = "4848debd"
RIGHT_GLOVE_SN = "db397317"

NUM_JOINTS = 25
VALUES_PER_JOINT = 7
SHORT_IDX = [23, 24, 4, 5, 9, 10, 19, 20, 14, 15]

CALIBRATION_FILE = os.path.join(project_root, "manus_calibration.json")
FINGER_SCALES = [1.0, 1.0, 1.0, 1.0, 1.0]
WRIST_OFFSET = [0.0, 0.0, 0.0]
FINGER_POS_OFFSETS = [[0.0, 0.0, 0.0] for _ in range(5)]

if os.path.exists(CALIBRATION_FILE):
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            calib = json.load(f)
            FINGER_SCALES = calib.get("FINGER_SCALES", FINGER_SCALES)
            WRIST_OFFSET = calib.get("WRIST_OFFSET", WRIST_OFFSET)
            FINGER_POS_OFFSETS = calib.get("FINGER_POS_OFFSETS", FINGER_POS_OFFSETS)
            print(f"[初始化] 成功加载校准文件: {CALIBRATION_FILE}")
    except Exception as e:
        print(f"[初始化] 加载校准文件失败: {e}")

# ============================================================================
# 坐标系 & 参数 (与 run_teleop_manus.py 保持一致)
# ============================================================================
TRANSLATION_SCALE = 1.0
T265_TO_UR_ALIGN = np.array([
    [ 0,  0, -1,  0],
    [-1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0,  1]
], dtype=np.float64)


# ============================================================================
# 辅助函数 (与 run_teleop_manus.py 保持一致)
# ============================================================================
def create_pose_matrix(translation, rotation_quat):
    """根据 pyrealsense2 的 translation 和 rotation 创建 4x4 齐次矩阵"""
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_quat([
        rotation_quat.x, rotation_quat.y,
        rotation_quat.z, rotation_quat.w
    ]).as_matrix()
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def matrix_to_pose_vector(matrix):
    """4x4 矩阵 → [x,y,z, rx,ry,rz] (旋转向量表示)"""
    pos = matrix[:3, 3]
    rot = R.from_matrix(matrix[:3, :3]).as_rotvec()
    return [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]]


def pose_vector_to_matrix(pose_vec):
    """[x,y,z, rx,ry,rz] → 4x4 齐次矩阵"""
    matrix = np.eye(4)
    matrix[:3, 3] = pose_vec[:3]
    matrix[:3, :3] = R.from_rotvec(pose_vec[3:]).as_matrix()
    return matrix


# ============================================================================
# GloveDataReceiver — 与 run_teleop_manus.py 一致
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
# RYHandIK — 与 run_teleop_manus.py 一致 (DIRECT 模式)
# ============================================================================
class RYHandIK:
    def __init__(self):
        self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)

        urdf_path = os.path.join(project_root, "Bidex_Manus_Teleop", "ryhand_left", "ruihand15z.urdf")
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0],
                                    p.getQuaternionFromEuler([0, 0, np.pi / 2]),
                                    useFixedBase=True)

        self.actuated_joints = []
        self.link_name_to_idx = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            self.link_name_to_idx[info[12].decode('utf-8')] = i
            if info[2] == p.JOINT_REVOLUTE:
                self.actuated_joints.append(i)

        fingertip_links = ["fz15_Link", "fz25_Link", "fz35_Link", "fz45_Link", "fz55_Link"]
        self.ee_indices = [self.link_name_to_idx[name]
                           for name in fingertip_links if name in self.link_name_to_idx]

        self.joint_positions = np.zeros(20)

    def compute_ik(self, glove_data):
        if glove_data is None or 'fingers' not in glove_data:
            return None
        short_skeleton = glove_data['fingers']
        if short_skeleton is None or len(short_skeleton) < 10:
            return None

        hand_pos = []
        for i, pos in enumerate(short_skeleton):
            finger_idx = i // 2
            offset = FINGER_POS_OFFSETS[finger_idx]
            hand_pos.append([pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]])

        tip_indices = [1, 3, 5, 7, 9]
        fingertip_positions = []
        for i, tip_idx in enumerate(tip_indices):
            pos = hand_pos[tip_idx]
            scale = FINGER_SCALES[i]
            fingertip_positions.append([pos[0] * scale, pos[1] * scale, pos[2] * scale])

        num_ee = min(len(fingertip_positions), len(self.ee_indices))
        p.stepSimulation()

        try:
            joint_poses = p.calculateInverseKinematics2(
                self.robot_id,
                self.ee_indices[:num_ee],
                fingertip_positions[:num_ee],
                solver=p.IK_DLS,
                maxNumIterations=100,
                residualThreshold=0.001,
            )
            for i, joint_idx in enumerate(self.actuated_joints):
                if i < len(joint_poses):
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_poses[i],
                        targetVelocity=0,
                        force=500,
                        positionGain=0.3,
                        velocityGain=1,
                    )
            self.joint_positions = np.array(joint_poses[:20], dtype=np.float32)
            return self.joint_positions
        except Exception as e:
            print(f"[IK计算错误] {e}")
            return None

    def close(self):
        p.disconnect(self.physics_client)


# ============================================================================
# IK 20 维 → 灵巧手 15 维关节角映射 (与 run_teleop_manus.py 一致)
# ============================================================================
def ik_to_hand_angles(ik_joints):
    """
    将 IK 模拟的 20 关节 → 真机 15 电机关节角 (弧度)。

    每根手指 4 个仿真关节 → 3 个物理电机通道:
      fzX1 (侧摆) → side_swing     [-30°, +30°]
      fzX2 (MCP)  → proximal_bend  [0°, 90°]
      fzX3+fzX4   → distal_bend    [0°, 75°]  (平均后缩放)
    """
    hand_angles = np.zeros(15, dtype=np.float64)
    limit_side = np.deg2rad(30)
    limit_prox = np.deg2rad(90)
    limit_dist = np.deg2rad(75)

    for finger in range(5):
        ik_base = finger * 4
        hand_base = finger * 3

        # 侧摆 — 仅拇指保留
        side_swing = ik_joints[ik_base]
        if finger == 0:
            hand_angles[hand_base] = np.clip(side_swing, -limit_side, limit_side)
        else:
            hand_angles[hand_base] = 0.0

        # 近端弯折
        proximal = ik_joints[ik_base + 1]
        hand_angles[hand_base + 1] = np.clip(proximal, 0, limit_prox)

        # 远端联动
        pip_angle = ik_joints[ik_base + 2]
        dip_angle = ik_joints[ik_base + 3]
        combined_distal = (pip_angle + dip_angle) * 0.5
        scaled_distal = combined_distal * (75.0 / 90.0)
        hand_angles[hand_base + 2] = np.clip(scaled_distal, 0, limit_dist)

    return hand_angles


# ============================================================================
# 全局状态
# ============================================================================
class RecordingState:
    """管理录制 & 离合 状态的线程安全类"""

    def __init__(self):
        self.lock = threading.Lock()
        self.recording = False
        self.should_quit = False
        self.should_discard = False
        self.clutch_active = False  # 离合: 空格键切换

    def toggle_recording(self):
        with self.lock:
            self.recording = not self.recording
            return self.recording

    def toggle_clutch(self):
        with self.lock:
            self.clutch_active = not self.clutch_active
            return self.clutch_active

    def request_quit(self):
        with self.lock:
            self.should_quit = True

    def request_discard(self):
        with self.lock:
            self.should_discard = True

    def clear_discard(self):
        with self.lock:
            self.should_discard = False

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


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="数据采集脚本 (diffusion_policy 格式)")
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

    # ---------- 输出路径 ----------
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(project_root, "data", f"collected_{timestamp}.zarr")
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    img_w, img_h = args.img_width, args.img_height
    control_dt = 1.0 / args.control_rate
    record_interval = 1.0 / args.record_rate  # 数据记录间隔

    print("=" * 70)
    print("     数据采集系统 (diffusion_policy zarr 格式)")
    print("=" * 70)
    print(f"  输出路径     : {args.output}")
    print(f"  控制频率     : {args.control_rate} Hz")
    print(f"  记录频率     : {args.record_rate} Hz")
    print(f"  图像分辨率   : {img_w} x {img_h}")
    print(f"  Dry-run      : {args.dry_run}")
    print("-" * 70)

    # ====================================================================
    # 1. 初始化所有硬件
    # ====================================================================
    hw_config = get_hardware_config()
    state = RecordingState()

    # --- UR5 机械臂 ---
    ur_arm = None
    if not args.dry_run:
        try:
            ur_arm = UR5Arm(ip=hw_config['ur_arm']['ip'])
            print("[✓] UR5 机械臂已连接")
        except Exception as e:
            print(f"[✗] UR5 连接失败: {e}")
    else:
        print("[~] Dry-run: 跳过 UR5 连接")

    # --- Ruiyan 灵巧手 ---
    hand_ctrl = None
    if not args.dry_run:
        try:
            hand_ctrl = RyHandController(port=hw_config['ruiyan_hand']['port'])
            print("[✓] Ruiyan 灵巧手已连接")
        except Exception as e:
            print(f"[✗] Ruiyan 手连接失败: {e}")
    else:
        print("[~] Dry-run: 跳过 Ruiyan 手连接")

    # --- RealSense RGB 相机 ---
    rs_camera = None
    rs_configs = hw_config.get('cameras', {}).get('realsense', [])
    if rs_configs:
        rs_cfg = rs_configs[0]
        cam_offset = rs_cfg.get('offset', 0)
        cam_zoom = rs_cfg.get('zoom', 1.0)
        cam_id = rs_cfg.get('id', '')
        video_idx = cam_offset

        by_path_dir = "/dev/v4l/by-path"
        if os.path.isdir(by_path_dir):
            for entry in os.listdir(by_path_dir):
                if cam_id in entry and "index0" in entry:
                    real_path = os.path.realpath(os.path.join(by_path_dir, entry))
                    try:
                        video_idx = int(real_path.replace("/dev/video", "")) + cam_offset
                    except ValueError:
                        pass
                    break

        rs_camera = RealSenseWorker(camera_index=video_idx, width=img_w, height=img_h)
        rs_camera.set_zoom(cam_zoom)
        rs_camera.daemon = True
        rs_camera.start()
        for _ in range(50):
            if rs_camera.get_latest_frame() is not None:
                break
            time.sleep(0.1)
        if rs_camera.get_latest_frame() is not None:
            print(f"[✓] RealSense RGB 相机已启动 (video{video_idx})")
        else:
            print(f"[✗] RealSense RGB 相机无法获取画面 (video{video_idx})")
    else:
        print("[✗] 未在配置中找到 RealSense 相机设置")

    # --- T265 追踪相机 ---
    t265_pipeline = None
    try:
        t265_pipeline = rs.pipeline()
        t265_config = rs.config()
        t265_config.enable_stream(rs.stream.pose)
        t265_pipeline.start(t265_config)
        frames = t265_pipeline.wait_for_frames(timeout_ms=3000)
        if frames.get_pose_frame():
            print("[✓] T265 追踪相机已启动")
        else:
            print("[✗] T265 无法获取位姿帧")
            t265_pipeline.stop()
            t265_pipeline = None
    except Exception as e:
        print(f"[✗] T265 连接失败: {e}")
        t265_pipeline = None

    # --- Manus 手套 + IK ---
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
    # 2. T265 + UR 初始基座校准 (与 run_teleop_manus.py 一致)
    # ====================================================================
    base_t265_matrix = None
    base_ur_matrix = None

    if t265_pipeline is not None and ur_arm is not None:
        print("[系统] 正在执行启动校准...")
        time.sleep(1.0)
        frames = t265_pipeline.wait_for_frames()
        pose_frame = frames.get_pose_frame()
        if pose_frame:
            pose_data = pose_frame.get_pose_data()
            base_t265_matrix = create_pose_matrix(pose_data.translation, pose_data.rotation)
            base_ur_matrix = pose_vector_to_matrix(ur_arm.rtde_r.getActualTCPPose())
            print("[✓] 初始校准完成: T265/UR 基座位姿已锁定")
        else:
            print("[✗] 校准失败: 无法获取 T265 初始帧")
    elif t265_pipeline is not None and args.dry_run:
        print("[~] Dry-run: 跳过 T265/UR 校准")
    else:
        print("[!] T265 或 UR 不可用，机械臂遥操作功能受限")

    # ====================================================================
    # 3. 键盘监听
    # ====================================================================
    def on_press(key):
        try:
            if key == keyboard.Key.space:
                is_clutch = state.toggle_clutch()
                if is_clutch:
                    print("\n[状态] 🤖 离合已接合(ENGAGED) - 机械臂跟随移动")
                else:
                    print("\n[状态] ⏸️  离合已解除(DISENGAGED) - 机械臂暂停")
            elif hasattr(key, 'char'):
                if key.char == 's':
                    is_rec = state.toggle_recording()
                    if is_rec:
                        print("\n>>> [REC] 开始录制 episode <<<")
                    else:
                        print("\n>>> [STOP] 结束录制 episode <<<")
                elif key.char == 'd':
                    state.request_discard()
                    print("\n>>> [DISCARD] 丢弃当前 episode <<<")
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
    print("  按 'd'   — 丢弃当前正在录制的 episode")
    print("  按 'q'   — 保存数据集并退出")
    print("  Ctrl+C   — 紧急退出 (会尝试保存)")
    print("=" * 70)

    # ====================================================================
    # 4. zarr 存储初始化
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
    arr_hand_joints = data_group.zeros('hand_joint_angles', shape=(0, 15), dtype=np.float32,
                                        chunks=(chunk_time, 15), compressor=compressor)
    arr_camera = data_group.zeros('camera_0', shape=(0, img_h, img_w, 3), dtype=np.uint8,
                                   chunks=(1, img_h, img_w, 3), compressor=img_compressor)
    arr_action_hand = data_group.zeros('action_hand_joints', shape=(0, 15), dtype=np.float32,
                                        chunks=(chunk_time, 15), compressor=compressor)
    arr_action_delta = data_group.zeros('action_eef_delta', shape=(0, 6), dtype=np.float32,
                                          chunks=(chunk_time, 6), compressor=compressor)

    # episode 内临时缓存
    ep_arm_eef = []
    ep_hand_joints = []
    ep_camera = []
    ep_action_hand = []
    ep_action_delta = []

    total_steps = 0
    total_episodes = 0

    # ====================================================================
    # 5. 主控制循环 (与 run_teleop_manus.py 控制逻辑一致)
    # ====================================================================
    print("\n系统就绪，等待按键操作...\n")

    # 离合切换时的基准位姿
    clutch_t265_matrix = None
    clutch_ur_matrix = None
    was_clutch_active = False

    # 当前 retarget 到的手部关节角 (15D, 弧度)
    current_hand_action = np.zeros(15, dtype=np.float32)

    # 上一次录制步的 UR 末端位姿矩阵 (用于计算 action_eef_delta)
    prev_arm_pose_for_delta = None

    # 手套连接状态
    glove_connected = False

    # 上次记录数据的时间
    last_record_time = 0.0

    last_time = time.time()

    def _flush_episode():
        """将当前 episode 缓冲写入 zarr"""
        nonlocal total_steps, total_episodes

        ep_len = len(ep_arm_eef)
        if ep_len == 0:
            return

        new_total = total_steps + ep_len

        arr_arm_eef.resize(new_total, 6)
        arr_arm_eef[total_steps:new_total] = np.array(ep_arm_eef, dtype=np.float32)

        arr_hand_joints.resize(new_total, 15)
        arr_hand_joints[total_steps:new_total] = np.array(ep_hand_joints, dtype=np.float32)

        arr_camera.resize(new_total, img_h, img_w, 3)
        arr_camera[total_steps:new_total] = np.array(ep_camera, dtype=np.uint8)

        arr_action_hand.resize(new_total, 15)
        arr_action_hand[total_steps:new_total] = np.array(ep_action_hand, dtype=np.float32)

        arr_action_delta.resize(new_total, 6)
        arr_action_delta[total_steps:new_total] = np.array(ep_action_delta, dtype=np.float32)

        episode_ends_arr.resize(total_episodes + 1)
        episode_ends_arr[total_episodes] = new_total

        total_steps = new_total
        total_episodes += 1
        print(f"\n[SAVE] Episode {total_episodes} 已写入 ({ep_len} 步, 总计 {total_steps} 步)")

    try:
        while not state.quit_requested:
            start_time = time.time()
            dt = start_time - last_time
            if dt <= 0:
                dt = 0.001
            last_time = start_time

            # ------ 丢弃请求处理 ------
            if state.discard_requested:
                ep_arm_eef.clear()
                ep_hand_joints.clear()
                ep_camera.clear()
                ep_action_hand.clear()
                ep_action_delta.clear()
                prev_arm_pose_for_delta = None
                if state.is_recording:
                    state.toggle_recording()
                state.clear_discard()
                print("[INFO] Episode 已丢弃")

            # ==============================================================
            # A) 手部控制 — 始终运行 (与 run_teleop_manus.py 一致)
            # ==============================================================
            if glove_receiver is not None and ryhand_ik is not None:
                skeleton_data = glove_receiver.get_data()

                if skeleton_data is not None and not glove_connected:
                    print(f"\n[信息] 成功与 MANUS 穿戴套件建立心跳反馈!")
                    glove_connected = True

                if skeleton_data is not None:
                    ik_positions = ryhand_ik.compute_ik(skeleton_data)
                    if ik_positions is not None:
                        hand_angles = ik_to_hand_angles(ik_positions)
                        current_hand_action = hand_angles.astype(np.float32)

                        # >>> 发送关节角到真实灵巧手 <<<
                        if hand_ctrl is not None:
                            hand_ctrl.set_angles(hand_angles, speed=1000, radians=True)
                else:
                    if not glove_connected:
                        pass  # 静默等待

            # ==============================================================
            # B) 机械臂控制 — 离合激活时跟随 T265 (与 run_teleop_manus.py 一致)
            # ==============================================================
            if t265_pipeline is not None:
                frames = t265_pipeline.poll_for_frames()
                if frames:
                    pose_frame = frames.get_pose_frame()
                    if pose_frame:
                        pose_data = pose_frame.get_pose_data()
                        current_t265_matrix = create_pose_matrix(
                            pose_data.translation, pose_data.rotation)

                        # --- 离合控制 (机械臂跟随) ---
                        if state.is_clutch_active and ur_arm is not None:
                            if not was_clutch_active:
                                # 刚接合: 记录离合基准
                                clutch_t265_matrix = current_t265_matrix.copy()
                                clutch_ur_matrix = pose_vector_to_matrix(
                                    ur_arm.rtde_r.getActualTCPPose())
                                was_clutch_active = True

                            if base_t265_matrix is not None and base_ur_matrix is not None:
                                # 旋转增量
                                rot_delta_t265 = np.linalg.inv(base_t265_matrix) @ current_t265_matrix
                                rot_delta_t265[:3, 3] = 0
                                mapped_rot_delta = T265_TO_UR_ALIGN @ rot_delta_t265 @ T265_TO_UR_ALIGN.T
                                mapped_rot_vec = R.from_matrix(mapped_rot_delta[:3, :3]).as_rotvec()

                                # 自定义旋转轴校准 (与 run_teleop_manus.py 一致)
                                adj_ry = -mapped_rot_vec[1]
                                adj_rx = mapped_rot_vec[2]
                                adj_rz = mapped_rot_vec[0]
                                adjusted_rot_delta_matrix = R.from_rotvec(
                                    [adj_rx, adj_ry, adj_rz]).as_matrix()

                                target_rotation = base_ur_matrix[:3, :3] @ adjusted_rot_delta_matrix

                                # 平移增量
                                trans_delta_t265 = (current_t265_matrix[:3, 3]
                                                    - clutch_t265_matrix[:3, 3])
                                mapped_trans_delta = (T265_TO_UR_ALIGN[:3, :3]
                                                      @ trans_delta_t265 * TRANSLATION_SCALE)

                                target_ur_matrix = np.eye(4)
                                target_ur_matrix[:3, :3] = target_rotation
                                target_ur_matrix[:3, 3] = clutch_ur_matrix[:3, 3] + mapped_trans_delta

                                # >>> 发送 servoL 到机械臂 <<<
                                try:
                                    ur_arm.rtde_c.servoL(
                                        matrix_to_pose_vector(target_ur_matrix),
                                        0.5, 0.5, 0.002, 0.1, 300)
                                except Exception:
                                    pass
                        else:
                            if was_clutch_active:
                                # 刚解除离合
                                if ur_arm is not None:
                                    try:
                                        ur_arm.rtde_c.servoStop()
                                    except Exception:
                                        pass
                                was_clutch_active = False

            # ==============================================================
            # C) 非录制 → 录制刚结束时写入 episode
            # ==============================================================
            if not state.is_recording:
                if ep_arm_eef:
                    _flush_episode()
                    ep_arm_eef.clear()
                    ep_hand_joints.clear()
                    ep_camera.clear()
                    ep_action_hand.clear()
                    ep_action_delta.clear()
                    prev_arm_pose_for_delta = None

                # 空闲时显示状态 (低频刷新避免刷屏)
                if int(start_time * 2) % 2 == 0:
                    clutch_str = "ENGAGED" if state.is_clutch_active else "DISENGAGED"
                    print(f"\r[IDLE] Ep: {total_episodes} | 步: {total_steps} | "
                          f"离合: {clutch_str} | 按 's' 开始录制", end="")

            # ==============================================================
            # D) 录制中: 按 record_rate 频率采样一帧数据
            # ==============================================================
            if state.is_recording:
                now = time.time()
                if now - last_record_time >= record_interval:
                    last_record_time = now

                    # (1) UR 机械臂末端位姿 (观测)
                    arm_pose = np.zeros(6, dtype=np.float32)
                    if ur_arm is not None:
                        try:
                            tcp_pose = ur_arm.rtde_r.getActualTCPPose()
                            arm_pose = np.array(tcp_pose, dtype=np.float32)
                        except Exception:
                            pass

                    # (2) Ruiyan 灵巧手关节角 (观测)
                    hand_angles_obs = np.zeros(15, dtype=np.float32)
                    if hand_ctrl is not None:
                        try:
                            hand_angles_obs = hand_ctrl.get_angles(radians=True).astype(np.float32)
                        except Exception:
                            pass

                    # (3) RealSense 相机画面 (观测)
                    camera_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                    if rs_camera is not None:
                        frame = rs_camera.get_latest_frame()
                        if frame is not None:
                            if frame.shape[0] != img_h or frame.shape[1] != img_w:
                                frame = cv2.resize(frame, (img_w, img_h))
                            camera_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # (4) 手套 retarget 关节角 (动作)
                    action_hand = current_hand_action.copy()

                    # (5) UR 末端位姿增量 (动作) — 从实际 UR TCP 位姿计算
                    #     delta = inv(prev_pose) @ curr_pose  (body-frame 相对变换)
                    action_delta = np.zeros(6, dtype=np.float32)
                    curr_arm_mat = pose_vector_to_matrix(arm_pose)
                    if prev_arm_pose_for_delta is not None:
                        delta_mat = np.linalg.inv(prev_arm_pose_for_delta) @ curr_arm_mat
                        action_delta = np.array(matrix_to_pose_vector(delta_mat),
                                                dtype=np.float32)
                    prev_arm_pose_for_delta = curr_arm_mat.copy()

                    # 追加到 episode 缓冲
                    ep_arm_eef.append(arm_pose)
                    ep_hand_joints.append(hand_angles_obs)
                    ep_camera.append(camera_img)
                    ep_action_hand.append(action_hand)
                    ep_action_delta.append(action_delta)

                    step_in_ep = len(ep_arm_eef)
                    clutch_str = "ON" if state.is_clutch_active else "OFF"
                    print(f"\r[REC] Ep {total_episodes + 1} | 步: {step_in_ep} | "
                          f"离合: {clutch_str} | "
                          f"arm: [{arm_pose[0]:.3f},{arm_pose[1]:.3f},{arm_pose[2]:.3f}] | "
                          f"hand: {np.rad2deg(hand_angles_obs[:3]).astype(int)} | "
                          f"delta: [{action_delta[0]:.4f},{action_delta[1]:.4f},{action_delta[2]:.4f}]",
                          end="")

            # --- 频率控制 ---
            elapsed = time.time() - start_time
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

    except KeyboardInterrupt:
        print("\n\n[!] Ctrl+C 收到，准备保存并退出...")

    # ====================================================================
    # 6. 清理与保存
    # ====================================================================
    print("\n" + "=" * 70)
    print("正在保存数据集...")

    # 写入未完成的 episode
    if ep_arm_eef:
        _flush_episode()

    # 关闭硬件
    print("正在关闭硬件连接...")

    if hand_ctrl is not None:
        try:
            print("[系统] 重置灵巧手至初始状态...")
            hand_ctrl.set_angles(np.zeros(15), speed=500, radians=True)
            time.sleep(0.5)
            hand_ctrl.close()
        except Exception:
            pass

    if ur_arm is not None:
        try:
            ur_arm.stop()
        except Exception:
            pass

    if rs_camera is not None:
        rs_camera.stop()

    if t265_pipeline is not None:
        try:
            t265_pipeline.stop()
        except Exception:
            pass

    if glove_receiver is not None:
        glove_receiver.close()
    if ryhand_ik is not None:
        ryhand_ik.close()

    kb_listener.stop()

    # 打印数据集摘要
    print("\n" + "=" * 70)
    print("数据集摘要:")
    print(f"  路径          : {args.output}")
    print(f"  总 episodes   : {total_episodes}")
    print(f"  总步数        : {total_steps}")
    print(f"  记录频率      : {args.record_rate} Hz")
    if total_episodes > 0:
        ends = episode_ends_arr[:]
        lengths = np.diff(np.concatenate([[0], ends]))
        print(f"  Episode 长度  : min={lengths.min()}, max={lengths.max()}, "
              f"mean={lengths.mean():.1f}")
    print("-" * 70)
    print("zarr 数据结构:")
    print(f"  data/arm_eef_pose        : {arr_arm_eef.shape}")
    print(f"  data/hand_joint_angles   : {arr_hand_joints.shape}")
    print(f"  data/camera_0            : {arr_camera.shape}")
    print(f"  data/action_hand_joints  : {arr_action_hand.shape}")
    print(f"  data/action_eef_delta    : {arr_action_delta.shape}")
    print(f"  meta/episode_ends        : {episode_ends_arr.shape}")
    print("=" * 70)
    print("数据集保存完毕!")

    # 验证
    try:
        sys.path.insert(0, os.path.join(project_root, "external", "diffusion_policy"))
        from diffusion_policy.common.replay_buffer import ReplayBuffer
        rb = ReplayBuffer.create_from_path(args.output, mode='r')
        print(f"\n[验证] ReplayBuffer 读取成功:")
        print(f"  n_episodes = {rb.n_episodes}")
        print(f"  n_steps    = {rb.n_steps}")
        print(f"  keys       = {list(rb.keys())}")
    except Exception as e:
        print(f"\n[验证] ReplayBuffer 读取测试: {e}")


if __name__ == "__main__":
    main()
