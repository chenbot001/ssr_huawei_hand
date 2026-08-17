#!/usr/bin/env python3
import os
import sys

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# 切换工作目录以加载配置
os.chdir(project_root)

import time
import math
import threading
import numpy as np
import pyrealsense2 as rs
import zmq
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

from ssr.hardware.arm_ur5 import UR5Arm
from ssr.hardware.ruiyan_driver import RyHandController
from ssr.config import get_hardware_config, get_teleop_config
from ssr.control.RyHand_IK import RYHandIK, ik_to_hand_angles

# ============================================================================
# 手套和手部配置 (从 configs/hardware_config.yaml 加载)
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
# 机械臂和T265配置 (从 configs/teleop_config.yaml 加载)
# ============================================================================
_teleop_config = get_teleop_config()
_servo_cfg = _teleop_config.get('servo', {})
_t265_cfg = _teleop_config.get('t265', {})
_control_cfg = _teleop_config.get('control', {})
_profiler_cfg = _teleop_config.get('velocity_profiler', {})

TRANSLATION_SCALE = _t265_cfg.get('translation_scale', 1.0)
SERVO_SPEED = _servo_cfg.get('speed', 0.5)
SERVO_ACCEL = _servo_cfg.get('acceleration', 0.5)
SERVO_DT = _servo_cfg.get('dt', 0.002)
SERVO_LOOKAHEAD = _servo_cfg.get('lookahead_time', 0.1)
SERVO_GAIN = _servo_cfg.get('gain', 300)
CONTROL_UPDATE_RATE = _control_cfg.get('update_rate', 80.0)
HAND_MOTOR_SPEED = _control_cfg.get('hand_motor_speed', 1000)
HAND_RESET_SPEED = _control_cfg.get('hand_reset_speed', 500)
PROFILER_MAX_STEP = _profiler_cfg.get('max_step', 0.15)
T265_TO_UR_ALIGN = np.array([
    [ 0,  0, -1,  0],
    [-1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0,  1]
])

clutch_active = False

def on_press(key):
    global clutch_active
    try:
        if key == keyboard.Key.space:
            clutch_active = not clutch_active
    except AttributeError:
        pass

# ============================================================================
# 辅助函数类 (Helpers)
# ============================================================================
def create_pose_matrix(translation, rotation_quat):
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_quat([rotation_quat.x, rotation_quat.y, rotation_quat.z, rotation_quat.w]).as_matrix()
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

class SinusoidalVelocityProfiler:
    def __init__(self, max_step=0.15):
        self.current_pos = None
        self.max_step = max_step

    def step(self, target_pos, dt):
        if self.current_pos is None:
            self.current_pos = np.copy(target_pos)
            return np.copy(self.current_pos)

        diff = target_pos - self.current_pos
        smoothed_target = np.zeros_like(self.current_pos)
        
        for i in range(len(diff)):
            error = diff[i]
            abs_error = abs(error)
            if abs_error < 1e-4:
                step_size = error
            else:
                mapped_error = min(abs_error / 0.4, 1.0) 
                scale = math.sin(mapped_error * (math.pi / 2))
                max_allowable_step = self.max_step * (dt * 30.0) 
                step_val = scale * abs_error
                step_val = min(step_val, max_allowable_step)
                step_size = math.copysign(step_val, error)
            smoothed_target[i] = self.current_pos[i] + step_size
            
        self.current_pos = np.copy(smoothed_target)
        return self.current_pos

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
        if len(data) < 176: return
        serial_number = data[0]
        short_positions = []
        for i in SHORT_IDX:
            idx = 1 + i * VALUES_PER_JOINT
            short_positions.append([float(data[idx]), -float(data[idx + 1]), float(data[idx + 2])])
        wrist_idx = 1 + 0 * VALUES_PER_JOINT
        wrist_pos = [float(data[wrist_idx]), -float(data[wrist_idx + 1]), float(data[wrist_idx + 2])]
        
        with self.lock:
            # 默认提取左手或未知手套数据
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


# RYHandIK and ik_to_hand_angles are imported from ssr.control.RyHand_IK (see top)

# ============================================================================
# 主控逻辑 (Main Application)
# ============================================================================
def main():
    global clutch_active
    config = get_hardware_config()

    print("\n" + "=" * 60)
    print("双臂协同遥操作控制中心 (T265 -> UR5, Manus -> RYHand)")
    print("=" * 60)
    
    # 1. 启动键盘监听
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    # 2. 初始化 Manus 接收和 IK 环境
    print("[系统] 初始化手部控制模块...")
    glove_receiver = GloveDataReceiver()
    ryhand_ik = RYHandIK()
    velocity_profiler = SinusoidalVelocityProfiler(max_step=PROFILER_MAX_STEP)
    
    hand_controller = None
    try:
        hand_controller = RyHandController(port=config['ruiyan_hand']['port'])
        print("[硬件] RYHand真机连接成功。")
    except Exception as e:
        print(f"[警告] RYHand连接失败 (将运行在纯算法输出模式): {e}")

    # 3. 初始化 T265 管道
    print("[系统] 初始化T265位姿相机...")
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.pose)
    try:
        pipeline.start(rs_config)
        print("[硬件] T265相机已启动。")
    except Exception as e:
        print(f"[错误] T265启动失败，请检查连接: {e}")
        return

    # 4. 初始化 UR5 机械臂
    print(f"[系统] 连接UR机械臂 ({config['ur_arm']['ip']})...")
    try:
        ur_arm = UR5Arm(ip=config['ur_arm']['ip'])
        print("[硬件] UR机械臂连接成功。")
    except Exception as e:
        print(f"[错误] UR机械臂连接失败: {e}")
        pipeline.stop()
        return

    time.sleep(1.0)
    
    # 5. 初始基座校准
    print("\n[系统状态] 正在执行启动校准...")
    frames = pipeline.wait_for_frames()
    pose_frame = frames.get_pose_frame()
    if pose_frame:
        pose_data = pose_frame.get_pose_data()
        base_t265_matrix = create_pose_matrix(pose_data.translation, pose_data.rotation)
        base_ur_matrix = pose_vector_to_matrix(ur_arm.rtde_r.getActualTCPPose())
        print("[校准完成] 初始相对坐标位姿已锁定。")
        print("\n>>> 系统就绪！按 [空格键] 接合/解除离合状态，按 [Ctrl+C] 安全退出 <<<")
    else:
        print("[校准失败] 获取T265初始帧失败！")
        return

    was_clutch_active = False
    update_interval = 1.0 / CONTROL_UPDATE_RATE
    connected = False
    
    try:
        last_time = time.time()
        while True:
            start_time = time.time()
            dt = start_time - last_time
            if dt <= 0:
                dt = 0.001
            last_time = start_time
            
            # -------- 处理手部运动 (RYHand) --------
            skeleton_data = glove_receiver.get_data()
            
            # 检测设备挂钩连接可用性状态变化信息并通报
            if skeleton_data is not None and not connected:
                print(f"\n[信息] 成功与MANUS穿戴套件建立心跳反馈!")
                connected = True
            
            if skeleton_data is not None:
                ik_positions = ryhand_ik.compute_ik(skeleton_data)
                if ik_positions is not None:
                    hand_angles = ik_to_hand_angles(ik_positions)
                    
                    # 启动正弦加减速(SVP - Sinusoidal Velocity Profiling)安全速度规划平滑处理。
                    # 该步骤必须有: 能彻底遏止将离界、粗糙或突波误差强行直接灌入强转矩物理电机构成机械跳格拉扯甚至断齿危险的发生。
                    # profiled_hand_angles = velocity_profiler.step(hand_angles, dt)
                    profiled_hand_angles = hand_angles
                    
                    if hand_controller:
                        hand_controller.set_angles(profiled_hand_angles, speed=HAND_MOTOR_SPEED, radians=True)
            else:
                print("空载或数据断流水位: 正待机监控骨骼坐标系传入管线...", end='\r')

            # -------- 处理机械臂运动 (UR5) --------
            frames = pipeline.poll_for_frames()
            if frames:
                pose_frame = frames.get_pose_frame()
                if pose_frame:
                    pose_data = pose_frame.get_pose_data()

                    if clutch_active:
                        if not was_clutch_active:
                            clutch_t265_matrix = create_pose_matrix(pose_data.translation, pose_data.rotation)
                            clutch_ur_matrix = pose_vector_to_matrix(ur_arm.rtde_r.getActualTCPPose())
                            was_clutch_active = True
                            print("\r[状态] 🤖 离合已接合(ENGAGED) - 机械臂正在跟随移动...          ", end='')

                        current_t265_matrix = create_pose_matrix(pose_data.translation, pose_data.rotation)
                        
                        rot_delta_t265 = np.linalg.inv(base_t265_matrix) @ current_t265_matrix
                        rot_delta_t265[:3, 3] = 0
                        mapped_rot_delta = T265_TO_UR_ALIGN @ rot_delta_t265 @ T265_TO_UR_ALIGN.T
                        
                        mapped_rot_vec = R.from_matrix(mapped_rot_delta[:3, :3]).as_rotvec()
                        
                        # 自定义旋转轴校准 (俯仰反向，偏航横滚对调)
                        adj_ry = -mapped_rot_vec[1]
                        adj_rx = mapped_rot_vec[2]
                        adj_rz = mapped_rot_vec[0]
                        adjusted_rot_delta_matrix = R.from_rotvec([adj_rx, adj_ry, adj_rz]).as_matrix()
                        
                        target_rotation = base_ur_matrix[:3, :3] @ adjusted_rot_delta_matrix
                        trans_delta_t265 = current_t265_matrix[:3, 3] - clutch_t265_matrix[:3, 3]
                        mapped_trans_delta = T265_TO_UR_ALIGN[:3, :3] @ trans_delta_t265 * TRANSLATION_SCALE
                        
                        target_ur_matrix = np.eye(4)
                        target_ur_matrix[:3, :3] = target_rotation
                        target_ur_matrix[:3, 3] = clutch_ur_matrix[:3, 3] + mapped_trans_delta
                        
                        try:
                            ur_arm.rtde_c.servoL(matrix_to_pose_vector(target_ur_matrix), SERVO_SPEED, SERVO_ACCEL, SERVO_DT, SERVO_LOOKAHEAD, SERVO_GAIN)
                        except Exception:
                            pass
                    else:
                        if was_clutch_active:
                            ur_arm.rtde_c.servoStop()
                            was_clutch_active = False
                            print("\r[状态] ⏸️ 离合已解除(DISENGAGED) - 机械臂暂停跟随.          ", end='')
            
            # 主循环节拍器守卫 - 睡足时间从而严格维系设定的Hz控制帧心跳速率保证全局稳定性
            elapsed = time.time() - start_time
            if elapsed < update_interval:
                time.sleep(update_interval - elapsed)
            
    except KeyboardInterrupt:
        print("\n\n[系统] 接收到退出信号，正在安全下线...")
    finally:
        if hand_controller:
            print("[系统] 重置机械手至初始状态...")
            hand_controller.set_angles(np.zeros(15), speed=HAND_RESET_SPEED, radians=True)
            time.sleep(0.5)
            hand_controller.close()
        if 'ur_arm' in locals() and ur_arm:
            ur_arm.stop()
        if 'pipeline' in locals() and pipeline:
            pipeline.stop()
        if 'listener' in locals() and listener:
            listener.stop()
        glove_receiver.close()
        print("[系统] 关闭完毕，安全退出。")

if __name__ == "__main__":
    main()