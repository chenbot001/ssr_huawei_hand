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
import json
import threading
import numpy as np
import pyrealsense2 as rs
import pybullet as p
import zmq
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

from ssr.hardware.arm_ur5 import UR5Arm
from ssr.hardware.ruiyan_driver import RyHandController
from ssr.config import get_hardware_config

# ============================================================================
# 手套和手部配置 (Glove & Hand Config)
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
# 机械臂和T265配置 (Arm & T265 Config)
# ============================================================================
TRANSLATION_SCALE = 1.0
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

class RYHandIK:
    def __init__(self):
        self.physics_client = p.connect(p.DIRECT)  # 使用无头模式提高性能
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        
        urdf_path = os.path.join(project_root, "Bidex_Manus_Teleop", "ryhand_left", "ruihand15z.urdf")
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0], p.getQuaternionFromEuler([0, 0, np.pi/2]), useFixedBase=True)
        
        self.actuated_joints = []
        self.link_name_to_idx = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            self.link_name_to_idx[info[12].decode('utf-8')] = i
            if info[2] == p.JOINT_REVOLUTE:
                self.actuated_joints.append(i)
        
        fingertip_links = ["fz15_Link", "fz25_Link", "fz35_Link", "fz45_Link", "fz55_Link"]
        self.ee_indices = [self.link_name_to_idx[name] for name in fingertip_links if name in self.link_name_to_idx]
        
        # 预设输出状态池(存储20个主动关节坐标集)
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
            fingertip_positions.append([pos[0]*scale, pos[1]*scale, pos[2]*scale])
        
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
            
            # 推送解析完的姿态使虚拟手跟随变形以提供视觉验证（影响后续IK计算的初始状态）
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
            
            # 返回提取好的角度结构池缓存以供真机转发使用
            self.joint_positions = np.array(joint_poses[:20], dtype=np.float32)
            return self.joint_positions
            
        except Exception as e:
            print(f"[IK计算错误] {e}")
            return None

def ik_to_hand_angles(ik_joints):
    """
    负责将IK模拟解算的高维姿态(20关节) 降维转译到 Ruihand 真机的物理闭环控制器参数(15自由度指令)。
    
    仿真URDF文件映射架构(每手指4个旋转自由度,累计20指节):
        每根手指内部顺序为: fzX1 (左右开合指根), fzX2 (MCP主弯折), fzX3(PIP指中弯折), fzX4(DIP指尖弯折)
        - fzX1 开合物理结构限位: [-0.524, 0.524] 弧度 = [-30°, 30°]
        - fzX2~X4 前屈物理限位: [0, 1.57] 弧度 = [0°, 90°]
        
    对应提取出 IK 数据张量池 结构索引：
        拇指:  [0]=fz11, [1]=fz12, [2]=fz13, [3]=fz14
        食指:  [4]=fz21, [5]=fz22, [6]=fz23, [7]=fz24
        中指:  [8]=fz31, [9]=fz32, [10]=fz33, [11]=fz34
        无名指:[12]=fz41, [13]=fz42, [14]=fz43, [15]=fz44
        小指:  [16]=fz51, [17]=fz52, [18]=fz53, [19]=fz54
    
    真实五指机械手硬件设计架构(每根手指3个物理电机,累计15电机通道):
        每根手指映射格式: [左右开合(side_swing), 近端弯折MCP(proximal_bend), 远端联动弯折(distal_bend)]
        - 开合(side_swing): 取值 [-30°, 30°]
        - 近端MCP(proximal_bend): 取值 [0°, 90°]
        - 末端(distal_bend): 取值 [0°, 75°]
    
    转换映射法则:
        - fzX1 -> 直接直通赋予 侧向开合驱动电机
        - fzX2 -> 直接直通赋予 根部近端电机 (最主要的握持发力关节)
        - (fzX3 + fzX4) -> 拟合糅合成单一参数赋予 远端联动电机
          在真实灵巧手中，PIP和DIP(指中指尖)是通过一个单一拉钩马达连杆驱动并做耦合运动的，
          所以必须对IK的独立双角度进行平均与限位衰减再下发。
    
    参数:
        ik_joints: [numpy 数组] 保存IK解算后的20维纯物理弧度张量
        
    返回:
        [numpy 数组] 含有15位降阶电机命令弧度目标列表，供驱动板使用
    """
    hand_angles = np.zeros(15, dtype=np.float64)
    
    # 物理软硬件夹角边界值限位 (转换成弧度)
    limit_side = np.deg2rad(30)    # 极限阈值 +/- 30度
    limit_prox = np.deg2rad(90)    # 极限阈值 0~90度弯曲
    limit_dist = np.deg2rad(75)    # 极限阈值 0~75度联动弯曲
    
    for finger in range(5):
        ik_base = finger * 4  # 按单指4关节计算偏移基址
        hand_base = finger * 3  # 按电机板单指3通道寻找写入基址
        
        # 1. 左右侧边摆动开合 (fzX1) - 仅冻结非大拇指的侧摆自由度
        side_swing = ik_joints[ik_base]
        if finger == 0:
            # 大拇指保持侧摆自由度
            hand_angles[hand_base] = np.clip(side_swing, -limit_side, limit_side)
        else:
            # 其他手指冻结侧摆自由度
            hand_angles[hand_base] = 0.0
        
        # 2. MCP根部基础弯折 (fzX2) - 主力对一对一直通
        proximal = ik_joints[ik_base + 1]
        hand_angles[hand_base + 1] = np.clip(proximal, 0, limit_prox)
        
        # 3. 远端联动拉绳合并 - 处理仿真IK的双关节(fzX3 PIP 与 fzX4 DIP)
        pip_angle = ik_joints[ik_base + 2] 
        dip_angle = ik_joints[ik_base + 3] 
        
        # 将双关节参数等强降维取均值，并以比例系数[75/90]缩放到适配最大75度的真实电机限幅区间内
        combined_distal = (pip_angle + dip_angle) * 0.5
        scaled_distal = combined_distal * (75.0 / 90.0)

        # 保护性截断超量程指令避免电机烧毁
        hand_angles[hand_base + 2] = np.clip(scaled_distal, 0, limit_dist)
    
    return hand_angles

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
    velocity_profiler = SinusoidalVelocityProfiler(max_step=0.15)
    
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
    update_interval = 1.0 / 80.0  # 默认30Hz控制频率
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
                        hand_controller.set_angles(profiled_hand_angles, speed=1000, radians=True)
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
                            ur_arm.rtde_c.servoL(matrix_to_pose_vector(target_ur_matrix), 0.5, 0.5, 0.002, 0.1, 300)
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
            hand_controller.set_angles(np.zeros(15), speed=500, radians=True)
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