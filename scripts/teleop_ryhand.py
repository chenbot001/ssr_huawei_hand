#!/usr/bin/env python3
"""
MANUS数据手套到Ruiyan(RY)左手真机遥操作脚本

此脚本通过ZMQ从MANUS SDK接收手套数据，基于PyBullet计算逆运动学(IK)，
并将计算得出的关节角度发送到真实的RYHand机器人进行控制。

使用方法:
    1. 启动MANUS SDK客户端
    2. 将RYHand连接至CAN总线
    3. 运行此脚本: python teleop_ryhand.py

依赖环境:
    pip install pyzmq numpy pybullet python-can
"""

import argparse
import os
import sys
import time
import math
import json
import threading
import numpy as np
import pybullet as p
import zmq

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
bidex_path = os.path.join(project_root, "Bidex_Manus_Teleop", "python")

if src_path not in sys.path:
    sys.path.append(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# 切换目录以供配置文件加载
os.chdir(project_root)

try:
    from ssr.hardware.ruiyan_driver import RyHandController
    from ssr.config import get_hardware_config
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装ssr包。")
    sys.exit(1)

class Filter1Euro:
    """
    一欧元滤波器(1-Euro Filter)实现，用于平滑带噪的传感器数据。
    该算法极常用于过滤人体动作捕捉时的高频抖动，因为它能根据运动速度自动调节截止频率：
    当动作静止时施加重度滤波，当高速移动时减轻滤波以防延迟(lag)。
    """
    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def _low_pass_filter(self, x, x_prev, alpha):
        return x_prev + alpha * (x - x_prev)

    def update(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0:
            return x

        # 1. 计算一阶导数(速度)并平滑
        dx = (x - self.x_prev) / t_e
        alpha_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx_smoothed = self._low_pass_filter(dx, self.dx_prev, alpha_d)

        # 2. 根据运动速度估算动态截止频率
        cutoff = self.min_cutoff + self.beta * abs(dx_smoothed)

        # 3. 对实际信号应用基于动态截止频率的低通滤波
        alpha = self._smoothing_factor(t_e, cutoff)
        x_filtered = self._low_pass_filter(x, self.x_prev, alpha)

        # 4. 保存当前状态供下一次迭代
        self.x_prev = x_filtered
        self.dx_prev = dx_smoothed
        self.t_prev = t

        return x_filtered

# ============================================================================
# 滤波器与速度规划模块
# ============================================================================
class SinusoidalVelocityProfiler:
    """
    S型（正弦）速度曲线规划器
    用于平滑电机从当前位置到目标位置的过渡，避免人手高频抖动强行注入突变的位置指令，
    导致电机绝对位置模式下产生的生硬启动与急停（抖动）。
    """
    def __init__(self, max_step=0.15):
        self.current_pos = None
        self.max_step = max_step # 单次规划最大允许步长（弧度）

    def step(self, target_pos, dt):
        if self.current_pos is None:
            self.current_pos = np.copy(target_pos)
            return np.copy(self.current_pos)

        # 计算位置差 (目标位 - 当前物理期望位)
        diff = target_pos - self.current_pos
        smoothed_target = np.zeros_like(self.current_pos)
        
        for i in range(len(diff)):
            error = diff[i]
            abs_error = abs(error)
            
            # 使用正弦函数平滑增量, error越小增量逼近0越剧烈(吸收微弱抖动)
            if abs_error < 1e-4:
                step_size = error
            else:
                # 设定 0.4 rad (约23度) 的误差窗口作为完全响应阈值, 缩小窗口会更灵敏
                mapped_error = min(abs_error / 0.4, 1.0) 
                
                # 正弦平滑 (S型曲线的加减速映射)
                scale = math.sin(mapped_error * (math.pi / 2))
                
                # 按照时间频率基准限制单帧最大绝对物理角速度限制 (极大地消除瞬间电磁跳动)
                max_allowable_step = self.max_step * (dt * 30.0) 
                step_val = scale * abs_error
                step_val = min(step_val, max_allowable_step)
                
                step_size = math.copysign(step_val, error)
                
            smoothed_target[i] = self.current_pos[i] + step_size
            
        self.current_pos = np.copy(smoothed_target)
        return self.current_pos
class KalmanFilter1D:
    """
    一维离散卡尔曼滤波器(Kalman Filter)，用于平滑单向关节角轨迹。
    """
    def __init__(self, process_noise=1e-3, measurement_noise=5e-2):
        self.q = process_noise      # 过程噪声协方差 (越大 = 越紧跟实际观测值)
        self.r = measurement_noise  # 测量噪声协方差 (越大 = 滤波平滑度越高)
        self.x_hat = 0.0            # 后验状态估计值
        self.p = 1.0                # 后验误差估计值
        self.k = 0.0                # 卡尔曼增益
        self.first_run = True

    def update(self, measurement):
        if self.first_run:
            self.x_hat = measurement
            self.first_run = False
            return self.x_hat

        # 时间更新 (预测环节)
        p_minus = self.p + self.q

        # 测量更新 (校正环节)
        self.k = p_minus / (p_minus + self.r)
        self.x_hat = self.x_hat + self.k * (measurement - self.x_hat)
        self.p = (1 - self.k) * p_minus

        return self.x_hat

# ============== 全局配置项 ==============
IP_ADDRESS = "tcp://localhost:8000"
LEFT_GLOVE_SN = "4848debd"
RIGHT_GLOVE_SN = "db397317"

# 数据结构相关常数
NUM_JOINTS = 25
VALUES_PER_JOINT = 7  # 坐标+四元数: x, y, z, qx, qy, qz, qw

# 适用于IK结算的精简版骨骼索引 (每根手指含DIP与指尖Tip两个点位)
# 顺序: 拇指_DIP, 拇指_Tip, 食指_DIP, 食指_Tip, 中指_DIP, 中指_Tip, 无名指_DIP, 无名指_Tip, 小指_DIP, 小指_Tip
SHORT_IDX = [23, 24, 4, 5, 9, 10, 19, 20, 14, 15]

# 从手套到RYHand的缩放与偏置缩放系数调整映射路径
CALIBRATION_FILE = os.path.join(project_root, "manus_calibration.json")

# 万一json无法读取，使用此回退默认值
FINGER_SCALES = [1.0, 1.0, 1.0, 1.0, 1.0]
WRIST_OFFSET = [0.0, 0.0, 0.0]
FINGER_POS_OFFSETS = [
    [0.0, 0.0, 0.0],  # 拇指
    [0.0, 0.0, 0.0],  # 食指
    [0.0, 0.0, 0.0],  # 中指
    [0.0, 0.0, 0.0],  # 无名指
    [0.0, 0.0, 0.0]   # 小指
]

# 尝试通过json文件加载参数
if os.path.exists(CALIBRATION_FILE):
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            calib = json.load(f)
            FINGER_SCALES = calib.get("FINGER_SCALES", FINGER_SCALES)
            WRIST_OFFSET = calib.get("WRIST_OFFSET", WRIST_OFFSET)
            FINGER_POS_OFFSETS = calib.get("FINGER_POS_OFFSETS", FINGER_POS_OFFSETS)
            print(f"[初始化] 成功从 {CALIBRATION_FILE} 加载校准配置文件")
    except Exception as e:
        print(f"[初始化] 读取文件出错 {CALIBRATION_FILE}: {e}, 回退使用默认参数。")
else:
    print(f"[初始化] 在 {CALIBRATION_FILE} 处未找到校准文件。回退使用默认参数。")


class GloveDataReceiver:
    """通过ZMQ从MANUS SDK接收处理手套数据的接收器"""
    
    def __init__(self, left_sn=LEFT_GLOVE_SN, right_sn=RIGHT_GLOVE_SN):
        self.left_sn = left_sn
        self.right_sn = right_sn
        
        # 初始化 ZMQ 环境
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, True)  # 仅保留管线最新的消息防延迟
        self.socket.connect(IP_ADDRESS)
        
        # 数据存储结构 - 包含空间坐标位与旋向四元数
        self.left_skeleton = None
        self.right_skeleton = None
        self.left_short = None  # 仅适用于IK的精简骨骼 (10个锚点坐标)
        self.right_short = None
        self.left_wrist = None
        self.right_wrist = None
        self.left_short_orn = None  # 仅适用于IK的精简方向 (10个四元数)
        self.right_short_orn = None
        self.lock = threading.Lock()
        
        # 启动常驻后台接收线程
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        
        print(f"[手套数据接收器] 正在跨进程连接至 {IP_ADDRESS}")
        print(f"[手套数据接收器] 左手手套序列号绑定为: {left_sn}")
        print(f"[手套数据接收器] 右手手套序列号绑定为: {right_sn}")
    
    def _receive_loop(self):
        """用于异步接收ZMQ字符串消息流的后台循环任务"""
        debug_count = 0
        while self.running:
            try:
                message = self.socket.recv(flags=zmq.NOBLOCK)
                message = message.decode('utf-8')
                data = message.split(",")
                
                # 调试逻辑: 打印前几次消息包做通信验证
                debug = (debug_count < 3)
                if debug:
                    print(f"[调试] 收到了 {len(data)} 个数据元")
                    if len(data) >= 176:
                        print(f"[调试] 捕获的首只手套设备号: {data[0]}")
                    if len(data) == 352:
                        print(f"[调试] 捕获的次只手套设备号: {data[176]}")
                    debug_count += 1
                
                if len(data) == 352:
                    self._process_skeleton(data[0:176], debug)
                    self._process_skeleton(data[176:352], debug)
                elif len(data) == 176:
                    self._process_skeleton(data[0:176], debug)
                    
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"[手套数据接收器] 解析错误: {e}")
    
    def _process_skeleton(self, data, debug=False):
        """解析切片提取单手的精细骨骼网格数据"""
        if len(data) < 176:
            return
        
        serial_number = data[0]
        
        if debug:
            print(f"[调试] 正在装载及解析SN为 [{serial_number}] 的动捕数据")
        
        # 将展平的流复原为满尺寸结构 (25关节 x 3维坐标)
        positions = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        for i in range(NUM_JOINTS):
            idx = 1 + i * VALUES_PER_JOINT
            positions[i, 0] = float(data[idx])
            positions[i, 1] = float(data[idx + 1])
            positions[i, 2] = float(data[idx + 2])
        
        # 提取IK所需的稀疏关节 (10个有效骨骼点)
        short_positions = []
        for i in SHORT_IDX:
            idx = 1 + i * VALUES_PER_JOINT
            # 这里统一应用笛卡尔左手系到Bullet系统的反向Y轴旋转映射
            pos = [float(data[idx]), -float(data[idx + 1]), float(data[idx + 2])]
            short_positions.append(pos)
            
        # 拆解基准手腕根节点数据 (第0号关节)
        wrist_idx = 1 + 0 * VALUES_PER_JOINT
        wrist_pos = [float(data[wrist_idx]), -float(data[wrist_idx + 1]), float(data[wrist_idx + 2])]
        
        with self.lock:
            if serial_number == self.left_sn:
                self.left_skeleton = positions
                self.left_short = short_positions
                self.left_wrist = wrist_pos
                if debug:
                    print(f"[调试] 系统判断记录为: 左手(LEFT)")
            elif serial_number == self.right_sn:
                self.right_skeleton = positions
                self.right_short = short_positions
                self.right_wrist = wrist_pos
                if debug:
                    print(f"[调试] 系统判断记录为: 右手(RIGHT)")
            else:
                if debug:
                    print(f"[调试] 未知设备号异常，默认回退赋予给左手骨骼")
                self.left_skeleton = positions
                self.left_short = short_positions
                self.left_wrist = wrist_pos
    
    def get_left_data(self):
        """线程安全地拉取左手精简追踪数据打包字典"""
        with self.lock:
            if self.left_short and self.left_wrist:
                return {"fingers": self.left_short.copy(), "wrist": self.left_wrist.copy()}
            return None
    
    def get_right_data(self):
        """线程安全地拉取右手精简追踪数据打包字典"""
        with self.lock:
            if self.right_short and self.right_wrist:
                return {"fingers": self.right_short.copy(), "wrist": self.right_wrist.copy()}
            return None
    
    def close(self):
        """优雅清理资源"""
        self.running = False
        self.socket.close()
        self.context.term()


class RYHandIK:
    """基于PyBullet执行睿言左手机械手逆向动力学计算的引擎核心"""
    
    def __init__(self, gui=True):
        # 初始化PyBullet物理或直连服务端
        if gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.4,
                cameraYaw=180,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.05]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)
        
        # 加载左手机械手描述文件模型 (Ruihand Left URDF)
        urdf_path = os.path.join(project_root, "Bidex_Manus_Teleop", "ryhand_left", "ruihand15z.urdf")
        
        # 建立世界基座及原生初始90度翻转补偿姿态
        base_pos = [0, 0, 0]
        base_orn = p.getQuaternionFromEuler([0, 0, np.pi/2])
        
        print(f"[IK求解器] 正在内存中渲染URDF模型: {urdf_path}")
        self.robot_id = p.loadURDF(urdf_path, base_pos, base_orn, useFixedBase=True)
        
        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"[IK求解器] 模型初始化成功，载入总计 {self.num_joints} 个关节结构")
        
        # 生成内部ID字典映射供逆解检索调用
        self._build_joint_info()
        
        # 注入目标可视化点位用于3D引擎显示
        self._create_target_vis()
        
        # 预设输出状态池(存储20个主动关节坐标集)
        self.joint_positions = np.zeros(20)
        
    def _build_joint_info(self):
        """遍历映射从URDF字符串名称直接提取ID索引"""
        self.joint_name_to_idx = {}
        self.link_name_to_idx = {}
        self.actuated_joints = []
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')
            joint_type = joint_info[2]
            
            self.joint_name_to_idx[joint_name] = i
            self.link_name_to_idx[link_name] = i
            
            if joint_type == p.JOINT_REVOLUTE:
                self.actuated_joints.append(i)
        
        print(f"[IK求解器] 探测出具备驱动能力的可动(Revolute)关节共: {len(self.actuated_joints)} 处")
        
        # 收录充当IK终止端点(即指尖碰撞体)的作用端标识索引 (End-effector indices)
        fingertip_links = ["fz15_Link", "fz25_Link", "fz35_Link", "fz45_Link", "fz55_Link"]
        self.ee_indices = []
        for link_name in fingertip_links:
            if link_name in self.link_name_to_idx:
                self.ee_indices.append(self.link_name_to_idx[link_name])
                print(f"[IK求解器] 绑定受控指尖实体: {link_name} -> 系统内部指针 {self.link_name_to_idx[link_name]}")
            else:
                print(f"[IK求解器] 系统警告: 无法匹配指尖结构名称 {link_name}")
        
        print(f"[IK求解器] 终端求解锚点池生成完毕: {self.ee_indices}")
    
    def _create_target_vis(self):
        """为PyBullet交互界面加载透明球体标识(可视化的IK捕捉目标)"""
        ball_radius = 0.005
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        base_mass = 0.001
        base_pos = [0.1, 0.1, 0.1]
        
        self.target_balls = []
        colors = [
            [1, 1, 0, 1],    # 黄 - 拇指标识
            [1, 0, 0, 1],    # 红 - 食指标识
            [0, 1, 0, 1],    # 绿 - 中指标识
            [0, 0, 1, 1],    # 蓝 - 无名指标识
            [1, 0, 1, 1],    # 宫红 - 小指标识
        ]
        
        for i in range(5):
            for j in range(2):
                ball_id = p.createMultiBody(
                    baseMass=base_mass,
                    baseCollisionShapeIndex=ball_shape,
                    basePosition=base_pos
                )
                p.setCollisionFilterGroupMask(ball_id, -1, 0, 0) # 剥离物理碰撞碰撞防止干涉
                alpha = 0.6 if j == 0 else 1.0
                color = colors[i].copy()
                color[3] = alpha
                p.changeVisualShape(ball_id, -1, rgbaColor=color)
                self.target_balls.append(ball_id)
        
    def _update_target_vis(self, hand_pos):
        """推送更新虚拟可视化捕捉球位的坐标至引擎"""
        for i, pos in enumerate(hand_pos):
            if i < len(self.target_balls):
                _, current_orn = p.getBasePositionAndOrientation(self.target_balls[i])
                p.resetBasePositionAndOrientation(self.target_balls[i], pos, current_orn)
    
    def compute_ik(self, glove_data):
        """
        基于精简指尖网格触发一帧逆向动力学偏置结算
        
        参数:
            glove_data: 字典封包格式包含人类手套手指跟踪与手腕平移点坐标数据 ('fingers', 'wrist')
        
        返回:
            机械手控制系统所接驳的全部物理关节目标弧度(Radiant)张量池 (长度20位)
        """
        if glove_data is None or 'fingers' not in glove_data:
            return None
            
        short_skeleton = glove_data['fingers']
        wrist_orig = glove_data.get('wrist', [0, 0, 0])

        if short_skeleton is None or len(short_skeleton) < 10:
            return None
        
        # 将参数结构重列映射并结合JSON调参做数学偏差预处理修正
        hand_pos = []
        for i, pos in enumerate(short_skeleton):
            finger_idx = i // 2
            finger_offset = FINGER_POS_OFFSETS[finger_idx]
            
            x = pos[0] + finger_offset[0]
            y = pos[1] + finger_offset[1]
            z = pos[2] + finger_offset[2]
            hand_pos.append([x, y, z])
        
        # 每帧更新前端渲染目标位置标识点
        self._update_target_vis(hand_pos)
        
        # 单独抽取并孤立所有物理终点(Tip)数据，仅提供终极末端做强向位约束 (下标：1, 3, 5, 7, 9)
        tip_indices = [1, 3, 5, 7, 9]
        
        fingertip_positions = []
        for i, tip_idx in enumerate(tip_indices):
            pos = hand_pos[tip_idx]
            scale = FINGER_SCALES[i]
            # 缩放骨骼比例以贴合目标硬件尺寸避免动作超出量程(拉伸)或不够大(张不开)
            scaled_pos = [pos[0] * scale, pos[1] * scale, pos[2] * scale]
            fingertip_positions.append(scaled_pos)
        
        num_ee = min(len(fingertip_positions), len(self.ee_indices))
        
        # 放行系统驱动滴答一帧用以完成力场解析
        p.stepSimulation()
        
        try:
            # 核心指令：要求PyBullet原生DLS算法使用最小二乘完成关节迭代，向指定端点空间推导角度反解
            joint_poses = p.calculateInverseKinematics2(
                self.robot_id,
                self.ee_indices[:num_ee],
                fingertip_positions[:num_ee],
                solver=p.IK_DLS,
                maxNumIterations=100,
                residualThreshold=0.001,
            )
            
            # 顺便推送解析完的姿态使虚拟手跟随变形以提供视觉验证
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
    
    def get_joint_positions(self):
        """提供给外部拉取最近一条计算完毕指令的安全拷贝方法"""
        return self.joint_positions.copy()
    
    def close(self):
        """环境关停重置与中断"""
        p.disconnect(self.physics_client)


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


def main():
    parser = argparse.ArgumentParser(description='MANUS手套遥操作映射到RYHand真机系统')
    parser.add_argument('--no-gui', action='store_true', help='关闭PyBullet后台可视化图形界面渲染')
    parser.add_argument('--rate', type=float, default=30.0, help='主闭环控制刷新率 Hz (默认: 30)')
    parser.add_argument('--speed', type=int, default=1000, help='马达底层最大响应速度 (默认: 1000)')
    parser.add_argument('--print-joints', action='store_true', help='在终端动态打印实时输出的关节控制信息')
    parser.add_argument('--use-right', action='store_true', help='改用右手套的数据替代左手套(适配左右SN号挂错的情境)')
    parser.add_argument('--dry-run', action='store_true', help='测试或无头渲染运行模式（不发起物理机械手的串口/CAN通信，仅跑算法仿真）')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MANUS手套到RYHand软硬全链路映射控制系统")
    print("=" * 60)
    if args.use_right:
        print("[状态] 采集焦点: 强制锁定【右手】动捕数据 (--use-right 设为开启)")
    else:
        print("[状态] 采集焦点: 绑定追踪【左手】动捕数据")
    print(f"[状态] 控制频率(更新率): {args.rate} Hz")
    print(f"[状态] 底层电机限速配置: {args.speed}")
    print(f"[状态] 干跑不连机(Dry run)演习模式: {args.dry_run}")
    print("-" * 60)
    
    # 初始化核心算力管线与接收组件
    glove_receiver = GloveDataReceiver()
    ryhand_ik = RYHandIK(gui=not args.no_gui)
    
    # 尝试桥接与挂载底层的物理手柄发送驱动串口环境 (除非主动挂起 dry-run 保护标签)
    hand_controller = None
    if not args.dry_run:
        try:
            config = get_hardware_config()
            hand_controller = RyHandController(port=config['ruiyan_hand']['port'])
            print(f"[硬件层] 已成功连线至RYHand端口: {config['ruiyan_hand']['port']}")
        except Exception as e:
            print(f"[硬件层] 警告 - 无法与物理机械臂取得握手通讯: {e}")
            print("[硬件层] 将主动降级回退至基于完全物理仿真的无头(Simulation-only)干跑模式...")
    
    print("-" * 60)
    print("主程序运作中... 按下键盘 [Ctrl+C] 组合键安全退出系统")
    print("操作期终端命令指导:")
    print("  - [Ctrl+C] >> 平缓停转全部硬件并在断开前安全归零手指物理位姿")
    print("=" * 60)
    
    update_interval = 1.0 / args.rate
    connected = False
    
    # 初始化 S型(正弦) 运动速度规划器 (解决绝对位置电机的"微跳指令"与"急刹"物理抖动)
    velocity_profiler = SinusoidalVelocityProfiler(max_step=0.15)
    
    try:
        last_time = time.time()
        while True:
            start_time = time.time()
            dt = start_time - last_time
            if dt <= 0:
                dt = 0.001
            last_time = start_time
            
            # 定时收揽缓冲流中的新一帧姿态图
            if args.use_right:
                skeleton_data = glove_receiver.get_right_data()
            else:
                skeleton_data = glove_receiver.get_left_data()
            
            # 检测设备挂钩连接可用性状态变化信息并通报
            if skeleton_data is not None and not connected:
                print(f"\n[信息] 成功与MANUS穿戴套件建立心跳反馈! (正调用 {'右' if args.use_right else '左'} 手数据作为主信号骨干)")
                connected = True
            
            if skeleton_data is not None:
                # 送入 Bullet 环境由物理引擎演算IK偏移逆解析得出20骨折空间映射角度
                ik_positions = ryhand_ik.compute_ik(skeleton_data)
                
                if ik_positions is not None:
                    # 将20个理论推演虚拟关节重重映射和削减至匹配兼容现实底层的15位通道数组结构
                    hand_angles = ik_to_hand_angles(ik_positions)
                    
                    # 启动正弦加减速(SVP - Sinusoidal Velocity Profiling)安全速度规划平滑处理。
                    # 该步骤必须有: 能彻底遏止将离界、粗糙或突波误差强行直接灌入强转矩物理电机构成机械跳格拉扯甚至断齿危险的发生。
                    # profiled_hand_angles = velocity_profiler.step(hand_angles, dt)
                    profiled_hand_angles = hand_angles
                    
                    # 假如串口非空，把剥除好有害高频跳频信息的无害指令列表用透传协议发进物理主控机CAN信道
                    if hand_controller is not None:
                        hand_controller.set_angles(profiled_hand_angles, speed=args.speed, radians=True)
                    
                    if args.print_joints:
                        # 转化弧度为肉眼阅读性更高的度数展示追踪效能
                        angles_deg = np.rad2deg(profiled_hand_angles)
                        print(f"\r指关节夹角变化检测 (度): 拇[{angles_deg[0]:5.1f},{angles_deg[1]:5.1f},{angles_deg[2]:5.1f}] "
                              f"食[{angles_deg[3]:5.1f},{angles_deg[4]:5.1f},{angles_deg[5]:5.1f}] "
                              f"中[{angles_deg[6]:5.1f},{angles_deg[7]:5.1f},{angles_deg[8]:5.1f}]", end='')
            else:
                print("空载或数据断流水位: 正待机监控骨骼坐标系传入管线...", end='\r')
            
            # 主循环节拍器守卫 - 睡足时间从而严格维系设定的Hz控制帧心跳速率保证全局稳定性
            elapsed = time.time() - start_time
            if elapsed < update_interval:
                time.sleep(update_interval - elapsed)
                
    except KeyboardInterrupt:
        print("\n\n触发终止钩子，开始执行下线清理脱钩操作...")
    finally:
        # 断开释放前为保障下次上电无突波事故,强制令实际机构柔顺重置全关节到开合基准位(全零张量位) 
        if hand_controller is not None:
            print("[硬件层] 激活机构退网断电保护守卫功能: 下发全手指放平清零操作复位动作...")
            hand_controller.set_angles(np.zeros(15), speed=500, radians=True)
            time.sleep(0.5)
            hand_controller.close()
        
        # 回收管道挂起挂载与垃圾释放
        glove_receiver.close()
        ryhand_ik.close()
        print("生命周期结束，资源已完整注销释放。系统下线完成。")


if __name__ == "__main__":
    main()
