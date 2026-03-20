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
import numpy as np
import pyrealsense2 as rs
from pynput import keyboard
from ssr.hardware.arm_ur5 import UR5Arm
from ssr.config import get_hardware_config, get_teleop_config
from scipy.spatial.transform import Rotation as R

# ============================================================================
# 配置 (从 configs/ 加载)
# ============================================================================
ROBOT_IP = get_hardware_config()['ur_arm']['ip']

_teleop_config = get_teleop_config()
_servo_cfg = _teleop_config.get('servo', {})
_t265_cfg = _teleop_config.get('t265', {})

TRANSLATION_SCALE = _t265_cfg.get('translation_scale', 1.0)
ROTATION_SCALE = 1.0
SERVO_SPEED = _servo_cfg.get('speed', 0.5)
SERVO_ACCEL = _servo_cfg.get('acceleration', 0.5)
SERVO_DT = _servo_cfg.get('dt', 0.002)
SERVO_LOOKAHEAD = _servo_cfg.get('lookahead_time', 0.1)
SERVO_GAIN = _servo_cfg.get('gain', 300)

# 坐标对齐矩阵: 将T265坐标系映射到UR机械臂坐标系
# 修复前后和上下运动反转的问题:
# Camera Left (-X)    -> maps to -> UR Left (+Y) -> y_new = -x
# Camera Up (+Y)      -> maps to -> UR Up (+Z) -> z_new = y
# Camera Forward (-Z) -> maps to -> UR Fwd (+X) -> x_new = -z
T265_TO_UR_ALIGN = np.array([
    [ 0,  0, -1,  0],
    [-1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0,  1]
])

# ============================================================================
# 全局状态 (Global State)
# ============================================================================
clutch_active = False

def on_press(key):
    global clutch_active
    try:
        if key == keyboard.Key.space:
            clutch_active = not clutch_active
    except AttributeError:
        pass

def create_pose_matrix(translation, rotation_quat):
    """
    根据平移和四元数(x,y,z,w)创建4x4齐次变换矩阵
    pyrealsense2的位姿格式为(x, y, z, w)
    """
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_quat([rotation_quat.x, rotation_quat.y, rotation_quat.z, rotation_quat.w]).as_matrix()
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix

def matrix_to_pose_vector(matrix):
    """
    将4x4矩阵转换为UR风格的位姿向量 [x, y, z, rx, ry, rz]
    """
    pos = matrix[:3, 3]
    rot = R.from_matrix(matrix[:3, :3]).as_rotvec()
    return [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]]

def pose_vector_to_matrix(pose_vec):
    """
    将UR风格的位姿向量 [x,y,z,rx,ry,rz] 转换为4x4矩阵
    """
    matrix = np.eye(4)
    matrix[:3, 3] = pose_vec[:3]
    matrix[:3, :3] = R.from_rotvec(pose_vec[3:]).as_matrix()
    return matrix

def main():
    global clutch_active

    # 1. 设置键盘监听
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("键盘监听已启动。可直接按Ctrl+C退出。")

    # 2. 设置RealSense T265管道
    pipeline = rs.pipeline()
    config = rs.config()
    # 启用T265位姿流
    config.enable_stream(rs.stream.pose)

    try:
        pipeline.start(config)
        print("T265管道已启动。")
    except Exception as e:
        print(f"启动T265管道失败，请检查设备是否连接。错误: {e}")
        return

    # 3. 连接UR5机械臂
    print(f"正在连接UR机械臂 {ROBOT_IP}...")
    try:
        ur_arm = UR5Arm(ip=ROBOT_IP)
    except Exception as e:
        print(f"连接UR机械臂失败: {e}")
        pipeline.stop()
        return

    # 检查是否支持servoL进行笛卡尔空间控制
    if not hasattr(ur_arm.rtde_c, 'servoL'):
        print("警告: rtde_c不支持servoL命令 (请检查ur_rtde是否最新更新)")

    initial_t265_matrix = None
    initial_ur_matrix = None
    was_clutch_active = False

    time.sleep(1.0) # 等待流稳定
    
    # --- 启动校准逻辑 ---
    print("\n[启动校准] 正在读取初始位姿以对齐T265与UR机械臂...")
    frames = pipeline.wait_for_frames()
    pose_frame = frames.get_pose_frame()
    if pose_frame:
        pose_data = pose_frame.get_pose_data()
        base_t265_matrix = create_pose_matrix(pose_data.translation, pose_data.rotation)
        
        current_ur_pose = ur_arm.rtde_r.getActualTCPPose()
        base_ur_matrix = pose_vector_to_matrix(current_ur_pose)
        print("[启动校准] 机械臂与相机构型已锁定。校准全部完成。")
        print("\n>>> 系统已就绪。按空格键(SPACE)启用离合进入跟踪模式。 <<<")
    else:
        print("[启动校准] 未能获取初始T265数据帧！")
        base_t265_matrix = np.eye(4)
        base_ur_matrix = np.eye(4)
    # ---------------------------------

    try:
        while True:
            # 获取T265数据帧
            frames = pipeline.wait_for_frames()
            pose_frame = frames.get_pose_frame()
            if not pose_frame:
                continue

            pose_data = pose_frame.get_pose_data()

            if clutch_active:
                # 4. 离合触发逻辑 (离合接合瞬间)
                if not was_clutch_active:
                    clutch_t265_matrix = create_pose_matrix(pose_data.translation, pose_data.rotation)
                    
                    # 读取机械臂当前实际TCP位姿
                    current_ur_pose = ur_arm.rtde_r.getActualTCPPose()
                    clutch_ur_matrix = pose_vector_to_matrix(current_ur_pose)
                    
                    was_clutch_active = True
                    print("\n>>> 离合已接合(ENGAGED)。运动跟踪激活中。按空格键解除。 <<<")

                # 5. 坐标系转换与计算
                current_t265_matrix = create_pose_matrix(pose_data.translation, pose_data.rotation)
                
                # 计算自脚本启动以来的绝对旋转变化
                rot_delta_t265 = np.linalg.inv(base_t265_matrix) @ current_t265_matrix
                rot_delta_t265[:3, 3] = 0 # 剥离平移部分
                
                # 将旋转映射至UR结构空间
                mapped_rot_delta = T265_TO_UR_ALIGN @ rot_delta_t265 @ T265_TO_UR_ALIGN.T
                
                # --- 旋转坐标轴校正 ---
                # 转换至UR笛卡尔空间的旋转向量
                # [rx, ry, rz] 粗略对应取决于TCP方向的 [横滚Roll, 俯仰Pitch, 偏航Yaw]
                mapped_rot_vec = R.from_matrix(mapped_rot_delta[:3, :3]).as_rotvec()
                
                ur_rx = mapped_rot_vec[0] # 当前X轴旋转 (Roll)
                ur_ry = mapped_rot_vec[1] # 当前Y轴旋转 (Pitch)
                ur_rz = mapped_rot_vec[2] # 当前Z轴旋转 (Yaw)
                
                # 基于用户经验的自定义校准映射:
                # 1. "俯仰轴正确但反向" (假设Pitch在Y轴)
                adj_ry = -ur_ry
                
                # 2. "偏航轴与横滚轴交换"
                adj_rx = ur_rz  # 原Yaw映射至新Roll
                adj_rz = ur_rx  # 原Roll映射至新Yaw
                
                # (注意: 若调整后轴相反请尝试加负号，例如: adj_rx = -ur_rz)
                
                # 利用调整后的旋转向量重新生成旋转增量矩阵
                adjusted_rot_delta_matrix = R.from_rotvec([adj_rx, adj_ry, adj_rz]).as_matrix()
                # ---------------------------------
                
                # 目标旋转姿态 = 机械臂启动基准姿态 + 调整后的绝对旋转增量
                target_rotation = base_ur_matrix[:3, :3] @ adjusted_rot_delta_matrix
                
                # 计算自离合接合起的相对平移量 (保证不连续的局部跟踪)
                trans_delta_t265 = current_t265_matrix[:3, 3] - clutch_t265_matrix[:3, 3]
                
                # 将相机的平移向量同样通过矩阵变换对齐到UR坐标系下
                mapped_trans_delta = T265_TO_UR_ALIGN[:3, :3] @ trans_delta_t265
                mapped_trans_delta *= TRANSLATION_SCALE
                
                # 组合最终的目标位姿 (绝对旋转矩阵 + 本地化相对平移向量)
                target_ur_matrix = np.eye(4)
                target_ur_matrix[:3, :3] = target_rotation
                target_ur_matrix[:3, 3] = clutch_ur_matrix[:3, 3] + mapped_trans_delta
                
                target_pose_vec = matrix_to_pose_vector(target_ur_matrix)

                # 6. 发送指令至UR机械臂 (通过servoL)
                # servoL(位姿, 速度, 加速度, 间隔dt, 预测前瞻延时, 增益系数)
                try:
                    ur_arm.rtde_c.servoL(target_pose_vec, SERVO_SPEED, SERVO_ACCEL, SERVO_DT, SERVO_LOOKAHEAD, SERVO_GAIN)
                except Exception as e:
                    print(f"发送servoL命令出错: {e}")
                    
            else:
                if was_clutch_active:
                    # 平滑停止伺服运动
                    ur_arm.rtde_c.servoStop()
                    was_clutch_active = False
                    print("\n>>> 离合已解除(DISENGAGED)。跟踪暂停中。按空格键恢复。 <<<")  
                
            time.sleep(0.002) # ~500Hz 控制循环匹配 RealSense / UR RTDE
            
    except KeyboardInterrupt:
        print("\n正在退出...")
    finally:
        if 'ur_arm' in locals() and ur_arm is not None:
            ur_arm.stop()
        if 'pipeline' in locals() and pipeline is not None:
            pipeline.stop()
        if 'listener' in locals() and listener is not None:
            listener.stop()
        print("系统关闭完毕。")

if __name__ == "__main__":
    main()
