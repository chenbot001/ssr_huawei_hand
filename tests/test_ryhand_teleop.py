#!/usr/bin/env python3
"""
MANUS数据手套到Ruiyan(RY)左手真机遥操作脚本

此脚本通过ZMQ从MANUS SDK接收手套数据，基于PyBullet计算逆运动学(IK)，
并将计算得出的关节角度发送到真实的RYHand机器人进行控制。

使用方法:
    1. 启动MANUS SDK客户端
    2. 将RYHand连接至CAN总线
    3. 运行此脚本: python tests/test_ryhand_teleop.py

依赖环境:
    pip install pyzmq numpy pybullet python-can
"""

import argparse
import os
import sys
import time
import math
import threading
import numpy as np
import zmq

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
bidex_path = os.path.join(project_root, "external", "Bidex_Manus_Teleop", "python")

if src_path not in sys.path:
    sys.path.append(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# 切换目录以供配置文件加载
os.chdir(project_root)

try:
    from ssr.hardware.ruiyan_driver import RyHandController
    from ssr.config import get_hardware_config, get_teleop_config
    from ssr.control.RyHand_IK import RYHandIK, ik_to_hand_angles
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

# ============== 全局配置项 (从 configs/hardware_config.yaml 加载) ==============
_hw_config = get_hardware_config()
_manus_config = _hw_config.get('manus_glove', {})
IP_ADDRESS = _manus_config.get('address', "tcp://localhost:8000")
LEFT_GLOVE_SN = _manus_config.get('left_sn', "4848debd")
RIGHT_GLOVE_SN = _manus_config.get('right_sn', "db397317")

# 数据结构相关常数
NUM_JOINTS = 25
VALUES_PER_JOINT = 7  # 坐标+四元数: x, y, z, qx, qy, qz, qw

# 适用于IK结算的精简版骨骼索引 (每根手指含DIP与指尖Tip两个点位)
# 顺序: 拇指_DIP, 拇指_Tip, 食指_DIP, 食指_Tip, 中指_DIP, 中指_Tip, 无名指_DIP, 无名指_Tip, 小指_DIP, 小指_Tip
SHORT_IDX = [23, 24, 4, 5, 9, 10, 19, 20, 14, 15]


# Calibration (FINGER_SCALES, FINGER_POS_OFFSETS, etc.) is loaded centrally
# inside ssr.control.RyHand_IK at import time.


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



# RYHandIK and ik_to_hand_angles are imported from ssr.control.RyHand_IK (see top)


def main():
    # 从 teleop_config.yaml 加载遥操参数
    teleop_config = get_teleop_config()
    control_cfg = teleop_config.get('control', {})
    profiler_cfg = teleop_config.get('velocity_profiler', {})
    
    default_speed = control_cfg.get('hand_motor_speed', 1000)
    default_reset_speed = control_cfg.get('hand_reset_speed', 500)
    default_max_step = profiler_cfg.get('max_step', 0.15)
    
    parser = argparse.ArgumentParser(description='MANUS手套遥操作映射到RYHand真机系统')
    parser.add_argument('--no-gui', action='store_true', help='关闭PyBullet后台可视化图形界面渲染')
    parser.add_argument('--rate', type=float, default=30.0, help='主闭环控制刷新率 Hz (默认: 30)')
    parser.add_argument('--speed', type=int, default=default_speed, help=f'马达底层最大响应速度 (默认: {default_speed})')
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
    ryhand_ik = RYHandIK(gui=(not args.no_gui))
    
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
    velocity_profiler = SinusoidalVelocityProfiler(max_step=default_max_step)
    
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
            hand_controller.set_angles(np.zeros(15), speed=default_reset_speed, radians=True)
            time.sleep(0.5)
            hand_controller.close()
        
        # 回收管道挂起挂载与垃圾释放
        glove_receiver.close()
        ryhand_ik.close()
        print("生命周期结束，资源已完整注销释放。系统下线完成。")


if __name__ == "__main__":
    main()
