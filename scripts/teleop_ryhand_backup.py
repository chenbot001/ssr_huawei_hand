#!/usr/bin/env python3
"""
MANUS Glove to RYHand Real Robot Teleoperation Script

This script receives glove data from MANUS SDK via ZMQ, computes inverse kinematics
using PyBullet, and sends the resulting joint angles to the real RYHand robot.

Usage:
    1. Start MANUS SDK
    2. Connect RYHand to CAN bus
    3. Run this script: python teleop_ryhand.py

Dependencies:
    pip install pyzmq numpy pybullet python-can
"""

import argparse
import os
import sys
import time
import math
import threading
import numpy as np
import pybullet as p
import zmq

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
bidex_path = os.path.join(project_root, "Bidex_Manus_Teleop", "python")

if src_path not in sys.path:
    sys.path.append(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# Change dir for config loading
os.chdir(project_root)

try:
    from ssr.hardware.ruiyan_driver import RyHandController
    from ssr.config import get_hardware_config
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure you have the ssr package installed.")
    sys.exit(1)

class Filter1Euro:
    """
    1-Euro Filter implementation for noisy sensor data smoothing.
    Particularly excellent at filtering out human tracking jitter because it adapts the cutoff 
    frequency based on the velocity of the movement: heavy filtering when standing still, 
    light filtering when moving fast to prevent lag.
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

        # 1. Compute velocity and smooth it
        dx = (x - self.x_prev) / t_e
        alpha_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx_smoothed = self._low_pass_filter(dx, self.dx_prev, alpha_d)

        # 2. Compute the dynamic cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(dx_smoothed)

        # 3. Filter the actual signal
        alpha = self._smoothing_factor(t_e, cutoff)
        x_filtered = self._low_pass_filter(x, self.x_prev, alpha)

        # 4. Save state
        self.x_prev = x_filtered
        self.dx_prev = dx_smoothed
        self.t_prev = t

        return x_filtered

# ============================================================================
# Filters & Velocity Planning
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
    A simple discrete 1D Kalman Filter for smoothing joint angle trajectories.
    """
    def __init__(self, process_noise=1e-3, measurement_noise=5e-2):
        self.q = process_noise      # Process noise covariance (higher = tracks actual measurement closer)
        self.r = measurement_noise  # Measurement noise covariance (higher = smooths more heavily)
        self.x_hat = 0.0            # Posteriori state estimate
        self.p = 1.0                # Posteriori error estimate
        self.k = 0.0                # Kalman gain
        self.first_run = True

    def update(self, measurement):
        if self.first_run:
            self.x_hat = measurement
            self.first_run = False
            return self.x_hat

        # Time Update (Prediction)
        p_minus = self.p + self.q

        # Measurement Update (Correction)
        self.k = p_minus / (p_minus + self.r)
        self.x_hat = self.x_hat + self.k * (measurement - self.x_hat)
        self.p = (1 - self.k) * p_minus

        return self.x_hat

# ============== Configuration ==============
IP_ADDRESS = "tcp://localhost:8000"
LEFT_GLOVE_SN = "4848debd"
RIGHT_GLOVE_SN = "db397317"

# Data structure constants
NUM_JOINTS = 25
VALUES_PER_JOINT = 7  # x, y, z, qx, qy, qz, qw

# Short skeleton indices for IK (DIP and Tip for each finger)
# Order: Thumb_DIP, Thumb_Tip, Index_DIP, Index_Tip, Middle_DIP, Middle_Tip, Ring_DIP, Ring_Tip, Pinky_DIP, Pinky_Tip
SHORT_IDX = [23, 24, 4, 5, 9, 10, 19, 20, 14, 15]

# Scaling factors from glove to RYHand (per-finger: [thumb, index, middle, ring, pinky])
FINGER_SCALES = [1.0, 1.0, 1.0, 1.0, 1.0]


class GloveDataReceiver:
    """Receives glove data from MANUS SDK via ZMQ"""
    
    def __init__(self, left_sn=LEFT_GLOVE_SN, right_sn=RIGHT_GLOVE_SN):
        self.left_sn = left_sn
        self.right_sn = right_sn
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, True)  # Only keep latest message
        self.socket.connect(IP_ADDRESS)
        
        # Data storage - positions and orientations
        self.left_skeleton = None
        self.right_skeleton = None
        self.left_short = None  # Short skeleton positions for IK (10 positions)
        self.right_short = None
        self.left_short_orn = None  # Short skeleton orientations for IK (10 quaternions)
        self.right_short_orn = None
        self.lock = threading.Lock()
        
        # Start receiver thread
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        
        print(f"[GloveReceiver] Connecting to {IP_ADDRESS}")
        print(f"[GloveReceiver] Left glove SN: {left_sn}")
        print(f"[GloveReceiver] Right glove SN: {right_sn}")
    
    def _receive_loop(self):
        """Background thread to receive ZMQ messages"""
        debug_count = 0
        while self.running:
            try:
                message = self.socket.recv(flags=zmq.NOBLOCK)
                message = message.decode('utf-8')
                data = message.split(",")
                
                # Debug: print first few messages
                debug = (debug_count < 3)
                if debug:
                    print(f"[DEBUG] Received {len(data)} data points")
                    if len(data) >= 176:
                        print(f"[DEBUG] First hand SN: {data[0]}")
                    if len(data) == 352:
                        print(f"[DEBUG] Second hand SN: {data[176]}")
                    debug_count += 1
                
                if len(data) == 352:
                    self._process_skeleton(data[0:176], debug)
                    self._process_skeleton(data[176:352], debug)
                elif len(data) == 176:
                    self._process_skeleton(data[0:176], debug)
                    
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"[GloveReceiver] Error: {e}")
    
    def _process_skeleton(self, data, debug=False):
        """Process skeleton data for one hand"""
        if len(data) < 176:
            return
        
        serial_number = data[0]
        
        if debug:
            print(f"[DEBUG] Processing glove SN: {serial_number}")
        
        # Parse full skeleton (25 joints x 3 positions)
        positions = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        for i in range(NUM_JOINTS):
            idx = 1 + i * VALUES_PER_JOINT
            positions[i, 0] = float(data[idx])
            positions[i, 1] = float(data[idx + 1])
            positions[i, 2] = float(data[idx + 2])
        
        # Extract short skeleton for IK (10 positions)
        short_positions = []
        for i in SHORT_IDX:
            idx = 1 + i * VALUES_PER_JOINT
            pos = [float(data[idx]), -float(data[idx + 1]), float(data[idx + 2])]
            short_positions.append(pos)
        
        with self.lock:
            if serial_number == self.left_sn:
                self.left_skeleton = positions
                self.left_short = short_positions
                if debug:
                    print(f"[DEBUG] Stored as LEFT hand")
            elif serial_number == self.right_sn:
                self.right_skeleton = positions
                self.right_short = short_positions
                if debug:
                    print(f"[DEBUG] Stored as RIGHT hand")
            else:
                if debug:
                    print(f"[DEBUG] Unknown SN, storing as LEFT")
                self.left_skeleton = positions
                self.left_short = short_positions
    
    def get_left_skeleton(self):
        """Get left hand short skeleton data for IK"""
        with self.lock:
            return self.left_short.copy() if self.left_short else None
    
    def get_right_skeleton(self):
        """Get right hand short skeleton data for IK"""
        with self.lock:
            return self.right_short.copy() if self.right_short else None
    
    def close(self):
        """Clean up"""
        self.running = False
        self.socket.close()
        self.context.term()


class RYHandIK:
    """PyBullet-based IK for RYHand Left"""
    
    def __init__(self, gui=True):
        # Connect to PyBullet
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
        
        # Load RYHand URDF
        urdf_path = os.path.join(project_root, "Bidex_Manus_Teleop", "ryhand_left", "ruihand15z.urdf")
        
        # Base position and orientation
        base_pos = [0, 0, 0]
        base_orn = p.getQuaternionFromEuler([0, 0, np.pi/2])
        
        print(f"[RYHandIK] Loading URDF: {urdf_path}")
        self.robot_id = p.loadURDF(urdf_path, base_pos, base_orn, useFixedBase=True)
        
        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"[RYHandIK] Loaded RYHand with {self.num_joints} joints")
        
        # Build joint info
        self._build_joint_info()
        
        # Create target visualization balls
        self._create_target_vis()
        
        # Joint positions storage (20 actuated joints)
        self.joint_positions = np.zeros(20)
        
    def _build_joint_info(self):
        """Build joint name to index mapping"""
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
        
        print(f"[RYHandIK] Actuated joints: {len(self.actuated_joints)}")
        
        # Get end effector indices for IK (fingertip links)
        fingertip_links = ["fz15_Link", "fz25_Link", "fz35_Link", "fz45_Link", "fz55_Link"]
        self.ee_indices = []
        for link_name in fingertip_links:
            if link_name in self.link_name_to_idx:
                self.ee_indices.append(self.link_name_to_idx[link_name])
                print(f"[RYHandIK] End effector: {link_name} -> index {self.link_name_to_idx[link_name]}")
            else:
                print(f"[RYHandIK] Warning: Link {link_name} not found")
        
        print(f"[RYHandIK] End effector indices: {self.ee_indices}")
    
    def _create_target_vis(self):
        """Create visualization balls for IK targets"""
        ball_radius = 0.005
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        base_mass = 0.001
        base_pos = [0.1, 0.1, 0.1]
        
        self.target_balls = []
        colors = [
            [1, 1, 0, 1],    # Yellow - Thumb
            [1, 0, 0, 1],    # Red - Index
            [0, 1, 0, 1],    # Green - Middle
            [0, 0, 1, 1],    # Blue - Ring
            [1, 0, 1, 1],    # Magenta - Pinky
        ]
        
        for i in range(5):
            for j in range(2):
                ball_id = p.createMultiBody(
                    baseMass=base_mass,
                    baseCollisionShapeIndex=ball_shape,
                    basePosition=base_pos
                )
                p.setCollisionFilterGroupMask(ball_id, -1, 0, 0)
                alpha = 0.6 if j == 0 else 1.0
                color = colors[i].copy()
                color[3] = alpha
                p.changeVisualShape(ball_id, -1, rgbaColor=color)
                self.target_balls.append(ball_id)
        
    def _update_target_vis(self, hand_pos):
        """Update visualization ball positions"""
        for i, pos in enumerate(hand_pos):
            if i < len(self.target_balls):
                _, current_orn = p.getBasePositionAndOrientation(self.target_balls[i])
                p.resetBasePositionAndOrientation(self.target_balls[i], pos, current_orn)
    
    def compute_ik(self, short_skeleton):
        """
        Compute IK from short skeleton data
        
        Args:
            short_skeleton: List of 10 positions [thumb_dip, thumb_tip, index_dip, index_tip, ...]
        
        Returns:
            Joint positions for RYHand (20 values in radians)
        """
        if short_skeleton is None or len(short_skeleton) < 10:
            return None
        
        # Scale and transform positions
        hand_pos = []
        for i, pos in enumerate(short_skeleton):
            x = pos[0]
            y = pos[1]
            z = pos[2]
            hand_pos.append([x, y, z])
        
        # Update visualization
        self._update_target_vis(hand_pos)
        
        # Extract fingertip positions only (indices 1, 3, 5, 7, 9 are tips)
        tip_indices = [1, 3, 5, 7, 9]
        
        fingertip_positions = []
        for i, tip_idx in enumerate(tip_indices):
            pos = hand_pos[tip_idx]
            scale = FINGER_SCALES[i]
            scaled_pos = [pos[0] * scale, pos[1] * scale, pos[2] * scale]
            fingertip_positions.append(scaled_pos)
        
        num_ee = min(len(fingertip_positions), len(self.ee_indices))
        
        # Compute IK
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
            
            # Update PyBullet visualization
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
            
            # Store joint positions
            self.joint_positions = np.array(joint_poses[:20], dtype=np.float32)
            return self.joint_positions
            
        except Exception as e:
            print(f"[RYHandIK] IK error: {e}")
            return None
    
    def get_joint_positions(self):
        """Get current joint positions"""
        return self.joint_positions.copy()
    
    def close(self):
        """Clean up"""
        p.disconnect(self.physics_client)


def ik_to_hand_angles(ik_joints):
    """
    Convert IK joint positions (20 joints) to RyHandController angles (15 joints).
    
    URDF structure (20 revolute joints, 4 per finger):
        Each finger has: fzX1 (side swing), fzX2, fzX3, fzX4 (bend joints)
        - fzX1: side swing, limits [-0.524, 0.524] rad = [-30°, 30°]
        - fzX2, fzX3, fzX4: bend joints, limits [0, 1.57] rad = [0°, 90°]
        
    IK output indices (20 joints):
        Thumb:  [0]=fz11, [1]=fz12, [2]=fz13, [3]=fz14
        Index:  [4]=fz21, [5]=fz22, [6]=fz23, [7]=fz24
        Middle: [8]=fz31, [9]=fz32, [10]=fz33, [11]=fz34
        Ring:   [12]=fz41, [13]=fz42, [14]=fz43, [15]=fz44
        Pinky:  [16]=fz51, [17]=fz52, [18]=fz53, [19]=fz54
    
    Real hand structure (15 joints, 3 per finger):
        Each finger: [side_swing, proximal_bend, distal_bend]
        - side_swing: [-30°, 30°]
        - proximal_bend: [0°, 90°]
        - distal_bend: [0°, 75°]
    
    Mapping strategy:
        - fzX1 -> side_swing (direct mapping)
        - fzX2 -> proximal_bend (direct mapping, this is the main MCP joint)
        - fzX3 + fzX4 -> distal_bend (combine PIP and DIP joints)
          The real hand's distal motor controls both PIP and DIP together,
          so we sum them and scale to fit within 0-75° limit.
    
    Args:
        ik_joints: numpy array of 20 joint positions in radians
        
    Returns:
        numpy array of 15 joint angles in radians for RyHandController
    """
    hand_angles = np.zeros(15, dtype=np.float64)
    
    # Limits in radians
    limit_side = np.deg2rad(30)    # +/- 30 degrees
    limit_prox = np.deg2rad(90)    # 0-90 degrees
    limit_dist = np.deg2rad(75)    # 0-75 degrees
    
    for finger in range(5):
        ik_base = finger * 4  # IK has 4 joints per finger
        hand_base = finger * 3  # Real hand has 3 joints per finger
        
        # Side swing (fzX1) - direct mapping
        side_swing = ik_joints[ik_base]
        hand_angles[hand_base] = np.clip(side_swing, -limit_side, limit_side)
        
        # Proximal bend (fzX2) - direct mapping to MCP joint
        proximal = ik_joints[ik_base + 1]
        hand_angles[hand_base + 1] = np.clip(proximal, 0, limit_prox)
        
        # Distal bend - combine fzX3 (PIP) and fzX4 (DIP)
        # In the real hand, one motor controls both PIP and DIP
        # Sum the two angles and scale to fit 0-75° range
        pip_angle = ik_joints[ik_base + 2]  # fzX3
        dip_angle = ik_joints[ik_base + 3]  # fzX4
        
        # Combined angle: average of PIP and DIP, then scale
        # Both joints have max 90°, so max combined would be 90°
        # We scale this to fit within 0-75° limit
        combined_distal = (pip_angle + dip_angle) * 0.5
        # Scale from [0, 90°] to [0, 75°]
        scaled_distal = combined_distal * (75.0 / 90.0)
        hand_angles[hand_base + 2] = np.clip(scaled_distal, 0, limit_dist)
    
    return hand_angles


def main():
    parser = argparse.ArgumentParser(description='MANUS Glove to RYHand Real Robot Teleoperation')
    parser.add_argument('--no-gui', action='store_true', help='Run without PyBullet GUI')
    parser.add_argument('--rate', type=float, default=30.0, help='Update rate in Hz (default: 30)')
    parser.add_argument('--speed', type=int, default=1000, help='Motor speed (default: 1000)')
    parser.add_argument('--print-joints', action='store_true', help='Print joint positions')
    parser.add_argument('--use-right', action='store_true', 
                        help='Use right glove data instead of left')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run without connecting to real robot (simulation only)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MANUS Glove to RYHand Real Robot Teleoperation")
    print("=" * 60)
    if args.use_right:
        print("Hand: Using RIGHT glove data (--use-right flag)")
    else:
        print("Hand: Using LEFT glove data")
    print(f"Update rate: {args.rate} Hz")
    print(f"Motor speed: {args.speed}")
    print(f"Dry run mode: {args.dry_run}")
    print("-" * 60)
    
    # Initialize components
    glove_receiver = GloveDataReceiver()
    ryhand_ik = RYHandIK(gui=not args.no_gui)
    
    # Initialize real hand controller (unless dry run)
    hand_controller = None
    if not args.dry_run:
        try:
            config = get_hardware_config()
            hand_controller = RyHandController(port=config['ruiyan_hand']['port'])
            print(f"[Hand] Connected to RYHand on {config['ruiyan_hand']['port']}")
        except Exception as e:
            print(f"[Hand] Failed to connect to real hand: {e}")
            print("[Hand] Continuing in simulation-only mode...")
    
    print("-" * 60)
    print("Running... Press Ctrl+C to exit")
    print("Commands during operation:")
    print("  - Press Ctrl+C to stop and reset hand")
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
            
            # Get glove data
            if args.use_right:
                skeleton = glove_receiver.get_right_skeleton()
            else:
                skeleton = glove_receiver.get_left_skeleton()
            
            # Track connection status
            if skeleton is not None and not connected:
                print(f"\n[INFO] Glove connected! (using {'right' if args.use_right else 'left'} data)")
                connected = True
            
            if skeleton is not None:
                # Compute IK (20 joint positions)
                ik_positions = ryhand_ik.compute_ik(skeleton)
                
                if ik_positions is not None:
                    # Convert to real hand angles (15 joints)
                    hand_angles = ik_to_hand_angles(ik_positions)
                    
                    # 正弦加减速 (Sinusoidal Velocity Profiling) 消除物理机械冲击
                    # This prevents the absolute-position motors from "jumping" to the target
                    profiled_hand_angles = velocity_profiler.step(hand_angles, dt)
                    
                    # Send profiled angles to real hand
                    if hand_controller is not None:
                        hand_controller.set_angles(profiled_hand_angles, speed=args.speed, radians=True)
                    
                    if args.print_joints:
                        # Print in degrees for readability
                        angles_deg = np.rad2deg(profiled_hand_angles)
                        print(f"\rAngles (deg): T[{angles_deg[0]:5.1f},{angles_deg[1]:5.1f},{angles_deg[2]:5.1f}] "
                              f"I[{angles_deg[3]:5.1f},{angles_deg[4]:5.1f},{angles_deg[5]:5.1f}] "
                              f"M[{angles_deg[6]:5.1f},{angles_deg[7]:5.1f},{angles_deg[8]:5.1f}]", end='')
            else:
                print("Waiting for glove data...", end='\r')
            
            # Maintain update rate
            elapsed = time.time() - start_time
            if elapsed < update_interval:
                time.sleep(update_interval - elapsed)
                
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        # Reset hand to zero position before closing
        if hand_controller is not None:
            print("[Hand] Resetting hand to zero position...")
            hand_controller.set_angles(np.zeros(15), speed=500, radians=True)
            time.sleep(0.5)
            hand_controller.close()
        
        glove_receiver.close()
        ryhand_ik.close()
        print("Done.")


if __name__ == "__main__":
    main()
