#!/usr/bin/env python3
"""
数据回放脚本 — 在真实硬件上回放 collect_data.py 采集的轨迹。

流程:
  1. 加载 zarr 数据集, 选择要回放的 episode
  2. 将 UR 机械臂移动到该 episode 的初始末端位姿 (arm_eef_pose[0])
  3. 将 Ruiyan 灵巧手设置为初始关节角 (hand_joint_angles[0])
  4. 等待用户确认安全后，开始逐帧执行:
       - 机械臂: 通过 servoL 跟随记录的 arm_eef_pose 轨迹 (或累积 action_eef_delta)
       - 灵巧手: 通过 set_angles 执行 action_hand_joints 动作
  5. 同时在 OpenCV 窗口中同步显示记录的相机画面与实时状态

用法:
    python scripts/replay_data.py <zarr_path> [选项]

示例:
    # 回放第 1 个 episode (默认)
    python scripts/replay_data.py data/collected_20260317.zarr

    # 回放第 3 个 episode, 0.5 倍速
    python scripts/replay_data.py data/collected_20260317.zarr -e 3 --speed 0.5

    # 使用 action delta 模式回放 (而非直接跟踪观测位姿)
    python scripts/replay_data.py data/collected_20260317.zarr --action-mode

    # 仅回放灵巧手, 不动机械臂
    python scripts/replay_data.py data/collected_20260317.zarr --hand-only

    # 打印数据集摘要 (不执行回放)
    python scripts/replay_data.py data/collected_20260317.zarr --info

依赖: pip install zarr numpy opencv-python scipy
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# ============================================================================
# 路径设置
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
dp_path = os.path.join(project_root, "external", "diffusion_policy")

for p in [src_path, project_root, dp_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(project_root)

from ssr.hardware.arm_ur5 import UR5Arm
from ssr.hardware.ruiyan_driver import RyHandController
from ssr.config import get_hardware_config, get_teleop_config
from diffusion_policy.common.replay_buffer import ReplayBuffer

# ============================================================================
# 坐标变换工具
# ============================================================================

def pose_vec_to_matrix(pose_vec):
    """[x,y,z,rx,ry,rz] → 4x4 齐次变换矩阵"""
    mat = np.eye(4)
    mat[:3, 3] = pose_vec[:3]
    mat[:3, :3] = R.from_rotvec(pose_vec[3:6]).as_matrix()
    return mat


def matrix_to_pose_vec(mat):
    """4x4 齐次变换矩阵 → [x,y,z,rx,ry,rz]"""
    pos = mat[:3, 3]
    rot = R.from_matrix(mat[:3, :3]).as_rotvec()
    return np.concatenate([pos, rot]).astype(np.float64)


# ============================================================================
# 数据集工具
# ============================================================================

def print_dataset_info(rb, zarr_path):
    """打印数据集摘要"""
    print("=" * 70)
    print(f"  数据集: {zarr_path}")
    print("=" * 70)
    print(f"  总 episodes : {rb.n_episodes}")
    print(f"  总步数      : {rb.n_steps}")
    print(f"  数据 keys   : {list(rb.keys())}")
    print("-" * 70)

    if rb.n_episodes > 0:
        ends = rb.episode_ends[:]
        starts = np.concatenate([[0], ends[:-1]])
        lengths = ends - starts
        print("  Episode 列表:")
        for i in range(rb.n_episodes):
            s, e = starts[i], ends[i]
            arm_start = rb['arm_eef_pose'][s] if 'arm_eef_pose' in rb.data else None
            info = ""
            if arm_start is not None:
                info = (f"  起始位姿=[{arm_start[0]:.3f}, {arm_start[1]:.3f}, "
                        f"{arm_start[2]:.3f}]")
            print(f"    Episode {i+1}: {lengths[i]:5d} 步{info}")
    print("=" * 70)


def get_episode_data(rb, episode_idx):
    """提取指定 episode 的所有数据"""
    ep_slice = rb.get_episode_slice(episode_idx)
    data = {}
    for key in rb.keys():
        data[key] = rb[key][ep_slice]
    ep_len = ep_slice.stop - ep_slice.start
    return data, ep_len


# ============================================================================
# 主回放逻辑
# ============================================================================

def replay_episode(ur_arm, hand_ctrl, data, ep_len, args):
    """
    在真实硬件上回放一段 episode 的轨迹。

    参数:
        ur_arm:    UR5Arm 实例 (如果 --hand-only 则为 None)
        hand_ctrl: RyHandController 实例 (如果 --arm-only 则为 None)
        data:      episode 数据字典
        ep_len:    episode 长度(步数)
        args:      命令行参数
    """
    has_arm_pose = 'arm_eef_pose' in data
    has_hand_angles = 'hand_joint_angles' in data
    has_action_hand = 'action_hand_joints' in data
    has_action_delta = 'action_eef_delta' in data
    has_camera_env = 'camera_env' in data
    has_camera_wrist = 'camera_wrist' in data
    has_camera = has_camera_env or has_camera_wrist

    control_dt = 1.0 / args.rate
    speed_factor = args.speed

    # ====================================================================
    # 1. 移动到初始状态
    # ====================================================================
    print("\n[Phase 1] 移动到初始状态...")

    # --- 机械臂: moveL 到初始位姿 ---
    if ur_arm is not None and has_arm_pose:
        init_pose = data['arm_eef_pose'][0].astype(np.float64).tolist()
        print(f"  机械臂初始位姿: [{init_pose[0]:.4f}, {init_pose[1]:.4f}, "
              f"{init_pose[2]:.4f}, {init_pose[3]:.4f}, {init_pose[4]:.4f}, "
              f"{init_pose[5]:.4f}]")
        print("  正在移动机械臂到初始位姿 (moveL)...")
        try:
            ur_arm.rtde_c.moveL(init_pose, speed=0.1, acceleration=0.3)
            print("  [✓] 机械臂已到位")
        except Exception as e:
            print(f"  [✗] 机械臂 moveL 失败: {e}")
            print("      尝试使用 servoL 缓慢接近...")
            try:
                for _ in range(200):
                    ur_arm.rtde_c.servoL(init_pose, 0.1, 0.3, 0.008, 0.1, 300)
                    time.sleep(0.008)
                ur_arm.rtde_c.servoStop()
                print("  [✓] 机械臂已到位 (servoL)")
            except Exception as e2:
                print(f"  [✗] servoL 也失败: {e2}")
                return False

    # --- 灵巧手: 设置初始关节角 ---
    if hand_ctrl is not None and has_hand_angles:
        init_hand = data['hand_joint_angles'][0].astype(np.float64)
        print(f"  灵巧手初始角度 (deg): "
              f"{np.rad2deg(init_hand[:3]).astype(int)} / "
              f"{np.rad2deg(init_hand[3:6]).astype(int)} / ...")
        print("  正在设置灵巧手到初始角度...")
        try:
            hand_ctrl.set_angles(init_hand, speed=300, radians=True)
            time.sleep(1.0)  # 等待手部运动到位
            print("  [✓] 灵巧手已到位")
        except Exception as e:
            print(f"  [✗] 灵巧手设置失败: {e}")

    # ====================================================================
    # 2. 等待用户确认
    # ====================================================================
    print("\n" + "-" * 50)
    print(f"  初始状态就绪。即将回放 {ep_len} 步轨迹")
    print(f"  回放速度: x{speed_factor}  |  控制频率: {args.rate} Hz")
    print(f"  机械臂模式: {'action delta 累积' if args.action_mode else '直接跟踪观测位姿'}")
    print("-" * 50)

    if not args.yes:
        print("\n  ⚠ 请确认机器人周围安全！")
        ans = input("  输入 'y' 开始回放, 'q' 取消: ").strip().lower()
        if ans != 'y':
            print("  已取消回放。")
            return False

    # ====================================================================
    # 3. 逐帧回放
    # ====================================================================
    print(f"\n[Phase 2] 开始回放... (按 Ctrl+C 紧急停止)")

    # 如果使用 action_mode, 需要从初始位姿累积 delta
    if args.action_mode and has_arm_pose:
        current_pose_mat = pose_vec_to_matrix(data['arm_eef_pose'][0].astype(np.float64))

    # 可视化窗口
    show_vis = has_camera and not args.no_vis
    window_name = "Replay" if show_vis else None
    paused = False

    try:
        for step in range(ep_len):
            loop_start = time.time()

            # ------ 机械臂控制 ------
            if ur_arm is not None and not args.hand_only:
                if args.action_mode and has_action_delta:
                    # 模式 B: 累积 action_eef_delta
                    delta = data['action_eef_delta'][step].astype(np.float64)
                    delta_mat = np.eye(4)
                    delta_mat[:3, 3] = delta[:3]
                    if np.linalg.norm(delta[3:6]) > 1e-8:
                        delta_mat[:3, :3] = R.from_rotvec(delta[3:6]).as_matrix()
                    current_pose_mat = current_pose_mat @ delta_mat
                    target_pose = matrix_to_pose_vec(current_pose_mat).tolist()
                elif has_arm_pose:
                    # 模式 A (默认): 直接跟踪记录的观测位姿
                    target_pose = data['arm_eef_pose'][step].astype(np.float64).tolist()
                else:
                    target_pose = None

                if target_pose is not None:
                    try:
                        ur_arm.rtde_c.servoL(
                            target_pose, 0.5, 0.5,
                            control_dt / speed_factor,
                            0.1, 300
                        )
                    except Exception as e:
                        print(f"\n  [WARN] servoL 第 {step} 步出错: {e}")

            # ------ 灵巧手控制 ------
            if hand_ctrl is not None and not args.arm_only:
                if has_action_hand:
                    target_hand = data['action_hand_joints'][step].astype(np.float64)
                elif has_hand_angles:
                    target_hand = data['hand_joint_angles'][step].astype(np.float64)
                else:
                    target_hand = None

                if target_hand is not None:
                    try:
                        hand_ctrl.set_angles(target_hand, speed=500,
                                             radians=True, wait=0.0)
                    except Exception as e:
                        if step % 50 == 0:
                            print(f"\n  [WARN] 手部第 {step} 步出错: {e}")

            # ------ 可视化 (三栏: Wrist | Env | Stats) ------
            if show_vis:
                panel_w, panel_h = 400, 360

                # — 左栏: Wrist 相机 —
                if has_camera_wrist:
                    wrist_frame = data['camera_wrist'][step].copy()
                    wrist_frame = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)
                    wrist_panel = cv2.resize(wrist_frame, (panel_w, panel_h),
                                             interpolation=cv2.INTER_LINEAR)
                else:
                    wrist_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                    cv2.putText(wrist_panel, "Wrist: N/A",
                                (panel_w // 2 - 70, panel_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (100, 100, 100), 2)

                # — 中栏: Env 相机 —
                if has_camera_env:
                    env_frame = data['camera_env'][step].copy()
                    env_frame = cv2.cvtColor(env_frame, cv2.COLOR_RGB2BGR)
                    env_panel = cv2.resize(env_frame, (panel_w, panel_h),
                                           interpolation=cv2.INTER_LINEAR)
                else:
                    env_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                    cv2.putText(env_panel, "Env: N/A",
                                (panel_w // 2 - 55, panel_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (100, 100, 100), 2)

                # — 右栏: 数值信息面板 —
                stats_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                y0 = 30
                dy = 24

                def _put(text, y, color=(220, 220, 220), scale=0.50, thick=1):
                    cv2.putText(stats_panel, text, (12, y),
                                font, scale, color, thick)

                _put(f"Step: {step+1} / {ep_len}", y0,
                     color=(0, 255, 0), scale=0.60, thick=2)
                _put(f"Speed: x{speed_factor:.1f}  |  Rate: {args.rate} Hz",
                     y0 + dy)

                line_y = y0 + dy * 2 + 8
                cv2.line(stats_panel, (10, line_y), (panel_w - 10, line_y),
                         (60, 60, 60), 1)

                cy = line_y + dy
                if has_arm_pose:
                    ap = data['arm_eef_pose'][step]
                    _put("Arm EEF Pose:", cy, color=(180, 220, 255))
                    cy += dy
                    _put(f"  pos  [{ap[0]:+.4f}, {ap[1]:+.4f}, {ap[2]:+.4f}]",
                         cy)
                    cy += dy
                    _put(f"  rot  [{ap[3]:+.4f}, {ap[4]:+.4f}, {ap[5]:+.4f}]",
                         cy)
                    cy += dy + 4

                if has_action_delta:
                    ad = data['action_eef_delta'][step]
                    _put("Action EEF Delta:", cy, color=(180, 220, 255))
                    cy += dy
                    _put(f"  dpos [{ad[0]:+.5f}, {ad[1]:+.5f}, {ad[2]:+.5f}]",
                         cy)
                    cy += dy
                    _put(f"  drot [{ad[3]:+.5f}, {ad[4]:+.5f}, {ad[5]:+.5f}]",
                         cy)
                    cy += dy + 4

                if has_hand_angles:
                    hj = np.rad2deg(data['hand_joint_angles'][step])
                    _put("Hand Joints (deg):", cy, color=(180, 220, 255))
                    cy += dy
                    _put(f"  Thumb  [{hj[0]:+5.1f},{hj[1]:+5.1f},{hj[2]:+5.1f}]",
                         cy)
                    cy += dy
                    _put(f"  Index  [{hj[3]:+5.1f},{hj[4]:+5.1f},{hj[5]:+5.1f}]",
                         cy)
                    cy += dy
                    _put(f"  Mid    [{hj[6]:+5.1f},{hj[7]:+5.1f},{hj[8]:+5.1f}]",
                         cy)
                    cy += dy
                    _put(f"  Ring   [{hj[9]:+5.1f},{hj[10]:+5.1f},{hj[11]:+5.1f}]",
                         cy)
                    cy += dy
                    _put(f"  Pinky  [{hj[12]:+5.1f},{hj[13]:+5.1f},{hj[14]:+5.1f}]",
                         cy)
                    cy += dy + 4

                if has_action_hand:
                    ah = np.rad2deg(data['action_hand_joints'][step])
                    _put("Action Hand (deg):", cy, color=(180, 220, 255))
                    cy += dy
                    _put(f"  [{ah[0]:+5.1f},{ah[1]:+5.1f},{ah[2]:+5.1f},...,"
                         f"{ah[12]:+5.1f},{ah[13]:+5.1f},{ah[14]:+5.1f}]", cy)

                # 面板标签
                cv2.putText(wrist_panel, "WRIST", (10, 25),
                            font, 0.65, (0, 200, 255), 2)
                cv2.putText(env_panel, "ENV", (10, 25),
                            font, 0.65, (0, 200, 255), 2)

                # 拼接三栏
                canvas = np.hstack([wrist_panel, env_panel, stats_panel])

                # 底部进度条 (跨越整个画布)
                total_w = canvas.shape[1]
                bar_y = canvas.shape[0] - 12
                progress = step / max(ep_len - 1, 1)
                cv2.rectangle(canvas, (8, bar_y),
                              (total_w - 8, bar_y + 6),
                              (60, 60, 60), -1)
                cv2.rectangle(canvas, (8, bar_y),
                              (8 + int((total_w - 16) * progress), bar_y + 6),
                              (0, 200, 0), -1)

                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print(f"\n  用户中断 (第 {step+1} 步)")
                    break
                elif key == ord(' '):
                    paused = not paused
                    if paused:
                        print(f"\n  [PAUSED] 第 {step+1}/{ep_len} 步  "
                              f"(按空格继续, q退出)")
                    while paused:
                        key2 = cv2.waitKey(50) & 0xFF
                        if key2 == ord(' '):
                            paused = False
                            print("  [RESUMED]")
                        elif key2 == ord('q') or key2 == 27:
                            paused = False
                            print(f"\n  用户中断 (第 {step+1} 步)")
                            step = ep_len  # 触发退出
                            break

            # 打印进度
            if not show_vis and step % 10 == 0:
                print(f"\r  进度: {step+1}/{ep_len}", end="")

            # ------ 频率控制 ------
            elapsed = time.time() - loop_start
            target_dt = control_dt / speed_factor
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("\n\n  [!] Ctrl+C — 紧急停止!")

    # ====================================================================
    # 4. 停止
    # ====================================================================
    print("\n\n[Phase 3] 停止运动...")

    if ur_arm is not None:
        try:
            ur_arm.rtde_c.servoStop()
            print("  [✓] 机械臂伺服已停止")
        except Exception:
            pass

    if show_vis:
        cv2.destroyAllWindows()

    print(f"  回放完成。\n")
    return True


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="数据回放脚本 — 在真实硬件上回放采集的轨迹")
    parser.add_argument("zarr_path", type=str, nargs="?", default=None,
                        help="zarr 数据集路径 (省略则自动选择 data/ 下最新的 .zarr 数据集)")
    parser.add_argument("--episode", "-e", type=int, default=None,
                        help="要回放的 episode 编号, 从 1 开始 (默认 1)")
    parser.add_argument("--rate", type=float, default=15.0,
                        help="回放控制频率 Hz (应与采集时一致, 默认 15)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="回放速度倍率 (默认 1.0, 建议先用 0.5 测试)")
    parser.add_argument("--action-mode", action="store_true",
                        help="使用 action_eef_delta 累积驱动机械臂 "
                             "(默认直接跟踪记录的 arm_eef_pose 观测值)")
    parser.add_argument("--hand-only", action="store_true",
                        help="仅回放灵巧手, 不控制机械臂")
    parser.add_argument("--arm-only", action="store_true",
                        help="仅回放机械臂, 不控制灵巧手")
    parser.add_argument("--no-vis", action="store_true",
                        help="不显示可视化窗口")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="跳过安全确认 (危险, 仅调试用)")
    parser.add_argument("--info", action="store_true",
                        help="仅打印数据集信息, 不执行回放")
    parser.add_argument("--loop", action="store_true",
                        help="循环回放直到手动停止")
    args = parser.parse_args()

    # --- 自动发现最新 zarr 数据集 ---
    if args.zarr_path is None:
        data_dir = os.path.join(project_root, "data")
        candidates = sorted(glob.glob(os.path.join(data_dir, "*.zarr")),
                             key=os.path.getmtime)
        if not candidates:
            print(f"[ERROR] data/ 目录下未找到任何 .zarr 数据集")
            sys.exit(1)
        zarr_path = candidates[-1]  # 最新修改的
        print(f"[自动选择] 最新数据集: {zarr_path}")
    else:
        zarr_path = os.path.expanduser(args.zarr_path)

    if not os.path.exists(zarr_path):
        print(f"[ERROR] 路径不存在: {zarr_path}")
        sys.exit(1)

    # 加载数据集
    print(f"\n加载数据集: {zarr_path}")
    rb = ReplayBuffer.create_from_path(zarr_path, mode='r')

    # 打印信息
    print_dataset_info(rb, zarr_path)

    if args.info:
        return

    # --- 默认回放最新 episode ---
    if args.episode is None:
        args.episode = rb.n_episodes  # 最后一个 episode
        print(f"[自动选择] 回放最新 episode: {args.episode}")

    # 验证 episode 编号
    ep_idx = args.episode - 1
    if ep_idx < 0 or ep_idx >= rb.n_episodes:
        print(f"[ERROR] Episode {args.episode} 不存在 (共 {rb.n_episodes} 个)")
        sys.exit(1)

    # 提取 episode 数据
    data, ep_len = get_episode_data(rb, ep_idx)
    print(f"\n选中 Episode {args.episode}: {ep_len} 步")

    # ====================================================================
    # 初始化硬件
    # ====================================================================
    hw_config = get_hardware_config()
    ur_arm = None
    hand_ctrl = None

    if not args.hand_only:
        print("\n连接 UR 机械臂...")
        try:
            ur_arm = UR5Arm(ip=hw_config['ur_arm']['ip'])
            print("[✓] UR5 已连接")
        except Exception as e:
            print(f"[✗] UR5 连接失败: {e}")
            if not args.arm_only:
                print("    继续 (仅控制灵巧手)")
            else:
                print("    无法回放 (--arm-only 模式需要机械臂)")
                sys.exit(1)

    if not args.arm_only:
        print("连接 Ruiyan 灵巧手...")
        try:
            hand_ctrl = RyHandController(port=hw_config['ruiyan_hand']['port'])
            print("[✓] Ruiyan 灵巧手已连接")
        except Exception as e:
            print(f"[✗] Ruiyan 手连接失败: {e}")
            if not args.hand_only:
                print("    继续 (仅控制机械臂)")
            else:
                print("    无法回放 (--hand-only 模式需要灵巧手)")
                sys.exit(1)

    if ur_arm is None and hand_ctrl is None:
        print("[ERROR] 机械臂和灵巧手均未连接, 无法回放")
        sys.exit(1)

    # ====================================================================
    # 执行回放
    # ====================================================================
    try:
        while True:
            success = replay_episode(ur_arm, hand_ctrl, data, ep_len, args)
            if not args.loop:
                break
            if not success:
                break
            print("\n[LOOP] 重新开始回放...")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\n[!] 收到 Ctrl+C, 正在安全退出...")

    finally:
        # 安全关闭
        if ur_arm is not None:
            try:
                ur_arm.rtde_c.servoStop()
            except Exception:
                pass
            ur_arm.stop()
            print("[✓] UR5 已断开")

        if hand_ctrl is not None:
            try:
                # 回到开手姿态
                hand_ctrl.set_angles(np.zeros(15), speed=300, radians=True)
                time.sleep(0.5)
            except Exception:
                pass
            hand_ctrl.close()
            print("[✓] Ruiyan 手已断开 (已复位)")

        cv2.destroyAllWindows()
        print("系统关闭完毕。")


if __name__ == "__main__":
    main()
