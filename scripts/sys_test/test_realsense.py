#!/usr/bin/env python3
"""
RealSense 相机连接测试

从 configs/hardware_config.yaml 读取已绑定的 Env 和 Wrist 相机配置,
通过 USB ID 定位设备节点, 打开实时画面并标注角色名称。
按 q 键退出。

该脚本仅用于验证已分配相机的连接是否正常, 不做任何绑定操作。
如需重新分配相机角色, 请运行 scripts/sys_test/assign_cam.py 。

Usage:
    python scripts/sys_test/test_realsense.py
"""

import sys
import os
import time

import cv2

# ============================================================================
# Path setup
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

os.chdir(project_root)

from ssr.config import get_hardware_config
from ssr.utils.camera_utils import get_video_index_by_id


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  RealSense 相机连接测试")
    print("=" * 60)

    hw_config = get_hardware_config()
    rs_configs = hw_config.get("cameras", {}).get("realsense", [])

    if not rs_configs:
        print("[错误] hardware_config.yaml 中未配置任何 RealSense 相机。")
        print("       请先运行: python scripts/sys_test/assign_cam.py")
        return

    # Build a list of cameras to open
    cameras = []  # (role_label, video_index, usb_id)
    for cfg in rs_configs:
        name = cfg.get("name", "")
        cam_id = cfg.get("id", "")
        cam_offset = cfg.get("offset", 0)

        # Derive a human-readable role label from the config name
        if "env" in name.lower():
            label = "ENV"
        elif "wrist" in name.lower():
            label = "WRIST"
        else:
            label = name.upper() or "UNKNOWN"

        video_idx = get_video_index_by_id(cam_id, cam_offset)
        if video_idx is None:
            print(f"[✗] {label} 相机 (USB: {cam_id}) 无法定位设备节点 — 请检查连接")
            continue

        cameras.append((label, video_idx, cam_id))
        print(f"[✓] {label} 相机已定位: /dev/video{video_idx} (USB: {cam_id})")

    if not cameras:
        print("\n[错误] 没有可用的相机。请检查连接后重试。")
        return

    # Open all cameras
    caps = {}  # label -> VideoCapture
    for label, video_idx, _ in cameras:
        cap = cv2.VideoCapture(video_idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            caps[label] = cap
            print(f"[✓] {label} 相机已打开")
        else:
            print(f"[✗] {label} 相机无法打开 (/dev/video{video_idx})")
            cap.release()

    if not caps:
        print("\n[错误] 无法打开任何相机。")
        return

    print("\n" + "-" * 60)
    print("  按 'q' 键退出测试")
    print("-" * 60 + "\n")

    try:
        while True:
            for label, cap in caps.items():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Draw role label on the frame
                    cv2.putText(frame, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 0), 2)
                    cv2.imshow(f"RealSense - {label}", frame)
                # If frame read fails, the window simply won't update

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("[退出] 测试结束")
                break

    except KeyboardInterrupt:
        print("\n[退出] Ctrl+C — 测试结束")

    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
