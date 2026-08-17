#!/usr/bin/env python3
"""
RealSense 相机连接测试

从 configs/hardware_config.yaml 读取已绑定的 Env 和 Wrist 相机配置。
若配置了 ``serial`` 则使用 pyrealsense2（与 collect_data 一致）；
否则通过 USB ID + offset 定位 ``/dev/video*`` 并用 OpenCV 打开。
按 q 键退出。

每个相机窗口提供两个控件（trackbar）：
  • Zoom x10  — 数字变焦 1.0x–3.0x（步进 0.1x）；通过放大后居中裁剪保持原始分辨率。
  • Resolution — 分辨率预设下拉列表（1080p / 720p / 480p / VGA / QVGA）；切换时自动
                 重启相机流。

该脚本仅用于验证已分配相机的连接是否正常, 不做任何绑定操作。
如需重新分配相机角色, 请运行 tests/assign_cam.py 。

Usage:
    python tests/test_realsense.py
"""

import sys
import os
import time

import cv2
import numpy as np

# ============================================================================
# Path setup
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

os.chdir(project_root)

from ssr.config import get_hardware_config
from ssr.utils.camera_utils import get_video_index_by_id
from ssr.hardware.realsense_worker import RealSenseWorker


# ============================================================================
# Resolution presets  (label, width, height)
# ============================================================================
RESOLUTIONS = [
    ("1080p", 1920, 1080),
    ("720p",  1280,  720),
    ("480p",   848,  480),
    ("VGA",    640,  480),
    ("QVGA",   424,  240),
]
DEFAULT_RES_IDX = 3   # 640 × 480


# ============================================================================
# Per-camera control state
# ============================================================================
class CameraControl:
    """Owns one camera source (RS worker or OpenCV cap) plus its UI state."""

    def __init__(self, label: str, serial: str = "", cam_id: str = "",
                 cam_offset: int = 0, worker=None, cap=None):
        self.label      = label
        self.serial     = serial
        self.cam_id     = cam_id
        self.cam_offset = cam_offset
        self.worker     = worker
        self.cap        = cap
        self.win_name   = f"RealSense - {label}"
        self.res_idx    = DEFAULT_RES_IDX
        self._prev_res  = DEFAULT_RES_IDX

    # ------------------------------------------------------------------
    # Window / trackbar setup
    # ------------------------------------------------------------------
    def setup_window(self) -> None:
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        # Zoom: trackbar value = zoom × 10  (range 10–30 → 1.0x–3.0x)
        cv2.createTrackbar("Zoom x10", self.win_name, 10, 30, lambda _: None)
        # Resolution: integer index into RESOLUTIONS list
        cv2.createTrackbar(
            "Resolution", self.win_name, self.res_idx, len(RESOLUTIONS) - 1,
            lambda _: None,
        )

    # ------------------------------------------------------------------
    # Zoom  (applied in display thread; preserves output dimensions)
    # ------------------------------------------------------------------
    def _zoom_factor(self) -> float:
        raw = cv2.getTrackbarPos("Zoom x10", self.win_name)
        return max(10, raw) / 10.0   # clamp to ≥ 1.0

    def apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        factor = self._zoom_factor()
        if factor <= 1.0:
            return frame
        h, w   = frame.shape[:2]
        new_w  = max(1, int(w / factor))
        new_h  = max(1, int(h / factor))
        cx, cy = w // 2, h // 2
        top    = max(0, min(cy - new_h // 2, h - new_h))
        left   = max(0, min(cx - new_w // 2, w - new_w))
        return cv2.resize(frame[top:top + new_h, left:left + new_w], (w, h),
                          interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # Resolution change detection & camera restart
    # ------------------------------------------------------------------
    def check_and_apply_resolution(self) -> None:
        idx = cv2.getTrackbarPos("Resolution", self.win_name)
        if idx == self._prev_res:
            return
        self._prev_res = idx
        self.res_idx   = idx
        label_str, new_w, new_h = RESOLUTIONS[idx]
        print(f"[{self.label}] 分辨率切换 → {label_str} ({new_w}×{new_h})")
        if self.worker is not None:
            self._restart_worker(new_w, new_h)
        elif self.cap is not None:
            self._restart_cap(new_w, new_h)

    def _restart_worker(self, new_w: int, new_h: int) -> None:
        self.worker.stop()
        self.worker = None
        try:
            w = RealSenseWorker(width=new_w, height=new_h, serial_number=self.serial)
            w.start()
            self.worker = w
            print(f"[{self.label}] RS worker 重启成功")
        except Exception as e:
            print(f"[{self.label}] RS worker 重启失败: {e}")

    def _restart_cap(self, new_w: int, new_h: int) -> None:
        self.cap.release()
        self.cap = None
        video_idx = get_video_index_by_id(self.cam_id, self.cam_offset)
        if video_idx is None:
            print(f"[{self.label}] 无法重新定位设备节点，分辨率未切换")
            return
        cap = cv2.VideoCapture(video_idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  new_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_h)
        if cap.isOpened():
            self.cap = cap
            print(f"[{self.label}] VideoCapture 重启成功")
        else:
            print(f"[{self.label}] VideoCapture 重启失败")
            cap.release()

    # ------------------------------------------------------------------
    # Frame retrieval
    # ------------------------------------------------------------------
    def get_frame(self) -> "np.ndarray | None":
        if self.worker is not None:
            return self.worker.get_latest_frame()
        if self.cap is not None:
            ret, frame = self.cap.read()
            return frame if ret and frame is not None else None
        return None

    # ------------------------------------------------------------------
    # Overlay text  (zoom + resolution shown on frame)
    # ------------------------------------------------------------------
    def annotate(self, frame: np.ndarray) -> np.ndarray:
        res_label = RESOLUTIONS[self.res_idx][0]
        zoom_val  = self._zoom_factor()
        lines = [
            self.label,
            f"Zoom: {zoom_val:.1f}x",
            f"Res:  {res_label}",
        ]
        y = 30
        for text in lines:
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 28
        return frame

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def release(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        if self.cap is not None:
            self.cap.release()


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  RealSense 相机连接测试")
    print("=" * 60)

    hw_config  = get_hardware_config()
    rs_configs = hw_config.get("cameras", {}).get("realsense", [])

    if not rs_configs:
        print("[错误] hardware_config.yaml 中未配置任何 RealSense 相机。")
        print("       请先运行: python tests/assign_cam.py")
        return

    controls: list[CameraControl] = []

    for cfg in rs_configs:
        name       = cfg.get("name", "")
        cam_id     = cfg.get("id", "")
        cam_offset = cfg.get("offset", 0)
        serial     = (cfg.get("serial") or cfg.get("serial_number") or "").strip()

        if "env" in name.lower():
            label = "ENV"
        elif "wrist" in name.lower():
            label = "WRIST"
        else:
            label = name.upper() or "UNKNOWN"

        _, init_w, init_h = RESOLUTIONS[DEFAULT_RES_IDX]

        if serial:
            try:
                w = RealSenseWorker(width=init_w, height=init_h, serial_number=serial)
                w.start()
                ctrl = CameraControl(label, serial=serial, cam_id=cam_id,
                                     cam_offset=cam_offset, worker=w)
                controls.append(ctrl)
                print(f"[✓] {label} 相机已启动 (pyrealsense2 serial={serial})")
            except Exception as e:
                print(f"[✗] {label} SDK 启动失败: {e}")
            continue

        video_idx = get_video_index_by_id(cam_id, cam_offset)
        if video_idx is None:
            print(f"[✗] {label} 相机 (USB: {cam_id}) 无法定位设备节点 — 请检查连接")
            continue

        cap = cv2.VideoCapture(video_idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  init_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, init_h)
        if cap.isOpened():
            ctrl = CameraControl(label, cam_id=cam_id, cam_offset=cam_offset, cap=cap)
            controls.append(ctrl)
            print(f"[✓] {label} 相机已打开: /dev/video{video_idx} (USB: {cam_id})")
        else:
            print(f"[✗] {label} 相机无法打开 (/dev/video{video_idx})")
            cap.release()

    if not controls:
        print("\n[错误] 没有可用的相机。请检查连接后重试。")
        return

    # Create all windows and attach trackbars
    for ctrl in controls:
        ctrl.setup_window()

    res_names = " | ".join(f"{i}={r[0]}" for i, r in enumerate(RESOLUTIONS))
    print("\n" + "-" * 60)
    print(f"  Resolution trackbar: {res_names}")
    print("  Zoom trackbar: value ÷ 10 = zoom factor  (10 = 1.0x, 30 = 3.0x)")
    print("  按 'q' 键退出测试")
    print("-" * 60 + "\n")

    try:
        while True:
            for ctrl in controls:
                # Check if resolution slider moved → restart camera
                ctrl.check_and_apply_resolution()

                frame = ctrl.get_frame()
                if frame is None:
                    continue

                if ctrl.label == "WRIST":
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                frame = ctrl.apply_zoom(frame)
                frame = ctrl.annotate(frame)
                cv2.imshow(ctrl.win_name, frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("[退出] 测试结束")
                break

    except KeyboardInterrupt:
        print("\n[退出] Ctrl+C — 测试结束")

    finally:
        for ctrl in controls:
            ctrl.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
