import cv2
import time
from threading import Thread, Lock
import numpy as np

try:
    import pyrealsense2 as rs

    _HAS_RS = True
except ImportError:
    rs = None
    _HAS_RS = False


class RealSenseWorker(Thread):
    """
    后台读取 RealSense 彩色图。

    - **推荐**：传入 ``serial_number``，使用 librealsense 按序列号打开，
      不依赖 ``/dev/video`` 编号是否随重启变化。
    - **兼容**：仅传 ``camera_index`` 时沿用 OpenCV + V4L2（依赖 offset 正确）。
    """

    def __init__(self, width=None, height=None, camera_index=None, serial_number=None):
        super().__init__()
        self.width = int(width) if width else 640
        self.height = int(height) if height else 480
        self.camera_index = camera_index
        sn = (serial_number or "").strip()
        self.serial_number = sn if sn else None
        self._use_rs_sdk = self.serial_number is not None

        if not self._use_rs_sdk and camera_index is None:
            raise ValueError("RealSenseWorker: 需要 camera_index 或 serial_number")
        if self._use_rs_sdk and not _HAS_RS:
            raise ImportError("未安装 pyrealsense2，无法使用 serial_number 打开相机")

        self.running = True
        self.lock = Lock()
        self.latest_frame = None
        self.zoom_factor = 1.0

    def stop(self):
        self.running = False
        self.join()

    def set_zoom(self, factor):
        self.zoom_factor = factor

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def _apply_zoom(self, frame):
        if self.zoom_factor <= 1.0:
            return frame
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        new_w = int(w / self.zoom_factor)
        new_h = int(h / self.zoom_factor)
        top = cy - new_h // 2
        left = cx - new_w // 2
        cropped = frame[top : top + new_h, left : left + new_w]
        return cv2.resize(cropped, (w, h))

    def _run_opencv(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            print(f"[RealSenseWorker] Error: Could not open camera (video{self.camera_index})")
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if ret:
                frame = self._apply_zoom(frame)
                with self.lock:
                    self.latest_frame = frame.copy()
            else:
                time.sleep(0.01)

        cap.release()

    def _run_rs_sdk(self):
        # D435/D435i 的 Color 往往没有 320x240 等分辨率，直接 enable 会失败。
        # 先尝试目标分辨率，再回退到常见模式，最后用 cv2 缩放到 self.width x self.height。
        want = (self.width, self.height)
        try_list = [want, (640, 480), (424, 240)]
        seen = set()
        resolution_order = []
        for wh in try_list:
            if wh not in seen:
                seen.add(wh)
                resolution_order.append(wh)

        pipeline = None
        for sw, sh in resolution_order:
            for fps in (30, 15, 6):
                cfg = rs.config()
                cfg.enable_device(self.serial_number)
                cfg.enable_stream(rs.stream.color, sw, sh, rs.format.bgr8, fps)
                p = rs.pipeline()
                try:
                    p.start(cfg)
                    pipeline = p
                    if (sw, sh) != want:
                        print(
                            f"[RealSenseWorker] serial={self.serial_number}: "
                            f"使用 SDK 分辨率 {sw}x{sh}@{fps}Hz，输出缩放到 {self.width}x{self.height}"
                        )
                    break
                except RuntimeError:
                    try:
                        p.stop()
                    except Exception:
                        pass
            if pipeline is not None:
                break

        if pipeline is None:
            print(
                f"[RealSenseWorker] SDK 无法启动彩色流 (serial={self.serial_number})。"
                f" 已尝试分辨率 {resolution_order}。"
            )
            self.running = False
            return

        try:
            while self.running:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError:
                    time.sleep(0.01)
                    continue
                color = frames.get_color_frame()
                if not color:
                    continue
                frame = np.asanyarray(color.get_data())
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                frame = self._apply_zoom(frame)
                with self.lock:
                    self.latest_frame = frame.copy()
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass

    def run(self):
        if self._use_rs_sdk:
            self._run_rs_sdk()
        else:
            self._run_opencv()
