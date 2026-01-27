import cv2
import time
from threading import Thread, Lock
import numpy as np

class RealSenseWorker(Thread):
    def __init__(self, camera_index, width=None, height=None):
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
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

    def run(self):
        # Using cv2.CAP_V4L2 as suggested in view_realsense.py
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        
        # Reduced buffer size to avoid lag and help with cleanup
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if self.width is not None:
             cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
             print(f"[RealSenseWorker] Error: Could not open camera (video{self.camera_index})")
             self.running = False
             return

        while self.running:
            ret, frame = cap.read()
            if ret:
                # Apply digital zoom if set
                if self.zoom_factor > 1.0:
                    h, w = frame.shape[:2]
                    cx, cy = w // 2, h // 2
                    new_w = int(w / self.zoom_factor)
                    new_h = int(h / self.zoom_factor)
                    
                    # Crop center
                    top = cy - new_h // 2
                    left = cx - new_w // 2
                    frame = frame[top : top + new_h, left : left + new_w]
                    
                    # Resize back
                    frame = cv2.resize(frame, (w, h))

                with self.lock:
                    self.latest_frame = frame.copy()
            else:
                # Add a small sleep to avoid busy waiting if camera fails
                time.sleep(0.01)
        
        cap.release()
