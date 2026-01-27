import cv2
import numpy as np
import os
import time
from datetime import datetime

class TeleopDashboard:
    def __init__(self):
        self.window_name = "SSR Teleop Dashboard"
        self.is_recording = False
        self.video_writer = None
        self.recording_dir = "demo"
        self.target_fps = 30.0
        self.last_frame_time = 0
        self.first_run = True
        if not os.path.exists(self.recording_dir):
            os.makedirs(self.recording_dir)
            
    def toggle_recording(self):
        if not self.is_recording:
            # Start Recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.recording_dir, f"teleop_{timestamp}.mp4")
            self.is_recording = True
            print(f"Started recording: {filename}")
            # Writer will be initialized on first frame in update()
        else:
            # Stop Recording
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print("Stopped recording.")
    
    def close(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        cv2.destroyAllWindows()

    def update(self, gello_q, gello_gripper, ur_q, fingertip_data, realsense_data, hand_state):
        try:
            # Dashboard Config
            # We want both columns to have the same total height
            # Let's assume 2 RS cameras and 2 Fingertip cameras for default layout scaling
            # RS Target: 540 Height each (Total 1080)
            target_rs_h = 540
            target_ft_h = 540
            
            # 1. Process RealSense (Vertically Stacked)
            rs_imgs = []
            for key in sorted(realsense_data.keys()):
                data = realsense_data.get(key)
                if data and 'image' in data and data['image'] is not None:
                     rs_img = data['image'].copy()
                     h, w = rs_img.shape[:2]
                     target_w = int(w * (target_rs_h / h))
                     rs_img = cv2.resize(rs_img, (target_w, target_rs_h))
                     
                     cv2.putText(rs_img, f"{key} ({w}x{h})", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                     rs_imgs.append(rs_img)
            
            # 2. Process Fingertip (Vertically Stacked)
            ft_imgs = []
            # Sort reversed to put 'thumb' above 'index' (T > I)
            for cam_key in sorted(fingertip_data.keys(), reverse=True):
                data = fingertip_data[cam_key]
                if data and 'arrows' in data and data['arrows'] is not None:
                     arrow_img = data['arrows'].copy()
                     h, w = arrow_img.shape[:2]
                     # We use target_ft_h to match the height of one RS feed
                     target_w = int(w * (target_ft_h / h))
                     arrow_img = cv2.resize(arrow_img, (target_w, target_ft_h))
                     
                     cv2.putText(arrow_img, str(cam_key), (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                     ft_imgs.append(arrow_img)

            # Combine into columns
            # Column 1: RealSense Stack
            col1 = np.vstack(rs_imgs) if rs_imgs else np.zeros((1080, 10, 3), dtype=np.uint8)
            # Column 2: Fingertip Stack
            col2 = np.vstack(ft_imgs) if ft_imgs else np.zeros((1080, 10, 3), dtype=np.uint8)
            
            # Match heights if number of cameras differ
            # (e.g., if one list is shorter, pad it with black)
            max_h = max(col1.shape[0], col2.shape[0])
            if max_h > 0:
                if col1.shape[0] < max_h:
                    col1 = np.vstack([col1, np.zeros((max_h - col1.shape[0], col1.shape[1], 3), dtype=np.uint8)])
                if col2.shape[0] < max_h:
                    col2 = np.vstack([col2, np.zeros((max_h - col2.shape[0], col2.shape[1], 3), dtype=np.uint8)])
            
            # Combine columns side-by-side
            combined_cameras = np.hstack([col1, col2])
            
            # Info Panel (Scale height to match total combined height)
            h, w = combined_cameras.shape[:2]
            panel = np.zeros((h, 350, 3), dtype=np.uint8)
            
            y = 50
            cv2.putText(panel, "SYSTEM STATUS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            y += 60
            cv2.putText(panel, f"Hand State: {hand_state}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            y += 60
            cv2.putText(panel, "UR5 Joint Angles (deg):", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            y += 40
            for i, q in enumerate(ur_q):
                if i < 6: 
                    cv2.putText(panel, f"Joint {i}: {np.degrees(q):8.1f}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    y += 35
            
            # Recording Indicator
            if self.is_recording:
                cv2.circle(panel, (30, h - 30), 15, (0, 0, 255), -1)
                cv2.putText(panel, "RECORDING", (60, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
            final_img = np.hstack([combined_cameras, panel])

            # 4. Handle Video Output
            if self.is_recording:
                current_time = time.time()
                # Initialize writer on first frame
                if self.video_writer is None:
                    fh, fw = final_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(self.recording_dir, f"teleop_{timestamp}.mp4")
                    self.video_writer = cv2.VideoWriter(filename, fourcc, self.target_fps, (fw, fh))
                    self.last_frame_time = current_time
                
                # Only write frame if enough time has passed to maintain real-time speed
                if (current_time - self.last_frame_time) >= (1.0 / self.target_fps):
                    self.video_writer.write(final_img)
                    self.last_frame_time = current_time
            
            # Use WINDOW_NORMAL so user can resize
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            if self.first_run:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                self.first_run = False
                
            cv2.imshow(self.window_name, final_img)
            return cv2.waitKey(1)
        except Exception as e:
            print(f"Dashboard Error: {e}")
            return -1
