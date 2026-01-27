#!/usr/bin/env python3
import sys
import os
import time
import cv2
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# Change dir for config loading
os.chdir(project_root)

try:
    from ssr.hardware.realsense_worker import RealSenseWorker
    from ssr.utils.camera_utils import get_video_index_by_id
    from ssr.config import get_hardware_config
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    config = get_hardware_config()
    rs_configs = config.get('cameras', {}).get('realsense', [])
    
    print("Testing RealSense Cameras...")
    
    workers = []
    
    try:
        for cfg in rs_configs:
            idx = get_video_index_by_id(cfg['id'], cfg.get('offset', 0))
            if idx is None:
                print(f"Warning: Could not find camera index for {cfg['name']} (ID: {cfg['id']})")
                continue
                
            print(f"Starting camera {cfg['name']} (index {idx})...")
            w = RealSenseWorker(camera_index=idx, width=1920, height=1080) 
            w.start()
            workers.append(w)
            
        if not workers:
            print("No RealSense cameras found or initialized. Exiting.")
            return
            
        print("Cameras started. Press 'q' to quit.")
        
        cv2.namedWindow("RealSense Cameras", cv2.WINDOW_NORMAL)
        
        while True:
            images = []
            for w in workers:
                frame = w.get_latest_frame()
                if frame is not None:
                    img = frame.copy()
                    
                    cv2.putText(img, f"Cam {w.camera_index} ({frame.shape[1]}x{frame.shape[0]})", (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    images.append(img)
                else:
                    # Placeholder
                    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    cv2.putText(blank, f"Waiting {w.camera_index}...", (50, 540),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    images.append(blank)
            
            if images:
                display = np.hstack(images)
                cv2.imshow("RealSense Cameras", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("\nStopping workers...")
        for w in workers:
            w.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
