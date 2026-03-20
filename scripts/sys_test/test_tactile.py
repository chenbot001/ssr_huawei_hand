#!/usr/bin/env python3
import sys
import os
import time
import cv2
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# Change dir for config loading
os.chdir(project_root)

try:
    from ssr.hardware.fingertip import CameraWorker
    from ssr.utils.camera_utils import get_video_index_by_id
    from ssr.config import get_hardware_config
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    config = get_hardware_config()
    fingertip_conf = config.get('cameras', {}).get('fingertips', {})
    
    print("Testing Tactile Sensors...")
    
    workers = []
    
    try:
        # Resolve indices
        thumb_idx = get_video_index_by_id(fingertip_conf['thumb']['id'], fingertip_conf['thumb'].get('offset', 0))
        index_idx = get_video_index_by_id(fingertip_conf['index']['id'], fingertip_conf['index'].get('offset', 0))

        if thumb_idx is not None:
            print(f"Starting Thumb sensor (index {thumb_idx})...")
            workers.append(CameraWorker(thumb_idx))
        else:
            print("Warning: Thumb sensor not found.")

        if index_idx is not None:
            print(f"Starting Index sensor (index {index_idx})...")
            workers.append(CameraWorker(index_idx))
        else:
            print("Warning: Index sensor not found.")
            
        if not workers:
            print("No tactile sensors found or initialized. Exiting.")
            return

        for w in workers:
            w.start()
            
        print("Sensors started. Press 'q' to quit.")
        
        while True:
            images = []
            for w in workers:
                data = w.get_latest_data()
                if data and 'arrows' in data and data['arrows'] is not None:
                    # Resize to standard size for concatenation
                    img = cv2.resize(data['arrows'], (320, 240))
                    # Add labels
                    cv2.putText(img, f"Cam {w.camera_index}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    images.append(img)
                else:
                    # Placeholder if no data yet
                    blank = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(blank, f"Waiting {w.camera_index}...", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    images.append(blank)
            
            if images:
                # Concatenate horizontally
                display = np.hstack(images)
                cv2.imshow("Tactile Sensors", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping workers...")
        for w in workers:
            w.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
