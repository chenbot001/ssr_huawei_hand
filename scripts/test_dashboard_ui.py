#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import cv2

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

os.chdir(project_root)

try:
    from ssr.hardware.realsense_worker import RealSenseWorker
    from ssr.hardware.fingertip import CameraWorker
    from ssr.utils.visualization import TeleopDashboard
    from ssr.utils.camera_utils import get_video_index_by_id
    from ssr.config import get_hardware_config
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    print("Starting Dashboard UI Test (Cameras Only)...")
    hw_config = get_hardware_config()
    dashboard = TeleopDashboard()
    
    # 1. Initialize RealSense Workers
    rs_workers = {}
    rs_configs = hw_config.get('cameras', {}).get('realsense', [])
    for cfg in rs_configs:
        idx = get_video_index_by_id(cfg['id'], cfg.get('offset', 0))
        if idx is not None:
            print(f"Starting RealSense: {cfg['name']} (index {idx})")
            # Request 1080p source
            w = RealSenseWorker(camera_index=idx, width=1920, height=1080)
            
            # Apply digital zoom if specified in config
            zoom = cfg.get('zoom', 1.0)
            if zoom > 1.0:
                 print(f"  Applying {zoom}x digital zoom to {cfg['name']}")
                 w.set_zoom(zoom)
            
            w.start()
            rs_workers[cfg['name']] = w
        else:
            print(f"Warning: RealSense {cfg['name']} (ID: {cfg['id']}) not found.")

    # 2. Initialize Fingertip Workers
    ft_workers = []
    ft_configs = hw_config.get('cameras', {}).get('fingertips', {})
    for name, cfg in ft_configs.items():
        idx = get_video_index_by_id(cfg['id'], cfg.get('offset', 0))
        if idx is not None:
            print(f"Starting Fingertip: {name} (index {idx})")
            w = CameraWorker(camera_index=idx)
            w.start()
            ft_workers.append({'name': name, 'worker': w})
        else:
            print(f"Warning: Fingertip {name} (ID: {cfg['id']}) not found.")

    print("\nUI Running. Press 'q' in the window to quit.")
    
    try:
        while True:
            # Mock Robot State
            mock_q = np.zeros(6)
            mock_gripper = 0.0
            mock_hand_state = "SIMULATED"
            
            # 3. Collect Fingertip Data
            fingertip_data = {}
            for item in ft_workers:
                # Assuming CameraWorker provides get_latest_data()
                data = item['worker'].get_latest_data()
                fingertip_data[item['name']] = data
            
            # 4. Collect RealSense Data
            rs_data = {}
            for name, w in rs_workers.items():
                rs_data[name] = {"image": w.get_latest_frame()}

            # 5. Update Dashboard
            key = dashboard.update(
                gello_q=mock_q,
                gello_gripper=mock_gripper,
                ur_q=mock_q,
                fingertip_data=fingertip_data,
                realsense_data=rs_data,
                hand_state=mock_hand_state
            )
            
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord(' '):
                dashboard.toggle_recording()
            
            # Small delay to prevent CPU maxing out
            # Normally handled by cv2.waitKey in dashboard.update
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping workers...")
        dashboard.close()
        
        # Stop all workers
        for w in rs_workers.values():
            w.running = False
        for item in ft_workers:
            item['worker'].running = False
            
        # Give them a moment to finish current frame
        time.sleep(0.2)
        
        # Join specifically
        for name, w in rs_workers.items():
            print(f"  Joining {name}...")
            w.join(timeout=1.0)
        for item in ft_workers:
            print(f"  Joining {item['name']}...")
            item['worker'].join(timeout=1.0)
            
        print("Done.")

if __name__ == "__main__":
    main()
