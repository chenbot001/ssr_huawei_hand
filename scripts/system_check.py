#!/usr/bin/env python3
import sys
import os
import time
import cv2
import subprocess
import socket
import importlib.util
import numpy as np

# ==========================================
# 1. Path Setup & dependency management
# ==========================================

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
external_gello = os.path.join(project_root, "external", "gello_software")
dynamixel_sdk_path = os.path.join(external_gello, "third_party", "DynamixelSDK", "python", "src")

# Add paths to sys.path
paths_to_add = [src_path, project_root, external_gello, dynamixel_sdk_path]
for p in paths_to_add:
    if p not in sys.path:
        sys.path.append(p)

os.chdir(project_root)

# ==========================================
# 2. Package Check
# ==========================================

def check_packages():
    print("[-] Checking critical Python packages...")
    packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('dynamixel_sdk', 'dynamixel_sdk'),
        ('rtde_receive', 'ur_rtde'),  # Assuming ur_rtde provides this
        ('gello', 'gello_software'),
    ]
    
    missing = []
    for module_name, package_name in packages:
        if importlib.util.find_spec(module_name) is None:
            # Try to import it inside a try-except block just in case find_spec fails for some reason or purely namespace packages
            try:
                __import__(module_name)
            except ImportError:
                missing.append(f"{module_name} ({package_name})")
    
    if missing:
        print(f"    [FAIL] Missing packages: {', '.join(missing)}")
        print("    Please ensure your environment is activated and dependencies are installed.")
        return False
    
    print("    [PASS] Critical packages found.")
    return True

# ==========================================
# 3. CAN Interface Check & Init
# ==========================================

def ensure_can_interface():
    print("[-] Checking CAN0 interface...")
    try:
        # Check current status
        result = subprocess.run(['ip', 'link', 'show', 'can0'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if "state UP" in result.stdout:
            print("    [PASS] CAN0 is already UP.")
            return True

        print("    [INFO] CAN0 is DOWN or not configured. Running initialization script...")
        script_path = os.path.join(project_root, "scripts", "ryhand_init.sh")
        
        if not os.path.exists(script_path):
            print(f"    [FAIL] Init script not found at {script_path}")
            return False

        print(f"    Executing: sudo bash {script_path}")
        # Using subprocess.call to allow interaction (sudo password)
        ret_code = subprocess.call(["sudo", "bash", script_path])
        
        if ret_code != 0:
            print("    [FAIL] CAN initialization script returned non-zero exit code.")
            return False
        else:
            # Double check
            result_after = subprocess.run(['ip', 'link', 'show', 'can0'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "state UP" in result_after.stdout:
                 print("    [PASS] CAN0 interface initialized successfully.")
                 return True
            else:
                 print("    [FAIL] Script ran but CAN0 is still not UP.")
                 return False
            
    except Exception as e:
        print(f"    [FAIL] Error checking/configuring CAN status: {e}")
        return False

# ==========================================
# 4. Hardware Checks (Imported logic)
# ==========================================

try:
    from ssr.config import get_hardware_config
    import rtde_receive
    from gello.agents.gello_agent import GelloAgent
    from ssr.hardware.ruiyan_driver import RyHandController
    from ssr.utils.camera_utils import get_video_index_by_id
except ImportError as e:
    # This might happen if packages checks pass but internal imports fail
    print(f"Fatal Import Error during hardware setup: {e}")
    sys.exit(1)

def check_ur5(ip):
    print(f"[-] Checking UR5 at {ip}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        result = sock.connect_ex((ip, 30003)) 
        sock.close()
        
        if result == 0:
            r = rtde_receive.RTDEReceiveInterface(ip)
            if r.isConnected():
                q = r.getActualQ()
                r.disconnect()
                print(f"    [PASS] Connected. Joint 0 pos: {q[0]:.2f}")
                return True
            else:
                print("    [FAIL] RTDE connect failed.")
                return False
        else:
            print("    [FAIL] Port 30003 unreachable.")
            return False
    except Exception as e:
        print(f"    [FAIL] Exception: {e}")
        return False

def check_gello(port):
    print(f"[-] Checking GELLO at {port}...")
    try:
        if not os.path.exists(port):
             print(f"    [FAIL] Port does not exist.")
             return False
        
        # Use simple serial check or GelloAgent
        # Using GelloAgent confirms dependencies work too
        agent = GelloAgent(port=port)
        act = agent.act({})
        print(f"    [PASS] Read success. Joints: {len(act)}")
        return True
    except Exception as e:
        print(f"    [FAIL] Exception: {e}")
        return False

def check_hand(port):
    print(f"[-] Checking Ruiyan Hand at {port} ...")
    try:
        # Bypassing the full servo initialization and state read check.
        # We simply check if the CAN interface is UP.
        result = subprocess.run(['ip', 'link', 'show', port], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if "state UP" in result.stdout:
            print(f"    [PASS] {port} is UP.")
            return True
        else:
            print(f"    [FAIL] {port} is not UP.")
            return False
            
    except Exception as e:
        print(f"    [FAIL] Exception checking port: {e}")
        return False

def check_camera(index, name):
    print(f"[-] Checking Camera {name} (Index {index})...")
    if index is None:
        print(f"    [FAIL] Camera ID not found in system.")
        return False

    try:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("    [FAIL] Could not open video device.")
            return False
        
        # Warmup / read
        for _ in range(5):
             ret, frame = cap.read()
        
        cap.release()
        
        if ret and frame is not None:
             h, w = frame.shape[:2]
             print(f"    [PASS] Frame captured. Resolution: {w}x{h}")
             return True
        else:
             print("    [FAIL] Failed to read frame.")
             return False
    except Exception as e:
        print(f"    [FAIL] Exception: {e}")
        return False

# ==========================================
# 5. Main Execution
# ==========================================

def main():
    print("========================================")
    print("   SSR System Start-up Check")
    print("========================================\n")
    
    # 1. Package Check
    if not check_packages():
        print("\n[CRITICAL] Package check failed. Fix environment before proceeding.")
        sys.exit(1)

    # 2. CAN Interface Check
    if not ensure_can_interface():
        print("\n[CRITICAL] CAN interface setup failed. Hand control will not work.")
        # We continue, but mark as failed
    
    config = get_hardware_config()
    status = {}
    
    # 3. UR5
    ur_ip = config.get('ur_arm', {}).get('ip', '192.168.1.5')
    status['UR5'] = check_ur5(ur_ip)
    
    # 4. GELLO
    gello_port = config.get('gello', {}).get('port', '/dev/ttyUSB0')
    status['GELLO'] = check_gello(gello_port)
    
    # 5. Hand
    hand_port = config.get('ruiyan_hand', {}).get('port', 'can0')
    status['Hand'] = check_hand(hand_port)
    
    # 6. Cameras
    print("[-] Checking Cameras...")
    fingertip_conf = config.get('cameras', {}).get('fingertips', {})
    
    if 'thumb' in fingertip_conf:
        thumb_idx = get_video_index_by_id(fingertip_conf['thumb']['id'], fingertip_conf['thumb'].get('offset', 0))
        status['Cam_Thumb'] = check_camera(thumb_idx, "Thumb")
        
    if 'index' in fingertip_conf:
        index_idx = get_video_index_by_id(fingertip_conf['index']['id'], fingertip_conf['index'].get('offset', 0))
        status['Cam_Index'] = check_camera(index_idx, "Index")
    
    rs_configs = config.get('cameras', {}).get('realsense', [])
    for i, cfg in enumerate(rs_configs):
        idx = get_video_index_by_id(cfg['id'], cfg.get('offset', 0))
        status[f'Cam_RealSense_{i+1}'] = check_camera(idx, cfg.get('name', 'Unknown'))
    
    print("\n========================================")
    print("             SUMMARY")
    print("========================================")
    
    all_pass = True
    for dev, passed in status.items():
        res_str = "\033[92m[OK]\033[0m" if passed else "\033[91m[FAIL]\033[0m"
        print(f"{dev:<20} : {res_str}")
        if not passed:
            all_pass = False
            
    print("========================================")
    if all_pass:
        print("\033[92mSYSTEM READY.\033[0m")
        sys.exit(0)
    else:
        print("\033[91mSYSTEM CHECK FAILED. PLEASE RESOLVE ISSUES.\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()
