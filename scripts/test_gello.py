#!/usr/bin/env python3
import sys
import os
import time
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
external_gello = os.path.join(project_root, "external", "gello_software")
src_path = os.path.join(project_root, "src")

if external_gello not in sys.path:
    sys.path.append(external_gello)
if src_path not in sys.path:
    sys.path.append(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# Change dir for config loading
os.chdir(project_root)

try:
    from ssr.hardware.gello_interface import GelloController
    from ssr.hardware.arm_ur5 import UR5Arm
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    print("Testing GELLO -> UR5 Control (Joint 7 Disabled)")
    
    try:
        # Initialize
        ur = UR5Arm()
        gello = GelloController()
        
        print("Reading GELLO start position...")
        time.sleep(1)
        gello_q, _ = gello.get_joint_state()
        print(f"Moving UR5 to start: {np.degrees(gello_q)}")
        ur.move_j(gello_q.tolist(), 1.05, 1.4)
        time.sleep(3)
        
        print("Starting Loop. Ctrl+C to stop.")
        
        while True:
            # Get Gello State (6 joints)
            gello_q, _ = gello.get_joint_state()
            
            # UR5 only needs 6 joints. 
            # If gello_q has more (e.g. gripper mapped to 7th), we just take first 6.
            # Usually get_joint_state returns 6 for arm + 1 val for gripper separately
            # In gello_interface.py, it returns joint_angles[:6].
            
            target_q = gello_q[:6]
            
            # Send to UR5
            ur.servo_j(target_q.tolist(), time_step=0.002, lookahead_time=0.1, gain=300)
            
            time.sleep(0.002)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        ur.stop()
    except Exception as e:
        print(f"Error: {e}")
        ur.stop()

if __name__ == "__main__":
    main()
