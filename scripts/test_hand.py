#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import re

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
    from ssr.hardware.ruiyan_driver import RyHandController
    from ssr.config import get_hardware_config
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    print("Testing Ruiyan Hand Control")
    print("Format: <finger><joint><angle>")
    print("Fingers: t(humb), i(ndex), m(iddle), r(ing), p(inky)")
    print("Joints: s(ide), p(roximal), d(istal)")
    print("Example: 'ip45' -> Index Proximal 45 degrees")
    print("Commands: 'q' to quit, 'r' to reset, 's' to show current state, 'c' for closure test")

    config = get_hardware_config()
    try:
        hand = RyHandController(port=config['ruiyan_hand']['port'])
    except Exception as e:
        print(f"Failed to init hand: {e}")
        return

    # Current state in angles (degrees)
    # Order: [T_s, T_p, T_d, I_s, I_p, I_d, M_s, M_p, M_d, R_s, R_p, R_d, P_s, P_p, P_d]
    current_angles = np.zeros(15)

    # Mappings
    finger_map = {'t': 0, 'i': 3, 'm': 6, 'r': 9, 'p': 12}
    joint_map = {'s': 0, 'p': 1, 'd': 2}

    try:
        while True:
            cmd = input("\nEnter command: ").strip().lower()
            
            if cmd == 'q':
                break
            
            if cmd == 'r':
                current_angles = np.zeros(15)
                hand.set_angles(current_angles, speed=1000, radians=False)
                print("Reset to 0.")
                continue

            if cmd == 's':
                angles = hand.get_angles(radians=False)
                print("\n" + "=" * 40)
                print(f"{'Finger':<10} | {'Side':>6} | {'Prox':>6} | {'Dist':>6}")
                print("-" * 40)
                finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
                for i, name in enumerate(finger_names):
                    f_angles = angles[i*3 : i*3+3]
                    print(f"{name:<10} | {f_angles[0]:>6.1f} | {f_angles[1]:>6.1f} | {f_angles[2]:>6.1f}")
                print("=" * 40)
                continue

            if cmd == 'c':
                print("Running closure test (Fingers I, M, R, P)...")
                # Create target: Max Proximal (90) and Distal (75) for fingers i,m,r,p
                test_angles = np.zeros(15)
                for base_idx in [3, 6, 9, 12]:
                    test_angles[base_idx + 1] = 90.0
                    test_angles[base_idx + 2] = 75.0
                
                hand.set_angles(test_angles, speed=1000, radians=False)
                time.sleep(1.0)
                
                current_angles = np.zeros(15)
                hand.set_angles(current_angles, speed=1000, radians=False)
                print("Sequence complete. Reset to 0.")
                continue

            # Parse command
            # Regex: ([timrp])([spd])(-?\d+)
            match = re.match(r"([timrp])([spd])(-?\d+)", cmd)
            
            if match:
                f_char, j_char, angle_str = match.groups()
                angle = float(angle_str)

                # Check limits
                # Limits from RyHandController source:
                # Side (s): +/- 30
                # Proximal (p): 0-90
                # Distal (d): 0-75
                
                valid = True
                if j_char == 's':
                    if not (-30 <= angle <= 30):
                        print(f"Angle {angle} out of range for Side swing (+/- 30)")
                        valid = False
                elif j_char == 'p':
                    if not (0 <= angle <= 90):
                        print(f"Angle {angle} out of range for Proximal bend (0-90)")
                        valid = False
                elif j_char == 'd':
                    if not (0 <= angle <= 75):
                        print(f"Angle {angle} out of range for Distal bend (0-75)")
                        valid = False
                
                if valid:
                    base_idx = finger_map[f_char]
                    offset = joint_map[j_char]
                    idx = base_idx + offset
                    
                    # Update state
                    current_angles[idx] = angle
                    
                    print(f"Setting {f_char}-{j_char} to {angle} degrees...")
                    print(f"Target Array: {current_angles}")
                    
                    # Send command
                    # Note: RyHandController.set_angles expects degrees if radians=False
                    hand.set_angles(current_angles, speed=1000, radians=False)
                
            else:
                print("Invalid format. Usage: <finger><joint><angle> e.g. 'ip45'")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        hand.close()

if __name__ == "__main__":
    main()
