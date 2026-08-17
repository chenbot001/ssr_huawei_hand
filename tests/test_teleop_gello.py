#!/usr/bin/env python3
import sys
import os

# Add external libraries to python path if they are not installed in environment
# This allows running without pip install -e external/gello_software
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
external_gello = os.path.join(project_root, "external", "gello_software")
external_openpi = os.path.join(project_root, "external", "openpi")
src_path = os.path.join(project_root, "src")

if external_gello not in sys.path:
    sys.path.append(external_gello)

if src_path not in sys.path:
    sys.path.append(src_path)

if project_root not in sys.path:
    sys.path.append(project_root)

os.chdir(project_root)

from ssr.control.teleop_controller import TeleopController

def main():
    print("Starting SSR Teleoperation...")
    try:
        controller = TeleopController(
            init_hand=True,
            init_fingertip=True,
            init_realsense=True
        )
        controller.run()
    except Exception as e:
        print(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
