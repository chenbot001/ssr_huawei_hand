import time
import numpy as np
import cv2
from ..config import get_teleop_config, get_hardware_config

# Import hardware interfaces
from ..hardware.arm_ur5 import UR5Arm
from ..hardware.gello_interface import GelloController
from ..hardware.ruiyan_driver import RyHandController
from ..hardware.fingertip import CameraWorker
from ..hardware.realsense_worker import RealSenseWorker

# Import utils
from ..utils.visualization import TeleopDashboard
from ..utils.camera_utils import get_video_index_by_id

class TeleopController:
    """
    Main controller class for UR5 Teleoperation via GELLO.
    """
    def __init__(self, init_hand=True, init_fingertip=True, init_realsense=True):
        self.config = get_teleop_config()
        self.hw_config = get_hardware_config()
        
        # 1. Initialize Arm
        print("Initializing UR5...")
        self.arm = UR5Arm()
        
        # 2. Initialize Gello
        print("Initializing GELLO...")
        self.gello = GelloController()
        
        # 3. Initialize Hand
        self.hand = None
        if init_hand:
            print("Initializing Hand...")
            try:
                self.hand = RyHandController(port=self.hw_config['ruiyan_hand']['port']) 
            except Exception as e:
                print(f"Failed to init Hand: {e}")

        # 4. Initialize Fingertip Sensors
        self.fingertip_workers = {}
        if init_fingertip:
            print("Initializing Fingertip Sensors...")
            try:
                cam_conf = self.hw_config.get('cameras', {}).get('fingertips', {})
                
                for name, cfg in cam_conf.items():
                    idx = get_video_index_by_id(cfg['id'], cfg.get('offset', 0))
                    if idx is not None:
                        print(f"  {name.capitalize()} camera found at index {idx}")
                        w = CameraWorker(camera_index=idx)
                        w.start()
                        self.fingertip_workers[name] = w
                    else:
                        print(f"  Warning: {name.capitalize()} camera not found with ID {cfg['id']}")
            except Exception as e:
                print(f"Failed to init Fingertip: {e}")

        # 5. Initialize RealSense
        self.realsense_workers = {}
        if init_realsense:
            print("Initializing RealSense...")
            try:
                rs_configs = self.hw_config.get('cameras', {}).get('realsense', [])
                for cfg in rs_configs:
                    idx = get_video_index_by_id(cfg['id'], cfg.get('offset', 0))
                    if idx is not None:
                        print(f"  RealSense ({cfg['name']}) found at index {idx}")
                        rs = RealSenseWorker(camera_index=idx, width=1920, height=1080)
                        
                        # Apply digital zoom if specified in config
                        zoom = cfg.get('zoom', 1.0)
                        if zoom > 1.0:
                             print(f"    Applying {zoom}x digital zoom to {cfg['name']}")
                             rs.set_zoom(zoom)
                             
                        rs.start()
                        self.realsense_workers[cfg['name']] = rs
                    else:
                        print(f"  Warning: RealSense ({cfg['name']}) not found with ID {cfg['id']}")
            except Exception as e:
                print(f"Failed to init RealSense: {e}")

        self.dashboard = TeleopDashboard()
        self.last_hand_state = None

    def move_to_home(self):
        """Move arm to GELLO start position"""
        print("Reading GELLO start position...")
        time.sleep(1)
        gello_q, _ = self.gello.get_joint_state()
        print(f"Moving to: {np.degrees(gello_q)}")
        self.arm.move_j(gello_q.tolist(), 1.05, 1.4)
        time.sleep(3)
        print("Home reached.")

    def run(self):
        """Main control loop"""
        self.move_to_home()
        
        print("\nStarting Teleop Loop... (Press 'q' in Dashboard or Ctrl+C to stop)")
        conf = self.config['servoj']
        safety = self.config['safety']
        
        # Hand poses
        hand_open = np.array(self.config['hand_poses']['open'])
        hand_close = np.array(self.config['hand_poses']['close'])
        
        # Initialize last_target (commanded position) with current actual position
        last_target = self.arm.get_actual_q()

        try:
            while True:
                t_start = time.time()
                
                # 1. Read Inputs
                gello_q, gripper_val = self.gello.get_joint_state()
                current_q = self.arm.get_actual_q()
                
                # 2. Safety Check (Max Velocity)
                if safety.get('safe_mode_default', True):
                    # Clamp maximum change per step (velocity limit)
                    max_step = safety['max_joint_velocity']
                    
                    # Calculate delta from last COMMANDED position to avoid jitter
                    delta = gello_q - last_target
                    delta_clamped = np.clip(delta, -max_step, max_step)
                    
                    target_q = last_target + delta_clamped
                    last_target = target_q
                else:
                    # Direct Control
                    target_q = gello_q
                    last_target = target_q
                
                # 3. Control Arm
                self.arm.servo_j(
                    target_q.tolist(),
                    time_step=conf['time_step'],
                    lookahead_time=conf['lookahead_time'],
                    gain=conf['gain']
                )
                
                # 4. Control Hand
                if self.hand:
                    target_state = "CLOSE" if gripper_val > 0.5 else "OPEN"
                    if target_state != self.last_hand_state:
                         angles = hand_close if target_state == "CLOSE" else hand_open
                         self.hand.set_angles(angles, speed=2000, radians=False)
                         self.last_hand_state = target_state

                # 5. Visualization
                # Collect data
                fingertip_data = {}
                for name, w in self.fingertip_workers.items():
                    fingertip_data[name] = w.get_latest_data()
                
                rs_data = {}
                for name, w in self.realsense_workers.items():
                    rs_data[name] = {"image": w.get_latest_frame()}

                key = self.dashboard.update(
                    gello_q=gello_q,
                    gello_gripper=gripper_val,
                    ur_q=current_q,
                    fingertip_data=fingertip_data,
                    realsense_data=rs_data,
                    hand_state=self.last_hand_state
                )

                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord(' '):
                    self.dashboard.toggle_recording()
                
                # Maintain Loop Rate
                elapsed = time.time() - t_start
                if elapsed < conf['time_step']:
                    time.sleep(conf['time_step'] - elapsed)
                    
        except KeyboardInterrupt:
            print("\nInterrupt received...")
        except Exception as e:
            print(f"\nError in teleop loop: {e}")
        finally:
            print("Stopping Teleop components...")
            self.arm.stop()
            self.dashboard.close()
            
            # Signal all workers to stop
            for w in self.fingertip_workers.values(): w.running = False
            for w in self.realsense_workers.values(): w.running = False
            
            time.sleep(0.2)
            
            # Join workers
            for w in self.fingertip_workers.values(): w.join(timeout=1.0)
            for w in self.realsense_workers.values(): w.join(timeout=1.0)
            
            print("Exit Complete.")
