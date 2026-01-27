try:
    import rtde_control
    import rtde_receive
    HAS_RTDE = True
except ImportError:
    HAS_RTDE = False
    print("Warning: ur_rtde not found. UR5 control will not work.")

from ..config import get_hardware_config, get_teleop_config

class UR5Arm:
    def __init__(self, ip=None):
        if not HAS_RTDE:
            raise ImportError("ur_rtde library is missing")
            
        self.ip = ip or get_hardware_config()['ur_arm']['ip']
        print(f"Connecting to UR5 at {self.ip}...")
        self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
        print("UR5 Connected.")

    def move_j(self, joint_positions, speed=1.05, acceleration=1.4):
        self.rtde_c.moveJ(joint_positions, speed, acceleration)

    def servo_j(self, joint_positions, velocity=0.0, acceleration=0.0, time_step=None, lookahead_time=None, gain=None):
        # Use config defaults if not provided
        conf = get_teleop_config()['servoj']
        ts = time_step or conf['time_step']
        lh = lookahead_time or conf['lookahead_time']
        gn = gain or conf['gain']
        
        self.rtde_c.servoJ(joint_positions, velocity, acceleration, ts, lh, gn)
        
    def stop(self):
        try:
            self.rtde_c.servoStop()
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
        except Exception:
            pass
        
    def get_actual_q(self):
        return self.rtde_r.getActualQ()
