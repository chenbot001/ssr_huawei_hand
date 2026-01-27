import numpy as np
import time
from ..config import get_hardware_config
import sys
import os

# Ensure external gello_software is in path if not installed via pip
# This is a fallback if the environment setup didn't install it
# external_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../external/gello_software"))
# if external_path not in sys.path:
#     sys.path.append(external_path)

try:
    from gello.agents.gello_agent import GelloAgent
except ImportError:
    print("Warning: gello package not found.")
    GelloAgent = None

class GelloController:
    def __init__(self, port=None):
        if GelloAgent is None:
            raise ImportError("Gello package is missing")
            
        self.port = port or get_hardware_config()['gello']['port']
        print(f"Connecting to GELLO at {self.port}...")
        self.agent = GelloAgent(port=self.port)
        print("GELLO Connected.")

    def get_joint_state(self):
        """
        Returns (joint_angles, gripper_state)
        """
        # GelloAgent.act({}) returns the joint angles
        try:
            joint_angles = self.agent.act({})
            # Assume 7 joints: 6 arm + 1 gripper
            if len(joint_angles) >= 7:
                 # Normalize gripper if needed, or return raw
                 # Using convention: last element is gripper
                return joint_angles[:6], joint_angles[6]
            elif len(joint_angles) == 6:
                return joint_angles, 0.0
            else:
                return joint_angles, 0.0
        except Exception as e:
            print(f"Error reading GELLO: {e}")
            return np.zeros(6), 0.0
