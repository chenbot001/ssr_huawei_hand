"""
Combined Ruiyan Hand driver and controller.
Contains low-level CAN communication, high-level servo control, and angle-based control.
"""

import ctypes
import threading
import time
import os
import can  # Requires: pip install python-can
import numpy as np
import math
from typing import Optional, Union, List

# ==================================================================================
# PART 1: Low-Level Driver (formerly ruiyan_lowlevel.py)
# ==================================================================================

# --- C Type Definitions (RyHandLib.h) ---

class CanMsg_t(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("ulId", ctypes.c_uint32),
        ("ucLen", ctypes.c_uint8),
        ("pucDat", ctypes.c_uint8 * 64),
    ]

# Callback type: s8_t (*BusWrite_t)(CanMsg_t stuMsg)
BusWrite_Type = ctypes.CFUNCTYPE(ctypes.c_int8, CanMsg_t)

class RyCanServoBus_t(ctypes.Structure):
    _fields_ = [
        ("pusTicksMs", ctypes.POINTER(ctypes.c_uint16)),
        ("usTicksPeriod", ctypes.c_uint16),
        ("usHookNum", ctypes.c_uint16),
        ("usListenNum", ctypes.c_uint16),
        ("pstuHook", ctypes.c_void_p),
        ("pstuListen", ctypes.c_void_p),
        ("pfunWrite", BusWrite_Type),
    ]

class ServoData_Union(ctypes.Union):
    _fields_ = [
        ("raw_u64", ctypes.c_uint64),
        ("pucDat", ctypes.c_uint8 * 64)
    ]

EN_SERVO_OK = 0

class RyHand:
    def __init__(self, port, bitrate=1000000, bustype='socketcan', lib_path=None):
        """
        Initialize the RyHand control class.
        
        Args:
            port (str): The CAN interface name (e.g., 'can0', 'vcan0', 'COM3').
            bitrate (int): Bus speed (default 1000000 for robotic hands).
            bustype (str): Interface type ('socketcan', 'slcan', 'pcan', 'serial', etc.).
            lib_path (str): Path to libRyhand64.so. If None, looks in ./lib/ relative to this file.
        """
        self.port = port
        self.bitrate = bitrate
        self.bustype = bustype
        
        if lib_path is None:
            # Look for lib in ./lib/libRyhand64.so
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(current_dir, "lib", "libRyhand64.so")
            
        self.lib_path = lib_path
        
        self.bus_hw = None
        self.running = False
        
        # Load C Library
        self._load_library()
        
        # Initialize Hardware and Logic
        self._init_hardware()
        self._init_logic()

    def _load_library(self):
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"Library not found at {self.lib_path}")
            
        self.lib = ctypes.CDLL(self.lib_path)

        # Define Signatures
        self.lib.RyCanServoBusInit.argtypes = [ctypes.POINTER(RyCanServoBus_t), BusWrite_Type, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16]
        self.lib.RyCanServoBusInit.restype = ctypes.c_uint8

        self.lib.RyCanServoLibRcvMsg.argtypes = [ctypes.POINTER(RyCanServoBus_t), CanMsg_t]
        self.lib.RyCanServoLibRcvMsg.restype = ctypes.c_int8

        self.lib.RyFunc_Reset.argtypes = [ctypes.POINTER(RyCanServoBus_t), ctypes.c_uint8, ctypes.c_uint16]
        self.lib.RyFunc_Reset.restype = ctypes.c_uint8

        self.lib.RyMotion_ServoMove_Speed.argtypes = [ctypes.POINTER(RyCanServoBus_t), ctypes.c_uint8, ctypes.c_int16, ctypes.c_uint16, ctypes.POINTER(ServoData_Union), ctypes.c_uint16]
        self.lib.RyMotion_ServoMove_Speed.restype = ctypes.c_uint8

        self.lib.RyFunc_GetServoInfo.argtypes = [ctypes.POINTER(RyCanServoBus_t), ctypes.c_uint8, ctypes.POINTER(ServoData_Union), ctypes.c_uint16]
        self.lib.RyFunc_GetServoInfo.restype = ctypes.c_uint8
        
        self.lib.RyParam_SetRigidity.argtypes = [ctypes.POINTER(RyCanServoBus_t), ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint16]
        self.lib.RyParam_SetRigidity.restype = ctypes.c_uint8

    def _init_hardware(self):
        """Connects to the physical CAN port using python-can."""
        try:
            print(f"Opening {self.bustype} interface on {self.port}...")
            self.bus_hw = can.Bus(interface=self.bustype, channel=self.port, bitrate=self.bitrate)
        except Exception as e:
            raise ConnectionError(f"Failed to open CAN port: {e}")

    def _init_logic(self):
        """Initializes the C library structures and background threads."""
        self.bus_struct = RyCanServoBus_t()
        self.timer_val = ctypes.c_uint16(0)
        self.running = True

        # 1. Setup Write Callback (Python -> C Library -> Python -> Hardware)
        def c_write_wrapper(msg):
            try:
                # Convert C array to bytes
                data = bytes(bytearray(msg.pucDat)[:msg.ucLen])
                # Create python-can Message
                can_msg = can.Message(arbitration_id=msg.ulId, data=data, is_extended_id=False)
                self.bus_hw.send(can_msg)
                return 1 # Success
            except can.CanError:
                return 0 # Fail

        # Keep reference to avoid Garbage Collection
        self.write_cb_ref = BusWrite_Type(c_write_wrapper)

        # 2. Start Threads
        # Thread A: Library Timer (1ms)
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()

        # Thread B: CAN Receiver (Hardware -> Python -> C Library)
        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.rx_thread.start()

        # 3. Call C Init
        ret = self.lib.RyCanServoBusInit(
            ctypes.byref(self.bus_struct), 
            self.write_cb_ref, 
            ctypes.byref(self.timer_val), 
            1000
        )
        if ret != 0:
            raise RuntimeError(f"Library Init Failed: {ret}")

    def _timer_loop(self):
        """Simulates 1ms tick for library internal timeouts."""
        while self.running:
            time.sleep(0.001)
            self.timer_val.value += 1
            if self.timer_val.value >= 1000:
                self.timer_val.value = 0

    def _rx_loop(self):
        """Reads from Hardware and feeds the C library."""
        while self.running:
            try:
                # Blocking read with 0.1s timeout to allow clean thread exit
                msg = self.bus_hw.recv(0.1) 
                if msg:
                    # Convert to C structure
                    c_msg = CanMsg_t()
                    c_msg.ulId = msg.arbitration_id
                    c_msg.ucLen = len(msg.data)
                    for i, b in enumerate(msg.data):
                        c_msg.pucDat[i] = b
                    
                    # Feed to C library
                    self.lib.RyCanServoLibRcvMsg(ctypes.byref(self.bus_struct), c_msg)
            except Exception as e:
                if not self.running:
                    break  # Suppress errors during shutdown
                print(f"RX Error: {e}")

    def close(self):
        self.running = False
        # Wait for background threads to finish before destroying the bus
        if hasattr(self, 'rx_thread') and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=0.5)
        if hasattr(self, 'timer_thread') and self.timer_thread.is_alive():
            self.timer_thread.join(timeout=0.5)
        if self.bus_hw:
            self.bus_hw.shutdown()

    # --- User Control API ---

    def reset_joint(self, servo_id):
        """Reboots a specific joint."""
        return self.lib.RyFunc_Reset(ctypes.byref(self.bus_struct), servo_id, 100) == EN_SERVO_OK

    def move_joint(self, servo_id, angle, speed=1000):
        """
        Move joint to angle (0-4095).
        """
        data = ServoData_Union()
        # Timeout 20ms
        ret = self.lib.RyMotion_ServoMove_Speed(
            ctypes.byref(self.bus_struct), 
            servo_id, 
            angle, 
            speed, 
            ctypes.byref(data), 
            20
        )
        return ret == EN_SERVO_OK

    def set_rigidity(self, servo_id, rigidity):
        """Set stiffness (0-255)."""
        return self.lib.RyParam_SetRigidity(ctypes.byref(self.bus_struct), servo_id, rigidity, 50) == EN_SERVO_OK

    def get_joint_state(self, servo_id):
        """
        Get current state (Pos, Vel, Current, Status).
        """
        data = ServoData_Union()
        # 50ms timeout for read
        ret = self.lib.RyFunc_GetServoInfo(ctypes.byref(self.bus_struct), servo_id, ctypes.byref(data), 50)
        
        if ret != EN_SERVO_OK:
            return None

        # Parse bitfields from the 64-bit union
        raw = data.raw_u64
        
        # Bit packing based on FingerServoInfo_t:
        # [Cmd:8][Status:8][Pos:12][Vel:12][Cur:12][Force:12]
        
        status = (raw >> 8) & 0xFF
        pos = (raw >> 16) & 0xFFF
        
        vel_raw = (raw >> 28) & 0xFFF
        vel = self._to_signed(vel_raw, 12)
        
        cur_raw = (raw >> 40) & 0xFFF
        current = self._to_signed(cur_raw, 12)

        return {
            "id": servo_id,
            "status": status,
            "position": pos,
            "velocity": vel,
            "current": current
        }

    def _to_signed(self, val, bits):
        if val & (1 << (bits - 1)):
            val -= 1 << bits
        return val


# ==================================================================================
# PART 2: Driver (formerly ruiyan_driver.py)
# ==================================================================================

class RyHandDriver:
    """
    High-level interface for Ruiyan robotic hand control.
    
    Provides numpy array-based control for all servos simultaneously,
    with simple initialization and convenient hand control methods.
    All positions are normalized to [0, 1] where 0 = fully open, 1 = fully closed.
    
    Example:
        # Simple initialization
        hand = RyHandDriver('can0')
        
        # Control all servos with normalized numpy arrays [0, 1]
        positions = np.array([0.2, 0.5, 0.4, 0.6, 0.3, 0.5, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.3])
        speeds = np.array([500] * 15)
        hand.set_positions(positions, speeds)
        
        # Read all servo states (normalized to [0, 1])
        states = hand.get_all_states()
        print(states['positions'])  # Values in [0, 1]
        
        # High-level hand control
        hand.open()   # 0.0
        hand.close()  # 1.0
        hand.set_grasp(positions)
    """
    
    # Hardware constants
    POS_MIN_HW = 0
    POS_MAX_HW = 4000
    POS_RANGE_HW = POS_MAX_HW - POS_MIN_HW
    
    def __init__(self, 
                 port: str = 'can0',
                 num_servos: int = 15,
                 bitrate: int = 1000000,
                 bustype: str = 'socketcan',
                 lib_path: Optional[str] = None):
        """
        Initialize the Ruiyan Hand controller.
        
        Args:
            port: CAN interface name (e.g., 'can0', 'vcan0')
            num_servos: Number of servos in the hand (default: 15)
            bitrate: CAN bus bitrate (default: 1000000)
            bustype: CAN interface type (default: 'socketcan')
            lib_path: Path to libRyhand64.so library. Defaults to None (auto-detect).
        """
        self.num_servos = num_servos
        self.port = port
        
        # Initialize low-level handler
        self._hand = RyHand(
            port=port,
            bitrate=bitrate,
            bustype=bustype,
            lib_path=lib_path
        )
        
        # Default servo IDs (1-indexed, typical for robotic hands)
        self.servo_ids = np.arange(1, num_servos + 1, dtype=np.uint8)

        self.reset_all()
        
        # Cache for last known states
        self._last_states = None
        
        print(f"RyHandDriver initialized: {num_servos} servos on {port}")
    
    @staticmethod
    def _normalize_to_hw(normalized: np.ndarray) -> np.ndarray:
        """Convert normalized [0, 1] positions to hardware [0, 4000]."""
        return (1 - np.clip(normalized, 0.0, 1.0)) * RyHandDriver.POS_RANGE_HW
    
    @staticmethod
    def _normalize_from_hw(hw_positions: np.ndarray) -> np.ndarray:
        """Convert hardware [0, 4000] positions to normalized [0, 1]."""
        return 1 - np.clip(hw_positions, RyHandDriver.POS_MIN_HW, RyHandDriver.POS_MAX_HW) / RyHandDriver.POS_RANGE_HW
    
    def close(self):
        """Close the hand connection and cleanup resources."""
        if self._hand:
            self._hand.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    # ==================== Array-based Control ====================
    
    def set_positions(self, 
                     positions: Union[np.ndarray, List[float]],
                     speeds: Optional[Union[np.ndarray, List[int]]] = None,
                     timeout: float = 0.1) -> np.ndarray:
        """
        Set positions for all servos using numpy arrays (normalized [0, 1]).
        
        Args:
            positions: Target positions in normalized range [0, 1] for each servo.
                      0.0 = fully open, 1.0 = fully closed.
                      Can be numpy array or list. Shape: (num_servos,)
            speeds: Movement speeds for each servo. If None, uses default 1000.
                    Can be numpy array, list, or single value.
                    Shape: (num_servos,) or scalar
            timeout: Timeout per servo command in seconds
            
        Returns:
            Boolean array indicating success for each servo
            
        Example:
            # Move all servos to different positions (normalized)
            positions = np.array([0.2, 0.5, 0.4, 0.6, 0.3, 0.5])
            hand.set_positions(positions)
            
            # Different speeds for each servo
            positions = np.array([0.5] * 6)
            speeds = np.array([500, 1000, 1500, 2000, 500, 1000])
            hand.set_positions(positions, speeds)
        """
        # Convert to numpy arrays
        positions = np.asarray(positions, dtype=np.float64)
        if speeds is None:
            speeds = np.full(self.num_servos, 1000, dtype=np.uint16)
        else:
            speeds = np.asarray(speeds, dtype=np.uint16)
            if speeds.ndim == 0:  # Scalar
                speeds = np.full(self.num_servos, speeds.item(), dtype=np.uint16)
        
        # Validate shapes
        if positions.shape != (self.num_servos,):
            raise ValueError(f"positions must have shape ({self.num_servos},), got {positions.shape}")
        if speeds.shape != (self.num_servos,):
            raise ValueError(f"speeds must have shape ({self.num_servos},) or be scalar, got {speeds.shape}")
        
        # Normalize positions to [0, 1] and convert to hardware range [0, 4095]
        positions_normalized = np.clip(positions, 0.0, 1.0)
        positions_hw = self._normalize_to_hw(positions_normalized).astype(np.int16)
        
        # Validate speed ranges
        speeds = np.clip(speeds, 1, 10000)
        
        # Send commands to all servos
        results = np.zeros(self.num_servos, dtype=bool)
        for i, (servo_id, pos_hw, speed) in enumerate(zip(self.servo_ids, positions_hw, speeds)):
            results[i] = self._hand.move_joint(servo_id, int(pos_hw), int(speed))
            if timeout > 0:
                time.sleep(timeout)
        
        return results
    
    def set_speeds(self,
                   speeds: Union[np.ndarray, List[int], int],
                   timeout: float = 0.1) -> np.ndarray:
        """
        Set movement speeds for all servos (without changing positions).
        
        Args:
            speeds: Speed values for each servo or single value for all.
                    Can be numpy array, list, or scalar.
            timeout: Timeout per servo command in seconds
            
        Returns:
            Boolean array indicating success for each servo
        """
        # Get current positions first
        states = self.get_all_states()
        current_positions = states['positions']
        
        # Use current positions with new speeds
        return self.set_positions(current_positions, speeds, timeout)
    
    def get_all_states(self, timeout: float = 0.05) -> dict:
        """
        Read state information from all servos (positions normalized to [0, 1]).
        
        Args:
            timeout: Timeout per servo read in seconds
            
        Returns:
            Dictionary with numpy arrays containing:
                - 'ids': Servo IDs (shape: num_servos,)
                - 'positions': Current positions normalized to [0, 1] (shape: num_servos,)
                - 'velocities': Current velocities (shape: num_servos,)
                - 'currents': Current values (shape: num_servos,)
                - 'statuses': Status bytes (shape: num_servos,)
                - 'valid': Boolean array indicating successful reads (shape: num_servos,)
        
        Example:
            states = hand.get_all_states()
            print(f"Positions (normalized): {states['positions']}")  # Values in [0, 1]
            print(f"Velocities: {states['velocities']}")
        """
        ids = []
        positions_hw = []
        velocities = []
        currents = []
        statuses = []
        valid = []
        
        for servo_id in self.servo_ids:
            state = self._hand.get_joint_state(servo_id)
            if state:
                ids.append(state['id'])
                positions_hw.append(state['position'])
                velocities.append(state['velocity'])
                currents.append(state['current'])
                statuses.append(state['status'])
                valid.append(True)
            else:
                ids.append(servo_id)
                positions_hw.append(0)
                velocities.append(0)
                currents.append(0)
                statuses.append(0)
                valid.append(False)
            
            if timeout > 0:
                time.sleep(timeout)
        
        # Convert hardware positions to normalized [0, 1]
        positions_hw_array = np.array(positions_hw, dtype=np.uint16)
        positions_normalized = self._normalize_from_hw(positions_hw_array)
        
        result = {
            'ids': np.array(ids, dtype=np.uint8),
            'positions': positions_normalized.astype(np.float64),
            'velocities': np.array(velocities, dtype=np.int16),
            'currents': np.array(currents, dtype=np.int16),
            'statuses': np.array(statuses, dtype=np.uint8),
            'valid': np.array(valid, dtype=bool)
        }
        
        self._last_states = result
        return result
    
    def get_positions(self) -> np.ndarray:
        """
        Get current positions of all servos.
        
        Returns:
            Numpy array of positions (shape: num_servos,)
        """
        states = self.get_all_states()
        return states['positions']
    
    def get_velocities(self) -> np.ndarray:
        """Get current velocities of all servos."""
        states = self.get_all_states()
        return states['velocities']
    
    def get_currents(self) -> np.ndarray:
        """Get current values of all servos."""
        states = self.get_all_states()
        return states['currents']
    
    # ==================== High-Level Hand Control ====================
    
    def open_hand(self, speed: int = 1000, timeout: float = 0.1):
        """
        Fully open the hand (all fingers extended).
        
        Args:
            speed: Movement speed for all servos
            timeout: Timeout per servo command
        """
        positions = np.zeros(self.num_servos, dtype=np.float64)  # 0.0 = fully open
        self.set_positions(positions, speed, timeout)
        print("Hand opened")
    
    def close_hand(self, position: float = 1.0, speed: int = 1000, timeout: float = 0.1):
        """
        Close the hand (make a fist).
        
        Args:
            position: Target position for closing in normalized range [0, 1] (default: 1.0 = fully closed)
            speed: Movement speed for all servos
            timeout: Timeout per servo command
        """
        positions = np.full(self.num_servos, position, dtype=np.float64)
        self.set_positions(positions, speed, timeout)
        print("Hand closed")
    
    def set_grasp(self,
                  positions: Union[np.ndarray, List[float]],
                  speed: int = 1000,
                  timeout: float = 0.1):
        """
        Set a specific grasp configuration (normalized [0, 1]).
        
        Args:
            positions: Target positions in normalized range [0, 1] for each finger/servo
            speed: Movement speed (same for all)
            timeout: Timeout per servo command
        """
        self.set_positions(positions, speed, timeout)
        print(f"Grasp set: {positions}")
    
    def pinch_grip(self, position: float = 0.7, speed: int = 1000):
        """
        Perform a pinch grip (thumb + index finger).
        
        Args:
            position: Grip position in normalized range [0, 1] (default: 0.7)
            speed: Movement speed
        """
        positions = np.zeros(self.num_servos, dtype=np.float64)
        if self.num_servos >= 2:
            positions[0] = position  # Thumb
            positions[1] = position  # Index
        self.set_positions(positions, speed)
        print("Pinch grip")
    
    def point(self, speed: int = 1000):
        """
        Point gesture (index finger extended, others closed).
        
        Args:
            speed: Movement speed
        """
        positions = np.full(self.num_servos, 1.0, dtype=np.float64)  # All closed
        if self.num_servos >= 2:
            positions[1] = 0.0  # Index finger extended (normalized)
        self.set_positions(positions, speed)
        print("Point gesture")
    
    def reset_all(self, wait_time: float = 2.0):
        """
        Reset all servos.
        
        Args:
            wait_time: Time to wait after reset for servos to reboot
        """
        for servo_id in self.servo_ids:
            self._hand.reset_joint(servo_id)
        print(f"Resetting all servos, waiting {wait_time}s...")
        time.sleep(wait_time)
        print("Reset complete")
    
    # ==================== Trajectory Control ====================
    
    def move_trajectory(self,
                       trajectory: np.ndarray,
                       speeds: Optional[np.ndarray] = None,
                       dt: float = 0.1):
        """
        Execute a trajectory (sequence of positions over time, normalized [0, 1]).
        
        Args:
            trajectory: Position trajectory array (shape: num_steps, num_servos)
                        Each row is a set of normalized positions [0, 1] for all servos
            speeds: Speed array (shape: num_steps, num_servos) or scalar.
                    If None, uses default 1000
            dt: Time step between trajectory points in seconds
            
        Example:
            # Create a simple trajectory: open -> close -> open (normalized)
            traj = np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # Open
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],      # Close
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # Open again
            ])
            hand.move_trajectory(traj, dt=0.5)
        """
        if trajectory.ndim != 2:
            raise ValueError(f"trajectory must be 2D (num_steps, num_servos), got shape {trajectory.shape}")
        
        if trajectory.shape[1] != self.num_servos:
            raise ValueError(f"trajectory must have {self.num_servos} columns, got {trajectory.shape[1]}")
        
        num_steps = trajectory.shape[0]
        
        if speeds is None:
            speeds = np.full((num_steps, self.num_servos), 1000, dtype=np.uint16)
        else:
            speeds = np.asarray(speeds, dtype=np.uint16)
            if speeds.ndim == 1:
                speeds = np.tile(speeds, (num_steps, 1))
            elif speeds.ndim == 0:  # Scalar
                speeds = np.full((num_steps, self.num_servos), speeds.item(), dtype=np.uint16)
        
        for step in range(num_steps):
            self.set_positions(trajectory[step], speeds[step], timeout=0)
            if dt > 0:
                time.sleep(dt)
    
    # ==================== Utility Methods ====================
    
    def set_rigidity(self,
                    rigidity: Union[np.ndarray, List[int], int],
                    timeout: float = 0.05) -> np.ndarray:
        """
        Set rigidity (stiffness) for all servos.
        
        Args:
            rigidity: Rigidity values (0-255) for each servo or single value
            timeout: Timeout per servo command
            
        Returns:
            Boolean array indicating success
        """
        rigidity = np.asarray(rigidity, dtype=np.uint8)
        if rigidity.ndim == 0:  # Scalar
            rigidity = np.full(self.num_servos, rigidity.item(), dtype=np.uint8)
        
        rigidity = np.clip(rigidity, 0, 255)
        
        results = np.zeros(self.num_servos, dtype=bool)
        for i, (servo_id, rig) in enumerate(zip(self.servo_ids, rigidity)):
            results[i] = self._hand.set_rigidity(servo_id, int(rig))
            if timeout > 0:
                time.sleep(timeout)
        
        return results
    
    def print_state(self):
        """Print current state of all servos in a readable format (positions normalized [0, 1])."""
        states = self.get_all_states()
        
        print("\n" + "=" * 85)
        print(f"{'ID':<4} {'Position':<12} {'Velocity':<10} {'Current':<10} {'Status':<8} {'Valid':<6}")
        print(f"{'':4} {'[0.0-1.0]':<12} {'':10} {'':10} {'':8} {'':6}")
        print("-" * 85)
        
        for i in range(self.num_servos):
            pos_str = f"{states['positions'][i]:.4f}"
            print(f"{states['ids'][i]:<4} "
                  f"{pos_str:<12} "
                  f"{states['velocities'][i]:<10} "
                  f"{states['currents'][i]:<10} "
                  f"0x{states['statuses'][i]:02X}     "
                  f"{'✓' if states['valid'][i] else '✗'}")
        
        print("=" * 85 + "\n")
    
    def wait_for_positions(self,
                          target_positions: np.ndarray,
                          tolerance: float = 0.01,
                          timeout: float = 5.0,
                          check_interval: float = 0.1) -> bool:
        """
        Wait until all servos reach their target positions (normalized [0, 1]).
        
        Args:
            target_positions: Target positions in normalized range [0, 1] for each servo
            tolerance: Position tolerance in normalized units (default: 0.01 = ~1% of range)
            timeout: Maximum time to wait in seconds
            check_interval: Time between position checks in seconds
            
        Returns:
            True if all servos reached targets, False if timeout
        """
        target_positions = np.asarray(target_positions, dtype=np.float64)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            current_positions = self.get_positions()
            errors = np.abs(current_positions - target_positions)
            
            if np.all(errors <= tolerance):
                return True
            
            time.sleep(check_interval)
        
        return False

# ==================================================================================
# PART 3: Controller (formerly ruiyan_controller.py)
# ==================================================================================

class RyHandController(RyHandDriver):
    """
    灵巧手控制的扩展接口，使用关节角度进行控制。
    
    该类增加了使用直观关节角度（角度或弧度）控制手部的方法，
    而不是使用归一化的舵机位置。
    """
    
    def set_angles(self, 
                  angles: Union[np.ndarray, List[float]], 
                  speed: int = 500, 
                  radians: bool = True,
                  wait: float = 0.0) -> np.ndarray:
        """
        设置所有手指的关节角度。
        
        参数:
            angles: 15个角度值的列表或数组。
                    顺序：每个手指（拇指、食指、中指、无名指、小指）：
                           [侧摆, 近端弯曲, 远端弯曲]
            speed: 运动速度（默认 500）。
            radians: 如果为 True，角度以弧度表示。如果为 False，以角度表示。
            wait: 发送指令后等待的时间（秒）。
            
        返回:
            指示每个舵机执行是否成功的布尔数组。
        """
        angles = np.array(angles, dtype=np.float64)
        if len(angles) != 15:
            raise ValueError("角度必须是包含15个元素的列表或数组。")
            
        if not radians:
            angles = np.radians(angles)
            
        # 限制范围（参考 Windows 实现）
        # 处理每个手指（3个关节）
        angles_clamped = angles.copy()
        
        # 根据 Windows 实现逻辑限制范围
        # 侧摆: +/- 30 度
        # 近端: 0-90 度
        # 远端: 0-75 度
        limit_side = math.pi * 30 / 180
        limit_prox = math.pi * 90 / 180
        limit_dist = math.pi * 75 / 180

        for i in range(15):
            joint_idx = i % 3
            if joint_idx == 0: # 侧摆
                angles_clamped[i] = max(-limit_side, min(limit_side, angles_clamped[i]))
            elif joint_idx == 1: # 近端
                angles_clamped[i] = max(0, min(limit_prox, angles_clamped[i]))
            elif joint_idx == 2: # 远端
                angles_clamped[i] = max(0, min(limit_dist, angles_clamped[i]))

        # 计算电机值
        motor_values = np.zeros(15, dtype=np.int16)
        
        for finger in range(5):
            base_idx = finger * 3
            
            theta1 = angles_clamped[base_idx]     # 侧摆
            theta2 = angles_clamped[base_idx + 1] # 近端
            theta3 = angles_clamped[base_idx + 2] # 远端
            
            # 映射公式
            
            # 侧摆
            # P_main2 = 4095 * theta1 / (pi/2)
            p_main2 = int(4095 * theta1 / (math.pi / 2))
            
            # 近端
            # P_sub1 = 4095 * (1 - (2 * theta2 / pi))
            p_sub1 = int(4095 * (1 - (2 * theta2 / math.pi)))
            
            # 电机 1 和 2 的值
            # 电机 1 = P_sub1 + P_main2
            m1_val = p_sub1 + p_main2
            
            # 电机 2 = P_sub1 - P_main2
            m2_val = p_sub1 - p_main2
            
            # 限制在 [0, 4095] 范围内
            motor_values[base_idx] = max(0, min(4095, m1_val))
            motor_values[base_idx + 1] = max(0, min(4095, m2_val))
            
            # 远端 (电机 3)
            # motor3 = 4095 * (1 - theta3 / (75 deg in rad))
            limit_75 = 75.0 * math.pi / 180.0
            m3_val = 4095 * (1 - theta3 / limit_75)
            motor_values[base_idx + 2] = int(max(0, min(4095, m3_val)))
            
        # 发送指令
        results = np.zeros(self.num_servos, dtype=bool)
        for i, (servo_id, pos_hw) in enumerate(zip(self.servo_ids, motor_values)):
            results[i] = self._hand.move_joint(servo_id, int(pos_hw), int(speed))
            
        if wait > 0:
            time.sleep(wait)
            
        return results

    def get_servo_pos(self) -> np.ndarray:
        """
        从灵巧手获取原始电机位置。

        返回:
            包含 15个 原始电机位置（0-4095）的 Numpy 数组。
        """
        raw_positions = np.zeros(15, dtype=np.int16)
        
        # 读取所有舵机
        for i, servo_id in enumerate(self.servo_ids):
            # get_joint_state 返回包含 'position' 键的字典（原始硬件值）
            state = self._hand.get_joint_state(servo_id)
            if state and 'position' in state:
                raw_positions[i] = state['position']
            else:
                # 如果读取失败，可能返回 nan 或之前的值。
                # 目前使用 0 作为回退值。
                pass
        return raw_positions

    def get_angles(self, radians: bool = True, raw_positions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取当前关节角度。
        
        参数:
           radians: 如果为 True，返回弧度。如果为 False，返回角度。
           raw_positions: 15个原始电机位置的可选数组。如果为 None，则从硬件获取。
           
        返回:
            包含 15个 角度值的 Numpy 数组。
        """
        if raw_positions is None:
            raw_positions = self.get_servo_pos()

        angles = np.zeros(15, dtype=np.float64)
        
        for finger in range(5):
            base_idx = finger * 3
            
            # 获取电机值
            m1 = raw_positions[base_idx]
            m2 = raw_positions[base_idx + 1]
            m3 = raw_positions[base_idx + 2]
            
            # 计算中间值
            p_sub1 = (m1 + m2) / 2.0
            p_main2 = (m1 - m2) / 2.0
            
            # Theta 1 (侧摆): 实际上是 +/- 90 的范围
            # theta1 = (P_main2 * 90) / 4095  (公式中使用角度值，这里我们统一使用弧度)
            # 根据写入公式: p_main2 = 4095 * theta1 / (pi/2)
            # theta1 = p_main2 * (pi/2) / 4095
            theta1 = p_main2 * (math.pi / 2.0) / 4095.0
            
            # Theta 2 (近端)
            # 根据写入公式: p_sub1 = 4095 * (1 - 2*theta2/pi)
            # 2*theta2/pi = 1 - p_sub1/4095
            # theta2 = (1 - p_sub1/4095) * pi/2
            theta2 = (1.0 - p_sub1 / 4095.0) * (math.pi / 2.0)
            
            # Theta 3 (远端)
            # 根据写入公式: m3 = 4095 * (1 - theta3/limit_75)
            # theta3/limit_75 = 1 - m3/4095
            limit_75 = 75.0 * math.pi / 180.0
            theta3 = (1.0 - m3 / 4095.0) * limit_75
            
            angles[base_idx] = theta1
            angles[base_idx + 1] = theta2
            angles[base_idx + 2] = theta3
            
        if not radians:
            angles = np.degrees(angles)
            
        return angles

