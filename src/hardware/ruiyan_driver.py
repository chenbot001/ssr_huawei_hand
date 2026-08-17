from __future__ import annotations

import ctypes
import math
from pathlib import Path
import threading
import time
from collections.abc import Sequence

import numpy as np


RYHAND_LIBRARY_PATH = Path(__file__).resolve().parent / "lib" / "libRyhand64.so"
SERVO_COUNT = 15
SERVO_IDS = tuple(range(1, SERVO_COUNT + 1))


class CanMsg(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("ulId", ctypes.c_uint32),
        ("ucLen", ctypes.c_uint8),
        ("pucDat", ctypes.c_uint8 * 64),
    ]


BusWrite = ctypes.CFUNCTYPE(ctypes.c_int8, CanMsg)


class RyCanServoBus(ctypes.Structure):
    _fields_ = [
        ("pusTicksMs", ctypes.POINTER(ctypes.c_uint16)),
        ("usTicksPeriod", ctypes.c_uint16),
        ("usHookNum", ctypes.c_uint16),
        ("usListenNum", ctypes.c_uint16),
        ("pstuHook", ctypes.c_void_p),
        ("pstuListen", ctypes.c_void_p),
        ("pfunWrite", BusWrite),
    ]


class ServoData(ctypes.Union):
    _fields_ = [("raw_u64", ctypes.c_uint64), ("pucDat", ctypes.c_uint8 * 64)]


def angles_to_motor_positions(angles: Sequence[float]) -> np.ndarray:
    """Convert 15 RYHand joint angles in radians to 15 raw motor targets."""
    values = np.asarray(angles, dtype=np.float64)
    if values.shape != (SERVO_COUNT,):
        raise ValueError(f"angles must have shape ({SERVO_COUNT},), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("angles must contain only finite values")

    motor = np.zeros(SERVO_COUNT, dtype=np.int16)
    side_limit = math.radians(30.0)
    proximal_limit = math.radians(90.0)
    distal_limit = math.radians(75.0)

    for finger in range(5):
        base = finger * 3
        side = float(np.clip(values[base], -side_limit, side_limit))
        proximal = float(np.clip(values[base + 1], 0.0, proximal_limit))
        distal = float(np.clip(values[base + 2], 0.0, distal_limit))

        differential = 4095.0 * side / (math.pi / 2.0)
        common = 4095.0 * (1.0 - (2.0 * proximal / math.pi))
        motor[base] = int(np.clip(common + differential, 0.0, 4095.0))
        motor[base + 1] = int(np.clip(common - differential, 0.0, 4095.0))
        motor[base + 2] = int(
            np.clip(4095.0 * (1.0 - distal / distal_limit), 0.0, 4095.0)
        )
    return motor


def motor_positions_to_angles(raw_positions: Sequence[int]) -> np.ndarray:
    """Convert 15 raw motor positions to RYHand joint angles in radians."""
    raw = np.asarray(raw_positions, dtype=np.float64)
    if raw.shape != (SERVO_COUNT,):
        raise ValueError(
            f"raw_positions must have shape ({SERVO_COUNT},), got {raw.shape}"
        )
    raw = np.clip(raw, 0.0, 4095.0)
    angles = np.zeros(SERVO_COUNT, dtype=np.float64)
    distal_limit = math.radians(75.0)

    for finger in range(5):
        base = finger * 3
        first, second, third = raw[base : base + 3]
        common = (first + second) / 2.0
        differential = (first - second) / 2.0
        angles[base] = differential * (math.pi / 2.0) / 4095.0
        angles[base + 1] = (1.0 - common / 4095.0) * (math.pi / 2.0)
        angles[base + 2] = (1.0 - third / 4095.0) * distal_limit
    return angles


class _RyHandBus:
    """Vendor-library and SocketCAN transport used by RyHandController."""

    def __init__(self, port: str, library_path: Path = RYHAND_LIBRARY_PATH) -> None:
        if not library_path.is_file():
            raise FileNotFoundError(
                f"RYHand vendor library is missing: {library_path}. "
                "Copy libRyhand64.so to this fixed path."
            )
        try:
            import can
        except ImportError as exc:
            raise ImportError("python-can is required for RYHand control") from exc

        self._can = can
        self._library = ctypes.CDLL(str(library_path))
        self._bind_library()
        self._hardware_bus = can.Bus(
            interface="socketcan", channel=port, bitrate=1_000_000
        )
        self._bus = RyCanServoBus()
        self._timer_value = ctypes.c_uint16(0)
        self._running = threading.Event()
        self._running.set()
        self._write_callback = BusWrite(self._write)

        result = self._library.RyCanServoBusInit(
            ctypes.byref(self._bus),
            self._write_callback,
            ctypes.byref(self._timer_value),
            1000,
        )
        if result != 0:
            self._hardware_bus.shutdown()
            raise RuntimeError(f"RYHand vendor library initialization failed: {result}")

        self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._timer_thread.start()
        self._receiver_thread.start()

    def _bind_library(self) -> None:
        self._library.RyCanServoBusInit.argtypes = [
            ctypes.POINTER(RyCanServoBus),
            BusWrite,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_uint16,
        ]
        self._library.RyCanServoBusInit.restype = ctypes.c_uint8
        self._library.RyCanServoLibRcvMsg.argtypes = [
            ctypes.POINTER(RyCanServoBus),
            CanMsg,
        ]
        self._library.RyCanServoLibRcvMsg.restype = ctypes.c_int8
        self._library.RyFunc_Reset.argtypes = [
            ctypes.POINTER(RyCanServoBus),
            ctypes.c_uint8,
            ctypes.c_uint16,
        ]
        self._library.RyFunc_Reset.restype = ctypes.c_uint8
        self._library.RyMotion_ServoMove_Speed.argtypes = [
            ctypes.POINTER(RyCanServoBus),
            ctypes.c_uint8,
            ctypes.c_int16,
            ctypes.c_uint16,
            ctypes.POINTER(ServoData),
            ctypes.c_uint16,
        ]
        self._library.RyMotion_ServoMove_Speed.restype = ctypes.c_uint8
        self._library.RyFunc_GetServoInfo.argtypes = [
            ctypes.POINTER(RyCanServoBus),
            ctypes.c_uint8,
            ctypes.POINTER(ServoData),
            ctypes.c_uint16,
        ]
        self._library.RyFunc_GetServoInfo.restype = ctypes.c_uint8

    def _write(self, message: CanMsg) -> int:
        try:
            payload = bytes(message.pucDat[: message.ucLen])
            self._hardware_bus.send(
                self._can.Message(
                    arbitration_id=message.ulId,
                    data=payload,
                    is_extended_id=False,
                )
            )
            return 1
        except self._can.CanError:
            return 0

    def _timer_loop(self) -> None:
        while self._running.is_set():
            time.sleep(0.001)
            self._timer_value.value = (self._timer_value.value + 1) % 1000

    def _receive_loop(self) -> None:
        while self._running.is_set():
            message = self._hardware_bus.recv(0.1)
            if message is None:
                continue
            incoming = CanMsg()
            incoming.ulId = message.arbitration_id
            incoming.ucLen = len(message.data)
            for index, byte in enumerate(message.data):
                incoming.pucDat[index] = byte
            self._library.RyCanServoLibRcvMsg(ctypes.byref(self._bus), incoming)

    def reset(self, servo_id: int) -> bool:
        return self._library.RyFunc_Reset(
            ctypes.byref(self._bus), servo_id, 100
        ) == 0

    def move(self, servo_id: int, position: int, speed: int) -> bool:
        data = ServoData()
        return self._library.RyMotion_ServoMove_Speed(
            ctypes.byref(self._bus),
            servo_id,
            position,
            speed,
            ctypes.byref(data),
            20,
        ) == 0

    def position(self, servo_id: int) -> int | None:
        data = ServoData()
        result = self._library.RyFunc_GetServoInfo(
            ctypes.byref(self._bus), servo_id, ctypes.byref(data), 50
        )
        if result != 0:
            return None
        return int((data.raw_u64 >> 16) & 0xFFF)

    def close(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        self._receiver_thread.join(timeout=0.5)
        self._timer_thread.join(timeout=0.5)
        self._hardware_bus.shutdown()


class RyHandController:
    """RYHand communication and 15-angle control."""

    def __init__(self, port: str = "can0") -> None:
        self.port = port
        self._bus = _RyHandBus(port)
        self._closed = False
        failed = [servo_id for servo_id in SERVO_IDS if not self._bus.reset(servo_id)]
        if failed:
            self._bus.close()
            self._closed = True
            raise RuntimeError(f"Failed to reset RYHand servos: {failed}")
        time.sleep(2.0)

    def set_angles(
        self, angles: Sequence[float], speed: int = 500, radians: bool = True
    ) -> np.ndarray:
        values = np.asarray(angles, dtype=np.float64)
        if not radians:
            values = np.radians(values)
        motor_positions = angles_to_motor_positions(values)
        bounded_speed = int(np.clip(speed, 1, 10_000))
        results = np.array(
            [
                self._bus.move(servo_id, int(position), bounded_speed)
                for servo_id, position in zip(SERVO_IDS, motor_positions)
            ],
            dtype=bool,
        )
        if not np.all(results):
            failed = [
                servo_id for servo_id, succeeded in zip(SERVO_IDS, results) if not succeeded
            ]
            raise RuntimeError(f"RYHand command failed for servos: {failed}")
        return results

    def get_angles(self, radians: bool = True) -> np.ndarray:
        positions: list[int] = []
        for servo_id in SERVO_IDS:
            position = self._bus.position(servo_id)
            if position is None:
                raise RuntimeError(f"Failed to read RYHand servo {servo_id}")
            positions.append(position)
        angles = motor_positions_to_angles(positions)
        return angles if radians else np.degrees(angles)

    def close(self) -> None:
        if not self._closed:
            self._bus.close()
            self._closed = True

    def __enter__(self) -> "RyHandController":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
