from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from config import get_hardware_config, get_teleop_config


class UR5Arm:
    """UR RTDE control with servo parameters hidden behind a small interface."""

    def __init__(
        self,
        ip: str | None = None,
        servo_config: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import rtde_control
            import rtde_receive
        except ImportError as exc:
            raise ImportError(
                "ur_rtde is required for UR5 control; install the project dependencies"
            ) from exc

        self.ip = ip or str(get_hardware_config()["ur_arm"]["ip"])
        self._servo = dict(servo_config or get_teleop_config()["servo"])
        self._control = rtde_control.RTDEControlInterface(self.ip)
        try:
            self._receive = rtde_receive.RTDEReceiveInterface(self.ip)
        except Exception:
            self._control.disconnect()
            raise
        if not self._control.isConnected() or not self._receive.isConnected():
            self._control.disconnect()
            self._receive.disconnect()
            raise ConnectionError(f"UR RTDE did not connect to {self.ip}")
        self._closed = False

    def get_tcp_pose(self) -> list[float]:
        return list(self._receive.getActualTCPPose())

    def servo_l(self, pose: Sequence[float]) -> None:
        if len(pose) != 6:
            raise ValueError(f"UR TCP pose must contain 6 values, got {len(pose)}")
        accepted = self._control.servoL(
            list(pose),
            float(self._servo["speed"]),
            float(self._servo["acceleration"]),
            float(self._servo["dt"]),
            float(self._servo["lookahead_time"]),
            float(self._servo["gain"]),
        )
        if accepted is False:
            raise RuntimeError("UR rejected the servoL command")

    def is_pose_within_safety_limits(self, pose: Sequence[float]) -> bool:
        if len(pose) != 6:
            raise ValueError(f"UR TCP pose must contain 6 values, got {len(pose)}")
        checker = getattr(self._control, "isPoseWithinSafetyLimits", None)
        if checker is None:
            raise RuntimeError("ur_rtde does not provide isPoseWithinSafetyLimits")
        return bool(checker(list(pose)))

    def servo_stop(self) -> None:
        if not self._closed:
            self._control.servoStop()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._control.servoStop()
        finally:
            try:
                self._control.stopScript()
            finally:
                try:
                    self._control.disconnect()
                finally:
                    self._receive.disconnect()
                    self._closed = True

    def __enter__(self) -> "UR5Arm":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
