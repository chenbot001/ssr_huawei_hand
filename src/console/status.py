from __future__ import annotations

import ctypes
import importlib.util
import math
import socket
import subprocess
import threading
import time
from typing import Any

from config import get_hardware_config
from control.ryhand_ik import URDF_PATH
from hardware.manus import MANUS_SDK_LIBRARY_PATH, MANUS_SDK_VERSION
from hardware.ruiyan_driver import RYHAND_LIBRARY_PATH


def _result(
    state: str,
    label: str,
    detail: str,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "state": state,
        "label": label,
        "detail": detail,
        "metadata": metadata or {},
    }


def _shared_library(path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"Missing {path}"
    try:
        ctypes.CDLL(str(path))
    except OSError as exc:
        return False, str(exc)
    return True, str(path)


class StatusScanner:
    """Read-only readiness probes used by the System tab."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._last = self._base_state()
        self._ur_receiver: Any | None = None
        self._ur_ip: str | None = None

    @staticmethod
    def _base_state() -> dict[str, Any]:
        hardware = get_hardware_config()
        return {
            "updated_at": None,
            "ur": _result(
                "unknown",
                "Not checked",
                "No RTDE session verified",
                {"IP": str(hardware["ur_arm"]["ip"]), "Backend": "RTDE receive"},
            ),
            "ryhand": _result(
                "unknown",
                "Not checked",
                "CAN state not checked",
                {"Interface": str(hardware["ruiyan_hand"]["port"]), "Bitrate": "1 Mbps"},
            ),
            "manus": _result(
                "unknown",
                "Not checked",
                f"Manus Core {MANUS_SDK_VERSION}",
                {"SDK": MANUS_SDK_VERSION, "Transport": str(hardware["manus_glove"]["address"])},
            ),
            "vive": _result(
                "unknown",
                "Not checked",
                "Configured left/right serials",
                {
                    "Left": str(hardware["vive_tracker"]["left_serial"]),
                    "Right": str(hardware["vive_tracker"]["right_serial"]),
                },
            ),
            "cameras": [],
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._ur_receiver is not None and self._ur_ip is not None:
                result = self._read_ur(self._ur_receiver, self._ur_ip)
                self._last = {**self._last, "ur": result, "updated_at": time.time()}
                if result["state"] != "connected":
                    self._disconnect_ur_locked()
            return self._last

    def scan(self) -> dict[str, Any]:
        hardware = get_hardware_config()
        state = self._base_state()
        ur_ip = str(hardware["ur_arm"]["ip"])
        with self._lock:
            if self._ur_receiver is not None and self._ur_ip == ur_ip:
                state["ur"] = self._read_ur(self._ur_receiver, self._ur_ip)
            else:
                state["ur"] = self.connect_ur(ur_ip)
        state["ryhand"] = self._check_ryhand(
            str(hardware["ruiyan_hand"]["port"])
        )
        state["manus"] = self._check_manus()
        state["vive"] = self._check_vive(hardware["vive_tracker"])
        state["cameras"] = self._check_cameras(hardware.get("rgb_cameras", []))
        state["updated_at"] = time.time()
        with self._lock:
            self._last = state
        return state

    def connect_ur(self, ip: str) -> dict[str, Any]:
        with self._lock:
            self._disconnect_ur_locked()
            receiver, error = self._open_ur(ip)
            if receiver is None:
                result = error
            else:
                result = self._read_ur(receiver, ip)
                if result["state"] == "connected":
                    self._ur_receiver = receiver
                    self._ur_ip = ip
                else:
                    try:
                        receiver.disconnect()
                    except Exception:
                        pass
            self._last = {**self._last, "ur": result, "updated_at": time.time()}
            return result

    def release_ur(self, detail: str = "RTDE status session closed") -> None:
        with self._lock:
            ip = self._ur_ip or str(get_hardware_config()["ur_arm"]["ip"])
            self._disconnect_ur_locked()
            self._last = {
                **self._last,
                "ur": _result(
                    "unknown",
                    "Released",
                    detail,
                    {"IP": ip, "Backend": "RTDE receive"},
                ),
                "updated_at": time.time(),
            }

    def close(self) -> None:
        with self._lock:
            self._disconnect_ur_locked()

    def _disconnect_ur_locked(self) -> None:
        receiver = self._ur_receiver
        self._ur_receiver = None
        self._ur_ip = None
        if receiver is not None:
            try:
                receiver.disconnect()
            except Exception:
                pass

    def refresh_ryhand(self) -> dict[str, Any]:
        interface = str(get_hardware_config()["ruiyan_hand"]["port"])
        result = self._check_ryhand(interface)
        self._last = {**self._last, "ryhand": result, "updated_at": time.time()}
        return result

    @staticmethod
    def _open_ur(ip: str) -> tuple[Any | None, dict[str, Any]]:
        metadata = {"IP": ip, "Backend": "RTDE receive"}
        try:
            with socket.create_connection((ip, 30004), timeout=0.7):
                pass
        except OSError as exc:
            return None, _result("offline", "Disconnected", f"{ip}:30004 — {exc}", metadata)

        try:
            import rtde_receive

            receiver = rtde_receive.RTDEReceiveInterface(ip)
            connected = getattr(receiver, "isConnected", lambda: True)()
            if not connected:
                raise ConnectionError("RTDE receive interface is not connected")
            return receiver, _result("connected", "Connected", "", metadata)
        except Exception as exc:
            return None, _result(
                "offline",
                "Disconnected",
                f"RTDE connection failed: {exc}",
                metadata,
            )

    @staticmethod
    def _read_ur(receiver: Any, ip: str) -> dict[str, Any]:
        metadata = {"IP": ip, "Backend": "RTDE receive"}
        try:
            connected = getattr(receiver, "isConnected", lambda: True)()
            if not connected:
                raise ConnectionError("RTDE receive interface is not connected")
            pose = list(receiver.getActualTCPPose())
            if len(pose) != 6 or not all(math.isfinite(float(value)) for value in pose):
                raise RuntimeError("UR returned an invalid TCP pose")
            robot_mode = int(receiver.getRobotMode())
            safety_mode = int(receiver.getSafetyMode())
            robot_modes = {
                -1: "No controller",
                0: "Disconnected",
                1: "Confirm safety",
                2: "Booting",
                3: "Power off",
                4: "Power on",
                5: "Idle",
                6: "Backdrive",
                7: "Running",
            }
            safety_modes = {
                1: "Normal",
                2: "Reduced",
                3: "Protective stop",
                4: "Recovery",
                5: "Safeguard stop",
                6: "System emergency stop",
                7: "Robot emergency stop",
                8: "Violation",
                9: "Fault",
            }
            metadata.update(
                {
                    "Robot mode": robot_modes.get(robot_mode, str(robot_mode)),
                    "Safety": safety_modes.get(safety_mode, str(safety_mode)),
                    "TCP xyz": " ".join(f"{float(value):.3f}" for value in pose[:3]),
                }
            )
            if robot_mode <= 0:
                raise ConnectionError(f"robot mode is {metadata['Robot mode']}")
            return _result(
                "connected",
                "Connected",
                f"Working RTDE session at {ip}",
                metadata,
            )
        except Exception as exc:
            return _result("offline", "Disconnected", f"RTDE connection failed: {exc}", metadata)

    @classmethod
    def _check_ur(cls, ip: str) -> dict[str, Any]:
        receiver, error = cls._open_ur(ip)
        if receiver is None:
            return error
        try:
            return cls._read_ur(receiver, ip)
        finally:
            try:
                receiver.disconnect()
            except Exception:
                pass

    @staticmethod
    def _check_ryhand(interface: str) -> dict[str, Any]:
        metadata = {"Interface": interface, "Bitrate": "1 Mbps"}
        library_ok, detail = _shared_library(RYHAND_LIBRARY_PATH)
        if not library_ok:
            return _result("offline", "Library unavailable", detail, metadata)
        if not URDF_PATH.is_file():
            return _result("offline", "URDF unavailable", str(URDF_PATH), metadata)
        try:
            process = subprocess.run(
                ["ip", "-details", "link", "show", interface],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _result("offline", "CAN check failed", str(exc), metadata)
        first_line = process.stdout.splitlines()[0] if process.stdout else ""
        is_up = process.returncode == 0 and (
            "state UP" in process.stdout
            or ("<" in first_line and "UP" in first_line.split(">", 1)[0])
        )
        if not is_up:
            return _result(
                "offline",
                "CAN is down",
                f"Bring up {interface} with scripts/ryhand_init.sh",
                metadata,
            )
        metadata["Vendor library"] = "Loaded"
        return _result("ready", "CAN ready", f"{interface}; vendor library loaded", metadata)

    @staticmethod
    def _check_manus() -> dict[str, Any]:
        glove = get_hardware_config()["manus_glove"]
        metadata = {
            "SDK": MANUS_SDK_VERSION,
            "Transport": str(glove["address"]),
            "Gloves": f"L {glove['left_id']} · R {glove['right_id']}",
        }
        library_ok, detail = _shared_library(MANUS_SDK_LIBRARY_PATH)
        if not library_ok:
            return _result("offline", "SDK unavailable", detail, metadata)
        return _result(
            "ready",
            "SDK ready",
            f"Manus Core {MANUS_SDK_VERSION}; connection opens on task start",
            metadata,
        )

    @staticmethod
    def _check_vive(config: dict[str, Any]) -> dict[str, Any]:
        serials = f"{config['left_serial']} / {config['right_serial']}"
        metadata = {
            "Backend": "OpenVR",
            "Left": str(config["left_serial"]),
            "Right": str(config["right_serial"]),
        }
        if importlib.util.find_spec("openvr") is None:
            return _result("offline", "OpenVR unavailable", "Install project dependencies", metadata)
        try:
            from hardware.vive import _find_steamvr_runtime, _vrserver_is_running
            import openvr

            running = _vrserver_is_running()
            runtime = _find_steamvr_runtime(openvr)
        except Exception as exc:
            return _result("offline", "SteamVR check failed", str(exc), metadata)
        if running:
            return _result("ready", "SteamVR running", serials, metadata)
        if runtime is None:
            return _result("offline", "SteamVR unavailable", serials, metadata)
        return _result("ready", "Runtime ready", f"{serials}; starts with task", metadata)

    @staticmethod
    def _check_cameras(configured: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not configured:
            return []
        try:
            import pyrealsense2 as rs
        except ImportError:
            return [
                {
                    "name": str(camera["name"]),
                    "serial": str(camera["serial"]),
                    **_result(
                        "offline",
                        "Driver unavailable",
                        "Install ssr-teleop[rgb]",
                        {"Serial": str(camera["serial"]), "Stream": "RGB"},
                    ),
                }
                for camera in configured
            ]
        try:
            context = rs.context()
            connected = {
                str(device.get_info(rs.camera_info.serial_number))
                for device in context.query_devices()
            }
        except Exception as exc:
            return [
                {
                    "name": str(camera["name"]),
                    "serial": str(camera["serial"]),
                    **_result(
                        "offline",
                        "Enumeration failed",
                        str(exc),
                        {"Serial": str(camera["serial"]), "Stream": "RGB"},
                    ),
                }
                for camera in configured
            ]
        return [
            {
                "name": str(camera["name"]),
                "serial": str(camera["serial"]),
                **_result(
                    "ready" if str(camera["serial"]) in connected else "offline",
                    "Connected" if str(camera["serial"]) in connected else "Not connected",
                    str(camera["serial"]),
                    {"Serial": str(camera["serial"]), "Stream": "RGB"},
                ),
            }
            for camera in configured
        ]
