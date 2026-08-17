from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np


MANUS_SDK_VERSION = "3.1.1"
MANUS_SDK_ROOT = Path(__file__).resolve().parent / "manus_sdk"
MANUS_SDK_CLIENT_PATH = MANUS_SDK_ROOT / "client.py"
MANUS_SDK_LIBRARY_PATH = MANUS_SDK_ROOT / "lib" / "libManusSDK_Integrated.so"
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
TARGET_JOINTS = ("distal", "tip")


@dataclass(frozen=True)
class ManusSample:
    glove_id: int
    hand: str
    fingers: np.ndarray
    wrist: np.ndarray
    timestamp: float
    source_timestamp_ns: int | None = None

    def age(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.timestamp


def _position(bone: Mapping[str, Any]) -> np.ndarray:
    value = bone.get("rawPos", bone.get("pos"))
    position = np.asarray(value, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("Manus bone position must contain three finite values")
    # Preserve the validated Manus-to-RYHand axis convention used by the IK path.
    return np.array((position[0], -position[1], position[2]), dtype=np.float64)


def parse_manus_message(
    message: bytes | str | Mapping[str, Any],
    known_glove_ids: Mapping[str, int] | None = None,
    timestamp: float | None = None,
) -> ManusSample | None:
    """Parse one EgoTac4D Manus 3.1.1 UDP JSON skeleton packet.

    Status packets are ignored. Skeleton packets are fail-closed: the side, glove
    ID, coordinate frame, and all distal/tip positions must be valid.
    """
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    if isinstance(message, str):
        packet = json.loads(message)
    elif isinstance(message, Mapping):
        packet = dict(message)
    else:
        raise ValueError("Manus packet must be a JSON object")
    if not isinstance(packet, dict):
        raise ValueError("Manus packet must be a JSON object")
    packet_type = packet.get("type")
    if packet_type == "manus_glove_status":
        return None
    if packet_type != "manus_hand_skeleton":
        raise ValueError(f"Unexpected Manus packet type: {packet_type!r}")

    hand = packet.get("hand")
    if hand not in ("left", "right"):
        raise ValueError(f"Unknown Manus hand side: {hand!r}")
    glove_id = packet.get("gloveId")
    if isinstance(glove_id, bool) or not isinstance(glove_id, int) or glove_id <= 0:
        raise ValueError("Manus gloveId must be a positive integer")
    expected_id = int((known_glove_ids or {}).get(hand, 0))
    if expected_id and glove_id != expected_id:
        raise ValueError(
            f"Unknown {hand} Manus glove ID {glove_id}; expected {expected_id}"
        )
    coordinate_frame = packet.get("coordinateFrame")
    if coordinate_frame != "openvr_raw_uncalibrated_meters":
        raise ValueError(f"Unexpected Manus coordinate frame: {coordinate_frame!r}")

    bones = packet.get("bones")
    if not isinstance(bones, list):
        raise ValueError("Manus skeleton bones must be a list")
    selected: dict[tuple[str, str], np.ndarray] = {}
    wrist: np.ndarray | None = None
    for bone in bones:
        if not isinstance(bone, Mapping):
            raise ValueError("Every Manus bone must be an object")
        chain = bone.get("chainType")
        joint = bone.get("fingerJointType")
        if chain == "hand" and wrist is None:
            wrist = _position(bone)
        key = (str(chain), str(joint))
        if key in selected:
            raise ValueError(f"Duplicate Manus bone for {key[0]} {key[1]}")
        if key[0] in FINGER_ORDER and key[1] in TARGET_JOINTS:
            selected[key] = _position(bone)

    missing = [
        f"{finger}/{joint}"
        for finger in FINGER_ORDER
        for joint in TARGET_JOINTS
        if (finger, joint) not in selected
    ]
    if missing:
        raise ValueError("Manus skeleton is missing bones: " + ", ".join(missing))
    if wrist is None:
        raise ValueError("Manus skeleton is missing the hand root")

    fingers = np.vstack(
        [selected[(finger, joint)] for finger in FINGER_ORDER for joint in TARGET_JOINTS]
    )
    source_value = packet.get("sourceTimestampNs")
    try:
        source_timestamp_ns = int(source_value) if source_value is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid Manus sourceTimestampNs") from exc
    received_at = time.monotonic() if timestamp is None else float(timestamp)
    fingers.setflags(write=False)
    wrist.setflags(write=False)
    return ManusSample(
        glove_id=glove_id,
        hand=hand,
        fingers=fingers,
        wrist=wrist,
        timestamp=received_at,
        source_timestamp_ns=source_timestamp_ns,
    )


def _parse_udp_address(address: str) -> tuple[str, int]:
    prefix = "udp://"
    if not address.startswith(prefix):
        raise ValueError(f"Manus address must use udp://, got {address!r}")
    host_port = address[len(prefix) :]
    host, separator, port_text = host_port.rpartition(":")
    if not separator or not host:
        raise ValueError(f"Manus address must include host and port: {address!r}")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"Invalid Manus UDP port in {address!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid Manus UDP port in {address!r}")
    return host, port


class ManusReceiver:
    """Own the Manus Core 3.1.1 bridge and expose fresh validated hand samples."""

    def __init__(
        self,
        address: str,
        left_id: int = 0,
        right_id: int = 0,
        *,
        sdk_library: Path = MANUS_SDK_LIBRARY_PATH,
        start_bridge: bool = True,
        command_port: int = 9003,
    ) -> None:
        if left_id < 0 or right_id < 0:
            raise ValueError("Configured Manus glove IDs cannot be negative")
        if not MANUS_SDK_CLIENT_PATH.is_file():
            raise FileNotFoundError(
                f"Manus SDK client is missing: {MANUS_SDK_CLIENT_PATH}"
            )
        if start_bridge and not sdk_library.is_file():
            raise FileNotFoundError(
                f"Manus Core {MANUS_SDK_VERSION} library is missing: {sdk_library}. "
                "Copy libManusSDK_Integrated.so from the Manus 3.1.1 Linux SDK."
            )

        self.address = address
        self._host, self._port = _parse_udp_address(address)
        self._command_port = int(command_port)
        if not 1 <= self._command_port <= 65535:
            raise ValueError("Manus command port must be between 1 and 65535")
        self._configured_ids = {"left": int(left_id), "right": int(right_id)}
        self._resolved_ids = dict(self._configured_ids)
        self._samples: dict[str, ManusSample] = {}
        self._frames: dict[str, dict[str, Any]] = {}
        self._device_status: dict[str, dict[str, Any]] = {}
        self._arrivals: dict[str, deque[float]] = {
            "left": deque(maxlen=120),
            "right": deque(maxlen=120),
        }
        self._calibration: dict[str, Any] = {"active": False, "inProgress": False}
        self._settings: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self.last_error: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._log_thread: threading.Thread | None = None

        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            receiver.bind((self._host, self._port))
        except Exception:
            receiver.close()
            raise
        receiver.settimeout(0.1)
        self._socket = receiver
        self._thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="manus-udp-receiver"
        )
        self._thread.start()
        if start_bridge:
            try:
                self._start_bridge(sdk_library)
            except Exception:
                self.close()
                raise

    def _start_bridge(self, sdk_library: Path) -> None:
        env = os.environ.copy()
        env.update(
            {
                "MANUS_CONNECTION_MODE": "integrated",
                "MANUS_SDK_LIBRARY": str(sdk_library.resolve()),
                "MANUS_OUT_HOST": self._host,
                "MANUS_OUT_PORT": str(self._port),
                "MANUS_COMMAND_PORT": str(self._command_port),
                "MANUS_SEND_RATE": "80",
                "MANUS_SEED_GLOBAL_SETTINGS": "1",
                "MANUS_CALIBRATION_ROOT": str(
                    Path.home() / ".config" / "ssr-teleop" / "manus"
                ),
            }
        )
        self._process = subprocess.Popen(
            [sys.executable, "-u", "-m", "hardware.manus_sdk.client"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._log_thread = threading.Thread(
            target=self._read_bridge_log, daemon=True, name="manus-sdk-log"
        )
        self._log_thread.start()

    def _read_bridge_log(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            message = line.strip()
            if message.startswith("ERROR:"):
                self.last_error = message

    def _receive_loop(self) -> None:
        while self._running.is_set():
            try:
                raw, _sender = self._socket.recvfrom(65535)
            except socket.timeout:
                process = self._process
                if process is not None and process.poll() is not None:
                    self.last_error = self.last_error or (
                        f"Manus SDK bridge exited with code {process.returncode}"
                    )
                continue
            except OSError:
                return
            try:
                packet = json.loads(raw.decode("utf-8"))
                if not isinstance(packet, dict):
                    raise ValueError("Manus packet must be a JSON object")
                if packet.get("type") == "manus_glove_status":
                    self._store_device_status(packet)
                    continue
                with self._lock:
                    known_ids = dict(self._resolved_ids)
                sample = parse_manus_message(packet, known_ids)
                assert sample is not None
                with self._lock:
                    expected = self._resolved_ids[sample.hand]
                    if expected == 0:
                        self._resolved_ids[sample.hand] = sample.glove_id
                    elif expected != sample.glove_id:
                        raise ValueError(
                            f"Unknown {sample.hand} Manus glove ID {sample.glove_id}; "
                            f"expected {expected}"
                        )
                    self._samples[sample.hand] = sample
                    self._frames[sample.hand] = {
                        "bones": self._display_bones(packet),
                        "coordinateFrame": "openvr_raw_uncalibrated_meters",
                        "sourceTimestampNs": sample.source_timestamp_ns,
                    }
                    self._arrivals[sample.hand].append(sample.timestamp)
                self.last_error = None
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self.last_error = str(exc)

    def _store_device_status(self, packet: dict[str, Any]) -> None:
        hand = packet.get("hand")
        if hand not in ("left", "right"):
            raise ValueError(f"Unknown Manus hand side: {hand!r}")
        glove_id = packet.get("gloveId")
        if isinstance(glove_id, bool) or not isinstance(glove_id, int) or glove_id <= 0:
            raise ValueError("Manus gloveId must be a positive integer")
        battery = packet.get("batteryPercentage")
        if not isinstance(battery, int) or isinstance(battery, bool) or not 0 <= battery <= 100:
            battery = None
        raw_family = packet.get("deviceFamily")
        family = None
        if isinstance(raw_family, Mapping):
            family_id = raw_family.get("id")
            family_name = raw_family.get("name")
            if isinstance(family_id, int) and not isinstance(family_id, bool) and isinstance(family_name, str):
                family = {"id": family_id, "name": family_name}
        raw_tunables = packet.get("calibrationTunables")
        tunables = None
        if isinstance(raw_tunables, Mapping):
            tunables = {
                "pinchCompensation": bool(raw_tunables.get("pinchCompensation")),
                "casingCompensation": bool(raw_tunables.get("casingCompensation")),
            }
        with self._lock:
            expected = self._resolved_ids[hand]
            if expected and expected != glove_id:
                raise ValueError(
                    f"Unknown {hand} Manus glove ID {glove_id}; expected {expected}"
                )
            self._device_status[hand] = {
                "gloveId": glove_id,
                "batteryPercentage": battery,
                "deviceFamily": family,
                "calibrationTunables": tunables,
            }

    @staticmethod
    def _display_bones(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for bone in packet.get("bones") or []:
            if not isinstance(bone, Mapping):
                continue
            position = bone.get("rawPos")
            if not isinstance(position, list) or len(position) != 3:
                continue
            try:
                values = [float(value) for value in position]
            except (TypeError, ValueError):
                continue
            if not np.all(np.isfinite(values)):
                continue
            node_id = bone.get("nodeId")
            parent_id = bone.get("parentId")
            if not isinstance(node_id, int) or isinstance(node_id, bool):
                continue
            if not isinstance(parent_id, int) or isinstance(parent_id, bool):
                parent_id = None
            result.append(
                {
                    "nodeId": node_id,
                    "parentId": parent_id,
                    "chainType": str(bone.get("chainType") or "unknown"),
                    "fingerJointType": str(bone.get("fingerJointType") or ""),
                    "rawPos": values,
                }
            )
        return result

    @staticmethod
    def _fps(arrivals: deque[float], now: float) -> float:
        recent = [stamp for stamp in arrivals if now - stamp <= 2.0]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / max(0.001, recent[-1] - recent[0])

    def status(self, include_frames: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            samples = dict(self._samples)
            frames = {hand: dict(frame) for hand, frame in self._frames.items()}
            devices = {hand: dict(value) for hand, value in self._device_status.items()}
            arrivals = {hand: deque(value) for hand, value in self._arrivals.items()}
            calibration = dict(self._calibration)
            settings = dict(self._settings)
        process = self._process
        if process is None:
            state = "disconnected"
        elif process.poll() is not None:
            state = "error"
        else:
            state = "running"
        hands: dict[str, dict[str, Any]] = {}
        for hand in ("left", "right"):
            sample = samples.get(hand)
            device = devices.get(hand) or {}
            age = None if sample is None else max(0.0, sample.age(now))
            glove_id = sample.glove_id if sample is not None else device.get("gloveId")
            calibrating = bool(calibration.get("active")) and int(
                calibration.get("gloveId") or 0
            ) == int(glove_id or 0)
            connected = (sample is not None and age is not None and age <= 2.0) or calibrating
            device_matches = device.get("gloveId") == glove_id
            item: dict[str, Any] = {
                "connected": connected,
                "calibrating": calibrating,
                "streamPaused": calibrating and (sample is None or age is None or age > 2.0),
                "gloveId": glove_id,
                "batteryPercentage": device.get("batteryPercentage") if device_matches else None,
                "deviceFamily": device.get("deviceFamily") if device_matches else None,
                "calibrationTunables": device.get("calibrationTunables") if device_matches else None,
                "boneCount": len((frames.get(hand) or {}).get("bones") or []),
                "fps": round(self._fps(arrivals[hand], now), 1),
                "ageMs": None if age is None else round(age * 1000, 1),
                "coordinateFrame": (frames.get(hand) or {}).get("coordinateFrame"),
                "sourceTimestampNs": (frames.get(hand) or {}).get("sourceTimestampNs"),
            }
            if include_frames and hand in frames:
                item["frame"] = frames[hand]
            hands[hand] = item
        error = self.last_error
        if state == "error" and not error and process is not None:
            error = f"Manus SDK bridge exited with code {process.returncode}"
        return {
            "state": state,
            "error": error,
            "sdkVersion": MANUS_SDK_VERSION,
            "settings": settings,
            "calibration": calibration,
            "hands": hands,
        }

    def command(self, action: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError("Manus bridge is not running")
        request_id = str(time.time_ns())
        payload = json.dumps(
            {"id": request_id, "action": action, "params": dict(params or {})},
            separators=(",", ":"),
        ).encode("utf-8")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as command_socket:
            command_socket.settimeout(5.0)
            try:
                command_socket.sendto(payload, ("127.0.0.1", self._command_port))
                while True:
                    raw, _sender = command_socket.recvfrom(65535)
                    result = json.loads(raw.decode("utf-8"))
                    if not isinstance(result, dict) or result.get("id") != request_id:
                        continue
                    if not result.get("ok"):
                        raise RuntimeError(str(result.get("error") or "Manus SDK command failed"))
                    command_result = result.get("result") or {}
                    if not isinstance(command_result, dict):
                        raise RuntimeError("Manus SDK returned an invalid command result")
                    with self._lock:
                        if action.startswith("calibration_"):
                            self._calibration = dict(command_result)
                        elif action in ("get_settings", "apply_settings"):
                            self._settings = dict(command_result)
                    return dict(command_result)
            except socket.timeout as exc:
                raise RuntimeError("Manus command timed out after 5.0s") from exc
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Manus command failed: {exc}") from exc

    def get_latest(self, use_right: bool = False) -> ManusSample | None:
        hand = "right" if use_right else "left"
        with self._lock:
            return self._samples.get(hand)

    def wait_for_sample(
        self, use_right: bool = False, timeout: float = 5.0
    ) -> ManusSample | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sample = self.get_latest(use_right)
            if sample is not None:
                return sample
            process = self._process
            if process is not None and process.poll() is not None:
                return None
            time.sleep(0.01)
        return None

    def close(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        self._socket.close()
        self._thread.join(timeout=1.0)
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        if self._log_thread is not None:
            self._log_thread.join(timeout=1.0)

    def __enter__(self) -> "ManusReceiver":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
