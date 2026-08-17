from __future__ import annotations

from collections import deque
import binascii
from datetime import datetime
import struct
import threading
import time
from typing import Any
import zlib

import numpy as np

from config import get_hardware_config, get_teleop_config
from control.ryhand_ik import RYHandIK, load_calibration, save_calibration
from control.teleop import TeleopController, TeleopSettings
from hardware.arm_ur5 import UR5Arm
from hardware.manus import MANUS_SDK_VERSION, ManusReceiver
from hardware.ruiyan_driver import RyHandController
from hardware.vive import ViveTracker


TELEOP_MODES = {
    "full": {"label": "Full system", "arm": True, "hand": True, "virtual": False},
    "arm": {"label": "UR + Vive", "arm": True, "hand": False, "virtual": False},
    "hand": {"label": "RYHand + Manus", "arm": False, "hand": True, "virtual": False},
    "simulation": {
        "label": "Virtual UR + RYHand",
        "arm": True,
        "hand": True,
        "virtual": True,
    },
}


def encode_rgba_png(image: np.ndarray) -> bytes:
    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("PNG input must have shape (height, width, 4)")
    height, width, _channels = pixels.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(
            ">I", binascii.crc32(payload) & 0xFFFFFFFF
        )

    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, 6))
        + chunk(b"IEND", b"")
    )


class EventLog:
    def __init__(self, limit: int = 240) -> None:
        self._lines: deque[dict[str, str]] = deque(maxlen=limit)
        self._lock = threading.Lock()

    def add(self, source: str, message: str, level: str = "info") -> None:
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "source": source,
            "level": level,
            "message": str(message),
        }
        with self._lock:
            self._lines.append(entry)

    def snapshot(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._lines)


class VirtualArm:
    def __init__(self) -> None:
        self.pose = np.array([0.42, 0.0, 0.32, 0.0, 0.0, 0.0], dtype=np.float64)

    def get_tcp_pose(self) -> list[float]:
        return self.pose.tolist()

    def servo_l(self, pose) -> None:
        self.pose = np.asarray(pose, dtype=np.float64).copy()

    def servo_stop(self) -> None:
        return None

    def is_pose_within_safety_limits(self, pose) -> bool:
        return bool(np.isfinite(np.asarray(pose, dtype=np.float64)).all())

    def close(self) -> None:
        return None


class VirtualHand:
    def __init__(self) -> None:
        self.angles = np.zeros(15, dtype=np.float64)

    def set_angles(self, angles, speed: int, radians: bool = True) -> None:
        values = np.asarray(angles, dtype=np.float64)
        self.angles = values.copy() if radians else np.radians(values)

    def close(self) -> None:
        return None


def _close(resources: list[Any]) -> None:
    for resource in reversed(resources):
        try:
            resource.close()
        except Exception:
            pass


class _BorrowedResource:
    """Let a controller use a console-owned input without closing it."""

    def __init__(self, resource: Any) -> None:
        self._resource = resource

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resource, name)

    def close(self) -> None:
        return None


MANUS_DESTINATIONS = frozenset({"system", "manus", "ryhand", "teleop"})


class RoutedManusInput:
    """Expose MANUS samples only while one console tab owns their destination."""

    def __init__(self, runtime: "ManusRuntime", destination: str) -> None:
        self._runtime = runtime
        self.destination = destination
        self._closed = False

    @property
    def last_error(self) -> str | None:
        return self._runtime.input_error(self.destination)

    def get_latest(self, use_right: bool = False):
        return self._runtime.get_latest(self.destination, use_right)

    def status(self, include_frames: bool = False) -> dict[str, Any]:
        return self._runtime.input_status(self.destination, include_frames)

    def close(self) -> None:
        if self._closed:
            return
        self._runtime.release(self.destination)
        self._closed = True


class ManusRuntime:
    """Own and route the console's single MANUS SDK stream."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._receiver: ManusReceiver | None = None
        self._destination = "system"
        self._consumers: dict[str, int] = {}
        self._state = "stopped"
        self._detail = "Start the shared MANUS stream explicitly"

    def active(self) -> bool:
        with self._lock:
            return self._receiver is not None

    def start(self) -> ManusReceiver:
        with self._ready:
            while self._state == "starting":
                self._ready.wait()
            if self._receiver is not None:
                return self._receiver
            self._state = "starting"
            self._detail = f"Opening the MANUS {MANUS_SDK_VERSION} bridge"
        receiver: ManusReceiver | None = None
        try:
            hardware = get_hardware_config()["manus_glove"]
            receiver = ManusReceiver(
                str(hardware["address"]),
                int(hardware["left_id"]),
                int(hardware["right_id"]),
            )
            with self._ready:
                self._receiver = receiver
                self._state = "running"
                self._detail = "Shared SDK bridge running; waiting for gloves"
                self._ready.notify_all()
            self.log.add("manus", f"Shared MANUS Core {MANUS_SDK_VERSION} stream started")
            return receiver
        except Exception as exc:
            if receiver is not None:
                receiver.close()
            with self._ready:
                self._state = "fault"
                self._detail = str(exc)
                self._ready.notify_all()
            self.log.add("manus", f"Start failed: {exc}", "error")
            raise

    def stop(self) -> None:
        with self._ready:
            while self._state == "starting":
                self._ready.wait()
            receiver = self._receiver
            self._receiver = None
            self._state = "stopped"
            self._detail = "Shared MANUS stream stopped"
        if receiver is not None:
            receiver.close()
            self.log.add("manus", "Shared MANUS stream stopped")

    def route_to(self, destination: str) -> None:
        if destination not in MANUS_DESTINATIONS:
            raise ValueError(f"Unknown MANUS destination: {destination}")
        with self._lock:
            previous = self._destination
            self._destination = destination
        if previous != destination:
            self.log.add("manus", f"Stream routed to {destination} tab")

    def open_input(self, destination: str) -> RoutedManusInput:
        if destination not in ("ryhand", "teleop"):
            raise ValueError(f"{destination} cannot consume MANUS control samples")
        while True:
            receiver = self.start()
            with self._lock:
                if self._receiver is receiver:
                    self._consumers[destination] = (
                        self._consumers.get(destination, 0) + 1
                    )
                    return RoutedManusInput(self, destination)

    def release(self, destination: str) -> None:
        with self._lock:
            count = self._consumers.get(destination, 0)
            if count <= 1:
                self._consumers.pop(destination, None)
            else:
                self._consumers[destination] = count - 1

    def get_latest(self, destination: str, use_right: bool = False):
        with self._lock:
            receiver = self._receiver
            selected = self._destination
        if receiver is None or selected != destination:
            return None
        return receiver.get_latest(use_right)

    def input_error(self, destination: str) -> str | None:
        with self._lock:
            receiver = self._receiver
            selected = self._destination
            detail = self._detail
        if selected != destination:
            return f"MANUS stream is routed to the {selected} tab"
        return detail if receiver is None else receiver.last_error

    def input_status(
        self, destination: str, include_frames: bool = False
    ) -> dict[str, Any]:
        with self._lock:
            receiver = self._receiver
            selected = self._destination
        if receiver is None:
            return self.snapshot()
        return receiver.status(include_frames=include_frames and selected == destination)

    def _require_receiver(self) -> ManusReceiver:
        with self._lock:
            receiver = self._receiver
        if receiver is None:
            raise RuntimeError("Start the shared MANUS stream first")
        return receiver

    def _require_manus_control(self) -> ManusReceiver:
        receiver = self._require_receiver()
        with self._lock:
            destination = self._destination
        if destination != "manus":
            raise RuntimeError(f"MANUS controls belong to the {destination} tab")
        return receiver

    def settings(self) -> dict[str, Any]:
        result = self._require_manus_control().command("get_settings")
        self.log.add("manus", "Loaded active SDK skeleton settings")
        return result

    def apply_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        params = {
            "handMotion": int(values.get("handMotion", 4)),
            "pinchCompensation": bool(values.get("pinchCompensation", False)),
            "casingCompensation": float(values.get("casingCompensation", 0.0)),
        }
        result = self._require_manus_control().command("apply_settings", params)
        self.log.add("manus", "Applied active SDK skeleton settings")
        return result

    def calibration_start(self, hand: str) -> dict[str, Any]:
        if hand not in ("left", "right"):
            raise ValueError("MANUS calibration hand must be left or right")
        receiver = self._require_manus_control()
        item = receiver.status()["hands"][hand]
        glove_id = item.get("gloveId")
        if not item.get("connected") or not glove_id:
            raise RuntimeError(f"{hand} MANUS glove is not connected")
        result = receiver.command("calibration_start", {"gloveId": int(glove_id)})
        self.log.add("manus", f"Official calibration started for {hand} glove {glove_id}")
        return result

    def _calibration_command(self, action: str, **extra: Any) -> dict[str, Any]:
        receiver = self._require_manus_control()
        calibration = receiver.status()["calibration"]
        glove_id = int(calibration.get("gloveId") or 0)
        if glove_id <= 0:
            raise RuntimeError("No MANUS calibration session is active")
        return receiver.command(action, {"gloveId": glove_id, **extra})

    def calibration_step(self) -> dict[str, Any]:
        receiver = self._require_manus_control()
        calibration = receiver.status()["calibration"]
        step_index = int(calibration.get("completedStepIndex", -1)) + 1
        result = self._calibration_command("calibration_step", stepIndex=step_index)
        self.log.add("manus", f"Calibration step {step_index + 1} started")
        return result

    def calibration_status(self) -> dict[str, Any]:
        return self._calibration_command("calibration_status")

    def calibration_finish(self) -> dict[str, Any]:
        result = self._calibration_command("calibration_finish")
        self.log.add("manus", "Official MANUS calibration saved")
        return result

    def calibration_cancel(self) -> dict[str, Any]:
        result = self._calibration_command("calibration_cancel")
        self.log.add("manus", "Official MANUS calibration cancelled", "warning")
        return result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            receiver = self._receiver
            destination = self._destination
            consumers = sorted(self._consumers)
            state = self._state
            detail = self._detail
        if receiver is None:
            return {
                "active": False,
                "destination": destination,
                "consumers": consumers,
                "state": state,
                "detail": detail,
                "error": detail if state == "fault" else None,
                "sdkVersion": MANUS_SDK_VERSION,
                "settings": {},
                "calibration": {"active": False, "inProgress": False},
                "hands": {
                    hand: {
                        "connected": False,
                        "gloveId": None,
                        "batteryPercentage": None,
                        "boneCount": 0,
                        "fps": 0.0,
                        "ageMs": None,
                    }
                    for hand in ("left", "right")
                },
            }
        result = receiver.status(include_frames=destination == "manus")
        result["active"] = True
        result["destination"] = destination
        result["consumers"] = consumers
        result["detail"] = result.get("error") or detail
        if result["state"] == "error":
            with self._lock:
                self._state = "fault"
                self._detail = result["detail"]
        return result


class TeleopRuntime:
    def __init__(self, log: EventLog, manus_runtime: ManusRuntime) -> None:
        self.log = log
        self._manus_runtime = manus_runtime
        self._lock = threading.Lock()
        self._inputs_ready = threading.Condition(self._lock)
        self._preview_change = threading.Lock()
        self._preview_stop = threading.Event()
        self._preview_thread: threading.Thread | None = None
        self._preview_state = "stopped"
        self._preview_detail = "Open the Teleop tab to preview live inputs"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._controller: TeleopController | None = None
        self._vive: ViveTracker | None = None
        self._manus: RoutedManusInput | None = None
        self._mode: str | None = None
        self._vive_side = "left"
        self._manus_side = "left"
        self._state = "stopped"
        self._detail = "Select a mode and start explicitly"
        self._started_at: float | None = None
        self._arm_connected = False
        self._hand_connected = False

    def active(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def preview_active(self) -> bool:
        with self._lock:
            return self._preview_thread is not None and self._preview_thread.is_alive()

    def start_preview(
        self, vive_side: str = "left", manus_side: str = "left"
    ) -> None:
        if vive_side not in ("left", "right"):
            raise ValueError("Vive side must be left or right")
        if manus_side not in ("left", "right"):
            raise ValueError("MANUS side must be left or right")
        with self._preview_change:
            with self._lock:
                control_active = self._thread is not None and self._thread.is_alive()
                current_preview = (
                    self._preview_thread is not None
                    and self._preview_thread.is_alive()
                )
                if control_active and (
                    self._vive_side != vive_side or self._manus_side != manus_side
                ):
                    raise RuntimeError("Stop teleoperation before changing input side")
                if (
                    current_preview
                    and self._vive_side == vive_side
                    and self._manus_side == manus_side
                ):
                    return
            self._stop_preview()
            with self._lock:
                self._vive_side = vive_side
                self._manus_side = manus_side
                self._preview_stop.clear()
                self._preview_state = "starting"
                self._preview_detail = (
                    f"Opening {vive_side} Vive and {manus_side} MANUS inputs"
                )
                self._preview_thread = threading.Thread(
                    target=self._run_preview,
                    args=(vive_side, manus_side),
                    daemon=True,
                    name="console-teleop-preview",
                )
                self._preview_thread.start()
        self.log.add(
            "teleop",
            f"Starting {vive_side} Vive + {manus_side} MANUS preview",
        )

    def _run_preview(self, vive_side: str, manus_side: str) -> None:
        resources: list[Any] = []
        errors: list[str] = []
        hardware = get_hardware_config()
        serial_key = f"{vive_side}_serial"
        try:
            try:
                vive = ViveTracker(str(hardware["vive_tracker"][serial_key]))
                resources.append(vive)
                with self._inputs_ready:
                    self._vive = vive
                    self._inputs_ready.notify_all()
            except Exception as exc:
                errors.append(f"Vive: {exc}")
            try:
                manus = self._manus_runtime.open_input("teleop")
                resources.append(manus)
                with self._inputs_ready:
                    self._manus = manus
                    self._inputs_ready.notify_all()
            except Exception as exc:
                errors.append(f"MANUS: {exc}")
            with self._inputs_ready:
                if self._vive is not None or self._manus is not None:
                    self._preview_state = "preview"
                    self._preview_detail = (
                        "Live tracker and skeleton preview"
                        if not errors
                        else "; ".join(errors)
                    )
                else:
                    self._preview_state = "fault"
                    self._preview_detail = "; ".join(errors) or "No preview input opened"
                self._inputs_ready.notify_all()
            while resources and not self._preview_stop.wait(0.1):
                pass
        finally:
            _close(resources)
            with self._inputs_ready:
                self._vive = None
                self._manus = None
                if self._preview_state != "fault":
                    self._preview_state = "stopped"
                    self._preview_detail = "Live-input preview stopped"
                self._inputs_ready.notify_all()

    def _stop_preview(self) -> None:
        self._preview_stop.set()
        with self._lock:
            thread = self._preview_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=12.0)

    def stop_preview(self) -> None:
        with self._preview_change:
            if self.active():
                return
            self._stop_preview()

    def _wait_for_inputs(
        self, require_vive: bool, require_manus: bool, timeout: float
    ) -> tuple[ViveTracker | None, RoutedManusInput | None]:
        deadline = time.monotonic() + timeout
        with self._inputs_ready:
            while True:
                vive_ready = not require_vive or self._vive is not None
                manus_ready = not require_manus or self._manus is not None
                if vive_ready and manus_ready:
                    return self._vive, self._manus
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(self._preview_detail)
                self._inputs_ready.wait(min(0.05, remaining))

    def start(
        self, mode: str, vive_side: str = "left", manus_side: str = "left"
    ) -> None:
        if mode not in TELEOP_MODES:
            raise ValueError(f"Unknown teleop mode: {mode}")
        self.start_preview(vive_side, manus_side)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Teleoperation is already running")
            self._stop.clear()
            self._mode = mode
            self._vive_side = vive_side
            self._manus_side = manus_side
            self._state = "starting"
            self._detail = "Opening selected inputs"
            self._started_at = time.monotonic()
            self._arm_connected = False
            self._hand_connected = False
            self._thread = threading.Thread(
                target=self._run,
                args=(mode, manus_side),
                daemon=True,
                name="console-teleop",
            )
            self._thread.start()
        self.log.add(
            "teleop",
            f"Starting {TELEOP_MODES[mode]['label']} "
            f"(Vive {vive_side}, MANUS {manus_side})",
        )

    def _report(self, message: str) -> None:
        level = (
            "warning"
            if message.startswith("CLUTCH ENGAGED:")
            and "operator engaged" not in message
            else "info"
        )
        self.log.add("teleop", message, level)

    def _run(self, mode: str, manus_side: str) -> None:
        resources: list[Any] = []
        controller: TeleopController | None = None
        spec = TELEOP_MODES[mode]
        try:
            hardware = get_hardware_config()
            settings = TeleopSettings.from_config(get_teleop_config())
            vive, manus = self._wait_for_inputs(
                bool(spec["arm"]), bool(spec["hand"]), 8.0
            )
            ik = None
            arm = None
            hand = None

            if spec["hand"]:
                ik = RYHandIK(gui=False)
                resources.append(ik)

            with self._lock:
                self._state = "waiting"
                self._detail = "Waiting for fresh selected inputs"
            deadline = time.monotonic() + 8.0
            while not self._stop.is_set() and time.monotonic() < deadline:
                vive_ready = not spec["arm"] or vive.get_latest() is not None
                manus_ready = (
                    not spec["hand"]
                    or manus.get_latest(manus_side == "right") is not None
                )
                if vive_ready and manus_ready:
                    break
                self._stop.wait(0.02)
            if self._stop.is_set():
                return
            if spec["arm"] and vive.get_latest() is None:
                raise RuntimeError(vive.last_error or "No fresh Vive tracker sample")
            if spec["hand"] and manus.get_latest(manus_side == "right") is None:
                raise RuntimeError(manus.last_error or "No fresh Manus glove sample")

            if spec["virtual"]:
                arm = VirtualArm()
                hand = VirtualHand()
            else:
                if spec["arm"]:
                    arm = UR5Arm(ip=str(hardware["ur_arm"]["ip"]))
                    resources.append(arm)
                    with self._lock:
                        self._arm_connected = True
                    if self._stop.is_set():
                        return
                if spec["hand"]:
                    hand = RyHandController(port=str(hardware["ruiyan_hand"]["port"]))
                    resources.append(hand)
                    with self._lock:
                        self._hand_connected = True
                    if self._stop.is_set():
                        return

            controller = TeleopController(
                arm,
                hand,
                None if vive is None else _BorrowedResource(vive),
                None if manus is None else _BorrowedResource(manus),
                ik,
                settings,
                use_right_manus=manus_side == "right",
                report=self._report,
                control_arm=bool(spec["arm"]),
                control_hand=bool(spec["hand"]),
            )
            resources.clear()
            with self._lock:
                self._controller = controller
                self._state = "clutched" if spec["arm"] else "hand-tracking"
                self._detail = (
                    "RYHand tracking active; UR held. Press Space to release clutch"
                    if spec["arm"] and spec["hand"]
                    else "UR held. Press Space to release clutch"
                    if spec["arm"]
                    else "RYHand follows fresh MANUS samples continuously"
                )
            interval = 1.0 / settings.update_rate
            while not self._stop.is_set():
                started = time.monotonic()
                controller.step(started)
                with self._lock:
                    hand_detail = controller.last_hand_stop_reason
                    if spec["arm"] and not controller.clutch_engaged:
                        self._state = "tracking"
                        self._detail = "Tracker deltas drive UR"
                    elif spec["arm"]:
                        self._state = "clutched"
                        self._detail = controller.last_stop_reason or "UR held"
                    else:
                        self._state = "hand-tracking"
                        self._detail = "RYHand follows MANUS continuously"
                    if hand_detail:
                        self._detail += f"; {hand_detail}"
                self._stop.wait(max(0.0, interval - (time.monotonic() - started)))
        except Exception as exc:
            with self._lock:
                self._state = "fault"
                self._detail = str(exc)
            self.log.add("teleop", f"Fault: {exc}", "error")
        finally:
            if controller is not None:
                try:
                    controller.close()
                except Exception as exc:
                    self.log.add("teleop", f"Cleanup warning: {exc}", "warning")
            else:
                _close(resources)
            with self._lock:
                self._controller = None
                self._arm_connected = False
                self._hand_connected = False
                if self._state not in ("fault",):
                    self._state = "stopped"
                    self._detail = "Stopped; outputs held at their last pose"
            self.log.add("teleop", "Stopped; UR servo stopped and RYHand held")

    def toggle(self) -> None:
        with self._lock:
            controller = self._controller
        if controller is None:
            raise RuntimeError("Teleoperation is not ready")
        controller.request_toggle()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=12.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            controller = self._controller
            vive = self._vive
            manus = self._manus
            mode = self._mode
            state = self._state
            detail = self._detail
            vive_side = self._vive_side
            manus_side = self._manus_side
            started_at = self._started_at
            arm_connected = self._arm_connected
            hand_connected = self._hand_connected
            preview_active = (
                self._preview_thread is not None and self._preview_thread.is_alive()
            )
            preview_state = self._preview_state
            preview_detail = self._preview_detail
        now = time.monotonic()
        vive_sample = vive.get_latest() if vive is not None else None
        manus_sample = (
            manus.get_latest(manus_side == "right") if manus is not None else None
        )
        manus_hand = None
        if manus is not None:
            manus_status = manus.status(include_frames=True)["hands"][manus_side]
            manus_hand = {
                "side": manus_side,
                "connected": bool(manus_status.get("connected")),
                "glove_id": manus_status.get("gloveId"),
                "frame": manus_status.get("frame"),
            }
        target = controller.last_target_pose if controller is not None else None
        angles = controller.last_hand_angles if controller is not None else None
        delta = (
            controller.motion_delta
            if controller is not None and controller.armed
            else np.zeros(6, dtype=np.float64)
        )
        vive_pose = None
        if vive_sample is not None:
            vive_pose = {
                "serial": vive.serial,
                "position": np.round(vive_sample.position, 6).tolist(),
                "quaternion": np.round(vive_sample.quaternion, 7).tolist(),
            }
        return {
            "active": self.active(),
            "mode": mode,
            "mode_label": TELEOP_MODES[mode]["label"] if mode else None,
            "vive_side": vive_side,
            "manus_side": manus_side,
            "state": preview_state if not self.active() and state == "stopped" else state,
            "detail": preview_detail if not self.active() and state == "stopped" else detail,
            "preview_active": preview_active,
            "armed": bool(controller and controller.armed),
            "arm_tracking": bool(controller and controller.armed),
            "clutch_engaged": bool(
                controller is None or controller.clutch_engaged
            ),
            "arm_connected": arm_connected,
            "hand_connected": hand_connected,
            "uptime": None if started_at is None else max(0.0, now - started_at),
            "vive_age": None if vive_sample is None else max(0.0, vive_sample.age(now)),
            "manus_age": None if manus_sample is None else max(0.0, manus_sample.age(now)),
            "vive_pose": vive_pose,
            "manus_hand": manus_hand,
            "target_pose": None if target is None else np.round(target, 5).tolist(),
            "motion_delta": {
                "translation_m": np.round(delta[:3], 7).tolist(),
                "rotation_rad": np.round(delta[3:], 7).tolist(),
            },
            "hand_angles": None if angles is None else np.round(angles, 5).tolist(),
        }


class CalibrationRuntime:
    def __init__(self, log: EventLog, manus_runtime: ManusRuntime) -> None:
        self.log = log
        self._manus_runtime = manus_runtime
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ik: RYHandIK | None = None
        self._manus: RoutedManusInput | None = None
        self._hand: RyHandController | None = None
        self._state = "stopped"
        self._detail = "Calibration is not running"
        self._side = "left"
        self._live_output = False
        self._frame: bytes | None = None
        self._angles: list[float] | None = None
        self._scales, self._offsets = load_calibration()

    def active(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self, use_right: bool = False, live_output: bool = False) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Calibration is already running")
            self._stop.clear()
            self._side = "right" if use_right else "left"
            self._live_output = bool(live_output)
            self._state = "starting"
            self._detail = "Opening shared Manus input and PyBullet"
            self._frame = None
            self._angles = None
            self._thread = threading.Thread(
                target=self._run,
                args=(use_right, bool(live_output)),
                daemon=True,
                name="console-calibration",
            )
            self._thread.start()
        output = "live RYHand output" if live_output else "simulation only"
        self.log.add("calibrate", f"Starting {self._side} calibration ({output})")

    def _run(self, use_right: bool, live_output: bool) -> None:
        resources: list[Any] = []
        try:
            hardware = get_hardware_config()
            manus = self._manus_runtime.open_input("ryhand")
            resources.append(manus)
            ik = RYHandIK(gui=False)
            resources.append(ik)
            with self._lock:
                self._manus = manus
                self._ik = ik
                ik.set_calibration(self._scales, self._offsets)
                self._state = "waiting"
                self._detail = "Waiting for a fresh Manus sample"
            deadline = time.monotonic() + 8.0
            sample = manus.get_latest(use_right)
            while sample is None and not self._stop.is_set() and time.monotonic() < deadline:
                self._stop.wait(0.02)
                sample = manus.get_latest(use_right)
            if self._stop.is_set():
                return
            if sample is None:
                raise RuntimeError(manus.last_error or "No fresh Manus glove sample")
            if live_output:
                hand = RyHandController(port=str(hardware["ruiyan_hand"]["port"]))
                resources.append(hand)
                if self._stop.is_set():
                    return
                with self._lock:
                    self._hand = hand
            with self._lock:
                self._state = "running"
                self._detail = "Adjust thumb/index mapping; Save writes atomically"
            next_render = 0.0
            while not self._stop.is_set():
                sample = manus.get_latest(use_right)
                if sample is None or sample.age() > 0.25:
                    with self._lock:
                        self._state = "stale"
                        self._detail = "Manus input stale; RYHand held"
                    self._stop.wait(0.02)
                    continue
                angles = ik.compute_hand_angles(sample.fingers)
                if angles is None:
                    raise RuntimeError("RYHand IK failed")
                if live_output:
                    hand.set_angles(angles, speed=500, radians=True)
                now = time.monotonic()
                frame = None
                if now >= next_render:
                    frame = encode_rgba_png(ik.render_rgba(720, 540))
                    next_render = now + 0.1
                with self._lock:
                    self._state = "running"
                    self._angles = np.round(angles, 5).tolist()
                    if frame is not None:
                        self._frame = frame
                self._stop.wait(0.0125)
        except Exception as exc:
            with self._lock:
                self._state = "fault"
                self._detail = str(exc)
            self.log.add("calibrate", f"Fault: {exc}", "error")
        finally:
            _close(resources)
            with self._lock:
                self._ik = None
                self._manus = None
                self._hand = None
                if self._state != "fault":
                    self._state = "stopped"
                    self._detail = "Stopped; RYHand held at its last pose"
            self.log.add("calibrate", "Calibration stopped")

    def update(self, values: dict[str, Any]) -> None:
        scales = self._scales.copy()
        offsets = self._offsets.copy()
        for finger, index in (("thumb", 0), ("index", 1)):
            item = values.get(finger)
            if not isinstance(item, dict):
                raise ValueError(f"Missing {finger} calibration")
            scale = float(item["scale"])
            offset = np.array(
                [float(item[axis]) for axis in ("x", "y", "z")],
                dtype=np.float64,
            )
            if not 0.1 <= scale <= 3.0:
                raise ValueError(f"{finger} scale must be between 0.1 and 3.0")
            if np.any(np.abs(offset) > 0.2):
                raise ValueError(f"{finger} offsets must be within ±0.2 m")
            scales[index] = scale
            offsets[index] = offset
        with self._lock:
            self._scales = scales
            self._offsets = offsets
            ik = self._ik
            if ik is not None:
                ik.set_calibration(scales, offsets)

    def save(self) -> None:
        with self._lock:
            scales = self._scales.copy()
            offsets = self._offsets.copy()
        save_calibration(scales, offsets)
        self.log.add("calibrate", "Saved thumb/index calibration atomically")

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=12.0)

    def frame(self) -> bytes | None:
        with self._lock:
            return self._frame

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._thread is not None and self._thread.is_alive(),
                "state": self._state,
                "detail": self._detail,
                "side": self._side,
                "live_output": self._live_output,
                "angles": self._angles,
                "frame_ready": self._frame is not None,
                "thumb": {
                    "scale": float(self._scales[0]),
                    "x": float(self._offsets[0, 0]),
                    "y": float(self._offsets[0, 1]),
                    "z": float(self._offsets[0, 2]),
                },
                "index": {
                    "scale": float(self._scales[1]),
                    "x": float(self._offsets[1, 0]),
                    "y": float(self._offsets[1, 1]),
                    "z": float(self._offsets[1, 2]),
                },
            }


class CameraRuntime:
    def __init__(self, log: EventLog) -> None:
        self.log = log
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: bytes | None = None
        self._state = "stopped"
        self._detail = "No preview active"
        self._serial: str | None = None

    def start(self, serial: str) -> None:
        configured = get_hardware_config().get("rgb_cameras", [])
        if serial not in {str(item["serial"]) for item in configured}:
            raise ValueError("Camera serial is not configured")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("A camera preview is already running")
            self._serial = serial
            self._state = "starting"
            self._detail = f"Opening {serial}"
            self._frame = None
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(serial,),
                daemon=True,
                name="console-rgb-camera",
            )
            self._thread.start()

    def _run(self, serial: str) -> None:
        pipeline = None
        try:
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgba8, 30)
            pipeline.start(config)
            with self._lock:
                self._state = "running"
                self._detail = f"Live RGB preview from {serial}"
            while not self._stop.is_set():
                frames = pipeline.wait_for_frames(1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                frame = np.asanyarray(color.get_data())
                encoded = encode_rgba_png(frame)
                with self._lock:
                    self._frame = encoded
        except Exception as exc:
            with self._lock:
                self._state = "fault"
                self._detail = str(exc)
            self.log.add("camera", f"Fault: {exc}", "error")
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            with self._lock:
                if self._state != "fault":
                    self._state = "stopped"
                    self._detail = "Preview stopped"

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)

    def frame(self) -> bytes | None:
        with self._lock:
            return self._frame

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._thread is not None and self._thread.is_alive(),
                "state": self._state,
                "detail": self._detail,
                "serial": self._serial,
                "frame_ready": self._frame is not None,
            }
