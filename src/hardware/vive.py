from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import signal
import subprocess
import threading
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ViveSample:
    position: np.ndarray
    quaternion: np.ndarray
    timestamp: float
    generation: int

    def age(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.timestamp


def _matrix34_to_components(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.array(
        [[float(matrix[row][column]) for column in range(3)] for row in range(3)],
        dtype=np.float64,
    )
    position = np.array(
        [float(matrix[row][3]) for row in range(3)], dtype=np.float64
    )
    quaternion = Rotation.from_matrix(rotation).as_quat()
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("Vive pose contains non-finite values")
    return position, quaternion


def _find_steamvr_runtime(openvr: Any) -> Path | None:
    override = os.environ.get("STEAMVR_RUNTIME")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_dir():
            return candidate
    try:
        candidate = Path(openvr.getRuntimePath())
        if candidate.is_dir():
            return candidate
    except Exception:
        pass
    home = Path.home()
    candidates = (
        home / ".steam/debian-installation/steamapps/common/SteamVR",
        home / ".steam/steam/steamapps/common/SteamVR",
        home / ".local/share/Steam/steamapps/common/SteamVR",
    )
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def _headless_vrserver_command(runtime_root: Path | str) -> list[str]:
    root = Path(runtime_root)
    return [
        str(root / "bin" / "vrenv.sh"),
        str(root / "bin" / "linux64" / "vrserver"),
        "-keepalive",
    ]


def _is_vrserver_process(pid: int) -> bool:
    try:
        command = (Path("/proc") / str(pid) / "comm").read_text(
            encoding="utf-8"
        ).strip()
        state = (
            (Path("/proc") / str(pid) / "stat")
            .read_text(encoding="utf-8")
            .rsplit(")", 1)[1]
            .split()[0]
        )
    except (IndexError, OSError):
        return False
    return command == "vrserver" and state != "Z"


def _vrserver_is_running() -> bool:
    try:
        return any(
            entry.name.isdigit() and _is_vrserver_process(int(entry.name))
            for entry in Path("/proc").iterdir()
        )
    except OSError:
        return False


def _start_headless_vrserver(openvr: Any) -> subprocess.Popen[bytes] | None:
    if not platform.system().lower().startswith("linux") or _vrserver_is_running():
        return None
    runtime = _find_steamvr_runtime(openvr)
    if runtime is None:
        return None
    command = _headless_vrserver_command(runtime)
    if not all(Path(path).is_file() for path in command[:2]):
        return None
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _stop_owned_vrserver(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5.0)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass


class ViveTracker:
    """Own one serial-selected Vive tracker stream through background OpenVR."""

    def __init__(self, serial: str, poll_rate_hz: float = 120.0) -> None:
        if not serial.strip():
            raise ValueError("Vive tracker serial must not be empty")
        try:
            import openvr
        except ImportError as exc:
            raise ImportError("openvr is required for Vive tracker input") from exc

        self.serial = serial.strip()
        self.poll_rate_hz = max(1.0, float(poll_rate_hz))
        self._openvr = openvr
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: ViveSample | None = None
        self._generation = 0
        self._last_error = ""
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"vive-{self.serial}",
        )
        self._thread.start()

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def get_latest(self) -> ViveSample | None:
        with self._lock:
            return self._latest

    def _invalidate(self, error: str = "") -> None:
        with self._lock:
            if self._latest is not None:
                self._generation += 1
            self._latest = None
            if error:
                self._last_error = error

    def _publish(self, matrix: Any) -> None:
        position, quaternion = _matrix34_to_components(matrix)
        position.setflags(write=False)
        quaternion.setflags(write=False)
        with self._lock:
            self._latest = ViveSample(
                position=position,
                quaternion=quaternion,
                timestamp=time.monotonic(),
                generation=self._generation,
            )
            self._last_error = ""

    def _tracker_index(self, vr_system: Any) -> int | None:
        for index in range(self._openvr.k_unMaxTrackedDeviceCount):
            if not vr_system.isTrackedDeviceConnected(index):
                continue
            if (
                vr_system.getTrackedDeviceClass(index)
                != self._openvr.TrackedDeviceClass_GenericTracker
            ):
                continue
            try:
                serial = vr_system.getStringTrackedDeviceProperty(
                    index, self._openvr.Prop_SerialNumber_String
                )
            except Exception:
                continue
            if str(serial) == self.serial:
                return index
        return None

    def _pose_is_usable(self, vr_system: Any, index: int, pose: Any) -> bool:
        if not pose.bDeviceIsConnected or not pose.bPoseIsValid:
            return False
        running_ok = getattr(self._openvr, "TrackingResult_Running_OK", None)
        if running_ok is not None and getattr(pose, "eTrackingResult", None) != running_ok:
            return False
        try:
            current_serial = vr_system.getStringTrackedDeviceProperty(
                index, self._openvr.Prop_SerialNumber_String
            )
        except Exception:
            return False
        return str(current_serial) == self.serial

    def _run(self) -> None:
        vr_system = None
        owned_vrserver = None
        try:
            owned_vrserver = _start_headless_vrserver(self._openvr)
            deadline = time.monotonic() + (15.0 if owned_vrserver is not None else 0.0)
            while not self._stop.is_set():
                try:
                    vr_system = self._openvr.init(
                        self._openvr.VRApplication_Background
                    )
                    break
                except Exception:
                    if owned_vrserver is None or time.monotonic() >= deadline:
                        raise
                    if owned_vrserver.poll() is not None:
                        raise RuntimeError(
                            f"headless vrserver exited with code {owned_vrserver.returncode}"
                        )
                    self._stop.wait(0.25)
            if vr_system is None:
                return

            tracker_index: int | None = None
            refresh_at = 0.0
            interval = 1.0 / self.poll_rate_hz
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    if tracker_index is None or started >= refresh_at:
                        tracker_index = self._tracker_index(vr_system)
                        refresh_at = started + 1.0
                    poses = vr_system.getDeviceToAbsoluteTrackingPose(
                        self._openvr.TrackingUniverseRawAndUncalibrated,
                        0,
                        self._openvr.k_unMaxTrackedDeviceCount,
                    )
                    if tracker_index is None or tracker_index >= len(poses):
                        self._invalidate(f"Vive tracker {self.serial} is not connected")
                    else:
                        pose = poses[tracker_index]
                        if self._pose_is_usable(vr_system, tracker_index, pose):
                            self._publish(pose.mDeviceToAbsoluteTracking)
                        else:
                            self._invalidate(
                                f"Vive tracker {self.serial} pose is not valid"
                            )
                            tracker_index = None
                            refresh_at = started
                except Exception as exc:
                    self._invalidate(f"Vive tracker read failed: {exc}")
                    tracker_index = None
                    refresh_at = started
                self._stop.wait(max(0.0, interval - (time.monotonic() - started)))
        except Exception as exc:
            self._invalidate(f"Vive tracker startup failed: {exc}")
        finally:
            if vr_system is not None:
                try:
                    self._openvr.shutdown()
                except Exception:
                    pass
            _stop_owned_vrserver(owned_vrserver)

    def wait_for_sample(self, timeout: float = 5.0) -> ViveSample | None:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            sample = self.get_latest()
            if sample is not None:
                return sample
            if not self._thread.is_alive():
                break
            self._stop.wait(0.01)
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        self._thread.join(timeout=8.0)
        self._closed = True

    def __enter__(self) -> "ViveTracker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
