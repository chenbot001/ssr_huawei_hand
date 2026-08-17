from http.client import HTTPConnection
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from console.app import ConsoleApplication, ConsoleServer
from console.runtime import (
    EventLog,
    ManusRuntime,
    TELEOP_MODES,
    TeleopRuntime,
    VirtualArm,
    VirtualHand,
    encode_rgba_png,
)
from console.security import request_is_local
from console.status import StatusScanner
from hardware.manus import ManusSample
from hardware.vive import ViveSample


WEB_ROOT = Path(__file__).parents[1] / "src" / "console" / "web"


def test_console_page_has_four_tabs_and_four_panels_per_tab():
    page = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    for tab in ("system", "manus", "ryhand", "teleop"):
        section = page.split(f'id="tab-{tab}"', 1)[1].split("</main>", 1)[0]
        assert 'class="left"' in section
        assert 'class="right"' in section
        assert "visual-panel" in section
        assert "log-section" in section
        assert "controls-panel" in section
        assert "status-section" in section


def test_manus_tab_retains_egotac_monitor_and_official_calibration_controls():
    page = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    script = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    section = page.split('id="tab-manus"', 1)[1].split("</main>", 1)[0]
    assert 'id="manus-canvas"' in section
    assert 'id="manus-left-battery"' in section
    assert 'id="manus-right-battery"' in section
    assert 'id="manus-view-yaw"' in section
    assert 'id="manus-settings-apply"' in section
    assert 'id="manus-calibration-start"' in section
    assert 'id="manus-calibration-next"' in section
    assert 'id="manus-calibration-save"' in section
    assert 'data-tab="calibrate"' not in page
    assert 'data-tab="ryhand"' in page
    assert 'raw === null || raw === undefined || raw === ""' in script
    assert 'api("/api/manus/visual")' in script
    assert "setTimeout(pollManusVisual, 50)" in script


def test_teleop_tab_displays_live_controller_owned_motion_delta():
    page = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    script = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    section = page.split('id="tab-teleop"', 1)[1].split("</main>", 1)[0]
    assert 'id="teleop-canvas"' not in section
    assert 'id="teleop-view-fit"' not in section
    assert 'id="teleop-view-reset"' not in section
    assert 'id="teleop-view-yaw"' not in section
    assert 'id="teleop-view-pitch"' not in section
    assert 'id="teleop-show-axes"' not in section
    for axis in ("x", "y", "z", "rx", "ry", "rz"):
        assert f'id="teleop-delta-{axis}"' in section
    assert 'id="teleop-mode"' in section
    assert 'id="teleop-config-fields"' in section
    assert 'id="teleop-config-save"' in section
    assert 'id="teleop-vive-side"' in section
    assert 'id="teleop-manus-side"' in section
    assert 'id="teleop-side"' not in section
    assert "Mount Correction" not in section
    assert "Release captures new tracker/TCP references" in section
    assert ">Release clutch</button>" in section
    assert 'id="teleop-modes"' not in section
    assert "Start a teleoperation mode to inspect live inputs" not in section
    assert 'action("/api/teleop/config"' in script
    assert "vive_side" in script
    assert "manus_side" in script
    assert "mount_correction" not in script
    assert "teleop.motion_delta?.translation_m" in script
    assert "teleop.motion_delta?.rotation_rad" in script
    assert "Number(raw) * scale" in script
    assert "preview_active" in script
    assert 'teleop.arm_tracking ? "TRACKING" : "CLUTCH ENGAGED"' in script
    assert "function drawTeleop" not in script
    assert "pointerdown" not in script


def test_console_uses_egotac_shell_geometry_with_equal_right_panels():
    styles = (WEB_ROOT / "styles.css").read_text(encoding="utf-8")
    assert "--header-h: 48px" in styles
    assert "--rail: 380px" in styles
    assert "--log-h: 150px" in styles
    assert "grid-template-columns: minmax(480px, 1fr) var(--rail)" in styles
    assert "grid-template-rows: minmax(0, 1fr) minmax(0, 1fr)" in styles


def test_system_tab_uses_hardware_inventory_and_explicit_connection_controls():
    page = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    assert 'id="hardware-inventory"' in page
    assert 'id="ur-ip"' in page
    assert 'id="ur-connect"' in page
    assert 'id="ryhand-init"' in page
    assert "system-map" not in page


def test_ur_status_requires_a_working_rtde_receive_session(monkeypatch):
    class Receiver:
        disconnected = False

        def __init__(self, ip):
            self.ip = ip

        def isConnected(self):
            return True

        def getActualTCPPose(self):
            return [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]

        def getRobotMode(self):
            return 7

        def getSafetyMode(self):
            return 1

        def disconnect(self):
            type(self).disconnected = True

    monkeypatch.setattr(
        "console.status.socket.create_connection",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setitem(
        sys.modules,
        "rtde_receive",
        SimpleNamespace(RTDEReceiveInterface=Receiver),
    )

    result = StatusScanner._check_ur("192.168.0.2")

    assert result["state"] == "connected"
    assert result["metadata"]["Robot mode"] == "Running"
    assert result["metadata"]["Safety"] == "Normal"
    assert Receiver.disconnected


def test_ur_status_stays_disconnected_when_rtde_session_fails(monkeypatch):
    class Receiver:
        def __init__(self, ip):
            raise RuntimeError("handshake failed")

    monkeypatch.setattr(
        "console.status.socket.create_connection",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setitem(
        sys.modules,
        "rtde_receive",
        SimpleNamespace(RTDEReceiveInterface=Receiver),
    )

    result = StatusScanner._check_ur("192.168.0.2")

    assert result["state"] == "offline"
    assert result["label"] == "Disconnected"
    assert "handshake failed" in result["detail"]


def test_ur_connect_keeps_a_monitored_session_until_release(monkeypatch):
    class Receiver:
        disconnected = False

        def __init__(self, ip):
            pass

        def isConnected(self):
            return True

        def getActualTCPPose(self):
            return [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]

        def getRobotMode(self):
            return 7

        def getSafetyMode(self):
            return 1

        def disconnect(self):
            type(self).disconnected = True

    monkeypatch.setattr(
        "console.status.socket.create_connection",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setitem(
        sys.modules,
        "rtde_receive",
        SimpleNamespace(RTDEReceiveInterface=Receiver),
    )
    scanner = StatusScanner()

    assert scanner.connect_ur("192.168.0.2")["state"] == "connected"
    assert scanner.snapshot()["ur"]["state"] == "connected"
    assert not Receiver.disconnected

    scanner.release_ur()
    assert Receiver.disconnected
    assert scanner.snapshot()["ur"]["state"] == "unknown"


def test_ryhand_init_button_executes_the_retained_script(monkeypatch):
    app = ConsoleApplication()
    hardware = {"ruiyan_hand": {"port": "can7"}}
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "can7 is UP", "")

    monkeypatch.setattr("console.app.get_hardware_config", lambda: hardware)
    monkeypatch.setattr("console.app.subprocess.run", run)
    monkeypatch.setattr(
        app.system,
        "refresh_ryhand",
        lambda: {"state": "ready", "label": "CAN ready", "detail": "can7", "metadata": {}},
    )
    try:
        result = app.initialize_ryhand()
    finally:
        app.close()

    assert result["state"] == "ready"
    assert calls[0][:3] == ["sudo", "-n", "bash"]
    assert calls[0][-1] == "can7"


def test_manus_console_runtime_exposes_live_frames_and_calibration(monkeypatch):
    class Receiver:
        def __init__(self, address, left_id, right_id):
            self.closed = False
            self.calibration = {"active": False, "inProgress": False}

        def status(self, include_frames=False):
            return {
                "state": "running",
                "error": None,
                "sdkVersion": "3.1.1",
                "settings": {"handMotion": 4},
                "calibration": dict(self.calibration),
                "hands": {
                    "left": {"connected": True, "gloveId": 17, "frame": {"bones": []}},
                    "right": {"connected": False, "gloveId": None},
                },
            }

        def command(self, action, params=None):
            if action == "calibration_start":
                self.calibration = {
                    "active": True,
                    "inProgress": False,
                    "gloveId": params["gloveId"],
                    "stepCount": 3,
                    "completedStepIndex": -1,
                }
            return dict(self.calibration) if action.startswith("calibration_") else {"handMotion": 4}

        def close(self):
            self.closed = True

    monkeypatch.setattr("console.runtime.ManusReceiver", Receiver)
    monkeypatch.setattr(
        "console.runtime.get_hardware_config",
        lambda: {"manus_glove": {"address": "udp://127.0.0.1:9001", "left_id": 0, "right_id": 0}},
    )
    runtime = ManusRuntime(EventLog())

    runtime.start()
    assert runtime.snapshot()["hands"]["left"]["connected"]
    runtime.route_to("manus")
    assert runtime.calibration_start("left")["gloveId"] == 17
    receiver = runtime._receiver
    runtime.stop()

    assert receiver.closed
    assert not runtime.active()


def test_teleop_runtime_snapshot_exposes_independent_inputs():
    class Vive:
        serial = "LHR-TEST"

        @staticmethod
        def get_latest():
            return ViveSample(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                10.0,
                0,
            )

    class Manus:
        sample = ManusSample(17, "left", np.zeros((10, 3)), np.zeros(3), 10.0)

        @classmethod
        def get_latest(cls, use_right=False):
            return cls.sample

        @staticmethod
        def status(include_frames=False):
            return {
                "hands": {
                    "left": {
                        "connected": True,
                        "gloveId": 17,
                        "frame": {"bones": [{"nodeId": 1, "parentId": 1, "rawPos": [0, 0, 0]}]},
                    }
                }
            }

    log = EventLog()
    runtime = TeleopRuntime(log, ManusRuntime(log))
    runtime._vive = Vive()
    runtime._manus = Manus()

    snapshot = runtime.snapshot()

    assert snapshot["vive_pose"]["serial"] == "LHR-TEST"
    assert snapshot["vive_pose"]["position"] == [1.0, 2.0, 3.0]
    assert snapshot["vive_pose"]["quaternion"] == [0.0, 0.0, 0.0, 1.0]
    assert set(snapshot["vive_pose"]) == {"serial", "position", "quaternion"}
    assert snapshot["vive_side"] == "left"
    assert snapshot["manus_side"] == "left"
    assert snapshot["manus_hand"]["glove_id"] == 17
    assert snapshot["manus_hand"]["frame"]["bones"][0]["nodeId"] == 1
    assert snapshot["clutch_engaged"]
    assert not snapshot["arm_tracking"]
    assert snapshot["motion_delta"] == {
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_rad": [0.0, 0.0, 0.0],
    }


def test_teleop_preview_opens_both_live_inputs_without_starting_a_mode(monkeypatch):
    vive_instances = []
    manus_instances = []

    class Vive:
        def __init__(self, serial):
            self.serial = serial
            self.closed = False
            vive_instances.append(self)

        def get_latest(self):
            return ViveSample(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                time.monotonic(),
                0,
            )

        def close(self):
            self.closed = True

    class Manus:
        def __init__(self):
            self.closed = False
            manus_instances.append(self)

        def get_latest(self, use_right=False):
            return ManusSample(
                17,
                "right" if use_right else "left",
                np.zeros((10, 3)),
                np.zeros(3),
                time.monotonic(),
            )

        def status(self, include_frames=False):
            frame = {"bones": [{"nodeId": 1, "parentId": 1, "rawPos": [0, 0, 0]}]}
            return {
                "hands": {
                    "left": {"connected": True, "gloveId": 17, "frame": frame},
                    "right": {"connected": False, "gloveId": None},
                }
            }

        def close(self):
            self.closed = True

    class SharedManus:
        def open_input(self, destination):
            assert destination == "teleop"
            return Manus()

    monkeypatch.setattr("console.runtime.ViveTracker", Vive)
    monkeypatch.setattr(
        "console.runtime.get_hardware_config",
        lambda: {"vive_tracker": {"left_serial": "LHR-LEFT", "right_serial": "LHR-RIGHT"}},
    )
    runtime = TeleopRuntime(EventLog(), SharedManus())

    runtime.start_preview("right", "left")
    with runtime._inputs_ready:
        assert runtime._inputs_ready.wait_for(
            lambda: runtime._vive is not None and runtime._manus is not None,
            timeout=1.0,
        )
    snapshot = runtime.snapshot()

    assert not snapshot["active"]
    assert snapshot["preview_active"]
    assert snapshot["vive_side"] == "right"
    assert snapshot["manus_side"] == "left"
    assert snapshot["vive_pose"]["serial"] == "LHR-RIGHT"
    assert snapshot["manus_hand"]["side"] == "left"
    assert snapshot["manus_hand"]["frame"]["bones"][0]["nodeId"] == 1
    runtime.start_preview("right", "left")
    assert len(vive_instances) == 1
    assert len(manus_instances) == 1

    runtime.stop_preview()
    assert vive_instances[0].closed
    assert manus_instances[0].closed


def test_shared_manus_stream_routes_one_receiver_to_the_active_tab(monkeypatch):
    sample = ManusSample(17, "left", np.zeros((10, 3)), np.zeros(3), 10.0)
    receivers = []

    class Receiver:
        last_error = None

        def __init__(self, address, left_id, right_id):
            self.closed = False
            receivers.append(self)

        def get_latest(self, use_right=False):
            return sample

        def status(self, include_frames=False):
            return {
                "state": "running",
                "error": None,
                "sdkVersion": "3.1.1",
                "settings": {},
                "calibration": {},
                "hands": {"left": {"connected": True}, "right": {"connected": False}},
            }

        def close(self):
            self.closed = True

    monkeypatch.setattr("console.runtime.ManusReceiver", Receiver)
    monkeypatch.setattr(
        "console.runtime.get_hardware_config",
        lambda: {"manus_glove": {"address": "udp://127.0.0.1:9001", "left_id": 0, "right_id": 0}},
    )
    runtime = ManusRuntime(EventLog())
    teleop_input = runtime.open_input("teleop")
    ryhand_input = runtime.open_input("ryhand")

    assert len(receivers) == 1
    runtime.route_to("teleop")
    assert teleop_input.get_latest() is sample
    assert ryhand_input.get_latest() is None
    runtime.route_to("ryhand")
    assert teleop_input.get_latest() is None
    assert ryhand_input.get_latest() is sample

    runtime.route_to("manus")
    with pytest.raises(RuntimeError, match="MANUS controls belong"):
        runtime.route_to("teleop")
        runtime.settings()
    runtime.stop()
    assert receivers[0].closed
    assert teleop_input.get_latest() is None
    runtime.start()
    assert len(receivers) == 2
    assert teleop_input.get_latest() is sample
    teleop_input.close()
    ryhand_input.close()
    runtime.stop()


def test_console_runtimes_share_the_same_manus_owner():
    app = ConsoleApplication()
    try:
        assert app.teleop._manus_runtime is app.manus
        assert app.calibration._manus_runtime is app.manus
    finally:
        app.close()


def test_teleop_modes_cover_full_partial_and_virtual_control():
    assert set(TELEOP_MODES) == {"full", "arm", "hand", "simulation"}
    assert TELEOP_MODES["full"] == {
        "label": "Full system",
        "arm": True,
        "hand": True,
        "virtual": False,
    }
    assert TELEOP_MODES["arm"]["arm"] and not TELEOP_MODES["arm"]["hand"]
    assert TELEOP_MODES["hand"]["hand"] and not TELEOP_MODES["hand"]["arm"]
    assert TELEOP_MODES["simulation"]["virtual"]


def test_virtual_actuators_hold_computed_state():
    arm = VirtualArm()
    hand = VirtualHand()
    pose = np.arange(6, dtype=float) / 10
    angles = np.arange(15, dtype=float) / 20
    arm.servo_l(pose)
    hand.set_angles(angles, 500)
    np.testing.assert_allclose(arm.get_tcp_pose(), pose)
    np.testing.assert_allclose(hand.angles, angles)


def test_png_encoder_produces_rgba_png():
    payload = encode_rgba_png(np.zeros((3, 4, 4), dtype=np.uint8))
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert payload[12:16] == b"IHDR"
    assert int.from_bytes(payload[16:20], "big") == 4
    assert int.from_bytes(payload[20:24], "big") == 3


def test_local_request_validation_rejects_remote_and_cross_origin_hosts():
    assert request_is_local("127.0.0.1:8768", None, 8768)
    assert request_is_local("localhost:8768", "http://localhost:8768", 8768)
    assert not request_is_local("192.168.1.2:8768", None, 8768)
    assert not request_is_local("127.0.0.1:8768", "http://example.com", 8768)


def test_console_serves_state_with_security_headers():
    app = ConsoleApplication()
    server = ConsoleServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request("GET", "/api/state")
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert response.getheader("X-Frame-Options") == "DENY"
        assert set(payload) >= {"system", "manus", "calibration", "teleop", "teleop_modes"}
        assert "ur_ip" in payload

        connection.request(
            "POST",
            "/api/console/tab",
            body=json.dumps({"tab": "manus"}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        routed = json.loads(response.read())["state"]
        assert response.status == 200
        assert routed["active_tab"] == "manus"
        assert routed["manus"]["destination"] == "manus"

        connection.request("GET", "/api/teleop/visual")
        response = connection.getresponse()
        visual = json.loads(response.read())
        assert response.status == 200
        assert set(visual) >= {
            "vive_pose",
            "manus_hand",
            "target_pose",
            "motion_delta",
            "hand_angles",
        }

        connection.request("GET", "/api/manus/visual")
        response = connection.getresponse()
        manus_visual = json.loads(response.read())
        assert response.status == 200
        assert set(manus_visual) >= {"state", "hands", "calibration", "settings"}

        connection.request(
            "POST",
            "/api/teleop/start",
            body=json.dumps({"mode": "unknown"}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 409
        assert "Unknown teleop mode" in json.loads(response.read())["error"]

        connection.request(
            "POST",
            "/api/system/ur/connect",
            body=json.dumps({"ip": "not-an-ip"}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 409
        assert "valid UR IPv4" in json.loads(response.read())["error"]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        app.close()
        thread.join(timeout=2)
