from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import IPv4Address, ip_address
import json
from pathlib import Path
import subprocess
import threading
import webbrowser
from typing import Any

from config import (
    PROJECT_ROOT,
    get_console_teleop_config,
    get_hardware_config,
    save_console_teleop_config,
)
from console.runtime import (
    CalibrationRuntime,
    CameraRuntime,
    EventLog,
    ManusRuntime,
    TELEOP_MODES,
    TeleopRuntime,
)
from console.security import MAX_REQUEST_BYTES, SECURITY_HEADERS, is_loopback_host, request_is_local
from console.status import StatusScanner


WEB_ROOT = Path(__file__).resolve().parent / "web"


class ConsoleApplication:
    def __init__(self) -> None:
        self.log = EventLog()
        self.system = StatusScanner()
        self.manus = ManusRuntime(self.log)
        self.teleop = TeleopRuntime(self.log, self.manus)
        self.calibration = CalibrationRuntime(self.log, self.manus)
        self.camera = CameraRuntime(self.log)
        self._active_tab = "system"
        self.log.add("system", "Huawei teleoperation console ready")

    def set_active_tab(
        self, tab: str, vive_side: str = "left", manus_side: str = "left"
    ) -> None:
        self.manus.route_to(tab)
        self._active_tab = tab
        if tab == "teleop":
            self.teleop.start_preview(vive_side, manus_side)
        else:
            self.teleop.stop_preview()

    def select_teleop_inputs(self, vive_side: str, manus_side: str) -> None:
        if self._active_tab != "teleop":
            raise RuntimeError("Open the Teleop tab before changing preview side")
        self.teleop.start_preview(vive_side, manus_side)

    def save_teleop_config(self, values: dict[str, Any]) -> dict[str, Any]:
        if self.teleop.active() or self.calibration.active():
            raise RuntimeError("Stop teleoperation and RYHand calibration before saving configuration")
        saved = save_console_teleop_config(values)
        self.log.add(
            "teleop",
            "Saved teleoperation configuration to hardware_config.yaml and teleop_config.yaml",
        )
        return saved

    def stop_teleop(self) -> None:
        self.teleop.stop()
        if self._active_tab != "teleop":
            self.teleop.stop_preview()

    def connect_ur(self, value: str) -> dict[str, Any]:
        if self.teleop.active():
            raise RuntimeError("Stop teleoperation before changing the UR connection")
        try:
            parsed = ip_address(value.strip())
        except ValueError as exc:
            raise ValueError("Enter a valid UR IPv4 address") from exc
        if not isinstance(parsed, IPv4Address):
            raise ValueError("Enter a valid UR IPv4 address")
        ip = str(parsed)
        result = self.system.connect_ur(ip)
        if result["state"] == "connected":
            get_hardware_config()["ur_arm"]["ip"] = ip
            self.log.add("system", f"UR RTDE connection verified at {ip}")
        else:
            self.log.add("system", result["detail"], "error")
        return result

    def initialize_ryhand(self) -> dict[str, Any]:
        if self.teleop.active() or self.calibration.active():
            raise RuntimeError("Stop teleoperation and calibration before initializing RYHand CAN")
        interface = str(get_hardware_config()["ruiyan_hand"]["port"])
        script = PROJECT_ROOT / "scripts" / "ryhand_init.sh"
        if not script.is_file():
            raise FileNotFoundError(f"RYHand initialization script is missing: {script}")
        self.log.add("system", f"Running {script.name} for {interface}")
        try:
            process = subprocess.run(
                ["sudo", "-n", "bash", str(script), interface],
                capture_output=True,
                text=True,
                timeout=20.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.system.refresh_ryhand()
            raise RuntimeError(f"RYHand initialization failed: {exc}") from exc
        output = "\n".join(
            part.strip() for part in (process.stdout, process.stderr) if part.strip()
        )
        for line in output.splitlines():
            self.log.add("ryhand", line, "info" if process.returncode == 0 else "error")
        status = self.system.refresh_ryhand()
        if process.returncode != 0:
            if "password" in output.lower() or "sudo" in output.lower():
                raise RuntimeError(
                    "sudo authorization is unavailable. Run 'sudo -v' in a terminal, then retry RYHand Init."
                )
            raise RuntimeError(output or f"ryhand_init.sh exited with code {process.returncode}")
        if status["state"] != "ready":
            raise RuntimeError(status["detail"])
        self.log.add("system", f"RYHand CAN initialized on {interface}")
        return status

    def start_teleop(self, mode: str, vive_side: str, manus_side: str) -> None:
        if mode not in TELEOP_MODES:
            raise ValueError(f"Unknown teleop mode: {mode}")
        if self._active_tab != "teleop":
            raise RuntimeError("Open the Teleop tab before starting teleoperation")
        if self.calibration.active():
            raise RuntimeError("Stop RYHand calibration before starting teleoperation")
        if mode in TELEOP_MODES:
            spec = TELEOP_MODES[mode]
            if spec["arm"] and not spec["virtual"]:
                self.system.release_ur("RTDE status session released to teleoperation")
        self.teleop.start(mode, vive_side, manus_side)

    def start_calibration(self, use_right: bool, live_output: bool) -> None:
        if self._active_tab != "ryhand":
            raise RuntimeError("Open the RYHand tab before starting calibration")
        if self.teleop.active():
            raise RuntimeError("Stop teleoperation before starting RYHand calibration")
        self.calibration.start(use_right, live_output)

    def start_manus(self) -> None:
        if self._active_tab != "manus":
            raise RuntimeError("Open the Manus tab before starting the MANUS stream")
        self.manus.start()

    def snapshot(self) -> dict[str, Any]:
        system = dict(self.system.snapshot())
        teleop = self.teleop.snapshot()
        manus = self.manus.snapshot()
        calibration = self.calibration.snapshot()
        camera = self.camera.snapshot()
        configured_cameras = get_hardware_config().get("rgb_cameras", [])

        mode = teleop.get("mode")
        if teleop["active"] and mode in TELEOP_MODES:
            spec = TELEOP_MODES[mode]
            if spec["arm"]:
                system["vive"] = {
                    **system["vive"],
                    "state": "connected" if teleop["vive_age"] is not None else "waiting",
                    "label": "Live input" if teleop["vive_age"] is not None else "Waiting",
                    "detail": f"{teleop['vive_side']} tracker owned by {teleop['mode_label']}",
                }
                system["ur"] = {
                    **system["ur"],
                    "state": (
                        "simulated"
                        if spec["virtual"]
                        else "connected" if teleop["arm_connected"] else "waiting"
                    ),
                    "label": (
                        "Virtual output"
                        if spec["virtual"]
                        else "Control session open" if teleop["arm_connected"] else "Connecting"
                    ),
                    "detail": (
                        teleop["mode_label"]
                        if spec["virtual"] or teleop["arm_connected"]
                        else "No working UR RTDE control session yet"
                    ),
                }
            if spec["hand"]:
                system["manus"] = {
                    **system["manus"],
                    "state": "connected" if teleop["manus_age"] is not None else "waiting",
                    "label": "Live input" if teleop["manus_age"] is not None else "Waiting",
                    "detail": f"{teleop['manus_side']} glove owned by {teleop['mode_label']}",
                }
                system["ryhand"] = {
                    **system["ryhand"],
                    "state": (
                        "simulated"
                        if spec["virtual"]
                        else "connected" if teleop["hand_connected"] else "waiting"
                    ),
                    "label": (
                        "Virtual output"
                        if spec["virtual"]
                        else "CAN session open" if teleop["hand_connected"] else "Connecting"
                    ),
                    "detail": (
                        teleop["mode_label"]
                        if spec["virtual"] or teleop["hand_connected"]
                        else "No working RYHand CAN session yet"
                    ),
                }
        elif calibration["active"]:
            system["manus"] = {
                **system["manus"],
                "state": "connected",
                "label": "Calibration input",
                "detail": f"{calibration['side']} glove",
            }
            if calibration["live_output"]:
                system["ryhand"] = {
                    **system["ryhand"],
                    "state": "connected",
                    "label": "Calibration output",
                    "detail": "RYHand commands enabled",
                }
        if manus["active"]:
            connected_hands = [
                hand for hand, item in manus["hands"].items() if item.get("connected")
            ]
            system["manus"] = {
                **system["manus"],
                "state": "connected" if connected_hands else "waiting",
                "label": "Live input" if connected_hands else "Bridge running",
                "detail": (
                    f"Live {' + '.join(connected_hands)} glove stream · routed to {manus['destination']}"
                    if connected_hands
                    else manus["detail"]
                ),
                "metadata": {
                    **system["manus"].get("metadata", {}),
                    "SDK": manus.get("sdkVersion") or "3.1.1",
                    "Gloves": ", ".join(connected_hands) or "Waiting",
                    "Route": manus["destination"],
                },
            }
        if camera["active"] and camera["serial"]:
            camera_cards = [dict(item) for item in system.get("cameras", [])]
            if not camera_cards:
                camera_cards = [
                    {
                        "name": str(item["name"]),
                        "serial": str(item["serial"]),
                        "state": "unknown",
                        "label": "Not checked",
                        "detail": str(item["serial"]),
                        "metadata": {"Serial": str(item["serial"]), "Stream": "RGB"},
                    }
                    for item in configured_cameras
                ]
            for item in camera_cards:
                if item["serial"] == camera["serial"]:
                    item.update(
                        state="connected",
                        label="Preview active",
                        detail=camera["detail"],
                    )
            system["cameras"] = camera_cards

        return {
            "system": system,
            "teleop": teleop,
            "manus": manus,
            "calibration": calibration,
            "camera": camera,
            "cameras": configured_cameras,
            "ur_ip": str(get_hardware_config()["ur_arm"]["ip"]),
            "active_tab": self._active_tab,
            "teleop_config": get_console_teleop_config(),
            "teleop_modes": [
                {"id": key, **value} for key, value in TELEOP_MODES.items()
            ],
            "logs": self.log.snapshot(),
        }

    def close(self) -> None:
        self.camera.stop()
        self.calibration.stop()
        self.teleop.stop()
        self.teleop.stop_preview()
        self.manus.stop()
        self.system.close()


class ConsoleServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: ConsoleApplication) -> None:
        self.app = app
        super().__init__(address, ConsoleHandler)


class ConsoleHandler(BaseHTTPRequestHandler):
    server: ConsoleServer

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def _authorized(self) -> bool:
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if request_is_local(host, origin, self.server.server_port):
            return True
        self._json({"error": "Local same-origin requests only"}, HTTPStatus.FORBIDDEN)
        return False

    def _headers(self, content_type: str, length: int, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        for key, value in SECURITY_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()

    def _bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self._headers(content_type, len(payload), status)
        self.wfile.write(payload)

    def _json(self, value: Any, status: int = 200) -> None:
        payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._bytes(payload, "application/json; charset=utf-8", status)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("Request body is too large")
        raw = self.rfile.read(length)
        value = json.loads(raw or b"{}")
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_GET(self) -> None:
        if not self._authorized():
            return
        path = self.path.split("?", 1)[0]
        static = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        if path in static:
            filename, content_type = static[path]
            self._bytes((WEB_ROOT / filename).read_bytes(), content_type)
            return
        if path == "/api/state":
            self._json(self.server.app.snapshot())
            return
        if path == "/api/teleop/visual":
            self._json(self.server.app.teleop.snapshot())
            return
        if path == "/api/manus/visual":
            self._json(self.server.app.manus.snapshot())
            return
        if path == "/api/system/scan":
            self.server.app.system.scan()
            self.server.app.log.add("system", "Read-only readiness scan completed")
            self._json(self.server.app.snapshot())
            return
        if path == "/api/calibration/frame":
            frame = self.server.app.calibration.frame()
            if frame is None:
                self._json({"error": "No calibration frame"}, HTTPStatus.NOT_FOUND)
            else:
                self._bytes(frame, "image/png")
            return
        if path == "/api/camera/frame":
            frame = self.server.app.camera.frame()
            if frame is None:
                self._json({"error": "No camera frame"}, HTTPStatus.NOT_FOUND)
            else:
                self._bytes(frame, "image/png")
            return
        self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self._authorized():
            return
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_json()
            app = self.server.app
            if path == "/api/system/ur/connect":
                app.connect_ur(str(body.get("ip", "")))
            elif path == "/api/console/tab":
                app.set_active_tab(
                    str(body.get("tab", "")),
                    str(body.get("vive_side", "left")),
                    str(body.get("manus_side", "left")),
                )
            elif path == "/api/system/ryhand/init":
                app.initialize_ryhand()
            elif path == "/api/teleop/start":
                app.start_teleop(
                    str(body.get("mode", "")),
                    str(body.get("vive_side", "left")),
                    str(body.get("manus_side", "left")),
                )
            elif path == "/api/teleop/toggle":
                app.teleop.toggle()
            elif path == "/api/teleop/stop":
                app.stop_teleop()
            elif path == "/api/teleop/preview":
                app.select_teleop_inputs(
                    str(body.get("vive_side", "left")),
                    str(body.get("manus_side", "left")),
                )
            elif path == "/api/teleop/config":
                app.save_teleop_config(body)
            elif path == "/api/manus/start":
                app.start_manus()
            elif path == "/api/manus/stop":
                app.manus.stop()
            elif path == "/api/manus/settings":
                app.manus.settings()
            elif path == "/api/manus/settings/apply":
                app.manus.apply_settings(body)
            elif path == "/api/manus/calibration/start":
                app.manus.calibration_start(str(body.get("hand", "")))
            elif path == "/api/manus/calibration/step":
                app.manus.calibration_step()
            elif path == "/api/manus/calibration/status":
                app.manus.calibration_status()
            elif path == "/api/manus/calibration/finish":
                app.manus.calibration_finish()
            elif path == "/api/manus/calibration/cancel":
                app.manus.calibration_cancel()
            elif path == "/api/calibration/start":
                app.start_calibration(
                    bool(body.get("use_right", False)),
                    bool(body.get("live_output", False)),
                )
            elif path == "/api/calibration/update":
                app.calibration.update(body)
            elif path == "/api/calibration/save":
                app.calibration.save()
            elif path == "/api/calibration/stop":
                app.calibration.stop()
            elif path == "/api/camera/start":
                app.camera.start(str(body.get("serial", "")))
            elif path == "/api/camera/stop":
                app.camera.stop()
            else:
                self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            self._json({"ok": True, "state": app.snapshot()})
        except (ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except Exception as exc:
            self.server.app.log.add("system", f"Request failed: {exc}", "error")
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local visual console for Huawei UR5 + RYHand teleoperation"
    )
    parser.add_argument("--host", default="127.0.0.1", help="loopback bind host")
    parser.add_argument("--port", type=int, default=8768, help="local HTTP port")
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="do not open the console in the default browser",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not is_loopback_host(args.host):
        raise SystemExit("ssr-console only binds to a loopback host")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")

    app = ConsoleApplication()
    server = ConsoleServer((args.host, args.port), app)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Huawei teleoperation console: {url}")
    print("Live modes require Start; UR begins clutched while RYHand follows MANUS.")
    if not args.no_open_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping console...")
    finally:
        server.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
