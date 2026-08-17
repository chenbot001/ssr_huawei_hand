from __future__ import annotations

import argparse
import ctypes
import importlib.util
import subprocess
import sys
from collections.abc import Callable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Non-moving readiness check for UR5 + RYHand teleoperation"
    )
    parser.add_argument(
        "--vive-right",
        action="store_true",
        help="check the right Vive tracker",
    )
    parser.add_argument(
        "--manus-right",
        action="store_true",
        help="check the right MANUS glove",
    )
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="input wait timeout in seconds"
    )
    return parser


def _show(name: str, ok: bool, detail: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return ok


def _check_packages() -> bool:
    packages = {
        "numpy": "numpy",
        "yaml": "PyYAML",
        "scipy": "scipy",
        "rtde_control": "ur-rtde",
        "rtde_receive": "ur-rtde",
        "can": "python-can",
        "pybullet": "pybullet",
        "openvr": "openvr",
        "pynput": "pynput",
    }
    missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
    return _show(
        "Python dependencies",
        not missing,
        "all installed" if not missing else "missing " + ", ".join(sorted(set(missing))),
    )


def _check_assets() -> bool:
    from control.ryhand_ik import URDF_PATH
    from hardware.manus import MANUS_SDK_LIBRARY_PATH
    from hardware.ruiyan_driver import RYHAND_LIBRARY_PATH

    urdf_ok = _show("RYHand URDF", URDF_PATH.is_file(), str(URDF_PATH))
    library_ok = RYHAND_LIBRARY_PATH.is_file()
    detail = str(RYHAND_LIBRARY_PATH)
    if library_ok:
        try:
            ctypes.CDLL(str(RYHAND_LIBRARY_PATH))
        except OSError as exc:
            library_ok = False
            detail = str(exc)
    _show("RYHand vendor library", library_ok, detail)
    manus_sdk_ok = MANUS_SDK_LIBRARY_PATH.is_file()
    manus_detail = str(MANUS_SDK_LIBRARY_PATH)
    if manus_sdk_ok:
        try:
            ctypes.CDLL(str(MANUS_SDK_LIBRARY_PATH))
        except OSError as exc:
            manus_sdk_ok = False
            manus_detail = str(exc)
    _show("Manus Core 3.1.1 library", manus_sdk_ok, manus_detail)
    return urdf_ok and library_ok and manus_sdk_ok


def _check_can(interface: str) -> bool:
    try:
        result = subprocess.run(
            ["ip", "-details", "link", "show", interface],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _show("SocketCAN", False, str(exc))
    output = result.stdout
    is_up = result.returncode == 0 and (
        "state UP" in output or ("<" in output and "UP" in output.split(">", 1)[0])
    )
    return _show("SocketCAN", is_up, f"{interface} is {'UP' if is_up else 'not UP'}")


def _check_ur(ip: str) -> bool:
    try:
        import rtde_receive

        receiver = rtde_receive.RTDEReceiveInterface(ip)
        pose = receiver.getActualTCPPose()
        mode = receiver.getRobotMode()
        safety = receiver.getSafetyMode()
        receiver.disconnect()
        ok = len(pose) == 6 and mode == 7 and safety == 1
        return _show("UR RTDE", ok, f"mode={mode}, safety={safety}, ip={ip}")
    except Exception as exc:
        return _show("UR RTDE", False, str(exc))


def _check_vive(hardware: dict, use_right: bool, timeout: float) -> bool:
    try:
        from hardware.vive import ViveTracker

        side = "right" if use_right else "left"
        serial = str(hardware["vive_tracker"][f"{side}_serial"])
        tracker = ViveTracker(serial)
        try:
            sample = tracker.wait_for_sample(timeout)
        finally:
            tracker.close()
        return _show(
            "Vive",
            sample is not None,
            f"fresh {side} pose from {serial}"
            if sample
            else tracker.last_error or f"no pose from {serial}",
        )
    except Exception as exc:
        return _show("Vive", False, str(exc))


def _check_manus(hardware: dict, use_right: bool, timeout: float) -> bool:
    try:
        from hardware.manus import ManusReceiver

        glove = hardware["manus_glove"]
        receiver = ManusReceiver(
            str(glove["address"]), int(glove["left_id"]), int(glove["right_id"])
        )
        try:
            sample = receiver.wait_for_sample(use_right, timeout)
        finally:
            receiver.close()
        configured_id = int(glove["right_id"] if use_right else glove["left_id"])
        expected = f"ID {configured_id}" if configured_id else f"the selected {'right' if use_right else 'left'} glove"
        return _show(
            "Manus",
            sample is not None,
            f"sample from {expected}"
            if sample
            else receiver.last_error or f"no sample from {expected}",
        )
    except Exception as exc:
        return _show("Manus", False, str(exc))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        print("--timeout must be positive", file=sys.stderr)
        return 2

    from config import get_hardware_config

    hardware = get_hardware_config()
    checks: list[Callable[[], bool]] = [
        _check_packages,
        _check_assets,
        lambda: _check_can(str(hardware["ruiyan_hand"]["port"])),
        lambda: _check_ur(str(hardware["ur_arm"]["ip"])),
        lambda: _check_vive(hardware, args.vive_right, args.timeout),
        lambda: _check_manus(hardware, args.manus_right, args.timeout),
    ]
    passed = [check() for check in checks]
    print("SYSTEM READY" if all(passed) else "SYSTEM NOT READY")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
