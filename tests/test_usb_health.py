#!/usr/bin/env python3
"""
USB Health Diagnostic — verify RealSense cameras enumerate on USB 3.0 and
no stale processes hold video device nodes.

Usage:
    python tests/test_usb_health.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from ssr.config import get_hardware_config

# ============================================================================
# Helpers
# ============================================================================

def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout


def _parse_lsusb_tree() -> list[dict]:
    """
    Parse ``lsusb -t`` into a list of dicts with keys:
      bus, port, dev, driver, speed_m (int, in Mbps)
    """
    raw = _run(["lsusb", "-t"])
    entries: list[dict] = []
    current_bus: int | None = None

    for line in raw.splitlines():
        bus_m = re.match(r"/:\s+Bus\s+(\d+)\.Port\s+\d+:.*?(\d+)M", line)
        if bus_m:
            current_bus = int(bus_m.group(1))
            continue

        port_m = re.search(
            r"Port\s+(\d+):\s+Dev\s+(\d+).*?Driver=(\S+),\s*(\d+)M", line
        )
        if port_m and current_bus is not None:
            entries.append({
                "bus": current_bus,
                "port": int(port_m.group(1)),
                "dev": int(port_m.group(2)),
                "driver": port_m.group(3),
                "speed_m": int(port_m.group(4)),
            })
    return entries


def _find_realsense_usb_devices() -> list[dict]:
    """Return lsusb entries whose driver is ``uvcvideo``."""
    tree = _parse_lsusb_tree()
    return [e for e in tree if e["driver"] == "uvcvideo"]


def _get_realsense_lsusb_lines() -> list[str]:
    """Return raw ``lsusb`` lines that match Intel RealSense."""
    raw = _run(["lsusb"])
    return [l for l in raw.splitlines() if "RealSense" in l]


def _check_stale_processes(dev_path: str) -> list[str]:
    """Return list of PIDs holding *dev_path* open, if any."""
    r = subprocess.run(
        ["fuser", dev_path], capture_output=True, text=True
    )
    pids = r.stdout.strip().split()
    return [p.strip() for p in pids if p.strip()]


# ============================================================================
# Tests
# ============================================================================

def test_realsense_usb3() -> tuple[bool, list[str]]:
    """Verify every RealSense camera is connected at USB 3.0 (≥5000 Mbps)."""
    msgs: list[str] = []
    uvc_entries = _find_realsense_usb_devices()

    if not uvc_entries:
        msgs.append("[WARN] No uvcvideo devices found in lsusb -t output")
        return False, msgs

    bus_dev_set: set[tuple[int, int]] = set()
    for e in uvc_entries:
        bus_dev_set.add((e["bus"], e["dev"]))

    passed = True
    for bus, dev in sorted(bus_dev_set):
        speeds = [
            e["speed_m"]
            for e in uvc_entries
            if e["bus"] == bus and e["dev"] == dev
        ]
        max_speed = max(speeds)
        if max_speed < 5000:
            msgs.append(
                f"[FAIL] Bus {bus:03d} Dev {dev:03d}: {max_speed}M "
                f"(USB 2.0) — RealSense requires USB 3.0 (5000M)"
            )
            passed = False
        else:
            msgs.append(
                f"[PASS] Bus {bus:03d} Dev {dev:03d}: {max_speed}M (USB 3.0)"
            )
    return passed, msgs


def test_no_stale_processes() -> tuple[bool, list[str]]:
    """Make sure no zombie process is holding /dev/video* nodes used by configured cameras."""
    msgs: list[str] = []
    hw = get_hardware_config()
    rs_configs = hw.get("cameras", {}).get("realsense", [])

    from ssr.utils.camera_utils import list_v4l2_devices

    v4l2_map = list_v4l2_devices()

    dev_nodes: list[str] = []
    for cfg in rs_configs:
        cam_id = cfg.get("id", "")
        if cam_id in v4l2_map:
            dev_nodes.extend(v4l2_map[cam_id])

    if not dev_nodes:
        msgs.append("[INFO] No video device nodes matched config — skipping stale-process check")
        return True, msgs

    passed = True
    for node in dev_nodes:
        pids = _check_stale_processes(node)
        if pids:
            msgs.append(f"[FAIL] {node} held by PID(s): {', '.join(pids)}")
            passed = False
        else:
            msgs.append(f"[PASS] {node} — no stale processes")

    return passed, msgs


def test_realsense_detected() -> tuple[bool, list[str]]:
    """Confirm at least the expected number of RealSense cameras appear in lsusb."""
    msgs: list[str] = []
    hw = get_hardware_config()
    expected = len(hw.get("cameras", {}).get("realsense", []))
    lines = _get_realsense_lsusb_lines()

    if len(lines) >= expected:
        msgs.append(
            f"[PASS] {len(lines)} RealSense device(s) detected (expected ≥{expected})"
        )
        return True, msgs

    msgs.append(
        f"[FAIL] Only {len(lines)} RealSense device(s) detected, "
        f"expected ≥{expected}"
    )
    for l in lines:
        msgs.append(f"       {l.strip()}")
    return False, msgs


def test_shared_controller_bandwidth() -> tuple[bool, list[str]]:
    """Warn if multiple RealSense cameras share the same USB root hub."""
    msgs: list[str] = []
    uvc_entries = _find_realsense_usb_devices()

    buses: dict[int, set[int]] = {}
    for e in uvc_entries:
        buses.setdefault(e["bus"], set()).add(e["dev"])

    passed = True
    for bus, devs in buses.items():
        if len(devs) > 1:
            msgs.append(
                f"[WARN] Bus {bus:03d} has {len(devs)} RealSense cameras — "
                f"they share USB bandwidth; consider separate controllers"
            )
            passed = False
        else:
            msgs.append(f"[PASS] Bus {bus:03d}: single RealSense camera — no contention")

    if not buses:
        msgs.append("[INFO] No RealSense uvcvideo devices to evaluate")

    return passed, msgs


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    print("=" * 60)
    print("  USB Health Diagnostic")
    print("=" * 60)

    all_pass = True
    tests = [
        ("RealSense detection", test_realsense_detected),
        ("USB 3.0 link speed", test_realsense_usb3),
        ("Stale process check", test_no_stale_processes),
        ("Bandwidth contention", test_shared_controller_bandwidth),
    ]

    for name, fn in tests:
        print(f"\n[-] {name}")
        passed, msgs = fn()
        for m in msgs:
            print(f"    {m}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("  \033[92mUSB HEALTH OK\033[0m")
    else:
        print("  \033[91mUSB HEALTH ISSUES DETECTED\033[0m")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
