#!/usr/bin/env python3
"""
Live glove-to-hand calibration tool.

Receives Manus glove data, runs PyBullet IK, and sends commands to the
**real** RYHand so calibration changes are immediately visible on the
physical fingers rather than only in simulation.

A PyBullet GUI window provides real-time sliders for:
  - FINGER_SCALES  (5 sliders) — how far each fingertip must reach
  - FINGER_POS_OFFSETS  (5 × X/Y/Z = 15 sliders) — per-finger position bias

Slider values are written back into ssr.control.RyHand_IK's module-level
globals each frame, so the IK engine picks them up with zero latency.

On Ctrl+C the final values are saved to configs/manus_calibration.json.

Usage:
    python scripts/calibrate_hand_live.py
    python scripts/calibrate_hand_live.py --dry-run          # no CAN hardware
    python scripts/calibrate_hand_live.py --use-right        # right glove
    python scripts/calibrate_hand_live.py --rate 50          # 50 Hz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import pybullet as p
import zmq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
for _p in [src_path, project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(project_root)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import ssr.control.RyHand_IK as _rik_module          # for mutating globals
from ssr.control.RyHand_IK import RYHandIK, ik_to_hand_angles
from ssr.config import get_hardware_config, get_teleop_config

CALIBRATION_FILE = os.path.join(project_root, "configs", "manus_calibration.json")

# ---------------------------------------------------------------------------
# Hardware config
# ---------------------------------------------------------------------------
_hw_cfg = get_hardware_config()
_manus_cfg = _hw_cfg.get("manus_glove", {})
IP_ADDRESS      = _manus_cfg.get("address",   "tcp://localhost:8000")
LEFT_GLOVE_SN   = _manus_cfg.get("left_sn",   "4848debd")
RIGHT_GLOVE_SN  = _manus_cfg.get("right_sn",  "db397317")

_teleop_cfg   = get_teleop_config()
_control_cfg  = _teleop_cfg.get("control", {})
DEFAULT_SPEED = _control_cfg.get("hand_motor_speed", 800)
RESET_SPEED   = _control_cfg.get("hand_reset_speed",  400)

NUM_JOINTS      = 25
VALUES_PER_JOINT = 7
SHORT_IDX = [23, 24, 4, 5, 9, 10, 19, 20, 14, 15]

# ---------------------------------------------------------------------------
# Glove receiver
# ---------------------------------------------------------------------------
class GloveDataReceiver:
    """Background ZMQ thread that parses Manus skeleton packets."""

    def __init__(self):
        self.context = zmq.Context()
        self.socket  = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.CONFLATE, True)
        self.socket.connect(IP_ADDRESS)

        self._left_short  = None
        self._left_wrist  = None
        self._right_short = None
        self._right_wrist = None
        self._lock        = threading.Lock()

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Glove] Connecting to {IP_ADDRESS}")

    def _loop(self):
        while self._running:
            try:
                raw = self.socket.recv(flags=zmq.NOBLOCK)
                data = raw.decode("utf-8").split(",")
                if len(data) == 352:
                    self._parse(data[0:176])
                    self._parse(data[176:352])
                elif len(data) == 176:
                    self._parse(data[0:176])
            except zmq.Again:
                time.sleep(0.001)
            except Exception as exc:
                print(f"[Glove] parse error: {exc}")

    def _parse(self, data: list[str]):
        if len(data) < 176:
            return
        sn = data[0]
        short = []
        for idx in SHORT_IDX:
            base = 1 + idx * VALUES_PER_JOINT
            short.append([
                float(data[base]),
                -float(data[base + 1]),
                float(data[base + 2]),
            ])
        wrist_base = 1
        wrist = [
            float(data[wrist_base]),
            -float(data[wrist_base + 1]),
            float(data[wrist_base + 2]),
        ]
        with self._lock:
            if sn == LEFT_GLOVE_SN:
                self._left_short, self._left_wrist = short, wrist
            elif sn == RIGHT_GLOVE_SN:
                self._right_short, self._right_wrist = short, wrist
            else:
                # fallback: unknown SN → treat as left
                self._left_short, self._left_wrist = short, wrist

    def get_left_data(self) -> dict | None:
        with self._lock:
            if self._left_short and self._left_wrist:
                return {"fingers": list(self._left_short), "wrist": list(self._left_wrist)}
        return None

    def get_right_data(self) -> dict | None:
        with self._lock:
            if self._right_short and self._right_wrist:
                return {"fingers": list(self._right_short), "wrist": list(self._right_wrist)}
        return None

    def close(self):
        self._running = False
        self.socket.close()
        self.context.term()


# ---------------------------------------------------------------------------
# Slider setup (PyBullet GUI)
# ---------------------------------------------------------------------------
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Only thumb (0) and index (1) produce non-zero output on the real hand.
# Middle/ring/pinky are fully zeroed in ik_to_hand_angles, and all side
# swings are hardcoded (thumb = +10 deg, rest = 0) — none are IK-driven.
# Exposing sliders for disabled fingers would give false confidence that
# tweaking them affects the real hand, so we restrict to active fingers only.
_ACTIVE_FINGERS = [0, 1]   # thumb, index


def setup_sliders() -> dict:
    """
    Create PyBullet debug sliders for the two active fingers only.

    Active fingers  : Thumb (0), Index (1)
    Disabled fingers: Middle (2), Ring (3), Pinky (4) — always zeroed
    Swing DOFs      : not exposed — thumb hardcoded +10 deg, others 0

    Returns a dict of slider IDs: {"scales": [sid_t, sid_i],
                                   "offsets": [[sx,sy,sz], [sx,sy,sz]]}
    """
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    sliders: dict = {"scales": [], "offsets": []}

    for fi in _ACTIVE_FINGERS:
        name = FINGER_NAMES[fi]
        sid = p.addUserDebugParameter(
            f"{name} Scale", 0.3, 3.0,
            _rik_module.FINGER_SCALES[fi],
        )
        sliders["scales"].append(sid)

    for fi in _ACTIVE_FINGERS:
        name = FINGER_NAMES[fi]
        row = []
        for axis, ax_name in enumerate(["X", "Y", "Z"]):
            sid = p.addUserDebugParameter(
                f"{name} Offset {ax_name}", -0.2, 0.2,
                _rik_module.FINGER_POS_OFFSETS[fi][axis],
            )
            row.append(sid)
        sliders["offsets"].append(row)

    return sliders


def read_sliders(sliders: dict):
    """Read slider values and write them back into the RyHand_IK module globals.

    Only thumb and index calibration params are updated.  Middle/ring/pinky
    scales and offsets are intentionally left at their loaded values and are
    NOT exposed via sliders — ik_to_hand_angles zeros those fingers anyway.
    """
    for slot, fi in enumerate(_ACTIVE_FINGERS):
        _rik_module.FINGER_SCALES[fi] = p.readUserDebugParameter(sliders["scales"][slot])
        for axis in range(3):
            _rik_module.FINGER_POS_OFFSETS[fi][axis] = p.readUserDebugParameter(
                sliders["offsets"][slot][axis]
            )


# ---------------------------------------------------------------------------
# Calibration save / load helpers
# ---------------------------------------------------------------------------
def save_calibration():
    data = {
        "FINGER_SCALES":      [round(float(v), 4) for v in _rik_module.FINGER_SCALES],
        "WRIST_OFFSET":       [round(float(v), 4) for v in _rik_module.WRIST_OFFSET],
        "FINGER_POS_OFFSETS": [
            [round(float(v), 4) for v in row]
            for row in _rik_module.FINGER_POS_OFFSETS
        ],
    }
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[Calib] Saved → {CALIBRATION_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Live glove-to-hand calibration")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Skip CAN hardware — visual/IK only")
    parser.add_argument("--use-right", action="store_true",
                        help="Use right glove data instead of left")
    parser.add_argument("--rate",      type=float, default=30.0,
                        help="Control loop Hz (default: 30)")
    parser.add_argument("--speed",     type=int,   default=DEFAULT_SPEED,
                        help=f"Motor speed (default: {DEFAULT_SPEED})")
    parser.add_argument("--no-gui",    action="store_true",
                        help="Disable PyBullet GUI (no sliders; hardware-only mode)")
    args = parser.parse_args()

    use_gui = not args.no_gui

    print("=" * 65)
    print("  RYHand Live Calibration")
    print("=" * 65)
    print(f"  Glove hand : {'RIGHT' if args.use_right else 'LEFT'}")
    print(f"  Rate       : {args.rate} Hz")
    print(f"  Motor speed: {args.speed}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"  GUI sliders: {use_gui}")
    print("-" * 65)

    # ----------------------------------------------------------------
    # Initialise components
    # ----------------------------------------------------------------
    glove = GloveDataReceiver()

    # RYHandIK: gui=True opens a PyBullet window for visual feedback.
    # We then re-enable the GUI panel so our sliders are visible.
    ik_engine = RYHandIK(gui=use_gui)

    sliders: dict | None = None
    if use_gui:
        sliders = setup_sliders()
        print("[Calib] PyBullet GUI ready — adjust sliders to tune calibration.")

    hand_controller = None
    if not args.dry_run:
        try:
            from ssr.hardware.ruiyan_driver import RyHandController
            port = _hw_cfg["ruiyan_hand"]["port"]
            hand_controller = RyHandController(port=port)
            print(f"[HW] Connected to RYHand on {port}")
        except Exception as exc:
            print(f"[HW] WARNING — could not connect to hardware: {exc}")
            print("[HW] Falling back to dry-run mode.")

    print("-" * 65)
    print("Running — press Ctrl+C to stop and save calibration.")
    print("=" * 65)

    interval  = 1.0 / args.rate
    connected = False

    try:
        while True:
            t0 = time.time()

            # --- Read slider values into RyHand_IK module globals ---
            if sliders is not None:
                read_sliders(sliders)

            # --- Get glove data ---
            glove_data = glove.get_right_data() if args.use_right else glove.get_left_data()

            if glove_data is None:
                print("Waiting for glove data...", end="\r")
                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                continue

            if not connected:
                hand_label = "RIGHT" if args.use_right else "LEFT"
                print(f"\n[INFO] Glove connected ({hand_label}).")
                connected = True

            # --- IK ---
            ik_joints = ik_engine.compute_ik(glove_data)
            if ik_joints is None:
                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                continue

            # --- Map 20-joint IK → 15 motor angles ---
            hand_angles = ik_to_hand_angles(ik_joints)

            # --- Send to real hardware ---
            if hand_controller is not None:
                hand_controller.set_angles(hand_angles, speed=args.speed, radians=True)

            # --- Terminal display ---
            deg = np.rad2deg(hand_angles)
            print(
                # Only show thumb and index — middle/ring/pinky are always zero
                f"\rThumb[swing=10°(fixed) prox={deg[1]:5.1f} dist={deg[2]:5.1f}]  "
                f"Index[swing=0°(fixed) prox={deg[4]:5.1f} dist={deg[5]:5.1f}]  "
                f"Mid/Ring/Pinky=DISABLED  "
                f"Scale T={_rik_module.FINGER_SCALES[0]:.3f} I={_rik_module.FINGER_SCALES[1]:.3f}",
                end="",
            )

            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        print("\n\n[Calib] Stopping...")

    finally:
        # Reset hand to open position before disconnecting
        if hand_controller is not None:
            print("[HW] Resetting hand to open position...")
            hand_controller.set_angles(np.zeros(15), speed=RESET_SPEED, radians=True)
            time.sleep(0.6)
            hand_controller.close()

        glove.close()

        # Save calibration
        save_calibration()

        if use_gui:
            try:
                p.disconnect()
            except Exception:
                pass

        print("[Calib] Done.")


if __name__ == "__main__":
    main()
