from __future__ import annotations

import argparse
import sys
from typing import Any


class _DryRunArm:
    def __init__(self) -> None:
        self.pose = [0.0] * 6
        self.last_target = None

    def get_tcp_pose(self) -> list[float]:
        return list(self.pose)

    def servo_l(self, pose) -> None:
        self.last_target = list(pose)

    def servo_stop(self) -> None:
        return None

    def close(self) -> None:
        return None


class _DryRunHand:
    def __init__(self) -> None:
        self.last_angles = None

    def set_angles(self, angles, speed: int, radians: bool = True) -> None:
        self.last_angles = list(angles)

    def close(self) -> None:
        return None


def _close_resources(resources: list[Any]) -> None:
    for resource in reversed(resources):
        try:
            resource.close()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vive + Manus teleoperation for a UR5 and RYHand"
    )
    parser.add_argument(
        "--vive-right",
        action="store_true",
        help="use the right Vive tracker for UR control",
    )
    parser.add_argument(
        "--manus-right",
        action="store_true",
        help="use the right MANUS glove for RYHand control",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute targets without connecting either actuator",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=5.0,
        help="seconds to wait for fresh Vive and Manus samples",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.startup_timeout <= 0:
        print("--startup-timeout must be positive", file=sys.stderr)
        return 2

    resources: list[Any] = []
    controller = None
    listener = None
    try:
        from pynput import keyboard

        from config import get_hardware_config, get_teleop_config
        from control.ryhand_ik import RYHandIK
        from control.teleop import TeleopController, TeleopSettings
        from hardware.arm_ur5 import UR5Arm
        from hardware.manus import ManusReceiver
        from hardware.ruiyan_driver import RyHandController
        from hardware.vive import ViveTracker

        hardware = get_hardware_config()
        settings = TeleopSettings.from_config(get_teleop_config())
        glove = hardware["manus_glove"]
        tracker_config = hardware["vive_tracker"]
        tracker_side = "right" if args.vive_right else "left"
        manus_side = "right" if args.manus_right else "left"
        tracker_serial = str(tracker_config[f"{tracker_side}_serial"])

        manus = ManusReceiver(
            str(glove["address"]), int(glove["left_id"]), int(glove["right_id"])
        )
        resources.append(manus)
        vive = ViveTracker(tracker_serial)
        resources.append(vive)
        ik = RYHandIK(gui=False)
        resources.append(ik)

        print(
            f"Waiting for {tracker_side} Vive tracker and {manus_side} MANUS glove..."
        )
        if vive.wait_for_sample(args.startup_timeout) is None:
            detail = vive.last_error or "no pose received"
            raise RuntimeError(
                f"No pose from Vive tracker {tracker_serial} before startup timeout: {detail}"
            )
        if manus.wait_for_sample(args.manus_right, args.startup_timeout) is None:
            raise RuntimeError(
                f"No sample from the configured {manus_side} glove before startup timeout"
            )

        if args.dry_run:
            arm: Any = _DryRunArm()
            hand: Any = _DryRunHand()
        else:
            hand = RyHandController(port=str(hardware["ruiyan_hand"]["port"]))
            resources.append(hand)
            arm = UR5Arm(ip=str(hardware["ur_arm"]["ip"]))
            resources.append(arm)

        controller = TeleopController(
            arm,
            hand,
            vive,
            manus,
            ik,
            settings,
            use_right_manus=args.manus_right,
            report=print,
        )
        resources.clear()

        def on_press(key) -> None:
            if key == keyboard.Key.space:
                controller.request_toggle()

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        mode = "DRY RUN" if args.dry_run else "LIVE"
        print(
            f"{mode} ready. RYHand follows MANUS continuously; UR starts held. "
            "Press Space to release/engage the arm clutch; Ctrl+C to exit."
        )
        controller.run()
    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
        return 0
    except Exception as exc:
        print(f"Teleoperation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if listener is not None:
            listener.stop()
        if controller is not None:
            try:
                controller.close()
            except Exception as exc:
                print(f"Cleanup warning: {exc}", file=sys.stderr)
        else:
            _close_resources(resources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
