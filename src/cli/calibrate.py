from __future__ import annotations

import argparse
import sys
import time
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live Manus calibration for the enabled RYHand thumb and index"
    )
    parser.add_argument("--use-right", action="store_true", help="use the right glove")
    parser.add_argument("--dry-run", action="store_true", help="do not connect RYHand")
    parser.add_argument("--no-gui", action="store_true", help="disable PyBullet sliders")
    parser.add_argument("--rate", type=float, default=30.0, help="update rate in Hz")
    parser.add_argument("--speed", type=int, default=None, help="RYHand motor speed")
    parser.add_argument(
        "--startup-timeout", type=float, default=5.0, help="seconds to wait for Manus"
    )
    return parser


def _add_sliders(pybullet: Any, client: int, scales, offsets) -> dict[str, list]:
    pybullet.configureDebugVisualizer(
        pybullet.COV_ENABLE_GUI, 1, physicsClientId=client
    )
    sliders: dict[str, list] = {"scales": [], "offsets": []}
    for finger, name in ((0, "Thumb"), (1, "Index")):
        sliders["scales"].append(
            pybullet.addUserDebugParameter(
                f"{name} Scale", 0.3, 3.0, float(scales[finger]), client
            )
        )
        row = []
        for axis, axis_name in enumerate(("X", "Y", "Z")):
            row.append(
                pybullet.addUserDebugParameter(
                    f"{name} Offset {axis_name}",
                    -0.2,
                    0.2,
                    float(offsets[finger, axis]),
                    client,
                )
            )
        sliders["offsets"].append(row)
    return sliders


def _read_sliders(pybullet: Any, sliders: dict[str, list], scales, offsets) -> None:
    for slot, finger in enumerate((0, 1)):
        scales[finger] = pybullet.readUserDebugParameter(sliders["scales"][slot])
        for axis in range(3):
            offsets[finger, axis] = pybullet.readUserDebugParameter(
                sliders["offsets"][slot][axis]
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rate <= 0 or args.startup_timeout <= 0:
        print("--rate and --startup-timeout must be positive", file=sys.stderr)
        return 2

    glove_receiver = None
    ik = None
    hand = None
    ran = False
    try:
        import numpy as np
        import pybullet

        from config import get_hardware_config, get_teleop_config
        from control.ryhand_ik import RYHandIK, save_calibration
        from hardware.manus import ManusReceiver
        from hardware.ruiyan_driver import RyHandController

        hardware = get_hardware_config()
        teleop = get_teleop_config()
        glove = hardware["manus_glove"]
        timeout = float(teleop["control"]["input_timeout"])
        speed = args.speed or int(teleop["control"]["hand_motor_speed"])

        glove_receiver = ManusReceiver(
            str(glove["address"]), int(glove["left_id"]), int(glove["right_id"])
        )
        if glove_receiver.wait_for_sample(args.use_right, args.startup_timeout) is None:
            raise RuntimeError("No sample received from the selected Manus glove")
        ik = RYHandIK(gui=not args.no_gui)
        scales = ik.scales.copy()
        offsets = ik.offsets.copy()
        sliders = None
        if not args.no_gui:
            sliders = _add_sliders(pybullet, ik.physics_client, scales, offsets)
        if not args.dry_run:
            hand = RyHandController(port=str(hardware["ruiyan_hand"]["port"]))

        ran = True
        interval = 1.0 / args.rate
        print("Calibration running. Adjust thumb/index sliders; Ctrl+C saves and exits.")
        while True:
            started = time.monotonic()
            if sliders is not None:
                _read_sliders(pybullet, sliders, scales, offsets)
                ik.set_calibration(scales, offsets)
            sample = glove_receiver.get_latest(args.use_right)
            if sample is not None and sample.age(started) <= timeout:
                angles = ik.compute_hand_angles(sample.fingers)
                if angles is not None:
                    if hand is not None:
                        hand.set_angles(angles, speed=speed, radians=True)
                    degrees = np.degrees(angles)
                    print(
                        f"\rThumb prox={degrees[1]:5.1f} dist={degrees[2]:5.1f} | "
                        f"Index prox={degrees[4]:5.1f} dist={degrees[5]:5.1f}",
                        end="",
                    )
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nSaving calibration...")
        return_code = 0
    except Exception as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return_code = 1
    finally:
        if ran and ik is not None:
            try:
                from control.ryhand_ik import save_calibration

                save_calibration(ik.scales, ik.offsets)
            except Exception as exc:
                print(f"Could not save calibration: {exc}", file=sys.stderr)
                return_code = 1
        for resource in (hand, ik, glove_receiver):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    print(f"Cleanup warning: {exc}", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
