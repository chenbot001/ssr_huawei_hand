from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from hardware.manus import ManusSample
from hardware.vive import ViveSample


VIVE_WORLD_TO_UR_BASE = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class TeleopSettings:
    update_rate: float
    input_timeout: float
    translation_scale: float
    hand_motor_speed: int
    max_linear_speed: float
    max_angular_speed: float
    max_translation_from_reference: float
    max_rotation_from_reference: float
    max_tracker_translation_jump: float
    max_tracker_rotation_jump: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TeleopSettings":
        control = config["control"]
        safety = config["safety"]
        settings = cls(
            update_rate=float(control["update_rate"]),
            input_timeout=float(control["input_timeout"]),
            translation_scale=float(config["vive"]["translation_scale"]),
            hand_motor_speed=int(control["hand_motor_speed"]),
            max_linear_speed=float(safety["max_linear_speed"]),
            max_angular_speed=float(safety["max_angular_speed"]),
            max_translation_from_reference=float(
                safety["max_translation_from_reference"]
            ),
            max_rotation_from_reference=float(safety["max_rotation_from_reference"]),
            max_tracker_translation_jump=float(
                safety["max_tracker_translation_jump"]
            ),
            max_tracker_rotation_jump=float(safety["max_tracker_rotation_jump"]),
        )
        positive_values = (
            settings.update_rate,
            settings.input_timeout,
            settings.translation_scale,
            settings.max_linear_speed,
            settings.max_angular_speed,
            settings.max_translation_from_reference,
            settings.max_rotation_from_reference,
            settings.max_tracker_translation_jump,
            settings.max_tracker_rotation_jump,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("Teleoperation rates, scales, and limits must be positive")
        return settings


def pose_vector_to_matrix(pose: Sequence[float]) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"Pose vector must have shape (6,), got {values.shape}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = values[:3]
    matrix[:3, :3] = Rotation.from_rotvec(values[3:]).as_matrix()
    return matrix


def matrix_to_pose_vector(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"Pose matrix must have shape (4, 4), got {value.shape}")
    return np.concatenate(
        (value[:3, 3], Rotation.from_matrix(value[:3, :3]).as_rotvec())
    )


def vive_sample_to_matrix(sample: ViveSample) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = sample.position
    matrix[:3, :3] = Rotation.from_quat(sample.quaternion).as_matrix()
    return matrix


def compute_target_pose(
    reference_vive: np.ndarray,
    reference_ur: np.ndarray,
    current_vive: np.ndarray,
    translation_scale: float,
) -> np.ndarray:
    """Map motion relative to one clutch engagement into a UR TCP pose."""
    reference_vive = np.asarray(reference_vive, dtype=np.float64)
    reference_ur = np.asarray(reference_ur, dtype=np.float64)
    current_vive = np.asarray(current_vive, dtype=np.float64)
    if any(
        matrix.shape != (4, 4)
        for matrix in (reference_vive, reference_ur, current_vive)
    ):
        raise ValueError("All transforms must have shape (4, 4)")

    rotation_delta = current_vive[:3, :3] @ reference_vive[:3, :3].T
    mapped_rotation = (
        VIVE_WORLD_TO_UR_BASE @ rotation_delta @ VIVE_WORLD_TO_UR_BASE.T
    )
    translation_delta = current_vive[:3, 3] - reference_vive[:3, 3]
    mapped_translation = (
        VIVE_WORLD_TO_UR_BASE @ translation_delta * float(translation_scale)
    )
    target = np.eye(4, dtype=np.float64)
    target[:3, :3] = mapped_rotation @ reference_ur[:3, :3]
    target[:3, 3] = reference_ur[:3, 3] + mapped_translation
    return matrix_to_pose_vector(target)


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    delta = first[:3, :3] @ second[:3, :3].T
    return float(np.linalg.norm(Rotation.from_matrix(delta).as_rotvec()))


def _limit_target_pose(
    previous: np.ndarray,
    requested: np.ndarray,
    max_translation: float,
    max_rotation: float,
) -> np.ndarray:
    target = np.asarray(requested, dtype=np.float64).copy()
    translation_delta = target[:3, 3] - previous[:3, 3]
    translation_distance = float(np.linalg.norm(translation_delta))
    if translation_distance > max_translation:
        target[:3, 3] = previous[:3, 3] + (
            translation_delta * (max_translation / translation_distance)
        )

    rotation_delta = target[:3, :3] @ previous[:3, :3].T
    rotation_vector = Rotation.from_matrix(rotation_delta).as_rotvec()
    rotation_distance = float(np.linalg.norm(rotation_vector))
    if rotation_distance > max_rotation:
        limited_delta = Rotation.from_rotvec(
            rotation_vector * (max_rotation / rotation_distance)
        ).as_matrix()
        target[:3, :3] = limited_delta @ previous[:3, :3]
    return target


class TeleopController:
    """Fail-closed combined UR5 and RYHand teleoperation lifecycle."""

    def __init__(
        self,
        arm: Any | None,
        hand: Any | None,
        vive: Any | None,
        manus: Any | None,
        ik: Any | None,
        settings: TeleopSettings,
        use_right_manus: bool = False,
        report: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        control_arm: bool = True,
        control_hand: bool = True,
    ) -> None:
        if not control_arm and not control_hand:
            raise ValueError("At least one teleoperation output must be enabled")
        if control_arm and (arm is None or vive is None):
            raise ValueError("UR teleoperation requires an arm and Vive input")
        if control_hand and (hand is None or manus is None or ik is None):
            raise ValueError("RYHand teleoperation requires a hand, Manus input, and IK")
        self.arm = arm
        self.hand = hand
        self.vive = vive
        self.manus = manus
        self.ik = ik
        self.settings = settings
        self.use_right_manus = use_right_manus
        self.report = report or (lambda _: None)
        self.clock = clock
        self.control_arm = control_arm
        self.control_hand = control_hand
        self.clutch_engaged = True
        self.last_stop_reason: str | None = None
        self.last_hand_stop_reason: str | None = None
        self.last_hand_angles: np.ndarray | None = None
        self.last_target_pose: np.ndarray | None = None
        self.motion_delta = np.zeros(6, dtype=np.float64)
        self._reference_vive: np.ndarray | None = None
        self._reference_ur: np.ndarray | None = None
        self._reference_generation: int | None = None
        self._last_vive: np.ndarray | None = None
        self._last_target: np.ndarray | None = None
        self._last_command_time: float | None = None
        self._toggle_requested = threading.Event()
        self._closed = False

    def _samples(self) -> tuple[ViveSample | None, ManusSample | None]:
        vive_sample = self.vive.get_latest() if self.control_arm else None
        manus_sample = (
            self.manus.get_latest(self.use_right_manus) if self.control_hand else None
        )
        return vive_sample, manus_sample

    @staticmethod
    def _sample_fresh(
        sample: ViveSample | ManusSample | None, now: float, timeout: float
    ) -> bool:
        return sample is not None and 0.0 <= now - sample.timestamp <= timeout

    @property
    def armed(self) -> bool:
        """Compatibility alias: true only while the released clutch drives UR."""
        return self.control_arm and not self.clutch_engaged

    def request_toggle(self) -> None:
        self._toggle_requested.set()

    def toggle_clutch(self, now: float | None = None) -> bool:
        if not self.control_arm:
            self.report("Clutch unchanged: this mode has no UR output")
            return False
        if not self.clutch_engaged:
            self.engage_clutch("operator engaged clutch")
            return False

        return self.release_clutch(now)

    def release_clutch(self, now: float | None = None) -> bool:
        current_time = self.clock() if now is None else now
        vive_sample = self.vive.get_latest() if self.vive is not None else None
        if not self._sample_fresh(
            vive_sample, current_time, self.settings.input_timeout
        ):
            self.report("Cannot release clutch: Vive input must be fresh")
            return False

        assert vive_sample is not None and self.arm is not None
        self._reference_vive = vive_sample_to_matrix(vive_sample)
        self._reference_ur = pose_vector_to_matrix(self.arm.get_tcp_pose())
        self._reference_generation = vive_sample.generation
        self._last_vive = self._reference_vive.copy()
        self._last_target = self._reference_ur.copy()
        self._last_command_time = current_time
        self.motion_delta = np.zeros(6, dtype=np.float64)
        self.clutch_engaged = False
        self.last_stop_reason = None
        self.report("CLUTCH RELEASED: UR tracker-delta control active")
        return True

    def engage_clutch(self, reason: str) -> None:
        if not self.clutch_engaged and self.control_arm:
            assert self.arm is not None
            self.arm.servo_stop()
        self.clutch_engaged = True
        self._reference_vive = None
        self._reference_ur = None
        self._reference_generation = None
        self._last_vive = None
        self._last_target = None
        self._last_command_time = None
        self.motion_delta = np.zeros(6, dtype=np.float64)
        self.last_stop_reason = reason
        self.report(f"CLUTCH ENGAGED: {reason}; UR held")

    def step(self, now: float | None = None) -> None:
        current_time = self.clock() if now is None else now
        if self._toggle_requested.is_set():
            self._toggle_requested.clear()
            self.toggle_clutch(current_time)

        vive_sample, manus_sample = self._samples()
        vive_fresh = not self.control_arm or self._sample_fresh(
            vive_sample, current_time, self.settings.input_timeout
        )
        manus_fresh = not self.control_hand or self._sample_fresh(
            manus_sample, current_time, self.settings.input_timeout
        )

        try:
            hand_angles: np.ndarray | None = None
            if self.control_hand and manus_fresh:
                assert manus_sample is not None and self.ik is not None
                hand_angles = self.ik.compute_hand_angles(manus_sample.fingers)
                if hand_angles is None:
                    raise RuntimeError("RYHand IK failed")
                assert self.hand is not None
                hand_result = self.hand.set_angles(
                    hand_angles,
                    speed=self.settings.hand_motor_speed,
                    radians=True,
                )
                if hand_result is not None and not np.all(hand_result):
                    raise RuntimeError("RYHand rejected one or more joint commands")
                self.last_hand_angles = np.asarray(
                    hand_angles, dtype=np.float64
                ).copy()
                self.last_hand_stop_reason = None
            elif self.control_hand:
                self.last_hand_stop_reason = "Manus input stale; RYHand held"

            if self.control_arm:
                if not vive_fresh:
                    if not self.clutch_engaged:
                        self.engage_clutch("Vive input stale")
                    return
                if self.clutch_engaged:
                    return
                assert vive_sample is not None and self.arm is not None
                assert self._reference_vive is not None and self._reference_ur is not None
                assert self._last_vive is not None and self._last_target is not None
                if vive_sample.generation != self._reference_generation:
                    self.engage_clutch("tracker session changed")
                    return
                current_vive = vive_sample_to_matrix(vive_sample)
                tracker_translation_jump = float(
                    np.linalg.norm(current_vive[:3, 3] - self._last_vive[:3, 3])
                )
                tracker_rotation_jump = _rotation_distance(
                    current_vive, self._last_vive
                )
                if (
                    tracker_translation_jump
                    > self.settings.max_tracker_translation_jump
                    or tracker_rotation_jump
                    > self.settings.max_tracker_rotation_jump
                ):
                    self.engage_clutch("tracker pose jumped")
                    return
                requested_pose = compute_target_pose(
                    self._reference_vive,
                    self._reference_ur,
                    current_vive,
                    self.settings.translation_scale,
                )
                requested_matrix = pose_vector_to_matrix(requested_pose)
                if (
                    np.linalg.norm(
                        requested_matrix[:3, 3] - self._reference_ur[:3, 3]
                    )
                    > self.settings.max_translation_from_reference
                    or _rotation_distance(requested_matrix, self._reference_ur)
                    > self.settings.max_rotation_from_reference
                ):
                    self.engage_clutch("reference motion limit exceeded")
                    return

                motion_delta = np.concatenate(
                    (
                        requested_matrix[:3, 3] - self._reference_ur[:3, 3],
                        Rotation.from_matrix(
                            requested_matrix[:3, :3]
                            @ self._reference_ur[:3, :3].T
                        ).as_rotvec(),
                    )
                )

                nominal_period = 1.0 / self.settings.update_rate
                elapsed = (
                    nominal_period
                    if self._last_command_time is None
                    else max(0.0, current_time - self._last_command_time)
                )
                command_period = max(
                    nominal_period, min(elapsed, 2.0 * nominal_period)
                )
                target_matrix = _limit_target_pose(
                    self._last_target,
                    requested_matrix,
                    self.settings.max_linear_speed * command_period,
                    self.settings.max_angular_speed * command_period,
                )
                target_pose = matrix_to_pose_vector(target_matrix)
                safety_check = getattr(self.arm, "is_pose_within_safety_limits", None)
                if safety_check is not None and not safety_check(target_pose):
                    self.engage_clutch("UR safety limits rejected target")
                    return
                self.arm.servo_l(target_pose)
                self.last_target_pose = target_pose
                self.motion_delta = motion_delta
                self._last_vive = current_vive
                self._last_target = target_matrix
                self._last_command_time = current_time
        except Exception:
            if self.control_arm:
                self.engage_clutch("control fault")
            if self.control_hand:
                self.last_hand_stop_reason = "control fault; RYHand held"
            raise

    def run(self) -> None:
        interval = 1.0 / self.settings.update_rate
        while True:
            started = self.clock()
            self.step(started)
            remaining = interval - (self.clock() - started)
            if remaining > 0:
                time.sleep(remaining)

    def close(self) -> None:
        if self._closed:
            return
        errors: list[Exception] = []
        if self.control_arm:
            try:
                assert self.arm is not None
                self.arm.servo_stop()
            except Exception as exc:
                errors.append(exc)
        self.clutch_engaged = True
        self.motion_delta = np.zeros(6, dtype=np.float64)
        resources = (self.ik, self.manus, self.vive, self.hand, self.arm)
        for resource in (item for item in resources if item is not None):
            try:
                resource.close()
            except Exception as exc:
                errors.append(exc)
        self._closed = True
        if errors:
            raise RuntimeError(
                "Teleop cleanup failed: " + "; ".join(str(error) for error in errors)
            )

    def __enter__(self) -> "TeleopController":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
