import numpy as np
import pytest

from control.teleop import TeleopController, TeleopSettings
from hardware.manus import ManusSample
from hardware.vive import ViveSample


class FakeArm:
    def __init__(self):
        self.pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
        self.targets = []
        self.stop_count = 0
        self.closed = False
        self.safe = True

    def get_tcp_pose(self):
        return list(self.pose)

    def servo_l(self, pose):
        self.targets.append(np.asarray(pose))

    def servo_stop(self):
        self.stop_count += 1

    def is_pose_within_safety_limits(self, pose):
        return self.safe

    def close(self):
        self.closed = True


class FakeHand:
    def __init__(self):
        self.targets = []
        self.closed = False
        self.result = None

    def set_angles(self, angles, speed, radians):
        self.targets.append((np.asarray(angles), speed, radians))
        return self.result

    def close(self):
        self.closed = True


class FakeVive:
    def __init__(self, timestamp=10.0):
        self.sample = ViveSample(
            np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), timestamp, 0
        )
        self.closed = False

    def get_latest(self):
        return self.sample

    def close(self):
        self.closed = True


class FakeManus:
    def __init__(self, timestamp=10.0):
        self.sample = ManusSample(
            17, "left", np.zeros((10, 3)), np.zeros(3), timestamp
        )
        self.closed = False

    def get_latest(self, use_right=False):
        return self.sample

    def close(self):
        self.closed = True


class FakeIK:
    def __init__(self):
        self.result = np.arange(15, dtype=float) / 100.0
        self.closed = False

    def compute_hand_angles(self, fingers):
        return None if self.result is None else self.result.copy()

    def close(self):
        self.closed = True


def make_controller():
    resources = FakeArm(), FakeHand(), FakeVive(), FakeManus(), FakeIK()
    controller = TeleopController(
        *resources,
        TeleopSettings(80.0, 0.25, 1.0, 1000, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        clock=lambda: 10.0,
    )
    return controller, resources


def test_starts_clutched_with_ur_held_and_hand_tracking_active():
    controller, (arm, hand, *_rest) = make_controller()
    controller.step(10.0)
    assert controller.clutch_engaged
    assert not controller.armed
    assert arm.targets == []
    assert len(hand.targets) == 1
    np.testing.assert_array_equal(controller.motion_delta, np.zeros(6))


def test_cannot_release_clutch_with_stale_vive():
    controller, (_arm, hand, vive, _manus, _ik) = make_controller()
    vive.sample = ViveSample(
        np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), 9.0, 0
    )
    assert not controller.toggle_clutch(10.0)
    controller.step(10.0)
    assert controller.clutch_engaged
    assert not controller.armed
    assert len(hand.targets) == 1


def test_release_captures_references_and_commands_zero_delta():
    controller, (arm, hand, *_rest) = make_controller()
    assert controller.toggle_clutch(10.0)
    controller.step(10.0)
    assert not controller.clutch_engaged
    assert controller.armed
    np.testing.assert_allclose(arm.targets[-1], arm.pose)
    assert len(hand.targets) == 1
    np.testing.assert_array_equal(controller.motion_delta, np.zeros(6))


def test_tracker_position_delta_moves_ur_like_legacy_t265_mapping():
    controller, (arm, _hand, vive, _manus, _ik) = make_controller()
    controller.toggle_clutch(10.0)
    vive.sample = ViveSample(
        np.array([0.02, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        10.01,
        0,
    )

    controller.step(10.01)

    np.testing.assert_allclose(arm.targets[-1][:3], [0.1, 0.18, 0.3])
    np.testing.assert_allclose(controller.motion_delta, [0.0, -0.02, 0.0, 0.0, 0.0, 0.0])


def test_tracker_rotation_delta_is_reported_in_ur_base_frame():
    controller, (_arm, _hand, vive, _manus, _ik) = make_controller()
    controller.toggle_clutch(10.0)
    angle = 0.1
    vive.sample = ViveSample(
        np.zeros(3),
        np.array([np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)]),
        10.01,
        0,
    )

    controller.step(10.01)

    np.testing.assert_allclose(
        controller.motion_delta,
        [0.0, 0.0, 0.0, 0.0, -angle, 0.0],
        atol=1e-10,
    )


def test_stale_vive_engages_clutch_while_hand_continues():
    controller, (arm, hand, vive, manus, _ik) = make_controller()
    controller.toggle_clutch(10.0)
    controller.step(10.0)
    hand_command_count = len(hand.targets)

    manus.sample = ManusSample(
        17, "left", np.ones((10, 3)), np.zeros(3), 10.3
    )
    controller.step(10.3)
    assert controller.clutch_engaged
    assert not controller.armed
    assert arm.stop_count == 1
    assert len(hand.targets) == hand_command_count + 1
    assert controller.last_stop_reason == "Vive input stale"
    np.testing.assert_array_equal(controller.motion_delta, np.zeros(6))

    vive.sample = ViveSample(
        np.ones(3), np.array([0.0, 0.0, 0.0, 1.0]), 10.3, 0
    )
    controller.step(10.3)
    assert not controller.armed
    assert len(hand.targets) == hand_command_count + 2
    assert controller.toggle_clutch(10.3)


def test_stale_manus_holds_hand_while_ur_continues_and_recovers_automatically():
    controller, (arm, hand, vive, manus, _ik) = make_controller()
    controller.toggle_clutch(10.0)
    controller.step(10.0)
    hand_command_count = len(hand.targets)
    vive.sample = ViveSample(
        np.array([0.01, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        10.3,
        0,
    )

    controller.step(10.3)

    assert controller.armed
    assert len(arm.targets) == 2
    assert len(hand.targets) == hand_command_count
    assert controller.last_hand_stop_reason == "Manus input stale; RYHand held"

    manus.sample = ManusSample(
        17, "left", np.ones((10, 3)), np.zeros(3), 10.3
    )
    controller.step(10.3)
    assert len(hand.targets) == hand_command_count + 1
    assert controller.last_hand_stop_reason is None


def test_reengage_holds_ur_and_rerelease_has_no_jump():
    controller, (arm, hand, vive, _manus, _ik) = make_controller()
    controller.toggle_clutch(10.0)
    vive.sample = ViveSample(
        np.array([0.02, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        10.01,
        0,
    )
    controller.step(10.01)
    arm.pose = arm.targets[-1].tolist()

    assert not controller.toggle_clutch(10.01)
    assert controller.clutch_engaged
    np.testing.assert_array_equal(controller.motion_delta, np.zeros(6))
    held_target_count = len(arm.targets)
    vive.sample = ViveSample(
        np.array([0.2, -0.1, 0.1]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        10.02,
        0,
    )
    controller.step(10.02)
    assert len(arm.targets) == held_target_count
    assert len(hand.targets) == 2

    assert controller.toggle_clutch(10.02)
    controller.step(10.02)
    np.testing.assert_allclose(arm.targets[-1], arm.pose)


def test_control_fault_engages_arm_clutch():
    controller, (arm, _hand, _vive, _manus, ik) = make_controller()
    controller.toggle_clutch(10.0)
    ik.result = None
    with pytest.raises(RuntimeError, match="IK failed"):
        controller.step(10.0)
    assert controller.clutch_engaged
    assert not controller.armed
    assert arm.stop_count == 1


def test_rejected_hand_command_engages_arm_clutch():
    controller, (arm, hand, _vive, _manus, _ik) = make_controller()
    controller.toggle_clutch(10.0)
    hand.result = np.array([True] * 14 + [False])
    with pytest.raises(RuntimeError, match="rejected"):
        controller.step(10.0)
    assert controller.clutch_engaged
    assert not controller.armed
    assert arm.stop_count == 1


def test_close_stops_arm_without_opening_hand_and_closes_all_resources():
    controller, resources = make_controller()
    arm, hand, vive, manus, ik = resources
    controller.toggle_clutch(10.0)
    controller.step(10.0)
    original = hand.targets[-1][0].copy()
    controller.close()
    np.testing.assert_array_equal(hand.targets[-1][0], original)
    assert arm.stop_count == 1
    assert all(resource.closed for resource in (arm, hand, vive, manus, ik))


def test_tracker_generation_change_engages_clutch_and_requires_release():
    controller, (arm, _hand, vive, _manus, _ik) = make_controller()
    controller.toggle_clutch(10.0)
    vive.sample = ViveSample(
        np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), 10.0, 1
    )

    controller.step(10.0)

    assert controller.clutch_engaged
    assert not controller.armed
    assert controller.last_stop_reason == "tracker session changed"
    assert arm.stop_count == 1


def test_tracker_pose_jump_stops_before_commanding_arm():
    controller, (arm, _hand, vive, _manus, _ik) = make_controller()
    controller.settings = TeleopSettings(
        80.0, 0.25, 1.0, 1000, 10.0, 10.0, 10.0, 10.0, 0.05, 0.35
    )
    controller.toggle_clutch(10.0)
    vive.sample = ViveSample(
        np.array([0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        10.0,
        0,
    )

    controller.step(10.0)

    assert controller.clutch_engaged
    assert not controller.armed
    assert controller.last_stop_reason == "tracker pose jumped"
    assert arm.targets == []


def test_ur_safety_rejection_stops_before_servo_command():
    controller, (arm, _hand, _vive, _manus, _ik) = make_controller()
    controller.toggle_clutch(10.0)
    arm.safe = False

    controller.step(10.0)

    assert controller.clutch_engaged
    assert not controller.armed
    assert controller.last_stop_reason == "UR safety limits rejected target"
    assert arm.targets == []


def test_cartesian_target_change_is_speed_limited():
    controller, (arm, _hand, vive, _manus, _ik) = make_controller()
    controller.settings = TeleopSettings(
        80.0, 0.25, 1.0, 1000, 0.08, 10.0, 10.0, 10.0, 0.1, 10.0
    )
    controller.toggle_clutch(10.0)
    vive.sample = ViveSample(
        np.array([0.04, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        10.0,
        0,
    )

    controller.step(10.0)

    commanded_change = np.linalg.norm(arm.targets[-1][:3] - np.asarray(arm.pose[:3]))
    assert commanded_change == pytest.approx(0.001)


def test_vive_only_mode_requires_no_manus_or_hand():
    arm, vive = FakeArm(), FakeVive()
    controller = TeleopController(
        arm,
        None,
        vive,
        None,
        None,
        TeleopSettings(80.0, 0.25, 1.0, 1000, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        clock=lambda: 10.0,
        control_hand=False,
    )

    assert controller.toggle_clutch(10.0)
    controller.step(10.0)

    assert len(arm.targets) == 1
    controller.close()
    assert arm.closed and vive.closed


def test_manus_only_mode_requires_no_vive_or_arm():
    hand, manus, ik = FakeHand(), FakeManus(), FakeIK()
    controller = TeleopController(
        None,
        hand,
        None,
        manus,
        ik,
        TeleopSettings(80.0, 0.25, 1.0, 1000, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        clock=lambda: 10.0,
        control_arm=False,
    )

    controller.step(10.0)
    assert len(hand.targets) == 1

    controller.step(10.3)
    assert controller.clutch_engaged
    assert not controller.armed
    assert len(hand.targets) == 1
    manus.sample = ManusSample(
        17, "left", np.ones((10, 3)), np.zeros(3), 10.3
    )
    controller.step(10.3)
    assert len(hand.targets) == 2
    assert not controller.toggle_clutch(10.3)
    controller.close()
    assert hand.closed and manus.closed and ik.closed


def test_partial_modes_validate_only_their_selected_resources():
    settings = TeleopSettings(80.0, 0.25, 1.0, 1000, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
    with pytest.raises(ValueError, match="arm and Vive"):
        TeleopController(None, None, None, None, None, settings, control_hand=False)
    with pytest.raises(ValueError, match="hand, Manus input, and IK"):
        TeleopController(None, None, None, None, None, settings, control_arm=False)
