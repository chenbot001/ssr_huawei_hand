from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from hardware.vive import ViveTracker, _headless_vrserver_command, _matrix34_to_components


def test_openvr_matrix_converts_to_xyzw_pose():
    matrix = [
        [0.0, -1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
    ]

    position, quaternion = _matrix34_to_components(matrix)

    np.testing.assert_allclose(position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        quaternion, [0.0, 0.0, 2**-0.5, 2**-0.5], atol=1e-9
    )


def test_headless_vrserver_command_starts_server_without_gui():
    command = _headless_vrserver_command("/runtime")

    assert command == [
        "/runtime/bin/vrenv.sh",
        "/runtime/bin/linux64/vrserver",
        "-keepalive",
    ]
    assert all("vrmonitor" not in part and "vrcompositor" not in part for part in command)


def test_pose_components_reject_non_finite_values():
    matrix = [
        [1.0, 0.0, 0.0, float("nan")],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    with pytest.raises(ValueError, match="non-finite"):
        _matrix34_to_components(matrix)


def test_tracker_is_resolved_by_persistent_serial():
    tracker = ViveTracker.__new__(ViveTracker)
    tracker.serial = "LHR-TARGET"
    tracker._openvr = SimpleNamespace(
        k_unMaxTrackedDeviceCount=3,
        TrackedDeviceClass_GenericTracker=3,
        Prop_SerialNumber_String=1000,
    )
    vr_system = mock.Mock()
    vr_system.isTrackedDeviceConnected.side_effect = [True, True]
    vr_system.getTrackedDeviceClass.side_effect = [3, 3]
    vr_system.getStringTrackedDeviceProperty.side_effect = [
        "LHR-OTHER",
        "LHR-TARGET",
    ]

    assert tracker._tracker_index(vr_system) == 1


def test_pose_requires_running_tracking_and_matching_serial():
    tracker = ViveTracker.__new__(ViveTracker)
    tracker.serial = "LHR-TARGET"
    tracker._openvr = SimpleNamespace(
        TrackingResult_Running_OK=200,
        Prop_SerialNumber_String=1000,
    )
    vr_system = mock.Mock()
    vr_system.getStringTrackedDeviceProperty.return_value = "LHR-TARGET"

    good = SimpleNamespace(
        bDeviceIsConnected=True,
        bPoseIsValid=True,
        eTrackingResult=200,
    )
    calibrating = SimpleNamespace(
        bDeviceIsConnected=True,
        bPoseIsValid=True,
        eTrackingResult=100,
    )

    assert tracker._pose_is_usable(vr_system, 2, good)
    assert not tracker._pose_is_usable(vr_system, 2, calibrating)
