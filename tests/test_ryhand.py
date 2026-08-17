import numpy as np
import pytest
from pathlib import Path
import xml.etree.ElementTree as ET

from control.ryhand_ik import RYHandIK, URDF_PATH, ik_to_hand_angles
from hardware.ruiyan_driver import (
    _RyHandBus,
    angles_to_motor_positions,
    motor_positions_to_angles,
)


def test_ik_mapping_enables_only_thumb_and_index():
    joints = np.full(20, np.deg2rad(45.0))
    angles = ik_to_hand_angles(joints)
    assert np.isclose(angles[0], np.deg2rad(10.0))
    assert angles[1] > 0 and angles[2] > 0
    assert angles[3] == 0
    assert angles[4] > 0 and angles[5] > 0
    np.testing.assert_array_equal(angles[6:], np.zeros(9))


def test_ik_mapping_clamps_joint_limits():
    joints = np.full(20, 100.0)
    angles = ik_to_hand_angles(joints)
    assert angles[1] == pytest.approx(np.deg2rad(90.0))
    assert angles[2] == pytest.approx(np.deg2rad(75.0))
    assert angles[4] == pytest.approx(np.deg2rad(90.0))
    assert angles[5] == pytest.approx(np.deg2rad(75.0))


def test_motor_mapping_round_trip_with_quantization():
    one_finger = np.deg2rad([10.0, 40.0, 30.0])
    angles = np.tile(one_finger, 5)
    raw = angles_to_motor_positions(angles)
    restored = motor_positions_to_angles(raw)
    np.testing.assert_allclose(restored, angles, atol=1e-3)


def test_motor_mapping_clamps_out_of_range_angles():
    angles = np.tile([10.0, -10.0, 10.0], 5)
    raw = angles_to_motor_positions(angles)
    assert np.all((raw >= 0) & (raw <= 4095))


def test_mapping_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        ik_to_hand_angles(np.zeros(19))
    with pytest.raises(ValueError):
        angles_to_motor_positions(np.zeros(14))


def test_missing_vendor_library_fails_with_actionable_path(tmp_path: Path):
    missing = tmp_path / "libRyhand64.so"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        _RyHandBus("can0", missing)


def test_official_urdf_has_only_resolvable_relative_meshes():
    assert URDF_PATH.is_file()
    assert "Bidex" not in str(URDF_PATH)
    root = ET.parse(URDF_PATH).getroot()
    mesh_names = {
        mesh.attrib["filename"] for mesh in root.findall(".//mesh")
    }
    assert len(mesh_names) == 26
    assert all(not name.startswith("package://") for name in mesh_names)
    assert all((URDF_PATH.parent / name).resolve().is_file() for name in mesh_names)


def test_pybullet_console_render_has_rgba_shape():
    with RYHandIK(gui=False) as ik:
        frame = ik.render_rgba(32, 24)
    assert frame.shape == (24, 32, 4)
    assert frame.dtype == np.uint8
