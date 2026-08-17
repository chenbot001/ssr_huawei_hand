import numpy as np
from scipy.spatial.transform import Rotation

from control.teleop import (
    VIVE_WORLD_TO_UR_BASE,
    compute_target_pose,
    matrix_to_pose_vector,
    pose_vector_to_matrix,
)


def test_pose_vector_matrix_round_trip():
    pose = np.array([0.2, -0.1, 0.5, 0.3, -0.2, 0.1])
    actual = matrix_to_pose_vector(pose_vector_to_matrix(pose))
    np.testing.assert_allclose(actual, pose, atol=1e-9)


def test_vive_alignment_is_a_rotation():
    np.testing.assert_allclose(
        VIVE_WORLD_TO_UR_BASE @ VIVE_WORLD_TO_UR_BASE.T, np.eye(3)
    )
    assert np.isclose(np.linalg.det(VIVE_WORLD_TO_UR_BASE), 1.0)


def test_clutch_engagement_has_no_target_jump():
    vive = np.eye(4)
    vive[:3, 3] = [3.0, 4.0, 5.0]
    vive[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.3, -0.1]).as_matrix()
    ur_pose = np.array([0.4, -0.3, 0.2, 0.1, 0.2, 0.3])
    target = compute_target_pose(vive, pose_vector_to_matrix(ur_pose), vive, 1.0)
    np.testing.assert_allclose(target, ur_pose, atol=1e-9)


def test_translation_uses_vive_to_ur_alignment_and_scale():
    reference = np.eye(4)
    current = np.eye(4)
    current[:3, 3] = [1.0, 0.0, 0.0]
    target = compute_target_pose(reference, np.eye(4), current, 0.5)
    np.testing.assert_allclose(target[:3], [0.0, -0.5, 0.0])


def test_reengagement_uses_new_ur_and_vive_references():
    new_vive = np.eye(4)
    new_vive[:3, 3] = [8.0, -2.0, 1.0]
    new_ur = np.array([0.7, 0.2, -0.1, -0.2, 0.1, 0.4])
    target = compute_target_pose(
        new_vive, pose_vector_to_matrix(new_ur), new_vive.copy(), 1.0
    )
    np.testing.assert_allclose(target, new_ur, atol=1e-9)


def test_rotation_delta_is_mapped_in_vive_world_frame():
    reference = np.eye(4)
    reference[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.1]).as_matrix()
    world_delta = Rotation.from_rotvec([0.0, 0.25, 0.0]).as_matrix()
    current = reference.copy()
    current[:3, :3] = world_delta @ reference[:3, :3]
    ur_reference = pose_vector_to_matrix([0.4, -0.2, 0.3, 0.2, 0.1, -0.1])

    target = pose_vector_to_matrix(
        compute_target_pose(reference, ur_reference, current, 1.0)
    )
    mapped_delta = VIVE_WORLD_TO_UR_BASE @ world_delta @ VIVE_WORLD_TO_UR_BASE.T
    np.testing.assert_allclose(
        target[:3, :3], mapped_delta @ ur_reference[:3, :3], atol=1e-9
    )


def test_fixed_tracker_mount_rotation_cancels_from_world_delta():
    control_reference = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
    control_current = (
        Rotation.from_rotvec([0.15, 0.0, -0.1]).as_matrix() @ control_reference
    )
    mount = Rotation.from_euler("xyz", [-0.7, 0.4, 0.2]).as_matrix()
    mounted_reference = np.eye(4)
    mounted_reference[:3, :3] = control_reference @ mount
    mounted_current = np.eye(4)
    mounted_current[:3, :3] = control_current @ mount
    direct_reference = np.eye(4)
    direct_reference[:3, :3] = control_reference
    direct_current = np.eye(4)
    direct_current[:3, :3] = control_current

    mounted_target = compute_target_pose(
        mounted_reference, np.eye(4), mounted_current, 1.0
    )
    direct_target = compute_target_pose(
        direct_reference, np.eye(4), direct_current, 1.0
    )
    np.testing.assert_allclose(mounted_target, direct_target, atol=1e-9)
