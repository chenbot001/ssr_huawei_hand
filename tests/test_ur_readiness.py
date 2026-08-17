#!/usr/bin/env python3
"""
UR5 Readiness & Coordinate Transform Tests.

Part 1 — Hardware (requires UR5 on the network):
  * RTDE receive + control interface connectivity
  * Robot mode (must be in RUNNING state for remote control)
  * Safety status (no protective stop / emergency stop)

Part 2 — Pure-math transform tests (no hardware):
  * pose_vector_to_matrix ↔ matrix_to_pose_vector roundtrip
  * Identity transform invariance
  * Known rotation / translation mapping
  * T265_TO_UR_ALIGN axis-mapping sanity

Usage:
    python tests/test_ur_readiness.py
    python tests/test_ur_readiness.py --skip-hardware
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np
from scipy.spatial.transform import Rotation as R

from ssr.config import get_hardware_config

# ============================================================================
# Transform functions (identical to teleop scripts — defined locally so the
# tests stay self-contained and validate the exact formulas used at runtime)
# ============================================================================

T265_TO_UR_ALIGN = np.array([
    [ 0,  0, -1,  0],
    [-1,  0,  0,  0],
    [ 0,  1,  0,  0],
    [ 0,  0,  0,  1]
], dtype=np.float64)


def pose_vector_to_matrix(pose_vec):
    matrix = np.eye(4)
    matrix[:3, 3] = pose_vec[:3]
    matrix[:3, :3] = R.from_rotvec(pose_vec[3:]).as_matrix()
    return matrix


def matrix_to_pose_vector(matrix):
    pos = matrix[:3, 3]
    rot = R.from_matrix(matrix[:3, :3]).as_rotvec()
    return [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]]


# ============================================================================
# Part 1 — Hardware tests
# ============================================================================

def test_rtde_receive(ip: str) -> tuple[bool, list[str]]:
    """Check RTDE receive interface + read joint positions."""
    msgs: list[str] = []
    try:
        import rtde_receive
    except ImportError:
        msgs.append("[FAIL] ur_rtde (rtde_receive) not installed")
        return False, msgs

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        rc = s.connect_ex((ip, 30003))
        s.close()
        if rc != 0:
            msgs.append(f"[FAIL] TCP port 30003 unreachable at {ip}")
            return False, msgs
    except Exception as e:
        msgs.append(f"[FAIL] Socket error: {e}")
        return False, msgs

    try:
        r = rtde_receive.RTDEReceiveInterface(ip)
        if not r.isConnected():
            msgs.append("[FAIL] RTDE receive interface failed to connect")
            return False, msgs
        q = r.getActualQ()
        msgs.append(
            f"[PASS] RTDE receive connected — joints: "
            f"[{', '.join(f'{v:.2f}' for v in q)}]"
        )
        r.disconnect()
        return True, msgs
    except Exception as e:
        msgs.append(f"[FAIL] RTDE receive exception: {e}")
        return False, msgs


def test_rtde_control(ip: str) -> tuple[bool, list[str]]:
    """Check that the RTDE control interface can connect (needed for servoJ/servoL)."""
    msgs: list[str] = []
    try:
        import rtde_control
    except ImportError:
        msgs.append("[FAIL] ur_rtde (rtde_control) not installed")
        return False, msgs

    try:
        c = rtde_control.RTDEControlInterface(ip)
        if not c.isConnected():
            msgs.append("[FAIL] RTDE control interface could not connect")
            return False, msgs
        msgs.append("[PASS] RTDE control interface connected")
        c.disconnect()
        return True, msgs
    except Exception as e:
        msgs.append(f"[FAIL] RTDE control exception: {e}")
        msgs.append(
            "       Common causes: robot not in Remote Control mode, "
            "protective stop active, or another RTDE client connected"
        )
        return False, msgs


def test_robot_mode(ip: str) -> tuple[bool, list[str]]:
    """Verify robot is in RUNNING mode (mode 7) for remote control."""
    msgs: list[str] = []
    try:
        import rtde_receive
        r = rtde_receive.RTDEReceiveInterface(ip)
    except Exception as e:
        msgs.append(f"[FAIL] Cannot connect to read robot mode: {e}")
        return False, msgs

    try:
        mode = r.getRobotMode()
        # ur_rtde robot modes:
        # 7 = RUNNING, 5 = IDLE, 3 = POWER_OFF, etc.
        MODE_NAMES = {
            -1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY",
            2: "BOOTING", 3: "POWER_OFF", 4: "POWER_ON",
            5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING",
        }
        name = MODE_NAMES.get(mode, f"UNKNOWN({mode})")

        if mode == 7:
            msgs.append(f"[PASS] Robot mode: {name}")
        else:
            msgs.append(f"[FAIL] Robot mode: {name} — expected RUNNING (7)")
            msgs.append("       Switch the teach pendant to Remote Control mode")

        safety = r.getSafetyMode()
        SAFETY_NAMES = {
            1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP",
            4: "RECOVERY", 5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP",
            7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT",
        }
        sname = SAFETY_NAMES.get(safety, f"UNKNOWN({safety})")

        if safety == 1:
            msgs.append(f"[PASS] Safety mode: {sname}")
        else:
            msgs.append(f"[FAIL] Safety mode: {sname} — expected NORMAL (1)")

        r.disconnect()
        return mode == 7 and safety == 1, msgs
    except Exception as e:
        msgs.append(f"[FAIL] Error reading robot mode: {e}")
        try:
            r.disconnect()
        except Exception:
            pass
        return False, msgs


# ============================================================================
# Part 2 — Pure-math transform tests
# ============================================================================

def _assert_close(a, b, tol=1e-9, label=""):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    max_err = np.max(np.abs(a - b))
    ok = max_err < tol
    tag = "[PASS]" if ok else "[FAIL]"
    return ok, f"{tag} {label} (max err {max_err:.2e})"


def test_identity_roundtrip() -> tuple[bool, list[str]]:
    """pose_vector(0,0,0,0,0,0) → matrix → pose_vector should be identity."""
    msgs: list[str] = []
    zero = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mat = pose_vector_to_matrix(zero)
    ok1, m1 = _assert_close(mat, np.eye(4), label="zero vec → identity matrix")
    msgs.append(m1)

    back = matrix_to_pose_vector(mat)
    ok2, m2 = _assert_close(back, zero, label="identity matrix → zero vec")
    msgs.append(m2)
    return ok1 and ok2, msgs


def test_pure_translation() -> tuple[bool, list[str]]:
    """Translation-only pose should survive roundtrip exactly."""
    msgs: list[str] = []
    vec = [1.0, -2.5, 0.3, 0.0, 0.0, 0.0]
    mat = pose_vector_to_matrix(vec)

    ok1, m1 = _assert_close(mat[:3, 3], vec[:3], label="translation preserved in matrix")
    msgs.append(m1)
    ok2, m2 = _assert_close(mat[:3, :3], np.eye(3), label="rotation block is identity")
    msgs.append(m2)

    back = matrix_to_pose_vector(mat)
    ok3, m3 = _assert_close(back, vec, label="roundtrip vec→mat→vec")
    msgs.append(m3)
    return ok1 and ok2 and ok3, msgs


def test_known_rotation() -> tuple[bool, list[str]]:
    """90° rotation about Z axis should map X→Y, Y→-X."""
    msgs: list[str] = []
    angle = np.pi / 2
    vec = [0.0, 0.0, 0.0, 0.0, 0.0, angle]
    mat = pose_vector_to_matrix(vec)

    expected_rot = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ], dtype=np.float64)

    ok1, m1 = _assert_close(mat[:3, :3], expected_rot, tol=1e-9,
                             label="90° Z rotation matrix")
    msgs.append(m1)

    back = matrix_to_pose_vector(mat)
    ok2, m2 = _assert_close(back, vec, tol=1e-9, label="roundtrip 90° Z")
    msgs.append(m2)
    return ok1 and ok2, msgs


def test_random_roundtrips(n: int = 50) -> tuple[bool, list[str]]:
    """Random pose vectors should survive vec→mat→vec→mat roundtrip.

    Rotation vectors near ±π have a sign ambiguity, so we compare the
    resulting 4x4 matrices rather than raw rotation vectors.
    """
    msgs: list[str] = []
    rng = np.random.default_rng(42)
    worst = 0.0

    for _ in range(n):
        pos = rng.uniform(-5, 5, size=3)
        rotvec = rng.uniform(-np.pi, np.pi, size=3)
        vec = list(pos) + list(rotvec)

        mat1 = pose_vector_to_matrix(vec)
        back = matrix_to_pose_vector(mat1)
        mat2 = pose_vector_to_matrix(back)
        err = np.max(np.abs(mat1 - mat2))
        worst = max(worst, err)

    ok = worst < 1e-12
    tag = "[PASS]" if ok else "[FAIL]"
    msgs.append(f"{tag} {n} random roundtrips via matrix comparison (worst err {worst:.2e})")
    return ok, msgs


def test_t265_align_orthogonality() -> tuple[bool, list[str]]:
    """T265_TO_UR_ALIGN must be a proper rotation (det=+1, orthogonal)."""
    msgs: list[str] = []
    rot = T265_TO_UR_ALIGN[:3, :3].astype(np.float64)

    det = np.linalg.det(rot)
    ok1, m1 = _assert_close([det], [1.0], tol=1e-12,
                             label="T265_TO_UR_ALIGN det = +1")
    msgs.append(m1)

    product = rot @ rot.T
    ok2, m2 = _assert_close(product, np.eye(3), tol=1e-12,
                             label="T265_TO_UR_ALIGN orthogonality (R @ R^T = I)")
    msgs.append(m2)
    return ok1 and ok2, msgs


def test_t265_align_axis_mapping() -> tuple[bool, list[str]]:
    """
    Verify the intended camera→UR axis mapping:
      Camera -Z (forward) → UR +X
      Camera -X (left)    → UR +Y
      Camera +Y (up)      → UR +Z
    """
    msgs: list[str] = []
    rot = T265_TO_UR_ALIGN[:3, :3].astype(np.float64)
    all_ok = True

    cam_fwd = np.array([0, 0, -1.0])
    ur_fwd = rot @ cam_fwd
    ok, m = _assert_close(ur_fwd, [1, 0, 0], label="cam -Z → UR +X")
    msgs.append(m)
    all_ok &= ok

    cam_left = np.array([-1, 0, 0.0])
    ur_left = rot @ cam_left
    ok, m = _assert_close(ur_left, [0, 1, 0], label="cam -X → UR +Y")
    msgs.append(m)
    all_ok &= ok

    cam_up = np.array([0, 1, 0.0])
    ur_up = rot @ cam_up
    ok, m = _assert_close(ur_up, [0, 0, 1], label="cam +Y → UR +Z")
    msgs.append(m)
    all_ok &= ok

    return all_ok, msgs


def test_matrix_inverse_consistency() -> tuple[bool, list[str]]:
    """Composing a pose with its inverse should yield identity."""
    msgs: list[str] = []
    vec = [0.3, -0.7, 1.2, 0.5, -0.3, 0.8]
    mat = pose_vector_to_matrix(vec)
    inv_mat = np.linalg.inv(mat)
    product = mat @ inv_mat
    ok, m = _assert_close(product, np.eye(4), tol=1e-12,
                          label="T @ T_inv = I")
    msgs.append(m)
    return ok, msgs


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="UR5 readiness & transform tests")
    parser.add_argument("--skip-hardware", action="store_true",
                        help="Only run pure-math transform tests (no UR5 needed)")
    args = parser.parse_args()

    print("=" * 60)
    print("  UR5 Readiness & Coordinate Transform Tests")
    print("=" * 60)

    all_pass = True

    # --- Part 2: transforms (always run) ---
    print("\n[Transforms]")
    print("-" * 40)
    transform_tests = [
        ("Identity roundtrip", test_identity_roundtrip),
        ("Pure translation", test_pure_translation),
        ("Known 90° Z rotation", test_known_rotation),
        ("Random roundtrips (50)", test_random_roundtrips),
        ("T265_TO_UR_ALIGN orthogonality", test_t265_align_orthogonality),
        ("T265_TO_UR_ALIGN axis mapping", test_t265_align_axis_mapping),
        ("Matrix inverse consistency", test_matrix_inverse_consistency),
    ]
    for name, fn in transform_tests:
        print(f"\n[-] {name}")
        passed, msgs = fn()
        for m in msgs:
            print(f"    {m}")
        if not passed:
            all_pass = False

    # --- Part 1: hardware ---
    if not args.skip_hardware:
        print("\n\n[UR5 Hardware]")
        print("-" * 40)

        ip = get_hardware_config().get("ur_arm", {}).get("ip", "192.168.1.5")

        hw_tests = [
            ("RTDE receive interface", lambda: test_rtde_receive(ip)),
            ("RTDE control interface", lambda: test_rtde_control(ip)),
            ("Robot mode & safety", lambda: test_robot_mode(ip)),
        ]
        for name, fn in hw_tests:
            print(f"\n[-] {name}")
            passed, msgs = fn()
            for m in msgs:
                print(f"    {m}")
            if not passed:
                all_pass = False
    else:
        print("\n\n[UR5 Hardware] skipped (--skip-hardware)")

    print("\n" + "=" * 60)
    if all_pass:
        print("  \033[92mALL TESTS PASSED\033[0m")
    else:
        print("  \033[91mSOME TESTS FAILED\033[0m")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
