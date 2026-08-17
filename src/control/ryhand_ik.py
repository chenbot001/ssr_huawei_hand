from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from config import PACKAGE_ROOT, PROJECT_ROOT


CALIBRATION_PATH = PROJECT_ROOT / "configs" / "manus_calibration.json"
URDF_PATH = PACKAGE_ROOT / "assets" / "ruihand15z" / "urdf" / "ruihand15z.urdf"


def load_calibration(path: Path = CALIBRATION_PATH) -> tuple[np.ndarray, np.ndarray]:
    scales = np.ones(5, dtype=np.float64)
    offsets = np.zeros((5, 3), dtype=np.float64)
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        scales = np.asarray(data.get("FINGER_SCALES", scales), dtype=np.float64)
        offsets = np.asarray(
            data.get("FINGER_POS_OFFSETS", offsets), dtype=np.float64
        )
    if scales.shape != (5,):
        raise ValueError(f"FINGER_SCALES must contain 5 values, got {scales.shape}")
    if offsets.shape != (5, 3):
        raise ValueError(
            f"FINGER_POS_OFFSETS must have shape (5, 3), got {offsets.shape}"
        )
    return scales, offsets


def save_calibration(
    scales: Sequence[float],
    offsets: Sequence[Sequence[float]],
    path: Path = CALIBRATION_PATH,
) -> None:
    scale_array = np.asarray(scales, dtype=np.float64)
    offset_array = np.asarray(offsets, dtype=np.float64)
    if scale_array.shape != (5,) or offset_array.shape != (5, 3):
        raise ValueError("Calibration must contain 5 scales and a 5x3 offset matrix")

    wrist_offset = [0.0, 0.0, 0.0]
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        wrist_offset = existing.get("WRIST_OFFSET", wrist_offset)

    data = {
        "FINGER_SCALES": [round(float(value), 4) for value in scale_array],
        "WRIST_OFFSET": wrist_offset,
        "FINGER_POS_OFFSETS": [
            [round(float(value), 4) for value in row] for row in offset_array
        ],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=4)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def ik_to_hand_angles(ik_joints: Sequence[float]) -> np.ndarray:
    """Map 20 simulated joints to the validated thumb/index RYHand angles."""
    joints = np.asarray(ik_joints, dtype=np.float64)
    if joints.shape != (20,):
        raise ValueError(f"IK result must have shape (20,), got {joints.shape}")
    if not np.all(np.isfinite(joints)):
        raise ValueError("IK result must contain only finite values")

    angles = np.zeros(15, dtype=np.float64)
    proximal_limit = np.deg2rad(90.0)
    distal_limit = np.deg2rad(75.0)

    for finger in (0, 1):
        ik_base = finger * 4
        hand_base = finger * 3
        angles[hand_base] = np.deg2rad(10.0) if finger == 0 else 0.0
        proximal = float(np.clip(joints[ik_base + 1], 0.0, proximal_limit))
        angles[hand_base + 1] = proximal

        pip_angle = max(0.0, float(joints[ik_base + 2]))
        dip_angle = max(0.0, float(joints[ik_base + 3]))
        if finger == 0:
            effective = max(pip_angle, dip_angle, proximal * 0.5)
        else:
            effective = (pip_angle + dip_angle) * 0.5
        angles[hand_base + 2] = np.clip(
            effective * (75.0 / 90.0), 0.0, distal_limit
        )
    return angles


class RYHandIK:
    """PyBullet IK engine for Manus-to-RYHand retargeting."""

    def __init__(self, gui: bool = False, urdf_path: Path = URDF_PATH) -> None:
        if not urdf_path.is_file():
            raise FileNotFoundError(
                f"RYHand URDF is missing: {urdf_path}. "
                "Reinstall the package with its bundled RYHand assets."
            )
        try:
            import pybullet
        except ImportError as exc:
            raise ImportError("pybullet is required for RYHand IK") from exc

        self._p = pybullet
        self._gui = gui
        self._client = pybullet.connect(pybullet.GUI if gui else pybullet.DIRECT)
        self._closed = False
        pybullet.setGravity(0, 0, 0, physicsClientId=self._client)
        pybullet.setRealTimeSimulation(0, physicsClientId=self._client)
        if gui:
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_GUI, 0, physicsClientId=self._client
            )
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SHADOWS, 0, physicsClientId=self._client
            )
            pybullet.resetDebugVisualizerCamera(
                cameraDistance=0.4,
                cameraYaw=180,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.05],
                physicsClientId=self._client,
            )

        orientation = pybullet.getQuaternionFromEuler([0, 0, np.pi / 2.0])
        self._robot = pybullet.loadURDF(
            str(urdf_path),
            [0, 0, 0],
            orientation,
            useFixedBase=True,
            physicsClientId=self._client,
        )
        self._actuated_joints: list[int] = []
        link_indices: dict[str, int] = {}
        for joint_index in range(
            pybullet.getNumJoints(self._robot, physicsClientId=self._client)
        ):
            info = pybullet.getJointInfo(
                self._robot, joint_index, physicsClientId=self._client
            )
            link_indices[info[12].decode("utf-8")] = joint_index
            if info[2] == pybullet.JOINT_REVOLUTE:
                self._actuated_joints.append(joint_index)

        fingertip_names = (
            "fz15_Link",
            "fz25_Link",
            "fz35_Link",
            "fz45_Link",
            "fz55_Link",
        )
        try:
            self._end_effectors = [link_indices[name] for name in fingertip_names]
        except KeyError as exc:
            raise RuntimeError(f"RYHand URDF is missing expected link {exc.args[0]}") from exc
        self._thumb_dip = link_indices.get("fz14_Link")
        self.scales, self.offsets = load_calibration()
        self._joint_positions = np.zeros(20, dtype=np.float64)

    @property
    def physics_client(self) -> int:
        return self._client

    def set_calibration(
        self, scales: Sequence[float], offsets: Sequence[Sequence[float]]
    ) -> None:
        scale_array = np.asarray(scales, dtype=np.float64)
        offset_array = np.asarray(offsets, dtype=np.float64)
        if scale_array.shape != (5,) or offset_array.shape != (5, 3):
            raise ValueError("Calibration must contain 5 scales and a 5x3 offset matrix")
        self.scales = scale_array.copy()
        self.offsets = offset_array.copy()

    def compute_ik(self, fingers: Sequence[Sequence[float]]) -> np.ndarray | None:
        positions = np.asarray(fingers, dtype=np.float64)
        if positions.shape != (10, 3) or not np.all(np.isfinite(positions)):
            return None

        calibrated = positions.copy()
        for index in range(10):
            calibrated[index] += self.offsets[index // 2]

        tip_indices = (1, 3, 5, 7, 9)
        target_positions = [
            (calibrated[index] * self.scales[finger]).tolist()
            for finger, index in enumerate(tip_indices)
        ]
        end_effectors = list(self._end_effectors)
        if self._thumb_dip is not None:
            end_effectors.append(self._thumb_dip)
            target_positions.append((calibrated[0] * self.scales[0]).tolist())

        self._p.stepSimulation(physicsClientId=self._client)
        try:
            solved = self._p.calculateInverseKinematics2(
                self._robot,
                end_effectors,
                target_positions,
                solver=self._p.IK_DLS,
                maxNumIterations=100,
                residualThreshold=0.001,
                physicsClientId=self._client,
            )
        except Exception:
            return None

        for index, joint_index in enumerate(self._actuated_joints):
            if index >= len(solved):
                break
            self._p.setJointMotorControl2(
                bodyIndex=self._robot,
                jointIndex=joint_index,
                controlMode=self._p.POSITION_CONTROL,
                targetPosition=solved[index],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
                physicsClientId=self._client,
            )
        self._joint_positions = np.asarray(solved[:20], dtype=np.float64)
        if self._joint_positions.shape != (20,):
            return None
        return self._joint_positions.copy()

    def compute_hand_angles(
        self, fingers: Sequence[Sequence[float]]
    ) -> np.ndarray | None:
        joints = self.compute_ik(fingers)
        return None if joints is None else ik_to_hand_angles(joints)

    def render_rgba(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Render the current simulated hand for the local calibration console."""
        width = int(width)
        height = int(height)
        if width < 1 or height < 1:
            raise ValueError("Render dimensions must be positive")
        view = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 0.06],
            distance=0.42,
            yaw=180,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
        )
        projection = self._p.computeProjectionMatrixFOV(
            fov=50,
            aspect=width / height,
            nearVal=0.01,
            farVal=2.0,
        )
        _width, _height, rgba, _depth, _segmentation = self._p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=self._p.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )
        return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4).copy()

    def close(self) -> None:
        if not self._closed:
            self._p.disconnect(self._client)
            self._closed = True

    def __enter__(self) -> "RYHandIK":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
