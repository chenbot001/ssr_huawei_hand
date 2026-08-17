"""
Centralized Ruiyan Hand IK: PyBullet-based inverse kinematics and
20-joint IK → 15-motor retargeting for the RYHand dexterous hand.

All scripts that teleoperate the hand via Manus glove (collect_data,
test_teleop_manus, test_ryhand_teleop) should import from this module to
ensure a single, consistent mapping.

Usage:
    from ssr.control.RyHand_IK import RYHandIK, ik_to_hand_angles

    ik_engine = RYHandIK(gui=False)
    ...
    hand_angles = ik_engine.compute_hand_angles(glove_data)  # 15-element array
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pybullet as p

from ..config import PROJECT_ROOT

# ============================================================================
# Calibration — loaded once from configs/manus_calibration.json
# ============================================================================
_CALIBRATION_FILE = PROJECT_ROOT / "configs" / "manus_calibration.json"

FINGER_SCALES: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0]
FINGER_POS_OFFSETS: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(5)]
WRIST_OFFSET: list[float] = [0.0, 0.0, 0.0]

if _CALIBRATION_FILE.exists():
    try:
        with open(_CALIBRATION_FILE, "r") as _f:
            _calib = json.load(_f)
            FINGER_SCALES = _calib.get("FINGER_SCALES", FINGER_SCALES)
            WRIST_OFFSET = _calib.get("WRIST_OFFSET", WRIST_OFFSET)
            FINGER_POS_OFFSETS = _calib.get("FINGER_POS_OFFSETS", FINGER_POS_OFFSETS)
    except Exception as _e:
        print(f"[RyHand_IK] 加载校准文件失败: {_e}, 使用默认参数")


# ============================================================================
# ik_to_hand_angles — 20-joint IK → 15-motor retarget
# ============================================================================
def ik_to_hand_angles(ik_joints: np.ndarray) -> np.ndarray:
    """
    Map 20 IK simulation joints to 15 physical motor angles (radians).

    Per-finger IK layout (4 revolute joints each, 20 total):
        fzX1 (side swing)   fzX2 (MCP bend)   fzX3 (PIP bend)   fzX4 (DIP bend)

    Per-finger motor layout (3 channels each, 15 total):
        side_swing [-30°, +30°]   proximal_bend [0°, 90°]   distal_bend [0°, 75°]

    Mapping rules:
        fzX1          → side_swing  (only thumb retains this DOF; others frozen to 0)
        fzX2          → proximal_bend
        fzX3 + fzX4   → distal_bend (combined then scaled by 75/90)

    Thumb special handling:
        The distal channel uses max(PIP, DIP) instead of the average, with an
        MCP-coupled floor of proximal * 0.5 to guarantee curl when IK under-
        estimates PIP/DIP.
    """
    hand_angles = np.zeros(15, dtype=np.float64)

    limit_side = np.deg2rad(30)
    limit_prox = np.deg2rad(90)
    limit_dist = np.deg2rad(75)


    # 0: Thumb, 1: Index, 2: Middle, 3: Ring, 4: Pinky
    for finger in range(5):
        ik_base = finger * 4
        hand_base = finger * 3

        # Disable teleop for ring (3) and pinky (4)
        if finger in [2, 3, 4]:
            hand_angles[hand_base] = 0.0
            hand_angles[hand_base + 1] = 0.0
            hand_angles[hand_base + 2] = 0.0
            continue

        # Side swing — disabled for all fingers, thumb hardcoded to +10 degrees
        # Middle finger (2): swing disabled, bend enabled
        if finger == 0:
            hand_angles[hand_base] = np.deg2rad(10)
        else:
            hand_angles[hand_base] = 0.0

        # Proximal bend (MCP)
        proximal = ik_joints[ik_base + 1]
        hand_angles[hand_base + 1] = np.clip(proximal, 0, limit_prox)

        # Distal bend (PIP + DIP coupled into one motor)
        pip_angle = ik_joints[ik_base + 2]
        dip_angle = ik_joints[ik_base + 3]

        if finger == 0:
            # Thumb: take the larger of PIP / DIP (the IK has an extra DIP
            # target constraint so both values are meaningful); fall back to
            # MCP-coupled estimate when IK occasionally under-solves.
            ik_distal = max(max(0, pip_angle), max(0, dip_angle))
            coupled_distal = max(0, proximal) * 0.5
            effective_distal = max(ik_distal, coupled_distal)
            scaled_distal = effective_distal * (75.0 / 90.0)
        else:
            combined_distal = (pip_angle + dip_angle) * 0.5
            scaled_distal = combined_distal * (75.0 / 90.0)

        hand_angles[hand_base + 2] = np.clip(scaled_distal, 0, limit_dist)

    return hand_angles


# ============================================================================
# RYHandIK — PyBullet IK engine
# ============================================================================
class RYHandIK:
    """
    PyBullet-based inverse-kinematics engine for the Ruiyan left hand.

    Supports both headless (``gui=False``, for collect_data / test_teleop_manus)
    and interactive (``gui=True``, for test_ryhand_teleop) modes.
    """

    _URDF_REL = os.path.join(
        "external", "Bidex_Manus_Teleop", "ryhand_left", "ruihand15z.urdf"
    )

    def __init__(self, gui: bool = False):
        if gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.4,
                cameraYaw=180,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.05],
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        self._gui = gui
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)

        urdf_path = os.path.join(str(PROJECT_ROOT), self._URDF_REL)
        base_orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        self.robot_id = p.loadURDF(
            urdf_path, [0, 0, 0], base_orn, useFixedBase=True
        )

        # Build joint / link index maps
        self.actuated_joints: list[int] = []
        self.link_name_to_idx: dict[str, int] = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            self.link_name_to_idx[info[12].decode("utf-8")] = i
            if info[2] == p.JOINT_REVOLUTE:
                self.actuated_joints.append(i)

        fingertip_links = [
            "fz15_Link", "fz25_Link", "fz35_Link",
            "fz45_Link", "fz55_Link",
        ]
        self.ee_indices = [
            self.link_name_to_idx[n]
            for n in fingertip_links
            if n in self.link_name_to_idx
        ]

        # Thumb DIP link for extra IK constraint
        self.thumb_dip_idx = self.link_name_to_idx.get("fz14_Link")

        self.joint_positions = np.zeros(20)

        # GUI-mode visualization helpers
        self._target_balls: list[int] = []
        if self._gui:
            self._create_target_vis()

    # ------------------------------------------------------------------
    # GUI visualisation helpers
    # ------------------------------------------------------------------
    def _create_target_vis(self) -> None:
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.005)
        colors = [
            [1, 1, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1],
            [0, 0, 1, 1], [1, 0, 1, 1],
        ]
        for i in range(5):
            for j in range(2):
                bid = p.createMultiBody(
                    baseMass=0.001,
                    baseCollisionShapeIndex=ball_shape,
                    basePosition=[0.1, 0.1, 0.1],
                )
                p.setCollisionFilterGroupMask(bid, -1, 0, 0)
                c = colors[i].copy()
                c[3] = 0.6 if j == 0 else 1.0
                p.changeVisualShape(bid, -1, rgbaColor=c)
                self._target_balls.append(bid)

    def _update_target_vis(self, hand_pos: list) -> None:
        for i, pos in enumerate(hand_pos):
            if i < len(self._target_balls):
                _, orn = p.getBasePositionAndOrientation(self._target_balls[i])
                p.resetBasePositionAndOrientation(self._target_balls[i], pos, orn)

    # ------------------------------------------------------------------
    # Core IK
    # ------------------------------------------------------------------
    def compute_ik(self, glove_data: dict | None) -> np.ndarray | None:
        """
        Run one frame of inverse kinematics from Manus glove skeleton data.

        Args:
            glove_data: ``{"fingers": [[x,y,z], ...], "wrist": [x,y,z]}``
                        where ``fingers`` has 10 entries (DIP + Tip per finger).

        Returns:
            20-element float32 array of IK joint angles (radians), or None.
        """
        if glove_data is None or "fingers" not in glove_data:
            return None

        short_skeleton = glove_data["fingers"]
        if short_skeleton is None or len(short_skeleton) < 10:
            return None

        # Apply per-finger calibration offsets
        hand_pos = []
        for i, pos in enumerate(short_skeleton):
            finger_idx = i // 2
            off = FINGER_POS_OFFSETS[finger_idx]
            hand_pos.append([pos[0] + off[0], pos[1] + off[1], pos[2] + off[2]])

        if self._gui:
            self._update_target_vis(hand_pos)

        # Build IK target list: 5 fingertips + thumb DIP (extra constraint)
        tip_indices = [1, 3, 5, 7, 9]
        ee_list: list[int] = list(self.ee_indices)
        target_list: list[list[float]] = []

        for i, tip_idx in enumerate(tip_indices):
            pos = hand_pos[tip_idx]
            s = FINGER_SCALES[i]
            target_list.append([pos[0] * s, pos[1] * s, pos[2] * s])

        if self.thumb_dip_idx is not None:
            thumb_dip_pos = hand_pos[0]  # SHORT_IDX[0] = Thumb DIP
            s = FINGER_SCALES[0]
            ee_list.append(self.thumb_dip_idx)
            target_list.append([
                thumb_dip_pos[0] * s,
                thumb_dip_pos[1] * s,
                thumb_dip_pos[2] * s,
            ])

        p.stepSimulation()

        try:
            joint_poses = p.calculateInverseKinematics2(
                self.robot_id,
                ee_list,
                target_list,
                solver=p.IK_DLS,
                maxNumIterations=100,
                residualThreshold=0.001,
            )
            for i, jidx in enumerate(self.actuated_joints):
                if i < len(joint_poses):
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=jidx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_poses[i],
                        targetVelocity=0,
                        force=500,
                        positionGain=0.3,
                        velocityGain=1,
                    )
            self.joint_positions = np.array(joint_poses[:20], dtype=np.float32)
            return self.joint_positions
        except Exception as e:
            print(f"[RYHandIK] IK error: {e}")
            return None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def compute_hand_angles(self, glove_data: dict | None) -> np.ndarray | None:
        """
        End-to-end: glove skeleton → 15-element motor angle array (radians).

        Chains :meth:`compute_ik` and :func:`ik_to_hand_angles`.  Returns
        ``None`` when the glove data is unavailable or IK fails.
        """
        ik = self.compute_ik(glove_data)
        if ik is None:
            return None
        return ik_to_hand_angles(ik)

    def get_joint_positions(self) -> np.ndarray:
        """Return a copy of the most recent 20-joint IK solution."""
        return self.joint_positions.copy()

    def close(self) -> None:
        p.disconnect(self.physics_client)
