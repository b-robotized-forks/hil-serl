"""
Teleop helpers for applying translation/rotation deltas to poses.

This module centralizes pose update logic so teleop tools can apply deltas
consistently in either the base frame or the TCP (end-effector) frame.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from serl_framework.utils.rotations import normalize_quat, quat_multiply


def apply_pose_delta(
    pose: np.ndarray,
    delta_xyz: np.ndarray,
    delta_rotvec: np.ndarray,
    delta_in_base: bool = True,
) -> np.ndarray:
    """
    Apply a translation + rotation delta to a pose.

    Args:
        pose: Current pose as [x, y, z, qx, qy, qz, qw].
        delta_xyz: Translation delta (meters).
        delta_rotvec: Rotation-vector delta (radians).
        delta_in_base: True when deltas are expressed in base/world frame,
            False when deltas are expressed in TCP/tool frame.

    Returns:
        Updated pose as [x, y, z, qx, qy, qz, qw].
    """
    pose = np.array(pose, dtype=np.float32)
    delta_xyz = np.array(delta_xyz, dtype=np.float32)
    delta_rotvec = np.array(delta_rotvec, dtype=np.float32)

    if not delta_in_base:
        # Map local translation to world frame using current orientation.
        delta_xyz = Rotation.from_quat(pose[3:]).apply(delta_xyz)
        # Apply rotation in the local frame (post-multiply).
        delta_q = Rotation.from_rotvec(delta_rotvec).as_quat()
        pose[3:] = normalize_quat(quat_multiply(pose[3:], delta_q))
    else:
        # Apply rotation in the base/world frame (pre-multiply).
        delta_q = Rotation.from_rotvec(delta_rotvec).as_quat()
        pose[3:] = normalize_quat(quat_multiply(delta_q, pose[3:]))

    pose[:3] = pose[:3] + delta_xyz
    return pose


def transform_delta_to_base(
    delta_xyz: np.ndarray,
    delta_rotvec: np.ndarray,
    tcp_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform TCP-frame teleop deltas into the base frame.

    Args:
        delta_xyz: Translation delta in meters.
        delta_rotvec: Rotation-vector delta in radians.
        tcp_quat: Current TCP orientation as [qx, qy, qz, qw].

    Returns:
        Tuple of (delta_xyz_base, delta_rotvec_base).
    """
    delta_xyz = np.array(delta_xyz, dtype=np.float32)
    delta_rotvec = np.array(delta_rotvec, dtype=np.float32)
    tcp_quat = np.array(tcp_quat, dtype=np.float32)

    tcp_rot = Rotation.from_quat(tcp_quat)
    delta_xyz_base = tcp_rot.apply(delta_xyz)
    delta_rot = Rotation.from_rotvec(delta_rotvec)
    delta_rot_base = tcp_rot * delta_rot * tcp_rot.inv()
    return delta_xyz_base, delta_rot_base.as_rotvec()


def transform_translation_to_tip(
    delta_xyz: np.ndarray,
    tcp_quat: np.ndarray,
) -> np.ndarray:
    """
    Transform a base-frame translation vector into the TCP/tool frame.

    Args:
        delta_xyz: Translation vector in base/world coordinates.
        tcp_quat: Current TCP orientation as [qx, qy, qz, qw].

    Returns:
        Translation vector expressed in the TCP/tool frame.
    """
    delta_xyz = np.array(delta_xyz, dtype=np.float32)
    tcp_quat = np.array(tcp_quat, dtype=np.float32)

    tcp_rot = Rotation.from_quat(tcp_quat)
    return tcp_rot.inv().apply(delta_xyz)
