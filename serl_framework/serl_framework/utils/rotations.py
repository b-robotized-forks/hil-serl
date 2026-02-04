"""
Rotation helpers for RobotEnv.

Origin: hil-serl/serl_robot_infra/franka_env/utils/rotations.py
Modified: Yes
Changes:
  - Moved into serl_framework for ROS2-free reuse.
  - Replaced Franka-specific euler_2_quat with standard euler2quat.
  - Uses scipy conventions throughout (xyzw quaternions, xyz euler order).
"""

import numpy as np
from scipy.spatial.transform import Rotation


def quat2euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to XYZ Euler angles.

    Args:
        quat: Quaternion as [x, y, z, w] (scalar-last, scipy convention).

    Returns:
        Euler angles in radians as [roll, pitch, yaw] (XYZ convention).
    """
    return Rotation.from_quat(quat).as_euler("xyz")


def euler2quat(rpy: np.ndarray) -> np.ndarray:
    """
    Convert XYZ Euler angles to a quaternion.

    Args:
        rpy: Euler angles in radians as [roll, pitch, yaw] (XYZ convention).

    Returns:
        Quaternion as [x, y, z, w] (scalar-last, scipy/ROS2 convention).
    """
    return Rotation.from_euler("xyz", rpy).as_quat()


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion to unit length.

    Args:
        quat: Quaternion as [x, y, z, w].

    Returns:
        Normalized quaternion as [x, y, z, w].
    """
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (x, y, z, w).

    Args:
        q1: Left quaternion as [x, y, z, w].
        q2: Right quaternion as [x, y, z, w].

    Returns:
        Product quaternion as [x, y, z, w].
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """
    Convert a rotation vector to a quaternion (x, y, z, w).

    Args:
        rotvec: Rotation vector [rx, ry, rz].

    Returns:
        Quaternion as [x, y, z, w].
    """
    vec = np.array(rotvec, dtype=np.float32)
    angle = float(np.linalg.norm(vec))
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = vec / angle
    half = angle * 0.5
    sin_half = np.sin(half)
    return np.array(
        [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, np.cos(half)],
        dtype=np.float32,
    )


def quat_to_angvel(q_prev: np.ndarray, q_curr: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute angular velocity from consecutive quaternions.

    Args:
        q_prev: Previous quaternion as [x, y, z, w].
        q_curr: Current quaternion as [x, y, z, w].
        dt: Time delta in seconds.

    Returns:
        Angular velocity vector (wx, wy, wz).
    """
    if dt <= 0:
        return np.zeros((3,), dtype=np.float32)
    q_prev = normalize_quat(q_prev)
    q_curr = normalize_quat(q_curr)
    q_inv = np.array([-q_prev[0], -q_prev[1], -q_prev[2], q_prev[3]], dtype=np.float32)
    q_delta = quat_multiply(q_curr, q_inv)
    angle = 2.0 * np.arccos(np.clip(q_delta[3], -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros((3,), dtype=np.float32)
    axis = q_delta[:3] / np.sin(angle / 2.0)
    return axis * (angle / dt)
