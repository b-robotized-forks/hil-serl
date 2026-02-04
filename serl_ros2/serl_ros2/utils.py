"""
ROS2 message construction utilities.

Provides functions to create ROS2 messages from Python/numpy types for the adapter.
These functions are used internally by the RobotAdapter to transform outgoing commands
into ROS2 message format.
"""

from typing import Sequence

import numpy as np
from geometry_msgs.msg import PoseStamped, WrenchStamped


def pose_to_msg(
    pose: Sequence[float] | np.ndarray | PoseStamped,
    frame_id: str = "base",
    timestamp: object | None = None,
) -> PoseStamped:
    """
    Convert a 7D pose to a PoseStamped message.

    Args:
        pose: 7D sequence [x, y, z, qx, qy, qz, qw], or an existing PoseStamped.
        frame_id: Frame ID to set in the header (ignored if pose is already PoseStamped).

    Returns:
        PoseStamped message with the pose and frame_id.

    Raises:
        ValueError: If pose is not a 7D sequence or PoseStamped.
    """
    if isinstance(pose, PoseStamped):
        if timestamp is not None:
            pose.header.stamp = timestamp
        return pose

    if not hasattr(pose, "__len__") or len(pose) != 7:
        raise ValueError("pose must be a 7D sequence [x, y, z, qx, qy, qz, qw] or PoseStamped")

    msg = PoseStamped()
    msg.header.frame_id = frame_id
    if timestamp is not None:
        msg.header.stamp = timestamp
    msg.pose.position.x = float(pose[0])
    msg.pose.position.y = float(pose[1])
    msg.pose.position.z = float(pose[2])
    msg.pose.orientation.x = float(pose[3])
    msg.pose.orientation.y = float(pose[4])
    msg.pose.orientation.z = float(pose[5])
    msg.pose.orientation.w = float(pose[6])
    return msg


def wrench_to_msg(wrench: Sequence[float] | np.ndarray | WrenchStamped) -> WrenchStamped:
    """
    Convert a 6D wrench to a WrenchStamped message.

    Args:
        wrench: 6D sequence [fx, fy, fz, tx, ty, tz], or an existing WrenchStamped.

    Returns:
        WrenchStamped message with the wrench values.

    Raises:
        ValueError: If wrench is not a 6D sequence or WrenchStamped.
    """
    if isinstance(wrench, WrenchStamped):
        return wrench

    if not hasattr(wrench, "__len__") or len(wrench) != 6:
        raise ValueError("wrench must be a 6D sequence [fx, fy, fz, tx, ty, tz] or WrenchStamped")

    msg = WrenchStamped()
    msg.wrench.force.x = float(wrench[0])
    msg.wrench.force.y = float(wrench[1])
    msg.wrench.force.z = float(wrench[2])
    msg.wrench.torque.x = float(wrench[3])
    msg.wrench.torque.y = float(wrench[4])
    msg.wrench.torque.z = float(wrench[5])
    return msg
