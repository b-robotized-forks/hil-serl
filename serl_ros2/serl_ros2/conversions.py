"""
ROS2 message conversion utilities.

Provides functions to convert ROS2 messages to Python/numpy types for the adapter.
These functions are used internally by the RobotAdapter to transform incoming messages
into the format expected by HIL-SERL environments.
"""

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose, PoseStamped, Twist, TwistStamped, Wrench, WrenchStamped
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32


def pose_msg_to_list(msg: Pose | PoseStamped) -> list[float]:
    """
    Convert a Pose or PoseStamped message to a 7D list.

    Args:
        msg: ROS2 Pose or PoseStamped message.

    Returns:
        List of [x, y, z, qx, qy, qz, qw].
    """
    pose = msg.pose if hasattr(msg, "pose") else msg
    return [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]


def twist_msg_to_list(msg: Twist | TwistStamped) -> list[float]:
    """
    Convert a Twist or TwistStamped message to a 6D list.

    Args:
        msg: ROS2 Twist or TwistStamped message.

    Returns:
        List of [vx, vy, vz, wx, wy, wz] (linear then angular velocity).
    """
    twist = msg.twist if hasattr(msg, "twist") else msg
    return [
        float(twist.linear.x),
        float(twist.linear.y),
        float(twist.linear.z),
        float(twist.angular.x),
        float(twist.angular.y),
        float(twist.angular.z),
    ]


def wrench_msg_to_list(msg: Wrench | WrenchStamped) -> list[float]:
    """
    Convert a Wrench or WrenchStamped message to a 6D list.

    Args:
        msg: ROS2 Wrench or WrenchStamped message.

    Returns:
        List of [fx, fy, fz, tx, ty, tz] (force then torque).
    """
    wrench = msg.wrench if hasattr(msg, "wrench") else msg
    return [
        float(wrench.force.x),
        float(wrench.force.y),
        float(wrench.force.z),
        float(wrench.torque.x),
        float(wrench.torque.y),
        float(wrench.torque.z),
    ]


def joint_state_to_lists(msg: JointState) -> tuple[list[float], list[float]]:
    """
    Convert a JointState message to position and velocity lists.

    Args:
        msg: ROS2 JointState message.

    Returns:
        Tuple of (positions, velocities) as lists of floats.
        Empty lists if the corresponding field is not populated.
    """
    positions = list(msg.position) if msg.position else []
    velocities = list(msg.velocity) if msg.velocity else []
    return positions, velocities


def float_msg_to_float(msg: Float32) -> float:
    """
    Convert a std_msgs/Float32 message to a Python float.

    Args:
        msg: ROS2 Float32 message.

    Returns:
        The data value as a float.
    """
    return float(msg.data)


def stamp_to_seconds(stamp: Time | None) -> float | None:
    """
    Convert a ROS2 Time stamp to seconds as a float.

    Args:
        stamp: ROS2 builtin_interfaces/Time message, or None.

    Returns:
        Time in seconds (float), or None if input is None.
    """
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def image_msg_to_numpy(msg: Image) -> np.ndarray:
    """
    Convert a sensor_msgs/Image message to a numpy array.

    Handles common encodings: rgb8, bgr8, rgba8, bgra8, mono8, mono16.
    For bgr encodings, converts to RGB for consistency with HIL-SERL.

    Args:
        msg: ROS2 Image message.

    Returns:
        Numpy array with shape (height, width, channels) for color images,
        or (height, width) for mono images.

    Raises:
        ValueError: If the encoding is not supported.
    """
    encoding = msg.encoding.lower()
    height = msg.height
    width = msg.width

    # Determine dtype and channels based on encoding
    if encoding in ("rgb8", "bgr8"):
        dtype = np.uint8
        channels = 3
    elif encoding in ("rgba8", "bgra8"):
        dtype = np.uint8
        channels = 4
    elif encoding == "mono8":
        dtype = np.uint8
        channels = 1
    elif encoding == "mono16":
        dtype = np.uint16
        channels = 1
    elif encoding in ("32fc1", "32fc3", "32fc4"):
        dtype = np.float32
        channels = int(encoding[-1])
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    # Reshape data
    if channels == 1:
        img = np.frombuffer(msg.data, dtype=dtype).reshape(height, width)
    else:
        img = np.frombuffer(msg.data, dtype=dtype).reshape(height, width, channels)

    # Convert BGR to RGB for consistency
    if encoding == "bgr8":
        img = img[:, :, ::-1].copy()
    elif encoding == "bgra8":
        img = img[:, :, [2, 1, 0, 3]].copy()

    return img
