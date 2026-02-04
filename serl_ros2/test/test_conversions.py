"""Unit tests for the serl_ros2 conversions module."""

from __future__ import annotations

import numpy as np
import pytest

from serl_ros2.conversions import (
    image_msg_to_numpy,
    pose_msg_to_list,
    stamp_to_seconds,
    twist_msg_to_list,
    wrench_msg_to_list,
)


class MockPose:
    """Mock geometry_msgs/Pose."""

    def __init__(self):
        self.position = MockPoint(1.0, 2.0, 3.0)
        self.orientation = MockQuaternion(0.0, 0.0, 0.0, 1.0)


class MockPoseStamped:
    """Mock geometry_msgs/PoseStamped."""

    def __init__(self):
        self.pose = MockPose()


class MockPoint:
    """Mock geometry_msgs/Point."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class MockQuaternion:
    """Mock geometry_msgs/Quaternion."""

    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class MockTwist:
    """Mock geometry_msgs/Twist."""

    def __init__(self):
        self.linear = MockVector3(1.0, 2.0, 3.0)
        self.angular = MockVector3(0.1, 0.2, 0.3)


class MockTwistStamped:
    """Mock geometry_msgs/TwistStamped."""

    def __init__(self):
        self.twist = MockTwist()


class MockVector3:
    """Mock geometry_msgs/Vector3."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class MockWrench:
    """Mock geometry_msgs/Wrench."""

    def __init__(self):
        self.force = MockVector3(10.0, 20.0, 30.0)
        self.torque = MockVector3(0.5, 0.6, 0.7)


class MockWrenchStamped:
    """Mock geometry_msgs/WrenchStamped."""

    def __init__(self):
        self.wrench = MockWrench()


class MockTime:
    """Mock builtin_interfaces/Time."""

    def __init__(self, sec, nanosec):
        self.sec = sec
        self.nanosec = nanosec


class MockImage:
    """Mock sensor_msgs/Image."""

    def __init__(self, encoding, height, width, data):
        self.encoding = encoding
        self.height = height
        self.width = width
        self.data = data


def test_pose_msg_to_list_from_pose():
    """Convert Pose to list."""
    pose = MockPose()
    result = pose_msg_to_list(pose)
    assert result == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]


def test_pose_msg_to_list_from_pose_stamped():
    """Convert PoseStamped to list."""
    pose = MockPoseStamped()
    result = pose_msg_to_list(pose)
    assert result == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]


def test_twist_msg_to_list():
    """Convert TwistStamped to list."""
    twist = MockTwistStamped()
    result = twist_msg_to_list(twist)
    assert result == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]


def test_wrench_msg_to_list():
    """Convert WrenchStamped to list."""
    wrench = MockWrenchStamped()
    result = wrench_msg_to_list(wrench)
    assert result == [10.0, 20.0, 30.0, 0.5, 0.6, 0.7]


def test_stamp_to_seconds():
    """Convert Time to seconds."""
    stamp = MockTime(sec=123, nanosec=500_000_000)
    result = stamp_to_seconds(stamp)
    assert result == 123.5


def test_stamp_to_seconds_none():
    """Handle None stamp."""
    result = stamp_to_seconds(None)
    assert result is None


def test_image_msg_to_numpy_rgb8():
    """Convert rgb8 image to numpy."""
    height, width = 2, 3
    data = bytes(range(height * width * 3))
    msg = MockImage("rgb8", height, width, data)

    result = image_msg_to_numpy(msg)

    assert result.shape == (2, 3, 3)
    assert result.dtype == np.uint8


def test_image_msg_to_numpy_bgr8_converts_to_rgb():
    """Convert bgr8 image to numpy and verify BGR->RGB conversion."""
    height, width = 1, 1
    # BGR pixel: Blue=1, Green=2, Red=3
    data = bytes([1, 2, 3])
    msg = MockImage("bgr8", height, width, data)

    result = image_msg_to_numpy(msg)

    # Should be converted to RGB: Red=3, Green=2, Blue=1
    assert result[0, 0, 0] == 3  # R
    assert result[0, 0, 1] == 2  # G
    assert result[0, 0, 2] == 1  # B


def test_image_msg_to_numpy_mono8():
    """Convert mono8 image to numpy."""
    height, width = 2, 3
    data = bytes(range(height * width))
    msg = MockImage("mono8", height, width, data)

    result = image_msg_to_numpy(msg)

    assert result.shape == (2, 3)
    assert result.dtype == np.uint8


def test_image_msg_to_numpy_unsupported_encoding():
    """Raise error for unsupported encoding."""
    msg = MockImage("unsupported", 1, 1, b"\x00")

    with pytest.raises(ValueError, match="Unsupported image encoding"):
        image_msg_to_numpy(msg)
