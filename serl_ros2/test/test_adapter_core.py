"""Unit tests for the serl_ros2 adapter."""

import os
import pytest
import rclpy
from rclpy.node import Node

from serl_ros2.robot_adapter import RobotAdapter


class MockNode:
    """Mock ROS2 node that doesn't trigger ROS2 setup."""

    pass  # Doesn't have create_subscription, so _is_ros2_node returns False


def _make_ros2_node():
    """Create a ROS2 node for tests or skip if ROS2 is unavailable."""
    owns_rclpy = False
    if not rclpy.ok():
        if "ROS_LOG_DIR" not in os.environ:
            log_dir = os.path.join("/tmp", f"ros_log_{os.getpid()}")
            os.makedirs(log_dir, exist_ok=True)
            os.environ["ROS_LOG_DIR"] = log_dir
        rclpy.init(args=None)
        owns_rclpy = True

    try:
        node = Node("serl_ros2_adapter_test")
    except Exception as exc:
        if owns_rclpy:
            rclpy.shutdown()
        pytest.skip(f"ROS2 node creation not available in this environment: {exc}")

    return node, owns_rclpy


def test_get_observation_returns_cached_state():
    """Verify get_observation returns the cached snapshot."""
    adapter = RobotAdapter(node=MockNode())
    adapter.set_snapshot(
        state={"tcp_pose": [0.0] * 7},
        images={"front": "image"},
        timestamp=123.0,
    )

    obs = adapter.get_observation()

    assert obs["state"] == adapter._latest_state
    assert obs["images"] == adapter._latest_images
    assert obs["timestamp"] == 123.0


def test_send_pose_command_caches_value():
    """Verify send_pose_command stores the pose for testing."""
    adapter = RobotAdapter(node=MockNode())
    pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]

    adapter.send_pose_command(pose)

    assert adapter._last_pose_command == pose


def test_send_wrench_command_caches_value():
    """Verify send_wrench_command stores the wrench for testing."""
    adapter = RobotAdapter(node=MockNode())
    wrench = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]

    adapter.send_wrench_command(wrench)

    assert adapter._last_wrench_command == wrench


def test_call_service_returns_not_available():
    """Verify call_service returns not available for mock node."""
    adapter = RobotAdapter(node=MockNode())

    result = adapter.call_service("reset_robot")

    assert result["ok"] is False
    assert "not available" in result["message"].lower()


def test_get_observation_respects_max_age():
    """Verify get_observation returns None for stale data."""
    adapter = RobotAdapter(node=MockNode())
    adapter.set_snapshot(
        state={"tcp_pose": [0.0] * 7},
        images={},
        timestamp=100.0,
    )

    # With a fake "now" far in the future, the snapshot should be stale
    obs = adapter.get_observation(max_age_s=1.0, now=200.0)

    assert obs is None


def test_get_observation_returns_data_within_max_age():
    """Verify get_observation returns data when within max_age."""
    adapter = RobotAdapter(node=MockNode())
    adapter.set_snapshot(
        state={"tcp_pose": [0.0] * 7},
        images={},
        timestamp=100.0,
    )

    obs = adapter.get_observation(max_age_s=1.0, now=100.5)

    assert obs is not None
    assert obs["state"]["tcp_pose"] == [0.0] * 7


def test_start_stop_spin_thread():
    """Verify adapter start/stop is idempotent and manages the spin thread."""
    node, owns_rclpy = _make_ros2_node()
    adapter = RobotAdapter(node=node)

    try:
        # stop before start should be a no-op
        adapter.stop()
        assert adapter._executor is None

        adapter.start()
        assert adapter._executor is not None
        assert adapter._spin_thread is not None
        first_thread = adapter._spin_thread

        # starting twice should not create a new thread
        adapter.start()
        assert adapter._spin_thread is first_thread

        adapter.stop()
        assert adapter._executor is None
        assert adapter._spin_thread is None

        # stopping twice should be a no-op
        adapter.stop()
        assert adapter._executor is None
        assert adapter._spin_thread is None
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if owns_rclpy:
            rclpy.shutdown()


def test_run_rejects_start_mode(monkeypatch):
    """Verify run() is rejected after start() has been used."""
    node, owns_rclpy = _make_ros2_node()
    adapter = RobotAdapter(node=node)

    try:
        adapter.start()
        monkeypatch.setattr(rclpy, "spin", lambda _node: None)

        with pytest.raises(RuntimeError):
            adapter.run(callbacks=[])
    finally:
        try:
            adapter.stop()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if owns_rclpy:
            rclpy.shutdown()


def test_start_rejects_run_mode(monkeypatch):
    """Verify start() is rejected after run() has been used."""
    node, owns_rclpy = _make_ros2_node()
    adapter = RobotAdapter(node=node)

    try:
        monkeypatch.setattr(rclpy, "spin", lambda _node: None)
        adapter.run(callbacks=[])

        with pytest.raises(RuntimeError):
            adapter.start()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if owns_rclpy:
            rclpy.shutdown()
