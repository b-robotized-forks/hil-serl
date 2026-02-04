"""Integration test for ROS2 service calls via RobotAdapter."""

import os
import threading

import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from serl_msgs.srv import ResetRobot, SetCompliance, SetGripper
from serl_ros2.robot_adapter import RobotAdapter


class ServiceNode(Node):
    """ROS2 node that exposes dummy services for adapter integration tests."""

    def __init__(self) -> None:
        super().__init__("serl_ros2_service_test")
        self.last_gripper_request: SetGripper.Request | None = None
        self.last_compliance_request: SetCompliance.Request | None = None
        self.reset_robot_called = False
        self.reset_world_called = False
        self.clear_error_called = False
        self.reset_signal_called = False

        self.create_service(Trigger, "/hilserl/clear_error", self._on_clear_error)
        self.create_service(SetGripper, "/hilserl/set_gripper", self._on_set_gripper)
        self.create_service(ResetRobot, "/hilserl/reset_robot", self._on_reset_robot)
        self.create_service(SetCompliance, "/hilserl/set_compliance", self._on_set_compliance)
        self.create_service(Trigger, "/hilserl/reset_world", self._on_reset_world)
        self.create_service(
            Trigger,
            "/teleop/reset_signal",
            self._on_reset_signal,
        )

    def _on_clear_error(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.clear_error_called = True
        response.success = True
        response.message = "cleared"
        return response

    def _on_set_gripper(self, request: SetGripper.Request, response: SetGripper.Response) -> SetGripper.Response:
        self.last_gripper_request = request
        response.ok = True
        response.message = "gripper ok"
        return response

    def _on_reset_robot(self, request: ResetRobot.Request, response: ResetRobot.Response) -> ResetRobot.Response:
        self.reset_robot_called = True
        response.ok = True
        response.message = "reset ok"
        return response

    def _on_reset_world(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        self.reset_world_called = True
        response.success = True
        response.message = "world reset ok"
        return response

    def _on_set_compliance(
        self,
        request: SetCompliance.Request,
        response: SetCompliance.Response,
    ) -> SetCompliance.Response:
        self.last_compliance_request = request
        response.ok = True
        response.message = "compliance ok"
        return response

    def _on_reset_signal(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        self.reset_signal_called = True
        response.success = True
        response.message = "reset signal ok"
        return response


def _make_ros2() -> tuple[Node, bool]:
    owns_rclpy = False
    if not rclpy.ok():
        if "ROS_LOG_DIR" not in os.environ:
            log_dir = os.path.join("/tmp", f"ros_log_{os.getpid()}")
            os.makedirs(log_dir, exist_ok=True)
            os.environ["ROS_LOG_DIR"] = log_dir
        rclpy.init(args=None)
        owns_rclpy = True

    try:
        node = ServiceNode()
    except Exception as exc:
        if owns_rclpy:
            rclpy.shutdown()
        pytest.skip(f"ROS2 node creation not available in this environment: {exc}")

    return node, owns_rclpy


def _spin_executor(executor: MultiThreadedExecutor, stop_event: threading.Event) -> None:
    while rclpy.ok() and not stop_event.is_set():
        executor.spin_once(timeout_sec=0.05)


def test_adapter_service_calls_roundtrip() -> None:
    node, owns_rclpy = _make_ros2()
    adapter = RobotAdapter(node=node)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    stop_event = threading.Event()
    spin_thread = threading.Thread(target=_spin_executor, args=(executor, stop_event), daemon=True)
    spin_thread.start()

    try:
        result = adapter.call_service("clear_error", timeout_sec=2.0)
        assert result["ok"] is True
        assert node.clear_error_called is True

        result = adapter.call_service("set_gripper", mode=1, timeout_sec=2.0)
        assert result["ok"] is True
        assert node.last_gripper_request is not None
        assert node.last_gripper_request.mode == 1

        result = adapter.call_service("reset_robot", timeout_sec=2.0)
        assert result["ok"] is True
        assert node.reset_robot_called is True

        params = {"translational_stiffness": 2000.0, "rotational_damping": 7.0}
        result = adapter.call_service("set_compliance", params=params, timeout_sec=2.0)
        assert result["ok"] is True
        assert node.last_compliance_request is not None
        assert list(node.last_compliance_request.parameter_names) == list(params.keys())
        assert list(node.last_compliance_request.parameter_values) == [float(v) for v in params.values()]

        result = adapter.call_service("reset_world", timeout_sec=2.0)
        assert result["ok"] is True
        assert node.reset_world_called is True

        result = adapter.call_service("reset_signal", timeout_sec=2.0)
        assert result["ok"] is True
        assert node.reset_signal_called is True
    finally:
        stop_event.set()
        spin_thread.join(timeout=2.0)
        executor.remove_node(node)
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()
