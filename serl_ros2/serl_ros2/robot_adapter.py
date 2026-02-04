"""
ROS2 robot adapter for HIL-SERL.

This module provides a ROS2 implementation of the RobotAdapter interface. It subscribes
to robot state topics, publishes commands, and exposes robot services. The adapter keeps
ROS2 details internal; the env-facing API uses plain Python types (lists, dicts, numpy).

Topic naming: the adapter currently uses absolute names under the /hilserl namespace
(e.g. /hilserl/tcp_pose, /hilserl/command_pose); the robot side must publish and
subscribe on exactly these names.
"""

import concurrent.futures
import os
import threading
from typing import Any, Callable

import numpy as np
import rclpy
import time
import yaml

from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from serl_msgs.srv import ResetRobot, SetCompliance, SetGripper
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

from serl_framework.robot_adapter import RobotAdapter as BaseRobotAdapter
from serl_framework.teleop_adapter import TeleopAdapter
from serl_ros2.conversions import (
    float_msg_to_float,
    image_msg_to_numpy,
    joint_state_to_lists,
    pose_msg_to_list,
    stamp_to_seconds,
    twist_msg_to_list,
    wrench_msg_to_list,
)
from serl_ros2.utils import pose_to_msg, wrench_to_msg


class RobotAdapter(BaseRobotAdapter):
    """
    ROS2 implementation of the HIL-SERL robot adapter.

    Subscribes to robot state topics (tcp_pose, tcp_twist, joint_states, gripper_pos),
    synchronizes them using ApproximateTimeSynchronizer, and caches the latest snapshot
    for non-blocking access by the env.

    Args:
        node: ROS2 node handle (must support create_subscription, create_publisher, etc.).
        config_path: Optional path to YAML config file.
        thread_safe: If True, use locks for thread-safe access to cached state.
        executor_threads: Number of threads for the executor (default: config or 4).
        teleop_adapter: Optional TeleopAdapter to surface teleop inputs.

    Config options (YAML):
        cameras: List of camera names to subscribe to (e.g., ["front", "wrist"]).
        sync_slop: Time tolerance (seconds) for ApproximateTimeSynchronizer (default: 0.05).
        sync_queue_size: Queue size for synchronizer (default: 10).
        subscribe_wrench: Whether to subscribe to tcp_wrench topic (default: True).
        frame_id: Frame ID for outgoing pose commands (default: "base").
        service_timeout_sec: Timeout for service calls in seconds (default: 5.0).
        call_timeout_sec: Timeout for waiting on service responses (default: 5.0).
        executor_threads: Number of threads for the internal executor (default: 4).
    """

    def __init__(
        self,
        node: Node | Any,
        config_path: str | None = None,
        thread_safe: bool = False,
        executor_threads: int | None = None,
        teleop_adapter: TeleopAdapter | None = None,
        do_synchronize: bool = True,
    ) -> None:
        if node is None:
            raise ValueError("node is required")
        super().__init__(thread_safe=thread_safe, teleop_adapter=teleop_adapter)
        self.node = node
        self.config_path = config_path
        self._config: dict[str, Any] = {}
        self._last_pose_command: list[float] | np.ndarray | None = None
        self._last_wrench_command: list[float] | np.ndarray | None = None
        self._latest_wrench: list[float] | None = None
        self._state_sync: ApproximateTimeSynchronizer | None = None
        self._service_clients: dict[str, Any] = {}
        self._pose_pub: Any | None = None
        self._wrench_pub: Any | None = None
        self._image_subs: dict[str, Any] = {}
        self._wrench_sub: Any | None = None
        self._teleop_sub: Any | None = None
        self._executor_threads = executor_threads
        self._executor: MultiThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._spin_stop_event = threading.Event()
        self._spin_lock = threading.Lock()
        self._spin_strategy: str | None = None
        self._run_active = False
        self._run_timers: list[Any] = []
        self._owns_rclpy = False
        self._owns_node = False
        self._do_synchronize = do_synchronize
        self._latest_pose_msg: PoseStamped | None = None
        self._latest_twist_msg: TwistStamped | None = None
        self._latest_joint_msg: JointState | None = None
        self._latest_gripper_msg: Float32 | None = None

        if self._is_ros2_node(self.node):
            self._setup_ros2()
            # If teleop adapter is a JoyTeleopAdapter, set up its ROS2 subscription
            self._setup_teleop_ros2()

    def send_pose_command(self, pose: list[float] | np.ndarray) -> None:
        """
        Publish an absolute pose command to the robot.

        Args:
            pose: 7D pose as [x, y, z, qx, qy, qz, qw].
        """
        self._last_pose_command = pose
        if self._pose_pub is None:
            return
        self._publish_pose(pose)

    def now(self) -> float:
        """Return current time in seconds from the ROS2 clock (sim-time when available)."""
        if self._is_ros2_node(self.node):
            return self.node.get_clock().now().nanoseconds / 1e9
        return time.time()

    def time_difference(self, t1: float | None, t2: float | None) -> float:
        """
        Compute t1 - t2 using ROS2 time when timestamps are None.

        Args:
            t1: Timestamp in seconds, or None to use ROS2 now().
            t2: Timestamp in seconds, or None to use ROS2 now().

        Returns:
            Time difference in seconds.
        """
        now_sec = self.now()
        left = now_sec if t1 is None else float(t1)
        right = now_sec if t2 is None else float(t2)
        return left - right

    def send_wrench_command(self, wrench: list[float] | np.ndarray) -> None:
        """
        Publish a wrench command to the robot.

        Args:
            wrench: 6D wrench as [fx, fy, fz, tx, ty, tz].
        """
        self._last_wrench_command = wrench
        if self._wrench_pub is None:
            return
        self.node.get_logger().warn("Please double check: receiving a wrench command, "
                                    "and this wasn't yet tested. Does your robot subscribe to it?")
        self._publish_wrench(wrench)

    def call_service(self, service_name: str, timeout_sec: float | None = None, **kwargs: Any) -> dict[str, Any]:
        """
        Call a robot service by name.

        Currently supported services:
            - "clear_error": Trigger service to clear robot errors (no kwargs).
            - "set_gripper": Control gripper (mode, position).
            - "reset_robot": Reset robot to home position.
            - "set_compliance": Set compliance parameters.
            - "reset_world": Optional Trigger service for world/scene resets.
            - "reset_signal": Optional Trigger service for external reset-sync hooks.
        Future services:
            - "set_control_mode": Switch control mode.
            - "set_payload": Set payload parameters.

        Args:
            service_name: Name of the service to call.
            **kwargs: Service-specific arguments.

        Returns:
            Dict with "ok" (bool) and "message" (str).
        """
        timeout = self._config.get("call_timeout_sec", 5.0) if timeout_sec is None else float(timeout_sec)
        fut = self.call_service_async(service_name, **kwargs)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return {"ok": False, "message": f"Service call timed out after {timeout}s"}

    @classmethod
    def create(
        cls,
        node_name: str = "serl_ros2_adapter",
        config_path: str | None = None,
        thread_safe: bool = False,
        executor_threads: int | None = None,
        teleop_adapter: TeleopAdapter | None = None,
        do_synchronize: bool = True,
    ) -> "RobotAdapter":
        """
        Create an adapter with a managed ROS2 node.

        This helper initializes rclpy if needed and creates a node internally.
        Call start() explicitly to begin spinning.
        """
        owns_rclpy = False
        if not rclpy.ok():
            cls._ensure_log_dir()
            rclpy.init(args=None)
            owns_rclpy = True

        node = Node(node_name)
        if config_path:
            # get the use_sim_time flag from the config, so we can create the
            # ROS2 node correctly.
            try:
                with open(config_path, "r") as handle:
                    config = yaml.safe_load(handle) or {}
            except OSError:
                config = {}
            use_sim_time = config.get("use_sim_time", None)
            if use_sim_time is not None:
                node.set_parameters(
                    [Parameter("use_sim_time", Parameter.Type.BOOL, bool(use_sim_time))]
                )
        adapter = cls(
            node=node,
            config_path=config_path,
            thread_safe=thread_safe,
            executor_threads=executor_threads,
            teleop_adapter=teleop_adapter,
            do_synchronize=do_synchronize,
        )
        adapter._owns_rclpy = owns_rclpy
        adapter._owns_node = True
        return adapter

    def call_service_async(self, service_name: str, **kwargs: Any) -> concurrent.futures.Future[dict[str, Any]]:
        """
        Call a robot service asynchronously.

        Returns:
            concurrent.futures.Future resolving to a dict with "ok"/"message".
        """
        client = self._service_clients.get(service_name)
        if client is None:
            return self._completed_future(
                {"ok": False, "message": f"Service '{service_name}' not available"}
            )

        service_timeout_sec = float(
            kwargs.pop("service_timeout_sec", self._config.get("service_timeout_sec", 5.0))
        )
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=service_timeout_sec):
                return self._completed_future(
                    {
                        "ok": False,
                        "message": (
                            f"Service '{service_name}' not ready after "
                            f"{service_timeout_sec}s"
                        ),
                    }
                )

        # Dispatch based on service type
        if service_name == "clear_error":
            return self._call_trigger_service_async(client)
        if service_name == "set_gripper":
            return self._call_set_gripper_service_async(client, **kwargs)
        if service_name == "reset_robot":
            return self._call_reset_robot_service_async(client)
        if service_name == "set_compliance":
            return self._call_set_compliance_service_async(client, **kwargs)
        if service_name == "reset_world":
            return self._call_trigger_service_async(client)
        if service_name == "reset_signal":
            return self._call_trigger_service_async(client)
        if service_name == "set_control_mode":
            raise NotImplementedError("set_control_mode is not supported by this adapter")
        if service_name == "set_payload":
            raise NotImplementedError("set_payload is not supported by this adapter")

        return self._completed_future(
            {"ok": False, "message": f"Service '{service_name}' handler not implemented"}
        )

    def _call_trigger_service_async(self, client: Any) -> concurrent.futures.Future[dict[str, Any]]:
        """Call a std_srvs/Trigger service asynchronously."""
        py_fut: concurrent.futures.Future[dict[str, Any]] = concurrent.futures.Future()
        ros_fut = client.call_async(Trigger.Request())

        def _done(ros_future: Any) -> None:
            try:
                resp = ros_future.result()
                py_fut.set_result({"ok": bool(resp.success), "message": str(resp.message)})
            except Exception as exc:
                py_fut.set_result({"ok": False, "message": f"Service call failed: {exc!r}"})

        ros_fut.add_done_callback(_done)
        return py_fut

    def _call_set_gripper_service_async(
        self,
        client: Any,
        **kwargs: Any,
    ) -> concurrent.futures.Future[dict[str, Any]]:
        """Call SetGripper asynchronously."""
        if "mode" not in kwargs:
            return self._completed_future(
                {"ok": False, "message": "set_gripper requires 'mode' (0=open, 1=close, 2=position)"}
            )
        mode = int(kwargs["mode"])
        position = float(kwargs.get("position", 0.0))
        if mode == 2 and "position" not in kwargs:
            return self._completed_future(
                {
                    "ok": False,
                    "message": "set_gripper mode=2 requires 'position' in [0,1] (0=closed, 1=open)",
                }
            )
        request = SetGripper.Request()
        request.mode = mode
        request.position = position
        return self._call_custom_service_async(client, request)

    def _call_reset_robot_service_async(
        self,
        client: Any,
    ) -> concurrent.futures.Future[dict[str, Any]]:
        """Call ResetRobot asynchronously."""
        request = ResetRobot.Request()
        return self._call_custom_service_async(client, request)

    def _call_set_compliance_service_async(
        self,
        client: Any,
        **kwargs: Any,
    ) -> concurrent.futures.Future[dict[str, Any]]:
        """Call SetCompliance asynchronously."""
        params = kwargs.get("params")
        if params is None:
            return self._completed_future(
                {"ok": False, "message": "set_compliance requires 'params' dict"}
            )
        if not isinstance(params, dict):
            return self._completed_future(
                {"ok": False, "message": "set_compliance expects 'params' as dict[str, float]"}
            )
        param_names = list(params.keys())
        param_values = [float(params[name]) for name in param_names]
        request = SetCompliance.Request()
        request.parameter_names = param_names
        request.parameter_values = param_values
        return self._call_custom_service_async(client, request)

    def _call_custom_service_async(
        self,
        client: Any,
        request: Any,
    ) -> concurrent.futures.Future[dict[str, Any]]:
        """Call a custom service with "ok" and "message" result fields asynchronously."""
        py_fut: concurrent.futures.Future[dict[str, Any]] = concurrent.futures.Future()
        ros_fut = client.call_async(request)

        def _done(ros_future: Any) -> None:
            try:
                resp = ros_future.result()
                py_fut.set_result({"ok": bool(resp.ok), "message": str(resp.message)})
            except Exception as exc:
                py_fut.set_result({"ok": False, "message": f"Service call failed: {exc!r}"})

        ros_fut.add_done_callback(_done)
        return py_fut

    def _completed_future(self, result: dict[str, Any]) -> concurrent.futures.Future[dict[str, Any]]:
        """Return a completed Future with the given result."""
        fut: concurrent.futures.Future[dict[str, Any]] = concurrent.futures.Future()
        fut.set_result(result)
        return fut

    def _is_ros2_node(self, node: Any) -> bool:
        """Check if the provided object is a valid ROS2 node."""
        required = ("create_subscription", "create_publisher", "create_client")
        return all(hasattr(node, attr) for attr in required)

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path:
            return {}
        with open(config_path, "r") as handle:
            data = yaml.safe_load(handle) or {}
        return data

    def _setup_ros2(self) -> None:
        """Set up ROS2 subscriptions, publishers, and service clients."""
        self._config = self._load_config(self.config_path)
        cameras = self._config.get("cameras", [])
        sync_slop = self._config.get("sync_slop", 0.05)
        sync_queue_size = self._config.get("sync_queue_size", 10)
        subscribe_wrench = self._config.get("subscribe_wrench", True)
        self._do_synchronize = bool(self._config.get("do_synchronize", True))

        if cameras is None:
            cameras = []
        if not isinstance(cameras, (list, tuple)):
            raise ValueError("cameras must be a list of camera names")

        # State topic subscriptions
        if self._do_synchronize:
            self._tcp_pose_sub = Subscriber(self.node, PoseStamped, "/hilserl/tcp_pose")
            self._tcp_twist_sub = Subscriber(self.node, TwistStamped, "/hilserl/tcp_twist")
            self._joint_state_sub = Subscriber(self.node, JointState, "/hilserl/joint_states")
            self._gripper_sub = Subscriber(self.node, Float32, "/hilserl/gripper_pos")

            # TODO: Switch gripper_pos to a stamped message type and disable allow_headerless.
            self._state_sync = ApproximateTimeSynchronizer(
                [
                    self._tcp_pose_sub,
                    self._tcp_twist_sub,
                    self._joint_state_sub,
                    self._gripper_sub,
                ],
                queue_size=sync_queue_size,
                slop=sync_slop,
                allow_headerless=True,
            )
            self._state_sync.registerCallback(self._on_state)
        else:
            self._tcp_pose_sub = self.node.create_subscription(
                PoseStamped, "/hilserl/tcp_pose", self._on_pose, 10
            )
            self._tcp_twist_sub = self.node.create_subscription(
                TwistStamped, "/hilserl/tcp_twist", self._on_twist, 10
            )
            self._joint_state_sub = self.node.create_subscription(
                JointState, "/hilserl/joint_states", self._on_joint, 10
            )
            self._gripper_sub = self.node.create_subscription(
                Float32, "/hilserl/gripper_pos", self._on_gripper, 10
            )

        # Optional wrench subscription (not synchronized with state)
        if subscribe_wrench:
            self._wrench_sub = self.node.create_subscription(
                WrenchStamped, "/hilserl/tcp_wrench", self._on_wrench, 10
            )

        # Camera subscriptions
        for cam_name in cameras:
            cam_topic = f"/hilserl/camera/{cam_name}/image"
            self._image_subs[cam_name] = self.node.create_subscription(
                Image, cam_topic, lambda msg, name=cam_name: self._on_image(name, msg), 10
            )

        # Command publishers
        self._pose_pub = self.node.create_publisher(PoseStamped, "/hilserl/command_pose", 10)
        self._wrench_pub = self.node.create_publisher(WrenchStamped, "/hilserl/command_wrench", 10)

        # Service clients
        self._service_clients["clear_error"] = self.node.create_client(Trigger, "/hilserl/clear_error")
        self._service_clients["set_gripper"] = self.node.create_client(SetGripper, "/hilserl/set_gripper")
        self._service_clients["reset_robot"] = self.node.create_client(ResetRobot, "/hilserl/reset_robot")
        self._service_clients["set_compliance"] = self.node.create_client(SetCompliance, "/hilserl/set_compliance")
        self._service_clients["reset_world"] = self.node.create_client(
            Trigger,
            "/hilserl/reset_world",
        )
        self._service_clients["reset_signal"] = self.node.create_client(
            Trigger,
            "/teleop/reset_signal",
        )
        # Future: Add service clients
        # self._service_clients["set_control_mode"] = self.node.create_client(SetControlMode, "set_control_mode")
        # self._service_clients["set_payload"] = self.node.create_client(SetPayload, "set_payload")

    def _setup_teleop_ros2(self) -> None:
        """Set up ROS2 subscription for JoyTeleopAdapter if present."""
        teleop_adapter = self.get_teleop_adapter()
        if teleop_adapter is None:
            return
        # Check if the adapter has a setup_ros2 method (e.g., JoyTeleopAdapter)
        if hasattr(teleop_adapter, "setup_ros2"):
            teleop_adapter.setup_ros2(self.node)

    def start(self) -> None:
        """Start the internal executor spin thread."""
        with self._spin_lock:
            if self._spin_strategy == "run":
                raise RuntimeError("start() is not allowed after run() has been used")
            if self._spin_strategy is None:
                self._spin_strategy = "threaded"
            if self._executor is not None:
                return
            teleop_adapter = self.get_teleop_adapter()
            if teleop_adapter is not None:
                teleop_adapter.start()
            threads = self._executor_threads
            if threads is None:
                threads = int(self._config.get("executor_threads", 4))
            self._executor = MultiThreadedExecutor(num_threads=threads)
            self._executor.add_node(self.node)
            self._spin_stop_event.clear()
            self._spin_thread = threading.Thread(
                target=self._spin_loop,
                name="serl_ros2_adapter_spin",
                daemon=True,
            )
            self._spin_thread.start()

    def stop(self) -> None:
        """Stop the internal executor spin thread."""
        with self._spin_lock:
            if self._spin_strategy == "run":
                raise RuntimeError("stop() is not allowed after run() has been used")
            if self._executor is None:
                return

            executor = self._executor
            spin_thread = self._spin_thread

            # Signal stop and wake executor wait-set so spin loop can exit promptly.
            self._spin_stop_event.set()
            try:
                executor.wake()
            except Exception:
                pass

            # Wait until spin thread exits before touching executor ownership.
            # This avoids races where spin_once() submits to a thread-pool that
            # has already been torn down.
            if spin_thread is not None and spin_thread is not threading.current_thread():
                spin_thread.join()
            self._spin_thread = None

            # Shutdown executor - spin thread is no longer using it.
            try:
                executor.shutdown(timeout_sec=2.0)
            except Exception:
                pass
            try:
                executor.remove_node(self.node)
            except Exception:
                pass
            self._executor = None

            self._cleanup_owned_resources()

    def run(self, callbacks: list[tuple[Callable[[], None], float]]) -> None:
        """
        Run the adapter in the current thread using rclpy.spin().

        Args:
            callbacks: List of (callback, rate_hz) tuples. Each callback is invoked
                at a best-effort fixed rate. No real-time guarantees are provided.
        """
        if self._spin_strategy == "threaded":
            raise RuntimeError("run() is not allowed after start()/stop() has been used")
        if self._spin_strategy is None:
            self._spin_strategy = "run"
        if self._run_active:
            raise RuntimeError("run() is already active")
        if self._executor is not None:
            raise RuntimeError("run() is not allowed while start() is active")
        if not self._is_ros2_node(self.node):
            raise ValueError("node does not appear to be a ROS2 node")

        self._run_active = True
        self._run_timers = []
        try:
            for callback, rate_hz in callbacks:
                if rate_hz <= 0:
                    raise ValueError("callback rate_hz must be > 0")
                period = 1.0 / float(rate_hz)
                timer = self.node.create_timer(period, callback)
                self._run_timers.append(timer)
            rclpy.spin(self.node)
        finally:
            for timer in self._run_timers:
                try:
                    self.node.destroy_timer(timer)
                except Exception:
                    pass
            self._run_timers = []
            self._run_active = False
            self._cleanup_owned_resources()

    def _spin_loop(self) -> None:
        """Spin the executor until stop is requested."""
        executor = self._executor
        if executor is None:
            return
        while rclpy.ok() and not self._spin_stop_event.is_set():
            try:
                executor.spin_once(timeout_sec=0.1)
            except Exception:
                if self._spin_stop_event.is_set() or not rclpy.ok():
                    break
                raise

    def _cleanup_owned_resources(self) -> None:
        """Release node and rclpy resources owned by this adapter."""
        teleop_adapter = self.get_teleop_adapter()
        if teleop_adapter is not None:
            teleop_adapter.close()

        if self._owns_node:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self._owns_node = False

        if self._owns_rclpy:
            try:
                rclpy.shutdown()
            except Exception:
                pass
            self._owns_rclpy = False

    @staticmethod
    def _ensure_log_dir() -> None:
        if "ROS_LOG_DIR" in os.environ:
            return
        log_dir = os.path.join("/tmp", f"ros_log_{os.getpid()}")
        os.makedirs(log_dir, exist_ok=True)
        os.environ["ROS_LOG_DIR"] = log_dir

    def _on_state(
        self,
        pose_msg: PoseStamped,
        twist_msg: TwistStamped,
        joint_msg: JointState,
        gripper_msg: Float32,
    ) -> None:
        """Callback for synchronized state messages."""
        joint_pos, joint_vel = joint_state_to_lists(joint_msg)
        if self._latest_wrench is None:
            tcp_force = [0.0, 0.0, 0.0]
            tcp_torque = [0.0, 0.0, 0.0]
        else:
            tcp_force = self._latest_wrench[:3]
            tcp_torque = self._latest_wrench[3:]
        state = {
            "tcp_pose": pose_msg_to_list(pose_msg),
            "tcp_vel": twist_msg_to_list(twist_msg),
            "tcp_force": tcp_force,
            "tcp_torque": tcp_torque,
            "gripper_pose": float_msg_to_float(gripper_msg),
            "q": joint_pos,
            "dq": joint_vel,
        }
        self.set_state(state, stamp_to_seconds(pose_msg.header.stamp))

    def _on_wrench(self, msg: WrenchStamped) -> None:
        """Callback for wrench messages (not synchronized with state)."""
        self._latest_wrench = wrench_msg_to_list(msg)

    def _on_pose(self, msg: PoseStamped) -> None:
        """Callback for tcp_pose when synchronization is disabled."""
        self._latest_pose_msg = msg
        self._maybe_update_state()

    def _on_twist(self, msg: TwistStamped) -> None:
        """Callback for tcp_twist when synchronization is disabled."""
        self._latest_twist_msg = msg
        self._maybe_update_state()

    def _on_joint(self, msg: JointState) -> None:
        """Callback for joint_states when synchronization is disabled."""
        self._latest_joint_msg = msg
        self._maybe_update_state()

    def _on_gripper(self, msg: Float32) -> None:
        """Callback for gripper_pos when synchronization is disabled."""
        self._latest_gripper_msg = msg
        self._maybe_update_state()

    def _maybe_update_state(self) -> None:
        """Update cached state when all required messages are available."""
        if self._latest_pose_msg is None:
            return
        if self._latest_twist_msg is None:
            return
        if self._latest_joint_msg is None:
            return
        if self._latest_gripper_msg is None:
            return
        self._on_state(
            self._latest_pose_msg,
            self._latest_twist_msg,
            self._latest_joint_msg,
            self._latest_gripper_msg,
        )

    def _on_image(self, name: str, msg: Image) -> None:
        """Callback for camera image messages."""
        if self.latest_images is None:
            images: dict[str, np.ndarray] = {}
        else:
            images = dict(self.latest_images)

        # Convert ROS Image to numpy array
        images[name] = image_msg_to_numpy(msg)

        # TODO: Implement proper image synchronization with state.
        # Currently images are updated independently from state sync.
        # For tight synchronization, consider adding images to the ApproximateTimeSynchronizer
        # or implementing a custom sync policy based on timestamp comparison.
        timestamp = self.latest_timestamp
        if timestamp is None:
            timestamp = stamp_to_seconds(msg.header.stamp)
        if timestamp is None:
            self.node.get_logger().warning("Image received, but have no state timestamp yet; ignoring.")
            return
        self.set_images(images, timestamp)

    def _publish_pose(self, pose: list[float] | np.ndarray) -> None:
        """Publish a pose command."""
        frame_id = self._config.get("frame_id", "base")
        stamp = self.node.get_clock().now().to_msg() if self._is_ros2_node(self.node) else None
        msg = pose_to_msg(pose, frame_id=frame_id, timestamp=stamp)
        self._pose_pub.publish(msg)

    def _publish_wrench(self, wrench: list[float] | np.ndarray) -> None:
        """Publish a wrench command."""
        msg = wrench_to_msg(wrench)
        self._wrench_pub.publish(msg)
