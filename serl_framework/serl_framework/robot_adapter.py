"""
Abstract robot adapter interface for HIL-SERL.

This module defines the transport-agnostic API that HIL-SERL environments use to
communicate with robots. Concrete implementations (e.g., serl_ros2.RobotAdapter)
provide the actual transport wiring while keeping this interface ROS-free.

The adapter pattern allows HIL-SERL to work with different robot frameworks
(ROS2, direct drivers, simulation) by swapping implementations without changing
env code.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Callable

import threading
import time

from serl_framework.teleop_adapter import TeleopAdapter


class RobotAdapter(ABC):
    """
    Abstract base class for robot adapters.

    Provides caching and thread-safe access to robot state, images, and teleop input.
    Concrete implementations must provide transport-specific methods for sending
    commands and calling services.

    Please also note that timestamps could be specific to the adapter. Use
    time_difference() to compute time differences in timestamps returned through this interface.

    State:
        - tcp_pose: 7D list [x, y, z, qx, qy, qz, qw]
        - tcp_vel: 6D list [vx, vy, vz, wx, wy, wz]
        - tcp_force: 3D list [fx, fy, fz] (optional, defaults to zeros)
        - tcp_torque: 3D list [tx, ty, tz] (optional, defaults to zeros)
        - gripper_pose: float in [0, 1] (0=closed, 1=open)
        - q: list of joint positions
        - dq: list of joint velocities

    Images:
        - Dict mapping camera names (str) to numpy arrays (H, W, C)

    Teleop:
        - Tuple of (xyz: list[float], rpy: list[float], buttons: list[int])

    Args:
        thread_safe: If True, use locks for thread-safe access to cached state.
    """

    def __init__(
        self,
        thread_safe: bool = False,
        teleop_adapter: TeleopAdapter | None = None,
    ) -> None:
        """
        Initialize the adapter cache and locking behavior.

        Args:
            thread_safe: If True, wrap cache access with a re-entrant lock.
            teleop_adapter: Optional TeleopAdapter for teleop input passthrough.
        """
        self._latest_state: dict[str, Any] | None = None
        self._latest_images: dict[str, Any] | None = None
        self._latest_timestamp: float | None = None
        self._lock = threading.RLock() if thread_safe else None
        self._no_lock = nullcontext()
        self._teleop_adapter = teleop_adapter

    def _guard(self):
        """
        Return the appropriate context manager for thread safety.

        Returns:
            Context manager (either a lock or a no-op context).
        """
        return self._lock if self._lock is not None else self._no_lock

    @property
    def latest_state(self) -> dict[str, Any] | None:
        """Return the most recent cached robot state dict."""
        with self._guard():
            return self._latest_state

    @property
    def latest_images(self) -> dict[str, Any] | None:
        """Return the most recent cached images dict."""
        with self._guard():
            return self._latest_images

    @property
    def latest_timestamp(self) -> float | None:
        """Return the most recent observation timestamp (seconds)."""
        with self._guard():
            return self._latest_timestamp

    def get_observation(
        self,
        max_age_s: float | None = None,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Return the latest observation snapshot.

        Args:
            max_age_s: Maximum age in seconds. If the cached snapshot is older,
                returns None to indicate stale data.
            now: Current time in seconds (for testing). Defaults to adapter time.

        Returns:
            Dict with keys "state", "images", "timestamp", or None if stale/unavailable.
        """
        with self._guard():
            if max_age_s is not None:
                if self._latest_timestamp is None:
                    return None
                age_s = self.time_difference(now if now is not None else None, self._latest_timestamp)
                if age_s > max_age_s:
                    return None
            return {
                "state": self._latest_state,
                "images": self._latest_images,
                "timestamp": self._latest_timestamp,
            }

    def get_teleop(self) -> tuple[list[float], list[float], list[int], str] | None:
        """
        Return the latest teleop input from the configured TeleopAdapter.

        Returns:
            Tuple of (xyz, rpy, buttons, frame_id), or None if no TeleopAdapter
            is configured.
        """
        if self._teleop_adapter is None:
            return None
        return self._teleop_adapter.get_teleop()

    def get_teleop_adapter(self) -> TeleopAdapter | None:
        """
        Return the configured TeleopAdapter instance, if any.
        """
        return self._teleop_adapter

    def wait_for_state(
        self,
        timeout_s: float = 5.0,
        poll_s: float = 0.05,
        max_in_the_past_s: float | None = None,
    ) -> None:
        """
        Block until the adapter has received at least one *new* state update.

        Args:
            timeout_s: Maximum time to wait in seconds.
            poll_s: Sleep duration between checks in seconds.
            max_in_the_past_s: If set, require the latest state timestamp to be no older
                than this many seconds.

        Raises:
            TimeoutError: If no state is received within the timeout, or if the
                latest state is older than max_in_the_past_s for the entire wait.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        state = None
        errstr = ""

        while time.monotonic() < deadline:
            state = self.latest_state
            timestamp = self.latest_timestamp
            if state is not None:
                if max_in_the_past_s is None:
                    return
                if timestamp is not None:
                    age_s = self.time_difference(None, timestamp)
                    # Freshness gate: even if a state exists, reject it if it's too old.
                    if age_s <= max_in_the_past_s:
                        return
                    errstr = f"state at time {timestamp} is too old ({age_s}s)."
            time.sleep(poll_s)
        if state is None:
            errstr = "No state received at all."
        raise TimeoutError(f"Timed out waiting for robot state from adapter ({errstr}).")

    @abstractmethod
    def send_pose_command(self, pose: list[float]) -> None:
        """
        Send an absolute pose command to the robot.

        Args:
            pose: 7D pose as [x, y, z, qx, qy, qz, qw].
        """

    def set_step_wrench_boost(self, wrench: list[float]) -> None:
        """Set a transient wrench boost added to the next pose command only.

        The boost is added to the persistent feedforward_wrench for a single
        send_pose_command call, then auto-clears. Used for action-magnitude
        force boost (e.g., extra push when teleop input is near saturation).

        Args:
            wrench: 6D wrench as [fx, fy, fz, tx, ty, tz].
        """

    @abstractmethod
    def send_wrench_command(self, wrench: list[float]) -> None:
        """
        Send a wrench command to the robot.

        Args:
            wrench: 6D wrench as [fx, fy, fz, tx, ty, tz].
        """

    @abstractmethod
    def call_service(self, service_name: str, **kwargs: Any) -> dict[str, Any]:
        """
        Call a robot service by name.

        This generic interface supports different robots with varying service sets.

        Args:
            service_name: Service identifier (e.g., "clear_error", "set_gripper").
            **kwargs: Service-specific arguments.

        Returns:
            Dict with "ok" (bool) and "message" (str).
        """

    @classmethod
    @abstractmethod
    def create(
        cls,
        *args,
        **kwargs,
    ):
        """
        Create an adapter instance.

        Implementations may use this to hide transport-specific setup.
        """

    @abstractmethod
    def start(self) -> None:
        """Start background processing (e.g., spinning or IO loops). See also run()."""

    @abstractmethod
    def stop(self) -> None:
        """Stop background processing and release resources."""

    @abstractmethod
    def run(self, callbacks: list[tuple[Callable[[], None], float]]) -> None:
        """
        Run the adapter's processing loop in the current thread.

        Using this will disallow the use of start() and stop().

        Args:
            callbacks: List of (callback, rate_hz) tuples. Each callback is invoked
                at a best-effort fixed rate. No real-time guarantees are provided.
        """

    def now(self) -> float:
        """Return the current time in seconds from the adapter's clock.

        For ROS2 adapters with ``use_sim_time=true`` this returns sim-time.
        The default implementation returns wall-clock time so that environments
        work without a ROS2-specific adapter.
        """
        return time.time()

    @abstractmethod
    def time_difference(self, t1: float | None, t2: float | None) -> float:
        """
        Compute t1 - t2 using the adapter's time base.
        This should be used with all timestamps which have been received through the RobotAdapter.

        Args:
            t1: Timestamp in seconds from the adapter interface, or None to use "now".
            t2: Timestamp in seconds from the adapter interface, or None to use "now".

        Returns:
            Time difference (t1 - t2) in seconds. Can be negative.
        """

    def set_state(self, state: dict[str, Any], timestamp: float) -> None:
        """
        Update the cached state and timestamp.

        Args:
            state: Robot state dict (see class docstring for expected keys).
            timestamp: Observation timestamp in seconds.
        """
        with self._guard():
            self._latest_state = state
            self._latest_timestamp = timestamp

    def set_images(self, images: dict[str, Any], timestamp: float) -> None:
        """
        Update the cached images and timestamp.

        Args:
            images: Dict mapping camera names to image arrays.
            timestamp: Observation timestamp in seconds.
        """
        with self._guard():
            self._latest_images = images
            self._latest_timestamp = timestamp

    def set_snapshot(
        self,
        state: dict[str, Any],
        images: dict[str, Any],
        timestamp: float,
    ) -> None:
        """
        Update state, images, and timestamp atomically.

        Args:
            state: Robot state dict.
            images: Dict mapping camera names to image arrays.
            timestamp: Observation timestamp in seconds.
        """
        with self._guard():
            self._latest_state = state
            self._latest_images = images
            self._latest_timestamp = timestamp
