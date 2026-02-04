"""
Mock RobotAdapter implementations for tests and smoke runs.

Origin: New file for Phase 2.1
Modified: N/A
"""

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from serl_framework.robot_adapter import RobotAdapter


class MockRobotAdapter(RobotAdapter):
    """
    In-memory RobotAdapter that records commands and serves cached snapshots.

    This adapter is intended for unit tests and local smoke tests where a
    lightweight, ROS-free adapter is needed.
    """

    def __init__(self, snapshot: Optional[Dict[str, Any]] = None, auto_timestamp: bool = True) -> None:
        """
        Initialize the mock adapter.

        Args:
            snapshot: Optional initial snapshot dict with state/images/timestamp.
            auto_timestamp: If True, refresh the snapshot timestamp on each read.
        """
        super().__init__(thread_safe=False)
        self.auto_timestamp = auto_timestamp
        self.pose_commands: List[List[float]] = []
        self.wrench_commands: List[List[float]] = []
        self.service_calls: List[Dict[str, Any]] = []
        self.started = False
        self._set_snapshot_from_dict(snapshot)

    def _set_snapshot_from_dict(self, snapshot: Optional[Dict[str, Any]]) -> None:
        """
        Load an initial snapshot dict into the adapter cache.

        Args:
            snapshot: Snapshot dict with keys state/images/timestamp.
        """
        if snapshot is None:
            return
        required_keys = {"state", "images", "timestamp"}
        missing = required_keys - snapshot.keys()
        if missing:
            raise ValueError(f"Snapshot missing required keys: {sorted(missing)}")
        self.set_snapshot(snapshot["state"], snapshot["images"], snapshot["timestamp"])

    def update_snapshot(
        self,
        state: Optional[Dict[str, Any]] = None,
        images: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update the cached snapshot while preserving unspecified fields.

        Args:
            state: Optional state dict to update.
            images: Optional images dict to update.
            timestamp: Optional timestamp override (defaults to time.time()).
        """
        if state is None:
            if self.latest_state is None:
                raise ValueError("state is required for the first snapshot update")
            state = self.latest_state
        if images is None:
            images = self.latest_images or {}
        if timestamp is None:
            timestamp = time.time()
        self.set_snapshot(state, images, timestamp)

    def get_observation(
        self,
        max_age_s: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the cached snapshot, refreshing the timestamp if configured.

        Args:
            max_age_s: Maximum allowable age in seconds.
            now: Optional time override used for age checks.

        Returns:
            Cached snapshot dict or None if stale/unavailable.
        """
        if self.auto_timestamp and self.latest_timestamp is not None:
            current_time = time.time() if now is None else now
            with self._guard():
                self._latest_timestamp = current_time
        return super().get_observation(max_age_s=max_age_s, now=now)

    def send_pose_command(self, pose: List[float]) -> None:
        """
        Record pose commands issued by callers.

        Args:
            pose: 7D pose command.
        """
        self.pose_commands.append(list(pose))

    def time_difference(self, t1: float | None, t2: float | None) -> float:
        """
        Compute t1 - t2 using wall-clock time when timestamps are None.
        """
        now = time.time()
        left = now if t1 is None else float(t1)
        right = now if t2 is None else float(t2)
        return left - right

    def send_wrench_command(self, wrench: List[float]) -> None:
        """
        Record wrench commands issued by callers.

        Args:
            wrench: 6D wrench command.
        """
        self.wrench_commands.append(list(wrench))

    def call_service(self, service_name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Record service calls and return a success response.

        Args:
            service_name: Service identifier.
            **kwargs: Service arguments.

        Returns:
            Dict indicating success.
        """
        self.service_calls.append({"service_name": service_name, "kwargs": kwargs})
        return {"ok": True, "message": "ok"}

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Creation is not supported for the mock adapter.
        """
        raise NotImplementedError("MockRobotAdapter does not implement create()")

    def start(self) -> None:
        """
        Start is a no-op for the mock adapter.
        """
        teleop_adapter = self.get_teleop_adapter()
        if teleop_adapter is not None:
            teleop_adapter.start()
        self.started = True

    def stop(self) -> None:
        """
        Stop is a no-op for the mock adapter.
        """
        self.started = False

    def run(self, callbacks: List[Any]) -> None:
        """
        Run is a no-op for the mock adapter.

        Args:
            callbacks: Ignored callback list.
        """
        return None


def make_mock_snapshot(
    camera_names: Sequence[str],
    dof: int,
    tcp_pose: Optional[np.ndarray] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create a synthetic observation snapshot for tests.

    Args:
        camera_names: Names to populate in the images dict.
        dof: Degrees of freedom for joint arrays.
        tcp_pose: Optional tcp pose override (7D array).
        timestamp: Optional timestamp override.

    Returns:
        Snapshot dict with state/images/timestamp entries.
    """
    if tcp_pose is None:
        tcp_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    state = {
        "tcp_pose": np.array(tcp_pose, dtype=np.float32),
        "tcp_vel": np.zeros((6,), dtype=np.float32),
        "tcp_force": np.zeros((3,), dtype=np.float32),
        "tcp_torque": np.zeros((3,), dtype=np.float32),
        "gripper_pose": 1.0,
        "q": np.zeros((dof,), dtype=np.float32),
        "dq": np.zeros((dof,), dtype=np.float32),
    }
    images = {name: np.zeros((200, 300, 3), dtype=np.uint8) for name in camera_names}
    snapshot_time = time.time() if timestamp is None else timestamp
    return {"state": state, "images": images, "timestamp": snapshot_time}
