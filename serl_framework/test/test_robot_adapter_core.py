"""
Tests for the RobotAdapter base class behaviors.
"""

import sys
import time
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from serl_framework.robot_adapter import RobotAdapter


class DummyAdapter(RobotAdapter):
    """
    Minimal adapter implementation for testing the base class.
    """

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Create a dummy adapter instance.
        """
        return cls()

    def send_pose_command(self, pose):
        """
        Cache the last pose command.
        """
        self.last_pose = pose

    def send_wrench_command(self, wrench):
        """
        Cache the last wrench command.
        """
        self.last_wrench = wrench

    def call_service(self, service_name, **kwargs):
        """
        Return a default failure response.
        """
        return {"ok": False, "message": "not implemented"}

    def start(self):
        """
        Mark the adapter as started.
        """
        self.started = True

    def stop(self):
        """
        Mark the adapter as stopped.
        """
        self.started = False

    def run(self, callbacks):
        """
        Record callbacks passed to run for inspection.
        """
        self.run_callbacks = callbacks

    def time_difference(self, t1: float | None, t2: float | None) -> float:
        """
        Compute t1 - t2 using wall-clock time for tests.
        """
        if t1 is None:
            t1 = time.time()
        if t2 is None:
            t2 = time.time()
        return t1 - t2


class DummyTeleop:
    """
    Minimal teleop adapter stub for testing get_teleop passthrough.
    """

    def __init__(self, teleop):
        """
        Store the teleop payload to return.
        """
        self._teleop = teleop

    def get_teleop(self):
        """
        Return the configured teleop payload.
        """
        return self._teleop

    def close(self):
        """
        No-op close for the dummy teleop adapter.
        """
        return None


def test_get_observation_returns_cached_state():
    """
    Verify get_observation returns the cached snapshot.
    """
    adapter = DummyAdapter()
    adapter.set_snapshot(
        state={"tcp_pose": [0.0] * 7},
        images={"front": "image"},
        timestamp=123,
    )

    obs = adapter.get_observation()

    assert obs["state"] == adapter.latest_state
    assert obs["images"] == adapter.latest_images
    assert obs["timestamp"] == adapter.latest_timestamp


def test_get_teleop_returns_adapter_output():
    """
    Verify get_teleop returns the TeleopAdapter output when configured.
    """
    teleop = ([0.1, 0.2, 0.3], [0.0, -0.1, 0.2], [1, 0], "base")
    teleop_adapter = DummyTeleop(teleop)
    adapter = DummyAdapter(teleop_adapter=teleop_adapter)

    assert adapter.get_teleop() == teleop
    assert adapter.get_teleop_adapter() is teleop_adapter
