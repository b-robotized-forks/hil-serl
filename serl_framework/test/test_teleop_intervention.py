"""
Unit tests for TeleopIntervention frame-routing behavior.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from serl_framework.envs.robot_env import DefaultEnvConfig, RobotEnv
from serl_framework.testing.mock_adapter import MockRobotAdapter, make_mock_snapshot
from serl_framework.wrappers import TeleopIntervention


class TeleopFrameTestConfig(DefaultEnvConfig):
    """
    Minimal config for testing TeleopIntervention frame conversions.
    """

    ROBOT_DOF = 7
    CAMERAS = {"front": {}}
    IMAGE_CROP = {}
    IMAGE_CHANNEL_ORDER = "rgb"
    DISPLAY_IMAGE = False
    ENABLE_KEYBOARD_LISTENER = False
    ACTION_SCALE = (1.0, 1.0, 1.0)
    ABS_POSE_LIMIT_LOW = np.array([-2.0, -2.0, -2.0, -np.pi, -np.pi, -np.pi])
    ABS_POSE_LIMIT_HIGH = np.array([2.0, 2.0, 2.0, np.pi, np.pi, np.pi])
    TARGET_POSE = np.zeros((6,))
    RESET_POSE = np.zeros((6,))
    REWARD_THRESHOLD = np.ones((6,))
    JOINT_RESET_PERIOD = 0
    RANDOM_RESET = False


class StaticTeleop:
    """
    Teleop stub that returns a fixed reading.
    """

    def __init__(
        self,
        xyz: list[float],
        rpy: list[float],
        buttons: list[int],
        frame_id: str,
    ) -> None:
        self.xyz = xyz
        self.rpy = rpy
        self.buttons = buttons
        self.frame_id = frame_id

    def get_teleop(self) -> tuple[list[float], list[float], list[int], str]:
        return list(self.xyz), list(self.rpy), list(self.buttons), self.frame_id


@pytest.mark.parametrize(
    "frame_id,expected_delta_base",
    [
        ("base", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ("tcp", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("joy", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("mystery", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ],
)
def test_teleop_intervention_frame_routing(
    frame_id: str,
    expected_delta_base: np.ndarray,
) -> None:
    """
    Verify frame-driven routing and fallback behavior in TeleopIntervention.
    """
    # 90-degree yaw means tcp +x aligns with base +y.
    quat = Rotation.from_euler("xyz", [0.0, 0.0, np.pi / 2]).as_quat()
    pose = np.array([0.0, 0.0, 0.0, *quat], dtype=np.float32)
    snapshot = make_mock_snapshot(["front"], 7, tcp_pose=pose)
    adapter = MockRobotAdapter(snapshot)

    env = RobotEnv(
        hz=100,
        fake_env=False,
        save_video=False,
        config=TeleopFrameTestConfig(),
        adapter=adapter,
    )
    teleop = StaticTeleop(
        xyz=[1.0, 0.0, 0.0],
        rpy=[0.0, 0.0, 0.0],
        buttons=[0, 0],
        frame_id=frame_id,
    )
    wrapped = TeleopIntervention(
        env,
        teleop_adapter=teleop,
        base_frame_id="base",
        tcp_frame_id="tcp",
    )

    wrapped.reset()
    wrapped.step(np.zeros((7,), dtype=np.float32))

    assert adapter.pose_commands, "Expected pose command from teleop intervention step."
    last_command = np.asarray(adapter.pose_commands[-1], dtype=np.float32)
    expected_position = pose[:3] + expected_delta_base
    assert np.allclose(last_command[:3], expected_position, atol=1e-6)
