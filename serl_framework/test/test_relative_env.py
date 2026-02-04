"""
Unit tests for the RelativeFrame wrapper around RobotEnv.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from serl_framework.envs.relative_env import RelativeFrame
from serl_framework.envs.robot_env import DefaultEnvConfig, RobotEnv
from serl_framework.testing.mock_adapter import MockRobotAdapter, make_mock_snapshot
from serl_framework.utils.transformations import construct_transform_matrix


class RelativeFrameTestConfig(DefaultEnvConfig):
    """
    Minimal config for exercising RelativeFrame behavior with RobotEnv.
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


def _make_snapshot_with_state(
    pose: np.ndarray,
    tcp_vel: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Create a mock adapter snapshot with a specified pose and TCP velocity.
    """
    # Start from a standard mock snapshot and override the fields we care about.
    snapshot = make_mock_snapshot(["front"], 7, tcp_pose=pose)
    state = dict(snapshot["state"])
    if tcp_vel is not None:
        state["tcp_vel"] = np.array(tcp_vel, dtype=np.float32)
    snapshot["state"] = state
    return snapshot


def test_relative_frame_transforms_action_to_base_frame() -> None:
    """
    Verify RelativeFrame maps body-frame action deltas into the base frame.
    """
    # 90-degree yaw rotates +x (body) into +y (base).
    rot = Rotation.from_euler("xyz", [0.0, 0.0, np.pi / 2]).as_quat()
    pose = np.array([0.0, 0.0, 0.0, *rot], dtype=np.float32)
    # Use an adapter snapshot with the rotated pose so the wrapper builds its transform matrix.
    adapter = MockRobotAdapter(_make_snapshot_with_state(pose, np.zeros((6,))))
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=RelativeFrameTestConfig(), adapter=adapter)
    wrapped = RelativeFrame(env, include_relative_pose=False)
    wrapped.reset()

    # Command +x in body frame; wrapper should rotate this into base frame before sending.
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    wrapped.step(action)

    assert adapter.pose_commands, "Expected a pose command from the wrapped env"
    last_command = np.array(adapter.pose_commands[-1], dtype=np.float32)
    # Expected base-frame translation is R * [1, 0, 0] (no rotation component change).
    expected_delta = Rotation.from_quat(rot).as_matrix() @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    expected_pos = pose[:3] + expected_delta * env.action_scale[0]
    assert np.allclose(last_command[:3], expected_pos, atol=1e-6)
    assert np.allclose(last_command[3:], pose[3:], atol=1e-6)


def test_relative_frame_transforms_tcp_vel_to_body_frame() -> None:
    """
    Verify RelativeFrame rotates tcp_vel into the end-effector frame on reset.
    """
    # Give the TCP a rotated pose so the wrapper needs to rotate velocities into body frame.
    rot = Rotation.from_euler("xyz", [0.0, 0.0, np.pi / 2]).as_quat()
    pose = np.array([0.0, 0.0, 0.0, *rot], dtype=np.float32)
    tcp_vel = np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)
    adapter = MockRobotAdapter(_make_snapshot_with_state(pose, tcp_vel))
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=RelativeFrameTestConfig(), adapter=adapter)
    wrapped = RelativeFrame(env, include_relative_pose=False)

    obs, _ = wrapped.reset()

    # The wrapper uses the inverse transform to rotate into the body frame.
    transform_inv = np.linalg.inv(construct_transform_matrix(pose))
    expected = transform_inv @ tcp_vel
    assert np.allclose(obs["state"]["tcp_vel"], expected, atol=1e-6)


def test_relative_frame_relative_pose_after_reset() -> None:
    """
    Verify relative tcp_pose is expressed in the reset frame after stepping.
    """
    # Initialize at a non-zero reset pose so the relative frame should be identity at reset.
    reset_pose = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter = MockRobotAdapter(_make_snapshot_with_state(reset_pose, np.zeros((6,))))
    config = RelativeFrameTestConfig()
    config.RESET_POSE = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=config, adapter=adapter)
    wrapped = RelativeFrame(env, include_relative_pose=True)

    obs, _ = wrapped.reset()
    # At reset, relative translation should be zero because base and reset coincide.
    assert np.allclose(obs["state"]["tcp_pose"][:3], np.zeros((3,)), atol=1e-6)

    # Move +0.1 in base x; relative frame should report +0.1 in its x.
    new_pose = reset_pose.copy()
    new_pose[0] += 0.1
    updated = _make_snapshot_with_state(new_pose, np.zeros((6,)))
    adapter.update_snapshot(
        state=updated["state"],
        images=updated["images"],
        timestamp=updated["timestamp"],
    )

    obs, _, _, _, _ = wrapped.step(np.zeros((7,), dtype=np.float32))
    assert np.allclose(obs["state"]["tcp_pose"][:3], np.array([0.1, 0.0, 0.0]), atol=1e-6)
    assert np.allclose(obs["state"]["tcp_pose"][3:], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6)


class _InjectInterveneAction(gym.Wrapper):
    """
    Wrapper that injects a fixed intervene_action into env.step info.
    """

    def __init__(self, env: gym.Env, intervene_action: np.ndarray) -> None:
        super().__init__(env)
        self._intervene_action = np.array(intervene_action, dtype=np.float32)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Insert a copy so tests can compare values without mutation side effects.
        info["intervene_action"] = self._intervene_action.copy()
        return obs, reward, done, truncated, info


def test_relative_frame_transforms_intervene_action() -> None:
    """
    Verify intervene_action is converted back into the body frame.
    """
    # Use a rotated pose so the wrapper must rotate intervention signals into body frame.
    rot = Rotation.from_euler("xyz", [0.0, 0.0, np.pi / 2]).as_quat()
    pose = np.array([0.0, 0.0, 0.0, *rot], dtype=np.float32)
    adapter = MockRobotAdapter(_make_snapshot_with_state(pose, np.zeros((6,))))
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=RelativeFrameTestConfig(), adapter=adapter)
    # Intervene action is in base frame; wrapper should transform it into body frame.
    intervene_action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25], dtype=np.float32)
    injected = _InjectInterveneAction(env, intervene_action=intervene_action)
    wrapped = RelativeFrame(injected, include_relative_pose=False)
    wrapped.reset()

    _, _, _, _, info = wrapped.step(np.zeros((7,), dtype=np.float32))

    assert "intervene_action" in info
    # Rotate translation/rotation-vector components into the body frame.
    expected_translation = Rotation.from_quat(rot).as_matrix().T @ intervene_action[:3]
    expected_rotation = Rotation.from_quat(rot).as_matrix().T @ intervene_action[3:6]
    # Gripper stays as-is; only the 6D spatial component is rotated.
    expected = np.concatenate([expected_translation, expected_rotation, intervene_action[6:]])
    assert np.allclose(info["intervene_action"], expected, atol=1e-6)


def test_relative_frame_intervene_action_gripper_passthrough() -> None:
    """
    Verify the gripper component is not transformed by RelativeFrame.
    """
    # Any rotation should leave the gripper component unchanged.
    rot = Rotation.from_euler("xyz", [0.0, 0.0, np.pi / 4]).as_quat()
    pose = np.array([0.0, 0.0, 0.0, *rot], dtype=np.float32)
    adapter = MockRobotAdapter(_make_snapshot_with_state(pose, np.zeros((6,))))
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=RelativeFrameTestConfig(), adapter=adapter)
    intervene_action = np.array([0.1, 0.2, 0.3, -0.1, 0.2, -0.3, -0.75], dtype=np.float32)
    injected = _InjectInterveneAction(env, intervene_action=intervene_action)
    wrapped = RelativeFrame(injected, include_relative_pose=False)
    wrapped.reset()

    _, _, _, _, info = wrapped.step(np.zeros((7,), dtype=np.float32))

    assert np.isclose(info["intervene_action"][6], intervene_action[6], atol=1e-6)
