"""
Unit tests for the RobotEnv integration with a mock adapter.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional
import tempfile

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from serl_framework.envs.robot_env import DefaultEnvConfig, RobotEnv
from serl_framework.testing.mock_adapter import MockRobotAdapter, make_mock_snapshot
from serl_framework.utils.rotations import euler2quat, quat2euler

class TestConfig(DefaultEnvConfig):
    """
    Minimal config for exercising RobotEnv in tests.
    """

    ROBOT_DOF = 7
    CAMERAS = {"front": {}}
    IMAGE_CROP = {"front": lambda img: img[10:110, 20:120]}
    IMAGE_CHANNEL_ORDER = "rgb"
    DISPLAY_IMAGE = False
    ENABLE_KEYBOARD_LISTENER = False
    ACTION_SCALE = (0.01, 0.01, 1.0)
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    TARGET_POSE = np.zeros((6,))
    RESET_POSE = np.zeros((6,))
    REWARD_THRESHOLD = np.ones((6,))


def _make_snapshot() -> Dict[str, Any]:
    """
    Create a synthetic observation snapshot for tests.
    """
    return make_mock_snapshot(["front"], 7)


def test_robot_env_step_produces_observation() -> None:
    """
    Verify that RobotEnv returns observations with expected keys and shapes.
    """
    adapter = MockRobotAdapter(_make_snapshot())
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=TestConfig(), adapter=adapter)

    obs, reward, done, truncated, info = env.step(np.zeros((7,), dtype=np.float32))

    assert "state" in obs
    assert "images" in obs
    assert obs["images"]["front"].shape == (128, 128, 3)
    assert obs["state"]["q"].shape == (7,)
    assert obs["state"]["dq"].shape == (7,)
    assert isinstance(reward, int)
    assert isinstance(done, bool)
    assert truncated is False
    assert "succeed" in info


class RewardTestConfig(DefaultEnvConfig):
    """
    Config for verifying reward behavior on updated adapter snapshots.
    """

    ROBOT_DOF = 7
    CAMERAS = {"front": {}}
    IMAGE_CROP = {}
    IMAGE_CHANNEL_ORDER = "rgb"
    DISPLAY_IMAGE = False
    ENABLE_KEYBOARD_LISTENER = False
    ACTION_SCALE = (0.01, 0.01, 1.0)
    ABS_POSE_LIMIT_LOW = np.array([-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi])
    ABS_POSE_LIMIT_HIGH = np.array([1.0, 1.0, 1.0, np.pi, np.pi, np.pi])
    TARGET_POSE = np.array([0.2, -0.1, 0.1, 0.0, 0.0, 0.0])
    RESET_POSE = np.zeros((6,))
    REWARD_THRESHOLD = np.ones((6,)) * 0.01


def test_robot_env_step_updates_state_and_reward() -> None:
    """
    Verify that updated adapter snapshots change observations and rewards.
    """
    config = RewardTestConfig()
    initial_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter = MockRobotAdapter(make_mock_snapshot(["front"], config.ROBOT_DOF, tcp_pose=initial_pose))
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=config, adapter=adapter)

    obs, reward, _, _, _ = env.step(np.zeros((7,), dtype=np.float32))
    assert np.allclose(obs["state"]["tcp_pose"], initial_pose)
    assert reward == 0

    target_quat = Rotation.from_euler("xyz", config.TARGET_POSE[3:]).as_quat()
    target_pose = np.concatenate([config.TARGET_POSE[:3], target_quat]).astype(np.float32)
    updated = make_mock_snapshot(["front"], config.ROBOT_DOF, tcp_pose=target_pose)
    adapter.update_snapshot(
        state=updated["state"],
        images=updated["images"],
        timestamp=updated["timestamp"],
    )

    obs, reward, _, _, _ = env.step(np.zeros((7,), dtype=np.float32))
    assert np.allclose(obs["state"]["tcp_pose"][:3], config.TARGET_POSE[:3], atol=1e-6)
    assert reward == 1



def test_robot_env_step_reports_actual_action_delta_debug() -> None:
    """RobotEnv.step() should expose the physical delta magnitudes it applied."""
    adapter = MockRobotAdapter(_make_snapshot())
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=TestConfig(), adapter=adapter)

    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)

    action = np.array([2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)

    delta_debug = info["action_delta_debug"]
    assert delta_debug is not None
    assert delta_debug["action_norm"] == pytest.approx(np.sqrt(2.0))
    assert delta_debug["trans_m"] == pytest.approx(0.01)
    assert delta_debug["rot_rad"] == pytest.approx(0.01)
    assert env.last_action_delta_debug == delta_debug


class ResetTestConfig(DefaultEnvConfig):
    """
    Config for testing reset flow with non-zero poses.
    """

    ROBOT_DOF = 7
    CAMERAS = {"front": {}}
    IMAGE_CROP = {}
    IMAGE_CHANNEL_ORDER = "rgb"
    DISPLAY_IMAGE = False
    ENABLE_KEYBOARD_LISTENER = False
    ACTION_SCALE = (0.01, 0.01, 1.0)
    # Non-trivial pose limits to exercise clipping
    ABS_POSE_LIMIT_LOW = np.array([-1.0, -1.0, 0.0, -np.pi, -np.pi / 2, -np.pi])
    ABS_POSE_LIMIT_HIGH = np.array([1.0, 1.0, 1.0, np.pi, np.pi / 2, np.pi])
    # Non-zero reset pose with euler angles
    RESET_POSE = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
    TARGET_POSE = np.array([0.5, 0.0, 0.2, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD = np.ones((6,)) * 0.1
    COMPLIANCE_PARAM = {"stiffness": 100.0}
    PRECISION_PARAM = {"stiffness": 500.0}
    JOINT_RESET_PERIOD = 0
    RANDOM_RESET = False
    RESET_MOVE_TIMEOUT_SEC = 0.25


def _make_reset_snapshot(tcp_pose: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Create a snapshot with a specified TCP pose for reset tests.
    """
    if tcp_pose is None:
        tcp_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return make_mock_snapshot(["front"], 7, tcp_pose=tcp_pose)


class TrackingMockRobotAdapter(MockRobotAdapter):
    """
    Mock adapter that mirrors commanded poses into the cached TCP state.

    This lets reset/interpolation tests exercise convergence behavior without a
    real controller loop.
    """

    def send_pose_command(self, pose: list[float]) -> None:
        super().send_pose_command(pose)
        if self.latest_state is None:
            return
        next_state = dict(self.latest_state)
        next_state["tcp_pose"] = np.array(pose, dtype=np.float32)
        self.update_snapshot(state=next_state)


class CaptureResetTimeoutEnv(RobotEnv):
    """
    RobotEnv test double that records the timeout passed to interpolate_move().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interpolate_calls: list[dict[str, object]] = []

    def interpolate_move(self, goal: np.ndarray, timeout: float, in_euler: bool) -> None:
        self.interpolate_calls.append(
            {
                "goal": np.array(goal, copy=True),
                "timeout": float(timeout),
                "in_euler": bool(in_euler),
            }
        )
        self.nextpos = np.array(goal, copy=True)


class _ResetCountingTeleop:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self, obs=None, info=None) -> None:
        self.reset_calls += 1


class OptionalResetPoseConfig(DefaultEnvConfig):
    ROBOT_DOF = 7
    CAMERAS = {"front": {}}
    IMAGE_CROP = {}
    IMAGE_CHANNEL_ORDER = "rgb"
    DISPLAY_IMAGE = False
    ENABLE_KEYBOARD_LISTENER = False
    ACTION_SCALE = (0.01, 0.01, 1.0)
    ABS_POSE_LIMIT_LOW = np.array([-1.0, -1.0, 0.0, -np.pi, -np.pi / 2, -np.pi])
    ABS_POSE_LIMIT_HIGH = np.array([1.0, 1.0, 1.0, np.pi, np.pi / 2, np.pi])
    TARGET_POSE = np.zeros((6,))
    GRASP_POSE = np.zeros((6,))
    RESET_POSE = None
    REWARD_THRESHOLD = np.ones((6,)) * 0.1


class CustomResetEnv(RobotEnv):
    def go_to_reset(self, joint_reset: bool = False) -> None:
        del joint_reset
        self._set_task_params()


def test_robot_env_reset_returns_observation() -> None:
    """
    Verify that RobotEnv.reset() returns a valid observation and exercises
    the quaternion conversion code paths.
    """
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=ResetTestConfig(), adapter=adapter)

    obs, info = env.reset()

    assert "state" in obs
    assert "images" in obs
    assert obs["state"]["tcp_pose"].shape == (7,)
    assert "succeed" in info
    assert info["succeed"] is False

    # Verify services were called during reset
    service_names = [call["service_name"] for call in adapter.service_calls]
    assert "set_compliance" in service_names
    assert "clear_error" in service_names


def test_go_to_reset_uses_configured_reset_move_timeout() -> None:
    """
    Verify go_to_reset forwards RESET_MOVE_TIMEOUT_SEC into interpolate_move.
    """
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    env = CaptureResetTimeoutEnv(
        hz=100, fake_env=False, save_video=False, config=ResetTestConfig(), adapter=adapter,
    )

    env.go_to_reset(joint_reset=False)

    assert env.interpolate_calls, "Expected go_to_reset to call interpolate_move."
    assert env.interpolate_calls[-1]["timeout"] == ResetTestConfig.RESET_MOVE_TIMEOUT_SEC


def test_go_to_reset_resets_teleop_adapter() -> None:
    """
    Verify go_to_reset resets the configured teleop adapter after moving home.
    """
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    env = CaptureResetTimeoutEnv(
        hz=100, fake_env=False, save_video=False, config=ResetTestConfig(), adapter=adapter,
    )
    teleop = _ResetCountingTeleop()
    env._teleop_adapter = teleop

    env.go_to_reset(joint_reset=False)

    assert teleop.reset_calls == 1


def test_robot_env_can_construct_without_reset_pose_when_subclass_overrides_reset() -> None:
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    env = CustomResetEnv(
        hz=100,
        fake_env=False,
        save_video=False,
        config=OptionalResetPoseConfig(),
        adapter=adapter,
    )

    assert env.get_fixed_reset_pose_euler() is None
    assert env.resetpos is None

    obs, info = env.reset()

    assert "state" in obs
    assert info["succeed"] is False


def test_robot_env_go_to_reset_requires_fixed_reset_pose_by_default() -> None:
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    env = RobotEnv(
        hz=100,
        fake_env=False,
        save_video=False,
        config=OptionalResetPoseConfig(),
        adapter=adapter,
    )

    with pytest.raises(RuntimeError, match="does not define a fixed reset pose"):
        env.go_to_reset(joint_reset=False)


def test_robot_env_reset_quaternion_consistency() -> None:
    """
    Verify that quaternions are consistent between resetpos and currpos.

    This test catches the wxyz vs xyzw quaternion ordering bug by checking
    that clip_safety_box and interpolate_move don't produce invalid rotations.
    """
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    config = ResetTestConfig()
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=config, adapter=adapter)

    # The resetpos should have valid xyzw quaternion that scipy can parse
    resetpos_quat = env.resetpos[3:]
    assert resetpos_quat.shape == (4,), f"Expected 4D quaternion, got {resetpos_quat.shape}"

    # Verify the quaternion is valid (normalized, no NaN)
    assert not np.any(np.isnan(resetpos_quat)), "Quaternion contains NaN"
    quat_norm = np.linalg.norm(resetpos_quat)
    assert np.isclose(quat_norm, 1.0, atol=1e-6), f"Quaternion not normalized: {quat_norm}"

    # Verify scipy can parse it without error (this would fail with wxyz ordering)
    try:
        rot = Rotation.from_quat(resetpos_quat)
        euler = rot.as_euler("xyz")
    except Exception as e:
        raise AssertionError(f"scipy.Rotation.from_quat failed on resetpos quaternion: {e}")

    # Verify clip_safety_box doesn't corrupt the pose
    test_pose = env.resetpos.copy()
    clipped = env.clip_safety_box(test_pose)
    clipped_quat = clipped[3:]
    clipped_norm = np.linalg.norm(clipped_quat)
    assert np.isclose(clipped_norm, 1.0, atol=1e-6), f"Clipped quaternion not normalized: {clipped_norm}"


def test_robot_env_step_quaternion_rotation() -> None:
    """
    Verify that step() correctly applies rotation deltas using quaternions.

    This exercises the Rotation.from_quat(self.currpos[3:]) code path.
    """
    # Start with identity quaternion [0, 0, 0, 1] in xyzw format
    initial_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter = MockRobotAdapter(_make_reset_snapshot(initial_pose))
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=ResetTestConfig(), adapter=adapter)

    # Apply a small rotation action
    action = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)

    # Verify pose commands were sent
    assert len(adapter.pose_commands) > 0, "No pose commands were sent"

    # Verify the commanded pose has a valid quaternion
    last_command = adapter.pose_commands[-1]
    cmd_quat = np.array(last_command[3:])
    cmd_norm = np.linalg.norm(cmd_quat)
    assert np.isclose(cmd_norm, 1.0, atol=1e-6), f"Command quaternion not normalized: {cmd_norm}"

    # Verify scipy can parse the commanded quaternion
    try:
        Rotation.from_quat(cmd_quat)
    except Exception as e:
        raise AssertionError(f"scipy.Rotation.from_quat failed on command quaternion: {e}")


def test_interpolate_move_uses_action_scale_sized_steps() -> None:
    """
    Verify interpolate_move advances toward the goal using bounded deltas.

    The delta is the Euclidean norm of the translational action scale
    (ACTION_SCALE[0] * sqrt(3)), so each step's displacement must not exceed it.
    """
    start_pose = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    goal_pose = np.array([0.035, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot(start_pose))
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=ResetTestConfig(), adapter=adapter)

    env.interpolate_move(goal_pose, timeout=0.1, in_euler=False)

    assert adapter.pose_commands, "Expected interpolate_move to send pose commands."
    commanded = np.array(adapter.pose_commands, dtype=np.float32)
    step_norms = np.linalg.norm(np.diff(commanded[:, :3], axis=0), axis=1)
    delta = ResetTestConfig.ACTION_SCALE[0] * np.sqrt(3)
    if step_norms.size > 0:
        assert np.all(step_norms <= delta + 1e-6)
    np.testing.assert_allclose(env.currpos[:3], goal_pose[:3], atol=3e-3)


def test_interpolate_move_warns_instead_of_raising_when_goal_not_reached() -> None:
    """
    Verify interpolate_move remains non-fatal when the TCP never catches up.
    """
    start_pose = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    goal_pose = np.array([0.05, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter = MockRobotAdapter(_make_reset_snapshot(start_pose))
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=ResetTestConfig(), adapter=adapter)

    env.interpolate_move(goal_pose, timeout=0.1, in_euler=False)

    assert adapter.pose_commands, "Expected interpolate_move to issue commands even when stalled."
    np.testing.assert_allclose(env.nextpos, goal_pose, atol=1e-6)


# --- Rotation utility tests ---


def test_euler2quat_round_trip() -> None:
    """
    Verify that euler2quat and quat2euler are inverse operations.
    """
    # Test with various euler angles (roll, pitch, yaw)
    test_cases = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.2, 0.3]),
        np.array([-0.5, 0.0, 0.5]),
        np.array([np.pi / 4, -np.pi / 6, np.pi / 3]),
    ]

    for rpy in test_cases:
        quat = euler2quat(rpy)
        rpy_recovered = quat2euler(quat)
        assert np.allclose(rpy, rpy_recovered, atol=1e-10), (
            f"Round-trip failed for {rpy}: got {rpy_recovered}"
        )


def test_euler2quat_produces_normalized_quaternion() -> None:
    """
    Verify that euler2quat always produces unit quaternions.
    """
    test_cases = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.5, -0.3]),
        np.array([np.pi, np.pi / 2, -np.pi]),
    ]

    for rpy in test_cases:
        quat = euler2quat(rpy)
        assert quat.shape == (4,), f"Expected 4D quaternion, got {quat.shape}"
        norm = np.linalg.norm(quat)
        assert np.isclose(norm, 1.0, atol=1e-10), (
            f"Quaternion not normalized for {rpy}: norm={norm}"
        )


def test_euler2quat_xyzw_format() -> None:
    """
    Verify that euler2quat returns quaternions in xyzw (scalar-last) format.
    """

    rpy = np.array([0.1, 0.2, 0.3])
    quat = euler2quat(rpy)

    # scipy's from_euler also uses xyzw format, so they should match exactly
    expected = Rotation.from_euler("xyz", rpy).as_quat()
    assert np.allclose(quat, expected, atol=1e-10), (
        f"Quaternion format mismatch: got {quat}, expected {expected}"
    )


# --- Relative workspace tests ---


class RelWorkspaceConfig(DefaultEnvConfig):
    """Config for testing relative workspace limits."""

    ROBOT_DOF = 7
    CAMERAS = {"front": {}}
    IMAGE_CROP = {}
    IMAGE_CHANNEL_ORDER = "rgb"
    DISPLAY_IMAGE = False
    ENABLE_KEYBOARD_LISTENER = False
    ACTION_SCALE = (0.01, 0.01, 1.0)
    ABS_POSE_LIMIT_LOW = np.array([-1.0, -1.0, 0.0, -np.pi, -np.pi, -np.pi])
    ABS_POSE_LIMIT_HIGH = np.array([1.0, 1.0, 1.0, np.pi, np.pi, np.pi])
    RESET_POSE = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
    TARGET_POSE = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD = np.ones((6,)) * 0.1
    COMPLIANCE_PARAM = {}
    PRECISION_PARAM = {}
    JOINT_RESET_PERIOD = 0
    RANDOM_RESET = False
    RESET_MOVE_TIMEOUT_SEC = 0.1
    REL_POSE_LIMIT_LOW = np.array([-0.10, -0.10, -0.20])
    REL_POSE_LIMIT_HIGH = np.array([0.10, 0.10, 0.05])


class NoRelWorkspaceConfig(RelWorkspaceConfig):
    """Same as above but without relative limits."""

    REL_POSE_LIMIT_LOW = None
    REL_POSE_LIMIT_HIGH = None


class RelWorkspace6DConfig(RelWorkspaceConfig):
    """Same as above but with relative XYZ + RPY limits."""

    REL_POSE_LIMIT_LOW = np.array([-0.10, -0.10, -0.20, -0.20, -0.10, -0.30])
    REL_POSE_LIMIT_HIGH = np.array([0.10, 0.10, 0.05, 0.20, 0.10, 0.30])


def test_no_relative_limits_uses_absolute_workspace() -> None:
    """Without relative limits, effective workspace equals absolute."""
    tcp = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot(tcp))
    config = NoRelWorkspaceConfig()
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=config, adapter=adapter)

    env.reset()

    np.testing.assert_array_equal(env._effective_xyz_low, config.ABS_POSE_LIMIT_LOW[:3])
    np.testing.assert_array_equal(env._effective_xyz_high, config.ABS_POSE_LIMIT_HIGH[:3])


def test_relative_limits_centered_on_post_reset_tcp() -> None:
    """After reset, effective workspace is centered on currpos[:3]."""
    tcp = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot(tcp))
    config = RelWorkspaceConfig()
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=config, adapter=adapter)

    env.reset()

    # Expected: intersection of absolute and (tcp[:3] + rel)
    expected_low = np.maximum(config.ABS_POSE_LIMIT_LOW[:3], tcp[:3] + config.REL_POSE_LIMIT_LOW)
    expected_high = np.minimum(config.ABS_POSE_LIMIT_HIGH[:3], tcp[:3] + config.REL_POSE_LIMIT_HIGH)
    np.testing.assert_allclose(env._effective_xyz_low, expected_low, atol=1e-6)
    np.testing.assert_allclose(env._effective_xyz_high, expected_high, atol=1e-6)


def test_relative_limits_clamped_by_absolute_fence() -> None:
    """Effective box cannot exceed the absolute safety fence."""
    # Place TCP near the edge of the absolute workspace.
    tcp = np.array([0.95, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot(tcp))
    config = RelWorkspaceConfig()
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=config, adapter=adapter)

    env.reset()

    # x: tcp=0.95, rel_high=+0.10 => candidate=1.05, abs_high=1.0 => clamped to 1.0
    assert env._effective_xyz_high[0] <= config.ABS_POSE_LIMIT_HIGH[0] + 1e-9
    # x: tcp=0.95, rel_low=-0.10 => candidate=0.85, abs_low=-1.0 => 0.85
    assert env._effective_xyz_low[0] >= config.ABS_POSE_LIMIT_LOW[0] - 1e-9


def test_invalid_intersection_falls_back_to_absolute() -> None:
    """If relative envelope doesn't intersect absolute fence, fall back."""
    # Place TCP completely outside the absolute workspace.
    tcp = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot(tcp))
    config = RelWorkspaceConfig()
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=config, adapter=adapter)

    env._update_effective_workspace(tcp)

    # Should fall back to absolute limits.
    np.testing.assert_array_equal(env._effective_xyz_low, config.ABS_POSE_LIMIT_LOW[:3])
    np.testing.assert_array_equal(env._effective_xyz_high, config.ABS_POSE_LIMIT_HIGH[:3])


def test_mixed_relative_limit_shapes_raise_value_error() -> None:
    """Relative low/high limits must use the same dimensionality."""

    class BadShapeConfig(RelWorkspaceConfig):
        REL_POSE_LIMIT_LOW = np.array([-0.1, -0.1, -0.2, 0.0, 0.0, 0.0])

    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    with pytest.raises(ValueError, match="must have the same shape"):
        RobotEnv(
            hz=100,
            fake_env=False,
            save_video=False,
            config=BadShapeConfig(),
            adapter=adapter,
        )


def test_invalid_relative_limit_shape_raises_value_error() -> None:
    """Relative limits must be either 3-D XYZ or 6-D XYZ+RPY offsets."""

    class BadShapeConfig(RelWorkspaceConfig):
        REL_POSE_LIMIT_LOW = np.array([-0.1, -0.1, -0.2, 0.0])
        REL_POSE_LIMIT_HIGH = np.array([0.1, 0.1, 0.05, 0.2])

    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    with pytest.raises(ValueError, match="3-element \\(XYZ\\) or 6-element"):
        RobotEnv(
            hz=100,
            fake_env=False,
            save_video=False,
            config=BadShapeConfig(),
            adapter=adapter,
        )


def test_invalid_relative_limit_order_raises_value_error() -> None:
    """Relative low/high limits must be strictly ordered on each axis."""

    class BadOrderConfig(RelWorkspaceConfig):
        REL_POSE_LIMIT_LOW = np.array([0.1, -0.1, -0.2])
        REL_POSE_LIMIT_HIGH = np.array([0.05, 0.1, 0.05])

    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    with pytest.raises(ValueError, match="strictly less than"):
        RobotEnv(
            hz=100,
            fake_env=False,
            save_video=False,
            config=BadOrderConfig(),
            adapter=adapter,
        )


def test_invalid_relative_limit_order_raises_value_error_for_rpy() -> None:
    """Relative 6-D limits must also be strictly ordered on the RPY axes."""

    class BadOrder6DConfig(RelWorkspace6DConfig):
        REL_POSE_LIMIT_LOW = np.array([-0.1, -0.1, -0.2, 0.2, -0.1, -0.3])
        REL_POSE_LIMIT_HIGH = np.array([0.1, 0.1, 0.05, 0.1, 0.1, 0.3])

    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    with pytest.raises(ValueError, match="XYZ\\+RPY axes"):
        RobotEnv(
            hz=100,
            fake_env=False,
            save_video=False,
            config=BadOrder6DConfig(),
            adapter=adapter,
        )


def test_default_env_config_from_yaml_loads_relative_workspace_arrays() -> None:
    """YAML overrides should parse relative workspace arrays end-to-end."""
    yaml_text = """
hz: 20
target_pose: [0, 0, 0, 0, 0, 0]
grasp_pose: [0, 0, 0, 0, 0, 0]
reward_threshold: [1, 1, 1, 1, 1, 1]
action_scale: [0.01, 0.01, 1.0]
reset_pose: [0, 0, 0, 0, 0, 0]
abs_pose_limit_low: [-1, -1, 0, -3.14, -3.14, -3.14]
abs_pose_limit_high: [1, 1, 1, 3.14, 3.14, 3.14]
rel_pose_limit_low: [-0.10, -0.10, -0.20]
rel_pose_limit_high: [0.10, 0.10, 0.05]
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(yaml_text)
        path = handle.name

    try:
        cfg = DefaultEnvConfig.from_yaml(path)
    finally:
        Path(path).unlink(missing_ok=True)

    np.testing.assert_allclose(cfg.REL_POSE_LIMIT_LOW, np.array([-0.10, -0.10, -0.20]))
    np.testing.assert_allclose(cfg.REL_POSE_LIMIT_HIGH, np.array([0.10, 0.10, 0.05]))


def test_clip_safety_box_uses_effective_workspace() -> None:
    """clip_safety_box() should use the narrowed effective XYZ limits."""
    tcp = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot(tcp))
    config = RelWorkspaceConfig()
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=config, adapter=adapter)

    env.reset()

    # Try to clip a pose far outside the relative envelope.
    far_pose = np.array([0.9, 0.5, 0.8, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    clipped = env.clip_safety_box(far_pose.copy())

    # Translation should be clamped to the effective (relative) envelope.
    assert clipped[0] <= tcp[0] + config.REL_POSE_LIMIT_HIGH[0] + 1e-6
    assert clipped[1] <= tcp[1] + config.REL_POSE_LIMIT_HIGH[1] + 1e-6
    assert clipped[2] <= tcp[2] + config.REL_POSE_LIMIT_HIGH[2] + 1e-6


def test_clip_safety_box_rotation_uses_relative_rpy_limits_when_configured() -> None:
    """Rotation clipping should use post-reset-relative base-frame RPY bounds."""
    tcp_quat = Rotation.from_euler("xyz", [0.4, -0.2, 0.3]).as_quat()
    tcp = np.concatenate([[0.5, 0.0, 0.3], tcp_quat]).astype(np.float32)
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot(tcp))
    config = RelWorkspace6DConfig()
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=config, adapter=adapter)

    env.reset()

    # Create a pose outside the relative post-reset RPY window.
    large_rot_quat = Rotation.from_euler("xyz", [1.2, -0.6, 1.1]).as_quat()
    test_pose = np.concatenate([tcp[:3], large_rot_quat]).astype(np.float32)
    clipped = env.clip_safety_box(test_pose.copy())
    clipped_euler = Rotation.from_quat(clipped[3:]).as_euler("xyz")
    ref_rpy = Rotation.from_quat(tcp[3:]).as_euler("xyz")

    expected_low = np.maximum(
        config.ABS_POSE_LIMIT_LOW[3:],
        ref_rpy + config.REL_POSE_LIMIT_LOW[3:],
    )
    expected_high = np.minimum(
        config.ABS_POSE_LIMIT_HIGH[3:],
        ref_rpy + config.REL_POSE_LIMIT_HIGH[3:],
    )

    assert np.all(clipped_euler >= expected_low - 1e-6)
    assert np.all(clipped_euler <= expected_high + 1e-6)


def test_clip_safety_box_rotation_uses_absolute_limits_when_relative_is_xyz_only() -> None:
    """XYZ-only relative limits should leave rotation on the absolute fence."""
    tcp = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot(tcp))
    config = RelWorkspaceConfig()
    env = RobotEnv(hz=100, fake_env=False, save_video=False, config=config, adapter=adapter)

    env.reset()

    large_rot_quat = Rotation.from_euler("xyz", [2.0, 2.0, 2.0]).as_quat()
    test_pose = np.concatenate([tcp[:3], large_rot_quat]).astype(np.float32)
    clipped = env.clip_safety_box(test_pose.copy())
    clipped_euler = Rotation.from_quat(clipped[3:]).as_euler("xyz")

    assert np.all(clipped_euler[1:] >= config.ABS_POSE_LIMIT_LOW[4:] - 1e-6)
    assert np.all(clipped_euler[1:] <= config.ABS_POSE_LIMIT_HIGH[4:] + 1e-6)


# ---------------------------------------------------------------------------
# Free-space action scale tests
# ---------------------------------------------------------------------------

class FreeActionScaleConfig(ResetTestConfig):
    """Config with FREE_ACTION_SCALE set."""

    FREE_ACTION_SCALE = (0.03, 0.03, 1.0)


def test_free_action_scale_fallback_when_none() -> None:
    """When FREE_ACTION_SCALE is None, _free_action_scale falls back to ACTION_SCALE."""
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    config = ResetTestConfig()  # FREE_ACTION_SCALE is not set (inherits None)
    env = RobotEnv(
        hz=100, fake_env=False, save_video=False, config=config, adapter=adapter,
    )
    assert tuple(env._free_action_scale) == tuple(config.ACTION_SCALE)
    assert tuple(env._task_action_scale) == tuple(config.ACTION_SCALE)


def test_free_action_scale_used_when_configured() -> None:
    """When FREE_ACTION_SCALE is set, _free_action_scale uses it."""
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    config = FreeActionScaleConfig()
    env = RobotEnv(
        hz=100, fake_env=False, save_video=False, config=config, adapter=adapter,
    )
    assert tuple(env._free_action_scale) == tuple(config.FREE_ACTION_SCALE)
    assert tuple(env._task_action_scale) == tuple(config.ACTION_SCALE)


def test_set_action_scale_updates_self_action_scale() -> None:
    """_set_action_scale should directly update self.action_scale."""
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    env = RobotEnv(
        hz=100, fake_env=False, save_video=False, config=ResetTestConfig(), adapter=adapter,
    )
    new_scale = (0.05, 0.05, 1.0)
    env._set_action_scale(new_scale)
    assert env.action_scale == new_scale


def test_go_to_reset_restores_task_action_scale() -> None:
    """go_to_reset() should switch to free-space scale, then restore task scale."""
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    config = FreeActionScaleConfig()
    env = CaptureResetTimeoutEnv(
        hz=100, fake_env=False, save_video=False, config=config, adapter=adapter,
    )
    # Before reset, action_scale is the task scale.
    assert tuple(env.action_scale) == tuple(config.ACTION_SCALE)

    env.go_to_reset(joint_reset=False)

    # After go_to_reset, action_scale should be restored to the task scale.
    assert tuple(env.action_scale) == tuple(config.ACTION_SCALE)


def test_reset_sets_task_action_scale() -> None:
    """reset() should ensure task-phase action scale is active."""
    adapter = TrackingMockRobotAdapter(_make_reset_snapshot())
    config = FreeActionScaleConfig()
    env = RobotEnv(
        hz=100, fake_env=False, save_video=False, config=config, adapter=adapter,
    )
    # Manually set to free-space scale to simulate leftover state.
    env._set_action_scale(config.FREE_ACTION_SCALE)
    assert tuple(env.action_scale) == tuple(config.FREE_ACTION_SCALE)

    env.reset()

    # After reset, action_scale should be the task scale.
    assert tuple(env.action_scale) == tuple(config.ACTION_SCALE)


def test_default_env_config_from_yaml_loads_free_action_scale() -> None:
    """YAML override for free_action_scale should be parsed as a numpy array."""
    yaml_text = """
action_scale: [0.01, 0.01, 1.0]
free_action_scale: [0.03, 0.03, 1.0]
reset_pose: [0, 0, 0, 0, 0, 0]
abs_pose_limit_low: [-1, -1, 0, -3.14, -3.14, -3.14]
abs_pose_limit_high: [1, 1, 1, 3.14, 3.14, 3.14]
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(yaml_text)
        path = handle.name

    try:
        cfg = DefaultEnvConfig.from_yaml(path)
    finally:
        Path(path).unlink(missing_ok=True)

    np.testing.assert_allclose(cfg.FREE_ACTION_SCALE, np.array([0.03, 0.03, 1.0]))


def test_default_env_config_from_yaml_free_action_scale_absent() -> None:
    """When free_action_scale is not in YAML, FREE_ACTION_SCALE stays None."""
    yaml_text = """
action_scale: [0.01, 0.01, 1.0]
reset_pose: [0, 0, 0, 0, 0, 0]
abs_pose_limit_low: [-1, -1, 0, -3.14, -3.14, -3.14]
abs_pose_limit_high: [1, 1, 1, 3.14, 3.14, 3.14]
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(yaml_text)
        path = handle.name

    try:
        cfg = DefaultEnvConfig.from_yaml(path)
    finally:
        Path(path).unlink(missing_ok=True)

    assert cfg.FREE_ACTION_SCALE is None
