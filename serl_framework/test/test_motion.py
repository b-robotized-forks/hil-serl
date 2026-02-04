"""Unit tests for shared motion helpers."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from serl_framework.utils.motion import move_to_pose_interpolated


def test_reaches_goal_with_bounded_steps() -> None:
    """Each step should advance by at most delta_m toward the goal."""
    current_pose = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    commands: list[np.ndarray] = []

    def read_current_pose() -> np.ndarray:
        return current_pose.copy()

    def send_pose_command(pose: np.ndarray) -> None:
        nonlocal current_pose
        commands.append(np.array(pose, copy=True))
        current_pose = np.array(pose, copy=True)

    result = move_to_pose_interpolated(
        goal=np.array([0.025, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        timeout=0.5,
        in_euler=False,
        delta_m=0.01,
        delta_rad=0.1,
        hz=100.0,
        read_current_pose=read_current_pose,
        send_pose_command=send_pose_command,
        clip_pose=lambda pose: np.array(pose, dtype=np.float32),
    )

    assert result.reached is True
    assert commands
    if len(commands) > 1:
        step_norms = np.linalg.norm(np.diff(np.array(commands)[:, :3], axis=0), axis=1)
        assert np.all(step_norms <= 0.01 + 1e-6)


def test_reports_timeout_without_progress() -> None:
    """Should return a non-reaching result when the TCP never moves."""
    current_pose = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    result = move_to_pose_interpolated(
        goal=np.array([0.05, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        timeout=0.05,
        in_euler=False,
        delta_m=0.01,
        delta_rad=0.1,
        hz=50.0,
        read_current_pose=lambda: current_pose.copy(),
        send_pose_command=lambda pose: None,
        clip_pose=lambda pose: np.array(pose, dtype=np.float32),
    )

    assert result.reached is False
    assert result.pos_error_m > 0.0


def test_longer_move_produces_more_commands() -> None:
    """A longer distance should produce more commands than a shorter one."""
    commands_short: list[np.ndarray] = []
    commands_long: list[np.ndarray] = []

    def _run(distance: float, commands_out: list):
        pose = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        def read():
            return pose.copy()

        def send(p):
            nonlocal pose
            commands_out.append(np.array(p, copy=True))
            pose = np.array(p, copy=True)

        move_to_pose_interpolated(
            goal=np.array([distance, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            timeout=5.0,
            in_euler=False,
            delta_m=0.01,
            delta_rad=0.1,
            hz=100.0,
            read_current_pose=read,
            send_pose_command=send,
            clip_pose=lambda p: np.array(p, dtype=np.float32),
        )

    _run(0.02, commands_short)
    _run(0.08, commands_long)
    assert len(commands_long) > len(commands_short)


def test_rotation_independent_of_translation() -> None:
    """Rotation should advance at its own delta_rad rate, independent of translation."""
    current_pose = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    commands: list[np.ndarray] = []

    def read_current_pose() -> np.ndarray:
        return current_pose.copy()

    def send_pose_command(pose: np.ndarray) -> None:
        nonlocal current_pose
        commands.append(np.array(pose, copy=True))
        current_pose = np.array(pose, copy=True)

    goal_quat = Rotation.from_euler("xyz", [0.0, 0.0, 0.4]).as_quat()
    goal = np.concatenate([[0.04, 0.0, 0.3], goal_quat]).astype(np.float32)

    result = move_to_pose_interpolated(
        goal=goal,
        timeout=1.0,
        in_euler=False,
        delta_m=0.01,
        delta_rad=0.05,
        hz=100.0,
        read_current_pose=read_current_pose,
        send_pose_command=send_pose_command,
        clip_pose=lambda pose: np.array(pose, dtype=np.float32),
    )

    assert result.reached is True
    # 0.04m / 0.01 delta_m = 4 pos steps, 0.4rad / 0.05 delta_rad = 8 rot steps
    # Total steps driven by whichever finishes last (rotation).
    assert len(commands) >= 7


def test_pure_rotation() -> None:
    """A pure-rotation move should converge using delta_rad."""
    current_pose = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    commands: list[np.ndarray] = []

    def read_current_pose() -> np.ndarray:
        return current_pose.copy()

    def send_pose_command(pose: np.ndarray) -> None:
        nonlocal current_pose
        commands.append(np.array(pose, copy=True))
        current_pose = np.array(pose, copy=True)

    goal_quat = Rotation.from_euler("xyz", [0.0, 0.0, 0.1]).as_quat()
    goal = np.concatenate([[0.0, 0.0, 0.3], goal_quat]).astype(np.float32)

    result = move_to_pose_interpolated(
        goal=goal,
        timeout=1.0,
        in_euler=False,
        delta_m=0.01,
        delta_rad=0.02,
        hz=100.0,
        read_current_pose=read_current_pose,
        send_pose_command=send_pose_command,
        clip_pose=lambda pose: np.array(pose, dtype=np.float32),
    )

    assert result.reached is True
    # 0.1 rad / 0.02 delta_rad = 5 steps
    assert len(commands) >= 4


def test_sim_time_pacing() -> None:
    """Pacing should use the now-clock, not wall-clock sleep duration."""
    current_pose = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    sim_time = [0.0]
    sleep_calls = [0]

    def fake_now() -> float:
        return sim_time[0]

    def fake_sleep(dt: float) -> None:
        sleep_calls[0] += 1
        sim_time[0] += 0.01

    def send_pose_command(pose: np.ndarray) -> None:
        nonlocal current_pose
        current_pose = np.array(pose, copy=True)

    result = move_to_pose_interpolated(
        goal=np.array([0.02, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        timeout=5.0,
        in_euler=False,
        delta_m=0.01,
        delta_rad=0.1,
        hz=10.0,
        read_current_pose=lambda: current_pose.copy(),
        send_pose_command=send_pose_command,
        clip_pose=lambda pose: np.array(pose, dtype=np.float32),
        now=fake_now,
        sleep=fake_sleep,
    )

    assert result.reached is True
    assert sleep_calls[0] > 0
