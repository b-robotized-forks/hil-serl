"""Shared closed-loop pose motion helpers."""

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from serl_framework.utils.rotations import euler2quat


@dataclass
class InterpolatedMoveResult:
    """Result of a closed-loop interpolated move."""

    reached: bool
    goal_pose: np.ndarray
    final_pose: np.ndarray
    pos_error_m: float
    rot_error_rad: float
    last_commanded_pose: np.ndarray | None


def move_to_pose_interpolated(
    *,
    goal: np.ndarray,
    timeout: float,
    in_euler: bool,
    delta_m: float,
    delta_rad: float,
    hz: float,
    read_current_pose: Callable[[], np.ndarray],
    send_pose_command: Callable[[np.ndarray], None],
    clip_pose: Callable[[np.ndarray], np.ndarray],
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    pos_tol_m: float | None = None,
    rot_tol_rad: float | None = None,
) -> InterpolatedMoveResult:
    """Closed-loop linear point-to-point TCP move ("carrot on a stick").

    Each cycle reads the actual TCP pose, places a translational target
    ``delta_m`` ahead and a rotational target ``delta_rad`` ahead along
    the current-to-goal direction, and sends the result.  The arm chases
    these carrots at every tick, producing smooth, constant motion
    consistent with how HIL-SERL controls the arm in ``RobotEnv.step()``.

    Translation and rotation each advance independently at their own rate.

    Pacing uses the ``now`` clock (typically sim-time) with small ``sleep``
    polls, so motion speed is consistent in the simulation world regardless
    of the real-time factor.

    Args:
        goal: Target pose as 6-D euler or 7-D quaternion.
        timeout: Safety time budget in ``now``-clock seconds.
        in_euler: Whether ``goal`` is 6-D euler.
        delta_m: Translational carrot distance (metres).
        delta_rad: Rotational carrot distance (radians).
        hz: Control frequency for inter-step pacing.
        read_current_pose: Returns the latest 7-D quaternion TCP pose.
        send_pose_command: Sends a 7-D quaternion pose command.
        clip_pose: Clips a 7-D quaternion pose to safety limits.
        now: Clock function (should match the simulation clock).
        sleep: Micro-sleep for polling ``now`` between cycles.
        pos_tol_m: Optional override for the translational convergence
            tolerance (metres). When ``None`` (default), the tolerance is
            derived from ``delta_m`` as ``max(0.001, min(0.003, 0.5 * delta_m))``.
            Pass an explicit value to widen (e.g. when the controller has
            a large steady-state error against persistent disturbances such
            as gripped-payload weight) or tighten (e.g. for precise
            pre-contact alignment) the convergence criterion.
        rot_tol_rad: Optional override for the rotational convergence
            tolerance (radians). When ``None`` (default), the tolerance is
            derived from ``delta_rad`` as
            ``max(0.01, min(0.03, 0.5 * delta_rad))``.
    """
    if in_euler:
        goal = np.concatenate([goal[:3], euler2quat(goal[3:])])

    goal = clip_pose(np.array(goal, dtype=np.float32))
    start = np.array(read_current_pose(), dtype=np.float32)
    goal_rot = Rotation.from_quat(goal[3:])

    if delta_m <= 0.0:
        if float(np.linalg.norm(goal[:3] - start[:3])) > 0.0:
            raise RuntimeError(
                "ERROR: move_to_pose_interpolated requires delta_m > 0."
            )
    if delta_rad <= 0.0:
        if float((Rotation.from_quat(start[3:]).inv() * goal_rot).magnitude()) > 0.0:
            raise RuntimeError(
                "ERROR: move_to_pose_interpolated requires delta_rad > 0."
            )

    if pos_tol_m is None:
        pos_tol_m = max(0.001, min(0.003, 0.5 * delta_m)) if delta_m > 0.0 else 0.001
    else:
        pos_tol_m = float(pos_tol_m)
    if rot_tol_rad is None:
        rot_tol_rad = max(0.01, min(0.03, 0.5 * delta_rad)) if delta_rad > 0.0 else 0.01
    else:
        rot_tol_rad = float(rot_tol_rad)
    period = 1.0 / hz
    deadline = now() + max(timeout, period)
    last_commanded_pose = None

    while now() < deadline:
        current = np.array(read_current_pose(), dtype=np.float32)

        # Remaining errors.
        pos_error = goal[:3] - current[:3]
        pos_dist = float(np.linalg.norm(pos_error))
        current_rot = Rotation.from_quat(current[3:])
        rot_error = (goal_rot * current_rot.inv()).as_rotvec()
        rot_dist = float(np.linalg.norm(rot_error))

        # Convergence check.
        if pos_dist <= pos_tol_m and rot_dist <= rot_tol_rad:
            return InterpolatedMoveResult(
                reached=True,
                goal_pose=goal,
                final_pose=current,
                pos_error_m=pos_dist,
                rot_error_rad=rot_dist,
                last_commanded_pose=last_commanded_pose,
            )

        # Translation carrot: advance by delta_m toward goal.
        if pos_dist > delta_m:
            delta_pos = pos_error * (delta_m / pos_dist)
        else:
            delta_pos = pos_error

        # Rotation carrot: advance by delta_rad toward goal.
        if rot_dist > delta_rad:
            rot_delta = rot_error * (delta_rad / rot_dist)
        else:
            rot_delta = rot_error

        next_pose = current.copy()
        next_pose[:3] = current[:3] + delta_pos
        next_pose[3:] = (Rotation.from_rotvec(rot_delta) * current_rot).as_quat()
        next_pose = clip_pose(next_pose)

        send_pose_command(next_pose)
        last_commanded_pose = next_pose.copy()

        # Sim-time pacing: poll in small wall-time increments (same pattern
        # as RobotEnv.step()).
        step_target = now() + period
        while now() < step_target:
            sleep(0.001)

    # Final convergence check.
    final_pose = np.array(read_current_pose(), dtype=np.float32)
    pos_error_m = float(np.linalg.norm(goal[:3] - final_pose[:3]))
    rot_error_rad = float(
        (Rotation.from_quat(final_pose[3:]).inv() * goal_rot).magnitude()
    )
    reached = pos_error_m <= pos_tol_m and rot_error_rad <= rot_tol_rad
    return InterpolatedMoveResult(
        reached=reached,
        goal_pose=goal,
        final_pose=final_pose,
        pos_error_m=pos_error_m,
        rot_error_rad=rot_error_rad,
        last_commanded_pose=last_commanded_pose,
    )
