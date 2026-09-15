"""
Standalone teleop check: drive the robot through the RobotAdapter, without any RL.

HIL-SERL controls the robot like a carrot on a stick: each control step moves a
commanded TCP target by a small delta, and the robot's controller tracks that
target. This tool runs exactly that loop with a human teleop device instead of a
policy. If you cannot perform your task smoothly this way, a trained policy will
not either, so run this check before collecting any data.

Usage (after colcon build, from any directory):
    # SpaceMouse (default)
    ros2 run serl_ros2 teleop_check --config path/to/adapter_config.yaml
    ros2 run serl_ros2 teleop_check --config path/to/adapter_config.yaml --spacemouse-frame-id base

    # Joy (keyboard or gamepad via ROS2)
    ros2 run serl_ros2 teleop_check --config path/to/adapter_config.yaml --teleop joy

    # For keyboard input, run in a separate terminal:
    ros2 run serl_ros2 keyboard_joy --ros-args -p topic:=teleop_joy -p frame_id:=tcp

    # Training parity: read rate, action scales, and teleop settings from the
    # task config YAML (CLI flags still override):
    ros2 run serl_ros2 teleop_check --config path/to/adapter_config.yaml \\
        --task-config path/to/task_config.yaml --verbose

Reading the --verbose diagnostics (printed once per second):
  - loop: effective control rate, should match --rate. Jitter points to
    scheduling problems (CPU pinning helps on Intel hybrid CPUs).
  - |a|: mean clipped action norm. dxyz / dr: mean per-step translation (mm)
    and rotation (rad) deltas. max_input: largest single-axis input seen.
  - delta_vel: velocity requested by the teleop deltas. cmd_vel: velocity of
    the commanded target. arm_vel: actual TCP velocity. cmd_vel and arm_vel in
    the same ballpark means the controller keeps up. arm_vel far below cmd_vel
    means the target runs away from the robot: reduce the action scales or
    retune the controller.
  - BOOST: average extra wrench, only when the task config enables action boost.
"""

import argparse
import time
import numpy as np

from serl_framework.teleop_adapter import TeleopAdapter, TeleopButton
from serl_framework.utils.config import (
    compute_action_boost,
    load_yaml_dict,
    resolve_compliance_params,
)
from serl_framework.utils.teleop import apply_pose_delta, transform_translation_to_tip
from serl_ros2.robot_adapter import RobotAdapter

# Fallback defaults, used when neither CLI nor task config YAML provide a value.
DEFAULT_RATE_HZ = 30.0
DEFAULT_XYZ_SCALE = 0.05
DEFAULT_RPY_SCALE = 0.07
DEFAULT_INITIAL_POSE = [0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Flags default to None where the value can also come from the experiment
    YAML; the effective value is resolved in _resolve_settings with precedence
    CLI > task config YAML > fallback default.
    """
    parser = argparse.ArgumentParser(description="Teleop check (drive the robot through the RobotAdapter)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the adapter YAML config (cameras, sync, command frame).",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default=None,
        help=(
            "Optional path to a task config YAML. When given, rate, action scales, "
            "teleop device/topic/frames, spacemouse axis mapping, compliance, and "
            "action boost are read from it so the check matches training."
        ),
    )
    parser.add_argument(
        "--teleop",
        type=str,
        choices=["spacemouse", "joy"],
        default=None,
        help="Teleop device type: 'spacemouse' or 'joy' (fallback: spacemouse).",
    )
    parser.add_argument(
        "--joy-topic",
        type=str,
        default=None,
        help="Joy topic name when using --teleop joy (fallback: teleop_joy).",
    )
    parser.add_argument(
        "--joy-preset",
        type=str,
        choices=["default", "xbox"],
        default="default",
        help="Joy axis mapping preset: 'default' or 'xbox' (default: default).",
    )
    parser.add_argument(
        "--spacemouse-frame-id",
        type=str,
        default=None,
        help="Frame id published by SpaceMouse teleop samples (fallback: tcp).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help=f"Command publish rate in Hz (fallback: {DEFAULT_RATE_HZ}).",
    )
    parser.add_argument(
        "--xyz-scale",
        type=float,
        default=None,
        help=(
            "Translation scale in meters per control step at full deflection, "
            f"HIL-SERL ACTION_SCALE-style (fallback: {DEFAULT_XYZ_SCALE})."
        ),
    )
    parser.add_argument(
        "--rpy-scale",
        type=float,
        default=None,
        help=(
            "Rotation-vector scale in radians per control step at full deflection, "
            f"HIL-SERL ACTION_SCALE-style (fallback: {DEFAULT_RPY_SCALE})."
        ),
    )
    parser.add_argument(
        "--base-frame-id",
        type=str,
        default=None,
        help="Frame id treated as base-frame teleop (fallback: base).",
    )
    parser.add_argument(
        "--tcp-frame-id",
        type=str,
        default=None,
        help="Frame id treated as tcp-frame teleop (fallback: tcp).",
    )
    parser.add_argument(
        "--initial-pose",
        type=float,
        nargs=7,
        default=None,
        help=(
            "Fallback initial pose as x y z qx qy qz qw, used if robot state is "
            f"unavailable (fallback: {DEFAULT_INITIAL_POSE})."
        ),
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Raise if the current robot pose cannot be read (do not fall back to --initial-pose).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print once-per-second diagnostics: effective loop Hz, action norm, "
            "per-step deltas, commanded vs actual TCP velocity, max input."
        ),
    )
    return parser.parse_args()


def _resolve_settings(args: argparse.Namespace) -> dict:
    """
    Resolve effective settings with precedence CLI > task config YAML > fallback.

    Returns the loaded task config YAML dict (empty when --task-config is
    not given), with the resolved values written back onto ``args``.
    """
    full_config = load_yaml_dict(args.task_config) if args.task_config else {}

    action_scale = full_config.get("action_scale", full_config.get("ACTION_SCALE", []))
    if args.rate is None:
        args.rate = float(full_config.get("hz", full_config.get("HZ", DEFAULT_RATE_HZ)))
    if args.xyz_scale is None:
        args.xyz_scale = float(action_scale[0]) if len(action_scale) > 0 else DEFAULT_XYZ_SCALE
    if args.rpy_scale is None:
        args.rpy_scale = float(action_scale[1]) if len(action_scale) > 1 else DEFAULT_RPY_SCALE
    if args.teleop is None:
        args.teleop = str(full_config.get("teleop_device", "spacemouse"))
    if args.joy_topic is None:
        args.joy_topic = str(full_config.get("teleop_joy_topic", "teleop_joy"))
    if args.spacemouse_frame_id is None:
        args.spacemouse_frame_id = str(full_config.get("teleop_spacemouse_frame_id", "tcp"))
    if args.base_frame_id is None:
        args.base_frame_id = str(full_config.get("teleop_base_frame_id", "base"))
    if args.tcp_frame_id is None:
        args.tcp_frame_id = str(full_config.get("teleop_tcp_frame_id", "tcp"))
    if args.initial_pose is None:
        args.initial_pose = list(DEFAULT_INITIAL_POSE)
    args.spacemouse_axis_mapping = full_config.get("spacemouse_axis_mapping", None)
    return full_config


XBOX_AXIS_MAPPING = {
    # Left stick for XY, D-pad for Z and roll
    # Right stick is often captured by game engines for camera control
    "x": 1,      # Left stick Y (forward/back)
    "y": 0,      # Left stick X (left/right)
    "z": 7,      # up/down X
    "roll": 6,   # D-pad X (left/right for roll)
    "pitch": -1, # Not mapped (use -1 to disable)
    "yaw": -1,   # Not mapped (use -1 to disable)
}


def _create_teleop(args: argparse.Namespace) -> TeleopAdapter:
    """
    Create the appropriate TeleopAdapter based on args.
    """
    if args.teleop == "spacemouse":
        from serl_framework.spacemouse_teleop import SpaceMouseTeleop
        return SpaceMouseTeleop(
            frame_id=args.spacemouse_frame_id,
            axis_mapping=args.spacemouse_axis_mapping,
        )
    elif args.teleop == "joy":
        from serl_ros2.joy_teleop_adapter import JoyTeleopAdapter

        axis_mapping = None
        axis_scale = None

        if args.joy_preset == "xbox":
            axis_mapping = XBOX_AXIS_MAPPING.copy()
            print("Using Xbox controller preset (left stick + D-pad)")

        return JoyTeleopAdapter(
            topic=args.joy_topic,
            axis_mapping=axis_mapping,
            axis_scale=axis_scale,
        )
    else:
        raise ValueError(f"Unknown teleop type: {args.teleop}")


def _get_current_pose(adapter: RobotAdapter, fallback_pose: list[float], no_fallback: bool = False) -> np.ndarray:
    """
    Use the robot's current TCP pose as the initial teleop pose.

    Falls back to the provided pose if the robot state is not yet available.
    """
    try:
        adapter.wait_for_state(timeout_s=5.0, poll_s=0.05)
    except TimeoutError:
        if not no_fallback:
            print("Warning: timed out waiting for robot state; using fallback initial pose.")
            return np.array(fallback_pose, dtype=np.float32)
        else:
            raise

    state = adapter.latest_state
    if state is None:
        if no_fallback:
            raise RuntimeError("No robot state received.")
        print("Warning: no robot state available; using fallback initial pose.")
        return np.array(fallback_pose, dtype=np.float32)

    tcp_pose = state.get("tcp_pose")
    if tcp_pose is None:
        if no_fallback:
            raise RuntimeError("Could not get robot state")
        print("Warning: robot state missing tcp_pose; using fallback initial pose.")
        return np.array(fallback_pose, dtype=np.float32)

    return np.array(tcp_pose, dtype=np.float32)


def _apply_compliance(adapter: RobotAdapter, full_config: dict) -> None:
    """
    Apply the task config YAML's compliance parameters, if any, tolerating failure.
    """
    compliance = full_config.get("compliance_param", full_config.get("COMPLIANCE_PARAM", {}))
    if not compliance:
        return
    resolved = resolve_compliance_params(compliance)
    result = adapter.call_service("set_compliance", params=resolved)
    if not result.get("ok", False):
        print(f"Warning: set_compliance failed: {result.get('message')}")


def _pace_loop(adapter: RobotAdapter, step_start_sim: float, period: float) -> None:
    """
    Pace one loop iteration on the adapter clock, matching the training loop.

    Under sim time this waits until a full sim period has elapsed, like
    env.step() does during training. A stalled clock (e.g. a paused simulator)
    cannot hang the loop: after a generous wall-clock deadline the wait ends.
    """
    target_sim_time = step_start_sim + period
    stall_deadline = time.perf_counter() + max(5.0, 10.0 * period)
    while adapter.now() < target_sim_time:
        time.sleep(0.001)
        if time.perf_counter() > stall_deadline:
            print("Warning: adapter clock is not advancing; continuing without sim-time pacing.")
            break


def main() -> int:
    """
    Run the teleop loop.
    """
    args = _parse_args()
    full_config = _resolve_settings(args)

    teleop = _create_teleop(args)
    adapter = RobotAdapter.create(config_path=args.config, teleop_adapter=teleop, executor_threads=4)
    adapter.start()

    _apply_compliance(adapter, full_config)
    boost_cfg = full_config.get("action_boost", {})
    boost_enabled = bool(boost_cfg.get("enabled", False))

    print(f"Teleop check started with {args.teleop}")
    print(f"  rate={args.rate} Hz, xyz_scale={args.xyz_scale}, rpy_scale={args.rpy_scale}")
    if args.teleop == "spacemouse":
        print(f"  SpaceMouse frame_id: {args.spacemouse_frame_id}")
    if args.teleop == "joy":
        print(f"  Subscribed to Joy topic: {args.joy_topic}")
        print("  Make sure a Joy publisher is running (e.g., ros2 run serl_ros2 keyboard_joy)")

    pose = _get_current_pose(adapter, args.initial_pose, no_fallback=args.no_fallback)
    period = 1.0 / max(args.rate, 1e-3)
    last_buttons = [0] * len(TeleopButton)

    # Diagnostics accumulators for --verbose.
    prev_pose = pose.copy()
    prev_cmd = pose.copy()
    diag_delta_sum = 0.0
    diag_move_sum = 0.0
    diag_cmd_step_sum = 0.0
    diag_action_norm_sum = 0.0
    diag_rot_delta_sum = 0.0
    diag_max_input = 0.0
    diag_boost_vec_sum = np.zeros(3)
    diag_boost_count = 0

    try:
        last_print = time.perf_counter()
        loop_count = 0
        while True:
            step_start_sim = adapter.now()

            prev_pose = pose.copy()
            pose = _get_current_pose(adapter, args.initial_pose, no_fallback=args.no_fallback)
            currpos_snapshot = pose.copy()  # save before delta is applied

            xyz, rpy, _buttons, _frame_id = adapter.get_teleop()
            boost_input_tip = None
            if xyz and rpy:
                raw_action = np.array(list(xyz[:3]) + list(rpy[:3]), dtype=np.float32)
                clipped_action = np.clip(raw_action, -1.0, 1.0)
                delta_xyz = clipped_action[:3] * float(args.xyz_scale)
                delta_rpy = clipped_action[3:6] * float(args.rpy_scale)
                diag_delta_sum += np.linalg.norm(delta_xyz)
                diag_rot_delta_sum += np.linalg.norm(delta_rpy)
                diag_action_norm_sum += np.linalg.norm(clipped_action)
                diag_max_input = max(diag_max_input, np.max(np.abs(clipped_action[:3])))
                teleop_frame_id = str(_frame_id).strip()
                # Empty frame defaults to TCP, as defined by TeleopAdapter interface.
                if not teleop_frame_id:
                    teleop_frame_id = args.tcp_frame_id
                # Workaround for the ROS2 joy_node, which always publishes frame_id "joy".
                if teleop_frame_id == "joy":
                    teleop_frame_id = args.tcp_frame_id
                if teleop_frame_id == args.base_frame_id:
                    delta_in_base = True
                    boost_input_tip = transform_translation_to_tip(
                        np.array(xyz[:3], dtype=np.float32),
                        currpos_snapshot[3:],
                    )
                elif teleop_frame_id == args.tcp_frame_id:
                    delta_in_base = False
                    boost_input_tip = np.array(xyz[:3], dtype=np.float32)
                else:
                    print(
                        "Warning: ignoring teleop sample with invalid frame_id "
                        f"'{teleop_frame_id}' (expected '{args.base_frame_id}' or '{args.tcp_frame_id}')."
                    )
                    delta_in_base = None
                if delta_in_base is not None:
                    pose = apply_pose_delta(
                        pose, delta_xyz, delta_rpy, delta_in_base=delta_in_base
                    )

            # Action-magnitude boost: extra wrench when teleop input is near saturation.
            if boost_enabled and boost_input_tip is not None:
                boost = compute_action_boost(boost_input_tip.tolist(), boost_cfg)
                if np.linalg.norm(boost[:3]) > 0:
                    diag_boost_vec_sum += np.array(boost[:3])
                    diag_boost_count += 1
                adapter.set_step_wrench_boost(boost)

            diag_cmd_step_sum += np.linalg.norm(pose[:3] - prev_cmd[:3])
            prev_cmd = pose.copy()
            adapter.send_pose_command(pose.tolist())

            buttons = list(_buttons) if _buttons else []
            while len(buttons) < len(TeleopButton):
                buttons.append(0)

            close_pressed = int(buttons[TeleopButton.GRIPPER_CLOSE])
            open_pressed = int(buttons[TeleopButton.GRIPPER_OPEN])

            if close_pressed and not last_buttons[TeleopButton.GRIPPER_CLOSE]:
                result = adapter.call_service("set_gripper", mode=1)
                if not result.get("ok", False):
                    print(f"Warning: set_gripper close failed: {result.get('message')}")

            if open_pressed and not last_buttons[TeleopButton.GRIPPER_OPEN]:
                result = adapter.call_service("set_gripper", mode=0)
                if not result.get("ok", False):
                    print(f"Warning: set_gripper open failed: {result.get('message')}")

            last_buttons[TeleopButton.GRIPPER_CLOSE] = close_pressed
            last_buttons[TeleopButton.GRIPPER_OPEN] = open_pressed

            _pace_loop(adapter, step_start_sim, period)

            # Track actual arm movement over the step.
            diag_move_sum += np.linalg.norm(pose[:3] - prev_pose[:3])

            loop_count += 1
            elapsed_since_print = time.perf_counter() - last_print
            if elapsed_since_print >= 1.0:
                if args.verbose:
                    effective_hz = max(loop_count / elapsed_since_print, 1e-6)
                    target_vel = diag_delta_sum / elapsed_since_print * 1000  # mm/s
                    arm_vel = diag_move_sum / elapsed_since_print * 1000  # mm/s
                    cmd_vel = diag_cmd_step_sum / elapsed_since_print * 1000  # mm/s
                    boost_str = ""
                    if diag_boost_count > 0:
                        avg = diag_boost_vec_sum / diag_boost_count
                        boost_str = f" | BOOST=({avg[0]:+.1f},{avg[1]:+.1f},{avg[2]:+.1f})N"
                    avg_action_norm = diag_action_norm_sum / max(loop_count, 1)
                    avg_rot_delta = diag_rot_delta_sum / max(loop_count, 1)
                    print(
                        f"  [diag] loop={effective_hz:.1f} Hz | "
                        f"|a|={avg_action_norm:.2f} | "
                        f"dxyz={target_vel / effective_hz:.1f} mm | "
                        f"dr={avg_rot_delta:.3f} rad | "
                        f"delta_vel={target_vel:.0f} mm/s | "
                        f"cmd_vel={cmd_vel:.0f} mm/s | "
                        f"arm_vel={arm_vel:.0f} mm/s | "
                        f"max_input={diag_max_input:.2f}"
                        f"{boost_str}"
                    )
                last_print = time.perf_counter()
                loop_count = 0
                diag_delta_sum = 0.0
                diag_move_sum = 0.0
                diag_cmd_step_sum = 0.0
                diag_action_norm_sum = 0.0
                diag_rot_delta_sum = 0.0
                diag_max_input = 0.0
                diag_boost_vec_sum = np.zeros(3)
                diag_boost_count = 0
    except KeyboardInterrupt:
        return 0
    finally:
        teleop.close()
        adapter.stop()


if __name__ == "__main__":
    raise SystemExit(main())
