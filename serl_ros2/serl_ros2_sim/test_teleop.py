"""
Teleop test node for the Ursina sim.

This script reads teleop deltas via TeleopAdapter (SpaceMouse or Joy) and publishes
pose commands through serl_ros2.RobotAdapter, so the Ursina sim ball follows the
teleop input.

Usage:
    # SpaceMouse (default)
    python test_teleop.py --config path/to/config.yaml
    python test_teleop.py --config path/to/config.yaml --spacemouse-frame-id base

    # Joy (keyboard or gamepad via ROS2)
    python test_teleop.py --config path/to/config.yaml --teleop joy --joy-topic teleop_joy

    # For keyboard input, run in a separate terminal:
    ros2 run serl_ros2 keyboard_joy --ros-args -p topic:=teleop_joy -p frame_id:=base
"""

import argparse
import time
import numpy as np

from serl_framework.teleop_adapter import TeleopAdapter, TeleopButton
from serl_framework.utils.teleop import apply_pose_delta
from serl_ros2.robot_adapter import RobotAdapter


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Teleop test node")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to adapter YAML config.",
    )
    parser.add_argument(
        "--teleop",
        type=str,
        choices=["spacemouse", "joy"],
        default="spacemouse",
        help="Teleop device type: 'spacemouse' or 'joy' (default: spacemouse).",
    )
    parser.add_argument(
        "--joy-topic",
        type=str,
        default="teleop_joy",
        help="Joy topic name when using --teleop joy (default: teleop_joy).",
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
        default="tcp",
        help=(
            "Frame id published by SpaceMouse teleop samples "
            "(default: tcp)."
        ),
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=30.0,
        help="Command publish rate in Hz.",
    )
    parser.add_argument(
        "--xyz-scale",
        type=float,
        default=0.05,
        help="Translation scale in meters per control step at full deflection (HIL-SERL ACTION_SCALE-style).",
    )
    parser.add_argument(
        "--rpy-scale",
        type=float,
        default=0.07,
        help="Rotation-vector scale in radians per control step at full deflection (HIL-SERL ACTION_SCALE-style).",
    )
    parser.add_argument(
        "--base-frame-id",
        type=str,
        default="base",
        help="Frame id treated as base-frame teleop (default: base).",
    )
    parser.add_argument(
        "--tcp-frame-id",
        type=str,
        default="tcp",
        help="Frame id treated as tcp-frame teleop (default: tcp).",
    )
    parser.add_argument(
        "--initial-pose",
        type=float,
        nargs=7,
        default=[0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0],
        help="Fallback initial pose as x y z qx qy qz qw (used if robot state is unavailable).",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Raise if the current robot pose cannot be read (do not fall back to --initial-pose).",
    )
    return parser.parse_args()


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
        return SpaceMouseTeleop(frame_id=args.spacemouse_frame_id)
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


def main() -> int:
    """
    Run the teleop loop.
    """
    args = _parse_args()
    teleop = _create_teleop(args)
    adapter = RobotAdapter.create(config_path=args.config, teleop_adapter=teleop, executor_threads=4)
    adapter.start()

    print(f"Teleop test started with {args.teleop}")
    if args.teleop == "spacemouse":
        print(f"  SpaceMouse frame_id: {args.spacemouse_frame_id}")
    if args.teleop == "joy":
        print(f"  Subscribed to Joy topic: {args.joy_topic}")
        print("  Make sure a Joy publisher is running (e.g., ros2 run serl_ros2 keyboard_joy)")

    pose = _get_current_pose(adapter, args.initial_pose, no_fallback=args.no_fallback)
    period = 1.0 / max(args.rate, 1e-3)
    last_buttons = [0] * len(TeleopButton)

    try:
        last_print = time.perf_counter()
        loop_count = 0
        while True:
            now = time.perf_counter()

            pose = _get_current_pose(adapter, args.initial_pose, no_fallback=args.no_fallback)

            xyz, rpy, _buttons, _frame_id = adapter.get_teleop()
            if xyz and rpy:
                delta_xyz = (
                    np.array(xyz[:3], dtype=np.float32) * float(args.xyz_scale)
                )
                delta_rpy = (
                    np.array(rpy[:3], dtype=np.float32) * float(args.rpy_scale)
                )
                teleop_frame_id = str(_frame_id).strip()
                # Empty frame defaults to TCP, as defined by TeleopAdapter interface.
                if not teleop_frame_id:
                    teleop_frame_id = args.tcp_frame_id
                # Temporary workaround for upstream joy_node frame id behavior.
                if teleop_frame_id == "joy":
                    teleop_frame_id = args.tcp_frame_id
                if teleop_frame_id == args.base_frame_id:
                    delta_in_base = True
                elif teleop_frame_id == args.tcp_frame_id:
                    delta_in_base = False
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
            sleep_for = period - (time.perf_counter() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)

            loop_count += 1
            if (time.perf_counter() - last_print) >= 1.0:
                # print(f"teleop loop hz ~ {loop_count / (time.perf_counter() - last_print):.1f}")
                last_print = time.perf_counter()
                loop_count = 0
    except KeyboardInterrupt:
        return 0
    finally:
        teleop.close()
        adapter.stop()


if __name__ == "__main__":
    raise SystemExit(main())
