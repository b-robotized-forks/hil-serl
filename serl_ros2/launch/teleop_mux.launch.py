"""
Launch teleop Joy sources and mux them into one output Joy topic.

Default source order:
- pose_to_joy      -> teleop_joy_pose
- joy_node         -> teleop_joy_spacemouse
- keyboard_joy     -> teleop_joy_keyboard

joy_mux publishes unified output on teleop_joy.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(_context, *_args, **_kwargs):
    output_topic = LaunchConfiguration("output_topic")
    active_source_topic = LaunchConfiguration("active_source_topic")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")
    default_frame_id = LaunchConfiguration("default_frame_id")
    frame_id_all = LaunchConfiguration("frame_id_all").perform(_context)

    enable_pose_to_joy = LaunchConfiguration("enable_pose_to_joy")
    enable_joy_node = LaunchConfiguration("enable_joy_node")
    enable_keyboard = LaunchConfiguration("enable_keyboard")

    base_frame_id = LaunchConfiguration("base_frame_id")
    tcp_frame_id = LaunchConfiguration("tcp_frame_id")
    pose_output_command_frame = LaunchConfiguration("pose_output_command_frame")
    keyboard_frame_id = LaunchConfiguration("keyboard_frame_id")

    joy_device_id = LaunchConfiguration("joy_device_id")

    if frame_id_all:
        if frame_id_all not in ("tcp", "base"):
            raise ValueError("frame_id_all must be either 'tcp', 'base', or empty.")
        default_frame_id = frame_id_all
        pose_output_command_frame = frame_id_all
        keyboard_frame_id = frame_id_all
        spacemouse_frame_override = frame_id_all
    else:
        # Keep current behavior for joy_node source without global override.
        spacemouse_frame_override = "tcp"

    return [
        Node(
            package="serl_ros2",
            executable="joy_mux",
            name="joy_mux",
            output="screen",
            parameters=[
                {
                    "output_topic": output_topic,
                    "active_source_topic": active_source_topic,
                    "publish_rate_hz": publish_rate_hz,
                    "default_frame_id": default_frame_id,
                    "source_names": ["pose", "spacemouse", "keyboard"],
                    "source_topics": [
                        "teleop_joy_pose",
                        "teleop_joy_spacemouse",
                        "teleop_joy_keyboard",
                    ],
                    "source_priorities": [0, 1, 2],
                    "source_deadzones": [0.1, 0.1, 0.1],
                    "source_timeouts_s": [0.25, 0.25, 0.25],
                    "source_frame_overrides": ["", spacemouse_frame_override, ""],
                    # Per-source axis mappings [x, y, z, roll, pitch, yaw].
                    # For spacemouse-via-joy, use roll/pitch swap to mirror
                    # SpaceMouseTeleop behavior as closely as possible.
                    "source_axis_mappings": [
                        0, 1, 2, 3, 4, 5,  # pose
                        0, 1, 2, 3, 4, 5,  # spacemouse
                        0, 1, 2, 3, 4, 5,  # keyboard
                    ],
                    # Per-source axis scales [x, y, z, roll, pitch, yaw].
                    # spacemouse x is inverted to match SpaceMouseTeleop.
                    "source_axis_scales": [
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # pose
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # spacemouse
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # keyboard
                    ],
                }
            ],
        ),
        Node(
            package="serl_ros2",
            executable="pose_to_joy",
            name="pose_to_joy",
            output="screen",
            condition=IfCondition(enable_pose_to_joy),
            parameters=[
                {
                    "joy_topic": "teleop_joy_pose",
                    "output_command_frame": pose_output_command_frame,
                    "base_frame_id": base_frame_id,
                    "tcp_frame_id": tcp_frame_id,
                    "enable_external_preempt_disarm": True,
                    "active_source_topic": active_source_topic,
                    "active_source_name": "pose",
                }
            ],
        ),
        Node(
            package="joy",
            executable="joy_node",
            name="spacemouse_joy_node",
            output="screen",
            condition=IfCondition(enable_joy_node),
            parameters=[{"device_id": joy_device_id}],
            remappings=[("joy", "teleop_joy_spacemouse")],
        ),
        Node(
            package="serl_ros2",
            executable="keyboard_joy",
            name="keyboard_joy",
            output="screen",
            condition=IfCondition(enable_keyboard),
            parameters=[
                {
                    "topic": "teleop_joy_keyboard",
                    "frame_id": keyboard_frame_id,
                }
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("output_topic", default_value="teleop_joy"),
            DeclareLaunchArgument(
                "active_source_topic", default_value="/teleop/active_source"
            ),
            DeclareLaunchArgument("publish_rate_hz", default_value="50.0"),
            DeclareLaunchArgument("default_frame_id", default_value="tcp"),
            DeclareLaunchArgument(
                "frame_id_all",
                default_value="",
                description=(
                    "Global frame override for all teleop sources: "
                    "set to 'tcp' or 'base' to enforce one frame everywhere."
                ),
            ),
            DeclareLaunchArgument("enable_pose_to_joy", default_value="true"),
            DeclareLaunchArgument("enable_joy_node", default_value="true"),
            DeclareLaunchArgument("enable_keyboard", default_value="true"),
            DeclareLaunchArgument("base_frame_id", default_value="base"),
            DeclareLaunchArgument("tcp_frame_id", default_value="tcp"),
            DeclareLaunchArgument("pose_output_command_frame", default_value="tcp"),
            DeclareLaunchArgument("keyboard_frame_id", default_value="tcp"),
            DeclareLaunchArgument("joy_device_id", default_value="0"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
