"""
Joy message teleop adapter for HIL-SERL.

This module provides a TeleopAdapter that subscribes to sensor_msgs/Joy messages,
allowing any ROS2 joy publisher (e.g., keyboard teleop nodes, gamepads) to be used
for human intervention.

The Joy message axes are mapped to HIL-SERL's (xyz, rpy) convention:
  axes[0] -> x (forward/backward)
  axes[1] -> y (left/right)
  axes[2] -> z (up/down)
  axes[3] -> roll (rotation around x)
  axes[4] -> pitch (rotation around y)
  axes[5] -> yaw (rotation around z)

Buttons are passed through directly from the Joy message.

Usage:
    # Create adapter (subscription is created when setup_ros2 is called)
    teleop = JoyTeleopAdapter()
    
    # Pass to RobotAdapter - it will call setup_ros2() internally
    adapter = RobotAdapter.create(config_path="...", teleop_adapter=teleop)
    adapter.start()
"""

import threading
from typing import Any

from serl_framework.teleop_adapter import TeleopAdapter, TeleopButton


class JoyTeleopAdapter(TeleopAdapter):
    """
    TeleopAdapter that receives input from sensor_msgs/Joy messages.

    Unlike SpaceMouseTeleop which uses a background process for direct device access,
    this adapter is driven by ROS2 callbacks. The RobotAdapter calls setup_ros2()
    to create the Joy subscription on its node.

    Use with `ros2 run serl_ros2 keyboard_joy` for keyboard input, or any other
    node that publishes sensor_msgs/Joy messages (e.g., gamepads).

    Args:
        topic: Joy topic name (default: "teleop_joy").
        axis_mapping: Optional dict mapping axis names to Joy axes indices.
            Default: {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
        axis_scale: Optional dict mapping axis names to scale factors.
            Default: all 1.0
        button_mapping: Optional dict mapping TeleopButton to Joy button indices.
            Default: {GRIPPER_CLOSE: 0, GRIPPER_OPEN: 1}
    """

    DEFAULT_AXIS_MAPPING = {
        "x": 0,
        "y": 1,
        "z": 2,
        "roll": 3,
        "pitch": 4,
        "yaw": 5,
    }

    DEFAULT_BUTTON_MAPPING = {
        TeleopButton.GRIPPER_CLOSE: 0,
        TeleopButton.GRIPPER_OPEN: 1,
    }

    def __init__(
        self,
        topic: str = "teleop_joy",
        axis_mapping: dict[str, int] | None = None,
        axis_scale: dict[str, float] | None = None,
        button_mapping: dict[TeleopButton, int] | None = None,
    ) -> None:
        self._topic = topic
        self._axis_mapping = axis_mapping or self.DEFAULT_AXIS_MAPPING.copy()
        self._axis_scale = axis_scale or {}
        self._button_mapping = button_mapping or self.DEFAULT_BUTTON_MAPPING.copy()
        
        # Thread-safe state (no multiprocessing needed - ROS2 callbacks update this)
        self._lock = threading.Lock()
        self._xyz = [0.0, 0.0, 0.0]
        self._rpy = [0.0, 0.0, 0.0]
        self._buttons: list[int] = []
        self._frame_id = ""
        
        # ROS2 subscription (created by setup_ros2)
        self._subscription: Any | None = None
        self._node: Any | None = None
        
        # Skip parent's multiprocessing setup by not calling super().__init__()
        # We'll provide our own get_teleop(), start(), close() implementations

    def setup_ros2(self, node: Any) -> None:
        """
        Create the Joy subscription on the given ROS2 node.

        This is called by RobotAdapter during its setup phase.

        Args:
            node: ROS2 node to create subscription on.
        """
        if self._subscription is not None:
            return  # Already set up
            
        # Import here to avoid requiring ROS2 at module load time
        from sensor_msgs.msg import Joy
        
        self._node = node
        self._subscription = node.create_subscription(
            Joy,
            self._topic,
            self._on_joy,
            10,
        )
        node.get_logger().info(f"JoyTeleopAdapter subscribed to '{self._topic}'")

    def _on_joy(self, msg: Any) -> None:
        """
        ROS2 callback for Joy messages.
        """
        axes = list(msg.axes) if msg.axes else []
        raw_buttons = [int(b) for b in msg.buttons] if msg.buttons else []
        
        # Extract xyz from axes
        xyz = [0.0, 0.0, 0.0]
        for i, axis_name in enumerate(["x", "y", "z"]):
            axis_idx = self._axis_mapping.get(axis_name, i)
            if axis_idx >= 0 and axis_idx < len(axes):
                scale = self._axis_scale.get(axis_name, 1.0)
                xyz[i] = axes[axis_idx] * scale
        
        # Extract rpy from axes
        rpy = [0.0, 0.0, 0.0]
        for i, axis_name in enumerate(["roll", "pitch", "yaw"]):
            axis_idx = self._axis_mapping.get(axis_name, i + 3)
            if axis_idx >= 0 and axis_idx < len(axes):
                scale = self._axis_scale.get(axis_name, 1.0)
                rpy[i] = axes[axis_idx] * scale
        
        buttons = [0] * len(TeleopButton)
        for button, idx in self._button_mapping.items():
            if 0 <= idx < len(raw_buttons):
                buttons[int(button)] = int(raw_buttons[idx])
        frame_id = ""
        if hasattr(msg, "header") and hasattr(msg.header, "frame_id"):
            frame_id = str(msg.header.frame_id)

        with self._lock:
            self._xyz = xyz
            self._rpy = rpy
            self._buttons = buttons
            self._frame_id = frame_id

    def get_teleop(self) -> tuple[list[float], list[float], list[int], str]:
        """
        Return the latest teleop reading from Joy messages.
        """
        with self._lock:
            return (
                list(self._xyz),
                list(self._rpy),
                list(self._buttons),
                str(self._frame_id),
            )

    def start(self) -> None:
        """
        No-op for JoyTeleopAdapter.
        
        The adapter is driven by ROS2 callbacks, not a background process.
        The subscription is created by setup_ros2() which is called by RobotAdapter.
        """
        pass

    def close(self) -> None:
        """
        Clean up the Joy subscription.
        """
        if self._subscription is not None and self._node is not None:
            try:
                self._node.destroy_subscription(self._subscription)
            except Exception:
                pass
            self._subscription = None

    # These methods are required by TeleopAdapter ABC but not used
    # since we override start() and get_teleop() directly.
    
    def _default_teleop(self) -> tuple[list[float], list[float], list[int], str]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [], ""

    def _read_devices(self) -> list[tuple[list[float], list[float], list[int], str]]:
        # Not used - ROS2 callback updates state directly
        return []

    def _open_device(self) -> None:
        # Not used - subscription created by setup_ros2()
        pass

    def _close_device(self) -> None:
        # Not used - cleanup done in close()
        pass
