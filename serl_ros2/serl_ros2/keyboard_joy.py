#!/usr/bin/env python3
"""
Keyboard to Joy message publisher for ROS2.

This node captures keyboard input and publishes sensor_msgs/Joy messages,
allowing keyboard control of robots via the standard Joy interface.

Key bindings (WASD/RF + IJKL scheme):
  Translation (axes 0-2):
    W/S - X axis (forward/backward)
    A/D - Y axis (left/right)
    R/F - Z axis (up/down)

  Rotation (axes 3-5):
    J/L - Roll (rotation around X)
    I/K - Pitch (rotation around Y)
    U/O - Yaw (rotation around Z)

  Buttons:
    F4    - Button 0 (gripper close)
    F3    - Button 1 (gripper open)
    1-9   - Buttons 2-10

  Modifiers:
    SHIFT - 2x speed boost

Usage:
    ros2 run serl_ros2 keyboard_joy --ros-args -p topic:=teleop_joy -p frame_id:=base
    # or directly:
    python keyboard_joy.py --ros-args -p topic:=teleop_joy -p frame_id:=base

Note: This node requires terminal focus to capture keyboard input.
"""
import sys
import threading
from typing import Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

from pynput import keyboard

class KeyboardJoyNode(Node):
    """ROS2 node that publishes Joy messages from keyboard input."""

    # Default key mappings for axes.
    # Format: key -> (axis_index, direction)
    DEFAULT_AXIS_KEYS = {
        "w": (0, 1.0),   # +X (forward)
        "s": (0, -1.0),  # -X (backward)
        "a": (1, 1.0),   # +Y (left)
        "d": (1, -1.0),  # -Y (right)
        "r": (2, 1.0),   # +Z (up)
        "f": (2, -1.0),  # -Z (down)
        "j": (3, 1.0),   # +roll
        "l": (3, -1.0),  # -roll
        "i": (4, 1.0),   # +pitch
        "k": (4, -1.0),  # -pitch
        "u": (5, 1.0),   # +yaw
        "o": (5, -1.0),  # -yaw
    }

    @staticmethod
    def _build_axis_keys(mapping_list: list) -> dict[str, tuple[int, float]]:
        """Build axis_keys dict from YAML-style list.

        Each entry: [key_positive, key_negative, axis_index, sign].
        Returns dict like {"w": (0, 1.0), "s": (0, -1.0), ...}.
        """
        result = {}
        for entry in mapping_list:
            key_pos, key_neg, axis_idx, sign = entry
            result[str(key_pos)] = (int(axis_idx), float(sign))
            result[str(key_neg)] = (int(axis_idx), -float(sign))
        return result

    def __init__(self) -> None:
        super().__init__("keyboard_joy")
        
        # Declare ROS2 parameters
        self.declare_parameter("topic", "teleop_joy")
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("frame_id", "tcp")
        self.declare_parameter("config", "")

        topic = self.get_parameter("topic").get_parameter_value().string_value
        rate = self.get_parameter("rate").get_parameter_value().double_value
        self._frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        config_path = self.get_parameter("config").get_parameter_value().string_value

        self._publisher = self.create_publisher(Joy, topic, 10)
        self._rate = rate
        self._period = 1.0 / rate

        # Build per-frame axis key dicts from config YAML or defaults
        mapping = {}
        if config_path:
            import yaml
            import os
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    full_config = yaml.safe_load(f) or {}
                mapping = full_config.get("keyboard_axis_mapping", {})
        self._axis_keys_by_frame: dict[str, dict[str, tuple[int, float]]] = {}
        for frame in ("tcp", "base"):
            if frame in mapping:
                self._axis_keys_by_frame[frame] = self._build_axis_keys(mapping[frame])
            else:
                self._axis_keys_by_frame[frame] = self.DEFAULT_AXIS_KEYS.copy()

        # Track pressed keys
        self._pressed_keys: set[str] = set()
        self._pressed_special: set[Any] = set()
        self._lock = threading.Lock()
        
        # Keyboard listener
        self._listener: Any = None
        
        # Timer for publishing
        self._timer = self.create_timer(self._period, self._publish_joy)
        
        self.get_logger().info(
            f"Publishing Joy messages to '{topic}' at {rate} Hz with frame_id='{self._frame_id}'"
        )

    def start_keyboard_listener(self) -> None:
        """Start the keyboard listener."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _on_press(self, key: Any) -> None:
        """Handle key press events."""
        if key is None:
            return
        # Tab toggles between tcp and base frame
        if key == keyboard.Key.tab:
            self._frame_id = "base" if self._frame_id == "tcp" else "tcp"
            self.get_logger().info(f"Teleop frame: {self._frame_id}")
            return
        with self._lock:
            if isinstance(key, keyboard.KeyCode) and key.char:
                self._pressed_keys.add(key.char.lower())
            elif isinstance(key, keyboard.Key):
                self._pressed_special.add(key)

    def _on_release(self, key: Any) -> None:
        """Handle key release events."""
        if key is None:
            return
        with self._lock:
            if isinstance(key, keyboard.KeyCode) and key.char:
                self._pressed_keys.discard(key.char.lower())
            elif isinstance(key, keyboard.Key):
                self._pressed_special.discard(key)

    def _publish_joy(self) -> None:
        """Publish a Joy message based on current key state."""
        with self._lock:
            pressed_keys = set(self._pressed_keys)
            pressed_special = set(self._pressed_special)

        # Modifiers:
        #   Shift = speed boost (2x base scale)
        #   Space = force boost (output 1.0 instead of 0.8, triggering action_boost wrench)
        speed = 2.0 if (
            keyboard.Key.shift in pressed_special or
            keyboard.Key.shift_l in pressed_special or
            keyboard.Key.shift_r in pressed_special
        ) else 1.0
        force_boost = keyboard.Key.space in pressed_special
        # Without Space, scale to 0.8 so action stays below the boost threshold (0.9).
        # With Space, scale to 1.0 so the action triggers the force boost.
        base_scale = 1.0 if force_boost else 0.8

        # Compute axes (6 axes for xyz + rpy) using the active frame's mapping
        axis_keys = self._axis_keys_by_frame.get(self._frame_id, self.DEFAULT_AXIS_KEYS)
        axes = [0.0] * 6
        for key, (axis_idx, direction) in axis_keys.items():
            if key in pressed_keys:
                axes[axis_idx] += direction * speed * base_scale

        # Compute buttons
        buttons = [0] * 11  # Support up to 11 buttons
        
        # F4 -> button 0 (gripper close)
        if keyboard.Key.f4 in pressed_special:
            buttons[0] = 1
        # F3 -> button 1 (gripper open)
        if keyboard.Key.f3 in pressed_special:
            buttons[1] = 1

        # Create and publish Joy message
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.axes = [float(a) for a in axes]
        msg.buttons = buttons
        self._publisher.publish(msg)


def suppress_terminal_echo():
    """Suppress terminal echo while running."""
    import contextlib
    
    @contextlib.contextmanager
    def _suppress():
        old_settings = None
        fd = None
        try:
            import termios
            import tty
            if sys.stdin.isatty():
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                tty.setraw(fd)
                # Re-enable ISIG so Ctrl+C works
                new_settings = termios.tcgetattr(fd)
                new_settings[3] = new_settings[3] | termios.ISIG
                termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        except (ImportError, OSError):
            pass
        
        try:
            yield
        finally:
            if old_settings is not None and fd is not None:
                try:
                    import termios
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except (ImportError, OSError):
                    pass
    
    return _suppress()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardJoyNode()
    
    # Get parameters for logging
    topic = node.get_parameter("topic").get_parameter_value().string_value
    rate = node.get_parameter("rate").get_parameter_value().double_value
    
    node.start_keyboard_listener()

    print()
    print("Keyboard Joy publisher running. Press Ctrl+C to exit.")
    print(f"Publishing to '{topic}' at {rate} Hz")
    print("This terminal must have focus for keyboard input to work.")
    print()
    frame_id = node.get_parameter("frame_id").get_parameter_value().string_value
    config_path = node.get_parameter("config").get_parameter_value().string_value
    print("Controls:")
    print("  Translation: W/S (X), A/D (Y), R/F (Z)")
    print("  Rotation: J/L (roll), I/K (pitch), U/O (yaw)")
    print("  Buttons: F4 (close), F3 (open)")
    print("  Speed boost: Hold SHIFT (2x speed)")
    print("  Force boost: Hold SPACE (extra wrench at TCP)")
    print(f"  Toggle frame: TAB (starting: '{frame_id}')")
    if config_path:
        print(f"  Axis mapping loaded from: {config_path}")
    print()

    with suppress_terminal_echo():
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.stop_keyboard_listener()
            node.destroy_node()
            rclpy.shutdown()

    print("\nKeyboard Joy publisher stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
