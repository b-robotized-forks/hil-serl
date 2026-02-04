#!/usr/bin/env python3
"""
Joy multiplexer for selecting one active teleop source at a time.

This node subscribes to multiple Joy topics and republishes one selected stream
to an output Joy topic. A source is considered active when any mapped axis
exceeds its deadzone or any button is pressed.

Selection behavior:
- If the currently selected source is still active, keep it.
- Otherwise, select the highest-priority active source.
- If no source is active, publish zero axes/buttons.

Per-source axis mapping and scaling are supported to normalize different input
devices before arbitration.
"""

from dataclasses import dataclass
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String


_AXIS_COUNT = 6
_DEFAULT_AXIS_MAPPING = [0, 1, 2, 3, 4, 5]
_DEFAULT_AXIS_SCALE = [1.0] * _AXIS_COUNT


@dataclass
class _SourceConfig:
    name: str
    topic: str
    priority: int
    deadzone: float
    timeout_s: float
    frame_override: str
    axis_mapping: list[int]
    axis_scale: list[float]


@dataclass
class _SourceSample:
    axes: list[float]
    buttons: list[int]
    frame_id: str
    stamp_s: float


class JoyMux(Node):
    """ROS2 node that arbitrates multiple Joy inputs into one output stream."""

    def __init__(self) -> None:
        super().__init__("joy_mux")

        self.declare_parameter("output_topic", "teleop_joy")
        self.declare_parameter("active_source_topic", "/teleop/active_source")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("default_frame_id", "tcp")

        self.declare_parameter(
            "source_names",
            ["pose", "spacemouse", "keyboard"],
        )
        self.declare_parameter(
            "source_topics",
            ["teleop_joy_pose", "teleop_joy_spacemouse", "teleop_joy_keyboard"],
        )
        self.declare_parameter("source_priorities", [0, 1, 2])
        self.declare_parameter("source_deadzones", [0.1, 0.1, 0.1])
        self.declare_parameter("source_timeouts_s", [0.25, 0.25, 0.25])
        self.declare_parameter("source_frame_overrides", ["", "", ""])
        self.declare_parameter(
            "source_axis_mappings",
            _DEFAULT_AXIS_MAPPING * 3,
        )
        self.declare_parameter(
            "source_axis_scales",
            _DEFAULT_AXIS_SCALE * 3,
        )

        output_topic = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )
        active_source_topic = (
            self.get_parameter("active_source_topic")
            .get_parameter_value()
            .string_value
        )
        publish_rate_hz = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )
        self._default_frame_id = (
            self.get_parameter("default_frame_id").get_parameter_value().string_value
        )
        if publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be > 0.")
        if not self._default_frame_id:
            raise ValueError("default_frame_id must not be empty.")

        source_names = list(
            self.get_parameter("source_names")
            .get_parameter_value()
            .string_array_value
        )
        source_topics = list(
            self.get_parameter("source_topics")
            .get_parameter_value()
            .string_array_value
        )
        source_priorities = [
            int(value)
            for value in self.get_parameter("source_priorities")
            .get_parameter_value()
            .integer_array_value
        ]
        source_deadzones = [
            float(value)
            for value in self.get_parameter("source_deadzones")
            .get_parameter_value()
            .double_array_value
        ]
        source_timeouts_s = [
            float(value)
            for value in self.get_parameter("source_timeouts_s")
            .get_parameter_value()
            .double_array_value
        ]
        source_frame_overrides = list(
            self.get_parameter("source_frame_overrides")
            .get_parameter_value()
            .string_array_value
        )
        source_axis_mappings = [
            int(value)
            for value in self.get_parameter("source_axis_mappings")
            .get_parameter_value()
            .integer_array_value
        ]
        source_axis_scales = [
            float(value)
            for value in self.get_parameter("source_axis_scales")
            .get_parameter_value()
            .double_array_value
        ]

        source_count = len(source_topics)
        if source_count <= 0:
            raise ValueError("source_topics must contain at least one topic.")

        if len(source_names) != source_count:
            if len(source_names) == 0:
                source_names = [f"source_{index}" for index in range(source_count)]
            else:
                raise ValueError("source_names length must match source_topics length.")
        if len(source_priorities) != source_count:
            raise ValueError("source_priorities length must match source_topics length.")
        if len(source_deadzones) != source_count:
            raise ValueError("source_deadzones length must match source_topics length.")
        if len(source_timeouts_s) != source_count:
            raise ValueError("source_timeouts_s length must match source_topics length.")
        if len(source_frame_overrides) != source_count:
            raise ValueError(
                "source_frame_overrides length must match source_topics length."
            )
        if len(source_axis_mappings) != source_count * _AXIS_COUNT:
            raise ValueError(
                "source_axis_mappings length must be source_count * 6."
            )
        if len(source_axis_scales) != source_count * _AXIS_COUNT:
            raise ValueError("source_axis_scales length must be source_count * 6.")

        self._source_configs: list[_SourceConfig] = []
        for index in range(source_count):
            start = index * _AXIS_COUNT
            end = start + _AXIS_COUNT
            mapping = source_axis_mappings[start:end]
            scales = source_axis_scales[start:end]
            deadzone = source_deadzones[index]
            timeout_s = source_timeouts_s[index]
            if deadzone < 0.0:
                raise ValueError("source deadzones must be >= 0.")
            if timeout_s <= 0.0:
                raise ValueError("source timeouts must be > 0.")
            self._source_configs.append(
                _SourceConfig(
                    name=source_names[index],
                    topic=source_topics[index],
                    priority=source_priorities[index],
                    deadzone=deadzone,
                    timeout_s=timeout_s,
                    frame_override=str(source_frame_overrides[index]),
                    axis_mapping=mapping,
                    axis_scale=scales,
                )
            )

        self._samples: list[_SourceSample | None] = [None] * source_count
        self._active_source_index: int | None = None

        self._publisher = self.create_publisher(Joy, output_topic, 10)
        self._active_source_pub = self.create_publisher(
            String, active_source_topic, 10
        )
        self._subscriptions = []
        for index, source in enumerate(self._source_configs):
            sub = self.create_subscription(
                Joy,
                source.topic,
                lambda msg, src_idx=index: self._on_joy(src_idx, msg),
                10,
            )
            self._subscriptions.append(sub)

        period_s = 1.0 / publish_rate_hz
        self._timer = self.create_timer(period_s, self._on_timer)

        source_summary = ", ".join(
            f"{src.name}:{src.topic}(p={src.priority},dz={src.deadzone:.2f})"
            for src in self._source_configs
        )
        self.get_logger().info(
            f"JoyMux started. output='{output_topic}', default_frame_id='{self._default_frame_id}'"
        )
        self.get_logger().info(f"Active-source topic: '{active_source_topic}'")
        self.get_logger().info(f"Sources: {source_summary}")

    def _map_axes(self, source: _SourceConfig, incoming_axes: list[float]) -> list[float]:
        """Apply per-source axis remapping and scale into canonical 6-axis output."""
        axes = [0.0] * _AXIS_COUNT
        for axis_index in range(_AXIS_COUNT):
            source_idx = int(source.axis_mapping[axis_index])
            if source_idx < 0 or source_idx >= len(incoming_axes):
                axes[axis_index] = 0.0
                continue
            axes[axis_index] = (
                float(incoming_axes[source_idx]) * float(source.axis_scale[axis_index])
            )
        return axes

    def _on_joy(self, source_index: int, msg: Joy) -> None:
        """Store latest normalized sample for one source."""
        source = self._source_configs[source_index]
        incoming_axes = list(msg.axes) if msg.axes else []
        incoming_buttons = [int(value) for value in msg.buttons] if msg.buttons else []
        mapped_axes = self._map_axes(source, incoming_axes)
        frame_id = source.frame_override or str(msg.header.frame_id)
        self._samples[source_index] = _SourceSample(
            axes=mapped_axes,
            buttons=incoming_buttons,
            frame_id=frame_id,
            stamp_s=time.monotonic(),
        )

    def _is_sample_active(self, source: _SourceConfig, sample: _SourceSample) -> bool:
        """Return whether sample should be considered operator activity."""
        axis_activity = max(abs(value) for value in sample.axes) >= source.deadzone
        button_activity = any(int(value) != 0 for value in sample.buttons)
        return axis_activity or button_activity

    def _select_source(self, now_s: float) -> int | None:
        """Select active source index according to activity and priority."""
        active_candidates: list[int] = []
        for index, source in enumerate(self._source_configs):
            sample = self._samples[index]
            if sample is None:
                continue
            if now_s - sample.stamp_s > source.timeout_s:
                continue
            if self._is_sample_active(source, sample):
                active_candidates.append(index)

        if (
            self._active_source_index is not None
            and self._active_source_index in active_candidates
        ):
            return self._active_source_index

        if not active_candidates:
            return None

        # Lower priority value wins. Use source index to make ties deterministic.
        return min(
            active_candidates,
            key=lambda idx: (self._source_configs[idx].priority, idx),
        )

    def _publish_sample(self, sample: _SourceSample | None) -> None:
        """Publish selected sample or zero message when no source is active."""
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        if sample is None:
            msg.header.frame_id = self._default_frame_id
            msg.axes = [0.0] * _AXIS_COUNT
            msg.buttons = [0, 0]
        else:
            msg.header.frame_id = sample.frame_id or self._default_frame_id
            msg.axes = [float(value) for value in sample.axes]
            msg.buttons = [int(value) for value in sample.buttons]
        self._publisher.publish(msg)

    def _publish_active_source(self, selected_index: int | None) -> None:
        """Publish the currently selected source name (or empty string)."""
        msg = String()
        if selected_index is None:
            msg.data = ""
        else:
            msg.data = self._source_configs[selected_index].name
        self._active_source_pub.publish(msg)

    def _on_timer(self) -> None:
        """Periodic arbitration and output publication."""
        now_s = time.monotonic()
        selected_index = self._select_source(now_s)

        if selected_index != self._active_source_index:
            old_name = (
                self._source_configs[self._active_source_index].name
                if self._active_source_index is not None
                else "none"
            )
            new_name = (
                self._source_configs[selected_index].name
                if selected_index is not None
                else "none"
            )
            self.get_logger().info(f"Active teleop source: {old_name} -> {new_name}")
            self._active_source_index = selected_index

        sample = None
        if selected_index is not None:
            sample = self._samples[selected_index]
        self._publish_sample(sample)
        self._publish_active_source(selected_index)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = JoyMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
