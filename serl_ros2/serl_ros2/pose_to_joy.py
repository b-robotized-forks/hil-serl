#!/usr/bin/env python3
"""Pose-to-Joy teleoperation bridge.

This node converts a Cartesian target pose into normalized teleoperation commands
for HIL-SERL's Joy-based teleop path.

Design overview:
1. Subscribe to a target pose (e.g. from an RViz interactive marker).
2. Subscribe to current TCP pose.
3. At a fixed loop rate, compute a bounded incremental step from current -> target:
   - translation step length capped by `max_translation_step_m`
   - rotation step angle capped by `max_rotation_step_rad` via quaternion slerp
4. Convert that incremental step into normalized Joy axes in [-1, 1]:
   - if far from target, vector magnitude is 1.0 ("full stick")
   - if close to target, vector magnitude shrinks proportionally
5. Use a settle/resume hysteresis around the goal to avoid wobbling near target
   when TCP pose updates are noisy.
6. Optionally subscribe to a gripper command topic and emit short Joy button
   pulses for close/open actions.

Important integration note:
`output_command_frame` is the frame id written to `sensor_msgs/Joy.header.frame_id`.
That frame id also selects whether the node publishes base-frame deltas or
TCP-frame deltas (when it matches `base_frame_id` or `tcp_frame_id`).
"""

from dataclasses import dataclass
import time

from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from std_msgs.msg import UInt8


def _normalize_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Return a normalized quaternion in `[x, y, z, w]` format."""
    norm = float(np.linalg.norm(quat_xyzw))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat_xyzw / norm


def _rotation_from_quat(quat_xyzw: np.ndarray) -> Rotation:
    """Create a SciPy rotation from a quaternion in `[x, y, z, w]` format."""
    return Rotation.from_quat(_normalize_quat(quat_xyzw))


def _saturate_unit_ball(vec: np.ndarray) -> np.ndarray:
    """Clamp a vector to unit length while preserving direction."""
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0:
        return vec
    return vec / norm


@dataclass
class PoseSample:
    """Pose snapshot used by the controller."""

    position: np.ndarray
    quat_xyzw: np.ndarray
    frame_id: str


class PoseToJoy(Node):
    """Convert target and TCP poses into normalized Joy teleop commands.

    This node continuously drives from current TCP pose toward the latest target
    using bounded translational and rotational increments.
    """

    def __init__(self) -> None:
        """Initialize subscriptions, publisher, parameters, and control timer."""
        super().__init__("pose_to_joy")

        self.declare_parameter("target_pose_topic", "/pose_to_joy/target_pose")
        self.declare_parameter("tcp_pose_topic", "/hilserl/tcp_pose")
        self.declare_parameter("joy_topic", "teleop_joy")
        self.declare_parameter("reference_frame", "base_link")
        self.declare_parameter("output_command_frame", "tcp")
        self.declare_parameter("base_frame_id", "base")
        self.declare_parameter("tcp_frame_id", "tcp")
        self.declare_parameter("max_translation_step_m", 0.2)
        self.declare_parameter("max_rotation_step_rad", 0.35)
        self.declare_parameter("position_tolerance_m", 0.002)
        self.declare_parameter("rotation_tolerance_rad", 0.01)
        self.declare_parameter("position_resume_tolerance_m", 0.004)
        self.declare_parameter("rotation_resume_tolerance_rad", 0.02)
        self.declare_parameter("target_change_pos_eps_m", 0.001)
        self.declare_parameter("target_change_rot_eps_rad", 0.01)
        self.declare_parameter("publish_rate_hz", 40.0)
        self.declare_parameter("tcp_stale_timeout_s", 0.5)
        self.declare_parameter("require_frame_match", True)
        self.declare_parameter("enable_external_preempt_disarm", False)
        self.declare_parameter("active_source_topic", "/teleop/active_source")
        self.declare_parameter("active_source_name", "pose")
        self.declare_parameter("gripper_command_topic", "/pose_to_joy/gripper_command")
        self.declare_parameter("gripper_button_pulse_s", 0.2)

        self.target_pose_topic = (
            self.get_parameter("target_pose_topic").get_parameter_value().string_value
        )
        self.tcp_pose_topic = (
            self.get_parameter("tcp_pose_topic").get_parameter_value().string_value
        )
        self.joy_topic = (
            self.get_parameter("joy_topic").get_parameter_value().string_value
        )
        self.reference_frame = (
            self.get_parameter("reference_frame").get_parameter_value().string_value
        )
        self.output_command_frame = (
            self.get_parameter("output_command_frame")
            .get_parameter_value()
            .string_value
        )
        self.base_frame_id = (
            self.get_parameter("base_frame_id").get_parameter_value().string_value
        )
        self.tcp_frame_id = (
            self.get_parameter("tcp_frame_id").get_parameter_value().string_value
        )
        self.max_translation_step_m = (
            self.get_parameter("max_translation_step_m")
            .get_parameter_value()
            .double_value
        )
        self.max_rotation_step_rad = (
            self.get_parameter("max_rotation_step_rad")
            .get_parameter_value()
            .double_value
        )
        self.position_tolerance_m = (
            self.get_parameter("position_tolerance_m")
            .get_parameter_value()
            .double_value
        )
        self.rotation_tolerance_rad = (
            self.get_parameter("rotation_tolerance_rad")
            .get_parameter_value()
            .double_value
        )
        self.position_resume_tolerance_m = (
            self.get_parameter("position_resume_tolerance_m")
            .get_parameter_value()
            .double_value
        )
        self.rotation_resume_tolerance_rad = (
            self.get_parameter("rotation_resume_tolerance_rad")
            .get_parameter_value()
            .double_value
        )
        self.target_change_pos_eps_m = (
            self.get_parameter("target_change_pos_eps_m")
            .get_parameter_value()
            .double_value
        )
        self.target_change_rot_eps_rad = (
            self.get_parameter("target_change_rot_eps_rad")
            .get_parameter_value()
            .double_value
        )
        self.publish_rate_hz = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )
        self.tcp_stale_timeout_s = (
            self.get_parameter("tcp_stale_timeout_s").get_parameter_value().double_value
        )
        self.require_frame_match = (
            self.get_parameter("require_frame_match").get_parameter_value().bool_value
        )
        self.enable_external_preempt_disarm = (
            self.get_parameter("enable_external_preempt_disarm")
            .get_parameter_value()
            .bool_value
        )
        self.active_source_topic = (
            self.get_parameter("active_source_topic")
            .get_parameter_value()
            .string_value
        )
        self.active_source_name = (
            self.get_parameter("active_source_name")
            .get_parameter_value()
            .string_value
        )
        self.gripper_command_topic = (
            self.get_parameter("gripper_command_topic")
            .get_parameter_value()
            .string_value
        )
        self.gripper_button_pulse_s = (
            self.get_parameter("gripper_button_pulse_s")
            .get_parameter_value()
            .double_value
        )

        if self.max_translation_step_m <= 0.0:
            raise ValueError("max_translation_step_m must be > 0.")
        if self.max_rotation_step_rad <= 0.0:
            raise ValueError("max_rotation_step_rad must be > 0.")
        if self.position_tolerance_m < 0.0:
            raise ValueError("position_tolerance_m must be >= 0.")
        if self.rotation_tolerance_rad < 0.0:
            raise ValueError("rotation_tolerance_rad must be >= 0.")
        if self.position_resume_tolerance_m < 0.0:
            raise ValueError("position_resume_tolerance_m must be >= 0.")
        if self.rotation_resume_tolerance_rad < 0.0:
            raise ValueError("rotation_resume_tolerance_rad must be >= 0.")
        if self.target_change_pos_eps_m < 0.0:
            raise ValueError("target_change_pos_eps_m must be >= 0.")
        if self.target_change_rot_eps_rad < 0.0:
            raise ValueError("target_change_rot_eps_rad must be >= 0.")
        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be > 0.")
        if self.gripper_button_pulse_s < 0.0:
            raise ValueError("gripper_button_pulse_s must be >= 0.")
        if not self.base_frame_id:
            raise ValueError("base_frame_id must not be empty.")
        if not self.tcp_frame_id:
            raise ValueError("tcp_frame_id must not be empty.")
        if self.enable_external_preempt_disarm and not self.active_source_name:
            raise ValueError(
                "active_source_name must not be empty when "
                "enable_external_preempt_disarm is true."
            )
        if self.output_command_frame == "base":
            self.output_command_frame = self.base_frame_id
        elif self.output_command_frame == "tcp":
            self.output_command_frame = self.tcp_frame_id
        if self.output_command_frame not in (self.base_frame_id, self.tcp_frame_id):
            raise ValueError(
                "output_command_frame must match base_frame_id or tcp_frame_id."
            )

        if self.position_resume_tolerance_m < self.position_tolerance_m:
            self.position_resume_tolerance_m = self.position_tolerance_m
        if self.rotation_resume_tolerance_rad < self.rotation_tolerance_rad:
            self.rotation_resume_tolerance_rad = self.rotation_tolerance_rad

        self._target_pose: PoseSample | None = None
        self._tcp_pose: PoseSample | None = None
        self._last_tcp_rx_time = self.get_clock().now()
        self._last_warn_stale_tcp_s = 0.0
        self._last_warn_frame_mismatch_s = 0.0
        self._last_warn_invalid_gripper_cmd_s = 0.0
        self._target_active = False
        self._gripper_pulse_buttons = [0, 0]
        self._gripper_pulse_end_s = 0.0

        self._joy_pub = self.create_publisher(Joy, self.joy_topic, 10)
        self._target_sub = self.create_subscription(
            PoseStamped, self.target_pose_topic, self._on_target_pose, 10
        )
        self._tcp_sub = self.create_subscription(
            PoseStamped, self.tcp_pose_topic, self._on_tcp_pose, 10
        )
        self._gripper_cmd_sub = self.create_subscription(
            UInt8,
            self.gripper_command_topic,
            self._on_gripper_command,
            10,
        )
        self._active_source_sub = None
        if self.enable_external_preempt_disarm:
            self._active_source_sub = self.create_subscription(
                String,
                self.active_source_topic,
                self._on_active_source,
                10,
            )

        period_s = 1.0 / self.publish_rate_hz
        self._timer = self.create_timer(period_s, self._on_timer)

        self.get_logger().info(
            f"PoseToJoy started. target='{self.target_pose_topic}' "
            f"tcp='{self.tcp_pose_topic}' joy='{self.joy_topic}' "
            f"frame='{self.reference_frame}' "
            f"output_command_frame='{self.output_command_frame}'"
        )
        self.get_logger().info(
            "Goal settle/resume thresholds: "
            "position="
            f"{self.position_tolerance_m:.4f}/"
            f"{self.position_resume_tolerance_m:.4f} m, "
            "rotation="
            f"{self.rotation_tolerance_rad:.4f}/"
            f"{self.rotation_resume_tolerance_rad:.4f} rad."
        )
        self.get_logger().info(
            "Target-change activation thresholds: "
            f"position={self.target_change_pos_eps_m:.4f} m, "
            f"rotation={self.target_change_rot_eps_rad:.4f} rad."
        )
        self.get_logger().info(
            "TeleopIntervention will infer command frame from Joy.header.frame_id; "
            f"this node publishes frame_id='{self.output_command_frame}'."
        )
        self.get_logger().info(
            "Gripper command bridge: "
            f"topic='{self.gripper_command_topic}' pulse={self.gripper_button_pulse_s:.3f}s."
        )
        if self.enable_external_preempt_disarm:
            self.get_logger().info(
                "External preempt disarm enabled: "
                f"topic='{self.active_source_topic}', source='{self.active_source_name}'."
            )

    def _on_target_pose(self, msg: PoseStamped) -> None:
        """
        Store the latest target pose and arm teleop on meaningful target changes.

        Teleop is intentionally edge-triggered: once the current target is reached,
        output stays zero until a new target move event is detected.
        """
        new_target = self._pose_sample_from_msg(msg)
        previous_target = self._target_pose
        self._target_pose = new_target
        if self._target_changed_significantly(previous_target, new_target):
            self._target_active = True

    def _on_tcp_pose(self, msg: PoseStamped) -> None:
        """Store the latest measured TCP pose."""
        self._tcp_pose = self._pose_sample_from_msg(msg)
        self._last_tcp_rx_time = self.get_clock().now()

    def _on_gripper_command(self, msg: UInt8) -> None:
        """Convert gripper command events into short Joy button pulses."""
        command = int(msg.data)
        if command == 1:
            pulse_buttons = [1, 0]
        elif command == 0:
            pulse_buttons = [0, 1]
        else:
            now = time.monotonic()
            if now - self._last_warn_invalid_gripper_cmd_s > 1.0:
                self.get_logger().warn(
                    "Unsupported gripper command on "
                    f"'{self.gripper_command_topic}': {command} "
                    "(expected 0=open, 1=close)."
                )
                self._last_warn_invalid_gripper_cmd_s = now
            return

        now = time.monotonic()
        min_pulse_s = 1.0 / self.publish_rate_hz
        pulse_s = max(self.gripper_button_pulse_s, min_pulse_s)
        self._gripper_pulse_buttons = pulse_buttons
        self._gripper_pulse_end_s = now + pulse_s

    def _on_active_source(self, msg: String) -> None:
        """Disarm marker teleop when another source is selected by an external mux."""
        selected_source = str(msg.data).strip()
        if selected_source == self.active_source_name:
            return
        if not self._target_active:
            return
        self._target_active = False
        self._publish_zero_joy()
        self.get_logger().info(
            "PoseToJoy disarmed due to external source preemption "
            f"(selected='{selected_source or 'none'}')."
        )

    def _pose_sample_from_msg(self, msg: PoseStamped) -> PoseSample:
        """Convert a `PoseStamped` message to an internal `PoseSample`."""
        frame_id = msg.header.frame_id if msg.header.frame_id else self.reference_frame
        pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        quat = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        quat = _normalize_quat(quat)
        return PoseSample(position=pos, quat_xyzw=quat, frame_id=frame_id)

    def _tcp_is_fresh(self) -> bool:
        """Return whether the latest TCP sample is within timeout."""
        age = (self.get_clock().now() - self._last_tcp_rx_time).nanoseconds * 1e-9
        return age <= self.tcp_stale_timeout_s

    def _compute_error_norms(
        self, current: PoseSample, target: PoseSample
    ) -> tuple[float, float]:
        """Return positional and rotational error magnitudes."""
        pos_error = target.position - current.position
        pos_dist = float(np.linalg.norm(pos_error))
        r_curr = _rotation_from_quat(current.quat_xyzw)
        r_tgt = _rotation_from_quat(target.quat_xyzw)
        rot_error_angle = float((r_curr.inv() * r_tgt).magnitude())
        return pos_dist, rot_error_angle

    def _target_changed_significantly(
        self,
        previous_target: PoseSample | None,
        new_target: PoseSample,
    ) -> bool:
        """
        Return True when a target update is a meaningful user move event.

        The first received target establishes baseline and does not arm teleop.
        """
        if previous_target is None:
            return False
        if previous_target.frame_id != new_target.frame_id:
            return True
        pos_delta_m = float(np.linalg.norm(new_target.position - previous_target.position))
        rot_delta_rad = float(
            (
                _rotation_from_quat(previous_target.quat_xyzw).inv()
                * _rotation_from_quat(new_target.quat_xyzw)
            ).magnitude()
        )
        return (
            pos_delta_m > self.target_change_pos_eps_m
            or rot_delta_rad > self.target_change_rot_eps_rad
        )

    def _on_timer(self) -> None:
        """Publish periodic normalized Joy commands from target-vs-current pose."""
        if self._target_pose is None or self._tcp_pose is None:
            self._publish_zero_joy()
            return
        if not self._target_active:
            self._publish_zero_joy()
            return

        if not self._tcp_is_fresh():
            now = time.monotonic()
            if now - self._last_warn_stale_tcp_s > 1.0:
                self.get_logger().warn("TCP pose is stale; publishing zero Joy.")
                self._last_warn_stale_tcp_s = now
            self._publish_zero_joy()
            return

        target = self._target_pose
        tcp = self._tcp_pose

        if self.require_frame_match and target.frame_id != tcp.frame_id:
            now = time.monotonic()
            if now - self._last_warn_frame_mismatch_s > 1.0:
                self.get_logger().warn(
                    f"Frame mismatch target='{target.frame_id}' "
                    f"tcp='{tcp.frame_id}'; publishing zero Joy."
                )
                self._last_warn_frame_mismatch_s = now
            self._publish_zero_joy()
            return

        pos_dist, rot_error_angle = self._compute_error_norms(
            current=tcp, target=target
        )

        if (
            pos_dist <= self.position_tolerance_m
            and rot_error_angle <= self.rotation_tolerance_rad
        ):
            self._target_active = False
            self._publish_zero_joy()
            return

        joy_axes = self._compute_normalized_axes(current=tcp, target=target)
        self._publish_joy(joy_axes)

    def _compute_normalized_axes(
        self, current: PoseSample, target: PoseSample
    ) -> np.ndarray:
        """Compute normalized `[x, y, z, roll, pitch, yaw]` Joy axes."""
        # Cartesian error in base frame (meters): target minus current.
        pos_error_m = target.position - current.position
        pos_dist_m = float(np.linalg.norm(pos_error_m))

        # Bounded translational step toward target:
        # - zero when already inside position tolerance
        # - otherwise keep direction but cap magnitude to max_translation_step_m
        if pos_dist_m <= self.position_tolerance_m:
            translation_step_m = np.zeros(3, dtype=np.float64)
        else:
            step_m = min(pos_dist_m, self.max_translation_step_m)
            translation_step_m = pos_error_m * (step_m / pos_dist_m)

        r_curr = _rotation_from_quat(current.quat_xyzw)
        r_tgt = _rotation_from_quat(target.quat_xyzw)

        # Shortest-arc angular error (radians) from current orientation to target.
        rot_error_rad = float((r_curr.inv() * r_tgt).magnitude())
        if rot_error_rad <= self.rotation_tolerance_rad:
            step_rotvec = np.zeros(3, dtype=np.float64)
        else:
            # Bounded rotational step: slerp toward target with capped step angle.
            alpha = min(1.0, self.max_rotation_step_rad / rot_error_rad)
            key_times = np.array([0.0, 1.0], dtype=np.float64)
            key_rots = Rotation.from_quat(
                np.stack([r_curr.as_quat(), r_tgt.as_quat()], axis=0)
            )
            r_step = Slerp(key_times, key_rots)([alpha])[0]
            # Base-frame delta that maps current -> step (pre-multiply convention).
            step_rotvec = (r_step * r_curr.inv()).as_rotvec()

        if self.output_command_frame == self.tcp_frame_id:
            # Convert base-frame step into TCP-local command deltas.
            translation_step_m = r_curr.inv().apply(translation_step_m)
            step_rotation = Rotation.from_rotvec(step_rotvec)
            step_rotvec = (r_curr.inv() * step_rotation * r_curr).as_rotvec()

        # Normalize command steps to Joy range:
        # - full stick (norm 1) at max step limits
        # - proportionally smaller when the target is closer
        joy_xyz = translation_step_m / self.max_translation_step_m
        joy_xyz = _saturate_unit_ball(joy_xyz)
        joy_rpy = step_rotvec / self.max_rotation_step_rad
        joy_rpy = _saturate_unit_ball(joy_rpy)

        axes = np.concatenate([joy_xyz, joy_rpy], axis=0)
        return np.clip(axes, -1.0, 1.0)

    def _active_gripper_buttons(self) -> list[int]:
        """Return current gripper button pulse state."""
        if time.monotonic() <= self._gripper_pulse_end_s:
            return list(self._gripper_pulse_buttons)
        self._gripper_pulse_buttons = [0, 0]
        return [0, 0]

    def _publish_joy(self, axes: np.ndarray) -> None:
        """Publish a Joy command message for the provided six-axis vector."""
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.output_command_frame
        msg.axes = [float(value) for value in axes.tolist()]
        # TODO: Extend beyond binary open/close commands when needed.
        msg.buttons = self._active_gripper_buttons()
        self._joy_pub.publish(msg)

    def _publish_zero_joy(self) -> None:
        """Publish a zero Joy command."""
        self._publish_joy(np.zeros(6, dtype=np.float64))


def main(args: list[str] | None = None) -> None:
    """Start the PoseToJoy ROS2 node."""
    rclpy.init(args=args)
    node = PoseToJoy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
