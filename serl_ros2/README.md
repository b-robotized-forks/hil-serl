# serl_ros2

ROS2 adapter package for HIL-SERL.

This package provides a ROS2 implementation of the `RobotAdapter` interface, enabling
HIL-SERL environments to communicate with robots via ROS2 topics and services.

## Installation

This package depends on the ROS-free interface in `serl_framework`.
Install it first, then install `serl_ros2`:

```bash
pip install -e ../serl_framework
pip install -e .
```

For ROS2 workspace builds:
```bash
colcon build --packages-select serl_ros2
```

## Topic naming and remapping

All ROS2 topics use `/hilserl`-prefixed names by default (e.g., `/hilserl/tcp_pose`,
`/hilserl/command_pose`). Standard ROS2 remapping can still be used, but the current
RobotAdapter hard-codes these absolute topic names.

**Subscribed topics:**
- `/hilserl/tcp_pose` (geometry_msgs/PoseStamped)
- `/hilserl/tcp_twist` (geometry_msgs/TwistStamped)
- `/hilserl/joint_states` (sensor_msgs/JointState)
- `/hilserl/gripper_pos` (std_msgs/Float32)
- `/hilserl/tcp_wrench` (geometry_msgs/WrenchStamped) - optional
- `/hilserl/camera/<name>/image` (sensor_msgs/Image) - per camera

Teleop input is handled by a TeleopAdapter. If you pass one into
`serl_ros2.RobotAdapter`, it will be available via `get_teleop()`.
This package provides `JoyTeleopAdapter` which subscribes to `sensor_msgs/Joy`
messages, enabling keyboard, gamepad, or pose-target-driven control (via
`pose_to_joy`). See [Teleop](#teleop) below.

**Published topics:**
- `/hilserl/command_pose` (geometry_msgs/PoseStamped)
- `/hilserl/command_wrench` (geometry_msgs/WrenchStamped)

**Services:**
- `/hilserl/clear_error` (std_srvs/Trigger)
- `/hilserl/reset_world` (std_srvs/Trigger, optional)
- `/teleop/reset_signal` (std_srvs/Trigger, optional)

## Configuration

Copy and customize `config/adapter_example.yaml` for your setup. Key options:
- `cameras`: List of camera names to subscribe to
- `sync_slop`: Time tolerance for state synchronization (default: 0.05s)
- `subscribe_wrench`: Enable/disable wrench subscription
- `frame_id`: Frame ID for outgoing pose commands

## Usage

For the ROS-agnostic usage pattern (a plain Python main loop using
`RobotAdapter.create()` / `call_service()`, or the same-thread `run()` variant)
and interface notes, see `../serl_framework/README.md`.

## Smoke test

Run a simple adapter smoke test (requires ROS2 topics to be publishing):

```bash
ros2 run serl_ros2 serl_ros2_smoke --config config/adapter_example.yaml
```

## Running unit tests

```bash
# Via colcon (preferred for ROS2)
colcon test --packages-select serl_ros2 --base-paths .
colcon test-result --verbose

# Or directly with pytest
pytest -q ./test
```

## Testing framework

We use `pytest` for unit tests because it is the common choice in ROS2 Python
packages (including `rclpy`), provides concise assertions and fixtures, and is
well supported by `ament_pytest` and `colcon test`.

## Teleop

Teleop in the ROS2 integration is Joy-driven end-to-end:
- one or more teleop input nodes publish `sensor_msgs/Joy`
- `JoyTeleopAdapter` reads those Joy messages
- `TeleopIntervention` consumes adapter output and injects intervention actions

Frame routing is also Joy-driven: `TeleopIntervention` infers base-vs-tcp from
`Joy.header.frame_id`. Keep publisher frame ids consistent with experiment config
(`teleop_base_frame_id` / `teleop_tcp_frame_id`).

This package provides `JoyTeleopAdapter`, a `TeleopAdapter` subclass that subscribes
to `sensor_msgs/Joy` messages, enabling keyboard, gamepad, or pose-target-driven
control without a SpaceMouse.

When using RViz marker + `pose_to_joy`, `RobotEnv.go_to_reset()` can call the
optional service `/teleop/reset_signal` through `RobotAdapter` to notify
teleop-side nodes that a reset completed. The interactive marker node reacts by
realigning the marker target with current TCP pose.

For simulation setups, `RobotEnv.go_to_reset()` can also call optional
`/hilserl/reset_world` through `RobotAdapter` to request scene/world resets
(for example when task objects can topple or drift).

### JoyTeleopAdapter

`JoyTeleopAdapter` maps Joy axes/buttons to 6-DOF robot control:

| Joy Field | Robot Control |
|-----------|---------------|
| axes[0-2] | X, Y, Z translation |
| axes[3-5] | Roll, Pitch, Yaw rotation |
| buttons[0] | Gripper close / intervention trigger |
| buttons[1] | Gripper open |
| header.frame_id | Teleop source frame |

Usage in your experiment config:

```python
from serl_ros2.joy_teleop_adapter import JoyTeleopAdapter
from serl_ros2.robot_adapter import RobotAdapter

teleop = JoyTeleopAdapter(topic="teleop_joy")
adapter = RobotAdapter.create(config_path="config.yaml", teleop_adapter=teleop)
adapter.start()
```

### SpaceMouse (Direct Device)

Direct SpaceMouse input is provided by `serl_framework.SpaceMouseTeleop`
(not by this package's Joy adapter). In experiment configs, this is typically:

```python
use_teleop = True
teleop_device = "spacemouse"
teleop_spacemouse_frame_id = "tcp"  # or "base"
```

**Runtime frame toggle:** Press both SpaceMouse buttons simultaneously to switch
between `tcp` and `base` frame. Individual button presses (gripper open/close)
are suppressed during the combo to avoid false triggers.

**Axis mapping:** SpaceMouse axes can be remapped, inverted, and swapped per
frame via `spacemouse_axis_mapping` in `experiment_config.yaml`:

```yaml
spacemouse_axis_mapping:
  tcp:
    xyz_axes: [["x", 1], ["y", -1], ["z", -1]]
    rpy_axes: [["pitch", -1], ["roll", -1], ["yaw", 1]]
  base:
    xyz_axes: [["x", -1], ["y", 1], ["z", -1]]
    rpy_axes: [["pitch", 1], ["roll", 1], ["yaw", 1]]
```

Each entry is `[pyspacemouse_attr, sign]`. The order defines the output mapping:
entry 0 maps to X/Roll, entry 1 to Y/Pitch, entry 2 to Z/Yaw. Set `sign` to
`-1` to invert an axis. Swap entries to swap axes (e.g., `[["y", -1], ["x", 1], ["z", 1]]`
swaps X and Y and inverts the new X).

See `serl_framework/README.md` for SpaceMouse setup and permissions.

### Keyboard (`keyboard_joy`)

For keyboard-based control, this package provides a `keyboard_joy` node that publishes
Joy messages from keyboard input:

```bash
# Run with ros2 (requires colcon install)
ros2 run serl_ros2 keyboard_joy --ros-args -p frame_id:=tcp

# Or run directly without colcon
python serl_ros2/serl_ros2/keyboard_joy.py

# With axis mapping from config
ros2 run serl_ros2 keyboard_joy --ros-args \
  -p config:=path/to/adapter_config.yaml
```

Set `frame_id:=base` if you intentionally want base-frame keyboard commands.

**Runtime frame toggle:** Press **Tab** to switch between `tcp` and `base`
frame. The active frame is printed to the console on each toggle.

**Default keyboard controls:**

| Control | Keys | Joy Axis/Button |
|---------|------|-----------------|
| X translation | W / S | axes[0] |
| Y translation | A / D | axes[1] |
| Z translation | R / F | axes[2] |
| Roll | J / L | axes[3] |
| Pitch | I / K | axes[4] |
| Yaw | U / O | axes[5] |
| Gripper close | F4 | buttons[0] |
| Gripper open | F3 | buttons[1] |
| Speed boost | SHIFT (hold) | 2x multiplier |
| Toggle frame | TAB | tcp <-> base |

**Axis mapping:** Key-to-axis assignments can be remapped per frame via
`keyboard_axis_mapping` in `experiment_config.yaml`:

```yaml
keyboard_axis_mapping:
  tcp:
    - ["w", "s", 0, 1]    # W/S -> axis 0 (X), sign=1
    - ["a", "d", 1, 1]    # A/D -> axis 1 (Y)
    - ["r", "f", 2, 1]    # R/F -> axis 2 (Z)
    - ["j", "l", 3, 1]    # J/L -> axis 3 (Roll)
    - ["i", "k", 4, 1]    # I/K -> axis 4 (Pitch)
    - ["u", "o", 5, 1]    # U/O -> axis 5 (Yaw)
  base:
    - ["w", "s", 0, -1]   # W/S -> axis 0 (X), inverted
    - ["a", "d", 1, 1]    # A/D -> axis 1 (Y)
    - ["r", "f", 2, 1]    # R/F -> axis 2 (Z)
    - ["j", "l", 3, 1]    # J/L -> axis 3 (Roll)
    - ["i", "k", 4, 1]    # I/K -> axis 4 (Pitch)
    - ["u", "o", 5, 1]    # U/O -> axis 5 (Yaw)
```

Each entry is `[key_positive, key_negative, axis_index, sign]`. Change
`axis_index` to remap which Joy axis a key pair controls. Set `sign` to
`-1` to invert the direction. The mapping is loaded from the YAML at startup
via the `config` parameter. Without it, the default mapping is used for both
frames.

> **Note:** The keyboard node needs terminal focus to capture input. Run it in a
> separate terminal from your training script.

### Gamepad (`joy_node`)

Any ROS2-compatible gamepad can be used with `JoyTeleopAdapter` via the standard
`joy` package:

```bash
sudo apt install ros-${ROS_DISTRO}-joy
ros2 run joy joy_node --ros-args -p device_id:=0 -r joy:=teleop_joy
```

> [!NOTE]
> `joy_node` publishes `Joy.header.frame_id = "joy"` and currently has no
> frame-id parameter. Temporary HIL-SERL workaround interprets `"joy"` as
> TCP-frame teleop.

Check the output in a separate terminal:

```bash
ros2 topic echo teleop_joy
```

> [!NOTE]
> **Finding available devices:**
> 
> To see all detected joystick devices:
>
> ```bash
> # List joystick device nodes
> ls -la /dev/input/js*
> 
> # Show detailed info for all input devices
> cat /proc/bus/input/devices | grep -A 5 "Name"
> ```
> 
> Example output:
> ```
> /dev/input/js0  ->  3Dconnexion Universal Receiver (SpaceMouse)
> /dev/input/js1  ->  Microsoft X-Box One pad
> ```
> Use the `device_id` parameter (integer) to select which device to use:


Typical Xbox controller mapping:
- Left stick: axes[0] (L/R), axes[1] (U/D)
- Right stick: axes[3] (L/R), axes[4] (U/D)
- Triggers: axes[2] (LT), axes[5] (RT)


**Troubleshooting: Controller shows static values**

If your controller is detected but axes/buttons don't change when you press them,
the controller may be disconnected or in a bad state. Try:

1. **Unplug and replug** the controller
2. Verify it's detected: `ls /dev/input/js*`
3. For wireless controllers over USB, hold the Xbox button for 3 seconds while
   plugging in to ensure it enters wired mode

### Pose Targets (`pose_to_joy`)

`JoyTeleopAdapter` can also consume Joy commands generated from Cartesian pose
targets. This package includes `pose_to_joy`, which subscribes to:
- target pose topic (`geometry_msgs/PoseStamped`)
- current TCP pose topic (`/hilserl/tcp_pose`)
- optional gripper command topic (`std_msgs/UInt8`, `0=open`, `1=close`)

and publishes normalized Joy commands (`teleop_joy`) that drive the TCP toward
the target in bounded translational/rotational increments.

Example:

```bash
ros2 run serl_ros2 pose_to_joy
```

`pose_to_joy` publishes TCP-frame deltas by default (`output_command_frame:=tcp`).
To publish base-frame deltas instead:

```bash
ros2 run serl_ros2 pose_to_joy --ros-args -p output_command_frame:=base
```

With custom frame names, set all three consistently:

```bash
ros2 run serl_ros2 pose_to_joy --ros-args \
  -p base_frame_id:=base \
  -p tcp_frame_id:=tcp \
  -p output_command_frame:=tcp
```

`pose_to_joy` uses edge-triggered intervention: it only becomes active when the
target pose changes significantly, and it automatically deactivates once the
target is reached (within tolerance). This prevents snap-back to an unchanged
marker target while policy control is running.

Activation thresholds can be tuned:

```bash
ros2 run serl_ros2 pose_to_joy --ros-args \
  -p target_change_pos_eps_m:=0.001 \
  -p target_change_rot_eps_rad:=0.01
```

You will need a node which publishes a `Pose` topic which publishes the TCP target pose.
This can for example be out of an RViz interactive marker.
If your marker menu (or other UI) publishes gripper open/close events on
`/pose_to_joy/gripper_command`, they are converted to Joy button pulses.

> [!IMPORTANT]
> With `TeleopIntervention`, frame routing is inferred from `Joy.header.frame_id`.
> Ensure your Joy publishers use frame ids that match the environment config
> (typically in config.py, `teleop_base_frame_id` / `teleop_tcp_frame_id`).
>
> For `pose_to_joy`, `output_command_frame` controls which frame id gets
> published (`base_frame_id` or `tcp_frame_id`).

> [!NOTE]
> The upstream ROS2 `joy_node` publishes `Joy.header.frame_id = "joy"` and does
> not expose a `frame_id` parameter. Current temporary workaround in
> `TeleopIntervention`: `"joy"` is interpreted as TCP-frame commands.
> If you need a different frame id, use a small relay node that rewrites
> `Joy.header.frame_id` before forwarding to `teleop_joy`.

### Multiple Devices (`joy_mux`)

To use multiple teleop sources at once (e.g. RViz marker via `pose_to_joy`,
keyboard, and SpaceMouse via `joy_node`) while ensuring only one active source
drives commands at a time, use `joy_mux`.

`joy_mux` selects the source currently outside deadzone (or pressing buttons)
and republishes it to a single output topic (`teleop_joy` by default).

Example launch:

```bash
ros2 launch serl_ros2 teleop_mux.launch.py
```

This starts:
- `pose_to_joy` -> `teleop_joy_pose`
- `joy_node` -> `teleop_joy_spacemouse`
- `keyboard_joy` -> `teleop_joy_keyboard`
- `joy_mux` -> `teleop_joy`

By default, all sources publish/resolve to TCP frame. To force all sources to
use one frame explicitly, use:

```bash
# Force all teleop sources to TCP
ros2 launch serl_ros2 teleop_mux.launch.py frame_id_all:=tcp

# Force all teleop sources to BASE
ros2 launch serl_ros2 teleop_mux.launch.py frame_id_all:=base
```

`frame_id_all` overrides `pose_output_command_frame`, `keyboard_frame_id`,
and the mux override used for `joy_node` input.

Disable only `joy_node` (for example when no joystick/spacemouse device is connected):

```bash
ros2 launch serl_ros2 teleop_mux.launch.py \
  enable_joy:=false
```

`enable_joy_node` is kept as a legacy alias for `enable_joy`.

`joy_mux` supports per-source axis mapping and per-source axis scale. This lets
you correct device-specific axes (swap/invert) before arbitration, so you can
match SpaceMouse-like behavior even when the device arrives through `joy_node`.

With the provided launch defaults, `pose_to_joy` is automatically disarmed when
another source is selected by `joy_mux`. It only re-arms on the next meaningful
marker target move event, which avoids snap-back to stale marker targets after
switching to another teleop device.

### Axis Remapping

`JoyTeleopAdapter` supports remapping axes to match your device or preference.

#### `axis_mapping` - Reassign which Joy axis controls each DOF

```python
# Example: Swap X and Y axes
teleop = JoyTeleopAdapter(
    topic="teleop_joy",
    axis_mapping={
        "x": 1,  # Use Joy axis 1 for X (default is 0)
        "y": 0,  # Use Joy axis 0 for Y (default is 1)
        "z": 2,
        "roll": 3,
        "pitch": 4,
        "yaw": 5,
    }
)
```

#### `axis_scale` - Invert or scale axis values

```python
# Example: Invert Y axis and reduce Z sensitivity
teleop = JoyTeleopAdapter(
    topic="teleop_joy",
    axis_scale={
        "y": -1.0,   # Invert Y
        "z": 0.5,    # Half sensitivity for Z
    }
)
```
