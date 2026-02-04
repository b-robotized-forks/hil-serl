## serl_ros2_sim

[Ursina](https://www.ursinaengine.org/)-based ROS2 simulation node that mirrors `/hilserl/command_pose` into `/hilserl/tcp_pose` and
publishes the ROS2 topics expected by `serl_ros2.RobotAdapter`.

### Install Ursina

Ursina is a small open source Python game engine
([website](https://www.ursinaengine.org/), [GitHub](https://github.com/pokepetter/ursina)):

```bash
pip install ursina
```

### Run the sim

From the repository root:

```bash
python serl_ros2/serl_ros2_sim/ursina_sim.py --rate 50 --cameras front,wrist --image-width 128 --image-height 128 --publish-images
```

Run a **smoke test** to ensure that the RobotAdapter is correctly connected to the SIM topics:

```bash
cd serl_ros2/serl_ros2
python3 smoke.py --config ../../examples/experiments/cube_demo_ros2/adapter_config.yaml
```

> [!NOTE]
> `adapter_config.yaml` is not only for this smoke/test context.
> In HIL-SERL experiment runs, the same adapter config is loaded indirectly via
> `TrainConfig.adapter_config_path` in the experiment `config.py`.

You should be getting confirmation of the data having been received in the observation:

```bash
[INFO] [1769351738.651243354] [serl_ros2_smoke]: Waiting up to 10.0s for observation...
[INFO] [1769351738.802083101] [serl_ros2_smoke]: Observation received:
  timestamp: 1769351738.7383022
  state keys: ['tcp_pose', 'tcp_vel', 'tcp_force', 'tcp_torque', 'gripper_pose', 'q', 'dq']
  state: {'tcp_pose': [0.6000000238418579, 0.0, 0.4000000059604645, 0.0, 0.0, 0.0, 1.0], 'tcp_vel': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'tcp_force': [0.0, 0.0, 0.0], 'tcp_torque': [0.0, 0.0, 0.0], 'gripper_pose': 1.0, 'q': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'dq': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
```

Next, you can confirm that you can set the pose via message (and through the RobotAdapter):
```
ros2 topic pub /hilserl/command_pose geometry_msgs/msg/PoseStamped "header:
  stamp:
    sec: 1769350780
    nanosec: 338216213
  frame_id: base_link
pose:
  position:
    x: 0.6
    y: 0.0
    z: 0.4000000059604645
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
" --once
```

#### Understand "Robot (sim/real)" side vs. "HIL-SERL" side of the RobotAdapter

The SIM node is treated as the "robot endpoint": it only publishes state and
executes incoming `/hilserl/command_pose` updates. So the Ursina simulation does NOT use the `RobotAdapter`.
It only implements the ROS2 topics that are required for the `serl_ros2.RobotAdapter` implementation to work.

Teleop (e.g., SpaceMouse) runs on the HIL-SERL side and commands are received within a `TeleopAdapter`, which
is held by the `RobotAdapter`. It then is a `gymnasium` env wrapper which queries this TeleopAdapter for the
input coming from the SpaceMouse (or other teleop).
It then feeds this into the policy/env loop, which in turn then sends pose commands
through the RobotAdapter. The SIM itself does not read SpaceMouse input directly.

#### Control the ball with Teleop (SpaceMouse, Keyboard/Gamepad or Pose-To-Joy via ROS2 Joy)

To emulate how Hil-Serl `RobotAdapter` reacts to teleop input, and test it with the Ursina sim,
we have to use another script which explicitly (especially for demonstration) runs in a different node,
and uses `RobotAdapter`.

Run the test node from the `hil-serl` directory.

First launch the ursina simulator with the simple ball (simulates the end effector)

```
python3 serl_ros2/serl_ros2_sim/ursina_sim.py
```

Now, test the teleop. You have three options: spacemouse, keyboard/gamepad, or pose-to-joy
(all Joy-based options publish `sensor_msgs/Joy` messages).

**Option A: SpaceMouse**

```bash
python serl_ros2/serl_ros2_sim/test_teleop.py --config serl_ros2/serl_ros2_sim/adapter_config.yaml 
```

By default, SpaceMouse commands are tagged as TCP-frame (`--spacemouse-frame-id tcp`).
Use base frame instead with:

```bash
python serl_ros2/serl_ros2_sim/test_teleop.py --config serl_ros2/serl_ros2_sim/adapter_config.yaml --spacemouse-frame-id base
```

**Option B*: Keyboard or Gamepad (via ROS2 Joy)**

This requires two terminals. In Terminal 1, run one of:

```bash
# Keyboard
ros2 run serl_ros2 keyboard_joy

# >>OR<< Xbox/gamepad (device_id=1 for js1, adjust as needed)
ros2 run joy joy_node --ros-args -p device_id:=1 -r joy:=teleop_joy
```

> [!NOTE]
> ROS2 `joy_node` publishes `Joy.header.frame_id = "joy"` and currently has no
> parameter to override that frame id.
> The frame ID is used to distinguish in which frame the teleop is to be interpreted
> (tcp or base). Tose don't necessarily correspond to the ROS2/tf frame ID's - they
> are the frames as interpreted by the `TeleopIntervention`.
>
> Temporary HIL-SERL workaround: `"joy"` is interpreted as TCP-frame teleop.
> If you need another frame id, insert a relay node that rewrites
> `Joy.header.frame_id`.

In Terminal 2, run the teleop test (**NOTE:** Add argument `--joy-preset xbox` if you are using the xbox controller):
```bash
python serl_ros2/serl_ros2_sim/test_teleop.py --config serl_ros2/serl_ros2_sim/adapter_config.yaml  --teleop joy
```

**Option C: Pose-To-Joy (target pose -> Joy)**

This is not useful with the Ursina simulator - only when using some interface which
publishes a target pose, like an RViz marker.

This requires three terminals:

```bash
# Terminal 1: Pose-to-Joy bridge
ros2 run serl_ros2 pose_to_joy
```

```bash
# Terminal 2: Publish a target pose (or use your own marker publisher like from an RViz marker)
ros2 topic pub --rate 10 /pose_to_joy/target_pose geometry_msgs/msg/PoseStamped "
header: {frame_id: base_link}
pose:
  position: {x: 2.5, y: 0.0, z: 0.0}
  orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
"
```

```bash
# Terminal 3: Teleop test node consuming Joy
python serl_ros2/serl_ros2_sim/test_teleop.py --config serl_ros2/serl_ros2_sim/adapter_config.yaml --teleop joy
```

**Option D: Combined teleop sources via `joy_mux`**

If you want to run multiple sources at the same time (marker + keyboard + joy
device) and have only the currently active one drive teleop, launch:

```bash
ros2 launch serl_ros2 teleop_mux.launch.py
```

Then run:

```bash
python serl_ros2/serl_ros2_sim/test_teleop.py --config serl_ros2/serl_ros2_sim/adapter_config.yaml --teleop joy
```

See [serl_ros2 README - Teleop](../README.md#teleop) for more details regarding keyboard controls, device selection, and axis remapping.

> [!IMPORTANT]
> Ensure that your performance is OK - you may get unsmooth behavior with teleop control.
> While SIM and TELEOP are up as described above, check the frequency of the /hilserl/command_pose topic:
> ```
> ros2 topic hz /hilserl/command_pose 
> average rate: 1.632
>	min: 0.001s max: 2.136s std dev: 0.75184s window: 6
> ```
> If you see high `max` values, then you have an issue!
> 
> Please check the [performance notes in the walkthrough](../../docs/robot_walkthrough.md#ros2-timer-jitter-on-intel-hybrid-cpus) for details.
> 
> Shortcut for a fix: prepend `taskset -c <cpu>`:
> 
> ```
> taskset -c 2 python3 serl_ros2/serl_ros2_sim/ursina_sim.py
> ```
> and
> ```
> taskset -c 4 python serl_ros2/serl_ros2_sim/test_teleop.py --config examples/experiments/cube_demo_ros2/adapter_config.yaml --rate 40
> ```
