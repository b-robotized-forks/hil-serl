# Integrating Your Own Robot

This guide describes how to connect a new robot to HIL-SERL. The result is a small bridge on the
robot side that provides the topics and services listed in the
[contract](#robot-integration-contract) below. Once that bridge works, everything else
(teleop, data collection, training) runs unchanged, as described in the
[training walkthrough](robot_walkthrough.md).

Prerequisites: a ROS2 driver for your robot that can track Cartesian TCP pose targets
(e.g. an impedance or admittance controller), and the installed packages from
[INSTALL.md](../INSTALL.md).

A complete reference implementation of the robot side is the bundled simulator,
[serl_ros2/serl_ros2_sim/ursina_sim.py](../serl_ros2/serl_ros2_sim/ursina_sim.py).
It publishes and serves exactly this contract, so it is a good template for your bridge node.

## 1. Provide the interface

Write a node (or launch configuration) that maps your driver's topics and services to the
HIL-SERL names. Typically this means:

- republishing TCP pose, twist, joint states, gripper state, and camera streams under the
  `/hilserl/...` names,
- forwarding `/hilserl/command_pose` to your Cartesian controller,
- implementing the small set of services (gripper, reset, clear error, compliance).

The current `serl_ros2.RobotAdapter` hard-codes the absolute `/hilserl/...` names, so the robot
side must use exactly these names (see [serl_ros2/README.md](../serl_ros2/README.md)).

### Robot integration contract

**Observations** delivered to the env keep the upstream field names:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `tcp_pose` | float32[7] | **Required** | xyz position + quaternion (xyzw) |
| `tcp_vel` | float32[6] | **Required** | linear (3) + angular (3) velocity, from TCP twist |
| `tcp_force` | float32[3] | Optional | from the external TCP wrench, for contact tasks |
| `tcp_torque` | float32[3] | Optional | from the external TCP wrench, for contact tasks |
| `gripper_pose` | float32 | **Required** | normalized [0,1], 0=closed, 1=open |
| `q`, `dq` | float32[N] | Required | joint positions/velocities, from joint_states |

**Topics the robot publishes** (robot to adapter):

- `hilserl/tcp_pose` (`geometry_msgs/PoseStamped`). If unavailable, compute from joint states via FK.
- `hilserl/tcp_twist` (`geometry_msgs/TwistStamped`). If unavailable, compute from joint velocities
  and the Jacobian.
- `hilserl/tcp_wrench` (`geometry_msgs/WrenchStamped`), optional. The external wrench at the TCP,
  not the commanded one. If absent, the adapter fills zeros and tasks run without force/torque cues.
- `hilserl/gripper_pos` (`std_msgs/Float32`), normalized [0,1] (do not publish raw joint positions).
- `hilserl/joint_states` (`sensor_msgs/JointState`).
- `hilserl/camera/<name>/image` + `hilserl/camera/<name>/camera_info`. Any image size works,
  `RobotEnv` crops/resizes to 128x128, so publishing small images saves bandwidth.

**Topics the robot consumes** (adapter to robot):

- `hilserl/command_pose` (`geometry_msgs/PoseStamped`): absolute TCP target to track.
- `hilserl/command_wrench` (`geometry_msgs/WrenchStamped`), optional and not yet fully supported.

**Services the robot provides:**

- `hilserl/set_gripper` (`serl_msgs/SetGripper`). For `MODE_POSITION`, normalized [0,1] with
  0=closed, 1=open.
- `hilserl/reset_robot` (`serl_msgs/ResetRobot`): a **hard joint reset**, only called when
  explicitly requested (e.g. `JOINT_RESET_PERIOD > 0`). Normal per-episode resets are Cartesian
  moves to `RESET_POSE` (optionally randomized) and do not use this service.
- `hilserl/reset_world` (`std_srvs/Trigger`), optional. Called on every env reset to reset
  simulation objects.
- `hilserl/set_compliance` (`serl_msgs/SetCompliance`): map generic keys to controller-specific
  parameters.
- `hilserl/clear_error` (`std_srvs/Trigger`): clear robot error state (can be a no-op).

Further notes:

- Quaternions are xyzw and rotations rpy throughout (the Franka-specific `euler_2_quat`
  convention of the original code base was standardized away).
- Gripper feedback and commands must be normalized at the adapter boundary using calibrated
  open/closed endpoints. See
  [how grasping works](robot_walkthrough.md#how-grasping-works-learned-gripper) for the details.
- Teleop input can be a `sensor_msgs/Joy` publisher or a direct device reader like
  `SpaceMouseTeleop`.
- Dual-arm setups are not yet addressed by the adapter design.

## 2. Verify the interface

With the robot (or your bridge node) running, check that the topics are alive and steady:

```bash
ros2 topic list | grep hilserl
ros2 topic hz /hilserl/tcp_pose
```

Then run the adapter smoke test. It subscribes to the state topics, prints observations, and
exercises the services:

```bash
ros2 run serl_ros2 serl_ros2_smoke --config <path-to-your>/adapter_config.yaml
```

Start from [serl_ros2/config/adapter_example.yaml](../serl_ros2/config/adapter_example.yaml) for
the adapter config (camera names, sync tolerance, command frame).

## 3. Test motion and teleop before any RL

Drive the robot by hand through the adapter, using the same teleop path that interventions will
use later:

```bash
python serl_ros2/serl_ros2_sim/test_teleop.py --config <path-to-your>/adapter_config.yaml
```

This uses a SpaceMouse by default. For keyboard or gamepad, start a Joy source and pass
`--teleop joy` (see [serl_ros2/README.md](../serl_ros2/README.md#teleop) for the device options):

```bash
ros2 run serl_ros2 keyboard_joy
python serl_ros2/serl_ros2_sim/test_teleop.py --config <path-to-your>/adapter_config.yaml --teleop joy
```

Move slowly and confirm that the robot tracks pose targets, that frames behave as expected
(base vs TCP), and that the gripper opens and closes. Fix problems here, not during training.

## 4. Create an experiment and train

Copy [examples/experiments/cube_demo_ros2](../examples/experiments/cube_demo_ros2) as a template,
adjust poses, workspace bounds, and cameras for your task, and register it in your config mapping
module (see [examples/README.md](../examples/README.md)). Then continue with the
[training walkthrough](robot_walkthrough.md).
