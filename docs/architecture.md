# Architecture

How this fork connects the HIL-SERL learning core to robots. The original project drove a Franka
arm through a ROS1 + Flask/HTTP bridge. This fork replaces that bridge with a robot-agnostic
adapter layer and a ROS2 implementation of it. The learning core (`serl_launcher`) is untouched by
this design. For the complete list of changes vs upstream, see [CHANGES_FORK.md](../CHANGES_FORK.md).

## System overview

```
┌─────────────────────────────────────────────────────────────┐
│                              HIL-SERL                       │
│  ┌──────────────┐                       ┌──────────────┐    │
│  │ACTOR PROCESS │ ── env.step/reset ───►│  RobotEnv    │    │
│  │  (Robot PC)  │                       │  (gym)       │    │
│  └──────┬───────┘                       └──────┬───────┘    │
│         │ AgentLace (ZMQ)                      │            │
│         │ transitions / weight updates         │            │
│  ┌──────┴───────┐                              │            │
│  │  LEARNER     │                              │            │
│  │  (GPU PC)    │                              │            │
│  └──────────────┘                              │            │
└────────────────────────────────────────────────┼────────────┘
                                             in-process API
                                                 │
┌────────────────────┐   ROS2 topics   ┌─────────┴───────────┐   ROS2 topics   ┌────────────────────┐
│ Cameras / Sensors  │ ───────────────►│   Robot Adapter     │ ◄────────────── │ Teleop Device (Joy)│
└────────────────────┘                 │     (ROS2 node)     │                 └────────────────────┘
                                       └─────────┬───────────┘
                                                 ▼
                                       ┌──────────────────────┐
                                       │ Robot Driver + HW    │
                                       └──────────────────────┘
```

Three layers, with one-way dependencies from top to bottom:

1. **Learning core** (`serl_launcher`): SAC/RLPD agents, replay buffers, reward classifier,
   actor/learner networking. Knows nothing about robots. Almost unchanged from upstream.
2. **Robot-agnostic layer** (`serl_framework`): the Gymnasium env (`RobotEnv`), env wrappers, the
   `RobotAdapter` and `TeleopAdapter` interfaces, and the training entry points
   (`serl_framework.train`). ROS-free, it talks to the robot only through the adapter interface.
3. **ROS2 layer** (`serl_ros2`, `serl_msgs`): the ROS2 implementation of `RobotAdapter`, teleop
   nodes, and the ursina simulator. The only code that imports `rclpy`.

## The RobotAdapter

`RobotAdapter` is a plain Python interface between the env and whatever middleware talks to the
robot. Its design points:

- **In-process API.** Env and adapter run in the same process. The env calls
  `get_observation()` / `send_pose_command()` / `call_service()` directly, no IPC between them.
- **Adapter owns the middleware.** The ROS2 implementation spins its executor internally
  (`create()` / `start()` / `stop()`), so user code and the env never call `rclpy.spin*()`.
  Service calls are bridged to plain Python futures and are safe from any thread.
- **Streaming control.** Each control step publishes an absolute TCP pose target
  (`/hilserl/command_pose`), and the robot's controller (impedance, admittance, servo) tracks it.
  Actions are small deltas applied to that target.
- **State aggregation.** The adapter subscribes to robot state and camera streams, time-aligns
  them (best-effort `ApproximateTimeSynchronizer`), and serves the latest consistent snapshot.
  `get_observation()` never blocks waiting for perfect sync. Staleness is bounded, and stale
  state ends the episode as `truncated`.
- **Teleop through the same path.** A device-agnostic `TeleopAdapter` (SpaceMouse direct, or any
  `sensor_msgs/Joy` publisher) is attached to the RobotAdapter. Env wrappers read it to inject
  human interventions, and the same path drives demo recording and the `teleop_check` tool.

The exact topics, services, and observation fields a robot integration must provide are specified
in [robot_integration.md](robot_integration.md).

## Distributed actor and learner

Training runs as two asynchronous processes connected by
[agentlace](https://github.com/youliangtan/agentlace) (ZMQ):

- The **actor** rolls out the policy on the robot machine, injects teleop interventions, and
  uploads transitions and episode stats (request channel, port 5588).
- The **learner** updates the policy on a GPU machine and broadcasts fresh weights
  (broadcast channel, port 5589). The fork's actor also pulls weights once at startup and can
  save learner-step-aligned local checkpoints.

Because the two processes share nothing but this connection, the learner can run on any GPU
machine, including a cloud instance. Run the connection over a VPN or another private network:
ZMQ is unauthenticated and the ports must never be exposed publicly. The practical steps and an
interaction diagram are in the
[remote learner section of the walkthrough](task_walkthrough.md#running-the-learner-remotely).

## Where things are decided

- Task-specific configuration (poses, bounds, cameras, reward source) lives in per-experiment
  `config.py` + `adapter_config.yaml` under `examples/experiments/`, resolved by the training
  scripts through a config mapping module (`--config_mapping`).
- Reward comes either from a pose target (`TARGET_POSE`) or from a trained image classifier.
  Gripper control is either fixed or learned via a separate grasp critic
  (see [how grasping works](task_walkthrough.md#how-grasping-works-learned-gripper)).
- The original ROS1 + Flask/HTTP architecture is documented in the
  [original repository](https://github.com/rail-berkeley/hil-serl).
