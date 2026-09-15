# What changed in this fork

This describes the `ros2` branch compared with upstream `main`.

The main change is how HIL-SERL connects to a robot. The original project provided
a Franka setup using ROS1 and an HTTP server. This fork provides a robot-independent
environment with a ROS2 adapter, plus tools for collecting data and running training.
The HIL-SERL learning method remains the original authors' work.

## Connecting a robot

- **A separate robot interface.** The new `serl_framework` package defines
  `RobotAdapter` and `TeleopAdapter` interfaces using ordinary Python values and
  arrays. Other integrations can implement these interfaces without ROS2.
- **A ROS2 implementation.** `serl_ros2` implements the robot interface for ros2.
  It receives TCP pose and velocity, joint states, gripper state, camera images, and optional force/torque feedback.
  It publishes Cartesian pose targets and calls robot services. It handles ROS2
  spinning internally so the training loop can stay a normal Python loop.
- **Small service definitions.** `serl_msgs` supplies gripper, robot-reset, and
  compliance services. Optional world-reset and teleop-reset services support
  simulation and reset coordination. A robot-side bridge maps these interfaces
  to its driver and controller.
- **A mock adapter.** Environment behavior can be checked without ROS2 or hardware.

## The environment and motion tools

- **A generic `RobotEnv`.** The upstream Franka environment's capabilities (action scaling,
  pose bounds, camera cropping/resizing, optional image display/video recording, episode
  handling, a pose-based success reward) are carried over into a robot-agnostic environment
  that builds Gymnasium observations from the adapter's cached state and images. New: the
  effective workspace can be narrowed around the post-reset pose, and stale robot state ends
  the episode instead of being silently ignored.
- **Reusable wrappers.** Relative-frame actions and observations, quaternion
  conversion, teleop interventions, fixed-gripper actions, and classifier rewards
  live in `serl_framework` instead of the Franka package.
- **More flexible resets.** Randomized resets, joint resets, and subclass reset overrides
  existed upstream. New: a fixed `RESET_POSE` is no longer required when a task supplies its
  own reset logic, reset moves have a configurable timeout, and optional world and teleop
  reset hooks keep the surrounding setup in sync.
- **Shared motion interpolation.** Upstream's open-loop interpolated moves become closed-loop,
  with configurable position and rotation tolerances, and are reusable outside the environment.
  Free-space/reset movement can use different action scales from the task itself.
- **Adapter-aware timing.** Can use the adapter's clock,
  including simulation time. State-age checks help detect stale robot feedback.
- **Gripper and compliance support.** The upstream gripper toggle threshold and delay are
  kept, with the threshold now a config field (`GRIPPER_OPEN_THRESHOLD`) instead of a
  hard-coded value, and gripper feedback is normalized to [0,1] at the adapter boundary.

## Human control

- **Several input options.** Direct SpaceMouse input and ROS2 Joy input support
  demonstrations and interventions. The ROS2 package includes keyboard control
  and can consume gamepad messages.
- **Explicit control frames.** Teleop input identifies whether movement is in the
  base or TCP frame. Keyboard and SpaceMouse input can switch frames and support
  configurable axis mappings.
- **Pose-target teleop.** `pose_to_joy` turns target poses, such as those from an
  interactive marker, into bounded movement commands. It activates when the target
  changes and stops when the target is reached.

## Training and checkpointing

The runnable training scripts now live in the installed `serl_framework.train`
package. They can be launched with `python -m serl_framework.train.<script>`.
Most of the additions below came from operating the separate AIC (AI for industry challenge)
integration.

- **Experiments can live in their own package.** `--config_mapping` selects the
  module that provides task configurations. Mapping entries can be loaded
  lazily. Shared helpers apply class defaults, optional task config YAML, and CLI
  overrides in a consistent order for the main training and recording scripts.
- **CSV logs by default.** Learner updates, timing, and actor episode statistics
  go into separate files. New metric columns can be added as they appear. W&B
  remains available as an alternative.
- **Sturdier resume.** Checkpoint restore and replay-chunk save/reload existed upstream
  (enabled via `buffer_period`). This fork moves the buffer dumping from the actor to the
  learner, so persisted chunks survive actor restarts, and prunes CSV rows newer than the
  resumed checkpoint.
- **More reliable actor restarts.** An actor resets its learner-side cursors
  on startup, requests the current weights, and waits for the first policy before
  starting rollout.
- **Local policy backups on the actor.** Optional actor-side checkpoints use the
  learner step number carried with weight updates, making them easier to match to the
  training run.
- **Useful run diagnostics.** Logs include replay/demo sizes, sampled rewards,
  weight updates, and episode-upload timing.
- **Interactive learner pauses.** On an interactive terminal, the learner can
  periodically offer a short window to pause and resume training.
  Useful if you need just a break from training/intervening and don't want to later restart the whole learner.

These changes support the existing actor/learner arrangement: the actor runs with
the robot, and the learner can run on a separate GPU machine over AgentLace.
They do not introduce a new reinforcement-learning algorithm.

## Collecting demonstrations and training a reward classifier

- **Incremental demonstration saving.** Successful demonstrations are saved after
  each accepted episode. A recording can resume from an existing pickle, use a
  readable run name, and override episode length.
- **Configuration recording.** When a task config YAML is available, the demo
  recorder copies it beside the dataset, for later reference.
- **More deliberate success/failure labeling.** A key press enters success mode,
  another cancels it. Positive samples are rate-limited, and recent frames before a
  success label can be discarded from the negative set, and other minor improvements.
  This makes training the classifier easier and more robust.
- **Flexible classifier datasets.** Classifier training accepts image-key
  overrides and multiple success/failure globs for training files. It deduplicates selected paths
  and rejects a file selected for both classes.

## Installation, examples, and documentation

- Added Python 3.12 support and documented the tested JAX 0.4.36 stack in
  [INSTALL.md](INSTALL.md) and `requirements.txt`.
- Kept changes to `serl_launcher` small: optional/lazy TensorFlow imports in the
  affected utilities, configurable AgentLace request types, package initialization
  fixes, and an AgentLace dependency pointing to a pinned fork commit that removes
  an obsolete dependency. The full training requirements still include TensorFlow.
- Added a small Ursina simulator and a cube-reaching experiment. A ball represents
  the TCP, allowing teleop, data collection, classifier training, and actor/learner
  wiring to be explored without a physical robot.
- Added framework and ROS2 tests, smoke tools, and CI for Python 3.10/3.12,
  ROS2 Jazzy, and training-stack imports/JAX computation.
- Reworked the documentation around robot integration, a complete training
  walkthrough, remote learners, and practical training recommendations. The
  recommendations distinguish upstream guidance from observations made during
  our insertion-policy training runs during the AI for industry challenge.

## What was removed or left out

- Removed `serl_robot_infra`, the Franka/ROS1 controllers and HTTP robot servers,
  and the Franka-specific examples and walkthrough material.
- Removed the old runnable scripts from `examples/` after moving the supported
  workflows into `serl_framework.train`. BC and HG-DAgger baseline scripts were
  not ported. Their upstream versions remain available in the original repository.
