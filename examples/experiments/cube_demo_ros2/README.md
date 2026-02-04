## cube_demo_ros2

Minimal ROS2-backed HIL-SERL experiment for moving the TCP above a cube on the table.
This example uses `serl_framework.RobotEnv` plus the ROS2 `RobotAdapter` from `serl_ros2`.

This is not a robot example. It is a smoke test for the whole framework: a red ball stands in for
the robot TCP and is trained to move to the blue cube in the deliberately minimal
[ursina](https://www.ursinaengine.org/) simulator. Everything else (adapter, teleop, data
collection, classifier, actor/learner training) runs exactly as it would on hardware.

![Ursina cube demo](../../../docs/media/ursina_cube_demo.gif)

See [INSTALL.md](../../../INSTALL.md) for installation and environment setup.

## Typical setup steps [just FYI]: general approach for new experiments

1) Configure the `TARGET_POSE` and safety bounds (`ABS_POSE_LIMIT_*`) in `hil-serl/examples/experiments/cube_demo_ros2/config.py`.
2) Configure the camera list and ROS2 topic remapping in
   `hil-serl/examples/experiments/cube_demo_ros2/adapter_config.yaml`.
3) Ensure your ROS2 robot driver publishes the expected topics and services
   (`tcp_pose`, `tcp_twist`, `joint_states`, `gripper_pos`, `camera/<name>/image`, `clear_error`).

> [!NOTE]
> `adapter_config.yaml` is consumed by `RobotAdapter` in both modes:
> - standalone scripts (where you pass `--config .../adapter_config.yaml`), and
> - HIL-SERL experiment scripts (`record_demos.py`, `run_actor.sh`, etc.), because
>   `config.py` sets `TrainConfig.adapter_config_path` to this same file.

> [!NOTE]
> `TARGET_POSE` is only used by the built-in pose-based reward in `RobotEnv.compute_reward()`.
> If you use that reward, you must set `TARGET_POSE` and `REWARD_THRESHOLD` to a sensible
> success region. If you do not want a fixed pose target, use the reward classifier instead:
> keep `classifier_keys` enabled and train the classifier
> (the reward then comes from `MultiCameraBinaryRewardClassifierWrapper`, and TARGET_POSE is
> not consulted for reward at all).
> Set `TrainConfig.classifier_keys = None` to use TARGET_POSE reward.

In the next section, we'll be using the already configured scenario using a simple simulator. 

Notes:
- The reward classifier is optional; if you prefer the pose-based reward from `RobotEnv`,
  set `TrainConfig.classifier_keys = None` and skip the classifier steps above.
- If you are not using teleop, set `TrainConfig.use_teleop = False`.
- To use keyboard or gamepad instead of SpaceMouse, set `TrainConfig.teleop_device = "joy"`
  and run a separate ROS2 node that publishes `sensor_msgs/Joy` messages
  (see [serl_ros2 README - Teleop](../../../serl_ros2/README.md#teleop)).

## Start the simple Ursina simulation (ball -> cube)

This is the fastest way to test pose-based reward and then the classifier using the ROS2 `RobotAdapter` without any hardware.

1) Follow [the instructions in serl_ros2_sim](../../../serl_ros2/serl_ros2_sim/README.md) to install (and smoke-test)
  the simple Ursina environment **and do a performance check**. This will act like the real robot and provide the ROS2 to topics and services
  which the `RobotAdapter` will require.

2) Run the ursina sim **from the hil-serl root**:
```bash
python serl_ros2/serl_ros2_sim/ursina_sim.py --rate 50 --cameras front,wrist --image-width 128 --image-height 128 --publish-images
```
3) **Only if** you changed the cube pose in the ursina sim:
   In `hil-serl/examples/experiments/cube_demo_ros2/config.py`, set `TARGET_POSE`
   to the cube position used by the sim (the default cube is at `(3, 0, 0)` in
   `serl_ros2_sim/ursina_sim.py`) and expand `ABS_POSE_LIMIT_*` if needed.

> [!IMPORTANT]
> If the sim or teleop feels jittery on Intel hybrid CPUs, follow the
> [performance notes in the walkthrough](../../../docs/robot_walkthrough.md#ros2-timer-jitter-on-intel-hybrid-cpus)
> and pin the sim + node to P-cores (e.g., via `taskset`).

### (a) [only if **NOT** using TARGET_POSE] Train reward classifier 

In the first "free space" experiment using Ursina, we shall use TARGET_POSE, so you can skip it.
Suggestion: the first time you try it, stick with TARGET_POSE, then move on to using the classifier.

1) Collect success/failure data (press Space to mark a success), **from the `hil-serl` repo root**:
```bash
PYTHONPATH=examples python -m serl_framework.train.record_success_fail --exp_name cube_demo_ros2 --successes_needed 200
```
2) Train the classifier:
```bash
PYTHONPATH=examples python -m serl_framework.train.train_reward_classifier --exp_name cube_demo_ros2
```
This writes checkpoints under `hil-serl/examples/experiments/cube_demo_ros2/classifier_ckpt`.

### (b) Record demonstration data

> [!NOTE]
> If you haven't trained a classifier in (a), in order to avoid a crash in the below, also
> make sure to set `TrainConfig.classifier_keys = None` in the `config.py`

**Prepare:**
Make sure teleop input in `hil-serl/examples/experiments/cube_demo_ros2/config.py` is
`TrainConfig.use_teleop = True` (set to `False` for pure rollouts without human intervention).
Use `TrainConfig.teleop_device = "spacemouse"` or `"joy"` (for keyboard/gamepad/pose-to-joy via ROS2).
The latter is default.
For SpaceMouse teleop, set `TrainConfig.teleop_spacemouse_frame_id = "tcp"` (default)
or `"base"` depending on your intended command frame.


**Run the recording:**

Start the ursina sim first, **from the `hil-serl` root**:

```bash
python serl_ros2/serl_ros2_sim/ursina_sim.py --rate 50 --cameras front,wrist --image-width 128 --image-height 128 --publish-images
```

Then **from the `hil-serl` repo root**:

```bash
PYTHONPATH=examples python -m serl_framework.train.record_demos --exp_name cube_demo_ros2 --successes_needed 10
```
If you are using the keyboard, also launch
```bash
ros2 run serl_ros2 keyboard_joy
# or:
# python3 serl_ros2/serl_ros2/keyboard_joy.py
```
Or for the `Joy` message adapter (device and frame notes are in
[serl_ros2 README - Teleop](../../../serl_ros2/README.md#teleop))
```bash
ros2 run joy joy_node --ros-args -p device_id:=0 -r joy:=teleop_joy
```

> [!NOTE]
> `joy_node` publishes `Joy.header.frame_id = "joy"`. Because it cannot be changed,
> we cannot set it to the desired target frame (base, tcp...). So current temporary
> workaround in HIL-SERL treats `"joy"` as TCP-frame teleop.

This saves a demo dataset under `hil-serl/demo_data/cube_demo_ros2<timestamp>.pkl`.


### (c) Run training

Before starting training, review these settings in `config.py`:

* `max_steps`: Maximum training steps. Simple tasks often converge in 20-50k steps.
* `checkpoint_period`: Save checkpoint every N steps. Set to 0 to disable.

Checkpoints are saved to the `--checkpoint_path` directory (default: `first_run/`).
You can safely Ctrl+C the learner/actor at any time - training can be resumed from
the last checkpoint, or you can use the saved model for evaluation.

From the experiment folder, run `run_learner.sh` with the path to your recorded demonstration data:
```bash
cd examples/experiments/cube_demo_ros2
bash run_learner.sh ../../../demo_data/<your_demo>.pkl
# Checkpoints saved to ./first_run/ every 5000 steps
```

Launch the simulator:

```bash
python serl_ros2/serl_ros2_sim/ursina_sim.py --rate 50 --cameras front,wrist --image-width 128 --image-height 128 --publish-images
```

And in another terminal, run the actor:

```bash
cd examples/experiments/cube_demo_ros2
bash run_actor.sh --ip localhost
# Reads checkpoints from ./first_run/
```
> [!NOTE]
> You can run your learner (see above) on another computer, but it has to bee in the same network (e.g. VPN).
> If you are doing that, use the IP of your host instead of `localhost`.

If you are using the keyboard, also launch

```bash
ros2 run serl_ros2 keyboard_joy
```
Or for the `Joy` message adapter (device and frame notes are in
[serl_ros2 README - Teleop](../../../serl_ros2/README.md#teleop))
```bash
ros2 run joy joy_node --ros-args -p device_id:=0 -r joy:=teleop_joy
```

> [!NOTE]
> `--ip` is the learner's host address that the actor connects to over AgentLace (ZMQ).
> Use `localhost` if both processes are on the same machine; otherwise pass the learner
> machine's reachable IP.

> `--checkpoint_path` is the directory where the learner saves checkpoints/buffers and the
> actor loads them. The scripts here set `--checkpoint_path=first_run`, which is relative
> to the current working directory. Run both scripts from this folder so they point to the
> same location, or edit the scripts to use an absolute path if you launch from elsewhere.

### (d) Execute a trained policy

Use eval flags to run a trained checkpoint without learning:
```bash
cd examples/experiments/cube_demo_ros2
bash run_actor.sh --eval_checkpoint_step 50000 --eval_n_trajs 5
```

## Teleop

Teleop documentation is centralized in:
- [serl_ros2 README - Teleop](../../../serl_ros2/README.md#teleop)

Use that section for:
- keyboard/gamepad setup
- `pose_to_joy` and RViz-marker flow
- multi-device mux (`joy_mux`)
- frame-id behavior (`base` vs `tcp`, including `joy_node` caveats)
