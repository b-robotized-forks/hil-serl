# Training Walkthrough

This guide walks through the full HIL-SERL training pipeline with the ROS2 architecture of this
fork, for any robot that provides the adapter contract (see
[robot_integration.md](robot_integration.md)).
It generalizes the Franka-specific walkthrough of the original repository. For the original
version and additional Franka detail, see
[the upstream franka_walkthrough](https://github.com/rail-berkeley/hil-serl/blob/main/docs/franka_walkthrough.md).

Where concrete commands help, this guide quotes the simulated cube demo
([examples/experiments/cube_demo_ros2](../examples/experiments/cube_demo_ros2)), which runs the
identical pipeline against the bundled [ursina](https://www.ursinaengine.org/) simulator.
If you have no robot yet, do the cube demo
first. Every step below maps one-to-one.

## Overview of the pipeline

1. Bring up the robot (driver plus HIL-SERL topic/service interface).
2. Create and edit an experiment configuration.
3. Collect classifier data and train a reward classifier (for classifier-based rewards).
4. Record a small set of human demonstrations.
5. Train with RLPD (actor + learner), giving occasional human interventions.
6. Evaluate checkpoints.

## 1. Robot setup

Launch your robot driver and whatever node(s) publish the HIL-SERL interface topics
(`/hilserl/tcp_pose`, `/hilserl/command_pose`, cameras, ...) and provide the services
(`set_gripper`, `reset_robot`, ...). How to set this up for a new robot, including the full
topic and service contract, is documented in [robot_integration.md](robot_integration.md).

For the simulated demo this is a single command:

```bash
python serl_ros2/serl_ros2_sim/ursina_sim.py --rate 50 --cameras front,wrist \
    --image-width 128 --image-height 128 --publish-images
```

Verify the interface is alive before continuing:

```bash
ros2 topic hz /hilserl/tcp_pose
ros2 topic list | grep hilserl
```

## 2. Editing the training configuration

Each experiment is a folder holding its config, launch scripts, and generated data.
Create yours by copying [examples/experiments/cube_demo_ros2](../examples/experiments/cube_demo_ros2)
and registering it in your config mapping module
(see [examples/experiments/mappings.py](../examples/experiments/mappings.py), which the training
scripts find via `--config_mapping`, default `experiments.mappings`).

In your experiment's `config.py`:

- **Cameras**: list the camera streams in `CAMERAS` in the `EnvConfig` and set the matching camera
  names in the experiment's `adapter_config.yaml` (the adapter subscribes to
  `/hilserl/camera/<name>/image`). The camera keys used for policy training and for the reward
  classifier are listed in `TrainConfig.image_keys` and `TrainConfig.classifier_keys`.
  Note: the upstream `REALSENSE_CAMERAS` field is not supported in `serl_framework`, use `CAMERAS`.
  Image size does not matter, `RobotEnv` crops/resizes to 128x128, but publishing small images
  saves bandwidth. `IMAGE_CROP` can hold per-camera crop functions.
- **Poses and workspace bounds**: set `TARGET_POSE` (arm pose at task success), `RESET_POSE`
  (pose to reset to), and the exploration bounding box `ABS_POSE_LIMIT_HIGH` / `ABS_POSE_LIMIT_LOW`.
  With `RANDOM_RESET` enabled, resets are randomized around `RESET_POSE`
  (`RANDOM_XY_RANGE`, `RANDOM_RZ_RANGE`). To capture the current arm pose, read it live:
  ```bash
  ros2 topic echo --once /hilserl/tcp_pose
  ```
- **Control**: `hz` (control rate), `ACTION_SCALE`, and compliance parameters if your robot
  supports the `set_compliance` service.

> **TIP**: Keep the bounding box tight around the task at first. It is the main safety mechanism
> during exploration.

## 3. Training a reward classifier

For classifier-based rewards, success is detected by a binary classifier trained on camera images.
(Alternatively, simple tasks can use a pose-based reward. The cube demo shows both variants.)

Collect labeled data while teleoperating:

```bash
PYTHONPATH=examples python -m serl_framework.train.record_success_fail \
    --exp_name <your_exp> --successes_needed 200
```

The script stores transitions in `.pkl` files under `classifier_data/`. Classifier training uses
only the image observations selected by `classifier_keys` plus binary labels. Labels are per
transition (per control step), not per episode. Labeling controls:

- Press `Space` once to enter success mode. Press `Space` again to cancel it and discard the
  pending success samples.
- The last `--discard_before_success_seconds` immediately before the `Space` press are dropped
  (not labeled failure), so near-success frames stay out of the failure set.
- In success mode, at most `--success_samples_per_second` transitions per second are labeled
  success. All other transitions are dropped.
- In failure mode, transitions are labeled failure (`--negative_sample_stride` records only every
  Nth one).
- Success samples are staged and committed only on `Esc` reset or episode end.
- `--failures_needed` sets an explicit minimum failure count, and `--episode_length` overrides
  the episode length for the run.

Practical implication: you do not need to hold a key every frame. Enter success mode during the
successful visual state window, cancel it if it was a mistake, and reset with `Esc` (or let the
episode end) to commit.

> **TIP**: To train a classifier robust against false positives, collect 2-3x more negative than
> positive transitions, covering all failure modes: wrong locations, halfway-in insertions, or
> holding the object right next to the goal.

Train the classifier (saved to `classifier_ckpt/` in the current directory):

```bash
PYTHONPATH=examples python -m serl_framework.train.train_reward_classifier --exp_name <your_exp>
```

Verify it live while teleoperating, before recording demos:

```bash
PYTHONPATH=examples python -m serl_framework.train.stream_classifier_prob \
    --exp_name <your_exp> --threshold 0.75
```

## 4. Recording demonstrations

A small number of human demonstrations (typically 10-30) is crucial to accelerate training:

```bash
PYTHONPATH=examples python -m serl_framework.train.record_demos \
    --exp_name <your_exp> --successes_needed 20
```

Demos are accepted only when an episode ends with success (`info["succeed"]`, decided by the
classifier threshold for classifier-based tasks). Progress is saved incrementally after every
accepted demo, `--resume <pkl>` continues a previous session, and the experiment YAML in effect is
snapshotted next to the output pkl. Results land in `demo_data/`.

> **TIP**: If the classifier produces false positives (episode ends with reward without real
> success) or false negatives, collect additional classifier data targeting those failure modes
> rather than just adjusting the threshold.

Why ~20 demos are enough, and how they are used during training, is covered in
[training_recommendations.md](training_recommendations.md).

## 5. Policy training

Training runs as two asynchronous processes: an actor rolling out the policy on the robot and
streaming transitions, and a learner updating the policy and broadcasting weights. Edit
`checkpoint_path` in your experiment's `run_actor.sh` / `run_learner.sh`, then:

```bash
# Terminal 1 (GPU machine)
bash run_learner.sh /path/to/demo.pkl

# Terminal 2 (robot machine)
bash run_actor.sh --ip <learner-ip>
```

Useful behavior beyond upstream:

- **Resume**: if `checkpoint_path` already holds checkpoints, the learner resumes from the latest
  one, reloads saved replay/demo buffers (`buffer/`, `demo_buffer/`, written every
  `buffer_period` steps), and prunes stale CSV log rows.
- **Metrics**: with the default `--logger=csv`, the learner writes `learner_update_metrics.csv`,
  `learner_timer_metrics.csv`, and `actor_stats.csv` under `checkpoint_path`.
  Use `--logger=wandb` for Weights & Biases.
- **Actor-side checkpoints**: `--save_actor_checkpoint` saves policy checkpoints on the actor,
  numbered by learner step.
- **Pause**: `--pause_prompt_period N` lets the learner offer an interactive pause every N steps.

### Running the learner remotely

The actor and learner communicate only over AgentLace (ZMQ), so the learner can run on any machine
with a GPU, for example a single cloud GPU instance (we used an NVIDIA L4). The actor machine must
be able to reach the learner's AgentLace ports (5588 for requests and data upload, 5589 for the
weight broadcast), typically over a VPN. Never expose these ports publicly, the connection is
unauthenticated.

```
      ROBOT MACHINE                                GPU MACHINE (local or cloud)
┌────────────────────────┐                    ┌─────────────────────────────────────┐
│ ACTOR                  │                    │ LEARNER                             │
│                        │   transitions +    │                                     │
│  env.step() rollouts   │   episode stats    │  ┌──────────────┐  ┌─────────────┐  │
│  with teleop           │ ─────────────────► │  │ replay buffer│  │ demo+intvn  │  │
│  interventions         │   ZMQ, port 5588   │  │ (every step) │  │ buffer      │  │
│                        │                    │  └───────┬──────┘  └──────┬──────┘  │
│  policy inference      │                    │      50% │        50%     │         │
│  on latest weights     │   weight broadcast │          ▼                ▼         │
│                        │ ◄───────────────── │        SAC / RLPD updates           │
│                        │   ZMQ, port 5589   │                                     │
│                        │                    │  checkpoints, buffer dumps, CSVs    │
└────────────────────────┘                    └─────────────────────────────────────┘
```

Every environment step goes to the replay buffer, intervention steps additionally to the
demo/intervention buffer, and each learner batch samples 50/50 from the two. The actor refreshes
its weights from the periodic broadcast (and pulls them once at startup).

1. Copy the demo data to the learner machine:
   ```bash
   scp demo_data/<your_demo>.pkl <learner-host>:demo_data/
   ```
2. Start the learner there as usual: `bash run_learner.sh demo_data/<your_demo>.pkl`
3. Start the actor on the robot machine with the learner's address:
   `bash run_actor.sh --ip <learner-ip>`
4. If the simulator occupies the actor machine's GPU, force the actor to CPU with
   `JAX_PLATFORMS=cpu`. The actor only runs inference, so this costs little.

Checkpoints, resume buffers, and the metrics CSVs are written on the learner machine under
`checkpoint_path`. Fetch them back with scp/rsync when you want to evaluate locally, or use
`--save_actor_checkpoint` on the actor to keep local policy checkpoints numbered by learner step.

### Interventions during training

You intervene the same way as when recording demos: as soon as the teleop device leaves its
deadzone, the intervention overrides the policy action and the transition is tagged with
`intervene_action`. The learner mixes online replay with demo/intervention replay, so corrections
flow directly into training. Interventions are what make HIL-SERL work: demos teach how to do the
task, interventions teach how to recover when things go wrong.

When and how much to intervene matters a lot for training speed and final robustness. Follow
[training_recommendations.md](training_recommendations.md) for the intervention protocol
(frequent early, taper, targeted late), intervention style, and which metrics to watch.

### Episode end and reset semantics

- `done` triggers on max episode length, the success reward condition, or a manual terminate flag.
- `truncated` is used for external interruption/timeouts (e.g. stale state in `RobotEnv.step()`).
- The recording scripts and the actor reset on `done or truncated`.
- `Esc` globally ends the current episode early (a `RobotEnv` keyboard listener sets the
  terminate flag). This applies to recording and training rollouts alike.

## 6. Evaluating a policy

Add eval flags to the actor:

```bash
bash run_actor.sh --eval_checkpoint_step 50000 --eval_n_trajs 50
```

`--eval_argmax` switches to deterministic action selection during eval.

Training-time success is upward-biased (interventions count as successes), so evaluate candidate
checkpoints separately with enough trials, and do not assume a later checkpoint is better. See
the eval protocol in [training_recommendations.md](training_recommendations.md#2-evaluate-separately-the-training-metrics-are-not-ground-truth).

## How grasping works (learned gripper)

In learned-gripper setups (`setup_mode = single-arm-learned-gripper` or
`dual-arm-learned-gripper`), HIL-SERL uses a hybrid controller: arm motion is learned by the
regular SAC actor (continuous EE action), while the gripper action is selected by a separate grasp
critic (`grasp_critic`) using DQN-style action selection over 3 discrete bins
(`close`, `no-op`, `open`), mapped to `{-1, 0, 1}` in the final action output. Optionally, tasks
can add a `grasp_penalty` term (for example penalizing frequent toggle attempts). When present it
is logged into transitions and added to the grasp critic reward target, shaping gripper behavior
without changing the arm reward.

Two separate signals exist, in intentionally different domains:

- **Action command** (`action[6]`, range `[-1, 1]`): close/open intent from the policy or the
  intervention wrapper. Teleop buttons map to strong values (close in `[-1.0, -0.9]`, open in
  `[0.9, 1.0]`, else `0.0`) so intent stays away from the decision boundary.
- **Measured state** (`gripper_pose`, range `[0, 1]`): normalized feedback from the robot adapter
  with `0=closed`, `1=open`.

`RobotEnv` treats `action[6] <= -0.5` as a close candidate and `action[6] >= 0.5` as an open
candidate, but only sends the service call if the measured state says it is still needed
(close only if `gripper_pose > GRIPPER_OPEN_THRESHOLD`, open only if below). This
"open enough / closed enough" gate acts as hysteresis against noisy gripper feedback, and a
time-based debounce (`GRIPPER_SLEEP`) prevents rapid toggling. Start with
`GRIPPER_OPEN_THRESHOLD = 0.85` and tune if your gripper's normalized signal clusters differently
near the physical endpoints.

At the adapter boundary, both directions are normalized `[0, 1]` with `0=closed`, `1=open`
(state on `/hilserl/gripper_pos`, commands via `set_gripper` `MODE_POSITION`). Hardware with raw
joint units (where open can be numerically larger or smaller than closed) should be converted in
the adapter using calibrated open/closed endpoints, inferring the direction from the calibration
and clamping to `[0, 1]`. This keeps env logic and task configs robot-agnostic.

## Tips and troubleshooting

### Reducing GPU memory usage

If you hit JAX/XLA out-of-memory errors: reduce `batch_size` (e.g. 256 to 64), use a single camera
(keep `image_keys`, `CAMERAS`, and the adapter config consistent, otherwise the env raises
"Missing images for cameras"), lower the image resolution at the source, switch
`encoder_type = "resnet"` instead of `"resnet-pretrained"`, or lower
`XLA_PYTHON_CLIENT_MEM_FRACTION` per process:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.2 bash run_learner.sh /path/to/demo.pkl
XLA_PYTHON_CLIENT_MEM_FRACTION=.05 bash run_actor.sh --ip localhost
```

### Noisy force/torque feedback in simulation

Simulated F/T feedback (e.g. Gazebo) can be noisy. Either disable wrench observations
(`subscribe_wrench: false` in the adapter config, `tcp_force`/`tcp_torque` become zeros) or filter
the signal in the adapter before it reaches the observations.

### Monitoring command/feedback lag

If the actor appears unstable, check whether command publication and robot feedback keep up:

```bash
ros2 topic hz /hilserl/command_pose
ros2 topic hz /hilserl/tcp_pose
ros2 topic bw /hilserl/camera/<name>/image
```

The command rate should match the experiment's `hz`. Long gaps in `/hilserl/command_pose` point to
actor-side compute stalls, growing feedback age means the policy acts on stale state, and high
camera bandwidth increases CPU pressure.

### ROS2 timer jitter on Intel hybrid CPUs

On Intel hybrid CPUs (P-cores plus E-cores, e.g. i7-12700K and newer), Linux may schedule Python
ROS2 nodes onto E-cores. This causes severe timer jitter and unstable publish rates even at
30-50 Hz. It is an OS scheduling effect, not a ROS2 bug. Check with
`ros2 topic hz /hilserl/tcp_pose`: large `max` gaps between messages indicate the problem.
The fix is to pin each Python ROS2 node to a dedicated P-core (check the core layout with
`lscpu`, avoid hyper-thread siblings):

```bash
taskset -c 2 python serl_ros2/serl_ros2_sim/ursina_sim.py --rate 50
```

Optionally switch the CPU governor to `performance` (lasts until reboot):

```bash
sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done'
```

### Should `q`/`dq` be part of the observations?

Including joint positions/velocities in `proprio_keys` can help the policy infer kinematic context
and recover from configurations where Cartesian state is ambiguous, but it increases input
dimensionality, adds redundancy with `tcp_pose`/`tcp_vel`, and makes policies more
embodiment-specific. Practical guidance: start with task-minimal proprioception (`tcp_pose`,
`tcp_vel`, `gripper_pose`), and add `q`/`dq` only if you observe behavior that depends on the
joint configuration (for example joint-jump rejections around equivalent Cartesian targets).
