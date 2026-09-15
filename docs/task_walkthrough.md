# Setting up and Training a New Task

You have a robot with a ROS2 driver and a manipulation task in mind, and you want HIL-SERL to learn it.
This guide covers everything from the first teleop test to a trained, evaluated policy.

Prerequisite: your robot provides the adapter contract, verified with the smoke test
(see [robot_integration.md](robot_integration.md)). If you have no robot yet, do the simulated
[cube demo](../examples/experiments/cube_demo_ros2/README.md) first, it runs the identical
pipeline against the bundled [ursina](https://www.ursinaengine.org/) simulator, and every step
below maps one-to-one.

This guide generalizes the Franka-specific walkthrough of the original repository. For the
original version and additional Franka detail, see
[the upstream franka_walkthrough](https://github.com/rail-berkeley/hil-serl/blob/main/docs/franka_walkthrough.md).

## Overview of the pipeline

1. Validate teleop first: confirm a human can control the robot smoothly.
2. Design and configure the task (poses, bounds, cameras, reward source, experiment folder).
3. Collect classifier data and train a reward classifier (for classifier-based rewards).
4. Record a small set of human demonstrations.
5. Train with RLPD (actor + learner), giving occasional human interventions.
6. Evaluate checkpoints.

How to operate a training run well (demo counts, intervention protocol, metrics) is a topic of its
own: [training_recommendations.md](training_recommendations.md).

## 1. Validate teleop first

HIL-SERL moves the robot like a "carrot on a stick" (as in how to get a donkey to move :laughing:):
every step offsets the current TCP pose by a small delta and sends this as target.
The policy, the recorded demonstrations, and your interventions all act through this exact scheme.
That leads to the most important rule of adoption: **if you cannot perform the task smoothly by teleoperating through
this scheme, HIL-SERL will not be able to either.** Bad tracking, jitter, wrong action scales, or
an awkward input device all can be reasons for the control to not work smoothly, and that has to be tuned first.
So this is the first step after integrating the robot.

Bring up the robot (driver plus the HIL-SERL interface nodes), or the simulated stand-in:

```bash
python serl_ros2/serl_ros2_sim/ursina_sim.py --rate 50 --cameras front,wrist \
    --image-width 128 --image-height 128 --publish-images
```

Then drive the robot with the `teleop_check` tool. It runs the same control loop as training,
with your hands instead of a policy:

```bash
# SpaceMouse (recommended device)
ros2 run serl_ros2 teleop_check --config <your>/adapter_config.yaml --verbose

# Keyboard (run `ros2 run serl_ros2 keyboard_joy` in a second terminal) or gamepad
ros2 run serl_ros2 teleop_check --config <your>/adapter_config.yaml --teleop joy --verbose
```

What to do and look for:

- **Perform the task manually.** For example, for a peg insertion task: pick the peg, move it
  over the hole, insert it. Repeat until it feels controlled, not lucky. If you cannot do it,
  fix the setup (device, scales, controller tuning) before going further.
- **Read the `--verbose` diagnostics**: commanded and actual TCP velocity should be in the same
  ballpark (if the arm lags far behind, reduce the action scales or retune the controller), and a
  jittery loop rate points to scheduling problems (see the
  [performance notes](#ros2-timer-jitter-on-intel-hybrid-cpus)). The script header explains every
  printed metric.
- **Tune what later becomes training config**: the teleop device, base/tcp frames, and the
  `--rate` / `--xyz-scale` / `--rpy-scale` values you end up with translate directly to the
  experiment's `hz` and `ACTION_SCALE`. Once the task config YAML exists, re-run the check with
  `--task-config <task_config.yaml>` so it reads exactly the training parameters.

## 2. Design and configure the task

Each experiment/task is a folder holding its config, launch scripts, and generated data. Create yours
by copying [examples/experiments/cube_demo_ros2](../examples/experiments/cube_demo_ros2) and
registering it in your config mapping module
(see [examples/experiments/mappings.py](../examples/experiments/mappings.py), which the training
scripts find via `--config_mapping`, default `experiments.mappings`).

> [!NOTE]
> It isn't ideal to add this mapping in code, this is something that still isn't addressed in the fork, but planned for future work.

Design decisions, in the order they usually come up:

- **Reset pose and workspace bounds**: set `RESET_POSE` to a pose from which the task is
  reachable (e.g. hovering above the workspace with the object grasped), and the exploration bounding box
  `ABS_POSE_LIMIT_HIGH` / `ABS_POSE_LIMIT_LOW` tightly around the task volume. The bounding box
  is the main safety mechanism during exploration, keep it tight at first. With `RANDOM_RESET`
  enabled, resets are randomized around `RESET_POSE` (`RANDOM_XY_RANGE`, `RANDOM_RZ_RANGE`).
  Capture poses live while teleoperating:
  ```bash
  ros2 topic echo --once /hilserl/tcp_pose
  ```
- **Cameras**: list the camera streams in `CAMERAS` in the `EnvConfig` and the matching names in
  the experiment's `adapter_config.yaml` (the adapter subscribes to
  `/hilserl/camera/<name>/image`). A wrist camera plus one static view is a good default (e.g. for
  an insertion task, the wrist camera sees the hole up close). Policy and classifier keys are chosen in `TrainConfig.image_keys` and
  `TrainConfig.classifier_keys`. Image size does not matter, `RobotEnv` crops/resizes to 128x128,
  but publishing small images saves bandwidth. More important is `IMAGE_CROP` which holds per-camera crop:
  since images are scaled to 128x128, it matters that the images see the right thing.
  (Note: the upstream `REALSENSE_CAMERAS` field is not supported here, use `CAMERAS`).
- **Reward source**: pose-based (`TARGET_POSE` plus `REWARD_THRESHOLD`) works when success equals
  reaching a known pose - but that's usually just for testing and not what you may want.
  Typically, we want a trained image classifier which detects the success of the task in the image
  (e.g. for a peg insertion task, a classifier that recognizes a seated peg generalizes over
  holes, while `TARGET_POSE` would fix a single one).
  Section 3 covers the classifier path. Set `classifier_keys = None` to use the pose reward instead.
- **Control**: `hz` (control rate) and `ACTION_SCALE`, carried over from your teleop validation,
  plus compliance parameters if your robot supports the `set_compliance` service.
- **Gripper mode**: `setup_mode = "single-arm-fixed-gripper"` when the object is pre-grasped or
  the gripper stays fixed, `"single-arm-learned-gripper"` when the policy must learn to grasp
  (see [how grasping works](#how-grasping-works-learned-gripper)).

After editing, re-run `teleop_check --task-config <your_experiment.yaml>` once: it now uses
the exact training rate and scales, which is the final confirmation that the task is humanly
doable under training conditions.

## 3. Training a reward classifier

For classifier-based rewards, success is detected by a binary classifier trained on camera images.
(Alternatively, simple tasks can use a pose-based reward. The cube demo shows both variants.)

You can use any classifier (e.g. we tried a q-value based classifier in a separate project),
or use the one which this repo ships - which is the one which will be discussed here.

Collect labeled data while teleoperating:

```bash
PYTHONPATH=examples python -m serl_framework.train.record_success_fail \
    --exp_name <your_exp> --successes_needed 200
```

The script stores transitions in `.pkl` files under `classifier_data/`. Classifier training uses
only the image observations selected by `classifier_keys` in the config, plus binary labels.
Labels are per transition (per control step), not per episode: press `Space` to enter success
mode while the scene shows success, press `Esc` (or let the episode end) to commit and reset.
The script header and `--help` document all labeling controls.

> [!NOTE]
> **TIP**: To train a classifier robust against false positives, collect 2-3x more negative than
> positive transitions, covering all failure modes: wrong locations, halfway-in insertions, or
> holding the object right next to the goal.

Train the classifier (saved to `classifier_ckpt/` in the current directory):

```bash
PYTHONPATH=examples python -m serl_framework.train.train_reward_classifier --exp_name <your_exp>
```

## 4. Recording demonstrations

A small number of human demonstrations (typically 20) is crucial to accelerate training:

```bash
PYTHONPATH=examples python -m serl_framework.train.record_demos \
    --exp_name <your_exp> --successes_needed 20
```

Demos are accepted only when an episode ends with success (decided by the classifier threshold
for classifier-based tasks) and land in `demo_data/`. Progress is saved incrementally after every
accepted demo and interrupted sessions can be resumed. The script header and `--help` document the details.

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

Useful behavior beyond upstream (flags and details in the `train_rlpd.py` header and `--help`):

- **Resume**: the learner resumes from the latest checkpoint in `checkpoint_path`, reloads saved
  replay/demo buffers, and prunes stale CSV log rows.
- **Metrics**: CSV files under `checkpoint_path` by default (learner updates, timing, actor
  episode stats). Weights & Biases remains available (upstream behavior).
- **Pause**: the learner can periodically offer an interactive pause. Useful if you cannot keep
  intervening right now and do not want to restart the learner later.

### Running the learner remotely

The actor and learner communicate only over AgentLace (ZMQ), so the learner can run on any machine
with a GPU, for example a single cloud GPU instance. The actor machine must
be able to reach the learner's AgentLace ports (5588 for requests and data upload, 5589 for the
weight broadcast), typically over a VPN. Don't expose these ports publicly, the connection is
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
deadzone, the intervention overrides the policy action and the transition is tagged as intervention.
The learner mixes online replay with demo/intervention replay, so corrections
flow directly into training. Interventions are what make HIL-SERL work: demos teach how to do the
task, interventions teach how to recover when things go wrong or how to avoid useless actions
(e.g. moving away from insertion point).

When and how much to intervene matters a lot for training speed and final robustness. Follow
[training_recommendations.md](training_recommendations.md) for the intervention protocol
(frequent early, taper, targeted late), intervention style, and which metrics to watch.

## 6. Evaluating a policy

When you think the policy has converged well enough, it's time to evaluate it.

Add eval flags to the actor:

```bash
bash run_actor.sh --eval_checkpoint_step 50000 --eval_n_trajs 50
```

Training-time success is upward-biased (intervention-helped episodes also count as successes),
so it is important to evaluate candidate checkpoints separately with enough trials.
Do not assume a later checkpoint is better - see the eval protocol in [training_recommendations.md](training_recommendations.md#2-evaluate-separately-the-training-metrics-are-not-ground-truth).

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

- **Action command** (range `[-1, 1]`): close/open intent from the policy or the
  intervention wrapper. Teleop buttons map to strong values (close in `[-1.0, -0.9]`, open in
  `[0.9, 1.0]`, else `0.0`) so intent stays away from the decision boundary.
- **Measured state** (gripper pose in range `[0, 1]`): normalized feedback from the robot adapter
  with `0=closed`, `1=open`.

`RobotEnv` treats an action command `<= -0.5` as a close candidate and `>= 0.5` as an open
candidate, but only sends the service call if the measured state says it is still needed
(close only "open enough", specifically if gripper pose `> GRIPPER_OPEN_THRESHOLD`.
Start with `GRIPPER_OPEN_THRESHOLD = 0.85` and tune for your gripper.

At the adapter boundary, both directions are normalized `[0, 1]` with `0=closed`, `1=open`.
This keeps env logic and task configs robot-agnostic.
Hardware with raw joint units (where open can be numerically larger or smaller than closed)
should be converted in the adapter using calibrated open/closed endpoints. 

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
`lscpu` and avoid hyper-thread siblings):

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
