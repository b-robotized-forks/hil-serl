# HIL-SERL for ROS2

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://hil-serl.github.io/)
[![Tests](https://github.com/b-robotized-forks/hil-serl/actions/workflows/unit-tests.yml/badge.svg?branch=ros2)](https://github.com/b-robotized-forks/hil-serl/actions/workflows/unit-tests.yml)

A fork of [rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl), a framework for
training precise robotic manipulation policies with human-in-the-loop reinforcement learning
([paper](https://arxiv.org/abs/2410.21845), [project page](https://hil-serl.github.io/)).

This fork adds ROS2 support and a robot-agnostic architecture that also works in simulation.
The learning core (`serl_launcher`) is almost unchanged. The ROS1/Franka-specific parts were
removed and remain available in the original repository.
See [CHANGES_FORK.md](CHANGES_FORK.md) for the full list of changes.

## What this fork adds

- A robot-agnostic layer (`serl_framework`): ROS-free `RobotAdapter` and `TeleopAdapter`
  interfaces plus a generic `RobotEnv`.
- A ROS2 implementation of that interface (`serl_ros2`, `serl_msgs`). A new robot needs a handful
  of topics and a few services.
- A simulation-first workflow: the bundled [ursina](https://www.ursinaengine.org/) simulator runs
  the full pipeline without hardware.
- Improved training scripts (`serl_framework.train`): resumable training, CSV metrics,
  actor-side checkpoints.
- Improved data collection: incremental demo saving with resume, config snapshots, and better
  success/failure labeling.
- Teleop for SpaceMouse, keyboard, and gamepad, including a pose-to-Joy bridge
  (see [serl_ros2/README.md](serl_ros2/README.md)).
- Python 3.12 support, a unit test suite, and CI.

This code was used in the [AI for Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge) (team b-robotized),
where policies performed cable insertion tasks.

![Insertion](docs/media/aic_insertion.gif)

[Higher-resolution screencast in [docs/media/aic_insertion.mp4](docs/media/aic_insertion.mp4)]

> [!NOTE]
> This fork does **not** include the full running competition example, only the HIL-SERL base it
> was built on. In terms of this repository, the competition setup is basically an `experiment`
> (like the ones in `examples/experiments/`), but it also depended on the challenge framework as a
> whole (the `aic` toolkit packages and its pixi environment) and was developed separately.
> It still needs some grooming before it can be published, and it would have to be published as a
> fork of the `aic` repository itself rather than as part of this one.

## Prerequisites

Software:

- Linux (tested on Ubuntu 24.04).
- ROS 2, tested with **Jazzy** and **Kilted**. Only `serl_ros2` and `serl_msgs` touch ROS.
- Python 3.12 (3.10 is also covered by CI) with **JAX 0.4.36** (see the note below).

Hardware:

- **Learner**: an NVIDIA GPU (CUDA 12). One mid-range GPU is enough, we trained on a single
  NVIDIA L4 in the cloud
  ([running the learner remotely](docs/task_walkthrough.md#running-the-learner-remotely)).
- **Actor**: no GPU needed, inference runs on CPU. On Intel hybrid CPUs see the
  [performance notes](docs/task_walkthrough.md#ros2-timer-jitter-on-intel-hybrid-cpus).
- **Teleop device**: a SpaceMouse is strongly recommended, keyboard and gamepad also work.
- **Robot**: any robot with a ROS 2 driver that can track Cartesian TCP pose targets (e.g. an
  impedance or admittance controller), plus one or more cameras
  (see [docs/robot_integration.md](docs/robot_integration.md)). The simulated demo needs no
  hardware at all.

## Installation

See [INSTALL.md](INSTALL.md).

> [!IMPORTANT]
> This fork requires **JAX 0.4.36**. The upstream 0.4.35 breaks on Python 3.12 / Ubuntu 24.04,
> and newer JAX versions do not work yet (tried with 0.9, upgrading requires code changes in
> serl_launcher).

## Quickstart: simulated cube demo (no hardware)

This is not a robot example. It is a smoke test for the whole framework: a red ball stands in for
the robot TCP and is trained to move to the blue cube in a deliberately minimal
[ursina](https://www.ursinaengine.org/) simulator. Everything else is real: the ROS2 adapter,
teleop, data collection, the reward classifier, and the RLPD actor/learner loop run exactly as
they would on hardware.

![Ursina cube demo - training process](docs/media/ursina_cube_demo.gif)

The tutorial in [examples/experiments/cube_demo_ros2/README.md](examples/experiments/cube_demo_ros2/README.md)
walks through the full pipeline. The short version:

```bash
# Terminal 1: the simulated "robot"
python serl_ros2/serl_ros2_sim/ursina_sim.py --rate 50 --cameras front,wrist \
    --image-width 128 --image-height 128 --publish-images

# Terminal 2: record demos (teleop via SpaceMouse, keyboard, or gamepad)
PYTHONPATH=examples python -m serl_framework.train.record_demos --exp_name cube_demo_ros2 --successes_needed 10

# Terminals 3 + 4: learner and actor
cd examples/experiments/cube_demo_ros2
bash run_learner.sh ../../../demo_data/<your_demo>.pkl
bash run_actor.sh --ip localhost
```

The training scripts find task configs through a mapping module (`--config_mapping`,
default `experiments.mappings`). For the bundled examples this module lives in `examples/`, hence
`PYTHONPATH=examples`. For your own experiments, point `--config_mapping` at your own package.

## Architecture

The fork replaces the original HTTP/Flask robot bridge with a **RobotAdapter**: a plain Python
interface between the Gymnasium env and the robot. The ROS2 implementation of that adapter
(`serl_ros2`) is the only component that talks ROS2, so the learning stack stays robot- and
middleware-agnostic:

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

The actor and learner run as separate processes connected only by AgentLace, so the learner can
live on any GPU machine, including a cloud instance. Details on the adapter design, state
synchronization, and the distributed setup are in [docs/architecture.md](docs/architecture.md).

## Code structure

| Directory | Description |
| --- | --- |
| [serl_launcher](serl_launcher) | Learning core (agents, networks, replay buffer, reward classifier), almost unchanged from upstream |
| [serl_framework](serl_framework) | ROS-free robot-agnostic layer: adapter interfaces, `RobotEnv`, wrappers, utilities |
| [serl_framework/train](serl_framework/serl_framework/train) | Training entry points (`train_rlpd`, `record_demos`, ...), moved here from upstream's `examples/` |
| [serl_ros2](serl_ros2) | ROS2 adapter implementation, teleop nodes, ursina simulator |
| [serl_msgs](serl_msgs) | ROS2 service definitions |
| [examples](examples) | Example task configs (currently the simulated cube demo) |
| [docs](docs) | Guides and media |

## Using it on your own robot and task

1. [docs/robot_integration.md](docs/robot_integration.md): connect your robot (topic/service
   contract, verification).
2. [docs/task_walkthrough.md](docs/task_walkthrough.md): set up and train a new task, starting
   with the teleop check, through task design, data collection, training, and evaluation.
3. [docs/training_recommendations.md](docs/training_recommendations.md): how to operate a
   training run well (demos, the intervention protocol, metrics, checkpoint evaluation).

Package details live in [serl_framework/README.md](serl_framework/README.md) and
[serl_ros2/README.md](serl_ros2/README.md).

## Known limitations / future work

- Experiment configuration is split between a `config.py` and an `adapter_config.yaml` per
  experiment. Collecting all settings in a single YAML file is a planned improvement.
- The control loop publishes pose commands without an execution feedback channel. There is no
  built-in confirmation that a command was accepted or tracked.
- Episode horizon timeouts are reported as terminal (`RobotEnv.step()` folds the
  `max_episode_length` timeout into `done` instead of setting `truncated`), inherited from the
  original code base. Harmless for sparse rewards and short horizons, but it deviates from
  Gymnasium time-limit semantics and matters for long horizons or dense rewards. Changing it
  requires an audit of everything that consumes the `dones`/`masks` fields.
- Dual-arm setups are not yet addressed by the ROS2 adapter design.
- The BC and HG-DAgger baseline scripts were not ported. See the
  [original repository](https://github.com/rail-berkeley/hil-serl) for those.

## Citation

This fork builds on HIL-SERL. If you use this code for your research, please cite the original
paper:

```bibtex
@misc{luo2024hilserl,
      title={Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning},
      author={Jianlan Luo and Charles Xu and Jeffrey Wu and Sergey Levine},
      year={2024},
      eprint={2410.21845},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## License and acknowledgements

Licensed under the Apache License 2.0, like the original repository.
All credit for the HIL-SERL method and the core implementation goes to the original authors at
RAIL, UC Berkeley.
