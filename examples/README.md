# Examples

This directory holds example experiment configurations for the ROS2-based HIL-SERL pipeline.

- [experiments/cube_demo_ros2](experiments/cube_demo_ros2): a fully simulated end-to-end demo
  (teleop, reward classifier, demonstrations, RLPD training) against the bundled
  [ursina](https://www.ursinaengine.org/) simulator.
  Not a robot example, more of a smoke test for the whole framework: a red ball stands in for the
  TCP and is trained to move to the blue cube. No hardware required. Start here.

## Adding your own experiment

1. Copy `experiments/cube_demo_ros2` as a template: `config.py` (a `TrainConfig` deriving from
   `serl_framework.train.config.DefaultTrainingConfig`), `adapter_config.yaml` (cameras, topics),
   an optional env `wrapper.py`, and the `run_actor.sh` / `run_learner.sh` launch scripts.
2. Register the new experiment in [experiments/mappings.py](experiments/mappings.py)
   (or in your own mapping module, passed to the training scripts via `--config_mapping`).
3. Follow [docs/robot_walkthrough.md](../docs/robot_walkthrough.md) for the full training pipeline.

The training scripts themselves live in the `serl_framework.train` package
(`python -m serl_framework.train.train_rlpd ...`). Run them from the repo root with
`PYTHONPATH=examples` so the mapping module is importable.

The original Franka experiments (RAM insertion, USB pick-up, object handover, egg flip) and the
BC / HG-DAgger baselines are not part of this fork. You find them in the
[original repository](https://github.com/rail-berkeley/hil-serl).
