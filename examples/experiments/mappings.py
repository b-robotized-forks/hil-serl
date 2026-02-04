"""Experiment-name mapping for the bundled example experiments.

The training scripts (serl_framework.train.train_rlpd, etc.) look up
``CONFIG_MAPPING[experiment_name]`` to obtain the ``TrainConfig`` class for
a given experiment. Add new entries here when creating new experiments, or
point the scripts at your own mapping module with --config_mapping.

Entries are dotted import path strings resolved lazily on first access, so
heavy dependencies like jax are only loaded for the selected experiment.
"""

from serl_framework.train.mappings import LazyConfigMapping

CONFIG_MAPPING = LazyConfigMapping({
    "cube_demo_ros2": "experiments.cube_demo_ros2.config.TrainConfig",
})
