"""Base training configuration and experiment YAML loading.

``DefaultTrainingConfig`` originates from examples/experiments/config.py of the original
hil-serl repository (rail-berkeley/hil-serl); the YAML loading helpers are additions of this fork.

``DefaultTrainingConfig`` is the base class every experiment's ``TrainConfig``
derives from. Experiments override its fields and implement
``get_environment()`` / ``process_demos()``. ``build_train_config`` is the
shared entry point the training scripts use to construct a config with
consistent precedence: class defaults < experiment YAML < CLI overrides.
"""

from abc import abstractmethod
from typing import Any

from serl_framework.utils.config import apply_config_overrides, load_yaml_dict


def load_experiment_config(path: str) -> dict[str, Any]:
    """Load an experiment YAML and return top-level overrides (excluding ``adapter:``)."""
    data = load_yaml_dict(path)
    return {k: v for k, v in data.items() if k != "adapter"}


def build_train_config(
    config_path: str | None,
    exp_name: str,
    config_mapping: dict,
    extra_overrides: dict[str, Any] | None = None,
):
    """Build the experiment TrainConfig with the YAML as the settings source.

    Shared entry point for the training scripts, so every path applies the
    experiment YAML identically. Priority: class defaults < experiment YAML
    < extra_overrides.

    Args:
        config_path: Experiment YAML path. When None or empty, the TrainConfig's
            own default ``experiment_config_path`` (if any) is used.
        exp_name: Experiment key in ``config_mapping``.
        config_mapping: Mapping of experiment names to TrainConfig classes
            (see ``serl_framework.train.mappings``).
        extra_overrides: Optional final overrides on top of the YAML (CLI flags).

    Returns:
        The configured TrainConfig instance.
    """
    import os

    from serl_framework.train.mappings import resolve_experiment_config_class

    config = resolve_experiment_config_class(config_mapping, exp_name)()
    if config_path:
        config.experiment_config_path = config_path
    overrides: dict[str, Any] = {}
    effective_path = str(getattr(config, "experiment_config_path", "") or "")
    if effective_path and os.path.isfile(effective_path):
        overrides = load_experiment_config(effective_path)
    if extra_overrides:
        overrides.update(extra_overrides)
    apply_config_overrides(config, overrides, uppercase_keys=False)
    return config


class DefaultTrainingConfig:
    """Default training configuration."""

    agent: str = "drq"
    max_traj_length: int = 100
    batch_size: int = 256
    cta_ratio: int = 2
    discount: float = 0.97
    # SAC target entropy for the temperature auto-tuner. None (default) keeps
    # SAC's own behaviour: it computes -action_dim/2, so leaving this unset
    # changes nothing. Set an explicit float to override: raise it (less
    # negative, e.g. -2.0) to keep the policy more stochastic / exploratory
    # for longer; lower it (more negative) for a more deterministic,
    # exploitative policy. Changing this is a fresh-run setting (baked into
    # the agent at creation).
    target_entropy: float | None = None

    max_steps: int = 1000000
    replay_buffer_capacity: int = 200000

    random_steps: int = 0
    training_starts: int = 100
    steps_per_update: int = 50

    log_period: int = 10
    eval_period: int = 2000

    # "resnet" for ResNet10 from scratch and "resnet-pretrained" for frozen ResNet10 with pretrained weights
    encoder_type: str = "resnet-pretrained"
    demo_path: str | None = None
    checkpoint_period: int = 0
    buffer_period: int = 0

    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5

    image_keys: list[str] | None = None
    classifier_keys: list[str] | None = None
    proprio_keys: list[str] | None = None

    # Optional path to an experiment YAML applied on top of the class defaults
    # (see build_train_config). Experiments may set a default here so their
    # YAML is picked up without passing --config.
    experiment_config_path: str | None = None

    # "single-arm-learned-gripper", "dual-arm-learned-gripper" for with learned gripper,
    # "single-arm-fixed-gripper", "dual-arm-fixed-gripper" for without learned gripper (i.e. pregrasped)
    setup_mode: str = "single-arm-fixed-gripper"
    teleop_base_frame_id: str = "base"
    teleop_tcp_frame_id: str = "tcp"
    # Used only for teleop adapters that do not provide frame metadata (e.g. SpaceMouseTeleop).
    # Ignored when teleop_device == "joy".
    teleop_spacemouse_frame_id: str = "tcp"

    @classmethod
    def from_yaml(cls, path: str) -> "DefaultTrainingConfig":
        """Create a config instance with values overridden from a YAML file.

        Keys under ``adapter:`` are skipped (they belong to the adapter).
        All other top-level keys are applied as attribute overrides.
        """
        overrides = load_experiment_config(path)
        config = cls()
        apply_config_overrides(config, overrides, uppercase_keys=False)
        return config

    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        raise NotImplementedError

    @abstractmethod
    def process_demos(self, demo):
        raise NotImplementedError
