"""Tests for the base training config and experiment YAML loading."""

from pathlib import Path

import pytest

from serl_framework.train.config import (
    DefaultTrainingConfig,
    build_train_config,
    load_experiment_config,
)


class _DummyTrainConfig(DefaultTrainingConfig):
    batch_size: int = 64


def _write_yaml(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_load_experiment_config_skips_adapter_section(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "exp.yaml", "batch_size: 32\nadapter:\n  cameras: [front]\n")
    overrides = load_experiment_config(path)
    assert overrides == {"batch_size": 32}


def test_from_yaml_applies_overrides(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "exp.yaml", "discount: 0.5\nadapter:\n  frame_id: base\n")
    config = _DummyTrainConfig.from_yaml(path)
    assert config.discount == 0.5
    assert config.batch_size == 64


def test_build_train_config_priority_defaults_yaml_cli(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "exp.yaml", "batch_size: 32\ndiscount: 0.5\n")
    mapping = {"dummy": _DummyTrainConfig}

    config = build_train_config(path, "dummy", mapping, extra_overrides={"discount": 0.9})

    assert config.batch_size == 32  # YAML beats the class default (64)
    assert config.discount == 0.9  # CLI override beats the YAML (0.5)
    assert config.experiment_config_path == path


def test_build_train_config_without_yaml_uses_defaults() -> None:
    config = build_train_config(None, "dummy", {"dummy": _DummyTrainConfig})
    assert config.batch_size == 64


def test_build_train_config_unknown_experiment() -> None:
    with pytest.raises(KeyError, match="dummy"):
        build_train_config(None, "other", {"dummy": _DummyTrainConfig})
