"""Tests for the experiment config-mapping helpers."""

import sys

import pytest

from serl_framework.train.mappings import (
    LazyConfigMapping,
    load_config_mapping,
    resolve_experiment_config_class,
)


def _write_mapping_module(tmp_path, name: str, body: str) -> None:
    (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")


@pytest.fixture
def module_dir(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    # Drop any modules imported from tmp_path so tests stay independent.
    for mod in [m for m in sys.modules if m.startswith("mapping_fixture")]:
        del sys.modules[mod]


def test_lazy_mapping_resolves_dotted_path_once(module_dir) -> None:
    _write_mapping_module(module_dir, "mapping_fixture_target", "class TrainConfig:\n    pass\n")
    mapping = LazyConfigMapping({"exp": "mapping_fixture_target.TrainConfig"})

    resolved = mapping["exp"]
    assert resolved.__name__ == "TrainConfig"
    # Second access returns the cached class, not the string.
    assert mapping["exp"] is resolved


def test_lazy_mapping_passes_through_classes() -> None:
    class TrainConfig:
        pass

    mapping = LazyConfigMapping({"exp": TrainConfig})
    assert mapping["exp"] is TrainConfig


def test_load_config_mapping_default_attribute(module_dir) -> None:
    _write_mapping_module(module_dir, "mapping_fixture_default", "CONFIG_MAPPING = {'exp': object}\n")
    mapping = load_config_mapping("mapping_fixture_default")
    assert mapping == {"exp": object}


def test_load_config_mapping_custom_attribute(module_dir) -> None:
    _write_mapping_module(module_dir, "mapping_fixture_custom", "MY_MAPPING = {'exp': object}\n")
    mapping = load_config_mapping("mapping_fixture_custom:MY_MAPPING")
    assert mapping == {"exp": object}


def test_load_config_mapping_missing_module_mentions_pythonpath() -> None:
    with pytest.raises(ImportError, match="PYTHONPATH"):
        load_config_mapping("mapping_fixture_does_not_exist")


def test_load_config_mapping_missing_attribute(module_dir) -> None:
    _write_mapping_module(module_dir, "mapping_fixture_empty", "x = 1\n")
    with pytest.raises(AttributeError, match="CONFIG_MAPPING"):
        load_config_mapping("mapping_fixture_empty")


def test_resolve_experiment_config_class_unknown_name_lists_known() -> None:
    with pytest.raises(KeyError, match="cube_demo"):
        resolve_experiment_config_class({"cube_demo": object}, "typo_name")
