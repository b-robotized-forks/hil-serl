"""Experiment config-mapping helpers for the training scripts.

The training scripts look up ``mapping[experiment_name]`` to obtain the
``TrainConfig`` class for a given experiment. Mappings are ordinary dicts from
experiment name to either the class itself or a dotted import path string.
String values are resolved lazily so that heavy dependencies like ``jax`` are
only loaded when a specific experiment is actually selected.

Users provide their own mapping module (see ``load_config_mapping``) and point
the scripts at it with the ``--config_mapping`` flag.
"""

import importlib

DEFAULT_MAPPING_ATTRIBUTE = "CONFIG_MAPPING"


def _resolve(dotted_path: str):
    """Import and return the object at *dotted_path* (e.g. 'pkg.mod.Class')."""
    module_path, _, attr_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


class LazyConfigMapping(dict):
    """Dict whose string values are lazily resolved to the actual class on first access."""

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, str):
            resolved = _resolve(value)
            self[key] = resolved
            return resolved
        return value


def load_config_mapping(spec: str) -> dict:
    """Load an experiment config mapping from a module path specification.

    Args:
        spec: ``"module.path"`` or ``"module.path:ATTRIBUTE"``. The attribute
            defaults to ``CONFIG_MAPPING`` and must be a dict mapping
            experiment names to TrainConfig classes (or dotted path strings,
            see ``LazyConfigMapping``).

    Returns:
        The mapping dict exported by the module.
    """
    module_path, _, attr_name = spec.partition(":")
    attr_name = attr_name or DEFAULT_MAPPING_ATTRIBUTE
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import config mapping module '{module_path}': {exc}. "
            "Make sure the directory containing your experiments package is on PYTHONPATH "
            "(e.g. PYTHONPATH=examples when using the bundled experiments), or pass "
            "--config_mapping pointing at your own module."
        ) from exc
    mapping = getattr(module, attr_name, None)
    if not isinstance(mapping, dict):
        raise AttributeError(
            f"Config mapping module '{module_path}' has no dict attribute '{attr_name}'."
        )
    return mapping


def resolve_experiment_config_class(mapping: dict, exp_name: str):
    """Return the TrainConfig class for ``exp_name``, with a helpful error if unknown."""
    if exp_name not in mapping:
        known = ", ".join(sorted(str(k) for k in mapping)) or "<empty>"
        raise KeyError(f"Unknown experiment '{exp_name}'. Available experiments: {known}")
    return mapping[exp_name]
