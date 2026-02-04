"""
Configuration helpers for serl_framework.

Origin: New file for ROS2 migration
Modified: N/A
"""

from typing import Any

import numpy as np
import yaml


def load_yaml_dict(path: str) -> dict[str, Any]:
    """
    Load a YAML file into a dictionary.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed YAML content as a dict.

    Raises:
        ValueError: If the YAML content is not a mapping.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got {type(data).__name__}")
    return data


def resolve_compliance_params(params: dict[str, Any]) -> dict[str, Any]:
    """Compute cartesian_damping from stiffness * damping_ratio if needed.

    Accepts a dict with ``cartesian_stiffness`` and either:
    - ``cartesian_damping``: used as-is (backwards compatible).
    - ``damping_ratio``: scalar or 6-element list. Damping is computed as
      ``stiffness[i] * ratio[i]`` per DOF.

    All other keys (e.g. ``feedforward_wrench``) are preserved in the output.
    """
    stiffness = params.get("cartesian_stiffness")
    if stiffness is None:
        return params
    if "cartesian_damping" in params and "damping_ratio" not in params:
        return params
    ratio = params.get("damping_ratio")
    if ratio is None:
        return params
    if isinstance(ratio, (int, float)):
        ratios = [float(ratio)] * len(stiffness)
    else:
        ratios = [float(r) for r in ratio]
    resolved = dict(params)
    resolved.pop("damping_ratio", None)
    resolved["cartesian_stiffness"] = stiffness
    resolved["cartesian_damping"] = [s * r for s, r in zip(stiffness, ratios)]
    return resolved


def compute_action_boost(action_xyz: list[float], boost_cfg: dict[str, Any]) -> list[float]:
    """Compute directional wrench boost from translation action magnitude.

    The boost activates when any axis exceeds the threshold. The force is
    applied along the normalized direction of all saturated axes, with
    per-axis max force limits.

    For example, with max_force=[1, 1, 5] and X=1.0, Z=1.0 at threshold=0.9,
    the boost direction is normalized(1, 0, 1) but scaled per axis so X
    doesn't exceed 1N and Z doesn't exceed 5N.

    Args:
        action_xyz: Raw translation action values ([-1, 1] range), length >= 3.
            The values are expected to already be expressed in the same frame
            as the wrench output. Callers are expected to convert into the
            TCP/tip frame before invoking this helper.
        boost_cfg: Dict with keys ``enabled``, ``threshold``, ``max_force``.
            ``max_force`` can be a scalar (same for all axes) or a 3-element
            list [x, y, z] for per-axis limits.

    Returns:
        6D wrench list (translation boost + zero torques).
    """
    if not boost_cfg.get("enabled", False):
        return [0.0] * 6
    threshold = float(boost_cfg.get("threshold", 0.9))
    mf = boost_cfg.get("max_force", 2.0)
    if isinstance(mf, (int, float)):
        max_forces = np.full(3, float(mf))
    else:
        max_forces = np.array(mf[:3], dtype=np.float64)

    # Zero out axes below threshold, normalize the rest.
    # Each normalized component scales that axis's max_force.
    action = np.array(action_xyz[:3], dtype=np.float64)
    active = np.abs(action) > threshold
    if not np.any(active):
        return [0.0] * 6
    direction = np.where(active, action, 0.0)
    direction /= np.linalg.norm(direction)
    boost = direction * max_forces
    return [boost[0], boost[1], boost[2], 0.0, 0.0, 0.0]


def apply_config_overrides(
    config: object,
    overrides: dict[str, Any],
    aliases: dict[str, str] | None = None,
    array_keys: set[str] | None = None,
    uppercase_keys: bool = True,
    skip_keys: set[str] | None = None,
) -> object:
    """
    Apply overrides to a config object, supporting key aliases and array conversion.

    Args:
        config: Config instance to mutate.
        overrides: Mapping of keys to override values.
        aliases: Optional mapping of alias keys to canonical keys.
        array_keys: Optional set of keys to coerce list/tuple values into numpy arrays.
        uppercase_keys: If True (default), override keys are uppercased before
            matching against config attributes (suitable for UPPER_CASE EnvConfig
            fields).  If False, keys are used as-is (suitable for lowercase
            TrainConfig fields).
        skip_keys: Optional set of target-key names to skip. 
            Note: skip_keys are matched against the *target* key (after
            uppercasing / alias resolution when ``uppercase_keys=True``).
            If a caller needs to protect UPPER_CASE EnvConfig fields from
            CLI overrides, the skip_keys must also be uppercased.

    Returns:
        The mutated config instance.
    """
    if uppercase_keys:
        normalized_aliases = {k.upper(): v.upper() for k, v in (aliases or {}).items()}
        array_key_set = {k.upper() for k in (array_keys or set())}
    else:
        normalized_aliases = dict(aliases or {})
        array_key_set = set(array_keys or set())

    skip = skip_keys or set()

    for raw_key, value in overrides.items():
        key = str(raw_key).upper() if uppercase_keys else str(raw_key)
        target_key = normalized_aliases.get(key, key)

        if target_key in skip:
            continue
        if not hasattr(config, target_key):
            continue

        if target_key in array_key_set and isinstance(value, (list, tuple)):
            value = np.array(value)

        existing = getattr(config, target_key, None)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged = existing.copy()
            merged.update(value)
            setattr(config, target_key, merged)
        else:
            setattr(config, target_key, value)

    return config
