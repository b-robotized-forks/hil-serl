"""Small shared helpers for the training and recording scripts."""

import os
import shutil


def override_episode_length(env, episode_length: int, exp_name: str) -> None:
    """Apply an episode-length override on the unwrapped environment."""
    inner_env = env.unwrapped
    if not hasattr(inner_env, "max_episode_length"):
        raise ValueError(f"Could not override max_episode_length for experiment '{exp_name}'.")
    inner_env.max_episode_length = episode_length
    if hasattr(inner_env, "config") and hasattr(inner_env.config, "MAX_EPISODE_LENGTH"):
        inner_env.config.MAX_EPISODE_LENGTH = episode_length
    print(f"Overriding max_episode_length to {episode_length}")


def save_experiment_config_snapshot(config, pkl_path: str) -> None:
    """Copy the experiment config yaml next to ``pkl_path`` using the same basename.

    The resulting file (same stem as the pkl, .yaml extension) makes it clear
    which settings produced a given recording, and keeps the pair together
    regardless of where the pkl is later copied.
    """
    src = getattr(config, "experiment_config_path", None)
    if not src or not os.path.isfile(src):
        print(
            f"warning: no experiment_config_path on config (or file missing); "
            f"skipping config snapshot for {pkl_path}"
        )
        return
    dst = os.path.splitext(pkl_path)[0] + ".yaml"
    try:
        shutil.copyfile(src, dst)
        print(f"saved experiment config snapshot to {dst}")
    except OSError as exc:
        print(f"warning: could not save config snapshot to {dst}: {exc}")
