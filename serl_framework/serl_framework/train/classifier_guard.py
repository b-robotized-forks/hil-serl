"""Fail-fast guard for classifier-based training workflows."""

import os
from typing import Sequence

from flax.training import checkpoints


def ensure_classifier_checkpoint_exists(
    *,
    exp_name: str,
    classifier_keys: Sequence[str] | None,
    checkpoint_path: str,
    context: str,
) -> str:
    """
    Ensure classifier checkpoints exist when classifier keys are configured.

    Returns:
        Absolute normalized checkpoint path when the guard passes.

    Raises:
        FileNotFoundError: if classifier is configured but no checkpoint exists.
    """
    if not classifier_keys:
        return os.path.abspath(os.path.expanduser(checkpoint_path))

    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_path))
    latest_ckpt = checkpoints.latest_checkpoint(checkpoint_dir)
    if latest_ckpt is None:
        raise FileNotFoundError(
            f"[{context}] Experiment '{exp_name}' has classifier_keys={list(classifier_keys)} "
            f"but no classifier checkpoint was found in '{checkpoint_dir}'. "
            "Train one first with: "
            f"`python -m serl_framework.train.train_reward_classifier --exp_name {exp_name}`."
        )
    return checkpoint_dir
