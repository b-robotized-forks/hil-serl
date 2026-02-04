"""
Live reward-classifier probability monitor for HIL-SERL experiments.

This script is intended for quick classifier verification during teleoperation.
It mirrors the action flow of `record_success_fail.py` (policy action is zeros and
teleop can override via `TeleopIntervention`).

Reset behavior:
  - No automatic reset on normal episode end.
  - Pressing Esc triggers an immediate reset.

Typical use:
  1) Train a classifier checkpoint in `classifier_ckpt/`.
  2) Run `python -m serl_framework.train.stream_classifier_prob --exp_name <name>`
     (with your experiment mapping importable, see the --config_mapping flag).
  3) Teleoperate the robot and observe printed classifier probabilities.
"""

import os
import time

# Prefer CUDA for realtime classifier probing in sim; override with
# JAX_PLATFORMS=cpu when needed.
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from absl import app, flags
import jax
import jax.numpy as jnp
import numpy as np

from serl_framework.train.classifier_guard import ensure_classifier_checkpoint_exists
from serl_framework.train.mappings import load_config_mapping, resolve_experiment_config_class
from serl_launcher.networks.reward_classifier import load_classifier_func


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Experiment name in the config mapping.")
flags.DEFINE_string(
    "config_mapping",
    "experiments.mappings",
    "Dotted module path (optionally ':ATTRIBUTE') exporting the experiment config "
    "mapping. The module must be importable, e.g. PYTHONPATH=examples for the "
    "bundled experiments.",
)
flags.DEFINE_float("threshold", 0.75, "Probability threshold to print success flag.")
flags.DEFINE_integer("max_steps", 0, "Max loop steps (<=0 means run until Ctrl+C).")
flags.DEFINE_bool(
    "disable_episode_timeout",
    True,
    "Disable RobotEnv episode timeout for continuous monitoring.",
)
flags.DEFINE_string(
    "checkpoint_path",
    "classifier_ckpt/",
    "Path to classifier checkpoint directory (relative or absolute).",
)


def main(_):
    mapping = load_config_mapping(FLAGS.config_mapping)
    config = resolve_experiment_config_class(mapping, FLAGS.exp_name)()
    if not config.classifier_keys:
        raise ValueError(f"Experiment '{FLAGS.exp_name}' has no classifier_keys configured.")

    checkpoint_path = ensure_classifier_checkpoint_exists(
        exp_name=FLAGS.exp_name,
        classifier_keys=config.classifier_keys,
        checkpoint_path=FLAGS.checkpoint_path,
        context="stream_classifier_prob",
    )

    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    if FLAGS.disable_episode_timeout and hasattr(env.unwrapped, "max_episode_length"):
        env.unwrapped.max_episode_length = int(1e9)
    classifier_fn = load_classifier_func(
        key=jax.random.PRNGKey(0),
        sample=env.observation_space.sample(),
        image_keys=config.classifier_keys,
        checkpoint_path=checkpoint_path,
    )

    _, _ = env.reset()
    print("Reset done (initial only). No auto-reset on normal episode end.")
    print("Press ESC to trigger an immediate reset.")
    print(f"Loaded classifier checkpoint from: {checkpoint_path}")
    if FLAGS.disable_episode_timeout:
        print("Episode timeout disabled for continuous streaming.")

    step = 0
    start_time = time.time()
    was_terminal = False
    success_events = 0
    success_steps = 0
    prev_predicted_success = False
    try:
        while FLAGS.max_steps <= 0 or step < FLAGS.max_steps:
            action = np.zeros(env.action_space.sample().shape, dtype=np.float32)
            next_obs, _, done, truncated, info = env.step(action)

            prob = float(
                np.asarray(1.0 / (1.0 + jnp.exp(-classifier_fn(next_obs)))).reshape(-1)[0]
            )
            teleop_active = "intervene_action" in info
            predicted_success = prob > FLAGS.threshold
            elapsed_s = time.time() - start_time
            if predicted_success:
                success_steps += 1
            if predicted_success and not prev_predicted_success:
                success_events += 1
            prev_predicted_success = predicted_success

            print(
                f"step={step:06d} t={elapsed_s:8.2f}s prob={prob:.4f} "
                f"success@{FLAGS.threshold:.2f}={int(predicted_success)} "
                f"success_events={success_events} success_steps={success_steps} "
                f"teleop={int(teleop_active)} done={int(done)} truncated={int(truncated)}"
            )

            is_terminal = bool(done or truncated)
            esc_requested = bool(getattr(env.unwrapped, "terminate", False))
            if esc_requested:
                print("ESC detected -> resetting episode.")
                _, _ = env.reset()
                was_terminal = False
                prev_predicted_success = False
                step += 1
                continue

            if is_terminal and not was_terminal:
                print("Episode ended, continuing without reset (as requested).")
            was_terminal = is_terminal

            step += 1
    finally:
        env.close()


if __name__ == "__main__":
    app.run(main)
