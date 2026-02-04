"""Record labeled success/failure transitions for reward classifier training.

Adapted from examples/record_success_fail.py of the original hil-serl repository
(rail-berkeley/hil-serl); the core recording loop is the original authors' work.

Drive the robot via teleop while the script records transitions. Press SPACE to
enter success mode (rate limited success labels; press SPACE again to undo the
pending ones), press ESC to commit pending successes and reset the episode.
Failure transitions from just before a success are discarded via
--discard_before_success_seconds so mislabeled near-success frames stay out of
the failure set.
"""

import copy
import os
import time
from collections import deque
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard

from serl_framework.train.config import build_train_config
from serl_framework.train.mappings import load_config_mapping
from serl_framework.train.script_utils import override_episode_length

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string(
    "config_mapping",
    "experiments.mappings",
    "Dotted module path (optionally ':ATTRIBUTE') exporting the experiment config "
    "mapping. The module must be importable, e.g. PYTHONPATH=examples for the "
    "bundled experiments.",
)
flags.DEFINE_integer("successes_needed", 200, "Number of successful images to collect.")
flags.DEFINE_integer(
    "failures_needed",
    None,
    "Minimum number of failure images to collect. If omitted, no failure minimum is required.",
)
flags.DEFINE_float(
    "success_samples_per_second",
    5.0,
    "In success mode, label at most this many transitions per second as success.",
)
flags.DEFINE_float(
    "discard_before_success_seconds",
    1.0,
    "When Space is pressed, discard this many seconds of immediately preceding transitions.",
)
flags.DEFINE_integer(
    "episode_length",
    None,
    "Override EnvConfig.MAX_EPISODE_LENGTH for this run (steps).",
)
flags.DEFINE_integer(
    "max_episode_steps",
    None,
    "Alias for --episode_length.",
)
flags.DEFINE_integer(
    "negative_sample_stride",
    1,
    "Record only every Nth candidate failure transition. 1 records all failures.",
)
flags.DEFINE_string(
    "config",
    None,
    "Path to an experiment YAML. Overrides the default path from TrainConfig.",
)
flags.DEFINE_string(
    "teleop",
    None,
    "Override teleop device: 'joy' or 'spacemouse'. Default: read from config.",
)
flags.DEFINE_string(
    "output_dir",
    "./classifier_data",
    "Directory for generated success/failure pkl files.",
)
flags.DEFINE_string(
    "success_output",
    None,
    "Optional exact output path for successful transitions.",
)
flags.DEFINE_string(
    "failure_output",
    None,
    "Optional exact output path for failure transitions.",
)


space_pressed = False
enter_success_mode_requested = False
manual_reset_requested = False


def on_press(key):
    global space_pressed, enter_success_mode_requested, manual_reset_requested
    if key == keyboard.Key.space:
        if not space_pressed:
            enter_success_mode_requested = True
        space_pressed = True
    elif key == keyboard.Key.esc:
        manual_reset_requested = True


def on_release(key):
    global space_pressed
    if key == keyboard.Key.space:
        space_pressed = False


def _resolve_episode_length() -> int | None:
    """Combine --episode_length and its --max_episode_steps alias into one value."""
    if FLAGS.episode_length is not None and FLAGS.max_episode_steps is not None:
        if int(FLAGS.episode_length) != int(FLAGS.max_episode_steps):
            raise ValueError("--episode_length and --max_episode_steps disagree.")
    max_episode_steps = (
        FLAGS.max_episode_steps
        if FLAGS.max_episode_steps is not None
        else FLAGS.episode_length
    )
    if max_episode_steps is None:
        return None
    if max_episode_steps <= 0:
        raise ValueError("--max_episode_steps/--episode_length must be > 0 when provided.")
    return int(max_episode_steps)


def main(_):
    global enter_success_mode_requested, manual_reset_requested
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
    )
    listener.start()

    extra_overrides = {}
    if FLAGS.teleop:
        extra_overrides["teleop_device"] = FLAGS.teleop
        print(f"Using teleop device: {FLAGS.teleop}")
    mapping = load_config_mapping(FLAGS.config_mapping)
    config = build_train_config(FLAGS.config, FLAGS.exp_name, mapping, extra_overrides)

    requested_episode_length = _resolve_episode_length()
    if FLAGS.successes_needed < 0:
        raise ValueError("--successes_needed must be >= 0.")
    if FLAGS.failures_needed is not None and FLAGS.failures_needed < 0:
        raise ValueError("--failures_needed must be >= 0 when provided.")
    if FLAGS.negative_sample_stride <= 0:
        raise ValueError("--negative_sample_stride must be >= 1.")

    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    if requested_episode_length is not None:
        override_episode_length(env, requested_episode_length, FLAGS.exp_name)
    success_sample_period_s = 1.0 / max(1e-6, float(FLAGS.success_samples_per_second))
    discard_before_success_s = max(0.0, float(FLAGS.discard_before_success_seconds))
    print(
        "record_success_fail mode: "
        f"success_samples_per_second={FLAGS.success_samples_per_second:.3f} "
        f"(period={success_sample_period_s:.3f}s)"
    )
    print(
        "record_success_fail mode: "
        f"discard_before_success_seconds={discard_before_success_s:.3f}"
    )
    print(
        "record_success_fail mode: "
        f"negative_sample_stride={int(FLAGS.negative_sample_stride)}"
    )
    print(
        "Press SPACE to enter success mode (press SPACE again to undo pending successes). "
        "Press ESC to reset and commit pending successes."
    )

    obs, _ = env.reset()
    successes = []
    pending_successes = []
    failures = []
    pending_failures: deque[tuple[float, dict]] = deque()
    success_mode = False
    last_success_sample_s = float("-inf")
    success_needed = FLAGS.successes_needed
    failure_needed = FLAGS.failures_needed
    negative_sample_stride = int(FLAGS.negative_sample_stride)
    negative_candidate_count = 0
    success_pbar = tqdm(total=success_needed, desc="successes")
    failure_pbar = (
        tqdm(total=failure_needed, desc="failures") if failure_needed is not None else None
    )

    def collection_complete() -> bool:
        enough_successes = len(successes) >= success_needed
        enough_failures = failure_needed is None or len(failures) >= failure_needed
        return enough_successes and enough_failures

    def commit_pending_successes(reason: str) -> None:
        if not pending_successes:
            return
        remaining = success_needed - len(successes)
        commit_count = min(remaining, len(pending_successes))
        if commit_count > 0:
            successes.extend(pending_successes[:commit_count])
            success_pbar.update(commit_count)
            print(f"Committed {commit_count} successes on {reason}.")
        pending_successes.clear()

    def enter_failure_mode() -> None:
        nonlocal success_mode, negative_candidate_count
        success_mode = False
        pending_failures.clear()
        negative_candidate_count = 0

    while not collection_complete():
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        episode_done = done or truncated
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=episode_done,
            )
        )
        obs = next_obs
        now_s = time.monotonic()
        flush_before_s = now_s - discard_before_success_s
        while pending_failures and pending_failures[0][0] <= flush_before_s:
            failures.append(pending_failures.popleft()[1])
            if failure_pbar is not None:
                failure_pbar.update(1)

        if enter_success_mode_requested:
            enter_success_mode_requested = False
            if len(successes) >= success_needed:
                print("Ignoring SPACE because the success target is already satisfied.")
            elif not success_mode:
                success_mode = True
                pending_failures.clear()
                last_success_sample_s = float("-inf")
                pending_successes.clear()
                print("Success mode ON. Press SPACE again to undo and return to failure mode.")
            else:
                enter_failure_mode()
                pending_successes.clear()
                print("Success mode canceled. Pending successes discarded.")

        if manual_reset_requested:
            manual_reset_requested = False
            commit_pending_successes("ESC reset")
            enter_failure_mode()
            print("Manual reset requested (ESC). Success mode OFF.")
            obs, _ = env.reset()
            continue

        if success_mode:
            if now_s - last_success_sample_s >= success_sample_period_s:
                pending_successes.append(transition)
                last_success_sample_s = now_s
            # Remaining transitions in success mode are dropped on purpose.
        else:
            negative_candidate_count += 1
            if negative_candidate_count % negative_sample_stride == 0:
                pending_failures.append((now_s, transition))

        if done or truncated:
            commit_pending_successes("episode end")
            enter_failure_mode()
            print("Episode ended. Success mode OFF.")
            obs, _ = env.reset()

    output_dir = FLAGS.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    success_count = len(successes)
    failure_count = len(failures)

    success_file_name = FLAGS.success_output or os.path.join(
        output_dir,
        f"{FLAGS.exp_name}_{success_count}_success_images_{uuid}.pkl",
    )
    if success_count > 0 or FLAGS.success_output:
        success_dir = os.path.dirname(success_file_name)
        if success_dir:
            os.makedirs(success_dir, exist_ok=True)
        with open(success_file_name, "wb") as f:
            pkl.dump(successes, f)
            print(f"saved {success_count} successful transitions to {success_file_name}")
    else:
        print("no successful transitions recorded; not writing success pkl")

    failure_file_name = FLAGS.failure_output or os.path.join(
        output_dir,
        f"{FLAGS.exp_name}_{failure_count}_failure_images_{uuid}.pkl",
    )
    if failure_count > 0 or FLAGS.failure_output:
        failure_dir = os.path.dirname(failure_file_name)
        if failure_dir:
            os.makedirs(failure_dir, exist_ok=True)
        with open(failure_file_name, "wb") as f:
            pkl.dump(failures, f)
            print(f"saved {failure_count} failure transitions to {failure_file_name}")
    else:
        print("no failure transitions recorded; not writing failure pkl")

    success_pbar.close()
    if failure_pbar is not None:
        failure_pbar.close()
    listener.stop()
    env.close()


if __name__ == "__main__":
    app.run(main)
