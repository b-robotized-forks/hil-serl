"""Record successful teleop demonstrations for RLPD training.

Adapted from examples/record_demos.py of the original hil-serl repository
(rail-berkeley/hil-serl); the core recording loop is the original authors' work.

Episodes are driven via teleop intervention; only successful episodes are kept.
Progress is saved incrementally after every accepted demo, and a previous
session can be continued with --resume. The experiment YAML in effect is
snapshotted next to the output pkl so recordings stay reproducible.
"""

import copy
import datetime
import os
import pickle as pkl
import time

import numpy as np
from absl import app, flags
from tqdm import tqdm

from serl_framework.train.classifier_guard import ensure_classifier_checkpoint_exists
from serl_framework.train.config import build_train_config
from serl_framework.train.mappings import load_config_mapping
from serl_framework.train.script_utils import override_episode_length, save_experiment_config_snapshot

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string(
    "config_mapping",
    "experiments.mappings",
    "Dotted module path (optionally ':ATTRIBUTE') exporting the experiment config "
    "mapping. The module must be importable, e.g. PYTHONPATH=examples for the "
    "bundled experiments.",
)
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_integer(
    "episode_length",
    None,
    "Override EnvConfig.MAX_EPISODE_LENGTH for this run (steps).",
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
    "run_name",
    None,
    "Human-readable run identifier. Embedded in the output pkl filename in "
    "place of the timestamp, e.g. <exp>_20_demos_<run_name>.pkl.",
)
flags.DEFINE_string(
    "resume",
    None,
    "Path to an existing demo .pkl file to resume from. "
    "Previously recorded transitions are loaded and counting continues.",
)


def main(_):
    extra_overrides = {}
    if FLAGS.teleop:
        extra_overrides["teleop_device"] = FLAGS.teleop
        print(f"Using teleop device: {FLAGS.teleop}")
    mapping = load_config_mapping(FLAGS.config_mapping)
    config = build_train_config(FLAGS.config, FLAGS.exp_name, mapping, extra_overrides)

    ensure_classifier_checkpoint_exists(
        exp_name=FLAGS.exp_name,
        classifier_keys=getattr(config, "classifier_keys", None),
        checkpoint_path="classifier_ckpt/",
        context="record_demos",
    )
    requested_episode_length = None
    if FLAGS.episode_length is not None:
        if FLAGS.episode_length <= 0:
            raise ValueError("--episode_length must be > 0 when provided.")
        requested_episode_length = int(FLAGS.episode_length)
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    if requested_episode_length is not None:
        override_episode_length(env, requested_episode_length, FLAGS.exp_name)

    try:
        obs, info = env.reset()
        print("Reset done")
        transitions = []
        success_count = 0
        success_needed = FLAGS.successes_needed

        # Resume from a previous recording session.
        if FLAGS.resume:
            if not os.path.isfile(FLAGS.resume):
                raise FileNotFoundError(f"Resume file not found: {FLAGS.resume}")
            with open(FLAGS.resume, "rb") as f:
                transitions = pkl.load(f)
            # Count episodes: each done==True marks an episode boundary.
            success_count = sum(1 for t in transitions if t["dones"])
            print(f"Resumed {success_count} demos ({len(transitions)} transitions) from {FLAGS.resume}")
        pbar = tqdm(total=success_needed, initial=success_count)
        trajectory = []
        returns = 0
        step_hz_alpha = 0.2
        step_hz_avg = None
        last_step_call_s = None

        while success_count < success_needed:
            step_call_s = time.perf_counter()
            if last_step_call_s is not None:
                inter_call_dt = step_call_s - last_step_call_s
                step_hz_raw = 1.0 / max(1e-6, inter_call_dt)
                if step_hz_avg is None:
                    step_hz_avg = step_hz_raw
                else:
                    step_hz_avg = (1.0 - step_hz_alpha) * step_hz_avg + step_hz_alpha * step_hz_raw
            last_step_call_s = step_call_s

            actions = np.zeros(env.action_space.sample().shape)
            next_obs, rew, done, truncated, info = env.step(actions)
            returns += rew
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
                    infos=info,
                )
            )
            trajectory.append(transition)

            pbar.set_description(f"Return: {returns}")
            postfix = {}
            if step_hz_avg is not None:
                postfix["hz"] = f"{step_hz_avg:.1f}"
            if "classifier_prob" in info:
                postfix["p"] = f"{info['classifier_prob']:.3f}"
            pbar.set_postfix(postfix)

            obs = next_obs
            if episode_done:
                if info["succeed"]:
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                    # Save incrementally so progress is never lost.
                    if not os.path.exists("./demo_data"):
                        os.makedirs("./demo_data")
                    run_suffix = FLAGS.run_name if FLAGS.run_name else "incremental"
                    incremental_path = f"./demo_data/{FLAGS.exp_name}_demos_{run_suffix}_incremental.pkl"
                    with open(incremental_path, "wb") as f:
                        pkl.dump(transitions, f)
                    save_experiment_config_snapshot(config, incremental_path)
                trajectory = []
                returns = 0
                obs, info = env.reset()

        if not os.path.exists("./demo_data"):
            os.makedirs("./demo_data")
        if FLAGS.run_name:
            suffix = FLAGS.run_name
        else:
            suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{suffix}.pkl"
        with open(file_name, "wb") as f:
            pkl.dump(transitions, f)
            print(f"saved {success_needed} demos to {file_name}")
        save_experiment_config_snapshot(config, file_name)
    finally:
        env.close()


if __name__ == "__main__":
    app.run(main)
