"""Train a binary reward classifier from recorded success/failure transitions.

Adapted from examples/train_reward_classifier.py of the original hil-serl repository
(rail-berkeley/hil-serl); the training procedure is the original authors' work.

Loads success and failure pkl files (as produced by record_success_fail.py),
trains a sigmoid classifier on the configured image keys, and saves the
checkpoint to classifier_ckpt/ in the current working directory.
"""

import glob
import os
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import optax
from tqdm import tqdm
from absl import app, flags

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from serl_framework.train.config import build_train_config
from serl_framework.train.mappings import load_config_mapping


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string(
    "config_mapping",
    "experiments.mappings",
    "Dotted module path (optionally ':ATTRIBUTE') exporting the experiment config "
    "mapping. The module must be importable, e.g. PYTHONPATH=examples for the "
    "bundled experiments.",
)
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_string(
    "config",
    None,
    "Path to an experiment YAML. Overrides the default path from TrainConfig.",
)
flags.DEFINE_multi_string(
    "classifier_key",
    None,
    "Image key to train the classifier on. Repeatable. Default: read from config.",
)
flags.DEFINE_multi_string(
    "success_glob",
    None,
    "Glob for success pkl files. Repeatable. Default: classifier_data/*success*.pkl",
)
flags.DEFINE_multi_string(
    "failure_glob",
    None,
    "Glob for failure pkl files. Repeatable. Default: classifier_data/*failure*.pkl",
)


def _expand_globs(patterns, default_pattern: str) -> list[str]:
    selected = patterns or [default_pattern]
    paths: list[str] = []
    for pattern in selected:
        matches = glob.glob(pattern)
        if not matches:
            print(f"WARNING: no files matched {pattern!r}")
        paths.extend(matches)
    return sorted(set(paths))


def _ensure_disjoint_paths(success_paths: list[str], failure_paths: list[str]) -> None:
    success_real = {os.path.realpath(path): path for path in success_paths}
    failure_real = {os.path.realpath(path): path for path in failure_paths}
    overlap = sorted(set(success_real).intersection(failure_real))
    if overlap:
        details = "\n".join(
            f"  success={success_real[path]}\n  failure={failure_real[path]}"
            for path in overlap
        )
        raise ValueError(
            "The same pkl file matched both success and failure inputs. "
            "Use more specific globs.\n"
            f"{details}"
        )


def main(_):
    mapping = load_config_mapping(FLAGS.config_mapping)
    config = build_train_config(FLAGS.config, FLAGS.exp_name, mapping)
    if FLAGS.classifier_key:
        config.classifier_keys = list(FLAGS.classifier_key)
    if not config.classifier_keys:
        raise ValueError(
            "classifier_keys is empty. Set classifier_keys in the experiment YAML "
            "or pass --classifier_key=<image_key>."
        )
    print(f"classifier keys: {config.classifier_keys}")

    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )

    success_paths = _expand_globs(
        FLAGS.success_glob,
        os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"),
    )
    print(f"success paths: {success_paths}")
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        for trans in success_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 1
            trans['actions'] = env.action_space.sample()
            pos_buffer.insert(trans)

    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )
    failure_paths = _expand_globs(
        FLAGS.failure_glob,
        os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"),
    )
    _ensure_disjoint_paths(success_paths, failure_paths)
    print(f"failure paths: {failure_paths}")
    for path in failure_paths:
        failure_data = pkl.load(
            open(path, "rb")
        )
        for trans in failure_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 0
            trans['actions'] = env.action_space.sample()
            neg_buffer.insert(trans)

    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)
    classifier = create_classifier(key,
                                   sample["observations"],
                                   config.classifier_keys,
                                   )

    def data_augmentation_fn(rng, observations):
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["observations"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        batch = concat_batches(
            pos_sample, neg_sample, axis=0
        )
        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],
            }
        )

        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/"),
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )


if __name__ == "__main__":
    app.run(main)
