#!/usr/bin/env python3
"""RLPD actor/learner training entry point for HIL-SERL.

Adapted from examples/train_rlpd.py of the original hil-serl repository
(rail-berkeley/hil-serl); the core actor/learner loop is the original authors' work.

Run with --learner on the training machine and --actor on the robot machine
(see the experiment run_*.sh wrappers). Beyond the upstream hil-serl script,
this version adds CSV metrics logging, resumable training (checkpoint plus
replay/demo buffer dumps), actor-side checkpoints aligned to learner steps,
an actor weight pull on startup, and an optional interactive learner pause.
"""

import copy
import glob
import os
import pickle as pkl
import sys
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from gymnasium.wrappers import RecordEpisodeStatistics

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from serl_framework.train.classifier_guard import ensure_classifier_checkpoint_exists
from serl_framework.train.config import build_train_config
from serl_framework.train.csv_log_cleanup import prune_training_csv_logs
from serl_framework.train.csv_logger import CSVLogger
from serl_framework.train.mappings import load_config_mapping
from serl_framework.train.pause_prompt import PAUSE_PROMPT_DEFAULT_TIMEOUT_SEC, offer_training_pause

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string(
    "config_mapping",
    "experiments.mappings",
    "Dotted module path (optionally ':ATTRIBUTE') exporting the experiment config "
    "mapping. The module must be importable, e.g. PYTHONPATH=examples for the "
    "bundled experiments.",
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean(
    "save_actor_checkpoint",
    False,
    "Save local actor-side policy checkpoints when new learner weights arrive. "
    "Checkpoints are written to {checkpoint_path}/actor_checkpoint/.",
)
flags.DEFINE_integer(
    "actor_checkpoint_period",
    10,
    "Save an actor-side checkpoint every N learner weight syncs received by the actor.",
)
flags.DEFINE_integer(
    "client_timeout_ms",
    60000,
    "AgentLace REQ/REP timeout in milliseconds for actor->learner requests.",
)

flags.DEFINE_boolean(
    "debug", False, "Debug mode (disables wandb logging)."
)
flags.DEFINE_enum(
    "logger", "csv", ["csv", "wandb"],
    "Logging backend. 'csv' writes learner_update_metrics.csv, "
    "learner_timer_metrics.csv, and actor_stats.csv under checkpoint_path "
    "(default). 'wandb' uses Weights & Biases (requires login).",
)

flags.DEFINE_string(
    "config", None,
    "Path to an experiment YAML. Overrides the default path from TrainConfig.",
)

flags.DEFINE_string(
    "teleop", None,
    "Override teleop device: 'joy' or 'spacemouse'. Default: read from config.",
)

flags.DEFINE_integer(
    "pause_prompt_period",
    500,
    "Learner only: every N steps offer to pause training on the terminal "
    f"(the offer times out after {PAUSE_PROMPT_DEFAULT_TIMEOUT_SEC:.0f}s). "
    "Set to 0 to disable. Requires an interactive terminal (a tty on stdin).",
)

flags.DEFINE_boolean(
    "eval_argmax", False,
    "When evaluating (--eval_checkpoint_step > 0), sample actions with "
    "argmax=True for deterministic policy rollouts. Default False matches "
    "the historical stochastic eval behaviour.",
)


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


ACTOR_DATASTORE_NAMES = ("actor_env", "actor_env_intvn")
TRAINER_REQUEST_RESET_DATA_CURSORS = "reset-data-cursors"
TRAINER_REQUEST_GET_NETWORK = "get-network"
TRAINER_REQUEST_TYPES = [
    "send-stats",
    TRAINER_REQUEST_RESET_DATA_CURSORS,
    TRAINER_REQUEST_GET_NETWORK,
]
LEARNER_TQDM_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}"


class PendingTransitionRecorder:
    """Record new datastore inserts so the learner can persist resume buffers.

    The learner-side replay/demo buffers are the authoritative training state.
    Wrap their ``insert()`` methods so we can periodically dump only the new
    transitions since the last save, matching the existing pickle format.
    """

    def __init__(self, data_store):
        self._data_store = data_store
        self._original_insert = data_store.insert
        self._pending = []
        self._lock = threading.Lock()

    def install(self):
        def recording_insert(*args, **kwargs):
            transition = args[0] if args else kwargs.get("data_dict")
            self._original_insert(*args, **kwargs)
            if transition is None:
                return
            with self._lock:
                self._pending.append(copy.deepcopy(transition))

        self._data_store.insert = recording_insert

    def flush_to_disk(self, checkpoint_path: str, subdir: str, step: int) -> int:
        if not checkpoint_path:
            return 0
        with self._lock:
            if not self._pending:
                return 0
            transitions = self._pending
            self._pending = []
        out_dir = os.path.join(checkpoint_path, subdir)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"transitions_{step}.pkl"), "wb") as f:
            pkl.dump(transitions, f)
        return len(transitions)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def learner_network_payload(params, step: int) -> dict:
    """Package learner weights for broadcast to actors.

    The payload includes the true learner step so actor-side local checkpoints
    can use learner-aligned step numbers instead of local sync counters.
    """
    return {
        "params": params,
        "step": int(step),
    }


def parse_network_payload(payload) -> tuple[dict, int | None]:
    """Unpack a broadcast network payload from the learner.

    Returns the params tree plus an optional learner step.
    Accepts both the new ``{"params": ..., "step": ...}`` payload and the
    older params-only payload for backward compatibility.
    """
    if isinstance(payload, dict) and "params" in payload and "step" in payload:
        return payload["params"], int(payload["step"])
    return payload, None


def batch_reward_mean(batch) -> float:
    """Return the sampled batch reward mean as a host float."""
    return float(np.asarray(jax.device_get(batch["rewards"])).mean())


def sample_policy_action(agent, observations, rng, argmax: bool):
    """Sample one eval-time policy action.

    Splits the rng, samples from the agent, and returns the action as a
    float32 numpy array with leading batch dims squeezed, plus the advanced rng.
    """
    rng, key = jax.random.split(rng)
    action = agent.sample_actions(
        observations=jax.device_put(observations),
        seed=key,
        argmax=bool(argmax),
    )
    action_np = np.squeeze(np.asarray(jax.device_get(action), dtype=np.float32))
    return action_np, rng


def prune_resumed_csv_logs(csv_dir: str, resume_step: int) -> None:
    """Delete stale CSV rows newer than the checkpoint step being resumed."""
    results = prune_training_csv_logs(csv_dir, max_step=resume_step)
    removed_total = 0
    for result in results:
        if result.reason == "missing_or_empty":
            print(
                f"[learner resume] {result.path.name}: skipped (missing or empty)",
                flush=True,
            )
            continue
        if result.reason == "step_column_missing":
            print(
                f"[learner resume] {result.path.name}: skipped (no matching step column)",
                flush=True,
            )
            continue
        removed_total += result.removed_rows
        print(
            f"[learner resume] {result.path.name}: removed {result.removed_rows} "
            f"stale row(s) newer than learner step {resume_step}",
            flush=True,
        )
    print(
        f"[learner resume] total stale CSV rows removed: {removed_total}",
        flush=True,
    )


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []
        try:
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
                step=FLAGS.eval_checkpoint_step,
            )
            agent = agent.replace(state=ckpt)

            for episode in range(FLAGS.eval_n_trajs):
                obs, _ = env.reset()
                done = False
                truncated = False
                reward = 0.0
                start_time = time.time()
                while not (done or truncated):
                    actions, sampling_rng = sample_policy_action(
                        agent, obs, sampling_rng, argmax=bool(FLAGS.eval_argmax)
                    )
                    next_obs, reward, done, truncated, info = env.step(actions)
                    obs = next_obs

                success = bool(done and reward)
                if success:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(dt)
                success_counter += int(success)
                print(int(success))
                print(f"{success_counter}/{episode + 1}")

            print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
            if time_list:
                print(f"average time: {np.mean(time_list)}")
            else:
                print("average time: n/a (no successful episodes)")
            return  # after done eval, return and exit
        finally:
            env.close()

    start_step = 0
    if FLAGS.save_actor_checkpoint and not FLAGS.checkpoint_path:
        raise ValueError("--save_actor_checkpoint requires --checkpoint_path.")
    if FLAGS.actor_checkpoint_period <= 0:
        raise ValueError("--actor_checkpoint_period must be > 0.")

    datastore_dict = {
        ACTOR_DATASTORE_NAMES[0]: data_store,
        ACTOR_DATASTORE_NAMES[1]: intvn_data_store,
    }

    client = TrainerClient(
        ACTOR_DATASTORE_NAMES[0],
        FLAGS.ip,
        make_trainer_config(request_types=TRAINER_REQUEST_TYPES),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=FLAGS.client_timeout_ms,
    )
    reset_response = client.request(
        TRAINER_REQUEST_RESET_DATA_CURSORS,
        {"stores": list(ACTOR_DATASTORE_NAMES)},
    )
    if reset_response is None or not reset_response.get("success", False):
        print(
            "[actor] WARNING: failed to reset learner data cursors on startup; "
            "actor restart may stall data transfer.",
            flush=True,
        )
    else:
        print(
            "[actor] reset learner data cursors for actor datastores",
            flush=True,
        )

    # Function to update the agent with new params
    _last_param_update_wall = [time.time()]
    _param_update_count = [0]
    _last_learner_step = [None]
    _first_param_update_event = threading.Event()

    def update_params(payload):
        nonlocal agent
        params, learner_step = parse_network_payload(payload)
        if learner_step is not None and _last_learner_step[0] == learner_step:
            print(
                f"[actor] ignoring duplicate learner policy sync "
                f"(learner_step={learner_step})",
                flush=True,
            )
            return
        agent = agent.replace(state=agent.state.replace(params=params))
        _param_update_count[0] += 1
        _last_param_update_wall[0] = time.time()
        _last_learner_step[0] = learner_step
        _first_param_update_event.set()
        step_label = "unknown" if learner_step is None else learner_step
        print(
            f"[actor] received updated policy "
            f"(sync #{_param_update_count[0]}, learner_step={step_label})",
            flush=True,
        )
        if (
            FLAGS.save_actor_checkpoint
            and FLAGS.checkpoint_path
            and _param_update_count[0] % FLAGS.actor_checkpoint_period == 0
        ):
            actor_ckpt_dir = os.path.join(
                os.path.abspath(FLAGS.checkpoint_path),
                "actor_checkpoint",
            )
            os.makedirs(actor_ckpt_dir, exist_ok=True)
            checkpoint_step = (
                learner_step if learner_step is not None else _param_update_count[0]
            )
            checkpoints.save_checkpoint(
                actor_ckpt_dir,
                agent.state,
                step=checkpoint_step,
                keep=20,
            )
            print(
                f"[actor] saved learner-step checkpoint {checkpoint_step} "
                f"(sync #{_param_update_count[0]}) to {actor_ckpt_dir}",
                flush=True,
            )

    client.recv_network_callback(update_params)

    network_response = client.request(TRAINER_REQUEST_GET_NETWORK, {})
    if network_response is not None and network_response.get("success", False):
        print("[actor] fetched current learner policy via request", flush=True)
        update_params(network_response["payload"])
    else:
        print(
            "[actor] WARNING: could not fetch current learner policy; "
            "waiting for next broadcast.",
            flush=True,
        )

    print("[actor] waiting for first learner policy sync before rollout...", flush=True)
    while not _first_param_update_event.wait(timeout=5.0):
        print("[actor] still waiting for first learner policy sync...", flush=True)
    learner_step = _last_learner_step[0]
    print(
        "[actor] first learner policy sync received "
        f"(learner_step={'unknown' if learner_step is None else learner_step}); "
        "starting env reset and rollout.",
        flush=True,
    )

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            episode_done = done or truncated
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=episode_done,
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty'] = info['grasp_penalty']
            data_store.insert(transition)
            if already_intervened:
                intvn_data_store.insert(transition)

            obs = next_obs
            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                episode_return = running_return
                pbar.set_description(f"last return: {episode_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                actor_last_id = data_store.latest_data_id()
                intvn_last_id = intvn_data_store.latest_data_id()
                upload_start = time.time()
                # Use tqdm.write so the banner is not smeared by the
                # progress bar redraws, and frame the block with a rule
                # so the "teleop blocked" window is scannable in the log.
                _rule = "=" * 60
                tqdm.tqdm.write("")
                tqdm.tqdm.write(_rule)
                tqdm.tqdm.write(
                    f"  EPISODE END @ step={step}  last_return={episode_return:.3f}"
                )
                tqdm.tqdm.write(
                    "  ACTOR LOOP BLOCKED  -  teleop paused until reset completes"
                )
                tqdm.tqdm.write(
                    f"  uploading to learner: actor_last_id={actor_last_id} "
                    f"intvn_last_id={intvn_last_id} timeout_ms={FLAGS.client_timeout_ms}"
                )
                tqdm.tqdm.write(_rule)
                update_ok = client.update()
                upload_dt = time.time() - upload_start
                tqdm.tqdm.write(
                    f"  upload done (ok={update_ok}, elapsed={upload_dt:.3f}s)  -  "
                    "running env.reset() ..."
                )
                obs, _ = env.reset()
                tqdm.tqdm.write(
                    "  RESET COMPLETE  -  teleop ACTIVE again"
                )
                tqdm.tqdm.write(_rule)
                tqdm.tqdm.write("")

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)

            secs_since_param = time.time() - _last_param_update_wall[0]
            pbar.set_postfix({
                "replay": len(data_store),
                "intvn": len(intvn_data_store),
                "params": _param_update_count[0],
                "last_p": f"{secs_since_param:.0f}s",
                "l_step": (
                    "-"
                    if _last_learner_step[0] is None
                    else str(_last_learner_step[0])
                ),
            })


##############################################################################


def learner(
    rng,
    agent,
    replay_buffer,
    demo_buffer,
    loggers=None,
    replay_transition_recorder=None,
    demo_transition_recorder=None,
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt is not None:
            start_step = int(os.path.basename(latest_ckpt)[11:]) + 1
    step = start_step
    current_network_step = [max(start_step - 1, 0)]

    _prev_replay_size = [len(replay_buffer)]
    _prev_demo_size = [len(demo_buffer)]
    server = None

    def stats_callback(type: str, payload: dict) -> dict:
        """Handle trainer requests from the actor."""
        nonlocal server
        if type == TRAINER_REQUEST_RESET_DATA_CURSORS:
            stores = payload.get("stores", [])
            if not isinstance(stores, list) or not stores:
                return {"success": False, "message": "payload.stores must be a non-empty list"}
            if server is None:
                return {"success": False, "message": "trainer server not initialized"}

            reset_info = {}
            for store_name in stores:
                if store_name not in server.data_stores or store_name not in server.last_update_id_map:
                    return {"success": False, "message": f"invalid datastore name: {store_name}"}
                old_id = server.last_update_id_map.get(store_name, -1)
                server.last_update_id_map[store_name] = -1
                reset_info[store_name] = old_id

            print(
                f"[learner] reset actor data cursors: {reset_info}",
                flush=True,
            )
            return {"success": True, "payload": reset_info}

        if type == TRAINER_REQUEST_GET_NETWORK:
            return {
                "success": True,
                "payload": learner_network_payload(
                    agent.state.params, current_network_step[0]
                ),
            }

        assert type == "send-stats", f"Invalid request type: {type}"
        actor_stats_logger = None if loggers is None else loggers.get("actor_stats")
        if actor_stats_logger is not None:
            callback_payload = {"callback_learner_step": step, **payload}
            actor_stats_logger.log(callback_payload)

        # Log when new data arrives from the actor.
        replay_now = len(replay_buffer)
        demo_now = len(demo_buffer)
        replay_delta = replay_now - _prev_replay_size[0]
        demo_delta = demo_now - _prev_demo_size[0]
        if replay_delta > 0 or demo_delta > 0:
            print(
                f"[learner] received from actor: "
                f"+{replay_delta} replay (total={replay_now})  "
                f"+{demo_delta} intvn (total={demo_now})",
                flush=True,
            )
            _prev_replay_size[0] = replay_now
            _prev_demo_size[0] = demo_now

        return {}  # not expecting a response

    # Create server
    server = TrainerServer(
        make_trainer_config(request_types=TRAINER_REQUEST_TYPES),
        request_callback=stats_callback,
    )
    server.register_data_store(ACTOR_DATASTORE_NAMES[0], replay_buffer)
    server.register_data_store(ACTOR_DATASTORE_NAMES[1], demo_buffer)
    server.start(threaded=True)

    # Publish the current network immediately so a freshly started or restarted
    # actor can block until it has learner-owned weights before collecting data.
    # When resuming, the loaded state corresponds to the previous checkpoint
    # step rather than start_step.
    initial_publish_step = current_network_step[0]
    server.publish_network(
        learner_network_payload(agent.state.params, initial_publish_step)
    )
    print_green(f"sent initial network to actor (step {initial_publish_step})")

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
        bar_format=LEARNER_TQDM_BAR_FORMAT,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    pbar = tqdm.tqdm(
        range(start_step, config.max_steps),
        dynamic_ncols=True,
        desc="learner",
        bar_format=LEARNER_TQDM_BAR_FORMAT,
    )
    last_replay_reward_mean = float("nan")
    last_demo_reward_mean = float("nan")
    pause_prompt_enabled = FLAGS.pause_prompt_period > 0 and sys.stdin.isatty()
    if FLAGS.pause_prompt_period > 0 and not pause_prompt_enabled:
        print(
            "[learner] pause prompts disabled: stdin is not an interactive terminal",
            flush=True,
        )
    for step in pbar:
        if pause_prompt_enabled and step > 0 and step % FLAGS.pause_prompt_period == 0:
            _rule = "=" * 60
            tqdm.tqdm.write("")
            tqdm.tqdm.write(_rule)
            tqdm.tqdm.write(f"  PAUSE OPPORTUNITY @ learner step {step}")
            paused = offer_training_pause(write=tqdm.tqdm.write)
            if paused:
                tqdm.tqdm.write(f"  [learner] resumed at step {step}")
            tqdm.tqdm.write(_rule)
            tqdm.tqdm.write("")
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                replay_batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                last_replay_reward_mean = batch_reward_mean(replay_batch)
                last_demo_reward_mean = batch_reward_mean(demo_batch)
                batch = concat_batches(replay_batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            replay_batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            last_replay_reward_mean = batch_reward_mean(replay_batch)
            last_demo_reward_mean = batch_reward_mean(demo_batch)
            batch = concat_batches(replay_batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
            current_network_step[0] = step
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(learner_network_payload(agent.state.params, step))

        if step % config.log_period == 0:
            learner_update_logger = None if loggers is None else loggers.get("learner_update")
            learner_timer_logger = None if loggers is None else loggers.get("learner_timer")
            if learner_update_logger is not None:
                learner_update_payload = {
                    **update_info,
                    "sampled_batch": {
                        "replay_reward_mean": last_replay_reward_mean,
                        "demo_reward_mean": last_demo_reward_mean,
                    },
                    "buffer": {
                        "replay_size": len(replay_buffer),
                        "demo_size": len(demo_buffer),
                    },
                }
                learner_update_logger.log(learner_update_payload, step=step)
            if learner_timer_logger is not None:
                learner_timer_logger.log({"timer": timer.get_average_times()}, step=step)

            def _scalar(v, key=None):
                """Extract a plain float from a JAX/numpy scalar or nested dict."""
                if isinstance(v, dict):
                    # Some metrics are dicts (e.g. temperature: {value: ..., lr: ...}).
                    # Try common sub-keys, or return nan.
                    for sub in (key, "value", "mean"):
                        if sub and sub in v:
                            return _scalar(v[sub])
                    return float("nan")
                if hasattr(v, "item"):
                    return v.item()
                return float(v)

            critic_info = update_info.get("critic", {})

            q_mean = _scalar(critic_info.get("predicted_qs", float("nan")))
            target_q_mean = _scalar(critic_info.get("target_qs", float("nan")))
            pbar.set_postfix({
                "r_rep": f"{last_replay_reward_mean:.4f}",
                "r_demo": f"{last_demo_reward_mean:.4f}",
                "q": f"{q_mean:.3f}",
                "tq": f"{target_q_mean:.3f}",
            })

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )
        if (
            step > 0
            and config.buffer_period
            and step % config.buffer_period == 0
        ):
            replay_saved = 0
            demo_saved = 0
            if replay_transition_recorder is not None:
                replay_saved = replay_transition_recorder.flush_to_disk(
                    FLAGS.checkpoint_path, "buffer", step
                )
            if demo_transition_recorder is not None:
                demo_saved = demo_transition_recorder.flush_to_disk(
                    FLAGS.checkpoint_path, "demo_buffer", step
                )
            if replay_saved > 0 or demo_saved > 0:
                print(
                    f"[learner] saved resume buffers at step {step}: "
                    f"{replay_saved} replay, {demo_saved} demo transitions",
                    flush=True,
                )


##############################################################################


def main(_):
    global config
    # The experiment YAML is the settings source everywhere; build the config
    # through the shared helper so training and eval apply it identically.
    # Priority: class defaults < YAML < CLI flags.
    if FLAGS.config:
        print(f"Using experiment config: {FLAGS.config}")
    extra_overrides = {}
    if FLAGS.teleop:
        extra_overrides["teleop_device"] = FLAGS.teleop
        print(f"Using teleop device: {FLAGS.teleop}")
    mapping = load_config_mapping(FLAGS.config_mapping)
    config = build_train_config(
        FLAGS.config,
        FLAGS.exp_name,
        mapping,
        extra_overrides=extra_overrides or None,
    )

    ensure_classifier_checkpoint_exists(
        exp_name=FLAGS.exp_name,
        classifier_keys=getattr(config, "classifier_keys", None),
        checkpoint_path="classifier_ckpt/",
        context="train_rlpd",
    )

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)

    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            target_entropy=config.target_entropy,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            target_entropy=config.target_entropy,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            target_entropy=config.target_entropy,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    resumed_checkpoint_step = None
    if (
        FLAGS.checkpoint_path is not None
        and os.path.exists(FLAGS.checkpoint_path)
        and not FLAGS.eval_checkpoint_step
        and FLAGS.learner
    ):
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt is not None:
            resumed_checkpoint_step = int(os.path.basename(latest_ckpt)[11:])
            input("Checkpoint path already exists. Press Enter to resume training.")
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
            )
            agent = agent.replace(state=ckpt)
            print_green(
                f"Loaded previous checkpoint at step {resumed_checkpoint_step}."
            )

    def create_replay_buffer_and_loggers():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        if FLAGS.logger == "wandb":
            shared_logger = make_wandb_logger(
                project="hil-serl",
                description=FLAGS.exp_name,
                debug=FLAGS.debug,
            )
            loggers = {
                "learner_update": shared_logger,
                "learner_timer": shared_logger,
                "actor_stats": shared_logger,
            }
        else:
            csv_dir = FLAGS.checkpoint_path or "."
            os.makedirs(csv_dir, exist_ok=True)
            if resumed_checkpoint_step is not None:
                prune_resumed_csv_logs(csv_dir, resumed_checkpoint_step)
            loggers = {
                "learner_update": CSVLogger(os.path.join(csv_dir, "learner_update_metrics.csv")),
                "learner_timer": CSVLogger(os.path.join(csv_dir, "learner_timer_metrics.csv")),
                "actor_stats": CSVLogger(os.path.join(csv_dir, "actor_stats.csv")),
            }
        return replay_buffer, loggers

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, loggers = create_replay_buffer_and_loggers()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        replay_transition_recorder = PendingTransitionRecorder(replay_buffer)
        replay_transition_recorder.install()
        demo_transition_recorder = PendingTransitionRecorder(demo_buffer)
        demo_transition_recorder.install()

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            loggers=loggers,
            replay_transition_recorder=replay_transition_recorder,
            demo_transition_recorder=demo_transition_recorder,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
