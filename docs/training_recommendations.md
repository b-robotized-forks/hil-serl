# Training Recommendations

A practical guide for operating a HIL-SERL training run. The first part is distilled from the
HIL-SERL [paper](https://arxiv.org/abs/2410.21845) and the original authors'
[Franka walkthrough](https://github.com/rail-berkeley/hil-serl/blob/main/docs/franka_walkthrough.md),
the same protocol they used to reach 100% success on RAM and USB insertion in 1.5-2.5 hours.
The second part ("Operator observations") collects patterns we saw across our own training runs
that extend or sharpen the published guidance.

## TL;DR

1. Record ~20 successful demonstrations before training.
2. In early training, intervene frequently enough that **at least one in three episodes terminates
   with reward**, even if the reward was obtained via intervention.
3. Use **short, targeted** interventions, not long ones that lead to task success.
4. Taper interventions as the policy starts succeeding without help.
5. In late training, induce specific failure modes to practice recovery.
6. Evaluate checkpoints separately with enough trials. Training curves are not ground truth.

## Pre-training: demonstrations

A small number of high-quality demonstrations is critical to bootstrapping RL. The walkthrough is
explicit:

> "A small number of human demonstrations is crucial to accelerating the reinforcement learning
> process, and for this task, we use 20 demonstrations."

Practical setup:

- Use a SpaceMouse for demos (the walkthrough's "First and foremost" tip applies to demo
  collection as well as interventions).
- Collect ~20 successful demos per task configuration
  (see the [walkthrough](robot_walkthrough.md#4-recording-demonstrations)).

Demonstrations load into the demo buffer at learner startup and are mixed 50/50 with online replay
samples in every batch (paper: *"The learner process samples data equally from the demo and RL
replay buffers."*).

## Why interventions matter

Demonstrations alone are not enough. HIL-SERL's central empirical claim is that **online human
corrections during training** are what separates it from imitation-learning baselines and from
demo-only RL. Before discussing how to time interventions, it helps to understand what they do:

### 1. They make the problem tractable at all

From the paper:

> "We first note that RL from scratch, without any demonstrations or corrections, achieved 0%
> success rate on all tasks. To validate the importance of online human corrections, we increased
> the number of demonstrations in the offline buffer of SERL tenfold, from the usual 20 to 200.
> However, without any online corrections, this approach resulted in significantly lower success
> rates compared to HIL-SERL, including a complete failure (0% success) on the car dashboard
> assembly. This confirms the crucial role of online corrections in facilitating policy learning."

### 2. They rescue the policy from dead ends

From the paper:

> "This intervention is crucial in scenarios where the policy leads the robot to an unrecoverable
> or undesirable state, or when it becomes stuck in a local optimum that would otherwise require a
> significant amount of time to overcome without human assistance."

### 3. They densify the reward signal early

From the walkthrough:

> "We also found it beneficial to intervene to help the policy finish the task and get the reward
> semi-frequently even in the beginning (i.e. letting 1/3 or more of the episodes get reward) -
> the frequent rewards will help the value backup propagate faster and speed up training."

See Phase 1 below for the operational version of this rule.

### 4. They shape where the policy learns to live

From the paper:

> "Through policy learning, we observe the gradual development of a funnel-like shape connecting
> the initial states to the target location. [...] demonstrations and corrections can be regarded
> as 'nominal trajectories' around which RL methods develop funnels for stabilization."

### How intervention data is used

From the paper:

> "When a human intervenes, their action a_itv is applied to the robot instead of the policy's
> action a_RL. We store the intervention data in both the demonstration and RL data buffers.
> However, we add the policy's transitions (i.e., the states and actions before and after the
> intervention) only to the RL buffer."

In this codebase's actor loop, every environment step is inserted into the online replay buffer.
Only steps whose `info` contains `intervene_action` are also inserted into the intervention/demo
buffer. The learner samples transition batches, not whole trajectories: half the batch comes from
online replay and half from the demo/intervention buffer. So a 120-step episode with 10
intervention steps contributes all 120 transitions to replay, but only those 10
intervention-action transitions to the demo/intervention stream.

### Fine-tuning from a useful checkpoint without old buffers

When a checkpoint already has the right broad behavior but still fails on a specific hard mode,
prefer a clean-buffer fine-tune over continuing with all old replay. Copy only the checkpoint
directory into a new run directory and leave out `buffer/` and `demo_buffer/` unless you
intentionally want the old transitions to keep participating. On resume, the learner restores the
full SAC train state from the checkpoint (actor, critics, target critics, temperature, and
optimizer state), but the replay buffers start from the configured demos plus new online data.

The policy keeps the weights that already understand the task, while new interventions on the hard
states are no longer diluted by a large old replay history. The new online replay still contains
normal autonomous rollouts, so the fine-tune is not only hard-case data, but the intervention/demo
half of each learner batch becomes much more focused on the recovery behaviors you are currently
teaching.

## Training protocol: three phases of intervention

The walkthrough's "Additional Tips for Success" section lays out three regimes for intervention
rate across a training run.

### Phase 1: early training, intervene often

From the walkthrough:

> "Give interventions frequently at the beginning parts of training (this can be as frequent as
> intervening for some timesteps every episode or every other episode)."

Typical pattern:

1. Let the policy act for **20-30 timesteps** at the start of the episode. This is the exploration
   window.
2. If the policy is veering toward clearly useless behavior (moving away from the target, drifting
   into the workspace corner, ...), take over briefly and guide the end effector or object near
   the goal.
3. Hand control back and let the policy attempt the decisive part of the task unassisted.
4. Intervene again if necessary to complete the episode so it terminates with reward.

The key quantitative rule for this phase: **at least ~33% of episodes should terminate with
`reward = 1`**, even if the reward came via intervention. A direct measurement is available in
`learner_update_metrics.csv`, column `sampled_batch/replay_reward_mean`.

### Phase 2: middle training, taper

From the walkthrough:

> "As the policy starts to behave more reasonably (policy can successfully finish the task by
> itself with minimal/none interventions every once in a while), we can taper off the rate of
> interventions significantly. At this stage, we would mostly hold off on intervening unless the
> policy is repeatedly making the same mistake."

The signal that you are leaving phase 1 is the emergence of **intervention-free successes**. Watch
`actor_stats.csv`: `environment/succeed` together with `environment/episode/intervention_count`
tells you which successes happened without help.

### Phase 3: late training, targeted robustness

From the walkthrough:

> "Sometimes, we may want the trained policy to have a more robust retry behavior (meaning the
> policy can successfully complete the task even if it makes a mistake earlier on) or to be more
> robust to external disturbances. In that case, we can also use interventions to help it practice
> these edge cases. [...] Using interventions to bring the policy to a place to practice these
> recovery behaviors is effective for reaching 100% success rate as these mistakes may occur too
> infrequently otherwise yet the policy still needs to learn how to recover to improve from a 97%
> success rate policy to a 100% success rate policy."

In this phase, use interventions to *cause* the mistakes you want the policy to recover from, then
hand control back so the policy practices the recovery.

## Intervention style: short and targeted

From the paper:

> "As the policy improves, shorter interventions are sufficient to correct fewer mistakes."

And from the walkthrough:

> "It's important to strike a balance between letting the policy explore (essential for RL) while
> also guiding it to explore efficiently through interventions. [...] We would typically let it
> explore these random motions for 20-30 timesteps then intervene to guide the object close to the
> insertion port, where we would let the policy practice the insertion."

### Resist providing long interventions that lead to successes

From the paper:

> "It's also important to note that we should avoid providing long sparse interventions that lead
> to task successes. Such an intervention strategy will cause the overestimation of the value
> function, particularly in the early stages of the training process; which can result in unstable
> training dynamics."

### Prefer corrections that still reach reward

With sparse reward, an intervention is most valuable when it creates a trajectory that eventually
reaches success. Intervention transitions are sampled heavily through the demo/intervention
buffer, but they are still trained by SAC/RLPD Bellman targets, not by a separate supervised
imitation loss. A correction from a bad state teaches the most when the same episode later reaches
`reward = 1`, because the critic can back up value through the corrected states and actions.

Failed intervention episodes are not useless: they can still move the system into better-looking
states, and those states may get value from similar successful continuations elsewhere in replay.
But they are weaker than successful corrections, because no reward was actually demonstrated after
the rescue. If there are too few steps left to plausibly finish, avoid long doomed interventions
that only add unrewarded intervention-buffer transitions. Prefer either a short correction that
leaves the policy a real chance to finish, or let the episode fail cleanly and spend the next
reset on a recovery attempt that can reach success.

The "what not to do" signal is indirect. SAC is not doing explicit pairwise preference learning,
it learns Q-values for state-action pairs. For example, in an insertion task:

```text
state S: part misaligned at the goal
A_bad:  policy pushes on while misaligned -> no reward / poor continuation
A_good: intervention backs out and realigns -> later reaches success
```

Over training, the critic can assign a higher value to `Q(S, A_good)` than to `Q(S, A_bad)` in
similar states, and the actor update then moves the policy toward actions the critic rates highly.
This comparison is much weaker when the corrective intervention also times out, because there is
no successful continuation to back up through it.

## Time-to-converge benchmarks

From the walkthrough, for calibration (runs with randomization on and occasional interventions per
the protocol above):

| Task | Time to 100% success |
|------|----------------------|
| RAM insertion | ~1.5 hours |
| USB pickup + insertion | ~2.5 hours |

Tasks with comparable geometry and a reliable reward signal can expect similar wall-clock
convergence. Classifier noise, heavy reset randomization, or slower control rates extend this.

## Signals to watch during a run

The learner writes these to the CSV files under `checkpoint_path` (see the
[walkthrough](robot_walkthrough.md#5-policy-training)). The paper and walkthrough give no
numerical thresholds. The anchor for intervention density is the "1/3 of episodes get reward" rule
from Phase 1.

| Metric | Source CSV column |
|--------|-------------------|
| Sampled batch reward mean | `learner_update_metrics.csv`: `sampled_batch/replay_reward_mean` |
| Buffer sizes | `learner_update_metrics.csv`: `buffer/replay_size`, `buffer/demo_size` |
| Episode success | `actor_stats.csv`: `environment/succeed` |
| Interventions per episode | `actor_stats.csv`: `environment/episode/intervention_count` |
| Critic health | `learner_update_metrics.csv`: `critic/critic_loss`, `critic/predicted_qs` |
| Actor health | `learner_update_metrics.csv`: `actor/entropy`, `actor/temperature` |

# Operator observations

Everything above this line is distilled from the HIL-SERL paper and walkthrough. Everything in
this section is **not** HIL-SERL canon. It is a collection of patterns observed across our own
training runs (6-DoF insertion tasks in simulation) that extend, sharpen, or reconcile the
published guidance. Concrete numbers are from those runs, yours will differ.

## 1. Don't leave training unsupervised near the peak

**The pattern.** Once the policy reaches high success with near-zero interventions, it is tempting
to step away and let training continue unsupervised. In one run, that choice triggered a
regression even though reset randomization was already disabled: the run was at 100% success with
zero interventions around learner step 7000, degraded to 96% by 8000, and had regressed to 82%
with visibly longer episodes by 9000. Recovery took roughly 5000 learner steps of occasional
re-engaged interventions (0.2-0.5 per episode) before the run stabilized at 96-100% with no
further help needed.

**What to take away:**

- A policy at 100% success with zero interventions is **not** a safe state to walk away from
  indefinitely. It can drift while unsupervised.
- When you re-engage after a drift, you don't need heavy Phase-1-style interventions. Occasional
  corrections were enough to pull performance back.

**How this relates to the walkthrough.** The Phase 2 guidance is to *"mostly hold off on
intervening unless the policy is repeatedly making the same mistake."* In that run the policy
*was* repeatedly making the same mistake, the operator just wasn't in the room to see it. The
walkthrough assumes a human in the loop across the whole run. Leaving the seat is not explicitly
discussed.

## 2. Evaluate separately, the training metrics are not ground truth

**The pattern.** Training-time success on the actor curve is **systematically upward-biased**
relative to what the policy actually does at eval time. This is visible even when the training
curves look pristine and healthy by every metric in the "Signals to watch" table above.

Two biases inflate the training-time number:

- **Intervention-assisted successes count as successes.** Even with light supervision (0.1-0.5
  interventions per episode), many "successful" episodes ended that way *because the human nudged
  the policy back on track*. At eval the human is gone.
- **The intervention-free subset isn't random.** The operator, consciously or not, skips
  intervention when the policy looks fine early. So "success rate among intervention-free
  episodes" is conditional on "policy appeared to be doing OK", a self-selected subset of the
  easier rollouts.

**Example from one of our runs:** learner metrics at checkpoints 15k and 20k were both pristine
(entropy at target, temperature at equilibrium, critic loss at its floor, Q plateaued). The actor
success curve showed ~95% overall and ~100% on intervention-free episodes at both checkpoints.
Eval told a different story: 49/50 successes at 15k, but only 35/50 at 20k. The 20k policy had
drifted toward one specific failure mode (approaching next to the goal instead of into it) that
was essentially invisible on every training-side metric, but dominated the eval failures. Relying
on training metrics alone would have shipped 20k as the "later and therefore better" checkpoint.

**Practical protocol:**

- **Save checkpoints often** (a `checkpoint_period` of 2500 learner steps is a good default).
- **Eval candidate checkpoints on 50+ trials** before deciding which one to ship. 20 trials is too
  small to distinguish a 98% policy from a 70% one with any confidence.
- **Characterize failures**, not just the success rate. If the failures cluster in a specific
  region of state space, that is gold: it tells you exactly where to apply Observation 3 below.
- **Don't assume later is better.** In our runs, the earliest checkpoint with sustained success
  often outperformed later ones, even when the later ones looked equally healthy on training
  metrics.

**How this relates to the walkthrough.** The walkthrough mentions evaluation only briefly (a
single eval run via `--eval_checkpoint_step`). It does not discuss using eval as a
*development-time* signal for driving training decisions, which is what this observation argues
for.

## 3. Phase 3 fixes clustered failure modes efficiently

Once Observation 2 has revealed a clustered failure mode, the walkthrough's Phase 3 protocol
(*"use interventions to cause the policy to make this mistake and let it practice recovering"*)
can target that mode with a small number of induced resets. From one session where we verified
this explicitly:

**Starting state:** a checkpoint at 70% eval success (35/50), with all 14 failures clustered in
one mode (approaching next to the goal instead of into it).

**Protocol used.** Resume training from that checkpoint. On roughly every 3rd to 5th reset, take
manual control, drive the system into the problematic state, then hand control back so the policy
has to recover. Over ~5000 additional learner steps this amounted to roughly 20 induced Phase 3
starts in total, far fewer than initially guessed.

**Results:** 2500 steps later the eval was at 98% (49/50) with the target failure mode essentially
eliminated. Another 2500 steps later it was at 94% but with ~30% faster, visibly smoother
execution, plus one new failure mode that Phase 3 had surfaced.

**What this suggests:**

- Phase 3 can fix a cleanly identified, clustered failure mode in a surprisingly small number of
  induced resets.
- Improvement is detectable at save-and-eval intervals of 2500 learner steps. No need to run
  Phase 3 for hours before checking.
- The policy goes through a qualitative behavior change during Phase 3: first conservative and
  recovery-aware (longer episodes, visible "almost fail but recover" patterns), later the recovery
  knowledge is integrated (smoother, faster).
- Phase 3 can introduce or reveal **other** failure modes. Watch eval for new clusters, not just
  the one you targeted.
- A Phase-3-rehabilitated late checkpoint does not necessarily beat the pre-drift early checkpoint
  on raw success rate, but it can win on other metrics such as cycle time. Keep both in your eval
  candidate set.

**How this relates to the walkthrough.** The walkthrough describes Phase 3 as moving a 97% policy
to 100%. The observation here is slightly different: Phase 3 can also be useful much earlier, at
70%, as long as the remaining failures are clean and clustered. The 97% figure is a rule of thumb,
not a prerequisite.

## 4. What a healthy run looks like (short form)

For quick reference during a live run (plateau values are from our insertion runs, yours will
differ):

- **`actor/entropy`** settles at the target entropy. With the default
  `target_entropy = -action_dim / 2` that is -3 for a 6-DoF action. Sitting at the target is the
  auto-tuner at equilibrium, *not* a sign the policy has collapsed.
- **`actor/temperature`** settles at the equilibrium pressure needed to hold entropy at target.
  The magnitude is task-specific, only the trend matters.
- **`critic/predicted_qs`** climbs during convergence and plateaus. A rise *past* the plateau
  while actor success falls is a divergence fingerprint. A rise that flattens as episode length
  flattens is healthy.
- **`critic/critic_loss`** decays from its initial spike to a low noisy floor and stays there.
- **Actor success** climbs and saturates, and intervention-free success catches up to overall
  success as supervision tapers.
- **Episode length** drops to the task's natural minimum.

All learner-side metrics reach their plateau at roughly the same learner step. That simultaneous
plateau is the healthy-equilibrium fingerprint.

## Related documents

- [robot_walkthrough.md](robot_walkthrough.md): the training pipeline this guide assumes.
- [The upstream Franka walkthrough](https://github.com/rail-berkeley/hil-serl/blob/main/docs/franka_walkthrough.md):
  the original source of the quoted protocol.
- [HIL-SERL paper](https://arxiv.org/abs/2410.21845).
