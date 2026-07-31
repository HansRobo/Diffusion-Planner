# rlvr.posttrain

Batch post-training toolchain for the T4 planner. One JSON spec per arm, one
command to plan or launch, and objectives/rewards are plugins.

```bash
python -m rlvr.posttrain objectives          # awr | dpo | grpo
python -m rlvr.posttrain presets             # bundled arms
python -m rlvr.posttrain show   dual         # spec after inheritance + sweep
python -m rlvr.posttrain plan   dual         # exact command line, preflight, no launch
python -m rlvr.posttrain run    dual         # waits for free GPUs, then launches
python -m rlvr.posttrain run    sweep_anchor_weight   # 6 arms, back to back
```

`run` takes `--now` (skip the GPU wait), `--detach`, `--log_root`, and two switches for
runs too long to babysit:

```bash
python -m rlvr.posttrain run dual --supervise            # relaunch until the target epoch
python -m rlvr.posttrain run dual --after <path>         # start once that path exists
```

`--supervise` blocks on each attempt, and when one exits before the arm's `epochs` it
relaunches the arm against its **own** newest checkpoint — new `model_path`, new
`start_epoch`, both resume switches on. A node reboot, a preempted job or a transient
CUDA fault is a resume, not a decision. An attempt that comes back having finished no
epoch is a real failure repeating, so identical stalls are counted and it stops rather
than burning GPUs in a crash loop — with a growing pause between them, because a stall
before the trainer even starts is usually something still landing rather than a verdict.
Detach the supervisor itself, not the attempts.

`--after` waits for a path before starting, which is how one arm follows the artifact
another arm produces without either knowing the timestamped directory the other picked.
It resolves the specs it was given *before* it waits, so the presets are read at launch:
edit an arm the supervisor is already parked on and restart it, or it will run the version
it started with.

## Methods

An arm is a spec file; every arm runs the same way, `run <arm>`. Two things vary
independently and it is worth keeping them apart:

- **`objective`** — how a group's rewards become per-candidate weights.
- **`entrypoint`** — which trainer runs, default `rlvr.train_awr`.

Anchors, guidance and reward overrides are neither: they are flags, and they compose with
any objective. The bundled `dual` and `h1` arms are both `objective: awr` and differ only
in their anchor flags, not in method — see **Bundled arms** below.

They do not compose as equal halves at the first waypoint, though, which is worth knowing
before swapping either one. An anchor is a target column appended to the group holding the
logged trajectory, carrying `expert_anchor_weight` as written
(`_inject_expert_anchor_batch`; with `expert_anchor_replace_worst` it overwrites the
weakest column instead of appending). The objective never sees that column and cannot
scale it, and since every column is scored as a mean over its own horizon, the weight a
column spends *per step* is its weight over its horizon: a horizon-1 anchor puts all of it
on the one step the vehicle executes, while a candidate — and a full-horizon anchor —
spreads the same weight over the whole plan. So the anchor flags are what move the executed
waypoint, and swapping `awr` for `grpo` or `dpo` re-weights the candidates around them.
Anchor weights are therefore read as multiples of an average candidate weight, not as
absolute numbers: the objective normalises its own columns to mean `|w| = 1` and the anchor
column is appended after that.

**Candidate-weighting objectives** run inside `rlvr.train_awr`. It reduces
`(per_candidate_loss * weights).sum()` over a scene group, so an objective is just a rule
for turning that group's cached rewards into per-candidate weights. No sampling, no
reference pass, no critic — the rewards are already on disk in the replay buffer, which is
why these are cheap.

| `objective` | rule | `awr_beta` | `positive_advantage_only` clamps against |
|---|---|---|---|
| `awr` | `w = exp(beta * (R - mean) / std)`, group-normalised | temperature: higher is greedier | candidate 0, the deterministic sample |
| `grpo` | `w = beta * (R - mean) / std`, signed, clipped at 20 | advantage scale | the group mean |
| `dpo` | `+1` on the best candidate, `-beta` on the worst | weight of the rejected term | the chosen candidate must beat candidate 0 |

The flag name is shared but its meaning is not, and neither is the reference point
`positive_advantage_only` measures against — read the row, not the flag. All three
rescale a group to mean `|w| = 1` (`normalize`, on by default), so a group that is nearly
all zeros still contributes at full weight; that is a property of the group, not of the
learning rate.

**Full methods** bring their own optimization loop, and are separate trainers because they
need the *current* policy's log-probs at every step, which a weight computed from cached
rewards cannot express:

| arm | trainer | what it adds |
|---|---|---|
| `grpo_full` | `rlvr.train_grpo` | on-policy rollouts, KL to a frozen reference, and with `inner_epochs > 1` a PPO-clipped importance ratio for rollout reuse; behaviour set by `rlvr/configs/grpo_*.json` |
| `dpo_full` | `preference_optimization.train_dpo` | the log-ratio preference objective against a frozen reference, plus optional LoRA |

```bash
python -m rlvr.posttrain plan grpo_full          # same command for either kind
python -m rlvr.posttrain run  dpo_full
```

So: the objectives are the cheap reduction of these methods at the weighting seam, and the
full trainers are the methods themselves. Pick the reduction when you want to sweep a
weighting rule against a mined buffer; pick the full trainer when the ratio, the KL term or
the reference pass is the point. **PPO has no arm** — its clipped ratio lives inside
`rlvr.train_grpo` as the rollout-reuse mechanism, not as a standalone method.

Which trainer an arm uses is one field, `entrypoint`, and it defaults to `rlvr.train_awr`.
Any module exposing `build_parser()` can be one; its flags are then validated the same way.

## Bundled arms

`base.json` carries the shared method and pins one expert anchor on the **first waypoint**
(`expert_anchor_loss_horizon: 1`) — the only step the vehicle executes, and the one a
plan-wide loss barely reaches. The arms differ from there:

| arm | what it is |
|---|---|
| `mine` | fills the replay buffer every weighting arm reads. Run this first, and again whenever the base checkpoint changes |
| `mine_shared` | a re-mine of the **same** base that links the previous buffer's decoder context instead of rebuilding it |
| `h1` | that step-1 anchor alone |
| `dual` | step-1 anchor **plus** a second anchor over the full plan horizon; the step-1 anchor closes executed error, the full-horizon slot pays back the lane cost it accrues |
| `grpo`, `dpo` | `dual`'s configuration with the objective swapped, so the weighting rule is the only difference |
| `probe` | a few epochs from a frozen checkpoint on a tiny valid set — parent of the sensitivity sweeps, and what to run first on a new machine |
| `sweep_anchor_weight`, `sweep_reward` | anchor weights / reward shaping, from `dual` |
| `sensitivity_*` | one knob each (objective temperature, candidate construction, solver steps), from `probe` |

`show <arm>` prints any of them fully resolved, which is the cheapest way to see what an
arm actually sets. Copy one and edit rather than starting from scratch; `extends` and
`null`-to-delete mean an arm should be a handful of lines.

## Point it at your data

`presets/site.json` is the only file you edit — the dataset lists, the base checkpoint's
`model_args.json`, and where your mined replay buffer is. Every other preset inherits from it and
carries method only, so arms are portable between machines.

```
runs/base_model/model.pth            your starting checkpoint
runs/base_model/model_args.json      its args
runs/reference_model/                the frozen PlannerRFT reference (model + args)
runs/mine/<stamp>_mine/replay_buffer the mined replay buffer the weighting arms read
runs/shared_cache                    an earlier buffer to link context from (mine_shared only)
runs/<arm>/                          where each arm writes
datasets/                            symlink to your dataset root (git-ignored)
```

Every trainer run gets its own timestamped directory, which is why `site.json` globs for
the buffer (`{"newest": "runs/mine/*/replay_buffer"}`) rather than naming it: the arms that
read it cannot know the stamp the mine will pick, and the alternative is a symlink somebody
has to remember to make between two runs that may be days apart. Name a path instead if you
keep several buffers around. A run never overwrites a previous one, and preflight tells you
if the buffer is not where the arm expects.

`runs/` and `datasets/` are git-ignored: stage or symlink your own, and nothing about your machine
ends up in the repo. W&B is off by default — set `wandb`, `wandb_project` and `wandb_entity` in
your own preset if you want it.

## Gotchas that cost GPU time

- Resume from the **last** epoch checkpoint, never a "best" by any proxy — the two dynamic values
  `latest_epoch` / `resume_after_latest_epoch` do that for you.
- The buffer belongs to the checkpoint it was mined from. Change the base and every candidate in
  it, and the frozen encoder's output with it, describes a model you are no longer training — so
  re-mine. Budget for it: the encoder output dominates the buffer and scales with your corpus, in
  terabytes rather than gigabytes, over hours rather than minutes. Never delete a finished buffer
  to reclaim disk; the mine takes no optimizer step, so reproducing one costs the full mine again.
- `hdp_rollout_interval` is a budget, not a detail: every epoch where `(epoch - 1)` is a
  multiple of it re-samples the whole corpus instead of optimising, which costs a full mine of
  wall-clock during which no optimizer step is taken, and a second buffer's worth of disk that
  lives until the run ends. It re-samples the *same* scenes, so what a refresh buys is fresher
  candidate trajectories, not more data, and it does nothing for the anchors, whose target is
  the logged human. Set it above the epoch count to spend the whole budget on gradient epochs.
  Lowering it is not a free knob: a refresh epoch puts the arm back on the mining path, which
  refuses the candidate weighting an arm normally applies (next bullet), so preflight rejects the
  pair instead of letting it abort every rank once that epoch arrives. To sample fresher
  candidates, mine again and resume from the new buffer. It may never be `0` either — that
  switches the disk replay backend off entirely rather than switching refreshes off.
- Mining and replay want *opposite* candidate weighting, and the trainer refuses the combination
  rather than picking for you: guided enrichment scores candidates group-relative, so the
  candidate-0 behaviour-anchor flags cannot be set while mining. Cache rewards, not weights, and
  one buffer then serves any weighting rule you replay it with. This is why `mine` drops those
  four flags instead of setting them.
- The deliverable is the **EMA** weights; ONNX export needs `--use_ema`.
- Get a same-weight noise floor before believing any A/B delta, and re-derive it on whatever
  subset you are reading — a smaller subset does not mean a smaller floor. Re-running the same
  checkpoint twice is the cheapest way to get one.
- A new objective's diagnostics must carry the same keys on every call (see below): a
  rank-dependent key set desynchronises the DDP reduce and the epoch never finishes.
- `--nproc_per_node` changes the effective batch and the optimizer steps per epoch, so it is not
  a free knob when you are comparing arms.

## Spec

```json
{
  "name": "dual",
  "extends": "base.json",
  "objective": "awr",
  "entrypoint": "rlvr.train_awr",
  "reward":  {"low_speed_steer_penalty": 2.0},
  "launch":  {"nproc": 8, "env": {"OMP_NUM_THREADS": "1"}},
  "train":   {"expert_anchor_weight": 0.4, "config": "rlvr/configs/arm.json"},
  "prepare": [{"creates": "outputs/overlay_b{awr_beta}", "argv": ["...", "{creates}"]}],
  "sweep":   {"expert_anchor_weight": [0.2, 0.4], "reward.hdp_lane_weight": [0, 1]}
}
```

- `train` keys are the trainer's flags without the dashes, validated against its own
  `build_parser()` — a typo fails in a second, not after epoch 0.
- `objective` selects a candidate-weighting plugin; it is ignored by a trainer that
  does not take `--awr_objective`.
- `entrypoint` is the trainer module, default `rlvr.train_awr`. `launch.launcher:
  "direct"` skips torchrun for a trainer that never initialises a process group.
- `reward` keys are `RewardConfig` fields, passed through `--reward_override`.
- `extends` is chainable and relative; `null` deletes an inherited key.
- `sweep` expands to the cartesian product and gives each arm its own
  `output_dir` / `exp_name` / `wandb_run_name`.
- Paths are project-relative and resolve against the repo root, so a spec is not
  tied to the host that wrote it. Put datasets outside the tree behind a
  symlink (`datasets/`, git-ignored).
- `prepare` runs before the arm, from the repo root. `{flag}` and `{name}`
  interpolate this arm's own values — that is how a swept knob that owns an
  artifact gets one artifact per arm — and `creates` skips a step whose output
  is already there. Reference the flag that consumes it (`{resume_replay_root}`)
  so the path has one definition.
- Three dynamic values, so one arm can name another's output before it exists:
  `{"latest_epoch": "<run dir>"}` and `{"resume_after_latest_epoch": "<run dir>"}` keep a
  resume from drifting from the checkpoint, and `{"newest": "<glob>"}` is the newest path
  matching a pattern. All three read as a missing artifact, not a broken spec, until the
  arm they depend on has run.

## Adding an objective

The trainer reduces `(per_candidate_loss * weights).sum()` over a scene group,
so an objective is a function from that group's rewards to per-candidate
weights. Register one and it is selectable; nothing else changes.

"Nothing else" includes the anchors: an objective governs the candidate
columns only, and the expert-anchor columns are appended after it at their
configured weight (**Methods**). A new rule is measured alongside them, not in
place of them, so compare two objectives at fixed anchor flags.

```python
from rlvr.posttrain.objectives import objective, diagnostics

@objective("my_rule")
def my_weights(rewards, beta=1.0, candidate_valid_mask=None, **_):
    ...
    return torch.from_numpy(weights), diagnostics(...)
```

```bash
python -m rlvr.train_awr --awr_objective my_rule ...
```

Diagnostics must carry the same keys on every call, empty groups included —
they are reduced across DDP ranks and a rank-dependent key set desynchronises
the reduce. `diagnostics()` gives you that block; the test in
`tests/test_objectives.py` pins it.

Shipped: `awr` (`rlvr/awr.py:compute_awr_weights`, exponentiated group-normalised
advantage), `grpo` (signed group-relative advantage), `dpo` (pairwise
best/worst preference, no reference pass) — see **Methods** above for what each
one does and how it differs from the full trainer of the same name.

What does *not* fit here: anything needing the current policy's log-probs, an old
policy, or a critic — an importance ratio, a KL penalty, a value baseline. The
weights are a pure function of the cached rewards, so they are constant with
respect to the parameters being optimised. Those methods get their own
`entrypoint` instead.

## Preflight

`plan` and `run` check what the flags imply before taking a GPU: every path flag
exists, every `rank_XXXX` of the replay overlay is present,
`--expert_anchor_safe_step_scoped` has its `expert_hard_gate_step.npy` registered
in the manifest, `--resume_optimizer_state` has an optimizer in the checkpoint,
and, when the run is a continuation, that `--start_epoch` actually continues it.

That last one is gated on `--use_policy_state` on purpose. A run that starts from
someone else's checkpoint numbers its own epochs from scratch, so the epoch that
checkpoint happens to have been saved at says nothing about this run — and it is the
same switch that decides whether there is a live policy state to continue at all.
