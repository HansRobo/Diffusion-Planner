# HDP-RL paper fidelity: `--rl_paper_exact`

`--rl_paper_exact true` runs the RL post-training exactly as published, and refuses
to start otherwise. It pins every field listed below to its cited value, turns off
every real-vehicle extension this repository added on top of the paper, and raises
on any explicit command-line flag that contradicts a published value — so a run
either reproduces HDP-RL or fails loudly. The applied pin table is printed on rank 0
and written to `args.json` as `rl_paper_exact_changes`.

```bash
python train_hdp_rl_predictor.py ... \
  --rl_paper_exact true \
  --rl_paper_reward multi \      # multi = HDP-RL;  single = HDP-RL† (r_safety alone)
  --rl_replay_dir <dir>          # required: the mine/replay epoch state machine
```

## Sources

Everything below is anchored to a file that can be re-read:

| Short name | Path |
| --- | --- |
| `neurips_2026.tex` | `reference/papers/hyper_diffusion_planner_paper/src/neurips_2026.tex` |
| `code.tex` | `reference/papers/hyper_diffusion_planner_paper/src/code.tex` (Algorithm 1, hybrid loss) |
| `code_rl.tex` | `reference/papers/hyper_diffusion_planner_paper/src/code_rl.tex` (Algorithm 2, RL-hybrid loss) |
| `dp_vla_rl_agent.py` | `reference/external/Hyper-Diffusion-Planner/HDP-navsim/hdp_navsim/agent/dp_vla/dp_vla_rl_agent.py` |
| `dp_vla_agent.py` | `reference/external/Hyper-Diffusion-Planner/HDP-navsim/hdp_navsim/agent/dp_vla/dp_vla_agent.py` |
| `dp_vla_rl_agent.yaml` | `reference/external/Hyper-Diffusion-Planner/HDP-navsim/hdp_navsim/config/agent/dp_vla_rl_agent.yaml` |
| `agent_lightning_module.py` | `reference/external/Hyper-Diffusion-Planner/HDP-navsim/hdp_navsim/training/agent_lightning_module.py` |
| `traj_kinematics.py` | `reference/external/Hyper-Diffusion-Planner/HDP-nuplan/hdp_nuplan/utils/traj_kinematics.py` |
| `train_predictor.py` (nuplan) | `reference/external/Hyper-Diffusion-Planner/HDP-nuplan/train_predictor.py` |

Key anchors in `neurips_2026.tex`: `eq:optim` (line 605), `eq:awr` (615),
`eq:awr_hybrid` (622), the RL section text (635–642), Appendix C / MDP (1122),
the hybrid-loss appendix (1278), `ap:implementation` (1353), `app:rewards` (1366),
and Table `tab:param` (1380–1404).

## What the paper's RL actually is

KL-regularized advantage-weighted regression, one epoch-scale iteration at a time:

- Eq. `eq:optim`: `max_pi E[r] - (1/beta) KL(pi^k || pi^{k-1})`.
- Closed form: `pi^{k*} ∝ pi^{k-1} · exp(beta · r)`.
- Eq. `eq:awr`: `L_RL = E[ exp(beta·r) · || tau^{v;k}_theta(tau^v_t, t, C) - tau^v_0 ||^2 ]`,
  where the actions `tau^v_0` in `D` are **drawn from `pi^{k-1}`** — the regression
  target is the sampled candidate, not the human trajectory.
- Eq. `eq:awr_hybrid`: same expression with the hybrid norm `||·||^2_P`, i.e. the
  velocity MSE plus `omega` times the waypoint MSE of the `W`-detached integral.
- `ap:implementation`: reward group normalization, discard of constant-reward
  groups, and EMA for policy updates.

That is the whole objective. There is no expert anchor, no candidate filtering, no
gating, and no shaping term beyond the four rewards of `app:rewards`.

## Pin table

`repo default` is the value a normal (non-paper-exact) run of this repository uses;
`paper-exact` is what the mode pins. Rows where the two agree are still pinned, so
that a future default change cannot silently break a paper-exact run.

### Table 3 (`tab:param`) and Eq. (`eq:awr_hybrid`)

| Field | Repo default | Paper-exact | Source |
| --- | --- | --- | --- |
| `num_generations` | 8 | **32** | `tab:param` RL / Group size |
| `rl_reward_beta` | 0.5 | **1.0** | `tab:param` RL / Temperature β |
| `rl_ema_update_rate` | 0.05 | 0.05 | `tab:param` RL / EMA (decay = 1 − rate = 0.95) |
| `rl_reward_w_risk` | 1.0 | 1.0 | `tab:param` λ_risk |
| `rl_reward_w_follow` | 3.0 | 3.0 | `tab:param` λ_follow |
| `rl_reward_w_lane` | 2.5 | 2.5 | `tab:param` λ_lane |
| `planning_hybrid_loss` (ω) | 0.01 | *inherited from the frozen IL base* | `tab:param` publishes 0.1 — see contradiction 2 and "The one pair not taken from the paper" |
| `hybrid_loss_window` (W) | 10 | *inherited from the frozen IL base* | hybrid-loss appendix publishes W = L−1 — see contradiction 3 and the section below |
| `advantage_eps` | 1e-6 | 1e-6 | `code_rl.tex` `r.std() + 1e-6` |
| `rl_reward_normalize` | `group` | `group` | `ap:implementation`, reward group normalization |
| `rl_init_use_ema` | True | True | Sec. RL, `pi^0` is the hybrid-loss imitation model |

### The reward (`app:rewards`)

| Field | Repo default | Paper-exact (`multi`) | Paper-exact (`single`) | Source |
| --- | --- | --- | --- | --- |
| `rl_reward_w_safety` | 0.0 | 0.0 | **1.0** | "the single-reward baseline uses r_safety alone" |
| `rl_reward_w_risk` | 1.0 | 1.0 | **0.0** | Total Training Reward |
| `rl_reward_w_follow` | 3.0 | 3.0 | **0.0** | Total Training Reward |
| `rl_reward_w_lane` | 2.5 | 2.5 | **0.0** | Total Training Reward |
| `rl_reward_aggregation` | `weighted_sum` | `weighted_sum` | ditto | `r` is a plain weighted sum |
| `rl_behavior_gate` | `safety` | **`none`** | ditto | the published sum has no gate |
| `rl_reward_w_progress` | 3.0 | **0.0** | ditto | no progress term in the paper |
| `rl_reward_w_road_border` | 0.0 | 0.0 | ditto | no road-border term in the paper |
| `rl_red_light_constraint` | True | **False** | ditto | neither r_safety nor r_risk has a traffic-light factor |
| `rl_occupancy_use_road_border` | True | **False** | ditto | OCC is "occupancy distance to static/uncertain regions", not HD-map border geometry |
| `rl_reward_horizon_steps` | 0 (full) | 0 | ditto | rewards are "evaluated … over the planning horizon of L steps" |
| `rl_reward_source` | `native` | `native` | ditto | the published reward set, not the ported original-DP PDM objective |

### The rollout / replay schedule (`dp_vla_rl_agent.yaml`, `dp_vla_rl_agent.py`)

| Field | Repo default | Paper-exact | Source |
| --- | --- | --- | --- |
| `rl_rollout_steps` | 6 | **5** | `rl_config.rollout_steps: 5` |
| `rl_rollout_interval` | 0 (mine every epoch) | **10** | `rl_config.replay_buffer_update_epoch: 10`; the `compute_loss` / `on_train_epoch_start` epoch state machine |
| `rl_updates_per_rollout` | 1 | 1 | `rl_config.diffusion_repeat_size: 1` |
| `rl_noise_scale` | 1.5 | **1.0** | `_rl_rollout` samples the prior at unit temperature |

`--rl_paper_exact` therefore requires `--rl_replay_dir`; a 10-epoch mine interval
without a replay cache would silently train ten epochs on nothing.

### Extensions of this repository that the mode switches off

| Field | Repo default | Paper-exact | Why |
| --- | --- | --- | --- |
| `rl_bc_weight` | 0.0 | 0.0 | `eq:awr_hybrid` is the weighted regression alone |
| `rl_candidate_loss_horizon` | 0 | 0 | the regression covers the whole action `tau^v_0` |
| `rl_diffusion_t_min` / `rl_diffusion_t_max` | 0.0 / 1.0 | 0.0 / 1.0 | expectation over the full `t` range |
| `rl_candidate_aug_prob` | 0.0 | 0.0 | see "deliberate non-verbatim item" below |

`rl_first_waypoint_gate` used to be listed here (repo default True, paper-exact
False). The mechanism was **removed** on 2026-07-29 — it was never enabled in any
run and its implementation was a superseded form measured to reject nothing. See
`docs/hdp_rl_first_waypoint_gate_removal_20260729.md`. Paper-exact mode no longer
has to switch it off, because it no longer exists.

## Three contradictions between the sources

These are properties of the published material, not of this repository. Where the
paper and the released code disagree, paper-exact mode follows the **paper**,
because the `.tex` is the only self-consistent, citable configuration. The single
exception is the hybrid norm (ω, W) of contradictions 2 and 3, which is inherited
from the frozen IL base instead — see "The one pair not taken from the paper".

### 1. `code_rl.tex` Algorithm 2 does not parse

The listing reads, verbatim:

```python
def rl_hybrid_loss(r, beta, pred_v, gt_v, W, omega):
    r_n = (r - r.mean() / (r.std() + 1e-6)
    weight = torch.exp(beta * r_n).detach()
    return weight * hybrid_loss(pred_v, gt_v, W, omega)
```

The second line has an unbalanced parenthesis, and as written it divides only the
mean rather than the centred reward: `r - (r.mean() / (r.std() + 1e-6))`. The
intended expression is the group-relative normalization `(r - mean) / (std + 1e-6)`
that `ap:implementation` cites to deepseekmath and that `dp_vla_rl_agent.py:632-634`
implements:

```python
rewards = (reward_abs - reward_abs.mean(dim=1, keepdim=True)) / (
    reward_abs.std(dim=1, keepdim=True) + 1e-6
)
```

We implement the corrected form. `tests/test_hdp_rl_paper_exact.py` locks
`compute_reward_weights` against a verbatim reimplementation of it.

### 2. ω = 0.1 (paper) vs 0.01 / 0.05 / 0.0 (code)

- `tab:param`: hybrid loss weight **ω = 0.1**.
- `HDP-nuplan/train_predictor.py:77`: `--planning_hybrid_loss` default **0.01**.
- `HDP-navsim/config/agent/dp_vla_agent_hdp.yaml:33`: `hybrid_loss_weight: 0.05`.
- `dp_vla_rl_agent.yaml`: does not set `hybrid_loss_weight` at all, and
  `dp_vla_rl_agent.py:664` falls back to `0.0` — i.e. the released navsim RL config
  runs the RL-hybrid loss with **no waypoint term**, which is precisely the ablation
  baseline the paper argues against in "RL-Hybrid Loss Matters".

This repository's normal default is 0.01 (it follows the released nuplan trainer,
which is the codebase our planner descends from). Paper-exact mode does **not**
pin 0.1: it inherits ω from the IL base it post-trains — see "The one pair not
taken from the paper" below.

### 3. W = L−1 (paper) vs a real axis bug (nuplan) vs W = 1 (navsim)

- Paper, hybrid-loss appendix: "In practice, we set **W = L−1**" — i.e. essentially
  the full gradient, with only the oldest step detached.
- `traj_kinematics.py:13,17` writes, on a tensor documented as `u: (B, T=80, D)`:

  ```python
  shifted[:, :, :detach_window_size] = 0
  ```

  The third index is the **coordinate** axis `D`, not time. With `D ∈ {2,3}` and
  `detach_window_size = 10`, the slice covers the whole axis, so `shifted` becomes
  all-zero, `sum_recent = cum_normal`, `cum_detach_shifted = 0`, and the function
  returns a plain `cumsum`. The released nuplan hybrid loss therefore performs **no
  detaching at all** (effectively W = L), which happens to land near the paper's
  W = L−1.
- The navsim copy at `dp_vla_agent.py:230,234` uses the correct
  `[..., :detach_window_size, :]`, but its default is `detach_window_size = 1`
  (`dp_vla_rl_agent.py:669`), the opposite extreme.

Our `diffusion_planner/loss.py:80` `_detached_integral` uses the correct time-axis
slice, and `tests/test_hdp_rl_paper_exact.py` both checks it against a verbatim port
of the navsim function (values *and* gradients, for W = 1, 3, 7, 8) and guards
against the nuplan axis bug regressing into it. As with ω, paper-exact mode does
not pin W = L−1; it inherits it.

## The one pair not taken from the paper: the hybrid norm (ω, W)

`eq:awr_hybrid` weights the *hybrid* distance, and `code_rl.tex` Algorithm 2
forwards `W, omega` straight into `hybrid_loss()`, so ω and W genuinely belong to
the RL objective and not only to imitation pretraining. That is exactly why
paper-exact mode does not pin them.

**IL is never retrained in this pipeline.** RL post-trains a frozen IL base, and
ω and W are the geometry of the norm that base was fitted in. Our base80
checkpoint was fitted at ω = 0.01, W = 10. Running RL on it at ω = 0.1, W = L−1
rescales the waypoint term 10× and widens the detach window from 10 steps to 79 —
every RL gradient would then be measured in a norm the policy has never been
optimized for. Pinning the published pair would be paper-exact in the table and
not paper-exact in the thing the table describes.

So `--rl_paper_exact` reads ω and W out of the `args.json` beside the checkpoint
in `--init_weights_path` and copies them onto the run:

- the run log and `args.json` record the inherited value **and** the published
  value it departs from, with the citation, so the departure is never silent;
- an explicit `--planning_hybrid_loss` / `--hybrid_loss_window` that contradicts
  the base is **rejected**, with the same citation in the error;
- a base directory with no `args.json` is rejected rather than guessed at.

Which pair is better *for RL* is an empirical question, not a citation question,
and `--rl_hybrid_ablation True` is how it gets asked: it releases both fields and
stamps the run as an ω/W ablation arm in `args.json`, so a sweep reads as a sweep
instead of as a paper-exact run that quietly disagrees with its own base. The
published pair (ω = 0.1, W = L−1) is reachable only through that flag:

```bash
HDP_RL_HYBRID_ABLATION=True \
HDP_RL_PLANNING_HYBRID_LOSS=0.1 HDP_RL_HYBRID_LOSS_WINDOW=79 \
  sbatch diffusion_planner/slurm/run_hdp_rl.sbatch
```

## The corpus and the input perturbation must match the IL base

RL post-training moves the same policy on the same distribution. If the corpus or
the augmentation differs from the base run's, the RL delta is unattributable —
any change could be the objective or could be the data. Paper-exact mode therefore
compares the run against the base's `args.json` and refuses a mismatch:

- corpus: `train_set_list`, `valid_set_list`, `extra_train_set_list` (by basename,
  since the same artifact has a different absolute path in each checkout),
  `extra_train_set_repeat`, `filter_skipped`, `train_subsample_step`,
  `align_legacy_neighbor_futures`;
- perturbation: `use_data_augment`, `augment_type`, `augment_prob`, `num_refine`,
  `ego_past_noise_std`, `use_smoothing_future_trajectory`;
- the normalizer, compared by **resolved content** rather than by path.

`run_hdp_rl.sbatch` defaults every one of these to base train's value, so the
launcher reproduces the base corpus without being told to. Augmentation is applied
before the rollout in both the training path (`hdp_rl_epoch.py:512-519`) and the
mining path (`:757-762`), so the reward, the gate and the regression all see the
same augmented candidate — turning it on does not desynchronise the objective.
`--rl_base_corpus_check False` waives the check for a deliberately different
distribution.

## Two further places the released code departs from its own paper

Neither is a contradiction we had to resolve — in both, our implementation already
follows the paper — but they are worth knowing when comparing runs.

- **Constant-reward groups are not discarded in the released code.**
  `ap:implementation` says "we discard samples in which all actions receive
  identical rewards". `dp_vla_rl_agent.py:660` computes `weights = torch.exp(rewards)`
  unconditionally, so a group whose rewards are all equal contributes at
  `exp(0) = 1` — unweighted self-distillation on exactly the scenes that carry no
  preference signal. `compute_reward_weights` in
  `diffusion_planner/hdp_rl_utils.py:2057` drops those groups.
- **No EMA runs in the released navsim RL.** `tab:param` lists EMA = 0.05 and the
  RL section says "we employ Exponential Moving Average (EMA) for policy updates",
  but the `ModelEma(..., decay=0.999)` block in `agent_lightning_module.py:66-68`
  is commented out. We implement the paper: `decay = 1 − rl_ema_update_rate = 0.95`.

## The one deliberate non-verbatim item

`dp_vla_rl_agent.py:534-535` augments rollout candidates while `current_epoch < 5`
with `augment_trajectory_batch` (`scoring.py:131`), which adds a **constant
along-track / lateral offset** (both drawn at σ = 0.5, `scoring.py:139-140`) to the
whole waypoint sequence. Paper-exact mode leaves
`rl_candidate_aug_prob = 0.0`, i.e. it does not reproduce that augmentation.

Reason: the released agent's action is a waypoint sequence, where a constant offset
is a rigid translation of the trajectory. Our action is the **velocity** sequence
(`use_velocity_representation`), where the same constant offset applied to the action
is a first-step impulse followed by nothing — a physically different perturbation,
and one this repository's `augment_rollout_candidates` explicitly rejects
(`hdp_rl_utils.py:1820` hard-fails on `ramp_steps < 1` for exactly this reason).
Reproducing the *code* here would not reproduce the *perturbation*. The augmentation
is also absent from `neurips_2026.tex`, so switching it off is the paper-faithful
choice; a ramped equivalent remains available outside paper-exact mode.

## What this mode does not change

- **Scale.** `tab:param` IL gives 6 blocks / 256 hidden / 8 heads, which is what this
  repository implements. The released navsim reproduction is a different model
  (`_shared/model.yaml`: 1024 hidden, depth 12, 16 heads, 8 actions) trained on
  navsim with a PDM scorer, and cannot be run on Tier IV data at all.
- **Simulator.** The paper trains against a non-reactive pseudo-closed-loop
  simulator built on real logs (`ap:implementation`); our reward runs on logged
  Tier IV NPZ scenes with the same non-reactive assumption. The reward *definitions*
  are pinned; the underlying scene source is necessarily ours.
- **Checkpoint selection.** Unchanged and non-negotiable: always `latest.pth`
  (see `docs/checkpoint_selection.md`). The RL trainer's `best_model/` bookkeeping
  is internal accept/reject state that never rolls back the live policy, so it does
  not affect the trained trajectory and needs no paper-exact override.
