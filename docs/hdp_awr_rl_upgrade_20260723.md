# HDP AWR-RL upgrade from the original-DP post-training audits (2026-07-23)

## Scope

This note records which findings from the original-DP AWR research repository
(`Diffusion-Planner-t4-main`, branch `feature/original-dp-awr-post-training`) were
ported into the HDP-RL pipeline, which were deliberately not ported, and why. The
source evidence is that repository's `docs/awr_zero_risk_improvement_audit_20260718.md`,
`docs/plannerrft_guided_exploration_for_original_dp_awr.md`,
`docs/hdp_awr_ema_policy_commit_semantics.md`, and
`docs/hdp_augmentation_dp_representation_pitfall.md`.

## Why this is an upgrade, not a rewrite

The existing HDP-RL path already implements the AWR core that the original-DP work
converged on: group-relative `exp(beta * z-score)` weighting with tied/non-finite
group discarding, rollouts from a frozen EMA behavior policy, a once-per-epoch
conservative policy commit (`rl_ema_update_rate=0.05` interpolation with optimizer
reset), encoder-once candidate expansion, and a frozen `rl_eval_*` held-out reward
with source-relative selection guards. Those mechanisms match the audited winning
profile's semantics and stay unchanged.

## Ported (defaults preserve the historical objective unless noted)

1. **First-waypoint candidate gate** — removed 2026-07-29. Not in the code.
2. **Gate-aware advantage statistics** (`compute_reward_weights(candidate_valid_mask=...)`).
   Vetoed candidates are excluded from the group mean/std, and a group needs at
   least two surviving candidates to carry preference signal.
3. **PDM-style bounded objective option** (`rl_reward_aggregation=gated_product`).
   Continuous multiplicative collision/red-light/border gates times a normalized
   risk/follow/lane/progress quality mix, bounded [0, 1]. This mirrors the
   `hdp_pdm` gate reward behind both audited positive full-corpus results
   (deployment reward `0.933058 -> 0.933557`, and the PlannerRFT arm
   `0.933403 -> 0.933935`). Default remains `weighted_sum` (the historical
   objective); held-out selection has independent frozen `rl_eval_*` counterparts.
4. **Prefix scoring and matched prefix regression** (`rl_reward_horizon_steps`).
   The original-DP profile scores a 4 s prefix (40 steps) of the 8 s plan; its
   ablations showed that regressing the full horizon of prefix-scored candidates is
   significantly negative, so the candidate regression horizon always follows the
   scoring horizon. The expert/BC anchor keeps the full horizon.
5. **Active-groups-only expert anchor** (`rl_bc_active_groups_only`, default on;
   only relevant when `rl_bc_weight > 0`). A broad every-scene anchor was
   significantly negative upstream; anchoring only scenes with an active reward
   group preserved the gains.
6. **Deterministic deployment-aligned selection** (`rl_eval_deterministic`,
   default on — an intentional behavior change). The deployed planner executes one
   zero-noise plan; the original-DP
   pipeline selected and deployed deterministically while treating stochastic
   K-sample metrics as diagnostics. Validation now also scores that exact
   trajectory per scene (`deterministic_mean`) and checkpoint selection uses it.
   The K-sample metrics and all source guards are unchanged.

## Deliberately not ported

- **Disk rollout cache / mine-once-replay-nine state machine** (~10k lines,
  `train_awr.py` + `_DiskReplayWriter/Reader` + context codecs). It is entangled
  with original-DP's absolute x-start contract, and HDP-RL's
  `rl_updates_per_rollout` already amortizes rollout cost at batch granularity
  with the same frozen-behavior-policy semantics inside an epoch. Revisit only if
  rollout time dominates at full-corpus scale.
- **PlannerRFT guided exploration** (frozen reference, Beta-stratified offsets,
  native∪guided union ranking). Its verdict at 5.45M-scene scale is still pending
  upstream (the `plannerrft_gatefix_clean` campaign was still mining as of
  2026-07-23); the projection/internalization question is unresolved. Port only
  after that campaign reports.
- **Ramped / stretched variants of the candidate augmentation.** Only the
  release's own transform is implemented (see the augmentation section below); the
  onset ramp and the PlannerRFT speed stretch were repo inventions with no source
  and were removed.
- **Survival reward** — broken upstream (ranked longer post-crash paths highest);
  a faithful reimplementation recovered 0.53% of the corpus. Not implemented.
- **Learned exploration policy / PPO / critic** — the open-loop data cannot supply
  honest closed-loop returns; upstream deprecated these.
- **K/beta/noise-scale transplants.** Upstream used `K=10, beta=1.0, noise 0.5`
  on original-DP latents. HDP's `G=8, beta=0.5, noise 1.5` was tuned on HDP's own
  velocity-latent geometry; scales must not be copied across representations
  without a sweep. All remain CLI-sweepable.

## Rollout-candidate augmentation (added same day)

The first version of this note missed that HDP-RL had **no exploration augmentation
at all**: candidates came only from diffusion prior noise, so AWR could only reweight
the policy's own support (self-distillation). HDP's released RL does perturb rollout
candidates (`augment_trajectory_batch`, `scoring.py:131`) for `current_epoch < 5`,
before both scoring and the replay-buffer write, so the reward and the regression
consume the same perturbed candidates.

`augment_rollout_candidates` (`hdp_rl_utils.py`) reproduces that transform verbatim:
one route-frame offset per candidate, `a, b ~ N(0, std)` of shape `(B, 1)`, held
constant over every timestep, headings untouched.
`--rl_candidate_aug_epochs` / `--rl_candidate_aug_std` are the two knobs (release
values 5 / 0.5; 0 epochs disables).

Off by default. The release emits 8 waypoints at 2 Hz and re-simulates them through
a `PDMSimulator`, so 0.5 m is ~20% of its first waypoint; we emit 80 at 10 Hz and the
first is executed directly (measured 525 mm), so 0.5 m is ~95% of it, and under
velocity actions a rigid translation lands entirely in that one step. The paper never
mentions augmentation and `ap:implementation` names its RL devices exhaustively
without it. Full evidence and the measured multimodality baseline:
`docs/hdp_rl_augmentation_multimodality_evidence_20260729.md`.


## Recommended first RL experiment after the Base80 -> SFT chain

Same SFT checkpoint, frozen `rl_eval_*` selection, one arm per change:

1. Baseline: historical `weighted_sum` objective (gate on, deterministic selection).
2. **Ported objective arm:** `rl_reward_source=pdm_port
   rl_reward_horizon_steps=40 rl_reward_beta=1.0` (beta acts on dimensionless
   group z-scores, so the source value transfers directly; our 0.5 default is a
   local choice that halves weight contrast vs public HDP). The verbatim port of
   the objective behind both audited positive results; on the 512-scene real-data audit
   (`docs/rl_threshold_audit_20260723.md`) it is far more expert-consistent than
   the native reward (expert wins its group 91% vs 64% — the native progress term
   rewards overtaking the expert endpoint, the pdm EP ratio is capped at 1).
3. Native alternative: `rl_reward_aggregation=gated_product` +
   `rl_reward_horizon_steps=40`, if (2) underperforms the native baseline.
4. Road-border reward sweep `rl_reward_w_road_border in {0, 0.25, 0.5, 1.0}` on the
   best of 1-3 (the SFT objective no longer contains a road-border term; note the
   pdm_port objective already hard-gates border crossings).

Compare the deterministic deployment reward plus DAC/EPDMS, border distance, lane
keeping, progress, comfort, collision and red-light metrics; a higher training
reward alone is not evidence.
