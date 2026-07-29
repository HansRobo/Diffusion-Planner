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

1. **First-waypoint candidate gate** (`rl_first_waypoint_gate`, default on).
   > **REMOVED 2026-07-29.** This item did not survive. The 5 cm tangent floor
   > described below sits *above* the standstill regime it polices (stop-turn
   > first-step p95 is 0.039 m), so the gate rejected nothing across four cycles,
   > and it was superseded upstream on 2026-07-25. It was never enabled in any HDP
   > run (32/32 `args.json` False) while defaulting to True, which advertised a
   > safeguard that was not there. Deleted, with the measured replacement criterion
   > recorded for a future cycle in
   > `docs/hdp_rl_first_waypoint_gate_removal_20260729.md`.

   Low-speed candidates whose first waypoint jumps from the current pose are
   excluded from both the advantage statistics and the weights. Includes the
   mandatory 5 cm tangent floor (`rl_first_waypoint_gate_tangent_min_step_m`):
   without it, a numerically-zero standstill step reads as 90° off-tangent and
   entire low-speed groups are silently discarded (the audited original-DP defect,
   fixed 2026-07-23 upstream; 6.86M of 6.99M rejections were <5 cm steps and the
   expert GT itself failed in 35% of low-speed scenes). Gate-aware weighting also
   fixes the overlay-resurrection class of bug structurally: weights and validity
   are computed together from the same mask.
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
4. **Prefix scoring and matched prefix regression** (`rl_reward_horizon_steps`,
   `rl_candidate_loss_horizon`). The original-DP profile scores a 4 s prefix (40
   steps) of the 8 s plan; its ablations showed that regressing the full horizon
   of prefix-scored candidates is significantly negative, so the candidate loss
   horizon follows the scoring horizon by default and may never exceed it. The
   expert/BC anchor keeps the full horizon.
5. **Restricted diffusion-time draw** (`rl_diffusion_t_min/max`). The original-DP
   ablation preferred `[0.001, 0.2]` over the full range for the reweighted
   regression. Defaults keep the historical `[eps, 1]` distribution exactly.
6. **Active-groups-only expert anchor** (`rl_bc_active_groups_only`, default on;
   only relevant when `rl_bc_weight > 0`). A broad every-scene anchor was
   significantly negative upstream; anchoring only scenes with an active reward
   group preserved the gains.
7. **Deterministic deployment-aligned selection** (`rl_eval_deterministic`,
   `rl_selection_metric=deterministic`, both default on — an intentional behavior
   change). The deployed planner executes one zero-noise plan; the original-DP
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
- ~~Quintic-ramp trajectory augmentation~~ — **correction (same day):** the first
  version of this note wrongly classified the ramped augmentation as
  DP-representation-specific and excluded it. The opposite is true, and the
  transform is now ported (see the augmentation section below). What remains
  excluded is only the *unramped* released constant offset, which is the audited
  first-delta-impulse failure under velocity actions.
- **Survival reward** — broken upstream (ranked longer post-crash paths highest);
  a faithful reimplementation recovered 0.53% of the corpus. Not implemented.
- **Learned exploration policy / PPO / critic** — the open-loop data cannot supply
  honest closed-loop returns; upstream deprecated these.
- **K/beta/noise-scale transplants.** Upstream used `K=10, beta=1.0, noise 0.5`
  on original-DP latents. HDP's `G=8, beta=0.5, noise 1.5` was tuned on HDP's own
  velocity-latent geometry; scales must not be copied across representations
  without a sweep. All remain CLI-sweepable.

## Rollout-candidate augmentation: the representation analysis (added same day)

The first version of this note missed that HDP-RL had **no exploration
augmentation at all**: candidates came only from diffusion prior noise, so AWR
could only reweight the policy's own support (self-distillation). HDP's own RL
includes rollout trajectory augmentation (`augment_trajectory_batch`) as part of
the algorithm, and both audited positive original-DP results used a form of it
(Gaussian offsets + quintic ramp in the first cycle; PlannerRFT-guided candidates
in the second).

The representation math, worked through for our velocity (per-step displacement)
actions where the regression target is `v_k = w_k - w_{k-1}`:

- **Public HDP's own order of operations is waypoint-space augmentation**:
  `model_action_to_waypoint` → offset → `waypoint_to_model_action` before the
  loss. Our pipeline natively matches it: `sample_group` returns decoded
  waypoints and the loss re-encodes via `waypoints_to_velocity`.
- **Constant offset `d` (the released transform)**: `v'_0 = v_0 + d`, all other
  deltas unchanged — the entire 0.5 m offset becomes a single 5 m/s first-step
  impulse. Upstream measured exactly this defect on `kinematic_type: diff`
  ("translation cancels to later differences, concentrated in the first delta"),
  and the constant-offset training branch regressed validation reward by
  −0.01096 with −1.95 m path shortening. **Forbidden for us; the code rejects
  `ramp_steps < 1`.**
- **Quintic minimum-jerk onset ramp `r(k)` (20 steps ≈ 2 s)**: the velocity
  target picks up increments `(r(k) - r(k-1)) · d`, bounded by
  `1.875/ramp_steps ≈ 9.4%` of the offset per step — for a 0.5 m offset ≈
  4.7 cm/step ≈ 0.47 m/s peak, about 1σ of our per-step action normalization
  (`ego_velocity` std 0.5/0.25). The onset also starts near zero
  (`r(1) ≈ 0.001`), so ramped candidates pass the first-waypoint gate by
  construction. **The ramp is therefore not a DP-specific fix — velocity actions
  need it strictly more than DP's absolute x-start does.**
- **PlannerRFT candidate stretch (`speed_scale = 1 + λ·η`, λ=0.25)** scales every
  per-step displacement uniformly: in velocity space this is exact, smooth, has
  zero first-step discontinuity, and preserves headings bit-for-bit. Of all the
  PlannerRFT exploration mechanisms, this one fits a velocity model *best* — it
  is the natural longitudinal exploration for progress/red-light/follow rewards.
- **Heading channels stay preserved** (upstream heading-mode audit: preserve
  0.088% kinematic failure vs 23.35% for tangent reconstruction from stochastic
  x/y). For the stretch, preserve is exact; for the ramped offset the tangent
  error is bounded by ≈ atan(0.094·|b| / step length).

Implemented as `augment_rollout_candidates` (`rl_candidate_aug_*` flags), applied
between sampling and reward so the reward, gate, and regression all consume the
augmented candidates consistently. Magnitude distributions: `gaussian`
(N(0, 0.5 m) route-frame offsets — the released HDP transform, positive in the
first audited cycle) and `stratified_beta` (the PlannerRFT fixed explorer:
symmetric Beta with concentration softplus(0)+1 ≈ 1.693, one draw per
equal-probability CDF stratum so a small group cannot miss a maneuver region;
upstream λ_lat=1.0). The first `rl_candidate_aug_keep=1` candidate per group
stays untouched as the on-policy anchor (PlannerRFT keeps native candidate 0),
and near-stationary scenes are skipped (upstream 2.0 m/s guard). Default off —
enabling it is the explicit exploration arm below.

**Still not ported from PlannerRFT** (requires guided denoising and the frozen
reference contract, verdict pending upstream): guidance toward the offset
reference during sampling, the trigger-on-low-native-best policy, native∪guided
union ranking with the ≥0.01 admission gain, and the learned Beta exploration
head (upstream's own safe path is fixed Beta first, learned head only via a
shadow-logged offline comparison).

## Recommended first RL experiment after the Base80 -> SFT chain

Same SFT checkpoint, frozen `rl_eval_*` selection, one arm per change:

1. Baseline: historical `weighted_sum` objective (gate on, deterministic selection).
2. **Exploration arm:** `rl_candidate_aug_prob=1.0` (gaussian offsets, ramp 20, keep 1).
   This is the highest-priority arm: without candidate augmentation AWR only
   reweights the policy's own support, and both audited upstream gains included an
   exploration mechanism. Optional sub-arm: `rl_candidate_aug_stretch=0.25`.
3. **Ported objective arm:** (2) + `rl_reward_source=pdm_port
   rl_reward_horizon_steps=40 rl_reward_beta=1.0` (beta acts on dimensionless
   group z-scores, so the source value transfers directly; our 0.5 default is a
   local choice that halves weight contrast vs public HDP). For the
   `stratified_beta` augmentation scheme, `rl_candidate_aug_std=1.0` matches the
   source lambda_lat. The verbatim port of the objective behind both
   audited positive results; on the 512-scene real-data audit
   (`docs/rl_threshold_audit_20260723.md`) it is far more expert-consistent than
   the native reward (expert wins its group 91% vs 64% — the native progress term
   rewards overtaking the expert endpoint, the pdm EP ratio is capped at 1).
4. Native alternative: (2) + `rl_reward_aggregation=gated_product` +
   `rl_reward_horizon_steps=40`, if (3) underperforms the native baseline.
5. Best of 3-4 + `rl_diffusion_t_min=0.001 rl_diffusion_t_max=0.2`.
6. Road-border reward sweep `rl_reward_w_road_border in {0, 0.25, 0.5, 1.0}` on the
   best of 1-5 (the SFT objective no longer contains a road-border term; note the
   pdm_port objective already hard-gates border crossings).

Compare the deterministic deployment reward plus DAC/EPDMS, border distance, lane
keeping, progress, comfort, collision and red-light metrics; a higher training
reward alone is not evidence.
