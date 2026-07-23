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
- **Quintic-ramp trajectory augmentation.** That ramp is specifically a fix for
  original-DP's absolute x-start first-frame continuity. HDP uses velocity
  (per-step displacement) actions; a constant waypoint translation becomes a
  first-delta impulse in this representation, so the transform cannot be copied.
  The first-waypoint gate covers the corresponding failure mode defensively.
- **Survival reward** — broken upstream (ranked longer post-crash paths highest);
  a faithful reimplementation recovered 0.53% of the corpus. Not implemented.
- **Learned exploration policy / PPO / critic** — the open-loop data cannot supply
  honest closed-loop returns; upstream deprecated these.
- **K/beta/noise-scale transplants.** Upstream used `K=10, beta=1.0, noise 0.5`
  on original-DP latents. HDP's `G=8, beta=0.5, noise 1.5` was tuned on HDP's own
  velocity-latent geometry; scales must not be copied across representations
  without a sweep. All remain CLI-sweepable.

## Recommended first RL experiment after the Base80 -> SFT chain

Same SFT checkpoint, frozen `rl_eval_*` selection, one arm per change:

1. Baseline: historical `weighted_sum` objective (gate on, deterministic selection).
2. `rl_reward_aggregation=gated_product`.
3. (2) + `rl_reward_horizon_steps=40` (candidate loss follows automatically).
4. (3) + `rl_diffusion_t_min=0.001 rl_diffusion_t_max=0.2`.
5. Road-border reward sweep `rl_reward_w_road_border in {0, 0.25, 0.5, 1.0}` on the
   best of 1-4 (the SFT objective no longer contains a road-border term).

Compare the deterministic deployment reward plus DAC/EPDMS, border distance, lane
keeping, progress, comfort, collision and red-light metrics; a higher training
reward alone is not evidence.
