# HDP-RL Experimental Design

This document describes the current RL experiments only. It is provisional and is not part of
the settled Base/SFT model contract in `docs/hyper_diffusion_planner.md`. RL reward weights,
gates, EMA policy updates, and selection rules may change after real-data experiments.

## Scope and References

The implementation is checked against the local paper and released code first:

```text
reference/hyper_diffusion_planner_paper/src/code_rl.tex
reference/external/Hyper-Diffusion-Planner/HDP-navsim/hdp_navsim/agent/dp_vla/dp_vla_rl_agent.py
reference/external/Hyper-Diffusion-Planner/HDP-navsim/hdp_navsim/agent/dp_vla/model/rl_utils.py
reference/external/Hyper-Diffusion-Planner/HDP-navsim/hdp_navsim/config/agent/dp_vla_rl_agent.yaml
```

The branch adapts the objective to signals present in Tier IV NPZ files. It is not a byte-for-byte
reproduction of the NAVSIM runtime, PDM cache, Ray scorer, or replay infrastructure.

## Current Candidate

The current candidate uses the paper's reward-weighted RL-Hybrid form:

```text
group-normalize(reward)
exp(beta * normalized_reward) * hybrid_diffusion_loss
```

The temporary tuning defaults are:

```text
rl_reward_beta=0.5
rl_reward_w_risk=1.0
rl_reward_w_follow=3.0
rl_reward_w_lane=2.5
rl_reward_w_progress=3.0
rl_reward_w_road_border=0.0
rl_behavior_gate=safety
rl_occupancy_use_road_border=true
rl_stationary_progress_mode=distance
rl_red_light_constraint=true
rl_bc_weight=0.0
num_generations=8
rl_noise_scale=1.5
rl_eval_noise_scale=0.5
rl_eval_num_generations=32
rl_train_scope=decoder
```

These are experiment settings, not a final algorithm decision. Held-out evaluation uses its own
reward-weight and gate flags so ablations cannot silently change the checkpoint-selection metric.

## Reward and Safety Terms

The reward path currently contains:

- SAT collision safety, continuous TTC and THW risk;
- static/stopped-agent occupancy with rear-end attenuation;
- leader-conditioned following, lane-center scoring, and expert lane-change/off-lane masking;
- route progress and red-light stop-line constraints.

The logged expert chooses the leader independently of each candidate. Candidate motion controls the
gap, speed-match, comfort, and progress terms. A safety behavior gate prevents behavior rewards from
compensating for an actual collision; a risk gate is retained as an explicit near-miss ablation.

### Direct Road-Border Experiment

The direct HD-map road-border term is an opt-in real-vehicle extension, disabled by default. With
`rl_reward_w_road_border>0`, the implementation computes exact ego-footprint-to-border clearance
over all 80 predicted steps, maps `road_border_critical_m` to score 0 and
`road_border_safe_m` to score 1, and gates behavior rewards below the critical margin. It is
independent of the weaker road-border occupancy fallback, so disabling that fallback cannot disable
the direct term.

This term is being evaluated specifically because removing the SFT road-border loss coincided with
lower PDMS DAC and trajectories close to or beyond map borders. It is not claimed to be a published
HDP reward term. The experiment must be judged jointly on DAC, EPDMS, continuous border score,
collision/risk, lane/follow, and right-turn behavior.

## Policy Update and Efficiency

- Rollouts are sampled from the EMA shadow without autograd; the live decoder receives the
  reward-weighted hybrid update.
- The EMA shadow remains fixed during an epoch and is committed once at the policy boundary with
  the configured update rate; live weights are synchronized and stale Adam moments are cleared.
- Decoder-only RL freezes the scene encoder and turn-indicator classifier in eval mode.
- Scene encoding is computed once per group. Candidate groups stay intact for reward normalization;
  compiled calls are split only at complete scene boundaries at the configured candidate cap.
- Full stochastic reward/EPDMS validation runs at `rl_full_eval_utd`. `latest.pth` is retained even
  when a candidate fails a source-policy guard.

## Data and Checkpoint Rules

RL starts from the SFT EMA checkpoint with `--init_weights_path`; `--resume_model_path` is only for
continuing the exact same RL run. The production Slurm launcher fingerprints code, manifests,
normalization, checkpoint, and Python/CUDA/NCCL environment before starting distributed training.

By default RL uses the shared precomputed SFT manifests:
`/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_train_sft_is_skipped_filtered.json`
and
`/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_valid_sft_balanced_is_skipped_filtered.json`.
They are already `is_skipped`-filtered, so the launcher does not rescan sidecars. Traffic-light
features are kept unchanged, and manifest paths are not rewritten by the RL loader. Alternate
right-turn or causal manifests must be passed explicitly as an experiment override.

## Selection Guards

The source SFT policy is evaluated before the first update. A replacement must improve the held-out
selection reward while respecting source-relative guards for risk, safety, collision-only safety,
red-light compliance, TTC, THW, occupancy, comfort, collision rates, and EPDMS. When the direct
road-border term and EPDMS are enabled, continuous border reward and binary `valid_epdms_dac` are
also required not to regress beyond `rl_max_valid_epdms_regression`.

Checkpoint selection uses the deterministic deployment reward: validation additionally scores
one zero-noise plan per scene (`deterministic_mean`), which is exactly what the deployed planner
executes. The deterministic selection score uses the run's own training reward,
matching the source repository's train-and-select discipline. The frozen
`rl_eval_*` stochastic metrics are report-only diagnostics for comparing arms; acceptance is
protected by the independent EPDMS/DAC/safety source guards, which are deployment metrics, not
a second reward.

## AWR upgrades from the original-DP post-training audits (2026-07-23)

See `docs/hdp_awr_rl_upgrade_20260723.md` for the evidence record. In brief: a first-waypoint
candidate gate with a mandatory 5 cm tangent floor excludes low-speed standstill-jump candidates
from both the advantage statistics and the weights; `rl_reward_aggregation=gated_product` offers
the PDM-style bounded multiplicative-gate objective; `rl_reward_horizon_steps` scores a prefix with
the candidate regression horizon following it (regressing the unscored tail is a known-negative
configuration and is rejected); and the optional expert anchor applies only to scenes with an
active reward group. Training-objective defaults are unchanged; only checkpoint selection
switched to the deterministic metric.

`--rl_candidate_aug_epochs` / `--rl_candidate_aug_std` reproduce HDP's released
rollout-candidate augmentation verbatim: for the first N epochs, one constant route-frame offset
per candidate (`a, b ~ N(0, std)`, headings untouched), applied before reward and regression so
both consume the same candidates. Release values are 5 and 0.5 m. Off by default
(`epochs=0`): our first waypoint is at 0.1 s and is executed directly, so 0.5 m is ~95% of it,
versus ~20% of the release's 0.5 s waypoint. Guided denoising toward a frozen
reference (full PlannerRFT) remains deferred pending the upstream full-scale verdict.

The DAC guard is deliberate: a higher proxy reward must not be accepted if it worsens the binary
drivable-area-compliance metric that motivated this experiment.

## Faithfulness and Open Questions

Faithful elements include the reward-weighted RL-Hybrid objective, group normalization,
`exp(beta * normalized_reward)`, decoder-policy fine-tuning, and SFT initialization. The following
remain Tier IV choices under experiment:

- direct road-border shaping and safety gating;
- stationary progress shaping and red-light constraints;
- streaming one-update-per-rollout versus replay-buffer epochs;
- EMA policy-update schedule and candidate group size;
- final reward weights and whether road-border shaping should be promoted.

No RL setting in this document should be treated as a change to the settled SFT/ViT model until a
real-data comparison establishes a measurable benefit without regressions in DAC, EPDMS, safety,
or target behavior.
