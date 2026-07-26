# HDP trajectory augmentation: representation compatibility and DP adaptation

Last updated: 2026-07-22

## 中文结论（先读这一段）

这次失败不是“HDP augmentation 无效”，而是**同一个几何增强被直接套在了未审计的轨迹表示上**：

- HDP public code 先把模型 action 转成 waypoint，再对整段 waypoint 加约 `0.5 m` 的纵向/横向 offset。公开 HDP IL 配置是 `kinematic_type: diff`，但 RL YAML 默认是 `kinematic_type: waypoint`，因此不能笼统地说这个 transform 在“原生 HDP 表示”里天然成立。
- 原始 DP 输出的是从当前 ego state 出发的绝对 x-start future pose，第一个 waypoint 约为 `t=0.1 s`。若从这里立刻加入完整 offset，就会人为制造瞬时位置跳变，同时可能令 `x/y` 与 heading 不一致。
- AWR 再把这些候选当回归目标，会把“初始跳变”也学进去；本次负向分支实际表现为轨迹整体缩短、mode averaging 和减速，而不是有意义的安全收益。

当前修复只针对原始 DP：保持 deploy-time deterministic candidate 四通道 bitwise 不变（当该候选存在时）；其他候选从当前状态 `t=0` 的零 offset 开始，用约 2 秒 quintic minimum-jerk ramp 平滑达到 HDP 所采样的最终 offset，并像 released HDP transform 一样保留模型预测的 heading。这个修复已通过几何、anchor 和 cache 审计；matched 512-update 实验把 validation 退化约减半，但仍未得到正收益，因此结论是：**ramp 是必要的表示修复，但不是完整的性能修复。**

以后如果对真正的 HDP 做 RL，可以保留一个 exact-public-code reproduction branch，但正式训练前必须从 checkpoint 和 resolved config 确认 `kinematic_type`。`waypoint` 分支要做与本页相同的首帧连续性审计；`diff` 分支也不能直接放行，因为整段 waypoint translation 转回 delta action 后主要集中在第一个 delta。只有根据真实表示验证 constant/ramp 的 action 与 waypoint 连续性后，才能选择正式增强。

## Purpose

This note records a failure mode found while adapting HDP-style AWR post-training to the original T4 Diffusion Planner (DP). It is intended as an implementation reference for both:

- future RL work on the original DP; and
- future work on an actual HDP model, where the released HDP transform may be correct and should not be changed without checking the trajectory representation.

The central lesson is that an augmentation is defined on a representation, not only on a geometric curve. The public transform is applied after converting model actions to waypoints; whether it is acceptable depends on the checkpoint's actual `kinematic_type` and on what happens when the augmented target is converted back for training.

## Released HDP transform

The HDP reference samples one longitudinal offset `a` and one lateral offset `b` per candidate, normally with `a,b ~ N(0, 0.5 m)`. It first converts generated model actions to waypoints, then transforms every predicted waypoint through the candidate heading:

```text
x' = x + a cos(yaw) - b sin(yaw)
y' = y + a sin(yaw) + b cos(yaw)
```

The same scalar pair is applied over the whole future. Heading is left unchanged in the released transform. This broadens a K-sample trajectory group and gives the reward function alternatives to rank.

Do not conclude that this transform is universally wrong. Also do not assume that the release proves it universally correct: the public HDP IL config overrides `kinematic_type` to `diff`, while the public RL config currently inherits the shared `waypoint` default. For `diff`, a constant translation of all absolute waypoints cancels from later differences and is concentrated largely in the first delta; for `waypoint`, it remains a constant absolute translation. Both cases require an explicit continuity audit.

## Public HDP configuration ambiguity

The relevant public paths are:

- `HDP-navsim/hdp_navsim/agent/dp_vla/scoring.py::augment_trajectory_batch`: constant waypoint-space translation, heading preserved;
- `HDP-navsim/hdp_navsim/agent/dp_vla/dp_vla_rl_agent.py`: `model_action_to_waypoint` before augmentation and `waypoint_to_model_action` before loss;
- `HDP-navsim/hdp_navsim/config/agent/dp_vla_agent_hdp.yaml`: HDP IL sets `kinematic_type: diff`;
- `HDP-navsim/hdp_navsim/config/agent/dp_vla_rl_agent.yaml` plus `_shared/model.yaml`: the public RL path defaults to `kinematic_type: waypoint` unless overridden externally.

Therefore a future HDP RL run must save the fully resolved Hydra configuration next to the checkpoint. A model name such as “HDP” is not enough to determine target semantics.

### Public code must be reproduced critically, not copied blindly

The local audit used the clean official repository `ZhengYinan-AIR/Hyper-Diffusion-Planner` at commit `1ec9bd44910e17977680f7ad78fa342a13d5094d`. Several paper/code/config differences must be resolved explicitly in any future port:

1. The paper reports group size 32; the public NAVSIM RL YAML uses 10.
2. The paper discards identical-reward groups; the clean public NAVSIM release instead gives a tied finite group unit weights after zero z-score. A later local Tier IV implementation follows the paper's discard behavior.
3. The paper reports EMA 0.05 but does not define its update/decay convention or boundary. The clean public NAVSIM YAML disables EMA and comments out `ModelEma` initialization. A later local Tier IV implementation interprets it as one epoch of proposal training followed by `0.95 × old + 0.05 × proposal`, copying the accepted policy to live and clearing optimizer state. That is a useful local hypothesis, not verified public-code semantics.
4. HDP IL sets `kinematic_type: diff`; public RL defaults to the shared `waypoint` value unless an external override is supplied.
5. In `_rl_rollout`, `filter_mask` has shape `[B,G]`, but `torch.where(filter_mask)[0]` returns each valid scene index once per valid candidate. With normal finite PDM scores this inserts the same scene-group into the in-memory replay buffer roughly `G` times. Equal duplication may leave the ideal sampling distribution unchanged, but buffer size, memory use and diagnostics are wrong, and nonuniform invalid candidates can bias it.

The T4 implementation intentionally stores exactly one group per source scene and validates group shape/count at cache close. “Faithful” must name the exact reference snapshot: the paper, clean public release and later local Tier IV implementation have different observable contracts and must remain documented rather than silently blended. The 5% epoch-boundary commit is an explicitly tested local interpretation, not an official-code fact.

## Why direct reuse is unsafe for original DP

The original DP checkpoint predicts absolute x-start future poses. Its first output is the ego pose about 0.1 s after the current state. A constant `0.5 m` offset applied from that first point creates an apparent displacement of roughly `5 m/s` over the first 0.1 s, before accounting for the model's own motion. A lateral offset also changes path tangent without changing the stored heading.

This creates three mismatches:

1. **State discontinuity:** the candidate no longer starts continuously from the observed current state.
2. **Pose inconsistency risk:** a time-varying augmented `x/y` is no longer exactly the curve implied by the stored heading. However, reconstructing yaw from noisy stochastic x/y is substantially worse; preserve heading unless a representation-native action transform is available.
3. **Regression bias:** a diffusion MSE over multiple translated targets can average incompatible absolute poses, shortening or smoothing the deploy-time trajectory instead of learning the intended maneuver.

When a deterministic behavior candidate is included, it must remain completely untouched. All augmented x-start targets—whether the group is anchored or fully stochastic—need a continuous onset.

## Evidence from the 2026-07-18 T4 experiment

Evidence levels used below:

- **Confirmed:** directly measured from the stated cache/checkpoint.
- **Leading root cause:** explains all current observations and is supported by geometry audits, but still needs the matched ramp-vs-constant training comparison.
- **Pending:** implemented and validated structurally, but not yet proven to improve a trained checkpoint.

Source checkpoint SHA-256: `4ffaeea21cd29904da73349eea642e1d28f8ddbf02be363b7386e3a9b8ebcc75` (the original v5 best DP, before any AWR update).

The constant-offset source-best cache had healthy-looking ranking statistics, but a physically implausible first waypoint spread:

| Measurement | Constant HDP offset |
|---|---:|
| Source deterministic reward | 0.93245 |
| Groups with `Rbest - Rdet > 0.01` | 22.81% |
| First-waypoint max spread, P50 | 1.826 m |
| First-waypoint max spread, P90 | 2.364 m |
| Endpoint max spread, P50 | 2.149 m |

After 512 AWR updates with plain MSE and LR `5e-7`, the fixed train selector and independent validation both regressed:

| Metric | Source | After 512 updates | Delta |
|---|---:|---:|---:|
| Train-selector reward | 0.93250 | 0.93070 | -0.00180 |
| Validation reward | 0.93240 | 0.92144 | -0.01096 |
| Validation path length | 42.12 m | 40.17 m | -1.95 m |
| Validation progress score | 19.736 | 19.418 | -0.317 |
| Validation ADE | 2.602 m | 2.924 m | +0.323 m |

The observed failure was global shortening/mode averaging, not a reward gain traded for imitation error. The run was stopped during the next replay epoch. These metric changes are **confirmed**.

The direct constant offset is a confirmed representation defect and a **leading contributor** to this particular negative branch. It is not an isolated explanation of AWR performance: loss type, group construction, update count, learning rate, anchor strength, and augmentation onset must be compared on the same source checkpoint and scenes.

### Important counter-evidence: do not overclaim causality

An earlier full-corpus run used the less conservative, released-HDP-style group construction: all `K=10` members were stochastic, all finite members received group-relative exponential weights, the loss was plain MSE at LR `1e-6`, and the same constant offset was applied. Its full 46,262-scene validation trajectory was:

| Epoch | Validation reward | Delta vs source |
|---:|---:|---:|
| Source | 0.932158 | — |
| 2 | 0.931630 | -0.000528 |
| 3 | 0.936092 | +0.003934 |
| 4 | 0.938172 | +0.006014 |
| 8 | 0.938657 | +0.006500 |

Therefore the invalid first-step geometry did **not** prevent that different AWR process from eventually improving the measured reward. This is why the safe conclusion is representation-specific: the constant transform should not be carried into absolute x-start DP, but one negative 512-update branch cannot be used to claim that HDP group-relative AWR is ineffective.

## Original-DP-safe adaptation

For x-start DP, sample the same final route-frame offsets as HDP but multiply them by a minimum-jerk onset:

```text
u_t = clamp(t / T_ramp, 0, 1)
w_t = 10 u_t^3 - 15 u_t^4 + 6 u_t^5
a_t = w_t a
b_t = w_t b
```

The current T4 profile uses `T_ramp = 20` at 10 Hz, or about 2 s. The quintic has zero first and second derivatives at both ends, so it introduces no position, velocity, or acceleration jump at the current state. Heading is preserved from the sampled DP trajectory, matching the released HDP transform.

Candidate 0 is restored bitwise after this operation. Its `x`, `y`, `cos(yaw)`, and `sin(yaw)` must all equal the deploy-time deterministic original-DP output.

The first 131,072-source-scene ramp cache showed the intended **position-onset** geometry (133,120 groups on disk include DDP padding):

| Measurement | Quintic ramp, anchor fixed |
|---|---:|
| Source deterministic reward | 0.932414 |
| Best candidate reward | 0.937797 |
| Mean `Rbest - Rdet` | +0.005382 |
| All-zero reward groups | 1.94% |
| Groups with `Rbest - Rdet > 0.01` | 14.62% |
| Groups with `Rbest - Rdet > 0.05` | 0.30% |
| Mean active AWR targets per group | 1.255 |
| Behavior anchor active | 97.93% |
| First-waypoint max spread, P50 | 0.0078 m |
| First-waypoint max spread, P90 | 0.0408 m |
| Endpoint max spread, P50 | 2.151 m |
| Endpoint max spread, P90 | 3.388 m |
| Candidate pairwise ADE, P50 | 0.782 m |
| Candidate pairwise ADE, P90 | 0.996 m |

Compared with the constant transform, first-waypoint median spread fell from `1.826 m` to `0.0078 m`, while endpoint median spread stayed essentially unchanged (`2.149 m` to `2.151 m`). This is the desired position signature: remove the artificial initial jump without removing future geometric diversity. The lower positive-group rate is expected because candidates that looked different mainly because of the jump are no longer selected.

That cache still used tangent-reconstructed heading for augmented candidates. Its deterministic anchor kept the group-level all-zero rate low and initially hid the candidate-level kinematic problem. It is useful evidence for the position ramp and the negative 512-update result, but it is **not** the final preserve-heading cache and must not be reused for formal training.

The matched positive-only ramp + DP-geometry-loss run still regressed after 512 updates at LR `5e-7`:

| Metric | Source | Ramp after 512 updates | Delta | Constant-offset delta |
|---|---:|---:|---:|---:|
| Train-selector EMA reward | 0.932500 | 0.931832 | -0.000668 | -0.001800 |
| Validation policy reward | 0.932404 | 0.926732 | -0.005672 | -0.010962 |
| Validation path length | 42.125 m | 40.609 m | -1.515 m | -1.954 m |
| Validation ADE | 2.602 m | 2.867 m | +0.265 m | +0.323 m |

Thus the discontinuous onset explains a material part of the failure, but not all of it. The selected repair targets were not shorter on average: positive targets were `+0.479 m` longer than their deterministic anchor after AWR weighting, while the weighted group centroid was `+0.357 m` longer. The shorter deployed plan is therefore a regression/projection effect, not direct evidence that the reward prefers stopping.

The 512 updates are also only `14.4%` of one full-corpus replay epoch (`512 / 3,547` optimizer updates). The earlier faithful run dipped at epoch 2 and recovered from epoch 3 onward. Do not stop or accept a branch based only on this early point. The next controlled experiment keeps the previously successful group-relative AWR construction and changes only constant onset to the DP-safe ramp. Its 3,547-update epochs match full-corpus optimizer budget but repeatedly sample a 131,072-scene cache (about 40.9 draws per cached group per epoch), so it is a stress test rather than full-data evidence.

## Second pitfall: do not infer heading from stochastic x/y

An initial implementation recomputed augmented heading from consecutive transformed positions. This appears geometrically clean, but original-DP stochastic samples contain small x/y jitter. Differentiating it over `0.1 s` amplifies the jitter into yaw-rate and bicycle-model failures. A second attempt added only the smooth offset velocity to the model heading; it also failed on low-speed/stopped scenes, where a lateral translation cannot be realized as a normal vehicle yaw over 2 seconds.

Matched 5,120-padded-group smoke results:

| Heading mode | Candidate kinematic failure | All-zero groups | RB crossing | Mean stochastic reward |
|---|---:|---:|---:|---:|
| Preserve sampled heading | 0.088% | 1.74% | 6.23% | 0.879 |
| Add offset velocity | 22.90% | 21.58% | 17.22% | 0.657 |
| One-frame x/y tangent (131k cache) | 23.35% | 22.99% | 20.65% | 0.653 |
| Earlier successful constant-offset cache | 0.040% | 1.53% | 6.70% | 0.873 |

The formal DP-safe choice is therefore `heading_mode=preserve`. This is not a claim that pose/action consistency is unimportant. It is a constraint of this post-hoc waypoint-space exploration transform: a physically exact heading update requires generating the maneuver in a kinematic action representation, not differentiating a noisy sampled absolute path.

The preserve-heading smoke also retained the desired position diversity: first-waypoint max-spread P50/P90 was `0.0083/0.0409 m`, while endpoint max-spread P50/P90 was `2.23/3.55 m`. Thus preserving heading does not undo the ramp or collapse the future endpoint bundle.

## Anchor bug caught during implementation

The first ramp implementation correctly left candidate-0 position unchanged, but recomputed heading for every candidate. On real DP output, model heading is not always identical to the finite-difference path tangent. Candidate-0 reward fell from about `0.93` to `0.71`, and all-zero groups rose to `23.6%`.

Required invariant:

```text
augmented_candidates[:, 0, :, :] == deterministic_behavior[:, 0, :, :]
```

This is a four-channel equality, not only an `x/y` equality. The invalid cache was discarded. A regression test now uses a behavior heading deliberately different from the path tangent.

## Decision table for future projects

| Model/target representation | Default augmentation |
|---|---|
| Public HDP reproduction branch | Reproduce the resolved public config exactly, but keep it separate from the formal representation-safe branch |
| HDP with `kinematic_type=waypoint` | Audit first-step absolute-pose continuity; use a continuous onset if the same jump is present |
| HDP with `kinematic_type=diff` | Audit both waypoint and converted-delta targets; constant waypoint translation can become a first-delta impulse |
| Absolute x-start waypoints beginning at `t=0.1 s` | Use a continuous position onset; preserve sampled heading unless a representation-native kinematic transform is available |
| Velocity, acceleration, control, or delta-pose actions | Derive the transform in action space; do not reuse waypoint offsets blindly |
| Unknown representation | Disable augmentation until current-state continuity and inverse transform are proven |

For a true HDP RL run, keep an exact released-code branch for reproducibility, but do not promote it to the formal branch until the resolved checkpoint representation passes both waypoint-space and model-action-space continuity checks. Use the quintic adaptation only when those checks show that a continuous waypoint onset is the correct representation-native repair.

Do not mix two independent decisions:

1. **Representation decision:** constant versus continuous-onset offset.
2. **AWR decision:** all-stochastic group-relative weighting versus deterministic-anchor/positive-only weighting.

The ramp fixes decision 1. It does not prove decision 2, and the current local evidence favors testing the released group-relative construction before adding conservative anchor heuristics.

## Mandatory checks before mining a large cache

1. Candidate 0 is bitwise equal before and after augmentation.
2. First-waypoint position, velocity, yaw rate, and acceleration remain continuous from the current ego state.
3. Heading policy is explicit. Never silently reconstruct it from one-frame stochastic x/y differences; audit yaw-rate and bicycle-model failure rates against augmentation off.
4. First-waypoint spread and endpoint spread are reported separately; a large endpoint spread must not be achieved by a large first-step jump.
5. Candidate hard-gate rates are compared with augmentation off.
6. `Rbest - Rdet`, all-zero groups, and positive-weight target count are audited.
7. A small replay run must improve a fixed train selector before full-corpus mining.
8. Validation uses augmentation off; deployment is always the zero-noise K=1 trajectory.
9. Cache provenance records augmentation type, standard deviation, ramp duration, source checkpoint hash, reward config, and neighbor-future alignment.
10. Any cache that violates the behavior-anchor invariant is discarded, even if its ranking statistics look favorable.

## Local implementation and tests

- Configuration: `rlvr/configs/awr_original_dp_t4_hdp_stable.json`
- Faithful group-relative + DP-safe ramp configuration: `rlvr/configs/awr_original_dp_t4_hdp_group_relative_ramp.json`
- Sampling and augmentation: `rlvr/awr.py`
- Replay/training lifecycle: `rlvr/train_awr.py`
- Regression tests: `rlvr/test_awr_observability.py`

Relevant config fields:

```json
{
  "hdp_trajectory_augmentation": true,
  "hdp_trajectory_augmentation_std": 0.5,
  "hdp_trajectory_augmentation_ramp_steps": 20,
  "hdp_trajectory_augmentation_heading_mode": "preserve",
  "hdp_trajectory_augmentation_schedule": "every_refresh",
  "deterministic_first": false
}
```

The formal all-stochastic group has no special candidate 0; every one of the ten members is sampled and augmented. The bitwise candidate-0 invariant applies only to deterministic-anchor ablations, where candidate 0 represents the deployment policy and must not be perturbed.

Set `hdp_trajectory_augmentation_ramp_steps` to `0` for an exact constant-offset HDP reproduction.

## 2026-07-22 stop-turn safeguard added to Original-DP AWR

The intermediate checkpoint audit found a second failure mode that must be
tracked separately from the stored heading channel: at low speed, a large
first `x/y` waypoint changes the path tangent seen by the controller even if
the predicted `(cos yaw, sin yaw)` is unchanged.  This is exactly how a
constant `0.5 m` HDP offset can look like a large steering command while the
heading tensor itself appears normal.

The Original-DP AWR sampler now does the following on new mining processes:

1. HDP output-space offsets are skipped when the current longitudinal speed is
   below `2.0 m/s`; a ramped configuration still remains available for
   higher-speed exploration.
2. For low-speed groups, candidate 0 is retained exactly.  Non-anchor
   candidates are assigned zero AWR weight if their first `0.1 s` waypoint is
   farther than `0.25 m`, laterally farther than `0.20 m`, starts more than
   `0.05 m` backward, or has a first-path tangent above `75°`.
3. The gate is not a lane-change gate and does not inspect later waypoints, so
   a legitimate lane change remains eligible.  It is a local continuity guard,
   not a replacement for the collision/road-border reward.
4. Every rollout records first-waypoint displacement/tangent/backward rates and
   a separate `stop-turn` slice (`speed < 1 m/s` and current turn command left
   or right).  The batch implementation computes these on-device and transfers
   only compact diagnostics, so full-corpus mining does not perform a
   Python/CPU synchronization per scene.

The gate is applied to newly generated targets only.  Existing replay arrays
are immutable and remain valid historical artifacts; strict replay provenance
continues to distinguish a cache mined before and after the safeguard.  The
first corrected full-corpus refresh should therefore report, side by side,
`first_waypoint_*`, `stop_turn_*`, candidate rejection rate, reward and
controller kinematic failures.  A lower rejection rate is not itself a gain:
the intended outcome is to remove artificial start jumps without suppressing
the expert-like stop-and-turn trajectory.

## Experiment status

The constant-offset **positive-only 512-update branch** is proven negative on the stated fixed-cache experiment, while an earlier constant-offset **all-stochastic group-relative full run** eventually improved. The quintic transform passed geometry/anchor tests and cache-level signal audits and materially reduced the early positive-only regression, but did not turn that early point into a gain. Its 131,072-scene stress run remained non-selectable because repeated finite-support replay caused moving-scene contraction.

The required fresh full-corpus experiment has now completed one 10-epoch cycle over all 5,446,154 training scenes. Raw replay checkpoints over-shot the fixed train selector, but a retained `alpha=0.05` update along the epoch-4 direction improved the fixed 65,536-scene train selector by `+0.00008062` reward and the independent full 46,262-scene deterministic deployment evaluation by `+0.00049921` (paired bootstrap 95% interval `[+0.00021309, +0.00080759]`). Thus the ramp is no longer only a structurally correct prerequisite: it participates in a measured positive full-data AWR result. The gain is small and comes from a heavily retained update, so it does not prove that unconstrained raw AWR or repeated later augmentation is automatically beneficial. Cycle 2 therefore starts from the accepted checkpoint, re-mines on-policy, and uses a much smaller replay LR while retaining source fallback.

Local audit artifacts (paths are relative to the repository root):

- Constant-offset source cache: `outputs/awr_t4_stable_ablation/20260718-164943_source_best_stable_cache131072/`
- Negative 512-update branch: `outputs/awr_t4_stable_ablation/20260718-172151_ablation_plain_mse_lr5e7_1536updates/`
- Invalid ramp cache with candidate-0 heading corruption; never use for training: `outputs/awr_t4_stable_smoke/20260718-173656_smooth_ramp20_source_cache3072/`
- Anchor-restored tangent-heading smoke cache; historical audit only: `outputs/awr_t4_stable_smoke/20260718-173927_smooth_ramp20_anchorfix_cache1024/`
- Positive-only tangent-heading 131,072-scene cache; do not use for formal training: `outputs/awr_t4_stable_ablation/20260718-174104_source_best_smooth_ramp20_cache131072/`
- Matched DP-geometry-loss checkpoint ablation: `outputs/awr_t4_stable_ablation/20260718-175124_smooth_dp_geometry_lr5e7_512updates/`
- Earlier full-corpus group-relative run with positive epoch 3–8 validation: `outputs/awr_t4_full_sequence_filtered/20260717-225918_full_sequence_20260707_hdp_plain_mse_e100_restart_e2_full_all_train_select_8gpu/`
- Invalid group-relative + smooth-ramp tangent-heading cache; never replay: `outputs/awr_t4_stable_ablation/20260718-181157_source_best_group_relative_smooth_ramp20_cache131072/`
- Preserve-heading smoke: `outputs/awr_t4_stable_smoke/20260718-183016_group_relative_ramp20_heading_preserve_cache3072/`
- Velocity-offset-heading negative smoke: `outputs/awr_t4_stable_smoke/20260718-183152_group_relative_ramp20_heading_velocity_offset_cache3072/`
- Audited preserve-heading 131,072-scene cache: `outputs/awr_t4_stable_ablation/20260718-183426_source_best_group_relative_smooth_ramp20_preserve_cache131072/`
- Matched two-full-replay-epoch experiment from original best DP: `outputs/awr_t4_stable_ablation/20260718-184638_group_relative_smooth_ramp20_preserve_2fullreplay/`

The invalid-cache path is retained only to make the failure auditable. Any future cache inventory or launcher must explicitly exclude it.

## 2026-07-23 addendum: two defects in the first-waypoint safeguard chain

A full audit of the clean-start Cycle-1 cache (5,446,656 groups,
`outputs/awr_t4_full_sequence_filtered/plannerrft_clean_sft_offset0_full_cycle01_mine/20260723-002534_plannerrft_full_cycle01_mine`)
found the safeguard both necessary and, as first deployed, broken in two ways.

**1. The 75-degree tangent test had no displacement floor.** At standstill the
model emits `x = 0` exactly, so millimeter lateral sampler noise makes the raw
path tangent 90 degrees for every candidate. The gate zeroed 6,992,676 of
9,240,525 low-speed non-anchor candidates (75.7%); 99.4% of those rejections
fired only on the tangent test at a median displacement of 1.3 mm, the expert
GT itself failed the unfloored test in 35.4% of low-speed scenes, and 71.6% of
low-speed groups lost all nine sampled candidates. Fix: the tangent test now
requires `first_step >= original_dp_first_waypoint_gate_tangent_min_step_m`
(default 0.05 m). With the floor, rejections drop 53x to 130,718 genuine
near-field anomalies (first step p50 0.11 m, max 0.76 m at < 1 m/s) and GT
false rejections drop to 0.07% (all genuine reversing scenes).

**2. The positive-anchor replay overlay resurrected gate-rejected candidates.**
`build_positive_anchor_replay_overlay.py` recomputed weights from rewards alone
and ignored the mining-time zeros. Because the reward total is blind to
near-field geometry (89% of genuine anomalies sit within 0.01 of their group's
best reward; 46,126 were the group best), 8,686 gate-rejected anomalies were
active replay targets and held the top weight in 2,061 groups — including
7.8 cm backward starts at 0 m/s with weight 21.6. This, not the mining gate,
was the live leak feeding standstill-jump targets into every replay epoch.
Fix: the overlay now masks its recomputed weights with `source_weights > 0`
(`parameters.respect_source_zero_weights`), and the epoch-100 supervisor
fail-closes on overlays built without it.

Why the gate must stay despite the reward function: a first-waypoint jump of
tens of centimeters changes the 8-second reward total by ~1e-4, so
group-relative AWR would otherwise hand exactly these kinematically infeasible
candidates the largest positive-advantage weights — the same mechanism that
trained the original stop-turn steering-lock behavior.

## 2026-07-25 addendum：tangent floor 定得比它要治的现象还大

2026-07-23 为 first-waypoint gate 加的 tangent 位移下限取了 **5 cm**。实测停车工况
的首步位移 p95 只有 **3.9 cm**，也就是说这个下限**整体压在了它本该管辖的区间之上**：
静止时几乎每个候选都低于门槛，tangent 判据永不触发。四个 cycle 的
`mean_first_waypoint_gate_rejected_fraction` 与
`mean_stop_turn_first_waypoint_gate_rejected_fraction` **全部为 0** 就是证据 ——
safeguard 在低速区间等于关闭，动力学不可行的首点畅通无阻地成为训练目标。

后果与整机现象一致：首点横向 p95 从 0.0066 m 升到 0.0091 m（+38%），后退比例从
恒为 0 变成非 0，同期全量 46k 配对审计显示碰撞相对上升约 15%（P(improved)=0.003）。
方向盘在静止时抖动，就是这条链路的末端表现。

根因不只是数值取错，而是**判据本身选错了物理量**。绝对位移/横向阈值（25 cm / 20 cm）
无法表达转向可行性：3.9 cm 的首步配 0.91 cm 横向偏移对应曲率
`k = 2y/s² ≈ 12 /m`（转弯半径 8 cm），前轮转角 `atan(L·k) ≈ 88°` —— 打死。而这组数
在两个绝对阈值下都"合格"。

修复（`_implied_first_step_steer_rad`）改为直接门控**隐含前轮转角**
`delta = atan(wheel_base · 2|y| / s²)`，阈值 0.64 rad，并把 tangent 下限降到 **5 mm**
（真噪声量级，远低于静止工况）。在 cycle-1 真实挖掘候选上离线校准：低速非锚候选中
67.2% 属亚 5 mm 噪声（豁免），**29.4% 被判为打死并拒绝**（旧 gate 仅 2.2%）；可测候选
的隐含转角中位数 1.449 rad(83°)、p95 1.570 rad(90°)。17% 的低速场景会退化为只跟锚点，
锚点永不被拒因此不会出现空组。与 2026-07-23 那次 75.7% 的过度拒绝相比是有界收紧。

教训：给噪声加下限时，**必须先量出目标现象自身的尺度**，否则下限会顺手关掉判据。
同时，任何"看起来很小"的绝对量在静止工况都可能对应极端的转向指令 —— 低速判据要用
无量纲/几何量（曲率、转角），不要用绝对位移。

## 2026-07-25 addendum 2：报告不等于门槛

同期发现全量配对审计一直是 **report-only**：cycle 1-4 的提交判据只有 128 场景
train-selector 的 `mean_det_reward`。该聚合量上涨（四个 cycle 共 +0.2%）的同时，
progress/centerline/smoothness 的改善掩盖了 collision/kinematic/safety 的退化，而流水线
没有任何环节能阻止它。已加 `rlvr/autoresearch/tools/enforce_cycle_non_regression.py`：
reward 必须 `P(improved) >= 0.95` 且 improvement > 0，三个安全项必须
`P(improved) >= 0.10`（即不得"确信更差"），停车隐含转角 p95 不得上升；任一不满足则该
cycle 回退到起始 incumbent（`selection_kind=non_regression_veto_incumbent`），最坏情况
是"不变"而非退化。用 cycle 4 的真实审计回放验证过：该工具正确否决了它。
