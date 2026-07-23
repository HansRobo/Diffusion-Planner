# FeaXDrive vs. HDP: Performance Audit

Date: 2026-07-18

Source: [arXiv:2604.12656](https://arxiv.org/abs/2604.12656), v2 (30 April
2026). The local TeX source and PDF are under
`reference/papers/arxiv_2604.12656_feaxdrive/`.

## Executive conclusion

FeaXDrive is highly relevant to the current HDP problem, especially the
observed near-road-border behavior. It suggests three useful ideas:

1. impose curvature/kinematic feasibility on the clean predicted trajectory;
2. guide the clean trajectory toward a footprint-level drivable-area SDF during
   reverse diffusion;
3. include trajectory feasibility in RL group rewards.

However, FeaXDrive's headline architectural claim is not a reason to replace
HDP. HDP already predicts `x_start` and converts its velocity latent into
waypoints. In other words, the current HDP already exposes the clean trajectory
interface that FeaXDrive calls trajectory-centric. The useful difference is
where we apply feasibility constraints, not a need to replace the DiT, route
encoder, velocity representation, or DPM-Solver.

## What FeaXDrive actually does

### Trajectory-centric parameterization

FeaXDrive predicts a clean waypoint trajectory directly at every diffusion
time:

```text
x0_hat(t) = f_theta(x_t, t, condition)
```

The supervised loss is an L2 loss to the expert waypoint trajectory. It then
uses the same `x0_hat` for curvature regularization, reverse-sampling guidance,
and RL reward computation.

The paper contrasts this with a noise-prediction diffusion model. That contrast
does not map exactly to current HDP: `Decoder` already uses
`diffusion_model_type="x_start"`, predicts normalized per-step velocity, and
`velocity_to_waypoints` integrates it before geometry losses and output. We can
therefore evaluate FeaXDrive-style constraints on the existing integrated
trajectory without changing the action token contract.

### Adaptive curvature regularization

The method smooths predicted planar positions, estimates curvature using
arc-length derivatives, and imposes a speed-adaptive limit:

```text
kappa_bound(v) = min(kappa_geo, a_lat_max / (v^2 + eps))
L_cur = mean(relu(abs(kappa) - kappa_bound)^2)
```

Their NAVSIM settings use a Chrysler Pacifica minimum turning radius of about
6 m (`kappa_geo=0.166 m^-1`) and `a_lat_max=6 m/s^2`. These constants are not
valid defaults for our three vehicle data sources. HDP must use vehicle- and
dataset-specific geometry or a conservative bound validated against expert
trajectories.

### Footprint-level drivable-area guidance

FeaXDrive rasterizes a local drivable region into a signed distance field (SDF),
evaluates the four vehicle footprint corners at every predicted timestep, and
uses a softplus barrier with a safety margin. During every reverse-sampling
evaluation it computes a gradient with respect to the predicted clean trajectory
and nudges the trajectory away from the boundary before the next diffusion
update. Guidance is triggered for footprints that are outside or too close to
the boundary.

This is different from post-processing: the corrected clean trajectory becomes
the input to the following reverse step.

### Feasibility-aware GRPO

The paper samples a group of trajectories, adds a curvature-feasibility reward
to the benchmark reward, and uses group-relative advantages over the denoising
chain. It explicitly reports that score-only GRPO improves PDMS but worsens
curvature violations:

| Variant | PDMS | DAC | Curvature violation |
|---|---:|---:|---:|
| FeaXDrive-IL | 88.75 | 97.46 | 0.88% |
| Standard GRPO | 90.56 | 98.28 | 5.79% |
| Feasibility-aware GRPO | 90.00 | 98.31 | 2.40% |

This is evidence that RL can trade away physical feasibility even while the
benchmark score improves. The exact GRPO likelihood implementation is not
identical to the current HDP reward-weighted hybrid update, but the reward
design lesson transfers directly.

## Comparison with current HDP

| FeaXDrive component | Current HDP status | Assessment |
|---|---|---|
| Clean `x0` interface | x-start prediction of velocity, integrated to waypoints | Already present; no architecture replacement needed |
| Waypoint supervision | hybrid waypoint loss over a recent integration window | Stronger than replacing it with plain L2; retain it |
| Vehicle footprint | `compute_ego_bbox_corners` and `compute_ego_edge_points` use `ego_shape` | Training geometry already exists |
| Road-border training | optional `compute_road_border_penalty` over footprint edge points | Useful but unsigned nearest-border distance, not a true inside/outside SDF |
| Inference road guidance | no clean-trajectory guidance inside DPM sampling | Highest-value missing component |
| Curvature loss | no explicit curvature/kinematic term | Isolated SFT ablation candidate |
| RL feasibility | occupancy, TTC, THW, lane, progress, road-border and red-light terms | Add curvature as a separate reward only after validation |
| Sampling | DPM-Solver++ | Keep it; guidance must be integrated with its `x0` conversion, not copied from DDIM |

## What can improve the model

### Priority 0: measure before retraining

Run the current best Base/SFT checkpoints through a fixed feasibility audit:

- curvature violation rate under per-vehicle bounds;
- maximum curvature and curvature jerk by speed bucket;
- footprint clearance to road borders;
- fraction of timesteps with any footprint outside the drivable region;
- DAC/PDMS, red-light violations, right-turn success, and comfort together.

This tells us whether the current failure is genuinely curvature/geometry or
mostly a map/route semantic problem. A lower curvature rate is not a win if it
reduces right-turn progress or increases red-light violations.

### Priority 1: inference-only footprint guidance

This is the most attractive first experiment because it does not require a new
SFT checkpoint and inference speed is not a constraint.

The current line-string penalty cannot be used as an SDF unchanged:

- it measures nearest distance to road-border segments;
- it has no signed inside/outside information;
- from outside the road, a symmetric distance gradient can point in the wrong
  direction;
- nearest-segment minima can switch discontinuously at corners.

For a correct guidance experiment, construct a signed drivable mask from route
lane polygons or consistently oriented lane boundaries. Preserve `ego_shape`
from the input and evaluate all footprint corners (or edge samples, as the
current training penalty already does). Apply guidance only when the footprint
is outside or below a margin, and scale it by diffusion time: early high-noise
predictions should receive weak or no geometric correction; late denoising steps
should receive the strongest correction.

For DPM-Solver, the guidance must modify the clean `x0` used by the solver's
`x_start -> noise` conversion at each model evaluation. A final trajectory clamp
would be a different post-processor and could break DPM consistency.

Acceptance condition: DAC/road-border clearance improves without degrading
PDMS, red-light compliance, right-turn success, curvature rate, or comfort.

### Priority 1: adaptive curvature loss as an isolated SFT arm

If Priority 0 confirms real curvature spikes, add an optional
`curvature_feasibility_loss` after inverse-normalizing and integrating the
predicted velocity. Do not alter the action representation:

```text
pred_velocity_x0 -> velocity_to_waypoints -> smooth xy
                 -> curvature(v) -> adaptive bound -> penalty
```

Important differences from a naive copy:

- use actual `ego_shape`/vehicle dynamics instead of the paper's Pacifica
  constants;
- use the same 0.1 s time base as the dataset and mask invalid/padded points;
- stabilize near-zero displacement and stationary scenes;
- compute curvature on the integrated position trajectory, not normalized latent
  velocity;
- gate or weight the penalty at low/moderate diffusion noise. At very high
  noise, `x0_hat` is a weak conditional-mean estimate and a strong curvature
  gradient can bias the learned diffusion score;
- start with a very small coefficient and compare against road-border loss
  alone, not only against a no-safety-loss baseline.

The paper's ablation reports curvature violation falling from 7.51% to 0.13%
with little PDMS change, but its vehicle, map, trajectory resolution, and
curvature evaluator differ from ours. That result justifies an experiment, not
the coefficient or threshold.

### Priority 2: curvature-aware RL reward

The current HDP RL reward already contains occupancy, TTC, THW, lane, progress,
red-light, and optional road-border terms. Add curvature only as a separately
logged and configurable term, for example:

```text
reward = current_reward + w_curvature * curvature_feasibility_score
```

Use a continuous score or a margin-based penalty, not only a binary violation,
so groups retain useful reward variance. Normalize it within the same group and
keep the current safety gate. Checkpoint selection must continue to use the
held-out multi-metric guard; otherwise the RL policy can buy progress by making
sharp, infeasible turns. The FeaXDrive results make this failure mode concrete.

This should be a separate RL experiment after the SFT checkpoint is fixed; it
should not contaminate the primary SFT model.

## Risks in directly copying FeaXDrive

1. **The `x0` novelty is overstated for HDP.** Current HDP already predicts
   `x_start`; only the coordinate parameterization differs.
2. **Fixed vehicle limits are unsafe for our data.** We have multiple vehicle
   sources and real `ego_shape`; one Pacifica bound can over-penalize valid
   maneuvers or under-penalize unsafe ones.
3. **Curvature and road guidance can conflict.** FeaXDrive itself reports that
   drivable-area guidance raises curvature violations from 0.13% to 0.88%.
   Guidance must be evaluated jointly with smoothness and right-turn metrics.
4. **A nearest-border penalty is not a signed SDF.** Reusing the current
   road-border distance as a guidance gradient can push an already-off-road
   trajectory farther outside.
5. **High-noise curvature gradients can bias diffusion.** The clean estimate at
   large `t` is not a reliable physical trajectory; a uniformly weighted
   auxiliary constraint can hurt score calibration.
6. **The paper uses DDIM and a VLM-conditioned NAVSIM planner.** Its sampler,
   input contract, 100-epoch IL recipe, and 16-layer DiT are not evidence that
   those choices improve our Tier IV model.

## Recommended implementation order

1. Implement the feasibility audit metrics only and run them on the current
   checkpoint.
2. Prototype inference-only signed footprint guidance on a small fixed
   validation subset; compare no guidance, late-step guidance, and all-step
   guidance.
3. If curvature violations are material, train one isolated SFT arm with
   adaptive curvature loss, using vehicle-specific limits and noise gating.
4. After SFT selection, add curvature feasibility to RL as a separate reward
   coefficient and retain the existing group-normalized safety guards.
5. Do not merge any of these into the primary training configuration until
   PDMS/DAC, road-border clearance, red-light behavior, right-turn success, and
   curvature/comfort all improve or remain within the agreed tolerance.

## Final decision

FeaXDrive provides a strong, concrete improvement direction for our current
road-border problem. The best transferable part is **clean-trajectory,
footprint-level feasibility handling**, especially inference-time signed SDF
guidance and a vehicle-aware curvature metric. The paper does not justify
replacing HDP's architecture or loss. The first safe action is evaluation and
inference-only guidance; curvature training and RL reward shaping should remain
isolated, measurable experiments.
