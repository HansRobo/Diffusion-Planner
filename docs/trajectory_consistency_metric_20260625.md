# Trajectory consistency metric

Date: 2026-06-25

## Motivation

Autoware has two relevant production-side metrics:

- `trajectory_ranker` trajectory consistency: compares points predicted by current
  and historical trajectories at a common absolute future time.
- `planning_evaluator` stability metrics: compares two post-processed trajectories
  with geometric distances such as Frechet distance, lateral distance, and lookahead
  lateral displacement.

Those metrics evaluate full online planner trajectories after post-processing. The
model validation loop here only has one direct model output per sample, so it cannot
compute the exact historical-trajectory variance without a sequence-aware evaluator.

## Implemented online validation proxy

File:

```text
diffusion_planner/diffusion_planner/validate_model.py
```

New metric family:

```text
ego_trajectory_consistency_error
ego_trajectory_consistency_velocity_boundary
ego_trajectory_consistency_acceleration_boundary
ego_trajectory_consistency_jerk
ego_trajectory_consistency_heading_boundary
ego_trajectory_consistency_heading_rate_change
```

Definition:

- connect observed ego history/current state to the predicted ego future
- measure velocity discontinuity at the current-to-future boundary
- measure acceleration discontinuity at the current-to-future boundary
- measure future jerk RMS inside the predicted trajectory
- measure heading discontinuity at the current-to-future boundary
- measure heading-rate change inside the predicted trajectory

`ego_trajectory_consistency_error` is a lower-is-better dimensionless composite:

```text
sqrt(mean([
  (velocity_boundary / 1.0 m/s)^2,
  (acceleration_boundary / 2.0 m/s^2)^2,
  (jerk_rms / 5.0 m/s^3)^2,
  (heading_boundary / 0.2 rad)^2,
  (heading_rate_change / 0.5 rad/s)^2
]))
```

## Logging

The metric is returned by `validate_model`, logged to W&B through the existing
`valid_loss/ego_*` path, and written to `train_log.tsv` as:

```text
valid_loss_ego_trajectory_consistency
```

The detailed components are also available in `valid_dict` and saved by
`valid_predictor.py` because they start with `ego_`.

## Limitation

This is not the exact Autoware historical trajectory consistency metric. It is the
strongest metric available inside the current sample-independent model validation
loop.

The exact metric should be added as a second-stage full-sequence evaluator that uses:

- sample path
- adjacent JSON timestamp and ego pose
- predicted trajectory for consecutive samples
- overlap at the same absolute future time in a common ego/world frame
