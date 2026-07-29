# HDP turn-indicator SFT design

## Scope

This design predicts turn intent without allowing the vehicle's previous signal
command to influence the trajectory policy. It is an SFT auxiliary task, not part
of the HDP trajectory objective and not an RL reward.

## Data evidence

The read-only audit tool is
`diffusion_planner/util_scripts/audit_turn_indicator_events.py`. The full Base80
manifest audit is stored under `artifacts/turn_indicator_audit_20260721/`.

Results for 5,446,154 samples:

- Every `turn_indicators` tensor has 31 frames.
- Valid raw states are exactly `1=DISABLE`, `2=ENABLE_LEFT`, and
  `3=ENABLE_RIGHT`.
- There are zero class-0 states and zero invalid values.
- Current-frame distribution is 61.14% disable, 21.03% left, and 17.84% right.
- The reconstructed sequences contain 11,439 disable-to-left and 9,426
  disable-to-right activations.
- Only 55.9% of left activations and 63.2% of right activations have either 0.5 m
  same-side lateral motion or 5 degrees same-side yaw within the available 8 s
  future.
- For those geometry-confirmed events, median signal-to-motion time is 3.5 s for
  both directions, with broad 10th-to-90th percentile ranges of 1.6-6.4 s (left)
  and 1.4-6.8 s (right).

The last result rules out a universal exact-frame or time-to-maneuver target.
Human signal timing, cancellation, long lead time, and futures that end before the
maneuver make that target partially unobservable. The network therefore predicts
state probabilities; temporal stability is handled after the network.

The exact default SFT manifest was audited independently: all 4,578,036 samples
are valid, with 62.07% disable, 20.34% left, and 17.59% right. This distribution
does not justify the old 10x active-state class weighting; per-sample cross entropy
preserves calibration for the output state machine. The two cost terms added on top
of it are error-conditional rather than class-conditional; see "Head objective".

## Model contract

The policy encoder has no turn-indicator input or token. `turn_indicators` remains
in training NPZ batches only to create the auxiliary label. The deployable planner
cannot form a signal-command feedback loop.

The head has three dense internal classes:

| Internal | Meaning | Autoware raw report |
| --- | --- | --- |
| 0 | disable/off | 1 |
| 1 | enable left | 2 |
| 2 | enable right | 3 |

Class 0 in the raw Autoware message is invalid and raises during training instead
of becoming a learned driving state.

The neural head reads:

- 16 normalized planned poses at 0.5 s spacing through 8.0 s, including heading;
- a learned attention readout over scene tokens;
- the global ordered-route condition;
- normalized current `vx`, `vy`, `ax`, `ay`, steering angle, and yaw rate.

Trajectory, scene, route, and proprioception tensors are detached inside the head.
Its loss cannot reshape the scene encoder, AdaLN condition, or diffusion policy.
The trajectory-policy total loss and checkpoint selection also exclude the head
loss.

Head-only training uses two sequential modes:

- `expert`: expert future waypoints provide a clean intent signal without running
  the diffusion policy;
- `deployment`: expert waypoints and the detached final DPM x-start trajectory are
  weighted equally, matching inference exposure after the head has learned a stable
  representation.

The staged training protocol is:

1. `supervised_training_stage=policy`: initialize from the stopped Base EMA,
   freeze and completely skip the new head, and adapt the trajectory policy after
   removing signal feedback.
2. `supervised_training_stage=turn_indicator` and
   `turn_indicator_head_training_mode=expert`: initialize from the policy-stage
   latest EMA, freeze the complete planner, keep it in evaluation mode, and train
   only the head for one full epoch. The encoder runs once per batch and DiT is not
   evaluated.
3. `supervised_training_stage=turn_indicator` and
   `turn_indicator_head_training_mode=deployment`: initialize from the expert-head
   latest EMA and fine-tune for one full epoch. Generated inputs come from the final
   six-step DPM trajectory, not a random-time one-step proxy. The frozen scene
   encoding is computed once per batch and reused by all DPM evaluations.

A persistent encoder-feature cache is deliberately not used. The full Base80 data
would require roughly 1.8 TB even in bf16, and cached features would no longer match
the random geometric augmentation applied to the current sample. Batch-local reuse
keeps exact augmented inputs without redundant encoder evaluations.

The joint mode remains available for controlled experiments. In that mode, the
generated per-sample loss is weighted by `(1-t)` and normalized by the sum of
weights. The production staged run does not use joint training. No stale four-class
weights or transition-onset multiplier is used in any mode.

Both stages use the exact Base80 data contract: the 20260707 vehicle-parameter and
mirror manifest, the same three `is_skipped`-filtered right-turn manifests repeated
10 times, and the same balanced validation manifest. The staged launcher verifies
their SHA256 values and compares the paths/repeat count with the Base `args.json`
before loading a checkpoint.

## Head objective

`diffusion_planner.loss.turn_indicator_objective` computes the per-sample head loss
for every mode (joint, expert, deployment). Cross entropy stays the backbone: the
state machine gates on absolute probabilities, not on the argmax, so the head must
remain a proper scoring rule and the rejected 10x class weighting stays rejected.
Two terms correct the two costs plain cross entropy prices wrong.

**Opposite-direction expected cost.** Cross entropy treats a left-for-right swap as
just another log-loss increment, but in deployment the two errors are not
interchangeable: a late or leaked signal is a comfort defect, while commanding the
opposite direction actively misleads surrounding traffic. On active ground truth the
probability mass on the mirrored class therefore carries an explicit expected cost,
`turn_indicator_opposite_direction_weight` (default 1.0). The Bayes-optimal opposite
probability is 0 whenever the label is a definite direction, so this term does not
trade away calibration on the classes the gates actually read. It targets the
measured weakness of the expert head: epoch-1 direction accuracy 0.760 with
`enable_right` recall 0.724 against `enable_left` 0.798.

**Evidence-conditional one-sided label smoothing.** Median lever-to-motion time is
3.5 s (10th-to-90th percentile 1.6-6.8 s), so a large share of frames labelled off
belong to a turn the driver has already committed to and simply has not signalled
yet. `turn_indicator_geometry_evidence` re-reads the expert future with the audit
tool's own bars - weak at 5 degrees same-side yaw or 0.5 m same-side lateral offset,
saturating at 20 degrees or 2.0 m - and returns the implied direction plus a 0-to-1
strength in which opposite-side evidence cancels, so an out-and-back swerve is
discounted while a genuine lane change is not. For off-labelled frames with non-zero
evidence, `turn_indicator_implied_intent_smoothing` (default 0.2) moves
`smoothing * strength` of the target mass from off onto the implied direction only.
Active labels stay one-hot and left is never smoothed toward right, so the term
relieves the under-triggering visible in active precision 0.929 against recall 0.770
without ever teaching a direction error.

Both terms reduce to the legacy objective at weight/smoothing 0.0, and the
`supervised_training_stage=policy` stage returns before the head loss is computed, so
Base and policy-stage SFT runs are numerically unaffected by these defaults. Two
diagnostics are logged next to the head loss:
`turn_indicator_opposite_probability` (mean opposite-class probability over active
ground truth) and `turn_indicator_implied_intent_mass` (mean target mass moved off
the off class). The six fields are strict-resume `training_fields`, exempted only from
the missing-field check so that checkpoints predating them can still be resumed.

## Temporal output

`diffusion_planner.utils.turn_indicator.TurnIndicatorStateMachine` is a pure
postprocessor. It never feeds state back to the policy. Defaults at 10 Hz are:

- activation confidence at least 0.60 for 0.3 s;
- deactivation confidence at least 0.60 for 0.3 s;
- opposite-direction confidence at least 0.70 for 0.5 s;
- minimum active duration of 1.0 s;
- probability EMA alpha of 0.50;
- probability temperature of 1.0.

Every gate above is an absolute probability, so when a gate opens depends on how well
the head is calibrated, not only on what it ranks first. `probability_temperature`
divides the logits before the softmax: above 1.0 flattens an over-confident head,
below 1.0 sharpens an under-confident one, and 1.0 leaves the network distribution
untouched. Because the softmax is shift-invariant, the state machine applies it by
re-softmaxing scaled log-probabilities, which is exactly `softmax(logits / T)` and
therefore works whether `update` is fed logits or probabilities. Fit it with
`fit_probability_temperature`, a deterministic golden-section search on
log-temperature over sequential validation predictions, then freeze it as deployment
configuration. Rescaling cannot change the argmax, so every accuracy metric is
invariant; the temperature is not part of the network and does not appear in any ONNX
graph.

These values are initial safety-oriented defaults. Tune them on sequential SFT
validation predictions and then freeze them as deployment configuration. The
state machine returns internal state 0/1/2; `raw_report_state` maps it to 1/2/3.

## Validation

Exact 0.1 s transition-frame accuracy is intentionally removed from W&B. It
measures annotator timing more than usable intent quality. Log:

- overall accuracy;
- balanced accuracy and macro-F1;
- active precision, recall, and F1 (off versus either direction);
- direction accuracy over active ground truth;
- per-class recall and counts;
- negative log-likelihood and expected calibration error.

`turn_indicator_nll` and `turn_indicator_ece` come from
`diffusion_planner.validate_model.turn_indicator_calibration_counts`, which
accumulates a 10-bin reliability diagram over the top-class probability alongside the
summed NLL. Only summed counts cross the DDP boundary, so the reduction is
rank-count-independent; the accumulation is float64 and clamps probability 1.0 into
the last bin. ECE is the count-weighted mean gap between top-class confidence and
top-class correctness. These two numbers, not accuracy, say whether the state
machine's absolute-probability gates are reading a trustworthy scale, and they are
what `fit_probability_temperature` minimizes and diagnoses.

Sequential deployment evaluation should additionally report event onset delay,
early activations, missed events, false activation duration, and output switches
per minute after the state machine.

## Base80 to SFT migration

Base80 was trained with the historical signal token and four-logit head. SFT uses
weights-only initialization from Base80 EMA. The loader explicitly:

1. drops only `encoder.turn_indicator_encoder.*` tensors;
2. reinitializes the complete `decoder.turn_indicator_predictor.*` head when its
   tensor contract differs;
3. loads every other policy tensor strictly.

Strict resume never performs this migration. A partially trained old run cannot be
resumed under the new architecture. After migration, SFT trains the policy without
signal feedback and trains the new head from scratch.

New checkpoints persist `policy_uses_turn_indicator_history=false` and
`turn_indicator_output_dim=3` as architecture provenance. Strict resume requires
both fields and exact values. Weights-only initialization alone permits those fields
to be absent or different in a legacy checkpoint, after which the targeted module
migration above is still enforced by tensor keys and shapes.

## ONNX contract

The full and encoder ONNX graphs no longer accept `turn_indicators`. The standalone
head graph accepts `encoding`, `final_x0`, `global_route_condition`, and
`ego_current_state`, and returns three logits. Mapping and state-machine logic stay
outside the neural graph.
