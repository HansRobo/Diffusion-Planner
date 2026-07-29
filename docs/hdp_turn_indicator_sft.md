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
does not justify the old 10x active-state class weighting; ordinary per-sample
cross entropy preserves calibration for the output state machine.

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

Head-only training uses expert future waypoints, which provide a clean intent signal
without running the diffusion policy.

The staged training protocol is:

1. `supervised_training_stage=policy`: initialize from the stopped Base EMA,
   freeze and completely skip the new head, and adapt the trajectory policy after
   removing signal feedback.
2. `supervised_training_stage=turn_indicator`: initialize from the policy-stage
   latest EMA, freeze the complete planner, keep it in evaluation mode, and train
   only the head. The encoder runs once per batch and DiT is not evaluated.

There used to be a third stage, selected by a `deployment` head mode, that re-trained
the same head on the detached final six-step DPM trajectory so that its inputs matched
inference exposure. It was removed after being measured. Over full epochs of the
2026-07-29 head architecture A/B (jobs 1540/1541), the head's expert-conditioned and
generated-conditioned predictions agree to 0.08 accuracy points — 0.96982 vs 0.96911 —
and the expert cross-entropy is the *lower* of the two (0.25102 vs 0.25258). The extra
stage paid for six DPM steps per batch to re-learn what the expert stage already knew.
Removing it does not weaken the check: validation is unchanged and still scores
`turn_indicator_logit`, the head applied to the *generated* trajectory
(`validate_model.py`), so exposure drift would still be visible in the metrics.

A persistent encoder-feature cache is deliberately not used. The full Base80 data
would require roughly 1.8 TB even in bf16, and cached features would no longer match
the random geometric augmentation applied to the current sample. Batch-local reuse
keeps exact augmented inputs without redundant encoder evaluations.

The joint mode remains available for controlled experiments. In that mode, the
generated per-sample loss is weighted by `(1-t)`, and the generated and expert
cross-entropies are combined as an even mean. The production staged run does not use
joint training. No stale four-class weights or transition-onset multiplier is used in
any mode.

Both stages use the exact Base80 data contract: the 20260707 vehicle-parameter and
mirror manifest, the same three `is_skipped`-filtered right-turn manifests repeated
10 times, and the same balanced validation manifest. The staged launcher verifies
their SHA256 values and compares the paths/repeat count with the Base `args.json`
before loading a checkpoint.

## Temporal output

`diffusion_planner.utils.turn_indicator.TurnIndicatorStateMachine` is a pure
postprocessor. It never feeds state back to the policy. Defaults at 10 Hz are:

- activation confidence at least 0.60 for 0.3 s;
- deactivation confidence at least 0.60 for 0.3 s;
- opposite-direction confidence at least 0.70 for 0.5 s;
- minimum active duration of 1.0 s;
- probability EMA alpha of 0.50.

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
- per-class recall and counts.

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
