# HDP final state — what this branch is, and how to run it

Entry point for the Hyper Diffusion Planner work as of **2026-07-29**. Read this first;
`hyper_diffusion_planner.md` is the model-contract detail and `hdp_rl.md` is the RL
reference.

## The branch

There is exactly one: **`feature/hyper-diffusion-planner`**. Seven remote and thirteen
local HDP branches were deleted on 2026-07-29 after everything unmerged was pushed as an
annotated tag — nothing was lost, see [Archive tags](#archive-tags).

`--rl_paper_exact` ships here (`a8e55afc`, `c3611a52`, `8871aa8f`, `d848517b`), opt-in and
default-off.

One line of merged code is easy to mistake for a stylistic choice and is not:
`sorted(...)` over the metric dict in `utils/ddp.py` (#317). Built from a set literal it
gave 8 ranks 8 iteration orders, and no HDP-RL replay epoch could finish under DDP.

## Pipeline

| Stage | Launcher | Notes |
| --- | --- | --- |
| Base (80-token, ego-only) | `run_hdp_ego_only_base80_node02.sbatch` | Full vehicle-parameter corpus, no SFT init. Three `is_skipped`-filtered right-turn manifests, each repeated ×10. Source lists are immutable inputs; the job never rewrites them. |
| Base (node01 variant) | `run_hdp_ego_only_base_node01.sbatch` | |
| SFT | `run_hdp_ego_only_sft_node01.sbatch` | Asserts `BASE_RUN/latest.pth` is at the expected epoch. |
| Staged SFT | `run_hdp_staged_sft_node02.sbatch` | Stage 1 removes signal feedback and adapts only the trajectory planner; stage 2 trains the turn-indicator head. Each stage hands its `latest.pth` EMA to the next. |
| RL post-training | `run_hdp_rl.sbatch` | Same three manifests at the same ×10 repeat as Base, re-checked by the trainer, so an RL delta is attributable to the objective and not to a distribution shift. |
| Turn-indicator head | `run_hdp_turn_indicator_head_node01.sbatch` | Safe to run **concurrently** with RL — see below. |
| LR probe | `run_hdp_policy_lr_probe_node02.sbatch` | |

Entry points: `train_predictor.py`, `train_hdp_rl_predictor.py`, `valid_predictor.py`,
`valid_predictor_closed_loop.py`.

## Turn-indicator head

The head is a detached probe on frozen policy features. `supervised_training_stage=turn_indicator`
freezes the planner, keeps it in eval mode, and conditions the head on the expert future;
diffusion sampling is skipped entirely, so the scene encoder runs once per batch and DiT is
not evaluated. Validation scores `turn_indicator_logit` — the head applied to the
*generated* trajectory — which is what the vehicle sees.

`HDP_HEAD_EPOCHS` defaults to **2**, where the head turns over on this configuration; the
turnover epoch belongs to the architecture and lr, so re-measure it if either changes. This
is a training-length default and not a checkpoint selection — the deliverable is still
`latest.pth`, which after two epochs simply *is* the epoch-2 head.

Architecture and regularization default to 4 queries, 2 layers, dropout 0.1 and label
smoothing 0.05, overridable per run by `HDP_HEAD_NUM_QUERIES`, `HDP_HEAD_NUM_LAYERS`,
`HDP_HEAD_DROPOUT` and `HDP_LABEL_SMOOTHING`. Those values are the winning arm of a
6-epoch A/B against the original single-query single-layer probe, both arms trained from
the same frozen `base80` epoch-80 policy: the redesigned head leads at **all six epochs**
on accuracy, balanced accuracy, macro-F1, active-F1 and turn-direction accuracy, and on
the epoch-6 `latest.pth` by balanced accuracy 0.8121 vs 0.7934, active-F1 0.8229 vs
0.8059 and direction accuracy 0.7345 vs 0.7046. `valid_loss_ego` is 2.3241 in both, which
is the check that the head stayed detached from the planner.

`turn_indicator_label_smoothing` is detected by the launcher with its own grep, because a
source pinned before it existed would be handed an unknown flag by argparse. A pin that
predates any of these flags now aborts rather than running: with the defaults no longer
equal to the legacy probe's, omitting the flags would train a different architecture than
the run records.

Running the head concurrently with RL is safe by construction. Per
`configure_rl_trainable_parameters` (`train_hdp_rl_predictor.py:87-95`),
`--rl_train_scope decoder` trains only `decoder.dit.*` and `decoder.global_route_encoder.*`,
and `all` trains everything *except* `decoder.turn_indicator_predictor.*` — the head is
untouched under **both** values.

## Checkpoint rule — always `latest.pth`

Every downstream use takes `<run_dir>/latest.pth`: SFT/RL/staged-head init, validation,
closed-loop eval, ONNX export, and anything a report points at. **Never** `best_model/` or
`best_epdms_model/` — those come from a validation-proxy selector on a small split that
does not predict closed-loop or real-vehicle behaviour, and substituting one silently
desyncs a run from the launchers' epoch assertions and from the EMA state the next stage
expects. Full reasoning in `checkpoint_selection.md` and `../CLAUDE.md`.

If a validation metric peaked early, report it — that is a fact about overfitting. It is
not a reason to change the checkpoint.

Two non-exceptions: `epochNNNN/best_model.pth` is a filename quirk for a periodic
snapshot, and `best_model/` inside `train_hdp_rl_predictor.py` is internal trainer state
(last *accepted* RL policy + `best_valid_score` bookkeeping for strict resume).

## Precision: fp32, and TF32 off

The vehicle runs fp32. Under bf16 the first waypoint quantizes to a **39 mm** longitudinal
comb — larger than a real cycle delta by ~134× — which silently disables the anti-jitter
reward term on 67.5% of rows. `amp_dtype: "auto"` is bf16 in disguise. With AMP and TF32
both off, a 46k-scene eval is byte-identical across replicates and the A/B noise floor is
exactly zero, which is what makes small paired contrasts readable at all.

## Reproducibility contract

- **Base is not retrained.** It is a fixed premise, not a tuning knob.
- **RL trains on exactly the Base training set**, augmentation included.
- The turn-indicator head loss is gated by `training_stage == "turn_indicator"`
  (`train_epoch.py:189-190`), so policy-only Base/SFT never evaluates it. Head-side
  changes cannot affect base reproducibility.
- `_FINGERPRINT_RETIRED` pins retired reward fields, so all 32 historical
  `reward_fingerprint` values still reproduce byte-for-byte and a resume of a ~2.1 TB
  mined cycle reads its cache instead of aborting on drift.
- `effective_config.json` records 208 keys including anchor horizons, precision,
  `world_size` and lr — but **not** `model_path`, `seed`, `start_epoch` or `epochs`.
- Treat `batch_size // world_size` as load-bearing: bf16 went non-finite at local batch
  128 in 6/6 attempts with a byte-identical `args.json`.

## Reading a result

- **Never compare across reward definitions.** Job 1564 uses
  `rl_reward_aggregation: gated_product`; job 1519 used paper-exact. Their
  `valid_reward_deterministic_mean` are 0.6525 and 4.29598 — that is a definition change,
  not a regression. Compare on definition-independent evaluator flags
  (`epdms_no_at_fault_collision`, `collision_active`, `collision_rear`).
- **Never judge a cycle before its last epoch.** One campaign was negative for 6 epochs
  and then jumped positive.
- **Never quote one replicate.** Same-checkpoint eval sd is 7.29e-05; per-flag net sd is
  lane 4.2 / road-border 2.9 / collision 1.5 / kinematic 0.9. Epoch wander is the real A/B
  noise floor, and the free epoch-0 pair understates it ~7×.
- Argmax-over-epochs inflates a cycle gain: the null alone gives +1.45e-04. Per-cycle
  numbers are cumulative and must never be summed.
- `det_progress` is **anti-correlated** with the executed first waypoint — the proxy
  improves while executed advance falls. Judge on executed error against the human.

## Archive tags

Unmerged HDP work is on tags, not branches. `git tag -n99 -l 'archive/*' -l 'provenance/*'`
carries the full reasoning for each.

| Tag | Contents |
| --- | --- |
| `provenance/job1564-hdp-rl-gated-collision` | `8b0336e9` (+ `5d92b96c`). **Not** ancestors of the branch — #317 merged rebased copies — and job 1564 pins `8b0336e9` as `HDP_EXPECTED_COMMIT`, so this tag is the only thing keeping its provenance resolvable. |
| `archive/hdp-presquash-local-20260729` | The 28 local-only pre-squash commits (`b43c55c9`). |
| `archive/hdp-full-working-state-20260729` | Pre-prune full working state; also covers `backup/pre-rebase-20260729`. |
| `archive/hdp-pre-rebase-final-20260729` | Separate line: fp32/selected-arm campaign, per-scene rollout visualisation, advisory `det_progress` veto axis, step-1 expert anchor. |
| `archive/hdp-flow-matching` | x0-parameterized flow matching. Not merged: it would change the generative path the base model and every artifact on disk were trained with. |
| `archive/hdp-right-turn-traffic-light-mask` | Right-turn traffic-light masking (touches the SFT data path, unvalidated) + a 2026-07-10 snapshot of `hdp_rl_epoch.py`/`onnx_export.py`, both rewritten since. |
| `archive/hdp-turn-indicator-token-type` | Superseded: its reusable half (`FloatsEncoder` taking `class_type`) already landed, and `encoder.py:60-61` now *raises* on the turn-indicator input path it added. |
| `archive/hdp-third-pass-audit-20260711` | Conditioning alignment / training-state hardening. |

## Deployment

`torch2onnx.py` rglobs **every** `.pth` under the directory it is given, and the RL
deliverable is the EMA, so export with `--use_ema` from a directory staged with that
checkpoint alone. Latest packages under `outputs/model_upload/`.

## Verification

`.venv/bin/python -m pytest` from the repo root → **635 passed, 15 skipped** (the
`diffusion_planner/tests` subdirectory alone collects only 418);
`.venv/bin/python -m ruff check` → clean. CI runs `pre-commit`, which includes
`ruff format` — run `pre-commit run --all-files` before pushing. Note the system `python3`
has neither pytest nor ruff; always use `.venv/bin/python`.
