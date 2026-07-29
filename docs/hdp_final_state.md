# HDP final state — what this branch is, and how to run it

Entry point for the Hyper Diffusion Planner work as of **2026-07-29**. Read this first;
`hyper_diffusion_planner.md` is the model-contract detail and `hdp_rl.md` is the RL
reference.

Two things this document exists to prevent:

1. Concluding that experiments were lost because `git branch -a | grep hdp` shows one
   branch. Nothing was lost — see [Archive tags](#archive-tags).
2. Re-proposing a mechanism that was already tried and refused on measurement. Every
   "why not X" below is a measured result, not a preference.

## The branch

There is exactly one: **`feature/hyper-diffusion-planner`**. Seven remote and thirteen
local HDP branches were deleted on 2026-07-29 after everything unmerged was pushed as an
annotated tag.

| PR | Outcome | Basis |
| --- | --- | --- |
| #305 | merged | It *is* the branch; every artifact on disk derives from it. |
| #317 | merged | Hard blocker. A metric dict built from a set literal gave 8 ranks 8 iteration orders, so **no HDP-RL replay epoch could finish under DDP**. Fix is `sorted(...)` in `utils/ddp.py`. Its second half survives a restart that lands on a mine epoch, saving a ~3h re-mine. |
| #310 | closed, superseded | `--rl_paper_exact` ships here: `a8e55afc`, `c3611a52`, `8871aa8f`, `d848517b` are all ancestors. Opt-in, default-off. |
| #308 | merged then **fully reverted** (`3d15a2e3`) | Premise contradicted by the deployment target — see below. |

### Why #308 was reverted

Its three knobs assumed the node gates on absolute probabilities. It does not: per
`node_turn_indicator_manager_review_20260729.md`, the C++ manager is **per-cycle argmax
plus a time window — no softmax, no EMA, no probability threshold**. Therefore

- `fit_probability_temperature` is a **provable no-op**: dividing all logits by one
  positive scalar is monotone and cannot change an argmax.
- `turn_indicator_opposite_direction_weight` prices probability *mass* that an argmax
  consumer is invariant to.
- `turn_indicator_implied_intent_smoothing` is a soft-label mechanism already refuted for
  this head: only **0.300%** of frames sit at a label transition.

Shipping it inert at `0.0/0.0` was also rejected — three flags that are argparse'd,
validated and sbatch-passed but cannot change a result are the same defect class this
branch had just finished removing. **Do not re-propose it.** Move the operating point
downstream, in the C++ manager.

## Pipeline

| Stage | Launcher | Notes |
| --- | --- | --- |
| Base (80-token, ego-only) | `run_hdp_ego_only_base80_node02.sbatch` | Full vehicle-parameter corpus, no SFT init. Three `is_skipped`-filtered right-turn manifests, each repeated ×10. Source lists are immutable inputs; the job never rewrites them. |
| Base (node01 variant) | `run_hdp_ego_only_base_node01.sbatch` | |
| SFT | `run_hdp_ego_only_sft_node01.sbatch` | Asserts `BASE_RUN/latest.pth` is at the expected epoch. |
| Staged SFT | `run_hdp_staged_sft_node02.sbatch` | Stage 1 removes signal feedback and adapts only the trajectory planner; each stage hands its `latest.pth` EMA to the next. |
| RL post-training | `run_hdp_rl.sbatch` | Same three manifests at the same ×10 repeat as Base, re-checked by the trainer, so an RL delta is attributable to the objective and not to a distribution shift. |
| Turn-indicator head | `run_hdp_turn_indicator_head_node01.sbatch` | Safe to run **concurrently** with RL. Per `configure_rl_trainable_parameters` (`train_hdp_rl_predictor.py:87-95`), `--rl_train_scope decoder` trains only `decoder.dit.*` and `decoder.global_route_encoder.*`, and `all` trains everything *except* `decoder.turn_indicator_predictor.*` — so the head is untouched under **both** values. |
| LR probe | `run_hdp_policy_lr_probe_node02.sbatch` | |

Entry points: `train_predictor.py`, `train_hdp_rl_predictor.py`, `valid_predictor.py`,
`valid_predictor_closed_loop.py`.

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

## What was removed, and why it changes no result

`rl_first_waypoint_gate` and its six threshold flags are gone
(`hdp_rl_first_waypoint_gate_removal_20260729.md`). It was **never on** — all 32 `args.json`
on disk carry `False` — and its 5 cm tangent floor sat *above* the phenomenon it had to
police (stop-turn first-step p95 is 0.039 m), so `mean_first_waypoint_gate_rejected_fraction`
was 0 across cycles 1–4.

Cache safety is explicit: the field is pinned in
`_FINGERPRINT_RETIRED = {"rl_first_waypoint_gate": repr(False)}`, so all 32 historical
`reward_fingerprint` values reproduce byte-for-byte and a resume of a ~2.1 TB mined cycle
still reads instead of aborting on drift.

If the gate is ever wanted back, judge standstill on **absolute lateral offset**, not
direction: below ~9 cm the logged human's own implied steer averages 50°, so direction
there is localization noise. A `|y| > 5 mm` low-speed cap rejects 0.004% of human
standstill rows. `compute_reward_weights(candidate_valid_mask=...)` is retained as the
integration point and currently has no caller.

## Reproducibility contract

- **Base is not retrained.** It is a fixed premise, not a tuning knob.
- **RL trains on exactly the Base training set**, augmentation included.
- The turn-indicator head loss is gated by `training_stage == "turn_indicator"`
  (`train_epoch.py:189-190`), so policy-only Base/SFT never evaluates it. Head-side
  changes cannot affect base reproducibility.
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

`.venv/bin/python -m pytest` → **633 passed, 15 skipped**;
`.venv/bin/python -m ruff check` → clean. CI runs `pre-commit`, which includes
`ruff format` — run `pre-commit run --all-files` before pushing. Note the system `python3`
has neither pytest nor ruff; always use `.venv/bin/python`.
