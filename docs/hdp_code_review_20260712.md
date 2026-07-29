# HDP Branch Code Review — 4th round (2026-07-12)

- Target: `feature/hyper-diffusion-planner`, HEAD = `f033c28` (working tree clean).
- Baseline for the branch diff: `fix/all-data-model-quality-fixes` (413 files, +12.7k/−98.9k, 104 commits).
- Prior findings were consolidated in `code_review_findings_20260711.md` (+ the "third pass" commit `42e1183`). All fixes from those rounds are merged at this HEAD; the stale worktree branch `audit/hdp-third-pass-20260711` (`cc33481`) is byte-equivalent to `42e1183` and can be deleted.
- **This round therefore concentrated on the never-audited surface: the 96 commits / +6.3k lines landed after `42e1183`** (RL trainer evolution, reward geometry batching, policy selection/early stopping, grouped closed-loop eval, Slurm launchers), plus a dangling-reference sweep of the whole branch.
- Verification baseline at HEAD: full test suite **382 passed, 15 skipped**; `ruff check` and `compileall` clean.

Severity legend: **P1** fix before relying on the affected path · **P2** should fix, bounded blast radius · **P3** minor/hygiene. Confirmed = mechanism verified in code; Plausible = mechanism real, trigger conditional.

---

## P1 — correctness issues on live paths

### 1. Strict resume-compat list omits the safety reward weights (Confirmed)
`diffusion_planner/diffusion_planner/train.py:312/318` — `assert_checkpoint_compatible`'s `training_fields` includes risk/follow/lane/progress for **both** the optimization (`rl_reward_w_*`) and held-out (`rl_eval_reward_w_*`) families, but not `rl_reward_w_safety` / `rl_eval_reward_w_safety`. A strict resume with a changed `--rl_reward_w_safety` passes validation and **silently changes the RL objective mid-run**; a changed `--rl_eval_reward_w_safety` silently breaks comparability of `valid_reward_mean` against the checkpoint's stored best score and the baseline `selection_score`. `advantage_eps` (enters the reward-weight formula) and `rl_full_eval_utd` (drives patience counting, see #4) are also unchecked.
**Fix:** add the two safety weights (and consider `advantage_eps`, `rl_full_eval_utd`) to the tuple.

### 2. A run started without the baseline preflight can never be resumed (Confirmed)
`diffusion_planner/train_hdp_rl_predictor.py:1143-1146` — strict resume unconditionally requires `source_baseline_metrics.json`, but that file is only written when a *fresh* run had `rl_validate_before_training=True` (default True, launcher default True). Start a run with the flag off, or lose the sidecar, and **every resume attempt raises FileNotFoundError forever** — and the flag itself is in the strict-compat list, so it cannot be flipped on resume. Auto-resume tooling would crash-loop.
**Fix:** when the flag was recorded False in the checkpoint's args.json, allow resume with `baseline_valid_loss = inf` / `best = -inf` instead of requiring the artifact.

### 3. README points RL training at a deleted script (Confirmed)
`README.md:62` — "Official HDP-RL … using `train_grpo_predictor.py`" — that script was removed on this branch; the entry point is `train_hdp_rl_predictor.py`. Verified this is the only nonexistent script referenced in README.

---

## P2 — bounded correctness / robustness issues

### 4. Launcher post-run verification fails legitimate resumes (Confirmed)
`diffusion_planner/slurm/run_hdp_rl.sbatch:369` — the verification block asserts `float(rows[0]["valid_reward_mean"])` is finite. `rows[0]` is the *first epoch of the whole run history*; resuming a run whose history was produced with `rl_full_eval_utd > 1` (legal: the field is not strict-compat-checked, see #1) leaves that cell empty → `float('')` raises → a **successful** training run is reported as failed and `RL_RUN_OK` never appears.
**Fix:** verify the first *full-eval* row (or the last row) instead of `rows[0]`.

### 5. Best-model selection score mixes incomparable scales (Plausible)
`diffusion_planner/train_hdp_rl_predictor.py:118` (`best_valid_score_from_rows`) and `:1310-1314` — the fallback chain `reward_mean → epdms_total → −valid_loss_ego` mixes units (~0–9.5 vs 0–1 vs negative). Protected in-loop by the `run_full_eval` gate, but: resume best-score reconstruction takes a `max` over rows that may span configs/scales; `epdms_total == 0.0` exactly falls through to `−ego_loss`; and the logged `valid/selection_score` series changes units on off-cadence epochs.
**Fix:** store the score *kind* alongside the value (or only ever compare reward-scale scores) and log NaN on off-cadence epochs.

### 6. EMA deepcopies the DDP wrapper — currently safe, structurally fragile (Plausible)
`diffusion_planner/train_hdp_rl_predictor.py:992→1026` and `diffusion_planner/diffusion_planner/train.py:717→729` — `ModelEma(model)` is constructed **after** DDP wrapping, so `ema.ema` is a deepcopy of the DDP wrapper: EMA checkpoints carry `module.` prefixes (all current consumers strip them), and rank-0-only eval (`closed_loop_validate`) forwards through a DDP module. This only works because the model has **no buffers** (with any future `register_buffer`, DDP's default `broadcast_buffers=True` turns the rank-0-only forward into a collective → all ranks deadlock) and because torch tolerates `deepcopy(DDP)`.
**Fix:** build the EMA from `ddp.get_model(...)` and store/eval unwrapped; keeps checkpoints prefix-free too.

### 7. `args.json` records the wrong `fused_optimizer` after fallback (Confirmed)
`diffusion_planner/train_hdp_rl_predictor.py:1013-1018` — the fused-AdamW `TypeError` fallback flips `args.fused_optimizer = False` *after* rank 0 wrote `args.json` (line 887). Run provenance is wrong and strict resume validates against the stale value. (Also: modern torch raises `RuntimeError` at step time rather than `TypeError` at construction in some unsupported configs, which would bypass this fallback entirely.)
**Fix:** resolve the effective value before dumping `args.json`.

### 8. Eval reward-weight defaults exist in three places (Confirmed)
`diffusion_planner/diffusion_planner/hdp_rl_epoch.py:431-437` — `validate_hdp_reward_policy` hardcodes `(0.0, 1.0, 3.0, 2.5, 3.0)` as getattr fallbacks, duplicating `TrainConfig` (`train_config.py:143-147`) and the sbatch defaults. Any future retuning must hit all three or held-out scoring silently drifts for callers whose args lack the fields.
**Fix:** pull the fallbacks from `TrainConfig.__dataclass_fields__` or make the fields required.

### 9. THW is scored fully safe for a stopped ego behind a leader (Plausible — design review requested)
`diffusion_planner/diffusion_planner/hdp_rl_utils.py:463` — `thw_score = where(leader_present & (ego_speed > 0.1), thw_score, 1.0)`: a candidate that stops 0.3 m behind the lead vehicle gets THW = 1.0. The near-contact state is only penalized via occupancy *if* the leader qualifies as a stopped neighbor (speed & displacement thresholds), otherwise risk reads safe in stop-and-go traffic. THW at v≈0 is legitimately undefined — but consider a clearance-based floor when `leader_gap` is small.

---

## P3 — metrics, hygiene, minor

### 10. `progress_ratio` diagnostic unbounded below (Confirmed)
`hdp_rl_utils.py:1029` — clamped to ≤2.0 but not from below; one reversing candidate can drag the `reward_progress_ratio_score` mean arbitrarily negative. Metrics-only (the reward uses the [0,1]-clamped score). Fix: `clamp(-1.0, 2.0)`.

### 11. W&B profile/interval prediction vs. actual steps (Confirmed)
`hdp_rl_epoch.py:337-359` — the profile decision uses `requested_updates`, the crossing commit uses actual `optimizer_steps`. Around skipped batches the timing profile is either paid for nothing (extra `cuda.Event`/synchronize) or missing from the logged step → holes in the `train_step/time_*` series. Metrics-only.

### 12. Grouped eval writes unsanitized area names to disk (Plausible)
`scenario_generation/grouped_closed_loop_eval.py:181-207` — `videos/{metric_group}/{area_name}` and `by_area/{area_name}_metrics_summary.json` use the raw area string, while `wandb_closed_loop.py:54` sanitizes `/`. An area name containing a path separator nests directories unexpectedly and can break `mp4.relative_to(out_dir)`. Sanitize once, shared by both writers.

### 13. `PYTHONWARNINGS=error` in the production launcher (Plausible — deliberate tradeoff)
`run_hdp_rl.sbatch:144` — a first-occurrence warning on a rare code path (e.g. a numpy `RuntimeWarning` in a DataLoader worker at epoch N) aborts a multi-day run. Consider a curated `-W error::DeprecationWarning`-style list instead of the blanket error.

### 14. `resume_model` loads the whole checkpoint onto GPU per rank (Confirmed)
`diffusion_planner/diffusion_planner/utils/train_utils.py:229` — `torch.load(map_location=device)` materializes model + optimizer + EMA tensors on the GPU before `load_state_dict` copies; transient double-residency at resume on all ranks simultaneously. `load_weights_only` already uses `map_location="cpu"` — do the same here.

### 15. Doubled shape check in the decoder (Confirmed)
`diffusion_planner/diffusion_planner/model/module/decoder.py:552-561` — the `elif` validating the cached global-route-condition shape is immediately followed by an identical unconditional `if`; the `elif` branch is fully shadowed. Keep one.

### 16. Leftover helper duplication (Confirmed, known-open since round 1)
`boolean()` ×3 (`train_predictor.py:9`, `train_hdp_rl_predictor.py:64`, `valid_predictor.py:23`) and `_train_config_default()` ×2 — move to a shared util. Also `hdp_rl_utils.heading_to_cos_sin_if_needed` overlaps the idempotent `train_epoch.heading_to_cos_sin` (only difference: `[..., :4]` truncation).

### 17. Parity-test coverage note
The single-scene reward helpers (`_hdp_lane_score`, `_relative_progress_score`, …) are used only by tests; production runs the parallel batched implementations. This is properly guarded by parity tests (`test_hdp_core.py:130-242`) — good pattern — but the parity scenarios use axis-aligned headings and straight-line experts. Add rotated-heading / curved-expert parity cases, since batched-vs-single divergence on curves was exactly round-2 finding #6.

### 18. Small observations (no action strictly required)
- `train_hdp_rl_predictor.py:1118-1124`: `_wandb_global_step` fallback (`init_epoch × len(loader) × updates_per_rollout`) overestimates when batches were skipped; only affects old checkpoints lacking `global_step`.
- Periodic epoch checkpoints are written as `epochNNNN/best_model.pth` — the name suggests best-model selection but it's the cadence snapshot (historical convention; a comment would do).
- `validate_compiled_candidate_batch`'s hardcoded 2048 H100 limit is a documented workaround for a TorchInductor mis-compilation — fine as a guard, but leave a pointer to the upstream issue when one exists.
- `_wandb_scalar` drops `numpy` integer types (`np.int64` is not `int`); currently harmless because all row values are built with python casts — keep it that way.

---

## Verified sound in this round (do not re-flag)

- RL loss plumbing: group/batch/none reward normalization incl. equal-reward-group discard; DDP world-size × global-valid-count scaling; BC anchor keeps the graph identical for `static_graph`; `valid_group_fraction` uses `.any(dim=1)`.
- `commit_ema_policy_update`: EMA-of-DDP parameter pairing (strict zip), relative-L2 diagnostic, optimizer-state clear matches the launcher's post-run assert.
- `sample_group`: eval/train mode restore, `_sample_steps` restore in `finally`, cached encoding + cached global-route condition repeat-interleave shapes (incl. the `[::n]` expert slice), turn-head skip on both training and inference paths.
- Reward geometry: vmap'ed `_collision_and_leader_terms` (shapes/gathers checked), reference-leader gap logic, stopped-neighbor occupancy fusion with rear attenuation, road-border chunked exact clearance, risk = min(TTC, THW, OCC) over time, risk-gated progress.
- `BatchAlignedDistributedSampler` padding math (relies on parent `__iter__` reading overridden `num_samples`/`total_size` — coupled to torch internals but correct in 2.11).
- Strict resume: per-rank RNG capture/restore with world-size guard; scheduler stepped before checkpointing; absolute-epoch save cadence; wandb id persistence.
- `valid_predictor` reward-eval flag remapping (`reward_eval_*` → `rl_eval_*`) is complete; TF32/device/ddp propagated onto the checkpoint config.
- All `run_hdp_rl.sbatch` flags exist in the trainer's argparse (re-verified); `TrainConfig` has no unused fields; npz loads are `allow_pickle=False` with key allowlists throughout the touched paths.
- Post-audit deltas to loss/normalizer/unicycle/subscores are behavior-preserving optimizations (`_detached_integral` single-cumsum form, per-device+dtype stat caches, `lru_cache`d scalar DTD with `.clone()`, `clearance_only` fast path).

## Process note

This round's findings were produced by a single-session inline review of the post-`42e1183` diff (the previously audited surface was only spot-checked), after a parallel multi-agent sweep was aborted for cost reasons. Residual risk: files whose post-audit delta I only skimmed structurally — `data_augmentation.py` internals (guarded by the reworked `test_data_augmentation.py`), `closed_loop_cli.py`, `test_hdp_core.py` additions — and the C++ side (`cpp_tools`), unchanged in this window.
