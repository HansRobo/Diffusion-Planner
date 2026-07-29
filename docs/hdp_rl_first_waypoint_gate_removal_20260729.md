# Removing the RL first-waypoint candidate gate (2026-07-29)

`rl_first_waypoint_gate` and its six threshold flags are deleted. This is a code
removal with **no behavioural effect on any run ever launched**, and it closes a
default-ON safeguard that measurement showed was not doing anything.

## Why it went

**1. It was never on.** All 32 `args.json` on disk (`workspaces/wang/runs/*`,
`outputs/*`) carry `rl_first_waypoint_gate: False`. `train_config.py` defaulted it to
`True` and `run_hdp_rl.sbatch` defaulted `FIRST_WAYPOINT_GATE=True`, so every run was
False only because `--rl_paper_exact` (or an explicit pin) switched it off. The
gate-on configuration was never trained, never evaluated, never shipped.

**2. The implementation was a superseded form, measured inert.** The gate rejected a
candidate whose first step exceeded absolute limits (0.25 m step / 0.20 m lateral /
0.05 m reverse) or sat more than 75° off-tangent, with the tangent test suppressed
below `tangent_min_step_m = 0.05`. That 5 cm floor was set *above the phenomenon it
had to police*: measured stop-turn first-step displacement p95 is **0.039 m**, so at
standstill nearly every candidate fell below the floor and the tangent test never
fired. Upstream evidence: `mean_first_waypoint_gate_rejected_fraction` was **0 across
cycles 1–4**. The check was off exactly where steering jitter lives.

**3. Its own replacement is also known-defective**, so there was no settled design to
keep. The 2026-07-25 upstream replacement (`_implied_first_step_steer_rad`, reject
when `atan(L·2|y|/s²) > 0.64 rad`, floor lowered to 0.005 m) fails the same way one
level deeper: the implied `|y|` tolerance collapses as `s²`, so on 1.6M logged-human
first steps it rejects **the ground truth** at 89.2% of 5 mm steps and 0% above
25 cm. It is also structurally inert in eval, where K=1 and the gate needs
`count > 1` survivors.

Keeping a default-ON mechanism that advertises a real-vehicle safeguard it does not
provide is worse than not having one.

## What the measured replacement is, if the gate is ever wanted back

Do **not** re-port either historical criterion. The discriminating quantity at
standstill is **absolute lateral offset**, not direction or geometry — below ~9 cm the
human's own implied steer averages 50° (p95 89.4°), i.e. direction there is pure
localization noise for human and policy alike.

Measured on 1.6M logged-human first steps versus the policy's population:

| population (first step < 5 cm) | mean \|y\| | p99 | max |
| --- | --- | --- | --- |
| logged human | 0.196 mm | 1.607 mm | 7.779 mm |
| policy | 6.978 mm | 87.585 mm | **192.871 mm** |

A low-speed cap at **`|y| > 5 mm`** rejects **0.004%** of human standstill rows and
catches essentially all of the policy tail. The old `max_lateral_m = 0.20` cap rejects
0.000% of human rows and is inert.

Two constraints on doing it:

- It changes what the mine produces, so it is a **new-cycle item with a fresh
  `rl_replay_dir`**, never a mid-campaign edit.
- `compute_reward_weights(candidate_valid_mask=...)` is deliberately **retained** as
  the integration point. It is correct, tested, and it is the structural fix for the
  overlay-resurrection bug class (weights and validity computed from one mask). It
  currently has no caller.

## Cache compatibility (this protects a multi-terabyte cycle)

`rl_first_waypoint_gate` was a `_FINGERPRINT_FIELDS` member, and `reward_fingerprint`
builds values with `getattr(args, name, None)`. Deleting the field outright would turn
every historical fingerprint entry from `'False'` into `'None'`, and
`CycleReplayReader` **aborts** on any fingerprint drift — so a resume of an otherwise
valid ~2.1 TB mined cycle would fail instead of reading it. The field is therefore
pinned in `_FINGERPRINT_RETIRED = {"rl_first_waypoint_gate": repr(False)}`, which
reproduces all 32 historical fingerprints exactly. Covered by
`test_fingerprint_keeps_the_retired_first_waypoint_gate_field`.

## Why this changes no result

Both call sites passed the gate mask into `compute_reward_weights`. With the gate off
the mask was all-True, and for an all-True mask the masked branch is mathematically
identical to `candidate_valid_mask=None`: same group mean, same Bessel-corrected std
(`Σ(x−μ)²/(n−1)`), same finite check, and the extra `valid_count >= 2` test is vacuous
at `n = num_generations = 32`. At `n = 1` both branches reject the group anyway (the
masked one on `valid_count >= 2`, the unmasked one on a NaN std). So dropping the
argument reproduces every run on disk.

## Surface removed

| file | what |
| --- | --- |
| `hdp_rl_utils.py` | `first_waypoint_candidate_gate()` (72 lines) |
| `hdp_rl_epoch.py` | import + both call sites + both `candidate_valid_mask=` args |
| `train_config.py` | 7 fields |
| `train_hdp_rl_predictor.py` | 7 argparse flags + the positivity-validation loop |
| `hdp_rl_paper_exact.py` | its `PaperExactSetting` row |
| `hdp_rl_replay.py` | fingerprint member → replaced by the frozen retired entry |
| `slurm/run_hdp_rl.sbatch` | env default, `paper_pin` line, torchrun flag |
| `util_scripts/audit_rl_thresholds_on_real_data.py` | gate-calibration part (now 2 parts, not 3) |
| `tests/` | 5 gate tests removed, 1 cache-compatibility test added |

Test suite: **418 → 414** passing (−5 gate tests, +1 new), no failures. `ruff check`
clean on every touched file.
