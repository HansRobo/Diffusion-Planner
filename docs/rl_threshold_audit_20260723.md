# RL threshold audit on real scenes (2026-07-23, CPU-only)

512 scenes sampled (seed 3407) from the local copy of
`path_list_valid_fullseq_from_20260622_step3.json`; no model, no GPU. Script:
`util_scripts/audit_rl_thresholds_on_real_data.py`; machine-readable report:
`artifacts/rl_threshold_audit/report_valid512_20260723.json`. Groups are one
unperturbed logged expert plus seven velocity-safe augmented variants
(gaussian offsets std 0.5 m, ramp 20, stretch 0.25).

## Gate calibration — the 5 cm floor reproduces the upstream audit on our data

| Configuration | Expert false-reject (all) | Expert false-reject (speed < 1 m/s) |
|---|---:|---:|
| Floored gate (5 cm, our default) | **0.0%** | **0.0%** |
| Unfloored gate | 5.5% | **41.2%** |

13.3% of scenes are low-speed. The unfloored tangent test rejects 41% of real
low-speed logged experts — the same pathology the source repository measured
(35.4% on their corpus) — while the floored gate rejects none. Defaults
confirmed; no adjustment needed.

## Augmentation calibration — velocity-safety holds on real trajectories

- 79.7% of scenes augmented (the rest under the 2 m/s low-speed guard).
- p99 of the maximum per-step velocity increment introduced by augmentation:
  0.40 m absolute = **0.81 sigma** of the per-step action normalization
  (`ego_velocity` std 0.5/0.25) — inside the intended ~1 sigma budget.
- **100% of augmented candidates pass the first-waypoint gate**, confirming the
  ramp's near-zero onset guarantee on real data. Defaults confirmed.

## Reward discrimination — all three objectives work; pdm_port is best aligned

| Objective | Failures | Valid-group fraction | ESS | Expert wins its group |
|---|---:|---:|---:|---:|
| native `weighted_sum` | 0 | 0.797 | 0.66 | 64.5% |
| native `gated_product` | 0 | 0.797 | 0.66 | 64.5% |
| ported `hdp_pdm` | 0 | 0.793 | 0.72 | **91.2%** |

- All three run finite on real T4 scenes; valid-group fractions match the
  augmented-scene fraction exactly (unaugmented groups are identical-reward and
  correctly discarded), and ESS shows healthy, non-collapsed weights. The AWR
  exploration arm therefore has usable gradient signal.
- **The decision-relevant finding:** under the native reward, a randomly
  offset/stretched variant of the expert beats the logged expert in 35.5% of
  groups; under the ported hdp_pdm objective only 8.8%. The driver is the EP
  term: pdm progress is a ratio capped at 1.0, so a stretched (faster)
  candidate gains nothing past the expert, while the native progress term
  rewards overtaking the expert endpoint. Systematically preferring perturbed
  experts is the "reward-hacking via overprogress" family the source
  repository built explicit penalties against. Spearman rank correlation
  between the two objectives is 0.71 — related but genuinely different.
- The two native aggregations are indistinguishable here because these
  validation scenes are mostly gate-clean; they diverge only on unsafe
  candidates.

## Consequence for the experiment ladder

`rl_reward_source=pdm_port` is promoted to a first-class arm (it is the
verbatim port of the objective behind both audited positive results, and the
only one of the three whose group ranking is strongly expert-consistent on our
data). Native arms remain for comparison; held-out selection stays frozen on
the native evaluation objective for all arms.
