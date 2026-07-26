# W&B monitoring design for the gate-fixed AWR campaign (2026-07-24)

Everything below lives in the W&B project
`advanced-technology-department/original-dp-awr`.

## Layer 1 — always-on campaign status run

Run: **`gatefix-campaign-e100-status`** (fixed run id; daemon restarts resume
the same run, so the history never fragments).

Producer: `rlvr/autoresearch/tools/campaign_wandb_heartbeat.py`, restarted by a
keep-alive loop; (re)launch with
`bash rlvr/autoresearch/launch_wandb_heartbeat.sh`. Read-only: it parses stage
logs, the health monitor snapshot, `cycle_state.jsonl` and overlay manifests
every 60 s. Local mirror of the latest sample:
`campaign_e100/wandb_heartbeat_state.json`.

Published every 60 s:

| Group | Metrics |
|---|---|
| Lifecycle | `phase` / `phase_code` (mining=5, replay=4, probe=3, context build=2, transition=1, **stalled=0**, complete=6), `active_probe`, `active_stage_log`, `last_status_line` |
| Progress | `global_epoch`, `mining_scene_fraction`, `replay_batch_fraction`, `context_build_fraction`, `phase_progress_per_hour`, `phase_eta_hours`, `campaign_epochs_done`, `campaign_fraction` |
| Live quality | `mining_det_reward`, `mining_best_reward`, `mining_loss`, `replay_loss` (parsed from rank-0 progress lines — this is what makes the W&B-silent mine stages visible) |
| Cycle outcomes | `completed_cycles`, `latest_cycle_reward`, `latest_cycle_reward_delta`, `latest_cycle_epoch`, `overlay_active_targets`, `overlay_gate_masked_candidates` |
| Health | `health_ok`, `health_reason`, `progress_age_seconds`, `supervisor_alive`, `keeper_alive`, `supervisor_error_exits`, process counts, GPU util/mem, `nvme_free_tib` |

W&B alerts (cooldown 1 h per topic): supervisor crash loop (≥3 error exits in
15 min), stalled/unhealthy for ≥2 ticks, per-cycle completion (info), campaign
completion (info).

Reading it: `phase_code` flatlining at 0, `health_ok` at 0, or
`supervisor_error_exits` climbing means a real problem. Low GPU utilisation
during `phase=mining` with `mining_scene_fraction` rising is the benchmarked
serial reward bottleneck, not a stall (see
`acceleration_benchmark/mining_runtime/selection.json`).

## Layer 2 — native per-stage trainer runs

* Replay stages (dense per-epoch metrics: loss, eval reward, collision,
  first-waypoint gate rates, stop-turn slices):
  `original-dp-awr-plannerrft-cycleNN-eXX-eXX`, unchanged.
* Mine stages: the trainer only emits W&B data at epoch summaries, so live
  mining progress always comes from Layer 1. `run_mine` in
  `run_plannerrft_full_to_epoch100.sh` now passes `--wandb`
  (`…-cycleNN-mine-eE`) so each mine also leaves a durable end-of-epoch
  summary run; this takes effect the next time the keeper (re)starts the
  supervisor, because a running bash keeps its in-memory copy.

## Operational cautions

* The sensitivity probes refuse to start when any process command line matches
  `rlvr.train_awr|eval_awr_full_distributed|diag_plannerrft_sampler_ablation`.
  Never put those literals in ad-hoc shells; bracket one character
  (`…ablatio[n]`) as done throughout the heartbeat and
  `check_gatefix_campaign.sh`. The same rule caused a supervisor crash loop on
  2026-07-24 when a monitoring loop matched the guard.
* One-shot local verdict without W&B:
  `bash rlvr/autoresearch/check_gatefix_campaign.sh`.
* A node reboot kills training, keeper and heartbeat alike; relaunch the
  keeper and `launch_wandb_heartbeat.sh` — both resume from committed
  artifacts into the same W&B run.

## 2026-07-25 update — jitterfix campaign

Run: **`jitterfix-campaign-e100-status`** (project unchanged). Launch with
`RUN_ROOT=.../plannerrft_jitterfix_e100 bash rlvr/autoresearch/launch_wandb_heartbeat.sh`.

New metrics, all read from the newest committed eval summary so they always
correspond to a finished epoch:

| Metric | Meaning |
|---|---|
| `stop_turn_implied_steer_p95_rad` | **the headline jitter number** — implied front-wheel angle at standstill; was ~1.45 rad (83 deg) median on mined candidates |
| `stop_turn_displacement_p95_m` | standstill first-step length (~0.039 m measured) |
| `first_waypoint_lateral_p95_m` | first-step lateral offset |
| `gate_rejected_fraction`, `stop_turn_gate_rejected_fraction` | **must be > 0 now.** Flat 0 means the safeguard is off again — that is exactly the 2026-07-23 failure |
| `latest_cycle_vetoed` | 1 when the cycle kept its incumbent because the non-regression gate fired; a flat reward curve with this at 1 is the gate working, not a stall |

Reading it: `gate_rejected_fraction == 0` across a whole cycle is a red flag, not
a clean run. Expect roughly 0.29 of low-speed non-anchor candidates rejected.

## Operational cautions (extended 2026-07-25)

* The bracketing rule applies to **regex** guards too, not just literal greps.
  The keeper's `ALIVE_PATTERN` is a regex, so `run_..._epoch10[0]\.sh` still
  matches the plain text `run_..._epoch100.sh` sitting in some other shell's
  command line. A monitoring shell that merely *mentions* the entrypoint
  filename makes `supervisor_alive` return true and the keeper polls forever
  instead of retrying — this silently stalled the 2026-07-25 relaunch for 13
  minutes. Never put the entrypoint or daemon filename in an ad-hoc shell;
  target leftovers by PID instead.
* `[[ -s file ]]` cannot distinguish an empty JSON list (`[]`, 2 bytes) from real
  content. Scene-list guards must test `jq 'length'`. A remapped list can
  legitimately be empty — the 20260623 `x2_dev` right-turn scenes have no
  20260707 counterpart (0% coverage), and passing that empty list to the trainer
  crashed all 8 ranks three times before the keeper's cooldown.
