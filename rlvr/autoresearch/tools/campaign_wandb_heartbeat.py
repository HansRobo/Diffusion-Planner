#!/usr/bin/env python3
"""Always-on W&B mirror of the gate-fixed PlannerRFT/AWR campaign.

Design (docs/wandb_campaign_monitoring.md):

  Layer 1 (this daemon)   one persistent W&B run, fixed run id, resumes across
                          daemon restarts.  Every 60 s it publishes the phase
                          (mining/replay/probe/context build/transition/
                          stalled/complete), in-phase progress and live reward
                          parsed from the stage logs, throughput and ETA,
                          process/health/crash counters, GPU/NVMe state and
                          per-cycle outcomes.  It also raises W&B alerts on
                          stall, health degradation, supervisor crash loops
                          and cycle completion.
  Layer 2 (trainer runs)  the per-cycle replay runs continue to carry the
                          dense per-epoch training metrics; mine stages gain
                          native end-of-epoch summary runs once the supervisor
                          restarts with the --wandb mine flag.

The daemon is read-only: it parses logs and state files and never touches
training state.  Every pgrep pattern is bracketed so this process can never
match the mutual-exclusion guards the sensitivity probes use to detect
competing GPU work.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

PHASES = {
    "complete": 6,
    "mining": 5,
    "replay": 4,
    "sensitivity_probe": 3,
    "context_build": 2,
    "supervisor_transition": 1,
    "stalled": 0,
}

MINE_LINE = re.compile(
    r"epoch (\d+) rank0 scene (\d+)/(\d+) loss=([\d.eE+-]+) "
    r"Rdet=([+\-][\d.]+) Rbest=([+\-][\d.]+)"
)
REPLAY_LINE = re.compile(r"epoch (\d+) replay (\d+)/(\d+) loss=([\d.eE+-]+)")
CONTEXT_LINE = re.compile(r"compressed ([\d,]+)/([\d,]+)")
PROBE_NAMES = (
    ("beta sensitivity", "beta_sensitivity"),
    ("solver sensitivity", "solver_step_sensitivity"),
    ("candidate-generation sensitivity", "candidate_sensitivity"),
    ("loader concurrency", "eval_loader_benchmark"),
    ("deployment", "deployment_audit"),
)


def pgrep_count(pattern: str) -> int:
    result = subprocess.run(
        ["pgrep", "-fc", pattern], capture_output=True, text=True
    )
    try:
        return int(result.stdout.strip() or 0)
    except ValueError:
        return 0


def gpu_stats() -> tuple[float, float, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    utils: list[float] = []
    mems: list[float] = []
    for line in result.stdout.strip().splitlines():
        try:
            util, mem = line.split(",")
            utils.append(float(util))
            mems.append(float(mem))
        except ValueError:
            continue
    if not utils:
        return 0.0, 0.0, 0.0
    return sum(utils) / len(utils), max(utils), max(mems) / 1024.0


def tail_text(path: Path, max_bytes: int = 131072) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - max_bytes))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def newest(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.is_file()]
    return max(existing, key=lambda p: p.stat().st_mtime) if existing else None


class RateTracker:
    """Progress-fraction velocity and ETA from successive samples."""

    def __init__(self) -> None:
        self.key: str | None = None
        self.samples: deque[tuple[float, float]] = deque(maxlen=30)

    def update(self, key: str, fraction: float, now: float) -> tuple[float, float]:
        if key != self.key:
            self.key = key
            self.samples.clear()
        self.samples.append((now, fraction))
        if len(self.samples) < 2:
            return float("nan"), float("nan")
        (t0, f0), (t1, f1) = self.samples[0], self.samples[-1]
        if t1 <= t0 or f1 <= f0:
            return float("nan"), float("nan")
        per_second = (f1 - f0) / (t1 - t0)
        eta_hours = (1.0 - f1) / per_second / 3600.0
        return per_second * 3600.0, eta_hours


class Alerter:
    """W&B alerts with per-topic cooldowns; never raises."""

    def __init__(self, wandb_module, cooldown_seconds: float = 3600.0) -> None:
        self.wandb = wandb_module
        self.cooldown = cooldown_seconds
        self.last_sent: dict[str, float] = {}

    def send(self, topic: str, title: str, text: str, warn: bool = True) -> None:
        now = time.time()
        if now - self.last_sent.get(topic, 0.0) < self.cooldown:
            return
        try:
            level = (
                self.wandb.AlertLevel.WARN if warn else self.wandb.AlertLevel.INFO
            )
            self.wandb.alert(title=title, text=text, level=level)
            self.last_sent[topic] = now
        except Exception as error:  # noqa: BLE001
            print(f"alert '{topic}' failed: {error!r}", flush=True)


def collect(run_root: Path, campaign: Path, target_epoch: int) -> dict:
    now = time.time()
    metrics: dict[str, float | str] = {"heartbeat_unix_time": now}

    trainers = pgrep_count(r"[-]m rlvr[.]train_awr")
    compressors = pgrep_count(r"build_compressed_replay_contex[t]")
    probes = pgrep_count(r"diag_plannerrft_sampler_ablatio[n]")
    evals = pgrep_count(r"eval_awr_full_distribute[d]")
    keeper = pgrep_count(r"run_plannerrft_autonomous_campaig[n]")
    supervisor = pgrep_count(r"run_plannerrft_full_to_epoch10[0]")
    entry = pgrep_count(r"gatefix_clean_to_epoch10[0]")
    metrics.update(
        trainer_processes=trainers,
        compressor_processes=compressors,
        probe_processes=probes,
        eval_processes=evals,
        keeper_alive=float(keeper > 0),
        supervisor_alive=float(supervisor > 0 or entry > 0),
    )

    util_mean, util_max, mem_max_gib = gpu_stats()
    metrics.update(
        gpu_util_mean_pct=util_mean,
        gpu_util_max_pct=util_max,
        gpu_mem_max_gib=mem_max_gib,
        nvme_free_tib=shutil.disk_usage(run_root).free / 1024**4,
    )

    completion = campaign / f"epoch{target_epoch}_completion.json"
    phase = "supervisor_transition"
    stage_logs = (
        sorted(run_root.glob("cycle01_*.log"))
        + sorted(campaign.glob("cycle*_mine.log"))
        + sorted(campaign.glob("cycle*_replay.log"))
    )
    active_log = newest(stage_logs)
    if completion.is_file():
        phase = "complete"
    elif trainers > 0 and active_log is not None:
        text = tail_text(active_log)
        mine = MINE_LINE.findall(text)
        replay = REPLAY_LINE.findall(text)
        if replay and (not mine or text.rfind(" replay ") > text.rfind("rank0 scene")):
            epoch, done, total, loss = replay[-1]
            phase = "replay"
            metrics.update(
                global_epoch=int(epoch),
                replay_batch_fraction=int(done) / max(int(total), 1),
                replay_loss=float(loss),
            )
        elif mine:
            epoch, done, total, loss, rdet, rbest = mine[-1]
            phase = "mining"
            metrics.update(
                global_epoch=int(epoch),
                mining_scene_fraction=int(done) / max(int(total), 1),
                mining_loss=float(loss),
                mining_det_reward=float(rdet),
                mining_best_reward=float(rbest),
            )
        else:
            phase = "mining" if "mine" in active_log.name else "replay"
        metrics["active_stage_log"] = active_log.name
    elif compressors > 0:
        phase = "context_build"
        context_logs = sorted(
            run_root.glob("cycle01_mine/20*/replay_context_zstd_v1/logs/rank_0000.log")
        )
        for log in context_logs:
            found = CONTEXT_LINE.findall(tail_text(log, 4096))
            if found:
                done_s, total_s = found[-1]
                metrics["context_build_fraction"] = int(
                    done_s.replace(",", "")
                ) / max(int(total_s.replace(",", "")), 1)
    elif probes > 0 or evals > 0:
        phase = "sensitivity_probe"
    elif supervisor == 0 and keeper == 0 and entry == 0:
        phase = "stalled"

    status = campaign / "status.log"
    if status.is_file():
        text = tail_text(status)
        metrics["supervisor_error_exits"] = float(
            len(re.findall(r"supervisor exit status=[1-9]", text))
        )
        lines = [line for line in text.strip().splitlines() if line.strip()]
        if lines:
            metrics["last_status_line"] = lines[-1][-220:]
            if phase in ("sensitivity_probe", "supervisor_transition"):
                for needle, label in PROBE_NAMES:
                    if needle in lines[-1]:
                        metrics["active_probe"] = label
                        break

    metrics["phase"] = phase
    metrics["phase_code"] = PHASES[phase]

    health = campaign / "health_monitor_latest.json"
    if health.is_file():
        try:
            payload = json.loads(health.read_text())
            metrics["health_ok"] = float(payload.get("health") == "healthy")
            metrics["health_reason"] = str(payload.get("reason", ""))
            metrics["progress_age_seconds"] = float(
                payload.get("progress", {}).get("age_seconds", -1)
            )
        except (json.JSONDecodeError, OSError):
            pass

    state = campaign / "cycle_state.jsonl"
    if state.is_file():
        rows = [
            json.loads(line)
            for line in state.read_text().splitlines()
            if line.strip()
        ]
        if rows:
            latest = max(rows, key=lambda row: int(row["cycle"]))
            metrics.update(
                completed_cycles=float(len(rows)),
                latest_cycle=float(latest["cycle"]),
                latest_cycle_reward=float(latest["best_train_reward"]),
                latest_cycle_reward_delta=float(
                    latest["best_train_reward_delta_from_start"]
                ),
                latest_cycle_epoch=float(latest["best_train_epoch"]),
            )
            # A veto means the cycle kept its incumbent rather than committing a
            # regression; surface it so a flat reward curve is never mistaken for
            # a stall.
            metrics["latest_cycle_vetoed"] = float(
                str(latest.get("selection_kind", "")) == "non_regression_veto_incumbent"
            )
    else:
        metrics["completed_cycles"] = 0.0

    # Standstill steering jitter — the metric this campaign exists to reduce.
    #
    # Deliberately sourced from the per-cycle non-regression verdict rather than
    # the eval summaries: the aggregated summary.json does NOT carry the
    # implied-steer diagnostic (only the per-scene scenes.json does), so reading
    # it here silently published nothing.  The verdict also happens to be the
    # number the commit decision was actually made on.
    verdicts = sorted(campaign.glob("cycle*_replay_*/*/deployment_full10/non_regression_verdict.json"))
    verdicts += sorted(run_root.glob("cycle01_replay_*/*/deployment_full10/non_regression_verdict.json"))
    latest_verdict = newest(verdicts)
    if latest_verdict is not None:
        try:
            payload = json.loads(latest_verdict.read_text())
            metrics["non_regression_verdict_veto"] = float(
                payload.get("verdict") == "veto"
            )
            for row in payload.get("checks") or []:
                if row.get("check") != "standstill_steer_not_worse":
                    continue
                for field, metric_key in (
                    ("candidate", "stop_turn_infeasible_steer_fraction"),
                    ("baseline", "stop_turn_infeasible_steer_fraction_baseline"),
                    ("delta", "stop_turn_infeasible_steer_fraction_delta"),
                ):
                    value = row.get(field)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        metrics[metric_key] = float(value)
                metrics["jitter_check_skipped"] = float(bool(row.get("skipped")))
        except (json.JSONDecodeError, OSError):
            pass

    overlays = sorted(campaign.glob("cycle*_positive_overlay/overlay_manifest.json")) + sorted(
        run_root.glob("cycle01_positive_overlay/overlay_manifest.json")
    )
    latest_overlay = newest(overlays)
    if latest_overlay is not None:
        try:
            payload = json.loads(latest_overlay.read_text())
            metrics["overlay_active_targets"] = float(payload.get("active_targets", 0))
            metrics["overlay_gate_masked_candidates"] = float(
                payload.get("source_zero_masked_candidates", 0)
            )
        except (json.JSONDecodeError, OSError):
            pass

    epochs_done = 10.0 * float(metrics.get("completed_cycles", 0.0))
    if phase == "replay" and "global_epoch" in metrics:
        epochs_done = float(metrics["global_epoch"]) - 1.0 + float(
            metrics.get("replay_batch_fraction", 0.0)
        )
    elif phase == "mining" and "global_epoch" in metrics:
        epochs_done = float(metrics["global_epoch"]) - 1.0 + float(
            metrics.get("mining_scene_fraction", 0.0)
        )
    metrics["campaign_epochs_done"] = epochs_done
    metrics["campaign_fraction"] = epochs_done / float(target_epoch)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--target-epoch", type=int, default=100)
    parser.add_argument("--project", default="original-dp-awr")
    parser.add_argument("--entity", default="advanced-technology-department")
    parser.add_argument("--run-name", default="gatefix-campaign-e100-status")
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()

    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        id=args.run_name,
        resume="allow",
        tags=["campaign-status", "gatefix", "pipeline-heartbeat"],
        config={
            "run_root": str(args.run_root),
            "campaign": str(args.campaign),
            "target_epoch": args.target_epoch,
        },
    )
    print(f"heartbeat run: {run.url}", flush=True)
    state_path = args.campaign / "wandb_heartbeat_state.json"
    rate = RateTracker()
    alerter = Alerter(wandb)
    error_exit_history: deque[tuple[float, float]] = deque(maxlen=60)
    previous_cycles = -1.0
    degraded_ticks = 0

    while True:
        started = time.time()
        try:
            metrics = collect(args.run_root, args.campaign, args.target_epoch)
            phase = str(metrics.get("phase"))
            fraction_key = {
                "mining": "mining_scene_fraction",
                "replay": "replay_batch_fraction",
                "context_build": "context_build_fraction",
            }.get(phase)
            if fraction_key and fraction_key in metrics:
                key = f"{phase}:{metrics.get('global_epoch', 0)}"
                per_hour, eta = rate.update(
                    key, float(metrics[fraction_key]), started
                )
                metrics["phase_progress_per_hour"] = per_hour
                metrics["phase_eta_hours"] = eta

            exits = float(metrics.get("supervisor_error_exits", 0.0))
            error_exit_history.append((started, exits))
            recent = [
                value
                for stamp, value in error_exit_history
                if started - stamp <= 900
            ]
            if len(recent) >= 2 and recent[-1] - recent[0] >= 3:
                alerter.send(
                    "crash_loop",
                    "AWR campaign: supervisor crash loop",
                    f"{recent[-1] - recent[0]:.0f} supervisor error exits in 15 "
                    f"minutes; last status: {metrics.get('last_status_line', '')}",
                )

            unhealthy = phase == "stalled" or metrics.get("health_ok") == 0.0
            degraded_ticks = degraded_ticks + 1 if unhealthy else 0
            if degraded_ticks >= 2:
                alerter.send(
                    "stall",
                    "AWR campaign: stalled or unhealthy",
                    f"phase={phase} health_ok={metrics.get('health_ok')} "
                    f"reason={metrics.get('health_reason', '')} last="
                    f"{metrics.get('last_status_line', '')}",
                )

            cycles = float(metrics.get("completed_cycles", 0.0))
            if previous_cycles >= 0 and cycles > previous_cycles:
                alerter.send(
                    f"cycle_{cycles:.0f}",
                    f"AWR campaign: cycle {cycles:.0f}/10 complete",
                    f"reward={metrics.get('latest_cycle_reward')} "
                    f"delta={metrics.get('latest_cycle_reward_delta')} "
                    f"epoch={metrics.get('latest_cycle_epoch')}",
                    warn=False,
                )
            previous_cycles = cycles

            numeric = {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
            wandb.log(numeric)
            for key, value in metrics.items():
                run.summary[key] = value
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(metrics, indent=1, sort_keys=True))
            tmp.replace(state_path)
            if phase == "complete":
                alerter.send(
                    "complete",
                    "AWR campaign: epoch-100 target reached",
                    "strict completion artifact present",
                    warn=False,
                )
                print("campaign complete; final heartbeat sent", flush=True)
                break
        except Exception as error:  # noqa: BLE001 - the feed must survive anything
            print(f"heartbeat iteration failed: {error!r}", flush=True)
        time.sleep(max(5.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    main()
