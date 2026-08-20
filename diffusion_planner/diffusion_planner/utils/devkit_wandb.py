"""Optional W&B score logging for the shared T4 evaluation interfaces.

The devkit stays independent of experiment trackers.  This module owns the
optional logger boundary while the actual BEV rendering remains in the devkit.
Importing this module does not import the ``wandb`` package; callers may pass a
Lightning logger experiment explicitly or use the active W&B run lazily.
"""

from __future__ import annotations

import logging
from numbers import Real
from typing import Any, Mapping

log = logging.getLogger(__name__)

OFFICIAL_SCORE_KEYS = (
    "pdms",
    "nc",
    "dac",
    "ddc",
    "tlc",
    "ttc",
    "ep",
    "lk",
    "comfort",
    "ec",
)

REPORT_KEY_TO_WANDB_KEY = {
    "score": "pdms",
    "pdms": "pdms",
    "no_at_fault_collisions": "nc",
    "nc": "nc",
    "drivable_area_compliance": "dac",
    "dac": "dac",
    "driving_direction_compliance": "ddc",
    "ddc": "ddc",
    "traffic_light_compliance": "tlc",
    "tlc": "tlc",
    "time_to_collision_within_bound": "ttc",
    "ttc": "ttc",
    "ego_progress": "ep",
    "ep": "ep",
    "lane_keeping": "lk",
    "lk": "lk",
    "history_comfort": "comfort",
    "comfort": "comfort",
    "extended_comfort": "ec",
    "ec": "ec",
}


def define_wandb_score_metrics(
    run: Any,
    *,
    prefix: str = "devkit",
    step_metric: str | None = None,
) -> None:
    """Define the canonical score series on a W&B-like run object."""

    if step_metric is not None:
        run.define_metric(step_metric, hidden=True)
    for key in OFFICIAL_SCORE_KEYS:
        kwargs = {"summary": "max", "goal": "maximize", "hidden": False}
        if step_metric is not None:
            kwargs["step_metric"] = step_metric
        run.define_metric(f"{prefix}/{key}", **kwargs)


def score_report_to_wandb(
    report: Mapping[str, Any],
    *,
    prefix: str = "devkit",
    epoch: int | None = None,
    step: int | None = None,
    run: Any | None = None,
) -> dict[str, float]:
    """Log numeric score fields to an active or explicitly supplied run."""

    if run is None:
        import wandb

        run = wandb.run
    if run is None:
        raise RuntimeError("score_report_to_wandb requires an active W&B run")

    define_wandb_score_metrics(
        run,
        prefix=prefix,
        step_metric="epoch" if epoch is not None else None,
    )
    payload: dict[str, float] = {}
    for key, value in report.items():
        if isinstance(value, Real) and not isinstance(value, bool):
            metric_key = REPORT_KEY_TO_WANDB_KEY.get(key)
            if metric_key is not None:
                payload[f"{prefix}/{metric_key}"] = float(value)
    if payload:
        log_payload: dict[str, Any] = dict(payload)
        if epoch is not None:
            log_payload["epoch"] = int(epoch)
        run.log(log_payload, **({} if step is None else {"step": int(step)}))
    return payload


__all__ = [
    "OFFICIAL_SCORE_KEYS",
    "REPORT_KEY_TO_WANDB_KEY",
    "define_wandb_score_metrics",
    "score_report_to_wandb",
]
