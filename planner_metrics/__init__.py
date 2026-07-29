"""Neutral metrics library: raw EPDMS-style subscores + geometry + config,
importable without depending on ``rlvr``."""

from __future__ import annotations

from planner_metrics.aggregate import (  # noqa: F401
    compute_subscores_batch,
    compute_subscores_scene_batch,
)
from planner_metrics.config import RewardConfig  # noqa: F401  (re-export)
from planner_metrics.geometry import *  # noqa: F401,F403
from planner_metrics.subscores import *  # noqa: F401,F403

_PDMS_EXPORTS = {
    "add_synthetic_epdms",
    "epdms_human_filtered",
    "pdms_proxy",
    "pdms_proxy_masked",
    "synthetic_epdms",
}


def __getattr__(name):
    if name in _PDMS_EXPORTS:
        from planner_metrics import pdms_proxy as _pdms_proxy

        return getattr(_pdms_proxy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
