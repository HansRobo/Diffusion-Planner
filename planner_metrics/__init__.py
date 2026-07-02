from planner_metrics.aggregate import compute_planning_metrics  # noqa: F401
from planner_metrics.config import MetricConfig, default_metric_config  # noqa: F401
from planner_metrics.subscores import (  # noqa: F401
    compute_comfort_metrics,
    compute_collision_score,
    compute_progress_metrics,
    compute_score,
)


def __getattr__(name):
    if name in {"pdms_proxy", "pdms_proxy_masked"}:
        from planner_metrics.pdms_proxy import pdms_proxy, pdms_proxy_masked

        return {"pdms_proxy": pdms_proxy, "pdms_proxy_masked": pdms_proxy_masked}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
