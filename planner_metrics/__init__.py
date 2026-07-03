"""Planner metric helpers."""

__all__ = ["pdms_proxy", "pdms_proxy_masked"]


def __getattr__(name):
    if name in {"pdms_proxy", "pdms_proxy_masked"}:
        from planner_metrics.pdms_proxy import pdms_proxy, pdms_proxy_masked

        return {"pdms_proxy": pdms_proxy, "pdms_proxy_masked": pdms_proxy_masked}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
