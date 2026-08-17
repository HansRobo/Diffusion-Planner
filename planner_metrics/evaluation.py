"""Common result types for scenario-specific metric evaluation.

The Open-loop evaluation runner consumes :class:`MetricEvaluation` so metric
implementations can return both aggregateable per-sample scores and optional
metric-specific details without coupling the runner to individual metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class MetricEvaluation:
    """Per-sample scores and optional metric-specific detail fields.

    Attributes:
        scores: Mapping from score names to one-dimensional tensors with one
            value per evaluated sample. These values are aggregated into the
            validation summary.
        details: Mapping from detail sections to fields. Scalar fields are
            one-dimensional tensors aligned with ``scores``. Trajectory and
            polyline fields may be higher-rank tensors or a Python list of
            variable-length values. These values are written to per-sample
            detail records and are not included in the aggregate summary.
    """

    scores: dict[str, torch.Tensor]
    details: dict[str, dict[str, Any]] = field(default_factory=dict)


def _tensor_to_json(sampled: torch.Tensor) -> Any:
    sampled = sampled.detach().cpu()
    if sampled.ndim == 0:
        item = sampled.item()
        return bool(item) if sampled.dtype == torch.bool else item
    return sampled.tolist()


def detail_value_to_json(value: Any, batch_index: int) -> Any:
    """Serialize one sample of a detail field for JSONL output.

    Zero-dimensional tensors become Python numbers or bools. Higher-rank
    tensors become nested lists. Variable-length Python lists are indexed
    per sample; any tensor at that index is converted the same way.
    """
    if torch.is_tensor(value):
        return _tensor_to_json(value[batch_index])
    if isinstance(value, (list, tuple)):
        sampled = value[batch_index]
        if torch.is_tensor(sampled):
            return _tensor_to_json(sampled)
        return sampled
    raise TypeError(f"unsupported detail value type: {type(value)!r}")


__all__ = ["MetricEvaluation", "detail_value_to_json"]
