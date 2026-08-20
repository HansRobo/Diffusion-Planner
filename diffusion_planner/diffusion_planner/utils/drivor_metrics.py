"""DrivoR's metric taxonomy, ported for the Diffusion-Planner training loop.

Ported from ``DrivoR/navsim/planning/training/agent_lightning_module.py``: the
``_metric_name`` table, ``_trajectory_metrics``, ``_log_selection_metrics``,
``_log_loss_dict`` and ``_global_progress_means``.  DrivoR emits these through
Lightning's ``self.log``; Diffusion-Planner aggregates a per-step loss dict and
calls ``wandb.log`` once per epoch, so the same names are produced here as plain
``{path: scalar}`` mappings and the caller does the logging.

Names are the cross-repository contract, so they are reproduced value for value:
epoch aggregates under ``train/*`` and ``val/*``, live diagnostics under
``perf/*``.  Diffusion-Planner-specific series (turn indicator, neighbour
prediction, road-border and neighbour-collision penalties, the diffusion
``ego_planning_loss``) have no counterpart in this head and are not emitted.
"""

from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist

# Live-diagnostic namespace, shared with DrivoR.
STEP_NAMESPACE = "perf"

# Implementation name -> per-epoch W&B path (DrivoR's ``_metric_name`` table,
# minus the entries whose producing branch does not exist in this head).
METRIC_NAMES: dict[str, str] = {
    "loss": "loss/total",
    "trajectory_loss": "loss/trajectory",
    "final_score_loss": "loss/scorer_total",
    "da_loss": "loss/scorer/DAC",
    "ttc_loss": "loss/scorer/TTC",
    "noc_loss": "loss/scorer/NC",
    "progress_loss": "loss/scorer/EP",
    "ddc_loss": "loss/scorer/DDC",
    "comfort_loss": "loss/scorer/Comfort",
    "human_loss": "loss/scorer/HumanTeacher",
    # Soft-target cross-entropy is H(labels) + KL(labels||prediction), and only
    # the KL term has a gradient, so the learnable remainder is reported apart
    # from the constant entropy floor.
    "logit_absmax": "scorer/logit_absmax",
    "human_kl_loss": "loss/learnable/HumanTeacher",
    "score_kl_loss": "loss/learnable/scorer_total",
    "label_entropy": "loss/learnable/label_entropy_floor",
    "min_loss0": "trajectory/error_before_refinement",
    "min_loss": "trajectory/error_after_refinement",
    # ``score``/``best_score`` are legacy loss-dict aliases; the canonical
    # oracle selection series are emitted by :func:`selection_metrics`, so these
    # must not share their keys.
    "score": "selection/legacy_score",
    "best_score": "selection/legacy_best_score",
}

# Oracle component -> the short display name used in every W&B path.
COMPONENT_DISPLAY_NAMES: dict[str, str] = {
    "no_at_fault_collisions": "NC",
    "drivable_area_compliance": "DAC",
    "driving_direction_compliance": "DDC",
    "time_to_collision_within_bound": "TTC",
    "ego_progress": "EP",
    "history_comfort": "Comfort",
    "traffic_light_compliance": "TrafficLight",
    "lane_keeping": "LaneKeeping",
    "extended_comfort": "ExtendedComfort",
}

# Per-step live series, DrivoR's ``progress_metrics`` naming.
PROGRESS_LOSS_KEYS: tuple[tuple[str, str], ...] = (
    ("noc_loss", "loss_NC"),
    ("da_loss", "loss_DAC"),
    ("ddc_loss", "loss_DDC"),
    ("ttc_loss", "loss_TTC"),
    ("progress_loss", "loss_EP"),
    ("comfort_loss", "loss_Comfort"),
    ("human_loss", "loss_HumanTeacher"),
    ("logit_absmax", "logit_absmax"),
)

PROGRESS_TRAJECTORY_KEYS: tuple[str, ...] = (
    "selected_ADE",
    "selected_FDE",
    "minADE",
    "minFDE",
    "heading_MAE",
    "final_heading_error",
)


def metric_name(prefix: str, key: str) -> str:
    """Map an implementation name to its per-epoch W&B path."""
    return f"{prefix}/{METRIC_NAMES.get(key, key)}"


def metric_path(prefix: str, name: str) -> str:
    """Return a canonical epoch aggregate path."""
    return f"{prefix}/{name}"


def step_metric(name: str) -> str:
    """Full live-diagnostic path in the shared ``perf/*`` family."""
    return f"{STEP_NAMESPACE}/{name}"


@torch.no_grad()
def trajectory_metrics(
    predictions: Mapping[str, Any], target_trajectory: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Displacement and heading metrics for the selected trajectory + proposals.

    ADE/FDE are Euclidean distances in the ego-frame XY plane.  Heading is
    reported separately and wrapped to ``[-pi, pi]`` so an equivalent angle
    across the branch cut cannot create a spurious 2*pi error -- with this
    head's (cos, sin) pose columns the wrap is exactly ``atan2``.  ``minADE``
    and ``minFDE`` measure the proposal generator; the ``selected_*`` pair
    measures the scorer's final model output.
    """
    selected = predictions.get("trajectory")
    proposals = predictions.get("proposals")
    if not all(torch.is_tensor(value) for value in (selected, proposals, target_trajectory)):
        return {}

    selected = selected.detach()
    proposals = proposals.detach()
    target = target_trajectory.detach().to(device=selected.device, dtype=selected.dtype)
    if selected.ndim != 3 or selected.shape[-1] < 2:
        raise ValueError(
            f"selected trajectory must have shape [B,T,D>=2], got {tuple(selected.shape)}"
        )
    if target.shape != selected.shape:
        raise ValueError(
            f"target/selected trajectory shape mismatch: {tuple(target.shape)} vs "
            f"{tuple(selected.shape)}"
        )
    if (
        proposals.ndim != 4
        or proposals.shape[0] != selected.shape[0]
        or proposals.shape[2:] != selected.shape[1:]
    ):
        raise ValueError(
            "proposals must have shape [B,N,T,D] matching selected trajectory, got "
            f"{tuple(proposals.shape)} and {tuple(selected.shape)}"
        )

    selected_xy_error = torch.linalg.vector_norm(
        selected[..., :2] - target[..., :2], dim=-1
    )
    proposal_xy_error = torch.linalg.vector_norm(
        proposals[..., :2] - target[:, None, :, :2], dim=-1
    )
    metrics = {
        "selected_ADE": selected_xy_error.mean(dim=-1).mean(),
        "selected_FDE": selected_xy_error[:, -1].mean(),
        "minADE": proposal_xy_error.mean(dim=-1).amin(dim=-1).mean(),
        "minFDE": proposal_xy_error[..., -1].amin(dim=-1).mean(),
    }
    if selected.shape[-1] >= 4:
        # Poses are (x, y, cos, sin): the wrapped difference of the two angles is
        # atan2 of the cross/dot products, which needs no explicit unwrapping.
        cross = selected[..., 3] * target[..., 2] - selected[..., 2] * target[..., 3]
        dot = selected[..., 2] * target[..., 2] + selected[..., 3] * target[..., 3]
        wrapped_heading_error = torch.atan2(cross, dot).abs()
        metrics.update(
            {
                "heading_MAE": wrapped_heading_error.mean(),
                "final_heading_error": wrapped_heading_error[:, -1].mean(),
            }
        )
    return metrics


#: Tolerance for calling two oracle aggregates equal.  The aggregate is a product
#: of sub-metrics that are themselves exact 0/0.5/1 rationals, so genuine ties are
#: bit-identical and this only absorbs fp32 rounding.
_TIE_EPS = 1e-6


@torch.no_grad()
def selection_metrics(loss_dict: Mapping[str, Any], prefix: str) -> dict[str, torch.Tensor]:
    """Selection quality in oracle units, including ranking metrics."""
    oracle = loss_dict.get("_oracle_total")
    chosen = loss_dict.get("_chosen_index")
    if not torch.is_tensor(oracle) or not torch.is_tensor(chosen) or oracle.ndim != 2:
        return {}

    chosen = chosen.to(device=oracle.device, dtype=torch.long).reshape(-1)
    rows = torch.arange(oracle.shape[0], device=oracle.device)
    selected = oracle[rows, chosen]
    best = oracle.amax(dim=-1)
    rank = 1 + (oracle > selected[:, None]).sum(dim=-1).float()
    # Score equality, not index equality: the oracle aggregate ties heavily (a
    # third of samples have several proposals at the maximum, and every proposal
    # that collides sits at exactly 0), so comparing argmax indices scores a
    # tied-best pick as a miss and reads as chance-level selection.  This is the
    # convention ``rank`` above already uses.
    top1 = (selected >= best - _TIE_EPS).float()
    kth = torch.topk(oracle, k=min(5, oracle.shape[1]), dim=-1).values[:, -1]
    top5 = (selected >= kth - _TIE_EPS).float()

    selection = metric_path(prefix, "selection")
    out = {
        f"{selection}/oracle_selected": selected.mean(),
        f"{selection}/oracle_best": best.mean(),
        f"{selection}/oracle_gap": (best - selected).mean(),
        f"{selection}/oracle_rank": rank.mean(),
        f"{selection}/top1_hit": top1.mean(),
        f"{selection}/top5_hit": top5.mean(),
    }

    components = loss_dict.get("_oracle_components")
    component_names = tuple(loss_dict.get("_oracle_component_names", ()))
    if (
        torch.is_tensor(components)
        and components.ndim == 3
        and components.shape[-1] == len(component_names)
    ):
        selected_components = components[rows, chosen]
        best_components = components[rows, oracle.argmax(dim=-1)]
        for index, public_name in enumerate(component_names):
            name = COMPONENT_DISPLAY_NAMES.get(public_name, public_name)
            out[f"{selection}/selected_{name}"] = selected_components[:, index].mean()
            # These distributions make scorer-label pathologies visible: a large
            # BCE alone cannot tell whether NC is difficult or simply dominated
            # by one class / soft-label regime.
            out[metric_path(prefix, f"oracle/mean_{name}")] = components[..., index].mean()
            out[metric_path(prefix, f"oracle/low_{name}_fraction")] = (
                (components[..., index] < 0.5).float().mean()
            )
            out[metric_path(prefix, f"oracle/best_{name}")] = best_components[:, index].mean()
    return out


@torch.no_grad()
def step_metrics(
    loss_dict: Mapping[str, Any], traj_metrics: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The small live-diagnostic set that makes a slow launch visibly active."""
    oracle = loss_dict.get("_oracle_total")
    if not torch.is_tensor(oracle):
        return {}
    chosen = loss_dict["_chosen_index"].to(device=oracle.device, dtype=torch.long).reshape(-1)
    rows = torch.arange(oracle.shape[0], device=oracle.device)
    selected = oracle[rows, chosen]
    best = oracle.amax(dim=-1)
    out: dict[str, torch.Tensor] = {
        "loss_total": loss_dict["loss"].detach(),
        "oracle_selected": selected.mean(),
        "oracle_best": best.mean(),
        "oracle_gap": (best - selected).mean(),
    }
    # ``selected_*`` measures the scorer output while ``min*`` answers whether
    # the proposal set contains a good trajectory at all; without both, a long
    # epoch cannot be debugged before its first aggregate is emitted.
    for name in PROGRESS_TRAJECTORY_KEYS:
        value = traj_metrics.get(name)
        if torch.is_tensor(value):
            out[name] = value
    for key, name in PROGRESS_LOSS_KEYS:
        value = loss_dict.get(key)
        if torch.is_tensor(value) and value.numel() == 1:
            out[name] = value.detach()
    return out


@torch.no_grad()
def global_progress_means(metrics: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reduce live diagnostics over the whole DDP global batch in one collective.

    Logging rank zero's local batch would make the curves depend on 1/world_size
    of the samples actually processed at a step; stacking every scalar into one
    tiny all-reduce keeps monitoring faithful to the optimizer batch without
    adding a collective per metric.
    """
    names = list(metrics)
    if not names:
        return {}
    values = torch.stack([metrics[name].detach().float().reshape(()) for name in names])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= float(dist.get_world_size())
    return dict(zip(names, values.unbind()))


def epoch_metrics(
    loss_dict: Mapping[str, Any], prefix: str, traj_metrics: Optional[Mapping[str, Any]] = None
) -> dict[str, float]:
    """One clean epoch-level hierarchy of scalars, keyed by their W&B path.

    ``loss_dict`` may hold plain floats (Diffusion-Planner's epoch aggregate) or
    tensors (a single step).  ``_``-prefixed entries are internal plumbing and,
    for ``score``/``best_score``, superseded by the canonical selection series.
    """
    out: dict[str, float] = {}
    extra = dict(traj_metrics or {})
    has_oracle_selection = "_oracle_total" in loss_dict or any(
        name.startswith(f"{prefix}/selection/") for name in extra
    )
    for key, value in loss_dict.items():
        key = str(key)
        if key.startswith("_"):
            continue
        if has_oracle_selection and key in {"score", "best_score"}:
            continue
        scalar = _as_float(value)
        if scalar is None:
            continue
        out[metric_name(prefix, key)] = scalar

    total = _as_float(loss_dict.get("loss"))
    if total is not None:
        if prefix == "val":
            out["val/total"] = total
        elif prefix == "train":
            out["train/planning_epoch"] = total
            out["train/loss_epoch"] = total
    if prefix == "val":
        trajectory_loss = _as_float(loss_dict.get("trajectory_loss"))
        if trajectory_loss is not None:
            out["val/ego_planning_loss"] = trajectory_loss

    for name, value in extra.items():
        scalar = _as_float(value)
        if scalar is None:
            continue
        # ``selection/*`` and ``oracle/*`` paths already carry their prefix.
        out[name if name.startswith(f"{prefix}/") else metric_path(prefix, f"trajectory/{name}")] = (
            scalar
        )
        if prefix == "val":
            # The comparison-facing aliases DrivoR also emits.
            canonical = {
                "selected_ADE": "val/ade",
                "selected_FDE": "val/fde_4s",
                "heading_MAE": "val/heading_error",
            }.get(name)
            if canonical is not None:
                out[canonical] = scalar
    return out


def _as_float(value: Any) -> Optional[float]:
    if torch.is_tensor(value):
        return float(value.detach().float().reshape(-1)[0]) if value.numel() == 1 else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


__all__ = [
    "COMPONENT_DISPLAY_NAMES",
    "METRIC_NAMES",
    "PROGRESS_LOSS_KEYS",
    "PROGRESS_TRAJECTORY_KEYS",
    "STEP_NAMESPACE",
    "epoch_metrics",
    "global_progress_means",
    "metric_name",
    "metric_path",
    "selection_metrics",
    "step_metric",
    "step_metrics",
    "trajectory_metrics",
]
