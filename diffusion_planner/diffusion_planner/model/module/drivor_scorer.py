"""DrivoR proposal scorer: six independent PDM heads + demonstration head.

Ported from ``DrivoR/navsim/agents/drivoR/score_module/scorer.py``.  The
auxiliary DrivoR branches (``double_score``, agent-box prediction, drivable-area
raster, BEV semantic map) are intentionally not carried over: this head only has
to select one ego trajectory, so nothing but the six PDM logits and the optional
demonstration logit is required.
"""

import torch
import torch.nn as nn

# The six component heads DrivoR learns, in the model's parameter-name order.
# ``history_comfort`` is the public metric name of the ``comfort`` head; keeping
# both spellings in one table is what lets the oracle, the loss and the metric
# panel agree without a second mapping.
DRIVOR_HEAD_METRICS: tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "history_comfort",
)

HEAD_METRIC_ORDER = DRIVOR_HEAD_METRICS

HEAD_BY_METRIC: dict[str, str] = {
    "no_at_fault_collisions": "no_at_fault_collisions",
    "drivable_area_compliance": "drivable_area_compliance",
    "driving_direction_compliance": "driving_direction_compliance",
    "time_to_collision_within_bound": "time_to_collision_within_bound",
    "ego_progress": "ego_progress",
    "history_comfort": "comfort",
}

# Model-selection weight order, matching DrivoR's (noc, dac, ddc, ttc, ep, comfort).
SCORE_WEIGHT_ORDER: tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "history_comfort",
)


def _head(d_model: int, d_ffn: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, 1))


class Scorer(nn.Module):
    """Six independent BCE logit heads plus an optional demonstration head."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        human_teacher_weight: float = 0.0,
        logit_bound: float = 0.0,
    ):
        super().__init__()

        self.score_num = 6
        self.pred_score = nn.ModuleDict(
            {
                "no_at_fault_collisions": _head(d_model, d_ffn),
                "drivable_area_compliance": _head(d_model, d_ffn),
                "time_to_collision_within_bound": _head(d_model, d_ffn),
                "ego_progress": _head(d_model, d_ffn),
                "driving_direction_compliance": _head(d_model, d_ffn),
                "comfort": _head(d_model, d_ffn),
            }
        )

        # Optional seventh head: how close a proposal is to the demonstrated
        # trajectory.  The six PDM heads leave roughly half the proposals tied
        # at the maximum aggregate score, and inside a tie set argmax is
        # arbitrary; this head is what orders them.  It is only built when the
        # behaviour profile actually gives it weight.
        self.human_weight = float(human_teacher_weight or 0.0)
        if self.human_weight < 0.0:
            raise ValueError("human_teacher_weight must be non-negative")
        # 0 disables the bound (bit-identical to the original head outputs).
        self.logit_bound = float(logit_bound or 0.0)
        if self.logit_bound < 0.0:
            raise ValueError("logit_bound must be non-negative")
        self.human_head = _head(d_model, d_ffn) if self.human_weight > 0.0 else None

    def forward(self, proposal_feature: torch.Tensor) -> dict[str, torch.Tensor]:
        """``proposal_feature``: [B, N, d_model] -> per-head logits [B, N]."""

        pred_logit: dict[str, torch.Tensor] = {}
        for key, head in self.pred_score.items():
            raw = head(proposal_feature).squeeze(-1)
            if self.logit_bound > 0.0:
                # Hard logit bound: cap * tanh(raw / cap).  BCE against a hard
                # (even smoothed) label is minimised at logit -> +/-inf, and the
                # catastrophic tail is one confidently-wrong batch carrying a
                # loss of ~|logit| per element.  The bound leaves the smoothed
                # optimum (~4.6) and the normal operating range untouched but
                # makes that tail mathematically impossible.  The demonstration
                # head stays unbounded -- its soft target never saturates.
                raw = self.logit_bound * torch.tanh(raw / self.logit_bound)
            pred_logit[key] = raw
        if self.human_head is not None:
            pred_logit["human_closeness"] = self.human_head(proposal_feature).squeeze(-1)
        return pred_logit


def aggregate_pdm_score(
    pred_logit: dict[str, torch.Tensor],
    score_weights: torch.Tensor,
    human_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate six scorer heads exactly like the original PDMS.

    The original NAVSIM ``PDMScorer`` treats NC and DAC as the two
    multiplicative metrics.  DDC, TTC, EP and Comfort form a normalized weighted
    arithmetic mean.  ``noc``/``dac`` are retained in the six-value profile for
    API compatibility but are not exponents in the original aggregate.  Work in
    FP32 so mixed-precision logits cannot change proposal ordering at the
    selection boundary.

    Args:
        pred_logit: per-head logits, each ``[B, N]``.
        score_weights: ``[B, 6]`` in ``SCORE_WEIGHT_ORDER``.
        human_weight: additive weight for the demonstration head.

    Returns:
        ``(pdm_score [B, N], score_components [B, N, 6])``
    """

    logits = torch.stack(
        tuple(pred_logit[HEAD_BY_METRIC[name]] for name in SCORE_WEIGHT_ORDER), dim=-1
    )
    score_components = logits.float().sigmoid()
    weights = score_weights.float()

    multiplicative = score_components[..., 0] * score_components[..., 1]
    behavior_weights = weights[:, 2:]
    behavior_denominator = behavior_weights.sum(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    behavior_score = (score_components[..., 2:] * behavior_weights[:, None, :]).sum(
        dim=-1
    ) / behavior_denominator[:, None]
    pdm_score = multiplicative * behavior_score

    # The demonstration term is added to, not folded into, the PDMS aggregate:
    # roughly half of the 64 proposals sit at exactly the same PDMS, so a small
    # additive term orders the tie set without moving anything across the PDMS
    # ordering.
    human_logit = pred_logit.get("human_closeness")
    if human_logit is not None and human_weight > 0.0:
        pdm_score = pdm_score + human_weight * human_logit.float().sigmoid()
    return pdm_score, score_components


__all__ = [
    "DRIVOR_HEAD_METRICS",
    "HEAD_BY_METRIC",
    "HEAD_METRIC_ORDER",
    "SCORE_WEIGHT_ORDER",
    "Scorer",
    "aggregate_pdm_score",
]
