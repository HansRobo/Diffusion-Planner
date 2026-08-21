"""DrivoR training objective for the Diffusion-Planner DrivoR head.

Ported from ``DrivoR/navsim/agents/drivoR/layers/losses/drivor_loss.py``, T4
branch (``t4_local_terms`` / ``t4_finish``).  The two halves are kept separate
so a training step can back-propagate the oracle-free part immediately after
the forward -- holding the graph -- while the PDM oracle scores the very
proposals it was just handed, and finish the six BCE heads when the labels
arrive.  Gradients of ``local_loss`` plus gradients of ``_deferred_loss`` are
exactly the serial gradient.

What is *not* carried over from the reference: the Hungarian agent-detection
loss, the BEV semantic cross-entropy, the drivable-area raster head and the
``diversity_loss`` term (DrivoR runs it at ``inter_weight = 0``).  This head
only emits an ego trajectory, so none of those have a consumer.
"""

from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from diffusion_planner.model.module.drivor_scorer import HEAD_BY_METRIC
from diffusion_planner.utils.drivor_oracle import ORACLE_METRIC_NAMES, TTC_UNDEFINED

# Per-head weights inside the scorer objective.  DrivoR's ``trajectory_pdm_weight``
# is all-ones for the six heads it trains, and the aggregate's behaviour weights
# (0/5/5/2) apply to *selection*, not to supervision -- every head has to be
# learnable regardless of how much it moves the final score.
SCORER_HEAD_WEIGHTS: dict[str, float] = {
    "no_at_fault_collisions": 1.0,
    "drivable_area_compliance": 1.0,
    "driving_direction_compliance": 1.0,
    "time_to_collision_within_bound": 1.0,
    "ego_progress": 1.0,
    "history_comfort": 1.0,
}

# Loss-dict key per oracle metric, matching DrivoR's ``loss_key`` table.
LOSS_KEY_BY_METRIC: dict[str, str] = {
    "no_at_fault_collisions": "noc_loss",
    "drivable_area_compliance": "da_loss",
    "driving_direction_compliance": "ddc_loss",
    "time_to_collision_within_bound": "ttc_loss",
    "ego_progress": "progress_loss",
    "history_comfort": "comfort_loss",
}


def three_to_two_classes(x: torch.Tensor) -> torch.Tensor:
    """NAVSIM's 0.5 "at fault but not the ego's fault" class -> 0."""
    return torch.where(x == 0.5, torch.zeros_like(x), x)


@torch.no_grad()
def _label_entropy(target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Bernoulli entropy of a soft label -- the irreducible part of a BCE.

    Reported by both halves of the split loss: the demonstration head in
    :meth:`DrivoRLoss.local_terms` and the six scorer heads in
    :meth:`DrivoRLoss.finish`.
    """
    probability = target.float().clamp(1e-6, 1.0 - 1e-6)
    entropy = -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log())
    if weight is None:
        return entropy.mean()
    return (entropy * weight).sum() / weight.sum().clamp(min=1.0)


class DrivoRLoss(torch.nn.Module):
    """WTA trajectory regression + six BCE scorer heads (+ demonstration head)."""

    def __init__(
        self,
        trajectory_weight: float = 1.0,
        final_score_weight: float = 1.0,
        prev_weight: float = 1.0,
        label_smoothing: float = 0.02,
    ) -> None:
        super().__init__()
        self.trajectory_weight = float(trajectory_weight)
        self.final_score_weight = float(final_score_weight)
        self.prev_weight = float(prev_weight)
        if not 0.0 <= float(label_smoothing) < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.label_smoothing = float(label_smoothing)

    # -- oracle-free half --------------------------------------------------
    def local_terms(self, pred: Mapping[str, Any], target_trajectory: torch.Tensor) -> dict:
        """Every term that is a function of the prediction and the GT alone.

        Args:
            pred: the DrivoR head's output dict.
            target_trajectory: ``[B, T, 4]`` expert future (x, y, cos, sin).
        """
        proposal_list = pred["proposal_list"]
        target = target_trajectory

        trajectory_loss = target.new_zeros(())
        min_loss_list = []
        for proposals_i in proposal_list:
            # Winner-takes-all: each sample is trained only through the proposal
            # already closest to the demonstration, which is what keeps the 64
            # proposals from collapsing onto the mean trajectory.
            min_loss = (
                torch.linalg.norm(proposals_i - target[:, None], dim=-1, ord=1)
                .mean(-1)
                .amin(1)
                .mean()
            )
            trajectory_loss = self.prev_weight * trajectory_loss + min_loss
            min_loss_list.append(min_loss)

        proposals = pred["proposals"]
        # Optional demonstration teacher.  The six PDM heads are distilled from a
        # rule metric that cannot separate roughly half the proposals; this head
        # is distilled from the demonstration instead.  It supervises only the
        # scorer -- the target is detached and the proposals enter the scorer
        # detached -- so the trajectory generator is untouched and this is not an
        # imitation loss on the selected output.
        human_logit = pred["pred_logit"].get("human_closeness")
        human_loss = trajectory_loss.new_zeros(())
        human_entropy = trajectory_loss.new_zeros(())
        if human_logit is not None:
            with torch.no_grad():
                proposal_error = torch.linalg.vector_norm(
                    proposals.detach()[..., :2] - target[:, None, :, :2], dim=-1
                ).mean(-1)
                human_target = (1.0 / (1.0 + proposal_error)).float()
            human_loss = F.binary_cross_entropy_with_logits(human_logit.float(), human_target)
            human_entropy = _label_entropy(human_target)

        return {
            "trajectory_loss": trajectory_loss,
            "min_loss_list": min_loss_list,
            "human_loss": human_loss,
            "human_entropy": human_entropy,
            "local_loss": self.trajectory_weight * trajectory_loss
            + self.final_score_weight * human_loss,
        }

    # -- oracle-dependent half ---------------------------------------------
    def finish(
        self, local: Mapping[str, Any], pred: Mapping[str, Any], oracle: torch.Tensor
    ) -> dict:
        """Complete the objective once the ``[B, N, 7]`` oracle labels exist."""

        trajectory_loss = local["trajectory_loss"]
        min_loss_list = local["min_loss_list"]
        human_loss = local["human_loss"]
        human_entropy = local["human_entropy"]
        proposals = pred["proposals"]

        oracle = oracle.detach().to(device=proposals.device, dtype=torch.float32)
        if tuple(oracle.shape[:2]) != tuple(proposals.shape[:2]):
            raise ValueError(
                "oracle shape must match proposals in [batch, proposal], got "
                f"{tuple(oracle.shape[:2])} vs {tuple(proposals.shape[:2])}"
            )
        if oracle.shape[-1] != len(ORACLE_METRIC_NAMES):
            raise ValueError(
                f"oracle width must be {len(ORACLE_METRIC_NAMES)}, got {oracle.shape[-1]}"
            )
        oracle_by_name = {
            name: oracle[..., index] for index, name in enumerate(ORACLE_METRIC_NAMES)
        }

        smoothing = self.label_smoothing

        def _smooth(value: torch.Tensor) -> torch.Tensor:
            # BCE against a hard label is minimised only at |logit| -> inf.
            # Smoothing puts the optimum at a finite logit, which is what keeps
            # the heads from drifting into saturation over a long run.
            if smoothing <= 0.0:
                return value
            return value * (1.0 - smoothing) + 0.5 * smoothing

        score_losses: dict[str, torch.Tensor] = {}
        entropy_terms = []
        scorer_logits = []
        for metric_name in HEAD_BY_METRIC:
            target = oracle_by_name[metric_name]
            head_name = HEAD_BY_METRIC[metric_name]
            logit = pred["pred_logit"].get(head_name)
            if logit is None:
                raise KeyError(f"metric {metric_name!r} requires head {head_name!r}")
            scorer_logits.append(logit)

            if metric_name in {"no_at_fault_collisions", "driving_direction_compliance"}:
                target = three_to_two_classes(target)
            if metric_name == "time_to_collision_within_bound":
                # The sentinel marks "no evaluable step", not "no infraction";
                # supervising it would teach the head the sentinel's frequency.
                mask = (target != TTC_UNDEFINED).to(dtype=torch.float32)
                loss = F.binary_cross_entropy_with_logits(
                    logit.float(),
                    _smooth(target.clamp(0.0, 1.0)),
                    weight=mask,
                    reduction="sum",
                ) / mask.sum().clamp(min=1.0)
                # The floor of a BCE is the entropy of the target it is actually
                # trained against, so it has to be the *smoothed* label: with
                # smoothing 0.02 a hard label's minimum is H(0.01) = 0.056, not 0.
                entropy_terms.append(_label_entropy(_smooth(target.clamp(0.0, 1.0)), mask))
            else:
                loss = F.binary_cross_entropy_with_logits(logit.float(), _smooth(target))
                entropy_terms.append(_label_entropy(_smooth(target)))
            score_losses[metric_name] = loss

        weighted = [SCORER_HEAD_WEIGHTS[name] * value for name, value in score_losses.items()]
        score_head_sum = torch.stack(weighted).sum()
        # Weighted the same way as ``score_head_sum``, or ``score_kl_loss`` below
        # stops being a difference of comparable quantities.
        scorer_entropy = torch.stack(
            [SCORER_HEAD_WEIGHTS[name] * value for name, value in zip(score_losses, entropy_terms)]
        ).sum()

        final_score_loss = score_head_sum
        if pred["pred_logit"].get("human_closeness") is not None:
            final_score_loss = final_score_loss + human_loss

        result = {
            "loss": self.trajectory_weight * trajectory_loss
            + self.final_score_weight * final_score_loss,
            "trajectory_loss": trajectory_loss,
            "human_loss": human_loss,
            "final_score_loss": final_score_loss,
            "min_loss0": min_loss_list[0],
            "min_loss": min_loss_list[-1],
            "_local_loss": local["local_loss"],
            "_deferred_loss": self.final_score_weight * score_head_sum,
        }
        result.update({LOSS_KEY_BY_METRIC[name]: value for name, value in score_losses.items()})

        # Soft-target cross-entropy is H(labels) + KL(labels || prediction) and
        # only the KL term has a gradient, so the learnable remainder is
        # reported apart from the constant entropy floor.
        result["logit_absmax"] = torch.stack(
            [logit.detach().float().abs().amax() for logit in scorer_logits]
        ).amax()
        result["score_kl_loss"] = (score_head_sum - scorer_entropy).clamp(min=0.0).detach()
        result["label_entropy"] = scorer_entropy.detach()
        if pred["pred_logit"].get("human_closeness") is not None:
            result["human_kl_loss"] = (human_loss - human_entropy).clamp(min=0.0).detach()
            result["label_entropy"] = (result["label_entropy"] + human_entropy).detach()

        component_names = tuple(HEAD_BY_METRIC)
        result["_oracle_components"] = torch.stack(
            [oracle_by_name[name] for name in component_names], dim=-1
        ).detach()
        result["_oracle_component_names"] = component_names

        oracle_total = oracle_by_name["score"]
        chosen = pred["pdm_score"].detach().argmax(dim=1)
        rows = torch.arange(oracle_total.shape[0], device=oracle_total.device)
        result.update(
            {
                "score": oracle_total[rows, chosen].mean(),
                "best_score": oracle_total.amax(dim=-1).mean(),
                "_oracle_total": oracle_total.detach(),
                "_chosen_index": chosen.detach(),
            }
        )
        return result

    # -- serial convenience path -------------------------------------------
    def forward(
        self,
        pred: Mapping[str, Any],
        target_trajectory: torch.Tensor,
        oracle: torch.Tensor,
    ) -> dict:
        """The two halves back-to-back: the original serial computation."""
        local = self.local_terms(pred, target_trajectory)
        return self.finish(local, pred, oracle)


__all__ = [
    "DrivoRLoss",
    "LOSS_KEY_BY_METRIC",
    "SCORER_HEAD_WEIGHTS",
    "three_to_two_classes",
]
