"""Loss and metrics for next-turn-indicator classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .turn_indicator import TurnIndicatorModel


def compute_turn_indicator_loss(
    model: TurnIndicatorModel,
    batch: dict[str, torch.Tensor],
    transition_weight: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cross-entropy loss, correct count, and valid-label count.

    The first future sample is the next-state target. Raw values 1, 2, and 3
    map to DISABLE, LEFT, and RIGHT; zero or unsupported values are ignored.
    Samples whose target differs from the last valid history state receive the
    specified transition weight.
    """
    logits = model(batch)
    target = batch["turn_indicators_future"][:, 0].to(torch.long)
    current = batch["turn_indicators"][:, -1].to(torch.long)
    valid = (target >= 1) & (target <= 3)
    current_valid = (current >= 1) & (current <= 3)
    transition = valid & current_valid & (current != target)
    class_target = (target - 1).clamp(0, 2)
    per_sample_loss = F.cross_entropy(logits, class_target, reduction="none")
    sample_weight = torch.where(
        transition,
        per_sample_loss.new_tensor(transition_weight),
        per_sample_loss.new_tensor(1.0),
    )
    sample_weight = sample_weight * valid
    valid_count = valid.sum()
    loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
    correct = ((logits.argmax(dim=-1) == class_target) & valid).sum()
    return loss, correct, valid_count
