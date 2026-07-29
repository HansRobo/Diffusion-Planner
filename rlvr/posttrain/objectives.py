"""Candidate-weighting objectives.

The trainer reduces ``(per_candidate_loss * weights).sum()`` over one scene
group, so an objective here *is* a function from that group's rewards to
per-candidate weights.  Register a new one and select it with
``--awr_objective``; nothing else in the trainer changes.

Contract:
    fn(rewards, **kwargs) -> (weights: torch.Tensor[K], diagnostics: dict[str, float])

``kwargs`` is the union of the knobs the trainer passes (see
``rlvr.awr.compute_awr_weights``); ignore what you do not need.  Diagnostics
must carry the *same keys on every call*, empty groups included: the trainer
reduces them across DDP ranks and a per-rank key set desynchronises the reduce.
"""

from __future__ import annotations

import importlib
import math
from typing import Callable

import numpy as np
import torch

WeightFn = Callable[..., tuple[torch.Tensor, dict[str, float]]]

_REGISTRY: dict[str, WeightFn] = {}
# Imported on first use so this module stays importable from rlvr.awr itself.
_LAZY: dict[str, tuple[str, str]] = {"awr": ("rlvr.awr", "compute_awr_weights")}

ZERO_TOL = 1e-8


def register(name: str, fn: WeightFn) -> WeightFn:
    _REGISTRY[str(name)] = fn
    return fn


def objective(name: str) -> Callable[[WeightFn], WeightFn]:
    def decorate(fn: WeightFn) -> WeightFn:
        return register(name, fn)

    return decorate


def available() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))


def resolve(name: str) -> WeightFn:
    key = str(name)
    if key not in _REGISTRY:
        if key not in _LAZY:
            raise ValueError(
                f"unknown objective {key!r}; available: {', '.join(available())}"
            )
        module, attribute = _LAZY[key]
        _REGISTRY[key] = getattr(importlib.import_module(module), attribute)
    return _REGISTRY[key]


def _valid_mask(totals: np.ndarray, candidate_valid_mask) -> np.ndarray:
    if candidate_valid_mask is None:
        return np.isfinite(totals)
    mask = (
        candidate_valid_mask.detach().cpu().numpy()
        if isinstance(candidate_valid_mask, torch.Tensor)
        else candidate_valid_mask
    )
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size != totals.size:
        raise ValueError(
            f"candidate_valid_mask/reward count mismatch: {mask.size} != {totals.size}"
        )
    return np.isfinite(totals) & mask


def diagnostics(
    totals: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray,
    *,
    safe_candidate_count: int,
    invalid_candidate_count: int,
    all_zero_group: bool,
    dropped_all_zero_group: bool,
    extra: dict[str, float] | None = None,
) -> dict[str, float]:
    """The shared diagnostic block, with a key set independent of the inputs."""

    active = np.abs(weights) > 0.0
    magnitude = np.abs(weights[active])
    total = float(magnitude.sum())
    ess = float(total * total / (magnitude**2).sum()) if total > 0.0 else 0.0
    count = int(magnitude.size)
    reward = totals[valid]
    zero_reward = valid & (np.abs(totals) <= ZERO_TOL)
    result = {
        "reward_mean": float(reward.mean()) if reward.size else float("nan"),
        "reward_std": float(reward.std(ddof=1)) if reward.size > 1 else 0.0,
        "reward_max": float(reward.max()) if reward.size else float("nan"),
        "valid_group": float(reward.size > 0),
        "weight_mean": float(magnitude.mean()) if count else 0.0,
        "weight_max": float(magnitude.max()) if count else 0.0,
        "weight_min": float(magnitude.min()) if count else 0.0,
        "negative_weight_count": float((weights < 0.0).sum()),
        "effective_sample_size": ess,
        "active_weight_count": float(count),
        "effective_sample_size_fraction": float(ess / count) if count else 0.0,
        "top1_weight_share": float(magnitude.max() / total) if total > 0.0 else 0.0,
        "zero_reward_weight_share": (
            float(np.abs(weights[zero_reward]).sum() / total) if total > 0.0 else 0.0
        ),
        "safe_candidate_count": float(safe_candidate_count),
        "first_waypoint_invalid_candidate_count": float(invalid_candidate_count),
        "first_waypoint_invalid_candidate_fraction": float(
            invalid_candidate_count / max(1, totals.size)
        ),
        "all_zero_reward_group": float(all_zero_group),
        "dropped_all_zero_group": float(dropped_all_zero_group),
    }
    result.update(extra or {})
    return result


def _prepare(rewards, reward_config, candidate_valid_mask, safe_only):
    from rlvr.awr import safe_candidate_mask

    totals = np.asarray([float(r.total) for r in rewards], dtype=np.float64)
    valid = _valid_mask(totals, candidate_valid_mask)
    invalid_count = int((np.isfinite(totals) & ~valid).sum())
    safe = safe_candidate_mask(rewards, reward_config)
    safe_count = int((valid & safe).sum())
    if safe_only and safe_count > 0:
        valid = valid & safe
    all_zero = bool(valid.any() and np.all(np.abs(totals[valid]) <= ZERO_TOL))
    return totals, valid, safe_count, invalid_count, all_zero


@objective("grpo")
def compute_grpo_weights(
    rewards,
    beta: float = 1.0,
    weight_clip: float = 20.0,
    normalize: bool = True,
    min_group_std: float = 1e-5,
    safe_only: bool = False,
    reward_config=None,
    positive_advantage_only: bool = False,
    positive_advantage_margin: float = 0.0,
    drop_all_zero_groups: bool = False,
    candidate_valid_mask=None,
    **_ignored,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Group-relative advantage: ``w_i = beta * (R_i - mean) / (std + eps)``.

    Unlike AWR the weight keeps its sign, so candidates below the group mean
    push the diffusion loss up.  ``positive_advantage_only`` clamps at the group
    mean -- GRPO's baseline -- not at candidate 0 as in AWR.
    """

    totals, valid, safe_count, invalid_count, all_zero = _prepare(
        rewards, reward_config, candidate_valid_mask, safe_only
    )
    weights = np.zeros_like(totals, dtype=np.float32)
    if valid.sum() > 1:
        reward = totals[valid]
        std = float(reward.std(ddof=1)) + max(float(min_group_std), 1e-8)
        advantage = (totals - float(reward.mean())) / std
        clip = abs(float(weight_clip))
        weights[valid] = np.clip(
            float(beta) * advantage[valid], -clip, clip
        ).astype(np.float32)
        if positive_advantage_only:
            margin = max(float(positive_advantage_margin), 0.0)
            weights[weights <= margin] = 0.0
        if normalize:
            scale = float(np.abs(weights).mean())
            if scale > 0.0:
                weights /= max(scale, 1e-8)
    dropped = bool(drop_all_zero_groups and all_zero)
    if dropped:
        weights[:] = 0.0
    return torch.from_numpy(weights), diagnostics(
        totals,
        weights,
        valid,
        safe_candidate_count=safe_count,
        invalid_candidate_count=invalid_count,
        all_zero_group=all_zero,
        dropped_all_zero_group=dropped,
        extra={
            "objective_positive_only": float(positive_advantage_only),
            "objective_safe_only": float(safe_only),
        },
    )


@objective("dpo")
def compute_dpo_weights(
    rewards,
    beta: float = 1.0,
    safe_only: bool = False,
    reward_config=None,
    positive_advantage_only: bool = False,
    positive_advantage_margin: float = 0.0,
    drop_all_zero_groups: bool = False,
    candidate_valid_mask=None,
    normalize: bool = True,
    **_ignored,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pairwise preference on the best/worst candidate: ``+1`` chosen, ``-beta``
    rejected, everything else zero.

    This is the weighted-loss reduction of DPO, not its log-ratio form: it needs
    no reference-policy pass, which is what makes it usable at this seam.  With
    ``positive_advantage_only`` the pair is emitted only when the chosen
    candidate beats candidate 0 by the margin.
    """

    totals, valid, safe_count, invalid_count, all_zero = _prepare(
        rewards, reward_config, candidate_valid_mask, safe_only
    )
    weights = np.zeros_like(totals, dtype=np.float32)
    pair = 0.0
    indices = np.flatnonzero(valid)
    if indices.size > 1:
        chosen = int(indices[np.argmax(totals[indices])])
        rejected = int(indices[np.argmin(totals[indices])])
        margin = max(float(positive_advantage_margin), ZERO_TOL)
        separated = totals[chosen] > totals[rejected] + margin
        beats_behavior = (
            not positive_advantage_only
            or (bool(valid[0]) and totals[chosen] > totals[0] + margin)
        )
        if separated and beats_behavior:
            weights[chosen] = 1.0
            weights[rejected] = -abs(float(beta))
            pair = 1.0
            if normalize:
                scale = float(np.abs(weights).mean())
                if scale > 0.0:
                    weights /= max(scale, 1e-8)
    dropped = bool(drop_all_zero_groups and all_zero)
    if dropped:
        weights[:] = 0.0
        pair = 0.0
    return torch.from_numpy(weights), diagnostics(
        totals,
        weights,
        valid,
        safe_candidate_count=safe_count,
        invalid_candidate_count=invalid_count,
        all_zero_group=all_zero,
        dropped_all_zero_group=dropped,
        extra={
            "objective_positive_only": float(positive_advantage_only),
            "objective_safe_only": float(safe_only),
            "dpo_pair_emitted": pair,
        },
    )
