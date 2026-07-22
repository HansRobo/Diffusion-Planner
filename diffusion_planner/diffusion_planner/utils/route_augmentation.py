"""Input-side augmentations for the route.

Implements two of the high-priority proposals from
docs/route_augmentation_proposals.md:

- Route tail truncation: randomly shorten the forward extent of ``route_lanes``
  so the model sees the short routes that occur near the goal / map edge at
  deployment time.
- Speed-limit unknown dropout: drop ``*_has_speed_limit`` to False (scene-wide,
  lanes and route together) so the encoder's ``unknown_speed_emb`` path is
  actually trained.

(Traffic-light unknown dropout and scene flip live on separate branches.)

Both transforms preserve the all-zero padding convention and only ever remove
information (never fabricate a state that contradicts the ground truth
trajectory). Applied in train_epoch before StatePerturbation, on raw
(un-normalized) ego-centric inputs.
"""

import torch

# Geometry channels used for validity checks (x, y, dx, dy, LB, RB), matching
# the masks in data_augmentation.StatePerturbation.
_GEOM_DIM = 8


class RouteAugmentation:
    """Randomized route / speed-limit dropout augmentation.

    Each component is applied independently per sample with its own
    probability. Probabilities of 0 disable a component.
    """

    def __init__(
        self,
        device,
        truncation_prob: float = 0.0,
        truncation_min_m: float = 60.0,
        truncation_max_m: float = 200.0,
        speed_limit_unknown_prob: float = 0.0,
    ):
        assert 0.0 < truncation_min_m <= truncation_max_m
        self._device = device
        self._truncation_prob = truncation_prob
        self._truncation_min_m = truncation_min_m
        self._truncation_max_m = truncation_max_m
        self._speed_limit_unknown_prob = speed_limit_unknown_prob

    @torch.no_grad()
    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._truncation_prob > 0.0:
            self._truncate_route_tail(inputs)
        if self._speed_limit_unknown_prob > 0.0:
            self._drop_speed_limit(inputs)
        return inputs

    def _truncate_route_tail(self, inputs: dict[str, torch.Tensor]) -> None:
        """Zero out route segments starting beyond a sampled arc-length budget.

        Distance is accumulated along the route polyline from the first route
        segment (the ego's current lanelet), which approximates distance from
        ego. The first segment is always kept so the immediate intent is never
        destroyed.
        """
        route = inputs["route_lanes"]  # [B, P, V, D]
        B = route.shape[0]

        valid_pt = route[..., :_GEOM_DIM].abs().sum(-1) > 0  # [B, P, V]
        xy = route[..., :2]
        step = (xy[:, :, 1:] - xy[:, :, :-1]).norm(dim=-1)  # [B, P, V-1]
        pair_valid = (valid_pt[:, :, 1:] & valid_pt[:, :, :-1]).to(step.dtype)
        seg_len = (step * pair_valid).sum(-1)  # [B, P]

        # Arc length from the route start to each segment's start.
        cum_before = torch.cumsum(seg_len, dim=1) - seg_len  # [B, P]

        budget = self._truncation_min_m + (
            self._truncation_max_m - self._truncation_min_m
        ) * torch.rand(B, 1, device=route.device)
        apply = torch.rand(B, 1, device=route.device) < self._truncation_prob

        drop = (cum_before >= budget) & apply  # [B, P]
        drop[:, 0] = False

        inputs["route_lanes"][drop] = 0.0
        inputs["route_lanes_speed_limit"][drop] = 0.0
        inputs["route_lanes_has_speed_limit"][drop] = False

    def _drop_speed_limit(self, inputs: dict[str, torch.Tensor]) -> None:
        """Scene-wide speed-limit unknown dropout on lanes and route together.

        Scene-wide (not per-segment) so the same lanelet never carries a limit
        in ``lanes`` while missing it in ``route_lanes``.
        """
        B = inputs["route_lanes"].shape[0]
        scene = torch.rand(B, device=self._device) < self._speed_limit_unknown_prob  # [B]
        if not bool(scene.any()):
            return
        for prefix in ("lanes", "route_lanes"):
            inputs[f"{prefix}_speed_limit"][scene] = 0.0
            inputs[f"{prefix}_has_speed_limit"][scene] = False
