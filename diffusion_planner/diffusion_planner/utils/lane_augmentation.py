"""Input-side augmentations for the surrounding lanes.

Implements phases 1 and 2 of docs/lane_augmentation_design.md:

- Far-lane truncation (high): zero lanes whose nearest point lies beyond a
  sampled radius, simulating smaller map crops / map edges.
- Non-route lane dropout (high): independently drop non-route lanes,
  simulating missing lanelets / map maintenance gaps.
- Lane geometry noise (medium): constant per-segment lateral offset of
  non-route lane centerlines (map survey error).
- Lane width jitter (medium): per-side scaling of the relative boundary
  offsets of non-route lanes (lane width mapping error).

Lanes that are copies of route lanelets are protected by an exact-match mask
(the conversion pipeline writes the same lanelet bit-identically into both
tensors), so the route never points at a road that no longer exists in
``lanes``. Because the match relies on unmodified route tensors, this class
must run BEFORE RouteAugmentation in train_epoch.

Same conventions as RouteAugmentation: all-zero padding is preserved, nothing
contradicting the ground-truth trajectory is fabricated, no host-device syncs,
no full-batch copies, masks live on the tensors' own device.
"""

import torch
import torch.nn.functional as F

# Geometry channels used for validity checks (x, y, dx, dy, LB, RB).
_GEOM_DIM = 8


class LaneAugmentation:
    """Randomized surrounding-lane augmentation with route protection.

    Each component is applied independently per sample with its own
    probability. Probabilities of 0 disable a component.
    """

    def __init__(
        self,
        device,
        truncation_prob: float = 0.0,
        truncation_min_m: float = 50.0,
        truncation_max_m: float = 100.0,
        dropout_prob: float = 0.0,
        dropout_ratio: float = 0.1,
        geometry_noise_prob: float = 0.0,
        geometry_noise_std_m: float = 0.15,
        width_jitter_prob: float = 0.0,
        width_jitter_std: float = 0.05,
    ):
        assert 0.0 < truncation_min_m <= truncation_max_m
        assert 0.0 <= dropout_ratio <= 1.0
        assert geometry_noise_std_m >= 0.0
        assert width_jitter_std >= 0.0
        self._device = device
        self._truncation_prob = truncation_prob
        self._truncation_min_m = truncation_min_m
        self._truncation_max_m = truncation_max_m
        self._dropout_prob = dropout_prob
        self._dropout_ratio = dropout_ratio
        self._geometry_noise_prob = geometry_noise_prob
        self._geometry_noise_std_m = geometry_noise_std_m
        self._width_jitter_prob = width_jitter_prob
        self._width_jitter_std = width_jitter_std

    @torch.no_grad()
    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not (
            (self._dropout_prob > 0.0 and self._dropout_ratio > 0.0)
            or self._truncation_prob > 0.0
            or self._geometry_noise_prob > 0.0
            or self._width_jitter_prob > 0.0
        ):
            return inputs

        # The per-point validity mask and the per-lane ego distance are shared
        # by several components; computing them once avoids repeated full
        # passes over the large lanes tensor (~190 MB at batch 512).
        # ``valid_pt`` is kept current by ``_zero_lanes`` when rows are removed.
        lanes = inputs["lanes"]
        valid_pt = lanes[..., :_GEOM_DIM].abs().sum(-1) > 0  # [B, L, V]
        dist = torch.where(
            valid_pt,
            lanes[..., :2].norm(dim=-1),
            torch.full(valid_pt.shape, float("inf"), device=lanes.device, dtype=lanes.dtype),
        )
        min_dist = dist.amin(dim=2)  # [B, L]; padding rows -> inf

        protected = self._route_protection_mask(inputs, valid_pt, min_dist)  # [B, L]
        # Row-removing components first; the noise components then only touch
        # the surviving rows.
        if self._dropout_prob > 0.0 and self._dropout_ratio > 0.0:
            self._drop_lanes(inputs, protected, valid_pt)
        if self._truncation_prob > 0.0:
            self._truncate_far_lanes(inputs, protected, valid_pt, min_dist)
        if self._geometry_noise_prob > 0.0:
            self._jitter_lane_geometry(inputs, protected, valid_pt)
        if self._width_jitter_prob > 0.0:
            self._jitter_lane_width(inputs, protected)
        return inputs

    def _route_protection_mask(
        self, inputs: dict[str, torch.Tensor], valid_pt: torch.Tensor, min_dist: torch.Tensor
    ) -> torch.Tensor:
        """Boolean [B, L] mask of lanes that are copies of route lanelets.

        The conversion pipeline writes the same lanelet bit-identically into
        ``lanes`` and ``route_lanes`` (verified on real npz data), so matching
        the first centerline point exactly is sufficient. Samples with an
        empty route fall back to protecting the lane closest to ego, so the
        currently-driven lane can never be dropped.
        """
        lanes = inputs["lanes"]  # [B, L, V, D]
        route = inputs["route_lanes"]  # [B, R, V, D]

        lanes_valid = valid_pt.any(-1)  # [B, L]
        route_valid = route[..., :_GEOM_DIM].abs().sum(dim=(2, 3)) > 0  # [B, R]

        eq = (lanes[:, :, None, 0, :2] == route[:, None, :, 0, :2]).all(-1)  # [B, L, R]
        protected = (eq & route_valid[:, None, :]).any(-1) & lanes_valid  # [B, L]

        # Route-less fallback: protect the nearest valid lane (ego is at the
        # origin of the ego-centric frame).
        no_route = ~protected.any(dim=1)  # [B]
        nearest = min_dist.argmin(dim=1)  # [B]
        fallback = F.one_hot(nearest, lanes.shape[1]).bool() & no_route[:, None] & lanes_valid
        return protected | fallback

    def _zero_lanes(
        self, inputs: dict[str, torch.Tensor], drop: torch.Tensor, valid_pt: torch.Tensor
    ) -> None:
        inputs["lanes"][drop] = 0.0
        inputs["lanes_speed_limit"][drop] = 0.0
        inputs["lanes_has_speed_limit"][drop] = False
        valid_pt[drop] = False  # keep the shared mask current for later stages

    def _drop_lanes(
        self, inputs: dict[str, torch.Tensor], protected: torch.Tensor, valid_pt: torch.Tensor
    ) -> None:
        """Independently drop non-route lanes, simulating missing lanelets."""
        lanes = inputs["lanes"]
        B, L = lanes.shape[:2]
        scene = torch.rand(B, 1, device=lanes.device) < self._dropout_prob
        drop = (torch.rand(B, L, device=lanes.device) < self._dropout_ratio) & scene & ~protected
        self._zero_lanes(inputs, drop, valid_pt)

    def _truncate_far_lanes(
        self,
        inputs: dict[str, torch.Tensor],
        protected: torch.Tensor,
        valid_pt: torch.Tensor,
        min_dist: torch.Tensor,
    ) -> None:
        """Zero lanes whose nearest valid point lies beyond a sampled radius.

        Simulates smaller map crops / map edges than the deployment-time
        +-100m window. Distance is measured directly (no reliance on the
        distance-sorted slot order).
        """
        lanes = inputs["lanes"]
        B = lanes.shape[0]

        radius = self._truncation_min_m + (
            self._truncation_max_m - self._truncation_min_m
        ) * torch.rand(B, 1, device=lanes.device)
        apply = torch.rand(B, 1, device=lanes.device) < self._truncation_prob

        # Padding rows (inf distance) land in `drop`; re-zeroing zeros is a no-op.
        drop = (min_dist > radius) & apply & ~protected
        self._zero_lanes(inputs, drop, valid_pt)

    def _jitter_lane_geometry(
        self, inputs: dict[str, torch.Tensor], protected: torch.Tensor, valid_pt: torch.Tensor
    ) -> None:
        """Constant per-segment lateral offset of non-route lane centerlines.

        Same construction as RouteAugmentation._jitter_route_geometry: the
        offset is constant within a segment so the direction channels and the
        relative boundary offsets stay exactly consistent. Route copies are
        excluded so the same lanelet never appears at two different positions.
        """
        lanes = inputs["lanes"]  # [B, L, V, D]
        B, L = lanes.shape[:2]

        mean_dir = (lanes[..., 2:4] * valid_pt.unsqueeze(-1)).sum(dim=2)  # [B, L, 2]
        unit = mean_dir / mean_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal = torch.stack([-unit[..., 1], unit[..., 0]], dim=-1)  # [B, L, 2]

        offset = torch.randn(B, L, 1, device=lanes.device) * self._geometry_noise_std_m
        apply = (torch.rand(B, 1, 1, device=lanes.device) < self._geometry_noise_prob).to(
            lanes.dtype
        )
        eligible = (~protected).unsqueeze(-1).to(lanes.dtype)  # [B, L, 1]
        shift = normal * offset * apply * eligible  # [B, L, 2]

        lanes[..., :2] += shift.unsqueeze(2) * valid_pt.unsqueeze(-1)

    def _jitter_lane_width(self, inputs: dict[str, torch.Tensor], protected: torch.Tensor) -> None:
        """Scale the relative boundary offsets of non-route lanes per side.

        The centerline stays fixed; only the left/right boundary offset
        vectors (channels 4-5 / 6-7) are scaled by independent per-segment
        factors clipped to [0.85, 1.15], simulating lane-width mapping error
        without ever collapsing or inverting a lane.
        """
        lanes = inputs["lanes"]
        B, L = lanes.shape[:2]

        apply = torch.rand(B, 1, 1, 1, device=lanes.device) < self._width_jitter_prob
        eligible = apply & (~protected).view(B, L, 1, 1)
        for start in (4, 6):  # left / right boundary channels
            scale = 1.0 + torch.randn(B, L, 1, 1, device=lanes.device) * self._width_jitter_std
            scale = scale.clamp(0.85, 1.15)
            scale = torch.where(eligible, scale, torch.ones_like(scale))
            lanes[..., start : start + 2] *= scale
