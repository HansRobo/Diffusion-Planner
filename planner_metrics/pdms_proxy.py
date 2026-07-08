"""Label-free open-loop PDMS-proxy for ranking planner experiments.

Torch-facing wrapper over the **formula-faithful NAVSIM PDMS sub-metric port**
in :mod:`oneplanner.training.pdms_navsim` (comfort / ego-progress / NC / TTC,
1:1 from navsim_v2 + nuplan, scipy+shapely). The val step calls ``pdms_proxy``;
the metrics are reported in [0,1], higher is better.

Why this exists: open-loop ADE/FDE is a weak closed-loop predictor (ρ≈−0.36),
so it cannot tell whether a frontier lever (anchored multi-mode, inference
scorer, …) actually helps — a multi-mode probe can even *degrade* ADE while
producing a better mode. This proxy scores the dimensions those levers optimize,
and ``pdms_proxy_modes`` reports the ORACLE best-of-M score, which directly
measures multi-modal upside (does ANY mode do well?) — something ADE can't.

Honest scope (see ``pdms_navsim`` for the full deviation list): the sub-metric
FORMULAS are bit-exact to navsim, but this is NOT bit-exact PDMS — there is no
bicycle-model simulator (ego kinematics are derived from the pose path) and no
nuplan maps (DAC/DDC/LK/TLC default to 1.0; NC/TTC use the GT-future agent boxes
from the ``.gt.tar`` sidecar). Comfort + EP run with no extra data; NC + TTC run
only when per-future-step agent boxes are supplied (caller's responsibility to
put them in the ego-trajectory frame). A bit-exact PDMS needs navsim+nuplan and
is a separate closed-loop effort (issue #58).

All functions take metric trajectories ``[..., T, 4] = (x, y, cos_yaw, sin_yaw)``
in the ego frame at ``dt`` seconds, batched over leading dims.
"""

from __future__ import annotations

import numpy as np
import torch

from planner_metrics import pdms_navsim as _ns

DT = 0.1  # 10 Hz (OUTPUT_T=80 -> 8 s horizon)

# navsim weighted-metric weights (pdm_scorer.py::PDMScorerConfig). EP and comfort
# (as HISTORY_COMFORT) are the two we compute label-free; TTC joins when agent
# boxes are supplied. The aggregate renormalises over whatever is present.
_W_EP = _ns.PROGRESS_WEIGHT  # 5.0
_W_COMFORT = _ns.HISTORY_COMFORT_WEIGHT  # 2.0
_W_LK = _ns.LANE_KEEPING_WEIGHT  # 2.0
_W_EC = _ns.EXTENDED_COMFORT_WEIGHT  # 2.0 (EPDMS extension)
_W_TTC = _ns.TTC_WEIGHT  # 5.0


def comfort_score(traj: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """navsim comfort in {0,1} (all 6 metrics within bound at all timesteps).
    ``traj`` ``[..., T, 4]`` -> ``[...]``. Delegates to the faithful port."""
    arr = traj.detach().to(torch.float64).cpu().numpy()
    out = _ns.comfort_score(arr, dt)
    return torch.as_tensor(np.asarray(out), dtype=torch.float32, device=traj.device)


def _ego_progress_and_gate(
    pred: torch.Tensor, gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample (EP score, gate flag). The gate flag marks frames where navsim's
    5 m threshold branch fired (stationary/slow expert) — EP is 1.0 there
    REGARDLESS of the prediction, so those frames carry no ranking signal."""
    lead = pred.shape[:-2]
    T = pred.shape[-2]
    pred_n = pred.detach().to(torch.float64).cpu().numpy().reshape(-1, T, 4)
    gt_n = gt.detach().to(torch.float64).cpu().numpy().reshape(-1, T, 4)
    pairs = [_ns.ego_progress_with_gate(pred_n[i], gt_n[i]) for i in range(pred_n.shape[0])]
    ep = np.array([p[0] for p in pairs])
    gated = np.array([float(p[1]) for p in pairs])
    dev = pred.device
    return (
        torch.as_tensor(ep.reshape(lead), dtype=torch.float32, device=dev),
        torch.as_tensor(gated.reshape(lead), dtype=torch.float32, device=dev),
    )


def ego_progress_score(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """navsim ego-progress in [0,1]: predicted progress projected onto the EXPERT
    GT path, normalised by the expert's own progress (navsim's proposal-relative
    EP with the expert as the reference proposal; 5 m threshold gate). ``pred``/
    ``gt`` ``[..., T, 4]`` -> ``[...]``."""
    return _ego_progress_and_gate(pred, gt)[0]


def pdms_proxy(
    pred: torch.Tensor,
    gt: torch.Tensor,
    agent_boxes_per_t: list | None = None,
    ego_dims: "tuple[float, float] | np.ndarray | None" = None,
    agent_labels_per_t: list | None = None,
    static_labels: set | None = None,
    border_lines: list | None = None,
    route_polys: list | None = None,
    lane_rings: list | None = None,
    intersection_rings: list | None = None,
    route_centerlines: list | None = None,
    lk_exempt: list | None = None,
    dac_values: "np.ndarray | None" = None,
    ec_values: "np.ndarray | None" = None,
    dt: float = DT,
) -> dict[str, torch.Tensor]:
    """Aggregate the label-free PDMS-proxy in [0,1] (higher=better) with the
    navsim weighted-average (EP weight 5, comfort weight 2, TTC weight 5 when
    available) and the NC multiplicative gate when agent boxes are supplied.

    Returns ``{total, ego_progress, comfort}`` always, plus ``{ttc,
    no_collision}`` when ``agent_boxes_per_t`` (a length-B list; each element a
    length-T list of ``[N_t, 9]`` ego-frame agent-box arrays) and ``ego_dims``
    ``(length, width)`` are given. ``agent_labels_per_t`` (same nesting as
    ``agent_boxes_per_t``) + ``static_labels`` enable navsim's static-object
    half-penalty (0.5 vs 0.0) in NC; without them every at-fault collision
    scores 0.0 (all-dynamic). Each value is ``[...]`` matching ``pred``'s
    leading dims.

    ``ec_values``: precomputed per-sample EPDMS extended-comfort scores
    (``pdms_navsim.extended_comfort`` over consecutive plans — pairing needs
    cross-sample context the proxy doesn't have). NaN = unavailable (the
    reference's first-plan case): that SAMPLE's weighted average simply
    excludes the EC weight, matching the C++ "unavailable" semantics.
    """
    ep, ep_gated = _ego_progress_and_gate(pred, gt)
    cf = comfort_score(pred, dt)
    # ep_gated: fraction of frames where the navsim 5 m gate fired (EP forced to
    # 1.0 by a stationary/slow expert). Log alongside EP — a high EP mean is only
    # meaningful relative to how many frames actually carried signal.
    out = {"ego_progress": ep, "comfort": cf, "ep_gated": ep_gated}

    # DAC (drivable-area compliance) from road-border polylines: label-free
    # (the borders ride in every shard's line_strings tensor), so it applies
    # in BOTH branches as the navsim multiplicative term. ``border_lines`` is
    # a length-B list; each element a list of [P,2] center-frame polylines.
    # Needs ego dims for the footprint; without them DAC stays None (absent).
    dac_arr = None
    if dac_values is not None:
        # Precomputed (semantic-primary) DAC wins; NaN entries mean the map
        # sidecar lacked that scene -> fall back to the border-only test below.
        dac_arr = np.asarray(dac_values, dtype=np.float64).reshape(-1).copy()
    if (
        border_lines is not None
        and ego_dims is not None
        and (dac_arr is None or np.isnan(dac_arr).any())
    ):
        T_d = pred.shape[-2]
        pred_d = pred.detach().to(torch.float64).cpu().numpy().reshape(-1, T_d, 4)
        dims_d = np.asarray(ego_dims, dtype=np.float64)
        if dims_d.ndim == 1:
            dims_d = np.broadcast_to(dims_d, (pred_d.shape[0], dims_d.shape[0]))
        dac_vals = []
        for i in range(pred_d.shape[0]):
            if dac_arr is not None and not np.isnan(dac_arr[i]):
                dac_vals.append(float(dac_arr[i]))  # semantic value present
                continue
            if dims_d.shape[1] == 3:
                off_d = float(dims_d[i][0]) / 2.0
                len_d, wid_d = float(dims_d[i][1]), float(dims_d[i][2])
            else:
                off_d = 0.0
                len_d, wid_d = float(dims_d[i][0]), float(dims_d[i][1])
            dac_vals.append(
                _ns.dac_from_road_borders(
                    pred_d[i], border_lines[i], len_d, wid_d, center_offset=off_d
                )
            )
        dac_arr = np.array(dac_vals)

    if dac_arr is not None:
        if np.isnan(dac_arr).any():
            raise ValueError("dac_values still has NaN after border fallback")
        out["dac"] = torch.as_tensor(
            dac_arr.reshape(pred.shape[:-2]), dtype=torch.float32, device=pred.device
        )

    # DDC (driving-direction compliance) from on-route lane polygons —
    # label-free like DAC; navsim multiplicative term with 1.0/0.5/0.0 levels.
    ddc_arr = None
    if route_polys is not None:
        lead_c = pred.shape[:-2]
        T_c = pred.shape[-2]
        pred_c = pred.detach().to(torch.float64).cpu().numpy().reshape(-1, T_c, 4)
        ddc_arr = np.array(
            [
                _ns.ddc_from_route_lanes(pred_c[i], route_polys[i], dt)
                for i in range(pred_c.shape[0])
            ]
        )
        out["ddc"] = torch.as_tensor(
            ddc_arr.reshape(lead_c), dtype=torch.float32, device=pred.device
        )

    # LK (lane keeping; Autoware planning_data_analyzer EPDMS spec) — a
    # WEIGHTED term (weight 2), not multiplicative.
    lk_arr = None
    if route_centerlines is not None:
        lead_k = pred.shape[:-2]
        T_k = pred.shape[-2]
        pred_k = pred.detach().to(torch.float64).cpu().numpy().reshape(-1, T_k, 4)
        lk_arr = np.array(
            [
                _ns.lane_keeping_score(
                    pred_k[i],
                    route_centerlines[i],
                    [] if intersection_rings is None else intersection_rings[i],
                    dt,
                    lane_change_exempt=bool(lk_exempt[i]) if lk_exempt is not None else False,
                )
                for i in range(pred_k.shape[0])
            ]
        )
        out["lane_keeping"] = torch.as_tensor(
            lk_arr.reshape(lead_k), dtype=torch.float32, device=pred.device
        )

    # EC (extended comfort; EPDMS weighted term, weight 2). Computed by the
    # caller over consecutive plans; NaN = unavailable for that sample.
    ec_arr = None
    if ec_values is not None:
        ec_arr = np.asarray(ec_values, dtype=np.float64).reshape(-1)
        out["extended_comfort"] = torch.as_tensor(
            ec_arr.reshape(pred.shape[:-2]), dtype=torch.float32, device=pred.device
        )

    if agent_boxes_per_t is not None and ego_dims is not None:
        lead = pred.shape[:-2]
        T = pred.shape[-2]
        pred_n = pred.detach().to(torch.float64).cpu().numpy().reshape(-1, T, 4)
        ep_f = ep.reshape(-1)
        cf_f = cf.reshape(-1)
        # ego_dims: (length, width) [offset 0] or (wheel_base, length, width)
        # triplets — e.g. batch["ego_shape"] — where the footprint centre is
        # shifted wheel_base/2 along heading from the trajectory point
        # (loss.py::compute_ego_bbox_corners convention). Single tuple or
        # per-sample [B, 2|3].
        dims = np.asarray(ego_dims, dtype=np.float64)
        if dims.ndim == 1:
            dims = np.broadcast_to(dims, (pred_n.shape[0], dims.shape[0]))
        nc_list, ttc_list, total_list = [], [], []
        for i in range(pred_n.shape[0]):
            if dims.shape[1] == 3:
                offset = float(dims[i][0]) / 2.0
                length, width = float(dims[i][1]), float(dims[i][2])
            else:
                offset = 0.0
                length, width = float(dims[i][0]), float(dims[i][1])
            states = _ns.states_from_poses(pred_n[i], dt)  # [T, STATE_SIZE]
            # ego-area map gate (navsim _calculate_ego_area) from the shard's
            # lane + intersection polygons; None keeps the legacy lenient path
            flags = None
            if lane_rings is not None or intersection_rings is not None:
                flags = _ns.ego_area_flags(
                    states,
                    [] if lane_rings is None else lane_rings[i],
                    [] if intersection_rings is None else intersection_rings[i],
                    length,
                    width,
                    center_offset=offset,
                )
            # pair boxes with the (possibly clamped) prediction horizon
            boxes_t = list(agent_boxes_per_t[i])[:T]
            labels_t = list(agent_labels_per_t[i])[:T] if agent_labels_per_t is not None else None
            nc = _ns.no_at_fault_collision(
                states,
                boxes_t,
                length,
                width,
                agent_labels_per_t=labels_t,
                static_labels=static_labels,
                center_offset=offset,
                area_flags=flags,
            )
            ttc = _ns.time_to_collision(
                states, boxes_t, length, width, dt, center_offset=offset, area_flags=flags
            )
            ec_i = None
            if ec_arr is not None and not np.isnan(ec_arr[i]):
                ec_i = float(ec_arr[i])
            total = _ns.aggregate_pdms(
                nc=nc,
                dac=1.0 if dac_arr is None else float(dac_arr[i]),
                ddc=1.0 if ddc_arr is None else float(ddc_arr[i]),
                ego_progress=float(ep_f[i]),
                ttc=ttc,
                lane_keeping=None if lk_arr is None else float(lk_arr[i]),
                history_comfort=float(cf_f[i]),
                extended_comfort=ec_i,
            )
            nc_list.append(nc)
            ttc_list.append(ttc)
            total_list.append(total)
        dev = pred.device
        out["no_collision"] = torch.as_tensor(
            np.array(nc_list).reshape(lead), dtype=torch.float32, device=dev
        )
        out["ttc"] = torch.as_tensor(
            np.array(ttc_list).reshape(lead), dtype=torch.float32, device=dev
        )
        out["total"] = torch.as_tensor(
            np.array(total_list).reshape(lead), dtype=torch.float32, device=dev
        )
        return out

    # comfort + EP only: navsim weighted average renormalised over the provided
    # terms, with DAC as the multiplicative gate when borders were provided.
    # EC availability varies PER SAMPLE (NaN = unavailable), so its weight
    # joins numerator and denominator only where it is valid.
    num = _W_EP * ep + _W_COMFORT * cf
    den = torch.full_like(ep, _W_EP + _W_COMFORT)
    if lk_arr is not None:
        num = num + _W_LK * out["lane_keeping"]
        den = den + _W_LK
    if ec_arr is not None:
        ec_t = out["extended_comfort"]
        valid = ~torch.isnan(ec_t)
        num = num + torch.where(valid, _W_EC * ec_t, torch.zeros_like(ec_t))
        den = den + torch.where(valid, torch.full_like(ec_t, _W_EC), torch.zeros_like(ec_t))
    total_lf = num / den
    if dac_arr is not None:
        total_lf = total_lf * out["dac"]
    if ddc_arr is not None:
        total_lf = total_lf * out["ddc"]
    out["total"] = total_lf
    return out


def pdms_proxy_masked(
    pred: torch.Tensor,
    gt: torch.Tensor,
    agent_boxes_per_t: list,
    agent_labels_per_t: list | None = None,
    ego_dims: "np.ndarray | None" = None,
    available: "list | None" = None,
    static_labels: set | None = None,
    border_lines: list | None = None,
    route_polys: list | None = None,
    lane_rings: list | None = None,
    intersection_rings: list | None = None,
    route_centerlines: list | None = None,
    lk_exempt: list | None = None,
    dac_values: "np.ndarray | None" = None,
    ec_values: "np.ndarray | None" = None,
    dt: float = DT,
) -> dict[str, torch.Tensor]:
    """``pdms_proxy`` with a per-sample agent-GT availability mask.

    A missing ``.gt.tar`` sidecar is NOT "no agents": its empty box lists would
    silently award perfect NC/TTC and fold the TTC weight into ``total``,
    inflating the score vs the label-free aggregate. Samples with
    ``available[i]`` falsy are therefore scored with the LABEL-FREE formula
    (EP+comfort); the rest get the full NC/TTC one. ``total`` / ``ego_progress``
    / ``ep_gated`` / ``comfort`` cover all ``B`` samples; ``no_collision`` /
    ``ttc`` cover ONLY the available ones (length ``sum(available)``, in batch
    order) and are absent when none are. ``pred``/``gt`` must be ``[B, T, 4]``.
    """
    B = pred.shape[0]
    avail = [True] * B if available is None else [bool(a) for a in available]
    ai = [i for i, a in enumerate(avail) if a]
    if len(ai) == B:
        return pdms_proxy(
            pred,
            gt,
            agent_boxes_per_t=agent_boxes_per_t,
            ego_dims=ego_dims,
            agent_labels_per_t=agent_labels_per_t,
            static_labels=static_labels,
            border_lines=border_lines,
            route_polys=route_polys,
            lane_rings=lane_rings,
            intersection_rings=intersection_rings,
            route_centerlines=route_centerlines,
            lk_exempt=lk_exempt,
            # regression note: this all-available fast path shipped in #102
            # WITHOUT dac_values — semantic DAC was silently dropped on the
            # most common val batch shape. Keep every per-sample input here.
            dac_values=dac_values,
            ec_values=ec_values,
            dt=dt,
        )
    if not ai:
        # Map terms are label-free (shard map tensors) — they apply even when
        # no sample has agent GT.
        return pdms_proxy(
            pred,
            gt,
            ego_dims=ego_dims,
            border_lines=border_lines,
            route_polys=route_polys,
            intersection_rings=intersection_rings,
            route_centerlines=route_centerlines,
            lk_exempt=lk_exempt,
            dac_values=dac_values,
            ec_values=ec_values,
            dt=dt,
        )
    ui = [i for i, a in enumerate(avail) if not a]
    p_a = pdms_proxy(
        pred[ai],
        gt[ai],
        agent_boxes_per_t=[agent_boxes_per_t[i] for i in ai],
        ego_dims=None if ego_dims is None else np.asarray(ego_dims)[ai],
        agent_labels_per_t=(
            None if agent_labels_per_t is None else [agent_labels_per_t[i] for i in ai]
        ),
        static_labels=static_labels,
        border_lines=None if border_lines is None else [border_lines[i] for i in ai],
        route_polys=None if route_polys is None else [route_polys[i] for i in ai],
        lane_rings=None if lane_rings is None else [lane_rings[i] for i in ai],
        intersection_rings=(
            None if intersection_rings is None else [intersection_rings[i] for i in ai]
        ),
        route_centerlines=(
            None if route_centerlines is None else [route_centerlines[i] for i in ai]
        ),
        lk_exempt=None if lk_exempt is None else [lk_exempt[i] for i in ai],
        dac_values=None if dac_values is None else np.asarray(dac_values)[ai],
        ec_values=None if ec_values is None else np.asarray(ec_values)[ai],
        dt=dt,
    )
    p_u = pdms_proxy(
        pred[ui],
        gt[ui],
        ego_dims=None if ego_dims is None else np.asarray(ego_dims)[ui],
        border_lines=None if border_lines is None else [border_lines[i] for i in ui],
        route_polys=None if route_polys is None else [route_polys[i] for i in ui],
        intersection_rings=(
            None if intersection_rings is None else [intersection_rings[i] for i in ui]
        ),
        route_centerlines=(
            None if route_centerlines is None else [route_centerlines[i] for i in ui]
        ),
        lk_exempt=None if lk_exempt is None else [lk_exempt[i] for i in ui],
        dac_values=None if dac_values is None else np.asarray(dac_values)[ui],
        ec_values=None if ec_values is None else np.asarray(ec_values)[ui],
        dt=dt,
    )
    out: dict[str, torch.Tensor] = {}
    merge_keys = ["total", "ego_progress", "ep_gated", "comfort"]
    if "dac" in p_a and "dac" in p_u:
        merge_keys.append("dac")
    if "ddc" in p_a and "ddc" in p_u:
        merge_keys.append("ddc")
    if "lane_keeping" in p_a and "lane_keeping" in p_u:
        merge_keys.append("lane_keeping")
    if "extended_comfort" in p_a and "extended_comfort" in p_u:
        merge_keys.append("extended_comfort")
    for k in merge_keys:
        merged = torch.empty(B, dtype=p_a[k].dtype, device=p_a[k].device)
        merged[ai] = p_a[k]
        merged[ui] = p_u[k]
        out[k] = merged
    out["no_collision"] = p_a["no_collision"]
    out["ttc"] = p_a["ttc"]
    return out


# EPDMS human filter (planning_data_analyzer): |H_m| <= eps means the HUMAN
# reference also failed metric m -> the agent's failure is suppressed to 1.
EPS_HUMAN = 1.0e-9


def epdms_human_filtered(
    agent: dict, human: dict, available: "list | np.ndarray | None" = None
) -> torch.Tensor:
    """Human-filtered EPDMS aggregate per sample (planning_data_analyzer).

    ``filter_m(agent, human) = 1`` when both scores are available and the human
    reference also failed (``|H_m| <= 1e-9``); otherwise the agent score passes
    through. Applied to the multiplicative gate (NC * DAC * DDC — TLC pending)
    and the weighted terms (TTC, LK, HC), with two reference-faithful
    exceptions: EC bypasses the filter (it measures consistency between
    consecutive AGENT plans), and EP keeps the agent value — our EP is already
    navsim's agent/human progress ratio whose 5 m stationary gate forces 1.0
    whenever the human made ~no progress, subsuming ``|H_EP| <= eps``.

    ``agent`` / ``human`` are ``pdms_proxy``/``pdms_proxy_masked`` outputs for
    the SAME batch (human = GT scored as a trajectory). ``no_collision``/``ttc``
    cover only the agent-GT-available subset in masked mode, so ``available``
    (the same mask passed to both calls) scatters them back to batch order;
    samples without those terms simply renormalise without the TTC weight,
    like the label-free total. Weighted terms missing from BOTH dicts (e.g. LK
    without centerlines) are excluded the same way.
    """
    ep = agent["ego_progress"]
    B = ep.shape[0]
    one = torch.ones_like(ep)

    def _filtered(key: str) -> torch.Tensor | None:
        a, h = agent.get(key), human.get(key)
        if a is None or h is None:
            return None
        if a.shape[0] != B:  # subset-length (masked nc/ttc) -> scatter to B
            if available is None:
                return None
            idx = [i for i, av in enumerate(available) if bool(av)]
            if len(idx) != a.shape[0] or len(idx) != h.shape[0]:
                raise ValueError(f"{key}: subset length does not match available mask")
            a_f = torch.full_like(ep, float("nan"))
            h_f = torch.full_like(ep, float("nan"))
            a_f[idx], h_f[idx] = a.to(ep.dtype), h.to(ep.dtype)
            a, h = a_f, h_f
        return torch.where(h.abs() <= EPS_HUMAN, one, a.to(ep.dtype))

    mult = one.clone()
    for key in ("no_collision", "dac", "ddc"):
        f = _filtered(key)
        if f is not None:
            mult = mult * torch.where(torch.isnan(f), one, f)

    num = _W_EP * ep
    den = torch.full_like(ep, _W_EP)
    for key, w in (("ttc", _W_TTC), ("lane_keeping", _W_LK), ("comfort", _W_COMFORT)):
        f = _filtered(key)
        if f is not None:
            valid = ~torch.isnan(f)
            num = num + torch.where(valid, w * f, torch.zeros_like(f))
            den = den + torch.where(valid, torch.full_like(f, w), torch.zeros_like(f))
    ec = agent.get("extended_comfort")
    if ec is not None:  # EC bypasses the human filter (reference rule)
        ec = ec.to(ep.dtype)
        valid = ~torch.isnan(ec)
        num = num + torch.where(valid, _W_EC * ec, torch.zeros_like(ec))
        den = den + torch.where(valid, torch.full_like(ec, _W_EC), torch.zeros_like(ec))
    return mult * num / den


def pdms_proxy_modes(
    pred_modes: torch.Tensor,
    gt: torch.Tensor,
    chosen_idx: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Multi-mode proxy. ``pred_modes`` is ``[B, M, T, 4]``; scores each mode and
    returns the CHOSEN mode's score (by ``chosen_idx``, default mode 0) and the
    ORACLE best-of-M score. ``oracle >= chosen`` always; ``oracle > chosen``
    means a better mode existed than the one selected — the multi-modal upside
    signal that open-loop ADE cannot reveal. Returns ``{chosen, oracle,
    per_mode}`` (``per_mode`` is ``[B, M]``)."""
    b, m = pred_modes.shape[0], pred_modes.shape[1]
    per_mode = torch.stack(
        [pdms_proxy(pred_modes[:, i], gt)["total"] for i in range(m)], dim=1
    )  # [B, M]
    oracle = per_mode.max(dim=1).values
    if chosen_idx is None:
        chosen = per_mode[:, 0]
    else:
        chosen = per_mode[torch.arange(b, device=per_mode.device), chosen_idx]
    return {"chosen": chosen, "oracle": oracle, "per_mode": per_mode}
