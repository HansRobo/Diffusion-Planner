"""Single-segment closed-loop rollout with ablation/attention hooks, for the
token-importance / attention-analysis tools (``scripts/token_importance_closed_loop.py``,
``scripts/visualize_closed_loop_attention.py``).

``reproducer_rollout.py`` used to expose this as ``run_segment``; a later rollout-engine
rewrite there replaced the single-segment orchestrator with the batched
``run_segments_batched`` (plus the rendering-oriented ``render_segment``), dropping
``run_segment`` itself. Rather than reintroduce it into that shared file — where it would
sit alongside two other single-segment/batched implementations doing the same per-step
bookkeeping — this module rebuilds just the orchestration loop here, on top of
``reproducer_rollout``'s still-current private step helpers (``_seed_state``, ``_pre_step``,
``_feed_turn_indicator``, ``_post_step``, ``_finalize``). Those helpers, and the metrics
dict ``_finalize`` returns, are shared with ``render_segment``/``run_segments_batched`` and
evolve with them — this module has no state of its own to drift out of sync.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from scenario_generation.inference_compile import mark_inference_step
from scenario_generation.perf_timer import Timers
from scenario_generation.reproducer_rollout import (
    _add_static_inputs,
    _arrays_to_device,
    _feed_turn_indicator,
    _finalize,
    _post_step,
    _pre_step,
    _seed_state,
)
from scenario_generation.route_timeline import RouteTimeline


def _to_torch_batch_ablated(
    np_dicts: list[dict],
    model_args,
    device: str,
    ablate_fn: Callable[[dict], dict] | None = None,
) -> dict:
    """Like ``reproducer_rollout._to_torch_batch``, with an ablation hook.

    ``ablate_fn``, if given, is applied to the un-normalized batched tensor dict right
    before normalization (same convention as ``token_importance.py``'s ``apply_ablation``,
    which also runs pre-normalization so a zeroed input maps to the model's real padding
    value rather than an arbitrary normalized one). A local copy rather than a patch to the
    shared ``_to_torch_batch``, since this module doesn't otherwise touch
    ``reproducer_rollout.py``.
    """
    N = len(np_dicts)
    arrays = {k: np.concatenate([d[k] for d in np_dicts], axis=0) for k in np_dicts[0]}
    data = _arrays_to_device(arrays, device)
    _add_static_inputs(data, model_args, N, device)
    if ablate_fn is not None:
        data = ablate_fn(data)
    return model_args.observation_normalizer(data)


@torch.no_grad()
def run_segment(
    model,
    model_args,
    tl: RouteTimeline,
    start: int,
    end: int,
    device: str = "cuda",
    near_miss_thresh: float = 0.5,
    search_radius: float = 1.5,
    warmup_steps: int = 0,
    goal_reach_m: float = 5.0,
    max_stuck_steps: int = 0,
    max_steps: int | None = None,
    unstick_after: int = 300,
    unstick_advance_m: float = 5.0,
    unstick_radius_mult: float = 3.0,
    unstick_teleport_after: int = 300,
    neighbor_history_mode: str = "recorded",
    timers: Timers | None = None,
    ablate_fn: Callable[[dict], dict] | None = None,
    step_callback: Callable[[int, dict, dict], None] | None = None,
) -> dict:
    """Single-segment closed-loop reproducer rollout over recorded frames [start, end).

    Every step re-plans (no ``render_segment``-style ``replan_interval`` caching) and uses
    perfect tracking / pose-mode timeline progress / segment-local goal — i.e. the same
    fixed choices the original ``run_segment`` made, now expressed as the corresponding
    ``_seed_state`` knobs (``tracker_mode="perfect"``, ``replay_mode="pose"``,
    ``goal_mode="segment"``) that its current, more configurable replacements expose.

    Unstick is on by default (snap the ego forward after ``unstick_after`` steps of no
    progress). The only timeout is the step cap ``max_steps`` (default 3*(end-start)); the
    hard stuck-cutoff is off.

    ``ablate_fn``, if given, is forwarded to ``_to_torch_batch_ablated`` and applied to the
    un-normalized model input each step. ``step_callback``, if given, is invoked as
    ``step_callback(s.k, np_dict, outputs)`` right after each model forward — e.g. to
    capture attention hooks installed on ``model`` before calling this function. Both
    default to no-ops.

    Returns the same metrics dict as ``reproducer_rollout._finalize`` (route_completion,
    per-family clearance/collision blocks, etc.) — not the old ``SegmentResult`` shape.
    """
    timers = timers or Timers()
    s = _seed_state(
        tl=tl,
        start=start,
        end=end,
        search_radius=search_radius,
        warmup_steps=warmup_steps,
        near_miss_thresh=near_miss_thresh,
        goal_reach_m=goal_reach_m,
        max_stuck_steps=max_stuck_steps,
        timers=timers,
        max_steps=max_steps if max_steps is not None else 3 * (end - start),
        unstick_after=unstick_after,
        unstick_advance_m=unstick_advance_m,
        unstick_radius_mult=unstick_radius_mult,
        unstick_teleport_after=unstick_teleport_after,
        neighbor_history_mode=neighbor_history_mode,
        goal_mode="segment",
        replay_mode="pose",
        tracker_mode="perfect",
        strong_brake_mps2=-2.5,
        yaw_gate=True,
    )
    while not s.done:
        with timers("input_build"):
            pre = _pre_step(s)
        if pre is None:
            break
        np_dict, neighbors_live, idx, _slot_uuids, _world_by_uuid = pre
        with timers("to_torch"):
            data = _to_torch_batch_ablated([np_dict], model_args, device, ablate_fn=ablate_fn)
        with timers("model_forward"):
            mark_inference_step()  # no-op unless the model was compiled with cudagraphs
            _, outputs = model(data)
            pred = outputs["prediction"][0, 0].cpu().numpy()
        if step_callback is not None:
            step_callback(s.k, np_dict, outputs)
        _feed_turn_indicator(s, outputs)  # closed-loop turn signal
        _post_step(s, pred, neighbors_live, idx, device, timers, np_dict=np_dict)
    return _finalize(s)
