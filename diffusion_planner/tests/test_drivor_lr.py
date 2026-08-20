"""Parity tests for DrivoR's optimizer schedule (``utils/drivor_lr.py``).

The reference is ``navsim/agents/drivoR/drivor_agent.py:408-483`` plus its
``config/training/t4_training.yaml`` values.  What is pinned here is everything
that silently changed meaning when this port moved from the 2.5% subsample to
the full train list: the peak LR's sqrt batch scaling, the fact that the ramp
and cosine are measured in optimizer STEPS, and the divergence guard's LR cut
surviving the next scheduler step.
"""

import torch

from diffusion_planner.utils.drivor_lr import (
    DRIVOR_BASE_BATCH_SIZE,
    DRIVOR_BASE_LR,
    DRIVOR_WARMUP_RATIO,
    build_drivor_scheduler,
    build_lr_probe,
    scaled_peak_lr,
    scheduler_base_lrs,
)
from diffusion_planner.utils.drivor_train import DivergenceGuard


def _optimizer(lr: float):
    return torch.optim.AdamW([{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": lr}])


def _lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def test_sqrt_scaling_matches_drivor():
    # drivor_agent.py:415 -- sqrt, not linear.  The released reference is
    # base_lr 1e-4 at batch 64, so our global batch of 256 resolves to 2e-4.
    assert scaled_peak_lr(DRIVOR_BASE_LR, 256, DRIVOR_BASE_BATCH_SIZE) == 2e-4
    assert scaled_peak_lr(1e-4, 64, 64) == 1e-4
    # Linear scaling would have given 4e-4 here; make the difference explicit so
    # a future "simplification" to linear fails loudly.
    assert scaled_peak_lr(1e-4, 256, 64) != 4e-4
    # base_batch_size <= 0 is the opt-out: --learning_rate is taken literally.
    assert scaled_peak_lr(3e-4, 256, 0) == 3e-4


def test_ramp_and_cosine_are_measured_in_steps():
    peak, total = 2e-4, 1000
    ramp = int(total * DRIVOR_WARMUP_RATIO)
    optimizer = _optimizer(peak)
    scheduler = build_drivor_scheduler(
        optimizer, total_steps=total, warmup_ratio=DRIVOR_WARMUP_RATIO
    )

    # start_factor=1e-6 (drivor_agent.py:461): the ramp begins at ~zero, not at
    # a tenth of peak the way CosineAnnealingWarmUpRestarts does.
    assert _lr(optimizer) < peak * 1e-5

    seen = []
    for _ in range(total):
        seen.append(_lr(optimizer))
        scheduler.step()

    # Peak is hit exactly at the hand-off, and only there.
    assert seen[ramp] == max(seen)
    assert abs(seen[ramp] - peak) < 1e-12
    # Monotone up to the hand-off, monotone down after it.
    assert all(a <= b for a, b in zip(seen[:ramp], seen[1 : ramp + 1]))
    assert all(a >= b for a, b in zip(seen[ramp:], seen[ramp + 1 :]))
    # eta_min=0 with T_max = total - ramp, so the last step lands on ~0 rather
    # than stopping part-way down the cosine.
    assert seen[-1] < peak * 1e-4


def test_zero_warmup_ratio_is_pure_cosine():
    optimizer = _optimizer(1e-4)
    scheduler = build_drivor_scheduler(optimizer, total_steps=100, warmup_ratio=0.0)
    assert _lr(optimizer) == 1e-4  # no ramp: starts at peak
    for _ in range(100):
        scheduler.step()
    assert _lr(optimizer) < 1e-8


def _drivor(peak: float, total: int, warmup_ratio: float, advance: int):
    optimizer = _optimizer(peak)
    scheduler = build_drivor_scheduler(
        optimizer, total_steps=total, warmup_ratio=warmup_ratio
    )
    for _ in range(advance):
        scheduler.step()
    return optimizer, scheduler


def _probe(advance: int):
    optimizer = _optimizer(1e-5)
    probe = build_lr_probe(optimizer, steps=900, lr_min=1e-5, lr_max=3e-3)
    for _ in range(advance):
        probe.step()
    return optimizer, probe


def test_guard_cut_is_exactly_half_against_an_uncut_control():
    """Halving ``param_groups`` alone is not enough, and is not always enough.

    Where it holds depends on the scheduler, because PyTorch is inconsistent
    about it:

    * ``LambdaLR`` (the probe) computes ``base_lrs * lambda(step)`` -- not
      recursive, so a ``param_groups`` cut is gone on the next step.
    * ``CosineAnnealingLR``'s general branch is recursive on ``group["lr"]``, so
      a cut there does propagate on its own.
    * ``SequentialLR`` calls ``step(0)`` on the incoming scheduler at its
      milestone, which recomputes from ``base_lrs`` via ``_get_closed_form_lr``
      -- so a cut taken during the ramp is wiped at the ramp/cosine hand-off.

    Cutting ``base_lrs`` covers all three, and must not double-cut the case that
    already worked.  Each case is therefore compared against an uncut control
    advanced to the same step; the ratio has to be 0.5, never 0.25.
    """
    cases = {
        "probe (LambdaLR, cut wiped without the fix)": (
            lambda: _probe(600),
            lambda: _probe(601)[0],
        ),
        "cosine phase (recursive, cut already propagated)": (
            lambda: _drivor(2e-4, 1000, 0.1, 500),
            lambda: _drivor(2e-4, 1000, 0.1, 501)[0],
        ),
        "ramp then across the SequentialLR milestone": (
            lambda: _drivor(2e-4, 1000, 0.1, 50),
            lambda: _drivor(2e-4, 1000, 0.1, 110)[0],
        ),
    }
    # The third case has to keep stepping after the cut to reach the milestone.
    extra_steps = {"ramp then across the SequentialLR milestone": 60}

    for name, (build, build_control) in cases.items():
        optimizer, scheduler = build()
        guard = DivergenceGuard(enabled=True)
        guard.attach_scheduler(scheduler)
        guard._cut_lr(optimizer, floor=1e-12, reason="test")
        for _ in range(extra_steps.get(name, 1)):
            scheduler.step()

        control = _lr(build_control())
        assert abs(_lr(optimizer) / control - 0.5) < 1e-9, name


def test_guard_without_a_scheduler_only_touches_param_groups():
    # The legacy per-epoch path passes no scheduler; the cut must still apply,
    # and nothing may reach into base_lrs that are not there.
    optimizer = _optimizer(1e-4)
    guard = DivergenceGuard(enabled=True)
    assert guard.scheduler is None
    guard._cut_lr(optimizer, floor=1e-8, reason="test")
    assert _lr(optimizer) == 5e-5


def test_guard_cut_respects_the_floor():
    optimizer, scheduler = _drivor(2e-4, 1000, 0.1, 500)
    guard = DivergenceGuard(enabled=True)
    guard.attach_scheduler(scheduler)
    for _ in range(40):
        guard._cut_lr(optimizer, floor=1e-6, reason="test")
    assert _lr(optimizer) == 1e-6
    assert all(lrs == [1e-6] for lrs in scheduler_base_lrs(scheduler))


def test_scheduler_base_lrs_reaches_sequential_children():
    # SequentialLR keeps no base_lrs of its own; both wrapped schedulers do.
    optimizer = _optimizer(2e-4)
    scheduler = build_drivor_scheduler(optimizer, total_steps=1000, warmup_ratio=0.1)
    groups = scheduler_base_lrs(scheduler)
    assert len(groups) == 2
    assert all(lrs == [2e-4] for lrs in groups)
    # And the un-wrapped case still returns something writable.
    plain = build_drivor_scheduler(_optimizer(1e-4), total_steps=10, warmup_ratio=0.0)
    assert scheduler_base_lrs(plain) == [[1e-4]]


def test_lr_probe_sweeps_geometrically_between_the_endpoints():
    lr_min, lr_max, steps = 1e-5, 3e-3, 900
    optimizer = _optimizer(lr_min)
    probe = build_lr_probe(optimizer, steps=steps, lr_min=lr_min, lr_max=lr_max)

    assert abs(_lr(optimizer) - lr_min) < 1e-12
    seen = []
    for _ in range(steps):
        seen.append(_lr(optimizer))
        probe.step()
    assert abs(_lr(optimizer) - lr_max) / lr_max < 1e-9
    # Geometric means a constant ratio between consecutive steps.
    ratios = [b / a for a, b in zip(seen[:-1], seen[1:])]
    assert max(ratios) - min(ratios) < 1e-9
    # And it holds at lr_max instead of overshooting if the loop runs long.
    for _ in range(50):
        probe.step()
    assert abs(_lr(optimizer) - lr_max) / lr_max < 1e-9
