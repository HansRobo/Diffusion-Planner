"""Tests for the warmup + post-warmup lr schedules.

Covers the fix where the post-warmup phase held the lr constant (the old
`CosineAnnealingWarmUpRestarts`) vs. the new cosine decay, plus the
backward-compatible alias and the dispatcher.
"""

import pytest
import torch
from diffusion_planner.utils.lr_schedule import (
    CosineAnnealingWarmUpRestarts,
    WarmupConstantLR,
    WarmupCosineAnnealingLR,
    build_lr_scheduler,
)

PEAK_LR = 1e-3
WARMUP = 5
EPOCHS = 25
ETA_MIN = 1e-6


def _make_optimizer():
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=PEAK_LR)


def _run(scheduler, epochs=EPOCHS):
    """Return the lr seen at the start of each epoch (step() once per epoch)."""
    optimizer = scheduler.optimizer
    lrs = []
    for _ in range(epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return lrs


def test_warmup_ramps_from_start_factor_to_peak():
    opt = _make_optimizer()
    lrs = _run(WarmupCosineAnnealingLR(opt, EPOCHS, WARMUP, start_factor=0.1, eta_min=ETA_MIN))
    # first epoch starts at start_factor * peak, peak reached by end of warmup
    assert lrs[0] == pytest.approx(0.1 * PEAK_LR, rel=1e-6)
    assert max(lrs) == pytest.approx(PEAK_LR, rel=1e-6)
    assert lrs[WARMUP] == pytest.approx(PEAK_LR, rel=1e-6)


def test_constant_holds_peak_after_warmup():
    opt = _make_optimizer()
    lrs = _run(WarmupConstantLR(opt, EPOCHS, WARMUP, start_factor=0.1))
    # every post-warmup epoch stays at the peak lr
    post_warmup = lrs[WARMUP:]
    assert all(lr == pytest.approx(PEAK_LR, rel=1e-6) for lr in post_warmup)


def test_cosine_decays_after_warmup():
    opt = _make_optimizer()
    lrs = _run(WarmupCosineAnnealingLR(opt, EPOCHS, WARMUP, start_factor=0.1, eta_min=ETA_MIN))
    post_warmup = lrs[WARMUP:]
    # strictly decreasing after the peak, and lands near eta_min by the end
    assert all(b < a for a, b in zip(post_warmup, post_warmup[1:]))
    assert lrs[-1] < 0.5 * PEAK_LR
    assert lrs[-1] >= ETA_MIN


def test_cosine_differs_from_constant():
    opt_c = _make_optimizer()
    opt_k = _make_optimizer()
    cosine = _run(WarmupCosineAnnealingLR(opt_c, EPOCHS, WARMUP, eta_min=ETA_MIN))
    constant = _run(WarmupConstantLR(opt_k, EPOCHS, WARMUP))
    # identical through warmup, diverge afterwards
    assert cosine[:WARMUP] == pytest.approx(constant[:WARMUP], rel=1e-6)
    assert cosine[-1] < constant[-1]


def test_alias_preserves_legacy_constant_behavior():
    # The old public name must keep the exact (constant) behavior for compat.
    opt_alias = _make_optimizer()
    opt_const = _make_optimizer()
    alias = _run(CosineAnnealingWarmUpRestarts(opt_alias, EPOCHS, WARMUP))
    constant = _run(WarmupConstantLR(opt_const, EPOCHS, WARMUP))
    assert alias == pytest.approx(constant, rel=1e-6)


def test_build_lr_scheduler_dispatch():
    opt_cos = _make_optimizer()
    opt_con = _make_optimizer()
    cosine = _run(build_lr_scheduler(opt_cos, EPOCHS, WARMUP, schedule_type="cosine"))
    constant = _run(build_lr_scheduler(opt_con, EPOCHS, WARMUP, schedule_type="constant"))
    assert cosine[-1] < constant[-1]
    with pytest.raises(ValueError):
        build_lr_scheduler(_make_optimizer(), EPOCHS, WARMUP, schedule_type="bogus")


def test_cosine_handles_short_warmup_edge_case():
    # warm_up_epoch == 1 and epoch == warm_up_epoch must not raise.
    opt = _make_optimizer()
    WarmupCosineAnnealingLR(opt, 1, 1)
    opt2 = _make_optimizer()
    sched = WarmupCosineAnnealingLR(opt2, 10, 1)
    _run(sched, epochs=10)
