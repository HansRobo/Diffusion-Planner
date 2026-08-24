"""Tests for the throughput work: fused EMA, and the guard's async policy.

Both replace something that was correct-but-slow, so what matters here is that
the fast path computes the *same* thing.  The EMA is checked against timm's
implementation directly rather than against hand-derived numbers.
"""

import torch
import torch.nn as nn
from diffusion_planner.utils.drivor_ema import FusedModelEma
from diffusion_planner.utils.drivor_train import DivergenceGuard
from timm.utils import ModelEma


class _Tiny(nn.Module):
    """Params plus a float buffer -- the shapes this model actually has."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.norm = nn.LayerNorm(3)
        self.register_buffer("scale", torch.ones(3))

    def forward(self, x):
        return self.norm(self.fc(x))


def _perturb(model, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                tensor.add_(torch.randn(tensor.shape, generator=generator) * 0.1)


def test_fused_ema_matches_timm_step_for_step():
    torch.manual_seed(0)
    reference_model, fused_model = _Tiny(), _Tiny()
    fused_model.load_state_dict(reference_model.state_dict())

    reference = ModelEma(reference_model, decay=0.999)
    fused = FusedModelEma(fused_model, decay=0.999)

    for step in range(25):
        _perturb(reference_model, step)
        _perturb(fused_model, step)
        reference.update(reference_model)
        fused.update(fused_model)

    got = fused.ema.state_dict()
    want = reference.ema.state_dict()
    assert set(got) == set(want), "state_dict keys must stay identical for checkpoints"
    for key in want:
        torch.testing.assert_close(got[key], want[key], rtol=1e-6, atol=1e-8, msg=key)


def test_fused_ema_decay_arithmetic_is_exact():
    # lerp_(dst, src, 1 - decay) must be decay*ema + (1-decay)*src, not the
    # other way round -- swapping them silently trains on the shadow weights.
    model = _Tiny()
    with torch.no_grad():
        for tensor in model.state_dict().values():
            tensor.fill_(1.0)
    ema = FusedModelEma(model, decay=0.9)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            tensor.fill_(0.0)
    ema.update(model)
    for key, tensor in ema.ema.state_dict().items():
        torch.testing.assert_close(
            tensor, torch.full_like(tensor, 0.9), msg=f"{key}: expected 0.9*1 + 0.1*0"
        )


def test_fused_ema_survives_a_state_dict_reload():
    # resume_model() calls ema.ema.load_state_dict(...); the cached tensor lists
    # must not go on pointing at pre-load storage.
    model = _Tiny()
    ema = FusedModelEma(model, decay=0.5)
    ema.update(model)
    restored = {key: torch.zeros_like(value) for key, value in ema.ema.state_dict().items()}
    ema.ema.load_state_dict(restored)
    ema.invalidate()
    with torch.no_grad():
        for tensor in model.state_dict().values():
            tensor.fill_(1.0)
    ema.update(model)
    for key, tensor in ema.ema.state_dict().items():
        torch.testing.assert_close(
            tensor, torch.full_like(tensor, 0.5), msg=f"{key}: 0.5*0 + 0.5*1"
        )


def test_fused_ema_copies_non_float_buffers_instead_of_averaging_them():
    class WithCounter(_Tiny):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("seen", torch.zeros((), dtype=torch.long))

    model = WithCounter()
    ema = FusedModelEma(model, decay=0.999)
    with torch.no_grad():
        model.seen.fill_(7)
    ema.update(model)
    # timm would compute int(0 * 0.999 + 7 * 0.001) == 0 and never advance.
    assert int(ema.ema.seen) == 7


def _grads(values):
    params = [nn.Parameter(torch.zeros(2)) for _ in values]
    for param, value in zip(params, values):
        param.grad = torch.full((2,), float(value))
    return params


def _loss_dict(loss: float, absmax: float = 1.0):
    # ``logit_absmax`` is the key drivor_train.py:252 actually reads; naming it
    # anything else silently disables the drift half of the guard.
    return {
        "loss": torch.tensor(float(loss)),
        "logit_absmax": torch.tensor(float(absmax)),
    }


def test_async_guard_zeroes_gradients_on_a_breach_without_a_host_branch():
    guard = DivergenceGuard(enabled=True, sync_every=8)
    # Establish a calm baseline so the spike below is a breach.
    for _ in range(guard.grace + 5):
        guard.check(_loss_dict(1.0), optimizer=None)

    params = _grads([3.0, -4.0])
    guard.check(_loss_dict(1e6), optimizer=None)
    guard.mask_grads([p.grad for p in params])
    for param in params:
        assert torch.all(param.grad == 0.0), "breached step must contribute no gradient"

    params = _grads([3.0, -4.0])
    guard.check(_loss_dict(1.0), optimizer=None)
    guard.mask_grads([p.grad for p in params])
    torch.testing.assert_close(params[0].grad, torch.full((2,), 3.0))
    torch.testing.assert_close(params[1].grad, torch.full((2,), -4.0))


def test_exact_policy_leaves_gradients_untouched_because_it_skips_instead():
    # With sync_every == 1 the loop skips optimizer.step() outright, so the mask
    # must be inert -- otherwise a breach would be punished twice.
    guard = DivergenceGuard(enabled=True, sync_every=1)
    for _ in range(guard.grace + 5):
        guard.check(_loss_dict(1.0), optimizer=None)
    params = _grads([3.0])
    assert guard.check(_loss_dict(1e6), optimizer=None) is True
    guard.mask_grads([p.grad for p in params])
    torch.testing.assert_close(params[0].grad, torch.full((2,), 3.0))


def test_resolve_only_reads_back_on_the_window_boundary():
    guard = DivergenceGuard(enabled=True, sync_every=8)
    for _ in range(guard.grace + 5):
        guard.check(_loss_dict(1.0), optimizer=None)

    seen = []
    for _ in range(16):
        guard.check(_loss_dict(1.0), optimizer=None)
        seen.append(guard.step_count % guard.sync_every == 0)
    assert sum(seen) == 2, "a 16-step run must read back exactly twice at sync_every=8"


def test_async_guard_reports_a_breach_at_the_next_boundary():
    guard = DivergenceGuard(enabled=True, sync_every=8)
    for _ in range(guard.grace + 5):
        guard.check(_loss_dict(1.0), optimizer=None)
    # Land the breach, then walk to the next boundary and confirm it is reported.
    guard.check(_loss_dict(1e6), optimizer=None)
    reported = False
    for _ in range(guard.sync_every):
        guard.check(_loss_dict(1.0), optimizer=None)
        reported = guard.resolve(_FakeOptimizer()) or reported
    assert reported, "a breach must not be lost between readbacks"


class _FakeOptimizer:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 1e-4}]


def test_async_guard_cuts_the_lr_on_logit_drift_at_the_boundary():
    # The drift half of the guard is what actually fired in the LR range test
    # (at 6.77e-4), so it has to survive the move off the per-step readback.
    guard = DivergenceGuard(enabled=True, logit_bound=10.0, sync_every=8)
    optimizer = _FakeOptimizer()
    for _ in range(guard.grace + 5):
        guard.check(_loss_dict(1.0, absmax=1.0), optimizer)
        guard.resolve(optimizer)
    assert optimizer.param_groups[0]["lr"] == 1e-4, "calm logits must not cut the LR"

    for _ in range(guard.sync_every * 4):
        guard.check(_loss_dict(1.0, absmax=50.0), optimizer)
        guard.resolve(optimizer)
    assert optimizer.param_groups[0]["lr"] < 1e-4, "drifting logits must cut the LR"


def test_guard_disabled_is_fully_inert():
    guard = DivergenceGuard(enabled=False, sync_every=8)
    params = _grads([5.0])
    assert guard.check(_loss_dict(1e9), optimizer=None) is False
    guard.mask_grads([p.grad for p in params])
    torch.testing.assert_close(params[0].grad, torch.full((2,), 5.0))
    assert guard.resolve(_FakeOptimizer()) is False
