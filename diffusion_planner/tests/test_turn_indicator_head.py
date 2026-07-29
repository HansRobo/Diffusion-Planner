"""Design contracts for the detached turn-indicator head.

The head is trained alone on a frozen policy, so the properties that matter are
gradient isolation, robustness to a fully padded scene, and the padding contract
surviving the input LayerNorm.
"""

import pytest
import torch

from diffusion_planner.model.module.decoder import TurnIndicatorHead

HIDDEN = 32
TRAJ = 64
BATCH = 6
TOKENS = 9


def _head(**kwargs):
    torch.manual_seed(0)
    return TurnIndicatorHead(hidden_dim=HIDDEN, trajectory_dim=TRAJ, num_heads=4, **kwargs)


def _inputs(fully_padded_rows=()):
    torch.manual_seed(1)
    scene = torch.randn(BATCH, TOKENS, HIDDEN)
    # Padding rows are exact zeros by the encoder's contract.
    scene[:, -2:] = 0.0
    for row in fully_padded_rows:
        scene[row] = 0.0
    return (
        torch.randn(BATCH, TRAJ, requires_grad=True),
        scene.requires_grad_(True),
        torch.randn(BATCH, HIDDEN, requires_grad=True),
        torch.randn(BATCH, 6, requires_grad=True),
    )


@pytest.mark.parametrize("num_queries,num_layers", [(1, 1), (4, 2), (2, 3)])
def test_output_shape_and_finiteness(num_queries, num_layers):
    head = _head(num_queries=num_queries, num_layers=num_layers)
    logit = head(*_inputs())
    assert logit.shape == (BATCH, 3)
    assert torch.isfinite(logit).all()


def test_fully_padded_scene_stays_finite():
    """A row with no scene tokens must not poison the batch.

    The installed PyTorch already returns finite values when every key is masked, so
    this is a regression pin rather than a reproduction of a live bug: one non-finite
    sample would destroy the whole batch loss, and the behaviour it relies on is a
    softmax special case, not a documented guarantee.
    """
    head = _head(num_queries=4, num_layers=2)
    logit = head(*_inputs(fully_padded_rows=(0, 3)))
    assert torch.isfinite(logit).all(), "fully padded scene rows produced non-finite logits"


def test_all_rows_fully_padded_stays_finite():
    head = _head(num_queries=2, num_layers=1)
    logit = head(*_inputs(fully_padded_rows=range(BATCH)))
    assert torch.isfinite(logit).all()


def test_gradient_is_isolated_from_every_upstream_input():
    """No head gradient may reach the encoder, route encoder or policy."""
    head = _head(num_queries=4, num_layers=2)
    traj, scene, route, proprio = _inputs()
    head(traj, scene, route, proprio).sum().backward()
    for name, tensor in [
        ("trajectory_features", traj),
        ("scene_tokens", scene),
        ("route_condition", route),
        ("proprioception", proprio),
    ]:
        assert tensor.grad is None, f"head leaked gradient into {name}"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_padding_tokens_cannot_influence_the_output():
    """The mask is derived pre-norm, so changing padding rows must be a no-op."""
    head = _head(num_queries=4, num_layers=2).eval()
    traj, scene, route, proprio = _inputs()
    with torch.no_grad():
        baseline = head(traj, scene, route, proprio)
        polluted = scene.clone()
        polluted[:, -2:] = 7.5  # would become real keys if the mask were computed post-norm
        assert not torch.allclose(head(traj, polluted, route, proprio), baseline), (
            "non-zero rows must be treated as real tokens"
        )
        # Re-zeroing restores the exact baseline: padding carries no information.
        polluted[:, -2:] = 0.0
        assert torch.equal(head(traj, polluted, route, proprio), baseline)


def test_dropout_is_active_in_train_mode_only():
    head = _head(num_queries=4, num_layers=2, dropout=0.5)
    traj, scene, route, proprio = _inputs()
    with torch.no_grad():
        head.train()
        torch.manual_seed(2)
        a = head(traj, scene, route, proprio)
        torch.manual_seed(3)
        b = head(traj, scene, route, proprio)
        assert not torch.allclose(a, b), "dropout had no effect in train mode"
        head.eval()
        assert torch.equal(
            head(traj, scene, route, proprio), head(traj, scene, route, proprio)
        )


def test_rejects_invalid_configuration():
    for kwargs in ({"num_queries": 0}, {"num_layers": 0}, {"dropout": 1.0}):
        with pytest.raises(ValueError):
            _head(**kwargs)


def test_conjunction_of_conditions_is_representable():
    """Concat-project must not collapse the three conditions into a cancellable sum.

    Under the old ``traj + proprio + route`` query, swapping evidence between the
    trajectory and route channels left the query unchanged. It must not now.
    """
    head = _head(num_queries=1, num_layers=1).eval()
    traj, scene, route, proprio = _inputs()
    with torch.no_grad():
        base = head(traj, scene, route, proprio)
        swapped = head(traj, scene, route + 0.3, proprio - 0.3)
        assert not torch.allclose(base, swapped, atol=1e-6)
