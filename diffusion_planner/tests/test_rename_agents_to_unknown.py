"""rename_agents_to_unknown: the training-time, class-agnostic "rename to Unknown with
probability p" augmentation (data_augmentation.py). Off by default (prob=0.0)."""

from unittest.mock import patch

import torch

from diffusion_planner.grpo_utils import expand_batch
from diffusion_planner.utils.data_augmentation import rename_agents_to_unknown

B, N, T, D = 2, 4, 5, 12


def _make_batch():
    x = torch.zeros(B, N, T, D)
    # Two valid agents per scene (vehicle, pedestrian); the rest are padding.
    x[:, 0, :, 0] = 1.0
    x[:, 0, :, 8] = 1.0  # vehicle
    x[:, 1, :, 1] = 1.0
    x[:, 1, :, 9] = 1.0  # pedestrian
    return x


def test_prob_zero_is_a_true_noop_and_draws_no_random_numbers():
    x = _make_batch()
    original = x.clone()
    with patch("torch.rand") as mock_rand:
        out = rename_agents_to_unknown(x, 0.0)
    mock_rand.assert_not_called()
    assert out is x
    assert torch.equal(out, original)


def test_prob_one_renames_all_valid_agents_every_timestep():
    x = _make_batch()
    out = rename_agents_to_unknown(x, 1.0)

    # Valid agents (slots 0, 1) become Unknown at every past timestep.
    for slot in (0, 1):
        assert torch.equal(
            out[:, slot, :, 8:12], torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(B, T, 4)
        )
    # Padding slots (2, 3) are untouched (still all-zero).
    assert torch.all(out[:, 2:, :, :] == 0.0)
    # Kinematic columns (0:8, excluding the type one-hot) are bit-identical.
    original = _make_batch()
    assert torch.equal(out[..., :6], original[..., :6])
    assert torch.equal(out[..., 6:8], original[..., 6:8])


def test_renamed_agent_is_consistent_across_all_timesteps():
    """An agent's class must not flip frame-to-frame -- either all timesteps are Unknown or
    none are, never a mix."""
    torch.manual_seed(0)
    x = _make_batch()
    out = rename_agents_to_unknown(x, 0.5)
    is_unknown_per_t = out[..., 11] == 1.0  # [B, N, T]
    all_t_agree = (is_unknown_per_t.all(dim=-1) == is_unknown_per_t.any(dim=-1)).all()
    assert bool(all_t_agree)


def test_padding_slots_never_renamed():
    torch.manual_seed(1)
    x = _make_batch()
    out = rename_agents_to_unknown(x, 1.0)  # even at prob=1.0
    assert torch.all(out[:, 2:, :, 8:12] == 0.0)


def test_grpo_group_invariant_rename_before_expand():
    """Mirrors _grpo_step's actual call order: rename runs on the per-scene batch BEFORE
    expand_batch, so every one of the n expanded copies of a scene gets an identical rename
    decision -- required for comparable group advantages."""
    torch.manual_seed(2)
    raw = {"neighbor_agents_past": _make_batch()}
    n = 8
    raw["neighbor_agents_past"] = rename_agents_to_unknown(
        raw["neighbor_agents_past"], 0.5
    )
    exp = expand_batch(raw, n)
    onehot = exp["neighbor_agents_past"][..., 8:12]  # [B*n, N, T, 4]
    onehot = onehot.view(B, n, N, T, 4)
    for b in range(B):
        first = onehot[b, 0]
        for k in range(1, n):
            assert torch.equal(onehot[b, k], first)
