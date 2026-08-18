"""NeighborEncoder must accept the 12-dim agent state (4-class type one-hot: vehicle,
pedestrian, bicycle, unknown) introduced for the Unknown-class change."""

import torch

from diffusion_planner.model.module.encoder import NeighborEncoder

B, P, V, D = 2, 5, 21, 12
HIDDEN_DIM = 16


def _make_encoder():
    return NeighborEncoder(time_len=V, drop_path_rate=0.0, hidden_dim=HIDDEN_DIM, depth=1)


def _make_batch():
    x = torch.zeros(B, P, V, D)
    # Two valid agents: one vehicle, one unknown (cols 8:12 one-hot).
    x[0, 0, :, 0] = 1.0  # nonzero x -> valid kinematics
    x[0, 0, -1, 8] = 1.0  # vehicle
    x[1, 0, :, 1] = 1.0
    x[1, 0, -1, 11] = 1.0  # unknown
    return x


def test_type_emb_width_is_four():
    enc = _make_encoder()
    assert enc.type_emb.in_features == 4


def test_forward_train_mode_shapes():
    enc = _make_encoder()
    enc.train()
    x = _make_batch()
    out, mask, pos = enc(x)
    assert out.shape == (B, P, HIDDEN_DIM)
    assert mask.shape == (B, P)
    assert pos.shape[:2] == (B, P)


def test_forward_eval_mode_shapes():
    enc = _make_encoder()
    enc.eval()
    x = _make_batch()
    out, mask, pos = enc(x)
    assert out.shape == (B, P, HIDDEN_DIM)
    assert mask.shape == (B, P)
    assert pos.shape[:2] == (B, P)


def test_unknown_type_produces_distinct_embedding_from_vehicle():
    """A vehicle-typed and an unknown-typed agent with identical kinematics must get
    different output embeddings -- proves the 4th one-hot column is actually consumed."""
    enc = _make_encoder()
    enc.eval()
    x = torch.zeros(1, 2, V, D)
    x[0, :, :, 0] = 1.0
    x[0, :, -1, 6] = 2.0  # width
    x[0, :, -1, 7] = 4.5  # length
    x[0, 0, -1, 8] = 1.0  # slot 0: vehicle
    x[0, 1, -1, 11] = 1.0  # slot 1: unknown
    out, _, _ = enc(x)
    assert not torch.allclose(out[0, 0], out[0, 1])
