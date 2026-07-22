"""SelfAttentionBlock: pre-LN must be applied to query, key and value alike
(regression test for the q-only LayerNorm inherited from upstream)."""

import torch
from diffusion_planner.model.module.encoder import SelfAttentionBlock


def _block(dim=32, heads=4):
    torch.manual_seed(0)
    return SelfAttentionBlock(dim=dim, heads=heads, dropout=0.0).eval()


def test_attn_receives_normalized_qkv():
    blk = _block()
    x = torch.randn(2, 7, 32) * 10.0  # large scale so norm1(x) != x
    mask = torch.zeros(2, 7, dtype=torch.bool)

    captured = {}
    orig_forward = blk.attn.forward

    def spy(query, key, value, **kwargs):
        captured["q"], captured["k"], captured["v"] = query, key, value
        return orig_forward(query, key, value, **kwargs)

    blk.attn.forward = spy
    blk(x, mask)

    expected = blk.norm1(x)
    assert torch.allclose(captured["q"], expected)
    assert torch.allclose(captured["k"], expected)
    assert torch.allclose(captured["v"], expected)


def test_forward_shape_and_grad_with_padding_mask():
    blk = _block()
    x = torch.randn(2, 10, 32, requires_grad=True)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 7:] = True  # padded tail

    out = blk(x, mask)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert torch.isfinite(x.grad).all()
