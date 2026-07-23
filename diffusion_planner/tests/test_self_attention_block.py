"""Regression tests for the encoder pre-LN self-attention contract."""

import torch

from diffusion_planner.model.module.encoder import SelfAttentionBlock


def _block(dim=32, heads=4):
    torch.manual_seed(0)
    return SelfAttentionBlock(dim=dim, heads=heads, dropout=0.0).eval()


def test_attn_receives_normalized_qkv():
    block = _block()
    x = torch.randn(2, 7, 32) * 10.0
    mask = torch.zeros(2, 7, dtype=torch.bool)
    captured = {}
    original_forward = block.attn.forward

    def spy(query, key, value, **kwargs):
        captured["q"], captured["k"], captured["v"] = query, key, value
        return original_forward(query, key, value, **kwargs)

    block.attn.forward = spy
    block(x, mask)
    expected = block.norm1(x)
    assert torch.allclose(captured["q"], expected)
    assert torch.allclose(captured["k"], expected)
    assert torch.allclose(captured["v"], expected)


def test_forward_shape_and_grad_with_padding_mask():
    block = _block()
    x = torch.randn(2, 10, 32, requires_grad=True)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 7:] = True
    output = block(x, mask)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(x.grad).all()
