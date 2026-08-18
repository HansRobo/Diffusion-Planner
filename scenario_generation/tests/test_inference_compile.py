"""Bit-exactness contract for compiled closed-loop inference.

What is pinned here: a compiled module reproduces an eager module that gave up
``nn.MultiheadAttention``'s fused fastpath, bit-for-bit -- and does NOT reproduce one that kept
it. That is the whole reason ``ClosedLoopEvalConfig`` couples ``compile_backend`` to
``canonical_attention``; if the coupling is ever dropped, turning on a speed knob starts moving
closed-loop metrics.

Why the gate is at the attention layer and not the whole planner
---------------------------------------------------------------
A whole-model check needs weights that are actually sensitive to the difference, and a
randomly-initialised ``Diffusion_Planner`` is not: swept over seeds, input scales, batch sizes
and neighbour-mask densities on both an RTX 4090 and an H100, fastpath and decomposed agree
bit-for-bit on random weights every time. The same sweep with a TRAINED checkpoint disagrees
everywhere. So a whole-model test built on random weights would pass no matter what the code
did -- which is exactly how an earlier synthetic benchmark came to report "BIT-IDENTICAL" while
the real rollout diverged. The checkpoint-backed whole-model check therefore lives outside the
test suite (it needs a checkpoint and a dataset); what is pinned in CI is the mechanism, on an
input proven able to tell the two paths apart.

Every exactness assertion below is preceded by a guard asserting its input still discriminates.
If a torch upgrade makes the two attention paths agree, the guard fails loudly instead of
letting the exactness assertion pass vacuously.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from scenario_generation.closed_loop_evaluation import ClosedLoopEvalConfig, RolloutParams
from scenario_generation.inference_compile import (
    canonical_attention,
    compiled_for_inference,
    fastpath_enabled,
    mark_inference_step,
)

# The DiT's self-attention shape (hidden_dim=256, num_heads=8, 1 ego + 320 neighbour tokens),
# with part of the batch padding-masked -- the configuration the closed-loop rollout runs.
DIM, HEADS, TOKENS = 256, 8, 321


def _attention(device: str) -> nn.MultiheadAttention:
    torch.manual_seed(0)
    return nn.MultiheadAttention(DIM, HEADS, 0.1, batch_first=True).to(device).eval()


def _inputs(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    x = torch.randn(1, TOKENS, DIM, device=device)
    key_padding_mask = torch.zeros(1, TOKENS, dtype=torch.bool, device=device)
    key_padding_mask[:, TOKENS // 2 :] = True  # ego (index 0) is never masked
    return x, key_padding_mask


def _forward(module, x, mask, *, fastpath: bool) -> torch.Tensor:
    torch.backends.mha.set_fastpath_enabled(fastpath)
    try:
        with torch.no_grad():
            return module(x, x, x, key_padding_mask=mask, need_weights=False)[0].clone()
    finally:
        torch.backends.mha.set_fastpath_enabled(True)


def _assert_compiled_matches_decomposed(device: str, backend: str) -> None:
    mha = _attention(device)
    x, mask = _inputs(device)

    fused = _forward(mha, x, mask, fastpath=True)
    decomposed = _forward(mha, x, mask, fastpath=False)
    assert not torch.equal(fused, decomposed), (
        "the fused MHA fastpath and the decomposed path now agree on this input, so the "
        "bit-exactness assertion below would pass without testing anything. Pick an input that "
        "still separates them before trusting this test again."
    )

    with canonical_attention():
        assert not fastpath_enabled()
        compiled = torch.compile(mha, backend=backend)
        got = _forward(compiled, x, mask, fastpath=False)

    torch.testing.assert_close(got, decomposed, rtol=0, atol=0)
    assert not torch.equal(got, fused), (
        "a compiled module matched the FUSED path, which dynamo cannot trace. Either torch "
        "learned to trace it or this test stopped compiling; both invalidate the coupling in "
        "ClosedLoopEvalConfig."
    )


def test_compiled_attention_matches_decomposed_path_cpu() -> None:
    """The mechanism, on CPU so it runs everywhere: compiling forgoes the fused fastpath."""
    _assert_compiled_matches_decomposed("cpu", "aot_eager")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cudagraphs backend requires CUDA")
def test_compiled_attention_matches_decomposed_path_cudagraphs() -> None:
    """The backend the closed-loop evaluation actually uses."""
    _assert_compiled_matches_decomposed("cuda", "cudagraphs")


def test_canonical_attention_restores_previous_setting() -> None:
    assert fastpath_enabled()
    with canonical_attention():
        assert not fastpath_enabled()
    assert fastpath_enabled()


def test_canonical_attention_restores_after_an_exception() -> None:
    with pytest.raises(ValueError), canonical_attention():
        raise ValueError
    assert fastpath_enabled()


class _FakeDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dit = nn.Linear(4, 4)


class _FakePlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = _FakeDecoder()


def test_compiled_for_inference_requires_canonical_attention() -> None:
    """The trap: the fastpath flag is not in dynamo's guard set.

    Disabling it after compilation leaves the already-traced graph in place while the flag reads
    back as disabled, so the run silently measures the wrong graph. Compiling outside
    ``canonical_attention()`` is the only way to reach that state, so it is refused.
    """
    with pytest.raises(RuntimeError, match="canonical_attention"):
        with compiled_for_inference(_FakePlanner(), backend="cudagraphs"):
            pass


def test_compiled_for_inference_rejects_a_model_it_cannot_compile() -> None:
    """A non-torch model (e.g. the ONNX adapter) must fail loudly, not benchmark uncompiled."""

    class _OnnxLike:
        def __call__(self, data):
            return None, {}

    with canonical_attention(), pytest.raises(TypeError, match="no .encoder"):
        with compiled_for_inference(_OnnxLike(), backend="cudagraphs"):
            pass


def test_compiled_for_inference_restores_the_model() -> None:
    """The training loop hands over its live model; it must come back unwrapped.

    A left-behind wrapper renames the decoder's parameter keys (``_orig_mod.`` prefixes), which
    would then be written into the next checkpoint.
    """
    model = _FakePlanner()
    encoder, dit = model.encoder, model.decoder.dit

    with canonical_attention(), compiled_for_inference(model, backend="aot_eager"):
        assert model.encoder is not encoder
        assert model.decoder.dit is not dit

    assert model.encoder is encoder
    assert model.decoder.dit is dit
    assert set(model.state_dict()) == {
        "encoder.weight",
        "encoder.bias",
        "decoder.dit.weight",
        "decoder.dit.bias",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cudagraphs backend requires CUDA")
def test_compiled_for_inference_drives_a_training_shaped_model() -> None:
    """The training loop's shape: a live model whose parameters still require grad.

    ``closed_loop_validate`` hands its in-training model straight to the evaluator, so the
    cudagraphs path has to tolerate ``requires_grad`` parameters under ``no_grad`` and give the
    model back untouched. A module with no attention is used deliberately: with the arithmetic
    held fixed, compiled and eager must agree exactly, so any difference here is the plumbing.
    """
    model = _FakePlanner().cuda()
    assert all(p.requires_grad for p in model.parameters())
    x = torch.randn(2, 4, device="cuda")

    with torch.no_grad():
        expected = model.decoder.dit(model.encoder(x)).clone()

    with canonical_attention(), compiled_for_inference(model, backend="cudagraphs"):
        for _ in range(3):  # replay, not just capture
            mark_inference_step()
            with torch.no_grad():
                got = model.decoder.dit(model.encoder(x)).clone()

    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert all(p.requires_grad for p in model.parameters())
    assert model.encoder is not None and not hasattr(model.encoder, "_orig_mod")


def test_config_couples_compile_to_canonical_attention() -> None:
    """Asking for compile alone must not leave the eager baseline on different arithmetic."""
    config = ClosedLoopEvalConfig(
        out_dir="/tmp/unused", params=RolloutParams(), compile_backend="cudagraphs"
    )
    assert config.canonical_attention is True


def test_config_leaves_attention_alone_by_default() -> None:
    """The default must reproduce today's numbers exactly."""
    config = ClosedLoopEvalConfig(out_dir="/tmp/unused", params=RolloutParams())
    assert config.compile_backend is None
    assert config.canonical_attention is False
