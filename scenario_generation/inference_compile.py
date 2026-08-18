"""``torch.compile`` for closed-loop inference, in a form that keeps the numbers reproducible.

Compiling the planner is worth ~1.4x on the model forward, but turning it on silently moves
closed-loop metrics unless one other switch is moved with it. This module exists to make that
coupling explicit and hard to get wrong.

Why the numbers move
--------------------
``nn.MultiheadAttention`` (used by both the encoder and the DiT) takes a fused fastpath in eval
mode -- ``torch._native_multi_head_attention``. Dynamo cannot trace it, so a compiled model
always runs the decomposed ``F.multi_head_attention_forward`` path instead. Both are correct;
they round differently in the last float32 bit. So a compiled model can never be bit-identical
to an eager model that took the fastpath, and no amount of backend tuning changes that -- it is
structural.

It *can* be bit-identical to an eager model that did not take the fastpath. Measured on H100
over a full 2-route closed-loop rollout (5246 steps), ``backend="cudagraphs"`` under
``canonical_attention()`` reproduced the eager run under ``canonical_attention()`` bit-for-bit:
every dumped prediction identical, and identical route-level metrics. ``eager`` repeated against
itself is likewise bit-identical, so that comparison is meaningful.

The inductor backend is NOT bit-identical to any eager reference: it carries a second difference
of its own from the reductions it generates. If reproducibility is the criterion, use
``cudagraphs`` -- which was also the faster of the two on the real rollout (39.1 vs 50.8 ms per
model forward, against 55.1 eager).

The trap
--------
``torch.backends.mha.set_fastpath_enabled`` and the SDPA backend flags are NOT part of dynamo's
guard set. Flipping them after a module has been compiled does not invalidate the compiled code,
so the process keeps running the graph built under the OLD flags while
``torch.backends...get_fastpath_enabled()`` cheerfully reports the new value. Anything that
measures "compiled, flags off" by flipping flags after ``torch.compile`` is measuring the wrong
graph. ``compiled_for_inference`` therefore refuses to run outside ``canonical_attention()``,
which is the only ordering that is guaranteed correct: flags first, compile second, and compile
is lazy so the flags must still hold at the first forward.
"""

from __future__ import annotations

import contextlib
import logging

import torch

logger = logging.getLogger(__name__)

#: Backends that reproduce eager bit-for-bit under :func:`canonical_attention`. ``cudagraphs``
#: replays the same kernels in the same order; inductor generates its own reductions and does
#: not qualify, so it is accepted but warned about.
BITEXACT_BACKENDS = frozenset({"cudagraphs"})


def fastpath_enabled() -> bool:
    """Whether ``nn.MultiheadAttention`` would currently take its fused fastpath."""
    return torch.backends.mha.get_fastpath_enabled()


@contextlib.contextmanager
def canonical_attention():
    """Run attention through the decomposed path, the way a compiled model has to.

    This is the same fastpath switch ``onnx_export_backends`` already flips for the export --
    there for traceability, here so that an eager baseline and a compiled run are comparable.
    It does NOT touch the SDPA backend flags: matching a compiled run only requires the eager
    side to give up the fused MHA kernel, and forcing math SDPA on top would be a second,
    slower change that moves the numbers again (measured: fastpath-off alone and fastpath-off
    plus math SDPA do not agree with each other).

    Process-global, so it is restored on exit and must wrap the whole evaluation -- including
    the first forward of any compiled module, since compilation is lazy.
    """
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


class _CloneEncoderOutput(torch.nn.Module):
    """Clone the encoder's output so the DiT's next cudagraph replay cannot overwrite it.

    The encoding is computed once per inference and then read by every DPM-solver step, but a
    cudagraph-managed output buffer belongs to the graph that produced it and is reused on the
    next replay. Without the clone the decoder reads a buffer the DiT has already written over.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs).clone()


@contextlib.contextmanager
def compiled_for_inference(model, backend: str = "cudagraphs"):
    """Compile the encoder and the DiT for the duration, then put the model back as it was.

    Scoped rather than permanent because the training loop hands its LIVE model to closed-loop
    validation: compiling in place would leave ``model.encoder`` wrapped and rename the decoder's
    parameter keys (``_orig_mod.`` prefixes), which would then land in the next checkpoint.

    The DiT is what the DPM solver calls once per solver step, so that is where per-inference
    cost sits; the encoder runs once per inference and is untouched by compiling the DiT alone,
    hence both.

    Callers must additionally:

    - be inside :func:`canonical_attention` (enforced -- see the module docstring),
    - run inference under ``no_grad`` (an output carrying an autograd graph drops compile off the
      CUDA-graph fast path silently, giving the whole speedup back),
    - call :func:`mark_inference_step` once per inference.
    """
    if fastpath_enabled():
        raise RuntimeError(
            "compiled_for_inference() must be used inside canonical_attention(). The MHA "
            "fastpath is not in dynamo's guard set, so disabling it after compilation does not "
            "rebuild the graph -- the process would keep running the graph traced with the "
            "fastpath setting that was live at the first forward, and the compiled run would "
            "not be comparable to any eager baseline."
        )
    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)
    if not isinstance(encoder, torch.nn.Module) or not isinstance(
        getattr(decoder, "dit", None), torch.nn.Module
    ):
        # e.g. the ONNX adapter from simulate.load_onnx_model, which is a plain callable. Refuse
        # loudly: silently skipping would report a "compiled" benchmark that never compiled.
        raise TypeError(
            f"{type(model).__name__} has no .encoder / .decoder.dit to compile; "
            "compile is only supported for the torch Diffusion_Planner."
        )
    if backend not in BITEXACT_BACKENDS:
        logger.warning(
            "compile backend %r is not bit-exact against an eager baseline (only %s are); "
            "closed-loop metrics will shift by the last float32 bit.",
            backend,
            sorted(BITEXACT_BACKENDS),
        )

    original_encoder, original_dit = encoder, decoder.dit
    model.encoder = _CloneEncoderOutput(torch.compile(original_encoder, backend=backend))
    decoder.dit = torch.compile(original_dit, backend=backend)
    try:
        yield model
    finally:
        model.encoder = original_encoder
        decoder.dit = original_dit


def mark_inference_step() -> None:
    """Open a new cudagraph step. No-op unless a module was compiled with cudagraphs.

    Cheap and unconditional on purpose: the rollout should not have to know whether the model it
    was handed is compiled, and skipping this on a compiled model corrupts the replayed buffers.
    """
    torch.compiler.cudagraph_mark_step_begin()
