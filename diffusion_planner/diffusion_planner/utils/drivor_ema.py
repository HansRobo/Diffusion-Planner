"""Multi-tensor EMA, a drop-in replacement for ``timm.utils.ModelEma``.

``timm``'s version carries its own deprecation notice, and on this training loop
it was a top-three throughput cost.  Per call it

* builds two ``state_dict()`` OrderedDicts (pure Python, every step),
* then issues ``ema_v.copy_(ema_v * decay + (1 - decay) * model_v)`` per tensor
  -- three kernels each, several hundred launches per step.

Those launches land on the one Python thread that feeds the GPU, so they do not
overlap with anything: they directly lengthen the step.  This class does the
identical arithmetic with two fused ``_foreach_`` calls over tensor lists
captured once at construction.

Deliberately unchanged, so checkpoints stay interchangeable:

* ``self.ema`` is ``deepcopy(model)`` exactly as timm does it, including the
  case where ``model`` is already DDP-wrapped -- ``state_dict`` keys (and thus
  the ``ema_state_dict`` entry written by the training loop) are byte-identical.
* the update is ``decay * ema + (1 - decay) * src``, which is what
  ``_foreach_lerp_(ema, src, 1 - decay)`` computes.

One semantic difference, and it is a fix rather than a port: timm runs the EMA
over *every* state_dict entry, so an integer buffer gets integer-truncated
``ema * decay + ...`` arithmetic.  Non-floating tensors are copied here instead.
This model has no integer buffers today (the only registered buffers are the
float ``state_mean``/``state_std`` pair in ``drivor_decoder.py``), so the branch
exists to keep the class honest if one is ever added.
"""

from __future__ import annotations

from copy import deepcopy

import torch


class FusedModelEma:
    """EMA of the full ``state_dict`` using multi-tensor ops.

    Args:
        model: the live model.  Wrapped or unwrapped, matching timm's contract.
        decay: EMA decay; the update is ``decay * ema + (1 - decay) * model``.
        device: optional device to hold the shadow copy on.  Passing ``""``
            keeps it wherever ``deepcopy`` put it, as in timm.
    """

    def __init__(self, model, decay: float = 0.999, device: str = "") -> None:
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = float(decay)
        self.device = device
        self.ema_has_module = hasattr(self.ema, "module")
        if device:
            self.ema.to(device=device)
        for param in self.ema.parameters():
            param.requires_grad_(False)
        self._float_dst: list[torch.Tensor] = []
        self._float_src: list[torch.Tensor] = []
        self._other_dst: list[torch.Tensor] = []
        self._other_src: list[torch.Tensor] = []
        self._bound_to: int | None = None

    def _bind(self, model) -> None:
        """Capture the tensor lists once, so ``update`` builds no dicts.

        Both sides are views into live storage, so this stays valid for the rest
        of training.  It is invalidated only by something that rebinds a
        parameter's storage -- ``load_state_dict`` with ``assign=True``, or a
        resume that replaces ``self.ema`` -- and ``_bound_to`` catches that by
        identity.
        """
        needs_module = hasattr(model, "module") and not self.ema_has_module
        source = model.state_dict()
        self._float_dst, self._float_src = [], []
        self._other_dst, self._other_src = [], []
        for key, dst in self.ema.state_dict().items():
            src = source["module." + key if needs_module else key].detach()
            if self.device:
                src = src.to(device=self.device)
            if dst.is_floating_point() and src.is_floating_point():
                self._float_dst.append(dst)
                self._float_src.append(src)
            else:
                self._other_dst.append(dst)
                self._other_src.append(src)
        self._bound_to = id(model)

    @torch.no_grad()
    def update(self, model) -> None:
        if self._bound_to != id(model):
            self._bind(model)
        # lerp_(dst, src, w) == dst + w * (src - dst) == (1 - w) * dst + w * src,
        # so w = 1 - decay reproduces timm's decay * ema + (1 - decay) * model.
        if self._float_dst:
            torch._foreach_lerp_(self._float_dst, self._float_src, 1.0 - self.decay)
        for dst, src in zip(self._other_dst, self._other_src):
            dst.copy_(src)

    def set(self, model) -> None:
        """Hard-copy the live weights into the shadow copy (timm parity)."""
        if self._bound_to != id(model):
            self._bind(model)
        with torch.no_grad():
            for dst, src in zip(
                self._float_dst + self._other_dst, self._float_src + self._other_src
            ):
                dst.copy_(src)

    def invalidate(self) -> None:
        """Force a re-bind, e.g. after loading ``self.ema`` from a checkpoint."""
        self._bound_to = None
