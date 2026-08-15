import torch.nn as nn
from torch import optim

# Modules whose parameters are never decayed: normalization affine terms are scales, and
# embedding rows are looked up (not multiplied), so decay would only pull them to zero.
NO_DECAY_MODULES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.Embedding,
)

# Bare nn.Parameter tables (e.g. Encoder.route_position_embedding) are embeddings too, but
# have no owning module to key off, so they are matched by name.
NO_DECAY_NAME_KEYWORDS = ("embedding",)

# Output heads: still decayed, but kept on AdamW rather than Muon. Matched as substrings of
# the parameter's qualified name. NOTE: the encoders' own ``head`` submodules are internal
# token projections, not output heads, so they are intentionally not listed here.
MUON_EXCLUDE_NAME_KEYWORDS = (
    "trajectory_decoder.output_projection",
)


def classify_params(model: nn.Module) -> dict[str, list]:
    """Group ``model``'s trainable parameters as ``(name, parameter)`` lists.

    Returns the keys ``muon`` (decayed 2-D hidden weights), ``adamw_decay`` (decayed
    weights Muon does not handle: output heads and any non-2-D weight matrix) and
    ``adamw_no_decay`` (biases, normalization scales, embeddings).
    """
    groups: dict[str, list] = {"muon": [], "adamw_decay": [], "adamw_no_decay": []}

    for module_name, module in model.named_modules():
        no_decay_module = isinstance(module, NO_DECAY_MODULES)
        # recurse=False keeps every parameter with the module that owns it, so the
        # isinstance check above applies to exactly that module's parameters.
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            name = f"{module_name}.{param_name}" if module_name else param_name
            lowered = name.lower()

            if (
                no_decay_module
                or param.ndim <= 1
                or any(keyword in lowered for keyword in NO_DECAY_NAME_KEYWORDS)
            ):
                groups["adamw_no_decay"].append((name, param))
            elif param.ndim != 2 or any(
                keyword.lower() in lowered for keyword in MUON_EXCLUDE_NAME_KEYWORDS
            ):
                groups["adamw_decay"].append((name, param))
            else:
                groups["muon"].append((name, param))

    return groups


def _assert_muon_exclusions_matched(groups: dict[str, list]) -> None:
    """Fail loudly if a MUON_EXCLUDE keyword no longer matches anything.

    Without this, renaming a head module would silently move it back into Muon.
    """
    excluded = " ".join(name.lower() for name, _ in groups["adamw_decay"])
    missing = [k for k in MUON_EXCLUDE_NAME_KEYWORDS if k.lower() not in excluded]
    if missing:
        raise ValueError(
            f"MUON_EXCLUDE_NAME_KEYWORDS {missing} matched no parameter; "
            "update diffusion_planner/utils/optimizer.py to the current module names"
        )


class MuonWithAuxAdamW:
    """A :class:`torch.optim.Muon` and a :class:`torch.optim.AdamW` behind one interface.

    Muon cannot cover a whole model, so the trainer would otherwise have to juggle two
    optimizers through the LR scheduler, checkpointing and resume. This exposes the parts
    of the optimizer API those need: ``param_groups`` hands out the underlying group dicts
    (so an LR scheduler writing ``lr`` into them reaches the real optimizers, and each
    group keeps its own peak LR), plus ``step`` / ``zero_grad`` / ``state_dict`` /
    ``load_state_dict``.
    """

    def __init__(self, muon: optim.Muon, adamw: optim.AdamW) -> None:
        self.muon = muon
        self.adamw = adamw

    @property
    def optimizers(self) -> tuple[optim.Optimizer, ...]:
        return (self.muon, self.adamw)

    @property
    def param_groups(self) -> list[dict]:
        return [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    @property
    def state(self) -> dict:
        return {
            k: v for optimizer in self.optimizers for k, v in optimizer.state.items()
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def build_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float, verbose: bool = False
) -> MuonWithAuxAdamW:
    """Build the training optimizer for ``model``.

    Both halves share ``learning_rate``: Muon runs with ``adjust_lr_fn="match_rms_adamw"``,
    which rescales its update to AdamW's RMS so an LR tuned for AdamW carries over.
    """
    groups = classify_params(model)
    _assert_muon_exclusions_matched(groups)

    optimizer = MuonWithAuxAdamW(
        muon=optim.Muon(
            [{"params": [p for _, p in groups["muon"]]}],
            lr=learning_rate,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
        ),
        # One AdamW with two groups: the decayed weights Muon does not take, and the
        # parameters that must not be decayed at all.
        adamw=optim.AdamW(
            [
                {
                    "params": [p for _, p in groups["adamw_decay"]],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for _, p in groups["adamw_no_decay"]],
                    "weight_decay": 0.0,
                },
            ],
            lr=learning_rate,
        ),
    )

    if verbose:
        labels = ("Muon", "AdamW decay", "AdamW no-decay")
        for label, group in zip(labels, optimizer.param_groups, strict=False):
            num_params = sum(p.numel() for p in group["params"])
            print(
                f"Optimizer [{label}]: lr={group['lr']:g}, "
                f"weight_decay={group['weight_decay']:g}, "
                f"{len(group['params'])} tensors, {num_params} params"
            )

    return optimizer
