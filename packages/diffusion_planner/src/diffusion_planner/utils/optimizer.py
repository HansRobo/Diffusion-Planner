"""Muon optimizer construction with an auxiliary AdamW."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch import optim

ParamEntry = tuple[str, nn.Parameter]
ParamGroups = dict[str, list[ParamEntry]]

# Affine normalization parameters and lookup tables are optimized without decay.
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

def classify_params(
    model: nn.Module, output_layers: Sequence[nn.Module]
) -> ParamGroups:
    """Classify trainable parameters into Muon and auxiliary AdamW groups."""
    groups: ParamGroups = {
        "muon": [],
        "adamw_decay": [],
        "adamw_no_decay": [],
    }
    output_parameter_ids = {
        id(parameter)
        for layer in output_layers
        for parameter in layer.parameters()
        if parameter.requires_grad
    }

    for module_name, module in model.named_modules():
        no_decay_module = isinstance(module, NO_DECAY_MODULES)
        for param_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            name = f"{module_name}.{param_name}" if module_name else param_name

            if no_decay_module or parameter.ndim <= 1:
                groups["adamw_no_decay"].append((name, parameter))
            elif parameter.ndim != 2 or id(parameter) in output_parameter_ids:
                groups["adamw_decay"].append((name, parameter))
            else:
                groups["muon"].append((name, parameter))

    _validate_param_groups(model, groups, output_parameter_ids)
    return groups


def _validate_param_groups(
    model: nn.Module, groups: ParamGroups, output_parameter_ids: set[int]
) -> None:
    """Verify that every trainable parameter is classified exactly once."""
    expected = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    occurrences: dict[int, list[str]] = {}
    for entries in groups.values():
        for name, parameter in entries:
            occurrences.setdefault(id(parameter), []).append(name)

    missing = [
        name for identifier, name in expected.items() if identifier not in occurrences
    ]
    duplicated = [names for names in occurrences.values() if len(names) != 1]
    unexpected = [
        names[0]
        for identifier, names in occurrences.items()
        if identifier not in expected
    ]
    foreign_output_parameters = output_parameter_ids.difference(expected)
    if missing or duplicated or unexpected or foreign_output_parameters:
        raise ValueError(
            "Invalid optimizer parameter classification: "
            f"missing={missing}, duplicated={duplicated}, unexpected={unexpected}, "
            f"foreign_output_parameters={len(foreign_output_parameters)}"
        )

    invalid_muon = [name for name, parameter in groups["muon"] if parameter.ndim != 2]
    if invalid_muon:
        raise ValueError(f"Muon parameters must be 2-D: {invalid_muon}")

class MuonWithAuxAdamW(optim.Optimizer):
    """Expose a Muon and an auxiliary AdamW through one Optimizer interface."""

    def __init__(self, muon: optim.Muon, adamw: optim.AdamW) -> None:
        self.muon = muon
        self.adamw = adamw
        parameters = [
            parameter
            for optimizer in self.optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self._initializing_wrapper = True
        super().__init__(parameters, defaults={})
        self._initializing_wrapper = False
        # Schedulers and GradScaler must operate on the real inner groups.
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]
        self._refresh_state_view()

    @property
    def optimizers(self) -> tuple[optim.Optimizer, optim.Optimizer]:
        return self.muon, self.adamw

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(
        self, closure: Callable[[], torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        self._refresh_state_view()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self._refresh_state_view()

    def _refresh_state_view(self) -> None:
        state: defaultdict[torch.Tensor, Any] = defaultdict(dict)
        state.update(self.muon.state)
        state.update(self.adamw.state)
        self.state = state

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if getattr(self, "_initializing_wrapper", False):
            super().add_param_group(param_group)
            return
        raise RuntimeError(
            "Add parameters to the inner Muon or AdamW optimizer explicitly"
        )


def build_optimizer(
    model: nn.Module,
    output_layers: Sequence[nn.Module],
    learning_rate: float,
    weight_decay: float,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    muon_eps: float = 1e-7,
    adamw_betas: tuple[float, float] = (0.9, 0.999),
    adamw_eps: float = 1e-8,
    verbose: bool = False,
) -> MuonWithAuxAdamW:
    """Build Muon for hidden matrices and AdamW for explicit output layers and scalars."""
    groups = classify_params(model, output_layers)
    optimizer = MuonWithAuxAdamW(
        muon=optim.Muon(
            [{"params": [parameter for _, parameter in groups["muon"]]}],
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            eps=muon_eps,
            adjust_lr_fn="match_rms_adamw",
        ),
        adamw=optim.AdamW(
            [
                {
                    "params": [parameter for _, parameter in groups["adamw_decay"]],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [parameter for _, parameter in groups["adamw_no_decay"]],
                    "weight_decay": 0.0,
                },
            ],
            lr=learning_rate,
            betas=adamw_betas,
            eps=adamw_eps,
        ),
    )

    if verbose:
        labels = ("Muon", "AdamW decay", "AdamW no-decay")
        for label, group in zip(labels, optimizer.param_groups, strict=True):
            num_params = sum(parameter.numel() for parameter in group["params"])
            print(
                f"Optimizer [{label}]: lr={group['lr']:g}, "
                f"weight_decay={group['weight_decay']:g}, "
                f"{len(group['params'])} tensors, {num_params} params"
            )

    return optimizer
