"""load_weights_only must migrate a restructured intent head, not die on it.

The head is trained separately from the policy and is the one module we expect to
redesign, so initializing a new head from a policy checkpoint that carries an older
head is a routine operation. It must reinitialize the head as a unit while keeping
strict loading for the policy.
"""

import pytest
import torch
import torch.nn as nn
from diffusion_planner.train import load_weights_only

HEAD = "decoder.turn_indicator_predictor."


class _Stub(nn.Module):
    """Minimal stand-in exposing a policy tensor plus a named intent head."""

    def __init__(self, head_keys):
        super().__init__()
        self.policy = nn.Parameter(torch.zeros(4))
        self._head_keys = dict(head_keys)
        for i, (_name, shape) in enumerate(head_keys.items()):
            self.register_parameter(f"h{i}", nn.Parameter(torch.zeros(*shape)))
        self._names = [f"h{i}" for i in range(len(head_keys))]

    def state_dict(self, *a, **k):  # noqa: D102
        out = {"decoder.policy": self.policy.detach()}
        for name, key in zip(self._names, self._head_keys, strict=True):
            out[key] = getattr(self, name).detach()
        return out

    def load_state_dict(self, state, strict=True):  # noqa: D102
        expected = set(self.state_dict())
        got = set(state)
        return torch.nn.modules.module._IncompatibleKeys(
            sorted(expected - got), sorted(got - expected)
        )


def _write(tmp_path, head_keys):
    ckpt = {"model": {"decoder.policy": torch.ones(4)}}
    for key, shape in head_keys.items():
        ckpt["model"][key] = torch.ones(*shape)
    path = tmp_path / "init.pth"
    torch.save(ckpt, path)
    return str(path)


OLD_HEAD = {
    f"{HEAD}scene_attention.in_proj_weight": (12, 4),
    f"{HEAD}classifier.fc1.weight": (8, 16),
}
# Restructured at CONSTANT widths: classifier.fc1 keeps its exact shape and only the
# attention module is renamed. This is the num_queries=1 case that a shape-only check
# silently missed.
NEW_HEAD_SAME_SHAPES = {
    f"{HEAD}attention_layers.0.in_proj_weight": (12, 4),
    f"{HEAD}context_proj.weight": (4, 12),
    f"{HEAD}classifier.fc1.weight": (8, 16),
}
# Restructured with a widened classifier: the num_queries>1 case.
NEW_HEAD_WIDER = {
    f"{HEAD}attention_layers.0.in_proj_weight": (12, 4),
    f"{HEAD}classifier.fc1.weight": (8, 28),
}


@pytest.mark.parametrize("new_head", [NEW_HEAD_SAME_SHAPES, NEW_HEAD_WIDER])
def test_restructured_head_is_migrated_not_rejected(tmp_path, new_head):
    path = _write(tmp_path, OLD_HEAD)
    model = _Stub(new_head)
    load_weights_only(path, model, torch.device("cpu"))


def test_identical_head_contract_loads_with_no_migration(tmp_path):
    path = _write(tmp_path, OLD_HEAD)
    load_weights_only(path, _Stub(OLD_HEAD), torch.device("cpu"))


def test_a_genuinely_broken_policy_tensor_still_raises(tmp_path):
    """The migration must not become a blanket escape hatch for the policy."""
    ckpt = {"model": {"decoder.unrelated_policy_tensor": torch.ones(4)}}
    for key, shape in OLD_HEAD.items():
        ckpt["model"][key] = torch.ones(*shape)
    path = tmp_path / "bad.pth"
    torch.save(ckpt, path)
    with pytest.raises(RuntimeError, match="not compatible"):
        load_weights_only(str(path), _Stub(NEW_HEAD_WIDER), torch.device("cpu"))
