from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from diffusion_planner.utils.onnx_export import export_onnx, validate_hdp_export_contract


def _valid_config(**overrides):
    values = {
        "use_velocity_representation": True,
        "decoder_tokenization": "temporal",
        "predicted_neighbor_num": 0,
        "time_len": 31,
        "future_len": 80,
        "policy_uses_turn_indicator_history": False,
        "turn_indicator_output_dim": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_hdp_export_contract_accepts_deployment_shape():
    validate_hdp_export_contract(_valid_config())


def test_legacy_export_repairs_non_batch_output_shape_metadata(tmp_path):
    pytest.importorskip("onnx")

    class Pair(nn.Module):
        def forward(self, value):
            return value, value[:, :2]

    output_path = tmp_path / "pair.onnx"
    export_onnx(
        Pair().eval(),
        {"value": torch.randn(1, 4)},
        ["value"],
        ["encoding", "route"],
        output_path,
        use_simplify=False,
        opset_version=20,
        external_data=False,
    )
    import onnx

    model = onnx.load(str(output_path))
    shapes = [
        [
            dimension.dim_param or dimension.dim_value
            for dimension in output.type.tensor_type.shape.dim
        ]
        for output in model.graph.output
    ]
    assert shapes == [["batch", 4], ["batch", 2]]


@pytest.mark.parametrize(
    "override, message",
    [
        ({"use_velocity_representation": False}, "velocity-based"),
        ({"decoder_tokenization": "spatial"}, "temporal"),
        ({"predicted_neighbor_num": 1}, "ego-only"),
        ({"time_len": 30}, "time_len=31"),
        ({"future_len": 79}, "future_len=80"),
        ({"policy_uses_turn_indicator_history": True}, "policy_uses_turn_indicator_history"),
        ({"turn_indicator_output_dim": 4}, "exactly 3 logits"),
    ],
)
def test_hdp_export_contract_rejects_incompatible_checkpoint(override, message):
    with pytest.raises(RuntimeError, match=message):
        validate_hdp_export_contract(_valid_config(**override))
