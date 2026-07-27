import torch
from diffusion_planner.dimensions import LINE_STRING_TYPE_NUM, POINTS_PER_LINE_STRING
from diffusion_planner.model.module.encoder import (
    CLASS_TYPE_LINE_STRING,
    CLASS_TYPE_NUM,
    LineEncoder,
)

POINT_DIM = 2 + LINE_STRING_TYPE_NUM
STOP_LINE = 0
ROAD_BORDER = 1


def _make_encoder():
    torch.manual_seed(0)
    encoder = LineEncoder(
        line_len=POINTS_PER_LINE_STRING,
        class_type=CLASS_TYPE_LINE_STRING,
        drop_path_rate=0.0,
        hidden_dim=32,
        depth=1,
        num_types=LINE_STRING_TYPE_NUM,
    )
    return encoder.eval()


def _short_line(type_index=STOP_LINE):
    x = torch.zeros(1, 2, POINTS_PER_LINE_STRING, POINT_DIM)
    x[0, 0, 0, :2] = torch.tensor([0.25, -0.1])
    x[0, 0, 1, :2] = torch.tensor([0.25, 0.1])
    x[0, 0, :2, 2 + type_index] = 1.0
    return x


def _capture_mixer_input(encoder, x):
    captured = {}

    def hook(module, inputs):
        captured["x"] = inputs[0].detach().clone()

    handle = encoder.channel_pre_project.register_forward_pre_hook(hook)
    with torch.no_grad():
        encoder(x)
    handle.remove()
    return captured["x"]


def test_line_pos_is_valid_centroid_and_segment_direction():
    with torch.no_grad():
        _, _, pos = _make_encoder()(_short_line())

    assert pos.shape == (1, 2, 4 + CLASS_TYPE_NUM)
    torch.testing.assert_close(pos[0, 0, :2], torch.tensor([0.25, 0.0]))
    torch.testing.assert_close(pos[0, 0, 2:4], torch.tensor([0.0, 1.0]))


def test_line_pos_geometry_does_not_depend_on_type():
    encoder = _make_encoder()
    with torch.no_grad():
        _, _, stop_pos = encoder(_short_line(STOP_LINE))
        _, _, border_pos = encoder(_short_line(ROAD_BORDER))
    torch.testing.assert_close(stop_pos[..., :4], border_pos[..., :4])


def test_line_diff_ignores_valid_to_padding_boundary():
    features = _capture_mixer_input(_make_encoder(), _short_line())

    torch.testing.assert_close(features[0, 0, 2:4], torch.tensor([0.0, 0.2]))
    assert torch.all(features[0, 1, 2:4] == 0)
    assert torch.all(features[0, 2:] == 0)


def test_line_mixer_feature_layout_is_xy_diff_without_type_channels():
    encoder = _make_encoder()
    x = _short_line()
    features = _capture_mixer_input(encoder, x)

    assert encoder.channel_pre_project.fc1.in_features == 4
    assert features.shape == (2, POINTS_PER_LINE_STRING, 4)
    torch.testing.assert_close(features[0, :, :2], x[0, 0, :, :2])
    torch.testing.assert_close(
        features[0, :, 2:], torch.tensor([[0.0, 0.2]] + [[0.0, 0.0]] * (POINTS_PER_LINE_STRING - 1))
    )


def test_line_type_is_injected_after_geometric_pooling():
    encoder = _make_encoder()
    assert encoder.type_emb.in_features == LINE_STRING_TYPE_NUM
    assert encoder.type_emb.out_features == 128

    stop = _capture_mixer_input(encoder, _short_line(STOP_LINE))
    border = _capture_mixer_input(encoder, _short_line(ROAD_BORDER))
    torch.testing.assert_close(stop, border)


def test_line_padding_element_is_masked_and_finite():
    with torch.no_grad():
        tokens, mask, pos = _make_encoder()(_short_line())

    assert mask.tolist() == [[False, True]]
    assert torch.all(tokens[0, 1] == 0)
    assert torch.all(torch.isfinite(pos))
    torch.testing.assert_close(pos[0, 1, :4], torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_closed_line_direction_has_deterministic_fallback():
    encoder = _make_encoder()
    x = torch.zeros(1, 1, POINTS_PER_LINE_STRING, POINT_DIM)
    square = torch.tensor([[0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1], [0.1, -0.1], [0.1, 0.1]])
    x[0, 0, :5, :2] = square
    x[0, 0, :5, 2 + ROAD_BORDER] = 1.0

    with torch.no_grad():
        _, _, pos = encoder(x)
    assert torch.all(torch.isfinite(pos))
    torch.testing.assert_close(pos[0, 0, :2], torch.tensor([0.02, 0.02]), atol=1e-6, rtol=0)
    torch.testing.assert_close(pos[0, 0, 2:4], torch.tensor([1.0, 0.0]))
