import torch


def zero_padding_mask(x: torch.Tensor, feature_dim: int | None = None) -> torch.Tensor:
    features = x if feature_dim is None else x[..., :feature_dim]
    return torch.sum(torch.ne(features, 0), dim=-1) == 0


def pose_padding_mask(x: torch.Tensor) -> torch.Tensor:
    return zero_padding_mask(x)


def neighbor_past_padding_mask(x: torch.Tensor) -> torch.Tensor:
    return zero_padding_mask(x, feature_dim=8)


def lane_point_padding_mask(x: torch.Tensor) -> torch.Tensor:
    return zero_padding_mask(x, feature_dim=8)


def static_object_padding_mask(x: torch.Tensor) -> torch.Tensor:
    return zero_padding_mask(x, feature_dim=10)
