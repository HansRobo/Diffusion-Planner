import torch


def zero_padding_mask(x: torch.Tensor, feature_dim: int | None = None) -> torch.Tensor:
    features = x if feature_dim is None else x[..., :feature_dim]
    return torch.all(features == 0, dim=-1)


def pose_padding_mask(x: torch.Tensor) -> torch.Tensor:
    return zero_padding_mask(x)


def neighbor_future_padding_mask(x: torch.Tensor) -> torch.Tensor:
    """Mask a converter-produced contiguous future while preserving internal zero poses."""
    observed = ~zero_padding_mask(x)
    time_index = torch.arange(1, x.shape[-2] + 1, device=x.device)
    time_index = time_index.reshape(*([1] * (observed.ndim - 1)), -1)
    valid_length = torch.where(observed, time_index, 0).amax(dim=-1, keepdim=True)
    return time_index > valid_length


def neighbor_past_padding_mask(x: torch.Tensor) -> torch.Tensor:
    return zero_padding_mask(x, feature_dim=8)


def lane_point_padding_mask(x: torch.Tensor) -> torch.Tensor:
    return zero_padding_mask(x, feature_dim=8)


def static_object_padding_mask(x: torch.Tensor) -> torch.Tensor:
    return zero_padding_mask(x, feature_dim=10)
