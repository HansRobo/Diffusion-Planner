"""Generic x0-prediction flow-matching utilities."""

from __future__ import annotations

from collections.abc import Callable

import torch

X0Model = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
StateProjector = Callable[[torch.Tensor], torch.Tensor]


def sample_time(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    mean: float,
    std: float,
) -> torch.Tensor:
    """Sample `(B,)` flow times from a logistic-normal distribution."""
    normal = torch.randn(batch_size, device=device, dtype=dtype)
    return torch.sigmoid(normal * std + mean)


def predict_velocity(
    x0_model: X0Model,
    state: torch.Tensor,
    time: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Convert an x0 prediction into an ODE velocity."""
    x0_prediction = x0_model(state, time)
    time_view = time.reshape(time.shape[0], *([1] * (state.ndim - 1)))
    return (x0_prediction - state) / (1 - time_view).clamp_min(epsilon)


def euler_step(
    x0_model: X0Model,
    state: torch.Tensor,
    time: torch.Tensor,
    next_time: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Advance one x0-prediction ODE step with Euler integration."""
    velocity = predict_velocity(x0_model, state, time, epsilon)
    step_size = (next_time - time).reshape(
        time.shape[0], *([1] * (state.ndim - 1))
    )
    return state + step_size * velocity


def heun_step(
    x0_model: X0Model,
    state: torch.Tensor,
    time: torch.Tensor,
    next_time: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Advance one x0-prediction ODE step with Heun integration."""
    velocity = predict_velocity(x0_model, state, time, epsilon)
    step_size = (next_time - time).reshape(
        time.shape[0], *([1] * (state.ndim - 1))
    )
    euler_state = state + step_size * velocity
    next_velocity = predict_velocity(x0_model, euler_state, next_time, epsilon)
    return state + step_size * 0.5 * (velocity + next_velocity)


def sample(
    x0_model: X0Model,
    initial_state: torch.Tensor,
    num_steps: int,
    epsilon: float,
    project_state: StateProjector,
) -> torch.Tensor:
    """Integrate with Heun steps followed by one final Euler step."""
    batch = initial_state.shape[0]
    state = project_state(initial_state)
    timesteps = torch.linspace(
        0.0,
        1.0,
        num_steps + 1,
        device=state.device,
        dtype=state.dtype,
    )
    for step in range(num_steps - 1):
        time = timesteps[step].expand(batch)
        next_time = timesteps[step + 1].expand(batch)
        state = project_state(
            heun_step(x0_model, state, time, next_time, epsilon)
        )
    state = euler_step(
        x0_model,
        state,
        timesteps[-2].expand(batch),
        timesteps[-1].expand(batch),
        epsilon,
    )
    return project_state(state)
