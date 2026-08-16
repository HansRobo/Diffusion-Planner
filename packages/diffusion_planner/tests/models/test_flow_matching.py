"""Tests for generic x0-prediction flow-matching utilities."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.models.flow_matching import (
    compute_x0_flow_matching_loss,
    euler_step,
    heun_step,
    sample,
    x0_velocity_error,
)


class FlowMatchingTest(unittest.TestCase):
    def test_x0_loss_accepts_l1_function(self) -> None:
        target = torch.ones(1, 2, 3)
        loss = compute_x0_flow_matching_loss(
            x0_model=lambda _state, _time: target + 1.0,
            loss_function=lambda x_prediction, clean_target, time: x0_velocity_error(
                x_prediction - clean_target, time, 1e-5
            ).abs(),
            target=target,
            mask=torch.zeros(1, 2, dtype=torch.bool),
            time_mean=0.0,
            time_std=0.0,
            noise_scale=1.0,
        )

        torch.testing.assert_close(loss, torch.tensor(2.0))

    def test_x0_loss_masks_elements(self) -> None:
        target = torch.randn(2, 3, 4)
        mask = torch.tensor(
            [[False, False, True], [False, True, True]], dtype=torch.bool
        )

        loss = compute_x0_flow_matching_loss(
            x0_model=lambda _state, _time: target,
            loss_function=lambda x_prediction, clean_target, time: x0_velocity_error(
                x_prediction - clean_target, time, 1e-5
            ).square(),
            target=target,
            mask=mask,
            time_mean=-0.4,
            time_std=1.0,
            noise_scale=1.0,
        )

        torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-10, rtol=0)

    def test_euler_step_accepts_x0_lambda(self) -> None:
        state = torch.zeros(2, 3)
        time = torch.zeros(2)
        next_time = torch.full((2,), 0.5)

        result = euler_step(
            lambda _state, _time: torch.ones_like(state),
            state,
            time,
            next_time,
            epsilon=1e-5,
        )

        torch.testing.assert_close(result, torch.full_like(state, 0.5))

    def test_heun_is_exact_for_constant_x0_path(self) -> None:
        state = torch.zeros(1, 2)
        result = heun_step(
            lambda value, _time: torch.ones_like(value),
            state,
            torch.zeros(1),
            torch.full((1,), 0.5),
            epsilon=1e-5,
        )

        torch.testing.assert_close(result, torch.full_like(state, 0.5))

    def test_sample_projects_each_state(self) -> None:
        result = sample(
            x0_model=lambda state, _time: torch.ones_like(state),
            initial_state=torch.zeros(1, 2),
            num_steps=2,
            epsilon=1e-5,
            project_state=lambda state: state.masked_fill(
                torch.tensor([[False, True]]), 0.0
            ),
        )

        torch.testing.assert_close(result, torch.tensor([[1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
