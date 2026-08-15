"""Tests for generic x0-prediction flow-matching utilities."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.models.flow_matching import euler_step, heun_step, sample


class FlowMatchingTest(unittest.TestCase):
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
