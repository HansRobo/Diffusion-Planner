"""Tests for the flow-matching trajectory decoder."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.data.dimensions import TRAJECTORY_LENGTH
from diffusion_planner.models.decoder import TrajectoryDecoder, TrajectoryEncoder


class TrajectoryEncoderTest(unittest.TestCase):
    def test_encodes_one_token_per_agent(self) -> None:
        encoder = TrajectoryEncoder(
            hidden_dim=16,
            depth=2,
        )
        trajectories = torch.randn(2, 3, TRAJECTORY_LENGTH, 4, requires_grad=True)

        tokens = encoder(trajectories)

        self.assertEqual(tokens.shape, (2, 3, 16))
        tokens.sum().backward()
        self.assertIsNotNone(trajectories.grad)


class TrajectoryDecoderTest(unittest.TestCase):
    def test_preserves_trajectory_shape_and_backpropagates(self) -> None:
        decoder = TrajectoryDecoder(
            hidden_dim=16,
            num_heads=4,
            depth=2,
            feedforward_dim=32,
        )
        x = torch.randn(2, 3, TRAJECTORY_LENGTH, 4, requires_grad=True)
        scene = torch.randn(2, 5, 16)
        scene[:, -2:] = 0.0
        scene.requires_grad_()
        time = torch.tensor([0.25, 0.75])
        x_mask = torch.tensor([[False, False, True], [False, False, False]])
        scene_mask = torch.tensor(
            [[False, False, False, True, True], [False, False, False, True, True]]
        )

        output = decoder(x, x_mask, scene, scene_mask, time)

        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(output[x_mask], torch.zeros_like(output[x_mask]))
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(scene.grad)

    def test_accepts_column_flow_time(self) -> None:
        decoder = TrajectoryDecoder(
            hidden_dim=8,
            num_heads=2,
            depth=1,
            feedforward_dim=16,
        )
        output = decoder(
            torch.zeros(1, 2, TRAJECTORY_LENGTH, 4),
            torch.tensor([[False, False]]),
            torch.ones(1, 3, 8),
            torch.tensor([[False, False, False]]),
            torch.tensor([[0.5]]),
        )

        self.assertEqual(output.shape, (1, 2, TRAJECTORY_LENGTH, 4))

    def test_attention_operates_on_one_token_per_agent(self) -> None:
        decoder = TrajectoryDecoder(
            hidden_dim=8,
            num_heads=2,
            depth=1,
            feedforward_dim=16,
        )
        attention_input_shape = None

        def record_attention_input(
            _module: torch.nn.Module, args: tuple[torch.Tensor, ...]
        ) -> None:
            nonlocal attention_input_shape
            attention_input_shape = args[0].shape

        handle = decoder.blocks[0].agent_attention.register_forward_pre_hook(
            record_attention_input
        )
        decoder(
            torch.zeros(2, 3, TRAJECTORY_LENGTH, 4),
            torch.zeros(2, 3, dtype=torch.bool),
            torch.ones(2, 5, 8),
            torch.zeros(2, 5, dtype=torch.bool),
            torch.tensor([0.25, 0.75]),
        )
        handle.remove()

        self.assertEqual(attention_input_shape, (2, 3, 8))


if __name__ == "__main__":
    unittest.main()
