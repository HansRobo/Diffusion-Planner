"""Tests for the planner ONNX module boundaries."""

from __future__ import annotations

import unittest

import torch

from diffusion_planner.models.diffusion_planner import DiffusionPlanner
from diffusion_planner.models.onnx import (
    SCENE_INPUT_NAMES,
    DiffusionPlannerSamplerOnnxWrapper,
    SceneEncoderOnnxWrapper,
    TrajectoryDecoderOnnxWrapper,
)

from .test_diffusion_planner import make_input_data


class OnnxWrapperTest(unittest.TestCase):
    def test_wrappers_match_planner_forward(self) -> None:
        model = DiffusionPlanner(
            hidden_dim=16,
            num_heads=4,
            scene_fusion_depth=1,
            element_encoder_depth=1,
            decoder_depth=1,
            trajectory_encoder_depth=1,
            feedforward_dim=32,
            embed_dim=8,
        ).eval()
        input_data = make_input_data()
        scene_wrapper = SceneEncoderOnnxWrapper(model.scene_encoder).eval()
        decoder_wrapper = TrajectoryDecoderOnnxWrapper(model.trajectory_decoder).eval()

        with torch.inference_mode():
            scene, scene_mask, agent_pose, agent_mask = scene_wrapper(
                *(input_data[name] for name in SCENE_INPUT_NAMES)
            )
            x = torch.randn(1, agent_mask.shape[1], 80, 4)
            time = torch.full((1,), 0.5)
            actual = decoder_wrapper(x, agent_mask, scene, scene_mask, agent_pose, time)
            expected, _ = model(x, agent_mask, input_data, time)

        torch.testing.assert_close(actual, expected)

    def test_sampler_wrapper_matches_planner_sample(self) -> None:
        model = DiffusionPlanner(
            hidden_dim=16,
            num_heads=4,
            scene_fusion_depth=1,
            element_encoder_depth=1,
            decoder_depth=1,
            trajectory_encoder_depth=1,
            feedforward_dim=32,
            embed_dim=8,
        ).eval()
        input_data = make_input_data()
        wrapper = DiffusionPlannerSamplerOnnxWrapper(model).eval()
        initial_noise = torch.randn(
            1, 3, 80, 4, generator=torch.Generator().manual_seed(42)
        )

        with torch.inference_mode():
            actual_trajectory, actual_turn_indicator = wrapper(
                initial_noise,
                *(input_data[name] for name in SCENE_INPUT_NAMES),
                input_data["turn_indicators"],
            )
            expected_trajectory, expected_turn_indicator = model.sample(
                input_data, initial_noise, num_steps=10
            )

        torch.testing.assert_close(actual_trajectory, expected_trajectory)
        torch.testing.assert_close(actual_turn_indicator, expected_turn_indicator)


if __name__ == "__main__":
    unittest.main()
