"""``head_emits_turn_indicator``: which models get closed-loop turn indicators."""

from types import SimpleNamespace

import numpy as np
import torch
from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_KEEP

from scenario_generation.reproducer_rollout import (
    head_emits_turn_indicator,
    resolve_turn_indicator,
)
from scenario_generation.simulate import decode_turn_indicator


def test_diffusion_head_always_emits():
    assert head_emits_turn_indicator(SimpleNamespace(predictor_head="diffusion"))


def test_legacy_args_without_predictor_head_are_diffusion():
    assert head_emits_turn_indicator(SimpleNamespace())


def test_drivor_without_the_flag_falls_back_to_recorded():
    assert not head_emits_turn_indicator(SimpleNamespace(predictor_head="drivor"))


def test_drivor_with_the_flag_emits():
    assert head_emits_turn_indicator(
        SimpleNamespace(predictor_head="drivor", drivor_turn_indicator=True)
    )


def test_keep_holds_the_previous_state():
    assert resolve_turn_indicator(TURN_INDICATOR_OUTPUT_KEEP, 2) == 2
    assert resolve_turn_indicator(np.array([TURN_INDICATOR_OUTPUT_KEEP]), 1) == 1


def test_raw_states_pass_through():
    for state in (0, 1, 2, 3):
        assert resolve_turn_indicator(state, 2) == state


def test_decoded_keep_never_reaches_the_history():
    logit = torch.zeros(1, 5)
    logit[0, TURN_INDICATOR_OUTPUT_KEEP] = 10.0
    decoded = decode_turn_indicator(logit, 0.25)
    assert int(decoded[0]) == TURN_INDICATOR_OUTPUT_KEEP
    assert resolve_turn_indicator(decoded, 3) == 3
