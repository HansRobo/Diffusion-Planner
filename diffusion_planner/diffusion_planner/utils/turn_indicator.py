"""Turn-intent output stabilization independent of the trajectory policy."""

import math
from dataclasses import dataclass

import torch

from diffusion_planner.dimensions import TURN_INDICATOR_OUTPUT_DIM


@dataclass(frozen=True)
class TurnIndicatorStateMachineConfig:
    """Temporal hysteresis parameters for a 3-state OFF/LEFT/RIGHT classifier."""

    frequency_hz: float = 10.0
    activation_threshold: float = 0.60
    deactivation_threshold: float = 0.60
    direction_switch_threshold: float = 0.70
    activation_seconds: float = 0.30
    deactivation_seconds: float = 0.30
    direction_switch_seconds: float = 0.50
    minimum_active_seconds: float = 1.00
    probability_ema_alpha: float = 0.50

    def __post_init__(self):
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        for name in (
            "activation_threshold",
            "deactivation_threshold",
            "direction_switch_threshold",
            "probability_ema_alpha",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name in (
            "activation_seconds",
            "deactivation_seconds",
            "direction_switch_seconds",
            "minimum_active_seconds",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")


class TurnIndicatorStateMachine:
    """Debounce per-frame neural probabilities without feeding state to the planner.

    Internal states are 0=OFF, 1=LEFT, 2=RIGHT. ``raw_report_state`` maps the
    result back to Autoware's 1/2/3 report values. The object is intentionally
    scalar and stateful: instantiate one per vehicle/planner stream.
    """

    def __init__(self, config: TurnIndicatorStateMachineConfig | None = None):
        self.config = config or TurnIndicatorStateMachineConfig()
        self.reset()

    @staticmethod
    def _frames(seconds: float, frequency_hz: float) -> int:
        return max(1, math.ceil(seconds * frequency_hz - 1e-9))

    def reset(self, state: int = 0) -> None:
        if state not in range(TURN_INDICATOR_OUTPUT_DIM):
            raise ValueError(f"state must be in [0, {TURN_INDICATOR_OUTPUT_DIM - 1}]")
        self.state = int(state)
        self._state_frames = 0
        self._candidate = self.state
        self._candidate_frames = 0
        self._smoothed_probabilities: torch.Tensor | None = None

    @property
    def raw_report_state(self) -> int:
        return self.state + 1

    def _required_frames(self, candidate: int) -> int:
        cfg = self.config
        if self.state == 0:
            seconds = cfg.activation_seconds
        elif candidate == 0:
            seconds = cfg.deactivation_seconds
        else:
            seconds = cfg.direction_switch_seconds
        return self._frames(seconds, cfg.frequency_hz)

    def _threshold(self, candidate: int) -> float:
        if self.state == 0:
            return self.config.activation_threshold
        if candidate == 0:
            return self.config.deactivation_threshold
        return self.config.direction_switch_threshold

    def update(self, values: torch.Tensor, *, logits: bool = True) -> int:
        """Consume one frame of logits/probabilities and return the stable state."""
        if values.numel() != TURN_INDICATOR_OUTPUT_DIM:
            raise ValueError(
                f"Expected {TURN_INDICATOR_OUTPUT_DIM} values, got shape {tuple(values.shape)}"
            )
        values = values.detach().float().reshape(TURN_INDICATOR_OUTPUT_DIM).cpu()
        probabilities = torch.softmax(values, dim=0) if logits else values
        if not torch.isfinite(probabilities).all():
            raise ValueError("Turn-indicator probabilities must be finite")
        if not logits:
            if (probabilities < 0).any() or abs(float(probabilities.sum()) - 1.0) > 1e-4:
                raise ValueError("Probabilities must be non-negative and sum to one")

        alpha = self.config.probability_ema_alpha
        if self._smoothed_probabilities is None:
            self._smoothed_probabilities = probabilities
        else:
            self._smoothed_probabilities = (
                alpha * probabilities + (1.0 - alpha) * self._smoothed_probabilities
            )

        self._state_frames += 1
        candidate = int(self._smoothed_probabilities.argmax())
        confidence = float(self._smoothed_probabilities[candidate])
        if candidate == self.state or confidence < self._threshold(candidate):
            self._candidate = self.state
            self._candidate_frames = 0
            return self.state

        minimum_active_frames = self._frames(
            self.config.minimum_active_seconds, self.config.frequency_hz
        )
        if self.state != 0 and self._state_frames < minimum_active_frames:
            self._candidate = self.state
            self._candidate_frames = 0
            return self.state

        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_frames = 1
        else:
            self._candidate_frames += 1
        if self._candidate_frames >= self._required_frames(candidate):
            self.state = candidate
            self._state_frames = 0
            self._candidate = candidate
            self._candidate_frames = 0
        return self.state
