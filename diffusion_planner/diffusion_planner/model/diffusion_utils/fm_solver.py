"""Euler sampler for x0-parameterized flow matching on the linear path.

The probability path is ``x_t = (1 - t) * x0 + t * eps`` (t=0 data, t=1 noise), whose
ODE is ``dx/dt = (x_t - x0) / t``. With an x0-prediction model, one Euler step from
``t`` to ``t'`` is exactly the interpolation

    x_{t'} = (t' / t) * x_t + (1 - t' / t) * x0_hat(x_t, t)

whose coefficients stay in [0, 1], so nothing blows up as t -> 0. The final step
targets t' = 0, ending on the model's clean-sample prediction (the flow-matching
analogue of DPM's denoise-to-zero) — a k-step sample costs exactly k model
evaluations.
"""

import torch


class FMEulerSolver:
    def __init__(self, model_fn):
        """``model_fn(x[B, P, T, 4], t[B]) -> x0_hat[B, P, T, 4]``."""
        self.model_fn = model_fn

    def sample(self, x, steps):
        if x.ndim != 4:
            raise ValueError(f"FM solver expects [B,P,T,D], got {tuple(x.shape)}")
        if steps < 1:
            raise ValueError(f"FM solver needs steps >= 1, got {steps}")
        batch_size = x.shape[0]
        with torch.no_grad():
            # Model evaluation times are 1, (k-1)/k, ..., 1/k — all strictly above the
            # training-time floor for any practical step count.
            timesteps = torch.linspace(1.0, 0.0, steps + 1, device=x.device)
            for step in range(steps):
                t, t_next = timesteps[step], timesteps[step + 1]
                x0_pred = self.model_fn(x, t.expand(batch_size)).reshape(x.shape)
                ratio = t_next / t
                x = ratio * x + (1.0 - ratio) * x0_pred
        return x
