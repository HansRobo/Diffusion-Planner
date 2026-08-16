"""Train an x0-prediction flow model on a spiral toy dataset.

Example:
    uv run --package diffusion-planner python \
        packages/diffusion_planner/examples/example_flow_matching.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from torch import nn
from tqdm import trange

from diffusion_planner.models.flow_matching import (
    compute_x0_flow_matching_loss,
    sample,
    x0_velocity_error,
)
from diffusion_planner.utils.optimizer import build_optimizer


def make_spiral_dataset(
    batch_size: int,
    noise: float,
    ambient_dim: int,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Create a noisy 2D spiral embedded in `ambient_dim` dimensions."""
    intrinsic_dim = 2

    if ambient_dim < intrinsic_dim:
        raise ValueError("ambient_dim must be at least 2.")

    angle = 1.5 * np.pi * (1.0 + 2.0 * np.random.uniform(size=batch_size))
    data_2d = np.stack((angle * np.cos(angle), angle * np.sin(angle)), axis=1)
    if noise > 0:
        data_2d += noise * np.random.randn(batch_size, intrinsic_dim)

    data_2d /= 10.0
    data_2d *= np.sqrt(ambient_dim / 2.0)

    basis, _ = np.linalg.qr(np.random.randn(ambient_dim, intrinsic_dim))
    data = data_2d @ basis.T

    def project_back(values: np.ndarray) -> np.ndarray:
        return values @ basis

    return data.astype(np.float32), project_back


class SpiralX0Model(nn.Module):
    """Predict clean ambient-space samples from noisy states and flow time."""

    def __init__(self, ambient_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(ambient_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, ambient_dim),
        )

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Predict clean samples from state `(B, D)` and time `(B,)`."""
        return self.network(torch.cat((state, time[:, None]), dim=-1))


def train(
    model: SpiralX0Model,
    dataset: torch.Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    """Train the toy x0 model with the shared flow-matching loss."""
    optimizer = build_optimizer(
        model=model,
        output_layers=(model.network[-1],),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        verbose=True,
    )
    progress = trange(steps, desc="training")
    for step in progress:
        indices = torch.randint(dataset.shape[0], (batch_size,), device=dataset.device)
        target = dataset[indices]
        loss = compute_x0_flow_matching_loss(
            x0_model=model,
            loss_function=lambda x_prediction, clean_target, time: x0_velocity_error(
                x_prediction - clean_target, time, 1e-5
            ).square(),
            target=target,
            mask=torch.zeros(batch_size, dtype=torch.bool, device=dataset.device),
            time_mean=-0.4,
            time_std=1.0,
            noise_scale=1.0,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            progress.set_postfix(loss=f"{loss.item():.4f}")


def generate(
    model: SpiralX0Model,
    sample_count: int,
    ambient_dim: int,
    sampling_steps: int,
    device: torch.device,
) -> np.ndarray:
    """Generate ambient-space spiral samples with shared Heun integration."""
    initial_state = torch.randn(sample_count, ambient_dim, device=device)
    model.eval()
    with torch.no_grad():
        generated = sample(
            x0_model=model,
            initial_state=initial_state,
            num_steps=sampling_steps,
            epsilon=1e-5,
            project_state=lambda state: state,
        )
    return generated.cpu().numpy()


def save_plot(
    reference: np.ndarray,
    generated: np.ndarray,
    output: Path,
) -> None:
    """Save reference and generated 2D samples as an interactive HTML plot."""
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=reference[:, 0],
            y=reference[:, 1],
            mode="markers",
            name="target",
            marker={"size": 4, "opacity": 0.45},
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=generated[:, 0],
            y=generated[:, 1],
            mode="markers",
            name="generated",
            marker={"size": 4, "opacity": 0.45},
        )
    )
    figure.update_layout(
        title="x0-prediction flow matching on a spiral",
        xaxis={"scaleanchor": "y", "scaleratio": 1},
        template="plotly_white",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-size", type=int, default=10_000)
    parser.add_argument("--sample-count", type=int, default=5_000)
    parser.add_argument("--ambient-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--training-steps", type=int, default=5_000)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--data-noise", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spiral_flow_matching.html"),
    )
    return parser.parse_args()


def main() -> None:
    """Train the toy model, sample it, and save a projected visualization."""
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, project_back = make_spiral_dataset(
        args.dataset_size,
        args.data_noise,
        args.ambient_dim,
    )
    dataset = torch.from_numpy(data).to(device)
    model = SpiralX0Model(args.ambient_dim, args.hidden_dim).to(device)
    train(
        model,
        dataset,
        args.training_steps,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
    )
    generated = generate(
        model,
        args.sample_count,
        args.ambient_dim,
        args.sampling_steps,
        device,
    )
    save_plot(project_back(data), project_back(generated), args.output)
    print(f"saved visualization to {args.output.resolve()}")


if __name__ == "__main__":
    main()
