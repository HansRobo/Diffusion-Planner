"""Shared model/input helpers for the token-analysis command-line tools."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusion_planner.dimensions import MAX_NUM_AGENTS, OUTPUT_T, POSE_DIM
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config


def init_distributed(requested_device: str):
    """Initialize torchrun data parallelism and return device/rank metadata."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = requested_device
    if world_size > 1:
        if requested_device.startswith("cuda"):
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)
    return device, rank, local_rank, world_size


def prepare_inputs(inputs: dict, cfg, device: str, *, include_future: bool = False):
    """Apply the validation preprocessing used by the planner."""
    inputs = {k: v.to(device) for k, v in inputs.items()}
    batch_size = inputs["ego_current_state"].shape[0]
    inputs["sampled_trajectories"] = torch.zeros(
        batch_size, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, dtype=torch.float32, device=device
    )
    inputs["delay"] = torch.zeros(batch_size, dtype=torch.float32, device=device)
    inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
    inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])
    ego_future = heading_to_cos_sin(inputs["ego_agent_future"]) if include_future else None
    inputs = cfg.observation_normalizer(inputs)
    return (inputs, ego_future) if include_future else inputs


def latest_ckpt(run_dir: Path) -> Path:
    if (run_dir / "best_model.pth").exists():
        return run_dir / "best_model.pth"
    epoch_dirs = sorted(
        (d for d in run_dir.iterdir() if re.fullmatch(r"epoch\d+", d.name)),
        key=lambda d: int(d.name[5:]),
    )
    if epoch_dirs:
        return epoch_dirs[-1] / "best_model.pth"
    return run_dir / "best_model" / "best_model.pth"


def load_model(run_dir: Path, device: str):
    cfg = Config(str(run_dir / "args.json"))
    cfg.device = device
    cfg.ddp = False
    model = Diffusion_Planner(cfg).to(device)
    ckpt_path = latest_ckpt(run_dir)
    state = torch.load(ckpt_path, map_location=device)
    state = state["model"] if "model" in state else state
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg, ckpt_path


def find_fusion(encoder):
    for module in encoder.modules():
        if type(module).__name__ == "FusionEncoder":
            return module
    raise RuntimeError("FusionEncoder not found")


def neighbor_dist(neighbors: torch.Tensor) -> torch.Tensor:
    valid = (neighbors[:, :, -6:, :8] != 0).any(dim=(2, 3))
    distance = neighbors[:, :, -1, :2].norm(dim=-1)
    return torch.where(valid, distance, torch.full_like(distance, float("inf")))


def polyline_dist(values: torch.Tensor, geom_dims: int | None = None) -> torch.Tensor:
    valid = (
        (values != 0).any(dim=-1)
        if geom_dims is None
        else (values[..., :geom_dims] != 0).any(dim=-1)
    )
    distance = values[..., :2].norm(dim=-1)
    distance = torch.where(valid, distance, torch.full_like(distance, float("inf")))
    return distance.min(dim=-1).values


def patch_fusion(fusion, store):
    """Capture the model's pre-norm attention weights, inputs, and mask."""
    for layer_index, block in enumerate(fusion.blocks):

        def make_forward(layer, index):
            def forward(x, mask):
                query = layer.norm1(x)
                attention_output, weights = layer.attn(
                    query,
                    x,
                    x,
                    key_padding_mask=mask,
                    need_weights=True,
                    average_attn_weights=True,
                )
                store.append(
                    {
                        "layer": index,
                        "weights": weights.detach(),
                        "w": weights.detach(),
                        "kv": x.detach(),
                        "mask": mask.detach(),
                    }
                )
                x = x + layer.drop_path(attention_output)
                return x + layer.drop_path(layer.mlp(layer.norm2(x)))

            return forward

        block.forward = make_forward(block, layer_index)


def draw_closed_loop_scene(
    ax,
    sample: dict,
    ego_pred,
    *,
    view_half_m: float,
    distance_label_offset_m: float = 1.2,
) -> str | None:
    """Draw the reproducer/replay render style onto ``ax`` for one closed-loop rollout step.

    ``sample`` is one already-unbatched, live-ego-frame model-input dict, as produced by
    the closed-loop reproducer rollout (``{k: np.asarray(v)[0] for k, v in np_dict.items()}``,
    the same shape ``scenario_generation.reproducer_rollout._draw_step`` builds its
    ``SceneContext`` from). ``ego_pred`` is the model's ego-frame predicted trajectory
    for this step, e.g. ``outputs["prediction"][0, 0].cpu().numpy()``.

    Reuses the same base layer (lanes, road borders, traffic-light overlay, oriented
    agent boxes, ego plan, distance badges) as ``render_reproducer_segment.py``'s video
    output, so the two tools' scene rendering stays visually consistent. Imports are
    local: this keeps the heavier ``scenario_generation.replay`` dependency chain out of
    this module's import graph for callers that never use closed-loop rendering.

    Returns the ego-state title fragment (or ``None`` if the sample has no ego agent).
    """
    from scenario_generation import npz_loader as nl
    from scenario_generation.replay import draw_step_scene
    from scenario_generation.reproducer_rollout import _polylines_from_tensor
    from scenario_generation.scene_context import SceneContext

    es = np.asarray(sample["ego_shape"]).reshape(-1)
    ego = nl._extract_ego_agent(sample, float(es[0]), float(es[1]), float(es[2]))
    neighbors = nl._extract_neighbors(sample)
    scene = SceneContext(
        agents=[ego] + neighbors, map_data=nl._extract_map_data(sample), ego_agent_id="ego"
    )
    ax.set_facecolor("#f8f8f8")
    return draw_step_scene(
        ax,
        scene,
        {"ego": ego_pred},
        route_polylines=_polylines_from_tensor(sample["route_lanes"]),
        road_border_polylines=_polylines_from_tensor(sample["line_strings"], border_only=True),
        view_half_m=view_half_m,
        distance_label_offset_m=distance_label_offset_m,
    )
