from dataclasses import dataclass, field

from .config_cli import cli


@dataclass
class ClosedLoopConfig:
    closed_loop_npz_root: list[str] = cli(
        "JSON file(s) or folder(s) for closed-loop validation. "
        "Empty = disabled. Supports: folder, flat JSON (list), grouped JSON (dict). "
        "Multiple inputs each become their own top-level namespace.",
        default_factory=list,
        path=True,
    )
    closed_loop_object_modes: list[str] = cli(
        "object-mode(s): 'objects'=normal, 'noobj'=empty-world ablation",
        default_factory=lambda: ["objects"],
    )
    closed_loop_seg_len: int = 100000
    closed_loop_replan_interval: int = 1
    closed_loop_draw_every: int = 4
    closed_loop_draw_workers: int = cli("render on this many worker processes", default=4)
    closed_loop_fps: int = 10
    closed_loop_near_miss_thresh: float = 0.5
    closed_loop_search_radius: float = 1.5
    closed_loop_warmup_steps: int = 0
    closed_loop_unstick_after: int = 300
    closed_loop_unstick_advance_m: float = 5.0
    closed_loop_unstick_radius_mult: float = 10.0
    closed_loop_unstick_teleport_after: int = 300
    closed_loop_abort_deviation_m: float = 50.0
    closed_loop_abort_after: int = 30
    closed_loop_abort_max_snaps: int = 0
    closed_loop_wandb_video_pick: str = cli(
        "which episode gets video+colormap: 'worst'/'first'/'longest'",
        default="worst",
    )
    closed_loop_colormap_metrics: list[str] = field(
        default_factory=lambda: [
            "clearance",
            "collision",
            "near_miss",
            "speed",
            "road_border",
            "red_light",
            "strong_brake",
        ]
    )
    scenario_sim_driver: str = cli(
        "shell driver that evaluates a saved checkpoint against the OpenSCENARIO suite. "
        "Empty = disabled. It receives the checkpoint and an output directory in CKPT / OUT; "
        "every other knob is its own environment's.",
        default="",
        path=True,
    )
    render_media: bool = cli(
        "render video/colormap artifacts during wandb logging",
        default=True,
    )
    device: str = cli("device for model and evaluation", default="cuda")
    wandb_project_name: str = cli("Weights & Biases project name (empty=disabled)", default="")
    exp_name: str = cli("name of this run; appears in the wandb run name", default="")
    # Mirror BaseConfig.ddp so ddp_setup_universal(...) can be called on this config
    # directly; ``True`` here means "respect RANK/WORLD_SIZE if set" (single-process
    # CLI runs with no torchrun env vars stay non-distributed).
    ddp: bool = cli("enable DDP when RANK/WORLD_SIZE are present", default=True)
