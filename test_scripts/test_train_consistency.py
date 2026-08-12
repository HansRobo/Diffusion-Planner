#!/usr/bin/env python3
"""Check that the train-mode forward is identical between the current branch and tier4-main.

The v5.0 checkpoint in ``test_scripts/assets`` is loaded, one forward pass is run in
train mode on CPU with a fixed seed, and every output tensor is compared between the
current branch and ``tier4-main`` with a float32-level tolerance.
The original branch is checked out again before the script exits, even on failure.

This file is copied into ``test_scripts/assets/_train_worker.py`` and re-executed there
as the worker: the copy is git-ignored, so it survives the checkout that swaps the
branch under test.

Example:
    uv run python test_scripts/test_train_consistency.py
"""

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.dimensions import (
    INPUT_T,
    LINE_STRING_TYPE_NUM,
    LINE_TYPE_LEFT_START,
    LINE_TYPE_RIGHT_START,
    MAX_NUM_NEIGHBORS,
    NUM_LINE_STRINGS,
    NUM_POLYGONS,
    NUM_SEGMENTS_IN_LANE,
    NUM_SEGMENTS_IN_ROUTE,
    NUM_STATIC_OBJECTS,
    OUTPUT_T,
    POINTS_PER_LANELET,
    POINTS_PER_LINE_STRING,
    POINTS_PER_POLYGON,
    POSE_DIM,
    SEGMENT_POINT_DIM,
    TRAFFIC_LIGHT_GREEN,
)
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.train_utils import set_seed

TEST_SCRIPTS_DIR = Path(__file__).resolve().parent
ASSETS_DIR = TEST_SCRIPTS_DIR if TEST_SCRIPTS_DIR.name == "assets" else TEST_SCRIPTS_DIR / "assets"
REPO_ROOT = ASSETS_DIR.parent.parent

BASE_REF = "tier4-main"

MODEL_BASE_URL = "https://awf.ml.dev.web.auto/planning/models/diffusion_planner/v5.0"
CKPT_NAME = "diffusion_planner.pth"
PARAM_NAME = "diffusion_planner.param.json"

WORKER_PATH = ASSETS_DIR / "_train_worker.py"
RESULT_PATH = ASSETS_DIR / "forward_result.npz"
HEAD_RESULT_PATH = ASSETS_DIR / "forward_head.npz"
BASE_RESULT_PATH = ASSETS_DIR / "forward_base.npz"

SEED = 3407
BATCH_SIZE = 2

# Differences are measured against the largest value of the tensor. The arithmetic is
# float32 reduced in float64, so a reordering of the same operations lands around 1e-7.
# Anything above this is a real change of behavior.
RELATIVE_TOLERANCE = 1e-5

# Dummy scene layout, taken from the number of non-zero rows in a real frame.
NUM_VALID_NEIGHBORS = 12
NUM_VALID_STATIC_OBJECTS = 2
NUM_VALID_LANES = 52
NUM_VALID_ROUTE_LANES = 7
NUM_VALID_POLYGONS = 2
NUM_VALID_LINE_STRINGS = 25
SPEED_LIMIT = 2.778
DT = 0.1
EGO_SPEED = 5.0
EGO_YAW_RATE = 0.05
EGO_SHAPE = [2.65, 3.84, 1.60]


# --------------------------------------------------------------------------------------
# dummy data
# --------------------------------------------------------------------------------------
def build_ego_trajectory(num_steps: int, start_step: int) -> np.ndarray:
    """Constant-speed, constant-yaw-rate poses (x, y, heading), ego-centric at step 0."""
    steps = np.arange(start_step, start_step + num_steps, dtype=np.float64)
    heading = EGO_YAW_RATE * steps * DT
    # Exact arc integration so that the pose at step 0 is the origin.
    radius = EGO_SPEED / EGO_YAW_RATE
    x = radius * np.sin(heading)
    y = radius * (1.0 - np.cos(heading))
    return np.stack([x, y, heading], axis=-1).astype(np.float32)


def build_segments(rng: np.random.Generator, num_segments: int, num_valid: int) -> np.ndarray:
    """Lane / route segments: straight center lines with parallel left and right bounds."""
    segments = np.zeros((num_segments, POINTS_PER_LANELET, SEGMENT_POINT_DIM), dtype=np.float32)
    for i in range(num_valid):
        start = rng.uniform([-40.0, -20.0], [40.0, 20.0])
        heading = rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(heading), np.sin(heading)])
        normal = np.array([-direction[1], direction[0]])
        offsets = np.arange(POINTS_PER_LANELET, dtype=np.float64)[:, None]
        center = start + direction * offsets
        segments[i, :, 0:2] = center
        segments[i, :, 2:4] = direction
        segments[i, :, 4:6] = center + normal * 2.0
        segments[i, :, 6:8] = center - normal * 2.0
        segments[i, :, TRAFFIC_LIGHT_GREEN] = 1.0
        segments[i, :, LINE_TYPE_LEFT_START] = 1.0
        segments[i, :, LINE_TYPE_RIGHT_START] = 1.0
    return segments


def build_speed_limit(num_segments: int, num_valid: int) -> tuple[np.ndarray, np.ndarray]:
    speed_limit = np.zeros((num_segments, 1), dtype=np.float32)
    has_speed_limit = np.zeros((num_segments, 1), dtype=bool)
    speed_limit[:num_valid] = SPEED_LIMIT
    has_speed_limit[:num_valid] = True
    return speed_limit, has_speed_limit


def build_dummy_scene(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One frame of dummy input, with the same keys/shapes/dtypes as a dataset npz."""
    scene = {}

    scene["ego_agent_past"] = build_ego_trajectory(INPUT_T + 1, -INPUT_T)
    scene["ego_agent_future"] = build_ego_trajectory(OUTPUT_T, 1)
    scene["ego_current_state"] = np.array(
        [0.0, 0.0, 1.0, 0.0, EGO_SPEED, 0.0, 0.0, 0.0, 0.0, EGO_YAW_RATE], dtype=np.float32
    )
    scene["ego_shape"] = np.array(EGO_SHAPE, dtype=np.float32)

    # Neighbors: [x, y, cos, sin, vx, vy, width, length, one-hot type (3)]
    neighbor_past = np.zeros((MAX_NUM_NEIGHBORS, INPUT_T + 1, 11), dtype=np.float32)
    neighbor_future = np.zeros((MAX_NUM_NEIGHBORS, OUTPUT_T, POSE_DIM), dtype=np.float32)
    for i in range(NUM_VALID_NEIGHBORS):
        center = rng.uniform([-30.0, -12.0], [30.0, 12.0])
        heading = rng.uniform(-np.pi, np.pi)
        speed = rng.uniform(0.0, 8.0)
        direction = np.array([np.cos(heading), np.sin(heading)])
        past_steps = np.arange(-INPUT_T, 1, dtype=np.float64)[:, None]
        neighbor_past[i, :, 0:2] = center + direction * speed * past_steps * DT
        neighbor_past[i, :, 2] = np.cos(heading)
        neighbor_past[i, :, 3] = np.sin(heading)
        neighbor_past[i, :, 4] = speed * direction[0]
        neighbor_past[i, :, 5] = speed * direction[1]
        neighbor_past[i, :, 6] = rng.uniform(1.5, 2.5)
        neighbor_past[i, :, 7] = rng.uniform(3.5, 5.5)
        neighbor_past[i, :, 8 + i % 3] = 1.0
        future_steps = np.arange(1, OUTPUT_T + 1, dtype=np.float64)[:, None]
        neighbor_future[i, :, 0:2] = center + direction * speed * future_steps * DT
        neighbor_future[i, :, 2] = np.cos(heading)
        neighbor_future[i, :, 3] = np.sin(heading)
    scene["neighbor_agents_past"] = neighbor_past
    scene["neighbor_agents_future"] = neighbor_future

    static_objects = np.zeros((NUM_STATIC_OBJECTS, 10), dtype=np.float32)
    for i in range(NUM_VALID_STATIC_OBJECTS):
        static_objects[i, 0:2] = rng.uniform([-20.0, -10.0], [20.0, 10.0])
        static_objects[i, 2] = 1.0
        static_objects[i, 4] = rng.uniform(0.5, 1.5)
        static_objects[i, 5] = rng.uniform(0.5, 1.5)
        static_objects[i, 6 + i % 4] = 1.0
    scene["static_objects"] = static_objects

    scene["lanes"] = build_segments(rng, NUM_SEGMENTS_IN_LANE, NUM_VALID_LANES)
    lanes_speed_limit, lanes_has_speed_limit = build_speed_limit(
        NUM_SEGMENTS_IN_LANE, NUM_VALID_LANES
    )
    scene["lanes_speed_limit"] = lanes_speed_limit
    scene["lanes_has_speed_limit"] = lanes_has_speed_limit

    scene["route_lanes"] = build_segments(rng, NUM_SEGMENTS_IN_ROUTE, NUM_VALID_ROUTE_LANES)
    route_speed_limit, route_has_speed_limit = build_speed_limit(
        NUM_SEGMENTS_IN_ROUTE, NUM_VALID_ROUTE_LANES
    )
    scene["route_lanes_speed_limit"] = route_speed_limit
    scene["route_lanes_has_speed_limit"] = route_has_speed_limit

    polygons = np.zeros((NUM_POLYGONS, POINTS_PER_POLYGON, 3), dtype=np.float32)
    for i in range(NUM_VALID_POLYGONS):
        center = rng.uniform([-40.0, -40.0], [40.0, 40.0])
        angles = np.linspace(0.0, 2.0 * np.pi, POINTS_PER_POLYGON, endpoint=False)
        polygons[i, :, 0] = center[0] + 3.0 * np.cos(angles)
        polygons[i, :, 1] = center[1] + 3.0 * np.sin(angles)
        polygons[i, :, 2] = 1.0
    scene["polygons"] = polygons

    line_strings = np.zeros((NUM_LINE_STRINGS, POINTS_PER_LINE_STRING, 4), dtype=np.float32)
    for i in range(NUM_VALID_LINE_STRINGS):
        start = rng.uniform([-40.0, -30.0], [40.0, 30.0])
        heading = rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(heading), np.sin(heading)])
        offsets = np.arange(POINTS_PER_LINE_STRING, dtype=np.float64)[:, None]
        line_strings[i, :, 0:2] = start + direction * offsets
        line_strings[i, :, 2 + i % LINE_STRING_TYPE_NUM] = 1.0
    scene["line_strings"] = line_strings

    scene["goal_pose"] = build_ego_trajectory(1, OUTPUT_T)[0]
    scene["turn_indicators"] = np.ones(INPUT_T + 1, dtype=np.int32)
    return scene


def build_dummy_batch(device: str) -> dict[str, torch.Tensor]:
    """One collated batch, identical for every run."""
    rng = np.random.default_rng(SEED)
    scenes = [build_dummy_scene(rng) for _ in range(BATCH_SIZE)]
    batch = {}
    for key in scenes[0]:
        stacked = np.stack([scene[key] for scene in scenes], axis=0)
        batch[key] = torch.from_numpy(stacked).to(device)
    return batch


def add_diffusion_inputs(inputs: dict[str, torch.Tensor], config) -> None:
    """Add the three tensors that ``Decoder._forward_training`` reads.

    ``compute_training_loss`` normally builds these from the ground truth; they are
    rebuilt here so that the test does not depend on the loss, which is where the two
    branches differ in how many random numbers they draw.
    """
    num_agents = 1 + config.predicted_neighbor_num
    num_steps = 1 + config.future_len

    current_states = torch.cat(
        [
            inputs["ego_current_state"][:, None, :POSE_DIM],
            inputs["neighbor_agents_past"][:, : config.predicted_neighbor_num, -1, :POSE_DIM],
        ],
        dim=1,
    )
    future_states = torch.cat(
        [
            heading_to_cos_sin(inputs["ego_agent_future"])[:, None],
            inputs["neighbor_agents_future"][:, : config.predicted_neighbor_num],
        ],
        dim=1,
    )
    inputs["gt_trajectories"] = torch.cat([current_states[:, :, None, :], future_states], dim=2)
    inputs["sampled_trajectories"] = 0.5 * torch.randn(
        BATCH_SIZE, num_agents, num_steps, POSE_DIM, dtype=torch.float32
    )
    diffusion_time = torch.rand(BATCH_SIZE, 1, 1, 1, dtype=torch.float32)
    inputs["diffusion_time"] = diffusion_time.expand(
        BATCH_SIZE, num_agents, num_steps, 1
    ).contiguous()


# --------------------------------------------------------------------------------------
# worker: run the forward pass with the code of the currently checked out ref
# --------------------------------------------------------------------------------------
def build_config():
    config = Config(str(ASSETS_DIR / PARAM_NAME))
    # Overrides so that the run is single-process, deterministic and CPU-only.
    config.device = "cpu"
    config.ddp = False
    config.use_amp = False
    # Drop path is disabled so that the two runs draw the same random numbers. It is
    # stochastic depth, so any refactor that adds or removes a drop_path call shifts the
    # whole RNG stream and moves every output, which would drown out the numerical
    # difference this test is looking for.
    config.encoder_drop_path_rate = 0.0
    config.decoder_drop_path_rate = 0.0
    return config


def build_model(config) -> Diffusion_Planner:
    model = Diffusion_Planner(config)
    model = model.to(config.device)
    checkpoint = torch.load(str(ASSETS_DIR / CKPT_NAME), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state_dict)
    return model


def run_worker() -> int:
    torch.set_num_threads(4)
    set_seed(SEED)

    config = build_config()
    model = build_model(config)
    model.train()

    inputs = build_dummy_batch(config.device)
    # The same preprocessing train_epoch applies before it calls the model.
    inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
    inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])
    inputs = config.observation_normalizer(inputs)
    add_diffusion_inputs(inputs, config)

    with torch.no_grad():
        encoder_outputs, decoder_outputs = model(inputs)

    outputs = {}
    # The encoder returns the encoding tensor itself; the decoder returns a dict.
    for name, tensor in {"encoding": encoder_outputs, **decoder_outputs}.items():
        array = tensor.detach().cpu().numpy()
        outputs[name] = array
        print(f"  {name}: shape={array.shape} mean={array.mean():.9g} std={array.std():.9g}")

    np.savez(RESULT_PATH, **outputs)
    print(f"Saved {RESULT_PATH}")
    return 0


# --------------------------------------------------------------------------------------
# parent: run the worker on both refs and compare
# --------------------------------------------------------------------------------------
def run_git(git_args: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *git_args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def resolve_head_ref() -> str:
    """Return the current branch name, or the commit hash when HEAD is detached."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "symbolic-ref", "--quiet", "--short", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return run_git(["rev-parse", "HEAD"])


def assert_no_tracked_changes() -> None:
    """Abort unless every tracked file matches HEAD.

    Uncommitted edits would survive the checkout and silently leak into the tier4-main
    run, which would make the comparison meaningless.
    """
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--quiet", "HEAD"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        changed = run_git(["diff", "--name-only", "HEAD"])
        raise RuntimeError(
            "The working tree has uncommitted changes to tracked files:\n"
            f"{changed}\n"
            "Commit or stash them before running this test."
        )


def ensure_assets_dir() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    gitignore_path = ASSETS_DIR / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("*\n")


def download_if_absent(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already downloaded: {dest} ({dest.stat().st_size} bytes)")
        return
    print(f"Downloading {url}\n         -> {dest}")
    tmp_path = dest.with_name(dest.name + ".partial")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp_path.rename(dest)
    print(f"Downloaded: {dest} ({dest.stat().st_size} bytes)")


def run_forward(result_path: Path) -> None:
    """Run the worker copy with the code of the currently checked out ref."""
    # Keep this script's own logs interleaved correctly with the subprocess output.
    sys.stdout.flush()
    RESULT_PATH.unlink(missing_ok=True)
    env = dict(os.environ)
    # `uv run` re-resolves the environment for the checked out ref, so an inherited
    # VIRTUAL_ENV from the caller must not override it.
    env.pop("VIRTUAL_ENV", None)
    subprocess.run(["uv", "run", "python", str(WORKER_PATH)], check=True, cwd=REPO_ROOT, env=env)
    if not RESULT_PATH.exists():
        raise RuntimeError(f"The worker did not produce {RESULT_PATH}")
    RESULT_PATH.replace(result_path)


def compare_outputs() -> list[str]:
    head = np.load(HEAD_RESULT_PATH)
    base = np.load(BASE_RESULT_PATH)
    for name in sorted(set(head.files) - set(base.files)):
        print(f"INFO '{name}' exists only on head")
    for name in sorted(set(base.files) - set(head.files)):
        print(f"INFO '{name}' exists only on base")

    mismatches = []
    for name in sorted(set(head.files) & set(base.files)):
        head_array = head[name].astype(np.float64)
        base_array = base[name].astype(np.float64)
        if head_array.shape != base_array.shape:
            print(f"DIFF {name}: shape {head_array.shape} on head, {base_array.shape} on base")
            mismatches.append(name)
            continue

        difference = np.abs(head_array - base_array)
        # Measured against the largest value of the tensor: dividing element-wise would
        # turn round-off on a near-zero element into an enormous relative difference.
        scale = max(np.abs(head_array).max(), np.abs(base_array).max())
        relative = difference.max() / scale if scale > 0.0 else 0.0
        if relative <= RELATIVE_TOLERANCE:
            print(f"OK   {name}: shape={head_array.shape} max rel={relative:.3e}")
            continue
        index = np.unravel_index(difference.argmax(), difference.shape)
        print(f"DIFF {name}: shape={head_array.shape} max rel={relative:.3e} at {index}")
        print(f"       head={head_array[index]!r}  base={base_array[index]!r}")
        mismatches.append(name)
    return mismatches


def main() -> int:
    ensure_assets_dir()
    download_if_absent(f"{MODEL_BASE_URL}/{PARAM_NAME}", ASSETS_DIR / PARAM_NAME)
    download_if_absent(f"{MODEL_BASE_URL}/{CKPT_NAME}", ASSETS_DIR / CKPT_NAME)
    # The worker must survive the checkout, so it lives in the git-ignored assets dir.
    shutil.copyfile(Path(__file__).resolve(), WORKER_PATH)

    assert_no_tracked_changes()
    head_ref = resolve_head_ref()
    head_commit = run_git(["rev-parse", "HEAD"])
    base_commit = run_git(["rev-parse", "--verify", f"{BASE_REF}^{{commit}}"])
    print(f"\nhead: {head_ref} ({head_commit})")
    print(f"base: {BASE_REF} ({base_commit})")
    if head_commit == base_commit:
        print("Warning: head and base point at the same commit, the comparison is trivial")

    print(f"\n{'#' * 80}\nTrain-mode forward on {head_ref}\n{'#' * 80}")
    run_forward(HEAD_RESULT_PATH)

    print(f"\n{'#' * 80}\nTrain-mode forward on {BASE_REF}\n{'#' * 80}")
    run_git(["checkout", BASE_REF])
    try:
        run_forward(BASE_RESULT_PATH)
    finally:
        run_git(["checkout", head_ref])
        restored = run_git(["rev-parse", "HEAD"])
        print(f"\nRestored {head_ref} ({restored})")
        if restored != head_commit:
            raise RuntimeError(f"Failed to restore {head_ref}: HEAD is {restored}")

    print(f"\n{'#' * 80}\nComparing outputs\n{'#' * 80}")
    mismatches = compare_outputs()
    if mismatches:
        print(f"\nFAILED: {len(mismatches)} output(s) differ between {head_ref} and {BASE_REF}")
        print(f"  head: {HEAD_RESULT_PATH}")
        print(f"  base: {BASE_RESULT_PATH}")
        return 1
    print("\nPASSED: every output agrees within the tolerance")
    return 0


if __name__ == "__main__":
    if Path(__file__).resolve().parent.name == "assets":
        sys.exit(run_worker())
    sys.exit(main())
