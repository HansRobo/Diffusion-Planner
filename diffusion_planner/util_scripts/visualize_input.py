import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.path_list import (
    build_frame_index,
    load_frame,
    load_path_list,
)
from diffusion_planner.utils.visualize_input import visualize_inputs
from tqdm import tqdm

_PACKED_MARKER = "frame_indices"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("save_path", type=Path)
    return parser.parse_args()


def probe_frames(npz_path: Path):
    """Return ``(num_frames, is_packed)`` for one npz without decompressing feature arrays."""
    with np.load(npz_path, allow_pickle=True) as npz:
        if _PACKED_MARKER in npz.files:
            return int(len(npz[_PACKED_MARKER])), True
    return 1, False


def build_annotation(npz_path: Path, frame_idx: int, is_packed: bool) -> tuple[str, str]:
    """Build the top banner for a frame as (text, color).

    The banner is always two lines (status + npz path) drawn at a fixed size so the figure
    height stays constant across frames. Kept frames show a faint grey "OK"; frames the C++
    data_converter would drop show the reason in red. The filter result is read from the
    sibling ``<token>.json`` -- a single dict for legacy per-frame files, or a per-frame list
    for packed sequence files (indexed by ``frame_idx``).
    """
    status, color = "OK", "0.7"  # faint grey for kept frames
    label = f"{npz_path}" + (f"  [frame {frame_idx}]" if is_packed else "")
    json_path = npz_path.with_suffix(".json")
    if json_path.is_file():
        try:
            with open(json_path, "r") as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            info = {}
        if isinstance(info, list):
            info = info[frame_idx] if frame_idx < len(info) else {}
        if info.get("is_skipped"):
            details = info.get("skipping_info", {}).get("details", "skipped")
            status = f"WOULD BE DROPPED IN REAL GENERATION — {details}"
            color = "red"
    return f"{status}\n{label}", color


def process_one_frame(input_path: Path, frame_idx: int, is_packed: bool, save_path: Path):
    frame = load_frame(str(input_path), frame_idx)
    data = {}
    for key, value in frame.items():
        if key in ("map_name", "token"):
            continue
        # add batch size axis
        data[key] = torch.tensor(np.expand_dims(value, axis=0))
    data["goal_pose"] = heading_to_cos_sin(data["goal_pose"])
    data["ego_agent_past"] = heading_to_cos_sin(data["ego_agent_past"])

    annotation, annotation_color = build_annotation(input_path, frame_idx, is_packed)
    visualize_inputs(
        data,
        save_path,
        view_ranges=[60, 150],
        annotation=annotation,
        annotation_color=annotation_color,
    )


def _render_task(task):
    """Pool worker: ``(input_path, frame_idx, is_packed, out_png)`` -> render unless it exists."""
    input_path, frame_idx, is_packed, out_png = task
    if out_png.exists():
        return
    process_one_frame(input_path, frame_idx, is_packed, out_png)


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    save_path = args.save_path
    ext = input_path.suffix

    if ext == ".npz":
        num_frames, is_packed = probe_frames(input_path)
        if num_frames == 1:
            process_one_frame(input_path, 0, is_packed, save_path)
            print(f"Saved visualization to {save_path}")
        else:
            # A packed sequence renders one png per frame into save_path/ (treated as a dir).
            save_path.mkdir(parents=True, exist_ok=True)
            tasks = [
                (input_path, f, True, save_path / f"{input_path.stem}_f{f:06d}.png")
                for f in range(num_frames)
            ]
            with Pool(8) as pool, tqdm(total=len(tasks)) as pbar:
                for _ in pool.imap_unordered(_render_task, tasks):
                    pbar.update(1)
            print(f"Saved {len(tasks)} frame visualizations to {save_path}")
    elif ext == ".json":
        entries = load_path_list(input_path)
        frame_index = build_frame_index(entries)

        save_path.mkdir(parents=True, exist_ok=True)
        assert save_path.is_dir()
        rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        tasks = []
        for entry_idx, frame_idx in frame_index:
            entry = entries[entry_idx]
            path = Path(entry["path"])
            is_packed = entry["num_frames"] > 1 or entry["valid"] is not None
            suffix = f"_f{frame_idx:06d}" if is_packed else ""
            out_png = save_path / f"{path.parent.name}_{path.stem}{suffix}.png"
            tasks.append((path, frame_idx, is_packed, out_png))

        with Pool(8) as pool, tqdm(total=len(tasks)) as pbar:
            for _ in pool.imap_unordered(_render_task, tasks):
                pbar.update(1)
