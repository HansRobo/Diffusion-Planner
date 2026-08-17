#!/usr/bin/env python3
"""Launch pretrain/SFT training across all visible GPUs (Python replacement for train_run.sh).

Thin launcher only: it resolves the run dir, saves git info, sets NCCL env and runs
train_predictor.py under torch.distributed.run.

It takes exactly the same flags as train_predictor.py -- both parsers are generated from
the fields TrainConfig marks with ``cli(...)`` -- and forwards them verbatim. The only
thing it decides on its own is the run directory: it resolves ``save_dir`` once, here,
so the launcher and the trainer cannot end up pointing at different directories.
"""

import os
import subprocess
import sys
from pathlib import Path

from diffusion_planner.scenario_based_open_loop.open_loop import (
    load_scenario_based_open_loop_settings,
)
from diffusion_planner.train_cli import build_parser, resolve_paths, to_command_line
from diffusion_planner.train_config import TrainConfig
from diffusion_planner.utils.dist_init import dist_init_file_path
from run_utils import NCCL_ENV, gpu_count, tee_run


def write_git_info(save_path: Path, repo_dir: Path) -> None:
    def git_output(cmd: list[str]) -> str:
        return subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True).stdout

    branch = git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    (save_path / "git_show.txt").write_text(
        f"branch: {branch}\n\n" + git_output(["git", "show", "-s", "--decorate"])
    )
    (save_path / "git_diff.txt").write_text(git_output(["git", "diff"]))


def main() -> None:
    args = build_parser(description=__doc__).parse_args()
    resolve_paths(args)

    if args.scenario_based_open_loop_list:
        load_scenario_based_open_loop_settings(args.scenario_based_open_loop_list)

    # Resolved here rather than in each process: the child would otherwise derive its own
    # timestamp, and the directory this launcher writes git info and logs into would not
    # be the one the trainer saves checkpoints into.
    if not args.save_dir:
        args.save_dir = TrainConfig.build_save_dir(args.output_root, args.exp_name)
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    write_git_info(save_path, here)

    dist_init_file_path().unlink(missing_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(gpu_count()),
        "--standalone",
        "train_predictor.py",
        # output_root has already done its job: save_dir is now concrete.
        *to_command_line(args, exclude=("output_root",)),
    ]
    rc = tee_run(
        cmd, cwd=here, env={**os.environ, **NCCL_ENV}, log_path=save_path / "train_log.txt"
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
