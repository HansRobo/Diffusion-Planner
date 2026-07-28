"""Smoke-test that a v1-style dataset can still train every pipeline.

Point this at a dataset built by ``build_small_dataset.py`` and it runs a tiny,
fast pass of each training pipeline, reporting PASS/FAIL per pipeline:

* **SFT**   - ``diffusion_planner/train_predictor.py`` on the frame set (1 epoch,
  tiny batch): the frame NPZs load and train+val produce a finite loss.
* **RSFT**  - ``rlvr.autoresearch.run_experiment`` on the frame set (K-sample
  ranked SFT, 1 epoch): generation + reward + a LoRA checkpoint save.
* **R2LPL** - ``rlvr.autoresearch.tools.mine_direct_reproducer_chunks`` on the
  contiguous corpus. Default (no reward config) = plan-only: load the contiguous
  NPZs, detect the rollout lineage, and plan chunks by frame contiguity (does NOT
  run the model). Supplying ``danger_reward_config`` runs the full closed-loop model
  rollout (>=1 chunk simulated). PASS reflects whichever mode ran.

The intent is a fast "did I break training?" gate: change something, run this
against a dataset, get confirmation. Each pipeline can be skipped independently
(all run by default).

No paths are hard-coded. The only assumption is the dataset layout produced by
``build_small_dataset.py``::

    <dataset_root>/
      frame/train.json          # SFT + RSFT train scene list
      frame/val.json            # SFT + RSFT val scene list
      contiguous/window_scenes.json   # R2LPL ordered contiguous scene list
      normalization.json        # (or pass --normalization)

The base model (``--base_model``) is required for RSFT/R2LPL and must have an
``args.json`` beside it (the standard deploy-dir layout).

Usage:
    python -m rlvr.autoresearch.tools.verify_dataset_training \
        --dataset_root <v1_dir> --base_model <best_model.pth> \
        [--skip_sft] [--skip_rsft] [--skip_r2lpl] [--device auto]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# Repo root = three levels up from rlvr/autoresearch/tools/this_file.py. Subprocesses
# run from here so both ``rlvr`` and ``diffusion_planner`` import cleanly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parent / "dataset_smoke_configs"

# smoke-only keys the RSFT config carries but run_experiment must not receive as
# config fields (they are applied by this script instead).
_RSFT_SCRIPT_KEYS = {"_doc", "n_train_cap", "n_val_cap", "sft_batch_size", "train_epochs"}


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def _subsample(scene_list: Path, cap: int, seed: int, out_path: Path) -> int:
    """Write a seeded ``cap``-scene subset of a JSON scene list. Returns kept count."""
    scenes = _load_json(scene_list)
    if not isinstance(scenes, list):
        raise ValueError(f"{scene_list} must be a JSON list of NPZ paths")
    if cap and len(scenes) > cap:
        scenes = random.Random(seed).sample(scenes, cap)
    with open(out_path, "w") as f:
        json.dump(scenes, f)
    return len(scenes)


def _run(cmd: list[str], log_path: Path, env: dict) -> tuple[int, str]:
    """Run a subprocess from the repo root, teeing combined output to a log."""
    with open(log_path, "w") as log:
        proc = subprocess.run(
            cmd, cwd=str(_REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT
        )
    text = log_path.read_text(errors="replace")
    return proc.returncode, text


def resolve_device(requested: str) -> str:
    """Resolve the requested device to a concrete 'cuda' or 'cpu' using ACTUAL CUDA
    availability. 'auto' must check availability (hiding CUDA via an env var does not
    make it available), and 'cuda' on a CPU-only host fails loudly rather than
    launching subprocesses that then crash."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        avail = torch.cuda.is_available()
    except Exception:
        avail = False
    if requested == "cuda":
        if not avail:
            raise SystemExit("--device cuda requested but no CUDA device is available")
        return "cuda"
    return "cuda" if avail else "cpu"  # auto


def _env_for_device(device: str) -> dict:
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:  # cuda (already resolved concrete)
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Put the repo root on PYTHONPATH so repo-root packages (planner_metrics, rlvr,
    # diffusion_planner) resolve even when they are not pip-installed. train_predictor
    # runs as a script path (sys.path[0] = diffusion_planner/, not the repo root), so
    # without this a source-only planner_metrics is not importable.
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + prev if prev else "")
    return env


# --------------------------------------------------------------------------- #
# pipelines
# --------------------------------------------------------------------------- #
def run_sft(ds, cfg, args, env) -> tuple[bool, str]:
    train_list = _require(ds["frame_train"], "frame/train.json (SFT)")
    val_list = _require(ds["frame_val"], "frame/val.json (SFT)")
    norm = _require(ds["normalization"], "normalization.json (SFT)")
    work = args.out_dir / "sft"
    work.mkdir(parents=True, exist_ok=True)
    n_tr = _subsample(train_list, int(cfg.get("n_train_cap", 400)), args.seed, work / "train.json")
    n_va = _subsample(val_list, int(cfg.get("n_val_cap", 100)), args.seed, work / "val.json")
    cmd = [
        sys.executable,
        "diffusion_planner/train_predictor.py",
        "--exp_name",
        "verify_sft",
        "--save_dir",
        str(work / "out"),
        "--train_set_list",
        str(work / "train.json"),
        "--valid_set_list",
        str(work / "val.json"),
        "--normalization_file_path",
        str(norm),
        "--train_epochs",
        str(int(cfg.get("train_epochs", 1))),
        "--warm_up_epoch",
        str(int(cfg.get("warm_up_epoch", 0))),
        "--batch_size",
        str(int(cfg.get("batch_size", 8))),
        "--ddp",
        "false",
        # train_predictor defaults to --device cuda; pass the resolved concrete
        # device (args.device is 'cuda'/'cpu' after resolve_device) so --device cpu
        # (and auto on a CPU-only host) actually runs on CPU.
        "--device",
        args.device,
        "--use_data_augment",
        str(bool(cfg.get("use_data_augment", True))).lower(),
        "--augment_prob",
        str(float(cfg.get("augment_prob", 0.5))),
    ]
    print(f"[SFT] train {n_tr} / val {n_va} scenes -> {work / 'sft.log'}", flush=True)
    rc, text = _run(cmd, work / "sft.log", env)
    # capture inf/nan as full tokens too, so a non-finite loss is a clean FAIL rather
    # than a bare "-" that raises ValueError on float()
    num = r"(-?inf|nan|[0-9.eE+-]+)"
    tr_loss = re.findall(rf"epoch_mean_loss\['loss'\]={num}", text)
    va_loss = re.findall(rf"valid_loss_ego={num}", text)

    def _finite(xs):
        if not xs:
            return False
        try:
            return all(math.isfinite(float(x)) for x in xs)
        except ValueError:
            return False

    ok = rc == 0 and _finite(tr_loss) and _finite(va_loss)
    detail = (
        f"rc={rc} train_loss={tr_loss[-1] if tr_loss else 'NONE'} "
        f"valid_loss_ego={va_loss[-1] if va_loss else 'NONE'}"
    )
    return ok, detail


def run_rsft(ds, cfg, args, env) -> tuple[bool, str]:
    train_list = _require(ds["frame_train"], "frame/train.json (RSFT)")
    val_list = _require(ds["frame_val"], "frame/val.json (RSFT)")
    base_model = _require(args.base_model, "--base_model (RSFT)")
    _require(base_model.parent / "args.json", "args.json beside --base_model (RSFT)")
    work = args.out_dir / "rsft"
    work.mkdir(parents=True, exist_ok=True)
    n_tr = _subsample(train_list, int(cfg.get("n_train_cap", 40)), args.seed, work / "train.json")
    n_va = _subsample(val_list, int(cfg.get("n_val_cap", 40)), args.seed, work / "val.json")
    # hand run_experiment a clean config (strip the smoke-only keys)
    clean = {k: v for k, v in cfg.items() if k not in _RSFT_SCRIPT_KEYS}
    clean_path = work / "run_experiment_config.json"
    with open(clean_path, "w") as f:
        json.dump(clean, f, indent=2)
    cmd = [
        sys.executable,
        "-m",
        "rlvr.autoresearch.run_experiment",
        "--config",
        str(clean_path),
        "--name",
        "verify_rsft",
        "--model_path",
        str(base_model),
        "--train_scenes",
        str(work / "train.json"),
        "--val_scenes",
        str(work / "val.json"),
        "--output_dir",
        str(work / "out"),
        "--skip_baseline",
        "--train_epochs",
        str(int(cfg.get("train_epochs", 1))),
        "--sft_batch_size",
        str(int(cfg.get("sft_batch_size", 8))),
    ]
    print(f"[RSFT] train {n_tr} / val {n_va} scenes -> {work / 'rsft.log'}", flush=True)
    rc, text = _run(cmd, work / "rsft.log", env)
    # PASS requires evidence that TRAINING actually happened, not just rc=0 + a saved
    # adapter + a promoted summary reward:
    #  - a per-epoch training-loss line was emitted (`trained`) — this is the real
    #    guard: the trainer swallows train-load failures to {} for N=0, but then
    #    log_metrics raises (rc!=0) AND no per-epoch Loss= line is printed, so N=0
    #    can't satisfy `trained`. (kept_train below is informational only: the
    #    skip-filter line prints only when skip_filtered_scenes is enabled, so gating
    #    on it would false-FAIL a healthy run that disables that feature.)
    #  - a FINITE per-epoch validation reward (the "Eval [epochN-val]: reward=" line,
    #    which exists regardless of the model-quality promotion threshold; the summary
    #    "val_reward" stays -inf with --skip_baseline when the reward is <= -5, so
    #    gating on it would false-FAIL a real run).
    kept = re.findall(r"train:\s*kept\s+(\d+)\s*/", text)
    kept_train = int(kept[-1]) if kept else None
    # require a FINITE per-epoch training loss (a Loss=nan/inf line means training ran
    # but diverged — not evidence of a healthy pipeline)
    tl = re.findall(r"Epoch\s+\d+[^\n]*Loss=(-?inf|nan|[0-9.eE+-]+)", text)
    trained = False
    if tl:
        try:
            trained = math.isfinite(float(tl[-1]))
        except ValueError:
            trained = False
    ev = re.findall(r"Eval \[epoch\d+-val\][^\n]*reward=([+-]?(?:inf|nan|[0-9.eE]+))", text)
    ev_val = None
    if ev:
        try:
            v = float(ev[-1])
            ev_val = v if math.isfinite(v) else None
        except ValueError:
            ev_val = None
    lora = list((work / "out").rglob("adapter_model.safetensors"))
    ok = rc == 0 and trained and ev_val is not None and bool(lora)
    detail = (
        f"rc={rc} kept_train={kept_train} trained={trained} "
        f"epoch_val_reward={ev[-1] if ev else 'NONE'} lora_saved={'yes' if lora else 'no'}"
    )
    return ok, detail


def _resolve_cfg_path(value) -> Path | None:
    """Resolve a config-file reference: absolute as-is, else relative to the repo."""
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _scenes_loadable(scene_list: Path, k: int = 8) -> tuple[int, int]:
    """Sample k EVENLY-SPACED scenes across the whole list (incl. first + last), then
    run the canonical model-input loader (``load_npz_data``) on each — constructing
    the tensors the model actually consumes — not just a key/shape spot-check. Returns
    (checked, loadable). This makes the R2LPL check genuinely CONSUME the dataset: the
    miner's plan-only branch only parses path strings + frame suffixes, so nonexistent
    paths or zip members with junk payloads under the expected names would otherwise
    plan chunks and PASS. Even spacing (not just indices 0..k-1) means a corpus valid
    only at the start still fails."""
    scenes = _load_json(scene_list)
    if not isinstance(scenes, list) or not scenes:
        return 0, 0
    import torch

    from preference_optimization.utils import load_npz_data

    n = len(scenes)
    # Spacing denominator is (samples-1), where samples = min(k, n) — so the largest
    # index always lands on n-1 (the last scene) even when the corpus is smaller than k.
    m = min(k, n)
    idx = sorted({round(i * (n - 1) / max(m - 1, 1)) for i in range(m)}) if n > 1 else [0]
    checked = loadable = 0
    for i in idx:
        checked += 1
        try:
            # reuse the canonical model-input loader: constructs the tensors + runs
            # heading_to_cos_sin on ego/goal, so a scene the real pipeline cannot
            # consume (missing fields, incompatible shapes) raises here.
            load_npz_data(scenes[i], torch.device("cpu"))
            loadable += 1
        except Exception:
            pass
    return checked, loadable


def run_r2lpl(ds, cfg, args, env) -> tuple[bool, str]:
    scene_list = _require(ds["contig_scene_list"], "contiguous/window_scenes.json (R2LPL)")
    base_model = _require(args.base_model, "--base_model (R2LPL)")
    work = args.out_dir / "r2lpl"
    work.mkdir(parents=True, exist_ok=True)
    summary = work / "summary.json"
    cmd = [
        sys.executable,
        "-m",
        "rlvr.autoresearch.tools.mine_direct_reproducer_chunks",
        "--scene_list",
        str(scene_list),
        "--segments_jsonl",
        str(work / "segments.jsonl"),
        "--model_path",
        str(base_model),
        "--summary_json",
        str(summary),
        "--chunk_len",
        str(int(cfg.get("chunk_len", 80))),
        "--start_stride",
        str(int(cfg.get("start_stride", 80))),
        "--expected_frame_step",
        str(int(cfg.get("expected_frame_step", 1))),
    ]
    # A real closed-loop rollout needs the three danger_* configs (threshold/credit
    # ship in-repo; the reward config is an internal asset). When a reward config is
    # supplied we run the full model rollout; otherwise we fall back to --plan_only,
    # which still loads the contiguous NPZs, detects the rollout lineage and plans
    # chunks by frame-index contiguity (a self-contained dataset-consumable check;
    # pose/timeline continuity is validated only in the full rollout).
    # null danger_reward_config -> plan_only (self-contained). A SUPPLIED path must
    # exist: fail loudly on a typo/stale path rather than silently downgrading to
    # plan-only (which could report PASS without ever running the model).
    reward_ref = cfg.get("danger_reward_config")
    full_rollout = bool(reward_ref)
    if full_rollout:
        reward_cfg = _require(_resolve_cfg_path(reward_ref), "danger_reward_config (R2LPL rollout)")
        thr = _require(_resolve_cfg_path(cfg["danger_threshold_config"]), "danger_threshold_config")
        crd = _require(
            _resolve_cfg_path(cfg["danger_credit_window_config"]), "danger_credit_window_config"
        )
        cmd += [
            "--out_dir",
            str(work / "out"),
            "--out_jsonl",
            str(work / "mined.jsonl"),
            "--danger_reward_config",
            str(reward_cfg),
            "--danger_threshold_config",
            str(thr),
            "--danger_credit_window_config",
            str(crd),
            "--tracker_mode",
            str(cfg.get("tracker_mode", "mpc")),
            "--timeline_progress_mode",
            str(cfg.get("timeline_progress_mode", "clock")),
            "--neighbor_history_mode",
            str(cfg.get("neighbor_history_mode", "sim")),
            "--batch_size",
            str(int(cfg.get("batch_size", 8))),
        ]
    else:
        cmd += ["--plan_only"]
    # max_chunks bounds work in BOTH modes (plan-only still scans/plans the whole
    # corpus otherwise) — apply it regardless of the rollout branch.
    if cfg.get("max_chunks") is not None:
        cmd += ["--max_chunks", str(int(cfg["max_chunks"]))]
    mode = "full-rollout" if full_rollout else "plan-only"
    print(f"[R2LPL] reproducer ({mode}) on contiguous corpus -> {work / 'r2lpl.log'}", flush=True)
    rc, _ = _run(cmd, work / "r2lpl.log", env)
    sim = planned = None
    if summary.exists():
        s = _load_json(summary)
        sim = s.get("simulated_chunks")
        planned = s.get("planned_chunks")
    # The sampled contiguous scenes must actually load (guards plan-only, whose miner
    # branch only parses path strings and would PASS on a nonexistent corpus).
    checked, loadable = _scenes_loadable(scene_list, 5)
    scenes_ok = checked > 0 and loadable == checked
    # PASS = the dataset feeds R2LPL: full-rollout must simulate a chunk; plan-only
    # must plan at least one (lineage detected) AND the sampled scenes must load.
    key = sim if full_rollout else planned
    ok = rc == 0 and bool(key) and key >= 1 and scenes_ok
    detail = (
        f"rc={rc} mode={mode} planned_chunks={planned} simulated_chunks={sim} "
        f"scenes_loadable={loadable}/{checked}"
    )
    return ok, detail


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset_root", type=Path, required=True, help="dataset dir (build_small_dataset layout)"
    )
    ap.add_argument(
        "--base_model",
        type=Path,
        help="base model .pth (with args.json beside it); required unless SFT is the only pipeline",
    )
    ap.add_argument(
        "--normalization",
        type=Path,
        default=None,
        help="override normalization.json (default <dataset_root>/normalization.json)",
    )
    ap.add_argument("--sft_config", type=Path, default=_CONFIG_DIR / "sft.json")
    ap.add_argument("--rsft_config", type=Path, default=_CONFIG_DIR / "rsft.json")
    ap.add_argument("--r2lpl_config", type=Path, default=_CONFIG_DIR / "r2lpl.json")
    ap.add_argument("--skip_sft", action="store_true")
    ap.add_argument("--skip_rsft", action="store_true")
    ap.add_argument("--skip_r2lpl", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="scratch dir for logs+artifacts (default <dataset_root>/_verify_<ts>)",
    )
    args = ap.parse_args()

    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"--dataset_root not a directory: {root}")
    if args.out_dir is None:
        args.out_dir = root / f"_verify_{time.strftime('%Y%m%d-%H%M%S')}"
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = {
        "frame_train": root / "frame" / "train.json",
        "frame_val": root / "frame" / "val.json",
        "contig_scene_list": root / "contiguous" / "window_scenes.json",
        "normalization": args.normalization or (root / "normalization.json"),
    }
    args.device = resolve_device(args.device)  # 'auto' -> concrete cuda/cpu by availability
    env = _env_for_device(args.device)

    pipelines = []
    if not args.skip_sft:
        pipelines.append(("SFT", run_sft, _load_json(args.sft_config)))
    if not args.skip_rsft:
        pipelines.append(("RSFT", run_rsft, _load_json(args.rsft_config)))
    if not args.skip_r2lpl:
        pipelines.append(("R2LPL", run_r2lpl, _load_json(args.r2lpl_config)))
    if not pipelines:
        print("Nothing to do: all pipelines skipped.")
        return
    if any(name in ("RSFT", "R2LPL") for name, _, _ in pipelines) and not args.base_model:
        raise SystemExit("--base_model is required for RSFT / R2LPL")

    print(f"dataset_root: {root}")
    print(f"out_dir:      {args.out_dir}")
    print(f"device:       {args.device}\n")

    results = []
    for name, fn, cfg in pipelines:
        print(f"===== {name} =====", flush=True)
        t0 = time.time()
        try:
            ok, detail = fn(ds, cfg, args, env)
        except Exception as e:  # a setup failure is a pipeline failure, reported not raised
            ok, detail = False, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        status = "PASS" if ok else "FAIL"
        print(f"[{name}] {status}  ({dt:.0f}s)  {detail}\n", flush=True)
        results.append((name, ok, detail, dt))

    print("===== SUMMARY =====")
    for name, ok, detail, dt in results:
        print(f"  {name:6s} {'PASS' if ok else 'FAIL':4s}  {dt:5.0f}s  {detail}")
    all_ok = all(ok for _, ok, _, _ in results)
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
