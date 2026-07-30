"""Token-group importance via input ablation (feature-importance analog).

Zeroing raw input rows makes the encoder mask those tokens (verified: padding
masks propagate through normalizer -> encoder -> fusion -> decoder cross-attn),
so "remove token group and measure metric degradation" is a clean permutation-
importance analog. Groups: input class (drop:*), nearest-K sweeps for
neighbors / lanes / line_strings, and 100m distance cutoffs.

Usage (CPU, sshfs checkpoint):
  uv run python scripts/token_importance.py \
    --run_dir /home/isamuyamashita/work/diffusion_plannner/DP_exp/20260725-132546_plantf_V_tail \
    --valid_set_list /home/isamuyamashita/work/diffusion_plannner/mini_datasets/j6_2231_fullseq_mini_20260707/path_list_valid.json \
    --n_samples 128 --batch_size 8 --out_tsv token_importance_vtail.tsv
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.dimensions import MAX_NUM_AGENTS, OUTPUT_T, POSE_DIM
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.train_epoch import heading_to_cos_sin
from diffusion_planner.utils.config import Config
from diffusion_planner.utils.dataset import DiffusionPlannerData
from torch.utils.data import DataLoader, Subset


def prepare_inputs(inputs: dict, cfg, device: str):
    """Mirror validate_model's input prep (self-contained, tier4-main compatible).

    Returns (normalized model inputs, ego_future in metres/cos-sin).
    """
    inputs = {k: v.to(device) for k, v in inputs.items()}
    B = inputs["ego_current_state"].shape[0]
    inputs["sampled_trajectories"] = torch.zeros(
        B, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, dtype=torch.float32, device=device
    )
    inputs["delay"] = torch.zeros(B, dtype=torch.float32, device=device)
    inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
    inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])
    ego_future = heading_to_cos_sin(inputs["ego_agent_future"])
    inputs = cfg.observation_normalizer(inputs)
    return inputs, ego_future


ONNX_TYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(bool)": np.bool_,
    "tensor(int64)": np.int64,
    "tensor(double)": np.float64,
}


class OnnxModel:
    """Callable mimicking Diffusion_Planner(inputs) -> (_, outputs) on an ORT session.

    The ONNX graph contains the encoder's zeroing/masking logic, so raw-input
    ablation works identically to the PyTorch path. Normalization is NOT
    embedded in the ONNX; the caller must pass normalized inputs (same as the
    C++ node), which prepare_inputs already does.
    """

    def __init__(self, onnx_path: str, device: str = "cpu"):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_specs = self.sess.get_inputs()
        print(f"onnxruntime providers: {self.sess.get_providers()}", flush=True)

    def __call__(self, feed: dict):
        B = feed["ego_current_state"].shape[0]
        onnx_feed = {}
        for si in self.input_specs:
            dtype = ONNX_TYPE_MAP.get(si.type, np.float32)
            shp = [B if not isinstance(d, int) or d <= 0 else d for d in si.shape]
            if si.name in feed:
                arr = feed[si.name].cpu().numpy().astype(dtype)
                if list(arr.shape) != shp and arr.size == int(np.prod(shp)):
                    arr = arr.reshape(shp)  # e.g. delay [B] -> [B,1]
                onnx_feed[si.name] = arr
            else:
                onnx_feed[si.name] = np.zeros(shp, dtype=dtype)
        pred = self.sess.run(["prediction"], onnx_feed)[0]
        return None, {"prediction": torch.from_numpy(np.asarray(pred))}


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
    state = {k.removeprefix("module."): v for k, v in state.items()}  # DDP-saved ckpt
    model.load_state_dict(state)
    model.eval()
    return model, cfg, ckpt_path


# ---------------------------------------------------------------- ablations


def _neighbor_dist(nbr: torch.Tensor) -> torch.Tensor:
    """nbr: [B, N, T, 11] raw. Distance of current position; inf for padding."""
    valid = (nbr[..., :8] != 0).any(dim=(2, 3))  # [B, N]
    d = nbr[:, :, -1, :2].norm(dim=-1)  # [B, N]
    return torch.where(valid, d, torch.full_like(d, float("inf")))


def _polyline_dist(x: torch.Tensor, geom_dims: int | None) -> torch.Tensor:
    """x: [B, P, V, D] raw. Min distance over valid points; inf for padding."""
    pt_valid = (x != 0).any(dim=-1) if geom_dims is None else (x[..., :geom_dims] != 0).any(dim=-1)
    d = x[..., :2].norm(dim=-1)  # [B, P, V]
    d = torch.where(pt_valid, d, torch.full_like(d, float("inf")))
    return d.min(dim=-1).values  # [B, P]


def _keep_topk(x: torch.Tensor, dist: torch.Tensor, k: int) -> torch.Tensor:
    """Zero all element slots except the k nearest valid ones."""
    x = x.clone()
    B = x.shape[0]
    for b in range(B):
        order = torch.argsort(dist[b])  # inf (padding) sorts last
        drop = order[k:]
        finite = torch.isfinite(dist[b][drop])
        x[b, drop[finite]] = 0.0
    return x


def _keep_within(x: torch.Tensor, dist: torch.Tensor, radius: float) -> torch.Tensor:
    x = x.clone()
    far = torch.isfinite(dist) & (dist > radius)
    x[far] = 0.0
    return x


def apply_ablation(inputs: dict, name: str) -> dict:
    out = {k: v.clone() for k, v in inputs.items()}
    if name == "baseline":
        return out
    kind, _, arg = name.partition(":")
    if kind == "drop":
        key = {
            "neighbors": "neighbor_agents_past",
            "lanes": "lanes",
            "route": "route_lanes",
            "line_strings": "line_strings",
            "polygons": "polygons",
            "goal_pose": "goal_pose",
            "turn_indicators": "turn_indicators",
        }[arg]
        out[key] = torch.zeros_like(out[key])
        if key in ("lanes", "route_lanes"):
            out[f"{key}_speed_limit"] = torch.zeros_like(out[f"{key}_speed_limit"])
            out[f"{key}_has_speed_limit"] = torch.zeros_like(out[f"{key}_has_speed_limit"])
        return out
    if kind == "nbr_top":
        d = _neighbor_dist(out["neighbor_agents_past"])
        out["neighbor_agents_past"] = _keep_topk(out["neighbor_agents_past"], d, int(arg))
        return out
    if kind == "nbr_within":
        d = _neighbor_dist(out["neighbor_agents_past"])
        out["neighbor_agents_past"] = _keep_within(out["neighbor_agents_past"], d, float(arg))
        return out
    if kind == "lane_top":
        d = _polyline_dist(out["lanes"], 8)
        out["lanes"] = _keep_topk(out["lanes"], d, int(arg))
        return out
    if kind == "lane_within":
        d = _polyline_dist(out["lanes"], 8)
        out["lanes"] = _keep_within(out["lanes"], d, float(arg))
        return out
    if kind == "ls_top":
        d = _polyline_dist(out["line_strings"], None)
        out["line_strings"] = _keep_topk(out["line_strings"], d, int(arg))
        return out
    raise ValueError(name)


CONFIGS = [
    "baseline",
    # class-level removal (permutation-importance analog)
    "drop:neighbors",
    "drop:lanes",
    "drop:route",
    "drop:line_strings",
    "drop:polygons",
    "drop:goal_pose",
    "drop:turn_indicators",
    # nearest-K sweeps (token-count optimization curves)
    "nbr_top:2",
    "nbr_top:4",
    "nbr_top:8",
    "nbr_top:16",
    "nbr_within:50",
    "nbr_within:100",
    "lane_top:10",
    "lane_top:20",
    "lane_top:40",
    "lane_top:70",
    "lane_top:100",
    "lane_within:50",
    "lane_within:100",
    "ls_top:10",
    "ls_top:20",
    "ls_top:40",
]


# ---------------------------------------------------------------- metrics


@torch.no_grad()
def eval_config(model, cfg, batches, device, name):
    fde_top, ade_top, min_fde, min_ade = [], [], [], []
    for raw in batches:
        inputs = apply_ablation(raw, name)
        inputs_n, ego_future = prepare_inputs(inputs, cfg, device)
        _, outputs = model(inputs_n)
        gt = ego_future[..., :2]  # [B,T,2] metres
        if "trajectory" in outputs and "probability" in outputs:
            traj = outputs["trajectory"]  # [B,K,T,4] normalized
            std = cfg.state_normalizer.std[0].to(device)
            mean = cfg.state_normalizer.mean[0].to(device)
            xy = (traj * std + mean)[..., :2]  # [B,K,T,2]
            err = (xy - gt[:, None]).norm(dim=-1)  # [B,K,T]
            ade_k = err.mean(dim=-1)  # [B,K]
            fde_k = err[..., -1]  # [B,K]
            top = outputs["probability"].argmax(dim=-1)  # [B]
            bidx = torch.arange(gt.shape[0], device=fde_k.device)
            fde_top += fde_k[bidx, top].tolist()
            ade_top += ade_k[bidx, top].tolist()
            min_fde += fde_k.min(dim=-1).values.tolist()
            min_ade += ade_k.min(dim=-1).values.tolist()
        else:
            pred = outputs["prediction"][:, 0, :, :2]  # [B,T,2] metres
            err = (pred - gt).norm(dim=-1)
            fde_top += err[:, -1].tolist()
            ade_top += err.mean(dim=-1).tolist()
            min_fde += err[:, -1].tolist()
            min_ade += err.mean(dim=-1).tolist()
    return {
        "fde_top": float(np.mean(fde_top)),
        "ade_top": float(np.mean(ade_top)),
        "min_fde": float(np.mean(min_fde)),
        "min_ade": float(np.mean(min_ade)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", help="PyTorch run dir (args.json + epochXXXX/best_model.pth)")
    p.add_argument("--onnx", help="Path to diffusion_planner.onnx (forward-only backend)")
    p.add_argument("--args_json", help="args.json for --onnx (default: sibling of the onnx)")
    p.add_argument("--valid_set_list", required=True)
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--move_min_m", type=float, default=5.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_tsv", default="token_importance.tsv")
    args = p.parse_args()

    if args.onnx:
        args_json = args.args_json or str(Path(args.onnx).parent / "args.json")
        cfg = Config(args_json)
        cfg.device = args.device
        cfg.ddp = False
        model = OnnxModel(args.onnx, args.device)
        print(f"loaded ONNX: {args.onnx} (normalizer from {args_json})", flush=True)
    else:
        model, cfg, ckpt = load_model(Path(args.run_dir), args.device)
        print(f"loaded: {ckpt}")

    # Spread sample selection across the whole valid set (avoid one clip),
    # keeping only "moving" scenes so map/agent inputs actually matter.
    dataset = DiffusionPlannerData(args.valid_set_list)
    probe = DataLoader(dataset, batch_size=1, shuffle=False)
    stride = max(1, len(dataset) // (args.n_samples * 2))
    idxs = []
    for i, s in enumerate(probe):
        if i % stride:
            continue
        if float(np.linalg.norm(s["ego_agent_future"][0, -1, :2])) >= args.move_min_m:
            idxs.append(i)
        if len(idxs) >= args.n_samples:
            break
    print(f"selected {len(idxs)} moving samples (stride={stride})")

    loader = DataLoader(Subset(dataset, idxs), batch_size=args.batch_size, shuffle=False)
    batches = [{k: v for k, v in b.items()} for b in loader]

    rows = []
    base = None
    for name in CONFIGS:
        t0 = time.time()
        m = eval_config(model, cfg, batches, args.device, name)
        if name == "baseline":
            base = m
        d = {f"d_{k}": m[k] - base[k] for k in m}
        rows.append({"config": name, **m, **d})
        print(
            f"{name:<22} fde_top={m['fde_top']:6.2f} ({d['d_fde_top']:+5.2f})  "
            f"ade_top={m['ade_top']:5.2f} ({d['d_ade_top']:+5.2f})  "
            f"min_fde={m['min_fde']:6.2f} ({d['d_min_fde']:+5.2f})  "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )

    cols = list(rows[0].keys())
    with open(args.out_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"wrote {args.out_tsv}")


if __name__ == "__main__":
    main()
