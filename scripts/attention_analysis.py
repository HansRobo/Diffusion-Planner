"""Fusion attention analysis by token class (companion to scripts/token_importance.py).

Patches each fusion SelfAttentionBlock to expose attention weights, then
measures, per layer:
  - attention share received by each token class (ego-token query and
    all-valid-query average), vs the class's share of valid tokens
    -> selectivity ratio (=1 means indistinguishable from count-proportional
    dilution; >1 means the model actively prefers the class)
  - value-norm weighted share (alpha * ||W_v x||, approximation ignoring
    per-head structure / out-proj) — attention weight alone can overstate
    contribution of low-value tokens
  - distance-binned attention within lanes / neighbors (ego query)
  - route share in turning vs straight scenes (functional check: does
    attention shift with scenario?)

Usage:
  uv run python scripts/attention_analysis.py \
    --run_dir /home/isamuyamashita/work/diffusion_plannner/DP_exp/20260725-132546_plantf_V_tail \
    --valid_set_list /home/isamuyamashita/work/diffusion_plannner/mini_datasets/j6_2231_fullseq_mini_20260707/path_list_valid.json \
    --n_samples 128 --batch_size 8
"""

import argparse
import json
import re
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
    """Mirror validate_model's input prep (self-contained, tier4-main compatible)."""
    inputs = {k: v.to(device) for k, v in inputs.items()}
    B = inputs["ego_current_state"].shape[0]
    inputs["sampled_trajectories"] = torch.zeros(
        B, MAX_NUM_AGENTS, OUTPUT_T + 1, POSE_DIM, dtype=torch.float32, device=device
    )
    inputs["delay"] = torch.zeros(B, dtype=torch.float32, device=device)
    inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
    inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])
    inputs = cfg.observation_normalizer(inputs)
    return inputs


# token layout (dimensions.py defaults, matches V_tail args)
CLASSES = [
    ("ego", 1),
    ("neighbors", 320),
    ("static", 5),
    ("lanes", 140),
    ("route", 25),
    ("polygons", 10),
    ("line_strings", 60),
    ("goal_pose", 1),
    ("ego_shape", 1),
    ("turn_indicator", 1),
]
OFFSETS = {}
_o = 0
for _name, _n in CLASSES:
    OFFSETS[_name] = (_o, _o + _n)
    _o += _n
TOKEN_NUM = _o  # 564

DIST_BINS = [(0, 25), (25, 50), (50, 100), (100, float("inf"))]


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
    state = torch.load(latest_ckpt(run_dir), map_location=device)
    state = state["model"] if "model" in state else state
    state = {k.removeprefix("module."): v for k, v in state.items()}  # DDP-saved ckpt
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def find_fusion(model):
    for m in model.modules():
        if type(m).__name__ == "FusionEncoder":
            return m
    raise RuntimeError("FusionEncoder not found")


def patch_fusion(fusion, store):
    """Replace each block's forward to capture (attn_weights, kv_input, mask)."""
    for li, block in enumerate(fusion.blocks):

        def make_fwd(blk, layer_idx):
            def fwd(x, mask):
                attn_out, w = blk.attn(
                    blk.norm1(x),
                    x,
                    x,
                    key_padding_mask=mask,
                    need_weights=True,
                    average_attn_weights=True,
                )
                store.append(
                    {"layer": layer_idx, "w": w.detach(), "kv": x.detach(), "mask": mask.detach()}
                )
                x = x + blk.drop_path(attn_out)
                x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
                return x

            return fwd

        block.forward = make_fwd(block, li)


def value_norms(block, kv):
    """||W_v x + b_v|| per token (heads concatenated; out_proj ignored)."""
    E = kv.shape[-1]
    W = block.attn.in_proj_weight[2 * E : 3 * E]
    b = block.attn.in_proj_bias[2 * E : 3 * E]
    return (kv @ W.T + b).norm(dim=-1)  # [B, N]


def neighbor_dist(nbr):
    # Match NeighborEncoder, which retains only the last six history rows.
    valid = (nbr[:, :, -6:, :8] != 0).any(dim=(2, 3))
    d = nbr[:, :, -1, :2].norm(dim=-1)
    return torch.where(valid, d, torch.full_like(d, float("inf")))


def lane_dist(lanes):
    pt_valid = (lanes[..., :8] != 0).any(dim=-1)
    d = lanes[..., :2].norm(dim=-1)
    d = torch.where(pt_valid, d, torch.full_like(d, float("inf")))
    return d.min(dim=-1).values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--valid_set_list", required=True)
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--move_min_m", type=float, default=5.0)
    p.add_argument("--turn_deg", type=float, default=15.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_json", default="", help="write aggregated results to this JSON file")
    args = p.parse_args()

    model, cfg = load_model(Path(args.run_dir), args.device)
    fusion = find_fusion(model)
    store = []
    patch_fusion(fusion, store)
    n_layers = len(fusion.blocks)
    print(f"patched {n_layers} fusion blocks, token_num={TOKEN_NUM}", flush=True)

    dataset = DiffusionPlannerData(args.valid_set_list)
    stride = max(1, len(dataset) // (args.n_samples * 2))
    cand = list(range(0, len(dataset), stride))
    probe = DataLoader(Subset(dataset, cand), batch_size=1, shuffle=False)
    idxs, turning = [], []
    for i, s in zip(cand, probe):
        fut = s["ego_agent_future"][0]
        if float(np.linalg.norm(fut[-1, :2])) < args.move_min_m:
            continue
        bearing = abs(np.degrees(np.arctan2(float(fut[-1, 1]), float(fut[-1, 0]))))
        idxs.append(i)
        turning.append(bearing >= args.turn_deg)
        if len(idxs) >= args.n_samples:
            break
    turning = np.array(turning)
    print(
        f"selected {len(idxs)} moving samples ({int(turning.sum())} turning, dataset={len(dataset)})",
        flush=True,
    )

    loader = DataLoader(Subset(dataset, idxs), batch_size=args.batch_size, shuffle=False)

    # accumulators: [layer][class] lists over samples
    ego_share = [{c: [] for c, _ in CLASSES} for _ in range(n_layers)]
    all_share = [{c: [] for c, _ in CLASSES} for _ in range(n_layers)]
    vw_share = [{c: [] for c, _ in CLASSES} for _ in range(n_layers)]
    count_share = {c: [] for c, _ in CLASSES}
    lane_bin_share = [[] for _ in DIST_BINS]  # conditional on class presence
    nbr_bin_share = [[] for _ in DIST_BINS]
    route_share_per_sample = []  # ego-query, layer-averaged (for turning split)

    with torch.no_grad():
        for raw in loader:
            l_dist = lane_dist(raw["lanes"]).to(args.device)  # [B, P] raw metres
            n_dist = neighbor_dist(raw["neighbor_agents_past"]).to(args.device)
            inputs_n = prepare_inputs(dict(raw), cfg, args.device)
            store.clear()
            model(inputs_n)
            assert len(store) == n_layers
            B = store[0]["w"].shape[0]
            valid = ~store[0]["mask"]  # [B, N] True=valid
            vf = valid.float()
            n_valid = vf.sum(dim=1)  # [B]

            for c, _ in CLASSES:
                s, e = OFFSETS[c]
                count_share[c] += (vf[:, s:e].sum(dim=1) / n_valid).tolist()

            per_layer_route = []
            per_layer_lane_bins = [[] for _ in DIST_BINS]
            per_layer_nbr_bins = [[] for _ in DIST_BINS]
            for li, rec in enumerate(store):
                w = rec["w"]  # [B, Q, K], rows over valid keys sum to 1
                ego_w = w[:, 0]  # [B, K]
                # all-valid-query mean of attention received per key
                allq_w = (w * vf.unsqueeze(-1)).sum(dim=1) / n_valid.unsqueeze(-1)
                vn = value_norms(fusion.blocks[li], rec["kv"])  # [B, K]
                vw = ego_w * vn
                vw = vw / vw.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                for c, _ in CLASSES:
                    s, e = OFFSETS[c]
                    ego_share[li][c] += ego_w[:, s:e].sum(dim=1).tolist()
                    all_share[li][c] += allq_w[:, s:e].sum(dim=1).tolist()
                    vw_share[li][c] += vw[:, s:e].sum(dim=1).tolist()
                # distance bins (conditional within class, ego query)
                ls, le = OFFSETS["lanes"]
                ns, ne = OFFSETS["neighbors"]
                lane_w = ego_w[:, ls:le]
                nbr_w = ego_w[:, ns:ne]
                for bi, (lo, hi) in enumerate(DIST_BINS):
                    lm = (l_dist >= lo) & (l_dist < hi)
                    nm = (n_dist >= lo) & (n_dist < hi)
                    lt = lane_w.sum(dim=1)
                    nt = nbr_w.sum(dim=1)
                    per_layer_lane_bins[bi].append(
                        torch.where(
                            lt > 0, (lane_w * lm).sum(dim=1) / lt.clamp(min=1e-9), torch.nan
                        )
                    )
                    per_layer_nbr_bins[bi].append(
                        torch.where(nt > 0, (nbr_w * nm).sum(dim=1) / nt.clamp(min=1e-9), torch.nan)
                    )
                rs, re_ = OFFSETS["route"]
                per_layer_route.append(ego_w[:, rs:re_].sum(dim=1))
            route_share_per_sample += torch.stack(per_layer_route).mean(dim=0).tolist()
            for bi in range(len(DIST_BINS)):
                lane_bin_share[bi] += torch.stack(per_layer_lane_bins[bi]).mean(dim=0).tolist()
                nbr_bin_share[bi] += torch.stack(per_layer_nbr_bins[bi]).mean(dim=0).tolist()

    # ---------------- aggregate ----------------
    r_route = np.array(route_share_per_sample)
    results = {
        "n_samples": len(idxs),
        "n_turning": int(turning.sum()),
        "n_layers": n_layers,
        "classes": [c for c, _ in CLASSES],
        "count_share": {c: float(np.mean(count_share[c])) for c, _ in CLASSES},
        "ego_share_per_layer": {
            c: [float(np.mean(ego_share[li][c])) for li in range(n_layers)] for c, _ in CLASSES
        },
        "all_share_avg": {
            c: float(np.mean([np.mean(all_share[li][c]) for li in range(n_layers)]))
            for c, _ in CLASSES
        },
        "vw_share_avg": {
            c: float(np.mean([np.mean(vw_share[li][c]) for li in range(n_layers)]))
            for c, _ in CLASSES
        },
        "dist_bins": ["0-25m", "25-50m", "50-100m", "100m+"],
        "lane_bin_share": [float(np.nanmean(x)) for x in lane_bin_share],
        "nbr_bin_share": [float(np.nanmean(x)) for x in nbr_bin_share],
        "route_share_turning": float(r_route[turning].mean()) if turning.any() else None,
        "route_share_straight": float(r_route[~turning].mean()) if (~turning).any() else None,
    }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.out_json}", flush=True)

    # ---------------- report ----------------
    def pct(x):
        return f"{100 * np.nanmean(x):5.1f}%"

    print("\n=== class attention share (ego-token query), per fusion layer ===")
    hdr = (
        f"{'class':<14}{'count':>7}"
        + "".join(f"  L{li}" + " " * 4 for li in range(n_layers))
        + "   vw(L-avg)  select"
    )
    print(hdr)
    for c, _ in CLASSES:
        cs = np.mean(count_share[c])
        layer_avg = np.mean([np.mean(ego_share[li][c]) for li in range(n_layers)])
        vw_avg = np.mean([np.mean(vw_share[li][c]) for li in range(n_layers)])
        sel = layer_avg / cs if cs > 0 else float("nan")
        row = f"{c:<14}{100 * cs:6.1f}%" + "".join(
            f" {100 * np.mean(ego_share[li][c]):5.1f}%" for li in range(n_layers)
        )
        print(row + f"   {100 * vw_avg:5.1f}%   {sel:5.2f}x")

    print("\n=== class attention share (all-valid-query mean), layer-averaged ===")
    for c, _ in CLASSES:
        cs = np.mean(count_share[c])
        avg = np.mean([np.mean(all_share[li][c]) for li in range(n_layers)])
        sel = avg / cs if cs > 0 else float("nan")
        print(f"{c:<14} count={100 * cs:5.1f}%  attn={100 * avg:5.1f}%  select={sel:5.2f}x")

    print("\n=== ego-query attention within lanes / neighbors, by distance (layer-avg) ===")
    print(f"{'bin':<12}{'lanes':>8}{'neighbors':>11}")
    for bi, (lo, hi) in enumerate(DIST_BINS):
        label = f"{lo:.0f}-{hi:.0f}m" if np.isfinite(hi) else f"{lo:.0f}m+"
        print(f"{label:<12}{pct(lane_bin_share[bi]):>8}{pct(nbr_bin_share[bi]):>11}")

    print("\n=== route share (ego query, layer-avg): turning vs straight ===")
    r = np.array(route_share_per_sample)
    print(f"turning  (n={int(turning.sum())}): {100 * r[turning].mean():5.2f}%")
    print(f"straight (n={int((~turning).sum())}): {100 * r[~turning].mean():5.2f}%")


if __name__ == "__main__":
    main()
