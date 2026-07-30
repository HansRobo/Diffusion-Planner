"""T_fwd re-measurement with explicit CUDA synchronization.

Purpose: plan/08 recorded T_fwd = 28.8 ms for the validator's torch path, but the
measuring script was lost and we cannot confirm it synchronized the CUDA stream.
An unsynced timer misattributes GPU time, so the 28.8 ms (and the 2.9x gap versus
the production TensorRT node's ~10 ms) is unverified. This script establishes a
trustworthy baseline and doubles as the harness for measuring torch.compile /
CUDA-graph gains.

Scope matched to the production node's "inference only" measurement: one
``model(data)`` call = encoder (once) + DPM-Solver decoder (10 steps) +
turn indicator. Tensor preparation (``to_model_tensors``) is excluded, as in
plan/09's per-tick table.

Weights are randomly initialised: no ``best_model.pth`` is present locally (it
was pulled per-run via webauto and the scratchpad is gone). Latency does not
depend on weight values -- only on architecture and shapes, both of which come
from the production ``args.json``.
"""

import argparse
import contextlib
import json
import statistics
import subprocess
import time

import torch

# Input spec lifted from the exported graph's declared inputs (verified against
# autoware_diffusion_planner's dimensions.hpp: MAX_NUM_AGENTS=321, OUTPUT_T=80,
# INPUT_T=30, NUM_SEGMENTS_IN_LANE=140, NUM_SEGMENTS_IN_ROUTE=25).
INPUT_SPEC = {
    "sampled_trajectories": ((321, 81, 4), torch.float32),
    "ego_agent_past": ((31, 4), torch.float32),
    "ego_current_state": ((10,), torch.float32),
    "neighbor_agents_past": ((320, 31, 11), torch.float32),
    "static_objects": ((5, 10), torch.float32),
    "lanes": ((140, 20, 33), torch.float32),
    "lanes_speed_limit": ((140, 1), torch.float32),
    "lanes_has_speed_limit": ((140, 1), torch.bool),
    "route_lanes": ((25, 20, 33), torch.float32),
    "route_lanes_speed_limit": ((25, 1), torch.float32),
    "route_lanes_has_speed_limit": ((25, 1), torch.bool),
    "polygons": ((10, 40, 3), torch.float32),
    "line_strings": ((60, 20, 4), torch.float32),
    "goal_pose": ((4,), torch.float32),
    "ego_shape": ((3,), torch.float32),
    "turn_indicators": ((31,), torch.float32),
}


def build_inputs(batch: int, n_neighbors: int, device: str) -> dict:
    """Synthetic input dict. ``n_neighbors`` rows of ``neighbor_agents_past`` stay
    non-zero; the rest are zeroed so the DiT attention mask
    (``sum(neighbor_current != 0) == 0`` -> masked, dit.py:156) sees exactly that
    many valid neighbours."""
    data = {}
    for name, (shape, dtype) in INPUT_SPEC.items():
        full = (batch, *shape)
        if dtype is torch.bool:
            data[name] = torch.ones(full, dtype=dtype, device=device)
        else:
            data[name] = torch.randn(full, dtype=dtype, device=device)
    data["neighbor_agents_past"][:, n_neighbors:] = 0.0
    # delay is [1, 1] in the graph; 0 = no frozen prefix steps.
    data["delay"] = torch.zeros((1, 1), dtype=torch.float32, device=device)
    return data


def check_gpu_exclusive() -> list[str]:
    """Return the other compute processes on the GPU, if any.

    The dev box shares its single 4090 with other work (a CARLA sim, a running
    pilot-auto stack). Contention inflated a measured T_fwd from ~31 ms to
    ~40 ms with max 84 ms, so a silently-contended run produces numbers that
    look like a regression. Always report what else was resident."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    mine = str(subprocess.os.getpid())
    return [ln for ln in out.strip().splitlines() if ln.strip() and not ln.startswith(mine + ",")]


def time_synced(fn, iters: int) -> list[float]:
    """Per-iteration wall time with the CUDA stream drained before and after."""
    out = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


VARIANTS = (
    "eager",
    # CUDA graphs WITHOUT inductor: the same eager kernels in the same order, replayed from a
    # captured graph. Only launch overhead goes away, so unlike the inductor variants the
    # arithmetic -- and therefore the output -- cannot change.
    "cudagraphs",
    "compile",
    "compile_all",  # torch.compile, default mode
    "compile_ro",
    "compile_all_ro",  # mode="reduce-overhead" (uses CUDA graphs)
    "compile_loop_ro",  # + compile the whole decoder (unroll the DPM loop)
    "compile_whole_ro",  # compile the whole model as ONE cudagraph region
    "compile_ma",
    "compile_all_ma",  # mode="max-autotune"
    "bf16",
    "compile_bf16",
    "tf32",
    "compile_tf32",
)

# Suffix -> torch.compile mode. "reduce-overhead" enables CUDA graphs under the
# hood, which is the cheap path to graph capture for this launch-bound workload
# (bf16/TF32 do nothing here, so the cost is dispatch, not FLOPs).
_COMPILE_MODES = {"_ro": "reduce-overhead", "_ma": "max-autotune"}


def build_model(args, variant: str, device: str, seed: int = 0):
    """Fresh model per variant so compilation of one cannot leak into another.

    Seeded, because the weights are random and comparing two variants' OUTPUTS requires every
    variant to hold the same ones.
    """
    from diffusion_planner.model.diffusion_planner import Diffusion_Planner

    # One implementation of the clone, shared with the rollout that ships it.
    from scenario_generation.scenario_sim_rollout import _CloneEncoderOutput

    torch.manual_seed(seed)
    model = Diffusion_Planner(args).to(device).eval()
    # Match the validator path exactly: guidance is disabled there
    # (simulate.py:632 sets ``model.decoder._guidance_fn = None``).
    model.decoder._guidance_fn = None
    if variant == "cudagraphs":
        # Same two modules as the inductor variants, and the same clone around the encoder:
        # compiling the WHOLE model into one graph fails with "accessing tensor output of
        # CUDAGraphs that has been overwritten by a subsequent run" -- the encoding is held
        # across all DPM steps, which is exactly what _CloneEncoderOutput exists to fix.
        model.decoder.dit = torch.compile(model.decoder.dit, backend="cudagraphs")
        model.encoder = _CloneEncoderOutput(torch.compile(model.encoder, backend="cudagraphs"))
        return model
    mode = next((m for sfx, m in _COMPILE_MODES.items() if variant.endswith(sfx)), None)
    kw = {"mode": mode} if mode else {}
    if variant == "compile_whole_ro":
        # One region for encoder + decoder + the unrolled DPM loop: no cross-region
        # cudagraph buffer hazard, so no clone wrapper is needed.
        return torch.compile(model, mode="reduce-overhead")
    if variant == "compile_loop_ro":
        # Compile the decoder as a whole so the 10-step DPM loop is unrolled into one
        # graph, removing the ~0.3 ms/step of inter-step orchestration. The encoder is
        # left OUT of cudagraphs here: with both as cudagraph regions, dynamo fails
        # tracing the decoder on the encoder's already-overwritten buffers.
        model.decoder = torch.compile(model.decoder, mode="reduce-overhead")
        model.encoder = torch.compile(model.encoder)
        return model
    if "compile" in variant:
        # The DiT is the module the DPM solver calls 10x per inference, so it is
        # where per-step cost lives. ``_sample_x_start`` reads ``self.dit`` on
        # every forward, so replacing the attribute is enough.
        model.decoder.dit = torch.compile(model.decoder.dit, **kw)
    if "_all" in variant:
        # The encoder runs once per inference (~7 ms of ~26 ms) and is untouched
        # by compiling the DiT alone; compile it too to bound what compilation
        # can recover in total.
        compiled_encoder = torch.compile(model.encoder, **kw)
        # max-autotune also turns cudagraphs on, so it needs the same clone.
        model.encoder = _CloneEncoderOutput(compiled_encoder) if mode else compiled_encoder
    return model


def run_sequence(args, a) -> None:
    """Compare K consecutive calls on DIFFERENT inputs, eager vs each other variant.

    The single-input comparison the rest of this script does cannot distinguish "the arithmetic
    is the same" from "the arithmetic is the same the first time". A rollout calls the model
    over a thousand times, on a new input each replan, with other GPU work allocating in
    between -- and that is where the closed-loop metrics were seen to move. This reproduces
    that shape: K distinct inputs, replayed in the same order for every variant, compared call
    by call, with a scratch allocation between calls to stand in for the per-step scorers.
    """
    k = a.sequence
    n = a.neighbors[0]
    print(f"\n=== sequence check: {k} consecutive calls, n_neighbors={n} ===")

    def run(variant: str) -> list[torch.Tensor]:
        torch.set_float32_matmul_precision("high" if "tf32" in variant else "highest")
        model = build_model(args, variant, a.device, seed=a.seed)
        out = []
        with torch.no_grad():
            for i in range(k):
                torch.manual_seed(a.seed + 1000 + i)
                data = build_inputs(a.batch, n, a.device)
                torch.compiler.cudagraph_mark_step_begin()
                torch.manual_seed(a.seed)
                _, o = model(data)
                out.append(o["prediction"].detach().float().cpu().clone())
                # Stand-in for the per-step scorers: allocate and free between inferences, so
                # the caching allocator is exercised the way it is during a rollout.
                scratch = torch.randn(1 << 20, device=a.device)
                del scratch
        del model
        torch.cuda.empty_cache()
        return out

    ref = run(a.variants[0])
    for variant in a.variants[1:]:
        got = run(variant)
        first_bad = None
        worst = 0.0
        for i, (x, y) in enumerate(zip(ref, got)):
            if not torch.equal(x, y):
                first_bad = i if first_bad is None else first_bad
                worst = max(worst, float((x - y).abs().max()))
        if first_bad is None:
            print(f"  {variant:14s} all {k} calls BIT-IDENTICAL")
        else:
            n_bad = sum(1 for x, y in zip(ref, got) if not torch.equal(x, y))
            print(
                f"  {variant:14s} first differing call: #{first_bad} of {k} "
                f"({n_bad} differ)  worst max|d|={worst:.3e}"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    # Required rather than defaulted: the production args.json lives outside the repo
    # and its path differs per machine.
    p.add_argument("--args_json", required=True, help="model config (args.json)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--compile_warmup",
        type=int,
        default=25,
        help="warmup iters for compiled variants (first calls trigger compilation)",
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--neighbors", type=int, nargs="+", default=[1, 32, 320])
    p.add_argument(
        "--variants",
        nargs="+",
        default=["eager"],
        choices=VARIANTS,
        help="eager | compile (torch.compile the DiT) | bf16 (autocast) | compile_bf16",
    )
    p.add_argument("--seed", type=int, default=1234, help="weights + inputs, so outputs compare")
    p.add_argument(
        "--sequence",
        type=int,
        default=0,
        help="compare K CONSECUTIVE calls on DIFFERENT inputs, the way a rollout calls the model "
        "(one comparison on one input cannot see a difference that only appears once a graph has "
        "been replayed, or once other work has allocated between replays)",
    )
    p.add_argument("--json_out", default=None)
    a = p.parse_args()

    from diffusion_planner.utils.config import Config

    args = Config(a.args_json)

    others = check_gpu_exclusive()
    if others:
        print("!" * 78)
        print("WARNING: other compute processes are resident on this GPU. Timings below are")
        print("contended and NOT comparable to an idle-GPU baseline. Stop them or discard:")
        for ln in others:
            print(f"  {ln}")
        print("!" * 78)

    print(f"torch {torch.__version__} / {torch.cuda.get_device_name(0)}")

    if a.sequence:
        run_sequence(args, a)
        return

    results: dict[str, dict] = {}
    reference: dict[int, torch.Tensor] = {}
    equality: dict[str, dict] = {v: {} for v in a.variants}
    for variant in a.variants:
        # TF32 is a global matmul-precision switch, so set it per variant rather
        # than leaking the previous variant's setting into the next one.
        torch.set_float32_matmul_precision("high" if "tf32" in variant else "highest")
        model = build_model(args, variant, a.device, seed=a.seed)
        bf16 = "bf16" in variant
        warmup = a.compile_warmup if "compile" in variant else a.warmup
        if variant == a.variants[0]:
            n_params = sum(p.numel() for p in model.parameters())
            print(
                f"params {n_params / 1e6:.2f}M  hidden_dim={args.hidden_dim} "
                f"decoder_depth={args.decoder_depth} "
                f"predicted_neighbor_num={args.predicted_neighbor_num} future_len={args.future_len}"
            )
        print(
            f"\n=== variant: {variant} (warmup {warmup}, autocast bf16={bf16}, "
            f"fp32_matmul={torch.get_float32_matmul_precision()}) ==="
        )

        results[variant] = {}
        with torch.no_grad():
            for n in a.neighbors:
                torch.manual_seed(a.seed + n)
                data = build_inputs(a.batch, n, a.device)
                ctx = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if bf16
                    else contextlib.nullcontext()
                )

                def full():
                    # Same contract as the rollout: one inference == one cudagraph step.
                    torch.compiler.cudagraph_mark_step_begin()
                    with ctx:
                        model(data)

                def encoder_only():
                    with ctx:
                        model.encoder(data)

                torch.cuda.synchronize()
                t_warm = time.perf_counter()
                for _ in range(warmup):
                    full()
                torch.cuda.synchronize()
                warmup_s = time.perf_counter() - t_warm

                # Random weights x random inputs can diverge. The DPM loop is a fixed
                # 10 steps with no early exit, so NaNs would not change the timing --
                # but assert finiteness anyway so a reader does not have to re-derive
                # that argument to trust the numbers.
                torch.compiler.cudagraph_mark_step_begin()
                torch.manual_seed(a.seed)  # the sampler draws noise; pin it to compare outputs
                with ctx:
                    _, out = model(data)
                assert torch.isfinite(out["prediction"]).all(), (
                    f"non-finite prediction at n_neighbors={n}; timing is still "
                    "shape-determined but investigate before trusting the model output"
                )
                # Same weights, same inputs, same sampler noise -- so any difference here is the
                # variant's arithmetic, which is what decides whether it can be adopted without
                # re-baselining every metric.
                pred = out["prediction"].detach().float().cpu().clone()
                if variant == a.variants[0]:
                    reference[n] = pred
                    equality[variant][n] = {"bit_identical": True, "max_abs_diff": 0.0}
                else:
                    ref = reference.get(n)
                    same = ref is not None and torch.equal(ref, pred)
                    mad = float((ref - pred).abs().max()) if ref is not None else float("nan")
                    equality[variant][n] = {"bit_identical": bool(same), "max_abs_diff": mad}
                    print(
                        f"n_neighbors={n:4d}  vs {a.variants[0]}: "
                        f"{'BIT-IDENTICAL' if same else f'DIFFERS  max|d|={mad:.3e}'}"
                    )

                full_ms = time_synced(full, a.iters)
                enc_ms = time_synced(encoder_only, a.iters)

                # Unsynced loop timing: what a timer without synchronize() reports.
                # In a tight loop this converges to true throughput via queue
                # backpressure, so the interesting artifact is the FIRST call.
                t0 = time.perf_counter()
                full()
                first_unsynced_ms = (time.perf_counter() - t0) * 1e3
                torch.cuda.synchronize()

                unsynced = []
                for _ in range(a.iters):
                    t0 = time.perf_counter()
                    full()
                    unsynced.append((time.perf_counter() - t0) * 1e3)

                med_full = statistics.median(full_ms)
                med_enc = statistics.median(enc_ms)
                results[variant][n] = {
                    "full_median_ms": med_full,
                    "full_mean_ms": statistics.mean(full_ms),
                    "full_min_ms": min(full_ms),
                    "full_max_ms": max(full_ms),
                    "encoder_median_ms": med_enc,
                    "decoder_loop_median_ms": med_full - med_enc,
                    "unsynced_loop_median_ms": statistics.median(unsynced),
                    "unsynced_first_call_ms": first_unsynced_ms,
                    "warmup_total_s": warmup_s,
                }
                r = results[variant][n]
                print(
                    f"n_neighbors={n:4d}  full {med_full:7.2f} ms "
                    f"(mean {r['full_mean_ms']:.2f} / min {r['full_min_ms']:.2f} / max {r['full_max_ms']:.2f})\n"
                    f"              encoder {med_enc:6.2f} ms  decoder(10 DPM steps) "
                    f"{r['decoder_loop_median_ms']:6.2f} ms "
                    f"({r['decoder_loop_median_ms'] / 10:.2f} ms/step)\n"
                    f"              unsynced: loop {r['unsynced_loop_median_ms']:.2f} ms / "
                    f"first call {first_unsynced_ms:.2f} ms  |  warmup {warmup_s:.1f} s"
                )
        del model
        torch.cuda.empty_cache()

    if len(a.variants) > 1:
        base = a.variants[0]
        print(f"\n=== speedup vs {base} (full, median) ===")
        for variant in a.variants[1:]:
            for n in a.neighbors:
                b = results[base][n]["full_median_ms"]
                v = results[variant][n]["full_median_ms"]
                print(f"  {variant:14s} n={n:4d}: {b:6.2f} -> {v:6.2f} ms  ({b / v:.2f}x)")

    if len(a.variants) > 1:
        print(f"\n=== output vs {a.variants[0]} (same weights, inputs and sampler noise) ===")
        for variant in a.variants[1:]:
            for n, e in equality[variant].items():
                verdict = (
                    "BIT-IDENTICAL" if e["bit_identical"] else f"max|d|={e['max_abs_diff']:.3e}"
                )
                print(f"  {variant:14s} n={n:4d}: {verdict}")

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "batch": a.batch,
                    "iters": a.iters,
                    "weights": "random-init (architecture from production args.json)",
                    "gpu_contention": others,
                    "variants": a.variants,
                    "results": results,
                    "equality_vs_first_variant": equality,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
