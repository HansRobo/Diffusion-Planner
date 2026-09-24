#!/usr/bin/env bash
# Drive the scenario_sim suite through scenario_sim_pool: JOBS persistent workers, each running
# case after case for the whole run.
#
# The pooled counterpart of the per-case driver, which spawns one process per case. Everything
# around the launch is kept identical to that one -- the same suite resolution, the same work list
# and ordering, the same MPS setup, the same aggregation and the same viewer export -- so the two
# differ by how often the model, the CUDA context and the parsed maps are paid for, and nothing
# else.
#
# It satisfies the training hook's driver contract (``--scenario_sim_driver``): the hook passes
# the epoch's ``best_model.pth`` as CKPT, and the ONNX graph exported next to it just before is
# what this runs. A missing graph is fatal rather than a fall back to the checkpoint, so a failed
# export cannot turn into a slower run that reads as this one.
#
#   env: CKPT OUT OPENSCENARIOS_BASE SUITE SCENARIO_ROOT GPUS JOBS_PER_GPU MAX_STEPS
#        REPLAN_INTERVAL DRAW_EVERY MAX_CASES CASE_TIMEOUT USE_MPS CLAIM_DIR MAX_FAILED
#        ROS_DOMAIN_SPREAD EXPORT DELIVERY_ROOT RUN_NAME PYTHON SCENARIO_SIM_MODEL
#        SCENARIO_SIM_ORT_EP SCENARIO_SIM_TRT_CACHE SCENARIO_SIM_ORT_INTRA
#        CLOSED_LOOP_PNG_COMPRESS_LEVEL
set -uo pipefail

OPENSCENARIOS_BASE="${OPENSCENARIOS_BASE:-/mnt/storage_rdma/diffusion_planner/openscenarios}"

if [ -n "${PYTHON:-}" ]; then
  :
elif [ -x "$OPENSCENARIOS_BASE/.venv/bin/python3" ]; then
  PYTHON="$OPENSCENARIOS_BASE/.venv/bin/python3"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
  PYTHON="$VIRTUAL_ENV/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  PYTHON="python"
fi
export PYTHON

SUITE="${SUITE:-}"
SCENARIO_ROOT="${SCENARIO_ROOT:-$OPENSCENARIOS_BASE/scenarios}"
if [ -n "$SUITE" ]; then
  if [ -d "$SUITE" ]; then
    SCENARIO_ROOT="$SUITE"
  elif [ -d "$OPENSCENARIOS_BASE/suites/$SUITE/scenarios" ]; then
    SCENARIO_ROOT="$OPENSCENARIOS_BASE/suites/$SUITE/scenarios"
  elif [ -d "$OPENSCENARIOS_BASE/suites/$SUITE" ]; then
    SCENARIO_ROOT="$OPENSCENARIOS_BASE/suites/$SUITE"
  else
    echo "[pool] FATAL: cannot resolve SUITE=$SUITE under $OPENSCENARIOS_BASE" >&2
    exit 2
  fi
fi

[ -n "${CKPT:-}" ] || { echo "[pool] ERROR: CKPT is required." >&2; exit 1; }
[ -n "${OUT:-}" ] || { echo "[pool] ERROR: OUT is required." >&2; exit 1; }
[ -r "$CKPT" ] || { echo "[pool] FATAL: cannot read CKPT=$CKPT" >&2; exit 1; }

# onnx (default): run the exported graph. A .pth CKPT names the directory the hook exported it
# into. pth: run the checkpoint through torch, one model per worker all the same.
SCENARIO_SIM_MODEL="${SCENARIO_SIM_MODEL:-onnx}"
case "$SCENARIO_SIM_MODEL:$CKPT" in
  onnx:*.onnx) MODEL="$CKPT" ;;
  onnx:*) MODEL="$(dirname "$CKPT")/diffusion_planner.onnx" ;;
  pth:*.pth) MODEL="$CKPT" ;;
  *) echo "[pool] FATAL: SCENARIO_SIM_MODEL=$SCENARIO_SIM_MODEL does not fit CKPT=$CKPT" >&2; exit 2 ;;
esac
[ -r "$MODEL" ] || { echo "[pool] FATAL: cannot read the model to run: $MODEL" >&2; exit 1; }

# The training hook names every run directory `scenario_sim`, so a basename is not a name there:
# every epoch of every training run would be delivered over the same one.
default_run_name() {
  case "$(basename "$1")" in
    scenario_sim)
      name="$(id -un)_$(basename "$(dirname "$(dirname "$1")")")_$(basename "$(dirname "$1")")"
      [ -n "${SLURM_JOB_ID:-}" ] && name="$name-$SLURM_JOB_ID"
      printf '%s' "$name" ;;
    *) basename "$1" ;;
  esac
}
EXPORT=${EXPORT:-1}
case "$EXPORT" in 0|1) ;; *) echo "[pool] FATAL: EXPORT must be 0 or 1"; exit 2 ;; esac
DELIVERY_ROOT=${DELIVERY_ROOT:-/mnt/storage_rdma/diffusion_planner/validation_result/closed_loop_scenario}
RUN_NAME=${RUN_NAME:-$(default_run_name "$OUT")}
DELIVERY_TARGET="$DELIVERY_ROOT/$RUN_NAME"

# Without a scheduler nothing hands us a GPU count, so count what this process can see.
visible_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | grep -c .
  else
    nvidia-smi -L 2>/dev/null | grep -c '^GPU'
  fi
}
# Distinct physical cores in this process's CPU affinity. Hyperthread siblings share a core,
# and the rollout is bound by physical cores, so counting threads overstates the capacity.
physical_cores() {
  "$PYTHON" -c '
import os
seen = set()
for c in os.sched_getaffinity(0):
    t = f"/sys/devices/system/cpu/cpu{c}/topology"
    try:
        seen.add((open(f"{t}/physical_package_id").read(), open(f"{t}/core_id").read()))
    except OSError:
        seen.add(("", str(c)))
print(len(seen))'
}
GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-$(visible_gpus)}}"
[ "${GPUS:-0}" -ge 1 ] 2>/dev/null || { echo "[pool] FATAL: no GPU count (set GPUS)" >&2; exit 2; }
# The suite's wall is U-shaped in the slot count and bottoms out near 0.85 slots per physical
# core, independent of the host: fewer slots leave the GPUs idle, more starve the simulators
# below 10 Hz until cases time out.
if [ -z "${JOBS_PER_GPU:-}" ]; then
  JOBS_PER_GPU=$(( ($(physical_cores) * 85 + GPUS * 50) / (GPUS * 100) ))
  [ "$JOBS_PER_GPU" -ge 1 ] || JOBS_PER_GPU=1
fi
JOBS="$((GPUS * JOBS_PER_GPU))"
MAX_STEPS="${MAX_STEPS:-3000}"
REPLAN_INTERVAL="${REPLAN_INTERVAL:-}"
DRAW_EVERY="${DRAW_EVERY:-}"
MAX_CASES="${MAX_CASES:-}"
CASE_TIMEOUT="${CASE_TIMEOUT:-1800}"
MAX_FAILED="${MAX_FAILED:-}"
ROS_DOMAIN_SPREAD="${ROS_DOMAIN_SPREAD:-1}"
USE_MPS="${USE_MPS:-1}"

# TensorRT by default for a graph: it is what the vehicle runs. Exported for this script's children
# only, so a closed-loop step in the same process tree keeps whatever it had.
case "$MODEL" in
  *.onnx) export SCENARIO_SIM_ORT_EP="${SCENARIO_SIM_ORT_EP:-trt}" ;;
esac
export SCENARIO_SIM_TRT_CACHE="${SCENARIO_SIM_TRT_CACHE:-/var/tmp/${USER:-u}/trt_engine_cache}"

mkdir -p "$OUT"

if [ -f "/opt/ros/humble/setup.bash" ]; then
  set +u; source /opt/ros/humble/setup.bash; set -u
fi
if [ -f "$OPENSCENARIOS_BASE/install/setup.bash" ]; then
  set +u; source "$OPENSCENARIOS_BASE/install/setup.bash"; set -u
fi
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

{
  echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  $(date -Is)"
  echo "driver=pool  ckpt=$CKPT  model=$MODEL"
  echo "export=$EXPORT delivery_target=${DELIVERY_TARGET:-off}"
  echo "suite=${SUITE:-none}  scenario_root=$SCENARIO_ROOT"
  echo "cases=$(find "$SCENARIO_ROOT" -name '*.xosc' 2>/dev/null | wc -l)"
  echo "jobs=$JOBS jobs_per_gpu=$JOBS_PER_GPU gpus=$GPUS max_steps=$MAX_STEPS"
  echo "draw_every=${DRAW_EVERY:-off} max_cases=${MAX_CASES:-all} use_mps=$USE_MPS case_timeout=${CASE_TIMEOUT}s"
  echo "ort_ep=${SCENARIO_SIM_ORT_EP:-<n/a>} ort_intra=${SCENARIO_SIM_ORT_INTRA:-1} trt_cache=$SCENARIO_SIM_TRT_CACHE"
  echo "png_compress_level=${CLOSED_LOOP_PNG_COMPRESS_LEVEL:-default}"
} | tee "$OUT/run_context.txt"

if [ "$USE_MPS" = 1 ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/var/tmp/${USER:-default}/mps_pipe_${SLURM_JOB_ID:-$$}
  export CUDA_MPS_LOG_DIRECTORY="$OUT/mps_log"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if ! echo get_default_active_thread_percentage | nvidia-cuda-mps-control >/dev/null 2>&1; then
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY"/* 2>/dev/null || true
    nvidia-cuda-mps-control -d || true
    sleep 0.5
  fi
  if echo get_default_active_thread_percentage | nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "[pool] MPS daemon ready (pipe=$CUDA_MPS_PIPE_DIRECTORY)"
  else
    echo "[pool] WARN: MPS daemon unreachable at $CUDA_MPS_PIPE_DIRECTORY; continuing without MPS"
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  fi
  trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true; rm -rf "$CUDA_MPS_PIPE_DIRECTORY"/* 2>/dev/null || true' EXIT
fi

if ! "$PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "[pool] FATAL: torch.cuda.is_available() is False" >&2
  exit 3
fi

# Same work list, built the same way, so the two arms run the same cases in the same order.
find "$SCENARIO_ROOT" -name '*.xosc' | sort | awk -v r="$SCENARIO_ROOT/" '{p=$0; sub(r,"",p); print NR-1"\t"p}' > "$OUT/work.tsv"
[ -n "$MAX_CASES" ] && { n=$(wc -l < "$OUT/work.tsv"); awk -v n="$n" -v m="$MAX_CASES" 'NR % int((n+m-1)/m) == 1' "$OUT/work.tsv" | head -"$MAX_CASES" > "$OUT/work.sub" && mv "$OUT/work.sub" "$OUT/work.tsv"; }

"$PYTHON" - "$OUT" "$SCENARIO_ROOT" <<'PY'
import json, os, sys
out, root = sys.argv[1], sys.argv[2]
work = []
for line in open(os.path.join(out, "work.tsv")):
    rel = line.rstrip("\n").split("\t", 1)[-1]
    if not rel:
        continue
    key = rel[:-5] if rel.endswith(".xosc") else rel
    work.append([os.path.join(out, key.replace("/", "_")), os.path.join(root, rel)])
json.dump(work, open(os.path.join(out, "work.json"), "w"))
PY

# Claiming is the mechanism here, not an option: workers take work rather than being handed it,
# so a run without a claim directory would have every worker run every case.
CLAIM_DIR=${CLAIM_DIR:-$OUT/claims}
mkdir -p "$CLAIM_DIR"
echo "[pool] $(wc -l < "$OUT/work.tsv") cases over $JOBS persistent workers, claiming under $CLAIM_DIR"

# Build the TensorRT engine once, here, before any worker starts. Every worker on this host wants
# the same engine, and left to themselves all $JOBS of them would build it at once, each paying
# the build and competing for the memory to do it. Fatal on anything but a registered TensorRT
# provider: a prewarm that quietly lands on CUDA leaves a cold cache and hands the whole pool the
# stampede this exists to prevent.
if [ "${SCENARIO_SIM_ORT_EP:-}" = trt ]; then
  echo "[pool] warming the TensorRT engine cache under $SCENARIO_SIM_TRT_CACHE"
  t_warm=$(date +%s)
  # torch first: onnxruntime's GPU providers dlopen the CUDA libraries that live in the venv's
  # nvidia/* wheels, which are not on the loader path until torch has pulled them in.
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" - "$MODEL" <<'PY' || {
import os, sys
import torch  # noqa: F401
from scenario_generation.simulate import load_onnx_model, trt_engine_cache_dir

onnx = sys.argv[1]
print("engine cache:", trt_engine_cache_dir(os.environ["SCENARIO_SIM_TRT_CACHE"], onnx, 0),
      flush=True)
# load_onnx_model raises unless TensorRT actually registered, which is the check this needs.
model, _ = load_onnx_model(onnx, "cuda")
print("providers:", model.session.get_providers())
# The engine is built on the first run, not when the session opens.
model.warm_up()
PY
    echo "[pool] FATAL: TensorRT prewarm failed; not starting $JOBS workers on a cold cache" >&2
    exit 3
  }
  echo "[pool] engine cache warm after $(( $(date +%s) - t_warm ))s"
fi

echo "### SUITE $(date -Is)"
t0=$(date +%s)
# Threads per worker: the workers ARE the parallelism, each wants a slice rather than the whole
# machine. Same split the per-case arm makes.
tpw=$(( $(nproc) / JOBS )); [ "$tpw" -lt 1 ] && tpw=1
export OMP_NUM_THREADS=$tpw MKL_NUM_THREADS=$tpw OPENBLAS_NUM_THREADS=$tpw TORCHINDUCTOR_COMPILE_THREADS=1

worker_pids=()
for slot in $(seq 0 $((JOBS - 1))); do
  gpu=$(( slot % GPUS ))
  # The GPU is bound to the slot for the worker's whole life, which is what makes one context per
  # worker enough. One ROS domain per slot, for the reason the per-case driver spreads them:
  # participants in one domain all discover each other, so N of them cost N^2 of discovery.
  # env, not a bare assignment prefix: bash decides what is an assignment before it expands, so
  # an expanded ROS_DOMAIN_ID=... would be taken for the command name.
  slot_env=(CUDA_VISIBLE_DEVICES=$gpu)
  [ "$ROS_DOMAIN_SPREAD" = 1 ] && slot_env+=("ROS_DOMAIN_ID=$(( slot % 101 ))")
  env "${slot_env[@]}" \
  "$PYTHON" -m scenario_generation.scenario_sim_pool \
    --work_list "$OUT/work.json" --claim_dir "$CLAIM_DIR" --run_dir "$OUT" \
    --model_path "$MODEL" --device cuda --max_steps "$MAX_STEPS" \
    ${REPLAN_INTERVAL:+--replan_interval "$REPLAN_INTERVAL"} \
    ${DRAW_EVERY:+--draw_every "$DRAW_EVERY"} \
    --watchdog_sec "$CASE_TIMEOUT" --slot "$slot" --gpu "$gpu" \
    > "$OUT/pool_$slot.log" 2>&1 &
  worker_pids+=($!)
done
wait "${worker_pids[@]}"
echo
echo "### DONE wall=$(( $(date +%s) - t0 ))s  $(date -Is)"

# Aggregation, byte for byte what the per-case driver runs, so the two arms' summary.json are
# produced by the same code and not merely by the same intent.
MAX_FAILED_ENV=$MAX_FAILED "$PYTHON" - "$OUT" <<'PY'
import glob, json, os, sys
from scenario_generation.closed_loop_eval import aggregate, segment_row_for_json
out = sys.argv[1]
rows = []
for rp in sorted(glob.glob(os.path.join(out, "*", "row.json"))):
    try:
        r = json.load(open(rp))
        rows.append(segment_row_for_json(r))
    except Exception:
        pass
with open(os.path.join(out, "segments.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r, default=float) + "\n")
s = aggregate(rows, 1.0)
s["n_scenarios"] = len(rows)
json.dump({k: v for k, v in s.items() if k != "segments"}, open(os.path.join(out, "summary.json"), "w"),
          indent=2, default=float)
submitted = sum(1 for _ in open(os.path.join(out, "work.tsv")))
rj = os.path.join(out, "rejected.txt")
rejected = sum(1 for _ in open(rj)) if os.path.exists(rj) else 0
failed = submitted - len(rows) - rejected
print(f"rows={len(rows)}/{submitted}  failed={failed}  rejected={rejected}"
      f"  steps={s.get('total_steps')}  terminated={s.get('terminated_counts')}")

if not rows:
    print("[suite] FAIL: no case produced a row", file=sys.stderr)
    sys.exit(4)
limit = os.environ.get("MAX_FAILED_ENV", "")
if limit != "" and failed > int(limit):
    print(f"[suite] FAIL: {failed} of {submitted} cases failed, limit {limit}", file=sys.stderr)
    sys.exit(5)
PY
aggregate_rc=$?
[ "$aggregate_rc" = 0 ] || exit "$aggregate_rc"

if [ "$EXPORT" = 1 ]; then
  echo "[pool] exporting viewer dataset to $DELIVERY_TARGET" | tee -a "$OUT/run_context.txt"
  if ! env MPLBACKEND=Agg "$PYTHON" -m scenario_generation.scenario_sim_viewer_export \
      --run_dir "$OUT" --out_root "$DELIVERY_TARGET"; then
    echo "[pool] FAIL: viewer export failed" | tee -a "$OUT/run_context.txt"
    exit 6
  fi
  echo "[pool] export complete: $DELIVERY_TARGET" | tee -a "$OUT/run_context.txt"
fi
