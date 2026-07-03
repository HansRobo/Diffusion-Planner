# Latent OOD Stage 1.5 — Feature Formulation Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate two OOD feature formulations (raw kNN distance vs maneuver-conditioned residual) alongside EPDMS subscores in a LightGBM classifier for override detection, measured by AUPRC and per-category F1.

**Architecture:** Convert all 74 override bags on NAS to npz, pull maneuver-matched normal npz from sakurab, score everything against an enlarged kNN bank, join with existing EPDMS subscores, then train/evaluate LightGBM under 4 feature configurations (EPDMS-only baseline, +raw OOD, +residual OOD, +both). 5-fold CV stratified by bag.

**Tech Stack:** Python, onnxruntime, lightgbm, sklearn, numpy, matplotlib, scipy

## Global Constraints

- `/mnt/nas` is **read-only** — never write there.
- sakurab SSH via `BatchMode=yes` with key `id_ed25519_sakuraDatacentric`.
- Check `df -h /home/chenglin` before large operations; maintain > 50GB free.
- Model: `/opt/autoware/mlmodels/diffusion_planner_for_x2/diffusion_planner.onnx` and `args.json` (ONNX only).
- All Python via `uv run` from DP repo root (`/home/chenglin/workspace/Diffusion-Planner`).
- `parse_rosbag.py` requires `PYTHONPATH=$PWD/diffusion_planner_ros:$PYTHONPATH` and `--min_frames 0`.
- ROS2 scripts require `source external/pilot-auto.x2/install/setup.bash` (from the risk-scene repo, not `/opt/ros/humble`).
- Branch: `feat/latent-ood-stage1` in Diffusion-Planner repo.
- NEVER commit directly to `tier4-main`.
- Override bags: 75 total (15 takanawa + 60 odaiba), **not** 174.
- Bags use 5 distinct map versions. Each needs its own extracted `.osm` map:
  - takanawa v23: 15 bags
  - odaiba v521: 21 bags
  - odaiba v531: 9 bags
  - odaiba v532: 26 bags
  - odaiba v544: 4 bags
- Version-to-bag mapping saved at `data/bag_map_versions.json`.
- Takanawa v23 map already exists: `/home/chenglin/autoware_map/takanawa/lanelet2_map_v23.osm`.
- EPDMS CSV: `/home/chenglin/workspace/at-team-tools/lin/diffusion-planner-risk-scene/results/analysis/samples_epdms_all.csv` (86,463 frames, 157 bags). Only 68 of these overlap with bags on NAS.
- Override transitions: `/home/chenglin/workspace/at-team-tools/lin/diffusion-planner-risk-scene/results/override_transitions.json` (98 OR events).
- SFT annotations: `/home/chenglin/workspace/Diffusion-Planner-Meta-Repository/dataset/sft_filtering_json/x2_dev/2355_Takanawa_gateway_copied_from_Aisantec/` (2,411 scenes with driving_decisions).
- Binary map extraction from bags requires: `lanelet2.io.loadRobust(bin_path, MGRSProjector(Origin(lat, lon)))` with `proj.setMGRSCode('54SUE')`, then `lanelet2.io.writeRobust(osm_path, lmap, proj)`.

---

### Task 1: Extract Maps and Convert All Override Bags to NPZ

**Files:**
- Create: `scripts/convert_all_or_bags.py` (full batch converter with map version handling)
- Use: `ros_scripts/parse_rosbag.py` (existing)

**Interfaces:**
- Consumes: `.db3` bags from NAS (read-only), binary maps from `/map/vector_map` topic
- Produces: `data/or_scene_npz/<area>/<category>/<scene>/*.npz` + companion `.json` files, `data/or_scene_npz/path_list_all_override.json`

- [ ] **Step 1.1: Check storage**

```bash
df -h /home/chenglin | tail -1
# Need ~10GB for ~35k npz (74 bags × ~480 frames × ~150KB)
# Ensure >=60GB free
```

- [ ] **Step 1.2: Write the batch converter script**

Create `scripts/convert_all_or_bags.py`:

```python
#!/usr/bin/env python3
"""Convert all ORScene bags to npz with automatic per-bag map extraction.

For each bag:
1. Read /map/vector_map binary from the bag
2. Deserialize with lanelet2.io.loadRobust + MGRSProjector
3. Write to a cached .osm keyed by (area, map_version)
4. Run parse_rosbag.py with the matched .osm

Usage:
    source external/pilot-auto.x2/install/setup.bash
    PYTHONPATH=$PWD/diffusion_planner_ros:$PYTHONPATH \
      uv run python scripts/convert_all_or_bags.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import lanelet2
import rosbag2_py
from autoware_lanelet2_extension_python.projection import MGRSProjector
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

ORSCENE_ROOT = Path("/mnt/nas/private_workspace/chenglin/ORScene_bags")
OUTPUT_ROOT = Path("data/or_scene_npz")
MAP_CACHE_DIR = Path("/home/chenglin/autoware_map/extracted")
PARSE_SCRIPT = Path("ros_scripts/parse_rosbag.py")

MGRS_ORIGINS = {
    "takanawa": (35.63, 139.74),
    "odaiba": (35.62, 139.77),
}
MGRS_CODE = "54SUE"


def extract_map_from_bag(bag_path: Path) -> tuple[int, bytes]:
    """Read /map/vector_map from bag, return (version, binary_data)."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=""),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/map/vector_map"]))

    if reader.has_next():
        _, data, _ = reader.read_next()
        msg_type = get_message(type_map["/map/vector_map"])
        msg = deserialize_message(data, msg_type)
        return msg.version_map, bytes(msg.data)

    raise RuntimeError(f"No /map/vector_map in {bag_path}")


def get_or_create_osm(area: str, version: int, binary_data: bytes) -> Path:
    """Get cached .osm for (area, version), or create from binary."""
    osm_path = MAP_CACHE_DIR / f"{area}_v{version}.osm"
    if osm_path.exists():
        return osm_path

    lat, lon = MGRS_ORIGINS.get(area, (35.63, 139.74))
    proj = MGRSProjector(lanelet2.io.Origin(lat, lon))
    proj.setMGRSCode(MGRS_CODE)

    bin_path = MAP_CACHE_DIR / f"{area}_v{version}.bin"
    MAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bin_path.write_bytes(binary_data)

    lmap, errors = lanelet2.io.loadRobust(str(bin_path), proj)
    print(f"  Map {area} v{version}: {len(list(lmap.laneletLayer))} lanelets, {len(errors)} load errors")

    lanelet2.io.writeRobust(str(osm_path), lmap, proj)
    bin_path.unlink()
    print(f"  Wrote: {osm_path}")
    return osm_path


def convert_bag(bag_path: Path, osm_path: Path, save_dir: Path) -> int:
    """Run parse_rosbag.py and return number of npz produced."""
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(PARSE_SCRIPT),
        str(bag_path), str(osm_path), str(save_dir),
        "--step", "1", "--min_frames", "0",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd() / 'diffusion_planner_ros'}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        return 0

    npz_count = len(list(save_dir.glob("*.npz")))
    return npz_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-bags", type=int, default=999)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    count = 0
    total_npz = 0
    failed = 0

    for area in ["takanawa", "odaiba"]:
        area_root = ORSCENE_ROOT / area
        if not area_root.exists():
            continue

        for category in sorted(area_root.iterdir()):
            if not category.is_dir():
                continue

            for scene in sorted(category.iterdir()):
                if not scene.is_dir():
                    continue

                for uuid_dir in sorted(scene.iterdir()):
                    if not uuid_dir.is_dir():
                        continue

                    for bag_entry in sorted(uuid_dir.iterdir()):
                        if not bag_entry.name.endswith(".db3") or bag_entry.name.endswith("_0.db3"):
                            continue
                        if not bag_entry.is_dir():
                            continue

                        if count >= args.max_bags:
                            print(f"\nReached --max-bags={args.max_bags}")
                            print(f"Total: {count} bags, {total_npz} npz, {failed} failed")
                            return

                        save_dir = OUTPUT_ROOT / area / category.name / scene.name / bag_entry.stem
                        if args.skip_existing and save_dir.exists() and any(save_dir.glob("*.npz")):
                            existing = len(list(save_dir.glob("*.npz")))
                            print(f"[{count}] SKIP ({existing} npz): {bag_entry.name}")
                            total_npz += existing
                            count += 1
                            continue

                        # Check free space
                        stat = shutil.disk_usage("/home/chenglin")
                        free_gb = stat.free / (1024**3)
                        if free_gb < 50:
                            print(f"ERROR: Only {free_gb:.0f}GB free, stopping")
                            return

                        print(f"[{count}] {area}/{category.name}/{bag_entry.name}")

                        try:
                            version, binary = extract_map_from_bag(bag_entry)
                            osm_path = get_or_create_osm(area, version, binary)
                            n = convert_bag(bag_entry, osm_path, save_dir)
                            total_npz += n
                            print(f"  -> {n} npz")
                        except Exception as e:
                            print(f"  FAILED: {e}")
                            failed += 1

                        count += 1

    print(f"\nDone: {count} bags, {total_npz} npz, {failed} failed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.3: Run the batch conversion**

```bash
source /home/chenglin/workspace/at-team-tools/lin/diffusion-planner-risk-scene/external/pilot-auto.x2/install/setup.bash
cd /home/chenglin/workspace/Diffusion-Planner

PYTHONPATH="$PWD/diffusion_planner_ros:$PYTHONPATH" \
  uv run python scripts/convert_all_or_bags.py 2>&1 | tee data/conversion_log.txt
```

Expected: ~74 bags converted, ~35k npz total. Takes ~2-4 hours.
The script caches extracted maps, so same-version bags reuse the `.osm`.

- [ ] **Step 1.4: Build path list for all override npz**

```bash
uv run python -c "
import json
from pathlib import Path
npz_files = sorted(str(p) for p in Path('data/or_scene_npz').rglob('*.npz'))
with open('data/or_scene_npz/path_list_all_override.json', 'w') as f:
    json.dump(npz_files, f, indent=2)
print(f'Total override npz: {len(npz_files)}')
"
```

- [ ] **Step 1.5: Commit**

```bash
git add scripts/convert_all_or_bags.py
git commit -m "feat: batch converter for all ORScene bags with per-version map extraction"
```

---

### Task 2: Enlarge kNN Bank and Score Everything

**Files:**
- Use: `scripts/build_latent_ood_bank.py` (existing)
- Use: `scripts/score_latent_ood.py` (existing)

**Interfaces:**
- Consumes: Normal npz from sakurab, ONNX model, all override npz (Task 1)
- Produces: `data/latent_ood_bank_5k/`, `data/latent_ood_scores_all.jsonl`

- [ ] **Step 2.1: Pull 5,000 random normal npz from sakurab**

```bash
cd /home/chenglin/workspace/Diffusion-Planner

# Sample 5000 paths (we already have the index)
ssh -o BatchMode=yes sakurab \
  "find /mnt/nvme/dataset/20260425_takanawa_full/x2_dev/2355_Takanawa_gateway_copied_from_Aisantec \
   -name '*.npz' -type f | shuf -n 5000" > /tmp/normal_5k_paths.txt

wc -l /tmp/normal_5k_paths.txt

# Copy
mkdir -p data/normal_npz_5k
rsync -a --files-from=/tmp/normal_5k_paths.txt sakurab:/ data/normal_npz_5k/

# Build path list
uv run python -c "
import json
from pathlib import Path
paths = sorted(str(p) for p in Path('data/normal_npz_5k').rglob('*.npz'))
with open('data/normal_npz_5k/path_list_normal_5k.json', 'w') as f:
    json.dump(paths, f, indent=2)
print(f'Normal npz: {len(paths)}')
"
```

- [ ] **Step 2.2: Build enlarged bank**

```bash
source /home/chenglin/workspace/at-team-tools/lin/diffusion-planner-risk-scene/external/pilot-auto.x2/install/setup.bash
cd /home/chenglin/workspace/Diffusion-Planner

uv run python scripts/build_latent_ood_bank.py \
  --model_path /opt/autoware/mlmodels/diffusion_planner_for_x2/diffusion_planner.onnx \
  --args_path /opt/autoware/mlmodels/diffusion_planner_for_x2/args.json \
  --train_list data/normal_npz_5k/path_list_normal_5k.json \
  --output_dir data/latent_ood_bank_5k \
  --batch_size 16 --device cuda
```

Expected: ~15 minutes for 5k embeddings.

- [ ] **Step 2.3: Score all override npz**

```bash
uv run python scripts/score_latent_ood.py \
  --model_path /opt/autoware/mlmodels/diffusion_planner_for_x2/diffusion_planner.onnx \
  --args_path /opt/autoware/mlmodels/diffusion_planner_for_x2/args.json \
  --eval_list data/or_scene_npz/path_list_all_override.json \
  --bank_dir data/latent_ood_bank_5k \
  --output data/latent_ood_scores_override.jsonl \
  --batch_size 16 --device cuda
```

- [ ] **Step 2.4: Also score maneuver-matched normal npz for residual computation**

Pull 200 npz per maneuver group (stop, avoid, turn, straight) from sakurab
using the SFT annotation matching from Stage 1. Reuse the existing
`data/maneuver_npz_paths.json` and `data/maneuver_npz/` if still present,
or re-run the matching.

```bash
# Score them
uv run python -c "
import json
from pathlib import Path
# Combine existing maneuver npz paths
paths = sorted(str(p) for p in Path('data/maneuver_npz').rglob('*.npz'))
paths += sorted(str(p) for p in Path('data/normal_npz_5k').rglob('*.npz'))
# Deduplicate
paths = sorted(set(paths))
with open('data/path_list_all_normal.json', 'w') as f:
    json.dump(paths, f, indent=2)
print(f'Total normal npz to score: {len(paths)}')
"

uv run python scripts/score_latent_ood.py \
  --model_path /opt/autoware/mlmodels/diffusion_planner_for_x2/diffusion_planner.onnx \
  --args_path /opt/autoware/mlmodels/diffusion_planner_for_x2/args.json \
  --eval_list data/path_list_all_normal.json \
  --bank_dir data/latent_ood_bank_5k \
  --output data/latent_ood_scores_normal.jsonl \
  --batch_size 16 --device cuda
```

---

### Task 3: Build the Joined Feature Matrix

**Files:**
- Create: `scripts/build_feature_matrix.py`

**Interfaces:**
- Consumes: OOD score JSONLs (Task 2), EPDMS CSV, override_transitions.json, SFT annotations, bag directory structure
- Produces: `data/feature_matrix.csv` with columns: `bag, ts_sec, category, maneuver_type, label, nc, dac, ddc, tlc, ttc, lk, hc, ec, ep, epdms, knn_mean, ood_residual, is_override`

- [ ] **Step 3.1: Write the feature matrix builder**

Create `scripts/build_feature_matrix.py`:

```python
#!/usr/bin/env python3
"""Join OOD scores with EPDMS subscores into a single feature matrix.

Joins by bag name + timestamp (50ms tolerance). Labels frames as positive
if within [t-20s, t+10s] of an OR event. Computes ood_residual per
maneuver type.

Usage:
    uv run python scripts/build_feature_matrix.py \
      --ood_override data/latent_ood_scores_override.jsonl \
      --ood_normal data/latent_ood_scores_normal.jsonl \
      --epdms_csv /path/to/samples_epdms_all.csv \
      --or_transitions /path/to/override_transitions.json \
      --output data/feature_matrix.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CATEGORY_TO_MANEUVER = {
    "停止_車両": "stop",
    "停止_信号": "stop",
    "停止_自転車歩行者": "stop",
    "回避_車両": "avoid",
    "回避_路駐車": "avoid",
    "曲がり切れない": "turn",
    "曲がるタイミングが早い": "turn",
    "その他": "other",
}

OR_WINDOW_PRE = 20.0
OR_WINDOW_POST = 10.0
MATCH_TOLERANCE_SEC = 0.05


def load_ood_scores(jsonl_path: Path) -> list[dict]:
    """Load OOD scores with parsed bag name and timestamp."""
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            e = json.loads(line.strip())
            npz_path = e["npz_path"]

            # Extract bag name from npz path
            # Filename format: <bag_name>_<frame_id>.npz
            # The bag name contains the vehicle/date/chunk info
            stem = Path(npz_path).stem
            parts = stem.rsplit("_", 1)
            bag_name = parts[0] if len(parts) == 2 else stem

            # Get timestamp from companion JSON if available
            json_path = Path(npz_path).with_suffix(".json")
            if json_path.exists():
                with open(json_path) as jf:
                    meta = json.load(jf)
                ts_sec = meta["timestamp"] / 1e9
            else:
                ts_sec = None

            entries.append({
                "npz_path": npz_path,
                "bag_name": bag_name,
                "ts_sec": ts_sec,
                "knn_mean": e["knn_mean"],
            })
    return entries


def load_epdms(csv_path: Path) -> dict[str, list[dict]]:
    """Load EPDMS CSV keyed by bag name."""
    by_bag = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            by_bag[row["bag"]].append(row)
    return dict(by_bag)


def load_or_transitions(json_path: Path) -> dict[str, list[float]]:
    with open(json_path) as f:
        return json.load(f)


def is_in_or_window(ts: float, or_times: list[float]) -> bool:
    for ot in or_times:
        if (ot - OR_WINDOW_PRE) <= ts <= (ot + OR_WINDOW_POST):
            return True
    return False


def extract_category_from_path(npz_path: str) -> str:
    """Extract failure_mode category from npz directory path."""
    parts = Path(npz_path).parts
    for part in parts:
        if part in CATEGORY_TO_MANEUVER:
            return part
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood_override", type=Path, required=True)
    parser.add_argument("--ood_normal", type=Path, required=True)
    parser.add_argument("--epdms_csv", type=Path, required=True)
    parser.add_argument("--or_transitions", type=Path, required=True)
    parser.add_argument("--maneuver_npz_paths", type=Path, default=None,
                        help="JSON with maneuver group -> npz path lists")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print("Loading OOD scores...")
    override_ood = load_ood_scores(args.ood_override)
    normal_ood = load_ood_scores(args.ood_normal)
    print(f"  Override: {len(override_ood)}, Normal: {len(normal_ood)}")

    print("Loading EPDMS...")
    epdms = load_epdms(args.epdms_csv)
    print(f"  {len(epdms)} bags, {sum(len(v) for v in epdms.values())} frames")

    print("Loading OR transitions...")
    or_trans = load_or_transitions(args.or_transitions)
    print(f"  {len(or_trans)} bags with OR events")

    # Assign maneuver types to normal scenes
    normal_maneuver = {}
    if args.maneuver_npz_paths and args.maneuver_npz_paths.exists():
        with open(args.maneuver_npz_paths) as f:
            groups = json.load(f)
        for group, paths in groups.items():
            maneuver = group.replace("normal_", "")
            for p in paths:
                normal_maneuver[p] = maneuver

    # Compute per-maneuver median OOD from normal scores
    maneuver_scores = defaultdict(list)
    for e in normal_ood:
        maneuver = normal_maneuver.get(e["npz_path"], "straight")
        maneuver_scores[maneuver].append(e["knn_mean"])

    maneuver_medians = {}
    for m, scores in maneuver_scores.items():
        maneuver_medians[m] = float(np.median(scores))
    print(f"  Maneuver medians: {maneuver_medians}")

    # Join override OOD with EPDMS
    EPDMS_COLS = ["nc", "dac", "ddc", "tlc", "ttc", "lk", "hc", "ec", "ep", "epdms"]
    OUTPUT_COLS = ["bag", "ts_sec", "category", "maneuver_type", "label",
                   *EPDMS_COLS, "knn_mean", "ood_residual", "is_override",
                   "ade_full", "fde_full"]

    rows = []
    matched = 0
    unmatched = 0

    for ood_entry in override_ood:
        bag = ood_entry["bag_name"]
        ts = ood_entry["ts_sec"]
        if ts is None or bag not in epdms:
            unmatched += 1
            continue

        # Find nearest EPDMS frame
        best = None
        best_dt = float("inf")
        for erow in epdms[bag]:
            epdms_ts = float(erow["ts"])
            dt = abs(ts - epdms_ts)
            if dt < best_dt:
                best_dt = dt
                best = erow

        if best is None or best_dt > MATCH_TOLERANCE_SEC:
            unmatched += 1
            continue

        category = extract_category_from_path(ood_entry["npz_path"])
        maneuver = CATEGORY_TO_MANEUVER.get(category, "other")
        or_times = or_trans.get(bag, [])
        label = 1 if is_in_or_window(ts, or_times) else 0
        ood_residual = ood_entry["knn_mean"] - maneuver_medians.get(maneuver, maneuver_medians.get("straight", 0))

        row = {
            "bag": bag,
            "ts_sec": f"{ts:.3f}",
            "category": category,
            "maneuver_type": maneuver,
            "label": label,
            "knn_mean": f"{ood_entry['knn_mean']:.6f}",
            "ood_residual": f"{ood_residual:.6f}",
            "is_override": 1,
            "ade_full": best.get("ade_full", ""),
            "fde_full": best.get("fde_full", ""),
        }
        for col in EPDMS_COLS:
            row[col] = best.get(col, "")

        rows.append(row)
        matched += 1

    print(f"  Override: matched={matched}, unmatched={unmatched}")

    # Add normal frames (label=0, is_override=0)
    for ood_entry in normal_ood:
        maneuver = normal_maneuver.get(ood_entry["npz_path"], "straight")
        ood_residual = ood_entry["knn_mean"] - maneuver_medians.get(maneuver, maneuver_medians.get("straight", 0))

        row = {
            "bag": "normal",
            "ts_sec": f"{ood_entry['ts_sec']:.3f}" if ood_entry["ts_sec"] else "",
            "category": "normal",
            "maneuver_type": maneuver,
            "label": 0,
            "knn_mean": f"{ood_entry['knn_mean']:.6f}",
            "ood_residual": f"{ood_residual:.6f}",
            "is_override": 0,
            "ade_full": "",
            "fde_full": "",
        }
        for col in EPDMS_COLS:
            row[col] = ""

        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.output}")
    print(f"  Override frames: {sum(1 for r in rows if r['is_override'] == 1)}")
    print(f"  Normal frames: {sum(1 for r in rows if r['is_override'] == 0)}")
    print(f"  Positive labels: {sum(1 for r in rows if r['label'] == 1)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Run the feature matrix builder**

```bash
cd /home/chenglin/workspace/Diffusion-Planner

uv run python scripts/build_feature_matrix.py \
  --ood_override data/latent_ood_scores_override.jsonl \
  --ood_normal data/latent_ood_scores_normal.jsonl \
  --epdms_csv /home/chenglin/workspace/at-team-tools/lin/diffusion-planner-risk-scene/results/analysis/samples_epdms_all.csv \
  --or_transitions /home/chenglin/workspace/at-team-tools/lin/diffusion-planner-risk-scene/results/override_transitions.json \
  --maneuver_npz_paths data/maneuver_npz_paths.json \
  --output data/feature_matrix.csv
```

- [ ] **Step 3.3: Verify the matrix**

```bash
uv run python -c "
import csv
with open('data/feature_matrix.csv') as f:
    rows = list(csv.DictReader(f))
print(f'Total rows: {len(rows)}')
print(f'Override: {sum(1 for r in rows if r[\"is_override\"]==\"1\")}')
print(f'Normal: {sum(1 for r in rows if r[\"is_override\"]==\"0\")}')
print(f'Positive (in OR window): {sum(1 for r in rows if r[\"label\"]==\"1\")}')
print(f'Negative: {sum(1 for r in rows if r[\"label\"]==\"0\")}')
# Check EPDMS fill rate for override rows
override_with_epdms = sum(1 for r in rows if r['is_override']=='1' and r['nc'])
print(f'Override rows with EPDMS: {override_with_epdms}')
"
```

- [ ] **Step 3.4: Commit**

```bash
git add scripts/build_feature_matrix.py
git commit -m "feat: join OOD scores with EPDMS into feature matrix for classifier evaluation"
```

---

### Task 4: Train and Evaluate LightGBM Classifier

**Files:**
- Create: `scripts/evaluate_ood_formulations.py`

**Interfaces:**
- Consumes: `data/feature_matrix.csv` (Task 3)
- Produces: `data/evaluation_results/` with AUPRC/F1 tables, per-category breakdown, feature importance plots, comparison figures

- [ ] **Step 4.1: Write the evaluation script**

Create `scripts/evaluate_ood_formulations.py`:

```python
#!/usr/bin/env python3
"""Evaluate OOD feature formulations with LightGBM classifier.

Compares 4 feature sets:
  - Baseline: EPDMS subscores only
  - H-A: EPDMS + raw OOD (knn_mean)
  - H-B: EPDMS + residual OOD (ood_residual)
  - H-AB: EPDMS + knn_mean + ood_residual

5-fold CV stratified by bag. Reports AUPRC, F1, recall@precision.

Usage:
    uv run python scripts/evaluate_ood_formulations.py \
      --input data/feature_matrix.csv \
      --output_dir data/evaluation_results
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import GroupKFold


EPDMS_FEATURES = ["nc", "dac", "ddc", "tlc", "ttc", "lk", "hc", "ec", "ep"]

FEATURE_SETS = {
    "EPDMS-only": EPDMS_FEATURES,
    "H-A (EPDMS+rawOOD)": EPDMS_FEATURES + ["knn_mean"],
    "H-B (EPDMS+residualOOD)": EPDMS_FEATURES + ["ood_residual"],
    "H-AB (EPDMS+both)": EPDMS_FEATURES + ["knn_mean", "ood_residual"],
}


def load_data(csv_path: Path):
    """Load feature matrix, return X dict, y, groups, categories."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row["nc"]:
                continue
            rows.append(row)

    all_features = EPDMS_FEATURES + ["knn_mean", "ood_residual"]
    X = np.zeros((len(rows), len(all_features)), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int32)
    groups = []
    categories = []

    for i, row in enumerate(rows):
        for j, feat in enumerate(all_features):
            val = row.get(feat, "")
            X[i, j] = float(val) if val else 0.0
        y[i] = int(row["label"])
        groups.append(row["bag"])
        categories.append(row["category"])

    feature_idx = {f: i for i, f in enumerate(all_features)}
    return X, y, groups, categories, feature_idx


def evaluate_fold(X_train, y_train, X_test, y_test, feature_cols):
    """Train LightGBM and return predictions."""
    dtrain = lgb.Dataset(X_train[:, feature_cols], label=y_train)

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "is_unbalance": True,
    }

    model = lgb.train(params, dtrain, num_boost_round=200)
    y_pred = model.predict(X_test[:, feature_cols])
    importance = model.feature_importance(importance_type="gain")
    return y_pred, importance


def compute_metrics(y_true, y_pred):
    """Compute AUPRC, best F1, recall@precision thresholds."""
    auprc = average_precision_score(y_true, y_pred)

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

    recall_at_p50 = 0.0
    recall_at_p70 = 0.0
    for p, r in zip(precision, recall):
        if p >= 0.50:
            recall_at_p50 = max(recall_at_p50, r)
        if p >= 0.70:
            recall_at_p70 = max(recall_at_p70, r)

    return {
        "auprc": auprc,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "recall_at_p50": recall_at_p50,
        "recall_at_p70": recall_at_p70,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y, groups, categories, feature_idx = load_data(args.input)
    print(f"  {X.shape[0]} rows, {X.shape[1]} features")
    print(f"  Positive: {y.sum()}, Negative: {(1-y).sum()}")
    print(f"  Unique bags: {len(set(groups))}")

    # 5-fold CV stratified by bag
    unique_bags = list(set(groups))
    bag_to_int = {b: i for i, b in enumerate(unique_bags)}
    group_ids = np.array([bag_to_int[g] for g in groups])

    gkf = GroupKFold(n_splits=args.n_folds)

    results = {}
    all_importances = defaultdict(list)

    for fs_name, fs_features in FEATURE_SETS.items():
        feature_cols = [feature_idx[f] for f in fs_features]
        fold_metrics = []
        fold_predictions = []

        print(f"\n{'='*60}")
        print(f"Evaluating: {fs_name}")
        print(f"  Features: {fs_features}")

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, group_ids)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            y_pred, importance = evaluate_fold(X_train, y_train, X_test, y_test, feature_cols)
            metrics = compute_metrics(y_test, y_pred)
            fold_metrics.append(metrics)

            for fi, feat in enumerate(fs_features):
                all_importances[(fs_name, feat)].append(importance[fi])

            fold_predictions.append((test_idx, y_test, y_pred, [categories[i] for i in test_idx]))

            print(f"  Fold {fold}: AUPRC={metrics['auprc']:.4f}, F1={metrics['best_f1']:.4f}")

        # Aggregate
        avg_metrics = {}
        for key in fold_metrics[0]:
            vals = [m[key] for m in fold_metrics]
            avg_metrics[key] = float(np.mean(vals))
            avg_metrics[f"{key}_std"] = float(np.std(vals))

        results[fs_name] = avg_metrics
        print(f"  Mean AUPRC={avg_metrics['auprc']:.4f} +/- {avg_metrics['auprc_std']:.4f}")
        print(f"  Mean F1={avg_metrics['best_f1']:.4f} +/- {avg_metrics['best_f1_std']:.4f}")

        # Per-category breakdown
        all_test_idx = np.concatenate([fp[0] for fp in fold_predictions])
        all_y_test = np.concatenate([fp[1] for fp in fold_predictions])
        all_y_pred = np.concatenate([fp[2] for fp in fold_predictions])
        all_cats = []
        for fp in fold_predictions:
            all_cats.extend(fp[3])

        print(f"\n  Per-category:")
        cat_results = {}
        for cat in sorted(set(all_cats)):
            mask = np.array([c == cat for c in all_cats])
            if mask.sum() < 10 or all_y_test[mask].sum() == 0:
                continue
            cat_metrics = compute_metrics(all_y_test[mask], all_y_pred[mask])
            cat_results[cat] = cat_metrics
            print(f"    {cat:<25} AUPRC={cat_metrics['auprc']:.4f}, F1={cat_metrics['best_f1']:.4f}, n={mask.sum()}")

        results[fs_name]["per_category"] = cat_results

    # Save results
    with open(args.output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output_dir / 'evaluation_results.json'}")

    # Comparison table
    print(f"\n{'='*80}")
    print(f"{'Feature Set':<30} {'AUPRC':>10} {'F1':>10} {'R@P50':>10} {'R@P70':>10}")
    print(f"{'-'*80}")
    for fs_name in FEATURE_SETS:
        r = results[fs_name]
        print(f"{fs_name:<30} {r['auprc']:>10.4f} {r['best_f1']:>10.4f} "
              f"{r['recall_at_p50']:>10.4f} {r['recall_at_p70']:>10.4f}")

    # Feature importance plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (fs_name, fs_features) in zip(axes.flat, FEATURE_SETS.items()):
        importances = [np.mean(all_importances[(fs_name, f)]) for f in fs_features]
        sorted_idx = np.argsort(importances)
        ax.barh([fs_features[i] for i in sorted_idx], [importances[i] for i in sorted_idx])
        ax.set_title(fs_name)
        ax.set_xlabel("Mean Gain")
    plt.suptitle("Feature Importance by Configuration", fontsize=14)
    plt.tight_layout()
    fig.savefig(args.output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output_dir / 'feature_importance.png'}")

    # AUPRC comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(FEATURE_SETS.keys())
    auprcs = [results[n]["auprc"] for n in names]
    stds = [results[n]["auprc_std"] for n in names]
    bars = ax.bar(names, auprcs, yerr=stds, capsize=5)
    ax.set_ylabel("AUPRC")
    ax.set_title("Override Detection: AUPRC by Feature Set (5-fold CV)")
    for bar, val in zip(bars, auprcs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(args.output_dir / "auprc_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output_dir / 'auprc_comparison.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Install lightgbm if not present**

```bash
cd /home/chenglin/workspace/Diffusion-Planner
uv add lightgbm 2>/dev/null || uv pip install lightgbm
```

- [ ] **Step 4.3: Run the evaluation**

```bash
uv run python scripts/evaluate_ood_formulations.py \
  --input data/feature_matrix.csv \
  --output_dir data/evaluation_results
```

- [ ] **Step 4.4: Commit**

```bash
git add scripts/build_feature_matrix.py scripts/evaluate_ood_formulations.py
git commit -m "feat: LightGBM evaluation of OOD feature formulations vs EPDMS baseline"
```

---

### Task 5: Write Results and Recommendation

**Files:**
- Create: `docs/experiment_results/2026-07-latent-ood-stage1.5-evaluation.md`

**Interfaces:**
- Consumes: All evaluation outputs from Task 4
- Produces: Results document with comparison tables, feature importance, recommendation

- [ ] **Step 5.1: Write results document**

Include:
- Executive summary: which formulation wins?
- Setup: data sizes, feature sets, methodology
- Comparison table: AUPRC, F1, R@P50, R@P70 across all 4 configurations
- Per-category breakdown: where does OOD help? Where doesn't it?
- Feature importance: which features drive the classifier?
- Residual analysis: does maneuver normalization actually help?
- Recommendation: which formulation to use in Stage 2
- Honest limitations

- [ ] **Step 5.2: Commit**

```bash
git add docs/experiment_results/2026-07-latent-ood-stage1.5-evaluation.md
git commit -m "docs: Stage 1.5 evaluation results — OOD formulation comparison"
```
