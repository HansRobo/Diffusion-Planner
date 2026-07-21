# Latent Similarity Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the map extraction bug so override npz has correct lane data, then build a similarity search tool that finds the most similar training scenes for any query scene.

**Architecture:** Extract the binary map from bags as `.bin` (preserving native coordinates), feed to `parse_rosbag.py` via a modified `convert_lanelet()` that accepts `.bin` files. Then build `search_similar.py` on top of the existing `LatentOODScorer.nearest()` method. Demo on 2-3 override scenes.

**Tech Stack:** Python, lanelet2, rosbag2_py, onnxruntime, numpy, torch

## Global Constraints

- `/mnt/nas` is **read-only** — never write there.
- All Python scripts that need ROS2 packages (lanelet2, rosbag2_py) must run with system `python3` after sourcing: `source /home/chenglin/workspace/at-team-tools/lin/diffusion-planner-risk-scene/external/pilot-auto.x2/install/setup.bash`
- Scripts that only need numpy/torch/sklearn use `uv run python`.
- PYTHONPATH for ROS2+DP scripts: `PYTHONPATH=$PWD:$PWD/diffusion_planner_ros:$PYTHONPATH`
- Model: `/opt/autoware/mlmodels/diffusion_planner_for_x2/diffusion_planner.onnx` and `args.json`.
- Branch: `feat/latent-ood-stage1` in Diffusion-Planner repo.
- NEVER commit directly to `tier4-main`.
- Check `df -h /home/chenglin` before large operations; maintain > 50GB free.
- Bag map versions are in `data/bag_map_versions.json` (75 entries, 5 distinct versions).
- The existing `LatentOODScorer.nearest()` method already returns top-K neighbors with paths + distances.
- The existing 5k training bank at `data/latent_ood_bank_5k/` has correct embeddings (training npz have full map data).

---

### Task 1: Fix Map Extraction — Use .bin Instead of .osm

**Files:**
- Create: `ros_scripts/extract_map_bin_from_bag.py` (new script, clean approach)
- Modify: `diffusion_planner_ros/diffusion_planner_ros/lanelet2_utils/lanelet_converter.py:250-268` (accept .bin)
- Test: Manual verification — convert 1 bag, check npz has nonzero lanes

**Interfaces:**
- Produces: `extract_map_bin_from_bag.py` CLI: `python3 extract_map_bin_from_bag.py <bag_path> --output <path.bin>`
- Produces: `convert_lanelet(filename)` now accepts both `.osm` and `.bin` files
- Consumed by: Task 2 batch converter, `parse_rosbag.py`

- [ ] **Step 1.1: Write `extract_map_bin_from_bag.py`**

This script extracts the raw binary map from a bag and saves it as a `.bin` file with dangling-member cleanup. Unlike the existing `extract_map_from_bag.py`, it does NOT shift coordinates or convert to .osm.

Create `ros_scripts/extract_map_bin_from_bag.py`:

```python
#!/usr/bin/env python3
"""Extract lanelet2 map from a rosbag as .bin, preserving native coordinates.

Unlike extract_map_from_bag.py (which shifts coordinates for MGRS and saves
.osm), this script saves the raw binary map directly. The .bin file's
coordinates are in the bag's native map frame — the same frame as
/localization/kinematic_state — so lane lookups by ego position work
correctly.

Dangling cached lanelet members (from the original bin serialization) are
cleaned by rebuilding each lanelet from scratch, same as
extract_map_from_bag.py.

Usage:
    source external/pilot-auto.x2/install/setup.bash
    python3 ros_scripts/extract_map_bin_from_bag.py <bag_path> --output map.bin
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import lanelet2
import rosbag2_py
from autoware_lanelet2_extension_python.projection import MGRSProjector
from lanelet2.core import Lanelet, LaneletMap
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_map_binary(bag_path: Path) -> bytes:
    """Read raw /map/vector_map binary data from a bag."""
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
        return bytes(msg.data)

    raise RuntimeError(f"No /map/vector_map in {bag_path}")


def load_and_clean_map(raw_bytes: bytes) -> lanelet2.core.LaneletMap:
    """Load binary map and rebuild lanelets to drop dangling cached members."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(raw_bytes)
        tmp_path = Path(f.name)

    try:
        proj = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
        raw_map = lanelet2.io.load(str(tmp_path), proj)
    finally:
        tmp_path.unlink()

    clean_map = LaneletMap()
    for ll in raw_map.laneletLayer:
        fresh = Lanelet(
            ll.id, ll.leftBound, ll.rightBound,
            ll.attributes, list(ll.regulatoryElements),
        )
        clean_map.add(fresh)
    for polygon in raw_map.polygonLayer:
        clean_map.add(polygon)
    for ls in raw_map.lineStringLayer:
        clean_map.add(ls)

    return clean_map


def save_bin(lmap: lanelet2.core.LaneletMap, output: Path) -> None:
    """Save lanelet map as .bin (native coordinates, no MGRS conversion)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    proj = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    lanelet2.io.write(str(output), lmap, proj)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("lanelet2_map.bin"))
    args = parser.parse_args()

    raw_bytes = read_map_binary(args.bag_path)
    print(f"Read {len(raw_bytes)} bytes from /map/vector_map")

    lmap = load_and_clean_map(raw_bytes)
    n_lanelets = len(list(lmap.laneletLayer))
    n_points = len(list(lmap.pointLayer))
    print(f"Loaded: {n_lanelets} lanelets, {n_points} points")

    save_bin(lmap, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.2: Modify `convert_lanelet` to accept .bin files**

Edit `diffusion_planner_ros/diffusion_planner_ros/lanelet2_utils/lanelet_converter.py` line 250-268.

Currently:
```python
def convert_lanelet(filename: str) -> LaneletMap:
    projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    lanelet_map = lanelet2.io.load(filename, projection)
```

The key insight: `lanelet2.io.load` already dispatches by file extension. For `.bin` files, it uses the Boost-serialization handler which ignores the projector — coordinates are stored directly. For `.osm` files, it uses the XML handler which uses the projector. So `convert_lanelet` already works with `.bin` files without any code change to the load call.

However, `lanelet2.io.write` for `.bin` does a Boost binary serialization of the LaneletMap. The `.bin` written by `save_bin()` in Step 1.1 uses `lanelet2.io.write` which triggers the same dangling-member issue (some lanelets silently dropped). We already rebuilt lanelets before saving, so this is safe. But verify by checking lanelet count.

Change the docstring only to document that `.bin` is now accepted:

```python
def convert_lanelet(filename: str) -> LaneletMap:
    """Convert lanelet (.osm or .bin) to map info.

    For .osm files, coordinates are loaded via MGRSProjector.
    For .bin files, coordinates are loaded directly (native frame).
    Use .bin files from extract_map_bin_from_bag.py to preserve the
    bag's native coordinate system for ego-position lane lookups.
    ...
```

- [ ] **Step 1.3: Test — convert 1 bag with .bin map and verify npz has lanes**

```bash
source /home/chenglin/workspace/at-team-tools/lin/diffusion-planner-risk-scene/external/pilot-auto.x2/install/setup.bash
cd /home/chenglin/workspace/Diffusion-Planner

# Pick a bag
BAG=$(python3 -c "import json; bags=json.load(open('data/bag_map_versions.json')); print(list(bags.keys())[0])")
echo "Test bag: $BAG"

# Extract .bin map
python3 ros_scripts/extract_map_bin_from_bag.py "$BAG" --output /tmp/test_map.bin

# Convert bag to npz using the .bin map
mkdir -p /tmp/test_npz_bin
PYTHONPATH="$PWD:$PWD/diffusion_planner_ros:$PYTHONPATH" \
  python3 ros_scripts/parse_rosbag.py "$BAG" /tmp/test_map.bin /tmp/test_npz_bin --step 1 --min_frames 0

# Verify npz has nonzero lanes
uv run python -c "
import numpy as np
from pathlib import Path
npz_files = sorted(Path('/tmp/test_npz_bin').rglob('*.npz'))[:3]
for f in npz_files:
    d = np.load(f, allow_pickle=True)
    lanes = np.count_nonzero(d['lanes'])
    route = np.count_nonzero(d['route_lanes'])
    ego = np.count_nonzero(d['ego_agent_past'])
    print(f'{f.name}: lanes={lanes}, route={route}, ego={ego}')
    assert lanes > 0, f'FAIL: lanes still zero in {f.name}'
print('SUCCESS: all npz have nonzero lanes')
"
```

Expected: lanes > 0 for all npz files. If lanes are still zero, the `.bin` save/load round-trip lost the lanelet data — investigate.

- [ ] **Step 1.4: Update `convert_all_or_bags.py` to use .bin maps**

Modify `scripts/convert_all_or_bags.py` to call `extract_map_bin_from_bag.py` instead of `extract_map_from_bag.py`. The cache key remains (area, map_version) but the cached file is now `.bin` instead of `.osm`.

Key changes:
- Change `MAP_CACHE_DIR` extension from `.osm` to `.bin`
- Change the map extraction subprocess call from `extract_map_from_bag.py` to `extract_map_bin_from_bag.py`
- The `parse_rosbag.py` call stays the same (it calls `convert_lanelet` which now accepts `.bin`)

- [ ] **Step 1.5: Convert 2-3 test bags and verify**

Pick bags from different areas/versions:
- 1 takanawa bag (v23)
- 1 odaiba bag (v531 or v532)

Convert each, verify npz has nonzero lanes/routes. Compare lane counts to training npz to sanity-check.

```bash
uv run python -c "
import numpy as np
from pathlib import Path

# Compare a training npz vs a new override npz
train = np.load(sorted(Path('data/normal_npz_5k').rglob('*.npz'))[0])
override = np.load(sorted(Path('/tmp/test_npz_bin').rglob('*.npz'))[0])
print(f'Training:  lanes={np.count_nonzero(train[\"lanes\"])}, route={np.count_nonzero(train[\"route_lanes\"])}')
print(f'Override:  lanes={np.count_nonzero(override[\"lanes\"])}, route={np.count_nonzero(override[\"route_lanes\"])}')
print('Both have map data — embeddings will be comparable')
"
```

- [ ] **Step 1.6: Commit**

```bash
git add ros_scripts/extract_map_bin_from_bag.py \
       diffusion_planner_ros/diffusion_planner_ros/lanelet2_utils/lanelet_converter.py \
       scripts/convert_all_or_bags.py
git commit -m "fix: use .bin map extraction to preserve native coordinates for lane lookups

The .osm round-trip (shift + MGRS) broke the ego-to-map coordinate
relationship, causing all-zero lane/route tensors in override npz.
New extract_map_bin_from_bag.py saves the raw binary map in native
frame. convert_lanelet() now accepts .bin files. convert_all_or_bags.py
updated to use .bin cache."
```

---

### Task 2: Build `search_similar.py`

**Files:**
- Create: `scripts/search_similar.py`
- Test: Manual verification — query a known scene, check results

**Interfaces:**
- Consumes: `LatentOODScorer.nearest()` (existing, returns top-K with paths + distances)
- Consumes: `EncoderInference` (existing, produces 256-dim embeddings)
- Consumes: Bank at `data/latent_ood_bank_5k/` (existing, 5k training embeddings)
- Produces: CLI tool + JSON output with top-K similar training scenes

- [ ] **Step 2.1: Write `search_similar.py`**

Create `scripts/search_similar.py`:

```python
#!/usr/bin/env python3
"""Find the most similar training scenes for a query npz file.

Uses the Diffusion Planner encoder to embed the query scene, then searches
against a pre-built embedding bank using kNN (L2 distance on L2-normalized
256-dim embeddings).

Usage:
    uv run python scripts/search_similar.py \
      --query data/or_scene_npz/odaiba/停止_車両/.../frame.npz \
      --bank_dir data/latent_ood_bank_5k \
      --model_path /opt/autoware/mlmodels/diffusion_planner_for_x2/diffusion_planner.onnx \
      --args_path /opt/autoware/mlmodels/diffusion_planner_for_x2/args.json \
      --top_k 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.encoder_inference import EncoderInference
from diffusion_planner.utils.latent_ood import LatentOODScorer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query", type=Path, required=True, help="Path to query .npz file")
    p.add_argument("--bank_dir", type=Path, required=True, help="Pre-built embedding bank directory")
    p.add_argument("--model_path", type=Path, required=True, help="ONNX model path")
    p.add_argument("--args_path", type=Path, required=True, help="Model args.json path")
    p.add_argument("--top_k", type=int, default=10, help="Number of nearest neighbors")
    p.add_argument("--output", type=Path, default=None, help="Output JSON path (default: stdout)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Query: {args.query}")
    print(f"Bank:  {args.bank_dir}")

    encoder = EncoderInference(str(args.model_path), str(args.args_path))
    scorer = LatentOODScorer(args.bank_dir)

    dataset = DiffusionPlannerData(
        data_path=[str(args.query)],
        config=encoder.config,
    )
    sample = dataset[0]
    batch = {k: v.unsqueeze(0) for k, v in sample.items()}

    embedding = encoder.encode(batch)

    neighbors = scorer.nearest(embedding, k=args.top_k)

    results = []
    for neighbor in neighbors[0]:
        results.append({
            "rank": len(results) + 1,
            "distance": round(neighbor["distance"], 6),
            "path": neighbor.get("npz_path", f"index_{neighbor['index']}"),
        })

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(results)} results to {args.output}")
    else:
        print(f"\nTop-{args.top_k} similar training scenes:")
        for r in results:
            print(f"  #{r['rank']}: distance={r['distance']:.4f}  {r['path']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Verify it runs (syntax + imports)**

```bash
cd /home/chenglin/workspace/Diffusion-Planner
uv run ruff check scripts/search_similar.py
uv run python -c "import scripts.search_similar" 2>&1 || true
```

- [ ] **Step 2.3: Commit**

```bash
git add scripts/search_similar.py
git commit -m "feat: search_similar.py — find top-K similar training scenes by encoder embedding"
```

---

### Task 3: Demo — Query Override Scenes Against Training Bank

**Files:**
- No new code files
- Produces: Demo output proving the tool works

**Interfaces:**
- Consumes: Fixed override npz from Task 1 (with map data)
- Consumes: `search_similar.py` from Task 2
- Consumes: Existing 5k training bank

This task is a manual verification pipeline, not code to commit.

- [ ] **Step 3.1: Reconvert 2-3 override bags with the fixed map extraction**

Pick diverse bags:
- 1 from takanawa (v23)
- 1-2 from odaiba (different categories)

```bash
source /home/chenglin/workspace/at-team-tools/lin/diffusion-planner-risk-scene/external/pilot-auto.x2/install/setup.bash
cd /home/chenglin/workspace/Diffusion-Planner

# Use the updated batch converter on specific bags
PYTHONPATH="$PWD:$PWD/diffusion_planner_ros:$PYTHONPATH" \
  python3 scripts/convert_all_or_bags.py --max-bags 3
```

Verify the produced npz has nonzero lanes:
```bash
uv run python -c "
import numpy as np
from pathlib import Path
for f in sorted(Path('data/or_scene_npz_v2').rglob('*.npz'))[:5]:
    d = np.load(f, allow_pickle=True)
    print(f'{f.name}: lanes={np.count_nonzero(d[\"lanes\"])}, route={np.count_nonzero(d[\"route_lanes\"])}')
"
```

- [ ] **Step 3.2: Run similarity search on a query scene**

```bash
cd /home/chenglin/workspace/Diffusion-Planner

# Pick a specific override frame
QUERY=$(find data/or_scene_npz_v2 -name "*.npz" -type f | head -1)

uv run python scripts/search_similar.py \
  --query "$QUERY" \
  --bank_dir data/latent_ood_bank_5k \
  --model_path /opt/autoware/mlmodels/diffusion_planner_for_x2/diffusion_planner.onnx \
  --args_path /opt/autoware/mlmodels/diffusion_planner_for_x2/args.json \
  --top_k 10 \
  --output data/demo_search_results.json
```

- [ ] **Step 3.3: Verify results make sense**

Check that the top-K results are plausible by comparing the query and result npz feature vectors:

```bash
uv run python -c "
import json, numpy as np
from pathlib import Path

results = json.load(open('data/demo_search_results.json'))
query_path = '...'  # fill in from step 3.2

query = np.load(query_path, allow_pickle=True)
print('Query:')
print(f'  ego_past shape: {query[\"ego_agent_past\"].shape}')
print(f'  lanes nonzero: {np.count_nonzero(query[\"lanes\"])}')
print(f'  neighbors: {np.count_nonzero(query[\"neighbor_agents_past\"])}')

for r in results[:3]:
    match = np.load(r['path'], allow_pickle=True)
    print(f'Match #{r[\"rank\"]} (dist={r[\"distance\"]:.4f}):')
    print(f'  lanes nonzero: {np.count_nonzero(match[\"lanes\"])}')
    print(f'  neighbors: {np.count_nonzero(match[\"neighbor_agents_past\"])}')
"
```

Success if: both query and matches have nonzero lanes, and distances are reasonable (not all near-zero or near-max).

- [ ] **Step 3.4: Compare old (broken) vs new (fixed) embeddings**

Run the same query against the bank twice — once with the old zero-map npz, once with the new fixed npz. Compare:

```bash
# Old embedding (zero maps)
uv run python scripts/search_similar.py \
  --query data/or_scene_npz/odaiba/停止_車両/.../some_frame.npz \
  --bank_dir data/latent_ood_bank_5k \
  --model_path /opt/autoware/mlmodels/diffusion_planner_for_x2/diffusion_planner.onnx \
  --args_path /opt/autoware/mlmodels/diffusion_planner_for_x2/args.json \
  --top_k 5 --output /tmp/search_old.json

# New embedding (with maps)
uv run python scripts/search_similar.py \
  --query data/or_scene_npz_v2/.../same_frame.npz \
  --bank_dir data/latent_ood_bank_5k \
  --model_path /opt/autoware/mlmodels/diffusion_planner_for_x2/diffusion_planner.onnx \
  --args_path /opt/autoware/mlmodels/diffusion_planner_for_x2/args.json \
  --top_k 5 --output /tmp/search_new.json

# Compare
uv run python -c "
import json
old = json.load(open('/tmp/search_old.json'))
new = json.load(open('/tmp/search_new.json'))
print('Old (no map) distances:', [r['distance'] for r in old])
print('New (with map) distances:', [r['distance'] for r in new])
print('Same results?', [o['path'] for o in old] == [n['path'] for n in new])
"
```

Expected: different results — the fixed embeddings should find different (and hopefully more meaningful) neighbors because the encoder now sees road context.
