# Latent Similarity Search Tool — Design Spec

**Date:** 2026-07-21
**Author:** Cheng (Dio) Lin
**Goal:** Prove that encoder embeddings can find similar scenes between problem data and training data
**Users:** Lin (demo), Kawahara-san (DevOps, scaling)

---

## 1. Problem

Given a scene where DP failed (override, test failure, etc.), find the most
similar scenes in the training dataset using the Diffusion Planner encoder's
latent space. This enables:
- Understanding what training data exists for a failure mode
- Identifying training data gaps (failure with no similar training scenes)
- Curating targeted retraining datasets

## 2. Blocker: Map Extraction Bug

### Root cause

`ros_scripts/extract_map_from_bag.py` extracts the lanelet map from a bag and
saves it as .osm. To fit the MGRSProjector's [0, 100000) coordinate range, it
**shifts all point coordinates** by a constant vector. When `parse_rosbag.py`
later loads this .osm and searches for lanelets near the ego position
(kinematic_state, which is in the bag's native unshifted frame), it finds
nothing because the coordinate systems don't match.

Result: all override npz have zero lane/route/polygon/line_string tensors.
Training npz (produced by the standard pipeline with correct maps) have full
map data. Embeddings are not comparable.

### Fix: bypass the .osm round-trip

The binary map data from `/map/vector_map` is already in the bag's native
map frame — the same frame as kinematic_state. The coordinate mismatch is
introduced by the shift + MGRS conversion + MGRS reload pipeline. We bypass
it entirely:

1. Extract raw binary bytes from the bag's `/map/vector_map` topic
2. Write to a `.bin` temp file
3. Load with `lanelet2.io.load(bin_path, dummy_projector)` — this
   deserializes the Boost archive directly, preserving native coordinates
4. Rebuild lanelets from scratch (drop dangling cached members, same as
   the existing script does)
5. Save as `.bin` file (NOT .osm) — cached by (area, map_version)

Then modify `parse_rosbag.py` (or the batch converter) to load the `.bin`
file directly with the same dummy projector, so the map coordinates match
the ego positions.

**Key insight:** `convert_lanelet()` in `lanelet_converter.py` uses
`MGRSProjector(Origin(0.0, 0.0))` for loading. When loading a `.bin` file,
lanelet2 dispatches to the Boost-serialization handler, which ignores the
projector entirely — point x/y are stored directly. So loading a `.bin`
with any projector gives native coordinates. We just need to make sure
`convert_lanelet` can accept `.bin` files (it currently only handles `.osm`).

**Alternative considered:** Shift ego positions to match the shifted map.
Rejected because it requires modifying parse_rosbag.py's coordinate pipeline
and is fragile.

## 3. Similarity Search Tool

### Interface

```
search_similar.py \
  --query <path_to_query.npz> \
  --bank_dir <path_to_embedding_bank> \
  --model_path <onnx_model> \
  --args_path <args.json> \
  --top_k 10
```

Output: JSON with top-K matches:
```json
[
  {"path": "train/2026-04-22/14-46-29/..._00006211.npz", "distance": 0.123},
  {"path": "train/2026-03-05/10-14-20/..._00000505.npz", "distance": 0.145},
  ...
]
```

### Components

Uses existing infrastructure:
- `encoder_inference.py` — ONNX encoder embedding extraction
- `build_latent_ood_bank.py` — bank building (already works)
- `score_latent_ood.py` — kNN scoring (already works, just need to return
  indices instead of just distances)

New code: `search_similar.py` — thin wrapper that:
1. Loads the bank
2. Embeds the query npz
3. Finds top-K nearest neighbors (L2 distance)
4. Returns paths + distances

### Bank

Built from training npz on sakurab. Start with the existing 5k bank for
the demo. Kawahara-san can scale to the full dataset with FAISS later.

## 4. Demo Plan

Prove the concept works by:
1. Pick a known override scene (e.g., 停止_車両 at a specific intersection)
2. Search against the 5k training bank
3. Check: do the top-K results show scenes at the same/similar intersection
   with similar traffic patterns?
4. Compare: what does the same query look like when embeddings have map
   context vs without?

Success criteria: similar scenes should be at similar road geometry with
similar agent configurations. If the results are random, the embeddings
don't capture meaningful scene structure.

## 5. Scope

**In scope:**
- Fix map extraction (`.bin` path, no MGRS round-trip)
- Reconvert a few test bags (not all 74 — just enough to demo)
- `search_similar.py` script
- Demo showing it works on 2-3 query scenes

**Out of scope:**
- Full 74-bag reconversion (can do after demo)
- FAISS index for large-scale search (Kawahara-san's team)
- Visualization / UI
- Clustering analysis
- Re-running the LightGBM evaluation
