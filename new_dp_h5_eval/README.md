# New DP native-H5 evaluation

This directory is the compatibility boundary for evaluating the new DP ONNX model in
the old repository. It does **not** convert old NPZ features into new model inputs.
Only H5 files produced by the new DP preprocessing pipeline (format version 4) are read.

Open-loop uses the old repository's unchanged `planner_metrics` scorers and writes the
same `summary.json` and `details/<metric>/details.jsonl` hierarchy:

```bash
PYTHONPATH=.:diffusion_planner ../new-DP/.venv/bin/python -m new_dp_h5_eval.open_loop \
  ../data/new_DP_dataset/open_loop_native_h5/matrix.json \
  ../data/new_DP_dataset/open_loop_native_h5/index.parquet \
  ../data/hisaki_new_DP_model/diffusion_planner_sampler.onnx \
  ../data/new_DP_dataset/open_loop_native_h5/old_metric_result
```

The matrix paths are joined strictly through the Parquet `source_npz_path` column. A
missing or duplicate mapping aborts before inference, preventing silent frame mismatch.

Closed-loop keeps the existing reproducer, per-step scorers, `segments.jsonl`, and
`summary.json` aggregation. The old route list is used only to establish timeline order
and load recorded world-pose/UUID sidecars; scene/model data always comes from native H5:

```bash
PYTHONPATH=.:diffusion_planner ../new-DP/.venv/bin/python -m new_dp_h5_eval.closed_loop \
  /path/to/native/index.parquet \
  ../data/hisaki_new_DP_model/diffusion_planner_sampler.onnx \
  /path/to/path_list_closed_loop_by_site.json \
  /path/to/output
```

Coordinate recentering handles pose tensors, lane boundary offsets, intersection areas,
stop lines, and road borders according to the new schema. `--gpu_transform` is deliberately
not exposed: the old GPU transform understands the old packed schema and is rejected for
native H5 instead of silently producing wrong coordinates.
