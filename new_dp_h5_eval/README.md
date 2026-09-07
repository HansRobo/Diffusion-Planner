# New DP native-H5 evaluation

Only native H5 is accepted. Open-loop JSON maps each metric to editable
`{"h5_path": "...", "frame_index": N}` entries. Closed-loop JSON maps each group
to full-route H5 paths; an entry may instead be
`{"h5_path": "...", "frame_start": N, "frame_stop": M}`.

```bash
PYTHONPATH=.:diffusion_planner ../new-DP/.venv/bin/python -m new_dp_h5_eval.open_loop \
  /path/to/open_loop_basic_h5.json \
  /path/to/basic/index.parquet \
  /path/to/diffusion_planner_sampler.onnx \
  /path/to/output

PYTHONPATH=.:diffusion_planner ../new-DP/.venv/bin/python -m new_dp_h5_eval.run_all_groups_closed_loop \
  --closed_loop_h5_root /path/to/closed_loop_by_site_h5.json /path/to/closed_loop_override_h5.json \
  --closed_loop_object_modes objects noobj \
  --model_path /path/to/diffusion_planner_sampler.onnx \
  --out_root /path/to/output
```

The closed-loop command preserves the standard runner's rendering and aggregate-result
layout. Full-route H5 is created from ROS bags with
`new-DP/scripts/dataset/create_h5_dataset.py`; the one-frame `h5/basic` files are for
open-loop only.
