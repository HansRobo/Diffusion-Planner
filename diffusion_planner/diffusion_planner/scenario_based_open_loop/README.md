# Scenario-based Open-loop Validation

Evaluate predictions on metric-specific NPZ samples during training or standalone validation.

## Input JSON

The JSON maps each metric name to a list of NPZ files:

```json
{
  "centerline": ["/path/to/centerline_scene.npz"],
  "departure": ["/path/to/departure_scene.npz"]
}
```

Supported metrics are `centerline` and `departure`. The NPZ files must use the standard planner input format; centerline evaluation requires `route_lanes` or `lanes`, and departure evaluation requires `ego_current_state`.

Pass the file with:

```bash
--scenario_based_open_loop_list /path/to/open_loop_matrix.json
```

Centerline match uses `--scenario_centerline_match_threshold_m` (default `0.5`) for the fraction of steps whose heading-frame `|n|` is within that threshold. The PNG lateral-offset band uses the same value.

## Centerline scores

Summary averages these per-sample scores:

- `average_lateral_error_m` / `final_lateral_error_m`: unsigned distance to the selected **centerline segment's supporting line**. Endpoint overshoot is `longitudinal_error_m`, not lateral error. Zero-length segments are ignored.
- `lateral_in_band_rate`: percentage of horizon steps whose heading-frame `|n|` is at most `match_threshold_m`. The Frenet s-axis is the **prediction heading** (stored `(cos, sin)` when present, otherwise xy differences); `n` is the left normal (left of the nearest centerline foot is positive).

These two lateral quantities are not interchangeable: segment-axis `lateral_error_m` is the #327 geometry; `lateral_offset_m` is the prediction-heading offset used for `lateral_in_band_rate`.

Per-sample `details.jsonl` keeps scalar centerline fields. `centerline_xy`, `prediction_xy`, `closest_centerline_xy`, and `lateral_offset_m` are used only for the PNG and are not written to JSONL.

## Visualization

Centerline PNGs split the figure: the map (prediction, route centerline, nearest-centerline feet, correspondence lines) on top, and `lateral_offset_m` versus time (`dt = 0.1s`) below, with a zero line and the `|n| <= threshold` band. Departure and other metrics without `lateral_offset_m` keep the original map-only PNG. Offset values are never plotted as XY coordinates.

## Adding a Metric

1. Implement the scorer in `planner_metrics/<metric_name>.py`.
2. Return `MetricEvaluation` from the open-loop scorer, including `scores` and optional `details`.
3. Register it in `scenario_based_open_loop/open_loop.py`.
4. Add configuration fields and tests as needed.
