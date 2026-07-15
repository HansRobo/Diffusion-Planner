# Temporal stability validation

This PR adds three validation-only temporal stability metrics:

- `ego_mean_abs_jerk`: mean absolute XY jerk of each predicted ego trajectory.
- `ego_curvature_rate`: mean absolute curvature-rate proxy for steering oscillation.
- `replan_position_consistency` / `replan_heading_consistency`: inter-frame consistency between two adjacent replans.

`ego_mean_abs_jerk` and `ego_curvature_rate` use a single predicted trajectory and can be computed from any validation datalist.

Inter-frame consistency requires a Step-1 full-sequence validation datalist. It compares predictions from frame `t` and frame `t + 1` after transforming the overlapping future from the frame-`t` ego coordinate system into the frame-`t + 1` coordinate system. A normal skip-N datalist should not be used for this metric because it does not represent true adjacent replanning.

The default configuration enables temporal stability validation and requires `replan_consistency_expected_gap=1`. With a Step-1 validation list, the pair loader builds adjacent pairs and logs the inter-frame metrics. With a non-Step-1 list, no adjacent pairs are found and the inter-frame metric is skipped rather than reported with a misleading value.

For fair comparison, use the same training datalist as the experiment being compared and use a Step-1 validation datalist when inter-frame consistency is part of the comparison. If training still uses a skip-N list, keep a separate Step-1 validation list and pass it via `--valid_set_list`.

The metrics do not change model inputs, model outputs, training loss, checkpoint format, or inference behavior.
