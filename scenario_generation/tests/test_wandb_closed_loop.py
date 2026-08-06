"""Unit tests for wandb_closed_loop's per-site bar-chart builder."""

from __future__ import annotations

from scenario_generation.wandb_closed_loop import (
    build_full_closed_loop_wandb_log,
    build_sites_score_bar_charts,
)


def _summary(collisions=0, curb_hits=0, snaps=0, red_light=0, strong_brake=0, completion=1.0):
    return {
        "mean_route_completion": completion,
        "object": {"collision_count": collisions},
        "road_border": {"collision_count": curb_hits},
        "reproducer": {"snap_count": snaps},
        "red_light_violation": {"count": red_light},
        "strong_brake": {"count": strong_brake},
    }


def test_build_sites_score_bar_charts_one_chart_per_metric_with_all_sites():
    """Every SCORE_KEY gets its own chart; each chart has one bar per site."""
    summaries = {
        "site_a": _summary(collisions=2, curb_hits=1),
        "site_b": _summary(collisions=0, curb_hits=3),
    }

    log = build_sites_score_bar_charts(summaries)

    assert set(log.keys()) == {
        "closed_loop_scores_bar/mean_route_completion",
        "closed_loop_scores_bar/total_curb_hits",
        "closed_loop_scores_bar/total_snaps",
        "closed_loop_scores_bar/total_red_light_violations",
        "closed_loop_scores_bar/total_strong_brakes",
        "closed_loop_scores_bar/total_collision_events",
    }
    curb_hits_rows = log["closed_loop_scores_bar/total_curb_hits"].table.data
    assert dict(curb_hits_rows) == {"site_a": 1, "site_b": 3}
    collision_rows = log["closed_loop_scores_bar/total_collision_events"].table.data
    assert dict(collision_rows) == {"site_a": 2, "site_b": 0}


def test_build_sites_score_bar_charts_excludes_noobj_from_collision_events():
    """The empty-world ablation (__noobj) is always a meaningless 0 for collision events."""
    summaries = {
        "site_a": _summary(collisions=2),
        "site_a__noobj": _summary(collisions=0),
    }

    log = build_sites_score_bar_charts(summaries)

    collision_rows = dict(log["closed_loop_scores_bar/total_collision_events"].table.data)
    assert collision_rows == {"site_a": 2}
    # Comparison keys (not objects-only) still include the noobj label.
    completion_rows = dict(log["closed_loop_scores_bar/mean_route_completion"].table.data)
    assert set(completion_rows) == {"site_a", "site_a__noobj"}


def test_build_sites_score_bar_charts_empty_summaries_returns_empty_log():
    assert build_sites_score_bar_charts({}) == {}


def test_include_score_scalars_false_omits_per_site_score_keys():
    """Sites that only run once per training run skip the (now single-point) scalar trend."""
    log = build_full_closed_loop_wandb_log(
        _summary(collisions=1, curb_hits=2),
        site="site_a",
        render_media=False,
        include_score_scalars=False,
    )

    assert not any(key.startswith("closed_loop_scores/") for key in log)


def test_include_score_scalars_true_by_default_keeps_per_site_score_keys():
    """main (closed_loop_npz_root) still runs every cadence call -- its trend stays intact."""
    log = build_full_closed_loop_wandb_log(
        _summary(collisions=1, curb_hits=2), site="main", render_media=False
    )

    assert log["closed_loop_scores/total_collision_events/main"] == 1
    assert log["closed_loop_scores/total_curb_hits/main"] == 2
