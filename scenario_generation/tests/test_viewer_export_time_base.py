"""The run reports its draw setting as configured, so the export must survive a word."""

from __future__ import annotations

import pytest

from scenario_generation.scenario_sim_viewer_export import _as_number


@pytest.mark.parametrize("value", ["off", "", None, "none", "auto", [], {}])
def test_a_non_numeric_setting_reads_as_no_time_base(value):
    assert _as_number(value) == 0.0


@pytest.mark.parametrize(("value", "expected"), [("4", 4.0), (4, 4.0), ("10.0", 10.0), (0, 0.0)])
def test_a_numeric_setting_is_carried(value, expected):
    assert _as_number(value) == expected
