import importlib.util
from datetime import date, timedelta
from pathlib import Path

from coral_bleaching.sources.sentinel import Observation

# Load the script module directly; it lives under scripts/, not the package.
_spec = importlib.util.spec_from_file_location(
    "quantify_coverage", Path("scripts/quantify_coverage.py")
)
quantify_coverage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(quantify_coverage)
aggregate_coverage = quantify_coverage.aggregate_coverage


def _obs(day, fraction):
    return Observation(day, f"scene-{day}", None, None, fraction)


def test_usable_rule_matches_threshold_and_rejects_none():
    assert quantify_coverage._usable(0.1, 0.5) is True
    assert quantify_coverage._usable(0.5, 0.5) is True
    assert quantify_coverage._usable(0.6, 0.5) is False
    assert quantify_coverage._usable(None, 0.5) is False


def test_max_gap_detects_a_long_break():
    base = date(2020, 1, 1)
    tight = [base, base + timedelta(days=7), base + timedelta(days=14)]
    assert quantify_coverage._max_gap_ok(tight, 14) is True
    wide = [base, base + timedelta(days=20)]
    assert quantify_coverage._max_gap_ok(wide, 14) is False


def test_aggregate_counts_usable_days_and_feature_bar():
    base = date(2020, 1, 1)
    # cell A: 6 clean weekly acquisitions, no gap > 14 days -> meets the bar.
    clean = [_obs(base + timedelta(days=7 * n), 0.1) for n in range(6)]
    # cell B: 6 acquisitions but one is cloudy (fraction above threshold) and
    # the resulting usable series has a 14+ day gap -> fails the bar.
    gappy = [
        _obs(base, 0.1),
        _obs(base + timedelta(days=7), 0.9),  # cloudy, unusable
        _obs(base + timedelta(days=14), 0.9),  # cloudy, unusable
        _obs(base + timedelta(days=28), 0.1),
    ]
    report = aggregate_coverage(
        {"a": clean, "b": gappy},
        threshold=0.5,
        min_clean_acquisitions=6,
        max_gap_days=14,
    )
    assert report["cell_count"] == 2
    assert report["usable_days_per_cell"] == {"a": 6, "b": 2}
    assert report["cells_meeting_feature_bar"] == 1  # only A
    # six distinct usable acquisition days across the reef (A's six)
    assert report["acquisition_days_with_usable_pixels"] == 6
    # per-day fraction: A's dates have 1/2 or 2/2 usable cells
    first_day = base.isoformat()
    assert report["per_day_usable_cell_fraction"][first_day] == 1.0  # A and B both usable day 0
    # cloud distribution captured both cloudy and clear fractions
    dist = report["cloud_cover_fraction_distribution"]
    assert dist["count"] == len(clean) + len(gappy)
    assert dist["min"] == 0.1 and dist["max"] == 0.9


def test_aggregate_empty_input_is_safe():
    report = aggregate_coverage({}, threshold=0.5, min_clean_acquisitions=6, max_gap_days=14)
    assert report["cell_count"] == 0
    assert report["acquisition_days_with_usable_pixels"] == 0
    assert report["cells_meeting_feature_bar"] == 0
    assert report["cloud_cover_fraction_distribution"]["count"] == 0
