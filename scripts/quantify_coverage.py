"""Quantify usable Sentinel-2 reef coverage over the real pilot grid.

Turns "97 catalog hits" into the numbers that actually gate the pipeline: how
many acquisition days yield usable reef pixels, how many cells clear the
quality threshold per day, and how many cells reach enough clean acquisitions
with no gap over the configured limit to compute the rolling features at all.

This is a read-only diagnostic. It reuses the shipped adapter
(`EarthSearchClient.observations`), so "usable" means exactly what the pipeline
means (SCL masking, water retained, the configured unusable-fraction
threshold) — no new science and no new definition. It touches the live public
STAC catalog and COG windows, so it is network-gated: `aggregate_coverage` is
a pure function tested offline; `main` is the thin network entry point.

Running it does not establish bleaching accuracy or clear-pixel suitability for
a study. It establishes whether 2020 even has enough usable multi-date coverage
over Heron to compute features, before anyone spends effort on labels.
"""

import argparse
import json
from datetime import datetime, timezone


def _usable(cloud_cover_fraction, threshold):
    """A cell/day is usable when enough reef pixels survived masking.

    Mirrors the pipeline's rule: fraction present and within [0, threshold].
    None (no represented or no valid pixels) is not usable.
    """
    return (
        cloud_cover_fraction is not None
        and 0 <= cloud_cover_fraction <= threshold
    )


def _max_gap_ok(dates, max_gap_days):
    """True if no two consecutive sorted dates are more than max_gap_days apart."""
    ordered = sorted(dates)
    for earlier, later in zip(ordered, ordered[1:]):
        if (later - earlier).days > max_gap_days:
            return False
    return True


def aggregate_coverage(
    observations_by_cell,
    threshold,
    min_clean_acquisitions,
    max_gap_days,
):
    """Pure aggregation of adapter output into coverage statistics.

    observations_by_cell: {reef_cell_id: [Observation]} as returned by
    EarthSearchClient.observations (each Observation has .date and
    .cloud_cover_fraction). No network, no file I/O; all callers (the script
    and the tests) share this exact logic so the reported numbers match the
    pipeline's eligibility rule.
    """
    cell_count = len(observations_by_cell)
    usable_days_per_cell = {}
    usable_cells_per_day = {}
    all_days = set()
    cloud_fractions = []
    cells_meeting_feature_bar = 0

    for cell_id, observations in observations_by_cell.items():
        usable_dates = []
        for observation in observations:
            all_days.add(observation.date)
            if observation.cloud_cover_fraction is not None:
                cloud_fractions.append(float(observation.cloud_cover_fraction))
            if _usable(observation.cloud_cover_fraction, threshold):
                usable_dates.append(observation.date)
                usable_cells_per_day[observation.date] = (
                    usable_cells_per_day.get(observation.date, 0) + 1
                )
        usable_days_per_cell[cell_id] = len(usable_dates)
        if len(usable_dates) >= min_clean_acquisitions and _max_gap_ok(
            usable_dates, max_gap_days
        ):
            cells_meeting_feature_bar += 1

    acquisition_days_with_usable_pixels = len(usable_cells_per_day)
    per_day_usable_cell_fraction = {
        (day.isoformat() if hasattr(day, "isoformat") else str(day)): (
            count / cell_count if cell_count else 0.0
        )
        for day, count in sorted(usable_cells_per_day.items())
    }

    cloud_fractions.sort()

    def _percentile(values, pct):
        if not values:
            return None
        index = min(len(values) - 1, int(round(pct / 100 * (len(values) - 1))))
        return values[index]

    return {
        "cell_count": cell_count,
        "acquisition_days_total": len(all_days),
        "acquisition_days_with_usable_pixels": acquisition_days_with_usable_pixels,
        "per_day_usable_cell_fraction": per_day_usable_cell_fraction,
        "usable_days_per_cell": usable_days_per_cell,
        "cells_meeting_feature_bar": cells_meeting_feature_bar,
        "feature_bar": {
            "min_clean_acquisitions": min_clean_acquisitions,
            "max_gap_days": max_gap_days,
        },
        "cloud_cover_fraction_distribution": {
            "count": len(cloud_fractions),
            "min": cloud_fractions[0] if cloud_fractions else None,
            "p25": _percentile(cloud_fractions, 25),
            "median": _percentile(cloud_fractions, 50),
            "p75": _percentile(cloud_fractions, 75),
            "max": cloud_fractions[-1] if cloud_fractions else None,
        },
        "threshold": threshold,
    }


def _load_region(config_path):
    # Imported lazily so importing this module for tests pulls in no heavy deps.
    from coral_bleaching.cli import load_region

    return load_region(config_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Quantify usable Sentinel-2 reef coverage (read-only, network)"
    )
    parser.add_argument("--config", required=True, help="Region config JSON (e.g. examples/heron_island.json)")
    parser.add_argument("--output", required=True, help="Where to write the dated JSON report")
    parser.add_argument(
        "--min-clean-acquisitions",
        type=int,
        default=6,
        help="Clean acquisitions a cell needs for the rolling feature (default 6)",
    )
    args = parser.parse_args(argv)

    from coral_bleaching.grid import generate_grid, load_boundary
    from coral_bleaching.sources.sentinel import EarthSearchClient

    region = _load_region(args.config)
    boundary = load_boundary(region.boundary_geojson_path)
    cells = generate_grid(
        boundary,
        region.cell_size_meters,
        region.reef_id,
        region.min_overlap_fraction,
        region.utm_epsg,
        region.grid_origin_x,
        region.grid_origin_y,
    )
    if not cells:
        parser.exit(1, "Boundary generated no eligible cells\n")

    client = EarthSearchClient(collection=region.collection)
    observations_by_cell = client.observations(
        cells, boundary, region.start_date, region.end_date
    )
    report = aggregate_coverage(
        observations_by_cell,
        region.cloud_cover_drop_threshold,
        args.min_clean_acquisitions,
        region.max_history_gap_days,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["region"] = region.reef_id
    report["interval"] = [region.start_date.isoformat(), region.end_date.isoformat()]
    report["note"] = (
        "Catalog/clear-pixel coverage only. Not a bleaching-accuracy or "
        "clear-reef-suitability result."
    )
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)
    print(
        f"Wrote {args.output}: "
        f"{report['acquisition_days_with_usable_pixels']} usable acquisition days, "
        f"{report['cells_meeting_feature_bar']}/{report['cell_count']} cells reach the feature bar"
    )


if __name__ == "__main__":
    main()
