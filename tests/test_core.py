import json
from datetime import date, timedelta

import pandas as pd
import pytest
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from coral_bleaching.config import RegionConfig
from coral_bleaching.grid import generate_grid, load_boundary, reproject
from coral_bleaching.labels import build_training_frame
from coral_bleaching.schema import validate_training
from coral_bleaching.sources.aims import Survey, find_matching_label, load_aims_labels
from coral_bleaching.spectral import normalized_difference, rolling_index_shift


@pytest.mark.parametrize(
    "overrides",
    [
        {"cell_size_meters": 0},
        {"cloud_cover_drop_threshold": 1.1},
        {"max_history_gap_days": 0},
        {"reef_id": "../bad"},
    ],
)
def test_config_rejects_invalid(overrides):
    with pytest.raises(ValueError):
        RegionConfig(
            **(
                dict(
                    reef_id="heron",
                    boundary_geojson_path="x",
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 1, 2),
                )
                | overrides
            )
        )


def test_grid_exact_projected_coverage_and_stability(tmp_path):
    boundary = reproject(box(389000, 7406800, 389300, 7407100), "EPSG:32756", "EPSG:4326")
    path = tmp_path / "boundary.geojson"
    path.write_text(json.dumps(mapping(boundary)))
    cells = generate_grid(load_boundary(path), 100, "heron")
    assert len(cells) == 9
    assert len({c.reef_cell_id for c in cells}) == 9
    coverage = unary_union([reproject(c.geometry_wgs84, "EPSG:4326", "EPSG:32756") for c in cells])
    assert coverage.area == pytest.approx(90000, abs=0.01)
    smaller = reproject(box(389000, 7406800, 389200, 7407000), "EPSG:32756", "EPSG:4326")
    assert {c.reef_cell_id for c in generate_grid(smaller, 100, "heron")} <= {
        c.reef_cell_id for c in cells
    }


def test_spectral_missing_history_and_gaps():
    assert normalized_difference(0.6, 0.2) == pytest.approx(0.5)
    for a, b in [(0, 0), (None, 0.2), (float("nan"), 1), (float("inf"), 1)]:
        assert normalized_difference(a, b) is None
    dates = [date(2020, 1, 1) + timedelta(days=7 * i) for i in range(6)]
    history = list(zip(dates, [0.1] * 3 + [0.3] * 3))
    assert rolling_index_shift(history) == (pytest.approx(0.2), dates[0])
    assert rolling_index_shift(history[:3])[0] is None
    assert rolling_index_shift(history, max_gap_days=6)[0] is None
    with pytest.raises(ValueError):
        rolling_index_shift(list(reversed(history)))


def test_surveys_preserve_identity_ties_and_validate(tmp_path):
    records = [
        Survey("a", "c", date(2020, 1, 1), "healthy"),
        Survey("b", "c", date(2020, 1, 3), "bleached"),
    ]
    assert find_matching_label(records, "c", date(2020, 1, 1))[0] == records[0]
    match, diagnostic = find_matching_label(records, "c", date(2020, 1, 2))
    assert match is None and diagnostic["reason"] == "conflicting_equidistant_surveys"
    assert find_matching_label(records, "c", date(2021, 1, 1)) == (None, None)
    path = tmp_path / "surveys.csv"
    path.write_text("survey_id,reef_cell_id,survey_date,label\na,c,2020-01-01,healthy\n")
    assert load_aims_labels(path) == [records[0]]
    path.write_text("survey_id,reef_cell_id,survey_date,label\na,c,2020-01-01,invalid\n")
    with pytest.raises(ValueError):
        load_aims_labels(path)


def test_quality_precedes_confirmed_labels(training_frame):
    training_frame.loc[0, "data_quality"] = "insufficient"
    training_frame.loc[1, ["confirmed_label", "survey_id", "survey_date"]] = None
    result = build_training_frame(training_frame)
    assert len(result) == 23
    assert result.iloc[0].sample_weight == 0.5
    assert result.iloc[1].sample_weight == 1


def test_schema_rejects_missing_provenance_and_bad_keys(training_frame):
    validate_training(training_frame)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_training(pd.concat([training_frame, training_frame]))
    training_frame.loc[0, "survey_id"] = None
    with pytest.raises(ValueError, match="provenance"):
        validate_training(training_frame)


def test_survey_coordinate_mapping_and_unknown_cells(tmp_path):
    from coral_bleaching.grid import ReefCell

    cell = ReefCell("reef_c0", box(151, -24, 152, -23), -23.5, 151.5)
    path = tmp_path / "surveys.csv"
    path.write_text(
        "survey_id,survey_date,label,longitude,latitude\ns1,2020-01-01,healthy,151.5,-23.5\n"
    )
    assert load_aims_labels(path, [cell])[0].reef_cell_id == "reef_c0"
    path.write_text("survey_id,survey_date,label,reef_cell_id\ns1,2020-01-01,healthy,unknown\n")
    with pytest.raises(ValueError, match="Unknown cell"):
        load_aims_labels(path, [cell])


def test_prediction_schema_rejects_false_certainty(training_frame):
    from coral_bleaching.predict import generate_predictions
    from coral_bleaching.schema import validate_predictions

    training_frame["data_quality"] = "insufficient"
    predictions = generate_predictions(training_frame, None)
    predictions.loc[0, "probability"] = 0.9
    with pytest.raises(ValueError, match="abstain"):
        validate_predictions(predictions)
