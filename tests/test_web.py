import json
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import HTTPException

from coral_bleaching.schema import PREDICTION_SCHEMA
from coral_bleaching.web import create_app, load_run_geojson


def write_run(root):
    run = root / "demo-001"
    run.mkdir()
    grid = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"reef_cell_id": cell},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[151.9, -23.4], [151.91, -23.4], [151.91, -23.41], [151.9, -23.4]]
                    ],
                },
            }
            for cell in ("a", "b", "c")
        ],
    }
    (run / "grid.geojson").write_text(json.dumps(grid))
    rows = [
        {
            "reef_cell_id": "a",
            "date": date(2020, 1, 1),
            "probability": 0.2,
            "predicted_label": "healthy",
            "top_contributing_features": [],
            "confidence_band": "high",
            "data_quality": "ok",
        },
        {
            "reef_cell_id": "a",
            "date": date(2020, 2, 1),
            "probability": 0.8,
            "predicted_label": "bleached",
            "top_contributing_features": [{"feature": "dhw", "value": 5.0, "contribution": 0.7}],
            "confidence_band": "high",
            "data_quality": "ok",
        },
        {
            "reef_cell_id": "b",
            "date": date(2020, 2, 1),
            "probability": None,
            "predicted_label": None,
            "top_contributing_features": [],
            "confidence_band": "unavailable",
            "data_quality": "insufficient",
        },
    ]
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(rows), schema=PREDICTION_SCHEMA),
        run / "predictions.parquet",
    )


def test_load_run_uses_latest_prediction_and_marks_missing_cell(tmp_path):
    write_run(tmp_path)
    result = load_run_geojson(tmp_path, "demo-001")
    properties = {
        item["properties"]["reef_cell_id"]: item["properties"] for item in result["features"]
    }
    assert properties["a"]["date"] == "2020-02-01"
    assert properties["a"]["probability"] == 0.8
    assert properties["a"]["top_contributing_features"][0]["feature"] == "dhw"
    assert properties["b"]["probability"] is None
    assert properties["c"]["data_quality"] == "insufficient"


def test_load_run_reports_missing_and_rejects_traversal(tmp_path):
    with pytest.raises(HTTPException, match="not found") as missing:
        load_run_geojson(tmp_path, "missing")
    assert missing.value.status_code == 404
    with pytest.raises(HTTPException) as traversal:
        load_run_geojson(tmp_path, "..")
    assert traversal.value.status_code == 404


def test_index_includes_selected_default_run(tmp_path):
    app = create_app(tmp_path, "field-check")
    route = next(route for route in app.routes if route.path == "/")
    response = route.endpoint()
    assert 'const defaultRun = "field-check"' in response
