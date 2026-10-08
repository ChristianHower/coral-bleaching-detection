import hashlib
import json
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

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
    grid_path = run / "grid.geojson"
    grid_path.write_text(json.dumps(grid))
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
    predictions_path = run / "predictions.parquet"
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(rows), schema=PREDICTION_SCHEMA),
        predictions_path,
    )
    manifest = {
        "schema_version": 2,
        "sha256": {
            "grid.geojson": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
            "predictions.parquet": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_load_run_uses_one_global_date_and_distinguishes_missing_records(tmp_path):
    write_run(tmp_path)
    result = load_run_geojson(tmp_path, "demo-001")
    properties = {
        item["properties"]["reef_cell_id"]: item["properties"] for item in result["features"]
    }
    assert properties["a"]["date"] == "2020-02-01"
    assert properties["a"]["probability"] == 0.8
    assert properties["a"]["top_contributing_features"][0]["feature"] == "dhw"
    assert properties["b"]["probability"] is None
    assert properties["b"]["record_status"] == "available"
    assert properties["c"]["data_quality"] is None
    assert properties["c"]["record_status"] == "missing"
    assert result["metadata"] == {
        "selected_date": "2020-02-01",
        "available_dates": ["2020-01-01", "2020-02-01"],
    }

    earlier = load_run_geojson(tmp_path, "demo-001", "2020-01-01")
    earlier_properties = {
        item["properties"]["reef_cell_id"]: item["properties"] for item in earlier["features"]
    }
    assert earlier_properties["a"]["probability"] == 0.2
    assert earlier_properties["b"]["record_status"] == "missing"


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


def test_index_escapes_default_run_before_inserting_it_in_script(tmp_path):
    payload = "</script><script>globalThis.injected=true</script>"
    app = create_app(tmp_path, payload)
    route = next(route for route in app.routes if route.path == "/")
    response = route.endpoint()
    assert payload not in response
    assert "\\u003c/script\\u003e" in response


def test_load_run_rejects_tampered_files_and_unknown_dates(tmp_path):
    run = write_run(tmp_path)
    with pytest.raises(HTTPException) as missing_date:
        load_run_geojson(tmp_path, "demo-001", "2021-01-01")
    assert missing_date.value.status_code == 404

    (run / "grid.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    with pytest.raises(HTTPException, match="checksum") as invalid:
        load_run_geojson(tmp_path, "demo-001")
    assert invalid.value.status_code == 422


def test_http_endpoint_serves_a_valid_selected_date_and_compresses(tmp_path):
    write_run(tmp_path)
    client = TestClient(create_app(tmp_path, "demo-001"))
    response = client.get(
        "/api/runs/demo-001?date=2020-01-01",
        headers={"accept-encoding": "gzip"},
    )

    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert response.json()["metadata"]["selected_date"] == "2020-01-01"
