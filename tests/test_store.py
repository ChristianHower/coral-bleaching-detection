import hashlib
import json
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from coral_bleaching import store
from coral_bleaching.pipeline import write_dataset
from coral_bleaching.schema import PREDICTION_SCHEMA, TRAINING_SCHEMA
from coral_bleaching.store import IngestError, connect
from coral_bleaching.web import create_region_app


# Three cells: 'a' has two dates (one ok, one bleached+DHW flag), 'b' is
# insufficient on the second date and absent on the first (missing record),
# 'c' is healthy with a DHW heat-stress flag (the disagreement case).
def _grid():
    def polygon(lon):
        return {
            "type": "Polygon",
            "coordinates": [[[lon, -23.4], [lon + 0.01, -23.4], [lon + 0.01, -23.41], [lon, -23.4]]],
        }

    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"reef_cell_id": cell}, "geometry": polygon(151.9 + i * 0.02)}
            for i, cell in enumerate(("a", "b", "c"))
        ],
    }


def _predictions():
    rows = [
        dict(reef_cell_id="a", date=date(2020, 1, 1), probability=0.2, predicted_label="healthy",
             top_contributing_features=[], confidence_band="high", data_quality="ok"),
        dict(reef_cell_id="a", date=date(2020, 1, 15), probability=0.8, predicted_label="bleached",
             top_contributing_features=[{"feature": "dhw", "value": 6.0, "contribution": 0.7}],
             confidence_band="high", data_quality="ok"),
        dict(reef_cell_id="b", date=date(2020, 1, 15), probability=None, predicted_label=None,
             top_contributing_features=[], confidence_band="unavailable", data_quality="insufficient"),
        dict(reef_cell_id="c", date=date(2020, 1, 1), probability=0.1, predicted_label="healthy",
             top_contributing_features=[], confidence_band="high", data_quality="ok"),
        dict(reef_cell_id="c", date=date(2020, 1, 15), probability=0.15, predicted_label="healthy",
             top_contributing_features=[], confidence_band="high", data_quality="ok"),
    ]
    return pd.DataFrame(rows, columns=PREDICTION_SCHEMA.names)


def _training():
    # dhw/dhw_risk_flag joined into predictions at ingest come from here.
    rows = []
    quality = {
        ("a", date(2020, 1, 1)): ("ok", 1.0),
        ("a", date(2020, 1, 15)): ("ok", 6.0),
        ("b", date(2020, 1, 15)): ("insufficient", 2.0),
        ("c", date(2020, 1, 1)): ("ok", 7.0),
        ("c", date(2020, 1, 15)): ("ok", 7.0),
    }
    for (cell, day), (dq, dhw) in quality.items():
        usable = dq == "ok"
        rows.append(dict(
            reef_cell_id=cell, date=day,
            ndci_like_index=0.3 if usable else None,
            index_shift=None,
            dhw=dhw,
            cloud_cover_fraction=0.1 if usable else 0.9,
            data_quality=dq,
            dhw_risk_flag=dhw >= 4.0,
            confirmed_label=None, survey_date=None, survey_id=None,
            source_scene_id="scene" if usable else None,
            feature_start_date=day,
        ))
    return pd.DataFrame(rows, columns=TRAINING_SCHEMA.names)


def write_run(root, name="demo-001", synthetic=True):
    run = root / name
    run.mkdir()
    grid_path = run / "grid.geojson"
    grid_path.write_text(json.dumps(_grid()))
    predictions_path = run / "predictions.parquet"
    pq.write_table(pa.Table.from_pandas(_predictions(), schema=PREDICTION_SCHEMA), predictions_path)
    write_dataset(_training(), run / "training_dataset.parquet")

    checksums = {}
    for path in run.rglob("*"):
        if path.is_file():
            checksums[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 2,
        "region": {"reef_id": "heron", "utm_epsg": "EPSG:32756", "cell_size_meters": 75.0},
        "provenance": {"synthetic": synthetic},
        "validation_status": "evaluated",
        "sha256": checksums,
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


@pytest.fixture
def db(tmp_path):
    connection = connect(str(tmp_path / "reef.db"))
    store.init_db(connection)
    return connection


def test_ingest_loads_region_cells_and_joins_dhw(tmp_path, db):
    write_run(tmp_path)
    run_id = store.ingest_run(db, tmp_path / "demo-001", "Heron pilot")
    regions = store.list_regions(db)
    assert len(regions) == 1
    region = regions[0]
    assert region["region_id"] == "heron"
    assert region["published_run_id"] == run_id
    assert region["cell_count"] == 3
    assert region["available_dates"] == ["2020-01-01", "2020-01-15"]


def test_predictions_endpoint_marks_missing_insufficient_and_dhw(tmp_path, db):
    write_run(tmp_path)
    store.ingest_run(db, tmp_path / "demo-001")
    result = store.region_predictions(db, "heron", "2020-01-15")
    by_cell = {f["properties"]["reef_cell_id"]: f["properties"] for f in result["features"]}
    # 'a' bleached with the DHW value joined from training
    assert by_cell["a"]["probability"] == 0.8
    assert by_cell["a"]["dhw"] == 6.0 and by_cell["a"]["dhw_risk_flag"] is True
    assert by_cell["a"]["top_contributing_features"][0]["feature"] == "dhw"
    # 'b' insufficient on this date
    assert by_cell["b"]["data_quality"] == "insufficient"
    assert by_cell["b"]["record_status"] == "available"
    # metadata provenance
    assert result["metadata"]["synthetic"] is True
    assert result["metadata"]["selected_date"] == "2020-01-15"

    earlier = store.region_predictions(db, "heron", "2020-01-01")
    early_cells = {f["properties"]["reef_cell_id"]: f["properties"] for f in earlier["features"]}
    # 'b' has no record on the first date -> missing, distinct from insufficient
    assert early_cells["b"]["record_status"] == "missing"
    assert early_cells["b"]["data_quality"] is None


def test_default_date_is_latest_and_unknown_date_raises(tmp_path, db):
    write_run(tmp_path)
    store.ingest_run(db, tmp_path / "demo-001")
    assert store.region_predictions(db, "heron")["metadata"]["selected_date"] == "2020-01-15"
    with pytest.raises(ValueError):
        store.region_predictions(db, "heron", "2019-01-01")


def test_cell_history_returns_full_series_and_none_for_unknown(tmp_path, db):
    write_run(tmp_path)
    store.ingest_run(db, tmp_path / "demo-001")
    history = store.cell_history(db, "heron", "a")
    assert [row["date"] for row in history["history"]] == ["2020-01-01", "2020-01-15"]
    assert [row["probability"] for row in history["history"]] == [0.2, 0.8]
    assert store.cell_history(db, "heron", "nope") is None


def test_reingest_supersedes_prior_run(tmp_path, db):
    write_run(tmp_path, "demo-001")
    write_run(tmp_path, "demo-002")
    first = store.ingest_run(db, tmp_path / "demo-001")
    second = store.ingest_run(db, tmp_path / "demo-002")
    runs = {r["run_id"]: r["status"] for r in store.list_runs(db, "heron")}
    assert runs[first] == "superseded"
    assert runs[second] == "published"
    assert store.region_predictions(db, "heron")["metadata"]["run_id"] == second


def test_ingest_rejects_tampered_run(tmp_path, db):
    run = write_run(tmp_path)
    (run / "grid.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    with pytest.raises(IngestError, match="checksum"):
        store.ingest_run(db, run)
    # a failed ingest leaves no region behind
    assert store.list_regions(db) == []


def test_http_endpoints_return_expected_status_codes(tmp_path, db):
    write_run(tmp_path)
    store.ingest_run(db, tmp_path / "demo-001")
    client = TestClient(create_region_app(str(tmp_path / "reef.db"), "heron"))
    assert client.get("/api/regions").status_code == 200
    assert client.get("/api/regions/heron/grid").status_code == 200
    predictions = client.get("/api/regions/heron/predictions", headers={"accept-encoding": "gzip"})
    assert predictions.status_code == 200
    assert predictions.headers.get("content-encoding") == "gzip"
    assert client.get("/api/regions/heron/cells/a/history").status_code == 200
    assert client.get("/api/regions/heron/runs").status_code == 200
    # 404s
    assert client.get("/api/regions/nope/grid").status_code == 404
    assert client.get("/api/regions/heron/predictions?date=1999-01-01").status_code == 404
    assert client.get("/api/regions/heron/cells/nope/history").status_code == 404
    # index serves region.html with the default region injected
    index = client.get("/")
    assert index.status_code == 200
    assert 'const defaultRegion = "heron"' in index.text
    assert "__DEFAULT_REGION__" not in index.text
