"""SQLite serving store for the reef map.

The pipeline writes immutable, checksummed run directories (Build 1). This
module loads a verified run into SQLite once, at ingest, and the API reads
SQLite. The run directory stays the source of truth; the database is a
derived, rebuildable serving store. Nothing here imports the model or the
pipeline's build path — only the run's published files and their verification.

Geometry is stored as GeoJSON strings plus a bounding box. At single-reef
scale (a few hundred to low-thousands of cells) that is enough; a spatial
extension is the documented PostGIS upgrade, not a pilot need.
"""

import hashlib
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from coral_bleaching.pipeline import read_dataset
from coral_bleaching.schema import PREDICTION_SCHEMA, validate_predictions

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS region (
    region_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    crs TEXT NOT NULL,
    cell_size_m REAL NOT NULL,
    boundary_geojson TEXT
);
CREATE TABLE IF NOT EXISTS reef_cell (
    region_id TEXT NOT NULL REFERENCES region(region_id),
    reef_cell_id TEXT NOT NULL,
    geometry_geojson TEXT NOT NULL,
    min_lon REAL, min_lat REAL, max_lon REAL, max_lat REAL,
    PRIMARY KEY (region_id, reef_cell_id)
);
CREATE TABLE IF NOT EXISTS run (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id TEXT NOT NULL REFERENCES region(region_id),
    run_name TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    source_run_dir TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    synthetic INTEGER NOT NULL,
    validation_status TEXT,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prediction (
    run_id INTEGER NOT NULL REFERENCES run(run_id),
    reef_cell_id TEXT NOT NULL,
    date TEXT NOT NULL,
    probability REAL,
    predicted_label TEXT,
    confidence_band TEXT,
    data_quality TEXT,
    dhw REAL,
    dhw_risk_flag INTEGER,
    top_contributing_features_json TEXT NOT NULL,
    PRIMARY KEY (run_id, reef_cell_id, date)
);
CREATE INDEX IF NOT EXISTS prediction_cell ON prediction(run_id, reef_cell_id);
CREATE INDEX IF NOT EXISTS prediction_date ON prediction(run_id, date);
"""


class IngestError(RuntimeError):
    pass


def connect(db_path):
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(connection):
    connection.executescript(_SCHEMA)
    connection.commit()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_run(run_dir):
    """Re-run the Build 1 trust checks at load time; abort ingest on any failure.

    This is the same verification web.py did per request, moved to run once at
    ingest: schema version, SHA-256 of every file the manifest claims, grid
    shape, and prediction schema. A tampered or half-written run never loads.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise IngestError(f"Run '{run_dir.name}' is missing manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise IngestError("Unsupported or missing run schema version")

    checksums = manifest.get("sha256", {})
    if not checksums:
        raise IngestError("Manifest records no checksums")
    for name, expected in checksums.items():
        target = run_dir / name
        if not target.is_file() or _sha256(target) != expected:
            raise IngestError(f"Run checksum failed for {name}")

    grid = json.loads((run_dir / "grid.geojson").read_text())
    if grid.get("type") != "FeatureCollection" or not isinstance(grid.get("features"), list):
        raise IngestError("Grid is not a GeoJSON FeatureCollection")
    cell_ids = [f.get("properties", {}).get("reef_cell_id") for f in grid["features"]]
    if any(not cell_id for cell_id in cell_ids) or len(cell_ids) != len(set(cell_ids)):
        raise IngestError("Grid requires unique, nonempty reef_cell_id values")

    predictions_path = run_dir / "predictions.parquet"
    if pq.read_schema(predictions_path).remove_metadata() != PREDICTION_SCHEMA:
        raise IngestError("Prediction file does not match schema version 2")
    return manifest, grid


def _bounds(geometry):
    """Longitude/latitude bounding box of a GeoJSON geometry's coordinates."""
    lons, lats = [], []

    def walk(coords):
        if coords and isinstance(coords[0], (int, float)):
            lons.append(coords[0])
            lats.append(coords[1])
            return
        for part in coords:
            walk(part)

    walk(geometry.get("coordinates", []))
    if not lons:
        return (None, None, None, None)
    return (min(lons), min(lats), max(lons), max(lats))


def ingest_run(connection, run_dir, region_name=None):
    """Load one verified run into SQLite and publish it for its region.

    Verification runs first; a failure aborts without touching the database.
    On success the prior published run for the same region is demoted to
    'superseded' and the new run becomes the single 'published' run, in one
    transaction. Confirmations (Module 5) key on (region, cell, date), so they
    survive supersession; they are not stored here.
    """
    run_dir = Path(run_dir).resolve()
    manifest, grid = _verify_run(run_dir)
    region = manifest["region"]
    region_id = region["reef_id"]
    synthetic = bool(manifest.get("provenance", {}).get("synthetic"))

    predictions = pq.read_table(run_dir / "predictions.parquet").to_pandas()
    validate_predictions(predictions)
    training = read_dataset(run_dir / "training_dataset.parquet")
    dhw_by_key = {
        (row.reef_cell_id, row.date): (float(row.dhw), bool(row.dhw_risk_flag))
        for row in training.itertuples()
    }

    manifest_sha = hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()

    with connection:  # one transaction; rolls back on any error
        connection.execute(
            "INSERT INTO region(region_id, name, crs, cell_size_m, boundary_geojson) "
            "VALUES (?,?,?,?,?) ON CONFLICT(region_id) DO UPDATE SET "
            "name=excluded.name, crs=excluded.crs, cell_size_m=excluded.cell_size_m",
            (
                region_id,
                region_name or region_id,
                region.get("utm_epsg", ""),
                float(region.get("cell_size_meters", 0.0)),
                None,
            ),
        )
        for feature in grid["features"]:
            cell_id = feature["properties"]["reef_cell_id"]
            geometry = feature["geometry"]
            min_lon, min_lat, max_lon, max_lat = _bounds(geometry)
            connection.execute(
                "INSERT INTO reef_cell(region_id, reef_cell_id, geometry_geojson, "
                "min_lon, min_lat, max_lon, max_lat) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(region_id, reef_cell_id) DO UPDATE SET "
                "geometry_geojson=excluded.geometry_geojson",
                (region_id, cell_id, json.dumps(geometry), min_lon, min_lat, max_lon, max_lat),
            )

        connection.execute(
            "UPDATE run SET status='superseded' WHERE region_id=? AND status='published'",
            (region_id,),
        )
        cursor = connection.execute(
            "INSERT INTO run(region_id, run_name, ingested_at, source_run_dir, "
            "manifest_sha256, schema_version, synthetic, validation_status, status) "
            "VALUES (?,?,?,?,?,?,?,?,'published')",
            (
                region_id,
                run_dir.name,
                now,
                str(run_dir),
                manifest_sha,
                SCHEMA_VERSION,
                int(synthetic),
                manifest.get("validation_status"),
            ),
        )
        run_id = cursor.lastrowid

        rows = []
        for row in predictions.itertuples():
            dhw, dhw_flag = dhw_by_key.get((row.reef_cell_id, row.date), (None, None))
            raw_contributions = row.top_contributing_features
            if raw_contributions is None:
                raw_contributions = []
            contributions = [
                {
                    "feature": c["feature"],
                    "value": None if c["value"] is None else float(c["value"]),
                    "contribution": float(c["contribution"]),
                }
                for c in raw_contributions
            ]
            rows.append(
                (
                    run_id,
                    row.reef_cell_id,
                    row.date.isoformat(),
                    None if row.probability is None else float(row.probability),
                    row.predicted_label,
                    row.confidence_band,
                    row.data_quality,
                    dhw,
                    None if dhw_flag is None else int(dhw_flag),
                    json.dumps(contributions),
                )
            )
        connection.executemany(
            "INSERT INTO prediction(run_id, reef_cell_id, date, probability, "
            "predicted_label, confidence_band, data_quality, dhw, dhw_risk_flag, "
            "top_contributing_features_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    return run_id


def _published_run(connection, region_id):
    row = connection.execute(
        "SELECT * FROM run WHERE region_id=? AND status='published'", (region_id,)
    ).fetchone()
    if row is None:
        return None
    return row


def list_regions(connection):
    regions = []
    for region in connection.execute("SELECT * FROM region ORDER BY region_id").fetchall():
        run = _published_run(connection, region["region_id"])
        dates, cell_count = [], 0
        if run is not None:
            dates = [
                r["date"]
                for r in connection.execute(
                    "SELECT DISTINCT date FROM prediction WHERE run_id=? ORDER BY date",
                    (run["run_id"],),
                ).fetchall()
            ]
            cell_count = connection.execute(
                "SELECT COUNT(*) AS n FROM reef_cell WHERE region_id=?", (region["region_id"],)
            ).fetchone()["n"]
        regions.append(
            {
                "region_id": region["region_id"],
                "name": region["name"],
                "published_run_id": None if run is None else run["run_id"],
                "available_dates": dates,
                "cell_count": cell_count,
            }
        )
    return regions


def region_grid(connection, region_id):
    cells = connection.execute(
        "SELECT reef_cell_id, geometry_geojson FROM reef_cell WHERE region_id=? "
        "ORDER BY reef_cell_id",
        (region_id,),
    ).fetchall()
    if not cells:
        return None
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"reef_cell_id": cell["reef_cell_id"]},
                "geometry": json.loads(cell["geometry_geojson"]),
            }
            for cell in cells
        ],
    }


def _available_dates(connection, run_id):
    return [
        r["date"]
        for r in connection.execute(
            "SELECT DISTINCT date FROM prediction WHERE run_id=? ORDER BY date", (run_id,)
        ).fetchall()
    ]


def _prediction_feature(cell_row, pred_row):
    """One GeoJSON feature merging a prediction (or a missing marker) onto a cell."""
    properties = {"reef_cell_id": cell_row["reef_cell_id"]}
    if pred_row is None:
        properties.update(
            date=None,
            probability=None,
            predicted_label=None,
            confidence_band="unavailable",
            data_quality=None,
            dhw=None,
            dhw_risk_flag=None,
            top_contributing_features=[],
            record_status="missing",
        )
    else:
        properties.update(
            date=pred_row["date"],
            probability=pred_row["probability"],
            predicted_label=pred_row["predicted_label"],
            confidence_band=pred_row["confidence_band"],
            data_quality=pred_row["data_quality"],
            dhw=pred_row["dhw"],
            dhw_risk_flag=None
            if pred_row["dhw_risk_flag"] is None
            else bool(pred_row["dhw_risk_flag"]),
            top_contributing_features=json.loads(pred_row["top_contributing_features_json"]),
            record_status="available",
        )
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": json.loads(cell_row["geometry_geojson"]),
    }


def region_predictions(connection, region_id, observation_date=None):
    """GeoJSON FeatureCollection for one region on one date, published run only.

    Returns None if the region has no published run. Raises ValueError for a
    date outside the run (the API maps that to 404). One feature per cell;
    cells without a record on the date are marked record_status 'missing',
    distinct from an insufficient-imagery abstention.
    """
    run = _published_run(connection, region_id)
    if run is None:
        return None
    dates = _available_dates(connection, run["run_id"])
    if not dates:
        raise ValueError("Published run has no predictions")
    if observation_date is None:
        selected = dates[-1]
    else:
        selected = date.fromisoformat(observation_date).isoformat()
        if selected not in dates:
            raise ValueError(f"No observations on {observation_date}")

    cells = connection.execute(
        "SELECT reef_cell_id, geometry_geojson FROM reef_cell WHERE region_id=? "
        "ORDER BY reef_cell_id",
        (region_id,),
    ).fetchall()
    predictions = {
        row["reef_cell_id"]: row
        for row in connection.execute(
            "SELECT * FROM prediction WHERE run_id=? AND date=?", (run["run_id"], selected)
        ).fetchall()
    }
    return {
        "type": "FeatureCollection",
        "metadata": {
            "region_id": region_id,
            "run_id": run["run_id"],
            "run_name": run["run_name"],
            "synthetic": bool(run["synthetic"]),
            "selected_date": selected,
            "available_dates": dates,
        },
        "features": [
            _prediction_feature(cell, predictions.get(cell["reef_cell_id"])) for cell in cells
        ],
    }


def cell_history(connection, region_id, reef_cell_id):
    """Per-cell time series across every date in the published run.

    This is the trend signal a ranger reasons about. Kept per-cell so it stays
    cheap as the date count grows.
    """
    run = _published_run(connection, region_id)
    if run is None:
        return None
    exists = connection.execute(
        "SELECT 1 FROM reef_cell WHERE region_id=? AND reef_cell_id=?",
        (region_id, reef_cell_id),
    ).fetchone()
    if exists is None:
        return None
    rows = connection.execute(
        "SELECT date, probability, predicted_label, confidence_band, data_quality, "
        "dhw, dhw_risk_flag FROM prediction WHERE run_id=? AND reef_cell_id=? ORDER BY date",
        (run["run_id"], reef_cell_id),
    ).fetchall()
    return {
        "region_id": region_id,
        "reef_cell_id": reef_cell_id,
        "run_id": run["run_id"],
        "history": [
            {
                "date": r["date"],
                "probability": r["probability"],
                "predicted_label": r["predicted_label"],
                "confidence_band": r["confidence_band"],
                "data_quality": r["data_quality"],
                "dhw": r["dhw"],
                "dhw_risk_flag": None if r["dhw_risk_flag"] is None else bool(r["dhw_risk_flag"]),
            }
            for r in rows
        ],
    }


def list_runs(connection, region_id):
    rows = connection.execute(
        "SELECT run_id, run_name, ingested_at, manifest_sha256, schema_version, "
        "synthetic, validation_status, status FROM run WHERE region_id=? "
        "ORDER BY run_id DESC",
        (region_id,),
    ).fetchall()
    return [
        {
            "run_id": r["run_id"],
            "run_name": r["run_name"],
            "ingested_at": r["ingested_at"],
            "manifest_sha256": r["manifest_sha256"],
            "schema_version": r["schema_version"],
            "synthetic": bool(r["synthetic"]),
            "validation_status": r["validation_status"],
            "status": r["status"],
        }
        for r in rows
    ]
