import copy
import hashlib
import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse

from coral_bleaching.schema import PREDICTION_SCHEMA, validate_predictions

PREDICTION_FIELDS = (
    "date",
    "probability",
    "predicted_label",
    "confidence_band",
    "data_quality",
    "top_contributing_features",
)


def _json_value(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _script_json(value):
    return (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _load_verified_run(run_dir_value, manifest_mtime, grid_mtime, predictions_mtime):
    del manifest_mtime, grid_mtime, predictions_mtime
    run_dir = Path(run_dir_value)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported or missing run schema version")
    checksums = manifest.get("sha256", {})
    for name in ("grid.geojson", "predictions.parquet"):
        if checksums.get(name) != _sha256(run_dir / name):
            raise ValueError(f"Run checksum failed for {name}")

    grid = json.loads((run_dir / "grid.geojson").read_text())
    if grid.get("type") != "FeatureCollection" or not isinstance(grid.get("features"), list):
        raise ValueError("Grid is not a GeoJSON FeatureCollection")
    cell_ids = [feature.get("properties", {}).get("reef_cell_id") for feature in grid["features"]]
    if any(not cell_id for cell_id in cell_ids) or len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Grid requires unique, nonempty reef_cell_id values")

    predictions_path = run_dir / "predictions.parquet"
    if pq.read_schema(predictions_path).remove_metadata() != PREDICTION_SCHEMA:
        raise ValueError("Prediction file does not match schema version 2")
    dates = sorted(set(pq.read_table(predictions_path, columns=["date"])["date"].to_pylist()))
    if not dates:
        raise ValueError("Prediction file contains no observations")
    return grid, frozenset(cell_ids), tuple(dates)


@lru_cache(maxsize=16)
def _load_prediction_date(predictions_path_value, predictions_mtime, observation_date):
    del predictions_mtime
    predictions = pq.read_table(
        predictions_path_value,
        filters=[("date", "=", observation_date)],
    ).to_pandas()
    validate_predictions(predictions)
    return predictions


def load_run_geojson(runs_root, run_name, observation_date=None):
    root = Path(runs_root).resolve()
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise HTTPException(status_code=404, detail="Run not found")
    run_dir = (root / run_name).resolve()
    if run_dir.parent != root or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_name}' not found")

    manifest_path = run_dir / "manifest.json"
    grid_path = run_dir / "grid.geojson"
    predictions_path = run_dir / "predictions.parquet"
    missing = [
        path.name for path in (manifest_path, grid_path, predictions_path) if not path.is_file()
    ]
    if missing:
        names = ", ".join(missing)
        raise HTTPException(status_code=404, detail=f"Run '{run_name}' is missing {names}")

    try:
        grid, grid_cell_ids, available_dates = _load_verified_run(
            str(run_dir),
            manifest_path.stat().st_mtime_ns,
            grid_path.stat().st_mtime_ns,
            predictions_path.stat().st_mtime_ns,
        )
        if observation_date is None:
            selected_date = available_dates[-1]
        else:
            selected_date = date.fromisoformat(observation_date)
            if selected_date not in available_dates:
                raise HTTPException(
                    status_code=404,
                    detail=f"Run '{run_name}' has no observations on {observation_date}",
                )
        predictions = _load_prediction_date(
            str(predictions_path), predictions_path.stat().st_mtime_ns, selected_date
        )
    except HTTPException:
        raise
    except (KeyError, OSError, TypeError, ValueError, pa.ArrowInvalid) as exc:
        raise HTTPException(status_code=422, detail=f"Run '{run_name}' is invalid: {exc}") from exc

    prediction_cell_ids = set(predictions["reef_cell_id"])
    unknown_cells = prediction_cell_ids - grid_cell_ids
    if unknown_cells:
        raise HTTPException(
            status_code=422,
            detail=f"Run '{run_name}' has predictions for cells absent from its grid",
        )
    by_cell = predictions.set_index("reef_cell_id").to_dict("index")
    grid = copy.deepcopy(grid)
    grid["metadata"] = {
        "selected_date": selected_date.isoformat(),
        "available_dates": [value.isoformat() for value in available_dates],
    }

    for feature in grid.get("features", []):
        properties = feature.setdefault("properties", {})
        cell_id = properties.get("reef_cell_id")
        row = by_cell.get(cell_id)
        if row is None:
            properties.update(
                date=None,
                probability=None,
                predicted_label=None,
                confidence_band="unavailable",
                data_quality=None,
                top_contributing_features=[],
                record_status="missing",
            )
            continue
        properties.update(
            {field: _json_value(row[field]) for field in PREDICTION_FIELDS}
            | {"record_status": "available"}
        )
    return grid


def create_app(runs_root="runs", default_run="demo-001"):
    app = FastAPI(title="Coral bleaching prediction map", docs_url=None, redoc_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    index_path = Path(__file__).with_name("static") / "index.html"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        page = index_path.read_text()
        return page.replace('"__DEFAULT_RUN__"', _script_json(default_run))

    @app.get("/api/runs/{run_name}")
    def run_predictions(
        run_name: str, observation_date: str | None = Query(default=None, alias="date")
    ):
        return load_run_geojson(runs_root, run_name, observation_date)

    return app


app = create_app()


def serve(runs_root, run_name, host, port):
    import uvicorn

    uvicorn.run(create_app(runs_root, run_name), host=host, port=port)
