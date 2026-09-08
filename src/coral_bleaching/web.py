import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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


def load_run_geojson(runs_root, run_name):
    root = Path(runs_root).resolve()
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise HTTPException(status_code=404, detail="Run not found")
    run_dir = (root / run_name).resolve()
    if run_dir.parent != root or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_name}' not found")

    grid_path = run_dir / "grid.geojson"
    predictions_path = run_dir / "predictions.parquet"
    missing = [path.name for path in (grid_path, predictions_path) if not path.is_file()]
    if missing:
        names = ", ".join(missing)
        raise HTTPException(status_code=404, detail=f"Run '{run_name}' is missing {names}")

    grid = json.loads(grid_path.read_text())
    predictions = pq.read_table(predictions_path).to_pandas()
    # The grid has no date dimension. Keep the latest observation for each cell
    # before joining so every geometry appears at most once in the map response.
    latest = predictions.sort_values("date").groupby("reef_cell_id", as_index=False).tail(1)
    by_cell = latest.set_index("reef_cell_id").to_dict("index")

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
                data_quality="insufficient",
                top_contributing_features=[],
            )
            continue
        properties.update({field: _json_value(row[field]) for field in PREDICTION_FIELDS})
    return grid


def create_app(runs_root="runs", default_run="demo-001"):
    app = FastAPI(title="Coral bleaching prediction map", docs_url=None, redoc_url=None)
    index_path = Path(__file__).with_name("static") / "index.html"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        page = index_path.read_text()
        return page.replace('"__DEFAULT_RUN__"', json.dumps(default_run))

    @app.get("/api/runs/{run_name}")
    def run_predictions(run_name: str):
        return load_run_geojson(runs_root, run_name)

    return app


app = create_app()


def serve(runs_root, run_name, host, port):
    import uvicorn

    uvicorn.run(create_app(runs_root, run_name), host=host, port=port)
