import hashlib
import json
import os
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from coral_bleaching.grid import grid_geojson
from coral_bleaching.schema import (
    SCHEMA_VERSION,
    TRAINING_SCHEMA,
    validate_predictions,
    validate_training,
)
from coral_bleaching.sources.aims import find_matching_label
from coral_bleaching.spectral import normalized_difference, rolling_index_shift


class PipelineError(RuntimeError):
    pass


def build_training_dataset(
    region, cells, observations, fetch_dhw_fn, aims_records, expected_dates=()
):
    rows, diagnostics = [], []
    for cell in cells:
        history = []
        cell_observations = observations.get(cell.reef_cell_id, [])
        by_date = {o.date: o for o in cell_observations}
        if len(by_date) != len(cell_observations):
            raise ValueError("Duplicate acquisitions on a cell/date; select before joining")
        dates = sorted(set(by_date) | set(expected_dates))
        if not dates:
            dates = [region.start_date]  # Explicit no-imagery row for an empty catalog.
        for obs_date in dates:
            if not region.start_date <= obs_date <= region.end_date:
                raise ValueError("Observation outside configured interval")
            try:
                dhw = fetch_dhw_fn(cell.centroid_lon, cell.centroid_lat, obs_date)
                if dhw is None:
                    raise ValueError("Missing DHW")
                survey, diagnostic = find_matching_label(
                    aims_records, cell.reef_cell_id, obs_date, region.aims_label_window_days
                )
            except Exception as exc:
                raise PipelineError(
                    f"Required source join failed for {cell.reef_cell_id}/{obs_date}"
                ) from exc
            if diagnostic:
                diagnostics.append(
                    {"reef_cell_id": cell.reef_cell_id, "date": obs_date.isoformat(), **diagnostic}
                )
            obs = by_date.get(obs_date)
            cloud = obs.cloud_cover_fraction if obs else None
            value = normalized_difference(obs.band_green, obs.band_red) if obs else None
            usable = (
                value is not None
                and cloud is not None
                and 0 <= cloud <= region.cloud_cover_drop_threshold
            )
            shift, start = None, obs_date
            if usable:
                if not obs.source_scene_id:
                    raise ValueError("Missing source scene identity")
                history.append((obs_date, value))
                shift, start = rolling_index_shift(
                    history, region.rolling_window, region.max_history_gap_days
                )
            rows.append(
                {
                    "reef_cell_id": cell.reef_cell_id,
                    "date": obs_date,
                    "ndci_like_index": value if usable else None,
                    "index_shift": shift,
                    "dhw": dhw,
                    "cloud_cover_fraction": cloud,
                    "data_quality": "ok" if usable else "insufficient",
                    "dhw_risk_flag": dhw >= region.dhw_risk_threshold,
                    "confirmed_label": survey.label if survey else None,
                    "survey_date": survey.survey_date if survey else None,
                    "survey_id": survey.survey_id if survey else None,
                    "source_scene_id": obs.source_scene_id if obs else None,
                    "feature_start_date": start,
                }
            )
    frame = pd.DataFrame(rows, columns=TRAINING_SCHEMA.names)
    validate_training(frame)
    frame.attrs["diagnostics"] = diagnostics
    return frame


def write_dataset(df, output_dir):
    table = validate_training(df)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    if len(df):
        ds.write_dataset(
            table,
            destination,
            format="parquet",
            partitioning=ds.partitioning(
                pa.schema([TRAINING_SCHEMA.field("reef_cell_id")]), flavor="hive"
            ),
        )
    else:
        pq.write_table(table, destination / "empty.parquet")


def read_dataset(path):
    table = ds.dataset(
        path, format="parquet", partitioning="hive", schema=TRAINING_SCHEMA
    ).to_table()
    frame = table.to_pandas()
    validate_training(frame)
    return frame


def publish_run(output_dir, frame, predictions, model, metrics, region, cells, provenance):
    """Atomically publish a complete immutable run on the same filesystem."""
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        write_dataset(frame, staging / "training_dataset.parquet")
        pq.write_table(
            validate_predictions(predictions),
            staging / "predictions.parquet",
        )
        if model is not None:
            model.booster_.save_model(str(staging / "model.txt"))
        (staging / "validation.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
        (staging / "grid.geojson").write_text(json.dumps(grid_geojson(cells)))
        loaded = read_dataset(staging / "training_dataset.parquet")
        loaded_predictions = pq.read_table(staging / "predictions.parquet").to_pandas()
        if set(zip(loaded.reef_cell_id, loaded.date)) != set(
            zip(loaded_predictions.reef_cell_id, loaded_predictions.date)
        ):
            raise ValueError("Prediction keys do not cover the raw dataset")
        checksums = {
            str(p.relative_to(staging)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in staging.rglob("*")
            if p.is_file()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "region": region.to_dict(),
            "model_available": model is not None,
            "row_count": len(frame),
            "software": {
                name: version(name)
                for name in [
                    "coral-bleaching-detection",
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "rasterio",
                    "lightgbm",
                    "scikit-learn",
                    "pystac-client",
                    "pyproj",
                    "shapely",
                ]
            },
            "provenance": provenance,
            "diagnostics": frame.attrs.get("diagnostics", []),
            "sha256": checksums,
            "validation_status": metrics["status"],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        # Recheck immediately before rename; never replace a previously published run.
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return destination
