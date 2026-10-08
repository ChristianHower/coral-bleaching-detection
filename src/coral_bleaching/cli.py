import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

from shapely.geometry import box

from coral_bleaching.config import RegionConfig
from coral_bleaching.grid import generate_grid, load_boundary, reproject
from coral_bleaching.labels import build_training_frame
from coral_bleaching.pipeline import build_training_dataset, publish_run, read_dataset
from coral_bleaching.predict import generate_predictions, write_predictions
from coral_bleaching.sources.aims import Survey, load_aims_labels
from coral_bleaching.sources.noaa import NOAAClient
from coral_bleaching.sources.sentinel import EarthSearchClient, Observation
from coral_bleaching.train import cross_validate, train_model


def run_batch(region, cells, observations, fetch_dhw_fn, surveys, output, provenance, params=None):
    raw = build_training_dataset(region, cells, observations, fetch_dhw_fn, surveys)
    training = build_training_frame(raw)
    metrics = cross_validate(training, params, region.aims_label_window_days)
    model = train_model(training, params) if len(training) else None
    predictions = generate_predictions(raw, model)
    return publish_run(output, raw, predictions, model, metrics, region, cells, provenance)


def demo(output):
    """Synthetic mechanics demonstration; never evidence of field accuracy."""
    region = RegionConfig(
        "synthetic_heron", "synthetic", date(2020, 1, 1), date(2020, 9, 1), cell_size_meters=75
    )
    boundary = reproject(box(389025, 7406850, 389175, 7407000), region.utm_epsg, "EPSG:4326")
    cells = generate_grid(boundary, region.cell_size_meters, region.reef_id)
    observations, surveys = {}, []
    for i, cell in enumerate(cells):
        observations[cell.reef_cell_id] = []
        for block in range(3):
            start = date(2020, 1, 1) + timedelta(days=90 * block)
            label = "bleached" if i % 2 else "healthy"
            surveys.append(
                Survey(f"survey-{i}-{block}", cell.reef_cell_id, start + timedelta(days=21), label)
            )
            for n in range(6):
                day = start + timedelta(days=7 * n)
                cloudy = i == 0 and n == 3
                missing = i == 1 and n == 2
                green = (0.025 if label == "bleached" else 0.06) + n * 0.0002
                observations[cell.reef_cell_id].append(
                    Observation(
                        day,
                        f"synthetic-scene-{block}-{n}",
                        None if missing else green,
                        None if missing else 0.03,
                        None if missing else (0.9 if cloudy else 0.1),
                    )
                )
    return run_batch(
        region,
        cells,
        observations,
        lambda lon, lat, day: 5.0 if day.day >= 15 else 1.0,
        surveys,
        output,
        {"synthetic": True, "purpose": "Software demonstration only"},
        {"n_estimators": 20, "min_child_samples": 2},
    )


def load_region(path):
    data = json.loads(Path(path).read_text())
    for key in ("start_date", "end_date"):
        data[key] = date.fromisoformat(data[key])
    boundary = Path(data["boundary_geojson_path"])
    if not boundary.is_absolute():
        data["boundary_geojson_path"] = str((Path(path).resolve().parent / boundary).resolve())
    return RegionConfig(**data)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline coral bleaching research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("demo", help="Run synthetic data through the entire pipeline offline")
    sample.add_argument("--output", required=True)
    live = sub.add_parser(
        "run", help="Run live public imagery/NOAA using supplied boundary and normalized surveys"
    )
    live.add_argument("--config", required=True)
    live.add_argument("--surveys", required=True)
    live.add_argument("--output", required=True)
    inference = sub.add_parser(
        "predict", help="Predict from an existing raw dataset and LightGBM text model"
    )
    inference.add_argument("--dataset", required=True)
    inference.add_argument("--model", required=True)
    inference.add_argument("--output", required=True)
    ui = sub.add_parser("ui", help="Serve the local read-only prediction map")
    ui.add_argument("--runs-root", default="runs")
    ui.add_argument("--run", default="demo-001")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8000)
    ingest = sub.add_parser(
        "ingest", help="Load a verified run directory into the SQLite serving store"
    )
    ingest.add_argument("--db", required=True, help="SQLite database path (created if absent)")
    ingest.add_argument("--run-dir", required=True, help="Path to a published runs/<name>/ dir")
    ingest.add_argument("--region-name", default=None, help="Human-readable region name")
    region_ui = sub.add_parser(
        "serve-region", help="Serve the region-keyed read API and time-aware map from SQLite"
    )
    region_ui.add_argument("--db", required=True)
    region_ui.add_argument("--region", default=None, help="Default region to open")
    region_ui.add_argument("--host", default="127.0.0.1")
    region_ui.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        if args.command == "ui":
            from coral_bleaching.web import serve

            serve(args.runs_root, args.run, args.host, args.port)
            return
        if args.command == "ingest":
            from coral_bleaching.store import connect, ingest_run, init_db

            connection = connect(args.db)
            init_db(connection)
            run_id = ingest_run(connection, args.run_dir, args.region_name)
            connection.close()
            print(f"Ingested {args.run_dir} as run_id {run_id}")
            return
        if args.command == "serve-region":
            from coral_bleaching.web import serve_region

            serve_region(args.db, args.host, args.port, args.region)
            return
        if args.command == "demo":
            output = demo(args.output)
        elif args.command == "predict":
            from lightgbm import Booster

            if Path(args.output).exists():
                raise FileExistsError(args.output)
            frame = read_dataset(args.dataset)
            write_predictions(
                generate_predictions(frame, Booster(model_file=args.model)), args.output
            )
            output = args.output
        else:
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            region = load_region(args.config)
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
                raise ValueError("Boundary generated no eligible cells")
            surveys = load_aims_labels(args.surveys, cells)
            imagery = EarthSearchClient(collection=region.collection)
            noaa = NOAAClient()
            observations = imagery.observations(cells, boundary, region.start_date, region.end_date)
            provenance = {
                "synthetic": False,
                "sentinel": imagery.provenance,
                "noaa": noaa.provenance,
                "survey_sha256": hashlib.sha256(Path(args.surveys).read_bytes()).hexdigest(),
                "boundary_sha256": hashlib.sha256(
                    Path(region.boundary_geojson_path).read_bytes()
                ).hexdigest(),
            }
            output = run_batch(
                region, cells, observations, noaa.fetch_dhw, surveys, args.output, provenance
            )
        print(f"Wrote {output}")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Build failed: {exc}\n")


if __name__ == "__main__":
    main()
