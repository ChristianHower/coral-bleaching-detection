import json
from datetime import date, timedelta

import pandas as pd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from coral_bleaching.cli import demo
from coral_bleaching.config import RegionConfig
from coral_bleaching.grid import ReefCell
from coral_bleaching.pipeline import (
    PipelineError,
    build_training_dataset,
    publish_run,
    read_dataset,
    write_dataset,
)
from coral_bleaching.schema import PREDICTION_SCHEMA, TRAINING_SCHEMA, empty_frame
from coral_bleaching.sources.aims import Survey
from coral_bleaching.sources.sentinel import Observation


def setup_data():
    region = RegionConfig("heron", "x", date(2020, 1, 1), date(2020, 3, 1))
    cell = ReefCell("c", box(151.91, -23.45, 151.92, -23.44), -23.445, 151.915)
    obs = [
        Observation(date(2020, 1, 1) + timedelta(days=7 * i), f"s{i}", 0.06, 0.02, 0.1)
        for i in range(6)
    ]
    return region, cell, obs


def test_pipeline_nulls_history_and_provenance(tmp_path):
    region, cell, obs = setup_data()
    survey = Survey("a", "c", obs[2].date, "bleached")
    raw = build_training_dataset(region, [cell], {"c": obs}, lambda *a: 6.0, [survey])
    assert pd.isna(raw.iloc[0].index_shift)
    assert raw.iloc[-1].index_shift == pytest.approx(0)
    assert raw.iloc[-1].feature_start_date == obs[0].date
    assert raw.iloc[2].survey_date == survey.survey_date
    obs[2] = Observation(obs[2].date, "s2", None, None, None)
    raw = build_training_dataset(region, [cell], {"c": obs}, lambda *a: 6.0, [survey])
    assert raw.iloc[2].data_quality == "insufficient"
    assert raw.iloc[2].confirmed_label == "bleached"
    assert raw.index_shift.isna().all()
    out = tmp_path / "dataset"
    write_dataset(raw, out)
    assert len(read_dataset(out)) == 6
    with pytest.raises(FileExistsError):
        write_dataset(raw, out)


def test_required_source_failure_and_empty_catalog():
    region, cell, obs = setup_data()
    with pytest.raises(PipelineError):
        build_training_dataset(region, [cell], {"c": obs}, lambda *a: None, [])
    raw = build_training_dataset(region, [cell], {}, lambda *a: 1.0, [])
    assert len(raw) == 1 and raw.iloc[0].data_quality == "insufficient"


def test_empty_parquet_retains_schema(tmp_path):
    path = tmp_path / "empty"
    write_dataset(empty_frame(TRAINING_SCHEMA), path)
    assert read_dataset(path).empty
    assert pq.read_schema(path / "empty.parquet").remove_metadata() == TRAINING_SCHEMA


def test_integration_demo_round_trip_and_immutable_output(tmp_path):
    output = demo(tmp_path / "demo")
    raw = read_dataset(output / "training_dataset.parquet")
    predictions = pq.read_table(output / "predictions.parquet")
    assert predictions.schema.remove_metadata() == PREDICTION_SCHEMA
    assert len(raw) == predictions.num_rows
    assert predictions.column("probability").null_count > 0
    metrics = json.loads((output / "validation.json").read_text())
    assert metrics["scored_folds"] == 3
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["provenance"]["synthetic"] is True
    assert manifest["sha256"] and manifest["schema_version"] == 2
    with pytest.raises(FileExistsError):
        demo(output)


def test_failed_publication_removes_staging(tmp_path, monkeypatch):
    region, cell, obs = setup_data()
    raw = build_training_dataset(region, [cell], {"c": obs}, lambda *a: 1.0, [])

    def fail(*args, **kwargs):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(pq, "write_table", fail)
    with pytest.raises(RuntimeError, match="disk failure"):
        publish_run(
            tmp_path / "run",
            raw,
            empty_frame(PREDICTION_SCHEMA),
            None,
            {"status": "not_evaluable"},
            region,
            [cell],
            {},
        )
    assert list(tmp_path.iterdir()) == []
