from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from coral_bleaching.labels import build_training_frame
from coral_bleaching.predict import confidence_band, generate_predictions, write_predictions
from coral_bleaching.schema import PREDICTION_SCHEMA, TRAINING_SCHEMA, empty_frame
from coral_bleaching.train import cross_validate, leave_one_survey_date_out_splits, train_model

PARAMS = {"n_estimators": 10, "min_child_samples": 1}


def test_scored_folds_and_single_class_are_honest(training_frame):
    frame = build_training_frame(training_frame)
    results = cross_validate(frame, PARAMS)
    assert results["scored_folds"] == 3
    assert all(0 <= f["roc_auc"] <= 1 for f in results["folds"])
    one_class = frame[frame.label == "healthy"]
    skipped = cross_validate(one_class, PARAMS)
    assert skipped["status"] == "not_evaluable"
    assert len(skipped["folds"]) == 3 and all(f["roc_auc"] is None for f in skipped["folds"])
    assert cross_validate(frame.iloc[:0])["status"] == "not_evaluable"


def test_split_preserves_surveys_and_purges_weak_neighbors(training_frame):
    rows = training_frame.copy()
    extra = rows.iloc[[0]].copy()
    extra["date"] = extra.date + timedelta(days=7)
    extra["source_scene_id"] = "other-date"
    weak = rows.iloc[[1]].copy()
    weak["date"] = weak.date + timedelta(days=10)
    weak["source_scene_id"] = "weak-scene"
    weak[["confirmed_label", "survey_id", "survey_date"]] = None
    same_scene = rows.iloc[[8]].copy()
    same_scene["reef_cell_id"] = "far-weak"
    same_scene["source_scene_id"] = "scene0"
    same_scene[["confirmed_label", "survey_id", "survey_date"]] = None
    frame = build_training_frame(pd.concat([rows, extra, weak, same_scene], ignore_index=True))
    day, train_idx, test_idx = next(leave_one_survey_date_out_splits(frame))
    assert len(test_idx) == 9
    assert set(test_idx) == set(range(8)) | {24}
    assert not {24, 25, 26} & set(train_idx)
    assert set(train_idx) == set(range(8, 24))
    assert frame.iloc[test_idx].survey_date.nunique() == 1


def test_predictions_abstain_roundtrip_and_signed_contributions(training_frame, tmp_path):
    frame = training_frame.copy()
    model = train_model(build_training_frame(frame), PARAMS)
    frame.loc[0, "data_quality"] = "insufficient"
    frame.loc[0, "ndci_like_index"] = np.nan
    result = generate_predictions(frame, model)
    assert len(result) == len(frame)
    assert pd.isna(result.iloc[0].probability) and pd.isna(result.iloc[0].predicted_label)
    assert result.iloc[0].confidence_band == "unavailable"
    assert len(result.iloc[0].top_contributing_features) == 0
    assert all(
        set(v) == {"feature", "value", "contribution"}
        for v in result.iloc[1].top_contributing_features
    )
    path = tmp_path / "predictions.parquet"
    write_predictions(result, path)
    assert pq.read_table(path).schema.remove_metadata() == PREDICTION_SCHEMA
    assert pq.read_table(path).column("probability").null_count == 1


def test_no_model_calls_for_empty_or_insufficient(training_frame):
    model = Mock()
    training_frame["data_quality"] = "insufficient"
    assert generate_predictions(training_frame, model).probability.isna().all()
    assert generate_predictions(empty_frame(TRAINING_SCHEMA), model).empty
    assert model.mock_calls == []


@pytest.mark.parametrize(
    "value,band", [(0.5, "low"), (0.6, "medium"), (0.8, "high"), (0.05, "high")]
)
def test_confidence_boundaries(value, band):
    assert confidence_band(value) == band


def test_source_interval_overlap_and_survey_weighting(training_frame, monkeypatch):
    frame = build_training_frame(training_frame)
    frame.loc[8, "feature_start_date"] = frame.loc[0, "date"]
    _, train_idx, _ = next(leave_one_survey_date_out_splits(frame))
    assert 8 not in train_idx
    # Two images from the same source survey share one unit of evaluation weight.
    extra = frame.iloc[[0]].copy()
    extra["date"] = extra.date + timedelta(days=1)
    extra["source_scene_id"] = "next-scene"
    frame = pd.concat([frame, extra], ignore_index=True)
    import coral_bleaching.train as module

    actual = module.roc_auc_score
    captured = []

    def capture(truth, score, sample_weight):
        captured.append(sample_weight.tolist())
        return actual(truth, score, sample_weight=sample_weight)

    monkeypatch.setattr(module, "roc_auc_score", capture)
    report = cross_validate(frame, PARAMS)
    assert report["scored_folds"] == 3
    assert captured[0][0] == 0.5 and captured[0][-1] == 0.5
    assert sum(captured[0]) == 8


def test_saved_model_matches_reloaded_dataset(training_frame, tmp_path):
    from lightgbm import Booster

    from coral_bleaching.pipeline import read_dataset, write_dataset

    write_dataset(training_frame, tmp_path / "raw")
    reloaded = read_dataset(tmp_path / "raw")
    model = train_model(build_training_frame(reloaded), PARAMS)
    path = tmp_path / "model.txt"
    model.booster_.save_model(str(path))
    expected = generate_predictions(reloaded, model)
    actual = generate_predictions(reloaded, Booster(model_file=str(path)))
    np.testing.assert_allclose(actual.probability, expected.probability)


def test_single_class_test_fold_is_reported(training_frame):
    frame = build_training_frame(training_frame)
    frame.loc[:7, ["label", "confirmed_label"]] = "healthy"
    report = cross_validate(frame, PARAMS)
    assert report["folds"][0]["reason"] == "single_class_test"
    assert report["skipped_folds"] == 1
