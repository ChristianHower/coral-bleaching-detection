import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from coral_bleaching.schema import eligible_rows, feature_matrix


def train_model(df, params=None):
    if df.empty or df.label.nunique() != 2:
        raise ValueError("Model training requires both bleached and healthy examples")
    if not eligible_rows(df).all() or not df.label.isin(["bleached", "healthy"]).all():
        raise ValueError("Training contains ineligible observations")
    if not np.isfinite(df.sample_weight).all() or (df.sample_weight <= 0).any():
        raise ValueError("Invalid sample weights")
    settings = {"n_estimators": 100, "random_state": 42, "n_jobs": 1, "verbosity": -1}
    settings.update(params or {})
    model = LGBMClassifier(**settings)
    model.fit(
        feature_matrix(df), (df.label == "bleached").astype(int), sample_weight=df.sample_weight
    )
    return model


def leave_one_survey_date_out_splits(df, embargo_days=14):
    if embargo_days < 0:
        raise ValueError("Embargo must be nonnegative")
    df = df.reset_index(drop=True)
    confirmed = df.confirmed_label.notna() & eligible_rows(df)
    if df.loc[confirmed, ["survey_date", "survey_id"]].isna().any().any():
        raise ValueError("Confirmed observations require survey provenance")
    if df.loc[eligible_rows(df), ["source_scene_id", "feature_start_date"]].isna().any().any():
        raise ValueError("Usable observations require feature provenance")
    starts, ends = pd.to_datetime(df.feature_start_date), pd.to_datetime(df.date)
    if (starts > ends).any():
        raise ValueError("Invalid feature interval")
    for held_out in sorted(df.loc[confirmed, "survey_date"].unique()):
        test_mask = confirmed & df.survey_date.eq(held_out)
        test = df.loc[test_mask]
        train_mask = eligible_rows(df) & ~test_mask
        train_mask &= ~df.survey_id.isin(test.survey_id)
        train_mask &= ~df.source_scene_id.isin(test.source_scene_id)
        for idx in test.index:
            lo = starts.iloc[idx].to_datetime64() - np.timedelta64(embargo_days, "D")
            hi = ends.iloc[idx].to_datetime64() + np.timedelta64(embargo_days, "D")
            train_mask &= ~((starts <= hi) & (ends >= lo))
        yield held_out, np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def cross_validate(df, params=None, embargo_days=14):
    df = df.reset_index(drop=True)
    folds = []
    for day, train_idx, test_idx in leave_one_survey_date_out_splits(df, embargo_days):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        result = {
            "survey_date": str(day),
            "status": "skipped",
            "reason": None,
            "train_count": len(train),
            "test_count": len(test),
            "train_classes": train.label.value_counts().to_dict(),
            "test_classes": test.confirmed_label.value_counts().to_dict(),
            "unique_test_surveys": int(test.survey_id.nunique()),
            "roc_auc": None,
            "dhw_roc_auc": None,
            "dhw_balanced_accuracy": None,
        }
        if train.empty or test.empty:
            result["reason"] = "empty_train_or_test"
        elif train.label.nunique() < 2:
            result["reason"] = "single_class_train"
        elif test.confirmed_label.nunique() < 2:
            result["reason"] = "single_class_test"
        else:
            model = train_model(train, params)
            probs = model.predict_proba(feature_matrix(test))[:, 1]
            truth = (test.confirmed_label == "bleached").astype(int)
            weights = 1 / test.groupby("survey_id")["survey_id"].transform("count")
            result.update(
                status="scored",
                roc_auc=float(roc_auc_score(truth, probs, sample_weight=weights)),
                dhw_roc_auc=float(roc_auc_score(truth, test.dhw, sample_weight=weights)),
                dhw_balanced_accuracy=float(
                    balanced_accuracy_score(
                        truth, test.dhw_risk_flag.astype(int), sample_weight=weights
                    )
                ),
            )
        folds.append(result)
    scored = [f["roc_auc"] for f in folds if f["status"] == "scored"]
    return {
        "status": "evaluated" if scored else "not_evaluable",
        "folds": folds,
        "scored_folds": len(scored),
        "skipped_folds": len(folds) - len(scored),
        "mean_roc_auc": float(np.mean(scored)) if scored else None,
        "embargo_days": embargo_days,
        "evaluation_weighting": "equal_per_survey_record",
    }
