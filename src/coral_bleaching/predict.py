import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from coral_bleaching.schema import (
    FEATURE_COLUMNS,
    PREDICTION_SCHEMA,
    eligible_rows,
    feature_matrix,
    validate_predictions,
)


def confidence_band(probability):
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must lie in [0,1]")
    distance = abs(probability - 0.5)
    if distance < 0.1 - 1e-12:
        return "low"
    if distance < 0.3 - 1e-12:
        return "medium"
    return "high"


def generate_predictions(df, model):
    df = df.reset_index(drop=True)
    mask = eligible_rows(df)
    results = [
        {
            "reef_cell_id": row.reef_cell_id,
            "date": row.date,
            "probability": None,
            "predicted_label": None,
            "top_contributing_features": [],
            "confidence_band": "unavailable",
            "data_quality": "insufficient",
        }
        for row in df.itertuples()
    ]
    if mask.any():
        if model is None:
            raise ValueError("Usable observations require a trained model")
        features = feature_matrix(df.loc[mask])
        # Booster files avoid pickle execution when loading a persisted model.
        booster = model.booster_ if hasattr(model, "booster_") else model
        if booster.feature_name() != FEATURE_COLUMNS:
            raise ValueError("Model feature names/order do not match schema v2")
        probabilities = np.asarray(booster.predict(features))
        contributions = np.asarray(booster.predict(features, pred_contrib=True))
        for pos, idx in enumerate(np.flatnonzero(mask)):
            probability = float(probabilities[pos])
            band = confidence_band(probability)
            ranked = sorted(
                zip(FEATURE_COLUMNS, contributions[pos][:-1]), key=lambda p: abs(p[1]), reverse=True
            )[:3]
            top = [
                {
                    "feature": name,
                    "value": float(df.at[idx, name]) if pd.notna(df.at[idx, name]) else None,
                    "contribution": float(value),
                }
                for name, value in ranked
            ]
            results[idx].update(
                probability=probability,
                predicted_label="bleached" if probability >= 0.5 else "healthy",
                top_contributing_features=top,
                confidence_band=band,
                data_quality="ok",
            )
    return pa.Table.from_pylist(results, schema=PREDICTION_SCHEMA).to_pandas()


def write_predictions(df, path):
    pq.write_table(validate_predictions(df), path)
