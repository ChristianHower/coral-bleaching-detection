import numpy as np
import pandas as pd
import pyarrow as pa

SCHEMA_VERSION = 2
FEATURE_COLUMNS = ["ndci_like_index", "index_shift", "dhw", "cloud_cover_fraction"]
TRAINING_SCHEMA = pa.schema(
    [
        pa.field("reef_cell_id", pa.string(), False),
        pa.field("date", pa.date32(), False),
        pa.field("ndci_like_index", pa.float64()),
        pa.field("index_shift", pa.float64()),
        pa.field("dhw", pa.float64(), False),
        pa.field("cloud_cover_fraction", pa.float64()),
        pa.field("data_quality", pa.string(), False),
        pa.field("dhw_risk_flag", pa.bool_(), False),
        pa.field("confirmed_label", pa.string()),
        pa.field("survey_date", pa.date32()),
        pa.field("survey_id", pa.string()),
        pa.field("source_scene_id", pa.string()),
        pa.field("feature_start_date", pa.date32(), False),
    ]
)
CONTRIBUTION = pa.struct(
    [("feature", pa.string()), ("value", pa.float64()), ("contribution", pa.float64())]
)
PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("reef_cell_id", pa.string(), False),
        pa.field("date", pa.date32(), False),
        pa.field("probability", pa.float64()),
        pa.field("predicted_label", pa.string()),
        pa.field("top_contributing_features", pa.list_(CONTRIBUTION), False),
        pa.field("confidence_band", pa.string(), False),
        pa.field("data_quality", pa.string(), False),
    ]
)


def empty_frame(schema):
    return schema.empty_table().to_pandas()


def feature_matrix(df):
    """Normalize nullable numeric dtypes without imputing missing observations."""
    return df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)


def eligible_rows(df):
    numeric = feature_matrix(df)
    current = numeric[["ndci_like_index", "dhw", "cloud_cover_fraction"]]
    valid = np.isfinite(current).all(axis=1)
    valid &= numeric["index_shift"].isna() | np.isfinite(numeric["index_shift"])
    valid &= numeric["cloud_cover_fraction"].between(0, 1) & (numeric["dhw"] >= 0)
    return df["data_quality"].eq("ok") & valid


def validate_training(df):
    if set(df.columns) != set(TRAINING_SCHEMA.names):
        raise ValueError("Training columns differ from schema v2")
    if df.duplicated(["reef_cell_id", "date"]).any():
        raise ValueError("Duplicate cell/date rows")
    if not df["data_quality"].isin(["ok", "insufficient"]).all():
        raise ValueError("Unknown data_quality")
    if not df["confirmed_label"].dropna().isin(["bleached", "healthy"]).all():
        raise ValueError("Invalid confirmed_label")
    presence = df[["confirmed_label", "survey_date", "survey_id"]].notna()
    if not (presence.all(axis=1) | ~presence.any(axis=1)).all():
        raise ValueError("Survey provenance must be populated together")
    if not np.isfinite(df["dhw"].astype(float)).all() or (df["dhw"] < 0).any():
        raise ValueError("Invalid DHW")
    if (pd.to_datetime(df["feature_start_date"]) > pd.to_datetime(df["date"])).any():
        raise ValueError("Features cannot use future acquisitions")
    ok = df["data_quality"].eq("ok")
    if (ok & (~eligible_rows(df) | df["source_scene_id"].isna())).any():
        raise ValueError("Usable rows require valid features and scene provenance")
    table = pa.Table.from_pandas(df, schema=TRAINING_SCHEMA, preserve_index=False)
    for field in TRAINING_SCHEMA:
        if not field.nullable and table[field.name].null_count:
            raise ValueError(f"Required field {field.name} is null")
    return table


def validate_predictions(df):
    if set(df.columns) != set(PREDICTION_SCHEMA.names):
        raise ValueError("Prediction columns differ from schema v2")
    if df.duplicated(["reef_cell_id", "date"]).any():
        raise ValueError("Duplicate prediction keys")
    if not df.data_quality.isin(["ok", "insufficient"]).all():
        raise ValueError("Unknown prediction quality")
    ok = df.data_quality.eq("ok")
    if not df.loc[ok, "probability"].between(0, 1).all():
        raise ValueError("Usable predictions require finite probabilities")
    if not df.loc[ok, "predicted_label"].isin(["bleached", "healthy"]).all():
        raise ValueError("Invalid predicted label")
    if not df.loc[ok, "confidence_band"].isin(["low", "medium", "high"]).all():
        raise ValueError("Invalid confidence band")
    if df.loc[~ok, ["probability", "predicted_label"]].notna().any().any():
        raise ValueError("Insufficient observations must abstain")
    if not df.loc[~ok, "confidence_band"].eq("unavailable").all():
        raise ValueError("Insufficient observations have unavailable confidence")
    for row in df.itertuples():
        if row.top_contributing_features is None:
            raise ValueError("Missing contribution list")
        if row.data_quality == "insufficient" and len(row.top_contributing_features):
            raise ValueError("Insufficient observations cannot have contributions")
        for entry in row.top_contributing_features:
            if entry["feature"] not in FEATURE_COLUMNS or not np.isfinite(entry["contribution"]):
                raise ValueError("Invalid feature contribution")
    table = pa.Table.from_pandas(df, schema=PREDICTION_SCHEMA, preserve_index=False)
    for field in PREDICTION_SCHEMA:
        if not field.nullable and table[field.name].null_count:
            raise ValueError(f"Required field {field.name} is null")
    return table
