import pandas as pd

from coral_bleaching.schema import eligible_rows

CONFIRMED_WEIGHT = 1.0
WEAK_LABEL_WEIGHT = 0.5


def build_training_frame(df):
    if not df.confirmed_label.dropna().isin(["bleached", "healthy"]).all():
        raise ValueError("Invalid confirmed label")
    result = df.loc[eligible_rows(df)].copy().reset_index(drop=True)
    confirmed = result.confirmed_label.notna()
    result["label"] = result.confirmed_label.where(
        confirmed, result.dhw_risk_flag.map({True: "bleached", False: "healthy"})
    )
    result["sample_weight"] = pd.Series(WEAK_LABEL_WEIGHT, index=result.index).where(
        ~confirmed, CONFIRMED_WEIGHT
    )
    return result
