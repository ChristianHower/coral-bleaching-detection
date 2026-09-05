import socket
from datetime import date, timedelta

import pandas as pd
import pytest

from coral_bleaching.schema import TRAINING_SCHEMA


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network disabled in the test suite")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture
def training_frame():
    rows = []
    for block in range(3):
        day = date(2020, 1, 1) + timedelta(days=90 * block)
        for i in range(8):
            label = "healthy" if i % 2 == 0 else "bleached"
            rows.append(
                dict(
                    reef_cell_id=f"cell{i}",
                    date=day,
                    ndci_like_index=0.4 if label == "healthy" else -0.2,
                    index_shift=None,
                    dhw=1.0 if label == "healthy" else 6.0,
                    cloud_cover_fraction=0.1,
                    data_quality="ok",
                    dhw_risk_flag=label == "bleached",
                    confirmed_label=label,
                    survey_date=day,
                    survey_id=f"s{block}-{i}",
                    source_scene_id=f"scene{block}",
                    feature_start_date=day,
                )
            )
    return pd.DataFrame(rows, columns=TRAINING_SCHEMA.names)
