"""Additional regression tests written by the group."""
import logging

import pandas as pd
import pytest

import quakewatch as qw


def test_group_features_uses_depth_km(loaded):
    qw.integrate()
    feat = qw.features_stage()

    assert "depth_class" in feat.columns
    assert feat["depth_class"].notna().all()

def test_group_manila_time_conversion(loaded):
    qw.integrate()
    feat = qw.features_stage()

    top = feat.loc[feat["mag"].idxmax()]
    assert top["local"].strftime("%Y-%m-%d %H:%M") == "2023-12-02 22:37"

@pytest.mark.parametrize(
    "magnitude, expected",
    [(5.0, "Moderate"), (5.9, "Moderate"), (6.0, "Strong"),
     (6.9, "Strong"), (7.0, "Major"), (8.0, "Great")],
)
def test_group_magnitude_boundaries(magnitude, expected):
    """Magnitude class boundaries must match the published category ranges."""
    assert qw.mag_class(pd.Series([magnitude])).iloc[0] == expected

def test_group_mindanao_event_nearest_city(loaded):
    clean = qw.integrate()

    row = clean[
        clean["time"].dt.strftime("%Y-%m-%d %H:%M") == "2023-12-02 14:37"
    ]

    assert row["city_id"].iloc[0] == "C10"

def test_group_incremental_acquisition_uses_fake_api(monkeypatch):
    """A fake API call must begin just after the latest persisted event."""
    latest = pd.Timestamp("2024-06-01T12:30:00Z")
    observed = {}

    monkeypatch.setattr(qw, "watermark", lambda: latest)

    def fake_api_fetch(start, end):
        observed["start"] = start
        return pd.DataFrame(columns=["id", "time", "latitude", "longitude",
                                     "depth_km", "mag", "magType", "place"])

    monkeypatch.setattr(qw, "fetch_range", fake_api_fetch)
    assert qw.load_new() == 0
    assert observed["start"] == "2024-06-01T12:30:00.001000"

def test_group_rejected_rows_emit_warning(raw_snapshot, caplog):
    """Rows rejected by the contract must be logged at WARNING level."""
    bad = raw_snapshot.head(1).copy()
    bad.loc[bad.index[0], "depth_km"] = -5.0

    with caplog.at_level(logging.WARNING, logger="quakewatch"):
        good, rejects = qw.split_by_contract(bad)

    assert len(good) == 0
    assert len(rejects) == 1
    assert any(record.levelno == logging.WARNING and "Rejected 1" in record.message
               for record in caplog.records)

