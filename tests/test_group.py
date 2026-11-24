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