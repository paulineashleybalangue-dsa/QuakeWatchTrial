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