"""Unit tests for trend detection module."""

import pandas as pd
from src.analytics.trends import detect_trends, calculate_pct_change
from src.data.transformer import prepare_dataset


def test_calculate_pct_change_regular():
    # 85 -> 69: (69 - 85) / 85 * 100 = -18.8235...%
    pct = calculate_pct_change(69.0, 85.0)
    assert pct is not None
    assert round(pct, 1) == -18.8


def test_calculate_pct_change_zero_division():
    # 0 -> 0 => 0.0%
    assert calculate_pct_change(0, 0) == 0.0
    # 0 -> 10 => None (safe, undefined)
    assert calculate_pct_change(10, 0) is None


def test_detect_trends_threshold_filter():
    df = pd.DataFrame({
        "month": ["2026-07", "2026-08", "2026-07", "2026-08"],
        "district": ["DistA", "DistA", "DistB", "DistB"],
        "anc_coverage": [100.0, 80.0, 100.0, 98.0]  # DistA: -20%, DistB: -2%
    })
    df_clean, indicators = prepare_dataset(df)
    
    # Threshold 10%: DistA should be caught, DistB should be ignored
    findings = detect_trends(df_clean, indicators, trend_threshold_pct=10.0)
    assert len(findings) == 1
    assert findings[0].district == "DistA"
    assert findings[0].change_pct == -20.0
    assert findings[0].direction == "decrease"


def test_detect_trends_generic_column_names():
    # Test with arbitrary non-healthcare columns
    df = pd.DataFrame({
        "month": ["2025-01", "2025-02"],
        "district": ["RegionX", "RegionX"],
        "custom_metric_xyz": [50.0, 75.0]  # +50%
    })
    df_clean, indicators = prepare_dataset(df)
    findings = detect_trends(df_clean, indicators, trend_threshold_pct=15.0)
    assert len(findings) == 1
    assert findings[0].indicator == "custom_metric_xyz"
    assert findings[0].direction == "increase"
    assert findings[0].change_pct == 50.0
