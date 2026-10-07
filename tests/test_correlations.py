"""Unit tests for correlation detection module."""

import pandas as pd
from src.analytics.correlations import detect_correlations, calculate_correlation_matrix


def test_correlation_matrix_computation():
    df = pd.DataFrame({
        "var_x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "var_y": [2.0, 4.0, 6.0, 8.0, 10.0],  # r = +1.0
        "var_z": [5.0, 4.0, 3.0, 2.0, 1.0],  # r = -1.0 with var_x
    })
    corr = calculate_correlation_matrix(df, ["var_x", "var_y", "var_z"])
    assert round(corr.loc["var_x", "var_y"], 4) == 1.0
    assert round(corr.loc["var_x", "var_z"], 4) == -1.0


def test_correlation_no_duplicate_pairs():
    df = pd.DataFrame({
        "var_x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "var_y": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    findings = detect_correlations(df, ["var_x", "var_y"], correlation_threshold=0.7)
    # Must only contain 1 pair: (var_x, var_y), NOT (var_y, var_x) as well
    assert len(findings) == 1
    pair = {findings[0].indicator_1, findings[0].indicator_2}
    assert pair == {"var_x", "var_y"}


def test_correlation_small_sample_flag():
    df = pd.DataFrame({
        "m1": [1.0, 2.0, 3.0, 4.0],
        "m2": [2.0, 4.0, 6.0, 8.0]
    })
    findings = detect_correlations(df, ["m1", "m2"], correlation_threshold=0.5)
    assert len(findings) == 1
    assert findings[0].is_small_sample is True
