"""Unit tests for outlier detection module."""

import pandas as pd
from src.analytics.outliers import detect_outliers, detect_outliers_iqr, detect_outliers_zscore


def test_outliers_iqr_detection():
    # Regular values around 80-90, single extreme low 20
    data = [80, 82, 85, 87, 88, 89, 90, 20]
    df = pd.DataFrame({
        "month": [f"2026-0{i+1}" for i in range(len(data))],
        "district": [f"District_{i+1}" for i in range(len(data))],
        "metric_a": data
    })
    
    findings = detect_outliers_iqr(df, "metric_a", iqr_multiplier=1.5)
    assert len(findings) >= 1
    assert any(f.value == 20 for f in findings)
    assert findings[0].method == "IQR"


def test_outliers_zscore_detection():
    # Value 100 with standard deviation around 5, mean around 10
    data = [10, 10, 11, 9, 10, 10, 10, 100]
    df = pd.DataFrame({
        "month": [f"2026-0{i+1}" for i in range(len(data))],
        "district": [f"District_{i+1}" for i in range(len(data))],
        "metric_b": data
    })
    
    findings = detect_outliers_zscore(df, "metric_b", threshold=2.0)
    assert len(findings) == 1
    assert findings[0].value == 100
    assert findings[0].method == "Z-score"


def test_outliers_constant_column():
    # All values identical => std = 0, IQR = 0 => no crash, returns empty
    data = [50, 50, 50, 50, 50]
    df = pd.DataFrame({
        "month": [f"2026-0{i+1}" for i in range(len(data))],
        "district": [f"District_{i+1}" for i in range(len(data))],
        "constant_col": data
    })
    
    findings_z = detect_outliers_zscore(df, "constant_col", threshold=3.0)
    assert len(findings_z) == 0
