"""Unit tests for severity calculation engine."""

from src.insights.severity import (
    calculate_trend_severity,
    calculate_outlier_severity,
    calculate_correlation_severity,
    calculate_breach_severity,
)
from src.insights.models import (
    TrendFinding,
    OutlierFinding,
    CorrelationFinding,
    ThresholdBreachFinding,
)


def test_trend_severity_levels():
    # 25% change with 10% threshold -> ratio 2.5 >= 2.0 -> High
    tf_high = TrendFinding(
        district="D1", indicator="m1", period="2026-08",
        current_value=75, previous_value=100, change_pct=-25.0,
        direction="decrease", threshold=10.0, is_significant=True
    )
    assert calculate_trend_severity(tf_high) == "High"

    # 15% change with 10% threshold -> ratio 1.5 -> Medium
    tf_med = TrendFinding(
        district="D1", indicator="m1", period="2026-08",
        current_value=85, previous_value=100, change_pct=-15.0,
        direction="decrease", threshold=10.0, is_significant=True
    )
    assert calculate_trend_severity(tf_med) == "Medium"

    # 11% change with 10% threshold -> ratio 1.1 -> Low
    tf_low = TrendFinding(
        district="D1", indicator="m1", period="2026-08",
        current_value=89, previous_value=100, change_pct=-11.0,
        direction="decrease", threshold=10.0, is_significant=True
    )
    assert calculate_trend_severity(tf_low) == "Low"


def test_outlier_severity_zscore():
    of_high = OutlierFinding(
        district="D1", indicator="m1", month="2026-08",
        value=150, method="Z-score", score=4.5
    )
    assert calculate_outlier_severity(of_high) == "High"

    of_med = OutlierFinding(
        district="D1", indicator="m1", month="2026-08",
        value=120, method="Z-score", score=3.2
    )
    assert calculate_outlier_severity(of_med) == "Medium"


def test_correlation_severity():
    cf_high = CorrelationFinding(
        indicator_1="a", indicator_2="b", correlation=0.95,
        direction="positive", threshold=0.7, sample_size=12, is_small_sample=False
    )
    assert calculate_correlation_severity(cf_high) == "High"

    cf_med = CorrelationFinding(
        indicator_1="a", indicator_2="b", correlation=0.78,
        direction="positive", threshold=0.7, sample_size=12, is_small_sample=False
    )
    assert calculate_correlation_severity(cf_med) == "Medium"


def test_threshold_breach_severity():
    # Target 100, observed 60 (overshoot 40% >= 30%) -> High
    bf_high = ThresholdBreachFinding(
        district="D1", indicator="anc", period="2026-08",
        value=60, breach_type="below_minimum", threshold_value=100
    )
    assert calculate_breach_severity(bf_high) == "High"
