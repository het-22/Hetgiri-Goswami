"""Centralized, explainable severity calculation engine.

Calculates signal strength based on the magnitude of the statistical deviation
relative to configurable baseline thresholds, rather than arbitrary magic numbers.
"""

from typing import Union
from src.insights.models import (
    TrendFinding,
    OutlierFinding,
    CorrelationFinding,
    ThresholdBreachFinding,
)

# Severity Level Constants
SEVERITY_LOW = "Low"
SEVERITY_MEDIUM = "Medium"
SEVERITY_HIGH = "High"


def calculate_trend_severity(finding: TrendFinding) -> str:
    """Calculate severity for trend finding.
    
    Signal Strength = abs(change_pct) / trend_threshold
    - signal_strength >= 2.0 (e.g. >= 20% change when threshold is 10%): High
    - signal_strength >= 1.4 (e.g. >= 14% change when threshold is 10%): Medium
    - Otherwise: Low
    """
    if finding.threshold <= 0:
        return SEVERITY_MEDIUM
        
    signal_strength = abs(finding.change_pct) / finding.threshold
    if signal_strength >= 2.0:
        return SEVERITY_HIGH
    elif signal_strength >= 1.4:
        return SEVERITY_MEDIUM
    return SEVERITY_LOW


def calculate_outlier_severity(finding: OutlierFinding) -> str:
    """Calculate severity for outlier finding.
    
    For Z-score:
    - abs(z) >= 4.0: High
    - abs(z) >= 3.0: Medium
    - abs(z) < 3.0: Low
    
    For IQR:
    - score (distance beyond fence / IQR):
      score >= 1.0 (extreme beyond 2.5 * IQR): High
      score >= 0.5 (substantial beyond 2.0 * IQR): Medium
      score < 0.5: Low
    """
    if finding.method == "Z-score":
        if finding.score >= 4.0:
            return SEVERITY_HIGH
        elif finding.score >= 3.0:
            return SEVERITY_MEDIUM
        return SEVERITY_LOW
    else:  # IQR
        if finding.score >= 1.0:
            return SEVERITY_HIGH
        elif finding.score >= 0.3:
            return SEVERITY_MEDIUM
        return SEVERITY_LOW


def calculate_correlation_severity(finding: CorrelationFinding) -> str:
    """Calculate severity for correlation finding.
    
    Signal strength = abs(correlation) / correlation_threshold
    - abs(correlation) >= 0.90: High
    - abs(correlation) >= 0.75: Medium
    - Otherwise: Low
    """
    abs_r = abs(finding.correlation)
    if abs_r >= 0.90:
        return SEVERITY_HIGH
    elif abs_r >= 0.75:
        return SEVERITY_MEDIUM
    return SEVERITY_LOW


def calculate_breach_severity(finding: ThresholdBreachFinding) -> str:
    """Calculate severity for threshold breach finding.
    
    Proportional overshoot:
    overshoot_ratio = abs(value - threshold_value) / threshold_value (if threshold != 0)
    - overshoot_ratio >= 0.30: High
    - overshoot_ratio >= 0.15: Medium
    - Otherwise: Low
    """
    if finding.threshold_value == 0:
        return SEVERITY_HIGH if finding.value != 0 else SEVERITY_LOW
        
    overshoot = abs(finding.value - finding.threshold_value) / abs(finding.threshold_value)
    if overshoot >= 0.30:
        return SEVERITY_HIGH
    elif overshoot >= 0.15:
        return SEVERITY_MEDIUM
    return SEVERITY_LOW


def assess_severity(
    finding: Union[TrendFinding, OutlierFinding, CorrelationFinding, ThresholdBreachFinding]
) -> str:
    """Central dispatch function for severity assessment."""
    if isinstance(finding, TrendFinding):
        return calculate_trend_severity(finding)
    elif isinstance(finding, OutlierFinding):
        return calculate_outlier_severity(finding)
    elif isinstance(finding, CorrelationFinding):
        return calculate_correlation_severity(finding)
    elif isinstance(finding, ThresholdBreachFinding):
        return calculate_breach_severity(finding)
    return SEVERITY_LOW
