"""Generic threshold breach detection module."""

from typing import List, Dict
import pandas as pd
from src.insights.models import ThresholdBreachFinding


def detect_threshold_breaches(
    df_clean: pd.DataFrame,
    thresholds: Dict[str, Dict[str, float]]
) -> List[ThresholdBreachFinding]:
    """Detect indicator values that breach user-configured limits.
    
    thresholds format:
    {
        "anc_coverage": {"min": 50.0, "max": 100.0},
        "high_risk_cases": {"max": 20.0}
    }
    
    Args:
        df_clean: Cleaned DataFrame.
        thresholds: Dictionary of threshold boundaries per indicator.
        
    Returns:
        List of ThresholdBreachFinding instances.
    """
    findings: List[ThresholdBreachFinding] = []
    
    if not thresholds or df_clean.empty:
        return findings

    for indicator, limits in thresholds.items():
        if indicator not in df_clean.columns:
            continue

        min_val = limits.get("min")
        max_val = limits.get("max")

        for _, row in df_clean.dropna(subset=[indicator]).iterrows():
            val = float(row[indicator])
            district = str(row["district"])
            period = str(row["month"])

            if min_val is not None and val < min_val:
                findings.append(
                    ThresholdBreachFinding(
                        district=district,
                        indicator=indicator,
                        period=period,
                        value=val,
                        breach_type="below_minimum",
                        threshold_value=float(min_val)
                    )
                )

            if max_val is not None and val > max_val:
                findings.append(
                    ThresholdBreachFinding(
                        district=district,
                        indicator=indicator,
                        period=period,
                        value=val,
                        breach_type="above_maximum",
                        threshold_value=float(max_val)
                    )
                )

    return findings
