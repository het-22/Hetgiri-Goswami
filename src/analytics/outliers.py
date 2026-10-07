"""Outlier detection module using IQR and Z-score methods."""

from typing import List, Optional
import pandas as pd
import numpy as np
from src.insights.models import OutlierFinding


def detect_outliers_iqr(
    df: pd.DataFrame,
    indicator: str,
    iqr_multiplier: float = 1.5
) -> List[OutlierFinding]:
    """Detect outliers across all observations for a given indicator using IQR.
    
    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    Outlier if value < lower_bound or value > upper_bound.
    """
    findings: List[OutlierFinding] = []
    series = df[indicator].dropna()
    if len(series) < 4:
        # Not enough data points to compute meaningful quartiles
        return findings

    q1 = float(np.percentile(series, 25))
    q3 = float(np.percentile(series, 75))
    iqr = q3 - q1

    # If IQR is zero, all middle 50% are identical; only check non-identical extreme
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    for _, row in df.dropna(subset=[indicator]).iterrows():
        val = float(row[indicator])
        if val < lower_bound or val > upper_bound:
            # Score is normalized distance beyond the boundary relative to IQR (or 1.0 if IQR == 0)
            dist = max(lower_bound - val, val - upper_bound)
            score = (dist / iqr) if iqr > 0 else dist
            
            findings.append(
                OutlierFinding(
                    district=str(row["district"]),
                    indicator=indicator,
                    month=str(row["month"]),
                    value=val,
                    method="IQR",
                    score=round(score, 2),
                    lower_bound=round(lower_bound, 2),
                    upper_bound=round(upper_bound, 2)
                )
            )

    return findings


def detect_outliers_zscore(
    df: pd.DataFrame,
    indicator: str,
    threshold: float = 3.0
) -> List[OutlierFinding]:
    """Detect outliers across all observations for a given indicator using Z-score.
    
    z = (x - mean) / std
    Outlier if abs(z) >= threshold.
    """
    findings: List[OutlierFinding] = []
    series = df[indicator].dropna()
    if len(series) < 3:
        return findings

    mean = float(series.mean())
    std = float(series.std(ddof=1))

    if std == 0 or np.isnan(std):
        # Constant column, no variation, no outliers
        return findings

    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std

    for _, row in df.dropna(subset=[indicator]).iterrows():
        val = float(row[indicator])
        z = (val - mean) / std
        if abs(z) >= threshold:
            findings.append(
                OutlierFinding(
                    district=str(row["district"]),
                    indicator=indicator,
                    month=str(row["month"]),
                    value=val,
                    method="Z-score",
                    score=round(abs(z), 2),
                    lower_bound=round(lower_bound, 2),
                    upper_bound=round(upper_bound, 2)
                )
            )

    return findings


def detect_outliers(
    df_clean: pd.DataFrame,
    indicator_cols: List[str],
    method: str = "IQR",
    z_score_threshold: float = 3.0,
    iqr_multiplier: float = 1.5
) -> List[OutlierFinding]:
    """Master outlier detection function across all indicators.
    
    Args:
        df_clean: Cleaned DataFrame.
        indicator_cols: List of indicator column names.
        method: 'IQR' or 'Z-score'.
        z_score_threshold: Critical Z value for anomaly.
        iqr_multiplier: Multiplier for IQR (standard 1.5).
        
    Returns:
        List of OutlierFinding instances.
    """
    all_findings: List[OutlierFinding] = []
    
    for indicator in indicator_cols:
        if indicator not in df_clean.columns:
            continue

        if method.upper() == "Z-SCORE":
            findings = detect_outliers_zscore(df_clean, indicator, threshold=z_score_threshold)
        else:
            findings = detect_outliers_iqr(df_clean, indicator, iqr_multiplier=iqr_multiplier)
            
        all_findings.extend(findings)

    return all_findings
