"""Trend detection module for district-indicator time series."""

from typing import List, Optional
import pandas as pd
import numpy as np
from src.insights.models import TrendFinding


def calculate_pct_change(current: float, previous: float) -> Optional[float]:
    """Calculate percentage change safely without zero division.
    
    Formula:
        ((current - previous) / previous) * 100
        
    Returns:
        float if previous != 0, else None or 0.0 depending on transition.
    """
    if previous == 0:
        if current == 0:
            return 0.0
        # If previous was 0 and current is non-zero, mathematically undefined percentage
        # We can treat this safely as None or 100% * sign
        return None
    return ((current - previous) / previous) * 100.0


def detect_trends(
    df_clean: pd.DataFrame,
    indicator_cols: List[str],
    trend_threshold_pct: float = 10.0
) -> List[TrendFinding]:
    """Detect significant trends across all district-indicator pairs.
    
    For every district + indicator pair:
    1. Sort chronologically.
    2. Compare current value with previous value.
    3. Calculate pct_change = ((current - previous) / previous) * 100.
    4. If abs(pct_change) >= trend_threshold, create a trend finding.
    
    Args:
        df_clean: Cleaned and chronologically sorted DataFrame.
        indicator_cols: List of numerical indicator column names.
        trend_threshold_pct: Percentage change threshold to qualify as significant.
        
    Returns:
        List of TrendFinding instances.
    """
    findings: List[TrendFinding] = []
    
    if df_clean.empty or not indicator_cols:
        return findings

    # Ensure sorted by district and month_dt
    sort_cols = ["district", "month_dt"] if "month_dt" in df_clean.columns else ["district", "month"]
    df_sorted = df_clean.sort_values(by=sort_cols).reset_index(drop=True)

    districts = df_sorted["district"].unique()

    for district in districts:
        district_data = df_sorted[df_sorted["district"] == district]
        
        for indicator in indicator_cols:
            if indicator not in district_data.columns:
                continue

            # Drop missing values for this indicator
            valid_series = district_data[["month", indicator]].dropna(subset=[indicator])
            if len(valid_series) < 2:
                continue

            records = valid_series.to_dict(orient="records")
            
            for i in range(1, len(records)):
                prev_record = records[i - 1]
                curr_record = records[i]
                
                prev_val = float(prev_record[indicator])
                curr_val = float(curr_record[indicator])
                curr_period = str(curr_record["month"])

                pct_chg = calculate_pct_change(curr_val, prev_val)
                
                # Handle safe zero-division case: if prev was 0 and curr is > 0, consider it infinite/significant
                if pct_chg is None:
                    # Special signal: moved from 0 to non-zero
                    is_sig = True
                    direction = "increase" if curr_val > prev_val else "decrease"
                    # approximate signal pct as 100% or large change
                    effective_pct = 100.0 if curr_val > 0 else -100.0
                    findings.append(
                        TrendFinding(
                            district=district,
                            indicator=indicator,
                            period=curr_period,
                            current_value=curr_val,
                            previous_value=prev_val,
                            change_pct=effective_pct,
                            direction=direction,
                            threshold=trend_threshold_pct,
                            is_significant=is_sig
                        )
                    )
                    continue

                abs_chg = abs(pct_chg)
                is_sig = abs_chg >= trend_threshold_pct
                
                if is_sig:
                    direction = "increase" if curr_val >= prev_val else "decrease"
                    findings.append(
                        TrendFinding(
                            district=district,
                            indicator=indicator,
                            period=curr_period,
                            current_value=curr_val,
                            previous_value=prev_val,
                            change_pct=round(pct_chg, 2),
                            direction=direction,
                            threshold=trend_threshold_pct,
                            is_significant=is_sig
                        )
                    )

    return findings
