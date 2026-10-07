"""Correlation detection module using Pearson correlation."""

from typing import List, Tuple
import pandas as pd
import numpy as np
from src.insights.models import CorrelationFinding

SMALL_SAMPLE_THRESHOLD = 10


def calculate_correlation_matrix(
    df_clean: pd.DataFrame,
    indicator_cols: List[str]
) -> pd.DataFrame:
    """Compute Pearson correlation matrix across numerical indicators.
    
    Drops non-numeric columns and columns with zero variance.
    """
    valid_cols = [c for c in indicator_cols if c in df_clean.columns]
    if not valid_cols:
        return pd.DataFrame()
        
    numeric_df = df_clean[valid_cols].apply(pd.to_numeric, errors="coerce")
    # Exclude columns with zero standard deviation (constant)
    non_constant_cols = [c for c in valid_cols if numeric_df[c].std() > 0]
    
    if len(non_constant_cols) < 2:
        return pd.DataFrame()

    corr_matrix = numeric_df[non_constant_cols].corr(method="pearson")
    return corr_matrix


def detect_correlations(
    df_clean: pd.DataFrame,
    indicator_cols: List[str],
    correlation_threshold: float = 0.70
) -> List[CorrelationFinding]:
    """Detect significant pairwise correlations between indicators.
    
    Rules:
    - Calculates Pearson correlation between numerical indicators.
    - Flags pairs where abs(r) >= correlation_threshold.
    - Avoids duplicate pairs (e.g. returns A-B once, not both A-B and B-A).
    - Ignores diagonal (r = 1.0 with itself).
    - Flags small sample sizes (N < 10) with is_small_sample warning flag.
    - Language adheres to statistical best practice (no causation inference).
    """
    findings: List[CorrelationFinding] = []
    
    corr_matrix = calculate_correlation_matrix(df_clean, indicator_cols)
    if corr_matrix.empty:
        return findings

    cols = list(corr_matrix.columns)
    n_cols = len(cols)

    # Use upper triangle to guarantee each pair is visited exactly once
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            ind1 = cols[i]
            ind2 = cols[j]
            r = corr_matrix.iloc[i, j]

            if pd.isna(r) or np.isnan(r):
                continue

            # Calculate effective sample size of non-null pairs
            valid_pairs = df_clean[[ind1, ind2]].dropna()
            n_obs = len(valid_pairs)

            if abs(r) >= correlation_threshold:
                direction = "positive" if r > 0 else "negative"
                findings.append(
                    CorrelationFinding(
                        indicator_1=ind1,
                        indicator_2=ind2,
                        correlation=round(float(r), 4),
                        direction=direction,
                        threshold=correlation_threshold,
                        sample_size=n_obs,
                        is_small_sample=(n_obs < SMALL_SAMPLE_THRESHOLD)
                    )
                )

    # Sort findings by correlation magnitude descending
    findings.sort(key=lambda x: abs(x.correlation), reverse=True)
    return findings
