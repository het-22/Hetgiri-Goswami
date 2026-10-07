"""Data transformation and normalization module."""

from typing import List, Tuple
import pandas as pd


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Transform, clean, and chronologically sort the dataset.
    
    Operations:
    1. Make an explicit copy.
    2. Dynamically identify numerical indicator columns.
    3. Coerce numerical indicators to numeric types (invalid -> NaN).
    4. Convert 'month' column to sortable datetime/period representation.
    5. Sort deterministically by district and month.
    6. Return cleaned DataFrame and detected numerical indicator names.
    """
    df_clean = df.copy()

    # Dynamic identification of indicator columns
    non_indicator_cols = {"month", "district"}
    indicator_cols = [col for col in df_clean.columns if col not in non_indicator_cols]

    # Clean and coerce numerical indicators
    for col in indicator_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Clean district string
    df_clean["district"] = df_clean["district"].astype(str).str.strip()

    # Parse month chronologically
    df_clean["month_dt"] = pd.to_datetime(df_clean["month"], errors="coerce")
    df_clean["month_str"] = df_clean["month"].astype(str).str.strip()

    # Sort by district and date
    df_clean = df_clean.sort_values(by=["district", "month_dt"]).reset_index(drop=True)

    return df_clean, indicator_cols


def to_long_format(df_clean: pd.DataFrame, indicator_cols: List[str]) -> pd.DataFrame:
    """Transform wide DataFrame into generic long format:
    district, month, indicator, value.
    """
    id_vars = ["district", "month_str"]
    available_cols = [c for c in indicator_cols if c in df_clean.columns]
    
    long_df = pd.melt(
        df_clean,
        id_vars=id_vars,
        value_vars=available_cols,
        var_name="indicator",
        value_name="value"
    ).rename(columns={"month_str": "month"})
    
    return long_df.dropna(subset=["value"])
