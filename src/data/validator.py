"""Data validation engine for healthcare performance datasets."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
from src.config import REQUIRED_COLUMNS


@dataclass
class ValidationReport:
    """Detailed dataset validation report."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_values: Dict[str, int] = field(default_factory=dict)
    duplicate_rows_count: int = 0
    total_rows: int = 0
    total_columns: int = 0
    detected_indicators: List[str] = field(default_factory=list)
    districts: List[str] = field(default_factory=list)
    months: List[str] = field(default_factory=list)


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Validate DataFrame against schema and statistical consistency rules.
    
    Checks:
    - Empty DataFrame
    - Required columns existence
    - 'month' parseable as datetime
    - 'district' non-empty
    - numerical indicators are actually numeric
    - duplicate rows
    - missing values
    """
    report = ValidationReport(is_valid=True)
    
    if df is None or df.empty:
        report.is_valid = False
        report.errors.append("Dataset is empty.")
        return report

    report.total_rows = len(df)
    report.total_columns = len(df.columns)
    
    # 1. Required columns check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.is_valid = False
        report.errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        return report

    # 2. Missing values check
    missing_dict = df.isnull().sum().to_dict()
    report.missing_values = {k: int(v) for k, v in missing_dict.items() if v > 0}
    if report.missing_values:
        total_missing = sum(report.missing_values.values())
        report.warnings.append(f"Detected {total_missing} missing values across {len(report.missing_values)} columns.")

    # 3. Duplicate rows check
    duplicates = df.duplicated(subset=["month", "district"], keep=False).sum()
    report.duplicate_rows_count = int(duplicates)
    if duplicates > 0:
        report.warnings.append(f"Detected {duplicates} duplicate entries for (month, district) pairs.")

    # 4. District validation
    empty_districts = df["district"].astype(str).str.strip().eq("").sum()
    if empty_districts > 0:
        report.errors.append(f"Found {empty_districts} rows with empty district names.")
        report.is_valid = False

    # 5. Month validation
    try:
        parsed_months = pd.to_datetime(df["month"], errors="coerce")
        invalid_months = parsed_months.isna().sum()
        if invalid_months > 0:
            report.warnings.append(f"Found {invalid_months} rows with unparseable month values.")
    except Exception as e:
        report.warnings.append(f"Month column parsing issue: {str(e)}")

    # 6. Identify numerical indicators dynamically
    candidate_indicators = [
        col for col in df.columns 
        if col not in ["month", "district"]
    ]
    
    valid_indicators = []
    for col in candidate_indicators:
        # Check if column is numeric or can be safely coerced
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        coerced_nans = numeric_series.isna().sum() - df[col].isna().sum()
        if coerced_nans > 0:
            report.warnings.append(f"Indicator '{col}' contains {coerced_nans} non-numeric values which will be coerced to NaN.")
        valid_indicators.append(col)
        
    report.detected_indicators = valid_indicators
    report.districts = sorted(df["district"].dropna().unique().tolist())
    report.months = sorted(df["month"].dropna().astype(str).unique().tolist())

    if len(report.districts) == 0:
        report.is_valid = False
        report.errors.append("No valid districts found.")

    if len(report.detected_indicators) == 0:
        report.is_valid = False
        report.errors.append("No numerical indicator columns found.")

    return report
