"""Unit tests for dataset validator."""

import pandas as pd
from src.data.validator import validate_dataset


def test_validator_empty_dataframe():
    df = pd.DataFrame()
    report = validate_dataset(df)
    assert not report.is_valid
    assert any("empty" in e.lower() for e in report.errors)


def test_validator_missing_required_columns():
    df = pd.DataFrame({"month": ["2026-07"], "district": ["Surat"]})
    report = validate_dataset(df)
    assert not report.is_valid
    assert any("missing required columns" in e.lower() for e in report.errors)


def test_validator_detects_missing_values():
    df = pd.DataFrame({
        "month": ["2026-07", "2026-08"],
        "district": ["Surat", "Surat"],
        "anc_coverage": [85, None],
        "institutional_delivery": [90, 92],
        "immunization": [95, 96],
        "high_risk_cases": [10, 12]
    })
    report = validate_dataset(df)
    assert report.is_valid
    assert "anc_coverage" in report.missing_values
    assert report.missing_values["anc_coverage"] == 1


def test_validator_detects_duplicate_rows():
    df = pd.DataFrame({
        "month": ["2026-07", "2026-07"],
        "district": ["Surat", "Surat"],
        "anc_coverage": [85, 85],
        "institutional_delivery": [90, 90],
        "immunization": [95, 95],
        "high_risk_cases": [10, 10]
    })
    report = validate_dataset(df)
    assert report.duplicate_rows_count == 2
    assert any("duplicate" in w.lower() for w in report.warnings)


def test_validator_empty_district_strings():
    df = pd.DataFrame({
        "month": ["2026-07"],
        "district": ["  "],
        "anc_coverage": [85],
        "institutional_delivery": [90],
        "immunization": [95],
        "high_risk_cases": [10]
    })
    report = validate_dataset(df)
    assert not report.is_valid
    assert any("empty district" in e.lower() for e in report.errors)
