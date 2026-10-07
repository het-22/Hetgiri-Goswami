"""Unit and integration tests for Data Loader."""

import io
import pytest
import pandas as pd
from src.data.loader import load_csv


def test_load_csv_valid_string_io():
    csv_data = "month,district,anc_coverage\n2026-07,Ahmedabad,85\n2026-08,Ahmedabad,69"
    df = load_csv(io.StringIO(csv_data))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["month", "district", "anc_coverage"]


def test_load_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_csv("non_existent_directory/fake_file.csv")


def test_load_csv_empty_data():
    empty_csv = ""
    with pytest.raises(ValueError, match="no data"):
        load_csv(io.StringIO(empty_csv))


def test_load_csv_strips_column_whitespace():
    csv_data = " month , district , anc_coverage \n2026-07,Ahmedabad,85"
    df = load_csv(io.StringIO(csv_data))
    assert "month" in df.columns
    assert "district" in df.columns
    assert "anc_coverage" in df.columns
