"""Data loading module for CSV ingestion."""

from pathlib import Path
from typing import Union, IO
import pandas as pd


def load_csv(source: Union[str, Path, IO]) -> pd.DataFrame:
    """Safely load CSV data from a file path, string path, or file-like buffer.
    
    Args:
        source: File path or file buffer.
        
    Returns:
        pd.DataFrame containing the raw ingested data.
        
    Raises:
        FileNotFoundError: If file path does not exist.
        ValueError: If file is empty or cannot be parsed.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {path}")
            
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        raise ValueError("The provided CSV file contains no data.")
    except Exception as exc:
        raise ValueError(f"Failed to read CSV: {str(exc)}") from exc
        
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df
