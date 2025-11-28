import pandas as pd
from pathlib import Path
from .config import RED_WINE_CSV, WHITE_WINE_CSV
from typing import Union
import os

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the dataset from csvpath.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path, sep=';')  # winequality uses ';' separator
    return df

def load_wine_data(kind="red"):
    if kind == "red":
        return load_data(RED_WINE_CSV)
    elif kind == "white":
        return load_data(WHITE_WINE_CSV)
    else:
        raise ValueError("kind must be 'red' or 'white'")

def save_processed_data(df: pd.DataFrame, path: Union[Path, str]) -> None:
    """Save processed data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
