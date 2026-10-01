"""
data_loader.py
Base module for loading and preprocessing water meter datasets.
Handles UTF-8 BOM encoding, date parsing, sorting, and daily flow calculation.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_pokazaniya(file_path="test/pokazaniya.csv") -> pd.DataFrame:
    """
    Loads pokazaniya.csv, parses dates, sorts by device and date,
    and calculates daily consumption (daily_flow).
    
    Handles the first day of meter operation correctly (daily_flow = NaN).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at {file_path}")

    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    
    # Adhere strictly to expected column types & formatting
    df["anon_id"] = df["anon_id"].astype(str).str.strip()
    df["data"] = pd.to_datetime(df["data"])
    df["pokazanie"] = pd.to_numeric(df["pokazanie"], errors="coerce")
    df["paketov_za_sutki"] = pd.to_numeric(df["paketov_za_sutki"], errors="coerce")
    
    # Ensure correct sorting by device and date
    df = df.sort_values(["anon_id", "data"]).reset_index(drop=True)
    
    # Calculate previous record metadata per device
    df["prev_data"] = df.groupby("anon_id")["data"].shift(1)
    df["date_diff"] = (df["data"] - df["prev_data"]).dt.days
    
    df["prev_pokazanie"] = df.groupby("anon_id")["pokazanie"].shift(1)
    df["raw_diff"] = df["pokazanie"] - df["prev_pokazanie"]
    
    # Daily consumption:
    # - On the first day of reading (prev_pokazanie is NaN / date_diff is NaN), daily_flow is NaN.
    # - If readings are consecutive (date_diff == 1), daily_flow is raw_diff.
    # - If there is a gap (date_diff > 1), daily_flow is raw_diff / date_diff (average daily flow in gap).
    df["daily_flow"] = np.where(df["date_diff"] == 1, df["raw_diff"], np.nan)
    df["avg_daily_flow"] = np.where(df["date_diff"] > 0, df["raw_diff"] / df["date_diff"], np.nan)
    
    return df


def load_pribory(file_path="test/pribory.csv") -> pd.DataFrame:
    """Loads pribory.csv containing device metadata."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at {file_path}")

    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    # Clean string columns if needed
    for col in ["anon_id", "bs", "tip", "model", "mesyac_vypuska"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_sobytiya(file_path="test/sobytiya.csv") -> pd.DataFrame:
    """Loads sobytiya.csv containing device diagnostic events (магнит, сброс, Холл)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at {file_path}")

    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df["anon_id"] = df["anon_id"].astype(str).str.strip()
    df["data"] = pd.to_datetime(df["data"])
    df["sobytie"] = df["sobytie"].astype(str).str.strip()
    return df


def load_full_dataset():
    """Loads and merges all three datasets."""
    df_pok = load_pokazaniya()
    df_prib = load_pribory()
    df_sob = load_sobytiya()
    return df_pok, df_prib, df_sob


if __name__ == "__main__":
    df_pok, df_prib, df_sob = load_full_dataset()
    print(f"Loaded {len(df_pok)} readings for {df_pok['anon_id'].nunique()} devices.")
    print(f"Loaded metadata for {len(df_prib)} devices.")
    print(f"Loaded {len(df_sob)} diagnostic events.")
