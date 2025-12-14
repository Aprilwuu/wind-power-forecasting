"""
Data splitting utilities for wind forecasting
-supports time-based split(no leakage)
-can stratify by zone_id
-returns train/val/test DAtaFrames or index masks
"""

from __future__ import annotations
import pandas as pd
from typing import Tuple, Optional

def time_based_split(
    df: pd.DataFrame,
    time_col: str = "datetime",
    zone_col: Optional[str] = "zone_id",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_train: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split to avoid leakage.
    Works for both single-zone and multi-zone setups.
    """
    df = df.sort_values([zone_col, time_col]) if zone_col else df.sort_values(time_col)
    df = df.reset_index(drop=True)

    if zone_col is None:
        n = len(df)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_test - n_val

        if n_train < min_train:
            raise ValueError("Not enough samples for the requested split ratios")
        
        df_train = df.iloc[:n_train]
        df_val = df.iloc[n_train:n_train + n_val]
        df_test = df.iloc[n_train + n_val:]
    else:
        # group-wise split
        splits = []
        for _, g in df.groupby(zone_col):
            n = len(g)
            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)
            n_train = n - n_test - n_val

            if n_train < min_train:
                continue

            g_train = g.iloc[:n_train]
            g_val = g.iloc[n_train:n_train + n_val]
            g_test = g.iloc[n_train+n_val: ]
            splits.append((g_train, g_val, g_test))

        df_train = pd.concat([s[0] for s in splits], ignore_index=True)
        df_val = pd.concat([s[1] for s in splits], ignore_index=True)
        df_test = pd.concat([s[2] for s in splits], ignore_index=True)

    print(f"Split complete: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return df_train, df_val, df_test

def index_split(
    df: pd.DataFrame,
    time_col: str = "datetime",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[list[int], list[int], list[int]]: 
    """
    Return index lists instead of dataframes.
    Useful for PyTorch DataLoaders.
    """
    df = df.sort_values(time_col).reset_index()
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val

    idx_train = df.index[:n_train].tolist()
    idx_val = df.index[n_train:n_train + n_val].tolist()
    idx_test = df.index[n_train + n_val:].tolist()

    return idx_train, idx_val, idx_test

def train_valid_split(df, **kwargs):
    df_train, df_valid, df_test = time_based_split(df, **kwargs)
    return df_train, df_valid, df_test
