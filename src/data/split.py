# src/data/split.py

"""
Data splitting utilities for wind forecasting.

Supported protocols:
- Time-based split (no leakage)
- Cut-based temporal split (Track 1 style)
- LOFO (Leave-One-Farm/Site-Out) split (Track 2 style), with an optional inner
  time-based validation split for early stopping.

Returns train/val/test DataFrames (or folds) or index masks.
"""

from __future__ import annotations
import pandas as pd
from typing import Tuple, Optional, List, Dict, Iterator, Union
import numpy as np

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


# --- Cut-based temporal split ---

def cut_based_split(
    df: pd.DataFrame,
    train_cut: str,
    val_cut: str,
    time_col: str = "datetime",
    zone_col: Optional[str] = "zone_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split using explicit cut dates (no leakage).

    Train: time < train_cut
    Val:   train_cut <= time < val_cut
    Test:  time >= val_cut

    If zone_col is provided, the split is applied within each zone and then concatenated.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].isna().any():
        raise ValueError(f"[ERROR]'{time_col}' contains NaN after datetime parsing.")

    train_cut_dt = pd.to_datetime(train_cut)
    val_cut_dt = pd.to_datetime(val_cut)
    if not (train_cut_dt < val_cut_dt):
        raise ValueError("train_cut must be earlier than val_cut")

    df = df.sort_values([zone_col, time_col]) if zone_col else df.sort_values(time_col)
    df = df.reset_index(drop=True)

    if zone_col is None:
        df_train = df[df[time_col] < train_cut_dt]
        df_val = df[(df[time_col] >= train_cut_dt) & (df[time_col] < val_cut_dt)]
        df_test = df[df[time_col] >= val_cut_dt]
    else:
        splits = []
        for _, g in df.groupby(zone_col):
            g = g.sort_values(time_col)
            g_train = g[g[time_col] < train_cut_dt]
            g_val = g[(g[time_col] >= train_cut_dt) & (g[time_col] < val_cut_dt)]
            g_test = g[g[time_col] >= val_cut_dt]
            splits.append((g_train, g_val, g_test))

        df_train = pd.concat([s[0] for s in splits], ignore_index=True)
        df_val = pd.concat([s[1] for s in splits], ignore_index=True)
        df_test = pd.concat([s[2] for s in splits], ignore_index=True)

    print(f"Cut split complete: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
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
    """Wrapper split function.

    Supports two modes:
      1) Explicit cuts: pass train_cut=..., val_cut=... (preferred for Track 1)
      2) Ratio-based: pass val_ratio/test_ratio/min_train (fallback)
    """
    if "train_cut" in kwargs and "val_cut" in kwargs:
        train_cut = kwargs.pop("train_cut")
        val_cut = kwargs.pop("val_cut")
        return cut_based_split(df, train_cut=train_cut, val_cut=val_cut, **kwargs)

    df_train, df_valid, df_test = time_based_split(df, **kwargs)
    return df_train, df_valid, df_test


# --- Track 2: LOFO (Leave-One-Farm/Site-Out) splitting ---

def _ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].isna().any():
        raise ValueError(f"[ERROR]'{time_col}' contains NaN after datetime parsing.")
    return df


def inner_time_validation_split(
    df_train: pd.DataFrame,
    time_col: str = "datetime",
    val_days: int = 30,
    min_train: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create an *inner* time-based validation split from training data.

    This is intended for early stopping/hyperparameter selection when the *outer*
    split is LOFO. The split is global across all training groups (i.e., uses the
    max timestamp in df_train).

    Train_inner: time < (max_time - val_days)
    Val_inner:   time >= (max_time - val_days)
    """
    df_train = _ensure_datetime(df_train, time_col=time_col)
    df_train = df_train.sort_values(time_col).reset_index(drop=True)

    if len(df_train) < (min_train + 1):
        raise ValueError("Not enough samples to create an inner time validation split")

    max_dt = df_train[time_col].max()
    cut_dt = max_dt - pd.Timedelta(days=val_days)

    train_inner = df_train[df_train[time_col] < cut_dt]
    val_inner = df_train[df_train[time_col] >= cut_dt]

    # Safety: if val ends up empty due to short span, fall back to last 10%
    if len(val_inner) == 0:
        n = len(df_train)
        n_val = max(1, int(0.1 * n))
        train_inner = df_train.iloc[: n - n_val]
        val_inner = df_train.iloc[n - n_val :]

    if len(train_inner) < min_train:
        raise ValueError(
            f"Inner split produced too-small train set: {len(train_inner)} < {min_train}. "
            "Increase data, reduce val_days, or lower min_train."
        )

    return train_inner.reset_index(drop=True), val_inner.reset_index(drop=True)


def lofo_time_val_split(
    df: pd.DataFrame,
    held_out_group: Union[int, str],
    group_col: str = "zone_id",
    time_col: str = "datetime",
    val_days: int = 30,
    min_train: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """LOFO split for Track 2 with an inner time-based validation split.

    Outer protocol:
      - Outer test: all rows where group_col == held_out_group
      - Outer train: all remaining rows

    Inner protocol (for early stopping):
      - Split outer train into (train_inner, val_inner) by time (last `val_days`).

    Returns:
      train_inner_df, val_inner_df, outer_test_df
    """
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in df")

    df = _ensure_datetime(df, time_col=time_col)

    outer_test = df[df[group_col] == held_out_group].copy()
    outer_train = df[df[group_col] != held_out_group].copy()

    if len(outer_test) == 0:
        raise ValueError(f"No rows found for held_out_group={held_out_group}")

    # Inner time split for early stopping
    train_inner, val_inner = inner_time_validation_split(
        outer_train, time_col=time_col, val_days=val_days, min_train=min_train
    )

    # Sort outer test chronologically (and keep group ordering stable)
    outer_test = outer_test.sort_values([group_col, time_col]).reset_index(drop=True)

    print(
        f"LOFO split complete (held_out={held_out_group}): "
        f"train_inner={len(train_inner)}, val_inner={len(val_inner)}, outer_test={len(outer_test)}"
    )

    return train_inner, val_inner, outer_test


def lofo_time_val_folds(
    df: pd.DataFrame,
    group_col: str = "zone_id",
    time_col: str = "datetime",
    val_days: int = 30,
    min_train: int = 1000,
) -> List[Dict[str, Union[int, str, pd.DataFrame]]]:
    """Generate LOFO folds across all groups.

    Returns a list of dicts, each containing:
      - held_out_group
      - train_inner
      - val_inner
      - test (outer_test)

    This is convenient for iterating all sites in Track 2.
    """
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in df")

    groups = list(pd.Series(df[group_col].unique()).sort_values())
    folds: List[Dict[str, Union[int, str, pd.DataFrame]]] = []

    for g in groups:
        train_inner, val_inner, outer_test = lofo_time_val_split(
            df,
            held_out_group=g,
            group_col=group_col,
            time_col=time_col,
            val_days=val_days,
            min_train=min_train,
        )
        folds.append(
            {
                "held_out_group": g,
                "train_inner": train_inner,
                "val_inner": val_inner,
                "test": outer_test,
            }
        )

    return folds
