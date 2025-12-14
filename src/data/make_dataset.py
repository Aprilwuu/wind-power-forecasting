"""
make_dataset.py
- Sort by zone_id + datetime
- Impute target NA per zone: linear interpolation + ffill/bfill (default)
- Minimal sanity checks
"""
from __future__ import annotations
import pandas as pd

def impute_target_per_zone(
    df: pd.DataFrame,
    target_col: str = "target",
    zone_col : str = "zone_id",
    method: str = "linear", 
) -> pd.DataFrame:
    df = df.copy()
    df[target_col] = (
        df.groupby(zone_col, group_keys=False)[target_col]
          .apply(lambda s: s.interpolate(method="linear").ffill().bfill())
    )

    return df

def make_dataset(
        df: pd.DataFrame,
        target_col: str = "target",
        zone_col: str = "zone_id",
        time_col: str = "datetime",
        impute_method: str = "linear",
        drop_duplicates: bool = True,
) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors = "coerce")

    if drop_duplicates:
        df = df.drop_duplicates(subset=[zone_col, time_col]).reset_index(drop=True)
    
    df = df.sort_values([zone_col, time_col], kind="mergesort").reset_index(drop=True)

    na_before = int(df[target_col].isna().sum())

    df = impute_target_per_zone(
        df,
        target_col=target_col,
        zone_col=zone_col,
        method=impute_method,
    )
    #sanity check
    if df[time_col].isna().any():
        raise ValueError(f"[ERROR]'{time_col}' contains NaN after cleaning.")
    na_after = int(df[target_col].isna().sum())
    if na_after > 0:
        raise ValueError(f"[ERROR]'{target_col}' still has {na_after} NaNs after imputation.")
    print(f"[OK]'{target_col}'imputed:{na_before} -> {na_after}")

    return df

def process_file(
    input_path: str,
    output_path: str,
    target_col: str = "target",
    zone_col: str = "zone_id",
    time_col: str = "datetime",
    impute_method: str = "linear",
):
    """Convenience function: read → make_dataset → write (parquet/csv)."""
    # read 
    df = pd.read_csv(input_path)
    # clean:
    df_clean = make_dataset(
        df,
        target_col=target_col,
        zone_col=zone_col,
        time_col=time_col,
        impute_method=impute_method
    )
    # write
    df_clean.to_csv(output_path, index=False)
    print(f"[DONE] Saved cleaned data to: {output_path}")
