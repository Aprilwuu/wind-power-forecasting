import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from src.data.load import load_raw_data  

def build_features(
    raw_path: str,
    out_path: str,
    target_col: str,
    split_cfg: Optional[Dict] = None,
    feature_cfg: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    build basic features from raw data, and save as feature file
    """
    print(f"Loading raw data from{raw_path}")
    df = load_raw_data(raw_path)

    #1. basic time features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    #2. simple lag features
    feature_cfg = feature_cfg or {}
    lags: List[int] = feature_cfg.get("lags", [1, 24, 168])
    value_cols: List[str] = feature_cfg.get("value_cols", [target_col])
    rollings: Dict[str, List[int]] = feature_cfg.get(
        "rollings",
        {"mean": [3, 24], "std": [24]}
    )

    #sort rows properly
    by_zone = "zone_id" in df.columns
    sort_cols = ["datetime"] if not by_zone else ["zone_id", "datetime"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    #3. Lag& rolling per zone
    def add_lag_and_roll(g: pd.DataFrame)->pd.DataFrame:
        g = g.copy()
        for col in value_cols:
            if col not in g.columns:
                continue
            #Lag features
            for k in lags:
                g[f"{col}_lag{k}"] = g[col].shift(k)
            #Rolling features
            for agg, widnows in rollings.items():
                for w in widnows:
                    roll = g[col].rolling(w, min_periods=max(1,w//2))
                    if agg == "mean":
                        g[f"{col}_roll{w}_mean"] = roll.mean()
                    elif agg =="std":
                        g[f"{col}_roll{w}_std"] = roll.std(ddof=0)
        return g
    if by_zone:
        df = df.groupby("zone_id", group_keys=False).apply(add_lag_and_roll)
    else:
        df = df.apply(add_lag_and_roll)

    #4. Drop rows with NaN from lag/rolling
    df = df.dropna()

    #5. Save
    df.to_csv(out_path, index=False)
    print(f"[build_feature] Saved to {out_path}")

    return df

