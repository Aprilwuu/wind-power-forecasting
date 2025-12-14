"""
Feature building utilities for wind forecasting (GEFCom-style).
- Base features: time encodings, wind physics (U/V -> ws/wd/veer), zone features
- ML features: lag/rolling stats for tree models (LightGBM/XGB/RF)
- Seq features: windowing tensors + masks for deep models (TCN/Transformer)

"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, List, Dict, Optional

##angle helper
def angle_wrap_pi(x: np.ndarray | pd.Series) -> np.ndarray:
    """Wrap angles to (-pi, pi]"""
    return (np.asanyarray(x) + np.pi) % (2 * np.pi) - np.pi

def angle_diff(a,b):
    """Return minimal rotation a-b in (-pi, pi)"""
    return angle_wrap_pi(np.asanyarray(a) - np.asanyarray(b))

##atomic feature builders
def add_time_features(
        df: pd.DataFrame,
        time_col: str = "datetime",
        add_cyclical: bool = True
) -> pd.DataFrame:
    """
    Add time-based features (no leakage).
    Assumes df[time_col] is datetime64
    """
    df = df.copy()
    ts = pd.to_datetime(df[time_col], errors="coerce")
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek  #0=Mon
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    if add_cyclical:
        #Daily cycle
        df["sin_hour"] = np.sin(2*np.pi * df["hour"]/24.0)
        df["cos_hour"] = np.cos(2*np.pi * df["hour"]/24.0)
        #Yearly cycle ( 366 to be safe for leap year)
        doy = ts.dt.dayofyear.fillna(1).astype(int).clip(1, 366)
        df["sin_doy"] = np.sin(2 * np.pi * doy / 366.0)
        df["cos_doy"] = np.cos(2 * np.pi * doy / 366.0)

    return df

def add_wind_features(
    df: pd.DataFrame,
    zone_col: str = "zone_id",
    u10: str = "U10", v10: str = "V10",
    u100: str = "U100", v100: str = "V100",
    add_angles: bool = True,
    add_angle_sincos: bool = True
) -> pd.DataFrame:
    """
    Compute wind physics from U/V:
     - ws10/ws100, wd10/wd100 (radians), shear, veer, simple temporal diffs.
     - For deep models, prefer using wd_sin/wd_cos instead of raw angle.
    """
    df = df.copy()

    #magnitudes
    df["ws10"] = np.hypot(df[u10], df[v10])
    df["ws100"] = np.hypot(df[u100], df[v100])
    
    # directions( meteorological: wd = atan2(U, V))
    wd10 = np.arctan2(df[u10], df[v10])
    wd100 = np.arctan2(df[u100], df[v100])

    if add_angles:
        df["wd10"] = wd10
        df["wd100"] = wd100

    if add_angle_sincos:
        df["wd10_sin"] = np.sin(wd10); df["wd10_cos"] = np.cos(wd10)
        df["wd100_sin"] = np.sin(wd100); df["wd100_cos"] = np.cos(wd100)

    #vertical shear & veer
    df["shear"] = df["ws100"] - df["ws10"]
    df["shear_ratio"] = df["ws100"] / (df["ws10"] + 1e-6) 
    df["veer"] = angle_diff(wd100, wd10)   # (-pi, pi]

    #simple dynamics(groupwise diff to avoid cross-zone mixing)
    g = df.groupby(zone_col, group_keys=False)
    for col in ["ws10", "ws100"]:
        df[f"d_{col}"] = g[col].diff(1)

    return df   
#base orchestrator
def make_base_features(
    df: pd.DataFrame,
    zone_col: str = "zone_id",
    time_col: str = "datetime",
) -> pd.DataFrame:
    """
    Features shared by All models(LGBM/TCN/Transformer):
    - sorted, deputed
    - time features (including cyclical)
    - wind physicss from U/V(ws/wd/veer/shear)
    """
    df = (
        df
        .sort_values([zone_col, time_col])
        .drop_duplicates(subset=[zone_col, time_col])
        .reset_index(drop=True)
    )
    df = add_time_features(df, time_col=time_col, add_cyclical=True)
    df = add_wind_features(df, zone_col=zone_col)
    return df

def add_ml_stats(
    df: pd.DataFrame,
    zone_col: str = "zone_id",
    target_col: str = "target",
    lags: Iterable[int] = (1, 2, 3, 6, 12, 24),
    roll_windows: Iterable[int] = (3, 6, 12, 24),
    weather_cols: Iterable[str] = ("ws10", "ws100", "shear"),
    weather_lags: Iterable[int] = (1, 3, 6),
    weather_rolls: Iterable[int] = (3, 6, 12),
) -> pd.DataFrame:
    """
    Hand-crafted stats for tree models ONLY (avoid for deep models):
      - target lags
      - target rolling stats (mean/std) using only past info (shift(1))
      - minimal weather lags/rolling
    """
    df = df.copy()
    g = df.groupby(zone_col, group_keys=False)

    # target lags:
    for L in lags:
        df[f"{target_col}_lag{L}"] = g[target_col].shift(L)

    # target rolling stats(past only)
    for w in roll_windows:
        past = g[target_col].shift(1)
        df[f"{target_col}_roll{w}_mean"] = past.rolling(w, min_periods=1).mean()
        df[f"{target_col}_roll{w}_std"] = past.rolling(w, min_periods=1).std()
    
    return df

def make_ml_features(
    df: pd.DataFrame,
    zone_col: str = "zone_id",
    time_col: str = "datetime",
    target_col: str = "target",
    lags: Iterable[int] = (1, 2, 3, 6, 12, 24),
    roll_windows: Iterable[int] = (3, 6, 12, 24),
) -> pd.DataFrame:
    """
    Pipeline for tree models:
      base -> add_ml_stats
    """
    df = make_base_features(df, zone_col = zone_col, time_col=time_col)
    df = add_ml_stats(df, zone_col=zone_col, target_col=target_col,
                      lags=lags, roll_windows=roll_windows)
    return df

# sequence windowing for deep models
def _window_stack(arr: np.ndarray, L: int, H: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows:
       Inputs: past L steps
       Targets: next H steps(default 1)
    Returns:
       X:[N, L, D], y: [N, H] or [N, H, Dy]
    """
    T = arr.shape[0]
    N = T - L - H + 1 
    if N <= 0:
        raise ValueError(f"Not enough samples to make windows: T={T}, L={L}, H={H}")
    # inputs
    X = np.lib.stride_tricks.sliding_window_view(arr, window_shape=(L, arr.shape[1])[:-H, 0, : ])
    #targets:assume the target is in the first column of y
    yw = np.lib.stride_tricks.sliding_window_view(arr[:, 0], window_shape= H) # [T-H+1, H]
    y = yw[L : L + N]
    return X, y

def make_seq_features(
    df: pd.DataFrame,
    zone_col: str = "zone_id",
    time_col: str = "datetime",
    target_col: str = "target",
    lookback: int = 168, # e.g.: 7 days if hourly
    horizon: int = 1, # single_step or multi-step
    feature_cols: Optional[List[str]] = None,
    include_target_as_input: bool = True,
    add_missing_mask: bool = True
) -> Dict[str,np.ndarray]:
    """
    Build sequence tensors for deep models (TCN/Transformer).
    Returns a dict with arrays ready for DataLoader.
       -X: [N, L, D]
       -y: [N, H]
       -zone: [N] zone_id for embedding (optional)
       -mask_missing: [N, L, 1] (optional)
    Notes:
      1) This function assumes df already finished imputation (make_dataset).
      2) Use only base features + raw sensors for deep models (no rolling stats).
    """
    # base features first
    df = make_base_features(df, zone_col=zone_col, time_col=time_col)

    # choose input columns
    if feature_cols is None:
        # Minimal but strong default set
        feature_cols = [
            target_col,
            "U10", "V10", "U100", "V100",
            "ws10", "ws100", "shear",
            "wd10_sin", "wd10_cos", "wd100_sin", "wd100_cos",
            "sin_hour", "cos_hour", "sin_doy", "cos_doy",
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]

    #group per zone, then windw; concatenate
    arrays_X, arrays_y, arrays_zone, arrays_mask = [],[],[],[]
    for zid, g in df.sort_values([zone_col, time_col]).groupby(zone_col):
        g = g.reset_index(drop=True)

        # build raw matrix[T, D]
        mat = g[feature_cols].to_numpy(dtype=np.float32)

        # if not including target in input, move it to y only
        if not include_target_as_input:
            # ensure target_col is first for y
            y_vec = g[[target_col]].to_numpy(dtype=np.float32)
            in_cols = [c for c in feature_cols if c != target_col]
            mat_in = g[in_cols].to_numpy(dtype=np.float32)
        else:
            #target as part of input
            y_vec = g[[target_col]].to_numpy(dtype=np.float32)
            mat_in = mat

        # missing mask (1 = missing) for the actual input matrix
        miss = None
        if add_missing_mask:
            miss = np.isnan(mat_in).astype(np.float32)

        T = len(g)
        N = T - lookback - horizon + 1
        if N <= 0:
            continue

        #windowing
        #inputs
        X = np.lib.stride_tricks.sliding_window_view(
            mat_in, window_shape=(lookback, mat_in.shape[1])
        )[:-horizon, 0, :] #[N,L,D]
        # targets (aligned with X)
        yw = np.lib.stride_tricks.sliding_window_view(
            y_vec.squeeze(-1), window_shape=horizon
        ) # [T - horizon + 1, H]
        y = yw[lookback : lookback + N] # [N, H]
        if horizon == 1:
            y = y.squeeze(-1) #[N]

        arrays_X.append(X.astype(np.float32))
        arrays_y.append(y.astype(np.float32))
        arrays_zone.append(np.full((X.shape[0],),fill_value=zid, dtype=np.int64))

        if add_missing_mask and (miss is not None):
            M = np.lib.stride_tricks.sliding_window_view(
                miss, window_shape=(lookback, miss.shape[1])
            )[:-horizon, 0, :]
            arrays_mask.append(M[..., :1].astype(np.float32))  #keep 1-channel mask

    if not arrays_X:
        raise ValueError("No sequences were created. Check lookback/horizon and data length per zone.")
        
    X = np.concatenate(arrays_X, axis=0)      #[N, L, D]
    y = np.concatenate(arrays_y, axis=0)      #[N] or [N, H]
    zone_ids = np.concatenate(arrays_zone, axis=0)  #[N]
    out = {"X":X, "y":y, "zone":zone_ids}

    if add_missing_mask and arrays_mask:
        out["mask_missing"] = np.concatenate(arrays_mask, axis=0) #[N, L, 1]

    return out
    
# Convenience: column selection for ML
def select_ml_xy(
    df: pd.DataFrame,
    target_col: str = "target",
    drop_cols: Iterable[str] = ("datetime",),
    keep_zone: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    split X/y for ML models after make_ml_features().
    keep_zone = True -> keep zone_id as classification feature, otherwise drop zone_id
    """
    drops = set(drop_cols) | {target_col}
    if not keep_zone and "zone_id" in df.columns:
        drops.add("zone_id")
    X = df.drop(columns=[c for c in drops if c in df.columns])
    y = df[target_col]
    return X, y
