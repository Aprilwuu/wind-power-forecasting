# src/utils/conformal_widening.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

def _picp(y, lo, hi) -> float:
    y = np.asarray(y, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    return float(np.mean((y >= lo) & (y <= hi)))

def conformal_widen_from_csvs(
    *,
    val_csv: Path,
    test_csv: Path,
    target_picp: float = 0.9,
    y_col: str = "y_true",
    lo_col: str = "q05",
    hi_col: str = "q95",
    out_suffix: str = "_cal",
) -> dict:
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    yv = val_df[y_col].to_numpy(float)
    lo = val_df[lo_col].to_numpy(float)
    hi = val_df[hi_col].to_numpy(float)

    s = np.maximum(np.maximum(lo - yv, yv - hi), 0.0)
    try:
        t = float(np.quantile(s, target_picp, method="higher"))
    except TypeError:
        t = float(np.quantile(s, target_picp, interpolation="higher"))

    lo_cal = f"{lo_col}{out_suffix}"
    hi_cal = f"{hi_col}{out_suffix}"

    for df in (val_df, test_df):
        df[lo_cal] = df[lo_col].astype(float) - t
        df[hi_cal] = df[hi_col].astype(float) + t

    val_out = val_csv.with_name(val_csv.stem + out_suffix + val_csv.suffix)
    test_out = test_csv.with_name(test_csv.stem + out_suffix + test_csv.suffix)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    summary = {
        "t": t,
        "target_picp": target_picp,
        "raw": {
            "val_picp": _picp(val_df[y_col], val_df[lo_col], val_df[hi_col]),
            "test_picp": _picp(test_df[y_col], test_df[lo_col], test_df[hi_col]),
        },
        "cal": {
            "val_picp": _picp(val_df[y_col], val_df[lo_cal], val_df[hi_cal]),
            "test_picp": _picp(test_df[y_col], test_df[lo_cal], test_df[hi_cal]),
        },
        "val_out": str(val_out),
        "test_out": str(test_out),
    }

    (val_csv.parent / "conformal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def conformal_widen_exp_dir(
    exp_dir: str | Path,
    *,
    target_picp: float = 0.9,
    y_col: str = "y_true",
    lo_col: str = "q05",
    hi_col: str = "q95",
) -> dict:
    d = Path(exp_dir)
    # Track1 filenames
    val_csv = d / "preds_val_mc.csv"
    test_csv = d / "preds_test_mc.csv"
    if val_csv.exists() and test_csv.exists():
        return conformal_widen_from_csvs(val_csv=val_csv, test_csv=test_csv,
                                        target_picp=target_picp, y_col=y_col, lo_col=lo_col, hi_col=hi_col)

    # Track2 filenames
    val_csv = d / "preds_inner_val_mc.csv"
    test_csv = d / "preds_outer_test_mc.csv"
    if val_csv.exists() and test_csv.exists():
        return conformal_widen_from_csvs(val_csv=val_csv, test_csv=test_csv,
                                        target_picp=target_picp, y_col=y_col, lo_col=lo_col, hi_col=hi_col)

    raise FileNotFoundError(f"Cannot find MC csvs under {d}")
