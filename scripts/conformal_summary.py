import json
from pathlib import Path

import numpy as np
import pandas as pd


def _picp(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def _mpiw(lo: np.ndarray, hi: np.ndarray) -> float:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return float(np.mean(hi - lo))


def _conformal_t(val_df: pd.DataFrame, y_col: str, lo_col: str, hi_col: str, target_picp: float) -> float:
    """
    Split conformal widening for two-sided interval:
      s_i = max(lo - y, y - hi, 0)
      t = Quantile_{target_picp}(s)
    """
    y = val_df[y_col].to_numpy(dtype=float)
    lo = val_df[lo_col].to_numpy(dtype=float)
    hi = val_df[hi_col].to_numpy(dtype=float)
    s = np.maximum(np.maximum(lo - y, y - hi), 0.0)

    q = float(target_picp)
    # “higher”保证覆盖率不低于目标（更保守一点）
    try:
        t = float(np.quantile(s, q, method="higher"))
    except TypeError:
        # numpy 旧版本
        t = float(np.quantile(s, q, interpolation="higher"))
    return t


def conformal_widen_files(
    val_path: str,
    test_path: str,
    *,
    y_col: str = "y_true",
    lo_col: str = "q05",
    hi_col: str = "q95",
    target_picp: float = 0.9,
    out_suffix: str = "_cal",
) -> dict:
    val_path = Path(val_path)
    test_path = Path(test_path)

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # basic checks
    for df, name in [(val_df, "val"), (test_df, "test")]:
        for c in [y_col, lo_col, hi_col]:
            if c not in df.columns:
                raise KeyError(f"[{name}] missing column: {c}. available={list(df.columns)}")

    raw = {
        "val_picp": _picp(val_df[y_col], val_df[lo_col], val_df[hi_col]),
        "val_mpiw": _mpiw(val_df[lo_col], val_df[hi_col]),
        "test_picp": _picp(test_df[y_col], test_df[lo_col], test_df[hi_col]),
        "test_mpiw": _mpiw(test_df[lo_col], test_df[hi_col]),
    }

    t = _conformal_t(val_df, y_col, lo_col, hi_col, target_picp=target_picp)

    # widen
    lo_cal = f"{lo_col}{out_suffix}"
    hi_cal = f"{hi_col}{out_suffix}"

    val_df[lo_cal] = val_df[lo_col].astype(float) - t
    val_df[hi_cal] = val_df[hi_col].astype(float) + t
    test_df[lo_cal] = test_df[lo_col].astype(float) - t
    test_df[hi_cal] = test_df[hi_col].astype(float) + t

    val_df[lo_cal] = np.clip(val_df[lo_cal], 0.0, 1.0)
    val_df[hi_cal] = np.clip(val_df[hi_cal], 0.0, 1.0)
    test_df[lo_cal] = np.clip(test_df[lo_cal], 0.0, 1.0)
    test_df[hi_cal] = np.clip(test_df[hi_cal], 0.0, 1.0)


    cal = {
        "t": float(t),
        "target_picp": float(target_picp),
        "val_picp_cal": _picp(val_df[y_col], val_df[lo_cal], val_df[hi_cal]),
        "val_mpiw_cal": _mpiw(val_df[lo_cal], val_df[hi_cal]),
        "test_picp_cal": _picp(test_df[y_col], test_df[lo_cal], test_df[hi_cal]),
        "test_mpiw_cal": _mpiw(test_df[lo_cal], test_df[hi_cal]),
    }

    # write outputs next to original files
    val_out = val_path.with_name(val_path.stem + out_suffix + val_path.suffix)
    test_out = test_path.with_name(test_path.stem + out_suffix + test_path.suffix)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    summary = {
        "val_path": str(val_path),
        "test_path": str(test_path),
        "val_out": str(val_out),
        "test_out": str(test_out),
        "y_col": y_col,
        "lo_col": lo_col,
        "hi_col": hi_col,
        "raw": raw,
        "calibrated": cal,
    }
    summary_path = val_path.parent / f"conformal_summary_{lo_col}_{hi_col}_{target_picp:.2f}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


if __name__ == "__main__":
    val_path = r"E:\Projects\wind-power-forecasting\data\featured\tcn_track1_mc500\baseline\seed_42\preds_val_mc.csv"
    test_path = r"E:\Projects\wind-power-forecasting\data\featured\tcn_track1_mc500\baseline\seed_42\preds_test_mc.csv"

    summary = conformal_widen_files(val_path, test_path, target_picp=0.9)
    print("raw val PICP:", summary["raw"]["val_picp"])
    print("raw test PICP:", summary["raw"]["test_picp"])
    print("widen t:", summary["calibrated"]["t"])
    print("cal val PICP:", summary["calibrated"]["val_picp_cal"])
    print("cal test PICP:", summary["calibrated"]["test_picp_cal"])
    print("saved:", summary["val_out"])
    print("saved:", summary["test_out"])
    print("summary:", str(Path(val_path).parent / "conformal_summary.json"))
