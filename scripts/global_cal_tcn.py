from pathlib import Path
import pandas as pd
import numpy as np

from scripts.conformal_summary import _conformal_t, _picp, _mpiw 

def compute_global_t(val_paths, y_col="y_true", lo_col="q05", hi_col="q95", target_picp=0.9) -> float:
    dfs = []
    for p in val_paths:
        df = pd.read_csv(p, usecols=[y_col, lo_col, hi_col])
        dfs.append(df)
    all_val = pd.concat(dfs, axis=0, ignore_index=True)
    return _conformal_t(all_val, y_col, lo_col, hi_col, target_picp)

def apply_global_t_to_file(path, t, lo_col="q05", hi_col="q95", out_suffix="_gcal"):
    df = pd.read_csv(path)
    lo_new = f"{lo_col}{out_suffix}"
    hi_new = f"{hi_col}{out_suffix}"
    df[lo_new] = df[lo_col].astype(float) - t
    df[hi_new] = df[hi_col].astype(float) + t
    out = Path(path).with_name(Path(path).stem + out_suffix + Path(path).suffix)
    df.to_csv(out, index=False)
    return str(out)

if __name__ == "__main__":
    exp_root = Path(r"E:\Projects\wind-power-forecasting\data\featured\tcn_track2_mc500\baseline")

    val_paths = sorted(exp_root.rglob("preds_inner_val_mc.csv"))
    test_paths = sorted(exp_root.rglob("preds_outer_test_mc.csv"))

    t_global = compute_global_t(val_paths, target_picp=0.9)
    print("t_global =", t_global)

    rows = []
    for tp in test_paths:
        df = pd.read_csv(tp)
        y = df["y_true"].to_numpy(float)
        lo = df["q05"].to_numpy(float)
        hi = df["q95"].to_numpy(float)

        raw_picp = _picp(y, lo, hi)
        raw_mpiw = _mpiw(lo, hi)

        lo2 = lo - t_global
        hi2 = hi + t_global
        cal_picp = _picp(y, lo2, hi2)
        cal_mpiw = _mpiw(lo2, hi2)

        rows.append({
            "test_path": str(tp),
            "raw_picp": raw_picp,
            "raw_mpiw": raw_mpiw,
            "gcal_picp": cal_picp,
            "gcal_mpiw": cal_mpiw,
            "t_global": t_global,
        })

    out = pd.DataFrame(rows)
    out.to_csv(exp_root / "global_cal_report.csv", index=False)
    print("saved:", exp_root / "global_cal_report.csv")
