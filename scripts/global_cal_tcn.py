import argparse
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

def parse_args():
    p = argparse.ArgumentParser("Global conformal calibration")
    p.add_argument("--exp_root", type=str, required=True, help="experiment root directory")
    p.add_argument("--target_picp", type=float, default=0.9)
    p.add_argument("--val_glob", type=str, default="**/preds_inner_val_mc.csv")
    p.add_argument("--test_glob", type=str, default="**/preds_outer_test_mc.csv")
    p.add_argument("--y_col", type=str, default="y_true")
    p.add_argument("--lo_col", type=str, default="q05")
    p.add_argument("--hi_col", type=str, default="q95")
    p.add_argument("--report_name", type=str, default="global_cal_report.csv")
    p.add_argument("--write_calibrated_csv", action="store_true",
                   help="also write calibrated csv copies with suffix")
    p.add_argument("--out_suffix", type=str, default="_gcal")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    exp_root = Path(args.exp_root)

    val_paths = sorted(exp_root.rglob(args.val_glob.replace("**/", ""))) if args.val_glob.startswith("**/") else sorted(exp_root.glob(args.val_glob))
    test_paths = sorted(exp_root.rglob(args.test_glob.replace("**/", ""))) if args.test_glob.startswith("**/") else sorted(exp_root.glob(args.test_glob))

    if len(val_paths) == 0:
        raise FileNotFoundError(f"No val files found under {exp_root} with pattern {args.val_glob}")
    if len(test_paths) == 0:
        raise FileNotFoundError(f"No test files found under {exp_root} with pattern {args.test_glob}")

    t_global = compute_global_t(
        val_paths,
        y_col=args.y_col, lo_col=args.lo_col, hi_col=args.hi_col,
        target_picp=args.target_picp
    )
    print("t_global =", t_global)

    rows = []
    for tp in test_paths:
        df = pd.read_csv(tp)
        y = df[args.y_col].to_numpy(float)
        lo = df[args.lo_col].to_numpy(float)
        hi = df[args.hi_col].to_numpy(float)

        raw_picp = _picp(y, lo, hi)
        raw_mpiw = _mpiw(lo, hi)

        lo2 = lo - t_global
        hi2 = hi + t_global
        cal_picp = _picp(y, lo2, hi2)
        cal_mpiw = _mpiw(lo2, hi2)

        row = {
            "test_path": str(tp),
            "raw_picp": raw_picp,
            "raw_mpiw": raw_mpiw,
            "gcal_picp": cal_picp,
            "gcal_mpiw": cal_mpiw,
            "t_global": t_global,
        }

        if args.write_calibrated_csv:
            out_csv = apply_global_t_to_file(
                tp, t_global,
                lo_col=args.lo_col, hi_col=args.hi_col,
                out_suffix=args.out_suffix
            )
            row["calibrated_csv"] = out_csv

        rows.append(row)

    out = pd.DataFrame(rows)
    report_path = exp_root / args.report_name
    out.to_csv(report_path, index=False)
    print("saved:", report_path)
