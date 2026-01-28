from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def t_crit_95(n: int) -> float:
    """
    Two-sided 95% t critical value with df=n-1.
    If scipy is available, use exact; otherwise fall back to 1.96 (normal approx).
    """
    if n <= 1:
        return float("nan")
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(0.975, df=n - 1))
    except Exception:
        # normal approximation
        return 1.96


def mean_ci95(x: np.ndarray) -> Dict[str, float]:
    x = x.astype(float)
    x = x[~np.isnan(x)]
    n = int(len(x))
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    crit = t_crit_95(n)
    ci95 = float(crit * std / math.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "std": std, "ci95": ci95}


def summarize(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").to_numpy()
            stats = mean_ci95(vals)
            rows.append(
                {
                    **base,
                    "metric": m,
                    **stats,
                }
            )
    out = pd.DataFrame(rows)
    # nicer ordering
    if not out.empty:
        out = out.sort_values(group_cols + ["metric"]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_csv", type=str, required=True, help="Path to conformal_runs.csv")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <runs_csv_dir>/tables)")
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["apply_cal_picp", "apply_cal_mpiw", "apply_raw_picp", "apply_raw_mpiw", "t"],
        help="Metric columns to summarize",
    )
    ap.add_argument(
        "--track2_by_heldout",
        action="store_true",
        help="Also output Track2 summaries grouped by heldout",
    )
    args = ap.parse_args()

    runs_csv = Path(args.runs_csv).resolve()
    if not runs_csv.exists():
        raise FileNotFoundError(runs_csv)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (runs_csv.parent / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(runs_csv)

    # Ensure columns exist
    need_cols = ["track", "model", "candidate"] + args.metrics
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    # ---------- Overall (track × model × candidate) ----------
    overall = summarize(df, group_cols=["track", "model", "candidate"], metrics=args.metrics)
    out_overall = out_dir / "conformal_mean_ci95_overall.csv"
    overall.to_csv(out_overall, index=False, encoding="utf-8")
    print(f"[OK] Saved: {out_overall}")

    # ---------- Track2 by heldout (optional) ----------
    if args.track2_by_heldout:
        if "heldout" not in df.columns:
            print("[WARN] No 'heldout' column found; skip track2_by_heldout.")
        else:
            d2 = df[df["track"].astype(str) == "track2"].copy()
            # normalize heldout to numeric-ish (keeps NaN if not parseable)
            d2["heldout"] = pd.to_numeric(d2["heldout"], errors="coerce")
            d2 = d2.dropna(subset=["heldout"])
            if len(d2) == 0:
                print("[WARN] Track2 has no valid heldout rows after parsing; skip.")
            else:
                by_hold = summarize(
                    d2,
                    group_cols=["track", "heldout", "model", "candidate"],
                    metrics=args.metrics,
                )
                out_hold = out_dir / "conformal_mean_ci95_track2_by_heldout.csv"
                by_hold.to_csv(out_hold, index=False, encoding="utf-8")
                print(f"[OK] Saved: {out_hold}")

    print("[DONE]")


if __name__ == "__main__":
    main()
