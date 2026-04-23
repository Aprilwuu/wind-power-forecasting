from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def _ensure_dir(p: Path) -> None:
    """Create directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def _mean_ci95(x: np.ndarray) -> Tuple[float, float]:
    """
    Return mean and 95% CI half-width (mean ± ci95).
    NOTE: This is CI across runs (seeds), not a predictive interval.
    """
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    ci = 1.96 * s / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, ci


def summarize(df: pd.DataFrame, value_col: str, group_cols: List[str]) -> pd.DataFrame:
    """Group by group_cols and compute mean±95%CI for value_col."""
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = pd.to_numeric(g[value_col], errors="coerce").to_numpy(dtype=float)
        m, ci = _mean_ci95(vals)
        row = dict(zip(group_cols, keys))
        row.update({"mean": m, "ci95": ci, "n": len(g)})
        rows.append(row)
    return pd.DataFrame(rows)


def _prep_long(d: pd.DataFrame, raw_col: str, cal_col: str) -> pd.DataFrame:
    """
    Convert wide format (raw/cal columns) into long format with columns:
    model, stage in {raw, cal}, value
    """
    long = pd.concat(
        [
            d.assign(stage="raw", value=d[raw_col]),
            d.assign(stage="cal", value=d[cal_col]),
        ],
        ignore_index=True,
    )[["model", "stage", "value"]]
    return long


def _ordered_summary(long: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize long df and enforce a stable x-order:
    for each model: model-raw, model-cal
    """
    s = summarize(long, "value", ["model", "stage"])
    s["x"] = s["model"].astype(str) + "-" + s["stage"].astype(str)

    order = []
    for m in sorted(s["model"].unique()):
        order += [f"{m}-raw", f"{m}-cal"]

    s = s.set_index("x").reindex(order).reset_index()
    return s


def bar_with_error(ax, summary_df: pd.DataFrame, title: str, ylabel: str, nominal_line: float | None = None):
    """Draw a bar chart with 95% CI error bars."""
    xs = np.arange(len(summary_df))
    means = summary_df["mean"].to_numpy()
    errs = summary_df["ci95"].to_numpy()

    ax.bar(xs, means, yerr=errs, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(summary_df["x"].tolist(), rotation=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if nominal_line is not None:
        ax.axhline(nominal_line, linestyle="--", linewidth=1)


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_metric_by_track(
    df: pd.DataFrame,
    tracks: List[str],
    raw_col: str,
    cal_col: str,
    out_path: Path,
    ylabel: str,
    suptitle: str,
    y_lim: Tuple[float, float] | None = None,
    nominal_line: float | None = None,
):
    """
    Create a 1x2 figure: left=track1, right=track2.
    Each panel shows raw vs cal bars for each model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, track in zip(axes, tracks):
        d = df[df["track"] == track].copy()
        if len(d) == 0:
            ax.set_axis_off()
            continue

        long = _prep_long(d, raw_col=raw_col, cal_col=cal_col)
        s = _ordered_summary(long)

        bar_with_error(
            ax,
            s,
            title=f"{track.upper()}",
            ylabel=ylabel if ax is axes[0] else "",  # keep y-label on left only
            nominal_line=nominal_line,
        )

    if y_lim is not None:
        axes[0].set_ylim(*y_lim)

    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_scatter_by_track(df: pd.DataFrame, out_path: Path):
    """
    1x2 scatter plot: Track1 (left), Track2 (right).
    X = calibrated MPIW (apply), Y = calibrated PICP (apply).
    Keeps shared axes for fair visual comparison.
    """
    model_label = {
        "lgbm": "LGBM-QR + Conformal",
        "tcn": "TCN-MC+Noise + Conformal",
    }

    # Colors (you can remove explicit colors if you prefer default matplotlib cycling)
    colors = {
        ("track1", "lgbm"): "tab:blue",
        ("track1", "tcn"): "tab:orange",
        ("track2", "lgbm"): "tab:green",
        ("track2", "tcn"): "tab:red",
    }

    x_all = pd.to_numeric(df["apply_cal_mpiw"], errors="coerce")
    y_all = pd.to_numeric(df["apply_cal_picp"], errors="coerce")
    mask = x_all.notna() & y_all.notna()

    if mask.sum() == 0:
        print("[WARN] No valid apply_cal_mpiw/apply_cal_picp values for tradeoff scatter.")
        return

    x_all = x_all[mask].to_numpy()
    y_all = y_all[mask].to_numpy()

    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))

    # Add padding so points don't touch borders
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.01
    y_pad = 0.08 * (y_max - y_min) if y_max > y_min else 0.01
    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)

    for ax, track in zip(axes, ["track1", "track2"]):
        d = df[df["track"] == track].copy()
        d["x"] = pd.to_numeric(d["apply_cal_mpiw"], errors="coerce")
        d["y"] = pd.to_numeric(d["apply_cal_picp"], errors="coerce")
        d = d.dropna(subset=["x", "y"])

        for model in ["lgbm", "tcn"]:
            g = d[d["model"] == model]
            if len(g) == 0:
                continue
            ax.scatter(
                g["x"],
                g["y"],
                alpha=0.85,
                label=model_label.get(model, model),
                color=colors.get((track, model), None),
            )

        ax.axhline(0.90, linestyle="--", linewidth=1)
        ax.set_title(f"{track.upper()}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.legend(frameon=True)

    fig.suptitle("Coverage–Width tradeoff after conformal calibration", y=1.02)
    fig.supxlabel("Calibrated MPIW (apply)")
    fig.supylabel("Calibrated PICP (apply)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_track2_boxplots(df: pd.DataFrame, out_picp: Path, out_mpiw: Path):
    """
    Track2 per-heldout boxplots (calibrated only).
    These are diagnostic plots; consider putting them in an appendix if space is limited.
    """
    d2 = df[df["track"] == "track2"].copy()
    if len(d2) == 0 or "heldout" not in d2.columns:
        return

    d2["heldout"] = pd.to_numeric(d2["heldout"], errors="coerce")
    d2 = d2.dropna(subset=["heldout"])
    if len(d2) == 0:
        return

    heldouts = sorted(d2["heldout"].unique().tolist())
    heldout_labels = [str(int(h)) for h in heldouts]

    # PICP boxplot
    fig = plt.figure()
    ax = plt.gca()
    data = [pd.to_numeric(d2[d2["heldout"] == h]["apply_cal_picp"], errors="coerce").dropna().to_numpy() for h in heldouts]
    ax.boxplot(data, labels=heldout_labels, showfliers=False)
    ax.axhline(0.90, linestyle="--", linewidth=1)  # nominal coverage line
    ax.set_title("Track2 per-heldout: calibrated PICP (apply)")
    ax.set_xlabel("Heldout zone")
    ax.set_ylabel("PICP")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_picp, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # MPIW boxplot
    fig = plt.figure()
    ax = plt.gca()
    data = [pd.to_numeric(d2[d2["heldout"] == h]["apply_cal_mpiw"], errors="coerce").dropna().to_numpy() for h in heldouts]
    ax.boxplot(data, labels=heldout_labels, showfliers=False)
    ax.set_title("Track2 per-heldout: calibrated MPIW (apply)")
    ax.set_xlabel("Heldout zone")
    ax.set_ylabel("MPIW")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_mpiw, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs_csv",
        type=str,
        required=True,
        help="Path to conformal_runs.csv (generated by summarize_conformal.py)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for figures (default: <runs_csv_dir>/figures)",
    )
    args = ap.parse_args()

    runs_csv = Path(args.runs_csv).resolve()
    if not runs_csv.exists():
        raise FileNotFoundError(runs_csv)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (runs_csv.parent / "figures")
    _ensure_dir(out_dir)

    df = pd.read_csv(runs_csv)

    # Ensure expected columns exist (fill with NaN if missing)
    needed = [
        "track", "model",
        "apply_raw_picp", "apply_cal_picp",
        "apply_raw_mpiw", "apply_cal_mpiw",
        "heldout",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    tracks = ["track1", "track2"]

    # FIG A: combined PICP by track (raw vs cal)
    plot_metric_by_track(
        df=df,
        tracks=tracks,
        raw_col="apply_raw_picp",
        cal_col="apply_cal_picp",
        out_path=out_dir / "picp_apply_by_track.png",
        ylabel="PICP",
        suptitle="PICP before/after conformal calibration (apply)",
        y_lim=(0.80, 0.95),     # adjust if your values are outside this range
        nominal_line=0.90,
    )

    # FIG B: combined MPIW by track (raw vs cal)
    plot_metric_by_track(
        df=df,
        tracks=tracks,
        raw_col="apply_raw_mpiw",
        cal_col="apply_cal_mpiw",
        out_path=out_dir / "mpiw_apply_by_track.png",
        ylabel="MPIW",
        suptitle="MPIW before/after conformal calibration (apply)",
        y_lim=None,
        nominal_line=None,      # no nominal line for width
    )

    # FIG C: tradeoff scatter, two panels (Track1 / Track2)
    plot_tradeoff_scatter_by_track(
        df=df,
        out_path=out_dir / "tradeoff_scatter_cal_by_track.png",
    )

    # FIG D (optional): Track2 per-heldout boxplots (calibrated only)
    plot_track2_boxplots(
        df=df,
        out_picp=out_dir / "track2_box_picp.png",
        out_mpiw=out_dir / "track2_box_mpiw.png",
    )

    print(f"[OK] Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
