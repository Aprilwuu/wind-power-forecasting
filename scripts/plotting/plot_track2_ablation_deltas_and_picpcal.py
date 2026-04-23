from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns exist: {candidates}. Available: {list(df.columns)}")


def require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nAvailable columns: {list(df.columns)}")


def standardize_model_name(s: str) -> str:
    return str(s).strip().lower().replace("_", "-").replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to joined_all_runs.csv (or similar run-level summary).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for figures.")
    ap.add_argument("--track", type=str, default="track2", choices=["track1", "track2"])
    ap.add_argument("--noise_name", type=str, default="tcn-mc-noise", help="Model name for noise variant (case-insensitive).")
    ap.add_argument("--cal_name", type=str, default="tcn-mc-cal", help="Model name for conformal-only variant (case-insensitive).")
    ap.add_argument("--title_suffix", type=str, default="Track2 LOFO (noise − cal)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(args.csv)

    # --- column mapping (your csv may use slightly different names) ---
    col_track = pick_col(df, ["track", "protocol"])
    col_model = pick_col(df, ["model", "method"])
    col_seed = pick_col(df, ["seed", "random_seed"])
    col_heldout = pick_col(df, ["heldout", "fold", "zone"])

    col_picp_cal = pick_col(df, ["apply_cal_picp", "picp_cal", "PICP_cal", "apply_picp_cal"])
    col_mpiw_cal = pick_col(df, ["apply_cal_mpiw", "mpiw_cal", "MPIW_cal", "apply_mpiw_cal"])

    # basic requirements
    require_cols(df, [col_track, col_model, col_seed, col_heldout, col_picp_cal, col_mpiw_cal])

    # normalize
    df[col_track] = df[col_track].astype(str).str.lower().str.strip()
    df[col_model] = df[col_model].astype(str).map(standardize_model_name)
    df[col_seed] = pd.to_numeric(df[col_seed], errors="coerce")
    df[col_heldout] = pd.to_numeric(df[col_heldout], errors="coerce")
    df[col_picp_cal] = pd.to_numeric(df[col_picp_cal], errors="coerce")
    df[col_mpiw_cal] = pd.to_numeric(df[col_mpiw_cal], errors="coerce")

    df = df.dropna(subset=[col_seed, col_heldout, col_picp_cal, col_mpiw_cal])
    df[col_seed] = df[col_seed].astype(int)
    df[col_heldout] = df[col_heldout].astype(int)

    # --- filter to track2 ---
    d = df[df[col_track] == args.track].copy()
    if d.empty:
        raise ValueError(f"No rows found for {args.track}. Unique tracks: {sorted(df[col_track].unique())}")

    noise = standardize_model_name(args.noise_name)
    cal = standardize_model_name(args.cal_name)

    d_noise = d[d[col_model] == noise].copy()
    d_cal = d[d[col_model] == cal].copy()

    if d_noise.empty or d_cal.empty:
        raise ValueError(
            f"Missing variants in csv.\n"
            f"Found noise rows: {len(d_noise)} for model='{noise}'\n"
            f"Found cal rows:   {len(d_cal)} for model='{cal}'\n"
            f"Available models in {args.track}: {sorted(d[col_model].unique())}"
        )

    # ========== FIG A: heldout-wise deltas (noise - cal) ==========
    # join by (heldout, seed) so deltas are computed on matched runs
    keys = [col_heldout, col_seed]
    j = pd.merge(
        d_noise[keys + [col_picp_cal, col_mpiw_cal]],
        d_cal[keys + [col_picp_cal, col_mpiw_cal]],
        on=keys,
        suffixes=("_noise", "_cal"),
        how="inner",
    )

    if j.empty:
        raise ValueError("No matched (heldout, seed) pairs between noise and cal variants. Check heldout/seed fields.")

    j["delta_picp"] = j[f"{col_picp_cal}_noise"] - j[f"{col_picp_cal}_cal"]
    j["delta_mpiw"] = j[f"{col_mpiw_cal}_noise"] - j[f"{col_mpiw_cal}_cal"]

    # summarize per heldout (mean ± std across seeds)
    g = j.groupby(col_heldout, as_index=False).agg(
        mean_dpicp=("delta_picp", "mean"),
        std_dpicp=("delta_picp", "std"),
        mean_dmpiw=("delta_mpiw", "mean"),
        std_dmpiw=("delta_mpiw", "std"),
        n=("delta_picp", "count"),
    ).sort_values(col_heldout)

    g["std_dpicp"] = g["std_dpicp"].fillna(0.0)
    g["std_dmpiw"] = g["std_dmpiw"].fillna(0.0)

    heldouts = g[col_heldout].to_numpy()
    x = np.arange(len(heldouts))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharex=True)

    # left: ΔPICP
    axes[0].bar(x, g["mean_dpicp"].to_numpy(), yerr=g["std_dpicp"].to_numpy(), capsize=4)
    axes[0].axhline(0.0, linestyle="--", linewidth=1)
    axes[0].set_title("ΔPICP_cal (noise − cal)")
    axes[0].set_xlabel("Heldout zone")
    axes[0].set_ylabel("ΔPICP")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)

    # right: ΔMPIW
    axes[1].bar(x, g["mean_dmpiw"].to_numpy(), yerr=g["std_dmpiw"].to_numpy(), capsize=4)
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_title("ΔMPIW_cal (noise − cal)")
    axes[1].set_xlabel("Heldout zone")
    axes[1].set_ylabel("ΔMPIW")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(h)) for h in heldouts])

    fig.suptitle(f"{args.title_suffix}: Decomposing ΔWIS into coverage vs width", y=1.03)
    fig.tight_layout()
    out_a = out_dir / "track2_heldout_delta_picp_and_mpiw_noise_minus_cal.png"
    fig.savefig(out_a, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ========== FIG B: reliability (PICP_cal) by heldout ==========
    # Make a 1x2 panel: left=cal, right=noise (clean & readable)
    heldouts_sorted = sorted(d[col_heldout].unique().tolist())

    def collect_box_data(dd: pd.DataFrame) -> list[np.ndarray]:
        data = []
        for h in heldouts_sorted:
            v = pd.to_numeric(dd[dd[col_heldout] == h][col_picp_cal], errors="coerce").dropna().to_numpy()
            data.append(v)
        return data

    data_cal = collect_box_data(d_cal)
    data_noise = collect_box_data(d_noise)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)

    axes[0].boxplot(data_cal, labels=[str(int(h)) for h in heldouts_sorted], showfliers=False)
    axes[0].axhline(0.90, linestyle="--", linewidth=1)
    axes[0].set_title(f"{args.cal_name}: PICP_cal by heldout")
    axes[0].set_xlabel("Heldout zone")
    axes[0].set_ylabel("PICP_cal")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)

    axes[1].boxplot(data_noise, labels=[str(int(h)) for h in heldouts_sorted], showfliers=False)
    axes[1].axhline(0.90, linestyle="--", linewidth=1)
    axes[1].set_title(f"{args.noise_name}: PICP_cal by heldout")
    axes[1].set_xlabel("Heldout zone")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(f"{args.track.upper()} reliability after calibration: heldout-wise PICP_cal", y=1.03)
    fig.tight_layout()
    out_b = out_dir / "track2_heldout_picpcal_boxplots_cal_vs_noise.png"
    fig.savefig(out_b, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:")
    print(" -", out_a)
    print(" -", out_b)


if __name__ == "__main__":
    main()