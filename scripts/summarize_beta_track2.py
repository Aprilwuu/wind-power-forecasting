from pathlib import Path
import json
import pandas as pd


def first_existing_file(run_dir: Path, names):
    for name in names:
        p = run_dir / name
        if p.exists() and p.is_file():
            return p
    return None


def load_metrics(run_dir: Path):
    metrics_file = first_existing_file(run_dir, ["metrics.json", "metrics"])
    if metrics_file is None:
        return None

    with metrics_file.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    val = metrics.get("val", {})
    test = metrics.get("test", {})

    return {
        "val_rmse": val.get("rmse"),
        "val_mae": val.get("mae"),
        "val_r2": val.get("r2"),
        "val_picp": val.get("picp"),
        "val_mpiw": val.get("mpiw"),
        "test_rmse": test.get("rmse"),
        "test_mae": test.get("mae"),
        "test_r2": test.get("r2"),
        "test_picp": test.get("picp"),
        "test_mpiw": test.get("mpiw"),
    }


def main():
    base_dir = Path(r"E:\Projects\wind-power-forecasting\data\featured\beta_trans_track2")

    rows = []

    heldout_dirs = sorted([p for p in base_dir.glob("heldout_*") if p.is_dir()])
    if not heldout_dirs:
        print("No heldout_* folders found.")
        return

    for heldout_dir in heldout_dirs:
        heldout = heldout_dir.name

        seed_dirs = sorted([p for p in heldout_dir.glob("seed_*") if p.is_dir()])
        if not seed_dirs:
            continue

        for seed_dir in seed_dirs:
            seed = seed_dir.name

            metrics = load_metrics(seed_dir)
            if metrics is None:
                rows.append({
                    "heldout": heldout,
                    "seed": seed,
                    "val_rmse": None,
                    "val_mae": None,
                    "val_r2": None,
                    "val_picp": None,
                    "val_mpiw": None,
                    "test_rmse": None,
                    "test_mae": None,
                    "test_r2": None,
                    "test_picp": None,
                    "test_mpiw": None,
                })
                continue

            row = {
                "heldout": heldout,
                "seed": seed,
                **metrics
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("No runs found.")
        return

    print("\n=== Per-run results ===\n")
    print(df.to_string(index=False))

    metric_cols = [
        "val_rmse", "val_mae", "val_r2", "val_picp", "val_mpiw",
        "test_rmse", "test_mae", "test_r2", "test_picp", "test_mpiw"
    ]

    # ---------- per-heldout mean/std ----------
    heldout_summary = (
        df.groupby("heldout")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # flatten columns
    heldout_summary.columns = [
        "heldout" if col[0] == "heldout" else f"{col[0]}_{col[1]}"
        for col in heldout_summary.columns
    ]

    print("\n=== Per-heldout mean ± std across seeds ===\n")
    print(heldout_summary.to_string(index=False))

    # ---------- overall mean/std across all runs ----------
    overall_rows = []
    for col in metric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        overall_rows.append({
            "metric": col,
            "mean": s.mean(),
            "std": s.std(ddof=1) if len(s) > 1 else 0.0,
            "mean_std": f"{s.mean():.4f} ± {s.std(ddof=1):.4f}" if len(s) > 1 else f"{s.mean():.4f} ± 0.0000",
            "n_runs": len(s),
        })

    overall_summary = pd.DataFrame(overall_rows)

    print("\n=== Overall mean ± std across all heldouts and seeds ===\n")
    print(overall_summary.to_string(index=False))

    # ---------- nicer formatted per-heldout table ----------
    pretty_rows = []
    for _, row in heldout_summary.iterrows():
        pretty_row = {"heldout": row["heldout"]}
        for col in metric_cols:
            mean_val = row.get(f"{col}_mean")
            std_val = row.get(f"{col}_std")
            if pd.notna(mean_val):
                if pd.notna(std_val):
                    pretty_row[col] = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    pretty_row[col] = f"{mean_val:.4f} ± 0.0000"
            else:
                pretty_row[col] = ""
        pretty_rows.append(pretty_row)

    pretty_heldout_summary = pd.DataFrame(pretty_rows)

    print("\n=== Pretty per-heldout mean ± std table ===\n")
    print(pretty_heldout_summary.to_string(index=False))

    # ---------- save ----------
    df.to_csv(base_dir / "beta_track2_per_run.csv", index=False, encoding="utf-8-sig")
    heldout_summary.to_csv(base_dir / "beta_track2_summary_by_heldout.csv", index=False, encoding="utf-8-sig")
    pretty_heldout_summary.to_csv(base_dir / "beta_track2_summary_by_heldout_pretty.csv", index=False, encoding="utf-8-sig")
    overall_summary.to_csv(base_dir / "beta_track2_overall_mean_std.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved: {base_dir / 'beta_track2_per_run.csv'}")
    print(f"Saved: {base_dir / 'beta_track2_summary_by_heldout.csv'}")
    print(f"Saved: {base_dir / 'beta_track2_summary_by_heldout_pretty.csv'}")
    print(f"Saved: {base_dir / 'beta_track2_overall_mean_std.csv'}")


if __name__ == "__main__":
    main()