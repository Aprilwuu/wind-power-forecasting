from pathlib import Path
from datetime import datetime
import pandas as pd


def get_mtime(path: Path):
    if path.exists() and path.is_file():
        return datetime.fromtimestamp(path.stat().st_mtime)
    return None


def first_existing_file(run_dir: Path, names):
    for name in names:
        p = run_dir / name
        if p.exists() and p.is_file():
            return p
    return None


def estimate_training_time(run_dir: Path):
    start_file = first_existing_file(
        run_dir,
        ["runtime.json", "runtime", "config_snapshot.json", "config_snapshot"]
    )

    end_file = first_existing_file(
        run_dir,
        ["model.pt", "metrics.json", "metrics", "metrics_summary.json", "metrics_summary"]
    )

    if start_file is None or end_file is None:
        return None

    start_time = get_mtime(start_file)
    end_time = get_mtime(end_file)

    if start_time is None or end_time is None:
        return None

    duration_min = round((end_time - start_time).total_seconds() / 60.0, 2)

    if duration_min < 0:
        return None

    return {
        "start_file": start_file.name,
        "end_file": end_file.name,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_min": duration_min,
    }


def main():
    # 改成你的 tuning 实验总目录
    base_dir = Path(r"E:\Projects\wind-power-forecasting\data\featured")

    # 只扫 transformer tuning 这几个实验
    exp_names = [
        "trans_lr_3e4",
        "trans_small",
        "trans_mid",
        "trans_lr_1e4",
        "trans_lr_1e3",
        "trans_deeper",
    ]

    rows = []

    for exp_name in exp_names:
        exp_dir = base_dir / exp_name
        if not exp_dir.exists():
            rows.append({
                "exp_name": exp_name,
                "seed": "",
                "start_file": "",
                "end_file": "",
                "start_time": "",
                "end_time": "",
                "duration_min": None,
            })
            continue

        seed_dirs = sorted([p for p in exp_dir.glob("seed_*") if p.is_dir()])

        if not seed_dirs:
            rows.append({
                "exp_name": exp_name,
                "seed": "",
                "start_file": "",
                "end_file": "",
                "start_time": "",
                "end_time": "",
                "duration_min": None,
            })
            continue

        for seed_dir in seed_dirs:
            result = estimate_training_time(seed_dir)

            if result is None:
                rows.append({
                    "exp_name": exp_name,
                    "seed": seed_dir.name,
                    "start_file": "",
                    "end_file": "",
                    "start_time": "",
                    "end_time": "",
                    "duration_min": None,
                })
            else:
                rows.append({
                    "exp_name": exp_name,
                    "seed": seed_dir.name,
                    "start_file": result["start_file"],
                    "end_file": result["end_file"],
                    "start_time": result["start_time"],
                    "end_time": result["end_time"],
                    "duration_min": result["duration_min"],
                })

    df = pd.DataFrame(rows)

    print("\nEstimated runtime for each run:\n")
    print(df.to_string(index=False))

    valid_df = df.dropna(subset=["duration_min"])
    if valid_df.empty:
        print("\nNo valid runs found.")
        return

    summary_df = (
        valid_df.groupby("exp_name", as_index=False)["duration_min"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "avg_training_time_min",
            "min": "min_training_time_min",
            "max": "max_training_time_min",
            "count": "num_runs",
        })
    )

    print("\nSummary by experiment:\n")
    print(summary_df.to_string(index=False))

    df.to_csv(base_dir / "tuning_estimated_runtime_per_run.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(base_dir / "tuning_estimated_runtime_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved: {base_dir / 'tuning_estimated_runtime_per_run.csv'}")
    print(f"Saved: {base_dir / 'tuning_estimated_runtime_summary.csv'}")


if __name__ == "__main__":
    main()