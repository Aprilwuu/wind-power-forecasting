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


def detect_runs(base_dir: Path):
    """
    支持两种结构：
    1) Track 1: base_dir/seed_42
    2) Track 2: base_dir/heldout_1/seed_42
    """
    runs = []

    heldout_dirs = sorted([p for p in base_dir.glob("heldout_*") if p.is_dir()])
    if heldout_dirs:
        for heldout_dir in heldout_dirs:
            seed_dirs = sorted([p for p in heldout_dir.glob("seed_*") if p.is_dir()])
            for seed_dir in seed_dirs:
                runs.append((heldout_dir.name, seed_dir.name, seed_dir))
        return runs

    seed_dirs = sorted([p for p in base_dir.glob("seed_*") if p.is_dir()])
    for seed_dir in seed_dirs:
        runs.append(("", seed_dir.name, seed_dir))

    return runs


def estimate_training_time(run_dir: Path):
    # 开始文件：只允许这两个
    start_file = first_existing_file(
        run_dir,
        ["runtime.json", "runtime", "config_snapshot.json", "config_snapshot"]
    )

    # 结束文件：只允许训练结束核心文件
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
    # 改成你当前要计算的目录
    base_dir = Path(r"E:\Projects\wind-power-forecasting\data\featured\transformer_qr_track1_lb168\baseline")

    rows = []
    runs = detect_runs(base_dir)

    if not runs:
        print("No run folders found.")
        return

    for heldout, seed, run_dir in runs:
        result = estimate_training_time(run_dir)

        if result is None:
            rows.append({
                "heldout": heldout,
                "seed": seed,
                "start_file": "",
                "end_file": "",
                "start_time": "",
                "end_time": "",
                "duration_min": None,
            })
        else:
            rows.append({
                "heldout": heldout,
                "seed": seed,
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

    mean_time = valid_df["duration_min"].mean()
    min_time = valid_df["duration_min"].min()
    max_time = valid_df["duration_min"].max()

    print("\nOverall summary:")
    print(f"Average: {mean_time:.2f} min")
    print(f"Min:     {min_time:.2f} min")
    print(f"Max:     {max_time:.2f} min")
    print(f"Runs:    {len(valid_df)}")

    df.to_csv(base_dir / "estimated_runtime_per_run.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame([{
        "avg_training_time_min": round(mean_time, 2),
        "min_training_time_min": round(min_time, 2),
        "max_training_time_min": round(max_time, 2),
        "num_runs": len(valid_df),
    }])

    summary_df.to_csv(base_dir / "estimated_runtime_model_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved: {base_dir / 'estimated_runtime_per_run.csv'}")
    print(f"Saved: {base_dir / 'estimated_runtime_model_summary.csv'}")


if __name__ == "__main__":
    main()