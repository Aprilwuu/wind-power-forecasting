import json
import pandas as pd
from pathlib import Path

def main(grid_dir):
    rows = []
    grid_dir = Path(grid_dir)

    for combo_dir in grid_dir.iterdir():
        if not combo_dir.is_dir():
            continue
        for seed_dir in combo_dir.iterdir():
            metrics_file = seed_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    m = json.load(f)
                row = {
                    "combo": combo_dir.name,
                    "seed": seed_dir.name,
                    "val_rmse": m["val"]["rmse"],
                    "val_mae": m["val"]["mae"],
                    "test_rmse": m["test"]["rmse"],
                    "test_mae": m["test"]["mae"]
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    numeric_cols = ["val_rmse", "val_mae", "test_rmse", "test_mae"]

    summary = df.groupby("combo")[numeric_cols].agg(["mean", "std"])    
    out_path = grid_dir / "grid_summary.csv"
    summary.to_csv(out_path)
    print("Saved to", out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.grid_dir)
