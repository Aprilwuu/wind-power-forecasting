#scripts/wind_all_zones.py
from pathlib import Path
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Merge Zone CSV files into one")
    parser.add_argument("--input", type=str, default="data/raw", help="raw data path")
    parser.add_argument("--output", type=str, default="data/processed/gefcom_wind_all_zones.csv", help="output file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob("Task15_W_Zone*.csv"))
    if not files:
        print(f"Not finding Task15_W_Zone*.csv")
        return
    
    print(f" find {len(files)} files, will combine:")
    for f in files:
        print(" -", f.name)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "TIMESTAMP" not in df.columns:
            raise ValueError(f"{f.name} Missing TIMESTAMP column")
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors = "coerce")
        df = df.dropna(subset=["TIMESTAMP"]).copy()
        dfs.append(df)
    
    full_df = pd.concat(dfs, axis=0, ignore_index=True)
    full_df = full_df.drop_duplicates()
    full_df = full_df.sort_values(["ZONEID", "TIMESTAMP"])

    full_df.to_csv(output_path, index=False)
    print(f"Combination done!, output file:{output_path}")
    print(f"The number of rows of the dataset:{len(full_df)}, The number of columns of the dataset: {len(full_df.columns)}")

if __name__ == "__main__":
    main()