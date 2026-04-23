import pandas as pd
import glob
import os

valid_exps = [
    "trans_lr_1e4",
    "trans_lr_3e4",
    "trans_lr_1e3",
    "trans_small",
    "trans_mid",
    "trans_deeper"
]

files = glob.glob("reports/experiments/*/runs.csv")

rows = []

for f in files:
    exp_name = os.path.basename(os.path.dirname(f))
    
    if exp_name in valid_exps:
        df = pd.read_csv(f)
        df["exp_name"] = exp_name
        rows.append(df)

if len(rows) == 0:
    print("No matching experiments found!")
else:
    df_all = pd.concat(rows)
    df_all = df_all.sort_values("val_rmse")

    print("\n=== Transformer Tuning Results ===\n")
    print(df_all[["exp_name", "val_rmse", "train_minutes"]].to_string(index=False))