import pandas as pd
import glob
import os


valid_exps = [
    "tcn_lr_1e4",
    "tcn_lr_3e4",
    "tcn_lr_1e3",
    "tcn_ch_32",
    "tcn_ch_mid",
    "tcn_ch_64"
]
files = glob.glob("reports/experiments/*/runs.csv")

rows = []

for f in files:
    exp_name = os.path.basename(os.path.dirname(f))
    
    if exp_name in valid_exps:
        df = pd.read_csv(f)
        df["exp_name"] = exp_name
        rows.append(df)

df_all = pd.concat(rows)

print(df_all[["exp_name", "val_rmse", "train_minutes"]].sort_values("val_rmse"))