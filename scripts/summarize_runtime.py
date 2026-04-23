from pathlib import Path
import pandas as pd

base_dir = Path(r"E:\Projects\wind-power-forecasting\data\featured\runtime_summary")

files = {
    "LightGBM-QR (Track1)": "estimated_runtime_per_run_lgbm_track1.csv",
    "LightGBM-QR (Track2)": "estimated_runtime_per_run_lgbm_track2.csv",
    "TCN-MC (Track1)": "estimated_runtime_per_run_tcn_track1.csv",
    "TCN-MC (Track2)": "estimated_runtime_per_run_tcn_track2.csv",
    "Transformer-QR (Track1)": "estimated_runtime_per_run_transformerqr_track1.csv",
    "Transformer-QR (Track2)": "estimated_runtime_per_run_transformerqr_track2.csv",
    "Beta-Transformer (Track1)": "estimated_runtime_per_run_betatransformer_track1.csv",
    "Beta-Transformer (Track2)": "estimated_runtime_per_run_betatransformer_track2.csv",
}

rows = []

for model, filename in files.items():

    file_path = base_dir / filename

    if not file_path.exists():
        print(f"Missing: {file_path}")
        continue

    df = pd.read_csv(file_path)

    df = df[df["duration_min"] > 0]

    rows.append({
        "Model": model,
        "Avg runtime (min)": round(df["duration_min"].mean(), 2),
        "Min (min)": round(df["duration_min"].min(), 2),
        "Max (min)": round(df["duration_min"].max(), 2),
        "Runs": len(df)
    })

summary = pd.DataFrame(rows)

print("\nRuntime summary:\n")
print(summary.to_string(index=False))

summary_path = base_dir / "runtime_summary_table.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

print(f"\nSaved: {summary_path}")

print("\nLaTeX table:\n")
print(summary.to_latex(index=False))