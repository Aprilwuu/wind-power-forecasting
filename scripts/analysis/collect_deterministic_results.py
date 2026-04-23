import json
import re
from pathlib import Path
import pandas as pd


# =========================
# 1. Root Path
# =========================
ROOT = Path(__file__).resolve().parents[2] / "data" / "featured"

# scan deterministic menu
MODEL_DIRS = [
    "tcn_track1_det",
    "lgbm_track1_det_lb168",
    "transformer_det_track1_lb168",
    "tcn_track2_det",
    "lgbm_track2_det_lb168",
    "transformer_det_track2_lb168",
]

OUTPUT_DIR = ROOT / "summary_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Tool Functions
# =========================
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_model_name(folder_name: str) -> str:
    s = folder_name.lower()
    if "lgbm" in s:
        return "LightGBM"
    if "tcn" in s:
        return "TCN"
    if "transformer" in s:
        return "Transformer"
    return folder_name


def infer_track(folder_name: str, data: dict) -> str:
    track = str(data.get("track", "")).lower()
    if "track1" in track or "temporal" in track:
        return "Track 1"
    if "track2" in track or "lofo" in track:
        return "Track 2"

    s = folder_name.lower()
    if "track1" in s:
        return "Track 1"
    if "track2" in s:
        return "Track 2"
    return "Unknown"


def extract_seed(path: Path):
    for p in path.parts:
        m = re.match(r"seed_(\d+)", p.lower())
        if m:
            return int(m.group(1))
    return None


def extract_heldout(path: Path, data: dict):
    split = data.get("split", {})
    if "held_out_group" in split and split["held_out_group"] is not None:
        return int(split["held_out_group"])

    for p in path.parts:
        m = re.match(r"heldout_(\d+)", p.lower())
        if m:
            return int(m.group(1))
    return None


def get_section(data: dict, primary: str, fallback: str = None):
    if primary in data and isinstance(data[primary], dict):
        return data[primary]
    if fallback and fallback in data and isinstance(data[fallback], dict):
        return data[fallback]
    return {}


def get_metric(section: dict, key: str):
    return section.get(key, None)


def get_n(section: dict):
    if "n" in section:
        return section["n"]
    if "n_seq" in section:
        return section["n_seq"]
    return None


# =========================
# 3. scan and read all metrics.json
# =========================
rows = []
missing_files = []

for model_dir_name in MODEL_DIRS:
    model_root = ROOT / model_dir_name
    if not model_root.exists():
        print(f"[Warning] Folder not found: {model_root}")
        continue

    # find metrics.json
    metric_files = list(model_root.rglob("metrics.json"))

    if not metric_files:
        print(f"[Warning] No metrics.json found under: {model_root}")
        continue

    for metrics_path in metric_files:
        try:
            data = load_json(metrics_path)

            model = infer_model_name(model_dir_name)
            track = infer_track(model_dir_name, data)
            seed = extract_seed(metrics_path)
            heldout = extract_heldout(metrics_path, data)

            # Track 1 common: val / test
            # Track 2 common: inner_val / outer_test
            val_sec = get_section(data, "inner_val", "val")
            test_sec = get_section(data, "outer_test", "test")

            row = {
                "track": track,
                "model": model,
                "model_dir": model_dir_name,
                "heldout_group": heldout,
                "seed": seed,

                "rmse_val": get_metric(val_sec, "rmse"),
                "mae_val": get_metric(val_sec, "mae"),
                "r2_val": get_metric(val_sec, "r2"),
                "n_val": get_n(val_sec),

                "rmse_test": get_metric(test_sec, "rmse"),
                "mae_test": get_metric(test_sec, "mae"),
                "r2_test": get_metric(test_sec, "r2"),
                "n_test": get_n(test_sec),

                "source_file": str(metrics_path),
            }
            rows.append(row)

        except Exception as e:
            print(f"[Error] Failed to parse {metrics_path}: {e}")
            missing_files.append((str(metrics_path), str(e)))


df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No valid metrics.json files were parsed. Please check your folders.")


# =========================
# 4. Ordering
# =========================
track_order = {"Track 1": 1, "Track 2": 2, "Unknown": 99}
model_order = {"LightGBM": 1, "TCN": 2, "Transformer": 3}

df["track_order"] = df["track"].map(track_order).fillna(99)
df["model_order"] = df["model"].map(model_order).fillna(99)

df = df.sort_values(
    by=["track_order", "model_order", "heldout_group", "seed"],
    na_position="last"
).reset_index(drop=True)

df = df.drop(columns=["track_order", "model_order"])


# =========================
# 5. output table
# =========================
all_runs_csv = OUTPUT_DIR / "deterministic_all_runs.csv"
df.to_csv(all_runs_csv, index=False, encoding="utf-8-sig")


# =========================
# 6. Generate tables (Mean ± std)
# =========================
# Track 1: aggregate by seeds
track1_df = df[df["track"] == "Track 1"].copy()

track1_summary = (
    track1_df.groupby(["track", "model"], dropna=False)
    .agg(
        rmse_mean=("rmse_test", "mean"),
        rmse_std=("rmse_test", "std"),
        mae_mean=("mae_test", "mean"),
        mae_std=("mae_test", "std"),
        r2_mean=("r2_test", "mean"),
        r2_std=("r2_test", "std"),
        n_runs=("seed", "count"),
    )
    .reset_index()
)

# Track 2: aggregate by model（10个heldout × 3 seeds）
track2_df = df[df["track"] == "Track 2"].copy()

track2_summary = (
    track2_df.groupby(["track", "model"], dropna=False)
    .agg(
        rmse_mean=("rmse_test", "mean"),
        rmse_std=("rmse_test", "std"),
        mae_mean=("mae_test", "mean"),
        mae_std=("mae_test", "std"),
        r2_mean=("r2_test", "mean"),
        r2_std=("r2_test", "std"),
        n_runs=("seed", "count"),
    )
    .reset_index()
)

summary_df = pd.concat([track1_summary, track2_summary], ignore_index=True)

summary_csv = OUTPUT_DIR / "deterministic_summary_mean_std.csv"
summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")


# =========================
# 7. Track 2 summary table for heldout
# =========================
track2_heldout_summary = (
    track2_df.groupby(["model", "heldout_group"], dropna=False)
    .agg(
        rmse_mean=("rmse_test", "mean"),
        rmse_std=("rmse_test", "std"),
        mae_mean=("mae_test", "mean"),
        mae_std=("mae_test", "std"),
        r2_mean=("r2_test", "mean"),
        r2_std=("r2_test", "std"),
        n_runs=("seed", "count"),
    )
    .reset_index()
    .sort_values(["model", "heldout_group"])
)

track2_heldout_csv = OUTPUT_DIR / "deterministic_track2_by_heldout.csv"
track2_heldout_summary.to_csv(track2_heldout_csv, index=False, encoding="utf-8-sig")


# =========================
# 8. Generate “mean ± std” Table
# =========================
def mean_std_str(mean_val, std_val, digits=4):
    if pd.isna(mean_val):
        return ""
    if pd.isna(std_val):
        return f"{mean_val:.{digits}f}"
    return f"{mean_val:.{digits}f} ± {std_val:.{digits}f}"


paper_summary = summary_df.copy()
paper_summary["RMSE"] = paper_summary.apply(lambda x: mean_std_str(x["rmse_mean"], x["rmse_std"]), axis=1)
paper_summary["MAE"] = paper_summary.apply(lambda x: mean_std_str(x["mae_mean"], x["mae_std"]), axis=1)
paper_summary["R2"] = paper_summary.apply(lambda x: mean_std_str(x["r2_mean"], x["r2_std"]), axis=1)

paper_summary = paper_summary[["track", "model", "RMSE", "MAE", "R2", "n_runs"]]
paper_summary_csv = OUTPUT_DIR / "deterministic_summary_for_paper.csv"
paper_summary.to_csv(paper_summary_csv, index=False, encoding="utf-8-sig")


# =========================
# 9. Checking for missing information
# =========================
expected_track1 = 3   # seeds 42,43,44
expected_track2 = 30  # heldout 1-10 × seeds 42,43,44

check_rows = []
for model in ["LightGBM", "TCN", "Transformer"]:
    n1 = len(track1_df[track1_df["model"] == model])
    n2 = len(track2_df[track2_df["model"] == model])

    check_rows.append({
        "model": model,
        "track1_found": n1,
        "track1_expected": expected_track1,
        "track1_complete": n1 == expected_track1,
        "track2_found": n2,
        "track2_expected": expected_track2,
        "track2_complete": n2 == expected_track2,
    })

check_df = pd.DataFrame(check_rows)
check_csv = OUTPUT_DIR / "deterministic_completeness_check.csv"
check_df.to_csv(check_csv, index=False, encoding="utf-8-sig")


# =========================
# 10. Print results
# =========================
print("\nSaved files:")
print(all_runs_csv)
print(summary_csv)
print(track2_heldout_csv)
print(paper_summary_csv)
print(check_csv)

if missing_files:
    err_df = pd.DataFrame(missing_files, columns=["file", "error"])
    err_csv = OUTPUT_DIR / "deterministic_parse_errors.csv"
    err_df.to_csv(err_csv, index=False, encoding="utf-8-sig")
    print(err_csv)

print("\nCompleteness check:")
print(check_df)

print("\nPaper summary preview:")
print(paper_summary)