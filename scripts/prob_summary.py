import json
import re
from pathlib import Path
import pandas as pd


ROOT = Path(r"E:\Projects\wind-power-forecasting\data\featured")

MODEL_DIRS = [
    "transformer_qr_track1_lb168",
    "transformer_qr_track2_lb168",
    "beta_transformer_track1_lb168",
    "beta_transformer_track2_lb168",
    "lgbm_qr_track1_lb168",
    "lgbm_qr_track2_lb168",
    "tcn_track1_mc500",
    "tcn_track2_mc500",
]

OUTPUT_DIR = ROOT / "summary_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_model_name(folder_name: str) -> str:
    s = folder_name.lower()
    if "lgbm" in s and "qr" in s:
        return "LightGBM-QR"
    if "transformer" in s and "beta" in s:
        return "Beta-Transformer"
    if "transformer" in s and "qr" in s:
        return "Transformer-QR"
    if "tcn" in s and "mc" in s:
        return "TCN-MC"
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
    held = split.get("held_out_group", None)
    if held is not None:
        try:
            return int(held)
        except Exception:
            return held

    for p in path.parts:
        m = re.match(r"heldout_(\d+)", p.lower())
        if m:
            return int(m.group(1))
    return None


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def first_existing_file(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def fmt_mean_std(mean_val, std_val, digits=4):
    if pd.isna(mean_val):
        return ""
    if pd.isna(std_val):
        return f"{mean_val:.{digits}f}"
    return f"{mean_val:.{digits}f} ± {std_val:.{digits}f}"


def parse_raw_metrics(folder_name: str, metrics: dict):
    model_name = infer_model_name(folder_name)
    out = {
        "pinball_raw": None,
        "picp_raw": None,
        "mpiw_raw": None,
        "rmse_med": None,
        "mae_med": None,
        "r2_med": None,
        "n_test": None,
    }

    if model_name == "Transformer-QR":
        out["pinball_raw"] = safe_get(metrics, "test_prob", "pinball")
        out["picp_raw"] = safe_get(metrics, "test_prob", "picp")
        out["mpiw_raw"] = safe_get(metrics, "test_prob", "mpiw")
        out["rmse_med"] = safe_get(metrics, "test", "rmse")
        out["mae_med"] = safe_get(metrics, "test", "mae")
        out["r2_med"] = safe_get(metrics, "test", "r2")
        out["n_test"] = safe_get(metrics, "test", "n_seq") or safe_get(metrics, "outer_test", "n")
        return out

    if model_name == "LightGBM-QR":
        track = str(metrics.get("track", "")).lower()

        if "track1" in track:
            out["pinball_raw"] = safe_get(metrics, "test", "pinball")
            out["picp_raw"] = safe_get(metrics, "coverage", "default", "test", "coverage")
            out["mpiw_raw"] = safe_get(metrics, "coverage", "default", "test", "avg_width")
            out["rmse_med"] = safe_get(metrics, "test", "median", "rmse")
            out["mae_med"] = safe_get(metrics, "test", "median", "mae")
            out["r2_med"] = safe_get(metrics, "test", "median", "r2")
            out["n_test"] = safe_get(metrics, "test", "n")
        else:
            out["pinball_raw"] = safe_get(metrics, "outer_test", "pinball")
            out["picp_raw"] = safe_get(metrics, "coverage", "default", "outer_test", "coverage")
            out["mpiw_raw"] = safe_get(metrics, "coverage", "default", "outer_test", "avg_width")
            out["rmse_med"] = safe_get(metrics, "outer_test", "median", "rmse")
            out["mae_med"] = safe_get(metrics, "outer_test", "median", "mae")
            out["r2_med"] = safe_get(metrics, "outer_test", "median", "r2")
            out["n_test"] = safe_get(metrics, "outer_test", "n")
        return out

    if model_name == "Beta-Transformer":
        test_sec = metrics.get("test", {})
        if not test_sec:
            test_sec = metrics.get("outer_test", {})
        out["pinball_raw"] = safe_get(metrics, "test_prob", "pinball") or safe_get(metrics, "outer_test", "pinball")
        out["picp_raw"] = test_sec.get("picp")
        out["mpiw_raw"] = test_sec.get("mpiw")
        out["rmse_med"] = test_sec.get("rmse")
        out["mae_med"] = test_sec.get("mae")
        out["r2_med"] = test_sec.get("r2")
        out["n_test"] = test_sec.get("n_seq") or test_sec.get("n")
        return out

    if model_name == "TCN-MC":
        out["pinball_raw"] = (
            safe_get(metrics, "test_prob", "summary", "pinball")
            or safe_get(metrics, "outer_test_prob", "summary", "pinball")
        )
        out["picp_raw"] = (
            safe_get(metrics, "test_prob", "summary", "picp")
            or safe_get(metrics, "outer_test_prob", "summary", "picp")
        )
        out["mpiw_raw"] = (
            safe_get(metrics, "test_prob", "summary", "mpiw")
            or safe_get(metrics, "outer_test_prob", "summary", "mpiw")
        )
        out["rmse_med"] = safe_get(metrics, "test", "rmse") or safe_get(metrics, "outer_test", "rmse")
        out["mae_med"] = safe_get(metrics, "test", "mae") or safe_get(metrics, "outer_test", "mae")
        out["r2_med"] = safe_get(metrics, "test", "r2") or safe_get(metrics, "outer_test", "r2")
        out["n_test"] = safe_get(metrics, "test", "n_seq") or safe_get(metrics, "outer_test", "n_seq")
        return out

    return out


def parse_conformal_metrics(run_dir: Path, folder_name: str):
    out = {
        "t_conformal": None,
        "n_cal": None,
        "q_level_used": None,
        "picp_cal": None,
        "mpiw_cal": None,
        "picp_raw_from_conformal": None,
        "mpiw_raw_from_conformal": None,
        "conformal_file": None,
    }

    model_name = infer_model_name(folder_name)
    post_dir = run_dir / "postprocess"

    if model_name == "Transformer-QR":
        cf = first_existing_file([
            post_dir / "conformal_report.json",
            post_dir / "conformal_summary.json",
        ])
        if cf:
            d = load_json(cf)
            out["conformal_file"] = str(cf)
            out["t_conformal"] = safe_get(d, "conformal", "t") or d.get("t")
            out["n_cal"] = safe_get(d, "conformal", "n_cal") or d.get("n_cal")
            out["q_level_used"] = safe_get(d, "conformal", "q_level_used") or d.get("q_level_used")
            out["picp_raw_from_conformal"] = (
                safe_get(d, "metrics", "apply", "raw_picp")
                or safe_get(d, "metrics", "raw", "test_picp")
            )
            out["mpiw_raw_from_conformal"] = (
                safe_get(d, "metrics", "apply", "raw_mpiw")
                or safe_get(d, "metrics", "raw", "test_mpiw")
            )
            out["picp_cal"] = (
                safe_get(d, "metrics", "apply", "cal_picp")
                or safe_get(d, "metrics", "cal", "test_picp")
            )
            out["mpiw_cal"] = (
                safe_get(d, "metrics", "apply", "cal_mpiw")
                or safe_get(d, "metrics", "cal", "test_mpiw")
            )
        return out

    if model_name == "LightGBM-QR":
        candidates = list(post_dir.glob("*conformal*.json"))
        cf = candidates[0] if candidates else None
        if cf:
            d = load_json(cf)
            out["conformal_file"] = str(cf)
            out["t_conformal"] = safe_get(d, "conformal", "t") or d.get("t")
            out["n_cal"] = safe_get(d, "conformal", "n_cal") or d.get("n_cal")
            out["q_level_used"] = safe_get(d, "conformal", "q_level_used") or d.get("q_level_used")
            out["picp_raw_from_conformal"] = safe_get(d, "metrics", "apply", "raw_picp")
            out["mpiw_raw_from_conformal"] = safe_get(d, "metrics", "apply", "raw_mpiw")
            out["picp_cal"] = safe_get(d, "metrics", "apply", "cal_picp")
            out["mpiw_cal"] = safe_get(d, "metrics", "apply", "cal_mpiw")
        return out

    if model_name == "Beta-Transformer":
        cf = first_existing_file([
            run_dir / "conformal_summary.json",
            post_dir / "conformal_summary.json",
        ])
        if cf:
            d = load_json(cf)
            out["conformal_file"] = str(cf)
            out["t_conformal"] = d.get("t")
            out["n_cal"] = d.get("n_cal")
            out["q_level_used"] = d.get("q_level_used")
            out["picp_raw_from_conformal"] = safe_get(d, "metrics", "raw", "test_picp")
            out["mpiw_raw_from_conformal"] = safe_get(d, "metrics", "raw", "test_mpiw")
            out["picp_cal"] = safe_get(d, "metrics", "cal", "test_picp")
            out["mpiw_cal"] = safe_get(d, "metrics", "cal", "test_mpiw")
        return out

    if model_name == "TCN-MC":
        candidates = [
            post_dir / "preds_test_mc__conformal_0.90.json",
            post_dir / "preds_outer_test_mc__conformal_0.90.json",
            post_dir / "conformal_report.json",
            post_dir / "conformal_summary.json",
        ]
        cf = first_existing_file(candidates)

        if cf:
            d = load_json(cf)
            out["conformal_file"] = str(cf)
            out["t_conformal"] = safe_get(d, "conformal", "t") or d.get("t")
            out["n_cal"] = safe_get(d, "conformal", "n_cal") or d.get("n_cal")
            out["q_level_used"] = safe_get(d, "conformal", "q_level_used") or d.get("q_level_used")
            out["picp_raw_from_conformal"] = (
                safe_get(d, "metrics", "apply", "raw_picp")
                or safe_get(d, "metrics", "raw", "test_picp")
                or safe_get(d, "metrics", "raw", "outer_test_picp")
            )
            out["mpiw_raw_from_conformal"] = (
                safe_get(d, "metrics", "apply", "raw_mpiw")
                or safe_get(d, "metrics", "raw", "test_mpiw")
                or safe_get(d, "metrics", "raw", "outer_test_mpiw")
            )
            out["picp_cal"] = (
                safe_get(d, "metrics", "apply", "cal_picp")
                or safe_get(d, "metrics", "cal", "test_picp")
                or safe_get(d, "metrics", "cal", "outer_test_picp")
            )
            out["mpiw_cal"] = (
                safe_get(d, "metrics", "apply", "cal_mpiw")
                or safe_get(d, "metrics", "cal", "test_mpiw")
                or safe_get(d, "metrics", "cal", "outer_test_mpiw")
            )
        return out

    return out


rows = []
parse_errors = []

for model_dir_name in MODEL_DIRS:
    model_root = ROOT / model_dir_name
    if not model_root.exists():
        print(f"[Warning] Folder not found: {model_root}")
        continue

    metric_files = list(model_root.rglob("metrics.json"))
    if not metric_files:
        print(f"[Warning] No metrics.json found under: {model_root}")
        continue

    for metrics_path in metric_files:
        try:
            metrics = load_json(metrics_path)
            run_dir = metrics_path.parent

            model = infer_model_name(model_dir_name)
            track = infer_track(model_dir_name, metrics)
            seed = extract_seed(metrics_path)
            heldout = extract_heldout(metrics_path, metrics)

            raw = parse_raw_metrics(model_dir_name, metrics)
            cal = parse_conformal_metrics(run_dir, model_dir_name)

            picp_raw_final = cal["picp_raw_from_conformal"] if cal["picp_raw_from_conformal"] is not None else raw["picp_raw"]
            mpiw_raw_final = cal["mpiw_raw_from_conformal"] if cal["mpiw_raw_from_conformal"] is not None else raw["mpiw_raw"]

            row = {
                "track": track,
                "model": model,
                "model_dir": model_dir_name,
                "heldout_group": heldout,
                "seed": seed,

                "pinball_raw": raw["pinball_raw"],
                "picp_raw": picp_raw_final,
                "mpiw_raw": mpiw_raw_final,
                "picp_cal": cal["picp_cal"],
                "mpiw_cal": cal["mpiw_cal"],

                "delta_picp": (
                    cal["picp_cal"] - picp_raw_final
                    if cal["picp_cal"] is not None and picp_raw_final is not None else None
                ),
                "delta_mpiw": (
                    cal["mpiw_cal"] - mpiw_raw_final
                    if cal["mpiw_cal"] is not None and mpiw_raw_final is not None else None
                ),

                "rmse_med": raw["rmse_med"],
                "mae_med": raw["mae_med"],
                "r2_med": raw["r2_med"],

                "t_conformal": cal["t_conformal"],
                "n_cal": cal["n_cal"],
                "q_level_used": cal["q_level_used"],
                "n_test": raw["n_test"],

                "metrics_file": str(metrics_path),
                "conformal_file": cal["conformal_file"],
            }
            rows.append(row)

        except Exception as e:
            parse_errors.append({"file": str(metrics_path), "error": str(e)})

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No probabilistic results were parsed. Check the folder names and file structure.")

track_order = {"Track 1": 1, "Track 2": 2, "Unknown": 99}
model_order = {
    "LightGBM-QR": 1,
    "TCN-MC": 2,
    "Transformer-QR": 3,
    "Beta-Transformer": 4,
}

df["track_order"] = df["track"].map(track_order).fillna(99)
df["model_order"] = df["model"].map(model_order).fillna(99)

df = df.sort_values(
    by=["track_order", "model_order", "heldout_group", "seed"],
    na_position="last"
).reset_index(drop=True)

df = df.drop(columns=["track_order", "model_order"])

all_runs_csv = OUTPUT_DIR / "probabilistic_all_runs.csv"
df.to_csv(all_runs_csv, index=False, encoding="utf-8-sig")

track1_df = df[df["track"] == "Track 1"].copy()
track2_df = df[df["track"] == "Track 2"].copy()

track1_summary = (
    track1_df.groupby(["track", "model"], dropna=False)
    .agg(
        pinball_mean=("pinball_raw", "mean"),
        pinball_std=("pinball_raw", "std"),
        picp_raw_mean=("picp_raw", "mean"),
        picp_raw_std=("picp_raw", "std"),
        mpiw_raw_mean=("mpiw_raw", "mean"),
        mpiw_raw_std=("mpiw_raw", "std"),
        picp_cal_mean=("picp_cal", "mean"),
        picp_cal_std=("picp_cal", "std"),
        mpiw_cal_mean=("mpiw_cal", "mean"),
        mpiw_cal_std=("mpiw_cal", "std"),
        delta_picp_mean=("delta_picp", "mean"),
        delta_mpiw_mean=("delta_mpiw", "mean"),
        n_runs=("seed", "count"),
    )
    .reset_index()
)

track2_summary = (
    track2_df.groupby(["track", "model"], dropna=False)
    .agg(
        pinball_mean=("pinball_raw", "mean"),
        pinball_std=("pinball_raw", "std"),
        picp_raw_mean=("picp_raw", "mean"),
        picp_raw_std=("picp_raw", "std"),
        mpiw_raw_mean=("mpiw_raw", "mean"),
        mpiw_raw_std=("mpiw_raw", "std"),
        picp_cal_mean=("picp_cal", "mean"),
        picp_cal_std=("picp_cal", "std"),
        mpiw_cal_mean=("mpiw_cal", "mean"),
        mpiw_cal_std=("mpiw_cal", "std"),
        delta_picp_mean=("delta_picp", "mean"),
        delta_mpiw_mean=("delta_mpiw", "mean"),
        n_runs=("seed", "count"),
    )
    .reset_index()
)

summary_df = pd.concat([track1_summary, track2_summary], ignore_index=True)
summary_csv = OUTPUT_DIR / "probabilistic_summary_mean_std.csv"
summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

track2_heldout_summary = (
    track2_df.groupby(["model", "heldout_group"], dropna=False)
    .agg(
        pinball_mean=("pinball_raw", "mean"),
        pinball_std=("pinball_raw", "std"),
        picp_raw_mean=("picp_raw", "mean"),
        picp_raw_std=("picp_raw", "std"),
        mpiw_raw_mean=("mpiw_raw", "mean"),
        mpiw_raw_std=("mpiw_raw", "std"),
        picp_cal_mean=("picp_cal", "mean"),
        picp_cal_std=("picp_cal", "std"),
        mpiw_cal_mean=("mpiw_cal", "mean"),
        mpiw_cal_std=("mpiw_cal", "std"),
        n_runs=("seed", "count"),
    )
    .reset_index()
    .sort_values(["model", "heldout_group"])
)

track2_heldout_csv = OUTPUT_DIR / "probabilistic_track2_by_heldout.csv"
track2_heldout_summary.to_csv(track2_heldout_csv, index=False, encoding="utf-8-sig")

paper_summary = summary_df.copy()
paper_summary["Pinball"] = paper_summary.apply(lambda x: fmt_mean_std(x["pinball_mean"], x["pinball_std"]), axis=1)
paper_summary["PICP_raw"] = paper_summary.apply(lambda x: fmt_mean_std(x["picp_raw_mean"], x["picp_raw_std"]), axis=1)
paper_summary["MPIW_raw"] = paper_summary.apply(lambda x: fmt_mean_std(x["mpiw_raw_mean"], x["mpiw_raw_std"]), axis=1)
paper_summary["PICP_cal"] = paper_summary.apply(lambda x: fmt_mean_std(x["picp_cal_mean"], x["picp_cal_std"]), axis=1)
paper_summary["MPIW_cal"] = paper_summary.apply(lambda x: fmt_mean_std(x["mpiw_cal_mean"], x["mpiw_cal_std"]), axis=1)
paper_summary["Delta_PICP"] = paper_summary["delta_picp_mean"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
paper_summary["Delta_MPIW"] = paper_summary["delta_mpiw_mean"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")

paper_summary = paper_summary[
    ["track", "model", "Pinball", "PICP_raw", "MPIW_raw", "PICP_cal", "MPIW_cal", "Delta_PICP", "Delta_MPIW", "n_runs"]
]

paper_summary_csv = OUTPUT_DIR / "probabilistic_summary_for_paper.csv"
paper_summary.to_csv(paper_summary_csv, index=False, encoding="utf-8-sig")

expected_track1 = 3
expected_track2 = 30

model_list = sorted(df["model"].dropna().unique().tolist())

check_rows = []
for model in model_list:
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
check_csv = OUTPUT_DIR / "probabilistic_completeness_check.csv"
check_df.to_csv(check_csv, index=False, encoding="utf-8-sig")

if parse_errors:
    err_df = pd.DataFrame(parse_errors)
    err_csv = OUTPUT_DIR / "probabilistic_parse_errors.csv"
    err_df.to_csv(err_csv, index=False, encoding="utf-8-sig")
    print("Parse errors saved to:", err_csv)

print("\nSaved files:")
print(all_runs_csv)
print(summary_csv)
print(track2_heldout_csv)
print(paper_summary_csv)
print(check_csv)

print("\nCompleteness check:")
print(check_df)

print("\nPaper summary preview:")
print(paper_summary)