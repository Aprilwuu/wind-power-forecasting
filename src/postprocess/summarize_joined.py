import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------
# Utils
# ----------------------------
def read_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def parse_seed(path: Path):
    s = str(path).replace("\\", "/")
    m = re.search(r"seed_(\d+)", s)
    return int(m.group(1)) if m else None

def parse_heldout(path: Path):
    s = str(path).replace("\\", "/")
    m = re.search(r"heldout_(\d+)", s)
    return int(m.group(1)) if m else None

def mean_std(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return (np.nan, np.nan)
    return (float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0)

def compute_wis(picp, mpiw, alpha=0.10, lam=0.5):
    nominal = 1.0 - alpha
    return lam * abs(float(picp) - nominal) + (1.0 - lam) * float(mpiw)

def flatten_cols(df):
    df.columns = [f"{a}_{b}" if b else a for a, b in df.columns.to_flat_index()]
    return df

# ----------------------------
# Load deterministic
# ----------------------------
def load_deterministic(base_dir: str, track: int, model: str, det_glob: str):
    base = Path(base_dir)

    if track == 1:
        paths = sorted(base.glob(f"**/seed_*/{det_glob}"))
    else:
        paths = sorted(base.glob(f"**/heldout_*/seed_*/{det_glob}"))

    rows = []
    for p in paths:
        d = read_json(p)

        # support multiple formats:
        #  A) metrics.json: test.rmse/mae/r2
        #  B) summary style: test.median.rmse/mae/r2
        #  C) your current format: outer_test.rmse/mae/r2
        #  D) optional: inner_val.rmse/mae/r2
        rmse = mae = r2 = np.nan

        if isinstance(d.get("test"), dict):
            test = d.get("test", {})
            if isinstance(test.get("median"), dict):
                median = test.get("median", {})
                rmse = median.get("rmse", np.nan)
                mae  = median.get("mae", np.nan)
                r2   = median.get("r2", np.nan)
            else:
                rmse = test.get("rmse", np.nan)
                mae  = test.get("mae", np.nan)
                r2   = test.get("r2", np.nan)

        elif isinstance(d.get("outer_test"), dict):
            outer = d["outer_test"]
            rmse = outer.get("rmse", np.nan)
            mae  = outer.get("mae", np.nan)
            r2   = outer.get("r2", np.nan)

        elif isinstance(d.get("inner_val"), dict):
            inner = d["inner_val"]
            rmse = inner.get("rmse", np.nan)
            mae  = inner.get("mae", np.nan)
            r2   = inner.get("r2", np.nan)

        if np.isnan(rmse) or np.isnan(mae) or np.isnan(r2):
            print(f"[WARN] cannot parse det metrics (rmse/mae/r2) from: {p}")

        rows.append({
            "model": model,
            "track": track,
            "seed": parse_seed(p),
            "heldout": parse_heldout(p) if track == 2 else None,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "det_path": str(p),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["model","track","seed","heldout","rmse","mae","r2","det_path"])
    return df

# ----------------------------
# Load interval (conformal report json)
# ----------------------------
def load_interval(base_dir: str, track: int, model: str, int_glob: str):
    base = Path(base_dir)

    if track == 1:
        paths = sorted(base.glob(f"**/seed_*/postprocess/{int_glob}"))
        paths += sorted(base.glob(f"**/seed_*/{int_glob}"))
    else:
        paths = sorted(base.glob(f"**/heldout_*/seed_*/postprocess/{int_glob}"))
        paths += sorted(base.glob(f"**/heldout_*/seed_*/{int_glob}"))

    # dedupe
    paths = sorted({str(p): p for p in paths}.values(), key=lambda x: str(x))

    def find_noise_blocks_one_level(obj: dict):
        """
        Find a dict that contains keys: old_raw and new_noise_augmented.
        We search:
          - top-level obj
          - one-level nested dicts (obj.values()) if they are dicts
        Return (old_raw_dict, new_noise_augmented_dict) or (None, None).
        """
        if not isinstance(obj, dict):
            return None, None

        candidates = [obj]
        for v in obj.values():
            if isinstance(v, dict):
                candidates.append(v)

        for c in candidates:
            if isinstance(c.get("old_raw"), dict) and isinstance(c.get("new_noise_augmented"), dict):
                return c["old_raw"], c["new_noise_augmented"]
        return None, None

    rows = []
    for p in paths:
        r = read_json(p)

        raw_picp = raw_mpiw = cal_picp = cal_mpiw = np.nan
        t = np.nan

        # ---- Case A: long report format (metrics.apply.*) ----
        if isinstance(r.get("metrics", {}).get("apply", None), dict) and r["metrics"]["apply"]:
            apply = r["metrics"]["apply"]
            t = r.get("conformal", {}).get("t", np.nan)

            raw_picp = apply.get("raw_picp", np.nan)
            raw_mpiw = apply.get("raw_mpiw", np.nan)
            cal_picp = apply.get("cal_picp", np.nan)
            cal_mpiw = apply.get("cal_mpiw", np.nan)

        # ---- Case B: conformal_summary format (metrics.raw/cal.test_*) ----
        elif isinstance(r.get("metrics", {}).get("raw", None), dict) and isinstance(r.get("metrics", {}).get("cal", None), dict):
            t = r.get("t", np.nan)

            raw_picp = r["metrics"]["raw"].get("test_picp", np.nan)
            raw_mpiw = r["metrics"]["raw"].get("test_mpiw", np.nan)
            cal_picp = r["metrics"]["cal"].get("test_picp", np.nan)
            cal_mpiw = r["metrics"]["cal"].get("test_mpiw", np.nan)

        # ---- Case C: TCN noise-augmentation report (key may not be named "metrics") ----
        # old_raw = true raw (before noise aug)
        # new_noise_augmented = treated as "cal" here (your setup has t=0)
        else:
            old_raw, new_noise = find_noise_blocks_one_level(r)
            if old_raw is None or new_noise is None:
                print(f"[SKIP] Unknown interval report format: {p}")
                continue

            # some files may have t at top-level; if absent, default 0.0
            t = float(r.get("t", 0.0))

            raw_picp = old_raw.get("picp", np.nan)
            raw_mpiw = old_raw.get("mpiw", np.nan)

            cal_picp = new_noise.get("picp", np.nan)
            cal_mpiw = new_noise.get("mpiw", np.nan)

        if any(np.isnan(x) for x in [raw_picp, raw_mpiw, cal_picp, cal_mpiw]):
            print(f"[SKIP] Missing interval numbers in: {p}")
            continue

        rows.append({
            "model": model,
            "track": track,
            "seed": parse_seed(p),
            "heldout": parse_heldout(p) if track == 2 else None,
            "raw_picp": raw_picp,
            "raw_mpiw": raw_mpiw,
            "cal_picp": cal_picp,
            "cal_mpiw": cal_mpiw,
            "t": t,
            "raw_wis": compute_wis(raw_picp, raw_mpiw),
            "cal_wis": compute_wis(cal_picp, cal_mpiw),
            "int_path": str(p),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["model","track","seed","heldout",
                                   "raw_picp","raw_mpiw","cal_picp","cal_mpiw","t",
                                   "raw_wis","cal_wis","int_path"])
    return df

# ----------------------------
# Summaries
# ----------------------------
def summarize_track1(df):
    out = {"n_seeds": df["seed"].nunique()}
    for k in ["rmse","mae","r2","raw_picp","raw_mpiw","raw_wis","cal_picp","cal_mpiw","cal_wis","t"]:
        m, s = mean_std(df[k])
        out[k+"_mean"] = m
        out[k+"_std"] = s
    return out

def summarize_track2(df):
    overall = {"n_runs": len(df), "n_heldout": df["heldout"].nunique(), "n_seeds": df["seed"].nunique()}
    for k in ["rmse","mae","r2","raw_picp","raw_mpiw","raw_wis","cal_picp","cal_mpiw","cal_wis","t"]:
        m, s = mean_std(df[k])
        overall[k+"_mean"] = m
        overall[k+"_std"] = s

    per_fold = (
        df.groupby(["heldout"])[["rmse","mae","r2","raw_picp","raw_mpiw","raw_wis","cal_picp","cal_mpiw","cal_wis","t"]]
          .agg(["mean","std"])
          .reset_index()
    )
    per_fold = flatten_cols(per_fold)
    return overall, per_fold

# ----------------------------
# CONFIGS
# ----------------------------
CONFIGS = [

    {
        "model": "TCN-MC-noise",
        "track": 1,
        "det_base": r"E:\Projects\wind-power-forecasting\data\featured\tcn_track1_det\baseline",
        "det_glob": "metrics.json",
        "int_base": r"E:\Projects\wind-power-forecasting\data\featured\tcn_track1_mc500\baseline",
        "int_glob": "preds_test_mc_noise_summary.json",
    },
    {
        "model": "TCN-MC-noise",
        "track": 2,
        "det_base": r"E:\Projects\wind-power-forecasting\data\featured\tcn_track2_det\baseline",
        "det_glob": "metrics.json",
        "int_base": r"E:\Projects\wind-power-forecasting\data\featured\tcn_track2_mc500\baseline",
        "int_glob": "preds_outer_test_mc_noise_summary.json",
    },

      {
        "model": "TCN-MC-cal",
        "track": 1,
        "det_base": r"E:\Projects\wind-power-forecasting\data\featured\tcn_track1_det\baseline",
        "det_glob": "metrics.json",
        "int_base": r"E:\Projects\wind-power-forecasting\data\featured\tcn_track1_mc500\baseline",
        "int_glob": "preds_test_mc__conformal_0.90.json",
    },
    {
        "model": "TCN-MC-cal",
        "track": 2,
        "det_base": r"E:\Projects\wind-power-forecasting\data\featured\tcn_track2_det\baseline",
        "det_glob": "metrics.json",
        "int_base": r"E:\Projects\wind-power-forecasting\data\featured\tcn_track2_mc500\baseline",
        "int_glob": "preds_outer_test_mc__conformal_0.90.json",
    },
]

def main():
    out_root = Path(r"E:\Projects\wind-power-forecasting\results_summaries_tcn")
    out_root.mkdir(parents=True, exist_ok=True)

    all_joined = []
    t1_rows = []
    t2_rows = []
    t2_fold_tables = []

    for cfg in CONFIGS:
        model = cfg["model"]
        track = int(cfg["track"])

        det = load_deterministic(cfg["det_base"], track, model, cfg["det_glob"])
        interval = load_interval(cfg["int_base"], track, model, cfg["int_glob"])

        if det.empty:
            print(f"[WARN] deterministic empty: {model} Track{track} | {cfg['det_base']} | glob={cfg['det_glob']}")
        if interval.empty:
            print(f"[WARN] interval empty: {model} Track{track} | {cfg['int_base']} | glob={cfg['int_glob']}")

        if det.empty and interval.empty:
            continue

        keys = ["model","track","seed"] + (["heldout"] if track == 2 else [])
        joined = pd.merge(det, interval, on=keys, how="outer")

        all_joined.append(joined)

        if track == 1:
            s = summarize_track1(joined)
            s.update({"model": model, "track": 1})
            t1_rows.append(s)
        else:
            overall, per_fold = summarize_track2(joined)
            overall.update({"model": model, "track": 2})
            t2_rows.append(overall)
            per_fold["model"] = model
            t2_fold_tables.append(per_fold)

    if all_joined:
        df_all = pd.concat(all_joined, ignore_index=True)
        df_all.to_csv(out_root / "joined_all_runs.csv", index=False)

    if t1_rows:
        pd.DataFrame(t1_rows).to_csv(out_root / "track1_mean_std_by_model.csv", index=False)

    if t2_rows:
        pd.DataFrame(t2_rows).to_csv(out_root / "track2_mean_std_by_model.csv", index=False)

    if t2_fold_tables:
        pd.concat(t2_fold_tables, ignore_index=True).to_csv(out_root / "track2_per_heldout_mean_std.csv", index=False)

    print(f"[OK] Saved to {out_root}")

if __name__ == "__main__":
    main()