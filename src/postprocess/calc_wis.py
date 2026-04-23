import re
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------
# Core metrics
# ----------------------------
def _to_1d(a):
    return np.asarray(a, dtype=float).reshape(-1)

def picp(y, lo, hi):
    y, lo, hi = _to_1d(y), _to_1d(lo), _to_1d(hi)
    return float(np.mean((y >= lo) & (y <= hi)))

def mpiw(lo, hi):
    lo, hi = _to_1d(lo), _to_1d(hi)
    return float(np.mean(hi - lo))

def wis_from_picp_mpiw(picp_val, mpiw_val, *, alpha=0.10, lam=0.5):
    nominal = 1.0 - float(alpha)
    return float(lam * abs(float(picp_val) - nominal) + (1.0 - lam) * float(mpiw_val))

def eval_intervals(y, lo, hi, *, alpha=0.10, lam=0.5):
    p = picp(y, lo, hi)
    w = mpiw(lo, hi)
    s = wis_from_picp_mpiw(p, w, alpha=alpha, lam=lam)
    return {"PICP": p, "MPIW": w, "WIS": s}

# ----------------------------
# Helpers
# ----------------------------
def load_npz(path: Path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}

def parse_seed(p: Path):
    s = str(p).replace("\\", "/")
    m = re.search(r"seed_(\d+)", s)
    return int(m.group(1)) if m else None

def parse_heldout(p: Path):
    s = str(p).replace("\\", "/")
    m = re.search(r"heldout_(\d+)", s)
    return int(m.group(1)) if m else None

def flatten_cols(df):
    df.columns = [f"{a}_{b}" if b else a for a, b in df.columns.to_flat_index()]
    return df

def eval_raw_cal_files(raw_path: Path, cal_path: Path | None,
                       *, alpha=0.10, lam=0.5,
                       y_key="y_true", lo="q05", hi="q95"):
    raw = load_npz(raw_path)
    if y_key not in raw or lo not in raw or hi not in raw:
        raise KeyError(f"{raw_path}: need keys {y_key},{lo},{hi}. keys={list(raw.keys())}")
    y = raw[y_key]
    out = {"raw": eval_intervals(y, raw[lo], raw[hi], alpha=alpha, lam=lam)}

    if cal_path is not None and cal_path.exists():
        cal = load_npz(cal_path)
        lo_cal = "q05_cal" if "q05_cal" in cal else lo
        hi_cal = "q95_cal" if "q95_cal" in cal else hi
        y_cal = cal.get(y_key, y)
        out["conformal"] = eval_intervals(y_cal, cal[lo_cal], cal[hi_cal], alpha=alpha, lam=lam)

    return out

# ----------------------------
# Track 1 / Track 2 summarizers
# ----------------------------
def summarize_track1(base_dir: str, raw_name: str, cal_name: str, *, alpha=0.10, lam=0.5):
    base = Path(base_dir)
    raw_paths = sorted(base.glob(f"**/seed_*/{raw_name}"))
    if not raw_paths:
        raise FileNotFoundError(f"Track1: no raw files under {base} matching **/seed_*/{raw_name}")

    rows = []
    for rp in raw_paths:
        seed = parse_seed(rp)
        cp = rp.parent / "postprocess" / cal_name
        metrics = eval_raw_cal_files(rp, cp, alpha=alpha, lam=lam)

        for variant, m in metrics.items():
            rows.append({
                "track": 1,
                "job": base.name,
                "heldout": None,
                "seed": seed,
                "variant": variant,
                "raw_file": str(rp),
                "cal_file": str(cp) if cp.exists() else None,
                **m
            })

    df = pd.DataFrame(rows).sort_values(["seed", "variant"]).reset_index(drop=True)
    overall = df.groupby(["variant"])[["PICP", "MPIW", "WIS"]].agg(["mean", "std"]).reset_index()
    overall = flatten_cols(overall)
    return df, overall

def summarize_track2(base_dir: str, raw_name: str, cal_name: str, *, alpha=0.10, lam=0.5):
    base = Path(base_dir)
    raw_paths = sorted(base.glob(f"**/heldout_*/seed_*/{raw_name}"))
    if not raw_paths:
        raise FileNotFoundError(f"Track2: no raw files under {base} matching **/heldout_*/seed_*/{raw_name}")

    rows = []
    for rp in raw_paths:
        seed = parse_seed(rp)
        heldout = parse_heldout(rp)
        cp = rp.parent / "postprocess" / cal_name
        metrics = eval_raw_cal_files(rp, cp, alpha=alpha, lam=lam)

        for variant, m in metrics.items():
            rows.append({
                "track": 2,
                "job": base.name,
                "heldout": heldout,
                "seed": seed,
                "variant": variant,
                "raw_file": str(rp),
                "cal_file": str(cp) if cp.exists() else None,
                **m
            })

    df = pd.DataFrame(rows).sort_values(["heldout", "seed", "variant"]).reset_index(drop=True)

    per_fold = (
        df.groupby(["heldout", "variant"])[["PICP", "MPIW", "WIS"]]
          .agg(["mean", "std"])
          .reset_index()
    )
    per_fold = flatten_cols(per_fold)

    overall = df.groupby(["variant"])[["PICP", "MPIW", "WIS"]].agg(["mean", "std"]).reset_index()
    overall = flatten_cols(overall)
    return df, per_fold, overall

# ----------------------------
# Batch jobs config (你只改这里就行)
# ----------------------------
JOBS = [
    # Track 1 example
    {
        "track": 1,
        "base_dir": r"E:\Projects\wind-power-forecasting\data\featured\tcn_track1_mc500\baseline",
        "raw_name": "preds_test_mc.npz",
        "cal_name": "preds_test_mc_noise_cal.npz",
        "tag": "tcn_mc_track1_baseline",
    },
       # Track 2 example
    {
        "track": 2,
        "base_dir": r"E:\Projects\wind-power-forecasting\data\featured\tcn_track2_mc500\baseline",
        "raw_name": "preds_outer_test_mc.npz",
        "cal_name": "preds_outer_test_mc_noise_cal.npz",
        "tag": "tcn_mc_track2_baseline",
    },
]

if __name__ == "__main__":
    alpha = 0.10
    lam = 0.5

    for job in JOBS:
        track = job["track"]
        base_dir = job["base_dir"]
        raw_name = job["raw_name"]
        cal_name = job["cal_name"]
        tag = job.get("tag", Path(base_dir).name)

        base = Path(base_dir)
        print(f"\n==== Running {tag} (Track {track}) ====")

        if track == 1:
            df, overall = summarize_track1(base_dir, raw_name, cal_name, alpha=alpha, lam=lam)

            out_df = base / f"wis_{tag}_by_seed.csv"
            out_overall = base / f"wis_{tag}_overall_mean_std.csv"
            df.to_csv(out_df, index=False)
            overall.to_csv(out_overall, index=False)

            print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print("\nOverall mean±std:")
            print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print(f"\nSaved:\n- {out_df}\n- {out_overall}")

        elif track == 2:
            df, per_fold, overall = summarize_track2(base_dir, raw_name, cal_name, alpha=alpha, lam=lam)

            out_df = base / f"wis_{tag}_by_heldout_seed.csv"
            out_fold = base / f"wis_{tag}_per_heldout_mean_std.csv"
            out_overall = base / f"wis_{tag}_overall_mean_std.csv"
            df.to_csv(out_df, index=False)
            per_fold.to_csv(out_fold, index=False)
            overall.to_csv(out_overall, index=False)

            print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print("\nPer-heldout mean±std:")
            print(per_fold.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print("\nOverall mean±std:")
            print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print(f"\nSaved:\n- {out_df}\n- {out_fold}\n- {out_overall}")
        else:
            raise ValueError(f"Unknown track: {track}")