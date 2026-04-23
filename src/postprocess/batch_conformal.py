import argparse
import json
import re
from pathlib import Path

from src.postprocess.conformal import load_npz, save_npz, apply_conformal_to_npz
import numpy as np

def _to_1d(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)

def picp_np(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = _to_1d(y); lo = _to_1d(lo); hi = _to_1d(hi)
    return float(np.mean((y >= lo) & (y <= hi)))

def mpiw_np(lo: np.ndarray, hi: np.ndarray) -> float:
    lo = _to_1d(lo); hi = _to_1d(hi)
    return float(np.mean(hi - lo))

def parse_seed(p: Path):
    m = re.search(r"seed_(\d+)", str(p).replace("\\", "/"))
    return int(m.group(1)) if m else None


def parse_heldout(p: Path):
    m = re.search(r"heldout_(\d+)", str(p).replace("\\", "/"))
    return int(m.group(1)) if m else None


def run_one(cal_path: Path, test_path: Path, out_dir: Path,
            *, target_picp=0.90, y_key="y_true", lo_key="q05", hi_key="q95",
            clip_min=0.0, clip_max=1.0, report_name: str = "conformal_report.json"):
    cal_npz = load_npz(cal_path)
    test_npz = load_npz(test_path)

    # ---- raw metrics (before conformal) ----
    y_cal = cal_npz[y_key]
    lo_cal_raw = cal_npz[lo_key]
    hi_cal_raw = cal_npz[hi_key]

    y_app = test_npz[y_key]
    lo_app_raw = test_npz[lo_key]
    hi_app_raw = test_npz[hi_key]

    raw_cal_picp = picp_np(y_cal, lo_cal_raw, hi_cal_raw)
    raw_cal_mpiw = mpiw_np(lo_cal_raw, hi_cal_raw)
    raw_app_picp = picp_np(y_app, lo_app_raw, hi_app_raw)
    raw_app_mpiw = mpiw_np(lo_app_raw, hi_app_raw)

    # ---- conformal apply (writes *_cal keys into dicts) ----
    cal_out, test_out, summary = apply_conformal_to_npz(
        cal_npz,
        test_npz,
        y_key=y_key,
        lo_key=lo_key,
        hi_key=hi_key,
        target_picp=target_picp,
        out_suffix="_cal",
        clip_min=clip_min,
        clip_max=clip_max,
    )

    # ---- calibrated metrics ----
    lo_cal_cal = cal_out[f"{lo_key}_cal"]
    hi_cal_cal = cal_out[f"{hi_key}_cal"]
    lo_app_cal = test_out[f"{lo_key}_cal"]
    hi_app_cal = test_out[f"{hi_key}_cal"]

    cal_cal_picp = picp_np(y_cal, lo_cal_cal, hi_cal_cal)
    cal_cal_mpiw = mpiw_np(lo_cal_cal, hi_cal_cal)
    cal_app_picp = picp_np(y_app, lo_app_cal, hi_app_cal)
    cal_app_mpiw = mpiw_np(lo_app_cal, hi_app_cal)

    # ---- save outputs ----
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz_apply = out_dir / (test_path.stem + "_cal.npz")
    save_npz(out_npz_apply, test_out)

    out_npz_cal = out_dir / (cal_path.stem + "_cal.npz")
    save_npz(out_npz_cal, cal_out)

    # ---- long report json (like your example) ----
    report = {
        "format": "npz",
        "cal": str(cal_path),
        "apply": str(test_path),
        "saved": {
            "cal": str(out_npz_cal),
            "apply": str(out_npz_apply),
        },
        "params": {
            "y": y_key,
            "lo": lo_key,
            "hi": hi_key,
            "suffix": "_cal",
            "target_picp": float(target_picp),
            "clip_min": float(clip_min),
            "clip_max": float(clip_max),
        },
        "conformal": {
            "t": float(summary.t),
            "n_cal": int(summary.n_cal),
            "q_level_used": float(summary.q_level_used),
        },
        "metrics": {
            "cal": {
                "raw_picp": float(raw_cal_picp),
                "raw_mpiw": float(raw_cal_mpiw),
                "cal_picp": float(cal_cal_picp),
                "cal_mpiw": float(cal_cal_mpiw),
            },
            "apply": {
                "raw_picp": float(raw_app_picp),
                "raw_mpiw": float(raw_app_mpiw),
                "cal_picp": float(cal_app_picp),
                "cal_mpiw": float(cal_app_mpiw),
            },
        },
    }

    out_report = out_dir / report_name
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return out_npz_apply, out_report, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="Baseline folder (contains seed_*/ or heldout_*/seed_*/)")
    ap.add_argument("--track", type=int, choices=[1, 2], required=True, help="1=Track1, 2=Track2")
    ap.add_argument("--cal_name", type=str, required=True, help="Calibration npz file name (e.g., preds_val_xxx.npz or preds_outer_val_xxx.npz)")
    ap.add_argument("--test_name", type=str, required=True, help="Test npz file name (e.g., preds_test_xxx.npz or preds_outer_test_xxx.npz)")
    ap.add_argument("--out_subdir", type=str, default="postprocess", help="Where to save calibrated outputs under each seed folder")
    ap.add_argument("--target_picp", type=float, default=0.90)
    ap.add_argument("--y_key", type=str, default="y_true")
    ap.add_argument("--lo_key", type=str, default="q05")
    ap.add_argument("--hi_key", type=str, default="q95")
    ap.add_argument("--clip_min", type=float, default=0.0)
    ap.add_argument("--clip_max", type=float, default=1.0)
    args = ap.parse_args()

    base = Path(args.base_dir)

    if args.track == 1:
        # .../seed_42/<cal_name> and .../seed_42/<test_name>
        test_paths = sorted(base.glob(f"**/seed_*/{args.test_name}"))
        if not test_paths:
            raise FileNotFoundError(f"No test files found: **/seed_*/{args.test_name} under {base}")

        n_ok = 0
        n_skip = 0
        for test_path in test_paths:
            seed = parse_seed(test_path)
            cal_path = test_path.parent / args.cal_name
            if not cal_path.exists():
                print(f"[SKIP] seed={seed}: missing cal file: {cal_path}")
                n_skip += 1
                continue

            out_dir = test_path.parent / args.out_subdir
            out_npz, out_summary, summary = run_one(
                cal_path, test_path, out_dir,
                target_picp=args.target_picp,
                y_key=args.y_key, lo_key=args.lo_key, hi_key=args.hi_key,
                clip_min=args.clip_min, clip_max=args.clip_max
            )
            print(f"[OK] Track1 seed={seed}  t={summary.t:.6f}  -> {out_npz}")
            n_ok += 1

        print(f"\nDone. OK={n_ok}, SKIP={n_skip}")

    else:
        # Track2: .../heldout_1/seed_42/<cal_name> and <test_name>
        test_paths = sorted(base.glob(f"**/heldout_*/seed_*/{args.test_name}"))
        if not test_paths:
            raise FileNotFoundError(f"No test files found: **/heldout_*/seed_*/{args.test_name} under {base}")

        n_ok = 0
        n_skip = 0
        for test_path in test_paths:
            seed = parse_seed(test_path)
            heldout = parse_heldout(test_path)
            cal_path = test_path.parent / args.cal_name
            if not cal_path.exists():
                print(f"[SKIP] heldout={heldout} seed={seed}: missing cal file: {cal_path}")
                n_skip += 1
                continue

            out_dir = test_path.parent / args.out_subdir
            out_npz, out_summary, summary = run_one(
                cal_path, test_path, out_dir,
                target_picp=args.target_picp,
                y_key=args.y_key, lo_key=args.lo_key, hi_key=args.hi_key,
                clip_min=args.clip_min, clip_max=args.clip_max
            )
            print(f"[OK] Track2 heldout={heldout} seed={seed}  t={summary.t:.6f}  -> {out_npz}")
            n_ok += 1

        print(f"\nDone. OK={n_ok}, SKIP={n_skip}")


if __name__ == "__main__":
    main()