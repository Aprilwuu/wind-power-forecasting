# scripts/postprocess/run_conformal.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# import sys
# from pathlib import Path as _Path
# sys.path.append(str(_Path(__file__).resolve().parents[2]))

from src.postprocess.conformal import (
    load_npz,
    save_npz,
    apply_conformal_to_npz,
    apply_conformal_to_csv,
    picp,
    mpiw,
)


def is_npz(p: Path) -> bool:
    return p.suffix.lower() == ".npz"


def is_csv(p: Path) -> bool:
    return p.suffix.lower() == ".csv"


def main():
    ap = argparse.ArgumentParser(description="Generic split conformal widening for interval predictions (npz or csv).")
    ap.add_argument("--cal", type=str, required=True, help="Calibration file: npz or csv (e.g., inner_val preds)")
    ap.add_argument("--apply", type=str, required=True, help="Apply file: npz or csv (e.g., outer_test preds)")
    ap.add_argument("--target_picp", type=float, default=0.9, help="Target coverage (default 0.9)")
    ap.add_argument("--lo", type=str, default="q05", help="Lower interval key/column (default q05)")
    ap.add_argument("--hi", type=str, default="q95", help="Upper interval key/column (default q95)")
    ap.add_argument("--y", type=str, default="y_true", help="y_true key/column (default y_true)")

    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <apply_dir>/postprocess)")
    ap.add_argument("--suffix", type=str, default="_cal", help="Suffix for calibrated interval keys/cols (default _cal)")
    ap.add_argument("--clip_min", type=float, default=0.0, help="Clip min (default 0.0)")
    ap.add_argument("--clip_max", type=float, default=1.0, help="Clip max (default 1.0)")

    args = ap.parse_args()
    cal_path = Path(args.cal)
    apply_path = Path(args.apply)

    # smart default: avoid .../postprocess/postprocess
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = apply_path.parent if apply_path.parent.name.lower() == "postprocess" else (apply_path.parent / "postprocess")

    out_dir.mkdir(parents=True, exist_ok=True)


    summary_out = out_dir / f"{apply_path.stem}__conformal_{args.target_picp:.2f}.json"

    if is_npz(cal_path) and is_npz(apply_path):
        cal_npz = load_npz(cal_path)
        apply_npz = load_npz(apply_path)

        cal_out, apply_out, summ = apply_conformal_to_npz(
            cal_npz=cal_npz,
            apply_npz=apply_npz,
            y_key=args.y,
            lo_key=args.lo,
            hi_key=args.hi,
            target_picp=args.target_picp,
            out_suffix=args.suffix,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
        )

        cal_save = out_dir / f"{cal_path.stem}{args.suffix}.npz"
        apply_save = out_dir / f"{apply_path.stem}{args.suffix}.npz"
        save_npz(cal_save, cal_out)
        save_npz(apply_save, apply_out)

        # metrics (only if y exists in apply)
        metrics = {"cal": {}, "apply": {}}
        if args.y in cal_out:
            metrics["cal"]["raw_picp"] = picp(cal_out[args.y], cal_npz[args.lo], cal_npz[args.hi])
            metrics["cal"]["raw_mpiw"] = mpiw(cal_npz[args.lo], cal_npz[args.hi])
            metrics["cal"]["cal_picp"] = picp(cal_out[args.y], cal_out[f"{args.lo}{args.suffix}"], cal_out[f"{args.hi}{args.suffix}"])
            metrics["cal"]["cal_mpiw"] = mpiw(cal_out[f"{args.lo}{args.suffix}"], cal_out[f"{args.hi}{args.suffix}"])

        if args.y in apply_out:
            metrics["apply"]["raw_picp"] = picp(apply_out[args.y], apply_npz[args.lo], apply_npz[args.hi])
            metrics["apply"]["raw_mpiw"] = mpiw(apply_npz[args.lo], apply_npz[args.hi])
            metrics["apply"]["cal_picp"] = picp(apply_out[args.y], apply_out[f"{args.lo}{args.suffix}"], apply_out[f"{args.hi}{args.suffix}"])
            metrics["apply"]["cal_mpiw"] = mpiw(apply_out[f"{args.lo}{args.suffix}"], apply_out[f"{args.hi}{args.suffix}"])

        payload = {
            "format": "npz",
            "cal": str(cal_path),
            "apply": str(apply_path),
            "saved": {"cal": str(cal_save), "apply": str(apply_save)},
            "params": {
                "y": args.y, "lo": args.lo, "hi": args.hi,
                "suffix": args.suffix,
                "target_picp": args.target_picp,
                "clip_min": args.clip_min, "clip_max": args.clip_max,
            },
            "conformal": {
                "t": summ.t,
                "n_cal": summ.n_cal,
                "q_level_used": summ.q_level_used,
            },
            "metrics": metrics,
        }
        summary_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Saved:", cal_save)
        print("Saved:", apply_save)
        print("Summary:", summary_out)

    elif is_csv(cal_path) and is_csv(apply_path):
        cal_df = pd.read_csv(cal_path)
        apply_df = pd.read_csv(apply_path)

        cal_out, apply_out, summ = apply_conformal_to_csv(
            cal_df=cal_df,
            apply_df=apply_df,
            y_col=args.y,
            lo_col=args.lo,
            hi_col=args.hi,
            target_picp=args.target_picp,
            out_suffix=args.suffix,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
        )

        cal_save = out_dir / f"{cal_path.stem}{args.suffix}.csv"
        apply_save = out_dir / f"{apply_path.stem}{args.suffix}.csv"
        cal_out.to_csv(cal_save, index=False)
        apply_out.to_csv(apply_save, index=False)

        metrics = {"cal": {}, "apply": {}}
        if args.y in cal_out.columns:
            metrics["cal"]["raw_picp"] = float(((cal_df[args.y] >= cal_df[args.lo]) & (cal_df[args.y] <= cal_df[args.hi])).mean())
            metrics["cal"]["raw_mpiw"] = float((cal_df[args.hi] - cal_df[args.lo]).mean())
            metrics["cal"]["cal_picp"] = float(((cal_out[args.y] >= cal_out[f"{args.lo}{args.suffix}"]) & (cal_out[args.y] <= cal_out[f"{args.hi}{args.suffix}"])).mean())
            metrics["cal"]["cal_mpiw"] = float((cal_out[f"{args.hi}{args.suffix}"] - cal_out[f"{args.lo}{args.suffix}"]).mean())

        if args.y in apply_out.columns:
            metrics["apply"]["raw_picp"] = float(((apply_df[args.y] >= apply_df[args.lo]) & (apply_df[args.y] <= apply_df[args.hi])).mean())
            metrics["apply"]["raw_mpiw"] = float((apply_df[args.hi] - apply_df[args.lo]).mean())
            metrics["apply"]["cal_picp"] = float(((apply_out[args.y] >= apply_out[f"{args.lo}{args.suffix}"]) & (apply_out[args.y] <= apply_out[f"{args.hi}{args.suffix}"])).mean())
            metrics["apply"]["cal_mpiw"] = float((apply_out[f"{args.hi}{args.suffix}"] - apply_out[f"{args.lo}{args.suffix}"]).mean())

        payload = {
            "format": "csv",
            "cal": str(cal_path),
            "apply": str(apply_path),
            "saved": {"cal": str(cal_save), "apply": str(apply_save)},
            "params": {
                "y": args.y, "lo": args.lo, "hi": args.hi,
                "suffix": args.suffix,
                "target_picp": args.target_picp,
                "clip_min": args.clip_min, "clip_max": args.clip_max,
            },
            "conformal": {
                "t": summ.t,
                "n_cal": summ.n_cal,
                "q_level_used": summ.q_level_used,
            },
            "metrics": metrics,
        }
        summary_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Saved:", cal_save)
        print("Saved:", apply_save)
        print("Summary:", summary_out)

    else:
        raise ValueError("cal/apply must both be .npz or both be .csv (same format).")


if __name__ == "__main__":
    main()
