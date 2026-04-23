# scripts/postprocess/apply_conformal_npz.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../wind-power-forecasting
sys.path.insert(0, str(ROOT))

from src.postprocess.conformal import (
    apply_conformal_to_npz,
    load_npz,
    mpiw,
    picp,
    save_npz,
)


def _default_paths(exp_dir: Path, track: str) -> tuple[Path, Path]:
    """
    Provide default (cal, apply) npz paths for track1/track2.

    Adjust these names if your project uses different filenames.
    """
    if track == "track1":
        cal_npz = exp_dir / "intervals_val.npz"
        apply_npz = exp_dir / "intervals_test.npz"
    else:
        cal_npz = exp_dir / "intervals_inner_val.npz"
        apply_npz = exp_dir / "intervals_outer_test.npz"
    return cal_npz, apply_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True, help="experiment directory")
    ap.add_argument("--track", choices=["track1", "track2"], default="track1")

    # Optional explicit paths (override defaults)
    ap.add_argument("--cal_npz", type=str, default=None, help="calibration npz path (val / inner_val)")
    ap.add_argument("--apply_npz", type=str, default=None, help="apply npz path (test / outer_test)")

    # Keys inside npz
    ap.add_argument("--y_key", type=str, default="y_true")
    ap.add_argument("--lo_key", type=str, default="q05")
    ap.add_argument("--hi_key", type=str, default="q95")

    # Conformal params
    ap.add_argument("--target_picp", type=float, default=0.9)
    ap.add_argument("--out_suffix", type=str, default="_cal")
    ap.add_argument("--clip_min", type=float, default=0.0)
    ap.add_argument("--clip_max", type=float, default=1.0)

    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)

    if args.cal_npz is None or args.apply_npz is None:
        cal_path, apply_path = _default_paths(exp_dir, args.track)
    else:
        cal_path, apply_path = Path(args.cal_npz), Path(args.apply_npz)

    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration npz not found: {cal_path}")
    if not apply_path.exists():
        raise FileNotFoundError(f"Apply npz not found: {apply_path}")

    # Load npz dicts
    cal_npz = load_npz(cal_path)
    apply_npz = load_npz(apply_path)

    # Apply conformal widening (writes new keys into dicts)
    cal_out, apply_out, summary = apply_conformal_to_npz(
        cal_npz,
        apply_npz,
        y_key=args.y_key,
        lo_key=args.lo_key,
        hi_key=args.hi_key,
        target_picp=args.target_picp,
        out_suffix=args.out_suffix,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )

    # ---- metrics (raw vs calibrated) ----
    raw_val_picp = picp(cal_npz[args.y_key], cal_npz[args.lo_key], cal_npz[args.hi_key])
    raw_val_mpiw = mpiw(cal_npz[args.lo_key], cal_npz[args.hi_key])
    raw_test_picp = picp(apply_npz[args.y_key], apply_npz[args.lo_key], apply_npz[args.hi_key])
    raw_test_mpiw = mpiw(apply_npz[args.lo_key], apply_npz[args.hi_key])

    lo_cal = f"{args.lo_key}{args.out_suffix}"
    hi_cal = f"{args.hi_key}{args.out_suffix}"

    cal_val_picp = picp(cal_out[args.y_key], cal_out[lo_cal], cal_out[hi_cal])
    cal_val_mpiw = mpiw(cal_out[lo_cal], cal_out[hi_cal])
    cal_test_picp = picp(apply_out[args.y_key], apply_out[lo_cal], apply_out[hi_cal])
    cal_test_mpiw = mpiw(apply_out[lo_cal], apply_out[hi_cal])

    metrics_block = {
        "raw": {
            "val_picp": raw_val_picp,
            "val_mpiw": raw_val_mpiw,
            "test_picp": raw_test_picp,
            "test_mpiw": raw_test_mpiw,
        },
        "cal": {
            "val_picp": cal_val_picp,
            "val_mpiw": cal_val_mpiw,
            "test_picp": cal_test_picp,
            "test_mpiw": cal_test_mpiw,
        },
    }

    # Output paths
    cal_out_path = cal_path.with_name(cal_path.stem + args.out_suffix + cal_path.suffix)
    apply_out_path = apply_path.with_name(apply_path.stem + args.out_suffix + apply_path.suffix)

    # Save widened npz files
    save_npz(cal_out_path, cal_out)
    save_npz(apply_out_path, apply_out)

    # Summary JSON (include metrics + paths)
    payload = {
        **summary.__dict__,
        "metrics": metrics_block,
        "cal_in": str(cal_path),
        "apply_in": str(apply_path),
        "cal_out": str(cal_out_path),
        "apply_out": str(apply_out_path),
    }

    summary_path = exp_dir / "conformal_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()