# scripts/postprocess/run_add_noise.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from src.postprocess.noise import (
    load_npz,
    save_npz,
    estimate_sigma_resid,
    augment_intervals_with_residual_noise,
    summarize_intervals,
)


def main():
    ap = argparse.ArgumentParser(description="Add residual noise to MC-dropout intervals (post-hoc).")
    ap.add_argument("--inner", type=str, required=True, help="Path to preds_inner_val_mc.npz")
    ap.add_argument("--outer", type=str, required=True, help="Path to preds_outer_test_mc.npz")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <outer_dir>/postprocess)")
    ap.add_argument("--lo", type=float, default=0.05, help="Lower quantile (default 0.05)")
    ap.add_argument("--hi", type=float, default=0.95, help="Upper quantile (default 0.95)")
    ap.add_argument("--sigma_method", type=str, default="rmse", choices=["rmse", "std", "mad"],
                    help="How to estimate sigma_resid from inner_val residuals")
    ap.add_argument("--min_sigma", type=float, default=1e-8, help="Minimum sigma_resid floor")
    ap.add_argument("--tag", type=str, default="noise", help="Tag for output filenames")
    args = ap.parse_args()

    inner_path = Path(args.inner)
    outer_path = Path(args.outer)

    inner = load_npz(inner_path)
    outer = load_npz(outer_path)

    # z for symmetric central interval [lo, hi]
    # For (0.05, 0.95), z = inv_cdf(0.95) ≈ 1.645
    z = NormalDist().inv_cdf(args.hi)

    sigma_resid = estimate_sigma_resid(inner, method=args.sigma_method, min_sigma=args.min_sigma)

    augmented = augment_intervals_with_residual_noise(
        outer_npz=outer,
        sigma_resid=sigma_resid,
        lo=args.lo,
        hi=args.hi,
        z_value=z,
    )

    # clip intervals to [0,1] because target is normalized
    augmented["q05"] = np.clip(augmented["q05"], 0.0, 1.0).astype(np.float32)
    augmented["q95"] = np.clip(augmented["q95"], 0.0, 1.0).astype(np.float32)

    # Summaries: old vs new
    y = outer["y_true"]
    old_q05 = outer["q05"]
    old_q95 = outer["q95"]
    old_picp = float(np.mean((y.reshape(-1) >= old_q05.reshape(-1)) & (y.reshape(-1) <= old_q95.reshape(-1))))
    old_mpiw = float(np.mean(old_q95.reshape(-1) - old_q05.reshape(-1)))

    new_summary = summarize_intervals(
        y_true=augmented["y_true"],
        qlo=augmented["q05"],
        qhi=augmented["q95"],
        sigma_resid=sigma_resid,
        std_mc=augmented["std_mc"],
        std_total=augmented["std_total"],
        lo=args.lo,
        hi=args.hi,
    )

    # Output paths
    out_dir = Path(args.out_dir) if args.out_dir else (outer_path.parent / "postprocess")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"{outer_path.stem}_{args.tag}.npz"
    out_json = out_dir / f"{outer_path.stem}_{args.tag}_summary.json"

    save_npz(out_npz, augmented)

    payload = {
        "inputs": {
            "inner": str(inner_path),
            "outer": str(outer_path),
            "lo": args.lo,
            "hi": args.hi,
            "z": z,
            "sigma_method": args.sigma_method,
            "min_sigma": args.min_sigma,
        },
        "old_raw": {
            "picp": old_picp,
            "mpiw": old_mpiw,
        },
        "new_noise_augmented": {
            "picp": new_summary.picp,
            "mpiw": new_summary.mpiw,
            "sigma_resid": new_summary.sigma_resid,
            "mean_std_mc": new_summary.mean_std_mc,
            "mean_std_total": new_summary.mean_std_total,
        },
        "outputs": {
            "npz": str(out_npz),
        },
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Done ===")
    print("Saved:", out_npz)
    print("Summary:", out_json)
    print("OLD  PICP:", old_picp, "MPIW:", old_mpiw)
    print("NEW  PICP:", new_summary.picp, "MPIW:", new_summary.mpiw)
    print("sigma_resid:", new_summary.sigma_resid)
    print("mean std_mc:", new_summary.mean_std_mc, "mean std_total:", new_summary.mean_std_total)


if __name__ == "__main__":
    main()
