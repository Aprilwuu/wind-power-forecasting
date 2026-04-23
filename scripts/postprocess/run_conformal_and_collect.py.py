from __future__ import annotations

"""
Batch-run conformal calibration for all seed directories under one experiment root
and collect run-level summary metrics into a single JSON file.

This script:
    1. Searches for seed_* directories under the given experiment root
    2. Detects calibration/apply prediction file pairs using common naming patterns
    3. Calls scripts.postprocess.run_conformal for each run
    4. Collects key metrics into postprocess_conformal_summary.json
    5. Computes WIS from PICP and MPIW for both raw and calibrated intervals

Typical usage:
    python -m scripts.postprocess.run_conformal_and_collect \
        --root data/featured/lgbm_qr_track1_lb168 \
        --target_picp 0.9

Note:
    This is a convenience batch wrapper built on top of run_conformal.py.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def wis_from_picp_mpiw(
    picp_val: Optional[float],
    mpiw_val: Optional[float],
    *,
    alpha: float = 0.10,
    lam: float = 0.5,
) -> Optional[float]:
    if picp_val is None or mpiw_val is None:
        return None
    nominal = 1.0 - float(alpha)
    return float(lam * abs(float(picp_val) - nominal) + (1.0 - lam) * float(mpiw_val))


def pick_pair(seed_dir: Path) -> Optional[Tuple[Path, Path]]:
    """
    Return (cal_path, apply_path) for conformal, or None if not found.

    Supports common naming styles across models and tracks.

    Preferred order:
      1. Track2-style exact pairs
      2. Track1-style exact pairs
      3. Backward-compatible variants
    """
    candidates = [
        # Generic naming
        ("preds_inner_val.npz", "preds_outer_test.npz"),
        ("preds_val.npz", "preds_test.npz"),

        # MC-dropout / noise style
        ("preds_inner_val_mc.npz", "preds_outer_test_mc.npz"),
        ("preds_val_mc.npz", "preds_test_mc.npz"),

        # LGBM-QR style
        ("preds_inner_val_lgbm_qr.npz", "preds_outer_test_lgbm_qr.npz"),
        ("preds_val_lgbm_qr.npz", "preds_test_lgbm_qr.npz"),

        # Transformer / QR / Beta style
        ("preds_inner_val_qr.npz", "preds_outer_test_qr.npz"),
        ("preds_val_qr.npz", "preds_test_qr.npz"),
        ("preds_inner_val_beta.npz", "preds_outer_test_beta.npz"),
        ("preds_val_beta.npz", "preds_test_beta.npz"),

        # Backward compatibility variants
        ("preds_inner_val.npz", "preds_test.npz"),
        ("preds_val.npz", "preds_outer_test.npz"),
        ("preds_inner_val_mc.npz", "preds_test_mc.npz"),
        ("preds_val_mc.npz", "preds_outer_test_mc.npz"),
        ("preds_inner_val_lgbm_qr.npz", "preds_test_lgbm_qr.npz"),
        ("preds_val_lgbm_qr.npz", "preds_outer_test_lgbm_qr.npz"),
    ]

    for cal_name, apply_name in candidates:
        cal_path = seed_dir / cal_name
        apply_path = seed_dir / apply_name
        if cal_path.exists() and apply_path.exists():
            return cal_path, apply_path

    return None


def run_one(cal_path: Path, apply_path: Path, target_picp: float) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.postprocess.run_conformal",
        "--cal",
        str(cal_path),
        "--apply",
        str(apply_path),
        "--target_picp",
        str(target_picp),
        "--lo",
        "q05",
        "--hi",
        "q95",
        "--y",
        "y_true",
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch-run conformal calibration and collect summary metrics."
    )
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Experiment root, e.g. data/featured/lgbm_qr_track1_lb168",
    )
    ap.add_argument(
        "--target_picp",
        type=float,
        default=0.9,
        help="Target coverage level for conformal calibration (default: 0.9)",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip seed directories that already contain conformal summary JSON files",
    )
    ap.add_argument(
        "--lam",
        type=float,
        default=0.5,
        help="Weight for |PICP - nominal| in WIS surrogate (default: 0.5)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Experiment root not found: {root}")

    seed_dirs = sorted([p for p in root.glob("**/seed_*") if p.is_dir()])
    if not seed_dirs:
        raise RuntimeError(f"No seed_* directories found under: {root}")

    alpha = 1.0 - float(args.target_picp)

    results = []
    skipped = 0
    no_pair = 0

    for sd in seed_dirs:
        pair = pick_pair(sd)
        if pair is None:
            no_pair += 1
            print(f"[SKIP] No prediction pair found in: {sd}")
            continue

        cal_path, apply_path = pair
        out_dir = sd / "postprocess"
        out_dir.mkdir(parents=True, exist_ok=True)

        existing_summaries = list(out_dir.glob("*__conformal_*.json"))
        if args.skip_existing and existing_summaries:
            skipped += 1
            print(f"[SKIP] Existing conformal summary found in: {sd}")
            continue

        run_one(cal_path, apply_path, args.target_picp)

        new_summaries = sorted(
            out_dir.glob("*__conformal_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if new_summaries:
            sp = new_summaries[0]
            obj = json.loads(sp.read_text(encoding="utf-8"))

            cal_raw_picp = obj.get("metrics", {}).get("cal", {}).get("raw_picp")
            cal_cal_picp = obj.get("metrics", {}).get("cal", {}).get("cal_picp")
            apply_raw_picp = obj.get("metrics", {}).get("apply", {}).get("raw_picp")
            apply_cal_picp = obj.get("metrics", {}).get("apply", {}).get("cal_picp")

            cal_raw_mpiw = obj.get("metrics", {}).get("cal", {}).get("raw_mpiw")
            cal_cal_mpiw = obj.get("metrics", {}).get("cal", {}).get("cal_mpiw")
            apply_raw_mpiw = obj.get("metrics", {}).get("apply", {}).get("raw_mpiw")
            apply_cal_mpiw = obj.get("metrics", {}).get("apply", {}).get("cal_mpiw")

            results.append(
                {
                    "seed_dir": str(sd),
                    "cal_file": str(cal_path),
                    "apply_file": str(apply_path),
                    "summary": str(sp),
                    "t": obj.get("conformal", {}).get("t"),

                    "cal_raw_picp": cal_raw_picp,
                    "cal_cal_picp": cal_cal_picp,
                    "apply_raw_picp": apply_raw_picp,
                    "apply_cal_picp": apply_cal_picp,

                    "cal_raw_mpiw": cal_raw_mpiw,
                    "cal_cal_mpiw": cal_cal_mpiw,
                    "apply_raw_mpiw": apply_raw_mpiw,
                    "apply_cal_mpiw": apply_cal_mpiw,

                    "cal_raw_wis": wis_from_picp_mpiw(
                        cal_raw_picp, cal_raw_mpiw, alpha=alpha, lam=args.lam
                    ),
                    "cal_cal_wis": wis_from_picp_mpiw(
                        cal_cal_picp, cal_cal_mpiw, alpha=alpha, lam=args.lam
                    ),
                    "apply_raw_wis": wis_from_picp_mpiw(
                        apply_raw_picp, apply_raw_mpiw, alpha=alpha, lam=args.lam
                    ),
                    "apply_cal_wis": wis_from_picp_mpiw(
                        apply_cal_picp, apply_cal_mpiw, alpha=alpha, lam=args.lam
                    ),

                    "wis_alpha": alpha,
                    "wis_lambda": args.lam,
                }
            )
        else:
            print(f"[WARN] No conformal summary JSON found after running: {sd}")

    out_summary = root / "postprocess_conformal_summary.json"
    out_summary.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n[OK] Saved: {out_summary}")
    print(f"[INFO] n_done={len(results)} / n_seed_dirs={len(seed_dirs)}")
    print(f"[INFO] skipped_existing={skipped}")
    print(f"[INFO] no_pair_found={no_pair}")


if __name__ == "__main__":
    main()