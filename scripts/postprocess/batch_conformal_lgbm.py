from __future__ import annotations
import argparse
import sys
import json
import subprocess
from pathlib import Path


def pick_pair(seed_dir: Path):
    """
    Return (cal_path, apply_path) for conformal, or None if not found.

    Supports both naming styles:
      - Track1: preds_val_lgbm_qr.npz + preds_test_lgbm_qr.npz
      - Track2: preds_inner_val_lgbm_qr.npz + preds_outer_test_lgbm_qr.npz
    """
    candidates = [
        ("preds_inner_val_lgbm_qr.npz", "preds_outer_test_lgbm_qr.npz"),
        ("preds_val_lgbm_qr.npz", "preds_test_lgbm_qr.npz"),
        # Additional naming variants (for backward compatibility)
        ("preds_inner_val_lgbm_qr.npz", "preds_test_lgbm_qr.npz"),
        ("preds_val_lgbm_qr.npz", "preds_outer_test_lgbm_qr.npz"),
    ]
    for a, b in candidates:
        pa = seed_dir / a
        pb = seed_dir / b
        if pa.exists() and pb.exists():
            return pa, pb
    return None


def run_one(cal_path: Path, apply_path: Path, target_picp: float) -> None:
    cmd = [
        sys.executable, "-m", "scripts.postprocess.run_conformal",
        "--cal", str(cal_path),
        "--apply", str(apply_path),
        "--target_picp", str(target_picp),
        "--lo", "q05",
        "--hi", "q95",
        "--y", "y_true",
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Experiment root, e.g. data/featured/lgbm_qr_track1_lb168",
    )
    ap.add_argument("--target_picp", type=float, default=0.9)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    # Find all seed_*/ directories:
    #   - Track1: .../seed_42
    #   - Track2: .../heldout_1/seed_42
    seed_dirs = sorted(root.glob("**/seed_*"))
    if not seed_dirs:
        raise RuntimeError(f"No seed_* dirs found under {root}")

    results = []
    for sd in seed_dirs:
        pair = pick_pair(sd)
        if pair is None:
            # No prediction files found in this seed directory; skip.
            continue
        cal_path, apply_path = pair

        # If conformal outputs already exist, we may choose to skip.
        # (To always rerun, just keep it as-is; to skip existing, uncomment the block below.)
        out_dir = sd / "postprocess"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_glob = list(out_dir.glob("*__conformal_*.json"))
        # Uncomment to skip directories that already have conformal summaries:
        # if summary_glob:
        #     continue

        run_one(cal_path, apply_path, args.target_picp)

        # Collect the newest summary file.
        # run_conformal produces something like: <apply_stem>__conformal_0.90.json
        new_summaries = sorted(
            out_dir.glob("*__conformal_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if new_summaries:
            sp = new_summaries[0]
            obj = json.loads(sp.read_text(encoding="utf-8"))
            results.append({
                "seed_dir": str(sd),
                "summary": str(sp),
                "t": obj.get("conformal", {}).get("t"),
                "cal_raw_picp": obj.get("metrics", {}).get("cal", {}).get("raw_picp"),
                "cal_cal_picp": obj.get("metrics", {}).get("cal", {}).get("cal_picp"),
                "apply_raw_picp": obj.get("metrics", {}).get("apply", {}).get("raw_picp"),
                "apply_cal_picp": obj.get("metrics", {}).get("apply", {}).get("cal_picp"),
                "apply_raw_mpiw": obj.get("metrics", {}).get("apply", {}).get("raw_mpiw"),
                "apply_cal_mpiw": obj.get("metrics", {}).get("apply", {}).get("cal_mpiw"),
            })

    # Write a single summary JSON for easier reporting / table generation.
    out_summary = root / "postprocess_conformal_summary.json"
    out_summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_summary}")
    print(f"n_done={len(results)} / n_seed_dirs={len(seed_dirs)}")


if __name__ == "__main__":
    main()
