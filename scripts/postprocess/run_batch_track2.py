from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import subprocess
import sys


def run_cmd(cmd: List[str]) -> None:
    print("\n>>>", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:\n", r.stdout)
        print("STDERR:\n", r.stderr)
        raise RuntimeError(f"Command failed with code {r.returncode}")
    # print useful stdout
    if r.stdout.strip():
        print(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True,
                    help="Base dir containing heldout_*/seed_* folders")
    ap.add_argument("--heldouts", type=str, default="1-10",
                    help="Heldouts like '1-10' or '1,3,5'")
    ap.add_argument("--seeds", type=str, default="42,43,44",
                    help="Seeds like '42,43,44'")
    ap.add_argument("--tag", type=str, default="noise",
                    help="Tag used by run_add_noise outputs (default noise)")
    ap.add_argument("--target_picp", type=float, default=0.9)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)

    # parse heldouts
    if "-" in args.heldouts:
        a, b = args.heldouts.split("-")
        heldouts = list(range(int(a), int(b) + 1))
    else:
        heldouts = [int(x.strip()) for x in args.heldouts.split(",") if x.strip()]

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []

    for h in heldouts:
        for s in seeds:
            seed_dir = base_dir / f"heldout_{h}" / f"seed_{s}"
            post_dir = seed_dir / "postprocess"
            post_dir.mkdir(parents=True, exist_ok=True)

            inner_npz = seed_dir / "preds_inner_val_mc.npz"
            outer_npz = seed_dir / "preds_outer_test_mc.npz"

            if not inner_npz.exists() or not outer_npz.exists():
                print(f"[SKIP] missing npz for heldout={h} seed={s}: {seed_dir}")
                continue

            # 1) noise for inner (so conformal calibration uses same interval form)
            run_cmd([
                sys.executable, "-m", "scripts.postprocess.run_add_noise",
                "--inner", str(inner_npz),
                "--outer", str(inner_npz),
                "--out_dir", str(post_dir),
                "--tag", args.tag,
            ])

            # 2) noise for outer
            run_cmd([
                sys.executable, "-m", "scripts.postprocess.run_add_noise",
                "--inner", str(inner_npz),
                "--outer", str(outer_npz),
                "--out_dir", str(post_dir),
                "--tag", args.tag,
            ])

            inner_noise = post_dir / f"{inner_npz.stem}_{args.tag}.npz"           # preds_inner_val_mc_noise.npz
            outer_noise = post_dir / f"{outer_npz.stem}_{args.tag}.npz"           # preds_outer_test_mc_noise.npz

            if not inner_noise.exists() or not outer_noise.exists():
                print(f"[SKIP] noise outputs missing for heldout={h} seed={s}")
                continue

            # 3) conformal (widening) using inner_noise -> apply to outer_noise
            run_cmd([
                sys.executable, "-m", "scripts.postprocess.run_conformal",
                "--cal", str(inner_noise),
                "--apply", str(outer_noise),
                "--target_picp", str(args.target_picp),
                "--lo", "q05", "--hi", "q95", "--y", "y_true",
            ])

            # 4) read summary json and collect
            summ_path = post_dir / f"{outer_noise.stem}__conformal_{args.target_picp:.2f}.json"
            if not summ_path.exists():
                print(f"[WARN] summary not found: {summ_path}")
                continue

            summ = json.loads(summ_path.read_text(encoding="utf-8"))
            row = {
                "heldout": h,
                "seed": s,
                "t": summ["conformal"]["t"],
                "n_cal": summ["conformal"]["n_cal"],
                "q_level_used": summ["conformal"]["q_level_used"],
                # metrics (raw == cal if t=0 and same clip)
                "cal_picp": summ["metrics"]["cal"]["raw_picp"],
                "cal_mpiw": summ["metrics"]["cal"]["raw_mpiw"],
                "apply_picp": summ["metrics"]["apply"]["raw_picp"],
                "apply_mpiw": summ["metrics"]["apply"]["raw_mpiw"],
                "summary_path": str(summ_path),
            }
            rows.append(row)
            print(f"[OK] heldout={h} seed={s} apply_picp={row['apply_picp']:.4f} apply_mpiw={row['apply_mpiw']:.4f}")

    if rows:
        df = pd.DataFrame(rows).sort_values(["heldout", "seed"])
        out_csv = base_dir / "batch_summary_noise_conformal.csv"
        df.to_csv(out_csv, index=False)
        print("\nSaved batch summary:", out_csv)
    else:
        print("\nNo runs collected.")


if __name__ == "__main__":
    main()
