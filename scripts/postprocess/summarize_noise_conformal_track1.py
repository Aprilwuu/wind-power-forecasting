from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

SEED_RE = re.compile(r"seed_(\d+)")

def _safe_load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _parse_seed(dir_path: Path) -> Optional[int]:
    s = str(dir_path).replace("\\", "/")
    m = SEED_RE.search(s)
    return int(m.group(1)) if m else None

def _find_first(dir_path: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(dir_path.glob(pat))
        if hits:
            return hits[0]
    return None

def collect_one_run(run_dir: Path, target_picp: float) -> List[Dict[str, Any]]:
    """
    Track1 uses:
      - val summary:  preds_val_mc_noise_summary.json
      - test summary: preds_test_mc_noise_summary.json
      - conformal:    *__conformal_0.90.json
    stages:
      - mc_raw      (old_raw)
      - noise_aug   (new_noise_augmented)
      - conformal   (metrics.cal / metrics.apply)
    """
    seed = _parse_seed(run_dir)

    val_sum = _find_first(run_dir, ["preds_val_mc_noise_summary.json", "preds_val_mc_noise_summary*.json"])
    test_sum = _find_first(run_dir, ["preds_test_mc_noise_summary.json", "preds_test_mc_noise_summary*.json"])
    conf = _find_first(run_dir, [f"*__conformal_{target_picp:.2f}.json"])

    rows: List[Dict[str, Any]] = []

    def add_row(*, split: str, stage: str, picp: float, mpiw: float, extra: Dict[str, Any] | None = None):
        r = {
            "seed": seed,
            "run_dir": str(run_dir),
            "split": split,   # val / test
            "stage": stage,   # mc_raw / noise_aug / conformal
            "picp": float(picp),
            "mpiw": float(mpiw),
        }
        if extra:
            r.update(extra)
        rows.append(r)

    def parse_noise_summary(p: Path, split: str):
        d = _safe_load_json(p)
        if not d:
            return
        if "old_raw" in d and isinstance(d["old_raw"], dict):
            add_row(split=split, stage="mc_raw",
                    picp=d["old_raw"].get("picp", float("nan")),
                    mpiw=d["old_raw"].get("mpiw", float("nan")))
        if "new_noise_augmented" in d and isinstance(d["new_noise_augmented"], dict):
            extra = {
                "sigma_resid": d["new_noise_augmented"].get("sigma_resid", float("nan")),
                "mean_std_mc": d["new_noise_augmented"].get("mean_std_mc", float("nan")),
                "mean_std_total": d["new_noise_augmented"].get("mean_std_total", float("nan")),
            }
            add_row(split=split, stage="noise_aug",
                    picp=d["new_noise_augmented"].get("picp", float("nan")),
                    mpiw=d["new_noise_augmented"].get("mpiw", float("nan")),
                    extra=extra)

    if val_sum:
        parse_noise_summary(val_sum, "val")
    if test_sum:
        parse_noise_summary(test_sum, "test")

    # conformal (post-hoc)
    if conf:
        d = _safe_load_json(conf)
        if d and "metrics" in d:
            conf_info = d.get("conformal", {})
            extra_base = {
                "conformal_t": conf_info.get("t", float("nan")),
                "conformal_n_cal": conf_info.get("n_cal", float("nan")),
                "conformal_q_used": conf_info.get("q_level_used", float("nan")),
            }
            m_cal = d["metrics"].get("cal", {})   # calibration split metrics (val)
            m_app = d["metrics"].get("apply", {}) # apply split metrics (test)

            if "cal_picp" in m_cal and "cal_mpiw" in m_cal:
                add_row(split="val", stage="conformal",
                        picp=m_cal["cal_picp"], mpiw=m_cal["cal_mpiw"], extra=extra_base)
            if "cal_picp" in m_app and "cal_mpiw" in m_app:
                add_row(split="test", stage="conformal",
                        picp=m_app["cal_picp"], mpiw=m_app["cal_mpiw"], extra=extra_base)

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="root dir containing seed_* runs")
    ap.add_argument("--target_picp", type=float, default=0.90)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    # run dirs: any directory path that contains /seed_
    run_dirs = []
    for p in root.rglob("*"):
        if p.is_dir():
            s = str(p).replace("\\", "/")
            if "/seed_" in s:
                run_dirs.append(p)

    all_rows: List[Dict[str, Any]] = []
    for d in sorted(set(run_dirs)):
        rows = collect_one_run(d, target_picp=args.target_picp)
        if rows:
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No rows collected. Check filenames and root path.")
        return

    long_path = out_dir / "track1_interval_stage_long.csv"
    df.to_csv(long_path, index=False)

    agg = (
        df.groupby(["split", "stage"], dropna=False)
          .agg(
              n=("picp", "count"),
              picp_mean=("picp", "mean"),
              picp_std=("picp", "std"),
              mpiw_mean=("mpiw", "mean"),
              mpiw_std=("mpiw", "std"),
              t_mean=("conformal_t", "mean"),
          )
          .reset_index()
          .sort_values(["split", "stage"])
    )

    agg_path = out_dir / "track1_interval_stage_agg.csv"
    agg.to_csv(agg_path, index=False)

    print(f"[OK] wrote: {long_path}")
    print(f"[OK] wrote: {agg_path}")
    print("\nPreview (agg):")
    print(agg.to_string(index=False))

if __name__ == "__main__":
    main()