from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


HELDOUT_RE = re.compile(r"heldout_(\d+)")
SEED_RE = re.compile(r"seed_(\d+)")


def _safe_load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_ids(dir_path: Path) -> tuple[Optional[int], Optional[int]]:
    # Try to extract heldout_x and seed_y from the full path string
    s = str(dir_path).replace("\\", "/")
    mh = HELDOUT_RE.search(s)
    ms = SEED_RE.search(s)
    heldout = int(mh.group(1)) if mh else None
    seed = int(ms.group(1)) if ms else None
    return heldout, seed


def _find_first(dir_path: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(dir_path.glob(pat))
        if hits:
            return hits[0]
    return None


def collect_one_run(run_dir: Path, target_picp: float) -> List[Dict[str, Any]]:
    """
    Collect rows for:
      - stage=mc_raw       (old_raw)         from preds_*_mc_noise_summary.json
      - stage=noise_aug    (new_noise_aug)   from preds_*_mc_noise_summary.json
      - stage=conformal    (cal_*)           from *__conformal_0.90.json

    split:
      - inner_val
      - outer_test
    """
    heldout, seed = _parse_ids(run_dir)

    # You may have these two summaries (inner + outer)
    inner_sum = _find_first(run_dir, [
        "preds_inner_val_mc_noise_summary.json",
        "preds_inner_val_mc_noise_summary*.json",
    ])
    outer_sum = _find_first(run_dir, [
        "preds_outer_test_mc_noise_summary.json",
        "preds_outer_test_mc_noise_summary*.json",
    ])

    # Conformal summary produced by your npz-based conformal script
    conf = _find_first(run_dir, [f"*__conformal_{target_picp:.2f}.json"])

    rows: List[Dict[str, Any]] = []

    def add_row(*, split: str, stage: str, picp: float, mpiw: float, extra: Dict[str, Any] | None = None):
        r = {
            "heldout": heldout,
            "seed": seed,
            "run_dir": str(run_dir),
            "split": split,     # inner_val / outer_test
            "stage": stage,     # mc_raw / noise_aug / conformal
            "picp": float(picp),
            "mpiw": float(mpiw),
        }
        if extra:
            r.update(extra)
        rows.append(r)

    # ---- noise summaries: old_raw + new_noise_augmented ----
    def parse_noise_summary(p: Path, split: str):
        d = _safe_load_json(p)
        if not d:
            return
        if "old_raw" in d and isinstance(d["old_raw"], dict):
            add_row(
                split=split,
                stage="mc_raw",
                picp=d["old_raw"].get("picp", float("nan")),
                mpiw=d["old_raw"].get("mpiw", float("nan")),
            )
        if "new_noise_augmented" in d and isinstance(d["new_noise_augmented"], dict):
            extra = {
                "sigma_resid": d["new_noise_augmented"].get("sigma_resid", float("nan")),
                "mean_std_mc": d["new_noise_augmented"].get("mean_std_mc", float("nan")),
                "mean_std_total": d["new_noise_augmented"].get("mean_std_total", float("nan")),
            }
            add_row(
                split=split,
                stage="noise_aug",
                picp=d["new_noise_augmented"].get("picp", float("nan")),
                mpiw=d["new_noise_augmented"].get("mpiw", float("nan")),
                extra=extra,
            )

    if inner_sum:
        parse_noise_summary(inner_sum, "inner_val")
    if outer_sum:
        parse_noise_summary(outer_sum, "outer_test")

    # ---- conformal summary: cal metrics after conformal ----
    if conf:
        d = _safe_load_json(conf)
        if d and "metrics" in d:
            conf_info = d.get("conformal", {})
            extra_base = {
                "conformal_t": conf_info.get("t", float("nan")),
                "conformal_n_cal": conf_info.get("n_cal", float("nan")),
                "conformal_q_used": conf_info.get("q_level_used", float("nan")),
            }

            # metrics.cal.{raw_*, cal_*} and metrics.apply.{raw_*, cal_*}
            m_cal = d["metrics"].get("cal", {})
            m_app = d["metrics"].get("apply", {})

            # We store the FINAL (after conformal) as stage="conformal"
            if "cal_picp" in m_cal and "cal_mpiw" in m_cal:
                add_row(
                    split="inner_val",
                    stage="conformal",
                    picp=m_cal["cal_picp"],
                    mpiw=m_cal["cal_mpiw"],
                    extra=extra_base,
                )
            if "cal_picp" in m_app and "cal_mpiw" in m_app:
                add_row(
                    split="outer_test",
                    stage="conformal",
                    picp=m_app["cal_picp"],
                    mpiw=m_app["cal_mpiw"],
                    extra=extra_base,
                )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="root dir containing heldout_*/seed_* runs")
    ap.add_argument("--target_picp", type=float, default=0.90, help="target coverage used in conformal filename")
    ap.add_argument("--out_dir", type=str, default=None, help="output directory (default: root)")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    # find run dirs that look like .../heldout_x/seed_y
    run_dirs = []
    for p in root.rglob("*"):
        if p.is_dir():
            s = str(p).replace("\\", "/")
            if "/heldout_" in s and "/seed_" in s:
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

    # long table
    long_path = out_dir / "interval_stage_long.csv"
    df.to_csv(long_path, index=False)

    # aggregated table for paper: mean/std over runs
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

    agg_path = out_dir / "interval_stage_agg.csv"
    agg.to_csv(agg_path, index=False)

    print(f"[OK] wrote: {long_path}")
    print(f"[OK] wrote: {agg_path}")
    print("\nPreview (agg):")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()