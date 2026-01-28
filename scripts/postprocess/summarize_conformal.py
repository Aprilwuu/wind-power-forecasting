from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(p: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write.")
    p.parent.mkdir(parents=True, exist_ok=True)
    # stable column order: put common keys first, then others
    common = [
        "track",
        "model",
        "exp_root",
        "candidate",
        "heldout",
        "seed",
        "target_picp",
        "t",
        "cal_raw_picp",
        "cal_cal_picp",
        "apply_raw_picp",
        "apply_cal_picp",
        "apply_raw_mpiw",
        "apply_cal_mpiw",
        "summary_path",
        "seed_dir",
    ]
    keys = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = [k for k in common if k in keys] + sorted([k for k in keys if k not in common])

    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _mean_std(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    return {"mean": mean, "std": std, "min": min(vals), "max": max(vals), "n": n}


def _infer_track_from_path(exp_root: Path) -> str:
    # based on our naming: track1 vs track2
    name = exp_root.name.lower()
    if "track2" in name:
        return "track2"
    if "track1" in name:
        return "track1"
    # fallback: detect heldout_ directories
    if list(exp_root.glob("**/heldout_*")):
        return "track2"
    return "track1"


def _infer_model_from_path(exp_root: Path) -> str:
    name = exp_root.name.lower()
    if "lgbm" in name:
        return "lgbm"
    if "tcn" in name:
        return "tcn"
    return "unknown"


def _parse_heldout_from_seed_dir(seed_dir: Path) -> Optional[str]:
    # .../heldout_1/seed_42  -> "1"
    for part in seed_dir.parts:
        if part.lower().startswith("heldout_"):
            return part.split("_", 1)[1]
    return None

def _infer_candidate_from_name(name: str) -> str:
    n = name.lower()
    # looking for keyword "mc_noise"
    if "mc_noise" in n:
        return "mc_noise"
    # maybe have preds_*_mc__conformal_0.90.json in future
    if "_mc" in n:
        return "mc"
    # default: raw
    return "raw"


def load_batch_summary(exp_root: Path) -> Tuple[List[Dict[str, Any]], Path]:
    """
    Preferred: exp_root/postprocess_conformal_summary.json (from batch scripts)
    Fallback:  scan **/postprocess/*__conformal_*.json and parse them directly
    """
    exp_root = exp_root.resolve()
    summary_path = exp_root / "postprocess_conformal_summary.json"

    track = _infer_track_from_path(exp_root)
    model = _infer_model_from_path(exp_root)

    rows: List[Dict[str, Any]] = []

    if summary_path.exists():
        items = _read_json(summary_path)
        if not isinstance(items, list):
            raise TypeError(f"{summary_path} should be a list of runs.")
        for obj in items:
            seed_dir = Path(obj.get("seed_dir", ""))
            heldout = _parse_heldout_from_seed_dir(seed_dir)

            summary_file = Path(obj.get("summary", ""))

            cand = obj.get("candidate")
            if cand is None or str(cand).strip().lower() in ("", "baseline", "raw"):  # raw 你也可以不放
                cand = None

            pipeline = obj.get("pipeline")
            if pipeline is None or str(pipeline).strip().lower() in ("", "baseline"):
                pipeline = None

            base_variant = obj.get("base_variant")
            if base_variant is None or str(base_variant).strip().lower() in ("", "baseline"):
                base_variant = None

            candidate = str(
                cand
                or pipeline
                or base_variant
                or _infer_candidate_from_name(summary_file.name)
            )

            target_picp = None
            if summary_file.exists():
                try:
                    detail = _read_json(summary_file)
                    target_picp = detail.get("params", {}).get("target_picp")
                except Exception:
                    target_picp = None
        

            rows.append(
                {
                    "track": track,
                    "model": model,
                    "exp_root": str(exp_root),
                    "candidate": candidate,
                    "heldout": heldout,
                    "seed": int(str(seed_dir.name).split("_", 1)[1]) if seed_dir.name.startswith("seed_") else None,
                    "target_picp": _safe_float(target_picp),
                    "t": _safe_float(obj.get("t")),
                    "cal_raw_picp": _safe_float(obj.get("cal_raw_picp")),
                    "cal_cal_picp": _safe_float(obj.get("cal_cal_picp")),
                    "apply_raw_picp": _safe_float(obj.get("apply_raw_picp")),
                    "apply_cal_picp": _safe_float(obj.get("apply_cal_picp")),
                    "apply_raw_mpiw": _safe_float(obj.get("apply_raw_mpiw")),
                    "apply_cal_mpiw": _safe_float(obj.get("apply_cal_mpiw")),
                    "summary_path": str(obj.get("summary")),
                    "seed_dir": str(obj.get("seed_dir")),
                }
            )
        return rows, summary_path

    # -------------------------
    # Fallback: scan conformal jsons directly
    # -------------------------
    conformal_files = sorted(exp_root.glob("**/postprocess/*__conformal_*.json"))
    if not conformal_files:
        raise FileNotFoundError(
            f"Not found: {summary_path}\n"
            f"Also found no fallback files under: {exp_root}\\**\\postprocess\\*__conformal_*.json"
        )

    for sp in conformal_files:
        try:
            obj = _read_json(sp)
        except Exception:
            continue

        seed_dir = sp.parent.parent  # .../seed_42/postprocess/<file>
        heldout = _parse_heldout_from_seed_dir(seed_dir)

        params = obj.get("params", {}) if isinstance(obj.get("params"), dict) else {}
        conf = obj.get("conformal", {}) if isinstance(obj.get("conformal"), dict) else {}
        mets = obj.get("metrics", {}) if isinstance(obj.get("metrics"), dict) else {}
        cal = mets.get("cal", {}) if isinstance(mets.get("cal"), dict) else {}
        app = mets.get("apply", {}) if isinstance(mets.get("apply"), dict) else {}

        candidate = _infer_candidate_from_name(sp.name)
        
        rows.append(
            {
                "track": track,
                "model": model,
                "exp_root": str(exp_root),
                "candidate": candidate,
                "heldout": heldout,
                "seed": int(seed_dir.name.split("_", 1)[1]) if seed_dir.name.startswith("seed_") else None,
                "target_picp": _safe_float(params.get("target_picp")),
                "t": _safe_float(conf.get("t")),
                "cal_raw_picp": _safe_float(cal.get("raw_picp")),
                "cal_cal_picp": _safe_float(cal.get("cal_picp")),
                "apply_raw_picp": _safe_float(app.get("raw_picp")),
                "apply_cal_picp": _safe_float(app.get("cal_picp")),
                "apply_raw_mpiw": _safe_float(app.get("raw_mpiw")),
                "apply_cal_mpiw": _safe_float(app.get("cal_mpiw")),
                "summary_path": str(sp),
                "seed_dir": str(seed_dir),
            }
        )

    # use a "virtual" source path for logging
    return rows, exp_root / "__scanned_conformal_jsons__"

def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate conformal results into paper-friendly summaries.

    Group keys:
      - model (lgbm/tcn)
      - track (track1/track2)
      - candidate (baseline by default)
    For track2, we also provide per-heldout aggregates.
    """

    def key_of(r: Dict[str, Any]) -> Tuple[str, str, str]:
        return (str(r.get("model")), str(r.get("track")), str(r.get("candidate")))

    def summarize(group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Collect metrics (ignore None)
        def collect(field: str) -> List[float]:
            return [v for v in (_safe_float(x.get(field)) for x in group_rows) if v is not None]

        out = {
            "n": len(group_rows),
            "t": _mean_std(collect("t")),
            "cal_raw_picp": _mean_std(collect("cal_raw_picp")),
            "cal_cal_picp": _mean_std(collect("cal_cal_picp")),
            "apply_raw_picp": _mean_std(collect("apply_raw_picp")),
            "apply_cal_picp": _mean_std(collect("apply_cal_picp")),
            "apply_raw_mpiw": _mean_std(collect("apply_raw_mpiw")),
            "apply_cal_mpiw": _mean_std(collect("apply_cal_mpiw")),
        }
        return out

    # -------------------------
    # Aggregate by (model, track, candidate)
    # -------------------------
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(key_of(r), []).append(r)

    by_model_track: Dict[str, Any] = {}
    for (model, track, cand), grp in sorted(grouped.items(), key=lambda x: x[0]):
        k = f"{model}__{track}__{cand}"
        by_model_track[k] = summarize(grp)

    # -------------------------
    # Extra: track2 per-heldout aggregates
    # -------------------------
    track2_rows = [r for r in rows if str(r.get("track")) == "track2"]
    per_heldout: Dict[str, Any] = {}

    if track2_rows:
        # key: model__candidate__heldout
        hh: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for r in track2_rows:
            model = str(r.get("model"))
            cand = str(r.get("candidate"))
            held = str(r.get("heldout"))
            hh.setdefault((model, cand, held), []).append(r)

        for (model, cand, held), grp in sorted(hh.items(), key=lambda x: x[0]):
            k = f"{model}__{cand}__heldout_{held}"
            per_heldout[k] = summarize(grp)

    return {
        "by_model_track_candidate": by_model_track,
        "track2_per_heldout": per_heldout,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help=(
            "One or more experiment roots that contain postprocess_conformal_summary.json. "
            "Example: data/featured/lgbm_qr_track1_lb168 data/featured/lgbm_qr_track2_lb168"
        ),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <first_root>/__paper_results",
    )
    args = ap.parse_args()

    all_rows: List[Dict[str, Any]] = []
    loaded_from: List[str] = []

    for root_str in args.roots:
        root = Path(root_str).resolve()
        rows, src = load_batch_summary(root)
        all_rows.extend(rows)
        loaded_from.append(str(src))

    if not all_rows:
        raise RuntimeError("No rows loaded. Check that each root has postprocess_conformal_summary.json")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (Path(args.roots[0]).resolve() / "__paper_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) raw run-level table
    out_csv = out_dir / "conformal_runs.csv"
    _write_csv(out_csv, all_rows)

    # 2) aggregations (paper-friendly)
    agg = aggregate(all_rows)
    out_json = out_dir / "conformal_aggregates.json"
    _write_json(out_json, {"loaded_from": loaded_from, "n_runs": len(all_rows), **agg})

    print(f"[OK] Saved run table : {out_csv}")
    print(f"[OK] Saved aggregates: {out_json}")
    print(f"[OK] n_runs={len(all_rows)}")


if __name__ == "__main__":
    main()
