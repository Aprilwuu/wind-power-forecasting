"""
run_quantile_lgbm.py

Quantile (probabilistic) LightGBM experiment runner.


Typical usage (from project root):
    python -m src.experiments.run_quantile_lgbm \
        --config configs/base.yaml \
        --exp_name quantile_lgbm_track1 \
        --quantiles 0.3 0.5 0.7 \
        --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


from src.utils.config import load_config
from src.data.load import load_raw_data

# -----------------------------------------------------------------------------
# Import pipeline entrypoints
# -----------------------------------------------------------------------------
# Track 1: run_pipeline_quantile (temporal cut split)
# Track 2: run_pipeline_quantile_lofo (LOFO + inner time validation)
try:
    from src.pipelines.quantile_lgbm_pipeline import (
        run_pipeline_quantile as _pipeline_track1,
        run_pipeline_quantile_lofo as _pipeline_track2,
    )
except ImportError as e:
    raise ImportError(
        "Cannot import quantile pipeline entrypoints. Expected: "
        "`src.pipelines.quantile_lgbm_pipeline.run_pipeline_quantile` and "
        "`src.pipelines.quantile_lgbm_pipeline.run_pipeline_quantile_lofo`. "
        "Please check the file path and function names."
    ) from e

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    _ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def _read_quantiles_from_cfg(cfg: Dict[str, Any]) -> Optional[List[float]]:
    """
    Try a few common locations in YAML:
    - probabilistic.quantiles
    - model.quantiles
    - features.quantiles
    """
    for path in [("probabilistic", "quantiles"), ("model", "quantiles"), ("features", "quantiles")]:
        a, b = path
        if isinstance(cfg.get(a), dict) and cfg[a].get(b) is not None:
            q = cfg[a][b]
            if isinstance(q, list) and all(isinstance(x, (int, float)) for x in q):
                return [float(x) for x in q]
    return None


def _coerce_float_list(xs: Any, name: str) -> List[float]:
    if not isinstance(xs, list) or len(xs) == 0:
        raise TypeError(f"{name} must be a non-empty list of floats")
    out: List[float] = []
    for x in xs:
        if not isinstance(x, (int, float)):
            raise TypeError(f"{name} must be a list of floats. Got element={x!r}")
        out.append(float(x))
    return out


# -----------------------------------------------------------------------------
# Main runner logic
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Quantile LGBM experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name (used for output directory). If omitted, read from YAML (output.experiment_name).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[42, 43, 44, 45, 46],
        help="Random seeds for repeated runs",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="*",
        default=None,
        help="Quantiles to train, e.g. --quantiles 0.3 0.5 0.7. "
             "If omitted, try reading from YAML (probabilistic.quantiles / model.quantiles).",
    )
    parser.add_argument(
        "--keep_zone",
        action="store_true",
        dest="keep_zone_override",
        help="Override: keep zone_id as a feature (otherwise read from YAML)",
    )
    parser.add_argument(
        "--no_keep_zone",
        action="store_false",
        dest="keep_zone_override",
        help="Override: drop zone_id from features (otherwise read from YAML)",
    )
    parser.set_defaults(keep_zone_override=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(str(config_path))

    # fallback read yaml if load_config drops keys
    cfg = cfg or {}
    if cfg.get("split") is None:
        logger.info(
            "load_config() returned cfg without 'split'. Falling back to yaml.safe_load on the provided file."
        )
        with config_path.open("r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
        if not isinstance(raw_cfg, dict):
            raise TypeError("YAML config did not parse into a dict")
        cfg.update(raw_cfg)
        logger.info(f"After fallback load, cfg keys: {sorted(cfg.keys())}")

    project_root = Path(__file__).resolve().parents[2]

    exp_name = args.exp_name
    if exp_name is None:
        if isinstance(cfg.get("output"), dict) and cfg["output"].get("experiment_name"):
            exp_name = str(cfg["output"]["experiment_name"])
        else:
            exp_name = "quantile_lgbm_track1"

    exp_root = project_root / "reports" / "experiments" / exp_name
    _ensure_dir(exp_root)

    # ------------------------------------------------------------------
    # quantiles
    # ------------------------------------------------------------------
    quantiles: Optional[List[float]] = None
    if args.quantiles is not None and len(args.quantiles) > 0:
        quantiles = [float(x) for x in args.quantiles]
    else:
        q_from_cfg = _read_quantiles_from_cfg(cfg)
        if q_from_cfg is not None:
            quantiles = q_from_cfg

    if quantiles is None:
        # default fallback
        quantiles = [0.1, 0.5, 0.9]

    quantiles = _coerce_float_list(quantiles, "quantiles")

    # ------------------------------------------------------------------
    # keep_zone
    # ------------------------------------------------------------------
    keep_zone_yaml = True
    if isinstance(cfg.get("features"), dict) and "keep_zone" in cfg["features"]:
        keep_zone_yaml = bool(cfg["features"]["keep_zone"])
    keep_zone = keep_zone_yaml
    if args.keep_zone_override is not None:
        keep_zone = bool(args.keep_zone_override)

    # ------------------------------------------------------------------
    # Read run inputs from YAML
    # ------------------------------------------------------------------
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}

    raw_split = cfg.get("split")
    if isinstance(raw_split, dict):
        split_cfg = raw_split
    elif isinstance(raw_split, str):
        split_cfg = {"type": raw_split}
    else:
        split_cfg = {}
    logger.info(f"Loaded split config: {split_cfg!r}")

    feat_cfg = cfg.get("features", {}) if isinstance(cfg.get("features"), dict) else {}
    output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), dict) else {}

    data_path_raw = data_cfg.get("combined_data_path") or data_cfg.get("input_path")
    if not data_path_raw:
        raise KeyError(
            "Missing data path in config. Set data.combined_data_path (or data.input_path)."
        )

    # IMPORTANT: resolve relative paths against the PROJECT ROOT, not the configs/ directory.
    # This makes runs stable no matter where you launch the command from.
    data_path = Path(str(data_path_raw))
    if not data_path.is_absolute():
        # Prefer project-root-relative (recommended)
        cand1 = (project_root / data_path).resolve()
        # Backward-compat: some older configs were written relative to configs/
        cand2 = (config_path.parent.resolve() / data_path).resolve()
        if cand1.exists():
            data_path = cand1
        elif cand2.exists():
            data_path = cand2
        else:
            raise FileNotFoundError(
                "Data file not found. Tried:\n"
                f"  1) {cand1}\n"
                f"  2) {cand2}\n"
                "Fix by setting data.combined_data_path relative to the project root, e.g. 'data/processed/...csv'."
            )
    else:
        data_path = data_path.resolve()

    data_path = str(data_path)

    target_col = str(data_cfg.get("target_col") or data_cfg.get("target") or "target")
    zone_col = str(data_cfg.get("zone_col") or data_cfg.get("group_col") or "zone_id")
    time_col = str(data_cfg.get("time_col") or "datetime")

    split_type_raw = (
        split_cfg.get("type")
        or split_cfg.get("protocol")
        or split_cfg.get("split_type")
        or "temporal"
    )
    split_type = str(split_type_raw).strip().strip('"').strip("'")
    split_type_norm = split_type.lower()
    is_track2_lofo = split_type_norm in {
        "lofo_time_val",
        "lofo",
        "leave_one_farm_out",
        "leave_one_site_out",
        "leave_one_group_out",
    }
    logger.info(f"Split type resolved: raw={split_type_raw!r} -> '{split_type}'")

    # Track1 cuts
    train_cut = split_cfg.get("train_cut")
    val_cut = split_cfg.get("val_cut")
    if not is_track2_lofo:
        if not train_cut or not val_cut:
            raise KeyError(
                "Config missing split cuts: split.train_cut and split.val_cut are required for Track1 temporal runs. "
                "If you intended Track2 LOFO, set split.type: lofo_time_val (or split.protocol: lofo_time_val)."
            )

    # Track2 controls
    val_days = int(split_cfg.get("val_days", 30))
    min_train = int(split_cfg.get("min_train", 1000))
    group_col = str(split_cfg.get("group_col") or zone_col)

    lags = feat_cfg.get("lags", [1, 2, 3, 6, 12, 24])
    rolling_windows = feat_cfg.get("rolling_windows", [6, 24])

    if not isinstance(lags, list) or not all(isinstance(x, int) for x in lags):
        raise TypeError("features.lags must be a list[int]")
    if not isinstance(rolling_windows, list) or not all(isinstance(x, int) for x in rolling_windows):
        raise TypeError("features.rolling_windows must be a list[int]")

    # ------------------------------------------------------------------
    # Track2: determine held-out groups
    # ------------------------------------------------------------------
    held_out_groups: List[Any] = []
    if is_track2_lofo:
        cfg_groups = split_cfg.get("held_out_groups")
        if isinstance(cfg_groups, list) and len(cfg_groups) > 0:
            held_out_groups = cfg_groups
        else:
            logger.info(
                f"Inferring LOFO groups from data column: {group_col} (will fallback to load_raw_data normalization if needed)"
            )
            try:
                # Fast path: if the CSV already contains the normalized column name (e.g. 'zone_id'), this is cheap.
                tmp = pd.read_csv(data_path, usecols=[group_col])
                held_out_groups = sorted(tmp[group_col].dropna().unique().tolist())
            except ValueError:
                # Fallback: the raw dataset header may be uppercase (e.g. 'ZONEID').
                # load_raw_data() normalizes names (ZONEID->zone_id, TIMESTAMP->datetime, TARGETVAR->target).
                df_norm = load_raw_data(str(data_path))
                if group_col not in df_norm.columns:
                    raise ValueError(
                        f"Requested group_col not found after load_raw_data normalization: wanted={group_col!r}. "
                        f"Available columns={list(df_norm.columns)!r}"
                    )
                held_out_groups = sorted(df_norm[group_col].dropna().unique().tolist())

        if len(held_out_groups) == 0:
            raise ValueError("No held-out groups found for LOFO. Check split.group_col and the dataset.")

        logger.info(f"Track2 LOFO enabled | n_groups={len(held_out_groups)}")

    # where pipeline artifacts will be saved
    pipeline_root = output_cfg.get("pipeline_root") or output_cfg.get("processed_output") or "data/featured"
    pipeline_root = Path(str(pipeline_root))
    if not pipeline_root.is_absolute():
        pipeline_root = (project_root / pipeline_root).resolve()

    # ------------------------------------------------------------------
    # Candidate grid (same pattern as deterministic runner)
    # ------------------------------------------------------------------
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    base_params = model_cfg.get("base_params", {}) if isinstance(model_cfg.get("base_params"), dict) else {}

    candidates = model_cfg.get("candidates", [])
    if not isinstance(candidates, list) or len(candidates) == 0:
        candidates = [{"name": "baseline", "overrides": {}}]

    run_rows: List[Dict[str, Any]] = []

    for cand in candidates:
        cand_name = str(cand.get("name", "candidate"))
        overrides = cand.get("overrides", {})
        if not isinstance(overrides, dict):
            overrides = {}

        cand_params = dict(base_params)
        cand_params.update(overrides)

        for seed in args.seeds:
            _set_global_seed(seed)

            cand_params_seeded = dict(cand_params)
            cand_params_seeded.setdefault("seed", seed)
            cand_params_seeded.setdefault("data_random_seed", seed)
            cand_params_seeded.setdefault("feature_fraction_seed", seed)
            cand_params_seeded.setdefault("bagging_seed", seed)

            if is_track2_lofo:
                for held_out in held_out_groups:
                    pipeline_out_dir = (
                        pipeline_root / exp_name / cand_name / f"heldout_{held_out}" / f"seed_{seed}"
                    )
                    logger.info(
                        f"Running Track2 LOFO (quantile) | candidate={cand_name} | heldout={held_out} | "
                        f"seed={seed} | quantiles={quantiles} | out_dir={pipeline_out_dir}"
                    )

                    result = _pipeline_track2(
                        data_path=data_path,
                        out_dir=pipeline_out_dir,
                        held_out_group=held_out,
                        val_days=val_days,
                        min_train=min_train,
                        lags=lags,
                        rolling_windows=rolling_windows,
                        quantiles=quantiles,
                        target_col=target_col,
                        zone_col=zone_col,
                        group_col=group_col,
                        time_col=time_col,
                        keep_zone=keep_zone,
                        seed=seed,
                        lgbm_params=cand_params_seeded,
                    )

                    if isinstance(result, dict):
                        metrics = result.get("metrics")
                        out_dir = result.get("out_dir")
                        model_paths = result.get("model_paths")
                        metrics_path = result.get("metrics_path")
                    else:
                        metrics = getattr(result, "metrics", None)
                        out_dir = getattr(result, "out_dir", None)
                        model_paths = getattr(result, "model_paths", None)
                        metrics_path = getattr(result, "metrics_path", None)

                    if not isinstance(metrics, dict):
                        raise TypeError("Pipeline did not return a valid metrics dict")

                    val = metrics.get("inner_val", {})
                    test = metrics.get("outer_test", {})
                    # optional: coverage block (if pipeline outputs it)
                    cov = metrics.get("coverage") or metrics.get("intervals") or {}
                    cov_json = json.dumps(cov, sort_keys=True) if isinstance(cov, dict) else None

                    run_rows.append(
                        {
                            "candidate": cand_name,
                            "held_out_group": held_out,
                            "seed": seed,
                            "quantiles": json.dumps(quantiles),
                            "val_pinball_mean": val.get("pinball_mean"),
                            "val_crossing_rate": val.get("crossing_rate"),
                            "test_pinball_mean": test.get("pinball_mean"),
                            "test_crossing_rate": test.get("crossing_rate"),
                            "coverage": cov_json,
                            "pipeline_out_dir": out_dir,
                            "pipeline_model_paths": json.dumps(model_paths, sort_keys=True) if model_paths else None,
                            "pipeline_metrics_path": metrics_path,
                            "param_overrides": json.dumps(overrides, sort_keys=True),
                        }
                    )
            else:
                pipeline_out_dir = pipeline_root / exp_name / cand_name / f"seed_{seed}"
                logger.info(
                    f"Running Track1 temporal (quantile) | candidate={cand_name} | seed={seed} | "
                    f"quantiles={quantiles} | out_dir={pipeline_out_dir}"
                )

                result = _pipeline_track1(
                    data_path=data_path,
                    out_dir=pipeline_out_dir,
                    train_cut=train_cut,
                    val_cut=val_cut,
                    lags=lags,
                    rolling_windows=rolling_windows,
                    quantiles=quantiles,
                    target_col=target_col,
                    zone_col=zone_col,
                    time_col=time_col,
                    keep_zone=keep_zone,
                    seed=seed,
                    lgbm_params=cand_params_seeded,
                )

                if isinstance(result, dict):
                    metrics = result.get("metrics")
                    out_dir = result.get("out_dir")
                    model_paths = result.get("model_paths")
                    metrics_path = result.get("metrics_path")
                else:
                    metrics = getattr(result, "metrics", None)
                    out_dir = getattr(result, "out_dir", None)
                    model_paths = getattr(result, "model_paths", None)
                    metrics_path = getattr(result, "metrics_path", None)

                if not isinstance(metrics, dict):
                    raise TypeError("Pipeline did not return a valid metrics dict")

                val = metrics.get("val", {})
                test = metrics.get("test", {})

                # optional: coverage block (if pipeline outputs it)
                cov = metrics.get("coverage") or metrics.get("intervals") or {}
                cov_json = json.dumps(cov, sort_keys=True) if isinstance(cov, dict) else None

                run_rows.append(
                    {
                        "candidate": cand_name,
                        "seed": seed,
                        "quantiles": json.dumps(quantiles),
                        "val_pinball_mean": val.get("pinball_mean"),
                        "val_crossing_rate": val.get("crossing_rate"),
                        "test_pinball_mean": test.get("pinball_mean"),
                        "test_crossing_rate": test.get("crossing_rate"),
                        "coverage": cov_json,
                        "pipeline_out_dir": out_dir,
                        "pipeline_model_paths": json.dumps(model_paths, sort_keys=True) if model_paths else None,
                        "pipeline_metrics_path": metrics_path,
                        "param_overrides": json.dumps(overrides, sort_keys=True),
                    }
                )

    # ----------------------------------------------------------------------
    # Aggregate metrics across runs
    # ----------------------------------------------------------------------
    val_pinball_mean = [float(r["val_pinball_mean"]) for r in run_rows if r["val_pinball_mean"] is not None]
    test_pinball_mean = [float(r["test_pinball_mean"]) for r in run_rows if r["test_pinball_mean"] is not None]
    val_cross = [float(r["val_crossing_rate"]) for r in run_rows if r["val_crossing_rate"] is not None]
    test_cross = [float(r["test_crossing_rate"]) for r in run_rows if r["test_crossing_rate"] is not None]

    def _extract_default_cov_width(cov_obj: Any) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Return (val_cov, val_width, test_cov, test_width) from a coverage/intervals dict.

        Supports Track1 keys (val/test) and Track2 keys (inner_val/outer_test).
        """
        if not isinstance(cov_obj, dict) or not cov_obj:
            return None, None, None, None
        d = cov_obj.get("default")
        if not isinstance(d, dict):
            # fallback: pick first interval if 'default' missing
            try:
                d = next(iter(cov_obj.values()))
            except Exception:
                return None, None, None, None
            if not isinstance(d, dict):
                return None, None, None, None

        val_sec = d.get("val") or d.get("inner_val") or {}
        test_sec = d.get("test") or d.get("outer_test") or {}

        val_cov = val_sec.get("coverage") if isinstance(val_sec, dict) else None
        val_w = val_sec.get("avg_width") if isinstance(val_sec, dict) else None
        test_cov = test_sec.get("coverage") if isinstance(test_sec, dict) else None
        test_w = test_sec.get("avg_width") if isinstance(test_sec, dict) else None

        return (
            float(val_cov) if val_cov is not None else None,
            float(val_w) if val_w is not None else None,
            float(test_cov) if test_cov is not None else None,
            float(test_w) if test_w is not None else None,
        )

    val_default_cov: List[float] = []
    val_default_w: List[float] = []
    test_default_cov: List[float] = []
    test_default_w: List[float] = []

    for r in run_rows:
        cov_str = r.get("coverage")
        if not cov_str:
            continue
        try:
            cov_obj = json.loads(cov_str) if isinstance(cov_str, str) else cov_str
        except Exception:
            continue
        v_cov, v_w, t_cov, t_w = _extract_default_cov_width(cov_obj)
        if v_cov is not None:
            val_default_cov.append(v_cov)
        if v_w is not None:
            val_default_w.append(v_w)
        if t_cov is not None:
            test_default_cov.append(t_cov)
        if t_w is not None:
            test_default_w.append(t_w)

    summary: Dict[str, Any] = {
        "experiment": exp_name,
        "protocol": "track2_lofo" if is_track2_lofo else "track1_temporal",
        "config": str(config_path),
        "keep_zone": keep_zone,
        "quantiles": list(quantiles),
        "seeds": list(args.seeds),
        "n_runs": len(run_rows),
        "n_groups": len(held_out_groups) if is_track2_lofo else None,
        "val": {
            "pinball_mean": _aggregate(val_pinball_mean),
            "crossing_rate": _aggregate(val_cross),
            **({"default_coverage": _aggregate(val_default_cov)} if len(val_default_cov) > 0 else {}),
            **({"default_avg_width": _aggregate(val_default_w)} if len(val_default_w) > 0 else {}),
        },
        "test": {
            "pinball_mean": _aggregate(test_pinball_mean),
            "crossing_rate": _aggregate(test_cross),
            **({"default_coverage": _aggregate(test_default_cov)} if len(test_default_cov) > 0 else {}),
            **({"default_avg_width": _aggregate(test_default_w)} if len(test_default_w) > 0 else {}),
        },
        "runs": run_rows,
    }

    # Aggregate by candidate
    by_candidate: Dict[str, Any] = {}
    for cand_name in sorted({r["candidate"] for r in run_rows}):
        rows = [r for r in run_rows if r["candidate"] == cand_name]
        c_val_cov: List[float] = []
        c_val_w: List[float] = []
        c_test_cov: List[float] = []
        c_test_w: List[float] = []
        for rr in rows:
            cov_str = rr.get("coverage")
            if not cov_str:
                continue
            try:
                cov_obj = json.loads(cov_str) if isinstance(cov_str, str) else cov_str
            except Exception:
                continue
            v_cov, v_w, t_cov, t_w = _extract_default_cov_width(cov_obj)
            if v_cov is not None:
                c_val_cov.append(v_cov)
            if v_w is not None:
                c_val_w.append(v_w)
            if t_cov is not None:
                c_test_cov.append(t_cov)
            if t_w is not None:
                c_test_w.append(t_w)
        by_candidate[cand_name] = {
            "n_runs": len(rows),
            "val": {
                "pinball_mean": _aggregate([float(r["val_pinball_mean"]) for r in rows if r["val_pinball_mean"] is not None]),
                "crossing_rate": _aggregate([float(r["val_crossing_rate"]) for r in rows if r["val_crossing_rate"] is not None]),
                **({"default_coverage": _aggregate(c_val_cov)} if len(c_val_cov) > 0 else {}),
                **({"default_avg_width": _aggregate(c_val_w)} if len(c_val_w) > 0 else {}),
            },
            "test": {
                "pinball_mean": _aggregate([float(r["test_pinball_mean"]) for r in rows if r["test_pinball_mean"] is not None]),
                "crossing_rate": _aggregate([float(r["test_crossing_rate"]) for r in rows if r["test_crossing_rate"] is not None]),
                **({"default_coverage": _aggregate(c_test_cov)} if len(c_test_cov) > 0 else {}),
                **({"default_avg_width": _aggregate(c_test_w)} if len(c_test_w) > 0 else {}),
            },
        }

    summary["by_candidate"] = by_candidate

    _write_json(exp_root / "summary.json", summary)
    _write_csv(exp_root / "runs.csv", run_rows)

    logger.info(f"Saved summary to: {exp_root / 'summary.json'}")
    logger.info(f"Saved per-run table to: {exp_root / 'runs.csv'}")

    logger.info(
        f"VAL pinball_mean mean±std: {summary['val']['pinball_mean']['mean']:.6f} ± {summary['val']['pinball_mean']['std']:.6f}"
    )
    logger.info(
        f"TEST pinball_mean mean±std: {summary['test']['pinball_mean']['mean']:.6f} ± {summary['test']['pinball_mean']['std']:.6f}"
    )


if __name__ == "__main__":
    main()