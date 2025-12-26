"""
run_deterministic_lgbm.py

Deterministic LightGBM experiment runner.

This file is the *runner* (experiment orchestrator):
- It does NOT implement data loading, feature engineering, splitting, or training.
- It calls the deterministic LGBM pipeline once per run.
- It repeats runs across multiple random seeds.
- It aggregates metrics (mean / std / min / max) across runs.

Typical usage (from project root):
    python -m src.experiments.run_deterministic_lgbm \
        --config configs/base.yaml \
        --exp_name deterministic_lgbm_track1 \
        --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.utils.config import load_config

# -----------------------------------------------------------------------------
# Import pipeline entrypoints
# -----------------------------------------------------------------------------
# Track 1: run_pipeline (temporal cut split)
# Track 2: run_pipeline_lofo (LOFO + inner time validation)
try:
    from src.pipelines.deterministic_lgbm_pipeline import (
        run_pipeline as _pipeline_track1,
        run_pipeline_lofo as _pipeline_track2,
    )
except ImportError as e:
    raise ImportError(
        "Cannot import pipeline entrypoints. Expected: "
        "`src.pipelines.deterministic_lgbm_pipeline.run_pipeline` and "
        "`src.pipelines.deterministic_lgbm_pipeline.run_pipeline_lofo`. "
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
    """Set global random seeds for reproducibility.

    Note:
    - Full LightGBM determinism also depends on model parameters
      (e.g., seed / bagging_seed) and thread settings.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------------------
# Main runner logic
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic LGBM experiment runner")
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

    # ------------------------------------------------------------------
    # Fallback: if load_config() drops keys (e.g., split=None), read YAML directly.
    # This avoids silent failures where the runner thinks it's Track1.
    # ------------------------------------------------------------------
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
        # Prefer new style: output.experiment_name
        if isinstance(cfg.get("output"), dict) and cfg["output"].get("experiment_name"):
            exp_name = str(cfg["output"]["experiment_name"])
        else:
            # Fallback to old style default
            exp_name = "deterministic_lgbm_track1"

    exp_root = project_root / "reports" / "experiments" / exp_name
    _ensure_dir(exp_root)

    run_rows: List[Dict[str, Any]] = []

    keep_zone_yaml = True
    if isinstance(cfg.get("features"), dict) and "keep_zone" in cfg["features"]:
        keep_zone_yaml = bool(cfg["features"]["keep_zone"])

    keep_zone = keep_zone_yaml
    if args.keep_zone_override is not None:
        keep_zone = bool(args.keep_zone_override)

    # ------------------------------------------------------------------
    # Read run inputs from YAML (runner owns config parsing)
    # ------------------------------------------------------------------
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}

    raw_split = cfg.get("split")
    if isinstance(raw_split, dict):
        split_cfg = raw_split
    elif isinstance(raw_split, str):
        # Allow shorthand YAML like: split: lofo_time_val
        split_cfg = {"type": raw_split}
    else:
        split_cfg = {}
    logger.info(f"Loaded split config: {split_cfg!r}")

    feat_cfg = cfg.get("features", {}) if isinstance(cfg.get("features"), dict) else {}
    output_cfg = cfg.get("output", {}) if isinstance(cfg.get("output"), dict) else {}

    data_path = data_cfg.get("combined_data_path") or data_cfg.get("input_path")
    # Resolve relative paths from the config file location
    cfg_dir = config_path.parent.resolve()
    data_path = Path(str(data_path))
    if not data_path.is_absolute():
        data_path = (cfg_dir / data_path).resolve()
    data_path = str(data_path)

    # Accept both old/new key styles
    target_col = str(data_cfg.get("target_col") or data_cfg.get("target") or "target")
    zone_col = str(data_cfg.get("zone_col") or data_cfg.get("group_col") or "zone_id")
    time_col = str(data_cfg.get("time_col") or "datetime")

    # Split protocol (support multiple key names for robustness)
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

    # Track 1 cuts (required only for temporal protocol)
    train_cut = split_cfg.get("train_cut")
    val_cut = split_cfg.get("val_cut")
    if not is_track2_lofo:
        if not train_cut or not val_cut:
            raise KeyError(
                "Config missing split cuts: split.train_cut and split.val_cut are required for Track1 temporal runs. "
                "If you intended Track2 LOFO, set split.type: lofo_time_val (or split.protocol: lofo_time_val)."
            )

    # Track 2 LOFO controls
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
    # Track 2: determine LOFO held-out groups
    # ------------------------------------------------------------------
    held_out_groups: List[Any] = []
    if is_track2_lofo:
        # Optional: user can pin specific held-out groups in YAML
        cfg_groups = split_cfg.get("held_out_groups")
        if isinstance(cfg_groups, list) and len(cfg_groups) > 0:
            held_out_groups = cfg_groups
        else:
            # Infer from data (read only the group column)
            logger.info(f"Inferring LOFO groups from data column: {group_col}")
            tmp = pd.read_csv(data_path, usecols=[group_col])
            held_out_groups = sorted(tmp[group_col].dropna().unique().tolist())

        if len(held_out_groups) == 0:
            raise ValueError("No held-out groups found for LOFO. Check split.group_col and the dataset.")

        logger.info(f"Track2 LOFO enabled | n_groups={len(held_out_groups)}")

    # Where pipeline artifacts will be saved
    pipeline_root = output_cfg.get("pipeline_root") or output_cfg.get("processed_output") or "data/featured"
    pipeline_root = Path(str(pipeline_root))
    if not pipeline_root.is_absolute():
        pipeline_root = (project_root / pipeline_root).resolve()

    # ------------------------------------------------------------------
    # Candidate grid (hyperparameter overrides)
    #
    # Expected YAML (recommended):
    # model:
    #   base_params: { ... }
    #   candidates:
    #     - name: baseline
    #       overrides: {}
    #     - name: lr003_leaves31
    #       overrides: {learning_rate: 0.03, num_leaves: 31}
    #
    # If candidates are missing, we run a single "baseline" candidate.
    # ------------------------------------------------------------------
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    base_params = model_cfg.get("base_params", {}) if isinstance(model_cfg.get("base_params"), dict) else {}

    candidates = model_cfg.get("candidates", [])
    if not isinstance(candidates, list) or len(candidates) == 0:
        candidates = [{"name": "baseline", "overrides": {}}]

    for cand in candidates:
        cand_name = str(cand.get("name", "candidate"))
        overrides = cand.get("overrides", {})
        if not isinstance(overrides, dict):
            overrides = {}

        # Merge base params + overrides for this candidate
        cand_params = dict(base_params)
        cand_params.update(overrides)

        for seed in args.seeds:
            _set_global_seed(seed)

            # Add LightGBM seeds for stochastic components
            # (Using multiple threads may still introduce small nondeterminism.)
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
                        f"Running Track2 LOFO | candidate={cand_name} | heldout={held_out} | seed={seed} | out_dir={pipeline_out_dir}"
                    )

                    result = _pipeline_track2(
                        data_path=data_path,
                        out_dir=pipeline_out_dir,
                        held_out_group=held_out,
                        val_days=val_days,
                        min_train=min_train,
                        lags=lags,
                        rolling_windows=rolling_windows,
                        target_col=target_col,
                        zone_col=zone_col,
                        time_col=time_col,
                        keep_zone=keep_zone,
                        seed=seed,
                        lgbm_params=cand_params_seeded,
                    )

                    # Pipeline returns a dataclass (preferred) or a dict (fallback)
                    if isinstance(result, dict):
                        metrics = result.get("metrics")
                        out_dir = result.get("out_dir")
                        model_path = result.get("model_path")
                        metrics_path = result.get("metrics_path")
                    else:
                        metrics = getattr(result, "metrics", None)
                        out_dir = getattr(result, "out_dir", None)
                        model_path = getattr(result, "model_path", None)
                        metrics_path = getattr(result, "metrics_path", None)

                    if not isinstance(metrics, dict):
                        raise TypeError("Pipeline did not return a valid metrics dict")

                    # For Track2, map inner_val -> val and outer_test -> test for consistent tables
                    val = metrics.get("inner_val", {})
                    test = metrics.get("outer_test", {})

                    run_rows.append(
                        {
                            "candidate": cand_name,
                            "held_out_group": held_out,
                            "seed": seed,
                            "val_rmse": val.get("rmse"),
                            "val_mae": val.get("mae"),
                            "val_r2": val.get("r2"),
                            "test_rmse": test.get("rmse"),
                            "test_mae": test.get("mae"),
                            "test_r2": test.get("r2"),
                            "pipeline_out_dir": out_dir,
                            "pipeline_model_path": model_path,
                            "pipeline_metrics_path": metrics_path,
                            "param_overrides": json.dumps(overrides, sort_keys=True),
                        }
                    )
            else:
                pipeline_out_dir = pipeline_root / exp_name / cand_name / f"seed_{seed}"
                logger.info(
                    f"Running Track1 temporal | candidate={cand_name} | seed={seed} | out_dir={pipeline_out_dir}"
                )

                result = _pipeline_track1(
                    data_path=data_path,
                    out_dir=pipeline_out_dir,
                    train_cut=train_cut,
                    val_cut=val_cut,
                    lags=lags,
                    rolling_windows=rolling_windows,
                    target_col=target_col,
                    zone_col=zone_col,
                    time_col=time_col,
                    keep_zone=keep_zone,
                    seed=seed,
                    lgbm_params=cand_params_seeded,
                )

                # Pipeline returns a dataclass (preferred) or a dict (fallback)
                if isinstance(result, dict):
                    metrics = result.get("metrics")
                    out_dir = result.get("out_dir")
                    model_path = result.get("model_path")
                    metrics_path = result.get("metrics_path")
                else:
                    metrics = getattr(result, "metrics", None)
                    out_dir = getattr(result, "out_dir", None)
                    model_path = getattr(result, "model_path", None)
                    metrics_path = getattr(result, "metrics_path", None)

                if not isinstance(metrics, dict):
                    raise TypeError("Pipeline did not return a valid metrics dict")

                val = metrics.get("val", {})
                test = metrics.get("test", {})

                run_rows.append(
                    {
                        "candidate": cand_name,
                        "seed": seed,
                        "val_rmse": val.get("rmse"),
                        "val_mae": val.get("mae"),
                        "val_r2": val.get("r2"),
                        "test_rmse": test.get("rmse"),
                        "test_mae": test.get("mae"),
                        "test_r2": test.get("r2"),
                        "pipeline_out_dir": out_dir,
                        "pipeline_model_path": model_path,
                        "pipeline_metrics_path": metrics_path,
                        "param_overrides": json.dumps(overrides, sort_keys=True),
                    }
                )

    # ----------------------------------------------------------------------
    # Aggregate metrics across seeds
    # ----------------------------------------------------------------------
    val_rmse = [float(r["val_rmse"]) for r in run_rows if r["val_rmse"] is not None]
    test_rmse = [float(r["test_rmse"]) for r in run_rows if r["test_rmse"] is not None]
    val_mae = [float(r["val_mae"]) for r in run_rows if r["val_mae"] is not None]
    test_mae = [float(r["test_mae"]) for r in run_rows if r["test_mae"] is not None]
    val_r2 = [float(r["val_r2"]) for r in run_rows if r["val_r2"] is not None]
    test_r2 = [float(r["test_r2"]) for r in run_rows if r["test_r2"] is not None]

    summary: Dict[str, Any] = {
        "experiment": exp_name,
        "protocol": "track2_lofo" if is_track2_lofo else "track1_temporal",
        "config": str(config_path),
        "keep_zone": keep_zone,
        "seeds": list(args.seeds),
        "n_runs": len(run_rows),
        "n_groups": len(held_out_groups) if is_track2_lofo else None,
        "val": {
            "rmse": _aggregate(val_rmse),
            "mae": _aggregate(val_mae),
            "r2": _aggregate(val_r2),
        },
        "test": {
            "rmse": _aggregate(test_rmse),
            "mae": _aggregate(test_mae),
            "r2": _aggregate(test_r2),
        },
        "runs": run_rows,
    }

    # ----------------------------------------------------------------------
    # Aggregate by candidate
    # ----------------------------------------------------------------------
    by_candidate: Dict[str, Any] = {}
    for cand_name in sorted({r["candidate"] for r in run_rows}):
        rows = [r for r in run_rows if r["candidate"] == cand_name]
        by_candidate[cand_name] = {
            "n_runs": len(rows),
            "val": {
                "rmse": _aggregate([float(r["val_rmse"]) for r in rows if r["val_rmse"] is not None]),
                "mae": _aggregate([float(r["val_mae"]) for r in rows if r["val_mae"] is not None]),
                "r2": _aggregate([float(r["val_r2"]) for r in rows if r["val_r2"] is not None]),
            },
            "test": {
                "rmse": _aggregate([float(r["test_rmse"]) for r in rows if r["test_rmse"] is not None]),
                "mae": _aggregate([float(r["test_mae"]) for r in rows if r["test_mae"] is not None]),
                "r2": _aggregate([float(r["test_r2"]) for r in rows if r["test_r2"] is not None]),
            },
        }

    summary["by_candidate"] = by_candidate

    _write_json(exp_root / "summary.json", summary)
    _write_csv(exp_root / "runs.csv", run_rows)

    logger.info(f"Saved summary to: {exp_root / 'summary.json'}")
    logger.info(f"Saved per-run table to: {exp_root / 'runs.csv'}")
    if is_track2_lofo:
        logger.info(
            f"INNER VAL RMSE mean±std: {summary['val']['rmse']['mean']:.4f} ± {summary['val']['rmse']['std']:.4f}"
        )
        logger.info(
            f"OUTER TEST RMSE mean±std: {summary['test']['rmse']['mean']:.4f} ± {summary['test']['rmse']['std']:.4f}"
        )
    else:
        logger.info(
            f"VAL RMSE mean±std: {summary['val']['rmse']['mean']:.4f} ± {summary['val']['rmse']['std']:.4f}"
        )
        logger.info(
            f"TEST RMSE mean±std: {summary['test']['rmse']['mean']:.4f} ± {summary['test']['rmse']['std']:.4f}"
        )


if __name__ == "__main__":
    main()