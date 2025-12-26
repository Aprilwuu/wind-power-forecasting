# src/pipelines/deterministic_lgbm_pipeline.py
"""Deterministic LightGBM pipeline (one single run).

This module is intentionally "config-agnostic":
- It does NOT read YAML.
- It does NOT decide parameter grids / candidates.

A runner (experiment script) should:
- Load YAML configs
- Build the final LightGBM params for THIS run (base_params + candidate override)
- Loop over seeds / candidates
- Call `run_pipeline(...)` once per run

Pipeline responsibilities:
- Load + clean data
- Build ML features (lags + rolling stats)
- Split (Track 1 temporal: cut-based) OR Track 2 LOFO (leave-one-site-out + inner time val)
- Train LightGBM
- Evaluate on val/test
- Save artifacts (model, metrics, config snapshot)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from src.data.load import load_raw_data
from src.data.make_dataset import make_dataset
from src.data.split import train_valid_split, lofo_time_val_split
from src.features.build_features import make_ml_features, select_ml_xy
from src.metrics.deterministic import evaluate_mae, evaluate_r2, evaluate_rmse
from src.models.deterministic.lgbm import create_lgbm_model

logger = logging.getLogger(__name__)


@dataclass
class DeterministicLGBMResult:
    out_dir: str
    model_path: str
    metrics_path: str
    config_snapshot_path: str
    metrics: Dict[str, Any]


def _ensure_dir(p: str | Path) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def run_pipeline(
    *,
    data_path: str | Path,
    out_dir: str | Path,
    train_cut: str,
    val_cut: str,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
    target_col: str = "target",
    zone_col: str = "zone_id",
    time_col: str = "datetime",
    keep_zone: bool = True,
    seed: int | None = None,
    lgbm_params: Dict[str, Any] | None = None,
) -> DeterministicLGBMResult:
    """Run one deterministic LightGBM training/evaluation pipeline.

    Args:
        data_path: Input CSV path (processed combined dataset).
        out_dir: Output directory for this single run.
        train_cut: Train end cut timestamp (e.g. "2013-09-01").
        val_cut: Validation end cut timestamp (e.g. "2013-11-01").
        lags: Target lag steps for ML features.
        rolling_windows: Target rolling window sizes for ML features.
        target_col: Target column name.
        zone_col: Zone/site column name.
        time_col: Timestamp column name.
        keep_zone: Whether to keep zone_id as a feature.
        seed: Random seed for this run (passed into LGBM params).
        lgbm_params: Final LightGBM params for THIS run (runner must provide merged params).

    Returns:
        DeterministicLGBMResult with artifact paths and metrics.
    """
    if not isinstance(lgbm_params, dict) or len(lgbm_params) == 0:
        raise ValueError(
            "lgbm_params must be a non-empty dict. "
            "The runner should provide merged params (base_params + override)."
        )

    out_dir_p = _ensure_dir(out_dir).resolve()

    # ------------------------------------------------------------------
    # Snapshot runtime inputs (config-agnostic)
    # ------------------------------------------------------------------
    config_snapshot_path = out_dir_p / "config_snapshot.json"
    snapshot = {
        "runtime": {
            "data_path": str(Path(data_path).resolve()),
            "out_dir": str(out_dir_p),
            "train_cut": train_cut,
            "val_cut": val_cut,
            "lags": list(lags),
            "rolling_windows": list(rolling_windows),
            "target_col": target_col,
            "zone_col": zone_col,
            "time_col": time_col,
            "keep_zone": bool(keep_zone),
            "seed": seed,
            "lgbm_params": dict(lgbm_params),
        }
    }
    config_snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Load -> clean -> features
    # ------------------------------------------------------------------
    logger.info(f"Load: {Path(data_path).resolve()}")
    df = load_raw_data(str(data_path))

    logger.info("Clean + impute (fit rules are deterministic here)")
    df = make_dataset(df, target_col=target_col, zone_col=zone_col, time_col=time_col)

    logger.info(f"Build ML features: lags={list(lags)}, rolling={list(rolling_windows)}")
    df = make_ml_features(
        df,
        zone_col=zone_col,
        time_col=time_col,
        target_col=target_col,
        lags=lags,
        roll_windows=rolling_windows,
    )

    # Drop rows with NaNs introduced by lag/rolling features
    df = df.dropna().reset_index(drop=True)

    # ------------------------------------------------------------------
    # Split (Track 1 temporal: cut-based)
    # ------------------------------------------------------------------
    logger.info(f"Split Track1 by cuts: train_cut={train_cut}, val_cut={val_cut}")
    df_train, df_val, df_test = train_valid_split(
        df,
        time_col=time_col,
        zone_col=zone_col,
        train_cut=train_cut,
        val_cut=val_cut,
    )

    # ------------------------------------------------------------------
    # Train + evaluate
    # ------------------------------------------------------------------
    logger.info("Prepare X/y")
    X_train, y_train = select_ml_xy(
        df_train,
        target_col=target_col,
        drop_cols=(time_col,),
        keep_zone=keep_zone,
    )
    X_val, y_val = select_ml_xy(
        df_val,
        target_col=target_col,
        drop_cols=(time_col,),
        keep_zone=keep_zone,
    )
    X_test, y_test = select_ml_xy(
        df_test,
        target_col=target_col,
        drop_cols=(time_col,),
        keep_zone=keep_zone,
    )

    logger.info("Train LGBM")
    model = create_lgbm_model(params=lgbm_params, seed=seed)
    model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)

    logger.info("Predict + evaluate")
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    metrics: Dict[str, Any] = {
        "track": "track1_temporal",
        "split": {"train_cut": train_cut, "val_cut": val_cut},
        "features": {"lags": list(lags), "rolling_windows": list(rolling_windows)},
        "keep_zone": bool(keep_zone),
        "model": {"lgbm_params": dict(lgbm_params), "seed": seed},
        "val": {
            "rmse": evaluate_rmse(y_val, pred_val),
            "mae": evaluate_mae(y_val, pred_val),
            "r2": evaluate_r2(y_val, pred_val),
            "n": int(len(y_val)),
        },
        "test": {
            "rmse": evaluate_rmse(y_test, pred_test),
            "mae": evaluate_mae(y_test, pred_test),
            "r2": evaluate_r2(y_test, pred_test),
            "n": int(len(y_test)),
        },
        "n_rows": {
            "train": int(len(df_train)),
            "val": int(len(df_val)),
            "test": int(len(df_test)),
        },
    }

    metrics_path = out_dir_p / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model_path = out_dir_p / "model.txt"
    model.save(str(model_path))

    logger.info(f"Saved to {out_dir_p}")
    logger.info(f"VAL RMSE={metrics['val']['rmse']:.4f} | TEST RMSE={metrics['test']['rmse']:.4f}")

    return DeterministicLGBMResult(
        out_dir=str(out_dir_p),
        model_path=str(model_path),
        metrics_path=str(metrics_path),
        config_snapshot_path=str(config_snapshot_path),
        metrics=metrics,
    )


def run_pipeline_lofo(
    *,
    data_path: str | Path,
    out_dir: str | Path,
    held_out_group: int | str,
    val_days: int,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
    target_col: str = "target",
    zone_col: str = "zone_id",
    group_col: Optional[str] = None,
    time_col: str = "datetime",
    keep_zone: bool = True,
    seed: int | None = None,
    lgbm_params: Dict[str, Any] | None = None,
    min_train: int = 1000,
) -> DeterministicLGBMResult:
    """Run one deterministic LightGBM pipeline under Track 2 LOFO.

    Track 2 protocol (as used in your `lofo_time_val_split` helper):
      - OUTER TEST: all rows where zone_col == held_out_group
      - OUTER TRAIN: all remaining rows
      - INNER VAL (for early stopping): last `val_days` of OUTER TRAIN by time

    We train on INNER TRAIN, early-stop on INNER VAL, and report metrics on:
      - inner_val  (sanity check)
      - outer_test (the real Track2 generalization score)

    Args:
        data_path: Input CSV path (processed combined dataset).
        out_dir: Output directory for this single run.
        held_out_group: The zone/site to leave out as OUTER TEST.
        val_days: Number of days for the INNER time validation split.
        lags: Target lag steps for ML features.
        rolling_windows: Target rolling window sizes for ML features.
        target_col: Target column name.
        zone_col: Zone/site column name.
        time_col: Timestamp column name.
        keep_zone: Whether to keep zone_id as a feature.
        seed: Random seed for this run (passed into LGBM params).
        lgbm_params: Final LightGBM params for THIS run (runner must provide merged params).
        min_train: Guardrail for minimum inner-train size.

    Returns:
        DeterministicLGBMResult with artifact paths and metrics.
    """
    if not isinstance(lgbm_params, dict) or len(lgbm_params) == 0:
        raise ValueError(
            "lgbm_params must be a non-empty dict. "
            "The runner should provide merged params (base_params + override)."
        )

    out_dir_p = _ensure_dir(out_dir).resolve()

    # ------------------------------------------------------------------
    # Snapshot runtime inputs (config-agnostic)
    # ------------------------------------------------------------------
    config_snapshot_path = out_dir_p / "config_snapshot.json"
    snapshot = {
        "runtime": {
            "data_path": str(Path(data_path).resolve()),
            "out_dir": str(out_dir_p),
            "protocol": "track2_lofo_time_val",
            "held_out_group": held_out_group,
            "val_days": int(val_days),
            "lags": list(lags),
            "rolling_windows": list(rolling_windows),
            "target_col": target_col,
            "zone_col": zone_col,
            "group_col": (group_col or zone_col),
            "time_col": time_col,
            "keep_zone": bool(keep_zone),
            "seed": seed,
            "min_train": int(min_train),
            "lgbm_params": dict(lgbm_params),
        }
    }
    config_snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Load -> clean -> features
    # ------------------------------------------------------------------
    logger.info(f"Load: {Path(data_path).resolve()}")
    df = load_raw_data(str(data_path))

    logger.info("Clean + impute (fit rules are deterministic here)")
    df = make_dataset(df, target_col=target_col, zone_col=zone_col, time_col=time_col)

    logger.info(f"Build ML features: lags={list(lags)}, rolling={list(rolling_windows)}")
    df = make_ml_features(
        df,
        zone_col=zone_col,
        time_col=time_col,
        target_col=target_col,
        lags=lags,
        roll_windows=rolling_windows,
    )

    # Drop rows with NaNs introduced by lag/rolling features
    df = df.dropna().reset_index(drop=True)

    # ------------------------------------------------------------------
    # Split (Track 2 LOFO + inner time val)
    # ------------------------------------------------------------------
    logger.info(
        f"Split Track2 LOFO: held_out_group={held_out_group} | inner_val_days={val_days}"
    )
    df_train_inner, df_val_inner, df_outer_test = lofo_time_val_split(
        df,
        held_out_group=held_out_group,
        group_col=(group_col or zone_col),
        time_col=time_col,
        val_days=val_days,
        min_train=min_train,
    )

    # ------------------------------------------------------------------
    # Train + evaluate
    # ------------------------------------------------------------------
    logger.info("Prepare X/y")
    X_train, y_train = select_ml_xy(
        df_train_inner,
        target_col=target_col,
        drop_cols=(time_col,),
        keep_zone=keep_zone,
    )
    X_val, y_val = select_ml_xy(
        df_val_inner,
        target_col=target_col,
        drop_cols=(time_col,),
        keep_zone=keep_zone,
    )
    X_test, y_test = select_ml_xy(
        df_outer_test,
        target_col=target_col,
        drop_cols=(time_col,),
        keep_zone=keep_zone,
    )

    logger.info("Train LGBM (early stop on inner val)")
    model = create_lgbm_model(params=lgbm_params, seed=seed)
    model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)

    logger.info("Predict + evaluate")
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    metrics: Dict[str, Any] = {
        "track": "track2_lofo",
        "split": {
            "protocol": "lofo_time_val",
            "held_out_group": held_out_group,
            "val_days": int(val_days),
        },
        "features": {"lags": list(lags), "rolling_windows": list(rolling_windows)},
        "keep_zone": bool(keep_zone),
        "model": {"lgbm_params": dict(lgbm_params), "seed": seed},
        "inner_val": {
            "rmse": evaluate_rmse(y_val, pred_val),
            "mae": evaluate_mae(y_val, pred_val),
            "r2": evaluate_r2(y_val, pred_val),
            "n": int(len(y_val)),
        },
        "outer_test": {
            "rmse": evaluate_rmse(y_test, pred_test),
            "mae": evaluate_mae(y_test, pred_test),
            "r2": evaluate_r2(y_test, pred_test),
            "n": int(len(y_test)),
        },
        "n_rows": {
            "train_inner": int(len(df_train_inner)),
            "val_inner": int(len(df_val_inner)),
            "outer_test": int(len(df_outer_test)),
        },
    }

    metrics_path = out_dir_p / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model_path = out_dir_p / "model.txt"
    model.save(str(model_path))

    logger.info(f"Saved to {out_dir_p}")
    logger.info(
        f"INNER VAL RMSE={metrics['inner_val']['rmse']:.4f} | "
        f"OUTER TEST RMSE={metrics['outer_test']['rmse']:.4f}"
    )

    return DeterministicLGBMResult(
        out_dir=str(out_dir_p),
        model_path=str(model_path),
        metrics_path=str(metrics_path),
        config_snapshot_path=str(config_snapshot_path),
        metrics=metrics,
    )
def run_pipeline_any(
    *,
    data_path: str | Path,
    out_dir: str | Path,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
    target_col: str = "target",
    zone_col: str = "zone_id",
    time_col: str = "datetime",
    keep_zone: bool = True,
    seed: int | None = None,
    lgbm_params: Dict[str, Any] | None = None,
    # Track 1 temporal
    train_cut: Optional[str] = None,
    val_cut: Optional[str] = None,
    # Track 2 LOFO
    held_out_group: Optional[Union[int, str]] = None,
    group_col: Optional[str] = None,
    val_days: Optional[int] = None,
    min_train: int = 1000,
) -> DeterministicLGBMResult:
    """Convenience dispatcher to run Track1 or Track2 without the runner branching.

    - If `train_cut` and `val_cut` are provided -> Track1 temporal split.
    - Else if `held_out_group` and `val_days` are provided -> Track2 LOFO + inner time-val.

    This keeps the pipeline module config-agnostic while making callers simpler.
    """
    if train_cut is not None and val_cut is not None:
        return run_pipeline(
            data_path=data_path,
            out_dir=out_dir,
            train_cut=train_cut,
            val_cut=val_cut,
            lags=lags,
            rolling_windows=rolling_windows,
            target_col=target_col,
            zone_col=zone_col,
            time_col=time_col,
            keep_zone=keep_zone,
            seed=seed,
            lgbm_params=lgbm_params,
        )

    if held_out_group is not None and val_days is not None:
        return run_pipeline_lofo(
            data_path=data_path,
            out_dir=out_dir,
            held_out_group=held_out_group,
            val_days=int(val_days),
            lags=lags,
            rolling_windows=rolling_windows,
            target_col=target_col,
            zone_col=zone_col,
            group_col=group_col,
            time_col=time_col,
            keep_zone=keep_zone,
            seed=seed,
            lgbm_params=lgbm_params,
            min_train=min_train,
        )

    raise ValueError(
        "run_pipeline_any requires either (train_cut & val_cut) for Track1, "
        "or (held_out_group & val_days) for Track2 LOFO."
    )