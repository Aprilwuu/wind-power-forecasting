# src/pipelines/quantile_lgbm_pipeline.py
"""Quantile (probabilistic) LightGBM pipelines.

- Track 1: temporal cut-based split
- Track 2: LOFO (leave-one-group-out) + inner time validation

Strategy: train one LightGBM model per quantile using objective='quantile' and alpha=q.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np

from src.data.load import load_raw_data
from src.data.make_dataset import make_dataset
from src.data.split import train_valid_split, lofo_time_val_split
from src.features.build_features import make_ml_features, select_ml_xy
from src.metrics.deterministic import evaluate_mae, evaluate_r2, evaluate_rmse
from src.metrics.probabilistic import pinball_loss_multi
from src.models.deterministic.lgbm import create_lgbm_model

logger = logging.getLogger(__name__)


def _ensure_dir(p: str | Path) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


@dataclass
class QuantileLGBMResult:
    out_dir: str
    model_paths: Dict[str, str]  # quantile(str) -> model path
    metrics_path: str
    config_snapshot_path: str
    metrics: Dict[str, Any]


def _validate_quantiles(quantiles: Iterable[float]) -> list[float]:
    qs = [float(q) for q in quantiles]
    if len(qs) == 0:
        raise ValueError("quantiles must be a non-empty iterable of floats.")
    for q in qs:
        if not (0.0 < q < 1.0):
            raise ValueError(f"Each quantile must be in (0, 1). Got {q}.")
    qs = sorted(set(round(q, 6) for q in qs))
    return [float(q) for q in qs]


def _qkey(q: float) -> str:
    return f"{q:.6f}"


# Rearrangement helper to enforce non-crossing quantiles
def _rearrange_quantiles(q_preds: np.ndarray) -> np.ndarray:
    """Rearrangement to enforce non-crossing quantiles per row.

    For each sample (row), sort predicted quantiles in ascending order.
    This is the standard "rearrangement" fix for quantile crossing.
    """
    if q_preds.ndim != 2 or q_preds.shape[1] < 2:
        return q_preds
    # sort along quantile dimension (columns)
    return np.sort(q_preds, axis=1)


def _crossing_rate(q_preds: np.ndarray) -> float:
    """Fraction of rows where any adjacent quantiles cross."""
    if q_preds.ndim != 2 or q_preds.shape[1] < 2:
        return 0.0
    crosses = (q_preds[:, :-1] > q_preds[:, 1:]).any(axis=1)
    return float(crosses.mean())


def run_pipeline_quantile(
    *,
    data_path: str | Path,
    out_dir: str | Path,
    train_cut: str,
    val_cut: str,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
    quantiles: Iterable[float],
    interval_pairs: Optional[Iterable[tuple[float, float]]] = None,
    target_col: str = "target",
    zone_col: str = "zone_id",
    time_col: str = "datetime",
    keep_zone: bool = True,
    seed: int | None = None,
    lgbm_params: Dict[str, Any] | None = None,
) -> QuantileLGBMResult:
    """Track 1: cut-based temporal split."""
    if not isinstance(lgbm_params, dict) or len(lgbm_params) == 0:
        raise ValueError(
            "lgbm_params must be a non-empty dict. "
            "The runner should provide merged params (base_params + override)."
        )

    qs = _validate_quantiles(quantiles)
    out_dir_p = _ensure_dir(out_dir).resolve()

    # snapshot
    config_snapshot_path = out_dir_p / "config_snapshot.json"
    snapshot = {
        "runtime": {
            "data_path": str(Path(data_path).resolve()),
            "out_dir": str(out_dir_p),
            "train_cut": train_cut,
            "val_cut": val_cut,
            "lags": list(lags),
            "rolling_windows": list(rolling_windows),
            "quantiles": qs,
            "target_col": target_col,
            "zone_col": zone_col,
            "time_col": time_col,
            "keep_zone": bool(keep_zone),
            "seed": seed,
            "lgbm_params": dict(lgbm_params),
        }
    }
    config_snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    # load -> clean -> features
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
    df = df.dropna().reset_index(drop=True)

    # split track1
    logger.info(f"Split Track1 by cuts: train_cut={train_cut}, val_cut={val_cut}")
    df_train, df_val, df_test = train_valid_split(
        df,
        time_col=time_col,
        zone_col=zone_col,
        train_cut=train_cut,
        val_cut=val_cut,
    )

    # X/y
    logger.info("Prepare X/y")
    X_train, y_train = select_ml_xy(df_train, target_col=target_col, drop_cols=(time_col,), keep_zone=keep_zone)
    X_val, y_val = select_ml_xy(df_val, target_col=target_col, drop_cols=(time_col,), keep_zone=keep_zone)
    X_test, y_test = select_ml_xy(df_test, target_col=target_col, drop_cols=(time_col,), keep_zone=keep_zone)

    models_dir = _ensure_dir(out_dir_p / "models")
    model_paths: Dict[str, str] = {}

    q_pred_val = np.zeros((len(y_val), len(qs)), dtype=float)
    q_pred_test = np.zeros((len(y_test), len(qs)), dtype=float)

    logger.info(f"Train {len(qs)} quantile models: {qs}")
    for j, q in enumerate(qs):
        params_q = dict(lgbm_params)
        params_q["objective"] = "quantile"
        params_q["alpha"] = float(q)

        model = create_lgbm_model(params=params_q, seed=seed)
        model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)

        q_pred_val[:, j] = model.predict(X_val)
        q_pred_test[:, j] = model.predict(X_test)

        qk = _qkey(q)
        mp = Path(models_dir) / f"model_q{qk}.txt"
        model.save(str(mp))
        model_paths[qk] = str(mp)

    # Enforce non-crossing quantiles (rearrangement)
    q_pred_val = _rearrange_quantiles(q_pred_val)
    q_pred_test = _rearrange_quantiles(q_pred_test)

    # metrics
    val_pinball = pinball_loss_multi(y_val, q_pred_val, qs)
    test_pinball = pinball_loss_multi(y_test, q_pred_test, qs)

    median_val: Dict[str, Any] = {}
    median_test: Dict[str, Any] = {}
    if 0.5 in qs:
        j50 = qs.index(0.5)
        median_val = {
            "rmse": evaluate_rmse(y_val, q_pred_val[:, j50]),
            "mae": evaluate_mae(y_val, q_pred_val[:, j50]),
            "r2": evaluate_r2(y_val, q_pred_val[:, j50]),
        }
        median_test = {
            "rmse": evaluate_rmse(y_test, q_pred_test[:, j50]),
            "mae": evaluate_mae(y_test, q_pred_test[:, j50]),
            "r2": evaluate_r2(y_test, q_pred_test[:, j50]),
        }

    # Coverage/width for arbitrary quantile intervals.
    # By default we compute the outermost available interval (min_q, max_q)
    # so quantiles like [0.3, 0.5, 0.7] will produce coverage for [0.3, 0.7].
    coverage: Dict[str, Any] = {}

    pairs: list[tuple[float, float]] = []
    if interval_pairs is not None:
        pairs.extend([(float(a), float(b)) for (a, b) in interval_pairs])

    if len(qs) >= 2:
        pairs.insert(0, (qs[0], qs[-1]))  # default interval first

    # de-duplicate while preserving order
    seen: set[tuple[float, float]] = set()
    uniq_pairs: list[tuple[float, float]] = []
    for a, b in pairs:
        if (a, b) not in seen:
            uniq_pairs.append((a, b))
            seen.add((a, b))

    for q_lo, q_hi in uniq_pairs:
        if not (q_lo < q_hi):
            continue
        if q_lo not in qs or q_hi not in qs:
            continue

        j_lo, j_hi = qs.index(q_lo), qs.index(q_hi)

        # Repaired bounds: handle quantile crossing by swapping per-sample.
        lo_val = np.minimum(q_pred_val[:, j_lo], q_pred_val[:, j_hi])
        hi_val = np.maximum(q_pred_val[:, j_lo], q_pred_val[:, j_hi])
        lo_test = np.minimum(q_pred_test[:, j_lo], q_pred_test[:, j_hi])
        hi_test = np.maximum(q_pred_test[:, j_lo], q_pred_test[:, j_hi])

        name = "default" if (q_lo == qs[0] and q_hi == qs[-1]) else f"q{q_lo:.3f}_{q_hi:.3f}"
        nominal = float(q_hi - q_lo)

        coverage[name] = {
            "q_lo": float(q_lo),
            "q_hi": float(q_hi),
            "nominal": nominal,
            "val": {
                "coverage": float(((y_val >= lo_val) & (y_val <= hi_val)).mean()),
                "avg_width": float(np.mean(hi_val - lo_val)),
            },
            "test": {
                "coverage": float(((y_test >= lo_test) & (y_test <= hi_test)).mean()),
                "avg_width": float(np.mean(hi_test - lo_test)),
            },
        }

    metrics: Dict[str, Any] = {
        "track": "track1_temporal_quantile",
        "split": {"train_cut": train_cut, "val_cut": val_cut},
        "features": {"lags": list(lags), "rolling_windows": list(rolling_windows)},
        "quantiles": qs,
        "keep_zone": bool(keep_zone),
        "model": {"lgbm_params": dict(lgbm_params), "seed": seed},
        "val": {
            "pinball": val_pinball,
            "pinball_mean": float(np.mean(val_pinball)),
            "crossing_rate": _crossing_rate(q_pred_val),
            "n": int(len(y_val)),
            **({"median": median_val} if median_val else {}),
        },
        "test": {
            "pinball": test_pinball,
            "pinball_mean": float(np.mean(test_pinball)),
            "crossing_rate": _crossing_rate(q_pred_test),
            "n": int(len(y_test)),
            **({"median": median_test} if median_test else {}),
        },
        **({"coverage": coverage} if coverage else {}),
        "n_rows": {"train": int(len(df_train)), "val": int(len(df_val)), "test": int(len(df_test))},
    }

    metrics_path = out_dir_p / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info(f"Saved to {out_dir_p}")
    logger.info(f"VAL pinball_mean={metrics['val']['pinball_mean']:.6f} | TEST pinball_mean={metrics['test']['pinball_mean']:.6f}")

    return QuantileLGBMResult(
        out_dir=str(out_dir_p),
        model_paths=model_paths,
        metrics_path=str(metrics_path),
        config_snapshot_path=str(config_snapshot_path),
        metrics=metrics,
    )


def run_pipeline_quantile_lofo(
    *,
    data_path: str | Path,
    out_dir: str | Path,
    held_out_group: int | str,
    val_days: int,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
    quantiles: Iterable[float],
    interval_pairs: Optional[Iterable[tuple[float, float]]] = None,
    target_col: str = "target",
    zone_col: str = "zone_id",
    group_col: Optional[str] = None,
    time_col: str = "datetime",
    keep_zone: bool = True,
    seed: int | None = None,
    lgbm_params: Dict[str, Any] | None = None,
    min_train: int = 1000,
) -> QuantileLGBMResult:
    """Track 2: LOFO + inner time validation."""
    if not isinstance(lgbm_params, dict) or len(lgbm_params) == 0:
        raise ValueError(
            "lgbm_params must be a non-empty dict. "
            "The runner should provide merged params (base_params + override)."
        )

    qs = _validate_quantiles(quantiles)
    out_dir_p = _ensure_dir(out_dir).resolve()

    # snapshot
    config_snapshot_path = out_dir_p / "config_snapshot.json"
    snapshot = {
        "runtime": {
            "data_path": str(Path(data_path).resolve()),
            "out_dir": str(out_dir_p),
            "protocol": "track2_lofo_time_val_quantile",
            "held_out_group": held_out_group,
            "val_days": int(val_days),
            "lags": list(lags),
            "rolling_windows": list(rolling_windows),
            "quantiles": qs,
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

    # load -> clean -> features
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
    df = df.dropna().reset_index(drop=True)

    # split LOFO
    logger.info(f"Split Track2 LOFO: held_out_group={held_out_group} | inner_val_days={val_days}")
    df_train_inner, df_val_inner, df_outer_test = lofo_time_val_split(
        df,
        held_out_group=held_out_group,
        group_col=(group_col or zone_col),
        time_col=time_col,
        val_days=val_days,
        min_train=min_train,
    )

    # X/y
    logger.info("Prepare X/y")
    X_train, y_train = select_ml_xy(df_train_inner, target_col=target_col, drop_cols=(time_col,), keep_zone=keep_zone)
    X_val, y_val = select_ml_xy(df_val_inner, target_col=target_col, drop_cols=(time_col,), keep_zone=keep_zone)
    X_test, y_test = select_ml_xy(df_outer_test, target_col=target_col, drop_cols=(time_col,), keep_zone=keep_zone)

    models_dir = _ensure_dir(out_dir_p / "models")
    model_paths: Dict[str, str] = {}

    q_pred_val = np.zeros((len(y_val), len(qs)), dtype=float)
    q_pred_test = np.zeros((len(y_test), len(qs)), dtype=float)

    logger.info(f"Train {len(qs)} quantile models (LOFO): {qs}")
    for j, q in enumerate(qs):
        params_q = dict(lgbm_params)
        params_q["objective"] = "quantile"
        params_q["alpha"] = float(q)

        model = create_lgbm_model(params=params_q, seed=seed)
        model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)

        q_pred_val[:, j] = model.predict(X_val)
        q_pred_test[:, j] = model.predict(X_test)

        qk = _qkey(q)
        mp = Path(models_dir) / f"model_q{qk}.txt"
        model.save(str(mp))
        model_paths[qk] = str(mp)

    # Enforce non-crossing quantiles (rearrangement)
    q_pred_val = _rearrange_quantiles(q_pred_val)
    q_pred_test = _rearrange_quantiles(q_pred_test)

    inner_pinball = pinball_loss_multi(y_val, q_pred_val, qs)
    outer_pinball = pinball_loss_multi(y_test, q_pred_test, qs)

    # Coverage/width for arbitrary quantile intervals (inner_val + outer_test).
    coverage: Dict[str, Any] = {}

    pairs: list[tuple[float, float]] = []
    if interval_pairs is not None:
        pairs.extend([(float(a), float(b)) for (a, b) in interval_pairs])

    if len(qs) >= 2:
        pairs.insert(0, (qs[0], qs[-1]))

    seen: set[tuple[float, float]] = set()
    uniq_pairs: list[tuple[float, float]] = []
    for a, b in pairs:
        if (a, b) not in seen:
            uniq_pairs.append((a, b))
            seen.add((a, b))

    for q_lo, q_hi in uniq_pairs:
        if not (q_lo < q_hi):
            continue
        if q_lo not in qs or q_hi not in qs:
            continue

        j_lo, j_hi = qs.index(q_lo), qs.index(q_hi)

        lo_val = np.minimum(q_pred_val[:, j_lo], q_pred_val[:, j_hi])
        hi_val = np.maximum(q_pred_val[:, j_lo], q_pred_val[:, j_hi])
        lo_test = np.minimum(q_pred_test[:, j_lo], q_pred_test[:, j_hi])
        hi_test = np.maximum(q_pred_test[:, j_lo], q_pred_test[:, j_hi])

        name = "default" if (q_lo == qs[0] and q_hi == qs[-1]) else f"q{q_lo:.3f}_{q_hi:.3f}"
        nominal = float(q_hi - q_lo)

        coverage[name] = {
            "q_lo": float(q_lo),
            "q_hi": float(q_hi),
            "nominal": nominal,
            "inner_val": {
                "coverage": float(((y_val >= lo_val) & (y_val <= hi_val)).mean()),
                "avg_width": float(np.mean(hi_val - lo_val)),
            },
            "outer_test": {
                "coverage": float(((y_test >= lo_test) & (y_test <= hi_test)).mean()),
                "avg_width": float(np.mean(hi_test - lo_test)),
            },
        }

    metrics: Dict[str, Any] = {
        "track": "track2_lofo_quantile",
        "split": {"protocol": "lofo_time_val", "held_out_group": held_out_group, "val_days": int(val_days)},
        "features": {"lags": list(lags), "rolling_windows": list(rolling_windows)},
        "quantiles": qs,
        "keep_zone": bool(keep_zone),
        "model": {"lgbm_params": dict(lgbm_params), "seed": seed},
        "inner_val": {
            "pinball": inner_pinball,
            "pinball_mean": float(np.mean(inner_pinball)),
            "crossing_rate": _crossing_rate(q_pred_val),
            "n": int(len(y_val)),
        },
        "outer_test": {
            "pinball": outer_pinball,
            "pinball_mean": float(np.mean(outer_pinball)),
            "crossing_rate": _crossing_rate(q_pred_test),
            "n": int(len(y_test)),
        },
        "n_rows": {
            "train_inner": int(len(df_train_inner)),
            "val_inner": int(len(df_val_inner)),
            "outer_test": int(len(df_outer_test)),
        },
        **({"coverage": coverage} if coverage else {}),
    }

    metrics_path = out_dir_p / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info(f"Saved to {out_dir_p}")
    logger.info(f"INNER pinball_mean={metrics['inner_val']['pinball_mean']:.6f} | OUTER pinball_mean={metrics['outer_test']['pinball_mean']:.6f}")

    return QuantileLGBMResult(
        out_dir=str(out_dir_p),
        model_paths=model_paths,
        metrics_path=str(metrics_path),
        config_snapshot_path=str(config_snapshot_path),
        metrics=metrics,
    )


def run_pipeline_any_quantile(
    *,
    data_path: str | Path,
    out_dir: str | Path,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
    quantiles: Iterable[float],
    interval_pairs: Optional[Iterable[tuple[float, float]]] = None,
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
) -> QuantileLGBMResult:
    """Convenience dispatcher for quantile pipelines (Track1 or Track2)."""
    if train_cut is not None and val_cut is not None:
        return run_pipeline_quantile(
            data_path=data_path,
            out_dir=out_dir,
            train_cut=train_cut,
            val_cut=val_cut,
            lags=lags,
            rolling_windows=rolling_windows,
            quantiles=quantiles,
            interval_pairs=interval_pairs,
            target_col=target_col,
            zone_col=zone_col,
            time_col=time_col,
            keep_zone=keep_zone,
            seed=seed,
            lgbm_params=lgbm_params,
        )

    if held_out_group is not None and val_days is not None:
        return run_pipeline_quantile_lofo(
            data_path=data_path,
            out_dir=out_dir,
            held_out_group=held_out_group,
            val_days=int(val_days),
            lags=lags,
            rolling_windows=rolling_windows,
            quantiles=quantiles,
            interval_pairs=interval_pairs,
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
        "run_pipeline_any_quantile requires either (train_cut & val_cut) for Track1, "
        "or (held_out_group & val_days) for Track2 LOFO."
    )