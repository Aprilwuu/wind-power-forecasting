# src/postprocess/conformal.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConformalSummary:
    t: float
    target_picp: float
    n_cal: int
    q_level_used: float  # the quantile level used for t (finite-sample correction)
    clip_min: float
    clip_max: float


def _to_1d(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


def picp(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = _to_1d(y)
    lo = _to_1d(lo)
    hi = _to_1d(hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(lo: np.ndarray, hi: np.ndarray) -> float:
    lo = _to_1d(lo)
    hi = _to_1d(hi)
    return float(np.mean(hi - lo))


def conformal_scores(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Two-sided interval nonconformity score:
      s_i = max(lo - y, y - hi, 0)
    """
    y = _to_1d(y)
    lo = _to_1d(lo)
    hi = _to_1d(hi)
    return np.maximum(np.maximum(lo - y, y - hi), 0.0)


def conformal_t(
    scores: np.ndarray,
    target_picp: float,
) -> Tuple[float, float]:
    """
    Split conformal: choose t as a finite-sample-corrected quantile of scores.
    Standard choice:
      q = ceil((n+1)*(1-alpha))/n, alpha = 1-target_picp
      t = Quantile_q(scores) using 'higher' to avoid under-coverage.
    Returns (t, q_level_used).
    """
    s = _to_1d(scores)
    n = len(s)
    if n <= 0:
        raise ValueError("Empty calibration scores.")

    alpha = 1.0 - float(target_picp)
    q = float(np.ceil((n + 1) * (1.0 - alpha)) / n)
    q = min(1.0, max(0.0, q))

    try:
        t = float(np.quantile(s, q, method="higher"))
    except TypeError:  # older numpy
        t = float(np.quantile(s, q, interpolation="higher"))

    return t, q


def widen_interval(lo: np.ndarray, hi: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    lo2 = _to_1d(lo) - float(t)
    hi2 = _to_1d(hi) + float(t)
    return lo2, hi2


def clip_interval(lo: np.ndarray, hi: np.ndarray, clip_min: float = 0.0, clip_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    lo2 = np.clip(_to_1d(lo), clip_min, clip_max)
    hi2 = np.clip(_to_1d(hi), clip_min, clip_max)
    return lo2, hi2


def load_npz(npz_path: str | Path) -> Dict[str, np.ndarray]:
    p = Path(npz_path)
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}


def save_npz(npz_path: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    p = Path(npz_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)


def apply_conformal_to_npz(
    cal_npz: Dict[str, np.ndarray],
    apply_npz: Dict[str, np.ndarray],
    *,
    y_key: str = "y_true",
    lo_key: str = "q05",
    hi_key: str = "q95",
    target_picp: float = 0.9,
    out_suffix: str = "_cal",
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], ConformalSummary]:
    # compute t from calibration set
    if y_key not in cal_npz:
        raise KeyError(f"Calibration npz missing y_key={y_key}. keys={list(cal_npz.keys())}")
    for k in [lo_key, hi_key]:
        if k not in cal_npz:
            raise KeyError(f"Calibration npz missing key={k}. keys={list(cal_npz.keys())}")

    s = conformal_scores(cal_npz[y_key], cal_npz[lo_key], cal_npz[hi_key])
    t, q_used = conformal_t(s, target_picp=target_picp)

    # apply widening to BOTH (cal + apply) for convenience/diagnostics
    def _apply_one(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if lo_key not in d or hi_key not in d:
            raise KeyError(f"Apply npz missing interval keys: {lo_key}, {hi_key}. keys={list(d.keys())}")

        lo2, hi2 = widen_interval(d[lo_key], d[hi_key], t)
        lo2, hi2 = clip_interval(lo2, hi2, clip_min=clip_min, clip_max=clip_max)

        out = dict(d)
        out[f"{lo_key}{out_suffix}"] = lo2.reshape(-1, 1).astype(np.float32)
        out[f"{hi_key}{out_suffix}"] = hi2.reshape(-1, 1).astype(np.float32)
        out["conformal_t"] = np.array([t], dtype=np.float32)
        out["conformal_target_picp"] = np.array([target_picp], dtype=np.float32)
        out["conformal_q_used"] = np.array([q_used], dtype=np.float32)
        return out

    cal_out = _apply_one(cal_npz)
    apply_out = _apply_one(apply_npz)

    summary = ConformalSummary(
        t=float(t),
        target_picp=float(target_picp),
        n_cal=int(len(s)),
        q_level_used=float(q_used),
        clip_min=float(clip_min),
        clip_max=float(clip_max),
    )
    return cal_out, apply_out, summary


def apply_conformal_to_csv(
    cal_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    *,
    y_col: str = "y_true",
    lo_col: str = "q05",
    hi_col: str = "q95",
    target_picp: float = 0.9,
    out_suffix: str = "_cal",
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, ConformalSummary]:
    # compute t from calibration df
    for c in [y_col, lo_col, hi_col]:
        if c not in cal_df.columns:
            raise KeyError(f"Calibration csv missing column={c}. cols={list(cal_df.columns)}")
    for c in [lo_col, hi_col]:
        if c not in apply_df.columns:
            raise KeyError(f"Apply csv missing column={c}. cols={list(apply_df.columns)}")

    s = conformal_scores(cal_df[y_col].to_numpy(), cal_df[lo_col].to_numpy(), cal_df[hi_col].to_numpy())
    t, q_used = conformal_t(s, target_picp=target_picp)

    def _apply_one(df: pd.DataFrame) -> pd.DataFrame:
        lo2, hi2 = widen_interval(df[lo_col].to_numpy(), df[hi_col].to_numpy(), t)
        lo2, hi2 = clip_interval(lo2, hi2, clip_min=clip_min, clip_max=clip_max)
        out = df.copy()
        out[f"{lo_col}{out_suffix}"] = lo2
        out[f"{hi_col}{out_suffix}"] = hi2
        return out

    cal_out = _apply_one(cal_df)
    apply_out = _apply_one(apply_df)

    summary = ConformalSummary(
        t=float(t),
        target_picp=float(target_picp),
        n_cal=int(len(s)),
        q_level_used=float(q_used),
        clip_min=float(clip_min),
        clip_max=float(clip_max),
    )
    return cal_out, apply_out, summary
