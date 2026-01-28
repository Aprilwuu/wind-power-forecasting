# src/postprocess/noise.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class IntervalSummary:
    picp: float
    mpiw: float
    lo: float
    hi: float
    sigma_resid: float
    mean_std_mc: float
    mean_std_total: float


def load_npz(npz_path: str | Path) -> Dict[str, np.ndarray]:
    npz_path = Path(npz_path)
    d = np.load(npz_path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def save_npz(npz_path: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)


def reshape_1d(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1)


def compute_picp_mpiw(y_true: np.ndarray, qlo: np.ndarray, qhi: np.ndarray) -> Tuple[float, float]:
    y = reshape_1d(y_true)
    lo = reshape_1d(qlo)
    hi = reshape_1d(qhi)
    picp = float(np.mean((y >= lo) & (y <= hi)))
    mpiw = float(np.mean(hi - lo))
    return picp, mpiw


def estimate_sigma_resid(
    inner_npz: Dict[str, np.ndarray],
    method: str = "rmse",
    min_sigma: float = 1e-8,
) -> float:
    """
    Estimate residual noise scale from inner validation (NO leakage).
    method:
      - "rmse": sqrt(mean(resid^2))
      - "mad" : 1.4826 * median(|resid - median(resid)|)  (robust)
      - "std" : std(resid)  (close to rmse if mean(resid)~0)
    """
    y = reshape_1d(inner_npz["y_true"])
    mu = reshape_1d(inner_npz["mean"])
    resid = y - mu

    method = method.lower().strip()
    if method == "rmse":
        sigma = float(np.sqrt(np.mean(resid ** 2)))
    elif method == "std":
        sigma = float(np.std(resid))
    elif method == "mad":
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        sigma = float(1.4826 * mad)
    else:
        raise ValueError(f"Unknown method={method}. Use one of: rmse, std, mad")

    return max(sigma, float(min_sigma))


def augment_intervals_with_residual_noise(
    outer_npz: Dict[str, np.ndarray],
    sigma_resid: float,
    lo: float = 0.05,
    hi: float = 0.95,
    z_value: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Build augmented intervals using:
      std_total = sqrt(std_mc^2 + sigma_resid^2)
      qlo = mean - z * std_total
      qhi = mean + z * std_total

    Notes:
      - This assumes (approximately) symmetric central interval.
      - If z_value is None, caller should provide it (recommended) using NormalDist().inv_cdf(hi).
    """
    if z_value is None:
        raise ValueError("z_value must be provided by caller (e.g., NormalDist().inv_cdf(hi)).")

    y = outer_npz["y_true"]
    mu = outer_npz["mean"]
    std_mc = outer_npz["std"]  # this is your MC std

    std_total = np.sqrt(std_mc ** 2 + (sigma_resid ** 2)).astype(mu.dtype)
    qlo = (mu - z_value * std_total).astype(mu.dtype)
    qhi = (mu + z_value * std_total).astype(mu.dtype)

    out = dict(outer_npz)  # keep original keys
    # overwrite q05/q95 by default (since your current setup is 0.05/0.95)
    out["q05"] = qlo
    out["q95"] = qhi
    # keep q50; if missing, set to mean
    out["q50"] = outer_npz.get("q50", mu)

    # keep original mc std separately for debugging
    out["std_mc"] = std_mc
    out["std_total"] = std_total
    out["sigma_resid"] = np.array([sigma_resid], dtype=np.float32)

    return out


def summarize_intervals(
    y_true: np.ndarray,
    qlo: np.ndarray,
    qhi: np.ndarray,
    sigma_resid: float,
    std_mc: np.ndarray,
    std_total: np.ndarray,
    lo: float,
    hi: float,
) -> IntervalSummary:
    picp, mpiw = compute_picp_mpiw(y_true, qlo, qhi)
    return IntervalSummary(
        picp=picp,
        mpiw=mpiw,
        lo=lo,
        hi=hi,
        sigma_resid=float(sigma_resid),
        mean_std_mc=float(np.mean(std_mc)),
        mean_std_total=float(np.mean(std_total)),
    )
