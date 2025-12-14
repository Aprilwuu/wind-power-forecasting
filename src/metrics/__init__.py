"""
Metric package
==============

This package provides evaluation metrics for both deterministic and probabilistic models.

Modules
----------
- deterministic: Includes RMSE, MAE, R2 for point forecasts.
- probabilistic: Includes CRPS, Pinball Loss, PICP and AIW for distributional forecasts.
"""

from .deterministic import evaluate_mae, evaluate_r2, evaluate_rmse
from .probabilistic import (
    pinball_loss,
    pinball_loss_multi,
    crps_ensemble,
    picp,
    aiw,
)

__all__ = [
    "evaluate_rmse",
    "evaluate_mae",
    "evaluate_r2",
    "pinball_loss",
    "pinball_loss_multi",
    "crps_ensemble",
    "picp",
    "aiw",
]
