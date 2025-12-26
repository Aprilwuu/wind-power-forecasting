"""
MC Dropout utilities.

Key idea:
- Keep the model in eval mode so BatchNorm (and other eval-time behaviors) stay stable.
- Re-enable ONLY dropout layers during inference to sample from the approximate posterior.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


_DROPOUT_TYPES: Tuple[type, ...] = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
)


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during evaluation (MC Dropout),
    while keeping BatchNorm layers in eval mode.

    Notes:
        - Calling model.eval() puts the whole model in eval mode.
        - We then switch ONLY dropout modules back to train mode.
        - This keeps BatchNorm/LayerNorm/etc. in eval mode (BatchNorm won't update running stats).
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES):
            m.train()


def _maybe_squeeze_single_horizon(arr: np.ndarray) -> np.ndarray:
    """
    If the last dimension is singleton (e.g., [..., 1]), squeeze it to [...].
    Otherwise keep as-is (e.g., [B, H]).

    Works for:
        - [B, 1] -> [B]
        - [T, B, 1] -> [T, B]
        - [B, H] -> unchanged
        - [T, B, H] -> unchanged
    """
    if arr.ndim >= 2 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def _validate_quantiles(qs: Iterable[float]) -> Tuple[float, ...]:
    q_list = [float(q) for q in qs]
    if len(q_list) == 0:
        raise ValueError("quantiles must be a non-empty iterable of floats in [0, 1].")
    for q in q_list:
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"Invalid quantile {q}. Quantiles must be in [0, 1].")
    return tuple(q_list)


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    xb: torch.Tensor,
    device: torch.device | str,
    mc_runs: int = 50,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    squeeze_single_horizon: bool = True,
    move_to_device: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray], np.ndarray]:
    """
    Run MC Dropout inference.

    Args:
        model: Trained model (e.g., TCN). Must contain dropout layers for MC Dropout to work.
        xb: Input batch, typically [B, L, D].
        device: torch device or device string.
        mc_runs: Number of MC samples (T). Must be >= 1.
        quantiles: Quantiles to compute from the MC samples.
        squeeze_single_horizon: If True, converts output shaped [..., 1] to [...].
        move_to_device: If True, moves model/xb to device inside this function.
                        Set False if caller already moved them (slightly faster).

    Returns:
        preds_mean: [B] (single-step) OR [B, H] (multi-step)
        preds_std:  [B] (single-step) OR [B, H] (multi-step)
        q_dict:     {q: [B] or [B, H]} for each quantile
        preds_all:  [T, B] (single-step) OR [T, B, H] (multi-step)
    """
    if mc_runs < 1:
        raise ValueError(f"mc_runs must be >= 1, got {mc_runs}.")
    q_tuple = _validate_quantiles(quantiles)

    device = torch.device(device)

    if move_to_device:
        model = model.to(device)
        xb = xb.to(device)

    enable_mc_dropout(model)

    # First forward pass to determine output shape and dtype.
    out0 = model(xb)
    out0_np = out0.detach().cpu().numpy()
    preds_list_shape = out0_np.shape  # e.g., [B, 1] or [B, H]

    # Pre-allocate for speed: [T, *out_shape]
    preds_all = np.empty((mc_runs, *preds_list_shape), dtype=out0_np.dtype)
    preds_all[0] = out0_np

    for t in range(1, mc_runs):
        out = model(xb)
        preds_all[t] = out.detach().cpu().numpy()

    if squeeze_single_horizon:
        preds_all = _maybe_squeeze_single_horizon(preds_all)  # [T, B] or [T, B, H]

    preds_mean = preds_all.mean(axis=0)
    preds_std = preds_all.std(axis=0)

    q_dict: Dict[float, np.ndarray] = {float(q): np.quantile(preds_all, q, axis=0) for q in q_tuple}

    return preds_mean, preds_std, q_dict, preds_all